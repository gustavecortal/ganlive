"""The window the picture is shown in, and how big the picture should be for this screen.

Torch-free. The GPU that made a frame draws it into the window, or else SDL draws its BGRA bytes.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import traceback

from ganlive.curves import clamp01
from ganlive.engine.present import Screen

#: How far one arrow key pans a 1:1 view, as a fraction of the picture.
PAN_STEP = 0.1
#: Arrow key name -> (dx, dy). Resolved to pygame key codes once the window is open.
PAN_KEYS = {"left": (-PAN_STEP, 0.0), "right": (PAN_STEP, 0.0),
            "up": (0.0, -PAN_STEP), "down": (0.0, PAN_STEP)}


def parse_height(text: str) -> int | None:
    """A height argument: `auto` is None (fit the screen), `native` is 0, or a number of pixels."""
    if text == "auto":
        return None
    if text == "native":
        return 0
    try:
        height = int(text)
    except ValueError:
        height = -1
    if height <= 0:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a height; use auto, native, or a number of pixels")
    return height


def fit_height(native_h: int, native_w: int, want: int | None, screen=None) -> int:
    """The height a frame should be produced at: `want` (None fits the screen, 0 is native),
    never above native, and always even, since NV12 has half-height chroma."""
    if want is None:
        if not screen or min(screen) <= 0:
            return native_h
        sw, sh = screen
        want = min(sh, round(native_h * sw / native_w))
    return max(2, min(want, native_h)) // 2 * 2 if want else native_h


def screen_size():
    """The desktop's size in pixels, or None if it cannot be asked."""
    # Windows answers without starting a video subsystem; SDL answers everywhere.
    asks = (_win32_screen, _sdl_screen) if sys.platform == "win32" else (_sdl_screen,)
    for ask in asks:
        try:
            size = ask()
        except Exception:  # noqa: BLE001 -- no desktop at all, or no SDL: try the next
            continue
        if min(size) > 0:
            return size
    return None


def _win32_screen():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _sdl_screen():
    import pygame

    pygame.display.init()
    info = pygame.display.Info()
    return info.current_w, info.current_h


def window_size(w: int, h: int, screen_w: int, screen_h: int, overlay=None) -> tuple[int, int]:
    """How big to open the window for a `w` x `h` frame beside `overlay`."""
    panel = overlay.width if overlay is not None else 0
    floor = overlay.floor_height() if overlay is not None else 0
    s = min((screen_w - panel) * 0.88 / w, (screen_h - 90) / h, 1.0)
    return (max(320, round(w * s)) + panel,
            min(screen_h, max(240, round(h * s), floor)))


#: Whether the window gets a thread of its own. Not on macOS, where a window and its events
#: belong to the main thread; there each `publish` draws on the caller's thread instead.
THREADED = sys.platform != "darwin"


def placed(w: int, h: int, picture, one_to_one: bool, pan) -> tuple[tuple, tuple]:
    """`(dst, src)`: where a `w` x `h` frame goes in the `picture` area (width, height), and
    which part of it is shown, as `(x, y, w, h)` rects. Fitted whole, or at one frame pixel a
    window pixel, panned."""
    aw, vh = picture
    if one_to_one:
        cw, ch = min(w, aw), min(h, vh)
        src = (round((w - cw) * pan[0]), round((h - ch) * pan[1]), cw, ch)
        return ((aw - cw) // 2, (vh - ch) // 2, cw, ch), src
    s = min(aw / w, vh / h)
    dw, dh = round(w * s), round(h * s)
    return ((aw - dw) // 2, (vh - dh) // 2, dw, dh), (0, 0, w, h)


class Display:
    """A native window showing the latest frame `publish`ed to it, with `overlay`, the
    optional strip (`strip.DialPanel`), beside it.

    Given the wgpu `device` the frames are made on, that GPU draws the window
    (`engine.present.Screen`): `publish` takes a `screen.Stepped` frame and presents it at
    once, and nothing goes through host memory. Without one, or where the window offers wgpu
    no surface (`note` says why), `publish` takes the frame's BGRA bytes, `on_gpu` is False,
    and SDL uploads and draws them.

    The window has its own thread where the platform allows (`THREADED`), which paints the
    strip and takes the events, so neither holds up the frame loop.
    Keys: Esc/Q stop, F toggles fullscreen, N shows native pixels, arrows pan that view."""

    def __init__(self, size, title: str = "ganlive", overlay=None, fullscreen: bool = True,
                 threaded: bool = THREADED, device=None):
        self.size = size                      # (h, w) of the frames it will be given
        self.threaded = threaded
        self.stopped = False
        self.overlay = overlay
        self.fullscreen = fullscreen
        self.screen = None
        self.note = ""
        self._device = device
        self._frame = None
        self._seq = 0
        self._waiting = 0
        self._closed = False
        self._cond = threading.Condition()
        self._ready = threading.Event()
        self._error = None
        self._one_to_one = False
        self._pan = [0.5, 0.5]
        #: For the GPU screen: the strip's latest pixels, not yet uploaded, and the window's
        #: `(pixels, scale, picture, strip)` as its thread last laid it out.
        self._strip = None
        self._areas = None
        if not threaded:
            try:
                self._open(title)
            except Exception as e:  # noqa: BLE001  no display, no SDL, no pygame
                raise RuntimeError(f"display window failed to open: {type(e).__name__}: {e}") from e
            return
        threading.Thread(target=self._run, args=(title,), daemon=True).start()
        self._ready.wait(timeout=20)
        if self._error:
            raise RuntimeError(f"display window failed to open: {self._error}")

    @property
    def on_gpu(self) -> bool:
        """Whether `publish` takes a GPU frame rather than its bytes."""
        return self.screen is not None

    @property
    def wants(self) -> bool:
        """Whether a frame published now would be drawn, so one is worth converting."""
        return self.on_gpu or not self.threaded or self._waiting > 0

    def publish(self, frame) -> None:
        if self.stopped:
            return
        if self.screen is not None:
            self._guarded(self._present_gpu, frame)
            if not self.threaded:
                self._guarded(self._tend)
                return
        elif not self.threaded:
            self._guarded(self._step, frame)
            return
        with self._cond:
            self._frame, self._seq = frame, self._seq + 1
            self._cond.notify_all()

    def close(self) -> None:
        if not self.threaded:
            self._destroy()
            return
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _open(self, title: str) -> None:
        """Window, its drawing, and the overlay attached. Raises if there is no display."""
        if sys.platform == "win32":
            # Windows only: SDL refuses this driver name everywhere else.
            os.environ.setdefault("SDL_VIDEODRIVER", "windows")
        # Filter the picture when it is scaled to the window. SDL's default is nearest-neighbour,
        # which shimmers on a frame shrunk to fit.
        os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "linear")
        import pygame
        from pygame._sdl2.video import Renderer, Texture, Window

        self._pg, self._texture = pygame, Texture
        h, w = self.size
        # The display alone: `pygame.init` would also open an audio output and start polling
        # joysticks for the whole session. The strip starts its own fonts.
        pygame.display.init()
        info = pygame.display.Info()
        self._screen_h = info.current_h
        # Always opened windowed and then made fullscreen, so `f` has a window to go back to.
        self._win = Window(title, size=window_size(w, h, info.current_w, info.current_h,
                                                   self.overlay),
                           resizable=True)
        if self.fullscreen:
            self._win.set_fullscreen(True)
        if self._device is not None:
            try:
                self.screen = Screen(self._win, self._device)
            except Exception as e:  # noqa: BLE001  any window wgpu cannot draw: SDL draws it
                self.note = f"{type(e).__name__}: {e}"
        if self.screen is None:
            self._ren = Renderer(self._win, vsync=False)
            self._tex = Texture(self._ren, (w, h), depth=32, streaming=True)
            self._tex_size = (w, h)
            self._strip_tex = None
        self._pan_keys = {getattr(pygame, f"K_{name.upper()}"): d for name, d in PAN_KEYS.items()}
        if self.overlay is not None:
            self.overlay.attach()
        if self.screen is not None:
            self._tend()

    def _run(self, title: str) -> None:
        try:
            self._open(title)
        except Exception as e:  # noqa: BLE001  no display, no SDL, no pygame
            self._error = f"{type(e).__name__}: {e}"
            self._ready.set()
            return
        self._ready.set()
        self._guarded(self._serve)
        self._destroy()

    def _serve(self) -> None:
        """The window thread: draw each new frame as it is published (or, on the GPU screen,
        paint the strip for the next one), until `close`."""
        seq = 0
        while True:
            with self._cond:
                self._waiting += 1
                try:
                    self._cond.wait_for(lambda at=seq: self._seq != at or self._closed)
                finally:
                    self._waiting -= 1
                if self._closed:
                    return
                seq, frame = self._seq, self._frame
            if self.screen is not None:
                self._tend()
            else:
                self._step(frame)

    def _present_gpu(self, frame) -> None:
        """Draw `frame` on the GPU screen, with the strip the window's thread last painted."""
        strip = self._strip
        if strip is not None:
            self._strip = None
            self.screen.upload_strip(*strip)
        pixels, scale, picture, rect = self._areas
        dst, src = placed(frame.width, frame.height, picture, self._one_to_one, self._pan)
        self.screen.present(frame, pixels, [v * scale for v in dst], src,
                            [v * scale for v in rect])

    def _tend(self) -> None:
        """On the GPU screen, between frames: lay the window out, paint the strip, and take
        the events that arrived."""
        picture, rect = self._layout()
        if rect[2]:
            surface = self.overlay.compose(rect[2], rect[3])
            self._strip = (surface.get_buffer().raw, surface.get_pitch(), *surface.get_size())
        pixels = self.screen.pixels(self._win)
        self._areas = (pixels, pixels[0] / max(1, self._win.size[0]), picture, rect)
        self._events(rect)

    def _step(self, frame) -> None:
        """Draw one frame, then take the events that arrived."""
        self._events(self._present(frame))

    def _events(self, strip) -> None:
        for ev in self._pg.event.get():
            if strip[2] and self.overlay.handle(ev, strip):
                continue
            self._on_key(ev)

    def _guarded(self, fn, *args) -> None:
        """Run `fn`; if it raises, stop the session rather than go on timing frames nobody
        can see."""
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            self.stopped = True
            traceback.print_exc()

    def _destroy(self) -> None:
        try:
            self._win.destroy()
        except Exception:  # noqa: BLE001  never opened, or already gone
            pass

    def _present(self, frame) -> tuple[int, int, int, int]:
        """Upload one frame's bytes, draw it and the overlay, and show them. Returns the
        strip's rect."""
        pg, ren = self._pg, self._ren
        h, w = frame.shape[0], frame.shape[1]
        if (w, h) != self._tex_size:
            self._tex = self._texture(ren, (w, h), depth=32, streaming=True)
            self._tex_size = (w, h)
        self._tex.update(pg.image.frombuffer(memoryview(frame).cast("B"), (w, h), "BGRA"))
        ren.draw_color = (0, 0, 0, 255)
        ren.clear()
        picture, strip = self._layout()
        dst, src = placed(w, h, picture, self._one_to_one, self._pan)
        self._tex.draw(srcrect=src, dstrect=dst)
        if strip[2]:
            surface = self.overlay.compose(strip[2], strip[3])
            if self._strip_tex is None or self._strip_tex.width != strip[2] or self._strip_tex.height != strip[3]:
                self._strip_tex = self._texture(ren, (strip[2], strip[3]), streaming=True)
            self._strip_tex.update(surface)
            self._strip_tex.draw(dstrect=strip)
        ren.present()
        return strip

    def _layout(self) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        """`((picture width, height), strip rect)` for the window as it is now.

        Enforces the overlay's floor height every frame rather than once at open: the window is
        resizable, and the floor belongs to the loaded model, so a switch can raise it."""
        overlay, win = self.overlay, self._win
        vw, vh = win.size
        floor = 0 if overlay is None or self.fullscreen else overlay.floor_height()
        floor = min(floor, self._screen_h)
        if floor and vh < floor:
            win.size = (vw, floor)
            vw, vh = win.size
        pw = min(overlay.width, max(0, vw - 160)) if overlay is not None else 0
        return (vw - pw, vh), (vw - pw, 0, pw, vh)

    def _on_key(self, ev) -> None:
        """The window's own keys, for events the overlay did not take."""
        pg = self._pg
        if ev.type == pg.QUIT:
            self.stopped = True
        elif ev.type != pg.KEYDOWN:
            return
        elif ev.key in (pg.K_ESCAPE, pg.K_q):
            self.stopped = True
        elif ev.key == pg.K_n:
            self._one_to_one = not self._one_to_one
        elif ev.key == pg.K_f:
            # Kept here: pygame's `Window` can set fullscreen but not say whether it is.
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                self._win.set_fullscreen(True)
            else:
                self._win.set_windowed()
        elif ev.key in self._pan_keys:
            dx, dy = self._pan_keys[ev.key]
            self._pan = [clamp01(self._pan[0] + dx), clamp01(self._pan[1] + dy)]
