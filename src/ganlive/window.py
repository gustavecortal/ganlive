"""The window the picture is shown in, and how big the picture should be for this screen.

Torch-free: the frame arrives as BGRA bytes, which is what an SDL streaming texture already is.
"""
from __future__ import annotations

import sys
import threading
import traceback

from ganlive.curves import clamp01

#: How far one arrow key pans a 1:1 view, as a fraction of the picture.
PAN_STEP = 0.1
#: Arrow key name -> (dx, dy). Resolved to pygame key codes once the window is open.
PAN_KEYS = {"left": (-PAN_STEP, 0.0), "right": (PAN_STEP, 0.0),
            "up": (0.0, -PAN_STEP), "down": (0.0, PAN_STEP)}


def parse_height(text: str) -> int | None:
    """A height argument: `auto` is None (fit the screen), `native` is 0, or a number of pixels."""
    import argparse

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
    import ctypes

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


class Display:
    """A native window on its own thread, showing the latest frame `publish`ed to it.

    `overlay` is the optional strip drawn beside the picture (`strip.DialPanel`). Keys:
    Esc/Q stop, F toggles fullscreen, N shows native pixels, arrows pan that view."""

    def __init__(self, size, title: str = "ganlive", overlay=None, fullscreen: bool = True):
        self.size = size                      # (h, w) of the frames it will be given
        self.stopped = False
        self.overlay = overlay
        self.fullscreen = fullscreen
        self._frame = None
        self._seq = 0
        self._waiting = 0
        self._closed = False
        self._cond = threading.Condition()
        self._ready = threading.Event()
        self._error = None
        self._one_to_one = False
        self._pan = [0.5, 0.5]
        threading.Thread(target=self._run, args=(title,), daemon=True).start()
        self._ready.wait(timeout=20)
        if self._error:
            raise RuntimeError(f"display window failed to open: {self._error}")

    @property
    def wants(self) -> bool:
        """Whether the window thread is waiting for a frame, so one is worth converting."""
        return self._waiting > 0

    def publish(self, frame) -> None:
        with self._cond:
            self._frame, self._seq = frame, self._seq + 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _open(self, title: str) -> None:
        """Window, renderer, texture, and the overlay attached. Raises if there is no display."""
        import os

        if sys.platform == "win32":
            # Windows only: SDL refuses this driver name everywhere else.
            os.environ.setdefault("SDL_VIDEODRIVER", "windows")
        import pygame
        from pygame._sdl2.video import Renderer, Texture, Window

        self._pg, self._texture = pygame, Texture
        h, w = self.size
        pygame.init()
        info = pygame.display.Info()
        self._screen_h = info.current_h
        if self.fullscreen:
            self._win = Window(title, size=(info.current_w, info.current_h),
                               fullscreen_desktop=True)
        else:
            self._win = Window(title, size=window_size(w, h, info.current_w, info.current_h,
                                                       self.overlay),
                               resizable=True)
        self._ren = Renderer(self._win, vsync=False)
        self._tex = Texture(self._ren, (w, h), depth=32, streaming=True)
        self._tex_size = (w, h)
        self._pan_keys = {getattr(pygame, f"K_{name.upper()}"): d for name, d in PAN_KEYS.items()}
        if self.overlay is not None:
            self.overlay.attach(self._ren)

    def _run(self, title: str) -> None:
        try:
            self._open(title)
        except Exception as e:  # noqa: BLE001  no display, no SDL, no pygame
            self._error = f"{type(e).__name__}: {e}"
            self._ready.set()
            return
        self._ready.set()
        seq = 0
        try:
            while True:
                with self._cond:
                    self._waiting += 1
                    try:
                        self._cond.wait_for(lambda at=seq: self._seq != at or self._closed)
                    finally:
                        self._waiting -= 1
                    if self._closed:
                        break
                    seq, frame = self._seq, self._frame
                strip = self._present(frame)
                for ev in self._pg.event.get():
                    if strip[2] and self.overlay.handle(ev, strip):
                        continue
                    self._on_key(ev)
        except Exception as e:  # noqa: BLE001
            # A dead window thread stops the session: otherwise the frame loop would go on
            # timing frames nobody can see.
            self._error = f"{type(e).__name__}: {e}"
            self.stopped = True
            traceback.print_exc()
        finally:
            try:
                self._win.destroy()
            except Exception:  # noqa: BLE001
                pass

    def _present(self, frame) -> tuple[int, int, int, int]:
        """Upload one frame, draw it and the overlay, and show them. Returns the strip's rect."""
        pg, ren = self._pg, self._ren
        h, w = frame.shape[0], frame.shape[1]
        if (w, h) != self._tex_size:
            self._tex = self._texture(ren, (w, h), depth=32, streaming=True)
            self._tex_size = (w, h)
        self._tex.update(pg.image.frombuffer(memoryview(frame).cast("B"), (w, h), "BGRA"))
        ren.draw_color = (0, 0, 0, 255)
        ren.clear()
        picture, strip = self._layout()
        aw, vh = picture
        if self._one_to_one:
            cw, ch = min(w, aw), min(h, vh)
            x = round((w - cw) * self._pan[0])
            y = round((h - ch) * self._pan[1])
            self._tex.draw(srcrect=(x, y, cw, ch),
                           dstrect=((aw - cw) // 2, (vh - ch) // 2, cw, ch))
        else:
            s = min(aw / w, vh / h)
            dw, dh = round(w * s), round(h * s)
            self._tex.draw(dstrect=((aw - dw) // 2, (vh - dh) // 2, dw, dh))
        if strip[2]:
            self.overlay.draw(ren, strip)
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
            if self._win.fullscreen:
                self._win.set_windowed()
            else:
                self._win.set_fullscreen(True)
        elif ev.key in self._pan_keys:
            dx, dy = self._pan_keys[ev.key]
            self._pan = [clamp01(self._pan[0] + dx), clamp01(self._pan[1] + dy)]
