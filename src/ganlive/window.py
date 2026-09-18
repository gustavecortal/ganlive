"""The window the frame is presented in: one streaming texture, never a JPEG."""

from __future__ import annotations

import threading
import traceback

from ganlive.dials.table import clamp01


def window_size(w: int, h: int, screen_w: int, screen_h: int, overlay=None) -> tuple[int, int]:
    """How big to open the window for a `w` x `h` frame beside `overlay`."""
    panel = overlay.width if overlay is not None else 0
    floor = overlay.floor_height() if overlay is not None else 0
    s = min((screen_w - panel) * 0.88 / w, (screen_h - 90) / h, 1.0)
    return (max(320, round(w * s)) + panel,
            min(screen_h, max(240, round(h * s), floor)))


class Display:
    """Native-window presentation: the frame goes to a streaming texture, never to JPEG."""

    PIXELS = "bgra"

    def __init__(self, size, title: str = "ganlive", overlay=None, fullscreen: bool = True):
        self.size = size                      # (h, w) of the frames it will be given
        self.stopped = False
        self.viewers = 1                      # a window is its own viewer
        self.overlay = overlay
        self.fullscreen = fullscreen
        self._frame = None
        self._seq = 0
        self._waiting = 0
        self._closed = False
        self._cond = threading.Condition()
        self._ready = threading.Event()
        self._error = None
        threading.Thread(target=self._run, args=(title,), daemon=True).start()
        self._ready.wait(timeout=20)
        if self._error:
            raise RuntimeError(f"display window failed to open: {self._error}")

    @property
    def wants(self) -> bool:
        return self._waiting > 0

    def publish(self, frame) -> None:
        with self._cond:
            self._frame, self._seq = frame, self._seq + 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _run(self, title: str) -> None:
        try:
            import os
            import sys

            if sys.platform == "win32":
                # Only here. Named unconditionally, SDL refuses it everywhere else and the
                # whole block falls into the `except` below as "no display".
                os.environ.setdefault("SDL_VIDEODRIVER", "windows")
            import pygame
            from pygame._sdl2.video import Renderer, Texture, Window

            h, w = self.size
            pygame.init()
            info = pygame.display.Info()
            sw, sh = info.current_w, info.current_h
            overlay = self.overlay
            if self.fullscreen:
                win = Window(title, size=(sw, sh), fullscreen_desktop=True)
            else:
                win = Window(title, size=window_size(w, h, sw, sh, overlay),
                             resizable=True)
            ren = Renderer(win, vsync=False)
            tex = Texture(ren, (w, h), depth=32, streaming=True)
            one_to_one, pan = False, [0.5, 0.5]
            if overlay is not None:
                overlay.attach(ren)
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
                fh, fw = frame.shape[0], frame.shape[1]
                if (fw, fh) != (w, h):
                    w, h = fw, fh
                    tex = Texture(ren, (w, h), depth=32, streaming=True)
                surf = pygame.image.frombuffer(memoryview(frame).cast("B"), (w, h),
                                               self.PIXELS.upper())
                tex.update(surf)
                ren.draw_color = (0, 0, 0, 255)
                ren.clear()
                # **A floor applied only to the opening size is not a floor**, because the window is
                # resizable: dragging it shorter reproduces exactly the defect `window_size` exists to
                # prevent -- the strip's scope, description, drum lights and status lines run off the bottom
                # with no error.
                vw, vh = win.size
                # Asked every frame, not once at open: the floor is the loaded model's, and a
                # switch to one with more dials raises it under a window that was already open.
                floor = 0 if overlay is None or self.fullscreen else overlay.floor_height()
                if floor and vh < floor:
                    win.size = (vw, floor)
                    vw, vh = win.size
                pw = min(overlay.width, max(0, vw - 160)) if overlay is not None else 0
                strip = (vw - pw, 0, pw, vh)
                aw = vw - pw
                if one_to_one:
                    cw, ch = min(w, aw), min(h, vh)
                    x = round((w - cw) * pan[0])
                    y = round((h - ch) * pan[1])
                    tex.draw(srcrect=(x, y, cw, ch),
                             dstrect=((aw - cw) // 2, (vh - ch) // 2, cw, ch))
                else:
                    s = min(aw / w, vh / h)
                    dw, dh = round(w * s), round(h * s)
                    tex.draw(dstrect=((aw - dw) // 2, (vh - dh) // 2, dw, dh))
                if pw:
                    overlay.draw(ren, strip)
                ren.present()
                for ev in pygame.event.get():
                    if pw and overlay.handle(ev, strip):
                        continue
                    if ev.type == pygame.QUIT:
                        self.stopped = True
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                            self.stopped = True
                        elif ev.key == pygame.K_n:          # native pixels, cropped
                            one_to_one = not one_to_one
                        elif ev.key == pygame.K_f:
                            win.set_fullscreen(True) if not win.fullscreen else win.set_windowed()
                        elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                            d = {pygame.K_LEFT: (-.1, 0), pygame.K_RIGHT: (.1, 0),
                                 pygame.K_UP: (0, -.1), pygame.K_DOWN: (0, .1)}[ev.key]
                            pan[0] = clamp01(pan[0] + d[0])
                            pan[1] = clamp01(pan[1] + d[1])
        except Exception as e:  # noqa: BLE001
            # **A dead window thread must stop the session, not freeze it.** This thread owns
            # the only picture; when it raised, the traceback went to stderr and the frame loop
            # carried on timing frames nobody could see and reporting that they fit. One
            # `KeyError` from the strip cost a whole measurement that way.
            self._error = f"{type(e).__name__}: {e}"
            self.stopped = True
            traceback.print_exc()
        finally:
            try:
                win.destroy()
            except Exception:  # noqa: BLE001
                pass
