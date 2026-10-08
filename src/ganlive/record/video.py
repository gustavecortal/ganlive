"""Writing finished frames to a video file on one encoder thread, and saving stills."""
from __future__ import annotations

import queue
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from ganlive.files import size_mb


def nv12_plane_views(frame, height: int, width: int):
    """numpy views onto an nv12 frame's own buffers, honouring each plane's line size."""
    return [np.frombuffer(p, dtype=np.uint8).reshape(rows, p.line_size)[:, :width]
            for p, rows in ((frame.planes[0], height), (frame.planes[1], height // 2))]

#: Hardware encoders, in the order they are tried. The first that opens on this machine
#: wins; `libx264` always does, on the CPU, and costs most of the frame rate at 4K.
CODECS = ("av1_qsv", "hevc_qsv", "h264_nvenc", "hevc_nvenc", "hevc_videotoolbox",
          "h264_amf", "libx264")
DEFAULT_CODEC = "auto"

#: Encoder frames reused in rotation, so the writer does not allocate one per frame.
POOL = 4


def save_still(path: Path, rgb) -> threading.Thread:
    """Write one RGB frame as a PNG, on its own thread, and hand the thread back.

    Through pygame, which the window already needs, rather than another imaging library."""
    def write():
        import pygame

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # `swapaxes`: a frame is (H, W, 3) and a pygame surface is indexed (x, y).
        pygame.image.save(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), str(path))

    thread = threading.Thread(target=write, name=f"still {Path(path).name}", daemon=True)
    thread.start()
    return thread


class Recorder:
    """A container, a bounded queue of NV12 frames, and the one thread that encodes them.

    With `realtime`, a frame offered while the queue is full is dropped and the timestamps
    follow the wall clock, so the file plays at the speed the take happened. Without it,
    `offer` blocks and every frame is kept."""

    def __init__(self, path, width: int, height: int, fps: float, codec: str = DEFAULT_CODEC,
                 *, realtime: bool = False, depth: int = 4) -> None:
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.codec = codec
        self.realtime = bool(realtime)
        self._q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.offered = 0
        self.dropped = 0
        self.written = 0
        self.failed: list[BaseException] = []
        self._thread: threading.Thread | None = None
        self._t0: float | None = None
        self._pts = -1
        #: Set once the encoder is open, or has failed to: see `wait_open`.
        self._opened = threading.Event()

    def start(self) -> Recorder:
        self._thread = threading.Thread(target=self._run, name=f"encode {self.path.name}",
                                        daemon=True)
        self._thread.start()
        return self

    def wait_open(self, timeout: float = 30.0) -> bool:
        """Block until the encoder is open, or has failed. A hardware encoder starts its media
        engine as it opens; on an Intel Arc, starting it while the same card is running the
        generator has lost the device, so a take started mid-session waits for this with the
        card idle."""
        return self._opened.wait(timeout)

    @property
    def seconds(self) -> float:
        """How long this take has been running. 0 until the first frame is offered."""
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    @property
    def wants(self) -> bool:
        """Whether a frame offered right now would actually be written."""
        return not self._q.full()

    def skip(self) -> None:
        """Count a frame the caller chose not to fetch, so the report still adds up."""
        self.offered += 1
        self.dropped += 1

    def offer(self, plane) -> bool:
        """Hand one host frame to the writer. False means the picture did not wait for it."""
        self.offered += 1
        if self._t0 is None:
            self._t0 = time.perf_counter()
        if not self.realtime:
            self._pts += 1
            self._q.put((self._pts, plane))
            return True
        pts = max(self._pts + 1, round(self.seconds * self.fps))
        try:
            self._q.put_nowait((pts, plane))
        except queue.Full:
            self.dropped += 1
            return False
        self._pts = pts
        return True

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until the writer has caught up with everything offered so far."""
        want = self.offered - self.dropped
        end = time.perf_counter() + timeout
        while self.written < want and not self.failed and time.perf_counter() < end:
            time.sleep(0.005)
        return self.written >= want

    def stop(self, timeout: float = 60.0) -> dict:
        """Close the file and say what is in it. Safe to call twice."""
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout)
            self._thread = None
        return self.report()

    def report(self) -> dict:
        """What is in the file. `offered == frames + dropped` unless the writer failed."""
        out = {"file": str(self.path), "codec": self.codec,
               "mb": round(size_mb(self.path), 1), "frames": self.written,
               "offered": self.offered,
               "dropped": self.dropped, "seconds": round(self.seconds, 1)}
        if self.failed:
            out["error"] = f"{type(self.failed[0]).__name__}: {self.failed[0]}"
        return out

    def _encoder(self, av, tb: Fraction) -> str:
        """The first wanted encoder that actually opens at this size on this machine.

        Opened for real rather than looked up: `add_stream` only checks that the build knows
        the name, so an NVENC encoder on a machine without an NVIDIA card would be accepted
        and then fail at the first frame. `libx264` is last in `CODECS` and always opens, on
        the CPU, so `auto` records slowly rather than not at all."""
        wanted = CODECS if self.codec == "auto" else (self.codec,)
        for name in wanted:
            try:
                ctx = av.CodecContext.create(name, "w")
                ctx.width, ctx.height, ctx.pix_fmt = self.width, self.height, "nv12"
                ctx.time_base, ctx.framerate = tb, round(self.fps)
                ctx.open()
                return name
            except Exception:                                        # noqa: BLE001
                continue
        raise RuntimeError(f"no encoder opened, tried {', '.join(wanted)}")

    def _run(self) -> None:
        container = stream = None
        try:
            import av

            tb = Fraction(1, round(self.fps))
            self.codec = self._encoder(av, tb)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(self.path), "w")
            stream = container.add_stream(self.codec, rate=round(self.fps))
            stream.width, stream.height, stream.pix_fmt = self.width, self.height, "nv12"
            # Opened now rather than at the first frame, so `wait_open` covers all of it.
            stream.codec_context.open()
            self._opened.set()
            pool = [av.VideoFrame(self.width, self.height, "nv12") for _ in range(POOL)]
            views = [nv12_plane_views(f, self.height, self.width) for f in pool]
            while True:
                item = self._q.get()
                if item is None:
                    break
                pts, plane = item
                k = self.written % POOL
                frame, (luma, chroma) = pool[k], views[k]
                np.copyto(luma, plane[:self.height])
                np.copyto(chroma, plane[self.height:])
                frame.pts, frame.time_base = pts, tb
                for packet in stream.encode(frame):
                    container.mux(packet)
                self.written += 1
        except BaseException as exc:                # noqa: BLE001  reported, not swallowed
            self.failed.append(exc)
            self._opened.set()                      # a failure is an answer too
            while self._q.get() is not None:        # keep the producer from blocking for ever
                pass
        finally:
            if container is not None:
                try:
                    if stream is not None:
                        for packet in stream.encode():
                            container.mux(packet)
                except BaseException as exc:        # noqa: BLE001
                    self.failed.append(exc)
                container.close()
