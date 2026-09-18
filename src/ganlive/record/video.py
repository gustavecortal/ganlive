"""Getting finished frames into a file: one writer thread, and where the files go."""
from __future__ import annotations

import queue
import threading
import time
from fractions import Fraction
from pathlib import Path

#: Hardware encoders, in the order they are tried. The first that opens on this machine
#: wins; `libx264` always does, on the CPU, and costs most of the frame rate at 4K.
CODECS = ("av1_qsv", "hevc_qsv", "h264_nvenc", "hevc_nvenc", "hevc_videotoolbox",
          "h264_amf", "libx264")
DEFAULT_CODEC = "auto"

POOL = 4


def next_path(folder: Path, stem: str, suffix: str) -> Path:
    """`folder/stem-01.suffix`, at the lowest number not already taken."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # One directory read, not one stat per candidate. This is called from the frame
    # loop, where a session that has saved 200 stills paid 200 sequential syscalls
    # inside a single frame.
    taken = {p.stem.rsplit("-", 1)[-1] for p in folder.glob(f"{stem}-*{suffix}")}
    used = {int(t) for t in taken if t.isdigit()}
    n = next((i for i in range(1, 1000) if i not in used), None)
    if n is None:
        raise FileExistsError(f"a thousand {stem} files in {folder}")
    return folder / f"{stem}-{n:02d}{suffix}"


def save_still(path: Path, rgb) -> threading.Thread:
    """Write one RGB frame as a PNG, on its own thread, and hand the thread back."""
    def write():
        from PIL import Image

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)

    thread = threading.Thread(target=write, name=f"still {Path(path).name}", daemon=True)
    thread.start()
    return thread


class Recorder:
    """A container, a bounded queue and the one thread that muxes into it."""

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


    def start(self) -> Recorder:
        self._thread = threading.Thread(target=self._run, name=f"encode {self.path.name}",
                                        daemon=True)
        self._thread.start()
        return self

    @property
    def seconds(self) -> float:
        """How long this take has been running. 0 until the first frame is offered."""
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    @property
    def wants(self) -> bool:
        """Whether a frame offered right now would actually be written."""
        return not self._q.full()

    def skip(self) -> None:
        """Count a frame the caller did not bother to fetch, so the report is still true."""
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
        mb = self.path.stat().st_size / 1e6 if self.path.exists() else 0.0
        out = {"file": str(self.path), "codec": self.codec,
               "mb": round(mb, 1), "frames": self.written,
               "offered": self.offered,
               "dropped": self.dropped, "seconds": round(self.seconds, 1)}
        if self.failed:
            out["error"] = f"{type(self.failed[0]).__name__}: {self.failed[0]}"
        return out


    def _run(self) -> None:
        import av
        import numpy as np

        from ganlive.models.graph import nv12_plane_views

        self.path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(self.path), "w")
        stream = None
        # `auto` tries the hardware encoders in turn and keeps the first this machine has.
        # `libx264` is last and always opens, on the CPU, which at 4K costs most of the frame
        # rate -- so a machine with no hardware encoder records, slowly, rather than failing.
        wanted = CODECS if self.codec == "auto" else (self.codec,)
        for name in wanted:
            try:
                stream = container.add_stream(name, rate=round(self.fps))
                self.codec = name
                break
            except Exception:                                        # noqa: BLE001
                continue
        if stream is None:
            raise RuntimeError(f"no encoder opened, tried {', '.join(wanted)}")
        stream.width, stream.height, stream.pix_fmt = self.width, self.height, "nv12"
        tb = Fraction(1, round(self.fps))
        pool = [av.VideoFrame(self.width, self.height, "nv12") for _ in range(POOL)]
        views = [nv12_plane_views(f, self.height, self.width) for f in pool]
        try:
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
            while self._q.get() is not None:        # keep the producer from blocking for ever
                pass
        finally:
            try:
                for packet in stream.encode():
                    container.mux(packet)
            except BaseException as exc:            # noqa: BLE001
                self.failed.append(exc)
            container.close()
