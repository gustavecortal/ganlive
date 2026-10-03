"""After the generator: resizing a frame and converting it to the bytes a window, an encoder or
a still wants, on a device queue of its own."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from ganlive.device import detect_backend, streams, synchronize
from ganlive.models.common import first_image
from ganlive.pixels import PinnedRing, to_nv12
from ganlive.pixels import to_bgra as _eager_bgra
from ganlive.pixels import to_rgb as _eager_rgb


class _Deferred:
    """A block inside which a stage's downloads do not each wait for the card.

    On exit they are waited for together -- or, for a `handoff`, not at all: `ticket` then
    holds what to wait on, and the caller waits when it is about to read the bytes."""

    def __init__(self, stage: FrameStage, keep: bool = False) -> None:
        self._stage, self._keep = stage, keep
        self.ticket = Ticket(None)

    def __enter__(self) -> _Deferred:
        self._stage._wait = False
        return self

    def __exit__(self, *_exc) -> bool:
        stage = self._stage
        stage._wait = True
        if stage._pending:
            stage._pending = False
            if self._keep and stage._side is not None:
                event = stage._streams.Event()
                event.record(stage._side)
                self.ticket = Ticket(event)
            else:
                stage.sync()
        return False


class Ticket:
    """The downloads of one frame, started and not yet waited for. `wait` before reading them.

    An event on the stage's own queue, so waiting on it waits for this frame's copies and
    nothing else -- not for the generator, which by then is drawing the next frame."""

    def __init__(self, event) -> None:
        self._event = event

    def wait(self) -> None:
        if self._event is not None:
            self._event.synchronize()
            self._event = None


class FrameStage:
    """The resize and the host conversions, with one pinned ring of host buffers per destination.

    All of it runs on the stage's own device queue, never the generator's: an eager op on a
    captured generator's queue between two replays slows every later replay (see
    `models.capture.Replay`). `step` waits on the generator's queue once and moves to this one;
    the conversions and host copies follow in order; and waiting for a frame's bytes is a host
    wait that adds work to neither queue."""

    def __init__(self, height: int, width: int, to_yuv=None, to_rgb=None,
                 to_bgra=None, device: str | None = None) -> None:
        self.height, self.width = int(height), int(width)
        self.device = device or detect_backend()
        self._streams = streams(self.device)
        self._side = self._streams.Stream() if self._streams is not None else None
        self._rings: dict[str, object] = {}
        self.to_yuv = to_yuv or to_nv12
        self.to_rgb = to_rgb or _eager_rgb
        self.to_bgra = to_bgra or _eager_bgra
        self.compiled = {"yuv": to_yuv is not None, "rgb": to_rgb is not None,
                         "bgra": to_bgra is not None}
        self._wait = True
        self._pending = False
        #: Recorded on this stage's queue after the last conversion, before its copy: once it
        #: has passed, the generator's frame has been read and the generator may write the
        #: next one into the same buffer -- see `release`.
        self._read = None

    def resize(self, height: int, width: int) -> None:
        """Produce frames at a different size from now on."""
        self.height, self.width = int(height), int(width)

    def eager(self) -> None:
        """Drop the compiled conversions for the plain ones."""
        self.to_yuv, self.to_rgb, self.to_bgra = to_nv12, _eager_rgb, _eager_bgra
        self.compiled = dict.fromkeys(self.compiled, False)

    def warm(self, staged: torch.Tensor) -> int:
        """Compile the three conversions now, on a frame of the real shape. Returns graphs built.

        Never fatal: a machine that cannot compile them gets them eager, and a line saying so,
        rather than a failure at the first frame with the window open."""
        from torch._dynamo.utils import counters

        try:
            before = counters["frames"]["ok"]
            with torch.no_grad():
                for fn in (self.to_yuv, self.to_rgb, self.to_bgra):
                    fn(staged)
            return counters["frames"]["ok"] - before
        except Exception as exc:  # noqa: BLE001 -- see `capture.compile_and_count`
            print(f"conversions: running eager -- {str(exc).splitlines()[0][:120]}", flush=True)
            self.eager()
            return 0

    def _take(self, name: str, tensor, depth: int = 3):
        """Get one converted frame to the host, through this destination's own ring."""
        key = (name, tuple(tensor.shape), tensor.dtype)
        ring = self._rings.get(key)
        if ring is None:
            ring = self._rings[key] = PinnedRing(depth, self.device)
        if self._side is not None:
            self._read = self._streams.Event()
            self._read.record(self._side)
        if self._wait:
            return ring.take(tensor)
        self._pending = True
        return ring.take(tensor, wait=False)

    def sync(self) -> None:
        """Wait for every copy this stage has started, on the device it started them on."""
        synchronize(self.device)

    def deferred(self) -> _Deferred:
        """Batch the downloads inside this block and wait for the card once, on exit."""
        return _Deferred(self)

    def handoff(self) -> _Deferred:
        """Start the downloads inside this block and do not wait for them: the block's `ticket`
        does, when the bytes are about to be read. Where there is no second queue to record
        on, this is `deferred` and the ticket is already done.

        **Call `release` before the generator runs again.** A captured generator writes every
        frame into the same buffer, and these conversions read it."""
        return _Deferred(self, keep=True)

    def release(self) -> None:
        """Wait until the last frame converted has been read out of the generator's buffer.

        A host wait on this stage's own queue: it passes once the conversions are done, while
        the copies behind them are still running, and it issues nothing on the generator's."""
        if self._read is not None:
            self._read.synchronize()
            self._read = None

    def pinned(self) -> dict[str, bool]:
        """Which destinations really got pinned host memory, one entry per destination."""
        out: dict[str, bool] = {}
        for (name, _shape, _dtype), ring in self._rings.items():
            out[name] = bool(ring.pinned) and out.get(name, True)
        return out

    def aside(self):
        """Run the block inside on this stage's own queue, after the frame `step` handed over."""
        return contextlib.nullcontext() if self._side is None else self._streams.stream(self._side)

    def step(self, outs) -> torch.Tensor:
        """One frame in, one frame out, at the size being shown. The one call that crosses from
        the generator's queue to this stage's, so it is the one that waits."""
        if self._side is not None:
            self._side.wait_stream(self._streams.current_stream())
        with self.aside():
            o = first_image(outs)
            h, w = int(o.shape[-2]), int(o.shape[-1])
            if (h, w) == (self.height, self.width):
                return o
            if self.height * self.width <= h * w:
                return F.interpolate(o, size=(self.height, self.width), mode="area")
            return F.interpolate(o, size=(self.height, self.width), mode="bilinear",
                                 align_corners=False)

    def nv12_bytes(self, frame: torch.Tensor, dest: str = "yuv", depth: int = 3):
        """A stepped frame as the `(H*3/2, W)` uint8 plane stack an encoder wants."""
        with self.aside():
            return self._take(dest, self.to_yuv(frame), depth)

    def rgb_bytes(self, frame: torch.Tensor):
        """A stepped frame as height-by-width-by-three bytes, for a window to blit."""
        with self.aside():
            return self._take("rgb", self.to_rgb(frame))

    def rgb_still(self, frame: torch.Tensor):
        """A stepped frame as its own host array, deliberately outside the staging rings."""
        with self.aside():
            return self.to_rgb(frame).cpu().numpy()

    def bgra_bytes(self, frame: torch.Tensor):
        """A stepped frame as height-by-width-by-four bytes, in the order a texture is.

        Four buffers rather than the default three, because one more may be in use at once: one
        downloading behind the next frame (see `handoff`), one published, and one the window
        thread may still be uploading from."""
        with self.aside():
            return self._take("bgra", self.to_bgra(frame), depth=4)
