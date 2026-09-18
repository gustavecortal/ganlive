"""Getting a finished picture from the generator to the screen."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from ganlive.models.fastgan import first_image
from ganlive.models.graph import to_bgra as _eager_bgra
from ganlive.models.graph import to_nv12
from ganlive.models.graph import to_rgb as _eager_rgb


class _Deferred:
    """The window inside which a stage's downloads do not each wait for the card."""

    __slots__ = ("_stage",)

    def __init__(self, stage: FrameStage) -> None:
        self._stage = stage

    def __enter__(self) -> FrameStage:
        self._stage._wait = False
        return self._stage

    def __exit__(self, *_exc) -> bool:
        stage = self._stage
        stage._wait = True
        if stage._pending:
            stage._pending = False
            stage.sync()
        return False


class FrameStage:
    """The downscale and the host conversions, with one ring per destination.

    **All of it on the stage's own queue, never the generator's.** An eager op on a captured
    generator's queue between two replays makes every later replay slower, without bound --
    see `models.graph.Replay`. So `step`, the first call after the generator,
    fences on its queue once and moves to this stage's; the conversions and the host copies
    follow it there, in order; and the frame's own wait for the card, `deferred` or
    `PinnedRing.take`, is a host wait that appends to neither queue."""

    HOST_COPY = "pinned"

    def __init__(self, height: int, width: int, to_yuv=None, to_rgb=None,
                 to_bgra=None, host_copy: str | None = None, device: str | None = None) -> None:
        from ganlive.device import detect_backend, streams

        self.height, self.width = int(height), int(width)
        self.device = device or detect_backend()
        self._streams = streams(self.device)
        self._side = self._streams.Stream() if self._streams is not None else None
        self.host_copy = host_copy or self.HOST_COPY
        self._rings: dict[str, object] = {}
        self.to_yuv = to_yuv or to_nv12
        self.to_rgb = to_rgb or _eager_rgb
        self.to_bgra = to_bgra or _eager_bgra
        self.compiled = {"yuv": to_yuv is not None, "rgb": to_rgb is not None,
                         "bgra": to_bgra is not None}
        self._wait = True
        self._pending = False

    def resize(self, height: int, width: int) -> None:
        """Produce frames at a different size from now on."""
        self.height, self.width = int(height), int(width)

    def eager(self) -> None:
        """Drop the compiled conversions for the plain ones."""
        self.to_yuv, self.to_rgb, self.to_bgra = to_nv12, _eager_rgb, _eager_bgra
        self.compiled = dict.fromkeys(self.compiled, False)

    def warm(self, staged: torch.Tensor) -> int:
        """Build the three conversions on a frame the real shape. Returns graphs built.

        Never fatal, on the same rule as `speedups.compile_and_count`: the conversions are the
        one compile that would otherwise fail at the first *frame*, with the window open, so a
        machine Inductor cannot build them on gets them in eager and a line saying so."""
        from ganlive.models.graph import warm

        try:
            return warm([self.to_yuv, self.to_rgb, self.to_bgra], staged)
        except Exception as exc:  # noqa: BLE001 -- see `compile_and_count`
            print(f"conversions: running eager -- {str(exc).splitlines()[0][:120]}", flush=True)
            self.eager()
            return 0

    def _take(self, name: str, tensor, depth: int = 3):
        """Get one converted frame to the host, through this destination's own ring."""
        key = (name, tuple(tensor.shape), tensor.dtype)
        ring = self._rings.get(key)
        if ring is None:
            from ganlive.models.graph import PinnedRing

            ring = self._rings[key] = PinnedRing(depth, self.device, self.host_copy)
        if self._wait:
            return ring.take(tensor)
        self._pending = True
        return ring.take(tensor, wait=False)

    def sync(self) -> None:
        """Wait for every copy this stage has started, on the device it started them on."""
        from ganlive.device import synchronize

        synchronize(self.device)

    def deferred(self) -> _Deferred:
        """Batch the downloads inside this block and wait for the card once, on exit."""
        return _Deferred(self)

    def pinned(self) -> dict[str, bool]:
        """Which destinations really got pinned staging. Reported per destination, not per
        shape: a bank of three sizes would otherwise print nine entries saying one thing."""
        out: dict[str, bool] = {}
        for (name, _shape, _dtype), ring in self._rings.items():
            out[name] = bool(ring.pinned) and out.get(name, True)
        return out

    def aside(self):
        """This stage's own queue, for the block inside. In order, so a conversion that follows
        `step` on it follows the frame; `step` is what fences on the generator's queue."""
        return contextlib.nullcontext() if self._side is None else self._streams.stream(self._side)

    def step(self, outs) -> torch.Tensor:
        """One frame in, one frame out, at the size being shown. The one call that crosses from
        the generator's queue to this stage's, so it is the one that waits."""
        if self._side is not None:
            # `wait_stream` records a fresh event each frame; a held `Event` re-recorded in its
            # place measured the same on the played loop, so the shorter spelling stays.
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
        """A stepped frame as height-by-width-by-four bytes, in the order a texture is."""
        with self.aside():
            return self._take("bgra", self.to_bgra(frame))
