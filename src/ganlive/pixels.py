"""Getting a finished frame out of the card and into whatever wants it.

Colour conversion, the pinned host staging the frames land in, and the one unit every
measurement in this project is quoted in. None of it is about models, which is why it is not
under `models/` -- it was, and `frame.py`, `record/video.py`, `walk.py` and `dials/steer.py`
all reached into a *model* module to get at it, two of them for a private name.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_KR, _KB = 0.2126, 0.0722


_KG = 1 - _KR - _KB


#: What "exact, not approximate" is allowed to mean for a rewrite that should be bit-identical.
#: The capture measured 0.0000 on both families; loose enough to survive a driver that
#: reassociates something, tight enough that a still picture -- 77 levels -- cannot pass.
EXACT_LEVELS = 0.5


def levels(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean difference between two frames in [-1, 1], in the 8-bit levels this repo judges by.

    `adopt.levels` is the same quantity for numpy arrays and carries the note about the unit;
    this one stays in torch and on the card. Scaling **after** the reduction rather than before
    it, and accumulating in float32, keeps a full-resolution frame from costing three more
    tensors of its own size -- 226 MB at 3072x2048 -- to answer one scalar question. The
    subtraction of two near-identical halves is exact, so nothing is lost by it."""
    return (a - b).abs().mean(dtype=torch.float32).item() * 127.5


def to_rgb(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the `(H, W, 3)` uint8 bytes a window blits, on the GPU."""
    x = out.add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1)[0].contiguous()


def compiled_to_rgb():
    """`to_rgb` through TorchInductor: 1.335 ms eager against 0.522 at 1536x1024."""
    return torch.compile(to_rgb, dynamic=False)


def to_bgra(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the `(H, W, 4)` uint8 bytes an SDL texture already is."""
    x = out.add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    b, g, r = x[:, 2], x[:, 1], x[:, 0]
    return torch.stack([b, g, r, torch.full_like(b, 255)], dim=-1)[0].contiguous()


def compiled_to_bgra():
    """`to_bgra` through TorchInductor. Same `dynamic=False` argument as the other two."""
    return torch.compile(to_bgra, dynamic=False)


def to_nv12(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the encoder's `(H*3/2, W)` uint8 plane stack, on the GPU.

    NV12: the Y plane, then the chroma interleaved as UVUVUV rows."""
    y, u, v = _yuv_planes(out)
    h, w = out.shape[-2:]
    chroma = torch.stack([u, v], dim=-1).reshape(-1)     # U,V,U,V... in row order
    return torch.cat([y.reshape(-1), chroma]).reshape(h * 3 // 2, w)


def compiled_to_nv12():
    """`to_nv12` through TorchInductor, which is a 5.8x on that function alone."""
    return torch.compile(to_nv12, dynamic=False)


def _yuv_planes(out: torch.Tensor):
    """The shared arithmetic: `(y, u, v)` as uint8, chroma at half resolution."""
    o = out.float().clamp(-1, 1)
    r, g, b = o[:, 0], o[:, 1], o[:, 2]
    luma = _KR * r + _KG * g + _KB * b
    y = (luma * 109.5 + 125.5).round().clamp(0, 255).to(torch.uint8)   # 219/2, 219/2 + 16

    pooled = F.avg_pool2d(o, 2)                      # one call for all three planes
    pr, pg, pb = pooled[:, 0], pooled[:, 1], pooled[:, 2]
    pluma = _KR * pr + _KG * pg + _KB * pb
    u = ((pb - pluma) * (224 / (4 * (1 - _KB))) + 128).round().clamp(0, 255).to(torch.uint8)
    v = ((pr - pluma) * (224 / (4 * (1 - _KR))) + 128).round().clamp(0, 255).to(torch.uint8)
    return y, u, v


def nv12_plane_views(frame, height: int, width: int):
    """numpy views onto an nv12 frame's own buffers, honouring each plane's line size."""
    return [np.frombuffer(p, dtype=np.uint8).reshape(rows, p.line_size)[:, :width]
            for p, rows in ((frame.planes[0], height), (frame.planes[1], height // 2))]


def pinned(shape, dtype) -> torch.Tensor:
    """A host buffer the card reads directly, or a plain one where pinning is refused -- the
    recording then fails on the pageable copy and says so there, rather than here."""
    try:
        return torch.zeros(shape, dtype=dtype, pin_memory=True)
    except (RuntimeError, NotImplementedError):
        return torch.zeros(shape, dtype=dtype)


class PinnedRing:
    """A ring of pinned host buffers to copy device frames into, instead of `Tensor.cpu()`."""

    def __init__(self, depth: int, device: str, mode: str = "pinned"):
        self.depth, self._device, self._mode = max(2, depth), device, mode
        self._buf: list = []
        self._key = None
        self._n = 0
        self.pinned = False

    def _alloc(self, src) -> None:
        if self._mode == "pinned":
            self._buf = [pinned(src.shape, src.dtype) for _ in range(self.depth)]
        else:
            self._buf = [torch.empty(src.shape, dtype=src.dtype) for _ in range(self.depth)]
        self.pinned = all(b.is_pinned() for b in self._buf)
        self._key = (tuple(src.shape), src.dtype)

    def take(self, src, wait: bool = True):
        """Copy `src` to the host and return a numpy view of the buffer it landed in."""
        if self._mode == "cpu":
            return src.cpu().numpy()            # the original path, kept as the baseline arm
        if self._key is None:
            self._alloc(src)
        if (tuple(src.shape), src.dtype) != self._key:
            return src.cpu().numpy()            # odd shape, e.g. a short final batch
        dst = self._buf[self._n % len(self._buf)]
        self._n += 1
        dst.copy_(src, non_blocking=True)
        if wait:
            self.sync()
        return dst.numpy()

    def sync(self) -> None:
        """Wait for everything queued on **this ring's** device, copies included."""
        from ganlive.device import synchronize

        synchronize(self._device)
