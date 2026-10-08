"""Getting a finished frame out of the card and into whatever wants it.

Colour conversion, the pinned host buffers frames land in, and the 8-bit level: the one unit
every measurement in this project is quoted in.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ganlive.device import synchronize
from ganlive.levels import EXACT_LEVELS, FLOOR_LEVELS, LEVEL, RANDOM_FLOOR

__all__ = ["EXACT_LEVELS", "FLOOR_LEVELS", "LEVEL", "RANDOM_FLOOR"]

#: BT.709 luma coefficients.
_KR, _KB = 0.2126, 0.0722
_KG = 1 - _KR - _KB


def levels(a, b) -> float:
    """Mean difference between two frames in [-1, 1], in 8-bit levels.

    Takes torch tensors, which stay on their device, or numpy arrays. A torch frame is
    reduced in float32 and scaled after the reduction, so a full-resolution frame costs no
    extra full-size tensors."""
    if isinstance(a, np.ndarray):
        return float(np.abs(a - b).mean() * LEVEL)
    return (a - b).abs().mean(dtype=torch.float32).item() * LEVEL


def worst_levels(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest difference between two frames in [-1, 1], in 8-bit levels."""
    return float((a - b).abs().max()) * LEVEL


def _bytes(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> uint8, channels first."""
    return out.add(1.0).mul_(LEVEL).round_().clamp_(0, 255).to(torch.uint8)


def to_rgb(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the `(H, W, 3)` uint8 bytes a window blits, on the GPU."""
    x = _bytes(out)
    return x.permute(0, 2, 3, 1)[0].contiguous()


def to_bgra(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the `(H, W, 4)` uint8 bytes an SDL texture already is."""
    x = _bytes(out)
    b, g, r = x[:, 2], x[:, 1], x[:, 0]
    return torch.stack([b, g, r, torch.full_like(b, 255)], dim=-1)[0].contiguous()


def to_nv12(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the encoder's `(H*3/2, W)` uint8 plane stack, on the GPU.

    NV12: the Y plane, then the chroma interleaved as UVUVUV rows."""
    y, u, v = _yuv_planes(out)
    h, w = out.shape[-2:]
    chroma = torch.stack([u, v], dim=-1).reshape(-1)     # U,V,U,V... in row order
    return torch.cat([y.reshape(-1), chroma]).reshape(h * 3 // 2, w)


def compiled_conversions():
    """`(to_nv12, to_rgb, to_bgra)` through TorchInductor. Functions of a frame, not of a net."""
    return tuple(torch.compile(f, dynamic=False) for f in (to_nv12, to_rgb, to_bgra))


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
    """A host buffer the card reads directly, or a plain one where pinning is refused."""
    try:
        return torch.zeros(shape, dtype=dtype, pin_memory=True)
    except (RuntimeError, NotImplementedError):
        return torch.zeros(shape, dtype=dtype)


class PinnedRing:
    """A ring of pinned host buffers to copy device frames into, instead of `Tensor.cpu()`.

    A frame taken from it stays valid until the ring comes back round to its buffer."""

    def __init__(self, depth: int, device: str):
        self.depth, self._device = max(2, depth), device
        self._buf: list = []
        self._key = None
        self._n = 0
        self.pinned = False

    def _alloc(self, src) -> None:
        self._buf = [pinned(src.shape, src.dtype) for _ in range(self.depth)]
        self.pinned = all(b.is_pinned() for b in self._buf)
        self._key = (tuple(src.shape), src.dtype)

    def take(self, src, wait: bool = True):
        """Copy `src` to the host and return a numpy view of the buffer it landed in."""
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
        """Wait for everything queued on this ring's device, copies included."""
        synchronize(self._device)
