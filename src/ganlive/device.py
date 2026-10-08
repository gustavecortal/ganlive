"""Which PyTorch accelerator a conversion runs on, and the few per-backend calls it needs.
Playing needs none of it: the engine chooses its own wgpu backend.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

Backend = Literal["xpu", "cuda", "mps", "cpu"]


def detect_backend() -> Backend:
    """Pick the best available accelerator. ``GANLIVE_DEVICE`` overrides."""
    forced = os.environ.get("GANLIVE_DEVICE", "").strip().lower()
    if forced:
        if forced not in ("xpu", "cuda", "mps", "cpu"):
            raise ValueError(f"GANLIVE_DEVICE={forced!r} is not one of xpu/cuda/mps/cpu")
        return forced  # type: ignore[return-value]

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _mod(name: str | None = None):
    """torch submodule for a named device (xpu, cuda, mps), or None for the cpu. `mps` has
    `synchronize` like the other two; the functions below `hasattr` for what it lacks."""
    b = (name or detect_backend()).split(":")[0]
    if b in ("xpu", "cuda", "mps"):
        return getattr(torch, b, None)
    return None


def playback_dtype(device: str | torch.device) -> torch.dtype:
    """The precision ganlive plays a model in when the caller did not say.

    Half on every accelerator; single on the CPU, where half-precision convolution is
    emulated and runs slower than float32."""
    return torch.float32 if str(device).split(":")[0] == "cpu" else torch.float16


def synchronize(name: str | None = None) -> None:
    """Wait for the device to finish. Pass the device a copy was issued to; see `_mod`."""
    m = _mod(name)
    if m is not None:
        m.synchronize()
