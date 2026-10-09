"""The kinds of generator file -- engine model, converted StyleGAN2, this project's FastGAN --
and what each says about itself before it is loaded.

Every model plays on the engine (`engine.load.open_model`). A checkpoint is described by its
conversion when it has a current one, so it is opened with PyTorch only while it has none.
"""
from __future__ import annotations

from dataclasses import dataclass

from ganlive.checkpoints import is_engine
from ganlive.engine.load import conversion_of, read_manifest
from ganlive.engine.player import EngineConfig
from ganlive.levels import RANDOM_FLOOR


@dataclass(frozen=True)
class LoadOptions:
    """Everything that decides how a checkpoint becomes a playable model.

    Carried on the `Bank`, so a model the shelf loads mid-session is prepared exactly like
    the ones loaded at launch."""

    #: Measure each model's grain gains at conversion rather than using a stock table.
    measure_grain: bool = True
    #: How many times a random direction's effect a derived direction must beat to earn a
    #: dial. See `levels.RANDOM_FLOOR`.
    direction_floor: float = RANDOM_FLOOR
    #: The wgpu backend to play on ("vulkan", "d3d12", "metal"), or None for the fastest here.
    backend: str | None = None


def _described(path) -> dict | None:
    """The manifest that describes `path`: an engine model's own, or a checkpoint's current
    conversion's. None for a checkpoint that has none."""
    if is_engine(path):
        return read_manifest(path)
    found = conversion_of(path)
    return None if found is None else found[1]


def family_of(path) -> str:
    """"engine", "stylegan2" or "fastgan". A StyleGAN2 shares the `.pt` suffix with FastGAN, so
    its conversion says, or the file's own tag."""
    if is_engine(path):
        return "engine"
    known = _described(path)
    if known is not None:
        return known.get("family", "fastgan")
    from ganlive.models import stylegan2 as S2  # PyTorch: an unconverted checkpoint

    return "stylegan2" if S2.is_stylegan2(path) else "fastgan"


def is_stylegan2(path) -> bool:
    return family_of(path) == "stylegan2"


def config_of(path):
    """A model's latent width and output size, without loading it."""
    known = _described(path)
    if known is not None:
        return EngineConfig.of(known)
    from ganlive.models import fastgan  # PyTorch: an unconverted checkpoint
    from ganlive.models import stylegan2 as S2

    return S2.config_of(path) if S2.is_stylegan2(path) else fastgan.config_of(path)
