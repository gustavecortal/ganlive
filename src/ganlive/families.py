"""The kinds of generator file -- engine model, converted StyleGAN2, this project's FastGAN --
and how each is opened and made ready to play.

Every model plays on the engine. A checkpoint is converted, at its first load, into the engine
model beside it (`lichen.pt` -> `lichen.engine/`), which later loads read directly, so a
checkpoint is opened with PyTorch only while it has no conversion beside it.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ganlive.checkpoints import ENGINE_SUFFIX, MANIFEST, is_engine
from ganlive.engine.load import open_model
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


@dataclass
class Prepared:
    """What a family hands back: one model ready to play."""

    net: object
    cfg: object
    settings: object
    #: The dials this model offers.
    layout: object
    #: The measured directions, or None.
    directions: object = None
    #: The `(bands, w_dim)` array a StyleGAN2 takes its `w` push in, None for one steered by `z`.
    push: object = None


def _manifest(folder: Path) -> dict | None:
    """The manifest of the engine model in `folder`, or None if there is none."""
    try:
        return json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _beside(path) -> dict | None:
    """The manifest of a checkpoint's conversion beside it, if it has one."""
    return _manifest(Path(path).with_suffix(ENGINE_SUFFIX))


def is_stylegan2(path) -> bool:
    """Whether this is a converted StyleGAN2. It shares the `.pt` suffix with FastGAN, so the
    file's own tag says, or its conversion's manifest."""
    known = _beside(path)
    if known is not None:
        return known.get("family") == "stylegan2"
    from ganlive.models import stylegan2 as S2  # PyTorch: an unconverted checkpoint

    return S2.is_stylegan2(path)


def engine_config_of(path) -> EngineConfig:
    return EngineConfig.of(_manifest(Path(path)) or {})


def _checkpoint_config(read: Callable) -> Callable:
    """A checkpoint's config off its conversion when it has one, else off the file."""
    def config_of(path):
        known = _beside(path)
        return EngineConfig.of(known) if known is not None else read(path)
    return config_of


def _fastgan_config(path):
    from ganlive.models import fastgan  # PyTorch: an unconverted checkpoint

    return fastgan.config_of(path)


def _stylegan2_config(path):
    from ganlive.models import stylegan2 as S2  # PyTorch: an unconverted checkpoint

    return S2.config_of(path)


def _prepare_engine(path, device, options: LoadOptions) -> Prepared:
    """A model on the engine (see `engine.load`): on `device`, the wgpu device the bank plays
    on, or else on this machine's fastest backend (or `options.backend`)."""
    grain = options.measure_grain if family_of(path).name == "fastgan" else True
    net, layout, directions = open_model(path, device, options.direction_floor, grain,
                                         backend=options.backend)
    return Prepared(net=net, cfg=net.cfg, settings=net.settings, layout=layout,
                    directions=directions, push=net.push)


@dataclass(frozen=True)
class Family:
    """One kind of generator file, and everything the bank asks about it."""

    name: str
    #: Is this file mine? Asked in `FAMILIES` order, so the last may simply say yes.
    owns: Callable
    #: `(path) -> config`: latent width and output size, without building the generator.
    config_of: Callable
    #: `(path, device, LoadOptions) -> Prepared`.
    prepare: Callable


#: Order matters: an engine folder first, then the file's own format tag, then what is left.
FAMILIES = (
    Family("engine", is_engine, engine_config_of, _prepare_engine),
    Family("stylegan2", is_stylegan2, _checkpoint_config(_stylegan2_config), _prepare_engine),
    Family("fastgan", lambda _path: True, _checkpoint_config(_fastgan_config), _prepare_engine),
)


def family_of(path) -> Family:
    """Which of `FAMILIES` this file belongs to. Never None: the last one takes anything."""
    return next(f for f in FAMILIES if f.owns(path))


def config_of(path):
    """A model's latent width and output size, whichever kind of file it is."""
    return family_of(path).config_of(path)
