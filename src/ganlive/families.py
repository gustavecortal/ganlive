"""The kinds of generator file -- engine model, exported ONNX graph, converted StyleGAN2, this
project's FastGAN -- and how each is opened and made ready to play.

A FastGAN plays on the engine: its checkpoint is converted, at its first load, into the engine
model beside it (`lichen.pt` -> `lichen.engine/`), which later loads read directly.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ganlive.checkpoints import MANIFEST, is_engine, is_onnx
from ganlive.device import playback_dtype
from ganlive.dials import steer
from ganlive.engine.load import open_model
from ganlive.engine.player import EngineConfig
from ganlive.models import fastgan, onnx_file
from ganlive.models import stylegan2 as S2
from ganlive.pixels import RANDOM_FLOOR


@dataclass(frozen=True)
class LoadOptions:
    """Everything that decides how a checkpoint becomes a playable model.

    Carried on the `Bank`, so a model the shelf loads mid-session is prepared exactly like
    the ones loaded at launch."""

    compile_net: bool = True
    #: Record the compiled forward as one device graph and replay it, instead of launching a
    #: hundred-odd kernels a frame from Python. Exact, and worth ~1-3 ms a frame.
    capture: bool = True
    #: Measure each model's grain gains at load rather than using a stock table. The dial
    #: sweep and the dial gate run either way.
    measure_grain: bool = True
    #: Run a converted StyleGAN2 in the precision its own file declares.
    exact: bool = False
    #: How many times a random direction's effect a derived direction must beat to earn a
    #: dial. See `pixels.RANDOM_FLOOR`.
    direction_floor: float = RANDOM_FLOOR


@dataclass
class Prepared:
    """What a family hands back: one generator made ready, before the bank captures and gates it."""

    net: object
    cfg: object
    settings: object
    #: The dials this model offers.
    layout: object
    #: The measured `dials.derive.Directions`, or None.
    directions: object = None
    graphs: int = 0
    compile_s: float = 0.0
    #: The `(bands, w_dim)` tensor a StyleGAN2 takes its `w` push in; None for families that
    #: steer `z`.
    push: object = None


def is_stylegan2(path) -> bool:
    """Whether this is a converted StyleGAN2. It shares the `.pt` suffix with FastGAN."""
    return not is_onnx(path) and S2.is_stylegan2(path)


def open_stylegan2(path, device, exact: bool = False, dtype=None):
    """A converted StyleGAN2 with its dials installed: `(net, settings, push, bands)`.

    Shared with `ganlive dials`, so the basis it derives is for the model exactly as it plays.
    `dtype` is the session's precision, the device's by default; see `S2.half_from_for`."""
    net = S2.from_file(path, device,
                       half_from=S2.half_from_for(dtype or playback_dtype(device), exact))
    settings = steer.install_stylegan2(net, device)
    # Read off the module tree now; the compile that follows hides it.
    return net, settings, net.mapping.push, S2.style_bands(net)


def engine_config_of(path) -> EngineConfig:
    return EngineConfig.of(json.loads((Path(path) / MANIFEST).read_text(encoding="utf-8")))


def _prepare_engine(path, device, dtype, options: LoadOptions) -> Prepared:
    """A model on the engine (see `engine.load`): on `device`, the wgpu device the bank plays
    on, or else on this machine's fastest backend. `dtype` is unused, since the engine is fp16."""
    grain = options.measure_grain if family_of(path).name == "fastgan" else True
    net, layout, directions = open_model(path, device, options.direction_floor, grain)
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
    #: `(path, device, dtype, LoadOptions) -> Prepared`.
    prepare: Callable
    #: Whether its forward can be recorded as one device graph. An ONNX graph runs under its
    #: own runtime, so a recording of the torch stream would hold none of its work.
    capturable: bool = True


#: Order matters: the suffix is decisive, then the file's own format tag, then what is left.
FAMILIES = (
    Family("engine", is_engine, engine_config_of, _prepare_engine, capturable=False),
    Family("onnx", is_onnx, onnx_file.config_of, _prepare_engine, capturable=False),
    Family("stylegan2", is_stylegan2, S2.config_of, _prepare_engine, capturable=False),
    Family("fastgan", lambda _path: True, fastgan.config_of, _prepare_engine, capturable=False),
)

#: The families that play on the engine, which is all a bank plays on.
ENGINE_FAMILIES = ("engine", "stylegan2", "fastgan")


def family_of(path) -> Family:
    """Which of `FAMILIES` this file belongs to. Never None: the last one takes anything."""
    return next(f for f in FAMILIES if f.owns(path))


def config_of(path):
    """A model's latent width and output size, whichever kind of file it is."""
    return family_of(path).config_of(path)

