"""The three kinds of generator file -- exported ONNX graph, converted StyleGAN2, this project's
FastGAN -- and how each is opened and made ready to play.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ganlive.checkpoints import is_onnx
from ganlive.device import playback_dtype
from ganlive.dials import derive, fastgan_dials, steer, table
from ganlive.dials.gate import directions_for
from ganlive.models import calibrate, fastgan, onnx_file
from ganlive.models import stylegan2 as S2
from ganlive.models.capture import compile_and_count
from ganlive.models.fold import prepare_for_inference
from ganlive.models.onnx import OnnxGenerator
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


def _compiled(net, nz: int, device, dtype, options: LoadOptions):
    """`(net, graphs, seconds)`: the net compiled if this load asks for it."""
    if not options.compile_net:
        return net, 0, 0.0
    return compile_and_count(net, nz, device, dtype)


def _onnx_layout(path):
    said = onnx_file.dials_of(path)
    # An export with no settings baked in is a plain graph with nothing to steer inside it.
    return (table.adopted(said["settings"], said["rests"], said["curves"], said["levels"])
            if said["curves"] else table.adopted((), (), (), ()))


def _prepare_onnx(path, device, dtype, options: LoadOptions) -> Prepared:
    """One exported graph made ready to play, with whatever dials it carries."""
    net = OnnxGenerator(path, device=device)
    print(net.report(), flush=True)
    found = directions_for(net, net.cfg.nz, device, dtype, floor=options.direction_floor,
                           path=path,
                           read=lambda _net, nz: derive.sefa_onnx(path, nz, count=derive.CANDIDATES))
    return Prepared(net=net, cfg=net.cfg, settings=net.settings, layout=_onnx_layout(path),
                    directions=found)


def _prepare_stylegan2(path, device, dtype, options: LoadOptions) -> Prepared:
    """A converted StyleGAN2 made ready to play, its dials swept rather than remembered."""
    net, settings, push, bands = open_stylegan2(path, device, exact=options.exact, dtype=dtype)
    cfg = net.cfg
    net, graphs, secs = _compiled(net, cfg.nz, device, dtype, options)

    # Directions before the layout: which style band a surviving direction belongs to, and
    # the strip labels them by band, is only known once `rank` has dropped the rest.
    found_dirs = directions_for(
        net, cfg.nz, device, dtype, into=push, floor=options.direction_floor, path=path,
        # `CANDIDATES` is per band.
        read=lambda _net, nz: derive.sefa_banded(bands, (derive.CANDIDATES,) * len(bands), nz))
    ranges = () if found_dirs is None or not found_dirs.ranges else tuple(
        (name, S2.BAND_PIXELS[name]) for name in found_dirs.ranges)

    # Not gated on `measure_grain`: this sweep is where a StyleGAN2's MODEL dials come from,
    # since a derived dial's curve is its measurement.
    found = calibrate.measured(calibrate.TorchProbe(net, settings, cfg.nz, device, dtype), settings.names,
                       size=(cfg.ladder.height, cfg.ladder.width))
    print(found.report(), flush=True)
    layout = table.stylegan2(found.names, [d.rest for d in found.dials],
                         [d.curve for d in found.dials],
                         [d.moved for d in found.dials], ranges)
    return Prepared(net=net, cfg=cfg, settings=settings, layout=layout, directions=found_dirs,
                    graphs=graphs, compile_s=secs, push=push)


def _prepare_fastgan(path, device, dtype, options: LoadOptions) -> Prepared:
    """This project's own generator made ready to play."""
    net, cfg = fastgan.load(path, device)
    fastgan.freeze_noise(net, seed=0)
    net = prepare_for_inference(net, cfg.nz, device, half=dtype is torch.float16)["net"]

    settings = steer.install(net, device, dtype)
    net, graphs, secs = _compiled(net, cfg.nz, device, dtype, options)
    gains = steer.calibrate_noise(net, settings, cfg.nz, device, dtype) if options.measure_grain else None
    return Prepared(net=net, cfg=cfg, settings=settings, layout=fastgan_dials.fastgan(noise_gains=gains),
                    directions=directions_for(net, cfg.nz, device, dtype, path=path,
                                              floor=options.direction_floor),
                    graphs=graphs, compile_s=secs)


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
    Family("onnx", is_onnx, onnx_file.config_of, _prepare_onnx, capturable=False),
    Family("stylegan2", is_stylegan2, S2.config_of, _prepare_stylegan2),
    Family("fastgan", lambda _path: True, fastgan.config_of, _prepare_fastgan),
)


def family_of(path) -> Family:
    """Which of `FAMILIES` this file belongs to. Never None: the last one takes anything."""
    return next(f for f in FAMILIES if f.owns(path))


def config_of(path):
    """A model's latent width and output size, whichever kind of file it is."""
    return family_of(path).config_of(path)

