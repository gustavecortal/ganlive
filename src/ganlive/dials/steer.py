"""Installing live dials on a prepared generator: the settings a drum machine can steer.

`install` grafts FastGAN's steerable modules in; `install_stylegan2` hands a StyleGAN2 its
truncation, noise gains and push buffer. Both return the `Settings` vector that drives them, and
both must run before `torch.compile`, or every dial is silently inert.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ganlive.dials.fastgan_dials import NOISE_BANDS, SETTINGS_WRITTEN
from ganlive.models.common import first_image, latent
from ganlive.models.fastgan import SkipLayerExcitation
from ganlive.models.fold import FoldedNoise
from ganlive.models.steerable import SteerableNoise, SteerableSLE
from ganlive.models.stylegan2 import BANDS, noise_sites
from ganlive.pixels import levels
from ganlive.settings import Settings


def _available(net: nn.Module) -> list[str]:
    """The settings this FastGAN has: each of `SETTINGS_WRITTEN` whose module it carries.
    A 256-pixel FastGAN has no 512-pixel stage, and is a supported size."""
    return [s for s in SETTINGS_WRITTEN
            if getattr(net, s.split(".", 1)[1], None) is not None]


def install(net: nn.Module, device, dtype=torch.float16) -> Settings:
    """Swap the steerable modules into a folded FastGAN and return the vector that drives them.

    Run it before `torch.compile`: installed after, every dial is silently inert."""
    settings = Settings(sorted(_available(net)), device, dtype)
    sites = {"noise": 0, "sle": 0}

    for parent_name, parent in list(net.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, FoldedNoise):
                stage = (parent_name or child_name).split(".")[0]
                index = settings.index.get(f"noise.{stage}")
                if index is None:
                    continue
                setattr(parent, child_name,
                        SteerableNoise(child.coeff, child.noise, settings, index))
                sites["noise"] += 1
            elif isinstance(child, SkipLayerExcitation):
                index = settings.index.get(f"sle.{child_name}")
                if index is None:
                    continue
                setattr(parent, child_name, SteerableSLE(child.gate, settings, index))
                sites["sle"] += 1

    if settings.names and not (sites["noise"] and sites["sle"]):
        raise RuntimeError(
            f"steering install matched nothing it needed: {sites}. The net must be folded "
            f"by `prepare_for_inference` before installing, or the noise gain is still "
            f"inside `NoiseInjection` and no dial will do anything.")
    return settings


def install_stylegan2(net, device) -> Settings:
    """Give a converted StyleGAN2 its dials: a truncation per style range, a noise gain per
    resolution, and the push buffer the direction dials write."""
    sites = noise_sites(net)
    styles = [name for name, _lo, _hi in BANDS]
    settings = Settings(styles + list(sites), device, torch.float32)
    # Truncation first in the vector, and so first on the strip.
    net.mapping.truncation = settings.span(styles)
    # The push buffer is allocated before `torch.compile` too, or the graph closes over
    # `None`. It is not part of the `Settings` vector: an offset whose neutral is 0, written
    # by the walk.
    net.mapping.push = torch.zeros(len(BANDS), net.cfg.w_dim, device=device,
                                   dtype=torch.float32)
    for name, layers in sites.items():
        view = settings.view(name)
        for layer in layers:
            layer.noise_gain = view
    return settings


@torch.no_grad()
def calibrate_noise(net: nn.Module, settings: Settings, nz: int, device,
                    dtype=torch.float16, seed: int = 0, probes: int = 7) -> dict[str, float]:
    """For this FastGAN, the gain each noise band needs to move the picture by its target in
    `NOISE_BANDS`, found by bisection in log space."""
    z = latent(nz, seed, device, dtype)
    settings.reset()
    base = first_image(net(z))

    out: dict[str, float] = {}
    for name, target, _start in NOISE_BANDS:
        if name not in settings.index:
            continue          # `noise_for` falls back to its default gain for this band
        lo, hi = math.log(0.5), math.log(3000.0)
        for _ in range(probes):
            mid = 0.5 * (lo + hi)
            settings.reset()
            settings.set(name, math.exp(mid))
            settings.commit()
            if levels(first_image(net(z)), base) < target:
                lo = mid
            else:
                hi = mid
        out[name] = round(math.exp(0.5 * (lo + hi)), 3)
    settings.reset()
    return out
