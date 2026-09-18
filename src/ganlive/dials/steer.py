"""Live handles on a prepared generator: the parameters a drum machine can steer."""
from __future__ import annotations

import math

import torch
from torch import nn

from ganlive.models.common import latent
from ganlive.models.steerable import SteerableNoise, SteerableSLE
from ganlive.pixels import levels
from ganlive.settings import Knobs

NOISE_RUNGS = ("feat_8", "feat_32", "feat_128", "feat_512", "feat_2048")


SLE_GATES = ("se_64", "se_128", "se_256", "se_512")


def available(net: nn.Module) -> set[str]:
    """Which settings this architecture actually has, by the parameters it carries."""
    have = {f"sle.{g}" for g in SLE_GATES if getattr(net, g, None) is not None}
    have |= {f"noise.{r}" for r in NOISE_RUNGS if getattr(net, r, None) is not None}
    return have


def install(net: nn.Module, device, dtype=torch.float16, wanted=None) -> Knobs:
    """Swap the steerable modules in and hand back the vector that drives them.

    **INVARIANT: this must run BEFORE `torch.compile`.** Installed after, every knob is
    silently inert -- no error, no effect. `bank._prepare_fastgan` does the two in order."""
    from ganlive.dials.fastgan_dials import SETTINGS_WRITTEN
    from ganlive.models.fastgan import SkipLayerExcitation
    from ganlive.models.fold import FoldedNoise

    have = available(net)
    if wanted is None:
        # What this architecture has, not what the table wants: a 256-pixel FastGAN has no
        # 512 rungs and is a supported size. Asked for a name explicitly, we still refuse --
        # that caller has a list, and a silently dropped name would be a dial that is not
        # there. Anything absent is drawn dark by the dead-dial gate either way.
        wanted = set(SETTINGS_WRITTEN) & have
    else:
        wanted = set(wanted)
        missing = sorted(wanted - have)
        if missing:
            raise RuntimeError(
                f"this checkpoint has no {', '.join(missing)}, and the control surface writes "
                f"them. It is a different architecture from the one the dials were measured "
                f"against; loading it would raise out of the frame loop instead of here.")

    knobs = Knobs(sorted(wanted), device, dtype)
    sites = {"noise": 0, "sle": 0}

    for parent_name, parent in list(net.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, FoldedNoise):
                rung = (parent_name or child_name).split(".")[0]
                name = f"noise.{rung}"
                if name not in knobs.index:
                    continue
                setattr(parent, child_name,
                        SteerableNoise(child.coeff, child.noise, knobs.view(name)))
                sites["noise"] += 1
            elif isinstance(child, SkipLayerExcitation):
                name = f"sle.{child_name}"
                if name not in knobs.index:
                    continue
                setattr(parent, child_name, SteerableSLE(child.gate, knobs.view(name)))
                sites["sle"] += 1

    reachable = {n for n in knobs.names if n.startswith(("noise.", "sle."))}
    if reachable and not (sites["noise"] and sites["sle"]):
        raise RuntimeError(
            f"steering install matched nothing it needed: {sites}. The net must be folded "
            f"(`prepare_for_inference(..., fold=True)`) before installing, or the noise gain "
            f"is still inside `NoiseInjection` and no knob will do anything.")
    knobs.sites = sites
    return knobs


def install_stylegan2(net, device, dtype=torch.float32) -> Knobs:
    """Give a converted StyleGAN2 its dials: one live noise gain per resolution."""
    from ganlive.models.stylegan2 import BANDS, noise_sites

    sites = noise_sites(net)
    styles = [name for name, _lo, _hi in BANDS]
    knobs = Knobs(styles + list(sites), device, dtype)
    # First in the vector and so first on the strip: truncation per style range is the control
    # this family is known for, and the grain sits under it.
    net.mapping.knob = knobs.span(styles)
    # The seam the direction dials push through, allocated before `torch.compile` for the same
    # reason the noise knobs are, or the graph closes over `None`. fp32, as `w` is on both sides.
    # Not part of the `Knobs` vector: an offset with a neutral of 0, driven by the walk.
    net.mapping.push = torch.zeros(len(BANDS), net.cfg.w_dim, device=device,
                                   dtype=torch.float32)
    for name, layers in sites.items():
        view = knobs.view(name)
        for layer in layers:
            layer.knob = view
    knobs.sites = {"noise": sum(len(v) for v in sites.values()), "style": len(styles)}
    return knobs


def _render(net: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """One frame as the generator left it, with the multi-output convention unwrapped once.

    **In the generator's own range, not denormalised to 0..1.** Both are only ever handed to
    `levels`, and `mean|a-b|` on [-1,1] at 127.5 is the same number as on [0,1] at 255 -- so
    the rescale bought nothing and cost a float32 copy of the whole frame on each side of every
    comparison, which at 3072x2048 is 75 MB apiece on the load path."""
    from ganlive.models.common import first_image

    return first_image(net(z))


@torch.no_grad()
def calibrate_noise(net: nn.Module, knobs: Knobs, nz: int, device,
                   dtype=torch.float16, seed: int = 0, probes: int = 7) -> dict[str, float]:
    """Find, for this model, the gain each grain band needs to buy the levels it should."""
    from ganlive.dials.fastgan_dials import NOISE_BANDS

    z = latent(nz, seed, device, dtype)
    knobs.reset()
    base = _render(net, z)

    out: dict[str, float] = {}
    for name, target, _start in NOISE_BANDS:
        if name not in knobs.index:
            continue          # `noise_for` falls back per band; saying so again here is a copy
        lo, hi = math.log(0.5), math.log(3000.0)
        for _ in range(probes):
            mid = 0.5 * (lo + hi)
            knobs.reset()
            knobs.set(name, math.exp(mid))
            knobs.commit()
            if levels(_render(net, z), base) < target:
                lo = mid
            else:
                hi = mid
        out[name] = round(math.exp(0.5 * (lo + hi)), 3)
    knobs.reset()
    return out
