"""Live handles on a prepared generator: the parameters a drum machine can steer."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from ganlive.dials.derive import _latent
from ganlive.pixels import levels

NOISE_RUNGS = ("feat_8", "feat_32", "feat_128", "feat_512", "feat_2048")


SLE_GATES = ("se_64", "se_128", "se_256", "se_512")


# INVARIANT: `install` must run BEFORE torch.compile. Installed after, every knob is silently
# inert -- no error, no effect. `bank._prepare_fastgan` does the two in order. Every knob is a
# device tensor, never a float: compile guards on scalar values and would recompile each frame.
class SteerableNoise(nn.Module):
    """`FoldedNoise` with a live gain: `x + (coeff * rung) * noise`."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor,
                 rung: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)
        # A plain attribute, NOT a buffer: register_buffer + .to() hands each module its
        # own detached copy, leaving the caller holding a tensor that steers nothing.
        self.rung = rung

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff * self.rung, self.noise)


class SteerableSLE(nn.Module):
    """`SkipLayerExcitation` with the gate blended toward identity: `high * (1 + b*(g-1))`."""

    def __init__(self, gate: nn.Module, blend: torch.Tensor) -> None:
        super().__init__()
        self.gate = gate
        self.blend = blend

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * (1.0 + self.blend * (self.gate(low) - 1.0))


class Knobs:
    """The generator's whole control state as one vector, plus the names to address it by."""

    STAGING = 3

    def __init__(self, names, device, dtype) -> None:
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        from ganlive.pixels import pinned

        self.vec = torch.ones(len(self.names), device=device, dtype=dtype)
        self.host = [pinned((len(self.names),), dtype).fill_(1.0) for _ in range(self.STAGING)]
        self._writes = [buf.numpy() for buf in self.host]
        self.pinned = all(buf.is_pinned() for buf in self.host)
        self._slot = 0
        self.write = self._writes[0]
        #: What the card holds. Starts at the neutral the vector is built holding, so a frame
        #: that moves nothing sends nothing -- see `commit`. Once `feed_from` is called it *is*
        #: the buffer the card reads, and the ring below stops being the sender.
        self._sent = np.ones(len(self.names), dtype=self._writes[0].dtype)
        self._fed = False
        self.skipped = 0
        self.sites: dict[str, int] = {}
        self.noise_gains: dict[str, float] | None = None

    def view(self, name: str) -> torch.Tensor:
        """The slice of the vector a module should hold. A view, so `commit` reaches it."""
        i = self.index[name]
        return self.vec[i:i + 1]

    def span(self, names) -> torch.Tensor:
        """Several adjacent settings as one view, for a module that wants them as a vector."""
        first = self.index[names[0]]
        if [self.index[n] for n in names] != list(range(first, first + len(names))):
            raise ValueError(
                f"{list(names)} are not adjacent in {self.names}, so they cannot be one view. "
                f"Order them together when the vector is built.")
        return self.vec[first:first + len(names)]

    def set(self, name: str, value: float) -> None:
        """Write one setting. A name this model does not have is accepted and dropped."""
        i = self.index.get(name)
        if i is not None:
            self.write[i] = value

    def reset(self) -> None:
        """Back to the trained neutral: every setting 1.0, the identity for all of them."""
        self.write[:] = 1.0
        self.commit()

    def committed(self) -> object:
        """The values last committed, as a host array."""
        return self._sent

    def feed_from(self, host: torch.Tensor) -> None:
        """From now on the card reads `vec` from `host` on every replay -- a captured graph
        uploads it itself, see `models.capture.capture` -- so a commit is a host write into it and
        nothing is sent. `host` already holds what the card holds."""
        self._sent, self._fed = host.numpy(), True

    def commit(self) -> None:
        """Send this frame's settings to the card -- unless the card already has them.

        **The copy is thirty-two bytes and it measured 1.6 ms.** `Tensor.copy_` drops the GIL,
        and taking it back from a window thread that is uploading a texture and presenting
        costs up to one switch interval: 5 ms by default, 1.64 median and 3.38 at p95 on the
        played loop, against 0.03 with no window open. A frame on which no dial moved was
        paying all of that for a copy of the numbers already there, and at rest no dial moves
        -- every dial on its own rest and nothing wired is what the instrument starts on.
        Comparing the eight floats first is about a microsecond against that."""
        if np.array_equal(self.write, self._sent):
            self.skipped += 1
            return
        np.copyto(self._sent, self.write)
        if self._fed:
            return                      # the graph reads `_sent` itself on the next replay
        self.vec.copy_(self.host[self._slot], non_blocking=True)
        nxt = (self._slot + 1) % self.STAGING
        self._writes[nxt][:] = self.write
        self._slot = nxt
        self.write = self._writes[nxt]


def available(net: nn.Module) -> set[str]:
    """Which settings this architecture actually has, by the parameters it carries."""
    have = {f"sle.{g}" for g in SLE_GATES if getattr(net, g, None) is not None}
    have |= {f"noise.{r}" for r in NOISE_RUNGS if getattr(net, r, None) is not None}
    return have


def install(net: nn.Module, device, dtype=torch.float16, wanted=None) -> Knobs:
    """Swap the steerable modules in and hand back the vector that drives them."""
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

    z = _latent(nz, seed, device, dtype)
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




@torch.no_grad()
def verify(net: nn.Module, knobs: Knobs, layout, nz: int, device, dtype=torch.float16,
           seed: int = 0):
    """Drive every unmeasured dial that writes the model to each end of its travel and measure
    what moved, through `Surface.apply` -- the path a hand takes. Returns the layout with
    `measured` filled in; `bank.live_dials` draws anything under `FLOOR_LEVELS` dark."""
    from ganlive.dials.table import Layout, Surface
    from ganlive.walk import WalkConfig

    surface, walk = Surface(layout=layout), WalkConfig()

    def frame(z, **held):
        surface.values.update(layout.rests)
        surface.set_held(held)
        surface.apply(knobs, walk)
        knobs.commit()
        return _render(net, z)

    z = _latent(nz, seed, device, dtype)
    base = frame(z)
    out = []
    for knob in layout.knobs:
        if not knob.writes or knob.measured is not None:
            out.append(knob)
            continue
        moved = max((levels(frame(z, **{knob.name: x}), base)
                     for x in (0.0, 1.0) if abs(x - knob.rest) > 1e-6), default=0.0)
        out.append(replace(knob, measured=round(moved, 3)))
    knobs.reset()
    return Layout(tuple(out))
