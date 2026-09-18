"""Driving a dial and measuring what it bought, on whatever kind of model it is.

A dial's *curve* is its measurement: the travel that buys an even share of the same change at
every point of the turn, so a turn feels like the same amount of picture on every model. That
is the whole argument for calibrating rather than shipping a range, and none of it is about
ONNX -- `bank` calibrates a converted StyleGAN2 through `TorchProbe` and this same `measured`.

It lived in `dials/onnx_dials.py`, which claimed to be one architecture's dial table beside
`fastgan_dials.py` and was in fact this plus the whole adoption pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ganlive.pixels import FLOOR_LEVELS

#: What a calibrated dial should buy at full travel, in mean 8-bit levels. Chosen so that
#: every dial on every model feels like the same amount of change under the hand, which is
#: the entire argument for calibrating rather than shipping a range.
TARGET_LEVELS = 25.0


#: Where a gain dial is allowed to be searched.
GAIN_RANGE = (1e-4, 4.0)


NOISE_RANGE = (1e-3, 400.0)


@dataclass
class Dial:
    """One derived control: where it was inserted, and what it turned out to do."""

    name: str
    #: The tensor the `Multiply` was placed on, for the report and for debugging a surprise.
    tensor: str
    #: What to feed this slot at each of `len(curve)` evenly spaced dial positions from 0 to 1. Three points
    #: was not enough: the response is logarithmic where a gain meets an instance norm, and a straight line
    #: through three of them put three of five rendered frames at visibly the same picture.
    curve: tuple[float, ...] = (0.0, 1.0, 2.0)
    #: Where on its own travel this dial sits when nothing is touching it. 0.5 for a control
    #: that works both ways; 0.0 or 1.0 for one that works only one way.
    rest: float = 0.5
    #: Mean 8-bit levels at each end, once measured. `None` before that.
    levels: tuple[float, float] | None = None

    @property
    def moved(self) -> float:
        return 0.0 if self.levels is None else max(self.levels)


@dataclass
class Adopted:
    """What adoption found, for the report and for the tests to assert against."""

    dials: list[Dial] = field(default_factory=list)
    #: How many noise patterns there are, however they got that way.
    noise: int = 0
    #: Whether they were already constants when the graph arrived, rather than live draws.
    was_baked: bool = False
    #: The `(height, width)` of every resolution band found in the graph.
    bands: tuple[tuple[int, int], ...] = ()
    #: What precision this graph survives, per `backend/device` -- one runtime breaking it is
    #: not evidence about another's rounding. Empty until something measures a pair.
    precision: dict = field(default_factory=dict)
    nz: int = 0
    size: tuple[int, int] = (0, 0)
    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dials]

    @property
    def dropped(self) -> list[str]:
        """Dials that were driven and moved nothing. Derived, so it cannot disagree."""
        return [d.name for d in self.dials if d.moved < FLOOR_LEVELS]

    def report(self) -> str:
        live = ", ".join(f"{d.name} {d.moved:.1f}" for d in self.dials) or "none"
        noise = (f"{self.noise} noise pattern(s) already frozen in the file"
                 if self.was_baked else f"{self.noise} random draw(s) frozen")
        bands = ", ".join(f"{w}x{h}" for h, w in self.bands)
        said = ", ".join(f"{k} {v}" for k, v in self.precision.items()) or "unmeasured"
        out = (f"{self.nz} latent, {self.size[1]}x{self.size[0]}, {said}, {noise}, "
               f"bands {bands}\n"
               f"  dials (mean 8-bit levels at full travel): {live}")
        if self.dropped:
            out += f"\n  dropped as inert: {', '.join(self.dropped)}"
        return out


def _latent(nz: int, seed: int = 0) -> np.ndarray:
    """The latent every measurement in this module is taken on.

    One spelling, because `usable_precision` decides fp16 against fp32 on it and `calibrate`
    then measures every dial on it: drawn differently in the two places, the precision verdict
    would belong to a picture no dial was ever calibrated against, with nothing to show it."""
    return np.random.default_rng(seed).standard_normal((1, nz)).astype(np.float32)


class _Probe:
    """The three things `calibrate` asks of a model, and nothing else about it."""

    nz: int
    settings: int

    def latent(self, seed: int = 0) -> np.ndarray:
        return _latent(self.nz, seed)

    def neutral(self) -> np.ndarray:
        return np.ones(self.settings, np.float32)

    def frame(self, z, k=None) -> np.ndarray:
        raise NotImplementedError


class Probe(_Probe):
    """The adopted graph, runnable on the host, so a dial can be asked what it does."""

    def __init__(self, model, device: str = "cpu", precision: str = "FP16") -> None:
        """`device` is `cpu` for ONNX Runtime, or an OpenVINO device name such as `GPU`."""
        from ganlive.models.runtime import open_graph

        cpu = device.lower() == "cpu"
        self.runner = open_graph(model, backend="ort" if cpu else "openvino",
                                 device="CPUExecutionProvider" if cpu else device,
                                 precision=precision)
        self.nz, self.settings = self.runner.nz, self.runner.settings

    def frame(self, z, k=None) -> np.ndarray:
        # A copy, not the view an OpenVINO request hands back: it dies at the next
        # submission, and every measurement here compares one frame with another taken later.
        return np.array(self.runner.infer(z, self.neutral() if k is None else k), np.float32)


def levels(a, b) -> float:
    """Mean absolute difference in 8-bit levels -- the unit every measurement here is in."""
    return float(np.abs(a - b).mean() * 127.5)


def deterministic(probe: Probe, seed: int = 0) -> float:
    """8-bit levels between two answers to the same question. Zero, or the graph is unusable."""
    z = probe.latent(seed)
    return levels(probe.frame(z), probe.frame(z))


def _sweep(probe: Probe, z, base, slot: int, limit: float,
           samples: int, stop: float | None = None) -> list[tuple[float, float]]:
    """What this slot does to the picture, logarithmically from rest out to `limit`.

    **It stops at `stop`, because nothing downstream can see past it.** `_knots` takes the
    *first* crossing of each target and the largest target is `stop`; `Dial.levels` is capped at
    it; the live-half test asks only for the smallest. So every sample after the first one at or
    above `stop` is a forward pass rendered and discarded, and dropping them changes no knot, no
    level and no verdict. A side that never gets there is untouched and still costs the lot.
    """
    # The first sample is `exp(0) == 1.0`, which is rest: the same frame `base` already is,
    # measuring 0.0 levels, which `_knots` can never choose. Rendering it was 22 wasted
    # forwards per adopt out of 531.
    out = [(1.0, 0.0)]
    k = probe.neutral()
    for value in np.exp(np.linspace(0.0, math.log(limit), samples))[1:]:
        k[slot] = value
        out.append((float(value), levels(probe.frame(z, k), base)))
        if stop is not None and out[-1][1] >= stop:
            break
    k[slot] = 1.0
    return out


def _knots(sweep, targets) -> list[float]:
    """The value that first reaches each target level, read off the sweep."""
    out = []
    for target in targets:
        value = sweep[-1][0]
        for (k0, l0), (k1, l1) in zip(sweep, sweep[1:], strict=False):
            if l1 >= target:
                share = 0.0 if l1 == l0 else (target - l0) / (l1 - l0)
                lo, hi = math.log(k0), math.log(k1)
                # Written out rather than through `dials.table.clamp01`: this holds an
                # interpolation share inside the pair it was read between, which is not the
                # dial clamp and must not follow it if that ever stops being a plain clamp --
                # and `models` importing from `dials` is the wrong way round besides.
                value = math.exp(lo + min(1.0, max(0.0, share)) * (hi - lo))
                break
        out.append(_sig(value))
    return out


def _sig(x: float, digits: int = 4) -> float:
    """A calibrated value, to four significant figures rather than four decimal places."""
    return 0.0 if x == 0 else float(f"{x:.{digits}g}")


class TorchProbe(_Probe):
    """The same contract as `Probe`, for a network that is a network rather than a graph."""

    def __init__(self, net, knobs, nz: int, device="cpu", dtype=None) -> None:
        self.net, self.knobs, self.nz, self.device = net, knobs, nz, device
        #: The dtype the latent arrives in *at play time*. Handing a compiled graph a latent
        #: of a different dtype than it was traced with recompiles the whole thing, which on
        #: a StyleGAN2 is a hundred seconds, in the middle of measuring it.
        self.dtype = dtype
        self.settings = len(getattr(knobs, "names", ()) or ())

    def frame(self, z, k=None) -> np.ndarray:
        import torch

        from ganlive.models.common import first_image

        if self.settings:
            self.knobs.write[:] = self.neutral() if k is None else k
            self.knobs.commit()
        latent = torch.from_numpy(z).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            out = first_image(self.net(latent))
        return out.cpu().float().numpy()


def measured(probe, names, size=(0, 0), **kw) -> Adopted:
    """Calibrate a set of dials that something else installed, and report them as `Adopted`.

    The dial's name *is* its tensor here: the installer already put them where they go, and
    nothing downstream of this reads `tensor` on a dial it did not insert itself."""
    found = Adopted(dials=[Dial(name=n, tensor=n) for n in names], nz=probe.nz, size=size)
    return calibrate(probe, found, **kw)


def calibrate(probe: Probe, found: Adopted, target: float = TARGET_LEVELS,
              knots: int = 4, samples: int = 24, seed: int = 0) -> Adopted:
    """Measure every derived dial and give each the travel that buys the same change."""
    z = probe.latent(seed)
    base = probe.frame(z)
    steps = [target * (i + 1) / knots for i in range(knots)]
    for slot, dial in enumerate(found.dials):
        noise = dial.name.startswith("noise_")
        floor_v, ceiling = NOISE_RANGE if noise else GAIN_RANGE
        up = _sweep(probe, z, base, slot, ceiling, samples, stop=steps[-1])
        down = _sweep(probe, z, base, slot, floor_v, samples, stop=steps[-1])
        reached = (max(l for _k, l in down), max(l for _k, l in up))
        # What full travel actually buys, which is the target for a dial that gets there and
        # less for one that does not -- so a weak dial reads as weak rather than as its
        # headroom. `reached` is what the range limit could do; the dial stops at its knots.
        dial.levels = (round(min(reached[0], target), 3), round(min(reached[1], target), 3))

        # **A side that cannot buy one knot is not a side.** The floor asks whether the dial does anything
        # at all; a *half* has to be worth turning.
        live = (reached[0] >= steps[0], reached[1] >= steps[0])

        # **A dial with a dead half is a dial that lies about where it is.** Scaling a feature band *up*
        # buys 0.2 levels on this StyleGAN and scaling it down buys 91, because the instance norm downstream
        # divides a constant gain straight back out.
        if all(live):
            dial.curve = tuple(_knots(down, steps)[::-1]) + (1.0,) + tuple(_knots(up, steps))
            dial.rest = 0.5
        elif live[0]:
            dial.curve = tuple(_knots(down, steps)[::-1]) + (1.0,)
            dial.rest = 1.0
        else:
            dial.curve = (1.0,) + tuple(_knots(up, steps))
            dial.rest = 0.0
    return found
