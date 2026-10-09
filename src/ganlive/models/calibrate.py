"""Driving a dial and measuring what it buys, on whatever kind of model it is.

A dial's *curve* is its measurement: the values that buy an even share of the same change at
every point of the turn, so a turn feels like the same amount of picture on every model.
`TorchProbe` runs a torch network with its settings installed, and `calibrate` takes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ganlive.curves import clamp01
from ganlive.models.common import first_image, host_latent
from ganlive.pixels import FLOOR_LEVELS, levels

#: What a calibrated dial should buy at full travel, in mean 8-bit levels, on every model.
TARGET_LEVELS = 25.0

#: Where a gain dial's value is searched.
GAIN_RANGE = (1e-4, 4.0)

#: Where a noise dial's value is searched.
NOISE_RANGE = (1e-3, 400.0)

#: Knots per side of a curve: the travel is cut into this many even shares of the change.
KNOTS = 4

#: Values tried per side, spaced logarithmically from rest to the end of the range.
SAMPLES = 24


@dataclass
class Dial:
    """One derived control: where it was inserted, and what it turned out to do."""

    name: str
    #: The tensor the `Mul` was placed on, for the report and for debugging a surprise.
    tensor: str
    #: The value to feed this slot at each of `len(curve)` evenly spaced positions from 0 to
    #: 1. More than three points, because where a gain meets an instance norm the response
    #: is logarithmic.
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
    #: How many noise patterns there are.
    noise: int = 0
    #: Whether they were already constants when the graph arrived, rather than live draws.
    was_baked: bool = False
    #: The `(height, width)` of every resolution band found in the graph.
    bands: tuple[tuple[int, int], ...] = ()
    #: The precision this graph survives, per `backend/device`: one runtime's rounding says
    #: nothing about another's. Empty until a pair is measured.
    precision: dict = field(default_factory=dict)
    nz: int = 0
    size: tuple[int, int] = (0, 0)

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dials]

    @property
    def dropped(self) -> list[str]:
        """Dials that were driven and moved nothing."""
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


class _Probe:
    """What `calibrate` asks of a model, and nothing else about it."""

    nz: int
    #: How many settings the model takes.
    width: int

    def latent(self, seed: int = 0) -> np.ndarray:
        """The latent a measurement is taken on. The precision check and the dials use the
        same one, so the verdict is about the picture the dials were calibrated against."""
        return host_latent(self.nz, seed)

    def neutral(self) -> np.ndarray:
        return np.ones(self.width, np.float32)

    def frame(self, z, values=None):
        """The picture for latent `z` with the settings at `values`, neutral when omitted."""
        raise NotImplementedError


class TorchProbe(_Probe):
    """A torch network with installed `Settings`, so a dial can be asked what it does."""

    def __init__(self, net, settings, nz: int, device="cpu", dtype=None) -> None:
        self.net, self.settings, self.nz, self.device = net, settings, nz, device
        #: The dtype the latent arrives in at play time. A compiled graph handed another
        #: dtype than it was traced with recompiles.
        self.dtype = dtype
        self.width = len(getattr(settings, "names", ()) or ())

    def frame(self, z, values=None):
        """The frame, left on the card as an owned float32 copy, so comparing two moves one
        scalar across the bus rather than two frames."""
        if self.width:
            self.settings.write[:] = self.neutral() if values is None else values
            self.settings.commit()
        latent = torch.from_numpy(z).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            return first_image(self.net(latent)).to(torch.float32, copy=True)


def deterministic(probe: _Probe, seed: int = 0) -> float:
    """8-bit levels between two answers to the same question. Zero, or the graph is unusable."""
    z = probe.latent(seed)
    return levels(probe.frame(z), probe.frame(z))


def _sweep(probe: _Probe, z, base, slot: int, limit: float,
           stop: float) -> list[tuple[float, float]]:
    """What this slot does to the picture, logarithmically from rest out to `limit`, as
    `(value, levels)` pairs.

    It stops at the first sample at or past `stop`, the largest knot target: nothing after
    that can change a knot or a level."""
    out = [(1.0, 0.0)]                      # rest is the base frame itself
    values = probe.neutral()
    for value in np.exp(np.linspace(0.0, math.log(limit), SAMPLES))[1:]:
        values[slot] = value
        out.append((float(value), levels(probe.frame(z, values), base)))
        if out[-1][1] >= stop:
            break
    return out


def _knots(sweep, targets) -> list[float]:
    """The value that first reaches each target level, interpolated in log space."""
    out = []
    for target in targets:
        value = sweep[-1][0]
        for (k0, l0), (k1, l1) in zip(sweep, sweep[1:], strict=False):
            if l1 >= target:
                share = 0.0 if l1 == l0 else (target - l0) / (l1 - l0)
                lo, hi = math.log(k0), math.log(k1)
                value = math.exp(lo + clamp01(share) * (hi - lo))
                break
        out.append(_sig(value))
    return out


def _sig(x: float, digits: int = 4) -> float:
    """A calibrated value, to four significant figures rather than four decimal places."""
    return 0.0 if x == 0 else float(f"{x:.{digits}g}")


def measured(probe: _Probe, names, size=(0, 0)) -> Adopted:
    """Calibrate dials that something else installed, and report them as `Adopted`.

    Each dial's `tensor` is its name: the installer already put them where they go."""
    found = Adopted(dials=[Dial(name=n, tensor=n) for n in names], nz=probe.nz, size=size)
    return calibrate(probe, found)


def calibrate(probe: _Probe, found: Adopted, target: float = TARGET_LEVELS,
              seed: int = 0) -> Adopted:
    """Measure every dial and give each the curve that buys the same change per turn."""
    z = probe.latent(seed)
    base = probe.frame(z)
    steps = [target * (i + 1) / KNOTS for i in range(KNOTS)]
    for slot, dial in enumerate(found.dials):
        noise = dial.name.startswith("noise_")
        floor_v, ceiling = NOISE_RANGE if noise else GAIN_RANGE
        up = _sweep(probe, z, base, slot, ceiling, stop=steps[-1])
        down = _sweep(probe, z, base, slot, floor_v, stop=steps[-1])
        reached = (max(l for _k, l in down), max(l for _k, l in up))
        # What full travel buys: the target, or less for a dial that cannot get there, so a
        # weak dial reads as weak.
        dial.levels = (round(min(reached[0], target), 3), round(min(reached[1], target), 3))

        # A side that cannot buy one knot is left off the dial. Gains often have one dead
        # half: an instance norm downstream divides a gain above 1 straight back out.
        live = (reached[0] >= steps[0], reached[1] >= steps[0])
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
