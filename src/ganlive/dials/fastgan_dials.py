"""FastGAN's own dials: its skip-layer gates and its noise, with hand-measured ranges.

`table.py` holds the dials every model shares and the layout types; this is what a FastGAN
adds to them. Other families derive their dials by measurement instead (see
`models.calibrate`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ganlive.curves import clamp01, evenly
from ganlive.dials import table


@dataclass(frozen=True)
class Span:
    """One dial writing one model setting over a measured range."""

    dial: str
    target: str
    lo: float
    mid: float
    hi: float

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        """The three values as a curve."""
        return evenly((self.lo, self.mid, self.hi))


#: The skip-layer gates, each blended toward identity below 1 and past its trained value
#: above. The player-facing text for each is in `DIALS`.
SPANS: tuple[Span, ...] = (
    Span("se_256", "sle.se_256", 0.08, 1.0, 2.0),
    Span("se_512", "sle.se_512", 0.08, 1.0, 2.6),
    Span("se_128", "sle.se_128", 0.10, 1.0, 2.7),
    Span("se_64", "sle.se_64", 1.0, 1.85, 2.7),
)

#: The noise bands the `noise` dial brings in, as `(setting, target levels, start)`: the mean
#: 8-bit levels `steer.calibrate_noise` tunes each band's full gain to buy, and where on the
#: dial's travel the band starts to ramp in. Finest first.
NOISE_BANDS = (("noise.feat_512", 22.0, 0.00), ("noise.feat_128", 41.4, 0.25),
               ("noise.feat_32", 60.2, 0.50), ("noise.feat_8", 52.7, 0.72))

#: Each band's full gain for a model whose noise has not been calibrated.
NOISE_FALLBACK_GAIN = {"noise.feat_512": 30.0, "noise.feat_128": 9.0,
                       "noise.feat_32": 8.0, "noise.feat_8": 60.0}

#: How much of the dial's travel each band takes to ramp from off to its full gain.
NOISE_RAMP = 0.45

#: Every model setting a FastGAN's dials write.
SETTINGS_WRITTEN = tuple(s.target for s in SPANS) + tuple(n for n, _, _ in NOISE_BANDS)

#: `(below, above, at rest)` for the readout. `noise` rests at off, and a gate rests where
#: it was trained.
POLES = {"noise": ("off", "on", "off"),
         **{s.dial: ("down", "up", "as trained") for s in SPANS}}

#: What this architecture offers a player: the shared dials, plus its own gates and noise.
DIALS: dict[str, tuple[float, str]] = {
    **table.DIALS,
    "se_64": (0.00, "the gate from the 4-tall base onto the 64-tall stage. One-sided: below "
                    "its trained value the same gate smears rather than breaks up, and that "
                    "half is not reachable from here."),
    "se_128": (0.50, "the gate from the 8-tall stage onto the 128-tall one."),
    "se_256": (0.50, "the gate from the 16-tall stage onto the 256-tall one."),
    "se_512": (0.50, "the gate from the 32-tall stage onto the 512-tall one."),
    "noise": (0.00, "the network's frozen noise patterns, brought in one band at a time as "
                    "it rises, finest mark first. The per-output-pixel band is left out: its "
                    "trained weight is 0.0001, so it needs a thousandfold gain to show."),
}

#: The MODEL block for this architecture.
MODEL = tuple(s.dial for s in SPANS) + ("noise",)


def noise_for(x: float, gains: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """The `noise` dial at `x`, as a gain for each noise band. Each band holds at 1 until
    its start, then ramps to its full gain over `NOISE_RAMP`."""
    x = clamp01(x)
    gains = gains or {}
    return [(name, 1.0 + (gains.get(name, NOISE_FALLBACK_GAIN[name]) - 1.0)
             * clamp01((x - start) / NOISE_RAMP))
            for name, _target, start in NOISE_BANDS]


def fastgan(noise_gains=None, directions: int = table.DIRECTIONS) -> table.Layout:
    """A FastGAN's layout, from the tables above and its calibrated noise gains."""
    knobs = table.shared_knobs(directions)
    for span in SPANS:
        knobs.append(table.Knob(span.dial, *DIALS[span.dial], group="MODEL",
                                poles=POLES.get(span.dial),
                                writes=(table.Write(span.target, span.points),)))
    writes = []
    for name, _target, start in NOISE_BANDS:
        # The knee is read off `noise_for`, so the layout and the played path cannot disagree.
        at_x = {x: dict(noise_for(x, noise_gains))[name]
                for x in (0.0, start, min(1.0, start + NOISE_RAMP), 1.0)}
        writes.append(table.Write(name, tuple(sorted(at_x.items()))))
    knobs.append(table.Knob("noise", *DIALS["noise"], group="MODEL",
                            poles=POLES["noise"], writes=tuple(writes)))
    return table.Layout(tuple(knobs))
