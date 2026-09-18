"""FastGAN's own dial vocabulary: what this project's generator offers, and what each buys.

`table.py` is the generic surface -- the spine every model shares, and the builders that turn
measurements into a layout. This is one architecture's contribution to it, beside
`onnx_dials.py`, which is another's. It lived in `table.py` as module state while the other two
families passed theirs in as arguments, which made "the per-model block" mean, literally,
whatever FastGAN had.
"""

from __future__ import annotations

from dataclasses import dataclass

from ganlive.dials.table import (
    DIALS as SPINE_DIALS,
)
from ganlive.dials.table import (
    DIRECTIONS,
    Knob,
    Layout,
    Write,
    clamp01,
    evenly,
    spine,
)


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
        """The same three numbers as a curve, so nothing has to retype them as one."""
        return evenly((self.lo, self.mid, self.hi))


#: The player-facing text for each of these lives in `DIALS`, which is what the strip reads.
#: `Span` used to carry a second copy as `note`; nobody read it, and it drifted.
SPANS: tuple[Span, ...] = (
    Span("se_256", "sle.se_256", 0.08, 1.0, 2.0),
    Span("se_512", "sle.se_512", 0.08, 1.0, 2.6),
    Span("se_128", "sle.se_128", 0.10, 1.0, 2.7),
    # No `pre_tanh`. A gain into the output squash is a contrast curve on the finished picture,
    # the same class as the deleted `zoom` and `chroma`: it explores nothing. Measured on the
    # shipping checkpoint over four latents it was also the weakest gate, 24 and 40 8-bit
    # levels at its two ends against 59 to 93 for `se_512`, and its two ends were one change
    # with the sign flipped (|cos| 0.91 between their difference images).
    Span("se_64", "sle.se_64", 1.0, 1.85, 2.7),
    # No `z_scale`, and with it the `where` field and the walk-writing branch it was the only
    # user of. Its up end moves 1.30x what a random direction moves on a fine-tuned FastGAN
    # and 1.53x on another -- under the bar every direction dial has to clear -- and its
    # down end is the flat wash its own description used to advertise, measuring |cos| 0.95
    # against `dir5`. On a StyleGAN2 the mapping network's pixel norm cancelled it outright.
)

NOISE_BANDS = (("noise.feat_512", 22.0, 0.00), ("noise.feat_128", 41.4, 0.25),
              ("noise.feat_32", 60.2, 0.50), ("noise.feat_8", 52.7, 0.72))

NOISE_FALLBACK_GAIN = {"noise.feat_512": 30.0, "noise.feat_128": 9.0,
                      "noise.feat_32": 8.0, "noise.feat_8": 60.0}
NOISE_RAMP = 0.45

SETTINGS_WRITTEN = tuple(s.target for s in SPANS) + tuple(n for n, _, _ in NOISE_BANDS)

#: `(below, above, at rest)` for the readout, for the dials this architecture adds. `noise`
#: rests at off, and a gate rests where it was trained.
POLES = {"noise": ("off", "on", "off"),
         **{gate: ("down", "up", "as trained")
            for gate in ("se_64", "se_128", "se_256", "se_512")}}

#: What this architecture offers a player: the shared spine, plus its own gates and grain.
DIALS: dict[str, tuple[float, str]] = {
    **SPINE_DIALS,
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

#: The per-model block for this architecture, named rather than derived by subtraction.
MODEL = tuple(s.dial for s in SPANS) + ("noise",)


def noise_for(x: float, gains: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """`noise` dial to a gain for each noise band."""
    x = clamp01(x)
    gains = gains or {}          # per-key fallback below already covers None and {}
    return [(name, 1.0 + (gains.get(name, NOISE_FALLBACK_GAIN[name]) - 1.0)
             * clamp01((x - start) / NOISE_RAMP))
            for name, _target, start in NOISE_BANDS]


def fastgan(noise_gains=None, directions: int = DIRECTIONS) -> Layout:
    """The layout this instrument was built around, from the tables above."""
    knobs = spine(directions)
    for span in SPANS:
        knobs.append(Knob(span.dial, *DIALS[span.dial], group="MODEL",
                          poles=POLES.get(span.dial),
                          writes=(Write(span.target, span.points),)))
    writes = []
    for name, _target, start in NOISE_BANDS:
        # The knee is read off `noise_for` rather than derived beside it, so the two spellings of a
        # hold-then-ramp cannot drift and the tests assert the function the played path runs.
        at_x = {x: dict(noise_for(x, noise_gains))[name]
                for x in (0.0, start, min(1.0, start + NOISE_RAMP), 1.0)}
        writes.append(Write(name, tuple(sorted(at_x.items()))))
    knobs.append(Knob("noise", *DIALS["noise"], group="MODEL",
                      poles=POLES["noise"], writes=tuple(writes)))
    return Layout(tuple(knobs))
