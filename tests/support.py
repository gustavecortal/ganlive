"""Stubs and fixtures more than one test module needs."""

from __future__ import annotations

import dataclasses
import pathlib
import time
import types

from ganlive.dials import table as _surface  # noqa: E402
from ganlive.dials.table import (
    Surface,
)
from ganlive.presets import Impulse, Macro, Preset  # noqa: E402
from ganlive.walk import (
    MusicalClock,
    SlerpWalk,
    WalkConfig,
)

NZ = 32
STILL = Preset(
    name="still",
    blurb="test fixture",
)
BREATHE = Preset(
    name="breathe",
    blurb="test fixture",
    dials={"spread": 0.12, "speed": 0.25},
    macros=[
        Macro("density", "spread", 2.5, 9.5, 0.10, 0.62, glide=1.6),
        Macro("density", "se_512", 2.5, 9.5, 0.40, 0.68, glide=1.2),
        Macro("density", "se_128", 2.5, 9.5, 0.42, 0.62, glide=1.4),
    ],
)
PULSE = Preset(
    name="pulse",
    blurb="test fixture",
    dials={"spread": 0.24},
    impulses=[
        Impulse("*", "dir1", amount=0.22, decay=0.13, velocity=0.0),
        Impulse("*", "noise", amount=0.16, decay=0.10, velocity=0.0),
    ],
)
VOICES = Preset(
    name="voices",
    blurb="test fixture",
    dials={"spread": 0.22},
    impulses=[
        Impulse("BD", "se_128", amount=0.34, decay=0.17, velocity=0.8),
        Impulse("CP", "se_512", amount=0.32, decay=0.16),
        Impulse("SD", "se_512", amount=0.24, decay=0.12),
        Impulse("OH", "se_256", amount=-0.30, decay=0.30),
        Impulse("CH", "noise", amount=0.10, decay=0.07, velocity=0.9),
        Impulse("CY", "noise", amount=0.30, decay=0.90),
        Impulse("LT", "dir1", amount=0.20, decay=0.22),
        Impulse("MT", "dir2", amount=0.18, decay=0.20),
        Impulse("HT", "dir3", amount=0.16, decay=0.18),
    ],
)
RELEASE = Preset(
    name="release",
    blurb="test fixture",
    dials={"hold": 0.78, "late": 0.15, "spread": 0.42, "speed": 0.5},
    impulses=[
        # A short attack, because these move where the picture IS rather than how fast it is
        # going, and shoving one instantly is a visible step. 60 ms is under four frames.
        Impulse("BD", "hold", amount=-0.62, decay=0.34, velocity=0.6, attack=0.06),
        Impulse("CP", "spread", amount=0.28, decay=0.45, attack=0.06),
        Impulse("OH", "late", amount=0.25, decay=0.30, attack=0.06),
    ],
    macros=[Macro("density", "speed", 2.5, 9.5, 0.30, 0.62, glide=1.8)],
)
FULL = Preset(
    name="full",
    blurb="test fixture",
    dials={"hold": 0.45, "late": 0.35, "spread": 0.20},
    impulses=list(VOICES.impulses) + [
        Impulse("BD", "hold", amount=-0.34, decay=0.30, velocity=0.6, attack=0.06),
        Impulse("RS", "dir1", amount=0.20, decay=0.20),
    ],
    macros=[
        Macro("density", "spread", 2.5, 9.5, 0.14, 0.55, glide=1.6),
        Macro("density", "hold", 2.5, 9.5, 0.62, 0.20, glide=1.8),
    ],
)
FIXTURES = {p.name: p for p in (STILL, BREATHE, PULSE, VOICES, RELEASE, FULL)}

def walk(**kw) -> SlerpWalk:
    return SlerpWalk(NZ, "cpu", WalkConfig(**kw))



def _step(w: SlerpWalk, n: int = 40) -> float:
    """Mean displacement between consecutive targets -- how far one segment actually travels."""
    return sum(float((w.seed_for(k) - w.seed_for(k + 1)).norm()) for k in range(n)) / n



def _pulses(clock, bpm, seconds, t0=0.0, jitter=0.0, rng=None):
    """Feed `seconds` of MIDI clock at `bpm`. Returns the timestamp reached."""
    period = 60.0 / (bpm * MusicalClock.PPQN)
    t = t0
    for _ in range(int(seconds / period)):
        t += period
        clock.on_pulse(t + (rng.uniform(-jitter, jitter) if rng else 0.0))
    return t



class FakeKnobs:
    """Enough of `Knobs` to record what a dial writes, with no card and no checkpoint."""

    def __init__(self, names=None):
        from ganlive.dials.table import SETTINGS_WRITTEN

        self.index = {n: i for i, n in enumerate(SETTINGS_WRITTEN if names is None else names)}
        self.written = {}
        self.noise_gains = None

    def set(self, name, value):
        if name not in self.index:
            raise KeyError(f"{name!r} is not a setting this generator has; "
                           f"have {sorted(self.index)}")
        self.written[name] = value

    def commit(self):
        pass



def _applied(**dials):
    """One set of dial values, applied to both of the surface's destinations."""
    surface = Surface(dials)
    knobs, walk = FakeKnobs(), WalkConfig()
    surface.apply(knobs, walk)
    return (dict(knobs.written), walk)



@dataclasses.dataclass
class _StubModel:
    """Just enough of `bank.Model` for the bank to be switched between."""

    rows: object = None
    directions: object = None
    dials_live: frozenset = frozenset()
    name: str = "stub"
    #: Not `None`: `bank.Model.path` has no default, so production always has one, and
    #: `_rewire` asks it which backend is loaded. A stub without one made that a crash.
    path: object = dataclasses.field(default_factory=lambda: pathlib.Path("stub.pt"))
    net: object = None
    #: A `ladder` beside the `nz`, because `_rewire` now moves the *stage* to the incoming
    #: model's native size as well as the walk to its latent width -- a bank no longer has
    #: one size. Fourth field this stub has grown for that reason; see the note below.
    cfg: object = dataclasses.field(default_factory=lambda: types.SimpleNamespace(
        nz=8, ladder=types.SimpleNamespace(width=96, height=64)))
    knobs: object = None
    graphs: int = 0
    compile_s: float = 0.0
    layout: object = dataclasses.field(default_factory=lambda: _surface.fastgan())
    #: `None` is what `bank.Model` defaults it to, and it is what every family except a converted StyleGAN2
    #: carries.
    push: object = None



def _panel(dials_live=None, levels=None):
    """A strip with a stub rig behind it, for the things that are facts about a model."""
    import types

    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner
    from ganlive.strip import DialPanel

    dirs = None if levels is None else types.SimpleNamespace(levels=levels)
    bank = None if dials_live is None else types.SimpleNamespace(
        current=_StubModel(dials_live=frozenset(dials_live), directions=dirs),
        models=[1], index=0, name="stub")
    return DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0), bank=bank)



def _pngs(folder):
    import hashlib
    from pathlib import Path

    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).glob("*.png"))}



def _drained(guide):
    """Wait for the writer thread to catch up, and hand the guide back."""
    for _ in range(500):
        if guide._q.empty():
            break
        time.sleep(0.002)
    return guide
