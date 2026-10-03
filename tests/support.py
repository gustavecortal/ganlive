"""Stubs and fixtures more than one test module needs."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import time
import types

from ganlive.dials import fastgan_dials as _fastgan
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

#: A Rytm over Overbridge: twelve tracks on eight voice channels, the pairs sharing one.
OVERBRIDGE = "BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9"

def tiny_stylegan2(**over):
    """A StyleGAN2 config small enough to run in a pre-commit loop, and the same shape as the
    real thing."""
    from ganlive.models import stylegan2 as S2

    return dataclasses.replace(
        S2.Config(z_dim=16, w_dim=16, img_resolution=32, channel_base=128, channel_max=32,
                  num_layers=2, num_fp16_res=0), **over)


def walk(**kw) -> SlerpWalk:
    return SlerpWalk(NZ, "cpu", WalkConfig(**kw))



def _step(w: SlerpWalk, n: int = 40) -> float:
    """Mean displacement between consecutive targets -- how far one segment actually travels."""
    return sum(float((w.seed_for(k) - w.seed_for(k + 1)).norm()) for k in range(n)) / n



def _pulses(clock, bpm, seconds, t0=0.0, jitter=0.0, rng=None):
    """Feed `seconds` of MIDI clock at `bpm`. Returns the timestamp reached."""
    period = MusicalClock.pulse_period(bpm)
    t = t0
    for _ in range(int(seconds / period)):
        t += period
        clock.on_pulse(t + (rng.uniform(-jitter, jitter) if rng else 0.0))
    return t



class FakeKnobs:
    """Enough of `Knobs` to record what a dial writes, with no card and no checkpoint."""

    def __init__(self, names=None):
        from ganlive.dials.fastgan_dials import SETTINGS_WRITTEN

        self.index = {n: i for i, n in enumerate(SETTINGS_WRITTEN if names is None else names)}
        self.written = {}

    def set(self, name, value):
        if name not in self.index:
            raise KeyError(f"{name!r} is not a setting this generator has; "
                           f"have {sorted(self.index)}")
        self.written[name] = value

    def commit(self):
        pass



def _applied(**dials):
    """One set of dial values, applied to both of the surface's destinations."""
    surface = Surface(dials, layout=_fastgan.fastgan())
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
    layout: object = dataclasses.field(default_factory=lambda: _fastgan.fastgan())
    #: `None` is what `bank.Model` defaults it to, and it is what every family except a converted StyleGAN2
    #: carries.
    push: object = None



def _runner(preset: str = "still"):
    """A runner for one fixture preset, on the twelve-track kit, with this project's dials."""
    from ganlive.control.kit import INDEX
    from ganlive.presets import PresetRunner

    return PresetRunner(FIXTURES[preset], INDEX, 60.0, layout=_fastgan.fastgan())


def stub_bank(current, **over):
    """Just enough of `bank.Bank` for a strip to read the playing model off."""
    return types.SimpleNamespace(**{"current": current, "models": [1], "index": 0,
                                    "name": "stub", **over})


def _panel(dials_live=None, levels=None, runner=None):
    """A strip with a stub bank behind it, for the things that are facts about a model."""
    from ganlive.strip import DialPanel

    dirs = None if levels is None else types.SimpleNamespace(levels=levels)
    bank = None if dials_live is None else stub_bank(
        _StubModel(dials_live=frozenset(dials_live), directions=dirs))
    return DialPanel(runner or _runner(), bank=bank)


@contextlib.contextmanager
def headless_renderer(size, title: str = "test"):
    """A real SDL renderer on the dummy video driver, for drawing the strip with no screen."""
    import pygame

    with dummy_display():
        from pygame._sdl2.video import Renderer, Window

        pygame.init()
        yield Renderer(Window(title, size=size), vsync=False)


@contextlib.contextmanager
def dummy_display():
    """SDL on its dummy video driver for the duration, then the display closed and the
    driver setting put back as it was."""
    import pygame

    before = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        yield
    finally:
        pygame.display.quit()
        if before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = before



def offline(audio, samplerate: int, fps: float, config=None) -> dict:
    """Run the onset extractor over a whole recording at a video frame rate."""
    import numpy as np

    from ganlive.control.features import FeatureExtractor

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    ex = FeatureExtractor(x.shape[0], samplerate, config)
    per_frame = samplerate / fps
    frames = int(x.shape[1] / per_frame)
    since = np.zeros((frames, x.shape[0]), dtype=np.float32)
    whole = {k: np.zeros(frames, dtype=np.float32) for k in ex.features()}
    onsets: list[list[tuple[int, float, float]]] = []
    for f in range(frames):
        a, b = int(f * per_frame), int((f + 1) * per_frame)
        ex.push(x[:, a:b])
        since[f] = ex.since
        for k, v in ex.features().items():
            whole[k][f] = v
        onsets.append(ex.drain())
    return {"since": since, "onsets": onsets, "frames": frames, "fps": fps, **whole}


def score_onsets(detected, fps: float, truth, channel_of: dict[str, int],
                 tolerance: float = 0.030) -> dict:
    """Precision, recall and timing error of per-frame onsets against the simulator's events."""
    import numpy as np

    got: list[tuple[float, int]] = []
    for f, items in enumerate(detected):
        for ch, _vel, ago in items:
            got.append(((f + 1) / fps - ago, ch))
    want = [(t, channel_of[name]) for t, name, _v in truth if name in channel_of]

    used = [False] * len(got)
    matched, errors = 0, []
    for t, ch in want:
        best, best_i = tolerance, -1
        for i, (gt, gch) in enumerate(got):
            if used[i] or gch != ch:
                continue
            d = abs(gt - t)
            if d < best:
                best, best_i = d, i
        if best_i >= 0:
            used[best_i] = True
            matched += 1
            errors.append(got[best_i][0] - t)
    return {
        "truth": len(want), "detected": len(got), "matched": matched,
        "recall": matched / max(len(want), 1),
        "precision": matched / max(len(got), 1),
        "mean_error_ms": float(np.mean(errors) * 1000) if errors else 0.0,
        "abs_error_ms": float(np.mean(np.abs(errors)) * 1000) if errors else 0.0,
        "max_error_ms": float(np.max(np.abs(errors)) * 1000) if errors else 0.0,
    }


def _drained(guide):
    """Wait for the writer thread to catch up, and hand the guide back."""
    for _ in range(500):
        if guide._q.empty():
            break
        time.sleep(0.002)
    return guide
