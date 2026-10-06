"""Musical time: a beat clock driven by a BPM or by MIDI clock, and where in the walk a beat falls.

Torch-free, so the MIDI and audio tools can keep time without loading the generator's stack.
`walk.SlerpWalk` turns a position from here into a latent.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ganlive.curves import clamp01

BEATS_PER_BAR = 4.0


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _ease_out(t):
    u = 1.0 - t
    return 1.0 - u * u * u


def _ease_in(t):
    return t * t * t


def shape(u: float, hold: float, when: float) -> float:
    """How far along the move the picture is, at phase `u` through the segment.

    `hold` is the share of the segment spent standing still, and `when` places the move
    inside it: 0 leaves at once and settles, 0.5 eases both ways, 1 waits and arrives late."""
    if hold > 0.0:
        u = clamp01((u - hold * when) / max(1.0 - hold, 1e-6))
    if when <= 0.5:
        a = when * 2.0
        return _ease_out(u) + (_smoothstep(u) - _ease_out(u)) * a
    a = (when - 0.5) * 2.0
    return _smoothstep(u) + (_ease_in(u) - _smoothstep(u)) * a


class MusicalClock:
    """Beats elapsed. Free-running from a BPM, or advanced by MIDI clock pulses."""

    #: MIDI clock sends 24 pulses per quarter note.
    PPQN = 24

    def __init__(self, bpm: float = 130.0) -> None:
        self._bpm = float(bpm)
        self._beats = 0.0
        self._pulses = 0
        self._external = False
        self._last_pulse: float | None = None
        self._pulse_period: float | None = None
        self.running = True

    @classmethod
    def pulse_period(cls, bpm: float) -> float:
        """Seconds between two MIDI clock pulses at `bpm`."""
        return 60.0 / (bpm * cls.PPQN)

    @classmethod
    def bpm_from(cls, pulses: float, seconds: float) -> float:
        """The tempo that fits `pulses` clock intervals into `seconds`."""
        return pulses / cls.PPQN / seconds * 60.0

    @property
    def beats(self) -> float:
        return self._beats

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def source(self) -> str:
        return "midi" if self._external else "internal"

    def advance(self, dt: float) -> None:
        """Advance by `dt` wall-clock seconds. Ignored once MIDI clock is driving."""
        if self._external or not self.running:
            return
        self._beats += dt * self._bpm / 60.0

    def on_pulse(self, now: float | None = None) -> None:
        """One MIDI clock pulse. Takes over from the internal clock on the first call.

        A machine that keeps sending clock while stopped still sets the tempo, but the picture
        stands still with the transport. With `now`, the tempo is a smoothed estimate from the
        pulse spacing."""
        self._external = True
        if self.running:
            self._pulses += 1
            self._beats = self._pulses / self.PPQN
        if now is not None:
            if self._last_pulse is not None:
                dt = now - self._last_pulse
                if 0.0005 < dt < 0.5:                     # 5000 BPM .. 5 BPM, i.e. sane
                    period = dt if self._pulse_period is None else (
                        0.85 * self._pulse_period + 0.15 * dt)
                    self._pulse_period = period
                    self._bpm = 60.0 / (period * self.PPQN)
            self._last_pulse = now

    def on_song_position(self, sixteenths: int) -> None:
        """MIDI Song Position Pointer: 16th notes since the start of the song."""
        self._pulses = int(round(sixteenths * self.PPQN / 4))
        self._beats = self._pulses / self.PPQN

    def on_start(self) -> None:
        self._pulses = 0
        self._beats = 0.0
        self.running = True

    def on_continue(self) -> None:
        self.running = True

    def on_stop(self) -> None:
        self.running = False


@dataclass
class WalkConfig:
    """What the walk does between seeds. Every field is safe to change while running."""

    beats_per_segment: float = 4.0

    spread: float = 1.0
    home_every: int = 0

    hold: float = 0.0
    when: float = 0.5
    step_grid: int = 0
    base_seed: int = 0
    loop_segments: int = 0

    #: `(n, nz)` unit rows: the generator's principal latent directions, from `dials.derive.sefa`.
    #: Set when a model is loaded; `None` until then, and the mechanism costs nothing.
    directions: object = None
    #: How far to push along each direction, one per row of `directions`. The surface writes
    #: this every frame from the `dir1..dirN` dials.
    amounts: Sequence[float] = ()

    #: Whether the loaded generator reads its latent from host memory, as the ONNX backend does.
    #: Then the walk hands over its host array and skips a device copy. Set by `bank._rewire`.
    latent_on_host: bool = False
    #: Where a `w` push goes on a generator that steers style rather than latent: the
    #: `(bands, w_dim)` tensor its mapping reads. `None` elsewhere, and the push folds into `z`.
    push_into: object = None


def position(cfg: WalkConfig, beats: float) -> tuple[int, float, float]:
    """`(segment, raw phase, how far along the move)` for a musical position."""
    b = max(0.0, beats) / max(cfg.beats_per_segment, 1e-6)
    k = math.floor(b)
    u = b - k
    t = shape(u, cfg.hold, cfg.when)
    if cfg.step_grid:
        t = math.floor(t * cfg.step_grid) / cfg.step_grid
    return k, u, t
