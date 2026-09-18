"""The latent walk, measured in beats instead of frames."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ganlive.dials.table import clamp01


def slerp_path(z0: torch.Tensor, z1: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """Latents along the great circle from `z0` to `z1`, one row per `t` in `ts`."""
    n0, n1 = z0 / z0.norm(), z1 / z1.norm()
    omega = torch.clamp((n0 * n1).sum(), -1.0, 1.0).acos()
    ts = ts.to(z0.device, z0.dtype).view(-1, 1)
    if omega.abs() < 1e-6:
        return z0.view(1, -1) + ts * (z1 - z0).view(1, -1)
    so = omega.sin()
    a = ((1.0 - ts) * omega).sin() / so
    b = (ts * omega).sin() / so
    return a * z0.view(1, -1) + b * z1.view(1, -1)

BEATS_PER_BAR = 4.0


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _ease_out(t):
    u = 1.0 - t
    return 1.0 - u * u * u


def _ease_in(t):
    return t * t * t


def shape(u: float, hold: float, when: float) -> float:
    """How far along the move the picture is, at phase `u` through the segment."""
    if hold > 0.0:
        u = clamp01((u - hold * when) / max(1.0 - hold, 1e-6))
    if when <= 0.5:
        a = when * 2.0
        return _ease_out(u) + (_smoothstep(u) - _ease_out(u)) * a
    a = (when - 0.5) * 2.0
    return _smoothstep(u) + (_ease_in(u) - _smoothstep(u)) * a


class MusicalClock:
    """Beats elapsed. Free-running from a BPM, or advanced by MIDI clock pulses."""

    PPQN = 24

    def __init__(self, bpm: float = 130.0) -> None:
        self._bpm = float(bpm)
        self._beats = 0.0
        self._pulses = 0
        self._external = False
        self._last_pulse: float | None = None
        self._pulse_period: float | None = None
        self.running = True

    @property
    def beats(self) -> float:
        return self._beats

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def bar_phase(self) -> float:
        """Position within the current bar, 0 on the downbeat."""
        return (self._beats % BEATS_PER_BAR) / BEATS_PER_BAR

    @property
    def source(self) -> str:
        return "midi" if self._external else "internal"

    def advance(self, dt: float) -> None:
        """Advance by `dt` wall-clock seconds. Ignored once MIDI clock is driving."""
        if self._external or not self.running:
            return
        self._beats += dt * self._bpm / 60.0

    def on_pulse(self, now: float | None = None) -> None:
        """One MIDI clock pulse. Takes over from the internal clock on the first call."""
        self._external = True
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

    #: `(n, nz)` unit rows: the generator's principal latent directions, from `gan.directions.sefa`.
    #: Set when a model is loaded; `None` until then, and the mechanism costs nothing.
    directions: object = None
    #: How far to push along each direction, one per row of `directions`. The surface writes
    #: this every frame from the `dir1..dirN` dials.
    amounts: Sequence[float] = ()

    #: Whether the loaded generator wants the latent on the host. The ONNX backend does: a device
    #: tensor made it download the same numbers back, 0.27 ms a frame. Set by `bank._rewire`.
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


class SlerpWalk:
    """Great-circle walk over a deterministic seed sequence, addressed by musical position."""

    STAGING = 3

    #: The width every seed is drawn at, before any model sees it. Larger than any latent this
    #: instrument has met, so the nesting above holds across a bank. One `randn(4096)` per segment.
    CANON = 4096

    def __init__(self, nz: int, device, config: WalkConfig | None = None,
                 dtype=torch.float32) -> None:
        self._slerp = slerp_path
        self.nz, self.device, self.dtype = nz, device, dtype
        self.cfg = config or WalkConfig()
        self._k: tuple | None = None
        self._z1 = None
        self._omega = self._so = None
        self.segment_index = 0
        self._offset_key: tuple | None = None
        self._offset_rows = None
        self._offset_vec = None
        #: The offset array last handed to a `w` seam, by identity. `_offset` returns the same array
        #: while the amounts hold still, which keeps a host-to-device copy off the frame path.
        self._pushed = None
        self._slot = 0
        self._staging, self._views = self._staging_ring()
        self._pair = None
        self._coef = np.zeros(2, dtype=np.float32)
        self._scratch = np.zeros(nz, dtype=np.float32)

    def _staging_ring(self):
        """The pinned host ring the latent is written into, at the current width."""
        from ganlive.models.graph import _pinned

        bufs = [_pinned((1, self.nz), self.dtype) for _ in range(self.STAGING)]
        return bufs, [buf.numpy().reshape(-1) for buf in bufs]

    def retarget(self, nz: int) -> None:
        """Hand this walk to a generator of a different latent width, mid-set."""
        nz = int(nz)
        if nz == self.nz:
            return
        self.nz = nz
        self._offset_key = self._offset_rows = self._offset_vec = None
        # The `w` seam is a different tensor on the incoming model, so the "already handed
        # over" memo has to go with it or the first frame after a switch writes nothing.
        self._pushed = None
        self._scratch = np.zeros(nz, dtype=np.float32)
        self._staging, self._views = self._staging_ring()
        # The cached endpoints and the angle are the incoming width's, so they go. The clock does not:
        # segment and phase are read back from the beat, so the walk resumes where it was.
        self._k = self._z1 = self._pair = None
        self._omega = self._so = None

    @property
    def canon(self) -> int:
        """The width the walk itself runs at, above every model that reads it."""
        return max(self.CANON, self.nz)

    def _draw(self, k: int, salt: int = 0) -> torch.Tensor:
        """A Gaussian draw that depends only on `(base_seed, salt, k)`. Stays on the HOST."""
        g = torch.Generator(device="cpu").manual_seed(
            (self.cfg.base_seed * 1_000_003 + salt * 7_919 + k) & 0x7FFF_FFFF)
        return torch.randn(self.canon, generator=g)

    def home_for(self, k: int) -> torch.Tensor:
        """The neighbourhood segment `k` is drawn around. Salted so `home_0` is not `seed_0`."""
        return self._home_canon(k)[:self.nz]

    def _home_canon(self, k: int) -> torch.Tensor:
        block = k // self.cfg.home_every if self.cfg.home_every else 0
        return self._draw(block, salt=1)

    def seed_for(self, k: int) -> torch.Tensor:
        """Segment `k`'s target latent, at the loaded model's width."""
        return self._seed_canon(k)[:self.nz]

    def _seed_canon(self, k: int) -> torch.Tensor:
        """The same target at the walk's own width. A pure function of `k`, which is what makes the whole
        sequence addressable rather than stepped."""
        if self.cfg.loop_segments:
            k %= self.cfg.loop_segments
        target = self._draw(k)
        if self.cfg.spread >= 1.0:
            return target
        return self._slerp(self._home_canon(k), target,
                           torch.tensor([max(0.0, self.cfg.spread)]))[0]

    # Everything that chooses WHICH destination a segment is must appear here, or a cache hit
    # returns the previous setting's. `spread` is out: it changes how far, not which one.
    SEED_FIELDS = ("base_seed", "loop_segments", "home_every")

    def _key(self, k: int) -> tuple:
        """What the cached endpoints for segment `k` actually depend on."""
        return (k, self.cfg.base_seed, self.cfg.loop_segments, self.cfg.home_every)

    def _load(self, k: int) -> None:
        """Cache the segment's endpoints and the two constants a slerp needs."""
        key = self._key(k)
        if self._k == key:
            return
        z0 = self._z1 if self._k == self._key(k - 1) and self._z1 is not None else None
        if z0 is None:
            z0 = self.seed_for(k).numpy().astype(np.float32)
        z1 = self.seed_for(k + 1).numpy().astype(np.float32)
        n0 = z0 / np.linalg.norm(z0)
        n1 = z1 / np.linalg.norm(z1)
        self._z1 = z1
        self._pair = np.stack([z0, z1])
        self._omega = float(np.arccos(np.clip(float((n0 * n1).sum()), -1.0, 1.0)))
        self._so = math.sin(self._omega)
        self._k = key
        self.segment_index = k

    def phase(self, beats: float) -> tuple[int, float]:
        """`(segment index, raw phase in [0,1))` for a musical position."""
        b = max(0.0, beats) / max(self.cfg.beats_per_segment, 1e-6)
        k = math.floor(b)
        return k, b - k

    def latent(self, beats: float) -> torch.Tensor:
        """The `(1, nz)` input for this musical position, on `device`."""
        k, u, t = position(self.cfg, beats)
        self._load(k)
        if abs(self._omega) < 1e-6:                      # near-parallel: the arc is unstable
            a, b = 1.0 - t, t
        else:
            a = math.sin((1.0 - t) * self._omega) / self._so
            b = math.sin(t * self._omega) / self._so

        self._slot = (self._slot + 1) % self.STAGING
        view = self._views[self._slot]
        self._coef[0], self._coef[1] = a, b
        offset = self._offset()
        if self.cfg.push_into is not None:
            # A `w` push never touches the latent: the mapping opens with a pixel norm that
            # cancels any scaling of `z`, so the push arrives after the mapping.
            self._hand_over(offset)
            offset = None
        # Seeds are shared across widths; the arc between them is each model's own.
        if view.dtype == np.float32:
            np.dot(self._coef, self._pair, out=view)
            if offset is not None:
                view += offset
        else:
            np.dot(self._coef, self._pair, out=self._scratch)
            if offset is not None:
                self._scratch += offset
            view[:] = self._scratch
        if self.cfg.latent_on_host:
            # Already the right numbers in the right place: the graph takes a host array and `infer`
            # widens it to f32 and copies, so nothing here outlives its slot.
            return view
        return self._staging[self._slot].to(self.device, non_blocking=True)

    def _hand_over(self, offset) -> None:
        """Put a `w` push where the generator reads it, and only when it changed."""
        if offset is self._pushed:
            return
        into = self.cfg.push_into
        if into.device.type == "cpu":
            # A captured generator reads the push from this host twin: a numpy write, no torch
            # small-tensor overhead and nothing issued to the card.
            view = into.numpy().reshape(-1)
            view[:] = 0.0 if offset is None else offset.reshape(-1)
        elif offset is None:
            into.zero_()
        else:
            into.copy_(torch.from_numpy(offset).reshape(into.shape))
        self._pushed = offset

    def _offset(self):
        """The summed direction push, or `None` when every dial is at rest."""
        rows = self.cfg.directions
        amounts = self.cfg.amounts
        if rows is None or not any(amounts):
            self._offset_key = self._offset_rows = self._offset_vec = None
            return None
        # A push folded into the latent has to be the latent's width, and this asks rather than assumes: a
        # mismatch means the walk is halfway between two models, which killed the process with a broadcast
        # error.
        if self.cfg.push_into is None and getattr(rows, "shape", (0,))[-1] != self.nz:
            return None
        # The basis is part of the key, not just the amounts: a switch does not move the dials, so
        # keying on amounts alone returned the previous generator's push. The array, not `id()`.
        amounts = tuple(amounts)
        if amounts != self._offset_key or rows is not self._offset_rows:
            n = min(len(amounts), len(rows))
            self._offset_vec = np.asarray(amounts[:n], dtype=np.float32) @ rows[:n]
            self._offset_key, self._offset_rows = amounts, rows
        return self._offset_vec

    def seconds_per_segment(self, bpm: float) -> float:
        return self.cfg.beats_per_segment * 60.0 / max(bpm, 1e-6)


class BeatDriver:
    """Adapts the beat-locked walk to the frame-counted seam the recorder already has."""

    def __init__(self, walk: SlerpWalk, clock: MusicalClock, fps: float) -> None:
        self.walk, self.clock, self.fps = walk, clock, float(fps)

    def next_z(self, _step: int):
        """`_step` is ignored: this driver's position comes from the clock, not a frame count."""
        self.clock.advance(1.0 / self.fps)
        return self.walk.latent(self.clock.beats)
