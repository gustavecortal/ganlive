"""Audio in, per-frame control features out. One implementation for offline and live."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

NEVER = 1e6

FULL_SCALE_HIT = 0.55


@dataclass
class FeatureConfig:
    """Thresholds for the follower and the onset detector."""

    hop: int = 128
    release: float = 0.12
    floor: float = 0.012
    onset_ratio: float = 1.9
    refractory: float = 0.018
    energy_window: float = 1.85


class FeatureExtractor:
    """Per-channel envelope, latched onsets, and a slow global energy."""

    def __init__(self, channels: int, samplerate: int,
                 config: FeatureConfig | None = None) -> None:
        self.n = channels
        self.sr = int(samplerate)
        self.cfg = config or FeatureConfig()
        h = self.cfg.hop / self.sr
        self._decay = float(np.exp(-h / max(self.cfg.release, 1e-4)))
        self._energy_decay = float(np.exp(-h / max(self.cfg.energy_window, 1e-4)))
        self._refractory_hops = max(1, int(self.cfg.refractory * self.sr / self.cfg.hop))

        self.env = np.zeros(channels, dtype=np.float32)
        self.peak = np.zeros(channels, dtype=np.float32)
        self.since = np.full(channels, NEVER, dtype=np.float32)
        self.pending: list[tuple[int, float, float]] = []
        self.energy = 0.0
        self.density = 0.0
        self.active = 0.0
        self.hops = 0
        self.hits = np.zeros(channels, dtype=np.int64)
        self._last_onset = np.full(channels, -10 ** 9, dtype=np.int64)
        self._tail = np.zeros((channels, 0), dtype=np.float32)
        self.tap = None

    def push(self, block: np.ndarray) -> None:
        """Consume audio. `block` is (channels, samples) or (samples, channels)."""
        x = np.asarray(block, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] != self.n and x.shape[1] == self.n:
            x = x.T                                      # sounddevice hands over (frames, ch)
        if x.shape[0] != self.n:
            raise ValueError(f"expected {self.n} channels, got {x.shape}")
        if self.tap is not None:
            self.tap(x)

        if self._tail.shape[1]:
            x = np.concatenate([self._tail, x], axis=1)
        hop = self.cfg.hop
        nhops = x.shape[1] // hop
        self._tail = x[:, nhops * hop:].copy()
        if nhops == 0:
            return

        blocks = x[:, : nhops * hop].reshape(self.n, nhops, hop)
        peaks = np.abs(blocks).max(axis=2)
        sq = np.einsum("chs,chs->ch", blocks, blocks)
        frame_rms = np.sqrt(sq.sum(axis=0) / hop)

        hop_s = hop / self.sr
        k = 1.0 - self._energy_decay
        rate = self.sr / hop
        np.maximum(self.peak, peaks.max(axis=1), out=self.peak)
        env, since = self.env.copy(), self.since.copy()
        for j in range(nhops):
            p = peaks[:, j]
            hit = ((p > self.cfg.floor)
                   & (p > self.cfg.onset_ratio * env)
                   & ((self.hops - self._last_onset) >= self._refractory_hops))
            np.multiply(env, self._decay, out=env)
            np.maximum(p, env, out=env)
            since += hop_s
            n_hit = np.count_nonzero(hit)
            if n_hit:
                since[hit] = 0.0
                ago = (nhops - 1 - j) * hop_s
                for c in np.flatnonzero(hit):
                    self.pending.append(
                        (int(c), min(1.0, float(p[c]) / FULL_SCALE_HIT), ago))
                    self._last_onset[c] = self.hops
                    self.hits[c] += 1
            self.energy = self.energy * self._energy_decay + float(frame_rms[j]) * k
            self.density = self.density * self._energy_decay + n_hit * rate * k
            self.active = self.active * self._energy_decay + (n_hit > 0) * k
            self.hops += 1
        self.env, self.since = env, since

    def played(self) -> int:
        """How many tracks have fired at all: whether the drums are reaching the instrument."""
        return int((self.since < NEVER * 0.1).sum())

    @property
    def floor(self) -> float:
        """The absolute level a hit must clear. Read beside `loudest`; see there."""
        return self.cfg.floor

    def loudest(self) -> list[float]:
        """The peak each channel ever carried, so a silent one can say WHY it was silent."""
        return [float(v) for v in self.peak]

    def heard(self) -> float:
        """Seconds of audio actually consumed, from the hop counter."""
        return self.hops * self.cfg.hop / self.sr

    def channel_of(self) -> dict[str, int] | None:
        """`track -> index into `since``, or None to mean "the caller's map is right"."""
        return None

    def features(self) -> dict[str, float]:
        """The whole-kit measurements a slow rule can be driven from."""
        return {"density": self.density, "energy": self.energy, "active": self.active}

    def drain(self) -> list[tuple[int, float, float]]:
        """Take every onset since the last call. Draining, not sampling: see the module note."""
        out, self.pending = self.pending, []
        return out


def offline(audio: np.ndarray, samplerate: int, fps: float,
            config: FeatureConfig | None = None) -> dict:
    """Run the extractor over a whole recording at a video frame rate."""
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


def score_onsets(detected: list[list[tuple[int, float, float]]], fps: float,
                 truth: list[tuple[float, str, float]], channel_of: dict[str, int],
                 tolerance: float = 0.030) -> dict:
    """Precision, recall and timing error against the simulator's ground truth."""
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


class NoteFeatures:
    """The same per-track features, from MIDI note-ons instead of from audio."""

    def __init__(self, tracks: int = 12, base_note: int = 0,
                 config: FeatureConfig | None = None,
                 channels: dict[int, int] | None = None,
                 notes: dict[int, int] | None = None) -> None:
        self.n = int(tracks)
        #: Which note is which track, when the kit shares one channel. `base_note` is the
        #: shorthand for a kit whose pads are consecutive from there; `notes` says it outright
        #: for one whose are not. See `kit.parse_notes`.
        self.notes = dict(notes) if notes else {int(base_note) + i: i for i in range(self.n)}
        self.channels = dict(channels) if channels else None
        self.unresolved = 0
        self.unclaimed: dict[tuple[int, int], int] = {}
        self.cfg = config or FeatureConfig()
        self.since = np.full(self.n, NEVER, dtype=np.float32)
        self.hits = np.zeros(self.n, dtype=np.int64)
        self.pending: list[tuple[int, float, float]] = []
        self.density = 0.0
        self.energy = 0.0
        self.active = 0.0
        self._last: float | None = None
        self._recent: deque[tuple[float, float]] = deque()
        self._now = 0.0

    def track_of(self, channel: int, note: int) -> int | None:
        """Which track a note-on IS, or None if it is not one of ours."""
        if self.channels is not None:
            index = self.channels.get(channel)
            if index is not None:
                return index if 0 <= index < self.n else None
        index = self.notes.get(note)
        return index if index is not None and 0 <= index < self.n else None

    def on_note(self, channel: int, note: int, velocity: int,
                when: float | None = None) -> int | None:
        """One trig or pad. Safe to call from the MIDI thread. Returns the track, or None."""
        index = self.track_of(channel, note)
        if index is None:
            self.unresolved += 1
            if len(self.unclaimed) < 32:            # a sample, not a log: this is a MIDI thread
                self.unclaimed[(channel, note)] = self.unclaimed.get((channel, note), 0) + 1
            return None
        self.on_track(index, min(1.0, velocity / 127.0), when)
        return index

    def on_track(self, index: int, strength: float, when: float | None = None) -> None:
        """One track fired, however that was learned."""
        now = self._now if when is None else when
        self.pending.append((index, strength, 0.0))
        self.since[index] = 0.0 if self._last is None else np.float32(self._last - now)
        self.hits[index] += 1
        self._recent.append((now, strength))

    def tick(self, now: float) -> None:
        """Advance time to `now`. Called once a frame, before anything reads `since`."""
        if self._last is not None:
            elapsed = max(0.0, now - self._last)
            if elapsed:
                self.since += np.float32(elapsed)
        self._last = self._now = now

        window = self.cfg.energy_window
        while self._recent and now - self._recent[0][0] > window:
            self._recent.popleft()
        self.density = len(self._recent) / window
        self.energy = (sum(v for _t, v in self._recent) / len(self._recent)
                       if self._recent else 0.0)
        self.active = float((self.since <= window).sum()) / max(1, self.n)

    def played(self) -> int:
        """How many tracks have fired at all. Same meaning as the audio source's."""
        return int((self.since < NEVER * 0.1).sum())

    def channel_of(self) -> dict[str, int]:
        """`track -> index into `since``, for the tracks this kit can actually reach.

        Derived from the wiring, not from the twelve names: a General MIDI kit wired with
        `--notes 36=BD,38=SD,42=CH,46=OH` reaches four. Returning all twelve gave the strip
        eight drum lights that could never fire and the end-of-run report eight silent
        tracks to complain about -- a fault that cannot exist."""
        from ganlive.control.kit import TRACKS

        reached = set(self.notes.values())
        if self.channels is not None:
            reached |= set(self.channels.values())
        return {TRACKS[i]: i for i in sorted(reached) if i < len(TRACKS)}

    def features(self) -> dict[str, float]:
        """The whole-kit measurements a slow rule can be driven from."""
        return {"density": self.density, "energy": self.energy, "active": self.active}

    def drain(self) -> list[tuple[int, float, float]]:
        """Take every hit since the last call. Draining, not sampling."""
        out, self.pending = self.pending, []
        return out


PAIRED_S = 0.075


class BothFeatures:
    """Pads by their MIDI note, sequencer trigs by their sound, in one twelve-track space."""

    def __init__(self, audio: FeatureExtractor, tracks: dict[str, int],
                 base_note: int = 0, config: FeatureConfig | None = None,
                 channels: dict[int, int] | None = None,
                 notes: dict[int, int] | None = None) -> None:
        self.audio = audio
        self.notes = NoteFeatures(len(tracks), base_note, config, channels, notes)
        self.n = self.notes.n
        index = self.notes.channel_of()
        self.on_channel: dict[int, list[int]] = {}
        for name, channel in tracks.items():
            if name in index:
                self.on_channel.setdefault(channel, []).append(index[name])
        self._voice_of = {i: ch for ch, idxs in self.on_channel.items() for i in idxs}
        self._noted: dict[int, float] = {}
        self._now = 0.0


    @property
    def since(self):
        return self.notes.since

    @property
    def hits(self):
        return self.notes.hits

    @property
    def unresolved(self) -> int:
        return self.notes.unresolved

    @property
    def unclaimed(self) -> dict[tuple[int, int], int]:
        return self.notes.unclaimed

    @property
    def sr(self) -> int:
        return self.audio.sr

    @property
    def tap(self):
        """The guide attaches here and reaches the audio half, which is the half with blocks."""
        return self.audio.tap

    @tap.setter
    def tap(self, fn) -> None:
        self.audio.tap = fn

    def push(self, block) -> None:
        self.audio.push(block)

    def on_note(self, channel: int, note: int, velocity: int,
                when: float | None = None) -> int | None:
        index = self.notes.on_note(channel, note, velocity, when)
        if index is not None:
            voice = self._voice_of.get(index)
            if voice is not None:
                self._noted[voice] = self._now if when is None else when
        return index

    def tick(self, now: float) -> None:
        """Fold this frame's audio onsets into the note space, then advance it."""
        self._now = now
        for channel, strength, ago in self.audio.drain():
            if now - self._noted.get(channel, -1e9) < PAIRED_S:
                continue                                  # a pad already reported this one
            for index in self.on_channel.get(channel, ()):
                self.notes.on_track(index, strength, now - ago)
        self.notes.tick(now)

    def drain(self):
        return self.notes.drain()

    def features(self) -> dict[str, float]:
        """The note half's counts, and the audio half's loudness."""
        out = self.notes.features()
        out["energy"] = self.audio.energy
        return out

    def played(self) -> int:
        return self.notes.played()

    def heard(self) -> float:
        return self.audio.heard()

    def loudest(self) -> list[float]:
        """The audio half's, in ITS channel space -- the only half that measures a level."""
        return self.audio.loudest()

    @property
    def floor(self) -> float:
        return self.audio.floor

    def channel_of(self) -> dict[str, int]:
        return self.notes.channel_of()
