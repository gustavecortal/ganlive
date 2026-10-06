"""Turning what the drums play into per-track features the presets read each frame.

Three sources with one interface: `FeatureExtractor` hears onsets in multichannel audio,
`NoteFeatures` reads MIDI note-ons, and `BothFeatures` combines them. Each keeps, per track,
the seconds since it last fired (`since`) and a queue of onsets (`drain`), plus three whole-kit
measurements: `density` (hits per second), `energy` (loudness) and `active`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ganlive.control.kit import TRACKS

#: `since` for a track that has never fired.
NEVER = 1e6

#: The peak level read as a full-strength hit, so audio onsets get a strength in [0, 1].
FULL_SCALE_HIT = 0.55


@dataclass
class FeatureConfig:
    """Thresholds for the envelope follower and the onset detector. Times are in seconds."""

    hop: int = 128
    release: float = 0.12
    floor: float = 0.012
    onset_ratio: float = 1.9
    refractory: float = 0.018
    energy_window: float = 1.85


class _Source:
    """The answers every source gives the same way, from `since`, `pending` and the three
    whole-kit measurements each keeps up to date."""

    def played(self) -> int:
        """How many tracks have fired at all: whether the drums are reaching ganlive."""
        return int((self.since < NEVER * 0.1).sum())

    def features(self) -> dict[str, float]:
        """The whole-kit measurements a slow rule (`presets.Macro`) can be driven from."""
        return {"density": self.density, "energy": self.energy, "active": self.active}

    def drain(self) -> list[tuple[int, float, float]]:
        """Take every onset since the last call, as `(track, strength, seconds ago)`.

        A queue rather than a snapshot, so two hits between frames are both seen."""
        out, self.pending = self.pending, []
        return out


class FeatureExtractor(_Source):
    """Onsets heard in audio: a per-channel envelope follower, latched onsets, and a slow
    global energy. `push` is called from the audio thread."""

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
        #: Called with every block pushed, before analysis; `record.sync.Guide` attaches here.
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
            # An onset: loud enough, well above the decaying envelope, and not too soon after
            # the last one on this channel.
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

    @property
    def floor(self) -> float:
        """The absolute level a hit must clear. Read beside `loudest`."""
        return self.cfg.floor

    def loudest(self) -> list[float]:
        """The peak each channel ever carried, so a report can tell a quiet send from none."""
        return [float(v) for v in self.peak]

    def heard(self) -> float:
        """Seconds of audio actually consumed, from the hop counter."""
        return self.hops * self.cfg.hop / self.sr

    def channel_of(self) -> dict[str, int] | None:
        """`track -> index into `since``, or None to mean "the caller's map is right"."""
        return None


class NoteFeatures(_Source):
    """The same per-track features, from MIDI note-ons instead of from audio."""

    def __init__(self, tracks: int = 12, channels: dict[int, int] | None = None,
                 notes: dict[int, int] | None = None) -> None:
        self.n = int(tracks)
        #: `{note: track}`, for a kit that shares one channel (see `kit.parse_notes`). Without
        #: it, note `i` is track `i`.
        self.notes = dict(notes) if notes else {i: i for i in range(self.n)}
        #: `{MIDI channel: track}`, for a kit that puts each track on its own channel. Checked
        #: first; a channel not in it falls through to the note.
        self.channels = dict(channels) if channels else None
        #: Note-ons that named no track: a count, and a small sample of `(channel, note)`.
        self.unresolved = 0
        self.unclaimed: dict[tuple[int, int], int] = {}
        self.cfg = FeatureConfig()
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
        """Which track a note-on is, or None if it is not one of ours."""
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

    def channel_of(self) -> dict[str, int]:
        """`track -> index into `since``, for the tracks this kit can actually reach.

        Derived from the wiring rather than the twelve names: a General MIDI kit wired with
        `--notes 36=BD,38=SD,42=CH,46=OH` reaches four, so the strip draws four drum lights
        and the end-of-run report checks four tracks."""
        reached = set(self.notes.values())
        if self.channels is not None:
            reached |= set(self.channels.values())
        return {TRACKS[i]: i for i in sorted(reached) if i < len(TRACKS)}


#: How close an audio onset may follow a note-on on the same voice and still be that hit.
PAIRED_S = 0.075


class BothFeatures:
    """Pads by their MIDI note, sequencer trigs by their sound, in one twelve-track space.

    For a machine whose sequencer does not send notes: pads still arrive as note-ons, and an
    audio onset on a voice's channel counts as a hit unless a note-on for it just arrived.
    `tracks` is `{track: audio channel}`."""

    def __init__(self, audio: FeatureExtractor, tracks: dict[str, int],
                 channels: dict[int, int] | None = None,
                 notes: dict[int, int] | None = None) -> None:
        self.audio = audio
        self.notes = NoteFeatures(len(tracks), channels, notes)
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
        """The audio half's tap: the half that has audio blocks to hand on."""
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
        """The audio half's, in its channel space -- the only half that measures a level."""
        return self.audio.loudest()

    @property
    def floor(self) -> float:
        return self.audio.floor

    def channel_of(self) -> dict[str, int]:
        return self.notes.channel_of()
