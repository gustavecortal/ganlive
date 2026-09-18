"""The twelve-track drum vocabulary, and a stand-in machine that speaks it.

Track names are the usual abbreviations -- bass drum, snare, rim shot, clap, four toms,
two hats, cymbal, cowbell -- and an Analog Rytm's twelve pads are exactly this order.
Any machine is wired to it by `parse_notes` or `parse_track_channels`."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

TRACKS = ("BD", "SD", "RS", "CP", "BT", "LT", "MT", "HT", "CH", "OH", "CY", "CB")
INDEX = {t: i for i, t in enumerate(TRACKS)}

VOICE_GROUPS = (("BD",), ("SD",), ("RS", "CP"), ("BT",), ("LT",), ("MT", "HT"),
                ("CH", "OH"), ("CY", "CB"))

STEPS_PER_BAR = 16

LONGEST_VOICE_S = 2.0


def _LONGEST_VOICE_SAMPLES(sr: int) -> int:
    return int(LONGEST_VOICE_S * sr)


def channel_map(layout: str = "tracks") -> dict[str, int]:
    """Which audio channel each drum arrives on."""
    if layout == "tracks":
        return dict(INDEX)
    if layout == "voices":
        return {track: i for i, group in enumerate(VOICE_GROUPS) for track in group}
    raise ValueError(f"unknown layout {layout!r}; have tracks, voices")


def track_index(name: str) -> int:
    """A track to its index, or raise. Its name, or the index itself for a
    machine whose pads are not called BD and SD -- `0` to `11`, the order the twelve tracks
    are wired in. Every parser here validates the same way."""
    name = name.strip().upper()
    if name.isdigit():
        index = int(name)
        if not 0 <= index < len(TRACKS):
            raise ValueError(f"track {name} is out of range; have 0 to {len(TRACKS) - 1}")
        return index
    if name not in INDEX:
        raise ValueError(f"unknown track {name!r}; have {', '.join(TRACKS)}, or 0 to "
                         f"{len(TRACKS) - 1}")
    return INDEX[name]


def parse_channel_map(text: str) -> dict[str, int]:
    """"BD=0,CH=4" to a map. Track names must be real ones, or a typo is a silent no-op."""
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, channel = part.partition("=")
        track_index(name)               # validated, but this map is by NAME
        out[name.strip().upper()] = int(channel)
    return out


def parse_notes(text: str) -> dict[int, int]:
    """Which note is which track, for a kit that shares one channel: `{note: track index}`.

    `"0"` -- a first note, for a kit whose pads send consecutive notes from there; a Rytm's
    send 0 to 11. Or `"36=BD,38=SD,42=CH,46=OH"` for one whose do not, which is every General
    MIDI kit: kick 36, snare 38, closed hat 42, open hat 46, and nothing consecutive about it.
    Same grammar as `parse_track_channels`, with a note where the channel was."""
    text = text.strip()
    if text.isdigit():
        return {int(text) + i: i for i in range(len(TRACKS))}
    out: dict[int, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        note, _, name = part.partition("=")
        number = int(note)
        if number in out:
            raise ValueError(f"note {number} is already {TRACKS[out[number]]}; two tracks on "
                             f"one note cannot be told apart")
        out[number] = track_index(name)
    if not out:
        raise ValueError("--notes needs a first note, as '0', or a map, as '36=BD,38=SD'")
    return out


def output_mode(seen: dict[int, set[int] | list[int] | dict[int, int]],
                known=range(len(TRACKS))) -> str | None:
    """`"auto"`, `"track"`, `"mixed"`, or None if the traffic cannot say. channel -> notes.

    `known` is the notes that name a track on this kit -- 0 to 11 unless told."""
    known = set(known)
    low = {ch for ch, notes in seen.items() if any(n in known for n in notes)}
    high = {ch for ch, notes in seen.items() if any(n not in known for n in notes)}
    high -= low                                   # a channel is classified by its lowest note
    if low and len(high) > 1:
        return "mixed"
    if low:
        return "auto"
    return "track" if len(high) > 1 else None


def parse_track_channels(text: str) -> dict[int, int]:
    """`"1-12"` or `"1=BD,2=SD"` to `{MIDI channel: track index}`."""
    out: dict[int, int] = {}
    text = text.strip()
    if "=" not in text:
        low, _, high = text.partition("-")
        first, span = int(low), int(high) - int(low) + 1
        if span != len(TRACKS):
            raise ValueError(f"{text!r} is {span} channels for {len(TRACKS)} tracks; "
                             f"the whole kit or an explicit map")
        return {first - 1 + i: i for i in range(len(TRACKS))}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        channel, _, name = part.partition("=")
        index = track_index(name)
        number = int(channel) - 1
        if number in out:
            raise ValueError(f"MIDI channel {number + 1} is already {TRACKS[out[number]]}; two "
                             f"tracks on one channel cannot be told apart")
        out[number] = index
    return out


def _env(n: int, decay: float, sr: int, hold: float = 0.0) -> np.ndarray:
    """Exponential decay with an optional flat hold, which is what a drum envelope is."""
    t = np.arange(n) / sr
    e = np.exp(-np.maximum(0.0, t - hold) / max(decay, 1e-4))
    return e.astype(np.float32)


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _band(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    """Keep `lo`..`hi` Hz, with a soft edge. One numpy call, where scipy was a whole package.

    These shape the noise in the stand-in kit -- a hi-hat, a snare's snap. What matters is
    which octaves survive, not the roll-off, so the exact shape of a Butterworth buys nothing
    here and cost a dependency larger than everything else combined."""
    n = x.size
    if n == 0:
        return x
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # A raised-cosine edge an octave wide at each end, so a hit does not ring on a brick wall.
    gain = np.ones_like(freqs)
    if lo > 0:
        gain *= np.clip(np.log2(np.maximum(freqs, 1e-6) / lo) + 1.0, 0.0, 1.0)
    if hi < sr / 2:
        gain *= np.clip(np.log2(hi / np.maximum(freqs, 1e-6)) + 1.0, 0.0, 1.0)
    return np.fft.irfft(np.fft.rfft(x) * gain, n).astype(np.float32)


def _hp(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    return _band(x, cutoff, sr / 2, sr)


def _bp(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    return _band(x, lo, hi, sr)


def _swept_sine(n: int, f0: float, f1: float, sweep: float, sr: int) -> np.ndarray:
    """A sine whose pitch falls from `f0` to `f1` with time constant `sweep`."""
    t = np.arange(n) / sr
    f = f1 + (f0 - f1) * np.exp(-t / max(sweep, 1e-5))
    return np.sin(2 * np.pi * np.cumsum(f) / sr).astype(np.float32)


def _voice(track: str, vel: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """One hit, as a mono float32 array. Amplitudes are rough but the shapes are the point."""
    v = 0.2 + 0.8 * vel

    if track == "BD":
        n = int(0.45 * sr)
        body = _swept_sine(n, 190.0, 47.0, 0.022, sr) * _env(n, 0.16, sr, hold=0.01)
        click = _hp(_noise(n, rng), 1200, sr) * _env(n, 0.0016, sr) * 0.5
        return (body * 0.95 + click) * v

    if track == "SD":
        n = int(0.30 * sr)
        tone = (np.sin(2 * np.pi * 186 * np.arange(n) / sr)
                + 0.7 * np.sin(2 * np.pi * 331 * np.arange(n) / sr)).astype(np.float32)
        tone *= _env(n, 0.075, sr)
        snap = _bp(_noise(n, rng), 900, 9000, sr) * _env(n, 0.085, sr)
        return (0.55 * tone + 0.75 * snap) * v

    if track == "RS":
        n = int(0.08 * sr)
        return (_bp(_noise(n, rng), 1400, 3200, sr) * _env(n, 0.012, sr)
                + 0.4 * np.sin(2 * np.pi * 1720 * np.arange(n) / sr).astype(np.float32)
                * _env(n, 0.010, sr)) * v

    if track == "CP":
        n = int(0.42 * sr)
        out = np.zeros(n, dtype=np.float32)
        src = _bp(_noise(n, rng), 700, 5200, sr)
        for k, off in enumerate((0.0, 0.0095, 0.019)):
            i = int(off * sr)
            seg = src[: n - i] * _env(n - i, 0.0055, sr) * (1.0 - 0.18 * k)
            out[i:] += seg
        out += src * _env(n, 0.115, sr) * 0.42
        return out * 0.8 * v

    if track in ("BT", "LT", "MT", "HT"):
        f0, f1, dec = {"BT": (150, 62, 0.28), "LT": (196, 84, 0.24),
                       "MT": (262, 116, 0.20), "HT": (344, 158, 0.17)}[track]
        n = int((dec * 4) * sr)
        body = _swept_sine(n, f0, f1, 0.055, sr) * _env(n, dec, sr)
        skin = _hp(_noise(n, rng), 2000, sr) * _env(n, 0.006, sr) * 0.22
        return (body + skin) * v

    if track in ("CH", "OH"):
        dec = 0.032 if track == "CH" else 0.34
        n = int(max(0.12, dec * 4.5) * sr)
        t = np.arange(n) / sr
        metal = np.zeros(n, dtype=np.float32)
        for f in (2380.0, 3140.0, 4270.0, 5630.0, 7180.0, 8890.0):
            metal += np.sign(np.sin(2 * np.pi * f * t)).astype(np.float32)
        metal = _bp(metal / 6.0, 5800, 13000, sr)
        return metal * _env(n, dec, sr) * 0.85 * v

    if track == "CY":
        n = int(1.5 * sr)
        t = np.arange(n) / sr
        metal = np.zeros(n, dtype=np.float32)
        for f in (1180.0, 1670.0, 2410.0, 3320.0, 4710.0, 6180.0, 8330.0):
            metal += np.sign(np.sin(2 * np.pi * f * t)).astype(np.float32)
        body = _bp(metal / 7.0, 2600, 12000, sr) * _env(n, 0.62, sr)
        wash = _hp(_noise(n, rng), 4000, sr) * _env(n, 0.34, sr) * 0.35
        return (body + wash) * 0.7 * v

    if track == "CB":
        n = int(0.34 * sr)
        t = np.arange(n) / sr
        tone = (np.sign(np.sin(2 * np.pi * 541 * t))
                + 0.8 * np.sign(np.sin(2 * np.pi * 812 * t))).astype(np.float32)
        return _bp(tone / 1.8, 480, 4200, sr) * _env(n, 0.13, sr) * 0.8 * v

    raise ValueError(f"unknown track {track!r}")


@dataclass
class Section:
    """`bars` of one pattern. `steps` maps a track to 16 velocities, 0 meaning no trig."""

    name: str
    bars: int
    steps: dict[str, list[float]] = field(default_factory=dict)


def _s(pattern: str, vel: float = 1.0, accent: float = 1.0) -> list[float]:
    """`'x...x...x...x...'` -> velocities. `X` is accented, `o` is a ghost note."""
    out = []
    for c in pattern[:STEPS_PER_BAR].ljust(STEPS_PER_BAR, "."):
        out.append({".": 0.0, "x": vel, "X": min(1.0, vel * accent), "o": vel * 0.42}[c])
    return out


FOUR_ON_THE_FLOOR = [
    Section("intro", 4, {
        "BD": _s("X...x...X...x..."),
        "CH": _s("..x...x...x...x.", 0.55),
    }),
    Section("build", 4, {
        "BD": _s("X...x...X...x..."),
        "CH": _s("..x...x...x...x.", 0.62),
        "OH": _s("....x.......x...", 0.55),
        "CP": _s("....X.......X..."),
    }),
    Section("drop", 4, {
        "BD": _s("X...x...X...x..."),
        "CH": _s("x.x.x.x.x.x.x.x.", 0.7, 1.3),
        "OH": _s("....x.......x..x", 0.7),
        "CP": _s("....X.......X..."),
        "SD": _s("..........o....o", 0.8),
        "CY": _s("X..............."),
    }),
    Section("break", 4, {
        "CH": _s("..x...x...x...x.", 0.5),
        "RS": _s("x...x.x...x.x..."),
        # The stand-in plays all twelve voices, so a silent channel in the end-of-run report
        # is always a fault. Without this one BT read silent on every simulated run, and the
        # line under it sent the player off to check send levels on a machine that was not there.
        "BT": _s("x.......x.......", 0.75),
        "LT": _s("........x.......", 0.8),
        "MT": _s("............x...", 0.8),
        "HT": _s("..............x.", 0.8),
        "CB": _s("....x.......x...", 0.6),
    }),
]


@dataclass
class Rendered:
    """Everything one render produced, in one object so nothing can drift out of step."""

    stems: np.ndarray               # (12, N) float32, one row per track
    mix: np.ndarray                 # (2, N) float32, stereo
    events: list[tuple[float, str, float]]      # (seconds, track, velocity), the ground truth
    samplerate: int
    bpm: float
    bars: int

    @property
    def seconds(self) -> float:
        return self.stems.shape[1] / self.samplerate

    def stems_for(self, channel_of: dict[str, int]) -> np.ndarray:
        """The take mixed down to the channels a given map addresses."""
        if not channel_of:
            return self.stems
        out = np.zeros((max(channel_of.values()) + 1, self.stems.shape[1]), dtype=np.float32)
        for track, channel in channel_of.items():
            if track in INDEX:
                out[channel] += self.stems[INDEX[track]]
        return out


class StemFeeder(threading.Thread):
    """Pushes rendered audio into an extractor at wall-clock rate, like the sound card will."""

    daemon = True

    def __init__(self, extractor, pcm, samplerate: int, blocksize: int) -> None:
        super().__init__()
        self.ex, self.pcm, self.sr, self.bs = extractor, pcm, int(samplerate), int(blocksize)
        if self.pcm.shape[1] < self.bs:
            raise ValueError(f"{self.pcm.shape[1]} samples is shorter than one {self.bs}-frame "
                             f"block; the feeder would spin without ever pushing")
        self.stop_flag = False
        self.calls = 0
        self.push_ms: deque[float] = deque(maxlen=20_000)
        self.late = 0

    def run(self) -> None:
        period = self.bs / self.sr
        n = self.pcm.shape[1]
        i = 0
        nxt = time.perf_counter() + period
        while not self.stop_flag:
            block = self.pcm[:, i:i + self.bs]
            if block.shape[1] < self.bs:
                i = 0
                continue
            i = (i + self.bs) % max(1, n - self.bs)
            t0 = time.perf_counter()
            self.ex.push(block)
            self.push_ms.append((time.perf_counter() - t0) * 1000)
            self.calls += 1
            slack = nxt - time.perf_counter()
            nxt += period
            if slack > 0:
                time.sleep(slack)
            else:
                self.late += 1

    def stop(self, timeout: float = 1.0) -> None:
        self.stop_flag = True
        self.join(timeout=timeout)

    def close(self) -> None:
        """Nothing to release -- a thread is not a device handle."""


def _span(pcm: np.ndarray, at: int, end: int) -> np.ndarray:
    """`pcm[:, at:end]`, wrapping round the end of the take."""
    n = pcm.shape[1]
    if end <= n:
        return pcm[:, at:end]
    return np.concatenate([pcm[:, at:], pcm[:, :end - n]], axis=1)


class MonitorFeeder:
    """The stand-in kit, played out of the speakers, measured from the same sample index."""

    def __init__(self, extractor, stems, mix, samplerate: int, blocksize: int,
                 device=None) -> None:
        self.ex, self.stems, self.mix = extractor, stems, mix
        self.sr, self.bs = int(samplerate), int(blocksize)
        self.device = device
        self.at = 0
        self.calls = 0
        self.late = 0
        self.push_ms: deque[float] = deque(maxlen=20_000)
        self.stream = None

    def start(self) -> None:
        import sounddevice as sd

        self.stream = sd.OutputStream(samplerate=self.sr, channels=self.mix.shape[0],
                                      blocksize=self.bs, dtype="float32",
                                      device=self.device, callback=self._block)
        self.stream.start()

    def _block(self, outdata, frames, _time, status) -> None:
        if status:
            self.late += 1
        end = self.at + frames
        mix, stems = _span(self.mix, self.at, end), _span(self.stems, self.at, end)
        self.at = end % self.mix.shape[1]
        outdata[:] = mix.T
        t0 = time.perf_counter()
        self.ex.push(stems)
        self.push_ms.append((time.perf_counter() - t0) * 1000)
        self.calls += 1

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class MachineSim:
    """Renders an arrangement to stems, a mix and a ground-truth event list."""

    def __init__(self, bpm: float = 130.0, samplerate: int = 48000, seed: int = 0,
                 sections: list[Section] | None = None, swing: float = 0.0,
                 humanise_ms: float = 0.0) -> None:
        self.bpm, self.sr = float(bpm), int(samplerate)
        self.sections = sections if sections is not None else FOUR_ON_THE_FLOOR
        self.rng = np.random.default_rng(seed)
        self.swing = swing
        self.humanise_ms = humanise_ms

    @property
    def step_seconds(self) -> float:
        return 60.0 / self.bpm / (STEPS_PER_BAR / 4)

    def render(self, bars: int | None = None, tail: float = 1.0) -> Rendered:
        total_bars = bars if bars is not None else sum(s.bars for s in self.sections)
        n = int((total_bars * STEPS_PER_BAR * self.step_seconds + tail) * self.sr)
        stems = np.zeros((len(TRACKS), n), dtype=np.float32)
        events: list[tuple[float, str, float]] = []

        fired: set[int] = set()
        group_of = {t: i for i, g in enumerate(VOICE_GROUPS) for t in g}

        bar = 0
        for section in self.sections:
            for _ in range(section.bars):
                if bar >= total_bars:
                    break
                for step in range(STEPS_PER_BAR):
                    t = (bar * STEPS_PER_BAR + step) * self.step_seconds
                    if self.swing and step % 2 == 1:
                        t += self.swing * self.step_seconds
                    if self.humanise_ms:
                        t += self.rng.normal(0.0, self.humanise_ms / 1000.0)
                    i0 = max(0, int(t * self.sr))
                    for track, vels in section.steps.items():
                        vel = vels[step]
                        if vel <= 0.0:
                            continue
                        vel = float(np.clip(vel * self.rng.normal(1.0, 0.06), 0.05, 1.0))
                        g = group_of[track]
                        if g in fired:
                            stop = min(n, i0 + _LONGEST_VOICE_SAMPLES(self.sr))
                            for other in VOICE_GROUPS[g]:
                                stems[INDEX[other], i0:stop] = 0.0
                        fired.add(g)
                        hit = _voice(track, vel, self.sr, self.rng)
                        end = min(n, i0 + hit.shape[0])
                        stems[INDEX[track], i0:end] += hit[: end - i0]
                        events.append((t, track, vel))
                bar += 1

        seen: dict[tuple[int, int], int] = {}
        for i, (t, track, _v) in enumerate(events):
            key = (group_of[track], int(t * self.sr))
            seen[key] = i                                # later entry wins, as the choke does
        keep = set(seen.values())                        # built once, not once per event
        events = [e for i, e in enumerate(events) if i in keep]

        events.sort()
        mix = self._mix(stems)
        return Rendered(stems=stems, mix=mix, events=events, samplerate=self.sr,
                        bpm=self.bpm, bars=total_bars)

    def _mix(self, stems: np.ndarray) -> np.ndarray:
        """Stereo mix with per-track level and pan, then a limiter."""
        gain = {"BD": 1.0, "SD": 0.72, "RS": 0.5, "CP": 0.66, "BT": 0.6, "LT": 0.6,
                "MT": 0.58, "HT": 0.56, "CH": 0.42, "OH": 0.46, "CY": 0.4, "CB": 0.44}
        pan = {"BD": 0.0, "SD": 0.0, "RS": -0.35, "CP": 0.2, "BT": -0.15, "LT": -0.3,
               "MT": 0.1, "HT": 0.3, "CH": 0.25, "OH": -0.2, "CY": 0.4, "CB": -0.4}
        weights = np.empty((2, len(INDEX)), dtype=np.float32)
        for track, i in INDEX.items():
            p = (pan[track] + 1.0) / 2.0                   # 0 left, 1 right
            weights[0, i] = gain[track] * np.sqrt(1.0 - p)
            weights[1, i] = gain[track] * np.sqrt(p)
        out = weights @ stems
        peak = float(np.abs(out).max())
        if peak > 0.89:
            out *= 0.89 / peak
        return out
