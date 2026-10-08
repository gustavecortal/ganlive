"""Lining a recorded take up with the multitrack in a DAW.

`Guide` writes, beside each take, a mono WAV of the audio input and a JSON sidecar marking
where every bar line fell in both the video and the audio, so the two can be aligned.
"""
from __future__ import annotations

import math
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

from ganlive.clock import BEATS_PER_BAR
from ganlive.files import write_json

#: How many audio blocks may wait for the writer thread before new ones are dropped.
DEPTH = 64

#: Float audio to 16-bit PCM.
FULL_SCALE = 32767.0

#: A guide whose peak stays below this is reported as silent.
SILENT = 1e-4


class Guide:
    """The mono guide track and the beat map for one take.

    The audio thread only copies each block into a queue; a writer thread does the disk work.
    `position` counts every sample the input has delivered, recording or not, so a take's
    start can be given as a sample index into the stream."""

    def __init__(self, depth: int = DEPTH) -> None:
        self.sr = 0
        self.channels = 0
        self.position = 0
        self.attached = False

        self._q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._thread: threading.Thread | None = None
        self._path: Path | None = None
        self._reset_take()

    def _reset_take(self) -> None:
        """Clear everything one take accumulates."""
        self._about: dict = {}
        self._head: dict = {}
        self._marks: list[dict] = []
        self._next_mark = 0.0
        self._t0 = 0.0
        self._first: int | None = None
        self._first_wall: float | None = None
        self._samples = 0
        self._peak = 0.0
        self.blocks = self.dropped = 0
        self._error: str | None = None

    def listen_to(self, extractor) -> None:
        """Take every block this extractor is pushed, and its samplerate and channel count."""
        if getattr(extractor, "tap", None) is not None:
            raise RuntimeError("that extractor already has a tap, and it holds only one")
        self.sr, self.channels = int(extractor.sr), int(extractor.n)
        extractor.tap = self.push
        self.attached = True

    def push(self, block: np.ndarray) -> None:
        """One block of input, from the audio thread. A copy and a queue put, and no more."""
        at = self.position
        self.position = at + block.shape[-1]
        if self._path is None:
            return
        try:
            self._q.put_nowait((at, time.perf_counter(), block.copy()))
        except queue.Full:
            self.dropped += 1
            return
        self.blocks += 1

    @property
    def running(self) -> bool:
        return self._path is not None

    @property
    def wav_path(self) -> Path:
        return self._path.with_suffix(".wav")

    @property
    def sidecar_path(self) -> Path:
        return self._path.with_suffix(".json")

    def start(self, take: Path, *, bpm: float, beat: float, beat_source: str,
              at: float | None = None, about: dict | None = None) -> str:
        """Begin a guide beside `take`, and say in one line what will be written."""
        self._path = Path(take)
        self._reset_take()
        self._head = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "bpm": round(float(bpm), 3), "beat": round(float(beat), 4),
                      "beat_source": beat_source}
        self._about = dict(about or {})
        self._next_mark = math.floor(float(beat) / BEATS_PER_BAR + 1) * BEATS_PER_BAR
        self._t0 = time.perf_counter() if at is None else float(at)
        if not self.attached:
            return f"{self.sidecar_path.name} -- no audio input, so bar lines only"
        self._thread = threading.Thread(target=self._run, name=f"guide {self._path.stem}",
                                        daemon=True)
        self._thread.start()
        return f"{self.sidecar_path.name} and {self.wav_path.name}"

    def mark(self, beat: float, at: float | None = None) -> None:
        """Called every frame; records a mark only when a bar line has been crossed.

        `at` is the frame's own time (`time.perf_counter`), when it is worked out after it was
        due, as a take catching up does: the audio that arrived since is taken off. The next bar
        line is recomputed from wherever the clock is now, so a transport restart (beats back
        to 0) or a song-position jump keeps marking, once per bar."""
        if self._path is None:
            return
        if beat >= self._next_mark:
            now = time.perf_counter()
            at = now if at is None else at
            audio = self._audio_seconds()
            self._marks.append({"beat": round(beat, 4),
                                "video_s": round(at - self._t0, 4),
                                "audio_s": None if audio is None else round(audio - (now - at), 4)})
        self._next_mark = math.floor(beat / BEATS_PER_BAR + 1) * BEATS_PER_BAR

    def _audio_seconds(self) -> float | None:
        """Seconds of input since the guide's first sample, or None when there is no audio."""
        first = self._first
        return None if first is None else round((self.position - first) / self.sr, 4)

    def stop(self, video: dict | None = None) -> dict:
        """Close the guide, write the sidecar, and hand back what is in it."""
        if self._path is None:
            return {}
        if self._thread is not None:
            try:
                self._q.put(None, timeout=5.0)
            except queue.Full:                             # pragma: no cover -- writer wedged
                self._error = "the writer never drained; the guide may be short"
            self._thread.join(timeout=5.0)
            self._thread = None
        report = self.report(video)
        write_json(self.sidecar_path, report)
        self._path = None
        return report

    def report(self, video: dict | None = None) -> dict:
        """Everything the sidecar says. Separate from `stop` so a test can read it directly."""
        out = {"take": self._path.name, **self._head, "about": self._about}
        if video is not None:
            out["video"] = video
        if self.attached:
            out["audio"] = {
                "guide": self.wav_path.name,
                "samplerate": self.sr,
                "channels": self.channels,
                "start_sample": self._first,
                "offset_ms": (None if self._first_wall is None
                              else round((self._first_wall - self._t0) * 1000, 3)),
                "seconds": round(self._samples / self.sr, 3),
                "peak": round(self._peak, 4),
                "blocks": self.blocks,
                "dropped_blocks": self.dropped,
                "stream_samples": self.position,
            }
            if self._error:
                out["audio"]["error"] = self._error
        else:
            out["audio"] = None
        out["marks"] = self._marks
        out["drift_ms"] = self._drift_ms()
        return out

    def _drift_ms(self) -> float | None:
        """How far the frame clock and the audio clock have come apart, at the last mark."""
        if not self._marks or self._marks[-1]["audio_s"] is None or self._first_wall is None:
            return None
        last = self._marks[-1]
        offset = self._first_wall - self._t0
        return round((last["video_s"] - last["audio_s"] - offset) * 1000, 3)

    def describe(self, report: dict) -> str:
        """One line about what was written, for the tool that wrote it."""
        bars = f"{len(report['marks'])} bar(s) marked"
        drift = report.get("drift_ms")
        if drift is not None:
            bars += f", drift {drift:+.1f} ms"
        audio = report.get("audio")
        if audio is None:
            return f"{bars}. No audio input, so there is no guide track -- match on the bars."
        if not audio["blocks"]:
            why = ("the stream is open and delivered nothing" if not audio["stream_samples"]
                   else f"the stream delivered {audio['stream_samples']} samples and the take "
                        f"kept none of them, so it was not running while they arrived")
            return f"{bars}. NO AUDIO REACHED THE GUIDE: the file is empty -- {why}."
        line = (f"{audio['guide']}  {audio['seconds']:.1f}s, peak {audio['peak']:.3f}, "
                f"starts {audio['offset_ms']:+.1f} ms from the first frame "
                f"at stream sample {audio['start_sample']}. {bars}.")
        if audio["peak"] < SILENT:
            line += " It is SILENT -- nothing to waveform-match against."
        if audio["dropped_blocks"]:
            line += (f" {audio['dropped_blocks']} block(s) dropped, so there are gaps: "
                     f"the writer could not keep up.")
        if "error" in audio:
            line += f" The writer failed: {audio['error']}"
        return line

    def _run(self) -> None:
        """The writer thread: each queued block, mixed to mono, into a 16-bit WAV, until the
        `None` that `stop` sends. On a failure it keeps draining so the audio thread never
        blocks."""
        wav = self.wav_path
        wav.parent.mkdir(parents=True, exist_ok=True)
        handle = wave.open(str(wav), "wb")
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(self.sr)
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                at, when, block = item
                if self._first is None:
                    # `when` is when the block arrived, which is when its LAST sample did.
                    self._first, self._first_wall = at, when - block.shape[-1] / self.sr
                mono = block.mean(axis=0) if block.ndim > 1 else block
                peak = float(np.abs(mono).max()) if mono.size else 0.0
                if peak > self._peak:
                    self._peak = peak
                handle.writeframesraw((mono * FULL_SCALE).astype("<i2"))
                self._samples += mono.shape[-1]
        except BaseException as exc:                # noqa: BLE001  reported, not swallowed
            self._error = f"{type(exc).__name__}: {exc}"
            while self._q.get() is not None:
                pass
        finally:
            handle.close()
