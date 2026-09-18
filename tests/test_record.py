"""Getting a take out: video, stills, and lining up with a DAW."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from ganlive.presets import Impulse, Macro, Preset  # noqa: E402
from ganlive.walk import (
    MusicalClock,
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

from tests.support import _drained, _pulses  # noqa: E402


class _TapSource:
    """The shape `Guide.listen_to` reads: a samplerate, a channel count and a free tap slot."""

    tap = None

    def __init__(self, n=10, sr=48000):
        self.n, self.sr = n, sr


def _guide_blocks(guide, count, channels=10, frames=256, level=0.5):
    """Push `count` blocks of the shape an ASIO callback hands over."""
    block = np.full((channels, frames), level, dtype=np.float32)
    for _ in range(count):
        guide.push(block)


def test_changing_tempo_mid_performance_moves_the_position_immediately():
    """Turning the tempo encoder must retime the video, and exactly."""
    c = MusicalClock(128.0)
    t = _pulses(c, 128.0, 4 * 4 * 60 / 128)
    _pulses(c, 160.0, 4 * 4 * 60 / 160, t0=t)
    assert c.beats == pytest.approx(32.0, abs=1e-9)


def test_a_take_drops_frames_rather_than_making_the_picture_wait():
    """x264 does not encode six megapixels at 60 fps here, so a blocking put would pace the
    whole instrument to the encoder. No thread is started: this is the queue policy alone."""
    import numpy as np

    from ganlive.record.video import Recorder

    rec = Recorder("unused.mp4", 8, 8, 60.0, realtime=True, depth=2)
    plane = np.zeros((12, 8), dtype=np.uint8)
    assert rec.offer(plane) is True
    assert rec.offer(plane) is True
    assert rec.offer(plane) is False, "a full queue blocked instead of dropping"
    assert rec.dropped == 1 and rec.offered == 3

    slow = Recorder("unused.mp4", 8, 8, 60.0, depth=2)
    assert [slow.offer(plane) for _ in range(2)] == [True, True]
    assert slow.dropped == 0


def test_a_dropped_frame_costs_the_take_a_held_frame_not_its_length():
    """With timestamps counted rather than clocked, dropped frames make the file play back
    faster than it was played -- fifteen minutes of performance as eleven minutes of video,
    which is the kind of wrong that gets believed."""
    import time

    import numpy as np

    from ganlive.record.video import Recorder

    rec = Recorder("unused.mp4", 8, 8, 60.0, realtime=True, depth=64)
    plane = np.zeros((12, 8), dtype=np.uint8)
    rec.offer(plane)
    time.sleep(0.1)                        # six frames of wall clock at 60 fps
    rec.offer(plane)
    stamps = [rec._q.get()[0] for _ in range(2)]
    assert stamps[0] == 0
    assert stamps[1] >= 4, f"a 100 ms gap became {stamps[1]} frames"

    counted = Recorder("unused.mp4", 8, 8, 60.0, depth=64)
    counted.offer(plane)
    time.sleep(0.05)
    counted.offer(plane)
    assert [counted._q.get()[0] for _ in range(2)] == [0, 1]


def test_a_take_is_a_playable_file_with_a_trailer(tmp_path):
    import numpy as np

    from ganlive.record.video import Recorder

    out = tmp_path / "take.mp4"
    rec = Recorder(out, 128, 64, 30.0, "libx264", realtime=True, depth=8).start()
    rng = np.random.default_rng(0)
    for _ in range(20):
        rec.offer(np.ascontiguousarray(rng.integers(0, 255, (96, 128), dtype=np.uint8)))
    report = rec.stop()
    assert "error" not in report, report
    assert report["frames"] == 20 - report["dropped"], report
    assert report["offered"] == 20, report
    assert out.exists() and out.stat().st_size > 0

    import av

    with av.open(str(out)) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded == report["frames"], (decoded, report)


def test_a_still_is_written_by_the_time_its_thread_is_joined(tmp_path):
    import numpy as np

    from ganlive.record.video import save_still

    rgb = np.zeros((16, 24, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    path = tmp_path / "shot.png"
    save_still(path, rgb).join(timeout=20.0)

    from PIL import Image

    with Image.open(path) as img:
        assert img.size == (24, 16)
        assert img.convert("RGB").getpixel((0, 0)) == (200, 0, 0)


def test_serial_names_do_not_collide_and_sort_in_the_order_they_were_made(tmp_path):
    from ganlive.record.video import next_path

    made = []
    for _ in range(3):
        path = next_path(tmp_path, "take", ".mp4")
        path.write_bytes(b"")
        made.append(path)
    assert [p.name for p in made] == ["take-01.mp4", "take-02.mp4", "take-03.mp4"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [p.name for p in made]


def test_the_encoder_can_be_made_to_open_before_the_clock_starts(tmp_path):
    """An encoder is not opened by `add_stream` -- it opens on the first frame it is given, and
    `av1_qsv` takes about a second and a half to do it while holding the GIL, so the thread
    producing frames stops wherever it happens to be. That is why the frame timings
    under-reported it by five: it lands between two frames as often as inside one. `drain` is
    how a caller pays it before the clock starts."""
    import numpy as np

    from ganlive.record.video import Recorder

    rec = Recorder(tmp_path / "warm.mp4", 64, 64, 30.0, "libx264",
                   realtime=True, depth=4).start()
    plane = np.zeros((96, 64), dtype=np.uint8)
    assert rec.offer(plane) is True
    assert rec.drain(timeout=30.0) is True, "the writer never caught up"
    assert rec.written == 1, rec.written

    rec.offer(plane)
    assert rec.drain(timeout=30.0) is True
    report = rec.stop()
    assert report["frames"] == 2 and report["offered"] == 2, report


def test_a_guide_is_the_length_that_was_played_and_cannot_clip(tmp_path):
    """**The one distortion that matters here is clipping**, because flattened peaks are
    exactly what a waveform match reads. Every channel is in [-1, 1] so their mean is too, and
    that is why this sums by averaging rather than adding: ten channels at full scale is the
    case where the obvious version writes a square wave and the file still looks fine."""
    import wave

    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=10))
    guide.start(tmp_path / "take-01.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    _guide_blocks(guide, 20, level=1.0)                   # every channel pinned at full scale
    _guide_blocks(guide, 20, level=-1.0)                  # ...and at the other end of it
    report = _drained(guide).stop()

    with wave.open(str(tmp_path / "take-01.wav")) as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 48000
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    assert len(pcm) == 40 * 256, f"{len(pcm)} samples for 40 blocks of 256"
    assert (pcm.max(), pcm.min()) == (32767, -32767), "the mean of ten full-scale channels " \
        "did not land on full scale, or wrapped past it"
    assert report["audio"]["seconds"] == round(40 * 256 / 48000, 3)
    assert report["audio"]["peak"] == 1.0


def test_the_guide_header_is_written_once_and_not_per_block(tmp_path):
    """`wave.writeframes` re-presets the RIFF header on EVERY call -- tell, seek, four bytes,
    seek, four bytes, seek back -- and each seek flushes the buffer, so a 512-byte block
    becomes a write syscall. Measured over one minute of audio: 12.26 us a block against 0.97
    for `writeframesraw`, which is what made the writer fall a queue behind and drop 28 blocks
    of a fourteen-second take. What `close` must still do is fix the header up, and that is
    what this pins -- the frame count on disk, read back by a reader that was not told it."""
    import wave

    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=2))
    guide.start(tmp_path / "take-07.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    _guide_blocks(guide, 9, channels=2)
    _drained(guide)
    wav = tmp_path / "take-07.wav"
    handed = 9 * 256 * 2
    assert wav.stat().st_size < handed // 2, (
        f"{wav.stat().st_size} bytes of {handed} on disk mid-take -- the header is being "
        f"patched per block again, and every preset flushes the buffer")
    guide.stop()
    with wave.open(str(wav)) as f:
        assert f.getnframes() == 9 * 256, "close did not fix the header up"


def test_the_audio_thread_drops_a_block_rather_than_waiting_for_the_disk(tmp_path):
    """A guide that made the sound card wait would cost dropped audio, and a dropped audio
    block is a missed HIT in the extractor rather than merely a gap in a file. No writer is
    started here: this is the queue policy alone, exactly as the frame recorder's is tested."""
    from ganlive.record.sync import Guide

    guide = Guide(depth=3)
    guide.attached = True                                 # ...but never started, so no writer
    guide._path = tmp_path / "take-01.mp4"
    _guide_blocks(guide, 5, channels=4)

    assert guide.blocks == 3, "the queue took more than it has room for"
    assert guide.dropped == 2, "a full queue blocked instead of dropping"
    assert guide.position == 5 * 256, "the stream position stopped at the dropped block"


def test_a_take_with_no_audio_still_gets_the_bar_lines(tmp_path):
    """`--triggers midi` and `--no-audio` open no input stream at all, and their takes still
    have a tempo, a start beat and a bar grid -- which alone places one in a DAW. Refusing to
    write the sidecar because half of it is unknown would take that away for nothing."""
    from ganlive.record.sync import Guide

    guide = Guide()                                       # never attached: no stream was open
    armed = guide.start(tmp_path / "take-02.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    assert "no audio input" in armed, armed
    side = guide.sidecar_path
    for beat in (0.0, 3.9, 4.0, 7.99, 8.0):
        guide.mark(beat)
    report = guide.stop()

    assert not (tmp_path / "take-02.wav").exists(), "an empty guide was written anyway"
    assert side.name == "take-02.json" and json.loads(side.read_text()) == report
    assert report["audio"] is None, "a run with no input claimed one"
    assert [m["beat"] for m in report["marks"]] == [4.0, 8.0]
    assert report["marks"][0]["audio_s"] is None
    assert report["drift_ms"] is None, "drift was reported with only one clock to read"


def test_the_sidecar_owns_its_own_field_names(tmp_path):
    """**Every field the spec asks for is a parameter of `start`, not a key in a dict handed
    over.** They were splatted at the top level for a while, so a caller with a key called
    `marks` or `audio` would have overwritten the sidecar's own and nothing would have said
    so -- and a reader written against the file would then be reading the call site."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.start(tmp_path / "take-08.mp4", bpm=99.6, beat=0.5417, beat_source="midi",
                about={"marks": "not the real ones", "model": "gv-warm-lr3"})
    report = guide.stop({"frames": 841})

    assert report["take"] == "take-08.mp4", "an absolute path where a name is enough"
    assert (report["bpm"], report["beat"], report["beat_source"]) == (99.6, 0.5417, "midi")
    assert report["marks"] == [], "a caller's key displaced the sidecar's own"
    assert report["about"]["marks"] == "not the real ones"
    assert report["video"] == {"frames": 841}
    assert report["started"][:2] == "20" and "T" in report["started"], report["started"]


def test_the_bar_lines_are_the_machine_s_and_not_the_take_s_own(tmp_path):
    """A mark exists to name a bar line the DAW also has. Counting four beats from wherever
    `v` happened to be pressed would put every mark a fraction of a bar off the grid, which is
    a beat map that is wrong everywhere and looks right."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.start(tmp_path / "take-03.mp4", bpm=130.0, beat=53.29, beat_source="midi")
    for beat in (53.29, 55.9, 56.1, 60.0, 63.5, 64.2):
        guide.mark(beat)
    assert [m["beat"] for m in guide.report()["marks"]] == [56.1, 60.0, 64.2]


def test_the_sidecar_anchors_the_take_on_the_audio_stream(tmp_path):
    """**Where the guide's first sample sits on the stream is the writer's answer, not the
    render thread's.** Reading the counter at `start` races the audio thread -- it may be
    mid-block, so the position read is one block out either way -- and it is the first block
    that actually lands in the file the sidecar has to describe. Two takes in one session are
    placed relative to each other by this number alone, with no waveform anywhere."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=4))
    _guide_blocks(guide, 7, channels=4)                   # played before `v` was pressed
    assert guide.position == 7 * 256 and guide.blocks == 0, "kept audio with no take running"

    guide.position += 256
    guide.start(tmp_path / "take-04.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    guide._q.put_nowait((7 * 256, time.perf_counter(),
                         np.zeros((4, 256), dtype=np.float32)))
    guide.blocks += 1
    _guide_blocks(guide, 3, channels=4)
    report = _drained(guide).stop()
    assert report["audio"]["start_sample"] == 7 * 256, \
        "the anchor was read from the counter on the render thread, one block out"
    assert report["audio"]["offset_ms"] > 0.0, report["audio"]
    assert not guide._q.qsize()


def test_a_mark_reads_both_clocks_at_one_instant_so_drift_is_measurable(tmp_path):
    """The frame clock is the system clock and the guide's is the sound card's crystal. Over a
    long take they part, and the video and its guide slide apart with them -- which is
    invisible in either file on its own. Both are read inside one `mark` for that reason."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=2))
    guide.start(tmp_path / "take-05.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    _guide_blocks(guide, 188, channels=2)                 # ~1.003 s of audio delivered
    _drained(guide).mark(4.0)
    report = guide.stop()

    mark = report["marks"][0]
    assert mark["audio_s"] == round(188 * 256 / 48000, 4), mark
    offset = report["audio"]["offset_ms"] / 1000.0
    assert report["drift_ms"] == round(
        (mark["video_s"] - mark["audio_s"] - offset) * 1000, 3)
    assert report["drift_ms"] < -500, report["drift_ms"]
