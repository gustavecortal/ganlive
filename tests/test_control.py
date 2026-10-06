"""What plays it: audio features, note features, and the track vocabulary."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ganlive.control.features import (
    PAIRED_S,
    BothFeatures,
    FeatureConfig,
    FeatureExtractor,
    NoteFeatures,
)
from ganlive.control.kit import (
    INDEX,
    TRACKS,
    VOICE_GROUPS,
    channel_map,
    output_mode,
    parse_channel_map,
    parse_track_channels,
)
from ganlive.control.simulate import (
    LONGEST_VOICE_S,
    MachineSim,
    MonitorFeeder,
    Section,
    StemFeeder,
    _s,
    _voice,
)
from ganlive.record.sync import Guide
from ganlive.timing import stat_ms
from ganlive.tools import doctor
from ganlive.tools.drum_map import WINDOW_S, Strikes, report, report_recall
from tests.support import OVERBRIDGE, _drained, offline, score_onsets

#: Every trigger source `--triggers` can choose, built fresh per test.
SOURCES = {
    "audio": lambda: FeatureExtractor(10, 48000),
    "midi": lambda: NoteFeatures(12),
    "both": lambda: BothFeatures(FeatureExtractor(10, 48000), parse_channel_map(OVERBRIDGE)),
}


def test_hit_detection_is_scored_against_the_simulator_rather_than_asserted():
    """Onset detection is scored for precision, recall and timing against the simulator's own
    list of the hits it played."""
    for seed in (1, 3, 7):
        take = MachineSim(bpm=130.0, seed=seed).render(bars=8, tail=0.5)
        feats = offline(take.stems, take.samplerate, fps=60.0)
        got = score_onsets(feats["onsets"], 60.0, take.events, INDEX)
        assert got["precision"] == 1.0, (seed, got)
        assert got["recall"] > 0.98, (seed, got)
        assert got["abs_error_ms"] < 4.0, (seed, got)


def test_the_ground_truth_drops_hits_that_were_silenced_before_they_sounded():
    """Two tracks sharing one voice on the same step: the second erases the first, so only the
    second may be listed as a hit, or the detector is scored against an inaudible event."""
    both = [Section("clash", 1, {"CH": _s("x..............."),
                                 "OH": _s("x...............")})]
    take = MachineSim(bpm=130.0, seed=0, sections=both).render(bars=1, tail=0.2)
    names = [name for _t, name, _v in take.events]
    assert names == ["OH"], names


def test_the_preflight_tool_and_the_stand_in_agree_on_the_drums():
    """`doctor` and the simulator use the same track names, so a channel map found with one
    means the same drums in the other."""

    assert doctor.TRACKS == TRACKS


def test_the_pessimistic_eight_channel_case_still_finds_the_hits():
    """With eight channels (one per analog voice), closed and open hat share a channel. Hits
    are still found precisely; the recall lost is the hat choke, not a detection failure."""
    take = MachineSim(bpm=130.0, seed=3).render(bars=8, tail=0.5)
    channel_of = channel_map("voices")
    voices = take.stems_for(channel_of)
    assert voices.shape[0] == len(VOICE_GROUPS) == 8
    feats = offline(voices, take.samplerate, fps=60.0)
    got = score_onsets(feats["onsets"], 60.0, take.events, channel_of)
    assert got["precision"] == 1.0, got
    assert 0.80 < got["recall"] < 1.0, got                # the choke, not the detector


def test_the_two_channel_layouts_are_what_the_hardware_might_give():
    """The two audio layouts a drum machine may stream: one channel per track (twelve), or one
    per analog voice (eight, with paired tracks sharing a channel)."""
    per_track = channel_map("tracks")
    assert len(set(per_track.values())) == len(TRACKS) == 12

    per_voice = channel_map("voices")
    assert len(set(per_voice.values())) == len(VOICE_GROUPS) == 8
    assert per_voice["CH"] == per_voice["OH"]
    assert per_voice["RS"] == per_voice["CP"]
    assert per_voice["CY"] == per_voice["CB"]
    assert per_voice["BD"] != per_voice["SD"]


def test_a_typo_in_a_discovered_layout_is_refused():
    """A channel map naming an unknown track raises, since a wrong map would silently route one
    drum's audio to another's controls."""
    assert parse_channel_map("BD=0, ch=3") == {"BD": 0, "CH": 3}
    with pytest.raises(ValueError, match="unknown track"):
        parse_channel_map("KICK=0")


def test_the_stand_in_produces_whatever_layout_the_map_asks_for():
    """The simulator mixes its stems into whatever channel layout a map asks for, so a
    discovered map can be rehearsed without the hardware."""
    take = MachineSim(bpm=130.0, seed=2).render(bars=2, tail=0.2)

    per_track = take.stems_for(channel_map("tracks"))
    assert per_track.shape == take.stems.shape
    assert np.allclose(per_track, take.stems)

    found = parse_channel_map("BD=0,SD=1,CH=2,OH=2")
    mixed = take.stems_for(found)
    assert mixed.shape[0] == 3, "sized by the channels the map addresses"
    assert np.allclose(mixed[0], take.stems[INDEX["BD"]])
    assert np.allclose(mixed[2], take.stems[INDEX["CH"]] + take.stems[INDEX["OH"]])
    mapped = sum(take.stems[INDEX[track]] for track in found)
    assert np.allclose(mixed.sum(axis=0), mapped)

    for layout, n in (("tracks", len(TRACKS)), ("voices", 8)):
        assert take.stems_for(channel_map(layout)).shape[0] == n


def test_the_stand_in_feeder_refuses_audio_shorter_than_one_block():
    """A take shorter than one block is refused, since the feeder would otherwise loop forever
    without pushing a sample and the run would just look quiet."""
    ex = FeatureExtractor(2, 48000)
    with pytest.raises(ValueError, match="shorter than one"):
        StemFeeder(ex, np.zeros((2, 100), dtype=np.float32), 48000, 256)
    assert StemFeeder(ex, np.zeros((2, 512), dtype=np.float32), 48000, 256) is not None


def test_the_feeder_reaches_the_extractor_at_something_like_wall_clock_rate():
    """`StemFeeder` pushes blocks at roughly real-time pace and records how long each push took;
    the live tool and the timing harness both use it."""
    take = MachineSim(bpm=130.0, seed=4).render(bars=2, tail=0.2)
    ex = FeatureExtractor(take.stems.shape[0], take.samplerate)
    feeder = StemFeeder(ex, take.stems, take.samplerate, 256)
    feeder.start()
    time.sleep(0.35)
    feeder.stop()

    expected = 0.35 / (256 / take.samplerate)
    assert 0.4 * expected < feeder.calls < 1.6 * expected, (feeder.calls, expected)
    assert ex.pending, "the extractor should have seen hits"
    assert feeder.push_ms, "the bookkeeping is always on now, not only in the timing harness"

    pushes = list(feeder.push_ms)
    steady = stat_ms(pushes[1:])
    assert steady["p95"] < 5.33, (steady, pushes[0])
    assert pushes[0] < 40.0, pushes[0]


def test_swing_and_humanising_move_the_hits_and_the_detector_still_finds_them():
    """Swing and humanising move hits off the grid, as real playing does, and the detector must
    still find them."""
    straight = MachineSim(bpm=130.0, seed=6).render(bars=4, tail=0.4)
    loose = MachineSim(bpm=130.0, seed=6, swing=0.18, humanise_ms=6.0).render(bars=4, tail=0.4)

    assert len(straight.events) == len(loose.events), "same pattern, different timing"
    moved = [abs(a - b) for a, b in zip(sorted(t for t, _n, _v in straight.events),
                                        sorted(t for t, _n, _v in loose.events), strict=True)]
    assert max(moved) > 0.01, "the settings must actually move something"

    feats = offline(loose.stems, loose.samplerate, fps=60.0)
    got = score_onsets(feats["onsets"], 60.0, loose.events, INDEX)
    assert got["recall"] > 0.95, got
    assert got["precision"] > 0.95, got


def test_no_voice_rings_longer_than_the_choke_reaches():
    """The choke silences a shared voice only up to `LONGEST_VOICE_S` after a trig, so no voice
    may ring longer than that; the bound is also checked to stay reasonably tight."""
    longest = {}
    for track in TRACKS:
        longest[track] = max(_voice(track, 1.0, 48000, np.random.default_rng(s)).shape[0]
                             for s in range(6)) / 48000
    worst = max(longest.values())
    assert worst <= LONGEST_VOICE_S, (
        f"{max(longest, key=longest.get)} rings for {worst:.3f}s, past the "
        f"{LONGEST_VOICE_S}s the choke reaches")
    assert LONGEST_VOICE_S < worst * 2.0, "the bound has drifted far past what it guards"


def test_a_choked_trig_leaves_no_tail_behind_it():
    """Checked on the rendered audio: a closed hat on top of a ringing open hat silences it
    from that sample on."""
    take = MachineSim(bpm=130.0, seed=3).render(bars=8, tail=0.5)
    sr = take.samplerate
    oh = take.stems[INDEX["OH"]]
    at = {name: sorted(int(t * sr) for t, n, _v in take.events if n == name)
          for name in ("CH", "OH")}
    assert at["CH"] and at["OH"], "the fixture needs both hats playing"

    checked = 0
    for start in at["CH"]:
        nxt = next((o for o in at["OH"] if o > start), len(oh))
        window = oh[start:min(nxt, start + int(1.6 * sr))]
        if window.size < sr // 50:                        # too short to say anything
            continue
        checked += 1
        assert np.abs(window).max() < 1e-6, (
            f"an open hat is still sounding {window.size / sr:.3f}s after a closed hat "
            f"took the voice at sample {start}")
    assert checked, "no closed hat landed on a ringing open hat; the fixture proves nothing"


def test_midi_notes_drive_the_same_features_audio_does():
    """A note-on is a hit with a strength, drained once, and timed like an audio onset."""
    notes = NoteFeatures(12)
    notes.tick(0.0)
    notes.on_note(14, 0, 100, when=0.0)      # BD, hard
    notes.on_note(14, 9, 40, when=0.0)       # OH, soft
    got = notes.drain()
    assert sorted((i, round(v, 2)) for i, v, _ago in got) == [(0, 0.79), (9, 0.31)]
    assert notes.drain() == [], "draining twice must not replay a hit"

    notes.tick(0.25)
    assert notes.since[0] == pytest.approx(0.25, abs=1e-3)
    assert notes.since[9] == pytest.approx(0.25, abs=1e-3)
    assert notes.since[1] > 100, "a track that never fired is long ago, not zero"


def test_midi_separates_the_drums_that_share_an_analog_voice():
    """Tracks that share an analog voice (and so one audio channel) still have their own MIDI
    notes, so note input tells them apart where audio cannot."""
    shared = [g for g in VOICE_GROUPS if len(g) > 1]
    assert shared, "this test is about the pairs; there must be some"

    notes = NoteFeatures(12)
    notes.tick(0.0)
    for pair in shared:
        first, second = pair[0], pair[1]
        notes.on_note(14, INDEX[first], 127, when=0.0)
        hits = {i for i, _v, _a in notes.drain()}
        assert hits == {INDEX[first]}, f"{first} must not also report {second}"
        assert notes.since[INDEX[second]] > 100, f"{second} did not fire and must not read so"
        notes.since[INDEX[first]] = 1e6


def test_the_note_source_reports_density_and_does_not_leave_energy_inert():
    """From notes, `density` counts hits in the window (every shipped macro reads it), and
    `energy` reflects how hard pads are struck; neither may sit at 0.0 while hits arrive."""
    notes = NoteFeatures(12)
    window = notes.cfg.energy_window
    notes.tick(0.0)
    for k in range(6):
        notes.on_note(14, k % 12, 127, when=k * 0.05)
    notes.tick(0.30)
    assert notes.density == pytest.approx(6 / window, rel=1e-6)
    assert notes.energy > 0.9, "six hits at full velocity is not zero energy"
    assert notes.active == pytest.approx(6 / 12)

    notes.tick(0.30 + window + 0.01)
    assert notes.density == 0.0
    assert notes.energy == 0.0
    assert notes.active == 0.0


def test_a_quiet_send_loses_its_quietest_drums_first_and_preflight_says_so():
    """`FeatureConfig.floor` is an absolute level, set against the simulator. This pins how much
    gain headroom it leaves, which is what `doctor`'s `HEADROOM` check assumes."""
    sr, block, tol = 48000, 256, 0.030
    render = MachineSim(bpm=130, samplerate=sr, seed=0).render(bars=2)
    pcm = render.stems.astype(np.float32)
    truth = [(t, INDEX[name]) for t, name, _v in render.events]
    assert truth, "the arrangement must actually trig something"

    def recall(gain):
        ex = FeatureExtractor(pcm.shape[0], sr)
        found = []
        for i in range(0, pcm.shape[1] - block, block):
            ex.push(pcm[:, i:i + block] * gain)
            for ch, _v, ago in ex.drain():
                found.append((ch, (i + block) / sr - ago))
        left, hit = sorted(truth), 0
        for ch, t in sorted(found, key=lambda e: e[1]):
            for k, (tt, tch) in enumerate(left):
                if tch == ch and abs(tt - t) <= tol:
                    hit += 1
                    left.pop(k)
                    break
        return hit / len(truth)

    assert recall(1.0) == 1.0
    assert recall(0.1) == 1.0

    quietest = min(np.abs(pcm[ch]).max() for _t, ch in truth)
    assert recall(0.002) < 0.6, "if this passes, the floor no longer binds and HEADROOM is moot"
    assert quietest > FeatureConfig().floor * 8, (
        "the quietest voice must clear preflight's HEADROOM at unity, or the check it prints "
        "would flag a healthy machine")


def test_the_stand_in_can_be_heard_and_measured_from_the_same_sample_index():
    """`MonitorFeeder` plays and analyses the simulator from one sample index, so what is heard
    and what drives the picture cannot drift apart over a set."""
    stems = np.zeros((2, 1000), dtype=np.float32)
    stems[0, 500] = 1.0
    mix = np.zeros((2, 1000), dtype=np.float32)
    mix[:, 500] = 0.5
    feeder = MonitorFeeder(FeatureExtractor(2, 48000), stems, mix, 48000, 256)

    out = np.zeros((256, 2), dtype=np.float32)
    heard, at = [], []
    for _ in range(3):
        at.append(feeder.at)
        feeder._block(out, 256, None, None)
        heard.append(float(np.abs(out).max()))
    assert at == [0, 256, 512]
    assert heard[1] == 0.5 and heard[0] == 0.0
    assert feeder.calls == 3


def test_both_sources_share_one_track_space_and_the_pairs_come_apart():
    """`BothFeatures` combines audio (heard sequencer trigs) and notes (pads) in the notes'
    twelve-track space: an audio onset on a shared voice fires both its tracks, and a note
    fires only its own."""
    tracks = parse_channel_map(OVERBRIDGE)
    both = BothFeatures(FeatureExtractor(10, 48000), tracks)
    assert both.n == 12, "the space is the channel map's ten, not the twelve tracks"
    assert both.channel_of() == dict(INDEX)

    both.audio.pending.append((8, 0.8, 0.0))
    both.tick(1.0)
    assert sorted(i for i, _v, _a in both.drain()) == sorted([INDEX["CH"], INDEX["OH"]])

    both.on_note(13, INDEX["OH"], 100, when=2.0)
    both.tick(2.0)
    assert [i for i, _v, _a in both.drain()] == [INDEX["OH"]]
    both.tick(2.5)
    assert both.since[INDEX["CH"]] > both.since[INDEX["OH"]], "the voice's other track fired"
    assert float(both.since[INDEX["OH"]]) == pytest.approx(0.5, abs=1e-4)


def test_a_pad_that_is_heard_as_well_as_read_counts_once():
    """A struck pad arrives twice: its MIDI note, then its sound over USB audio a few
    milliseconds later. Within `PAIRED_S` they count as one hit, so a pad weighs the same as a
    sequencer trig."""
    tracks = parse_channel_map(OVERBRIDGE)
    both = BothFeatures(FeatureExtractor(10, 48000), tracks)

    both.on_note(13, INDEX["CH"], 100, when=5.0)          # the note, first
    both.audio.pending.append((8, 0.8, 0.0))              # ...and its sound, just after
    both.tick(5.0 + PAIRED_S * 0.5)
    assert [i for i, _v, _a in both.drain()] == [INDEX["CH"]], "the same hit was counted twice"

    both.audio.pending.append((8, 0.8, 0.0))
    both.tick(5.0 + PAIRED_S * 2.0)
    assert sorted(i for i, _v, _a in both.drain()) == sorted([INDEX["CH"], INDEX["OH"]])


@pytest.mark.parametrize("kind", list(SOURCES))
def test_every_source_answers_everything_the_frame_loop_and_the_report_ask(kind):
    """The frame loop, the presets and the end-of-run report read these off whichever source
    `--triggers` chose, and none of them may know which it has."""
    src = SOURCES[kind]()
    for name in ("n", "since", "hits"):
        assert hasattr(src, name), name
    for name in ("drain", "features", "played", "channel_of"):
        assert callable(getattr(src, name, None)), name
    assert isinstance(src.n, int) and isinstance(src.played(), int)
    got = src.features()
    assert set(got) == {"density", "energy", "active"}, f"{kind} answered {sorted(got)}"
    assert all(isinstance(v, float) for v in got.values())
    src.channel_of()                          # may be empty; must not raise
    assert len(list(src.hits)) == src.n


def test_the_combined_source_forwards_the_audio_half_s_calls():
    """`BothFeatures` also stands in for the audio source the recording guide taps."""
    audio = FeatureExtractor(10, 48000)
    both = BothFeatures(audio, parse_channel_map(OVERBRIDGE))
    for name in ("heard", "push", "tick", "on_note"):
        assert callable(getattr(both, name, None)), name
    assert both.sr == 48000 and len(both.since) == 12 and len(both.hits) == 12

    both.tap = print
    assert audio.tap is print, "the tap landed on the wrapper instead of the audio"
    both.push(np.zeros((10, 256), dtype=np.float32))
    assert both.heard() > 0.0, "heard() did not reach the audio half"


def test_a_track_channel_map_is_written_one_based_and_read_zero_based():
    """A person writes the channel the machine's screen shows; the comparison happens against
    `status & 0x0F`. `midi.parse_controls` resolves the same mismatch the same way."""
    whole = parse_track_channels("1-12")
    assert whole == {i: i for i in range(len(TRACKS))}, whole
    assert parse_track_channels("1=BD,2=SD") == {0: INDEX["BD"], 1: INDEX["SD"]}
    with pytest.raises(ValueError, match="11 channels"):
        parse_track_channels("1-11")
    with pytest.raises(ValueError, match="already BD"):
        parse_track_channels("3=BD,3=SD")


def test_the_track_a_note_is_comes_from_the_channel_when_the_machine_sends_that_way():
    """In AUTO CH mode the note number (0-11) names the track; in TRACK CH mode each track has
    its own channel and the note is a pitch, so the track must come from the channel."""
    auto = NoteFeatures(12)
    auto.on_note(13, 5, 100)                       # pad: channel 14 on the wire, note 5
    assert auto.hits[5] == 1 and auto.hits.sum() == 1

    auto.on_note(13, 47, 100)
    assert auto.hits.sum() == 1, "a pitch was read as a track index"

    per_track = NoteFeatures(12, channels=parse_track_channels("1-12"))
    per_track.on_note(5, 47, 100)                  # track 6, pitch 47 -- the pitch is not read
    per_track.on_note(5, 12, 64)
    assert per_track.hits[5] == 2 and per_track.hits.sum() == 2
    per_track.on_note(15, 40, 100)
    assert per_track.hits.sum() == 2


def test_the_pair_source_claims_a_voice_by_the_same_rule_the_notes_use():
    """`BothFeatures` resolves a note to its track through `NoteFeatures`, channels included,
    so a pad and the audio onset it causes are one hit, not two."""
    tracks = parse_channel_map("BD=0,SD=1,RS=2,CP=2,BT=3,LT=3,MT=4,HT=4,CH=5,OH=5,CY=6,CB=6")
    audio = FeatureExtractor(7, 48000)
    both = BothFeatures(audio, tracks, channels=parse_track_channels("1-12"))

    assert both.on_note(7, 53, 110, when=1.0) == INDEX["HT"]
    assert both.hits[INDEX["HT"]] == 1

    audio.pending.append((4, 0.8, 0.0))
    both.tick(1.0 + PAIRED_S * 0.5)
    assert both.hits[INDEX["HT"]] == 1 and both.hits[INDEX["MT"]] == 0


def test_a_note_that_names_no_track_is_counted_rather_than_dropped():
    """Notes that resolve to no track are counted per channel and note (up to a cap), so a
    wrongly set output mode shows up in the report instead of looking like silence."""
    missing = NoteFeatures(12)                     # AUTO CH reader given TRACK CH pitches
    for note in (47, 53, 60):
        assert missing.on_note(5, note, 100) is None
    assert missing.played() == 0 and missing.unresolved == 3
    assert missing.unclaimed == {(5, 47): 1, (5, 53): 1, (5, 60): 1}

    for note in range(200):
        missing.on_note(9, note + 64, 100)
    assert len(missing.unclaimed) == 32 and missing.unresolved == 203

    wired = NoteFeatures(12, channels=parse_track_channels("1-12"))
    assert wired.on_note(0, 47, 100) == 0 and wired.unresolved == 0


def test_the_two_output_modes_are_told_apart_by_the_traffic_itself():
    """A few seconds of traffic identify the output mode (the flag only overrides it); an
    ambiguous sample returns None rather than a guess."""
    assert output_mode({13: {0, 1, 5, 11}}) == "auto"           # pads, one channel, low notes
    assert output_mode({0: {47}, 1: {53}, 8: {36}}) == "track"  # pitches across many channels
    assert output_mode({}) is None
    assert output_mode({4: {47, 53}}) is None
    assert output_mode({13: {0, 5, 11}, 0: {47}, 1: {53}, 7: {36}}) == "mixed"


def test_a_channel_map_and_a_note_base_are_both_live_because_the_machine_uses_both():
    """A machine can send sequencer trigs on per-track channels while its pads still use the
    auto channel, so a reader with a channel map must also resolve auto-channel notes."""
    both = NoteFeatures(12, channels=parse_track_channels("1-12"))
    assert both.on_note(6, 53, 100) == 6, "a mapped channel is the track, whatever the pitch"
    assert both.on_note(13, 9, 100) == 9, "the auto channel still names a track by its note"
    assert both.unresolved == 0
    assert both.on_note(13, 47, 100) is None
    assert both.unresolved == 1


def test_the_guide_hears_whatever_pushes_the_extractor(tmp_path):
    """The recording guide taps the feature extractor, so it hears every audio source that
    pushes blocks (sound card, `StemFeeder`, `MonitorFeeder`); driven here by direct pushes."""
    ex = FeatureExtractor(4, 48000)
    guide = Guide()
    guide.listen_to(ex)
    assert (guide.sr, guide.channels) == (48000, 4), "the extractor's own numbers were not read"
    guide.start(tmp_path / "take-06.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    ex.push(np.zeros((256, 4), dtype=np.float32))
    ex.push(np.zeros((4, 256), dtype=np.float32))
    report = _drained(guide).stop()
    assert report["audio"]["blocks"] == 2 and guide.position == 512
    assert report["audio"]["peak"] == 0.0
    assert "SILENT" in guide.describe(report), guide.describe(report)


def test_a_second_guide_is_refused_rather_than_displacing_the_first(tmp_path):
    """`FeatureExtractor.tap` holds one consumer, so a second guide is refused rather than
    silently cutting off the first."""
    ex = FeatureExtractor(2, 48000)
    Guide().listen_to(ex)
    with pytest.raises(RuntimeError, match="already has a tap"):
        Guide().listen_to(ex)


def test_learning_the_channel_map_votes_each_pad_onto_the_channel_that_answers_it(capsys):
    """`doctor --learn`: a note-on names the drum, and the channel that peaks just after it
    is where that drum is heard. A channel answering every pad is a mix bus, set aside."""
    strikes = Strikes(4)
    t = 0.0
    firsts = []
    for note, channel in ((0, 1), (1, 2), (0, 1), (1, 2), (0, 1)):
        firsts.append(strikes.on_note(note, t))
        block = np.zeros((64, 4), dtype=np.float32)
        block[:, channel] = 0.5
        block[:, 3] = 0.4                                  # the mix bus hears everything
        strikes.on_audio(block)
        t += WINDOW_S * 2
        strikes.settle(t)
    assert firsts == [True, True, False, False, False], "a pad is announced once"

    mapping = report(strikes.seen, strikes.votes, strikes.levels, strikes.struck,
                     strikes.silent, ["BD", "SD"])
    assert mapping == {"BD": 1, "SD": 2}
    report_recall(strikes.times, [(0.001, 1)], mapping, ["BD", "SD"])
    said = capsys.readouterr().out
    assert "mix bus" in said and "--map BD=1,SD=2" in said
