"""What plays it: audio features, note features, and the track vocabulary."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ganlive.timing import stat_ms
from tests.support import _drained


def test_hit_detection_is_scored_against_the_simulator_rather_than_asserted():
    """The reason the stand-in Rytm returns its event list at all."""
    from ganlive.control.features import offline, score_onsets
    from ganlive.control.kit import INDEX
    from ganlive.control.simulate import MachineSim

    for seed in (1, 3, 7):
        take = MachineSim(bpm=130.0, seed=seed).render(bars=8, tail=0.5)
        feats = offline(take.stems, take.samplerate, fps=60.0)
        got = score_onsets(feats["onsets"], 60.0, take.events, INDEX)
        assert got["precision"] == 1.0, (seed, got)
        assert got["recall"] > 0.98, (seed, got)
        assert got["abs_error_ms"] < 4.0, (seed, got)


def test_the_ground_truth_drops_hits_that_were_silenced_before_they_sounded():
    """Two tracks sharing one voice on the same step means the second erases the first, so
    listing the first as a hit would make the ground truth claim something nothing can hear --
    which once scored the detector at 90% on closed hats while it was completely correct."""
    from ganlive.control.simulate import MachineSim, Section, _s

    both = [Section("clash", 1, {"CH": _s("x..............."),
                                 "OH": _s("x...............")})]
    take = MachineSim(bpm=130.0, seed=0, sections=both).render(bars=1, tail=0.2)
    names = [name for _t, name, _v in take.events]
    assert names == ["OH"], names


def test_the_preflight_tool_and_the_stand_in_agree_on_the_drums():
    """A preset tuned against the stand-in has to address the same drums the hardware tool
    labels, or the channel map discovered with one is read with the other's names."""

    from ganlive.control.kit import TRACKS
    from ganlive.tools import doctor

    assert doctor.TRACKS == TRACKS


def test_the_pessimistic_eight_channel_case_still_finds_the_hits():
    """If the MKI turns out to expose voices rather than tracks, closed and open hat arrive on
    one channel. A preset has to survive that, so the loss it costs is measured rather than
    assumed -- and the loss is real hardware behaviour, not a detection failure."""
    from ganlive.control.features import offline, score_onsets
    from ganlive.control.kit import VOICE_GROUPS, channel_map
    from ganlive.control.simulate import MachineSim

    take = MachineSim(bpm=130.0, seed=3).render(bars=8, tail=0.5)
    channel_of = channel_map("voices")
    voices = take.stems_for(channel_of)
    assert voices.shape[0] == len(VOICE_GROUPS) == 8
    feats = offline(voices, take.samplerate, fps=60.0)
    got = score_onsets(feats["onsets"], 60.0, take.events, channel_of)
    assert got["precision"] == 1.0, got
    assert 0.80 < got["recall"] < 1.0, got                # the choke, not the detector


def test_the_two_channel_layouts_are_what_the_hardware_might_give():
    """One channel per track is the optimistic case; one per analog voice is what an MKI
    advertising ten inputs most likely means."""
    from ganlive.control.kit import TRACKS, VOICE_GROUPS, channel_map

    per_track = channel_map("tracks")
    assert len(set(per_track.values())) == len(TRACKS) == 12

    per_voice = channel_map("voices")
    assert len(set(per_voice.values())) == len(VOICE_GROUPS) == 8
    assert per_voice["CH"] == per_voice["OH"]
    assert per_voice["RS"] == per_voice["CP"]
    assert per_voice["CY"] == per_voice["CB"]
    assert per_voice["BD"] != per_voice["SD"]


def test_a_typo_in_a_discovered_layout_is_refused():
    """A wrong map is a machine where the kick drives what the hat should, and nothing
    anywhere reports a problem -- so a name that is not a real track has to raise."""
    from ganlive.control.kit import parse_channel_map

    assert parse_channel_map("BD=0, ch=3") == {"BD": 0, "CH": 3}
    with pytest.raises(ValueError, match="unknown track"):
        parse_channel_map("KICK=0")


def test_the_stand_in_produces_whatever_layout_the_map_asks_for():
    """The one command meant to rehearse a discovered map used to hand it per-track audio."""
    import numpy as np

    from ganlive.control.kit import INDEX, TRACKS, channel_map, parse_channel_map
    from ganlive.control.simulate import MachineSim
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
    """The wrap-around skips a short tail by `continue`, so a take shorter than one block spins
    the thread forever without ever pushing a sample -- silently, as a run that detects no hits
    and looks merely quiet. Both copies of this class had it before there was one class."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.control.simulate import StemFeeder

    ex = FeatureExtractor(2, 48000)
    with pytest.raises(ValueError, match="shorter than one"):
        StemFeeder(ex, np.zeros((2, 100), dtype=np.float32), 48000, 256)
    assert StemFeeder(ex, np.zeros((2, 512), dtype=np.float32), 48000, 256) is not None


def test_the_feeder_reaches_the_extractor_at_something_like_wall_clock_rate():
    """One class now, used by both the live tool and the timing harness. It used to be two,
    differing only in whether being late was counted -- so they could drift apart and only one
    of them could have noticed."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.control.simulate import MachineSim, StemFeeder

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
    """Two simulator settings that nothing turned. Rather than delete them, they are the
    pessimistic case the detector should be scored against: real drumming is not on the grid,
    and a detector tuned only against a perfect one is tuned against a machine that does not
    exist. This also makes two branches in the innermost render loop reachable.
    """
    from ganlive.control.features import offline, score_onsets
    from ganlive.control.kit import INDEX
    from ganlive.control.simulate import MachineSim

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
    """The choke silences a shared voice forward from the trig, and it stops at `LONGEST_VOICE_S` rather than
    at the end of the record -- which was averaging half a multi-megabyte row per trig, 625 MB written
    where 62 could be non-zero."""
    from ganlive.control.kit import TRACKS
    from ganlive.control.simulate import LONGEST_VOICE_S, _voice

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
    """The property the bound exists to preserve, asserted on the audio rather than on the
    arithmetic: a closed hat on top of a ringing open hat silences it from that sample on."""
    from ganlive.control.kit import INDEX
    from ganlive.control.simulate import MachineSim

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
    """`NoteFeatures` is a drop-in for `FeatureExtractor`: the frame loop reads exactly `n`,
    `since`, `drain()` and `features()`, and nothing downstream may know which it has."""
    from ganlive.control.features import FeatureExtractor, NoteFeatures

    audio = FeatureExtractor(12, 48000)
    notes = NoteFeatures(12)
    for name in ("n", "since", "drain", "features"):
        assert hasattr(notes, name), name
    assert set(notes.features()) == set(audio.features()), "a macro naming a source would break"

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
    """The whole reason this exists. On an MKI, CH and OH share one voice and arrive on one
    audio channel, so no audio rule can ever tell them apart -- and BT and LT are not streamed
    at all for want of USB bandwidth. Each still has its own note."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.kit import INDEX, VOICE_GROUPS

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
    """All six macros that ship read `density`, so it is the one that must be right. `energy`
    is a different quantity here -- how hard the pads are struck, not how loud the room is --
    and the failure to guard against is it silently reading 0.0 forever."""
    from ganlive.control.features import NoteFeatures

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
    """`FeatureConfig.floor` is an ABSOLUTE level, measured against the stand-in Rytm rather than against
    anybody's Overbridge gain staging. This pins how much room it actually has, because `doctor`'s
    `HEADROOM` is a claim about this number and nothing else checks it."""
    from ganlive.control.features import FeatureConfig, FeatureExtractor
    from ganlive.control.kit import INDEX
    from ganlive.control.simulate import MachineSim

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
    """Two clocks over one recording would agree at the start of a set and be half a bar apart
    by the end of it, so the picture would answer a kick that had already gone."""
    import numpy as np

    from ganlive.control.features import FeatureExtractor
    from ganlive.control.simulate import MonitorFeeder

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
    """**Neither source alone covers a live set on this machine.** The sequencer sends no notes
    at all and playing its trigs is how the instrument is used; the pads send an exact note per
    drum and are the only thing that can separate the four pairs sharing an analog voice. So the
    index space is the NOTE's, the coarse one maps into it, and an onset on a shared voice fires
    both of its tracks -- which is what the audio genuinely says and all it can say."""
    from ganlive.control.features import BothFeatures, FeatureExtractor
    from ganlive.control.kit import INDEX, parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
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


def test_every_source_answers_everything_the_end_of_run_report_asks_it():
    """The report reached for `.density` and `.energy`, which only the two single-source extractors have.
    `BothFeatures` is the DEFAULT -- `--triggers both` -- so the line raised at the end of every
    performance, after the playing was over, and took the whole report with it: the tracks that played, the
    hit counts, the diagnosis of what to fix. The one run that needed it most was the one where nothing
    arrived."""
    from ganlive.control.features import BothFeatures, FeatureExtractor, NoteFeatures
    from ganlive.control.kit import parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
    sources = {
        "audio": FeatureExtractor(10, 48000),
        "midi": NoteFeatures(12),
        "both": BothFeatures(FeatureExtractor(10, 48000), tracks),
    }
    for kind, src in sources.items():
        assert isinstance(src.played(), int), kind
        assert isinstance(src.n, int), kind
        got = src.features()
        assert set(got) == {"density", "energy", "active"}, f"{kind} answered {sorted(got)}"
        assert all(isinstance(v, float) for v in got.values()), kind
        src.channel_of()                      # may be empty; must not raise
        assert len(list(src.hits)) == src.n, kind


def test_a_pad_that_is_heard_as_well_as_read_counts_once():
    """A pad struck by hand arrives twice -- the note over USB MIDI, early, and the sound
    through Overbridge a few milliseconds later. Counting both would push every dial a rule
    drives twice as far for a hand hit as for a sequencer trig, which is a difference nobody
    asked for and nothing would report."""
    from ganlive.control.features import PAIRED_S, BothFeatures, FeatureExtractor
    from ganlive.control.kit import INDEX, parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
    both = BothFeatures(FeatureExtractor(10, 48000), tracks)

    both.on_note(13, INDEX["CH"], 100, when=5.0)          # the note, first
    both.audio.pending.append((8, 0.8, 0.0))              # ...and its sound, just after
    both.tick(5.0 + PAIRED_S * 0.5)
    assert [i for i, _v, _a in both.drain()] == [INDEX["CH"]], "the same hit was counted twice"

    both.audio.pending.append((8, 0.8, 0.0))
    both.tick(5.0 + PAIRED_S * 2.0)
    assert sorted(i for i, _v, _a in both.drain()) == sorted([INDEX["CH"], INDEX["OH"]])


def test_the_combined_source_answers_everything_a_source_is_asked():
    """The frame loop reads `n`, `since`, `drain()` and `features()`; the guide wants `tap` and
    `sr`; the summary wants `played()`, `hits` and `heard()`. A wrapper that answered most of
    them would fail at the one call site that used the rest, and several of those run once at
    the end of a performance -- the worst possible time to learn about it."""
    from ganlive.control.features import BothFeatures, FeatureExtractor
    from ganlive.control.kit import parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
    audio = FeatureExtractor(10, 48000)
    both = BothFeatures(audio, tracks)

    for name in ("n", "since", "hits", "sr", "tap"):
        assert hasattr(both, name), name
    for name in ("drain", "features", "played", "heard", "channel_of", "push", "tick",
                 "on_note"):
        assert callable(getattr(both, name, None)), name
    assert set(both.features()) == {"density", "energy", "active"}
    assert both.sr == 48000 and len(both.since) == 12 and len(both.hits) == 12

    both.tap = print
    assert audio.tap is print, "the tap landed on the wrapper instead of the audio"
    both.push(np.zeros((10, 256), dtype=np.float32))
    assert both.heard() > 0.0, "heard() did not reach the audio half"


def test_a_track_channel_map_is_written_one_based_and_read_zero_based():
    """A person writes the channel the machine's screen shows; the comparison happens against
    `status & 0x0F`. `midi.parse_controls` resolves the same mismatch the same way."""
    from ganlive.control.kit import INDEX, TRACKS, parse_track_channels

    whole = parse_track_channels("1-12")
    assert whole == {i: i for i in range(len(TRACKS))}, whole
    assert parse_track_channels("1=BD,2=SD") == {0: INDEX["BD"], 1: INDEX["SD"]}
    with pytest.raises(ValueError, match="11 channels"):
        parse_track_channels("1-11")
    with pytest.raises(ValueError, match="already BD"):
        parse_track_channels("3=BD,3=SD")


def test_the_track_a_note_is_comes_from_the_channel_when_the_machine_sends_that_way():
    """**The machine has two output modes and they identify a track differently.** On AUTO CH
    the kit shares one channel and the note says which track -- 0 to 11, what the pads send. On
    TRACK CH each track has its own channel and the note carries the trig's PITCH, 12 to 60,
    which names no track at all. Reading the note in that mode indexes the kit with a pitch."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.kit import parse_track_channels

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
    """`BothFeatures` had its own copy of `note - base_note`. It would have kept working while
    `NoteFeatures` learned about channels, so the pad-beats-onset rule would have gone on
    reading pitches and every trig would have been counted twice."""
    from ganlive.control.features import PAIRED_S, BothFeatures, FeatureExtractor
    from ganlive.control.kit import INDEX, parse_channel_map, parse_track_channels

    tracks = parse_channel_map("BD=0,SD=1,RS=2,CP=2,BT=3,LT=3,MT=4,HT=4,CH=5,OH=5,CY=6,CB=6")
    audio = FeatureExtractor(7, 48000)
    both = BothFeatures(audio, tracks, channels=parse_track_channels("1-12"))

    assert both.on_note(7, 53, 110, when=1.0) == INDEX["HT"]
    assert both.hits[INDEX["HT"]] == 1

    audio.pending.append((4, 0.8, 0.0))
    both.tick(1.0 + PAIRED_S * 0.5)
    assert both.hits[INDEX["HT"]] == 1 and both.hits[INDEX["MT"]] == 0


def test_a_note_that_names_no_track_is_counted_rather_than_dropped():
    """**Both ways of setting the mode wrongly are otherwise silent, and one of them looks
    like a working run.** With the map omitted on TRACK CH every pitch falls out of spread and
    the wire reads as dead. With it supplied on AUTO CH the whole kit shares one channel, so
    all twelve drums resolve to ONE index -- `played()` is 1, the hit watchdog stays quiet, and
    the picture runs with one drum driving what twelve should."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.kit import parse_track_channels

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
    """The flag was very nearly the only way to know the mode. It is an override: the modes do
    not overlap in what they put on the wire, so a few seconds of it settles the question --
    except for the one genuinely ambiguous case, which must not be guessed."""
    from ganlive.control.kit import output_mode

    assert output_mode({13: {0, 1, 5, 11}}) == "auto"           # pads, one channel, low notes
    assert output_mode({0: {47}, 1: {53}, 8: {36}}) == "track"  # pitches across many channels
    assert output_mode({}) is None
    assert output_mode({4: {47, 53}}) is None
    assert output_mode({13: {0, 5, 11}, 0: {47}, 1: {53}, 7: {36}}) == "mixed"


def test_a_channel_map_and_a_note_base_are_both_live_because_the_machine_uses_both():
    """**Read off the MKI manual, not guessed, and it overturned a guard added an hour before.**
    `TRK SEND MIDI` sends on the track's channel (11.8); `OUTPUT CH` selects auto or track for
    the PADS and knobs (13.4.2). Two settings, two sources: per-track channels with OUTPUT CH
    left on AUTO puts sequencer pitches on channels 1-12 and pad notes on the auto channel
    simultaneously, so a reader that insisted on one rule would drop half the kit."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.kit import parse_track_channels

    both = NoteFeatures(12, channels=parse_track_channels("1-12"))
    assert both.on_note(6, 53, 100) == 6, "a mapped channel is the track, whatever the pitch"
    assert both.on_note(13, 9, 100) == 9, "the auto channel still names a track by its note"
    assert both.unresolved == 0
    assert both.on_note(13, 47, 100) is None
    assert both.unresolved == 1


def test_the_guide_hears_whatever_pushes_the_extractor(tmp_path):
    """**The tap is on the extractor, not on the sound card.** Three things push blocks into
    it -- the live tool's ASIO callback, `StemFeeder` and `MonitorFeeder` -- so a hook beside
    one of them is two copies waiting to be written and a `--simulate` path that quietly does
    not do it. This drives it through the stand-in, which is the only path testable here."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.record.sync import Guide

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
    """`FeatureExtractor.tap` holds one consumer. A second attach used to overwrite it, which
    leaves the first reporting on a stream that no longer reaches it -- a fault that counts
    itself as fine, and this project's signature failure."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.record.sync import Guide

    ex = FeatureExtractor(2, 48000)
    Guide().listen_to(ex)
    with pytest.raises(RuntimeError, match="already has a tap"):
        Guide().listen_to(ex)
