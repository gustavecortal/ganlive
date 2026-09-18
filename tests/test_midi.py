"""What arrives on the wire: notes, clock, CCs, NRPN, pressure."""

from __future__ import annotations

import time

import pytest

from ganlive.dials import table as _surface  # noqa: E402
from ganlive.dials.fastgan_dials import fastgan
from ganlive.presets import Preset
from ganlive.walk import (
    MusicalClock,
)
from tests.support import FIXTURES


def test_midi_pulses_take_over_from_the_internal_clock():
    """The fallback must become invisible the moment real clock arrives, so a driver can call
    `advance` unconditionally without checking who is in charge."""
    c = MusicalClock(120.0)
    c.advance(1.0)
    assert c.source == "internal"
    for _ in range(MusicalClock.PPQN):
        c.on_pulse()
    assert c.source == "midi"
    assert c.beats == pytest.approx(1.0)                  # 24 pulses is one quarter note
    c.advance(10.0)                                       # must now be ignored
    assert c.beats == pytest.approx(1.0)


def test_the_five_midi_messages_that_matter_reach_the_clock():
    """Each one answers a different question, and getting any of them wrong produces video
    that still looks smooth. They are checked as pure arithmetic, with no hardware."""
    from ganlive.control import midi

    clock = MusicalClock(120.0)
    for _ in range(MusicalClock.PPQN * 2):
        assert midi.dispatch(clock, midi.CLOCK) == "clock"
    assert clock.beats == pytest.approx(2.0)
    assert clock.source == "midi"

    assert midi.dispatch(clock, midi.START) == "start"
    assert clock.beats == pytest.approx(0.0), "Start is what puts the video's bar on the music's"

    assert midi.dispatch(clock, midi.STOP) == "stop"
    assert not clock.running
    assert midi.dispatch(clock, midi.CONTINUE) == "continue"
    assert clock.running

    assert midi.dispatch(clock, midi.SONG_POSITION, 16, 0) == "song_position"
    assert clock.beats == pytest.approx(4.0)
    assert midi.dispatch(clock, midi.SONG_POSITION, 0, 1) == "song_position"
    assert clock.beats == pytest.approx(128 / 4)


def test_anything_else_on_the_wire_is_ignored():
    """A Rytm sends note-ons and controller moves down the same cable. None of them may
    disturb the position."""
    from ganlive.control import midi

    clock = MusicalClock(120.0)
    for _ in range(MusicalClock.PPQN):
        midi.dispatch(clock, midi.CLOCK)
    before = clock.beats
    assert midi.dispatch(clock, 0x90, 60, 100) == "note_on"
    assert midi.dispatch(clock, 0x9F, 60, 100) == "note_on", "any MIDI channel"
    assert midi.dispatch(clock, 0x90, 60, 0) == "note_off", "velocity 0 is a release"
    for status in (0x80, 0xE0, 0xF0, 0xFE, 0xF1):
        assert midi.dispatch(clock, status, 60, 100) is None
    for channel in range(16):
        assert midi.dispatch(clock, midi.CONTROL_CHANGE + channel, 60, 100) == "control_change"
    assert clock.beats == before


def test_the_reader_hands_pressure_on_and_not_just_notes_and_knobs():
    """**The seam the other pressure tests cannot reach.** `dispatch` naming a byte and
    `PressureMap` acting on one are both covered, and neither would have caught the reader
    having no branch to carry it between them -- which is exactly the state the aftertouch
    stream was in: classified nowhere, delivered nowhere, and silent about both."""
    from ganlive.control.midi import AFTERTOUCH_POLY, CONTROL_CHANGE, NOTE_ON, ClockReader

    got = {"pressure": [], "note": [], "control": []}
    reader = ClockReader(MusicalClock(120.0),
                         on_pressure=lambda c, n, v: got["pressure"].append((c, n, v)),
                         on_note=lambda c, n, v: got["note"].append((c, n, v)),
                         on_control=lambda c, n, v, _top: got["control"].append((c, n, v)))

    class _Port:
        """One burst, then nothing -- the shape `run`'s drain loop reads."""

        def __init__(self, events):
            self._events = list(events)

        def poll(self):
            return bool(self._events)

        def read(self, _n):
            out, self._events = self._events, []
            return [[e, 0] for e in out]

    port = _Port([[AFTERTOUCH_POLY + 13, 3, 90, 0], [NOTE_ON + 13, 5, 100, 0],
                  [CONTROL_CHANGE + 13, 35, 64, 0]])
    reader.open_ports = lambda: [("fake", port)]
    reader.start()
    for _ in range(200):                       # the thread polls; do not sleep a fixed guess
        if got["pressure"] and got["note"] and got["control"]:
            break
        time.sleep(0.005)
    reader.stop_flag = True
    reader.join(2.0)

    assert got["pressure"] == [(13, 3, 90)], got
    assert got["note"] == [(13, 5, 100)] and got["control"] == [(13, 35, 64)]
    assert reader.counts.get("aftertouch_poly") == 1, reader.counts


def test_a_reader_with_no_ports_says_so_rather_than_failing():
    """No MIDI is a real outcome -- the Rytm's USB setting is a choice between Overbridge and
    MIDI, and they may not coexist. It has to degrade to the internal tempo, loudly."""
    from ganlive.control.midi import ClockReader

    reader = ClockReader(MusicalClock(120.0), port_match="nothing-matches-this")
    assert reader.open_ports() == []
    said = reader.describe()
    assert said.startswith("MIDI:"), said
    assert "ignoring the machine" in said or "no MIDI" in said, said


def test_a_midi_port_filter_that_matched_nothing_says_so():
    """A typo in `--midi-port` and a machine with no MIDI at all produced the same message,
    and that message describes the failure this whole design fears most: the picture runs at
    its own tempo and looks entirely plausible while ignoring the drummer. Two different
    problems must not share one symptom when one of them is a typo."""
    from ganlive.control.midi import ClockReader
    from ganlive.walk import MusicalClock

    empty = ClockReader(MusicalClock(), "rytmm")
    assert "no input ports at all" in empty.describe()

    missed = ClockReader(MusicalClock(), "rytmm")
    missed.rejected = ["Elektron Analog Rytm MKI", "loopMIDI Port"]
    said = missed.describe()
    assert "rytmm" in said and "Elektron Analog Rytm MKI" in said
    assert "matched none of" in said

    listening = ClockReader(MusicalClock(), "rytm")
    listening.ports = ["Elektron Analog Rytm MKI"]
    listening.rejected = ["loopMIDI Port"]
    assert "listening to Elektron" in listening.describe()
    assert "ignoring loopMIDI Port" in listening.describe()


def test_the_preflight_certifies_the_dispatch_the_live_tool_actually_runs():
    """The tool exists to answer whether the clock survives Overbridge, on the one question
    where a wrong answer looks exactly like a right one. It used to re-declare the status bytes
    and re-implement the dispatch, so a green preflight was evidence about code that would not
    run."""
    import importlib.util
    from pathlib import Path

    from ganlive.control import midi
    from ganlive.walk import MusicalClock

    path = Path(__file__).resolve().parents[1] / "src" / "ganlive" / "tools" / "doctor.py"
    spec = importlib.util.spec_from_file_location("doctor_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.dispatch is midi.dispatch, "it must run the shipping dispatch, not its own"
    for name in ("CLOCK", "START", "CONTINUE", "STOP", "SPP", "SONG_POSITION"):
        assert not hasattr(mod, name), f"{name} is a second copy of a midi.py constant"

    listener = mod._MidiListener.__new__(mod._MidiListener)
    listener.clock_state = MusicalClock()
    listener.clock, listener.transport, listener.notes = 0, [], {}
    listener.first_clock = listener.last_clock = None
    assert midi.dispatch(listener.clock_state, midi.CLOCK, now=0.0) == "clock"
    assert listener.clock_state.source == "midi"


def test_a_dial_says_which_knob_is_on_it_and_not_only_which_drum():
    """**The routing grid shows the drums; the knobs had nothing.** A learned binding was
    visible for exactly as long as the learn, and afterwards the only way to find out which
    knob moved a dial was to turn all of them and watch the strip."""
    from ganlive.control.midi import EncoderMap, control_name, nrpn_number, parse_controls

    knobs = EncoderMap(parse_controls("16=noise,2:17=se_256,n1.3=dir1,3:16=noise"))
    assert knobs.where("se_256") == "2:17"
    assert knobs.where("dir1") == control_name(-1, nrpn_number(1, 3)) == "n1.3"
    assert knobs.where("noise") == "16 3:16", "both, because both move it"
    assert knobs.where("spread") == "", "and a dial no knob is on says nothing"

    # The spelling is the flag's own, so what the strip shows can be typed back in.
    assert parse_controls("2:17=se_256") == {(1, 17): "se_256"}


def test_a_knob_holds_a_dial_through_the_same_seam_a_hand_does():
    """The last piece of "the Rytm controls the knobs": no hole is cut in the frame loop for
    the hardware, and everything the strip already shows about a held dial shows an encoder's
    holds for free."""
    from ganlive.control.midi import EncoderMap, parse_controls
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0, layout=fastgan())
    knobs = EncoderMap(parse_controls("16=noise, 2:17=se_256"))

    assert knobs.apply(runner, 5, 16, 0) == "noise"
    assert runner.hands["noise"] == pytest.approx(0.0)
    assert knobs.apply(runner, 5, 16, 127) == "noise"
    assert runner.hands["noise"] == pytest.approx(1.0)

    assert knobs.apply(runner, 5, 17, 127) is None
    assert "se_256" not in runner.hands
    assert knobs.apply(runner, 1, 17, 64) == "se_256"
    assert runner.hands["se_256"] == pytest.approx(64 / 127)

    assert set(runner.hands) == {"noise", "se_256"}

    assert knobs.unmapped == {(5, 17): 1}

    knobs.release(runner, "noise")
    assert set(runner.hands) == {"se_256"}
    knobs.release(runner)
    assert runner.hands == {}


def test_a_knob_mapping_names_any_dial_and_is_checked_against_the_model_that_plays():
    """**Which dials exist is the loaded model's business, and this parses before one opens.**

    Checking a mapping against this project's own `DIALS` refused every converted StyleGAN2's
    whole MODEL block from `--cc` and `--pressure` -- and did worse than refuse to a learned
    one. `l` on `w_fine` binds and `flush` writes `1:16=w_fine` in the flag's own words; the
    next launch read it back, raised here, and `play.remembered` swallowed it. The learn
    was undone and the only trace was one line about a file that "does not parse"."""
    from ganlive.control.midi import EncoderMap, format_controls, parse_controls
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    assert parse_controls("") == {}
    assert parse_controls("16=noise") == {(-1, 16): "noise"}
    assert parse_controls(" 3:22 = se_64 ") == {(2, 22): "se_64"}
    assert parse_controls("16=w_fine") == {(-1, 16): "w_fine"}, (
        "a converted StyleGAN2's own dial, which this used to refuse outright")
    with pytest.raises(ValueError):
        parse_controls("16=")               # a mapping with no dial at all is still a typo

    # The whole round trip a learn takes: bound on the strip, written down, read back.
    knobs = EncoderMap({})
    knobs.learning = "w_fine"
    knobs.apply(PresetRunner(FIXTURES["still"], INDEX, 60.0, layout=fastgan()), 0, 16, 64)
    assert parse_controls(format_controls(knobs.controls)) == knobs.controls

    # And the check that used to live here, moved to where the model is known.
    sg2 = _surface.stylegan2(("w_fine",), (0.5,), ((),), (25.0,))
    assert not EncoderMap(parse_controls("16=w_fine")).unreachable(sg2)
    stray = EncoderMap(parse_controls("16=se_256,17=punch")).unreachable(sg2)
    assert "se_256" in stray and "punch" in stray, (
        "a dial this model does not have, and a name no model ever will, read the same here "
        "and are both worth saying")
    assert not EncoderMap({}).unreachable(sg2), "nothing wired is nothing to report"


def test_a_handler_that_raises_drops_one_message_and_not_every_message_after_it():
    """**This is how a live set goes deaf halfway through and nothing says so.** `run` had no guard: a handler
    raising unwound the thread, `run` returned, and every clock, note, knob and pad after that moment was
    gone for the rest of the session. The end-of-run counts then show whatever arrived before the fault, so
    a run that died at minute two reads exactly like a machine that was never sending -- and the fix it
    reports, CLOCK SEND = ON, is the wrong one."""
    from ganlive.control.midi import NOTE_ON, ClockReader, MusicalClock

    seen = []

    def explodes(_channel, note, _velocity):
        seen.append(note)
        if note == 2:
            raise ValueError("boom")

    reader = ClockReader(MusicalClock(), on_note=explodes)

    class _OneBurst:
        """One burst, then it stops the reader -- `run` polls until `stop_flag`."""

        def __init__(self, events):
            self._events = list(events)

        def poll(self):
            if not self._events:
                reader.stop_flag = True
                return False
            return True

        def read(self, _n):
            out, self._events = self._events, []
            return [[e, 0] for e in out]

    reader.open_ports = lambda: [
        ("fake", _OneBurst([[NOTE_ON, 1, 100], [NOTE_ON, 2, 100], [NOTE_ON, 3, 100]]))]
    reader.run()

    assert seen == [1, 2, 3], "the messages after the raising one never arrived"
    assert reader.counts.get("note_on") == 3, "counting stopped at the fault"
    assert reader.trouble().startswith("1 MIDI message"), reader.trouble()
    assert "boom" in reader.trouble(), reader.trouble()


def test_pressure_is_classified_at_all_which_it_previously_was_not():
    """`dispatch` had no branch for polyphonic aftertouch, so the one continuous gesture the
    machine has could never reach anything downstream. Measured on the wire at 122 messages a
    minute from ordinary playing, and dropped every one of them."""
    from ganlive.control.midi import AFTERTOUCH_POLY, dispatch
    from ganlive.walk import MusicalClock

    clock = MusicalClock()
    assert dispatch(clock, AFTERTOUCH_POLY, 3, 90) == "aftertouch_poly"
    assert dispatch(clock, AFTERTOUCH_POLY + 0x0D, 3, 90) == "aftertouch_poly", "any channel"
    assert clock.beats == 0.0


def test_a_pad_leaned_on_holds_a_dial_and_gives_it_back_when_released():
    """**The one real difference from a knob**, and why this is a subclass rather than a flag:
    a knob parked at zero means zero, a pad nobody is touching means nothing. Everything else
    is `EncoderMap`'s code and is not copied."""
    from ganlive.control.midi import EncoderMap, PressureMap, parse_pressure
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(Preset(name="t", blurb="", dials={"noise": 0.1, "se_256": 0.1}), INDEX,
                         60.0, layout=fastgan())
    pads = PressureMap(parse_pressure("BD=noise,SD=se_256"))
    assert pads.controls == {(-1, INDEX["BD"]): "noise", (-1, INDEX["SD"]): "se_256"}

    assert pads.apply(runner, 13, INDEX["BD"], 127) == "noise"
    assert runner.held_by(pads.source) == {"noise": 1.0}
    pads.apply(runner, 13, INDEX["BD"], 64)
    assert runner.held_by(pads.source)["noise"] == pytest.approx(64 / 127)
    assert pads.apply(runner, 13, INDEX["BD"], 0) == "noise"
    assert runner.held_by(pads.source) == {}
    assert pads.seen == 3, "held is current state and cannot say the machine ever spoke"

    pads.apply(runner, 13, INDEX["CH"], 100)
    assert pads.unmapped == {(13, INDEX["CH"]): 1}

    knobs = EncoderMap({(-1, 35): "noise", (-1, 36): "se_256"})
    assert knobs.source != pads.source, "a source cannot outrank itself"
    knobs.apply(runner, 13, 35, 20)
    knobs.apply(runner, 13, 36, 127)
    pads.apply(runner, 13, INDEX["BD"], 127)
    assert runner.hands["noise"] == 1.0, "the parked knob won a dial being actively squeezed"
    assert runner.hands["se_256"] == 1.0, "the squeeze wiped a knob it was not touching"
    pads.apply(runner, 13, INDEX["BD"], 0)
    assert runner.hands["noise"] == pytest.approx(20 / 127)


def test_a_pressure_wiring_refuses_a_name_that_is_not_a_track():
    """A track typo is a pad that can never fire, and no model can make it one. A dial typo is
    reported against the loaded model instead -- see the knob mapping's own test for why."""
    from ganlive.control.midi import parse_pressure
    from ganlive.control.tracks import INDEX

    assert parse_pressure("BD=w_fine") == {(-1, INDEX["BD"]): "w_fine"}
    with pytest.raises(ValueError, match="unknown track"):
        parse_pressure("XX=noise")
    with pytest.raises(ValueError):
        parse_pressure("BD=")


def test_nrpn_arrives_as_one_fourteen_bit_control():
    """Four CCs in, one control out, with the low byte as an update rather than a wait."""
    from ganlive.control.midi import DATA_LSB, DATA_MSB, NRPN_LSB, NRPN_MSB, Nrpn, nrpn_number

    n = Nrpn()
    assert n.feed(0, NRPN_MSB, 1) is None
    assert n.feed(0, NRPN_LSB, 3) is None
    assert n.feed(0, DATA_MSB, 64) == (nrpn_number(1, 3), 64 << 7, 16383)
    assert n.feed(0, DATA_LSB, 5) == (nrpn_number(1, 3), 64 << 7 | 5, 16383)
    assert n.feed(1, DATA_MSB, 9) == (6, 9, 127), "data entry with no parameter named is a plain CC"
    assert n.feed(0, 17, 9) is None, "an ordinary CC is not the state machine's business"


def test_the_reader_routes_nrpn_and_plain_ccs_to_the_same_handler():
    from ganlive.control.midi import CONTROL_CHANGE, ClockReader, nrpn_number

    got = []
    reader = ClockReader(MusicalClock(120.0), on_control=lambda *a: got.append(a))
    for cc, value in ((99, 1), (98, 3), (6, 64), (38, 5), (17, 100)):
        reader._handle([CONTROL_CHANGE + 2, cc, value, 0], 0.0)
    reader._handle([CONTROL_CHANGE + 5, 6, 50, 0], 0.0)     # data entry with no NRPN armed
    assert got == [(2, nrpn_number(1, 3), 64 << 7, 16383),
                   (2, nrpn_number(1, 3), 64 << 7 | 5, 16383),
                   (2, 17, 100, 127),
                   (5, 6, 50, 127)]


def test_the_knob_map_reads_nrpns_and_writes_itself_back_in_the_same_words():
    from ganlive.control.midi import format_controls, nrpn_number, number_name, parse_controls

    assert parse_controls("n1.3=dir1") == {(-1, nrpn_number(1, 3)): "dir1"}
    assert parse_controls("2:N1.3=dir1") == {(1, nrpn_number(1, 3)): "dir1"}
    assert number_name(nrpn_number(1, 3)) == "n1.3"
    assert number_name(17) == "17"
    controls = parse_controls("16=noise,2:17=se_256,n1.3=dir1")
    assert parse_controls(format_controls(controls)) == controls


def test_learn_binds_the_next_control_to_the_focused_dial_and_writes_it_down(tmp_path):
    """Click a dial, turn a knob: the pair is the map now, on disk, in `--cc`'s own words."""
    from ganlive.control.midi import EncoderMap, parse_controls
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0, layout=fastgan())
    saved = tmp_path / "cc.txt"
    knobs = EncoderMap({(-1, 16): "noise"}, remember=saved)

    knobs.learning = "se_256"
    assert knobs.apply(runner, 1, 17, 64) == "se_256", "bound and applied in the same turn"
    assert knobs.learning is None
    assert knobs.controls == {(-1, 16): "noise", (1, 17): "se_256"}
    assert not saved.exists(), "the reader's thread never touches the disk"
    knobs.flush()
    assert parse_controls(saved.read_text("utf-8")) == knobs.controls
    assert knobs.learned == ["2:17=se_256"]

    knobs.learning = "noise"
    knobs.apply(runner, 3, 20, 10)
    assert (-1, 16) not in knobs.controls, "a dial has one knob; relearning moves it"
    assert knobs.controls[(3, 20)] == "noise"

    knobs.learning = "dir1"
    knobs.learning = None
    assert knobs.apply(runner, 0, 99, 1) is None, "disarmed, so an unmapped control stays unmapped"

    assert knobs.apply(runner, 1, 17, 8191, top=16383) == "se_256"
    assert runner.hands["se_256"] == pytest.approx(0.5, abs=1e-3), "14-bit scales by its own top"


def test_a_machine_nobody_named_gets_advice_about_itself():
    """Four tools used to print one drum machine's menu paths at whoever ran them.

    Telling a Launchkey owner to set `MIDI CONFIG > PORT CONFIG > ENCODER DEST` is worse
    than saying nothing: it names a menu their hardware does not have."""
    from ganlive.control.machine import GENERIC, RYTM, profile
    from ganlive.control.midi import EncoderMap

    assert profile("") is GENERIC and profile("launchkey") is GENERIC
    assert profile("rytm") is RYTM, "matched on the words the user already typed"
    assert profile("Elektron Analog Rytm MKII") is RYTM, "anywhere in the port name"
    assert profile("", "Rytm Overbridge") is RYTM, "an audio device name counts too"

    said = EncoderMap({(0, 16): "noise"}, machine=GENERIC).SILENCE
    assert "ENCODER DEST" not in said and "MIDI CONFIG" not in said
    assert "knob/CC output setting" in said, said

    said = EncoderMap({(0, 16): "noise"}, machine=RYTM).SILENCE
    assert "ENCODER DEST" in said, "the machine it was built against keeps its exact words"


def test_every_silence_a_tool_reports_is_in_the_machines_own_words():
    """The generic profile must never leak a menu path from the one known machine."""
    from ganlive.control.machine import GENERIC, KNOWN
    from ganlive.tools.play import silence_words

    menus = [w for m in KNOWN for w in (m.clock, m.transport, m.notes, m.encoders)]
    for notes in (True, False):
        for audio in (True, False):
            _kind, fix = silence_words(notes, audio, GENERIC)
            for menu in menus:
                assert menu not in fix, f"a named machine's words reached everyone: {fix}"
