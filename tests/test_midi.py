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
from tests.support import _runner


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
    """A drum machine sends note-ons and controller moves down the same cable as its clock;
    they are classified but none of them may disturb the position."""
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
    """The reader delivers poly aftertouch to its pressure handler, alongside notes and CCs.
    Other tests cover classifying the byte and acting on it; this covers the link between."""
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
    """No MIDI input is a normal setup (some machines offer USB audio or USB MIDI, not both),
    so the reader must fall back to the internal tempo and say so."""
    from ganlive.control.midi import ClockReader

    reader = ClockReader(MusicalClock(120.0), port_match="nothing-matches-this")
    assert reader.open_ports() == []
    said = reader.describe()
    assert said.startswith("MIDI:"), said
    assert "ignoring the machine" in said or "no MIDI" in said, said


def test_a_midi_port_filter_that_matched_nothing_says_so():
    """A typo in `--midi-port` gets its own message, naming the ports it missed, distinct from
    a machine with no MIDI ports at all; both otherwise look like a picture at its own tempo."""
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


def test_the_preflight_tools_count_through_the_dispatch_the_live_tool_runs():
    """`doctor` answers whether the clock reaches the instrument, so it must classify
    messages with the same code `play` runs rather than a copy of its own."""
    from ganlive.control import midi
    from ganlive.tools import doctor

    assert doctor.Traffic is midi.Traffic
    for name in ("CONTINUE", "SPP", "SONG_POSITION"):
        assert not hasattr(doctor, name), f"{name} is a second copy of a midi.py constant"

    traffic = midi.Traffic()
    period = MusicalClock.pulse_period(120.0)
    for i in range(MusicalClock.PPQN * 4 + 1):
        assert traffic.take([midi.CLOCK, 0, 0, 0], i * period) == "clock"
    assert traffic.take([midi.NOTE_ON + 9, 36, 100, 0], 0.0) == "note_on"
    assert traffic.take([midi.NOTE_ON + 9, 36, 0, 0], 0.0) == "note_off"
    assert traffic.clock.source == "midi", "the messages went through `dispatch`"
    bpm, span = traffic.tempo()
    assert bpm == pytest.approx(120.0) and span == pytest.approx(2.0)
    assert traffic.notes == {9: {36: 1}}
    assert traffic.note_lines() == ["  channel 10  36:1"]


def test_every_status_has_a_name_and_the_instruments_kinds_are_classifys():
    from ganlive.control import midi
    from ganlive.tools.doctor import name_of

    assert name_of(midi.CLOCK) == "clock" and name_of(midi.NOTE_ON, 0) == "note_off"
    assert name_of(0x80) == "note_off" and name_of(0xE3) == "pitch_bend"
    assert name_of(0xFE) == "active_sensing" and name_of(0xF1) == "system_0xf1"


class _Port:
    """A MIDI input that hands over one burst of messages, then nothing."""

    def __init__(self, events):
        self._events = [list(e) for e in events]
        self.closed = False

    def poll(self):
        return bool(self._events)

    def read(self, _n):
        out, self._events = self._events, []
        return [[e, 0] for e in out]

    def close(self):
        self.closed = True


def test_the_doctors_midi_watch_counts_and_reports_what_arrived(capsys):
    """`doctor --listen`, `--drive`, `--meter --midi` and `--learn` all read MIDI through one
    watch: kinds per phase, transport, knobs, and note-ons handed to `--learn`."""
    from ganlive.control import midi
    from ganlive.control.machine import GENERIC
    from ganlive.tools.doctor import MidiWatch

    watch = MidiWatch("nothing-matches-this", GENERIC)
    port = _Port([[midi.START, 0, 0, 0], [midi.NOTE_ON + 9, 36, 100, 0],
                  [midi.CONTROL_CHANGE, 16, 64, 0], [0xFE, 0, 0, 0]])
    watch.inputs = [("fake", port)]
    struck = []
    watch.poll(on_note=lambda note, _now: struck.append(note))
    watch.close()

    assert struck == [36] and port.closed
    assert watch.transport == ["START"]
    assert watch.controls[0][16] == 1
    assert watch.total("active_sensing") == 1 and watch.total("note_on") == 1
    watch.report()
    said = capsys.readouterr().out
    assert "NO CLOCK" in said and "channel 10  36:1" in said and '--cc "1:16=<dial>"' in said


def test_a_doctor_that_hears_nothing_says_what_to_check(capsys):
    from ganlive.control.machine import RYTM
    from ganlive.tools.doctor import MidiWatch

    watch = MidiWatch("nothing-matches-this", RYTM)
    watch.inputs = [("fake", _Port([]))]
    watch.poll()
    watch.report()
    said = capsys.readouterr().out
    assert f"NOTHING AT ALL -- is {RYTM.name} on this port?" in said


def test_a_dial_says_which_knob_is_on_it_and_not_only_which_drum():
    """A dial can report which knob (CC or NRPN, in `--cc` spelling) is bound to it, so the
    strip can show a binding after it is learned."""
    from ganlive.control.midi import EncoderMap, control_name, nrpn_number, parse_controls

    knobs = EncoderMap(parse_controls("16=noise,2:17=se_256,n1.3=dir1,3:16=noise"))
    assert knobs.where("se_256") == "2:17"
    assert knobs.where("dir1") == control_name(-1, nrpn_number(1, 3)) == "n1.3"
    assert knobs.where("noise") == "16 3:16", "both, because both move it"
    assert knobs.where("spread") == "", "and a dial no knob is on says nothing"

    # The spelling is the flag's own, so what the strip shows can be typed back in.
    assert parse_controls("2:17=se_256") == {(1, 17): "se_256"}


def test_a_knob_holds_a_dial_through_the_same_seam_a_hand_does():
    """A hardware knob holds a dial through the same mechanism as the other holders (mouse,
    knobs, pads), so the frame loop needs no special case and the strip shows it as held."""
    from ganlive.control.midi import EncoderMap, parse_controls

    runner = _runner()
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
    """Parsing accepts any dial name, since mappings are read before a model is loaded; dials
    the loaded model lacks are reported by `unreachable` instead. A learned binding to a
    model-specific dial must therefore survive a save and reload."""
    from ganlive.control.midi import EncoderMap, format_controls, parse_controls

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
    knobs.apply(_runner(), 0, 16, 64)
    assert parse_controls(format_controls(knobs.controls)) == knobs.controls

    # The dial-name check, made against the model that is loaded.
    sg2 = _surface.stylegan2(("w_fine",), (0.5,), ((),), (25.0,))
    assert not EncoderMap(parse_controls("16=w_fine")).unreachable(sg2)
    stray = EncoderMap(parse_controls("16=se_256,17=punch")).unreachable(sg2)
    assert "se_256" in stray and "punch" in stray, (
        "a dial this model does not have, and a name no model ever will, read the same here "
        "and are both worth saying")
    assert not EncoderMap({}).unreachable(sg2), "nothing wired is nothing to report"


def test_a_handler_that_raises_drops_one_message_and_not_every_message_after_it():
    """A handler that raises loses only its own message: the reader thread keeps running and
    reports the fault, rather than silently ignoring all MIDI for the rest of the run."""
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
    """`dispatch` classifies polyphonic aftertouch on any channel, without moving the clock;
    it is the pads' one continuous gesture."""
    from ganlive.control.midi import AFTERTOUCH_POLY, dispatch
    from ganlive.walk import MusicalClock

    clock = MusicalClock()
    assert dispatch(clock, AFTERTOUCH_POLY, 3, 90) == "aftertouch_poly"
    assert dispatch(clock, AFTERTOUCH_POLY + 0x0D, 3, 90) == "aftertouch_poly", "any channel"
    assert clock.beats == 0.0


def test_a_pad_leaned_on_holds_a_dial_and_gives_it_back_when_released():
    """Unlike a knob parked at zero, which holds zero, a released pad lets go of its dial; and
    an actively pressed pad outranks a parked knob on the same dial."""
    from ganlive.control.kit import INDEX
    from ganlive.control.midi import EncoderMap, PressureMap, parse_pressure
    from ganlive.presets import PresetRunner

    runner = PresetRunner(Preset(name="t", blurb="", dials={"noise": 0.1, "se_256": 0.1}), INDEX,
                         60.0, layout=fastgan())
    pads = PressureMap(parse_pressure("BD=noise,SD=se_256"))
    assert pads.controls == {(-1, INDEX["BD"]): "noise", (-1, INDEX["SD"]): "se_256"}

    assert pads.apply(runner, 13, INDEX["BD"], 127) == "noise"
    assert runner.held_by(pads.SOURCE) == {"noise": 1.0}
    pads.apply(runner, 13, INDEX["BD"], 64)
    assert runner.held_by(pads.SOURCE)["noise"] == pytest.approx(64 / 127)
    assert pads.apply(runner, 13, INDEX["BD"], 0) == "noise"
    assert runner.held_by(pads.SOURCE) == {}
    assert pads.seen == 3, "held is current state and cannot say the machine ever spoke"

    pads.apply(runner, 13, INDEX["CH"], 100)
    assert pads.unmapped == {(13, INDEX["CH"]): 1}

    knobs = EncoderMap({(-1, 35): "noise", (-1, 36): "se_256"})
    assert knobs.SOURCE != pads.SOURCE, "a source cannot outrank itself"
    knobs.apply(runner, 13, 35, 20)
    knobs.apply(runner, 13, 36, 127)
    pads.apply(runner, 13, INDEX["BD"], 127)
    assert runner.hands["noise"] == 1.0, "the parked knob won a dial being actively squeezed"
    assert runner.hands["se_256"] == 1.0, "the squeeze wiped a knob it was not touching"
    pads.apply(runner, 13, INDEX["BD"], 0)
    assert runner.hands["noise"] == pytest.approx(20 / 127)


def test_a_pressure_wiring_refuses_a_name_that_is_not_a_track():
    """An unknown track name is refused at parse time, since no model could make that pad fire;
    unknown dial names are left for the loaded model to report."""
    from ganlive.control.kit import INDEX
    from ganlive.control.midi import parse_pressure

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

    runner = _runner()
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
    """An unrecognised machine gets generic advice; menu paths are given only for a machine
    the port or audio device name identifies, since others do not have those menus."""
    from ganlive.control.machine import GENERIC, RYTM, profile
    from ganlive.control.midi import EncoderMap

    assert profile("") is GENERIC and profile("launchkey") is GENERIC
    assert profile("rytm") is RYTM, "matched on the words the user already typed"
    assert profile("Elektron Analog Rytm MKII") is RYTM, "anywhere in the port name"
    assert profile("", "Rytm Overbridge") is RYTM, "an audio device name counts too"

    said = EncoderMap({(0, 16): "noise"}, machine=GENERIC).silence()
    assert "ENCODER DEST" not in said and "MIDI CONFIG" not in said
    assert "knob/CC output setting" in said, said

    said = EncoderMap({(0, 16): "noise"}, machine=RYTM).silence()
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


def test_no_tool_spells_a_known_machines_menu_itself():
    """No tool's source hard-codes a known machine's menu words; only the machine profile may
    say them, so every tool gives the same machine-appropriate advice."""
    import pathlib

    import ganlive.tools
    from ganlive.control.machine import KNOWN

    words = {"MIDI CONFIG", "TRANSPORT SEND", "CLOCK SEND", "ENCODER DEST", "TRK SEND"}
    words |= {w for m in KNOWN for w in (m.clock, m.transport, m.notes, m.encoders, m.stems) if w}
    for source in pathlib.Path(ganlive.tools.__file__).parent.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        leaked = sorted(w for w in words if w in text)
        assert not leaked, f"{source.name} spells {leaked} itself"
