"""A machine that is not an Analog Rytm: pads on other notes, tracks with no drum names.

The twelve tracks are the instrument's vocabulary; what a machine calls them and which note
each pad sends are its own business, and both are settings here rather than assumptions."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ganlive.control.audio import NoAudioDevice, pick_input
from ganlive.control.features import NoteFeatures
from ganlive.control.midi import parse_pressure
from ganlive.control.tracks import INDEX, output_mode, parse_notes, parse_track_channels

#: A General MIDI kit, which is what "another machine" usually means, and nothing about it
#: is consecutive: kick 36, snare 38, closed hat 42, open hat 46, crash 49.
GM = "36=BD,38=SD,42=CH,46=OH,49=CY"


def test_a_track_is_named_by_the_rytm_word_or_by_its_index():
    assert parse_track_channels("1=BD,2=3") == {0: INDEX["BD"], 1: 3}
    with pytest.raises(ValueError, match="out of range"):
        parse_track_channels("1=12")
    with pytest.raises(ValueError, match="or 0 to 11"):
        parse_track_channels("1=KICK")


def test_notes_are_a_first_note_or_a_map():
    assert parse_notes("0") == {i: i for i in range(12)}, "a Rytm's pads"
    assert parse_notes("36")[36] == 0 and parse_notes("36")[47] == 11
    gm = parse_notes(GM)
    assert gm == {36: INDEX["BD"], 38: INDEX["SD"], 42: INDEX["CH"], 46: INDEX["OH"],
                  49: INDEX["CY"]}
    with pytest.raises(ValueError, match="already BD"):
        parse_notes("36=BD,36=SD")
    with pytest.raises(ValueError, match="needs a first note"):
        parse_notes("")


def test_a_general_midi_kit_lands_each_pad_on_its_own_track():
    """The case a first-note offset cannot serve: 36 would put the snare on RS and the hats on
    MT and CY."""
    heard = NoteFeatures(12, notes=parse_notes(GM))
    assert heard.track_of(9, 38) == INDEX["SD"]
    assert heard.track_of(9, 42) == INDEX["CH"]
    assert heard.track_of(9, 46) == INDEX["OH"]
    assert heard.track_of(9, 37) is None, "a note the kit does not use names no track"
    consecutive = NoteFeatures(12, base_note=36)
    assert consecutive.track_of(9, 36) == 0 and consecutive.track_of(9, 47) == 11
    assert consecutive.track_of(9, 48) is None


def test_a_pad_is_pressed_on_the_note_its_kit_sends():
    gm = parse_notes(GM)
    assert parse_pressure("SD=dir1", gm) == {(-1, 38): "dir1"}
    assert parse_pressure(f"{INDEX['CH']}=noise", gm) == {(-1, 42): "noise"}
    assert parse_pressure("SD=dir1") == {(-1, INDEX["SD"]): "dir1"}, "a Rytm's press on 0-11"


def test_the_output_mode_is_read_against_the_kits_own_notes():
    """On a Rytm, 0-11 is AUTO CH and pitches are TRACK CH. On a GM kit on channel 10 the same
    notes are the pads, and calling them TRACK CH would send a user to the wrong flag."""
    gm = parse_notes(GM)
    assert output_mode({9: {36, 38, 42}}, gm) == "auto"
    assert output_mode({9: {36, 38, 42}}) is None, "read as a Rytm, these name nothing"
    assert output_mode({13: {0, 1, 5, 11}}) == "auto"


def _sd(devices, hostapis=("MME",), starts=True):
    """Just enough of `sounddevice` for the picker: a device table and host APIs."""
    opened = []

    class Stream:
        def __init__(self, **kw):
            opened.append(kw["device"])

        def start(self):
            if not starts:
                raise RuntimeError("busy")

        def stop(self):
            pass

        close = stop

    return SimpleNamespace(
        query_devices=lambda i=None: devices if i is None else devices[i],
        query_hostapis=lambda i=None: [{"name": n} for n in hostapis] if i is None
        else {"name": hostapis[i]},
        InputStream=Stream), opened


def test_an_explicit_input_is_taken_on_any_host_api():
    """A mixer on CoreAudio or a loopback on WASAPI: no ASIO, no Rytm in the name."""
    sd, opened = _sd([{"name": "Aggregate Device", "max_input_channels": 8, "hostapi": 0}],
                     hostapis=("Core Audio",))
    device, info, nch = pick_input(sd, 0)
    assert (device, nch) == (0, 8) and info["name"] == "Aggregate Device"
    assert opened == [0], "opened once to prove it starts"


def test_an_explicit_input_that_will_not_start_says_so():
    sd, _ = _sd([{"name": "Taken", "max_input_channels": 2, "hostapi": 0}], starts=False)
    with pytest.raises(NoAudioDevice, match="will not start"):
        pick_input(sd, 0)
    sd, _ = _sd([{"name": "Speakers", "max_input_channels": 0, "hostapi": 0}])
    with pytest.raises(NoAudioDevice, match="no inputs"):
        pick_input(sd, 0)


def test_a_named_input_is_found_on_any_host_api_when_asked_and_a_rytm_only_on_asio():
    table = [{"name": "Scarlett 2i2", "max_input_channels": 2, "hostapi": 0}]
    sd, _ = _sd(table, hostapis=("Core Audio",))
    with pytest.raises(NoAudioDevice, match="--audio-name"):
        pick_input(sd)
    device, _info, nch = pick_input(sd, pattern="scarlett", hostapi=None)
    assert (device, nch) == (0, 2)


def test_a_kit_only_advertises_the_tracks_it_can_actually_reach():
    """The strip draws one drum light per entry here, and the end-of-run report names a
    silent track per entry that never fired. On a five-pad General MIDI kit, returning all
    twelve gave seven lights that could not light and seven faults that could not exist --
    the same defect the audio half had fixed by deferring to the caller's map."""
    notes = NoteFeatures(notes=parse_notes(GM))
    assert notes.channel_of() == {"BD": 0, "SD": 1, "CH": 8, "OH": 9, "CY": 10}

    # A Rytm's twelve consecutive pads still get twelve.
    assert NoteFeatures().channel_of() == dict(INDEX)

    # A machine identifying tracks by channel contributes those too.
    both = NoteFeatures(notes=parse_notes(GM), channels=parse_track_channels("1=BT"))
    assert "BT" in both.channel_of()
