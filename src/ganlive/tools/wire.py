"""Do the sequencer's trigs reach MIDI? Drives the machine and watches both wires at once."""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from ganlive.control.midi import (
    AFTERTOUCH_POLY,
    CLOCK,
    CONTINUE,
    CONTROL_CHANGE,
    NOTE_ON,
    SONG_POSITION,
    START,
    STOP,
    find_ports,
)
from ganlive.control.tracks import TRACKS, output_mode
from ganlive.walk import MusicalClock

VOICE = {0x80: "note_off", NOTE_ON: "note_on", AFTERTOUCH_POLY: "aftertouch_poly",
         CONTROL_CHANGE: "control_change", 0xC0: "program_change",
         0xD0: "aftertouch_channel", 0xE0: "pitch_bend"}
SYSTEM = {0xF0: "sysex", SONG_POSITION: "song_position", CLOCK: "clock", START: "start",
          CONTINUE: "continue", STOP: "stop", 0xFE: "active_sensing", 0xFF: "reset"}

YIELD_S = 0.0005


def name_of(status: int) -> str:
    if status >= 0xF0:
        return SYSTEM.get(status, f"system_{status:#04x}")
    return VOICE.get(status & 0xF0, f"unknown_{status:#04x}")


class Mark(NamedTuple):
    """A phase boundary: onsets so far, note-ons so far, and audio consumed so far."""

    hits: np.ndarray | None
    notes: int
    heard: float


class Wire:
    """Every message that arrived, and which phase of the run it arrived in."""

    def __init__(self) -> None:
        self.by_phase: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.notes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.velocities: dict[int, list[int]] = defaultdict(list)
        self.phase = "listening"

    def take(self, status: int, d1: int, d2: int) -> None:
        kind = name_of(status)
        if kind == "note_on" and d2 == 0:
            kind = "note_off"                       # velocity 0 is a release, by convention
        self.by_phase[self.phase][kind] += 1
        if kind == "note_on":
            self.notes[status & 0x0F][d1] += 1
            self.velocities[d1].append(d2)

    def total(self, kind: str) -> int:
        return sum(counts[kind] for counts in self.by_phase.values())

    def kinds(self) -> list[str]:
        return sorted({k for counts in self.by_phase.values() for k in counts},
                      key=lambda k: -self.total(k))

    def bpm(self, span: float) -> float | None:
        """Tempo from counted pulses over a known span, never from an estimate."""
        pulses = self.total("clock")
        if pulses < MusicalClock.PPQN or span <= 0:
            return None
        return pulses / MusicalClock.PPQN / span * 60


def open_ports(port_match: str, want_output: bool):
    import pygame.midi

    ins, outs, _rejected = find_ports(port_match)
    if not ins:
        raise SystemExit(f"no MIDI input matching {port_match!r}. Is the machine connected over "
                         f"USB MIDI, and is anything that claims its port exclusively closed?")
    print("MIDI in  : " + ", ".join(f"{i} {n}" for i, n in ins))
    print("MIDI out : " + (", ".join(f"{i} {n}" for i, n in outs) or "NONE"))
    if want_output and not outs:
        raise SystemExit("--drive needs a MIDI output to send Start and clock on.")
    return ([pygame.midi.Input(i) for i, _ in ins],
            pygame.midi.Output(outs[0][0]) if (outs and want_output) else None,
            pygame.midi)


def open_audio():
    """The ASIO stream and the onset detector, or `(None, None)` if there is no card."""
    try:
        import sounddevice as sd

        from ganlive.control.audio import pick_input
        from ganlive.control.features import FeatureConfig, FeatureExtractor

        device, info, channels = pick_input(sd)
    except Exception as exc:                                      # noqa: BLE001
        print(f"audio    : none ({type(exc).__name__}: {exc})")
        return None, None
    print(f"audio    : {device} {info['name']}, {channels} ch")
    extractor = FeatureExtractor(channels, 48000, FeatureConfig())

    def callback(indata, _frames, _time_info, _status):
        extractor.push(indata)

    stream = sd.InputStream(device=device, channels=channels, samplerate=extractor.sr,
                            blocksize=256, dtype="float32", callback=callback)
    return stream, extractor


def run(seconds: float, port_match: str, drive: bool, bpm: float) -> int:
    ports, out, pm = open_ports(port_match, drive)
    stream, heard = open_audio()
    wire = Wire()
    marks: dict[str, Mark] = {}
    started = time.perf_counter()

    def drain() -> None:
        for port in ports:
            while port.poll():
                for event, _ts in port.read(128):
                    wire.take(event[0], event[1], event[2])

    def mark(label: str) -> None:
        marks[label] = Mark(heard.hits.copy() if heard is not None else None,
                            wire.by_phase[wire.phase].get("note_on", 0),
                            heard.heard() if heard is not None else 0.0)

    def elapse(span: float, pulse: float | None = None) -> None:
        """Drain for `span` seconds, sending a clock pulse every `pulse` if asked."""
        end = next_pulse = time.perf_counter()
        end += span
        while (now := time.perf_counter()) < end:
            if pulse is not None and now >= next_pulse:
                out.write_short(CLOCK)
                next_pulse += pulse
            drain()
            time.sleep(YIELD_S)

    if stream is not None:
        stream.start()
    try:
        mark("start")
        elapse(4.0 if drive else seconds)
        mark("listened")
        if drive:
            print(f"\n>>> Start, then {bpm:g} BPM for {seconds:g}s")
            out.write_short(START)
            wire.phase = "driven"
            mark("drive0")
            elapse(seconds, pulse=60.0 / (bpm * MusicalClock.PPQN))
            mark("drive1")
            print(">>> Stop")
            out.write_short(STOP)
            wire.phase = "after"
            elapse(3.0)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        for port in ports:
            port.close()
        if out is not None:
            out.close()
        pm.quit()

    return report(wire, marks, drive, time.perf_counter() - started)


def report(wire: Wire, marks: dict[str, Mark], drove: bool, span: float) -> int:
    print(f"\n{'kind':<20}{'total':>8}   by phase")
    for kind in wire.kinds():
        where = " ".join(f"{p}={wire.by_phase[p][kind]}"
                         for p in ("listening", "driven", "after") if wire.by_phase[p][kind])
        print(f"{kind:<20}{wire.total(kind):>8}   {where}")
    if not wire.by_phase:
        print("  NOTHING AT ALL -- the Control Panel is almost certainly closed.")
    if drove:
        print("\n  Sent messages arrive back on the input port, so the transport counts above")
        print("  include this script's own Start, clock and Stop. Only note-ons are the")
        print("  machine's, because nothing here ever sends one.")

    tempo = wire.bpm(span)
    print(f"\nclock pulses : {wire.total('clock')}" + (f"  ({tempo:.1f} BPM)" if tempo else ""))

    had_audio = marks["start"].hits is not None
    if had_audio:
        windows = [("idle  ", "start", "listened")]
        if drove:
            windows.append(("driven", "drive0", "drive1"))
        for label, a, b in windows:
            hits = marks[b].hits - marks[a].hits
            struck = ", ".join(f"{TRACKS[i]}:{int(n)}" for i, n in enumerate(hits) if n)
            print(f"audio, {label}: {int(hits.sum())} onsets over "
                  f"{marks[b].heard - marks[a].heard:.1f}s   {struck or 'silence'}")

    if wire.notes:
        print("\nnote-ons per channel (shown 1-based, the wire is 0-based):")
        for ch in sorted(wire.notes):
            items = ", ".join(f"{n}:{c}" for n, c in sorted(wire.notes[ch].items()))
            print(f"  channel {ch + 1:<3} {items}")
        every = sorted({n for notes in wire.notes.values() for n in notes})
        mode = output_mode(wire.notes)
        advice = {"auto": "the note is the track. Leave --midi-channels OFF.",
                  "track": "the channel is the track. Run `ganlive play --midi-channels 1-12`.",
                  "mixed": "sequencer on track channels, pads on the auto channel -- the "
                           "machine's normal shape. Run `ganlive play --midi-channels 1-12`. "
                           "the auto channel falls through to the note rule.",
                  None: "one channel of high notes is a single track in TRACK CH, or something "
                        "that is not the kit. Play more of the pattern."}[mode]
        print(f"  notes {every[0]}-{every[-1]} on {len(wire.notes)} channel(s): "
              f"{(mode or 'AMBIGUOUS').upper()}{' CH' if mode else ''}, {advice}")
        vel = {n: sorted(set(v)) for n, v in sorted(wire.velocities.items())}
        if any(len(v) > 1 for v in vel.values()):
            print("  velocity: varies, so MIDI carries how hard each hit was")
        else:
            print(f"  velocity: one value per note -- {({n: v[0] for n, v in vel.items()})}")
    else:
        print("\nnote-ons     : NONE")

    print("\nVERDICT")
    if not drove:
        print("  Listening only. Re-run with --drive to start the sequencer and get an answer.")
        return 0
    notes = marks["drive1"].notes - marks["drive0"].notes
    if not had_audio:
        print(f"  NO AUDIO, so nothing certifies the sequencer played. {notes} note-ons.")
        return 1
    onsets = int((marks["drive1"].hits - marks["drive0"].hits).sum())
    if onsets == 0:
        print("  INCONCLUSIVE -- the sequencer never ran, so its MIDI silence means nothing.")
        print("  It is slaved to our clock, so check the pattern has trigs and is not muted.")
        return 1
    if notes == 0:
        print(f"  Sequencer trigs send NO MIDI notes. {onsets} drum onsets were heard while it")
        print("  played, and not one note-on came with them.")
        print("  This is a SETTING, not a hardware limit: OS 1.50 added a per-track MIDI send")
        print("  to the MKI. Turn it on for each track and run this again.")
        return 0
    print(f"  Sequencer trigs DO send MIDI notes: {notes} note-ons alongside {onsets} onsets.")
    print("  The instrument can be driven by MIDI alone -- all twelve tracks, BT and LT")
    print("  included, and the four shared analog voices come apart.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ganlive wire", description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--port", default="rytm", metavar="TEXT",
                    help="substring of the MIDI port to listen to. The default suits an "
                         "Analog Rytm; pass your own, or '' for every port")
    ap.add_argument("--bpm", type=float, default=166.0)
    ap.add_argument("--drive", action="store_true",
                    help="be the clock master: Start, pulse, Stop, and certify against audio")
    args = ap.parse_args(argv)
    return run(args.seconds, args.port, args.drive, args.bpm)

