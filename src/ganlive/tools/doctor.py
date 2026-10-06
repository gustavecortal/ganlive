"""What this machine's audio and MIDI actually offer. Run it with the controller on.

  (no mode)  list every audio input, the sample rates it claims, and which one would be used
  --meter    live per-channel levels: hit one pad at a time and see which channel moves
             (add --midi to count clock, transport, notes and knobs at the same time)
  --listen   is MIDI arriving -- notes, clock, transport -- and is audio?
  --drive    start the machine's sequencer from here and certify that its trigs send notes,
             matched against the drums heard on the audio input
  --learn    hit every pad: map which audio channel each drum arrives on, and remember it
             for `ganlive play`
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from ganlive.clock import MusicalClock
from ganlive.control.audio import (
    DEFAULT_MATCH,
    HOSTAPI,
    NoAudioDevice,
    input_stream,
    named_inputs,
    pick_input,
    require_sounddevice,
)
from ganlive.control.features import FeatureConfig, FeatureExtractor
from ganlive.control.kit import TRACKS, output_mode
from ganlive.control.machine import Machine, profile
from ganlive.control.midi import (
    CLOCK,
    START,
    STOP,
    Traffic,
    classify,
    find_ports,
    messages,
    open_inputs,
)
from ganlive.files import CHANNEL_MAP, remember
from ganlive.tools import parser
from ganlive.tools.drum_map import Strikes, report, report_recall

#: Sample rates tried, in order, when none is given.
RATES = (48000, 44100, 96000)

#: The onset detector's level floor, and how far above it a send should peak.
FLOOR = FeatureConfig().floor
HEADROOM = 8.0

#: How long the polling loops sleep between MIDI reads.
YIELD_S = 0.0005

#: `--seconds` when it is not given, per mode.
SECONDS = {"meter": 20.0, "listen": 20.0, "drive": 20.0, "learn": 75.0}

EXCLUSIVE = ("An exclusive audio API (ASIO, or WASAPI in exclusive mode) gives the device to "
             "ONE program at a time: close any DAW or control panel holding it, then retry.")

#: Names for the statuses `classify` ignores, so a report can still say what they were.
VOICE = {0x80: "note_off", 0xC0: "program_change", 0xD0: "aftertouch_channel",
         0xE0: "pitch_bend"}
SYSTEM = {0xF0: "sysex", 0xFE: "active_sensing", 0xFF: "reset"}


def name_of(status: int, data2: int = 0) -> str:
    """What any MIDI status byte is, including the kinds ganlive ignores."""
    known = classify(status, data2)
    if known is not None:
        return known
    if status >= 0xF0:
        return SYSTEM.get(status, f"system_{status:#04x}")
    return VOICE.get(status & 0xF0, f"unknown_{status:#04x}")


class MidiWatch:
    """Everything that arrives on the matching MIDI inputs, and the report on it.

    Counts every kind of message per phase of the run, note-ons per channel, clock pulses and
    tempo (through `midi.Traffic`, the classification `play` runs), transport and knobs."""

    def __init__(self, port_match: str = "", machine: Machine | None = None) -> None:
        self.machine = machine or profile(port_match)
        self.inputs, self.rejected, self.error = open_inputs(port_match)
        self.traffic = Traffic()
        self.by_phase: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.phase = "listening"
        self.transport: list[str] = []
        self.controls: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.velocities: dict[int, list[int]] = defaultdict(list)

    def describe(self) -> str:
        if not self.inputs:
            return ("MIDI: NO INPUT PORTS" + (f" ({self.error})" if self.error else "")
                    + ". If a machine is in a mode that claims its USB for audio, that is the "
                      "finding -- the clock needs another source.")
        return "MIDI in  : " + ", ".join(name for name, _port in self.inputs)

    def poll(self, on_note=None) -> None:
        """Read every waiting message. `on_note(note, now)` is called for each note-on."""
        now = time.perf_counter()
        for event, _ts in messages(port for _name, port in self.inputs):
            status, data1, data2 = event[0], event[1], event[2]
            what = self.traffic.take(event, now)
            self.by_phase[self.phase][name_of(status, data2)] += 1
            if what in ("start", "continue", "stop"):
                self.transport.append(what.upper())
            elif what == "song_position":
                self.transport.append(f"SPP={data1 | (data2 << 7)}")
            elif what == "control_change":
                self.controls[status & 0x0F][data1] += 1
            elif what == "note_on":
                self.velocities[data1].append(data2)
                if on_note is not None:
                    on_note(data1, now)

    def close(self) -> None:
        for _name, port in self.inputs:
            port.close()

    def total(self, kind: str) -> int:
        return sum(counts[kind] for counts in self.by_phase.values())

    def kinds(self) -> list[str]:
        return sorted({k for counts in self.by_phase.values() for k in counts},
                      key=lambda k: -self.total(k))

    def report(self) -> None:
        """What arrived, and what to switch on for whatever did not."""
        machine = self.machine
        if not self.inputs:
            print("MIDI: no input ports were present at all.")
            return
        print(f"\n{'kind':<20}{'total':>8}   by phase")
        for kind in self.kinds():
            where = " ".join(f"{p}={self.by_phase[p][kind]}"
                             for p in ("listening", "driven", "after") if self.by_phase[p][kind])
            print(f"{kind:<20}{self.total(kind):>8}   {where}")
        if not self.kinds():
            print(f"  NOTHING AT ALL -- is {machine.name} on this port? If it is, "
                  f"{machine.says('clock')} and {machine.transport}.")

        tempo = self.traffic.tempo()
        print(f"\nclock pulses : {self.traffic.pulses}"
              + (f"  ({tempo[0]:.1f} BPM over {tempo[1]:.1f} s)" if tempo else ""))
        if not self.traffic.pulses:
            print(f"  NO CLOCK. On {machine.name}, {machine.says('clock')}, and make sure it")
            print("  sends over USB. Without it the picture runs at its own internal tempo")
            print("  and will look entirely plausible while doing so.")
        print(f"transport    : {self.transport or 'NONE -- ' + machine.says('transport')}")

        notes = self.traffic.notes
        if notes:
            print("\nnote-ons per MIDI channel (note:count):")
            print("\n".join(self.traffic.note_lines()))
            every = sorted({n for per in notes.values() for n in per})
            mode = output_mode(notes)
            advice = {"auto": "the note is the track. Leave --midi-channels OFF.",
                      "track": "the channel is the track. Run `ganlive play --midi-channels "
                               "1-12`.",
                      "mixed": "sequencer on per-track channels, pads on one shared channel. "
                               "Run `ganlive play --midi-channels 1-12`; the shared channel "
                               "falls through to the note.",
                      None: "one channel of notes outside the kit's is a single track on its "
                            "own channel, or something that is not the kit. Play more of the "
                            "pattern."}[mode]
            print(f"  notes {every[0]}-{every[-1]} on {len(notes)} channel(s): "
                  f"{(mode or 'AMBIGUOUS').upper()}{' CH' if mode else ''}, {advice}")
            vel = {n: sorted(set(v)) for n, v in sorted(self.velocities.items())}
            if any(len(v) > 1 for v in vel.values()):
                print("  velocity: varies, so MIDI carries how hard each hit was")
            else:
                print(f"  velocity: one value per note -- {({n: v[0] for n, v in vel.items()})}")
        else:
            print("\nnote-ons     : NONE. Per-track identity is what MIDI is for here, so on "
                  f"{machine.name}, {machine.says('notes')}.")

        if self.controls:
            print("\ncontrol changes (turn one knob at a time; these are what --cc takes):")
            for ch in sorted(self.controls):
                for cc, n in sorted(self.controls[ch].items()):
                    print(f"  ch {ch + 1:2} cc {cc:3}: {n:5} messages   "
                          f"--cc \"{ch + 1}:{cc}=<dial>\"")
        else:
            print("\ncontrol changes : NONE. Without them the knobs cannot hold a dial; the")
            print("                  sliders and the drums still can.")


def _pick(sd, args, rate: int | None = None):
    """`(device, info, channels, samplerate)` for the input the options name.

    Tried at each rate in `RATES` unless one is given: a device locked to 44.1 kHz refuses
    48 kHz. Raises `NoAudioDevice` with the last reason."""
    trouble = None
    for sr in (rate,) if rate else RATES:
        try:
            device, info, nch = pick_input(sd, args.audio_device, args.audio_name, samplerate=sr)
            return device, info, nch, sr
        except NoAudioDevice as exc:
            trouble = exc
    raise trouble


def find_devices(sd, pattern=DEFAULT_MATCH):
    """Every input-capable device whose name matches, on any host API, with that API's name.

    Through `named_inputs`, the same search `pick_input` makes, so this reports on the
    devices ganlive would actually consider."""
    out = []
    for i in named_inputs(sd, pattern, hostapi=None):
        d = sd.query_devices(i)
        out.append((i, d, sd.query_hostapis(d["hostapi"])["name"]))
    return out


def cmd_list(sd, pattern=DEFAULT_MATCH) -> int:
    apis = [ha["name"] for ha in sd.query_hostapis()]
    print(f"PortAudio : {sd.get_portaudio_version()[1]}")
    print(f"host APIs : {', '.join(apis)}")
    if HOSTAPI not in apis:
        print(f"\n  {HOSTAPI} is missing, and it is this platform's multi-channel API.")
        if HOSTAPI == "ASIO":
            print("  sounddevice ships an ASIO build behind SD_ENABLE_ASIO; if it is absent")
            print("  here the wrong DLL was loaded.")
        print("  An interface on another API still works; pass --audio-device with its index.")

    hits = find_devices(sd, pattern)
    matched = {i for i, _d, _api in hits}

    print("\ninput-capable devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        api = sd.query_hostapis(d["hostapi"])["name"]
        print(f"  {i:3} [{api:12}] in={d['max_input_channels']:3} "
              f"out={d['max_output_channels']:3} sr={d['default_samplerate']:6.0f}  "
              f"{d['name']}{'  <--' if i in matched else ''}")

    print()
    if not hits:
        print(f"  Nothing named {pattern!r}. ganlive plays without any audio input --")
        print("  MIDI notes alone drive it -- so this is a finding, not a failure. A driver")
        print("  may also register stubs for machines that are not plugged in, so a name in")
        print("  the list above is not a connection either.")
        known = profile(pattern)
        if known.stems:
            print(f"  For per-drum audio on {known.name}: {known.stems}.")
        return 0
    for i, d, api in hits:
        print(f"  {d['name']} on {api}: {d['max_input_channels']} in, "
              f"{d['max_output_channels']} out")
        print("    sample rates the driver claims (NOT proof it is connected): ",
              end="", flush=True)
        ok = [sr for sr in RATES if _openable(sd, i, d["max_input_channels"], sr)]
        print(", ".join(str(s) for s in ok) if ok else "NONE (is the driver running?)")
        if d["max_input_channels"] < len(TRACKS):
            print(f"    NOTE: {d['max_input_channels']} channels against {len(TRACKS)} "
                  f"tracks, so this is not one channel per track. Tracks that share an")
            print("          analog voice (CH/OH, RS/CP, CY/CB) cannot be separated here;")
            print("          use --meter to find out what the channels really are.")
    return 0


def cmd_meter(sd, args, seconds: float) -> int:
    """Live per-channel levels. Hit one pad at a time and read which channel moves."""
    try:
        device, info, nch, rate = _pick(sd, args, args.samplerate)
    except NoAudioDevice as exc:
        print(str(exc))
        print(f"  {EXCLUSIVE}")
        return 1

    peak = np.zeros(nch, dtype=np.float64)
    rms_acc = np.zeros(nch, dtype=np.float64)
    blocks = 0
    overflows = 0

    def callback(indata, frames, t, status):
        nonlocal blocks, overflows
        if status:
            overflows += 1
        a = np.abs(indata)
        np.maximum(peak, a.max(axis=0), out=peak)
        rms_acc[:] += (indata.astype(np.float64) ** 2).mean(axis=0)
        blocks += 1

    watch = MidiWatch(args.midi_port, profile(args.midi_port, args.audio_name)) \
        if args.midi else None
    print(f"device {device}: {info['name']}   {nch} ch @ {rate} Hz, blocksize {args.blocksize}")
    if watch:
        print(watch.describe())
    print(f"\nHit ONE PAD AT A TIME and watch which channel moves. {seconds:.0f} s.\n")

    try:
        with input_stream(sd, device, nch, rate, args.blocksize, callback):
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                time.sleep(0.1)
                if watch:
                    watch.poll()
                live = np.sqrt(rms_acc / max(blocks, 1))
                sys.stdout.write("\r" + _bars(live) + "  ")
                sys.stdout.flush()
                rms_acc[:] = 0.0
                blocks = 0
    except Exception as exc:                                      # noqa: BLE001
        print(f"\n\nstream failed: {type(exc).__name__}: {exc}")
        if isinstance(exc, sd.PortAudioError):
            print(f"  {EXCLUSIVE}")
        return 1
    finally:
        if watch:
            watch.close()

    print("\n\npeak per channel over the whole run:")
    for c in range(nch):
        db = 20 * np.log10(peak[c]) if peak[c] > 1e-9 else -120.0
        flag = "  SIGNAL" if peak[c] > 1e-4 else "  -- silent --"
        print(f"  ch {c:2}  {db:7.1f} dBFS  {'#' * int(max(0, (db + 60) / 3))}{flag}")
    silent = [c for c in range(nch) if peak[c] <= 1e-4]
    print(f"\n{nch - len(silent)}/{nch} channels carried signal; silent: {silent}")

    quiet = [c for c in range(nch) if 1e-4 < peak[c] < FLOOR * HEADROOM]
    print()
    print(f"onset floor {FLOOR:g} ({20 * math.log10(FLOOR):.1f} dBFS); "
          f"a send wants to peak {HEADROOM:g}x above it")
    if quiet:
        for c in quiet:
            print(f"  ch {c:2}  peaks at {peak[c] / FLOOR:.1f}x the floor -- turn this "
                  f"track's send up, or its hits drop out before any others")
    else:
        print("  every channel carrying signal has room; nothing to change")
    if overflows:
        print(f"WARNING: {overflows} stream overflows -- raise the blocksize.")
    if watch:
        watch.report()
    return 0


def _openable(sd, device, channels, sr) -> bool:
    try:
        sd.check_input_settings(device=device, channels=channels, samplerate=sr)
        return True
    except Exception:                                             # noqa: BLE001
        return False


LEVELS = "_.-=+*#%@"


def _bars(rms) -> str:
    """One character per channel, from `_` (silent) to `@` (loud)."""
    out = []
    for v in rms:
        n = 0 if v <= 1e-6 else min(8, max(1, int(20 * (v ** 0.4))))
        out.append(LEVELS[n])
    return "".join(out)


def _audio(args):
    """The input stream and its onset detector, or `(None, None)` with no audio input.

    Audio is optional for `--listen` and `--drive`: without it they still report MIDI."""
    try:
        sd = require_sounddevice()
        device, info, channels, rate = _pick(sd, args, args.samplerate or 48000)
    except SystemExit as exc:                                     # no audio extra installed
        print(f"audio    : none -- {exc}")
        return None, None
    except Exception as exc:                                      # noqa: BLE001
        print(f"audio    : none ({type(exc).__name__}: {exc})")
        return None, None
    print(f"audio    : {device} {info['name']}, {channels} ch @ {rate} Hz")
    extractor = FeatureExtractor(channels, rate)

    def callback(indata, _frames, _time_info, _status):
        extractor.push(indata)

    return input_stream(sd, device, channels, rate, args.blocksize, callback), extractor


class Mark(NamedTuple):
    """A phase boundary: onsets so far, note-ons so far, and audio consumed so far."""

    hits: object
    notes: int
    heard: float


def _drive_output(port_match: str):
    """The MIDI output to send Start and clock on, or `SystemExit`."""
    import pygame.midi

    outs = find_ports(port_match).outputs
    print("MIDI out : " + (", ".join(f"{i} {n}" for i, n in outs) or "NONE"))
    if not outs:
        raise SystemExit("--drive needs a MIDI output to send Start and clock on.")
    if len(outs) > 1 and not port_match:
        raise SystemExit(f"{len(outs)} MIDI outputs and nothing says which drives the machine; "
                         f"pass --midi-port.")
    return pygame.midi.Output(outs[0][0])


def cmd_listen(args, seconds: float, drive: bool) -> int:
    """`--listen` and `--drive`: watch MIDI and audio together, optionally as clock master."""
    import pygame.midi

    watch = MidiWatch(args.midi_port, profile(args.midi_port, args.audio_name))
    if not watch.inputs:
        watch.close()
        raise SystemExit(f"no MIDI input matching {args.midi_port!r}"
                         + (f" ({watch.error})" if watch.error else "")
                         + ". Is the machine connected over USB MIDI, and is anything that "
                           "claims its port exclusively closed?")
    print(watch.describe())
    out = None
    try:
        out = _drive_output(args.midi_port) if drive else None
        stream, heard = _audio(args)
        marks: dict[str, Mark] = {}

        def mark(label: str) -> None:
            marks[label] = Mark(heard.hits.copy() if heard is not None else None,
                                watch.by_phase[watch.phase].get("note_on", 0),
                                heard.heard() if heard is not None else 0.0)

        def elapse(span: float, pulse: float | None = None) -> None:
            """Read for `span` seconds, sending a clock pulse every `pulse` if asked."""
            end = next_pulse = time.perf_counter()
            end += span
            while (now := time.perf_counter()) < end:
                if pulse is not None and now >= next_pulse:
                    out.write_short(CLOCK)
                    next_pulse += pulse
                watch.poll()
                time.sleep(YIELD_S)

        if stream is not None:
            stream.start()
        try:
            mark("start")
            elapse(4.0 if drive else seconds)
            mark("listened")
            if drive:
                print(f"\n>>> Start, then {args.bpm:g} BPM for {seconds:g}s")
                out.write_short(START)
                watch.phase = "driven"
                mark("drive0")
                elapse(seconds, pulse=MusicalClock.pulse_period(args.bpm))
                mark("drive1")
                print(">>> Stop")
                out.write_short(STOP)
                watch.phase = "after"
                elapse(3.0)
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
    finally:
        watch.close()
        if out is not None:
            out.close()
        pygame.midi.quit()

    watch.report()
    if drive:
        print("\n  Sent messages arrive back on the input port, so the transport counts above")
        print("  include this command's own Start, clock and Stop. Only note-ons are the")
        print("  machine's, because nothing here ever sends one.")
    return _verdict(watch, marks, drive)


def _verdict(watch: MidiWatch, marks: dict[str, Mark], drove: bool) -> int:
    had_audio = marks["start"].hits is not None
    if had_audio:
        windows = [("idle  ", "start", "listened")]
        if drove:
            windows.append(("driven", "drive0", "drive1"))
        print()
        for label, a, b in windows:
            hits = marks[b].hits - marks[a].hits
            struck = ", ".join(f"{TRACKS[i] if i < len(TRACKS) else f'ch{i + 1}'}:{int(n)}"
                               for i, n in enumerate(hits) if n)
            print(f"audio, {label}: {int(hits.sum())} onsets over "
                  f"{marks[b].heard - marks[a].heard:.1f}s   {struck or 'silence'}")

    print("\nVERDICT")
    machine = watch.machine
    if not drove:
        print("  Listening only. Run with --drive to start the sequencer and get an answer.")
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
        print(f"  On {machine.name}, {machine.says('notes')}, and run this again.")
        return 0
    print(f"  Sequencer trigs DO send MIDI notes: {notes} note-ons alongside {onsets} onsets.")
    print("  ganlive can be driven by MIDI alone -- every track, including those that")
    print("  share an audio channel.")
    return 0


def cmd_learn(sd, args, seconds: float) -> int:
    """`--learn`: hit every pad; each note-on names a drum, and the audio says where it is."""
    import pygame.midi

    order = [t.strip().upper() for t in args.order.split(",") if t.strip()]
    try:
        device, info, nch, rate = _pick(sd, args, args.samplerate or 48000)
    except NoAudioDevice as exc:
        raise SystemExit(str(exc)) from exc

    watch = MidiWatch(args.midi_port, profile(args.midi_port, args.audio_name))
    if not watch.inputs:
        raise SystemExit(f"no MIDI input matching {args.midi_port!r}"
                         + (f" ({watch.error})" if watch.error else "")
                         + ". --learn needs MIDI: the notes are what identify the pads, and "
                           "nothing else can.")
    machine = watch.machine
    strikes = Strikes(nch)
    ex = FeatureExtractor(nch, rate)
    onsets: list[tuple[float, int]] = []

    def callback(indata, _frames, _t, _status):
        strikes.on_audio(indata)
        ex.push(indata)

    def on_note(note: int, now: float) -> None:
        if strikes.on_note(note, now):
            print(f"  note {note:>3}  ({len(strikes.seen)} of {len(TRACKS)} pads seen)",
                  flush=True)

    stream = input_stream(sd, device, nch, rate, args.blocksize, callback)
    try:
        try:
            stream.start()
        except sd.PortAudioError as exc:
            print(f"could not start the input: {exc}")
            print(f"  {EXCLUSIVE}")
            if machine.stems:
                print(f"  On {machine.name}: {machine.stems}.")
            raise SystemExit(1) from exc
        print(f"device {device}: {info['name']}  {nch} ch @ {rate} Hz")
        print(watch.describe())
        print(f"Hit every pad several times, in ANY order, for {seconds:g} s.")
        print("More hits is a better answer. Ctrl-C stops early and still reports.")
        print()
        end = time.perf_counter() + seconds
        try:
            while time.perf_counter() < end:
                now = time.perf_counter()
                strikes.settle(now)
                for ch, _vel, ago in ex.drain():
                    onsets.append((now - ago, int(ch)))
                watch.poll(on_note)
                time.sleep(0.002)
        except KeyboardInterrupt:
            print("(stopped early)")
        stream.stop()
    finally:
        stream.close()
        watch.close()
        pygame.midi.quit()

    if not strikes.seen:
        carried = dict(sorted(watch.by_phase["listening"].items()))
        print()
        print(f"No note-ons arrived, so nothing can be mapped. The wire carried: "
              f"{carried or 'nothing at all'}")
        if carried:
            print(f"  The port is FINE -- those messages came down it. The machine is not "
                  f"sending what you play: on {machine.name}, {machine.says('notes')}. For "
                  f"its sequencer to send notes too, each track usually needs its own MIDI "
                  f"channel.")
        else:
            print("  Nothing at all arrived, not even clock, so this is the port or the "
                  "cable rather than a setting.")
        return 1

    mapping = report(strikes.seen, strikes.votes, strikes.levels, strikes.struck,
                     strikes.silent, order)
    report_recall(strikes.times, onsets, mapping, order)
    if mapping:
        text = ",".join(f"{n}={c}" for n, c in sorted(mapping.items()))
        trouble = remember(CHANNEL_MAP, text)
        print(f"\nNOT SAVED: {trouble}" if trouble else
              f"\nsaved to {CHANNEL_MAP}; `ganlive play` uses it when no --map is given.")
    return 0


def main(argv=None) -> int:
    ap = parser("doctor", __doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true",
                      help="enumerate devices and sample rates; the default with no other mode")
    mode.add_argument("--meter", action="store_true",
                      help="live per-channel levels; hit one pad at a time")
    mode.add_argument("--listen", action="store_true",
                      help="watch MIDI (notes, clock, transport) and audio arriving")
    mode.add_argument("--drive", action="store_true",
                      help="be the clock master: Start, pulse, Stop, and certify the "
                           "sequencer's notes against the audio")
    mode.add_argument("--learn", action="store_true",
                      help="map which audio channel each drum arrives on, from the pads' "
                           f"notes, and save it to {CHANNEL_MAP}")
    ap.add_argument("--midi", action="store_true",
                    help="with --meter: watch MIDI at the same time")
    ap.add_argument("--seconds", type=float, default=None,
                    help="how long to run: 20 by default, 75 for --learn")
    ap.add_argument("--midi-port", default="", metavar="TEXT",
                    help="substring of the MIDI port to use; the default takes every input. "
                         "--drive sends on the matching output")
    ap.add_argument("--audio-device", type=int, default=None,
                    help="the input's index from the list, on any host API")
    ap.add_argument("--audio-name", default=DEFAULT_MATCH, metavar="TEXT",
                    help="substring of the input to look for. The default suits an Elektron "
                         "Analog Rytm drum machine over Overbridge, Elektron's USB audio; "
                         "pass your own interface's name")
    ap.add_argument("--samplerate", type=int, default=None,
                    help="default: the first of 48000, 44100, 96000 that opens (48000 for "
                         "--listen, --drive and --learn)")
    ap.add_argument("--blocksize", type=int, default=256,
                    help="audio buffer in frames; 256 at 48 kHz is 5.3 ms")
    ap.add_argument("--bpm", type=float, default=166.0, help="the tempo --drive sends")
    ap.add_argument("--order", default=",".join(TRACKS),
                    help="with --learn: track names in note order; the default is the pad "
                         "order of the Elektron Analog Rytm drum machine")
    args = ap.parse_args(argv)

    which = next((m for m in ("meter", "listen", "drive", "learn") if getattr(args, m)), "list")
    seconds = args.seconds if args.seconds is not None else SECONDS.get(which, 0.0)
    if which in ("listen", "drive"):
        return cmd_listen(args, seconds, drive=which == "drive")
    sd = require_sounddevice()
    if which == "meter":
        return cmd_meter(sd, args, seconds)
    if which == "learn":
        return cmd_learn(sd, args, seconds)
    return cmd_list(sd, args.audio_name)
