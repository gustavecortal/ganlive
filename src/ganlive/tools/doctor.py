"""What this machine's audio and MIDI actually offer. Run it with the controller on."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("SD_ENABLE_ASIO", "1")

RATES = (48000, 44100, 96000)

from ganlive.control.audio import HOSTAPI, NoAudioDevice, pick_input  # noqa: E402
from ganlive.control.features import FeatureConfig  # noqa: E402
from ganlive.control.midi import dispatch, open_inputs  # noqa: E402
from ganlive.control.tracks import TRACKS  # noqa: E402
from ganlive.walk import MusicalClock  # noqa: E402


def find_devices(sd, pattern="rytm"):
    """Every input-capable device whose name matches, with its host API."""
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        if pattern.lower() in d["name"].lower():
            out.append((i, d, sd.query_hostapis(d["hostapi"])["name"]))
    return out


def cmd_list(sd, pattern="rytm") -> int:
    apis = [ha["name"] for ha in sd.query_hostapis()]
    print(f"PortAudio : {sd.get_portaudio_version()[1]}")
    print(f"host APIs : {', '.join(apis)}")
    if HOSTAPI not in apis:
        print(f"\n  {HOSTAPI} is missing, and it is this platform's multi-channel API.")
        if HOSTAPI == "ASIO":
            print("  sounddevice ships an ASIO build behind SD_ENABLE_ASIO; if it is absent")
            print("  here the wrong DLL was loaded.")
        print("  An interface on another API still works; pass --device with its index.")

    print("\ninput-capable devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        api = sd.query_hostapis(d["hostapi"])["name"]
        mark = "  <--" if pattern.lower() in d["name"].lower() else ""
        print(f"  {i:3} [{api:12}] in={d['max_input_channels']:3} "
              f"out={d['max_output_channels']:3} sr={d['default_samplerate']:6.0f}  "
              f"{d['name']}{mark}")

    hits = find_devices(sd, pattern)
    print()
    if not hits:
        print(f"  Nothing named {pattern!r}. The instrument plays without any audio input --")
        print("  MIDI notes alone drive it -- so this is a finding, not a failure. A driver")
        print("  may also register stubs for machines that are not plugged in, so a name in")
        print("  the list above is not a connection either.")
        return 0
    for i, d, api in hits:
        print(f"  {d['name']} on {api}: {d['max_input_channels']} in, "
              f"{d['max_output_channels']} out")
        print("    sample rates the driver claims (NOT proof it is connected): ",
              end="", flush=True)
        ok = []
        for sr in RATES:
            try:
                sd.check_input_settings(device=i, channels=d["max_input_channels"],
                                        samplerate=sr)
                ok.append(sr)
            except Exception:                                     # noqa: BLE001
                pass
        print(", ".join(str(s) for s in ok) if ok else "NONE (is the driver running?)")
        if d["max_input_channels"] < len(TRACKS):
            print(f"    NOTE: {d['max_input_channels']} channels against {len(TRACKS)} "
                  f"tracks, so this is not one channel per track. Tracks that share an")
            print("          analog voice (CH/OH, RS/CP, CY/CB) cannot be separated here;")
            print("          use --meter to find out what the channels really are.")
    return 0


FLOOR = FeatureConfig().floor
HEADROOM = 8.0

def cmd_meter(sd, seconds: float, device: int | None, rate: int | None,
              blocksize: int, with_midi: bool, pattern: str = "rytm") -> int:
    """Live per-channel levels. Hit one pad at a time and read which channel moves."""
    import numpy as np

    try:
        device, info, nch = pick_input(sd, device, pattern)
    except NoAudioDevice as exc:
        print(str(exc))
        return 1

    if rate is None:
        rate = next((sr for sr in RATES
                     if _openable(sd, device, nch, sr)), None)
        if rate is None:
            print("Could not open the device at any sample rate.")
            print("  An exclusive API hands the device to one program at a time: a DAW or")
            print("  a control panel holding it will lock this out. Close them and retry.")
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

    listener = _MidiListener() if with_midi else None
    print(f"device {device}: {info['name']}   {nch} ch @ {rate} Hz, blocksize {blocksize}")
    if listener:
        print(listener.describe())
    print(f"\nHit ONE PAD AT A TIME and watch which channel moves. {seconds:.0f} s.\n")

    try:
        with sd.InputStream(device=device, channels=nch, samplerate=rate,
                            blocksize=blocksize, dtype="float32", callback=callback):
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                time.sleep(0.1)
                if listener:
                    listener.poll()
                live = np.sqrt(rms_acc / max(blocks, 1))
                sys.stdout.write("\r" + _bars(live) + "  ")
                sys.stdout.flush()
                rms_acc[:] = 0.0
                blocks = 0
    except Exception as exc:                                      # noqa: BLE001
        print(f"\n\nstream failed: {type(exc).__name__}: {exc}")
        if isinstance(exc, sd.PortAudioError):
            print("  The device may be held exclusively -- close any DAW or control panel "
                  "using it.")
        return 1

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
    if listener:
        print()
        listener.report()
    return 0


def _openable(sd, device, channels, sr) -> bool:
    try:
        sd.check_input_settings(device=device, channels=channels, samplerate=sr)
        return True
    except Exception:                                             # noqa: BLE001
        return False


LEVELS = "_.-=+*#%@"


def _bars(rms) -> str:
    out = []
    for v in rms:
        n = 0 if v <= 1e-6 else min(8, max(1, int(20 * (v ** 0.4))))
        out.append(LEVELS[n])
    return "".join(out)


class _MidiListener:
    """Whether MIDI survives an audio driver holding the interface, and what it carries."""

    def __init__(self) -> None:
        self.inputs, _rejected, self.error = open_inputs()
        self.clock = 0
        self.transport: list[str] = []
        self.notes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.controls: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.first_clock: float | None = None
        self.last_clock: float | None = None
        self.clock_state = MusicalClock()

    def describe(self) -> str:
        if not self.inputs:
            return ("MIDI: NO INPUT PORTS. If a machine is in a mode that claims its USB "
                    "for audio, that is the finding -- the clock needs another source.")
        return "MIDI inputs: " + ", ".join(name for name, _port in self.inputs)

    def poll(self) -> None:
        """Every message goes through `midi.dispatch`, the same call the live tool makes."""
        now = time.perf_counter()
        for _name, port in self.inputs:
            while port.poll():
                for event, _ts in port.read(64):
                    status = event[0]
                    what = dispatch(self.clock_state, status, event[1], event[2], now)
                    if what == "clock":
                        self.clock += 1
                        self.first_clock = self.first_clock or now
                        self.last_clock = now
                    elif what in ("start", "continue", "stop"):
                        self.transport.append(what.upper())
                    elif what == "song_position":
                        self.transport.append(f"SPP={event[1] | (event[2] << 7)}")
                    elif what == "control_change":
                        self.controls[status & 0x0F][event[1]] += 1
                    elif 0x90 <= status <= 0x9F and event[2] > 0:
                        self.notes[status & 0x0F][event[1]] += 1

    def report(self) -> None:
        if not self.inputs:
            print("MIDI: no input ports were present at all.")
            return
        print(f"MIDI clock pulses : {self.clock}")
        if self.clock > 24 and self.first_clock and self.last_clock > self.first_clock:
            span = self.last_clock - self.first_clock
            bpm = (self.clock - 1) / 24 / span * 60
            print(f"  implied tempo   : {bpm:.2f} BPM over {span:.1f} s")
        elif self.clock == 0:
            print("  NO CLOCK. Set MIDI CONFIG > SYNC > CLOCK SEND = ON, and check that")
            print("  PORT CONFIG > OUTPUT sends to USB. Without it the video runs at its")
            print("  own internal tempo and will look entirely plausible while doing so.")
        print(f"transport events  : {self.transport or 'NONE -- set TRANSPORT SEND = ON'}")
        if self.notes:
            print("note-ons per MIDI channel (channel -> note: count):")
            for ch in sorted(self.notes):
                items = ", ".join(f"{n}:{c}" for n, c in sorted(self.notes[ch].items()))
                label = TRACKS[ch] if ch < len(TRACKS) else "?"
                print(f"  ch {ch + 1:2} ({label:2}): {items}")
        else:
            print("note-ons          : NONE. Per-track identity is what MIDI is for here, so")
            print("                    check MIDI CONFIG > PORT CONFIG > OUT PORT FUNC.")
        if self.controls:
            print("control changes (turn one knob at a time; these are what --cc takes):")
            for ch in sorted(self.controls):
                for cc, n in sorted(self.controls[ch].items()):
                    print(f"  ch {ch + 1:2} cc {cc:3}: {n:5} messages   "
                          f"--cc \"{ch + 1}:{cc}=<dial>\"")
        else:
            print("control changes   : NONE. Without them the knobs cannot hold a dial; the")
            print("                    sliders and the drums still can.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ganlive doctor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="enumerate devices and sample rates; the default with no other mode")
    ap.add_argument("--meter", type=float, metavar="SECONDS", default=0.0,
                    help="live per-channel levels for this long; hit one pad at a time")
    ap.add_argument("--midi", action="store_true",
                    help="listen for clock, transport and note-ons at the same time")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--name", default="rytm", metavar="TEXT",
                    help="substring of the input to look for. The default suits an Analog "
                         "Rytm over Overbridge; pass your own interface's name")
    ap.add_argument("--rate", type=int, default=None)
    ap.add_argument("--blocksize", type=int, default=256,
                    help="audio buffer in frames; 256 at 48 kHz is 5.3 ms")
    args = ap.parse_args(argv)

    try:
        import sounddevice as sd
    except ImportError:
        print("this needs an audio input: pip install 'ganlive[audio]'")
        return 1

    if args.meter:
        return cmd_meter(sd, args.meter, args.device, args.rate, args.blocksize,
                         args.midi, args.name)
    return cmd_list(sd, args.name)

