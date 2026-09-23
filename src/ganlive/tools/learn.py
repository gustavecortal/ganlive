"""Discover which audio channel each drum arrives on, using MIDI as the ground truth."""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

from ganlive.control.audio import NoAudioDevice, pick_input
from ganlive.control.features import FeatureExtractor
from ganlive.control.kit import TRACKS, by_channel
from ganlive.control.machine import profile
from ganlive.control.midi import dispatch, open_inputs
from ganlive.walk import MusicalClock

WINDOW_S = 0.050

QUIET = 1e-4

AGREE = 0.8

RESPONDS = 0.25

DETECT_S = 0.075

BUS = 0.75


def open_midi(match: str = ""):
    """The first input port whose name contains `match`; any input port if it is empty."""
    opened, _rejected, error = open_inputs(match)
    if not opened:
        raise SystemExit(f"no MIDI input matching {match!r}{f' ({error})' if error else ''}. "
                         f"This tool needs MIDI: the notes are what identify the pads, and "
                         f"nothing else can. `ganlive doctor --midi` lists the ports.")
    return opened[0][1]


def report(seen, votes, levels, struck, silent, order):
    """Everything measured, turned into a map plus an honest account of what is shaky."""
    by_note = sorted(seen)
    named = {note: (order[note] if note < len(order) else f"note{note}") for note in by_note}

    played = [n for n in by_note if struck[n]]
    answers: dict[int, int] = defaultdict(int)
    for note in played:
        for ch in votes[note]:
            answers[ch] += 1
    buses = sorted(ch for ch, n in answers.items() if played and n >= BUS * len(played))

    print()
    print(f"{'track':>6} {'note':>5} {'channel':>8} {'hits':>5} {'agree':>6} {'peak':>8}"
          f"   also on")
    print("  " + "-" * 62)
    mapping: dict[str, int] = {}
    trouble: list[str] = []
    for note in by_note:
        name = named[note]
        counts = {ch: n for ch, n in votes.get(note, {}).items() if ch not in buses}
        hits = struck[note]
        rest = " ".join(str(c) for c in sorted(votes.get(note, {})) if c in buses)
        if not hits:
            trouble.append(f"{name}: {silent[note]} hits and no audio anywhere. Its separate "
                           f"send is off, or the track is muted.")
            print(f"{name:>6} {note:>5} {'--':>8} {0:>5} {'--':>6} {'--':>8}")
            continue
        loudest = max(levels[note])
        if not counts:
            trouble.append(f"{name}: {hits} hits, heard ONLY on the mix bus ({rest}). It has "
                           f"no separate send, so nothing can tell it from the other drums in "
                           f"audio -- only its MIDI note can.")
            print(f"{name:>6} {note:>5} {'bus':>8} {hits:>5} {'--':>6} {loudest:>8.4f}"
                  f"   {rest}")
            continue
        ch = max(counts, key=counts.get)
        agree = counts[ch] / hits
        mapping[name] = ch
        if agree < AGREE:
            trouble.append(f"{name}: only {agree:.0%} of {hits} hits reached channel {ch}, "
                           f"saw {counts}. Hit it more, or its send is too low.")
        print(f"{name:>6} {note:>5} {ch:>8} {hits:>5} {agree:>5.0%} {loudest:>8.4f}"
              f"   {rest}")

    print()
    if buses:
        print(f"mix bus (answers {BUS:.0%}+ of all drums, set aside): "
              + ", ".join(f"ch {c}" for c in buses))
    pairs = {ch: names for ch, names in by_channel(mapping).items() if len(names) > 1}
    print(f"{len(set(mapping.values()))} separate channels for {len(mapping)} of "
          f"{len(by_note)} tracks")
    if pairs:
        print("sharing a channel -- indistinguishable in audio, separable only by MIDI note:")
        for ch, names in sorted(pairs.items()):
            print(f"  ch {ch}: {'/'.join(sorted(names))}")
    if trouble:
        print()
        print("NEEDS ATTENTION:")
        for line in trouble:
            print(f"  {line}")
    if mapping:
        print()
        print("pass this to `ganlive play`:")
        print("  --map " + ",".join(f"{n}={c}" for n, c in sorted(mapping.items())))
    return mapping


def report_recall(strikes, onsets, mapping, order):
    """Per pad: how many strikes the ONSET DETECTOR caught, against the notes as truth."""
    print()
    print("what the ONSET DETECTOR caught, per pad -- this is what the picture actually sees")
    print(f"{'track':>6} {'note':>5} {'ch':>4} {'struck':>7} {'caught':>7} {'recall':>7}")
    print("  " + "-" * 46)
    spent = [False] * len(onsets)
    missing = []
    for note in sorted(strikes):
        name = order[note] if note < len(order) else f"note{note}"
        ch = mapping.get(name)
        struck = len(strikes[note])
        if ch is None:
            print(f"{name:>6} {note:>5} {'--':>4} {struck:>7} {'--':>7} {'--':>7}")
            continue
        caught = 0
        for t in strikes[note]:
            for i, (when, where) in enumerate(onsets):
                if not spent[i] and where == ch and -0.005 <= when - t <= DETECT_S:
                    spent[i] = True
                    caught += 1
                    break
        rate = caught / struck if struck else 0.0
        print(f"{name:>6} {note:>5} {ch:>4} {struck:>7} {caught:>7} {rate:>6.0%}")
        if struck >= 3 and rate < 0.8:
            missing.append((name, ch, struck, caught, rate))

    if missing:
        print()
        print("PADS THE PICTURE IS MISSING:")
        for name, ch, struck, caught, rate in missing:
            print(f"  {name} on ch {ch}: {caught} of {struck} strikes seen ({rate:.0%}). "
                  f"Any rule wired to it fires that fraction of the time.")
        print("  On a shared channel this is the pair effect, not a level: the quieter drum "
              "cannot clear onset_ratio against the louder one's tail.")
    unspent = sum(1 for x in spent if not x)
    print()
    print(f"{len(onsets)} detector onsets in all; {unspent} matched no pad "
          f"({unspent / max(1, len(onsets)):.0%}) -- sequencer trigs, or false positives.")


def main(argv=None) -> int:
    import pygame.midi
    try:
        import sounddevice as sd
    except ModuleNotFoundError:
        print("this needs an audio input: pip install 'ganlive[audio]'", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(prog="ganlive learn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--port", default="", metavar="TEXT",
                    help="substring of the MIDI port carrying the pad notes; the default takes the first input port")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--blocksize", type=int, default=256)
    ap.add_argument("--order", default=",".join(TRACKS),
                    help="track names in note order; the default is the Rytm's own")
    args = ap.parse_args(argv)

    order = [t.strip().upper() for t in args.order.split(",") if t.strip()]
    try:
        device, info, nch = pick_input(sd, args.device)
    except NoAudioDevice as exc:
        raise SystemExit(str(exc)) from exc

    pygame.midi.init()
    machine = profile(args.port)
    midi_in = open_midi(args.port)
    clock = MusicalClock(120.0)

    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    levels: dict[int, list[float]] = defaultdict(list)
    struck: dict[int, int] = defaultdict(int)
    silent: dict[int, int] = defaultdict(int)
    windows: list[list] = []

    ex = FeatureExtractor(nch, args.rate)
    onsets: list[tuple[float, int]] = []
    strikes: dict[int, list[float]] = defaultdict(list)

    def callback(indata, _frames, _t, _status):
        peak = np.abs(np.asarray(indata, dtype=np.float32)).max(axis=0)
        for w in windows:
            np.maximum(w[2], peak, out=w[2])
        ex.push(indata)

    stream = sd.InputStream(device=device, channels=nch, samplerate=args.rate,
                            blocksize=args.blocksize, dtype="float32", callback=callback)
    try:
        stream.start()
    except sd.PortAudioError as exc:
        midi_in.close()
        pygame.midi.quit()
        print(f"could not start the input: {exc}")
        print("  ASIO gives the device to ONE program at a time. Close the Overbridge "
              "Control Panel window (the Engine can stay), and any DAW holding the Rytm, "
              "then run this again.")
        raise SystemExit(1) from exc
    print(f"device {device}: {info['name']}  {nch} ch @ {args.rate} Hz")
    print(f"Hit every pad several times, in ANY order, for {args.seconds:g} s.")
    print("More hits is a better answer. Ctrl-C stops early and still reports.")
    print()

    seen: list[int] = []
    wire: dict[str, int] = defaultdict(int)
    end = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < end:
            now = time.perf_counter()
            for w in [w for w in windows if now - w[0] >= WINDOW_S]:
                windows.remove(w)
                peak = w[2]
                top = int(np.argmax(peak))
                if peak[top] <= QUIET:
                    silent[w[1]] += 1
                    continue
                struck[w[1]] += 1
                levels[w[1]].append(float(peak[top]))
                for ch in np.flatnonzero(peak >= RESPONDS * peak[top]):
                    votes[w[1]][int(ch)] += 1
            for ch, _vel, ago in ex.drain():
                onsets.append((now - ago, int(ch)))
            while midi_in.poll():
                for event, _ts in midi_in.read(16):
                    status, note, vel = event[0], event[1], event[2]
                    what = dispatch(clock, status, note, vel)
                    wire[what or f"status 0x{status:02X}"] += 1
                    if what == "note_on":
                        if note not in seen:
                            seen.append(note)
                            print(f"  note {note:>3}  ({len(seen)} of 12 pads seen)",
                                  flush=True)
                        struck_at = time.perf_counter()
                        strikes[note].append(struck_at)
                        windows.append([struck_at, note,
                                        np.zeros(nch, dtype=np.float32)])
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("(stopped early)")
    stream.stop()
    stream.close()
    midi_in.close()
    pygame.midi.quit()

    if not seen:
        print()
        print(f"No note-ons arrived, so nothing can be mapped. The wire carried: "
              f"{dict(sorted(wire.items())) or 'nothing at all'}")
        if wire:
            print(f"  The port is FINE -- those messages came down it. The machine is not "
                  f"sending what you play: on {machine.name}, {machine.says('notes')}. For "
                  f"its sequencer to send notes too, each track usually needs its own MIDI "
                  f"channel.")
        else:
            print("  Nothing at all arrived, not even clock, so this is the port or the "
                  "cable rather than a setting.")
        return 1

    mapping = report(seen, votes, levels, struck, silent, order)
    report_recall(strikes, onsets, mapping, order)
    return 0

