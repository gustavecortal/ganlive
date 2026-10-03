"""`ganlive doctor --learn`: which audio channel each drum arrives on, from MIDI as truth.

`Strikes` collects, for every note-on, the audio peaks that follow it; `report` turns those
into a channel map and `report_recall` scores the onset detector against the notes.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ganlive.control.kit import by_channel

#: How long after a note-on the audio peaks are collected for it.
WINDOW_S = 0.050
#: A window whose loudest channel stays below this heard nothing.
QUIET = 1e-4
#: The share of a pad's hits that must land on one channel for the answer to be trusted.
AGREE = 0.8
#: A channel answers a hit if it peaks at this share of the loudest channel or more.
RESPONDS = 0.25
#: How long after a strike an onset may come and still be that strike.
DETECT_S = 0.075
#: A channel answering this share of all pads is a mix bus, not one drum's own send.
BUS = 0.75


class Strikes:
    """Every pad struck, and which audio channels answered it.

    `on_note` opens a window per note-on; `on_audio` (the audio thread) raises each open
    window's per-channel peak; `settle` closes the windows that are old enough and votes."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.seen: list[int] = []
        self.votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.levels: dict[int, list[float]] = defaultdict(list)
        self.struck: dict[int, int] = defaultdict(int)
        self.silent: dict[int, int] = defaultdict(int)
        self.times: dict[int, list[float]] = defaultdict(list)
        #: Open windows: `[when, note, peak per channel so far]`.
        self.windows: list[list] = []

    def on_audio(self, block) -> None:
        peak = np.abs(np.asarray(block, dtype=np.float32)).max(axis=0)
        for w in list(self.windows):
            np.maximum(w[2], peak, out=w[2])

    def on_note(self, note: int, now: float) -> bool:
        """Open a window for one strike. True if this is the first time `note` was seen."""
        first = note not in self.seen
        if first:
            self.seen.append(note)
        self.times[note].append(now)
        self.windows.append([now, note, np.zeros(self.channels, dtype=np.float32)])
        return first

    def settle(self, now: float) -> None:
        """Close every window older than `WINDOW_S`; each votes for the channels that answered."""
        for w in [w for w in self.windows if now - w[0] >= WINDOW_S]:
            self.windows.remove(w)
            _when, note, peak = w
            top = int(np.argmax(peak))
            if peak[top] <= QUIET:
                self.silent[note] += 1
                continue
            self.struck[note] += 1
            self.levels[note].append(float(peak[top]))
            for ch in np.flatnonzero(peak >= RESPONDS * peak[top]):
                self.votes[note][int(ch)] += 1


def report(seen, votes, levels, struck, silent, order):
    """Print the measured map, and every pad whose answer is uncertain. Returns the map."""
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
        print("the map, as `ganlive play --map` takes it:")
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
