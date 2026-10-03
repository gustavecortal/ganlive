"""The twelve-track drum vocabulary, and the three ways a machine is wired to it.

Track names are the usual abbreviations -- bass drum, snare, rim shot, clap, four toms, two
hats, cymbal, cowbell -- in the pad order of the Elektron Analog Rytm drum machine, which is
the default kit. Any other machine is wired to it by `parse_notes` (which note is which pad),
`parse_track_channels` (which MIDI channel is which track) or `parse_channel_map` (which audio
channel is which drum). All three validate names through `track_index`.

Nothing here makes a sound; the stand-in machine that does is `control/simulate.py`.
"""
from __future__ import annotations

TRACKS = ("BD", "SD", "RS", "CP", "BT", "LT", "MT", "HT", "CH", "OH", "CY", "CB")


INDEX = {t: i for i, t in enumerate(TRACKS)}

#: Tracks that share one analog voice on the default kit, so one hit chokes the other and
#: they arrive on one audio channel in the `voices` layout.
VOICE_GROUPS = (("BD",), ("SD",), ("RS", "CP"), ("BT",), ("LT",), ("MT", "HT"),
                ("CH", "OH"), ("CY", "CB"))


def channel_map(layout: str = "tracks") -> dict[str, int]:
    """Which audio channel each drum arrives on: one per track (`tracks`), or one per shared
    voice (`voices`)."""
    if layout == "tracks":
        return dict(INDEX)
    if layout == "voices":
        return {track: i for i, group in enumerate(VOICE_GROUPS) for track in group}
    raise ValueError(f"unknown layout {layout!r}; have tracks, voices")


def by_channel(channel_of: dict[str, int]) -> dict[int, list[str]]:
    """`{track: channel}` turned round: which tracks share each channel, in the map's order."""
    out: dict[int, list[str]] = {}
    for track, channel in channel_of.items():
        out.setdefault(channel, []).append(track)
    return out


def track_index(name: str) -> int:
    """A track to its index, or raise. Accepts its name, or the index itself (`0` to `11`)
    for a machine whose pads are not called BD and SD."""
    name = name.strip().upper()
    if name.isdigit():
        index = int(name)
        if not 0 <= index < len(TRACKS):
            raise ValueError(f"track {name} is out of range; have 0 to {len(TRACKS) - 1}")
        return index
    if name not in INDEX:
        raise ValueError(f"unknown track {name!r}; have {', '.join(TRACKS)}, or 0 to "
                         f"{len(TRACKS) - 1}")
    return INDEX[name]


def pairs(text: str):
    """`"a=b,c=d"` as `(a, b), (c, d)`: the one comma-list grammar every flag here shares.
    Empty entries are skipped; each side is left for its own parser to strip and validate."""
    for part in text.split(","):
        part = part.strip()
        if part:
            left, _, right = part.partition("=")
            yield left, right


def parse_channel_map(text: str) -> dict[str, int]:
    """"BD=0,CH=4" to `{track: audio channel}`. Unknown track names are refused."""
    out = {}
    for name, channel in pairs(text):
        track_index(name)               # validated, but this map is by NAME
        out[name.strip().upper()] = int(channel)
    return out


def parse_notes(text: str) -> dict[int, int]:
    """Which note is which track, for a kit that shares one channel: `{note: track index}`.

    `"0"` -- a first note, for a kit whose pads send consecutive notes from there (the default
    kit sends 0 to 11). Or `"36=BD,38=SD,42=CH,46=OH"` for one that does not, such as a
    General MIDI kit. Same grammar as `parse_track_channels`, with a note where the channel
    was."""
    text = text.strip()
    if text.isdigit():
        return {int(text) + i: i for i in range(len(TRACKS))}
    out: dict[int, int] = {}
    for note, name in pairs(text):
        number = int(note)
        if number in out:
            raise ValueError(f"note {number} is already {TRACKS[out[number]]}; two tracks on "
                             f"one note cannot be told apart")
        out[number] = track_index(name)
    if not out:
        raise ValueError("--notes needs a first note, as '0', or a map, as '36=BD,38=SD'")
    return out


def output_mode(seen: dict[int, set[int] | list[int] | dict[int, int]],
                known=range(len(TRACKS))) -> str | None:
    """How a machine sends its drums, read from `{MIDI channel: notes seen}`.

    `"auto"` when the notes name tracks (the whole kit on one channel), `"track"` when several
    channels carry other notes (one channel per track), `"mixed"` for both at once, or None if
    the traffic cannot tell. `known` is the notes that name a track on this kit, 0 to 11
    unless told."""
    known = set(known)
    low = {ch for ch, notes in seen.items() if any(n in known for n in notes)}
    high = {ch for ch, notes in seen.items() if any(n not in known for n in notes)}
    high -= low                                   # a channel is classified by its lowest note
    if low and len(high) > 1:
        return "mixed"
    if low:
        return "auto"
    return "track" if len(high) > 1 else None


def parse_track_channels(text: str) -> dict[int, int]:
    """`"1-12"` or `"1=BD,2=SD"` to `{MIDI channel: track index}`."""
    out: dict[int, int] = {}
    text = text.strip()
    if "=" not in text:
        low, _, high = text.partition("-")
        first, span = int(low), int(high) - int(low) + 1
        if span != len(TRACKS):
            raise ValueError(f"{text!r} is {span} channels for {len(TRACKS)} tracks; "
                             f"the whole kit or an explicit map")
        return {first - 1 + i: i for i in range(len(TRACKS))}
    for channel, name in pairs(text):
        index = track_index(name)
        number = int(channel) - 1
        if number in out:
            raise ValueError(f"MIDI channel {number + 1} is already {TRACKS[out[number]]}; two "
                             f"tracks on one channel cannot be told apart")
        out[number] = index
    return out
