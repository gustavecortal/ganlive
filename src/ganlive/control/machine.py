"""Per-machine advice: which setting to switch on when clock, notes or knobs are silent.

The tools report silences in a machine's own words, taken from a `Machine` here. Hardware
nobody named gets `GENERIC`, which says what kind of setting to look for without naming a
menu it may not have.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Machine:
    """One controller's own words for the things that have to be on."""

    name: str
    #: Substring that finds its MIDI port, and its audio device where it has one. Empty means
    #: "no idea, take the first port", which is the right default for hardware nobody named.
    port: str = ""
    clock: str = "its clock-send setting"
    transport: str = "its transport-send setting"
    notes: str = "its pad/note output setting"
    encoders: str = "its knob/CC output setting"
    #: How per-voice audio reaches the host, for a machine whose drums can be heard
    #: separately. `None` for most controllers, and then the tools say nothing about it.
    stems: str | None = None

    def says(self, what: str) -> str:
        """`check <this machine's words for it>`, ready to drop into a sentence."""
        return f"check {getattr(self, what)}"


GENERIC = Machine(name="your controller")

#: The Elektron Analog Rytm drum machine, which this was built against: its menu paths are
#: exact. Overbridge is Elektron's USB audio, which carries each voice on its own channel.
RYTM = Machine(
    name="Analog Rytm",
    port="rytm",
    clock="MIDI CONFIG > SYNC > CLOCK SEND = ON",
    transport="MIDI CONFIG > SYNC > TRANSPORT SEND = ON",
    notes="MIDI CONFIG > PORT CONFIG > OUT PORT FUNC, and TRK SEND MIDI on each track "
          "(a setting, not a hardware limit -- OS 1.50 added it to the MKI)",
    encoders="MIDI CONFIG > PORT CONFIG > ENCODER DEST = INT+EXT or EXT",
    stems="its stems over Overbridge, which exposes them on ASIO only and hands the device "
          "to one program at a time",
)

KNOWN = (RYTM,)


def profile(port_match: str = "", name: str = "") -> Machine:
    """The machine a port filter or a device name points at, or `GENERIC`.

    Matched on what the user already typed -- `--midi-port`, `--audio-name` -- so nobody has
    to declare their hardware to get advice about it."""
    hay = f"{port_match} {name}".lower()
    return next((m for m in KNOWN if m.port and m.port in hay), GENERIC)
