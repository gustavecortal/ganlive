"""What a controller has to be set to, in whatever that controller calls it.

Every tool here reports a silence sooner or later -- no clock, no notes, no knobs -- and the
useful half of that report is *what to go and switch on*. That is the one thing which really
is per-machine, so it lives here as data rather than as sentences spread through four tools
that named one product's menus at everyone who ran them.

Unknown hardware gets `GENERIC`, which says what kind of setting to look for without
pretending to know the menu it is under. That is honest, and it is what most people see.
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
    #: How per-voice audio reaches the host, for kit whose drums can be heard separately.
    #: `None` where that is not a thing it does, which is most of them -- and `doctor` then
    #: says nothing about it rather than guessing that a controller has stems at all.
    stems: str | None = None

    def says(self, what: str) -> str:
        """`check <this machine's words for it>`, ready to drop into a sentence."""
        return f"check {getattr(self, what)}"


GENERIC = Machine(name="your controller")

#: The machine this was built against. Its menu paths are exact; nothing else assumes them.
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

    Matching on the words the user already typed -- `--port`, `--audio-name` -- rather than
    on a flag of its own, so nobody has to declare their hardware to get advice about it."""
    hay = f"{port_match} {name}".lower()
    return next((m for m in KNOWN if m.port and m.port in hay), GENERIC)
