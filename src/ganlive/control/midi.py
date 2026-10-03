"""Reading a machine's clock, transport and controls, and driving a `MusicalClock` from them."""
from __future__ import annotations

import threading
import time
from typing import NamedTuple

from ganlive.control.kit import pairs
from ganlive.dials.table import clamp01
from ganlive.files import remember
from ganlive.walk import MusicalClock


class Ports(NamedTuple):
    """What this machine offers MIDI-wise, as `(index, name)` pairs."""

    inputs: list
    outputs: list
    #: Ports whose name did not contain the filter. What to print when nothing matched.
    rejected: list


def find_ports(match: str = "") -> Ports:
    """Every MIDI port whose name contains `match`, case-insensitively; all of them if empty.

    The one enumeration. `play`'s clock reader, `wire`, `learn` and `doctor` each had their
    own, and two of them compared the name as bytes while two compared it as text."""
    import pygame.midi

    pygame.midi.init()
    want = match.lower()
    found = Ports([], [], [])
    for i in range(pygame.midi.get_count()):
        _interf, raw, is_input, _is_output, _open = pygame.midi.get_device_info(i)
        label = raw.decode(errors="replace")
        if want and want not in label.lower():
            found.rejected.append(label)
            continue
        (found.inputs if is_input else found.outputs).append((i, label))
    return found


def open_inputs(match: str = "") -> tuple[list, list, str | None]:
    """`(opened, rejected, error)` for every matching input, as `(name, Input)` pairs.

    `error` is why there is no MIDI at all, or the last port that refused to open -- never a
    raise, because three of the four callers are reporting on the machine rather than using it."""
    try:
        import pygame.midi

        found = find_ports(match)
    except ImportError as exc:
        return [], [], f"no MIDI library: {exc}"
    out, error = [], None
    for i, label in found.inputs:
        try:
            out.append((label, pygame.midi.Input(i)))
        except Exception as exc:                                  # noqa: BLE001
            error = f"{label}: {exc}"
    return out, found.rejected, error


CLOCK = 0xF8
START = 0xFA
CONTINUE = 0xFB
STOP = 0xFC
SONG_POSITION = 0xF2
CONTROL_CHANGE = 0xB0
NOTE_ON = 0x90
AFTERTOUCH_POLY = 0xA0


def dispatch(clock: MusicalClock, status: int, data1: int = 0, data2: int = 0,
             now: float | None = None) -> str | None:
    """Apply one MIDI message to a clock. Returns what it was, or None if it was ignored."""
    if status == CLOCK:
        clock.on_pulse(now)
        return "clock"
    if status == START:
        clock.on_start()
        return "start"
    if status == CONTINUE:
        clock.on_continue()
        return "continue"
    if status == STOP:
        clock.on_stop()
        return "stop"
    if status == SONG_POSITION:
        clock.on_song_position(data1 | (data2 << 7))
        return "song_position"
    if CONTROL_CHANGE <= status <= CONTROL_CHANGE + 0x0F:
        return "control_change"
    if NOTE_ON <= status <= NOTE_ON + 0x0F:
        return "note_on" if data2 > 0 else "note_off"
    if AFTERTOUCH_POLY <= status <= AFTERTOUCH_POLY + 0x0F:
        return "aftertouch_poly"
    return None


def _dial_or_raise(name: str) -> str:
    """A dial name, checked for being one at all. **Not checked against a list of them.**

    Which dials exist is a property of the loaded model, and this runs before any model is
    open. Checking against a fixed table refused every foreign model's whole MODEL block --
    and silently undid learned bindings, which are written in this flag's own words and read
    back through here.

    A name no loaded model turns out to have is reported by `EncoderMap.unreachable`, beside
    the preset rules the same model cannot run."""
    name = name.strip()
    if not name:
        raise ValueError("a mapping needs a dial after the '=', as in '16=noise', "
                         "'2:17=w_fine' or 'n1.3=dir1'")
    return name


#: NRPN parameters share the number space with CCs, above them: CC is 0..127, NRPN `msb.lsb`
#: is `NRPN_BASE + (msb << 7 | lsb)`. One key type, so a map holds both without a second dict.
NRPN_BASE = 128
NRPN_MSB, NRPN_LSB, DATA_MSB, DATA_LSB = 99, 98, 6, 38
NRPN_CCS = frozenset((NRPN_MSB, NRPN_LSB, DATA_MSB, DATA_LSB))


def nrpn_number(msb: int, lsb: int) -> int:
    return NRPN_BASE + ((msb & 0x7F) << 7 | (lsb & 0x7F))


def number_name(number: int) -> str:
    """`17` for a CC, `n1.3` for an NRPN -- the spelling `parse_controls` reads back."""
    if number < NRPN_BASE:
        return str(number)
    param = number - NRPN_BASE
    return f"n{param >> 7}.{param & 0x7F}"


def parse_number(text: str) -> int:
    text = text.strip().lower()
    if text.startswith("n"):
        msb, _, lsb = text[1:].partition(".")
        return nrpn_number(int(msb), int(lsb or 0))
    return int(text)


def parse_controls(text: str) -> dict[tuple[int, int], str]:
    """`"16=noise,2:17=se_256,n1.3=dir1"` to `{(channel, number): dial}`. `-1` is any channel."""
    out: dict[tuple[int, int], str] = {}
    for where, dial in pairs(text):
        channel, _, number = where.strip().rpartition(":")
        out[(int(channel) - 1 if channel else -1, parse_number(number))] = _dial_or_raise(dial)
    return out


def control_name(channel: int, number: int) -> str:
    """One control's address in `--cc`'s own words: `17`, `2:17`, `n1.3`. `-1` is any channel."""
    return (f"{channel + 1}:" if channel >= 0 else "") + number_name(number)


def format_controls(controls: dict[tuple[int, int], str]) -> str:
    """The inverse of `parse_controls`, so a learned map is written in the flag's own words."""
    return ",".join(f"{control_name(ch, n)}={dial}"
                    for (ch, n), dial in sorted(controls.items()))


class Nrpn:
    """The four-CC state machine, per channel: 99 and 98 name a parameter, 6 and 38 carry it.

    14-bit against a CC's 7, so a slow latent walk stops stair-stepping. Emits on 6 with the
    low byte at zero and again on 38 with it filled, since a device may send either alone."""

    def __init__(self) -> None:
        self.address: dict[int, list[int]] = {}       # channel -> [msb, lsb, data msb]

    def feed(self, channel: int, cc: int, value: int) -> tuple[int, int, int] | None:
        """`(number, value, top)` for a data byte -- 14-bit under a named parameter, else the
        plain 7-bit CC it is -- and None for the address bytes."""
        if cc == NRPN_MSB:
            self.address[channel] = [value, 0, 0]
            return None
        state = self.address.get(channel)
        if state is None:
            return (cc, value, 127) if cc in (DATA_MSB, DATA_LSB) else None
        if cc == NRPN_LSB:
            state[1] = value
            return None
        number = nrpn_number(state[0], state[1])
        if cc == DATA_MSB:
            state[2] = value
            return number, value << 7, 16383
        if cc == DATA_LSB:
            return number, state[2] << 7 | value, 16383
        return None


class EncoderMap:
    """Knobs on the machine holding dials, through the seam a hand on the strip uses."""

    SOURCE = "encoder"
    PRIORITY = 0
    FLAG, THING = "--cc", "knob"
    #: Which of `Machine`'s settings switches this kind of control on.
    SETTING = "encoders"

    @property
    def SILENCE(self) -> str:                                               # noqa: N802
        return (f"{self.FLAG} wired {len(self.controls)} {self.THING}(s) and not one reported. "
                f"On {self.machine.name}, {self.machine.says(self.SETTING)}.")

    def __init__(self, controls: dict[tuple[int, int], str], source: str = "",
                 remember=None, machine=None) -> None:
        from ganlive.control.machine import GENERIC

        self.machine = machine or GENERIC
        self.controls = dict(controls)
        self.source = source or self.SOURCE
        self.held: dict[str, float] = {}
        self.unmapped: dict[tuple[int, int], int] = {}
        self.seen = 0
        #: The dial the next control to arrive is bound to, or None. Set from the window's
        #: thread, read on the MIDI thread; one name, so no lock.
        self.learning: str | None = None
        self.learned: list[str] = []
        #: Bumped by every learn, so a poller can tell one from the last without counting.
        self.version = 0
        self.remember = remember
        self.trouble = ""
        self._unsaved = False

    def dial_for(self, channel: int, number: int) -> str | None:
        """What one control moves, or None. An any-channel entry is the fallback."""
        return self.controls.get((channel, number)) or self.controls.get((-1, number))

    def where(self, dial: str) -> str:
        """How the machine addresses this dial, in `--cc`'s own words, or "" if it does not.

        **The strip has a routing grid for the drums and had nothing at all for the knobs.** A
        map with eight bindings in it was invisible the moment each learn ended, and the only
        way to find out which knob moved a dial was to turn all of them and watch."""
        return " ".join(control_name(ch, n) for (ch, n), held in sorted(self.controls.items())
                        if held == dial)

    def unreachable(self, layout) -> str:
        """The controls wired to a dial this model does not have, as one line, or "".

        **Not an error, and not silence either.** A bank holds several families, and a knob
        parked on the outgoing model's dial is right again the moment that model comes back --
        so this is said on every switch rather than refused at the start. It has to be said,
        because the other reading of a knob that moves nothing is a knob whose messages are not
        arriving at all, and telling those two apart is what the rest of this class is for."""
        strays = sorted({dial for dial in self.controls.values() if dial not in layout})
        if not strays:
            return ""
        return (f"{self.source}: {', '.join(strays)} not on this model, so "
                f"{len(strays)} {self.THING}(s) move nothing until one that has them plays")

    def _bind(self, channel: int, number: int, dial: str) -> None:
        """One control now moves `dial`, and nothing else does. Written down by `flush`, off
        this thread: the clock is what the MIDI thread is for, not the disk."""
        self.controls = {k: d for k, d in self.controls.items() if d != dial}
        self.controls[(channel, number)] = dial
        self.learning = None
        self.learned.append(format_controls({(channel, number): dial}))
        self.version += 1
        self._unsaved = self.remember is not None

    def flush(self) -> None:
        """Write the map down if a learn changed it. Call from any thread but the reader's."""
        if not self._unsaved:
            return
        self._unsaved = False
        self.trouble = remember(self.remember, format_controls(self.controls))

    def report(self) -> list[str]:
        """The end-of-run lines about these controls: silence, strays, and what was learned."""
        out = []
        if self.controls and not self.seen:
            out.append(f"NOTHING ARRIVED. {self.SILENCE}")
        if self.unmapped:
            out.append(f"unwired {self.source} seen: "
                       + ", ".join(f"ch{c + 1} {number_name(n)} x{k}"
                                   for (c, n), k in sorted(self.unmapped.items())))
        if self.learned:
            out.append(f"learned    {', '.join(self.learned)}"
                       + (f"  -- NOT SAVED: {self.trouble}" if self.trouble
                          else f"  -- in {self.remember}"))
        return out

    def apply(self, runner, channel: int, number: int, value: int,
              top: int = 127) -> str | None:
        """One control change. Returns the dial it moved, or None."""
        if self.learning is not None:
            self._bind(channel, number, self.learning)
        dial = self.dial_for(channel, number)
        if dial is None:
            key = (channel, number)
            self.unmapped[key] = self.unmapped.get(key, 0) + 1
            return None
        self.held[dial] = clamp01(value / top)
        self.seen += 1
        runner.hold(self.source, self.held, self.PRIORITY)      # `hold` copies
        return dial

    def release(self, runner, dial: str | None = None) -> None:
        """Let a knob go, so the preset and the drums have that dial back."""
        if dial is None:
            self.held.clear()
        else:
            self.held.pop(dial, None)
        runner.free(self.source, dial)


def parse_pressure(text: str, notes: dict[int, int] | None = None) -> dict[tuple[int, int], str]:
    """`"BD=noise,SD=dir1"` to `{(channel, pad note): dial}`, for `PressureMap`.

    The pad is named as a track -- `BD`, or its index -- and `notes` (`kit.parse_notes`)
    says which note that track's pad presses on; a Rytm's press on 0 to 11."""
    from ganlive.control.kit import track_index

    note_of = {} if notes is None else {track: note for note, track in notes.items()}
    out: dict[tuple[int, int], str] = {}
    for name, dial in pairs(text):
        track = track_index(name)
        out[(-1, note_of.get(track, track))] = _dial_or_raise(dial)
    return out


class PressureMap(EncoderMap):
    """A pad leaned on holding a dial. `EncoderMap`'s mechanism at a different rank."""

    SOURCE = "pressure"
    PRIORITY = 5
    FLAG, THING, SETTING = "--pressure", "pad", "notes"

    def apply(self, runner, channel: int, number: int, value: int,
              top: int = 127) -> str | None:
        if value:
            return super().apply(runner, channel, number, value, top)
        dial = self.dial_for(channel, number)
        if dial is not None:
            self.seen += 1
            self.release(runner, dial)
        return dial


# A thread, not a callback: the frame loop must never wait on MIDI, nor MIDI on a frame.
# Handler exceptions are caught PER MESSAGE -- an unguarded raise used to end MIDI silently.
class ClockReader(threading.Thread):
    """Polls MIDI inputs and drives a clock. Silent and harmless if there are none."""

    daemon = True

    def __init__(self, clock: MusicalClock, port_match: str = "", poll: float = 0.001,
                 on_control=None, on_note=None, on_pressure=None) -> None:
        super().__init__()
        self.clock = clock
        self.on_pressure = on_pressure
        self.on_note = on_note
        self.on_control = on_control
        self.port_match = port_match.lower()
        self.poll = poll
        self.stop_flag = False
        self.counts: dict[str, int] = {}
        self.ports: list[str] = []
        self.rejected: list[str] = []
        self.error: str | None = None
        self.faults: dict[str, int] = {}
        self.first_fault: str | None = None
        self.nrpn = Nrpn()

    def open_ports(self):
        """Every matching input port, or every port if no filter was given."""
        out, self.rejected, self.error = open_inputs(self.port_match)
        self.ports = [label for label, _ in out]
        return out

    def run(self) -> None:
        ports = self.open_ports()
        if not ports:
            return
        import pygame.midi

        # PortMidi's clock, in ms, when it is running: then each message is placed at the time
        # it was stamped rather than the time of this poll, and a batch read after a stall does
        # not hand every clock pulse in it the same instant.
        stamped = pygame.midi.get_init()
        while not self.stop_flag:
            now = time.perf_counter()
            stamp_now = pygame.midi.time() if stamped else None
            for _label, port in ports:
                while port.poll():
                    for event, ts in port.read(128):
                        at = now if stamp_now is None else now - max(0, stamp_now - ts) / 1000.0
                        try:
                            self._handle(event, at)
                        except Exception as exc:                  # noqa: BLE001  see `faults`
                            key = f"{type(exc).__name__}: {exc}"
                            self.faults[key] = self.faults.get(key, 0) + 1
                            if self.first_fault is None:
                                self.first_fault = key
            time.sleep(self.poll)

    def _handle(self, event, now: float) -> None:
        """Classify one message and hand it on. Separate from `run` so it can be guarded."""
        what = dispatch(self.clock, event[0], event[1], event[2], now)
        if what:
            self.counts[what] = self.counts.get(what, 0) + 1
        if what == "control_change" and self.on_control is not None:
            channel, cc, value = event[0] & 0x0F, event[1], event[2]
            got = self.nrpn.feed(channel, cc, value) if cc in NRPN_CCS else (cc, value, 127)
            if got is not None:
                self.on_control(channel, *got)
        elif what == "note_on" and self.on_note is not None:
            self.on_note(event[0] & 0x0F, event[1], event[2])
        elif what == "aftertouch_poly" and self.on_pressure is not None:
            self.on_pressure(event[0] & 0x0F, event[1], event[2])

    def trouble(self) -> str:
        """What went wrong on this thread, for the end-of-run report. Empty if nothing did."""
        if not self.faults:
            return ""
        total = sum(self.faults.values())
        worst = max(self.faults, key=lambda k: self.faults[k])
        return (f"{total} MIDI message(s) were dropped by a handler that raised. "
                f"Most common: {worst}. First: {self.first_fault}")

    def describe(self) -> str:
        if self.error and not self.ports:
            return f"MIDI: {self.error}"
        if not self.ports and self.rejected:
            return (f"MIDI: --midi-port {self.port_match!r} matched none of "
                    f"{', '.join(self.rejected)}. The picture will run at its own tempo and "
                    f"look entirely plausible while ignoring the machine.")
        if not self.ports:
            return ("MIDI: no input ports at all. The picture will run at its own tempo and "
                    "look entirely plausible while ignoring the machine.")
        return (f"MIDI: listening to {', '.join(self.ports)}"
                + (f" (ignoring {', '.join(self.rejected)})" if self.rejected else ""))
