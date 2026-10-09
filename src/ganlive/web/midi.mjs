// MIDI in a browser, as ganlive's `control.midi` and `control.features.NoteFeatures`: clock and
// transport into a `MusicalClock`, note-ons as drum hits, and knobs (CC and NRPN) and pad
// pressure holding dials. `MidiIn` reads Web MIDI the way the desktop's `ClockReader` reads
// PortMidi. Times are in seconds on `performance.now()`'s clock.

import { clamp01 } from "./dials.mjs";
import { pairs, TRACKS, trackIndex } from "./kit.mjs";

/** `since` for a track that has never fired. */
export const NEVER = 1e6;
/** The window the whole-kit measurements (`density`, `energy`, `active`) average over. */
const ENERGY_WINDOW = 1.85;

/** Per-track drum features from note-ons: seconds since each track fired (`since`), the hits
 *  since the last frame (`drain`), and the whole-kit measurements a macro reads. */
export class NoteFeatures {
  /** `notes` is `{note: track}` for a kit on one channel; `channels`, `{channel: track}`, for
   *  one that puts each track on its own channel, checked first. */
  constructor(tracks = TRACKS.length, channels = null, notes = null) {
    this.n = tracks;
    this.notes = notes ?? Object.fromEntries(Array.from({ length: tracks }, (_, i) => [i, i]));
    this.channels = channels;
    this.unresolved = 0;
    /** Note-ons that named no track: a sample of `"channel:note" -> count`. */
    this.unclaimed = new Map();
    this.since = new Float32Array(tracks).fill(NEVER);
    this.hits = new Array(tracks).fill(0);
    this.pending = [];
    this.density = 0;
    this.energy = 0;
    this.active = 0;
    this.last = null;
    this.recent = [];
    this.now = 0;
  }
  trackOf(channel, note) {
    if (this.channels) {
      const index = this.channels[channel];
      if (index !== undefined) return index >= 0 && index < this.n ? index : null;
    }
    const index = this.notes[note];
    return index !== undefined && index >= 0 && index < this.n ? index : null;
  }
  /** One trig or pad. Returns the track, or null. */
  onNote(channel, note, velocity, when = null) {
    const index = this.trackOf(channel, note);
    if (index === null) {
      this.unresolved += 1;
      const key = `${channel}:${note}`;
      if (this.unclaimed.size < 32 || this.unclaimed.has(key)) this.unclaimed.set(key, (this.unclaimed.get(key) ?? 0) + 1);
      return null;
    }
    this.onTrack(index, Math.min(1, velocity / 127), when);
    return index;
  }
  onTrack(index, strength, when = null) {
    const now = when ?? this.now;
    this.pending.push([index, strength, 0]);
    // Counted from the last tick, so the next tick brings it to the right age.
    this.since[index] = this.last === null ? 0 : this.last - now;
    this.hits[index] += 1;
    this.recent.push([now, strength]);
  }
  /** Advance time to `now`, once a frame, before anything reads `since`. */
  tick(now) {
    if (this.last !== null) {
      const elapsed = Math.max(0, now - this.last);
      if (elapsed) for (let i = 0; i < this.n; i++) this.since[i] += elapsed;
    }
    this.last = this.now = now;
    while (this.recent.length && now - this.recent[0][0] > ENERGY_WINDOW) this.recent.shift();
    this.density = this.recent.length / ENERGY_WINDOW;
    this.energy = this.recent.length ? this.recent.reduce((a, [, v]) => a + v, 0) / this.recent.length : 0;
    let on = 0;
    for (const s of this.since) if (s <= ENERGY_WINDOW) on++;
    this.active = on / Math.max(1, this.n);
  }
  drain() {
    const out = this.pending;
    this.pending = [];
    return out;
  }
  features() {
    return { density: this.density, energy: this.energy, active: this.active };
  }
  /** How many tracks have fired at all. */
  played() {
    let n = 0;
    for (const s of this.since) if (s < NEVER * 0.1) n++;
    return n;
  }
  /** `{track: index into since}` for the tracks this wiring can reach. */
  channelOf() {
    const reached = new Set(Object.values(this.notes));
    if (this.channels) for (const t of Object.values(this.channels)) reached.add(t);
    return Object.fromEntries([...reached].filter((i) => i < TRACKS.length).sort((a, b) => a - b).map((i) => [TRACKS[i], i]));
  }
}

const SYSTEM = { 0xf8: "clock", 0xfa: "start", 0xfb: "continue", 0xfc: "stop", 0xf2: "song_position" };
const VOICE = { 0xb0: "control_change", 0xa0: "aftertouch_poly" };

/** What one MIDI message is, or null for the kinds ganlive ignores. A note-on at velocity 0
 *  is a release. */
export function classify(status, data2 = 0) {
  if (status >= 0xf0) return SYSTEM[status] ?? null;
  const kind = status & 0xf0;
  if (kind === 0x90) return data2 > 0 ? "note_on" : "note_off";
  return VOICE[kind] ?? null;
}

/** Apply one message to a clock. Returns what it was. */
export function dispatch(clock, status, data1 = 0, data2 = 0, now = null) {
  const what = classify(status, data2);
  if (what === "clock") clock.onPulse(now);
  else if (what === "start") clock.onStart();
  else if (what === "continue") clock.onContinue();
  else if (what === "stop") clock.onStop();
  else if (what === "song_position") clock.onSongPosition(data1 | (data2 << 7));
  return what;
}

/** NRPN parameters share the number space with CCs, above them. */
export const NRPN_BASE = 128;
const NRPN_MSB = 99, NRPN_LSB = 98, DATA_MSB = 6, DATA_LSB = 38;
export const NRPN_CCS = new Set([NRPN_MSB, NRPN_LSB, DATA_MSB, DATA_LSB]);
export const nrpnNumber = (msb, lsb) => NRPN_BASE + (((msb & 0x7f) << 7) | (lsb & 0x7f));

/** `17` for a CC, `n1.3` for an NRPN. */
export function numberName(number) {
  if (number < NRPN_BASE) return String(number);
  const param = number - NRPN_BASE;
  return `n${param >> 7}.${param & 0x7f}`;
}

export function parseNumber(text) {
  text = text.trim().toLowerCase();
  if (text.startsWith("n")) {
    const [msb, lsb] = text.slice(1).split(".");
    return nrpnNumber(Number.parseInt(msb, 10), Number.parseInt(lsb || "0", 10));
  }
  const n = Number.parseInt(text, 10);
  if (Number.isNaN(n)) throw new Error(`"${text}" is not a control number`);
  return n;
}

const dialOrThrow = (name) => {
  name = name.trim();
  if (!name) throw new Error("a mapping needs a dial after the '=', as in 16=noise, 2:17=se_256 or n1.3=dir1");
  return name;
};

/** "16=noise,2:17=se_256,n1.3=dir1" to a Map of "channel:number" -> dial, channel -1 meaning any. */
export function parseControls(text) {
  const out = new Map();
  for (const [where, dial] of pairs(text)) {
    const at = where.trim().lastIndexOf(":");
    const channel = at < 0 ? -1 : Number.parseInt(where.slice(0, at), 10) - 1;
    out.set(`${channel}:${parseNumber(at < 0 ? where : where.slice(at + 1))}`, dialOrThrow(dial));
  }
  return out;
}

/** One control in the map's own words: `17`, `2:17`, `n1.3`. */
export const controlName = (channel, number) => `${channel >= 0 ? `${channel + 1}:` : ""}${numberName(number)}`;

const split = (key) => key.split(":").map(Number);
const keyOrder = (a, b) => {
  const [ca, na] = split(a), [cb, nb] = split(b);
  return ca - cb || na - nb;
};

/** The inverse of `parseControls`. */
export const formatControls = (controls) =>
  [...controls].sort(([a], [b]) => keyOrder(a, b)).map(([key, dial]) => `${controlName(...split(key))}=${dial}`).join(",");

/** The four-CC NRPN state machine, per channel: 99 and 98 name a parameter, 6 and 38 carry
 *  it, 14-bit. Emits on 6 with the low byte at zero and again on 38 with it filled. */
export class Nrpn {
  constructor() {
    this.address = {};
  }
  /** `[number, value, top]` for a data byte, or null for an address byte. */
  feed(channel, cc, value) {
    if (cc === NRPN_MSB) {
      this.address[channel] = [value, 0, 0];
      return null;
    }
    const state = this.address[channel];
    if (!state) return cc === DATA_MSB || cc === DATA_LSB ? [cc, value, 127] : null;
    if (cc === NRPN_LSB) {
      state[1] = value;
      return null;
    }
    const number = nrpnNumber(state[0], state[1]);
    if (cc === DATA_MSB) {
      state[2] = value;
      return [number, value << 7, 16383];
    }
    if (cc === DATA_LSB) return [number, (state[2] << 7) | value, 16383];
    return null;
  }
}

/** Knobs on the machine holding dials, through the runner's `hold`, as the strip's mouse
 *  does. Learns a binding from the next control that moves (`learning`), and hands the map
 *  to `remember` (a function) to keep it. */
export class EncoderMap {
  static SOURCE = "encoder";
  static PRIORITY = 0;

  constructor(controls = new Map(), remember = null) {
    this.controls = new Map(controls);
    this.held = {};
    this.unmapped = new Map();
    this.seen = 0;
    this.learning = null;
    this.learned = [];
    this.version = 0;
    this.remember = remember;
  }
  get source() {
    return this.constructor.SOURCE;
  }
  get priority() {
    return this.constructor.PRIORITY;
  }
  dialFor(channel, number) {
    return this.controls.get(`${channel}:${number}`) ?? this.controls.get(`-1:${number}`) ?? null;
  }
  /** How the machine addresses a dial, in the map's own words, or "". */
  where(dial) {
    return [...this.controls].filter(([, d]) => d === dial).map(([k]) => k).sort(keyOrder)
      .map((k) => controlName(...split(k))).join(" ");
  }
  /** The controls wired to dials this model lacks, as one line, or "". */
  unreachable(layout) {
    const strays = [...new Set(this.controls.values())].filter((d) => !layout.has(d)).sort();
    return strays.length ? `${this.source}: ${strays.join(", ")} not on this model, so ${strays.length} control(s) move nothing until one that has them plays` : "";
  }
  bind(channel, number, dial) {
    for (const [k, d] of [...this.controls]) if (d === dial) this.controls.delete(k);
    this.controls.set(`${channel}:${number}`, dial);
    this.learning = null;
    this.learned.push(`${controlName(channel, number)}=${dial}`);
    this.version += 1;
    this.remember?.(formatControls(this.controls));
  }
  /** One control change. Returns the dial it moved, or null. */
  apply(runner, channel, number, value, top = 127) {
    if (this.learning !== null) this.bind(channel, number, this.learning);
    const dial = this.dialFor(channel, number);
    if (dial === null) {
      const key = `${channel}:${number}`;
      this.unmapped.set(key, (this.unmapped.get(key) ?? 0) + 1);
      return null;
    }
    this.held[dial] = clamp01(value / top);
    this.seen += 1;
    runner.hold(this.source, this.held, this.priority);
    return dial;
  }
  /** Let a knob go, so the preset and the drums have the dial back. */
  release(runner, dial = null) {
    if (dial === null) this.held = {};
    else delete this.held[dial];
    runner.free(this.source, dial);
  }
}

/** A pad leaned on (polyphonic aftertouch) holding a dial, above a knob. Zero pressure lets go. */
export class PressureMap extends EncoderMap {
  static SOURCE = "pressure";
  static PRIORITY = 5;

  apply(runner, channel, number, value, top = 127) {
    if (value) return super.apply(runner, channel, number, value, top);
    const dial = this.dialFor(channel, number);
    if (dial !== null) {
      this.seen += 1;
      this.release(runner, dial);
    }
    return dial;
  }
}

/** "BD=noise,SD=dir1" to the pressure map, a pad named as its track, whose note `notes`
 *  (`parseNotes`) says; without it, track i is note i. */
export function parsePressure(text, notes = null) {
  const noteOf = {};
  if (notes) for (const [note, track] of Object.entries(notes)) noteOf[track] = Number(note);
  const out = new Map();
  for (const [name, dial] of pairs(text)) {
    const track = trackIndex(name);
    out.set(`-1:${noteOf[track] ?? track}`, dialOrThrow(dial));
  }
  return out;
}

/** Every Web MIDI input whose name contains `match` (all of them if empty), read into `clock`
 *  and handed on: knobs to `onControl(channel, number, value, top)`, note-ons to
 *  `onNote(channel, note, velocity, when)`, pressure to `onPressure(channel, note, value)`.
 *  Inputs plugged in later are taken up as they appear. */
export class MidiIn {
  constructor(access, clock, { match = "", onControl = null, onNote = null, onPressure = null } = {}) {
    Object.assign(this, { access, clock, match: match.toLowerCase(), onControl, onNote, onPressure });
    this.counts = {};
    this.nrpn = new Nrpn();
    this.ports = [];
    this.rejected = [];
    this.open();
    access.onstatechange = () => this.open();
  }
  open() {
    this.ports = [];
    this.rejected = [];
    for (const input of this.access.inputs.values()) {
      const wanted = !this.match || input.name.toLowerCase().includes(this.match);
      input.onmidimessage = wanted ? (e) => this.message(e.data, e.timeStamp / 1000) : null;
      (wanted ? this.ports : this.rejected).push(input.name);
    }
  }
  message(data, now) {
    const [status, data1 = 0, data2 = 0] = data;
    const what = dispatch(this.clock, status, data1, data2, now);
    if (!what) return;
    this.counts[what] = (this.counts[what] ?? 0) + 1;
    const channel = status & 0x0f;
    if (what === "control_change" && this.onControl) {
      const got = NRPN_CCS.has(data1) ? this.nrpn.feed(channel, data1, data2) : [data1, data2, 127];
      if (got) this.onControl(channel, ...got);
    } else if (what === "note_on" && this.onNote) this.onNote(channel, data1, data2, now);
    else if (what === "aftertouch_poly" && this.onPressure) this.onPressure(channel, data1, data2);
  }
  stop() {
    for (const input of this.access.inputs.values()) input.onmidimessage = null;
    this.access.onstatechange = null;
  }
  describe() {
    if (!this.ports.length && this.rejected.length)
      return `"${this.match}" matches none of ${this.rejected.join(", ")}: the picture runs at its own tempo and ignores the machine.`;
    if (!this.ports.length) return "No MIDI inputs: the picture runs at its own tempo. Plug one in and it is taken up.";
    return `Listening to ${this.ports.join(", ")}${this.rejected.length ? ` (not ${this.rejected.join(", ")})` : ""}.`;
  }
}
