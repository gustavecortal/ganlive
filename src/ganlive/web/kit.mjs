// The twelve-track drum vocabulary, as ganlive's `control.kit`: the Analog Rytm's pad order,
// and the two ways a machine's MIDI is wired to it (which note, or which channel, is which track).

export const TRACKS = ["BD", "SD", "RS", "CP", "BT", "LT", "MT", "HT", "CH", "OH", "CY", "CB"];
export const INDEX = Object.fromEntries(TRACKS.map((t, i) => [t, i]));

/** A track's index from its name or its index ("0" to "11"), or an error. */
export function trackIndex(name) {
  name = name.trim().toUpperCase();
  if (/^\d+$/.test(name)) {
    const index = Number(name);
    if (index < 0 || index >= TRACKS.length) throw new Error(`track ${name} is out of range; have 0 to ${TRACKS.length - 1}`);
    return index;
  }
  if (!(name in INDEX)) throw new Error(`unknown track "${name}"; have ${TRACKS.join(", ")}, or 0 to ${TRACKS.length - 1}`);
  return INDEX[name];
}

/** "a=b,c=d" as [a, b], [c, d]: the comma-list grammar every map shares. */
export function* pairs(text) {
  for (let part of text.split(",")) {
    part = part.trim();
    if (!part) continue;
    const at = part.indexOf("=");
    yield at < 0 ? [part, ""] : [part.slice(0, at), part.slice(at + 1)];
  }
}

/** Which note is which track, for a kit on one channel, as `{note: track}`: a first note for
 *  pads that send consecutive notes ("0"), or a map ("36=BD,38=SD,42=CH,46=OH"). */
export function parseNotes(text) {
  text = text.trim();
  if (/^\d+$/.test(text)) return Object.fromEntries(TRACKS.map((_, i) => [Number(text) + i, i]));
  const out = {};
  for (const [note, name] of pairs(text)) {
    const number = Number.parseInt(note, 10);
    if (Number.isNaN(number)) throw new Error(`"${note}" is not a note number`);
    if (number in out) throw new Error(`note ${number} is already ${TRACKS[out[number]]}; two tracks on one note cannot be told apart`);
    out[number] = trackIndex(name);
  }
  if (!Object.keys(out).length) throw new Error("the notes need a first note, as 0, or a map, as 36=BD,38=SD");
  return out;
}

/** "1-12" or "1=BD,2=SD" to `{MIDI channel (from 0): track}`. */
export function parseTrackChannels(text) {
  text = text.trim();
  const out = {};
  if (!text.includes("=")) {
    const [low, high] = text.split("-").map(Number);
    const span = high - low + 1;
    if (span !== TRACKS.length) throw new Error(`${text} is ${span} channels for ${TRACKS.length} tracks; the whole kit or an explicit map`);
    TRACKS.forEach((_, i) => (out[low - 1 + i] = i));
    return out;
  }
  for (const [channel, name] of pairs(text)) {
    const index = trackIndex(name);
    const number = Number.parseInt(channel, 10) - 1;
    if (number in out) throw new Error(`MIDI channel ${number + 1} is already ${TRACKS[out[number]]}; two tracks on one channel cannot be told apart`);
    out[number] = index;
  }
  return out;
}

/** `{track: channel}` turned round: which tracks share each channel. */
export function byChannel(channelOf) {
  const out = {};
  for (const [track, channel] of Object.entries(channelOf)) (out[channel] ??= []).push(track);
  return out;
}

/** How a machine sends its drums, from `{channel: notes seen}`: "auto" (the kit on one
 *  channel), "track" (one channel a track), "mixed", or null when the traffic cannot tell. */
export function outputMode(seen, known = TRACKS.map((_, i) => i)) {
  known = new Set(known);
  const low = new Set();
  const high = new Set();
  for (const [ch, notes] of Object.entries(seen)) {
    if ([...notes].some((n) => known.has(n))) low.add(ch);
    if ([...notes].some((n) => !known.has(n))) high.add(ch);
  }
  for (const ch of low) high.delete(ch);
  if (low.size && high.size > 1) return "mixed";
  if (low.size) return "auto";
  return high.size > 1 ? "track" : null;
}
