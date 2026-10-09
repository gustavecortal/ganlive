// Presets in a browser, as ganlive's `presets`: dial values plus the rules connecting what
// the drums do to what the picture does. A preset is the same JSON as a desktop preset file
// (`to_dict`), so one saved on either plays on the other.

import { clamp01, MOTION, Surface } from "./dials.mjs";
import { walkConfig } from "./clock.mjs";
import { NEVER } from "./midi.mjs";

export const AMOUNT_MAX = 1;
export const clampAmount = (x) => Math.max(-AMOUNT_MAX, Math.min(AMOUNT_MAX, x));

const IMPULSE = { track: undefined, dial: undefined, amount: undefined, decay: 0.25, velocity: 0.7, attack: 0 };
const MACRO = { source: undefined, dial: undefined, src_lo: undefined, src_hi: undefined, out_lo: undefined, out_hi: undefined, glide: 0.8 };
const PRESET = { name: "", blurb: "", dials: {}, impulses: [], macros: [], base_seed: 5, loop_segments: 0, home_every: 0 };

/** One drum hit pushing one dial and letting go: `track` (or "*" for any) pushes `dial` by
 *  `amount`, decaying over `decay` s, rising over `attack`, scaled by the hit by `velocity`. */
export function impulse(fields) {
  const imp = { ...IMPULSE, ...fields };
  imp.amount = clampAmount(imp.amount);
  return imp;
}

/** The push `since` seconds after a hit of strength `velocity`. */
export function impulseValue(imp, since, velocity) {
  if (since >= imp.decay * 6) return 0;
  let shape = Math.exp(-since / Math.max(imp.decay, 1e-4));
  if (imp.attack > 0 && since < imp.attack) shape *= since / imp.attack;
  return imp.amount * shape * (1 - imp.velocity + imp.velocity * velocity);
}

/** A slow move: a whole-kit measurement (`density`, `energy` or `active`) mapped onto a dial. */
export const macro = (fields) => ({ ...MACRO, ...fields });

function macroMap(m, value) {
  if (m.src_hi === m.src_lo) return m.out_lo;
  const u = (value - m.src_lo) / (m.src_hi - m.src_lo);
  return m.out_lo + clamp01(u) * (m.out_hi - m.out_lo);
}

/** Every field that is not simply its default. */
function spare(obj, defaults, skip = []) {
  const out = {};
  for (const [key, value] of Object.entries(obj)) {
    if (skip.includes(key)) continue;
    if (JSON.stringify(value) !== JSON.stringify(defaults[key])) out[key] = value;
  }
  return out;
}

/** A preset as plain data: its name and blurb, then only what differs from a bare one, as a
 *  desktop preset file. */
export function toDict(preset) {
  const out = { name: preset.name, blurb: preset.blurb, ...spare(preset, PRESET, ["impulses", "macros"]) };
  if (preset.impulses.length) out.impulses = preset.impulses.map((r) => spare(r, IMPULSE));
  if (preset.macros.length) out.macros = preset.macros.map((r) => spare(r, MACRO));
  return out;
}

/** The other direction. Anything unrecognised is refused rather than dropped. */
export function fromDict(data) {
  for (const key of Object.keys(data))
    if (!(key in PRESET)) throw new Error(`"${key}" is not part of a preset; have ${Object.keys(PRESET).sort().join(", ")}`);
  const rules = (list, defaults, make, kind) =>
    (list ?? []).map((rule) => {
      const unknown = Object.keys(rule).filter((k) => !(k in defaults));
      if (unknown.length) throw new Error(`${unknown.sort().join(", ")} is not part of a ${kind}; have ${Object.keys(defaults).sort().join(", ")}`);
      return make(rule);
    });
  return {
    ...structuredClone(PRESET),
    ...data,
    dials: { ...(data.dials ?? {}) },
    impulses: rules(data.impulses, IMPULSE, impulse, "impulse"),
    macros: rules(data.macros, MACRO, macro, "macro"),
  };
}

/** What the interface starts on: every dial at rest and nothing wired. */
export const DEFAULT = fromDict({
  name: "default",
  blurb: "Every dial at rest, nothing routed. Wire the drums up in the routing grid and save what you find.",
});

/** Turns one frame of drum features into dial values, then applies them. Dials come from the
 *  preset's values, then its rules, then the holders (mouse, knobs, pads), which `hold` and
 *  `free`, the higher `priority` winning where two hold one dial. `channelOf` says which
 *  feature channel each track arrives on. */
export class PresetRunner {
  constructor(preset, channelOf, fps, layout, channels = 0) {
    this.channelOf = channelOf;
    this.fps = fps;
    this.channels = channels || Math.max(-1, ...Object.values(channelOf)) + 1;
    this.surface = new Surface(layout);
    this.walk = walkConfig();
    this.sources = {};
    this.rank = {};
    this.hands = {};
    this.handsFrom = {};
    this.macroState = {};
    this.velocity = {};
    /** The dials this model can write; empty means everything is playable. */
    this.live = new Set();
    this.model = null;
    this.load(preset);
  }

  playable(name) {
    return !this.live.size || this.live.has(name);
  }

  /** The model changed: take its dials and what is live, and re-read the rules. */
  useModel(model) {
    this.live = new Set(model.live);
    this.surface.relayout(model.layout);
    this.load(this.preset);
    this.replace({ ...this.sources });
    this.model = model;
  }

  unrunnable(dial) {
    if (!this.surface.layout.has(dial)) return "no such dial";
    if (!this.playable(dial)) return "not on this model";
    return null;
  }

  resolveImpulse(imp) {
    const why = this.unrunnable(imp.dial);
    if (why) return [null, why];
    if (imp.track === "*") return [-1, null];
    const channel = this.channelOf[imp.track];
    if (channel === undefined) return [null, "no such track in this kit"];
    if (channel >= this.channels) return [null, `channel ${channel}, kit has ${this.channels}`];
    return [channel, null];
  }

  /** Swap in a preset. Rules this model or kit cannot run are listed in `dropped`. */
  load(preset) {
    this.preset = structuredClone(preset);
    const base = { ...this.surface.layout.rests };
    const impulses = [];
    const dropped = [];
    for (const [name, value] of Object.entries(this.preset.dials)) {
      if (this.surface.layout.has(name)) base[name] = clamp01(value);
      else dropped.push(`${name} (no such dial)`);
    }
    for (const imp of this.preset.impulses) {
      const [channel, why] = this.resolveImpulse(imp);
      if (why) dropped.push(`${imp.track}->${imp.dial} (${why})`);
      else impulses.push([imp, channel]);
    }
    const macros = [];
    for (const m of this.preset.macros) {
      const why = this.unrunnable(m.dial);
      if (why) dropped.push(`${m.source}->${m.dial} (${why})`);
      else macros.push([m, 1 - Math.exp(-1 / Math.max(m.glide * this.fps, 1e-6))]);
    }
    Object.assign(this, { base, impulses, macros, dropped, macroState: {} });
    this.walk.baseSeed = this.preset.base_seed;
    this.walk.loopSegments = this.preset.loop_segments;
    this.walk.homeEvery = this.preset.home_every;
  }

  /** Wire drum tracks to a dial, or unwire them. True if the rule now exists. Takes every
   *  track on one channel at once, since a channel is what fires. */
  wire(tracks, dial, amount = null) {
    const unknown = tracks.filter((t) => !(t in this.channelOf));
    if (!this.surface.layout.has(dial) || unknown.length || !tracks.length)
      throw new Error(`cannot route ${tracks.join("/") || "<nothing>"} to ${dial}`);
    if (!this.playable(dial)) throw new Error(`${dial} is not on the loaded model`);
    const on = new Set(tracks);
    const kept = this.preset.impulses.filter((imp) => !(on.has(imp.track) && imp.dial === dial));
    const added = kept.length === this.preset.impulses.length;
    if (added) {
      if (amount === null) amount = (this.base[dial] ?? 0) > 0.6 ? -0.3 : 0.3;
      // A short rise on the motion dials, where an instant shove reads as a visible step.
      const attack = MOTION.includes(dial) ? 0.06 : 0;
      for (const track of tracks) kept.push(impulse({ track, dial, amount, attack }));
    }
    this.preset.impulses = kept;
    this.load(this.preset);
    return added;
  }

  /** `dial -> [[rule, channel]]` for the rules that will fire. -1 is any hit. */
  routing() {
    const out = {};
    for (const [imp, channel] of this.impulses) (out[imp.dial] ??= []).push([imp, channel]);
    return out;
  }

  rulesOn(dial, channel) {
    return this.impulses.filter(([imp, ch]) => imp.dial === dial && ch === channel).map(([imp]) => imp);
  }

  /** The strength at `(dial, channel)`, or null if nothing is wired there. */
  amountOn(dial, channel) {
    const rules = this.rulesOn(dial, channel);
    return rules.length ? rules.reduce((a, r) => (Math.abs(r.amount) > Math.abs(a) ? r.amount : a), rules[0].amount) : null;
  }

  setAmountOn(dial, channel, amount) {
    const rules = this.rulesOn(dial, channel);
    if (!rules.length) return null;
    amount = clampAmount(amount);
    for (const imp of rules) imp.amount = amount;
    return amount;
  }

  /** Holder `source` now holds exactly these dials. */
  hold(source, values, priority = 0) {
    this.rank[source] = priority;
    this.replace({ ...this.sources, [source]: { ...values } });
  }

  heldBy(source) {
    return { ...(this.sources[source] ?? {}) };
  }

  /** Let go of one dial for one holder, or of everything it holds. */
  free(source, name = null) {
    const held = this.sources[source];
    if (!held || !Object.keys(held).length) return;
    const keep = name === null ? {} : Object.fromEntries(Object.entries(held).filter(([k]) => k !== name));
    this.replace({ ...this.sources, [source]: keep });
  }

  replace(sources) {
    const merged = {};
    const owner = {};
    const order = Object.keys(sources);
    for (const name of [...order].sort((a, b) => (this.rank[a] ?? 0) - (this.rank[b] ?? 0) || order.indexOf(a) - order.indexOf(b))) {
      for (const [k, v] of Object.entries(sources[name])) {
        if (!this.playable(k)) continue;
        merged[k] = v;
        owner[k] = name;
      }
    }
    this.sources = sources;
    this.hands = merged;
    this.handsFrom = owner;
  }

  /** Record the strength of each hit that arrived this frame, per channel. */
  observe(onsets) {
    for (const [ch, vel] of onsets) this.velocity[ch] = clamp01(vel);
  }

  /** Write this frame's control state: preset values, held dials, macros, then impulses, into
   *  the model's `settings` and the walk. */
  apply(since, features, settings) {
    const s = this.surface;
    Object.assign(s.values, this.base);
    s.setHeld(this.hands);
    for (const [m, k] of this.macros) {
      const target = macroMap(m, Number(features[m.source] ?? 0));
      let cur = this.macroState[m.dial] ?? target;
      cur += (target - cur) * k;
      this.macroState[m.dial] = cur;
      s.set(m.dial, cur);
    }
    const reaction = s.values.reaction * 2;
    const sinceAny = since.length ? Math.min(...since) : NEVER;
    for (const [imp, ch] of this.impulses) {
      const [elapsed, velocity] = ch < 0 ? [sinceAny, 1] : [since[ch], this.velocity[ch] ?? 1];
      const add = impulseValue(imp, elapsed, velocity) * reaction;
      if (add) s.add(imp.dial, add);
    }
    s.apply(settings, this.walk);
    settings.commit();
  }
}
