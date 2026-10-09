// The dials in a browser: ganlive's `curves`, `dials.table` and the settings vector, the same
// arithmetic as the desktop strip, so a dial at one position does the same on both. A model's
// own dials (its layout) come from its program, measured when it was converted; the shared
// blocks' mechanics are here. tests/test_web.py checks these against the Python.

export const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** A curve given as values at even spacing, as `[position, value]` pairs. */
export function evenly(values) {
  const last = Math.max(1, values.length - 1);
  return values.map((v, i) => [i / last, v]);
}

/** A dial's value at position `x`, walking the line between its measured points. */
export function at(points, x) {
  x = clamp01(x);
  for (let i = 1; i < points.length; i++) {
    const [x1, v1] = points[i];
    if (x <= x1) {
      const [x0, v0] = points[i - 1];
      return x1 === x0 ? v0 : v0 + ((v1 - v0) * (x - x0)) / (x1 - x0);
    }
  }
  return points[points.length - 1][1];
}

/** How many latent directions get a dial. */
export const DIRECTIONS = 8;
/** The furthest a direction dial pushes, in units of the direction's own length. */
export const DIRECTION_RANGE = 2.5;
/** The fraction of full push that buys each even share of a direction's change (`table.DIRECTION_PUSH`). */
export const DIRECTION_PUSH = [0.0, 0.0662, 0.1342, 0.216, 0.3078, 0.4147, 0.5465, 0.7166, 1.0];

/** A direction dial's travel: rests centred, pushes `DIRECTION_RANGE` either way. */
export const DIRECTION_POINTS = (() => {
  const response = evenly(DIRECTION_PUSH);
  return [
    ...response.slice(1).reverse().map(([share, push]) => [0.5 - 0.5 * share, -DIRECTION_RANGE * push]),
    ...response.map(([share, push]) => [0.5 + 0.5 * share, DIRECTION_RANGE * push]),
  ];
})();

export const MASTER = ["reaction"];
export const MOTION = ["speed", "spread", "hold", "late", "grid"];
export const DIRECTION_DIALS = Array.from({ length: DIRECTIONS }, (_, i) => `dir${i + 1}`);
export const SPEED_BEATS = [8, 4, 2, 1, 0.5];
export const GRID_STEPS = [0, 4, 8, 16, 32];
/** The walk's spread against how far a move goes, measured: `[spread, distance]`. */
export const SPREAD_TABLE = [[0, 0], [0.05, 1.8], [0.1, 3.5], [0.2, 7.0], [0.35, 11.8], [0.5, 16.0], [0.7, 20.2], [1.0, 22.5]];
export const SPREAD_MAX = SPREAD_TABLE[SPREAD_TABLE.length - 1][1];
export const SPREAD_POINTS = SPREAD_TABLE.map(([near, far]) => [far / SPREAD_MAX, near]);
export const RANGE_WORDS = [[3.5, "breathing"], [16.0, "altering"]];
export const HOLD_MAX = 0.95;
export const POLES = { reaction: ["calmer", "harder", "as written"] };

/** Which direction a dial drives, counting from 0, or -1. */
export const directionIndex = (name) => DIRECTION_DIALS.indexOf(name);
const detent = (x, values) => values[Math.min(values.length - 1, Math.max(0, Math.round(x * (values.length - 1))))];
export const spreadFor = (x) => at(SPREAD_POINTS, x);
export const holdFor = (x) => Math.min(HOLD_MAX, clamp01(x));

/** What a dial's position means in its own units, as the strip says it. */
export function readout(name, value, layout) {
  const v = clamp01(value);
  const knob = layout?.get(name);
  if (name === "speed") {
    const beats = detent(v, SPEED_BEATS);
    return `${beats} beat${beats === 1 ? "" : "s"}`;
  }
  if (name === "grid") {
    const steps = detent(v, GRID_STEPS);
    return steps ? `${steps} steps` : "glide";
  }
  if (name === "spread") {
    const far = v * SPREAD_MAX;
    const word = RANGE_WORDS.find(([edge]) => far < edge)?.[1] ?? "new scene";
    return `${far.toFixed(1)} ${word}`;
  }
  if (name === "hold") return `${(holdFor(v) * 100).toFixed(0)}% still`;
  if (name === "late") {
    if (Math.abs(v - 0.5) < 0.02) return "spread evenly";
    return `${v > 0.5 ? "arrives" : "leaves"} ${(Math.abs(v - 0.5) * 200).toFixed(0)}%`;
  }
  if (directionIndex(name) >= 0) {
    const push = at(DIRECTION_POINTS, v);
    return Math.abs(push) < 0.02 ? "centred" : `${push >= 0 ? "+" : ""}${push.toFixed(2)}`;
  }
  const poles = knob?.poles ?? POLES[name];
  if (poles) {
    const rest = knob ? knob.rest : 0.5;
    const span = Math.max(rest, 1 - rest);
    const away = span ? (v - rest) / span : 0;
    return Math.abs(away) < 0.02 ? poles[2] : `${poles[away > 0 ? 1 : 0]} ${(Math.abs(away) * 100).toFixed(0)}%`;
  }
  return v.toFixed(2);
}

/** The dials one model offers, in drawing order: a program's `knobs` (`player.browser_layout`). */
export class Layout {
  constructor(knobs) {
    this.knobs = knobs;
    this.byName = new Map(knobs.map((k) => [k.name, k]));
    this.rests = Object.fromEntries(knobs.map((k) => [k.name, k.rest]));
    const blocks = new Map();
    for (const k of knobs) {
      if (!blocks.has(k.group)) blocks.set(k.group, []);
      blocks.get(k.group).push(k.name);
    }
    /** `[title, names]` per block, in first-seen order. */
    this.groups = [...blocks];
  }
  get(name) {
    return this.byName.get(name);
  }
  has(name) {
    return this.byName.has(name);
  }
  get names() {
    return [...this.byName.keys()];
  }
}

/** The dials whose meaning belongs to one model: its MODEL block and its directions. */
export const perModel = (layout) =>
  layout.knobs.filter((k) => k.group === "MODEL" || directionIndex(k.name) >= 0).map((k) => k.name);

/** A model's settings vector on the host: written, committed, uploaded when it changed. */
export class Settings {
  constructor(names) {
    this.names = names;
    this.index = new Map(names.map((n, i) => [n, i]));
    this.write = new Float32Array(names.length).fill(1);
    this.sent = new Float32Array(names.length).fill(1);
    /** Committed and not yet uploaded. True at first, so the first frame uploads. */
    this.changed = true;
  }
  set(name, value) {
    const i = this.index.get(name);
    if (i !== undefined) this.write[i] = value;
  }
  commit() {
    for (let i = 0; i < this.write.length; i++) {
      if (this.write[i] !== this.sent[i]) {
        this.sent.set(this.write);
        this.changed = true;
        return;
      }
    }
  }
}

/** The dial values, and the one method that turns them into everything downstream. */
export class Surface {
  constructor(layout) {
    this.layout = layout;
    this.values = { ...layout.rests };
    this.held = new Set();
  }
  /** Take a different model's dials, keeping every value the two have in common. */
  relayout(layout) {
    this.layout = layout;
    this.values = Object.fromEntries(Object.entries(layout.rests).map(([n, rest]) => [n, this.values[n] ?? rest]));
  }
  /** Put these dials where they are asked and stop `set` moving them. */
  setHeld(values) {
    this.held = new Set(Object.keys(values));
    for (const [name, value] of Object.entries(values)) if (name in this.values) this.values[name] = clamp01(value);
  }
  set(name, value) {
    if (name in this.values && !this.held.has(name)) this.values[name] = clamp01(value);
  }
  add(name, value) {
    if (name in this.values) this.values[name] = clamp01(this.values[name] + value);
  }
  /** Write every dial through to the model's settings and the walk, once a frame. */
  apply(settings, walk) {
    const v = this.values;
    walk.beatsPerSegment = detent(v.speed, SPEED_BEATS);
    walk.spread = spreadFor(v.spread);
    walk.hold = holdFor(v.hold);
    walk.when = clamp01(v.late);
    walk.stepGrid = detent(v.grid, GRID_STEPS);
    for (const knob of this.layout.knobs)
      for (const [setting, points] of knob.writes) settings.set(setting, at(points, v[knob.name]));
    walk.amounts = DIRECTION_DIALS.map((name) => (name in v ? at(DIRECTION_POINTS, v[name]) : 0));
  }
}
