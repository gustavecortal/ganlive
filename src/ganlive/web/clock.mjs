// Musical time in a browser, as ganlive's `clock`: a beat clock driven by a BPM or by MIDI
// clock, and where in the walk a beat falls. Times are in seconds.

import { clamp01 } from "./dials.mjs";

export const BEATS_PER_BAR = 4;

const smoothstep = (t) => t * t * (3 - 2 * t);
const easeOut = (t) => 1 - (1 - t) ** 3;
const easeIn = (t) => t * t * t;

/** How far along the move the picture is at phase `u`: `hold` of it still, placed by `when`. */
export function shape(u, hold, when) {
  if (hold > 0) u = clamp01((u - hold * when) / Math.max(1 - hold, 1e-6));
  if (when <= 0.5) {
    const a = when * 2;
    return easeOut(u) + (smoothstep(u) - easeOut(u)) * a;
  }
  const a = (when - 0.5) * 2;
  return smoothstep(u) + (easeIn(u) - smoothstep(u)) * a;
}

/** Beats elapsed: free-running from a BPM, or advanced by MIDI clock pulses. */
export class MusicalClock {
  static PPQN = 24;

  constructor(bpm = 130) {
    this.bpm = bpm;
    this.beats = 0;
    this.pulses = 0;
    this.external = false;
    this.lastPulse = null;
    this.period = null;
    this.running = true;
  }
  get source() {
    return this.external ? "midi" : "internal";
  }
  /** Advance by `dt` seconds. Ignored once MIDI clock is driving. */
  advance(dt) {
    if (this.external || !this.running) return;
    this.beats += (dt * this.bpm) / 60;
  }
  /** One MIDI clock pulse, at `now` seconds: it takes over from the internal clock, and the
   *  tempo is a smoothed estimate from the pulse spacing. */
  onPulse(now) {
    this.external = true;
    if (this.running) {
      this.pulses += 1;
      this.beats = this.pulses / MusicalClock.PPQN;
    }
    if (now != null) {
      if (this.lastPulse != null) {
        const dt = now - this.lastPulse;
        if (dt > 0.0005 && dt < 0.5) {
          this.period = this.period == null ? dt : 0.85 * this.period + 0.15 * dt;
          this.bpm = 60 / (this.period * MusicalClock.PPQN);
        }
      }
      this.lastPulse = now;
    }
  }
  /** MIDI Song Position Pointer: 16th notes since the start of the song. */
  onSongPosition(sixteenths) {
    this.pulses = Math.round((sixteenths * MusicalClock.PPQN) / 4);
    this.beats = this.pulses / MusicalClock.PPQN;
  }
  onStart() {
    this.pulses = 0;
    this.beats = 0;
    this.running = true;
  }
  onContinue() {
    this.running = true;
  }
  onStop() {
    this.running = false;
  }
}

/** What the walk does between seeds. Every field may change while it runs. */
export function walkConfig(fields = {}) {
  return {
    beatsPerSegment: 4, spread: 1, homeEvery: 0, hold: 0, when: 0.5, stepGrid: 0,
    baseSeed: 0, loopSegments: 0,
    /** Rows of the model's latent directions, and how far each is pushed. */
    directions: null, amounts: [],
    ...fields,
  };
}

/** `[segment, raw phase, how far along the move]` for a musical position. */
export function position(cfg, beats) {
  const b = Math.max(0, beats) / Math.max(cfg.beatsPerSegment, 1e-6);
  const k = Math.floor(b);
  const u = b - k;
  let t = shape(u, cfg.hold, cfg.when);
  if (cfg.stepGrid) t = Math.floor(t * cfg.stepGrid) / cfg.stepGrid;
  return [k, u, t];
}
