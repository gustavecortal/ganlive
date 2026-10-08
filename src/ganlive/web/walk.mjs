// The latent walk in a browser, as ganlive's `walk.SlerpWalk`: a great-circle path through
// seeded latents, addressed by beat. Each seed is drawn with the engine's own generator
// (`engine.noise.seeded_noise`), so a preset's seed visits the same pictures on the desktop.

import { position } from "./clock.mjs";

/** The engine's PCG hash of a 32-bit word. */
export function pcg(v) {
  const s = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

const f = Math.fround;
const unit = (v) => f((f(pcg(v) >>> 8) + 0.5) / 16777216);

/** `n` standard normal values from `seed`, as `engine.noise.seeded_noise` computes them. */
export function seededNoise(n, seed) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = unit(((2 * i) ^ seed) >>> 0);
    const b = unit(((2 * i + 1) ^ seed) >>> 0);
    out[i] = f(f(Math.sqrt(f(-2 * f(Math.log(a))))) * f(Math.cos(f(f(6.2831853) * b))));
  }
  return out;
}

/** The angle between two latents, in radians. */
function angle(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return Math.acos(Math.min(1, Math.max(-1, dot / Math.sqrt(na * nb))));
}

/** `[a, b]` such that `a * z0 + b * z1` is the point `t` of the way along the great circle. */
function weights(omega, t) {
  if (Math.abs(omega) < 1e-6) return [1 - t, t];
  const so = Math.sin(omega);
  return [Math.sin((1 - t) * omega) / so, Math.sin(t * omega) / so];
}

const mix = (a, wa, b, wb) => {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = wa * a[i] + wb * b[i];
  return out;
};

export class SlerpWalk {
  /** The width every seed is drawn at; a model reads the first `nz`. */
  static CANON = 4096;

  constructor(nz, cfg) {
    this.nz = nz;
    this.cfg = cfg;
    this.key = null;
    this.home = null;
  }
  get canon() {
    return Math.max(SlerpWalk.CANON, this.nz);
  }
  /** Hand the walk to a model of another latent width; it resumes where it was. */
  retarget(nz) {
    if (nz === this.nz) return;
    this.nz = nz;
    this.key = null;
  }
  /** A draw that depends only on `(baseSeed, salt, k)`. */
  draw(k, salt = 0) {
    const seed = (((this.cfg.baseSeed * 1000003 + salt * 7919 + k) % 2147483648) + 2147483648) % 2147483648;
    return seededNoise(this.canon, pcg(seed));
  }
  homeCanon(k) {
    const block = this.cfg.homeEvery ? Math.floor(k / this.cfg.homeEvery) : 0;
    const key = `${this.cfg.baseSeed},${block},${this.canon}`;
    if (this.home?.[0] !== key) this.home = [key, this.draw(block, 1)];
    return this.home[1];
  }
  /** Segment `k`'s target latent at the walk's own width: pulled toward its home by `spread`. */
  seedCanon(k) {
    if (this.cfg.loopSegments) k %= this.cfg.loopSegments;
    const target = this.draw(k);
    if (this.cfg.spread >= 1) return target;
    const home = this.homeCanon(k);
    const [a, b] = weights(angle(home, target), Math.max(0, this.cfg.spread));
    return mix(home, a, target, b);
  }
  seedFor(k) {
    return this.seedCanon(k).subarray(0, this.nz);
  }
  /** What a segment's endpoints depend on: which seeds, not how far (`spread`) they go,
   *  which the next segment takes up. */
  keyOf(k) {
    return `${k},${this.cfg.baseSeed},${this.cfg.loopSegments},${this.cfg.homeEvery}`;
  }
  load(k) {
    const key = this.keyOf(k);
    if (this.key === key) return;
    const z0 = this.key === this.keyOf(k - 1) && this.z1 ? this.z1 : this.seedFor(k).slice();
    this.z1 = this.seedFor(k + 1).slice();
    this.z0 = z0;
    this.omega = angle(z0, this.z1);
    this.key = key;
    this.segment = k;
  }
  /** The latent for this musical position, with the direction push added. */
  latent(beats) {
    const [k, , t] = position(this.cfg, beats);
    this.load(k);
    const [a, b] = weights(this.omega, t);
    const z = mix(this.z0, a, this.z1, b);
    const rows = this.cfg.directions;
    const amounts = this.cfg.amounts;
    if (rows) {
      const n = Math.min(amounts.length, rows.length);
      for (let r = 0; r < n; r++) {
        const amount = amounts[r];
        if (!amount || rows[r].length !== this.nz) continue;
        for (let i = 0; i < this.nz; i++) z[i] += amount * rows[r][i];
      }
    }
    return z;
  }
}
