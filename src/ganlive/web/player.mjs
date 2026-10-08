// A ganlive performance in a browser: what `ganlive play` runs each frame, on WebGPU. Models
// load from their program and weights (`runner.mjs`), the beat comes from MIDI clock or a
// BPM, drum hits from MIDI notes, and the dials from the preset, its rules and whatever holds
// them (the page's sliders, the machine's knobs and pads). A page draws the controls; this
// file owns none of them.

import { MusicalClock } from "./clock.mjs";
import { Layout, perModel, Settings } from "./dials.mjs";
import { NoteFeatures } from "./midi.mjs";
import { DEFAULT, PresetRunner } from "./presets.mjs";
import { loadModel, screen } from "./runner.mjs";
import { SlerpWalk } from "./walk.mjs";

/** The most the clock moves in one frame, so a stalled frame does not jump the picture. */
const LONGEST_STEP_S = 0.25;
/** Frames on the card at most, so a GPU slower than the screen never builds a queue. */
const DEPTH = 2;
/** Frames of a take drawn in one animation frame at most, catching up after a late one. */
const CATCH_UP = 4;
/** Frames of a take worked out in one animation frame at most: a tab hidden for minutes catches
 *  up over several rather than freezing in one. */
const MOST_LOGGED = 120;
/** Seconds a take may fall behind while playing. Beyond, its frames wait for `endTake`, so a
 *  card too slow for the take spends itself on the live picture. */
const LIVE_BEHIND_S = 1;
/** The holder the page's sliders hold dials under, above knobs and pads. */
export const HAND = "console";
export const HAND_PRIORITY = 10;

/** A WebGPU device with the adapter's largest buffers, which a 3072x2048 layer needs. */
export async function gpu() {
  if (!navigator.gpu)
    throw new Error("This browser has no WebGPU. Recent Chrome, Edge, Safari and Firefox have it.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU found no graphics card to use.");
  const { maxBufferSize, maxStorageBufferBindingSize, maxStorageBuffersPerShaderStage } = adapter.limits;
  return adapter.requestDevice({ requiredLimits: { maxBufferSize, maxStorageBufferBindingSize, maxStorageBuffersPerShaderStage } });
}

/** How late this browser says work is done, in ms: at once in Chrome and Safari, on a 100 ms
 *  poll in Firefox. The least of a few empty submissions. */
async function reportLag(device) {
  let least = Infinity;
  for (let i = 0; i < 3; i++) {
    const t = performance.now();
    device.queue.submit([]);
    await device.queue.onSubmittedWorkDone();
    least = Math.min(least, performance.now() - t);
  }
  return least;
}

/** One model ready to play: its program (`base` + program.json) and weights, its dials, and
 *  its latent directions. `onStage` is told "download", "compile" and "draw" as each begins:
 *  the first frame is drawn here, since some drivers finish a pipeline only when it first
 *  runs (in Firefox, lichen's first frames took 300 ms each). */
export async function loadPlayable(device, base, name, { onProgress, onStage, format = navigator.gpu.getPreferredCanvasFormat() } = {}) {
  const get = async (file) => {
    let got;
    try {
      got = await fetch(`${base}${file}`);
    } catch (e) {
      throw new Error(`The model did not download (${e.message}). Check the connection and try again.`);
    }
    if (!got.ok) throw new Error(`The model did not download (${file}: HTTP ${got.status}).`);
    return got;
  };
  const program = await (await get("program.json")).json();
  if (!program.knobs || !program.screen) throw new Error(`${name}: this program predates the web player; convert it again`);
  const model = await loadModel(device, program, await get("weights.bin"), { onProgress, onStage });
  // The first frame, drawn as the canvas will draw it, so the screen's pipeline is ready too.
  onStage?.("draw");
  const onto = screen(device, model, format);
  const target = device.createTexture({ size: [64, 64], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const first = device.createCommandEncoder();
  model.encode(first);
  onto.draw(first, target);
  device.queue.submit([first.finish()]);
  await device.queue.onSubmittedWorkDone();
  target.destroy();
  const found = program.dials?.directions;
  return {
    name, program, model, nz: program.nz, width: program.width, height: program.height,
    layout: new Layout(program.knobs),
    live: program.live,
    settings: new Settings(program.settings),
    directions: (found?.basis ?? []).map((row) => Float32Array.from(row)),
    levels: found?.levels ?? [],
    onto,
  };
}

/** The performance: one clock, one kit, one preset runner and walk, over the models loaded. */
export class Player {
  /** `canvas` shows the picture; `positions` remembers each model's held dials across
   *  switches and visits (`{get(key), set(key, value)}`). */
  constructor(device, canvas, { fps = 60, bpm = 130, notes = null, channels = null, positions = null } = {}) {
    Object.assign(this, { device, canvas, fps, positions });
    this.clock = new MusicalClock(bpm);
    this.features = new NoteFeatures(12, channels, notes);
    this.models = [];
    this.index = -1;
    this.runner = null;
    this.walk = null;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context = canvas.getContext("webgpu");
    this.context.configure({ device, format: this.format, alphaMode: "opaque" });
    this.pending = 0;
    this.lag = 0;
    this.last = null;
    this.shown = null;
    /** A take, while recording (`take.mjs`). */
    this.tap = null;
    /** The take's frame last drawn on the screen. */
    this.onScreen = null;
    this.redraw = true;
    this.stats = { frames: 0, since: performance.now(), fps: 0, ms: 0 };
  }

  get current() {
    return this.models[this.index];
  }

  /** Which note (`kit.parseNotes`), or which channel (`parseTrackChannels`), is which drum. */
  setKit(notes, channels) {
    this.features = new NoteFeatures(12, channels, notes);
    if (!this.runner) return;
    this.runner.channelOf = this.features.channelOf();
    this.runner.load(this.runner.preset);
  }

  /** Add a loaded model (`loadPlayable`); the first one starts playing. */
  async add(entry) {
    this.models.push(entry);
    if (this.index < 0) {
      this.lag = await reportLag(this.device);
      this.runner = new PresetRunner(DEFAULT, this.features.channelOf(), this.fps, entry.layout, this.features.n);
      this.walk = new SlerpWalk(entry.nz, this.runner.walk);
      this.use(0);
    }
    return this.models.length - 1;
  }

  /** Play model `index`. The page's holds on the outgoing model's own dials are kept with it
   *  and those on the incoming one come back. */
  use(index) {
    const was = this.current;
    if (was) this.stash(was);
    this.index = index;
    const m = this.current;
    this.runner.useModel(m);
    this.runner.walk.directions = m.directions;
    this.walk.retarget(m.nz);
    this.recall(m);
    this.fit();
  }

  /** The canvas is shown `w` x `h` device pixels big: draw it at that many pixels, never
   *  more than the model draws. */
  show(w, h) {
    this.box = [w, h];
    this.fit();
  }

  fit() {
    const m = this.current;
    if (m && this.box) {
      const s = Math.min(1, this.box[0] / m.width, this.box[1] / m.height);
      this.canvas.width = Math.max(1, Math.round(m.width * s));
      this.canvas.height = Math.max(1, Math.round(m.height * s));
    }
    this.redraw = true;
  }

  stash(m) {
    if (!this.positions) return;
    const mine = new Set(perModel(m.layout));
    const held = Object.fromEntries(Object.entries(this.runner.heldBy(HAND)).filter(([n]) => mine.has(n)));
    this.positions.set(m.name, held);
    for (const name of Object.keys(held)) this.runner.free(HAND, name);
  }

  recall(m) {
    const back = this.positions?.get(m.name);
    if (!back) return;
    const mine = new Set(perModel(m.layout));
    const found = Object.fromEntries(Object.entries(back).filter(([n]) => mine.has(n)));
    if (Object.keys(found).length) this.runner.hold(HAND, { ...this.runner.heldBy(HAND), ...found }, HAND_PRIORITY);
  }

  /** The animation frame at `now` (ms, a requestAnimationFrame time). Returns whether it drew.
   *
   *  While recording, the take sets the time: the drums, the dials and the beat are worked out
   *  at each frame of the take, at that frame's own time (`take.next`), and kept in its log.
   *  The encoder draws them in order while the card keeps up, and the newest is the one shown,
   *  so on a card that keeps up each picture is drawn once. Frames the card had no time for
   *  wait, and `endTake` draws them, so a take has every frame, evenly spaced, on any card. */
  frame(now) {
    const m = this.current;
    if (!m) return false;
    const take = this.tap;
    if (!take) return this.step(m, now);
    for (let n = 0, at; n < MOST_LOGGED && (at = take.next(now)) !== null; n++) take.log(this.update(m, at));
    const room = DEPTH + Math.ceil((this.lag * take.fps) / 1000);
    let shown = false;
    if (take.behind <= LIVE_BEHIND_S * take.fps)
      for (let n = 0; n < CATCH_UP && take.behind && take.ready() && this.pending < room; n++) {
        const f = take.oldest();
        const newest = f.index === take.latest.index;
        this.draw(f, newest, take);
        if (newest) this.onScreen = take.latest;
        shown ||= newest;
      }
    if (!shown && take.latest && take.latest !== this.onScreen && this.pending < room) {
      this.draw(take.latest, true, null);
      shown = true;
    }
    return shown;
  }

  /** The drums, the dials and the beat at `now`, then the picture, unless the card is behind or
   *  nothing changed. */
  step(m, now) {
    const f = this.update(m, now);
    const changed = m.settings.changed;
    m.settings.changed = false;
    // The card is behind: skip rather than queue, which would hold up the page around it.
    if (this.pending >= DEPTH + Math.ceil(this.lag / Math.max(this.dt * 1000, 4))) return false;
    const same = !this.redraw && !changed && this.shown && this.shown.every((v, i) => v === f.latent[i]);
    if (same) return false;
    this.draw(f, true, null);
    return true;
  }

  /** The drums, the dials and the beat at `now`: the frame model `m` would draw then. */
  update(m, now) {
    this.dt = this.last === null ? 0 : (now - this.last) / 1000;
    this.last = now;
    const features = this.features;
    features.tick(now / 1000);
    this.runner.observe(features.drain());
    this.runner.apply(features.since, features.features(), m.settings);
    this.clock.advance(Math.min(this.dt, LONGEST_STEP_S));
    return { entry: m, latent: this.walk.latent(this.clock.beats), settings: m.settings.sent.slice() };
  }

  /** Draw frame `f`: onto the screen, into `take`, or both. */
  draw(f, screen, take) {
    const m = f.entry;
    m.model.setLatent(f.latent);
    m.model.setSettings(f.settings);
    const encoder = this.device.createCommandEncoder();
    m.model.encode(encoder);
    if (screen) {
      m.onto.draw(encoder, this.context.getCurrentTexture());
      this.redraw = false;
      this.shown = f.latent;
      this.onScreen = f;
    }
    take?.record(encoder, f);
    this.device.queue.submit([encoder.finish()]);
    take?.recorded();
    this.pending++;
    const t = performance.now();
    this.device.queue.onSubmittedWorkDone().then(() => {
      this.pending--;
      this.stats.ms = 0.9 * this.stats.ms + 0.1 * (performance.now() - t);
    });
    if (!screen) return;
    const s = this.stats;
    s.frames++;
    if (t - s.since > 1000) {
      s.fps = (s.frames * 1000) / (t - s.since);
      s.frames = 0;
      s.since = t;
    }
  }

  /** Stop recording: draw the frames the take is still waiting for, `onProgress(fraction)` as
   *  it goes, then the video as a Blob. The picture plays on meanwhile. */
  async endTake(onProgress) {
    const take = this.tap;
    this.tap = null;
    const total = take.behind;
    while (take.behind && !take.failed) {
      if (!take.ready() || this.pending >= DEPTH + Math.ceil((this.lag * take.fps) / 1000)) {
        await new Promise((r) => setTimeout(r, 4));
        continue;
      }
      this.draw(take.oldest(), false, take);
      onProgress?.(1 - take.behind / total);
    }
    return take.stop();
  }

  /** The picture now, as a PNG at the model's own size. */
  async picture() {
    const m = this.current;
    const size = m.width * m.height * 4;
    const read = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(m.model.output, 0, read, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const pixels = new ImageData(new Uint8ClampedArray(read.getMappedRange().slice(0)), m.width, m.height);
    read.destroy();
    const still = new OffscreenCanvas(m.width, m.height);
    still.getContext("2d").putImageData(pixels, 0, 0);
    return still.convertToBlob({ type: "image/png" });
  }

  stop() {
    this.context.unconfigure();
    for (const m of this.models) m.model.destroy();
    this.models = [];
    this.index = -1;
  }
}
