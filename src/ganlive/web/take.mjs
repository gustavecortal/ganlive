// A take in a browser, in two steps. While playing, `Recording` logs what each frame of the take
// draws: the ends of the walk's segment and where between them the frame is, the direction
// pushes and the model's settings, a few bytes a frame, kept in the browser (IndexedDB) as it
// goes, so a take of hours costs the card nothing and survives a closed tab. After Stop,
// `renderTake` draws every logged frame again, exactly, at the take's size and as fast as the
// card goes: a compute shader turns each into I420 (the layout every WebCodecs encoder takes), a
// ring of mapped buffers brings it back, and WebCodecs encodes it, H.264 where the browser
// offers it, else VP9, at a constant quality where the encoder honours one, into an MP4 written
// on disk as it comes (the site's private storage). The desktop's takes are `record/video.py`.

import { screen } from "./runner.mjs";
import { compose, stylePush } from "./walk.mjs";

/** Frames a second of every take: drawn after the performance, so any card records them all. */
export const TAKE_FPS = 60;
/** A bitrate, in bits per pixel of each frame, where an encoder takes no quantizer or ignores
 *  it (Firefox): about 16 Mbit/s for 1620x1080 at 60 fps, above the 12 YouTube asks of 1080p60. */
const BITS_PER_PIXEL = 0.15;
/** A constant quality, where the encoder honours it. H.264 at 23 keeps lichen's fastest motion
 *  as close to the frames as 0.15 bits per pixel does (about 47 dB), and spends a third of that
 *  on slow motion, which is most of a set. */
const QUANTIZER = { avc: 23, vp9: 26 };
/** Frames the encoder may hold before the render waits for it. */
const QUEUE = 4;
/** Frames on their way back from the GPU at most. A read takes 15 to 35 ms in Edge and Firefox. */
const RING = 6;
/** What the I420 shader writes, as each frame tells the encoder. The MP4 says it too (`colr`). */
const BT709 = { primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false };
/** Seconds between key frames. */
const KEY_EVERY_S = 2;
/** Frames worked out in one animation frame at most: a tab hidden for a while catches up over
 *  several rather than freezing in one. */
export const MOST_LOGGED = 120;
/** Bytes of log kept in memory before they go to the browser's storage. */
const CHUNK = 16384;

/** The encoders worth trying, best first. A level fits the frame size and rate. */
const CODECS = [
  ["avc", "avc1.640034"], ["avc", "avc1.640033"], ["avc", "avc1.64002a"], ["avc", "avc1.4d0034"],
  ["vp9", "vp09.00.51.08"], ["vp9", "vp09.00.41.08"],
];

/** The take's frame (an rgba8unorm texture, drawn at its size) -> I420: the Y plane, then U and
 *  V at half resolution. The colour maths of `engine/screen.py`'s NV12 (BT.709, limited range),
 *  which the desktop's takes use. One thread a word of four bytes. */
const I420 = `
@group(0) @binding(0) var X: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<uniform> U: vec4u;      // h, w
const KR = 0.2126; const KB = 0.0722; const KG = 1.0 - 0.2126 - 0.0722;
fn rgb(y: u32, x: u32) -> vec3f {           // [-1, 1], as the generator drew it
  return textureLoad(X, vec2u(x, y), 0).rgb * 2.0 - 1.0;
}
fn byte(b: u32) -> u32 {
  let h = U[0]; let w = U[1];
  if (b < h * w) {
    let c = rgb(b / w, b % w);
    return u32(clamp(round((KR * c.r + KG * c.g + KB * c.b) * 109.5 + 125.5), 0.0, 255.0));
  }
  let quarter = (h / 2u) * (w / 2u);
  let i = (b - h * w) % quarter; let row = i / (w / 2u); let x = (i % (w / 2u)) * 2u;
  let c = (rgb(2u * row, x) + rgb(2u * row, x + 1u) + rgb(2u * row + 1u, x) + rgb(2u * row + 1u, x + 1u)) / 4.0;
  let l = KR * c.r + KG * c.g + KB * c.b;
  let v = select((c.b - l) * (224.0 / (4.0 * (1.0 - KB))), (c.r - l) * (224.0 / (4.0 * (1.0 - KR))), b - h * w >= quarter);
  return u32(clamp(round(v + 128.0), 0.0, 255.0));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let total = U[0] * U[1] * 3u / 2u;
  let word = id.y * 16776960u + id.x;
  if (word * 4u >= total) { return; }
  var out = 0u;
  for (var j = 0u; j < 4u; j++) {
    if (word * 4u + j < total) { out |= byte(word * 4u + j) << (8u * j); }
  }
  Y[word] = out;
}`;

/** The size a take of a `width` x `height` model has on a `screen` of device pixels: the
 *  model's shape, as large as the screen holds and never larger than the model, each side
 *  even (I420 halves them). The desktop's takes follow its window the same way. */
export function takeSize(width, height, screen) {
  const s = Math.min(1, screen[0] / width, screen[1] / height);
  const even = (n) => Math.max(2, 2 * Math.floor((n * s) / 2));
  return [even(width), even(height)];
}


/** What this browser answered for each configuration asked about. */
const answers = new Map();
const supported = (config) => {
  const key = JSON.stringify(config);
  if (!answers.has(key))
    answers.set(key, VideoEncoder.isConfigSupported(config).then((r) => r.supported, () => false));
  return answers.get(key);
};

/** Whether an encoder that takes a quantizer acts on it: Firefox's accepts one and encodes at
 *  its own bitrate whatever the value. Two small frames of noise, at a fine and a coarse one. */
const honours = new Map();
function honoured(kind, config) {
  const key = `${config.codec} ${config.hardwareAcceleration}`;
  if (!honours.has(key))
    honours.set(key, (async () => {
      const size = 256;
      const small = { ...config, width: size, height: size };
      if (!(await supported(small))) return false;
      const noise = Uint8Array.from({ length: (size * size * 3) / 2 }, (_, i) => (i * 2654435761) >>> 24);
      const bytes = [];
      for (const quantizer of [10, 40]) {
        let total = 0;
        const encoder = new VideoEncoder({ output: (chunk) => (total += chunk.byteLength), error: () => {} });
        encoder.configure(small);
        for (let i = 0; i < 2; i++) {
          const frame = new VideoFrame(noise, { format: "I420", codedWidth: size, codedHeight: size, timestamp: i * 16667 });
          encoder.encode(frame, { keyFrame: i === 0, [kind]: { quantizer } });
          frame.close();
        }
        await encoder.flush().catch(() => {});
        encoder.close();
        bytes.push(total);
      }
      return bytes[0] > 1.5 * bytes[1];
    })().catch(() => false));
  return honours.get(key);
}

/** The encoder configurations this browser takes for a `width` x `height` take at `fps`, best
 *  first: hardware before software, a constant quality before a bitrate. Each is asked about
 *  only when the one before is not enough. */
async function* configure(width, height, fps) {
  if (typeof VideoEncoder === "undefined") return;
  for (const hardwareAcceleration of ["prefer-hardware", "no-preference"])
    for (const [kind, codec] of CODECS) {
      const base = { codec, width, height, framerate: fps, hardwareAcceleration, latencyMode: "quality",
        ...(kind === "avc" ? { avc: { format: "avc" } } : {}) };
      const steady = { ...base, bitrateMode: "quantizer" };
      if ((await supported(steady)) && (await honoured(kind, steady))) yield { kind, config: steady, quantizer: QUANTIZER[kind] };
      const variable = { ...base, bitrateMode: "variable", bitrate: Math.round(BITS_PER_PIXEL * width * height * fps) };
      if (await supported(variable)) yield { kind, config: variable };
    }
}

/** Whether this browser can save a take as video at all. */
export const canRecord = () => typeof VideoEncoder !== "undefined";

// -- the log ---------------------------------------------------------------------------------

/** A log's records, each a tag byte and its fields, little-endian: the model the frames after it
 *  draw (its name, as JSON), a new segment of the walk (its end, and its start unless that is the
 *  last segment's end), and a frame (where between the two ends it is, then the direction pushes
 *  and the model's settings if they changed). */
const MODEL = 1;
const SEGMENT = 2;
const FRAME = 3;
const PUSHES = 1;
const SETTINGS = 2;

/** The takes not yet saved as video, and their logs, in IndexedDB: a closed tab or a lost
 *  graphics card leaves a take that can still be saved. */
let opened = null;
function store() {
  opened ??= new Promise((resolve, reject) => {
    const request = indexedDB.open("ganlive-takes", 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("takes", { keyPath: "id" });
      request.result.createObjectStore("log");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return opened;
}

/** Run `work` in one transaction; resolves to what its request found, once committed. */
async function transact(mode, work) {
  const db = await store();
  return new Promise((resolve, reject) => {
    const t = db.transaction(["takes", "log"], mode);
    const request = work(t.objectStore("takes"), t.objectStore("log"));
    t.oncomplete = () => resolve(request?.result);
    t.onerror = t.onabort = () => reject(t.error);
  });
}

const chunksOf = (id) => IDBKeyRange.bound([id, 0], [id, Infinity]);

/** The takes kept in this browser and not yet saved as video, oldest first. */
export async function keptTakes() {
  try {
    const takes = await transact("readonly", (takes) => takes.getAll());
    return takes.filter((t) => t.frames > 0).sort((a, b) => a.id - b.id);
  } catch {
    return [];
  }
}

/** Forget a kept take and its log. */
export const forgetTake = (id) =>
  transact("readwrite", (takes, log) => {
    takes.delete(id);
    log.delete(chunksOf(id));
  }).catch(() => {});

/** A take as it plays. While it is the player's `tap`, the player works out each of its frames at
 *  that frame's own time (`next`) and logs it (`log`). `hit(seconds)` is told when each drum hit
 *  arrives; `stop` resolves to the take, ready for `renderTake`. */
export class Recording {
  constructor() {
    this.fps = TAKE_FPS;
    this.id = Date.now();
    this.started = new Date().toISOString();
    this.start = null;
    this.slot = -1;
    this.frames = 0;
    /** The first and the last drum hit, in seconds of the video, for lining up its audio. */
    this.hits = null;
    this.models = [];
    this.bytes = new Uint8Array(2 * CHUNK);
    this.used = 0;
    /** The whole log, in memory for the render after Stop, and in IndexedDB in case it never comes. */
    this.chunks = [];
    this.saving = Promise.resolve();
    this.flushed = 0;
  }
  /** Seconds of video recorded so far. */
  get seconds() {
    return this.frames / this.fps;
  }
  /** The time (ms, as the player's) of the take's next frame, if it is due by the animation
   *  frame at `now`: up to half a frame early, so each animation frame of a screen at the
   *  take's rate works out one. The take starts at the first `now`. */
  next(now) {
    this.start ??= now;
    const at = this.start + ((this.slot + 1) * 1000) / this.fps;
    if (at > now + 500 / this.fps) return null;
    this.slot++;
    return at;
  }
  /** A drum hit arrived at `seconds` (`performance.now() / 1000`, as MIDI times are). */
  hit(seconds) {
    if (this.start === null) return;
    const at = seconds - this.start / 1000;
    if (at < 0) return;
    this.hits ??= { first: at, last: at };
    this.hits.last = at;
  }
  /** Log the frame `entry` draws now: the walk's latent and the model's settings as they stand. */
  log(entry, walk, settings) {
    if (entry !== this.entry) {
      this.entry = entry;
      if (!this.models.some((m) => m.name === entry.name))
        this.models.push({ name: entry.name, width: entry.width, height: entry.height });
      const name = new TextEncoder().encode(JSON.stringify({ name: entry.name }));
      this.reserve(3 + name.length);
      this.u8(MODEL);
      this.view.setUint16(this.used, name.length, true);
      this.used += 2;
      this.bytes.set(name, this.used);
      this.used += name.length;
      this.z1 = this.pushes = this.settings = null;
    }
    if (walk.z1 !== this.z1) {
      const both = walk.z0 !== this.z1;
      this.reserve(4 + 4 * walk.z1.length * (both ? 2 : 1));
      this.u8(SEGMENT);
      this.u8(both ? 1 : 0);
      this.view.setUint16(this.used, walk.z1.length, true);
      this.used += 2;
      if (both) this.floats(walk.z0, 32);
      this.floats(walk.z1, 32);
      this.z1 = walk.z1;
    }
    const pushes = walk.cfg.amounts;
    const k = settings.sent;
    const flags = (differ(pushes, this.pushes) ? PUSHES : 0) | (differ(k, this.settings) ? SETTINGS : 0);
    this.reserve(20 + 8 * pushes.length + 4 * k.length);
    this.u8(FRAME);
    this.u8(flags);
    this.floats(walk.mixed, 64);
    if (flags & PUSHES) {
      this.u8(pushes.length);
      this.floats(pushes, 64);
      this.pushes = pushes.slice();
    }
    if (flags & SETTINGS) {
      this.u8(k.length);
      this.floats(k, 32);
      this.settings = k.slice();
    }
    this.frames++;
    if (this.used >= CHUNK || this.frames - this.flushed >= 2 * this.fps) this.flush();
  }
  reserve(n) {
    if (this.used + n > this.bytes.length) {
      const grown = new Uint8Array(2 * (this.used + n));
      grown.set(this.bytes.subarray(0, this.used));
      this.bytes = grown;
    }
    if (this.view?.buffer !== this.bytes.buffer) this.view = new DataView(this.bytes.buffer);
  }
  u8(v) {
    this.bytes[this.used++] = v;
  }
  floats(values, bits) {
    for (const v of values) {
      if (bits === 32) this.view.setFloat32(this.used, v, true);
      else this.view.setFloat64(this.used, v, true);
      this.used += bits / 8;
    }
  }
  /** What the log holds so far, into IndexedDB, with what is known of the take. */
  flush() {
    this.flushed = this.frames;
    if (!this.used) return this.saving;
    const data = this.bytes.slice(0, this.used);
    this.used = 0;
    const n = this.chunks.push(data) - 1;
    const about = this.about();
    this.saving = this.saving.then(() => transact("readwrite", (takes, log) => {
      log.put(data.buffer, [this.id, n]);
      takes.put(about);
    })).catch(() => {});    // a browser without room or IndexedDB still saves the take after Stop
    return this.saving;
  }
  about() {
    const { id, started, fps, frames, hits, models } = this;
    return { id, started, fps, frames, hits: hits && { ...hits }, models: models.map((m) => ({ ...m })) };
  }
  /** The take, its log whole: what `renderTake` draws. */
  async stop() {
    await this.flush();
    return { ...this.about(), log: concat(this.chunks) };
  }
}

const differ = (a, b) => !b || a.length !== b.length || a.some((v, i) => v !== b[i]);

/** A kept take's log, from IndexedDB. */
async function storedLog(id) {
  const chunks = await transact("readonly", (_, log) => log.getAll(chunksOf(id)));
  return concat(chunks.map((c) => new Uint8Array(c)));
}

/** What a log holds, in order: `{model}`, the name of the model the frames after it draw, and
 *  each frame, `{z0, a, z1, b, pushes, settings}`: its latent is `compose(z0, a, z1, b,
 *  directions, pushes)`, and its settings are the last ones given since its model. */
export function* readLog(log) {
  const view = new DataView(log.buffer, log.byteOffset, log.byteLength);
  const f32 = (at, n) => new Float32Array(log.slice(at, at + 4 * n).buffer);
  let z0, z1, pushes = [], settings = null;
  for (let at = 0; at < log.length;) {
    const tag = log[at++];
    if (tag === MODEL) {
      const n = view.getUint16(at, true);
      const { name } = JSON.parse(new TextDecoder().decode(log.subarray(at + 2, at + 2 + n)));
      at += 2 + n;
      settings = null;
      yield { model: name };
    } else if (tag === SEGMENT) {
      const both = log[at];
      const n = view.getUint16(at + 1, true);
      at += 3;
      if (both) {
        z0 = f32(at, n);
        at += 4 * n;
      } else z0 = z1;
      z1 = f32(at, n);
      at += 4 * n;
    } else if (tag === FRAME) {
      const flags = log[at];
      const a = view.getFloat64(at + 1, true);
      const b = view.getFloat64(at + 9, true);
      at += 17;
      if (flags & PUSHES) {
        const n = log[at++];
        pushes = Array.from({ length: n }, (_, i) => view.getFloat64(at + 8 * i, true));
        at += 8 * n;
      }
      if (flags & SETTINGS) {
        const n = log[at++];
        settings = f32(at, n);
        at += 4 * n;
      }
      yield { z0, a, z1, b, pushes, settings };
    } else throw new Error(`the take's log is damaged at byte ${at - 1}`);
  }
}

// -- drawing a take --------------------------------------------------------------------------

/** Draw `take` (from `Recording.stop` or `keptTakes`) as an MP4 of `size` (`takeSize`), every
 *  frame at its own time. `model(name)` gives a loaded model (`loadPlayable`). `onProgress(done)`
 *  is told the share of frames encoded. The frames are drawn on the card the page plays on: a
 *  paused picture leaves the render the whole card. */
export async function renderTake(device, take, { size, model, onProgress = () => {} }) {
  const log = take.log ?? (await storedLog(take.id));
  let refused = null;
  for await (const candidate of configure(size[0], size[1], take.fps)) {
    const file = await openFile(BITS_PER_PIXEL * size[0] * size[1] * take.frames / 8);
    try {
      return await draw(device, take, log, candidate, file, size, model, onProgress);
    } catch (e) {
      await file.discard();
      // An encoder can claim a configuration and then refuse its first frame (Firefox did with
      // NV12): the next candidate draws the take from the start.
      if (!e.refused) throw e;
      refused = e;
    }
  }
  throw refused ?? new Error(`this browser has no video encoder for ${size[0]}×${size[1]}`);
}

async function draw(device, take, log, { kind, config, quantizer }, file, [width, height], model, onProgress) {
  const fps = take.fps;
  const format = "rgba8unorm";
  const size = (width * height * 3) / 2;
  const bytes = Math.ceil(size / 4) * 4;
  /** The frame at the take's size, drawn by the screen's own shader (engine/fit.py), which fits a
   *  model of another shape inside it. */
  const picture = device.createTexture({ size: [width, height], format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const out = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const shape = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(shape, 0, new Uint32Array([height, width, 0, 0]));
  const module = device.createShaderModule({ code: I420 });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: picture.createView() }, { binding: 1, resource: { buffer: out } },
    { binding: 2, resource: { buffer: shape } }] });
  const groups = Math.ceil(bytes / 4 / 256);
  const ring = Array.from({ length: RING }, () => ({
    buffer: device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    free: Promise.resolve(),
  }));
  const mux = new Mp4(kind, width, height, file);
  let failed = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => mux.add(chunk, meta),
    error: (e) => (failed ??= Object.assign(e, { refused: !mux.samples.length })),
  });
  encoder.configure(config);
  const dequeued = () => new Promise((resolve) => {
    encoder.addEventListener("dequeue", resolve, { once: true });
    setTimeout(resolve, 20);
  });
  // Frames reach the encoder in order through one chain of promises.
  let chain = Promise.resolve();
  let encoded = 0;
  const keyEvery = Math.round(KEY_EVERY_S * fps);
  const screens = new Map();
  let entry, onto, frame = 0;
  try {
    for (const item of readLog(log)) {
      if (item.model) {
        entry = await model(item.model);
        if (!screens.has(entry)) screens.set(entry, screen(device, entry.model, format));
        onto = screens.get(entry);
      } else {
        const { z0, a, z1, b, pushes, settings } = item;
        if (failed) throw failed;
        const slot = ring[frame % RING];
        await slot.free;
        entry.model.setLatent(compose(z0, a, z1, b, entry.directions, pushes));
        const width = entry.model.pushWidth;
        if (width) entry.model.setPush(stylePush(entry.directions, pushes, width));
        // The model's buffers are the page's too: its own settings go back after the frame.
        if (settings) entry.model.setSettings(settings);
        const commands = device.createCommandEncoder();
        entry.model.encode(commands);
        onto.draw(commands, picture);
        const pass = commands.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
        pass.end();
        commands.copyBufferToBuffer(out, 0, slot.buffer, 0, bytes);
        device.queue.submit([commands.finish()]);
        entry.model.setSettings(entry.settings.sent);
        const mapped = slot.buffer.mapAsync(GPUMapMode.READ);
        const index = frame++;
        chain = chain.then(() => mapped).then(async () => {
          while (!failed && encoder.encodeQueueSize >= QUEUE) await dequeued();
          if (!failed) {
            const image = new VideoFrame(new Uint8Array(slot.buffer.getMappedRange(), 0, size), {
              format: "I420", codedWidth: width, codedHeight: height, colorSpace: BT709,
              timestamp: Math.round((index * 1e6) / fps), duration: Math.round(1e6 / fps),
            });
            encoder.encode(image, { keyFrame: index % keyEvery === 0, ...(quantizer ? { [kind]: { quantizer } } : {}) });
            image.close();
            if (++encoded % 30 === 0) onProgress(encoded / take.frames);
          }
          slot.buffer.unmap();
        }).catch((e) => (failed ??= e));
        slot.free = chain;
      }
    }
    await chain;
    if (!failed) await encoder.flush().catch((e) => (failed ??= e));
    if (failed) throw failed;
    onProgress(1);
  } finally {
    // Each model's latent, push and picture still hold the take's last frame. Its settings,
    // marked unsent, make the player draw the live frame again, even while paused, so the
    // picture it saves is the one on screen.
    for (const used of screens.keys()) used.settings.changed = true;
    await chain;
    if (encoder.state !== "closed") encoder.close();
    for (const b of [out, shape, ...ring.map((r) => r.buffer)]) b.destroy();
    picture.destroy();
  }
  return mux.finish(fps);
}

// -- where a take's file goes ---------------------------------------------------------------

/** Takes on disk are named so, in the site's private storage (OPFS). */
const ON_DISK = "ganlive-take-";
let visited = false;

/** A file for the next take: on disk where the browser allows it, else in memory. A saved take
 *  is the user's own copy, so the site's are removed: all of an earlier visit's, and of this
 *  visit's all but the last, whose download may still be running. */
async function openFile(need) {
  try {
    const dir = await navigator.storage.getDirectory();
    const old = [];
    for await (const name of dir.keys()) if (name.startsWith(ON_DISK)) old.push(name);
    old.sort();
    for (const name of visited ? old.slice(0, -1) : old) await dir.removeEntry(name).catch(() => {});
    visited = true;
    const name = `${ON_DISK}${Date.now()}.mp4`;
    const handle = await dir.getFileHandle(name, { create: true });
    const { quota = Infinity, usage = 0 } = await navigator.storage.estimate().catch(() => ({}));
    // Firefox keeps a site to 10 GB at most unless the site asks to keep its files.
    if (quota - usage < need) await navigator.storage.persist?.().catch(() => {});
    return new DiskFile(dir, name, handle, await handle.createWritable());
  } catch {
    return new MemoryFile();     // no OPFS, or no createWritable (Safari before 26)
  }
}

/** A file written in order (`append`), with bytes patched in place (`patch`) before `close`. */
class MemoryFile {
  constructor() {
    this.parts = [];
    this.batch = [];
    this.batchBytes = 0;
    this.size = 0;
  }
  append(bytes) {
    this.batch.push(bytes);
    this.batchBytes += bytes.length;
    this.size += bytes.length;
    // A Blob a few tens of MB at a time, which a browser may keep on disk rather than in memory.
    if (this.batchBytes > 32e6) this.flush();
  }
  flush() {
    if (this.batch.length) this.parts.push(new Blob(this.batch));
    this.batch = [];
    this.batchBytes = 0;
  }
  /** Only the first bytes appended can be patched, which is all an MP4's header needs. */
  patch(position, bytes) {
    this.flush();
    this.head ??= this.parts.shift();
    this.patches = [...(this.patches ?? []), [position, bytes]];
  }
  async close(type) {
    this.flush();
    const head = new Uint8Array(await this.head.arrayBuffer());
    for (const [position, bytes] of this.patches ?? []) head.set(bytes, position);
    return new Blob([head, ...this.parts], { type });
  }
  discard() {
    this.parts = [];
  }
}

class DiskFile {
  constructor(dir, name, handle, writable) {
    Object.assign(this, { dir, name, handle, writable });
    this.size = 0;
    /** Writes in the order they were asked for; the first failure (a full disk) is kept. */
    this.chain = Promise.resolve();
    this.failed = null;
  }
  append(bytes) {
    const position = this.size;
    this.size += bytes.length;
    this.write({ type: "write", position, data: bytes });
  }
  patch(position, bytes) {
    this.write({ type: "write", position, data: bytes });
  }
  write(command) {
    this.chain = this.chain.then(() => this.failed || this.writable.write(command)).catch((e) => (this.failed ??= e));
  }
  async close(type) {
    await this.chain;
    await this.writable.close().catch((e) => (this.failed ??= e));
    if (this.failed) {
      await this.dir.removeEntry(this.name).catch(() => {});
      throw this.failed;
    }
    // A File on disk: the download reads it from there. Its type is set by the name.
    return this.handle.getFile();
  }
  async discard() {
    await this.chain;
    await this.writable.abort().catch(() => {});
    await this.dir.removeEntry(this.name).catch(() => {});
  }
}

// -- MP4 -------------------------------------------------------------------------------------

const ascii = (s) => Uint8Array.from(s, (c) => c.charCodeAt(0));

/** Big-endian fields: `[bits, value]` pairs, bits 8, 16, 32 or 64. */
function fields(...pairs) {
  const size = pairs.reduce((a, [bits]) => a + bits / 8, 0);
  const out = new DataView(new ArrayBuffer(size));
  let at = 0;
  for (const [bits, value] of pairs) {
    if (bits === 8) out.setUint8(at, value);
    else if (bits === 16) out.setUint16(at, value);
    else if (bits === 32) out.setUint32(at, value);
    else out.setBigUint64(at, BigInt(value));
    at += bits / 8;
  }
  return new Uint8Array(out.buffer);
}

/** Big-endian words of `bits` (32 or 64), one per value: a sample table, however long the take. */
function table(bits, values) {
  const out = new DataView(new ArrayBuffer((values.length * bits) / 8));
  values.forEach((v, i) => (bits === 32 ? out.setUint32(i * 4, v) : out.setBigUint64(i * 8, BigInt(v))));
  return new Uint8Array(out.buffer);
}

/** `values` as `[count, value]` runs of equal neighbours, flattened. */
function runs(values) {
  const out = [];
  for (const v of values) {
    if (out.length && out[out.length - 1] === v) out[out.length - 2]++;
    else out.push(1, v);
  }
  return out;
}

function concat(parts) {
  const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

const box = (type, ...parts) => {
  const body = concat(parts);
  return concat([fields([32, body.length + 8]), ascii(type), body]);
};
const full = (type, version, flags, ...parts) => box(type, fields([8, version], [8, flags >> 16], [16, flags & 0xffff]), ...parts);
const MATRIX = fields([32, 0x10000], [32, 0], [32, 0], [32, 0], [32, 0x10000], [32, 0], [32, 0], [32, 0], [32, 0x40000000]);

/** Bytes before the first sample: `ftyp` (32), then the `mdat` header with a 64-bit size (16),
 *  written as zeros and filled in by `finish`, when the codec and the size are known. */
const HEAD = 48;

/** An MP4 of one video track, written onto `file` as the encoder hands out its chunks: the
 *  samples, then the index (`moov`) after them, then the header patched in. */
class Mp4 {
  constructor(kind, width, height, file) {
    Object.assign(this, { kind, width, height, file });
    this.samples = [];        // [size, timestamp in us, key]
    this.description = null;
    if (!file.size) file.append(new Uint8Array(HEAD));
  }
  add(chunk, meta) {
    if (meta?.decoderConfig?.description) this.description = new Uint8Array(meta.decoderConfig.description);
    const data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);
    this.samples.push([data.length, chunk.timestamp, chunk.type === "key"]);
    this.file.append(data);
  }
  sampleEntry() {
    const visual = concat([fields([32, 0], [16, 0], [16, 1], [16, 0], [16, 0], [32, 0], [32, 0], [32, 0],
      [16, this.width], [16, this.height], [32, 0x480000], [32, 0x480000], [32, 0], [16, 1]),
      new Uint8Array(32), fields([16, 0x18], [16, 0xffff])]);
    // BT.709 primaries, transfer and matrix, limited range: the encoders leave the stream unmarked.
    const colr = box("colr", ascii("nclx"), fields([16, 1], [16, 1], [16, 1], [8, 0]));
    if (this.kind === "avc") return box("avc1", visual, box("avcC", this.description), colr);
    // VP9 profile 0, 8-bit 4:2:0, BT.709.
    const vpcC = full("vpcC", 1, 0, fields([8, 0], [8, 51], [8, 0x82], [8, 1], [8, 1], [8, 1], [16, 0]));
    return box("vp09", visual, vpcC);
  }
  /** The finished file. An encoder may reorder frames (B-frames): chunks come out in decoding
   *  order with presentation timestamps, so decoding times are the timestamps sorted, each
   *  sample's offset to its presentation is `ctts`, signed (version 1), so the first frame is
   *  presented at 0 with no edit list, which YouTube asks of an upload. */
  async finish(fps) {
    const n = this.samples.length;
    const timescale = 90000;
    const first = this.samples.reduce((a, [, t]) => Math.min(a, t), Infinity);
    const shown = this.samples.map(([, t]) => Math.round(((t - first) * timescale) / 1e6));
    const decoded = [...shown].sort((a, b) => a - b);
    const offsets = shown.map((t, i) => t - decoded[i]);
    const reordered = offsets.some((o) => o !== 0);
    const durations = decoded.map((t, i) => (i + 1 < n ? decoded[i + 1] - t : Math.round(timescale / fps)));
    const duration = durations.reduce((a, d) => a + d, 0);
    const stts = runs(durations);
    const ctts = runs(offsets);
    const keys = this.samples.map(([, , key], i) => (key ? i + 1 : 0)).filter(Boolean);
    const ftyp = box("ftyp", ascii("isom"), fields([32, 512]), ascii("isomiso2mp41"), ascii(this.kind === "avc" ? "avc1" : "vp09"));
    const mdatSize = 16 + this.samples.reduce((a, [size]) => a + size, 0);
    const mdatHead = concat([fields([32, 1]), ascii("mdat"), fields([64, mdatSize])]);
    let offset = HEAD;
    const placed = this.samples.map(([size]) => {
      const at = offset;
      offset += size;
      return at;
    });
    const stbl = box("stbl",
      full("stsd", 0, 0, fields([32, 1]), this.sampleEntry()),
      full("stts", 0, 0, fields([32, stts.length / 2]), table(32, stts)),
      // Negative offsets go in as their 32-bit two's complement, which version 1 reads signed.
      ...(reordered ? [full("ctts", 1, 0, fields([32, ctts.length / 2]), table(32, ctts))] : []),
      full("stss", 0, 0, fields([32, keys.length]), table(32, keys)),
      full("stsc", 0, 0, fields([32, 1], [32, 1], [32, 1], [32, 1])),
      full("stsz", 0, 0, fields([32, 0], [32, n]), table(32, this.samples.map(([size]) => size))),
      full("co64", 0, 0, fields([32, n]), table(64, placed)));
    const minf = box("minf", full("vmhd", 0, 1, fields([16, 0], [16, 0], [16, 0], [16, 0])),
      box("dinf", full("dref", 0, 0, fields([32, 1]), full("url ", 0, 1))), stbl);
    const mdia = box("mdia",
      full("mdhd", 0, 0, fields([32, 0], [32, 0], [32, timescale], [32, duration], [16, 0x55c4], [16, 0])),
      full("hdlr", 0, 0, fields([32, 0]), ascii("vide"), fields([32, 0], [32, 0], [32, 0]), ascii("ganlive\0")),
      minf);
    const tkhd = full("tkhd", 0, 3, fields([32, 0], [32, 0], [32, 1], [32, 0], [32, duration], [32, 0], [32, 0],
      [16, 0], [16, 0], [16, 0], [16, 0]), MATRIX, fields([32, this.width * 0x10000], [32, this.height * 0x10000]));
    const mvhd = full("mvhd", 0, 0, fields([32, 0], [32, 0], [32, timescale], [32, duration], [32, 0x10000], [16, 0x100],
      [16, 0], [32, 0], [32, 0]), MATRIX, new Uint8Array(24), fields([32, 2]));
    const moov = box("moov", mvhd, box("trak", tkhd, mdia));
    this.file.append(moov);
    this.file.patch(0, concat([ftyp, mdatHead]));
    return this.file.close("video/mp4");
  }
}
