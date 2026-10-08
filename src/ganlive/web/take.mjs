// A take in a browser: the picture at the model's own size, every frame, as an MP4. A compute
// shader turns each frame into I420 (the layout every WebCodecs encoder takes), a ring of
// mapped buffers brings it back without stalling the page, and WebCodecs encodes it: the GPU's
// H.264 encoder at a high bitrate where the browser offers one, else VP9, written into an MP4
// here. A browser without WebCodecs records a canvas with MediaRecorder instead. The desktop's
// takes are `record/video.py`.

import { screen } from "./runner.mjs";

/** A bitrate, in bits per pixel of each frame: about 57 Mbit/s for lichen at 60 fps. Every
 *  encoder honours one; Firefox takes a quantizer and ignores it (7 Mbit/s for lichen). */
const BITS_PER_PIXEL = 0.15;
/** Where only a constant quality is offered, its quantizer: about the desktop's CRF 18. */
const QUANTIZER = { avc: 20, vp9: 26 };
/** Frames the encoder may hold: the next waits (the player draws it at the next animation frame). */
const QUEUE = 4;
/** Frames on their way back from the GPU at most. A read takes 15 to 35 ms in Edge and Firefox. */
const RING = 6;
/** What the I420 shader writes, as each frame tells the encoder. The MP4 says it too (`colr`). */
const BT709 = { primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false };
/** Seconds between key frames. */
const KEY_EVERY_S = 2;

/** The encoders worth trying, best first. A level fits the frame size and rate. */
const CODECS = [
  ["avc", "avc1.640034"], ["avc", "avc1.640033"], ["avc", "avc1.64002a"], ["avc", "avc1.4d0034"],
  ["vp9", "vp09.00.51.08"], ["vp9", "vp09.00.41.08"],
];

/** The model's packed frame -> I420: the Y plane, then U and V at half resolution. The colour
 *  maths of `engine/screen.py`'s NV12 (BT.709, limited range), which the desktop's takes use.
 *  One thread a word of four bytes. `order` reads the frame's channels as RGB. */
const I420 = (order) => `
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<uniform> U: vec4u;      // h, w
const KR = 0.2126; const KB = 0.0722; const KG = 1.0 - 0.2126 - 0.0722;
fn rgb(y: u32, x: u32) -> vec3f {           // [-1, 1], as the generator drew it
  return unpack4x8unorm(X[y * U[1] + x]).${order} * 2.0 - 1.0;
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

/** What this browser answered for each configuration asked about. Edge takes 1.3 s over its
 *  first answer, so the page asks ahead (`prepareTake`) and Record starts at once. */
const answers = new Map();
const supported = (config) => {
  const key = JSON.stringify(config);
  if (!answers.has(key))
    answers.set(key, VideoEncoder.isConfigSupported(config).then((r) => r.supported, () => false));
  return answers.get(key);
};

/** The encoder configurations this browser says it takes for a `width` x `height` take at
 *  `fps`, best first: hardware before software, a bitrate before a constant quality. Each is
 *  asked about only when the one before is not enough. */
async function* configure(width, height, fps) {
  if (typeof VideoEncoder === "undefined") return;
  for (const hardwareAcceleration of ["prefer-hardware", "no-preference"])
    for (const [kind, codec] of CODECS)
      for (const bitrateMode of ["variable", "quantizer"]) {
        const config = {
          codec, width, height, framerate: fps, hardwareAcceleration, bitrateMode, latencyMode: "quality",
          ...(bitrateMode === "variable" ? { bitrate: Math.round(BITS_PER_PIXEL * width * height * fps) } : {}),
          ...(kind === "avc" ? { avc: { format: "avc" } } : {}),
        };
        // A configuration this browser cannot parse is one it does not support.
        if (await supported(config)) yield { kind, config };
      }
}

/** Ask, in the background, what a take of `entry` at `fps` will encode with. */
export async function prepareTake(entry, { fps = 60 } = {}) {
  for await (const _ of configure(entry.width, entry.height, fps)) break;
}

/** Start a take of `entry` (a loaded model) at `fps`. While it is the player's `tap`, the player
 *  logs the frame due at each time `next` gives, and draws the frames `waiting` in order into it
 *  (`record`, then `recorded` after the submit); `stop` resolves to the video as a Blob. */
export async function startTake(device, entry, { fps = 60 } = {}) {
  // I420 halves each side for its colour planes, so an odd size goes through a canvas instead.
  const even = entry.width % 2 === 0 && entry.height % 2 === 0;
  const candidates = even ? configure(entry.width, entry.height, fps) : null;
  const first = await candidates?.next();
  return first && !first.done ? new CodecTake(device, entry, fps, first.value, candidates) : new RecorderTake(device, entry, fps);
}

class TakeBase {
  constructor(entry, fps) {
    Object.assign(this, { entry, fps });
    this.start = null;
    this.slot = -1;
    /** Frames logged and not drawn yet, oldest first: `{index, entry, latent, settings}`. */
    this.waiting = [];
    this.latest = null;
    this.frames = 0;
    this.dropped = 0;
    this.failed = null;
  }
  /** Seconds recorded so far. */
  get seconds() {
    return this.start === null ? 0 : (performance.now() - this.start) / 1000;
  }
  /** The time (ms, as the player's) of the take's next frame, if it is due by the animation
   *  frame at `now`: up to half a frame early, so each animation frame of a screen at the
   *  take's rate makes one. The take starts at the first `now`. */
  next(now) {
    this.start ??= now;
    const at = this.start + ((this.slot + 1) * 1000) / this.fps;
    return at <= now + 500 / this.fps ? at : null;
  }
  /** Keep frame `f` (the player's `update`) as the take's next. */
  log(f) {
    f.index = ++this.slot;
    this.waiting.push(f);
    this.latest = f;
  }
  ready() {
    return true;
  }
}

class CodecTake extends TakeBase {
  constructor(device, entry, fps, first, candidates) {
    super(entry, fps);
    Object.assign(this, { device, candidates });
    const { width: w, height: h } = entry;
    this.size = (w * h * 3) / 2;
    const bytes = Math.ceil(this.size / 4) * 4;
    this.out = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.shape = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.shape, 0, new Uint32Array([h, w, 0, 0]));
    const groups = Math.ceil(bytes / 4 / 256);
    this.groups = [Math.min(groups, 65535), Math.ceil(groups / 65535)];
    this.ring = Array.from({ length: RING }, () => ({
      buffer: device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      busy: false,
    }));
    this.filling = null;
    /** The conversion for each model the take has drawn, made once. */
    this.binds = new Map();
    /** Each frame's encode waits for the one before, so frames reach the encoder in order. */
    this.chain = Promise.resolve();
    this.open(first);
  }
  /** Encode with `candidate`. An encoder can claim a configuration and then refuse its first
   *  frame (Firefox did with NV12), so one that fails before writing anything gives way to the
   *  next candidate; frames wait meanwhile (`ready`). */
  open(candidate) {
    Object.assign(this, candidate, { keyed: 0, switching: false });
    this.mux = new Mp4(this.kind, this.entry.width, this.entry.height);
    const encoder = (this.encoder = new VideoEncoder({
      output: (chunk, meta) => this.mux.add(chunk, meta),
      error: (e) => {
        if (encoder !== this.encoder) return;
        if (this.mux.samples.length) return void (this.failed ??= e);
        this.switching = true;
        this.reopened = this.candidates.next().then(({ value, done }) => (done ? (this.failed ??= e) : this.open(value)));
      },
    }));
    encoder.configure(this.config);
  }
  /** The conversion of `entry`'s frames: a model switched to mid-take has the take's size
   *  (the page refuses others). */
  use(entry) {
    if (!this.binds.has(entry)) {
      const module = this.device.createShaderModule({ code: I420(entry.program.output === "bgra8" ? "zyx" : "xyz") });
      const pipeline = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.binds.set(entry, [pipeline, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: entry.model.output } }, { binding: 1, resource: { buffer: this.out } },
        { binding: 2, resource: { buffer: this.shape } }] })]);
    }
    return this.binds.get(entry);
  }
  get label() {
    return `${this.config.codec}, ${this.config.bitrateMode === "quantizer" ? "constant quality" : "high bitrate"}`;
  }
  ready() {
    return !this.failed && !this.switching && this.encoder.encodeQueueSize < QUEUE && this.ring.some((r) => !r.busy);
  }
  /** Frame `f`, the oldest waiting, just drawn into the player's encoder: as I420, into a free
   *  buffer of the ring. */
  record(encoder, f) {
    const [pipeline, bind] = this.use(f.entry);
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...this.groups);
    pass.end();
    this.filling = this.ring.find((r) => !r.busy);
    this.filling.busy = true;
    this.filling.index = this.waiting.shift().index;
    encoder.copyBufferToBuffer(this.out, 0, this.filling.buffer, 0, this.out.size);
  }
  /** After the player's submit: read the frame back, then hand it to the encoder. */
  recorded() {
    const r = this.filling;
    const slot = r.index;
    const mapped = r.buffer.mapAsync(GPUMapMode.READ);
    this.chain = this.chain.then(() => mapped).then(async () => {
      if (this.switching) await this.reopened;      // a frame read back while the encoder changes
      if (!this.failed) {
        const frame = new VideoFrame(new Uint8Array(r.buffer.getMappedRange(), 0, this.size), {
          format: "I420", codedWidth: this.entry.width, codedHeight: this.entry.height, colorSpace: BT709,
          timestamp: Math.round((slot * 1e6) / this.fps), duration: Math.round(1e6 / this.fps),
        });
        const quantizer = this.config.bitrateMode === "quantizer" ? { [this.kind]: { quantizer: QUANTIZER[this.kind] } } : {};
        try {
          this.encoder.encode(frame, { keyFrame: this.keyed++ % Math.round(KEY_EVERY_S * this.fps) === 0, ...quantizer });
        } catch (e) {
          if (this.encoder.state !== "configured") throw e;   // an encoder that failed, since replaced
        }
        frame.close();
        this.frames++;
      }
      r.buffer.unmap();
    }).catch((e) => (this.failed ??= e)).finally(() => (r.busy = false));
  }
  async stop() {
    await this.chain;
    if (!this.failed) await this.encoder.flush().catch((e) => (this.failed ??= e));
    if (this.encoder.state !== "closed") this.encoder.close();
    for (const b of [this.out, this.shape, ...this.ring.map((r) => r.buffer)]) b.destroy();
    if (this.failed) throw this.failed;
    return this.mux.finish(this.fps);
  }
}

/** Where WebCodecs is missing: a canvas of the model's size through MediaRecorder, MP4 where
 *  offered, else WebM. */
class RecorderTake extends TakeBase {
  constructor(device, entry, fps) {
    super(entry, fps);
    this.canvas = Object.assign(document.createElement("canvas"), { width: entry.width, height: entry.height });
    this.context = this.canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format, alphaMode: "opaque" });
    Object.assign(this, { device, format });
    this.use(entry);
    const kind = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"].find((k) => MediaRecorder.isTypeSupported(k)) ?? "";
    this.stream = this.canvas.captureStream(0);
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream, { mimeType: kind, videoBitsPerSecond: Math.round(BITS_PER_PIXEL * entry.width * entry.height * fps) });
    this.recorder.addEventListener("dataavailable", (e) => e.data.size && this.chunks.push(e.data));
    this.recorder.start(1000);
    this.label = this.recorder.mimeType;
  }
  /** MediaRecorder stamps each frame as it arrives, so a frame drawn late cannot be put back
   *  in its place: only the newest waits, and those it replaces count as `dropped`. */
  log(f) {
    this.dropped += this.waiting.length;
    this.waiting = [];
    super.log(f);
  }
  use(entry) {
    if (entry !== this.entry || !this.onto) this.onto = screen(this.device, entry.model, this.format);
    this.entry = entry;
  }
  record(encoder, f) {
    this.use(f.entry);
    this.waiting.shift();
    this.onto.draw(encoder, this.context.getCurrentTexture());
  }
  recorded() {
    this.stream.getVideoTracks()[0].requestFrame?.();
    this.frames++;
  }
  stop() {
    return new Promise((resolve) => {
      this.recorder.addEventListener("stop", () => {
        this.context.unconfigure();
        resolve(new Blob(this.chunks, { type: this.recorder.mimeType }));
      });
      this.recorder.stop();
    });
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

/** An MP4 of one video track, its samples kept in memory as Blobs until `finish`. */
class Mp4 {
  constructor(kind, width, height) {
    Object.assign(this, { kind, width, height });
    this.samples = [];        // [size, timestamp in us, key]
    this.parts = [];
    this.batch = [];
    this.batchBytes = 0;
    this.description = null;
  }
  add(chunk, meta) {
    if (meta?.decoderConfig?.description) this.description = new Uint8Array(meta.decoderConfig.description);
    const data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);
    this.samples.push([data.length, chunk.timestamp, chunk.type === "key"]);
    this.batch.push(data);
    this.batchBytes += data.length;
    // A Blob a few tens of MB at a time, which a browser may keep on disk rather than in memory.
    if (this.batchBytes > 32e6) this.flush();
  }
  flush() {
    if (this.batch.length) this.parts.push(new Blob(this.batch));
    this.batch = [];
    this.batchBytes = 0;
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
   *  sample's offset to its presentation is `ctts`, and an edit list starts the track at the
   *  first presented frame. */
  finish(fps) {
    this.flush();
    const n = this.samples.length;
    const timescale = 90000;
    const first = this.samples.reduce((a, [, t]) => Math.min(a, t), Infinity);
    const shown = this.samples.map(([, t]) => Math.round(((t - first) * timescale) / 1e6));
    const decoded = [...shown].sort((a, b) => a - b);
    const delay = shown.reduce((a, t, i) => Math.max(a, decoded[i] - t), 0);
    const offsets = shown.map((t, i) => t + delay - decoded[i]);
    const durations = decoded.map((t, i) => (i + 1 < n ? decoded[i + 1] - t : Math.round(timescale / fps)));
    const duration = durations.reduce((a, d) => a + d, 0);
    const stts = runs(durations);
    const ctts = runs(offsets);
    const keys = this.samples.map(([, , key], i) => (key ? i + 1 : 0)).filter(Boolean);
    const ftyp = box("ftyp", ascii("isom"), fields([32, 512]), ascii("isomiso2mp41"), ascii(this.kind === "avc" ? "avc1" : "vp09"));
    const mdatSize = 16 + this.samples.reduce((a, [size]) => a + size, 0);
    const mdatHead = concat([fields([32, 1]), ascii("mdat"), fields([64, mdatSize])]);
    let offset = ftyp.length + 16;
    const placed = this.samples.map(([size]) => {
      const at = offset;
      offset += size;
      return at;
    });
    const stbl = box("stbl",
      full("stsd", 0, 0, fields([32, 1]), this.sampleEntry()),
      full("stts", 0, 0, fields([32, stts.length / 2]), table(32, stts)),
      ...(delay ? [full("ctts", 0, 0, fields([32, ctts.length / 2]), table(32, ctts))] : []),
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
    // Reordered frames start presenting `delay` in: the edit list skips it.
    const edts = delay ? [box("edts", full("elst", 0, 0, fields([32, 1], [32, duration], [32, delay], [16, 1], [16, 0])))] : [];
    const moov = box("moov", mvhd, box("trak", tkhd, ...edts, mdia));
    return new Blob([ftyp, mdatHead, ...this.parts, moov], { type: "video/mp4" });
  }
}
