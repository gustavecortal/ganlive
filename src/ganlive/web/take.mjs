// A take in a browser: the picture at the model's own size, every frame, as an MP4. Each frame
// is drawn a second time onto a canvas of the model's size, off the page, and encoded with
// WebCodecs: the GPU's H.264 encoder at a constant quality where the browser offers one, else
// VP9 or a high bitrate, written into an MP4 here. A browser without WebCodecs records the
// canvas with MediaRecorder instead. The desktop's takes are `record/video.py`.

import { screen } from "./runner.mjs";

/** Constant quality, where the encoder takes a quantizer: about the desktop's CRF 18. */
const QUANTIZER = { avc: 20, vp9: 26 };
/** Otherwise a bitrate, in bits per pixel of each frame. */
const BITS_PER_PIXEL = 0.15;
/** Frames the encoder may hold before new ones are skipped (counted in `dropped`). */
const QUEUE = 3;
/** Seconds between key frames. */
const KEY_EVERY_S = 2;

/** The encoders worth trying, best first. A level fits the frame size and rate. */
const CODECS = [
  ["avc", "avc1.640034"], ["avc", "avc1.640033"], ["avc", "avc1.64002a"], ["avc", "avc1.4d0034"],
  ["vp9", "vp09.00.51.08"], ["vp9", "vp09.00.41.08"],
];

/** The first encoder configuration this browser takes for a `width` x `height` take at `fps`:
 *  hardware before software, a constant quality before a bitrate. */
async function configure(width, height, fps) {
  if (typeof VideoEncoder === "undefined") return null;
  for (const hardwareAcceleration of ["prefer-hardware", "no-preference"])
    for (const [kind, codec] of CODECS)
      for (const bitrateMode of ["quantizer", "variable"]) {
        const config = {
          codec, width, height, framerate: fps, hardwareAcceleration, bitrateMode, latencyMode: "quality",
          ...(bitrateMode === "variable" ? { bitrate: Math.round(BITS_PER_PIXEL * width * height * fps) } : {}),
          ...(kind === "avc" ? { avc: { format: "avc" } } : {}),
        };
        try {
          if ((await VideoEncoder.isConfigSupported(config)).supported) return { kind, config };
        } catch {
          // a configuration this browser cannot parse is one it does not support
        }
      }
  return null;
}

/** Start a take of `entry` (a loaded model) at `fps`. The player calls `draw` and `drawn` for
 *  every frame it draws while the take is its `tap`; `stop` resolves to the video as a Blob. */
export async function startTake(device, entry, { fps = 60 } = {}) {
  const { width, height } = entry;
  const canvas = Object.assign(document.createElement("canvas"), { width, height });
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });
  const onto = screen(device, entry.model, format);
  const found = await configure(width, height, fps);
  return found ? new CodecTake(canvas, context, onto, entry, fps, found) : new RecorderTake(canvas, context, onto, entry, fps);
}

class TakeBase {
  constructor(canvas, context, onto, entry, fps) {
    Object.assign(this, { canvas, context, onto, entry, fps });
    this.start = null;
    this.slot = -1;
    this.due = false;
    this.frames = 0;
    this.dropped = 0;
  }
  /** Seconds recorded so far. */
  get seconds() {
    return this.start === null ? 0 : (performance.now() - this.start) / 1000;
  }
  /** Into the player's encoder: this frame at the take's size, if its slot is due. */
  draw(encoder, m) {
    if (m !== this.entry) return;    // a switch to another model is refused while recording
    const now = performance.now();
    this.start ??= now;
    // Which frame of the take this is: the next one, unless the screen ran ahead of the take
    // (too early) or a frame was missed by more than half a frame's time.
    const at = ((now - this.start) / 1000) * this.fps;
    if (at < this.slot + 0.5 || !this.ready()) {
      this.due = false;
      return;
    }
    const slot = at < this.slot + 1.5 ? this.slot + 1 : Math.round(at);
    this.dropped += slot - this.slot - 1;
    this.slot = slot;
    this.due = true;
    this.onto.draw(encoder, this.context.getCurrentTexture());
  }
  ready() {
    return true;
  }
}

class CodecTake extends TakeBase {
  constructor(canvas, context, onto, entry, fps, { kind, config }) {
    super(canvas, context, onto, entry, fps);
    this.kind = kind;
    this.config = config;
    this.mux = new Mp4(kind, entry.width, entry.height);
    this.failed = null;
    this.encoder = new VideoEncoder({
      output: (chunk, meta) => this.mux.add(chunk, meta),
      error: (e) => (this.failed = e),
    });
    this.encoder.configure(config);
  }
  get label() {
    return `${this.config.codec}, ${this.config.bitrateMode === "quantizer" ? "constant quality" : "high bitrate"}`;
  }
  ready() {
    return this.encoder.encodeQueueSize < QUEUE && !this.failed;
  }
  /** After the player's submit: hand the frame drawn for this slot to the encoder. */
  drawn() {
    if (!this.due) return;
    const frame = new VideoFrame(this.canvas, { timestamp: Math.round((this.slot * 1e6) / this.fps), duration: Math.round(1e6 / this.fps) });
    const quantizer = this.config.bitrateMode === "quantizer" ? { [this.kind]: { quantizer: QUANTIZER[this.kind] } } : {};
    this.encoder.encode(frame, { keyFrame: this.frames % Math.round(KEY_EVERY_S * this.fps) === 0, ...quantizer });
    frame.close();
    this.frames++;
  }
  async stop() {
    if (!this.failed) await this.encoder.flush();
    this.encoder.close();
    this.context.unconfigure();
    if (this.failed) throw this.failed;
    return this.mux.finish(this.fps);
  }
}

/** Where WebCodecs is missing: the canvas through MediaRecorder, MP4 where offered, else WebM. */
class RecorderTake extends TakeBase {
  constructor(canvas, context, onto, entry, fps) {
    super(canvas, context, onto, entry, fps);
    const kind = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"].find((k) => MediaRecorder.isTypeSupported(k)) ?? "";
    this.stream = canvas.captureStream(0);
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream, { mimeType: kind, videoBitsPerSecond: Math.round(BITS_PER_PIXEL * entry.width * entry.height * fps) });
    this.recorder.addEventListener("dataavailable", (e) => e.data.size && this.chunks.push(e.data));
    this.recorder.start(1000);
    this.label = this.recorder.mimeType;
  }
  drawn() {
    if (!this.due) return;
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
    if (this.kind === "avc") return box("avc1", visual, box("avcC", this.description));
    // VP9 profile 0, 8-bit 4:2:0, BT.709.
    const vpcC = full("vpcC", 1, 0, fields([8, 0], [8, 51], [8, 0x82], [8, 1], [8, 1], [8, 1], [16, 0]));
    return box("vp09", visual, vpcC);
  }
  /** The finished file. */
  finish(fps) {
    this.flush();
    const n = this.samples.length;
    const timescale = 90000;
    const ticks = (us) => Math.round((us * timescale) / 1e6);
    const durations = this.samples.map(([, t], i) => (i + 1 < n ? ticks(this.samples[i + 1][1] - t) : Math.round(timescale / fps)));
    const duration = durations.reduce((a, d) => a + d, 0);
    const stts = [];
    for (const d of durations) {
      if (stts.length && stts[stts.length - 1][1] === d) stts[stts.length - 1][0]++;
      else stts.push([1, d]);
    }
    const keys = this.samples.map(([, , key], i) => (key ? i + 1 : 0)).filter(Boolean);
    const ftyp = box("ftyp", ascii("isom"), fields([32, 512]), ascii("isomiso2mp41"), ascii(this.kind === "avc" ? "avc1" : "vp09"));
    const mdatSize = 16 + this.samples.reduce((a, [size]) => a + size, 0);
    const mdatHead = concat([fields([32, 1]), ascii("mdat"), fields([64, mdatSize])]);
    let offset = ftyp.length + 16;
    const offsets = this.samples.map(([size]) => {
      const at = offset;
      offset += size;
      return at;
    });
    const stbl = box("stbl",
      full("stsd", 0, 0, fields([32, 1]), this.sampleEntry()),
      full("stts", 0, 0, fields([32, stts.length], ...stts.flatMap(([count, d]) => [[32, count], [32, d]]))),
      full("stss", 0, 0, fields([32, keys.length], ...keys.map((k) => [32, k]))),
      full("stsc", 0, 0, fields([32, 1], [32, 1], [32, 1], [32, 1])),
      full("stsz", 0, 0, fields([32, 0], [32, n], ...this.samples.map(([size]) => [32, size]))),
      full("co64", 0, 0, fields([32, n], ...offsets.map((o) => [64, o]))));
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
    return new Blob([ftyp, mdatHead, ...this.parts, moov], { type: "video/mp4" });
  }
}
