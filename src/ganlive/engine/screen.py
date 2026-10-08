"""After an engine model: the frame resized to the take's size, as the bytes an encoder or a
still wants, and their downloads. `FrameStage`'s interface, in WGSL.

A conversion reads the model's output buffer and is queued behind the frame that wrote it and
ahead of the next one, so nothing waits between them. A download copies into a mappable buffer
of a ring and is read back when its ticket is waited on, behind the next frame's generation.
"""

from __future__ import annotations

import math

import numpy as np
import wgpu

from ganlive.engine.codegen import linear
from ganlive.engine.runner import MAPPABLE

STORAGE = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

# The model's BGRA8 frame -> the size shown, as BGRA8 (what an SDL texture is). Shrinking
# averages the same windows as torch's `area` (adaptive average pooling). Growing is bilinear.
RESIZE = """
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> U: array<u32>;      // H, W, h, w
fn px(y: u32, x: u32) -> vec4f { return unpack4x8unorm(X[y * U[1] + x]); }   // BGRA
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let H = U[0]; let W = U[1]; let h = U[2]; let w = U[3];
  if (id.x >= w || id.y >= h) { return; }
  var c = vec4f(0.0);
  if (h * w <= H * W) {
    let y0 = id.y * H / h; let y1 = ((id.y + 1u) * H + h - 1u) / h;
    let x0 = id.x * W / w; let x1 = ((id.x + 1u) * W + w - 1u) / w;
    for (var y = y0; y < y1; y++) { for (var x = x0; x < x1; x++) { c += px(y, x); } }
    c /= f32((y1 - y0) * (x1 - x0));
  } else {
    let sy = clamp((f32(id.y) + 0.5) * f32(H) / f32(h) - 0.5, 0.0, f32(H - 1u));
    let sx = clamp((f32(id.x) + 0.5) * f32(W) / f32(w) - 0.5, 0.0, f32(W - 1u));
    let y0 = u32(sy); let x0 = u32(sx);
    let y1 = min(y0 + 1u, H - 1u); let x1 = min(x0 + 1u, W - 1u);
    let fy = sy - f32(y0); let fx = sx - f32(x0);
    c = mix(mix(px(y0, x0), px(y0, x1), fx), mix(px(y1, x0), px(y1, x1), fx), fy);
  }
  Y[id.y * w + id.x] = pack4x8unorm(vec4f(c.xyz, 1.0));
}
"""

# A shown BGRA8 frame -> NV12, BT.709 limited range as `pixels.to_nv12`: the Y plane, then the
# chroma at half resolution as UVUV rows. One thread a word of four bytes.
NV12 = """
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> U: array<u32>;      // h, w
const KR = 0.2126; const KB = 0.0722; const KG = 1.0 - 0.2126 - 0.0722;
fn rgb(y: u32, x: u32) -> vec3f {           // [-1, 1], as the generator drew it
  return unpack4x8unorm(X[y * U[1] + x]).zyx * 2.0 - 1.0;
}
fn byte(b: u32) -> u32 {
  let h = U[0]; let w = U[1];
  if (b < h * w) {
    let c = rgb(b / w, b % w);
    return u32(clamp(round((KR * c.r + KG * c.g + KB * c.b) * 109.5 + 125.5), 0.0, 255.0));
  }
  let i = b - h * w; let row = i / w; let col = i % w; let x = (col / 2u) * 2u;
  let c = (rgb(2u * row, x) + rgb(2u * row, x + 1u) + rgb(2u * row + 1u, x) + rgb(2u * row + 1u, x + 1u)) / 4.0;
  let l = KR * c.r + KG * c.g + KB * c.b;
  let v = select((c.b - l) * (224.0 / (4.0 * (1.0 - KB))), (c.r - l) * (224.0 / (4.0 * (1.0 - KR))), col % 2u == 1u);
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
}
"""


class Stepped:
    """A frame at the size shown, as BGRA8 in `buffer` on the GPU."""

    def __init__(self, buffer, height: int, width: int) -> None:
        self.buffer, self.height, self.width = buffer, height, width

    @classmethod
    def native(cls, net) -> Stepped:
        """An engine model's last frame at its own size, in the model's own buffer."""
        return cls(net.model.output, net.cfg.ladder.height, net.cfg.ladder.width)


class Ticket:
    """The downloads of one frame, started and not yet read. `wait` before reading them."""

    def __init__(self, jobs=()) -> None:
        self._jobs = list(jobs)

    def wait(self) -> None:
        for staging, array, size in self._jobs:
            staging.map_sync(wgpu.MapMode.READ, 0, staging.size)
            np.copyto(array.reshape(-1).view(np.uint8),
                      np.frombuffer(staging.read_mapped(0, staging.size, copy=False), np.uint8)[:size])
            staging.unmap()
        self._jobs = []


class _Handoff:
    """A block whose downloads are submitted together on exit and read at its `ticket`."""

    def __init__(self, stage: EngineStage) -> None:
        self._stage = stage
        self.ticket = Ticket()

    def __enter__(self) -> _Handoff:
        self._stage._jobs = []
        return self

    def __exit__(self, *_exc) -> bool:
        jobs, self._stage._jobs = self._stage._jobs, None
        self._stage.flush()
        self.ticket = Ticket(jobs)
        return False


class EngineStage:
    """What comes after an engine model's frame: `step` it to the size shown, then
    `nv12_bytes` or `rgb_still`, with `handoff` to read the downloads behind the
    next frame. Everything a frame asks of it is recorded into one command buffer."""

    def __init__(self, device, height: int, width: int) -> None:
        self.device = device
        self.height, self.width = int(height), int(width)
        self._resize = self._pipeline(RESIZE)
        self._nv12 = self._pipeline(NV12)
        self._buffers: dict = {}
        self._rings: dict = {}
        self._binds: dict = {}
        self._written: dict = {}
        self._encoder = None
        #: The downloads of the open `handoff` block, or None outside one.
        self._jobs: list | None = None
        #: Whether a conversion may write straight into the buffer the host maps.
        self._mappable = MAPPABLE in device.features

    def _pipeline(self, code: str):
        return self.device.create_compute_pipeline(
            layout="auto", compute={"module": self.device.create_shader_module(code=code),
                                    "entry_point": "main"})

    def _buffer(self, name: str, size: int, usage=STORAGE):
        """A buffer kept by name and remade when a larger one is asked for."""
        size = max(16, math.ceil(size / 4) * 4)
        got = self._buffers.get(name)
        if got is None or got.size < size:
            got = self._buffers[name] = self.device.create_buffer(size=size, usage=usage)
            self._binds.clear()                  # they may hold the buffer this replaces
        return got

    def _encode(self):
        if self._encoder is None:
            self._encoder = self.device.create_command_encoder()
        return self._encoder

    def flush(self) -> None:
        """Submit what this frame recorded."""
        if self._encoder is not None:
            self.device.queue.submit([self._encoder.finish()])
            self._encoder = None

    def _run(self, pipeline, buffers, groups) -> None:
        key = (id(pipeline), *(id(b) for b in buffers))
        bind = self._binds.get(key)
        if bind is None:
            bind = self._binds[key] = self.device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                         for i, b in enumerate(buffers)])
        compute = self._encode().begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, bind)
        compute.dispatch_workgroups(*groups)
        compute.end()

    def _words(self, name: str, *values: int):
        """A small parameter buffer, written only when its values change."""
        buf = self._buffer(name, 16)
        if self._written.get(name) != (buf, values):
            self.device.queue.write_buffer(buf, 0, np.array([*values, 0, 0, 0, 0][:4], np.uint32))
            self._written[name] = (buf, values)
        return buf

    def resize(self, height: int, width: int) -> None:
        self.height, self.width = int(height), int(width)

    def sync(self) -> None:
        """Wait for everything queued: a read waits for the work before it. (wgpu-py 0.32's
        `on_submitted_work_done_sync` fails on a callback signature.)"""
        self.flush()
        self.device.queue.read_buffer(self._buffer("sync", 16), 0, 4)

    def handoff(self) -> _Handoff:
        return _Handoff(self)

    def step(self, net) -> Stepped:
        """The model's frame at the size shown, as BGRA8: the model's own buffer when it is
        shown at its size."""
        H, W = net.cfg.ladder.height, net.cfg.ladder.width
        h, w = self.height, self.width
        if (H, W) == (h, w):
            return Stepped.native(net)
        out = self._buffer("shown", h * w * 4)
        self._run(self._resize, [net.model.output, out, self._words("resize", H, W, h, w)],
                  (math.ceil(w / 8), math.ceil(h / 8), 1))
        return Stepped(out, h, w)

    def _take(self, name: str, source, size: int, shape, depth: int = 3, write=None) -> np.ndarray:
        """Bring `size` bytes to the host through this destination's ring, one per shape, so
        that switching models reuses rings: copied from `source`, or, where the device allows,
        written by `write(buffer)` straight into the buffer the host maps. The array holds them
        once the download is read: now, or at the `handoff` block's ticket."""
        padded = math.ceil(size / 4) * 4
        direct = write is not None and self._mappable
        usage = wgpu.BufferUsage.MAP_READ | (wgpu.BufferUsage.STORAGE if direct else wgpu.BufferUsage.COPY_DST)
        ring = self._rings.get((name, tuple(shape)))
        if ring is None:
            ring = self._rings[(name, tuple(shape))] = {"n": 0, "slots": [
                (self.device.create_buffer(size=padded, usage=usage), np.zeros(shape, np.uint8))
                for _ in range(max(2, depth))]}
        staging, array = ring["slots"][ring["n"] % len(ring["slots"])]
        ring["n"] += 1
        if direct:
            write(staging)
        else:
            if write is not None:
                source = self._buffer(f"{name}.out", padded)
                write(source)
            self._encode().copy_buffer_to_buffer(source, 0, staging, 0, padded)
        job = (staging, array, size)
        if self._jobs is None:
            self.flush()
            Ticket([job]).wait()
        else:
            self._jobs.append(job)
        return array

    def nv12_bytes(self, frame: Stepped, dest: str = "yuv", depth: int = 3) -> np.ndarray:
        """The `(h*3/2, w)` uint8 plane stack an encoder wants."""
        h, w = frame.height, frame.width
        size = h * w * 3 // 2

        def write(out):
            self._run(self._nv12, [frame.buffer, out, self._words("nv12", h, w)],
                      linear(math.ceil(size / 4), 256)[0])
        return self._take(dest, None, size, (h * 3 // 2, w), depth, write=write)

    def rgb_still(self, frame: Stepped) -> np.ndarray:
        """`(h, w, 3)` RGB, its own array, read now even inside a `handoff` block."""
        jobs, self._jobs = self._jobs, None
        try:
            bgra = self._take("still", frame.buffer, frame.height * frame.width * 4,
                              (frame.height, frame.width, 4), depth=2)
        finally:
            self._jobs = jobs
        return bgra[..., 2::-1].copy()
