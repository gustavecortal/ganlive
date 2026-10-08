"""After an engine model: the frame resized to the size being shown, as the bytes a window, an
encoder or a still wants, and their downloads. `FrameStage`'s interface, in WGSL.

A conversion reads the model's output buffer and is queued behind the frame that wrote it and
ahead of the next one, so nothing waits between them. A download copies into a mappable buffer
of a ring and is read back when its ticket is waited on, behind the next frame's generation.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import wgpu

STORAGE = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

# The model's RGBA8 frame -> the size shown, as BGRA8 (what an SDL texture is). Shrinking
# averages the same windows as torch's `area` (adaptive average pooling); growing is bilinear.
RESIZE = """
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> U: array<u32>;      // H, W, h, w
fn px(y: u32, x: u32) -> vec4f { return unpack4x8unorm(X[y * U[1] + x]); }
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
  Y[id.y * w + id.x] = pack4x8unorm(vec4f(c.b, c.g, c.r, 1.0));
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


class _Deferred:
    """A block inside which a stage's downloads are not read at once: on exit they are read
    together, or, for a `handoff`, left to the block's `ticket`."""

    def __init__(self, stage: EngineStage, keep: bool) -> None:
        self._stage, self._keep = stage, keep
        self.ticket = Ticket()

    def __enter__(self) -> _Deferred:
        self._stage._jobs = []
        return self

    def __exit__(self, *_exc) -> bool:
        jobs, self._stage._jobs = self._stage._jobs, None
        self.ticket = Ticket(jobs)
        if not self._keep:
            self.ticket.wait()
        return False


class EngineStage:
    """`frame.FrameStage` for engine models: `step`, then `bgra_bytes`, `nv12_bytes` or
    `rgb_still`, with `handoff` to read the downloads behind the next frame."""

    def __init__(self, device, height: int, width: int) -> None:
        self.device = device
        self.height, self.width = int(height), int(width)
        self.compiled = {"yuv": True, "rgb": True, "bgra": True}
        self._resize = self._pipeline(RESIZE)
        self._nv12 = self._pipeline(NV12)
        self._buffers: dict = {}
        self._rings: dict = {}
        #: The downloads of the open `deferred`/`handoff` block, or None outside one.
        self._jobs: list | None = None

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
        return got

    def _run(self, pipeline, buffers, groups) -> None:
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                     for i, b in enumerate(buffers)]))
        compute.dispatch_workgroups(*groups)
        compute.end()
        self.device.queue.submit([encoder.finish()])

    def _words(self, name: str, *values: int):
        buf = self._buffer(name, 16)
        self.device.queue.write_buffer(buf, 0, np.array([*values, 0, 0, 0, 0][:4], np.uint32))
        return buf

    def resize(self, height: int, width: int) -> None:
        self.height, self.width = int(height), int(width)

    def eager(self) -> None:
        """Nothing to fall back to: the conversions are shaders."""

    def warm(self, staged) -> int:
        """The conversions compile when the stage is made; nothing is left to build."""
        return 0

    def sync(self) -> None:
        """Wait for everything queued: a read waits for the work before it. (wgpu-py 0.32's
        `on_submitted_work_done_sync` fails on a callback signature.)"""
        self.device.queue.read_buffer(self._buffer("sync", 16), 0, 4)

    def release(self) -> None:
        """Nothing to wait for: a conversion is queued before the next frame's generation."""

    def deferred(self) -> _Deferred:
        return _Deferred(self, keep=False)

    def handoff(self) -> _Deferred:
        return _Deferred(self, keep=True)

    def aside(self):
        return contextlib.nullcontext()

    def pinned(self) -> dict[str, bool]:
        return {name: True for name in self._rings}

    def step(self, outs) -> Stepped:
        """The model's frame at the size shown, as BGRA8."""
        net = outs[0] if isinstance(outs, (list, tuple)) else outs
        H, W = net.cfg.ladder.height, net.cfg.ladder.width
        h, w = self.height, self.width
        out = self._buffer("shown", h * w * 4)
        self._run(self._resize, [net.model.output, out, self._words("resize", H, W, h, w)],
                  (math.ceil(w / 8), math.ceil(h / 8), 1))
        return Stepped(out, h, w)

    def _take(self, name: str, source, size: int, shape, depth: int = 3) -> np.ndarray:
        """Copy `size` bytes of `source` to the host through this destination's ring; the array
        holds them once the copy is read (now, or at the block's ticket)."""
        padded = math.ceil(size / 4) * 4
        ring = self._rings.get(name)
        if ring is None or ring["shape"] != tuple(shape):
            ring = self._rings[name] = {"shape": tuple(shape), "n": 0, "slots": [
                (self.device.create_buffer(size=padded, usage=wgpu.BufferUsage.MAP_READ
                                           | wgpu.BufferUsage.COPY_DST),
                 np.zeros(shape, np.uint8)) for _ in range(max(2, depth))]}
        staging, array = ring["slots"][ring["n"] % len(ring["slots"])]
        ring["n"] += 1
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(source, 0, staging, 0, padded)
        self.device.queue.submit([encoder.finish()])
        job = (staging, array, size)
        if self._jobs is None:
            Ticket([job]).wait()
        else:
            self._jobs.append(job)
        return array

    def bgra_bytes(self, frame: Stepped) -> np.ndarray:
        """`(h, w, 4)` BGRA8, the bytes an SDL texture is. Four in the ring: one downloading
        behind the next frame, one published, one the window thread may still be uploading."""
        return self._take("bgra", frame.buffer, frame.height * frame.width * 4,
                          (frame.height, frame.width, 4), depth=4)

    def nv12_bytes(self, frame: Stepped, dest: str = "yuv", depth: int = 3) -> np.ndarray:
        """The `(h*3/2, w)` uint8 plane stack an encoder wants."""
        h, w = frame.height, frame.width
        size = h * w * 3 // 2
        out = self._buffer(f"nv12.{dest}", size)
        words = math.ceil(size / 4)
        groups = math.ceil(words / 256)
        self._run(self._nv12, [frame.buffer, out, self._words("nv12", h, w)],
                  (min(groups, 65535), math.ceil(groups / 65535), 1))
        return self._take(dest, out, size, (h * 3 // 2, w), depth)

    def rgb_still(self, frame: Stepped) -> np.ndarray:
        """`(h, w, 3)` RGB, its own array, outside the rings."""
        bgra = self._take("still", frame.buffer, frame.height * frame.width * 4,
                          (frame.height, frame.width, 4), depth=2)
        if self._jobs is not None:
            self._jobs, jobs = [j for j in self._jobs if j[1] is not bgra], self._jobs
            Ticket([j for j in jobs if j[1] is bgra]).wait()
        return bgra[..., 2::-1].copy()
