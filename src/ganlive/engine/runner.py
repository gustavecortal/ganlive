"""Run a ganlive engine program (from `ganlive convert`) through wgpu-py: the same shaders and
dispatches `runner.mjs` runs in a browser, on Vulkan, Metal or DX12.

    model = Model.load("models/lichen")
    model.set_latent(z); model.set_settings(k)
    model.frame()                       # submits one frame
    pixels = model.read()               # (H, W, 4) uint8, or (3, H, W) float32 with output="f32"
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import wgpu

USAGE = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC


def default_device(fallback: bool = False):
    """The high-performance adapter (or, with `fallback`, the CPU one), with its largest
    buffers and timestamps if it has them."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance",
                                            force_fallback_adapter=fallback)
    limits = {k: adapter.limits[k] for k in (
        "max-buffer-size", "max-storage-buffer-binding-size", "max-storage-buffers-per-shader-stage")}
    features = [f for f in ("timestamp-query",) if f in adapter.features]
    return adapter.request_device_sync(required_features=features, required_limits=limits)


class Model:
    """One loaded program: its buffers, compiled pipelines and the steps of a frame."""

    def __init__(self, device, program: dict, weights: bytes, *, output: str = "rgba8",
                 noise: dict[str, np.ndarray] | None = None) -> None:
        self.device, self.program = device, program
        self.height, self.width = program["height"], program["width"]
        self.format = output
        limit = min(device.limits["max-storage-buffer-binding-size"], device.limits["max-buffer-size"])
        last = program["outputs"][output]
        self.buffers: dict[str, wgpu.GPUBuffer] = {}

        def create(name: str, size: int):
            if size > limit:
                raise ValueError(f"{name}: a {size / 2**20:.0f} MiB buffer is over this "
                                 f"device's {limit / 2**20:.0f} MiB limit")
            self.buffers[name] = device.create_buffer(size=size, usage=USAGE)
            return self.buffers[name]

        create(last["buffer"], last["size"])
        noise = noise or {}
        for name, spec in program["buffers"].items():
            buf = create(name, spec["size"])
            init = spec.get("init")
            if init == "weights":
                device.queue.write_buffer(buf, 0, _words(weights))
            elif init == "ones":
                device.queue.write_buffer(buf, 0, np.ones(spec["size"] // 4, np.float32))
            elif init == "noise":
                layer = name.removeprefix("noise.")
                if layer in noise:
                    device.queue.write_buffer(buf, 0, np.ascontiguousarray(noise[layer], np.float32))
                else:
                    self._seed(buf, spec["size"] // 4, spec["seed"])

        modules: dict[int, object] = {}
        self.steps = []
        for spec in [*program["steps"], last["step"]]:
            index = spec["shader"]
            if index not in modules:
                module = device.create_shader_module(code=program["shaders"][index])
                modules[index] = device.create_compute_pipeline(
                    layout="auto", compute={"module": module, "entry_point": "main"})
            pipeline = modules[index]
            bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {"binding": i, "resource": {"buffer": self.buffers[n], "offset": 0,
                                            "size": self.buffers[n].size}}
                for i, n in enumerate(spec["bind"])])
            self.steps.append((spec["name"], pipeline, bind, spec["groups"]))
        self.output = self.buffers[last["buffer"]]

    @classmethod
    def load(cls, folder, device=None, **options) -> Model:
        folder = Path(folder)
        program = json.loads((folder / "program.json").read_text(encoding="utf-8"))
        return cls(device or default_device(), program, (folder / "weights.bin").read_bytes(), **options)

    def set_latent(self, z) -> None:
        self.device.queue.write_buffer(self.buffers["Z"], 0, np.ascontiguousarray(z, np.float32))

    def set_settings(self, k) -> None:
        self.device.queue.write_buffer(self.buffers["K"], 0, np.ascontiguousarray(k, np.float32))

    def encode(self, encoder) -> None:
        """Records one frame into `encoder`, in one compute pass."""
        compute = encoder.begin_compute_pass()
        for _, pipeline, bind, (x, y, z) in self.steps:
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, bind)
            compute.dispatch_workgroups(x, y, z)
        compute.end()

    def frame(self) -> None:
        encoder = self.device.create_command_encoder()
        self.encode(encoder)
        self.device.queue.submit([encoder.finish()])

    def read(self) -> np.ndarray:
        """The last frame, waited for: (H, W, 4) uint8 RGBA, or (3, H, W) float32 in [-1, 1]."""
        data = self.device.queue.read_buffer(self.output)
        if self.format == "f32":
            return np.frombuffer(data, np.float32).reshape(3, self.height, self.width)
        return np.frombuffer(data, np.uint8).reshape(self.height, self.width, 4)

    def _seed(self, buf, n: int, seed: int) -> None:
        code = self.program["noise"].replace("${n}", str(n)).replace("${seed}", str(seed))
        pipeline = self.device.create_compute_pipeline(
            layout="auto", compute={"module": self.device.create_shader_module(code=code),
                                    "entry_point": "main"})
        groups = math.ceil(n / 256)
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}]))
        compute.dispatch_workgroups(min(groups, 65535), math.ceil(groups / 65535), 1)
        compute.end()
        self.device.queue.submit([encoder.finish()])


def _words(data: bytes) -> bytes:
    """GPU uploads come in whole 4-byte words."""
    return data if len(data) % 4 == 0 else data + bytes(4 - len(data) % 4)
