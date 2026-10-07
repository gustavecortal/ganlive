"""Run a ganlive engine program (from `ganlive convert`) through wgpu-py: the same shaders and
dispatches `runner.mjs` runs in a browser, on Vulkan, Metal or DX12.

    model = Model.load("models/lichen")
    model.set_latent(z); model.set_settings(k)
    model.frame()                       # submits one frame
    pixels = model.read()               # (H, W, 4) uint8, or (3, H, W) float32 with output="f32"
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import wgpu

USAGE = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC


def default_device(fallback: bool = False):
    """The high-performance adapter (or, with `fallback`, the CPU one), with its largest
    buffers and timestamps if it has them."""
    return _device(wgpu.gpu.request_adapter_sync(power_preference="high-performance",
                                                 force_fallback_adapter=fallback))


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
        """The model in `folder`, on `device`, or else on the fastest backend (`fastest`)."""
        program, weights = read(folder)
        if device is None:
            return fastest(program, weights, **options)[0]
        return cls(device, program, weights, **options)

    def destroy(self) -> None:
        for buf in self.buffers.values():
            buf.destroy()

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


def read(folder) -> tuple[dict, bytes]:
    folder = Path(folder)
    program = json.loads((folder / "program.json").read_text(encoding="utf-8"))
    return program, (folder / "weights.bin").read_bytes()


#: Where each machine's measured choice of backend is kept, per adapter, driver and program.
CHOICES = Path("runs/ganlive/backends.json")


def _identity(adapter) -> str:
    """An adapter as its backend, GPU and driver: a driver update measures again. Backends
    report the driver version in different fields, so all of them are kept."""
    i = adapter.info
    return " | ".join(str(i.get(k, "")) for k in (
        "backend_type", "device", "vendor_id", "device_id", "vendor", "description"))


def fastest(program: dict, weights: bytes, *, backend: str | None = None, frames: int = 20,
            choices: Path = CHOICES, **options) -> tuple[Model, dict]:
    """The model built on whichever GPU backend runs it fastest here, and what was measured.

    The same card can differ several times over between backends (on an Arc A770, wgpu-py's
    Vulkan runs a frame in 10 ms and its DX12 in 39), and which wins depends on the GPU and
    driver, so each backend is timed once and the winner remembered. `backend` ("vulkan",
    "d3d12", "metal", ...) skips the measurement and uses that one."""
    adapters = [a for a in wgpu.gpu.enumerate_adapters_sync()
                if a.info["adapter_type"] != "CPU" and a.info["backend_type"] != "OpenGL"]
    if backend:
        adapters = [a for a in adapters if a.info["backend_type"].lower() == backend.lower()]
        if not adapters:
            raise ValueError(f"no {backend} adapter; this machine has "
                             f"{', '.join(_identity(a) for a in wgpu.gpu.enumerate_adapters_sync())}")
    if not adapters:
        adapters = [wgpu.gpu.request_adapter_sync(power_preference="high-performance")]
    key = hashlib.sha1("\n".join(program["shaders"]).encode()).hexdigest()[:12]
    names = sorted(_identity(a) for a in adapters)
    known = _choices(choices).get(key, {})
    if len(adapters) == 1 or set(known.get("measured", {})) == set(names):
        best = known.get("best") if len(adapters) > 1 else names[0]
        adapter = next((a for a in adapters if _identity(a) == best), adapters[0])
        return Model(_device(adapter), program, weights, **options), known

    measured, built = {}, {}
    for adapter in adapters:
        try:
            built[_identity(adapter)] = Model(_device(adapter), program, weights, **options)
        except Exception as exc:  # noqa: BLE001 -- a backend that cannot build it just loses
            measured[_identity(adapter)] = {"error": str(exc)[:200]}
    if not built:
        raise RuntimeError(f"no backend could build this model: {measured}")
    # Rounds alternate between backends and each keeps its best, so a moment when something
    # else holds the GPU cannot decide the choice.
    best_ms = dict.fromkeys(built, math.inf)
    for _ in range(3):
        for name, model in built.items():
            model.frame()
            model.read()
            started = time.perf_counter()
            for _ in range(frames):
                model.frame()
            model.read()
            best_ms[name] = min(best_ms[name], (time.perf_counter() - started) / frames * 1000)
    measured.update({n: {"ms": round(ms, 2)} for n, ms in best_ms.items()})
    best = min(best_ms, key=best_ms.get)
    for name, model in built.items():
        if name != best:
            model.destroy()
    report = {"best": best, "measured": measured}
    saved = _choices(choices)
    saved[key] = report
    choices.parent.mkdir(parents=True, exist_ok=True)
    choices.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return built[best], report


def _choices(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _device(adapter):
    limits = {k: adapter.limits[k] for k in (
        "max-buffer-size", "max-storage-buffer-binding-size", "max-storage-buffers-per-shader-stage")}
    features = [f for f in ("timestamp-query",) if f in adapter.features]
    return adapter.request_device_sync(required_features=features, required_limits=limits)


def _words(data: bytes) -> bytes:
    """GPU uploads come in whole 4-byte words."""
    return data if len(data) % 4 == 0 else data + bytes(4 - len(data) % 4)
