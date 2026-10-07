"""Run a ganlive engine program (from `ganlive convert`) through wgpu-py: the same shaders and
dispatches `runner.mjs` runs in a browser, on Vulkan, Metal or DX12.

    model = Model.load("models/lichen")   # on this machine's fastest backend
    model.set_latent(z); model.set_settings(k)
    model.frame()                         # submits one frame
    pixels = model.read()                 # (H, W, 4) uint8, or (3, H, W) float32 with output="f32"
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import wgpu

from ganlive.engine.program import PROBE_LEVELS, box_means

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
        if len(weights) != program["bytes"]:
            raise ValueError(f"the weights are {len(weights)} bytes and the program expects "
                             f"{program['bytes']}: they come from different conversions")
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
                device.queue.write_buffer(buf, 0, weights)
            elif init == "ones":
                device.queue.write_buffer(buf, 0, np.ones(spec["size"] // 4, np.float32))
            elif init == "noise":
                layer = name.removeprefix("noise.")
                if layer in noise:
                    device.queue.write_buffer(buf, 0, np.ascontiguousarray(noise[layer], np.float32))
                else:
                    self._seed(buf, spec["size"] // 4, spec["seed"])

        pipelines: dict[int, object] = {}
        self.steps = []
        for spec in [*program["steps"], last["step"]]:
            index = spec["shader"]
            if index not in pipelines:
                module = device.create_shader_module(code=program["shaders"][index])
                pipelines[index] = device.create_compute_pipeline(
                    layout="auto", compute={"module": module, "entry_point": "main"})
            pipeline = pipelines[index]
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

    def strays(self) -> float:
        """How far this backend's picture of the program's probe is from PyTorch's, in 8-bit
        levels averaged over the probe's blocks. Leaves the settings at neutral."""
        probe = self.program["probe"]
        self.set_latent(np.asarray(probe["z"], np.float32))
        self.set_settings(np.ones(len(self.program["settings"]), np.float32))
        self.frame()
        image = self.read()
        if self.format != "f32":
            image = image[..., :3].transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
        return float(np.abs(box_means(image).ravel() - np.asarray(probe["means"])).mean() * 127.5)

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


def cache_dir() -> Path:
    """Where this user's machine-specific measurements are kept."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ganlive"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ganlive"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ganlive"


def fastest(program: dict, weights: bytes, *, backend: str | None = None, frames: int = 20,
            choices: Path | None = None, **options) -> tuple[Model, dict]:
    """The model built on whichever GPU backend draws it right and fastest here, and what was
    measured.

    One card can differ between backends (on an Arc A770, wgpu-py's DX12 runs amber in 8.1 ms
    and its Vulkan in 9.4; DX12 compiled with DXC runs it in 114), and the order depends on
    the GPU and driver, so each backend that matches the program's probe is timed and the
    winner remembered in `choices`. `backend` ("vulkan", "d3d12", "metal", ...) uses that
    backend without measuring."""
    choices = choices or cache_dir() / "backends.json"
    adapters = _gpus(backend)
    if backend:
        return _checked(adapters[0], program, weights, options), {"best": _names(adapters)[0]}
    names = _names(adapters)
    key = hashlib.sha1("\n".join(program["shaders"]).encode()).hexdigest()[:12]
    known = _choices(choices).get(key, {})
    if len(adapters) == 1 or sorted(known.get("measured", {})) == sorted(names):
        best = names[0] if len(adapters) == 1 else known.get("best")
        if best in names:
            try:
                return _checked(adapters[names.index(best)], program, weights, options), known
            except (RuntimeError, ValueError, wgpu.GPUError):
                pass                        # the remembered backend fails today: measure again

    measured, built = {}, {}
    for name, adapter in zip(names, adapters, strict=True):
        try:
            built[name] = _checked(adapter, program, weights, options)
        except (RuntimeError, ValueError, wgpu.GPUError) as exc:
            measured[name] = {"error": str(exc)[:200]}
    # Rounds alternate between backends and each keeps its best, so a moment when something
    # else holds the GPU cannot decide the choice. A backend that fails mid-way drops out.
    best_ms = dict.fromkeys(built, math.inf)
    for _ in range(3):
        for name in [n for n in best_ms if n in built]:
            model = built[name]
            try:
                model.frame()
                model.read()
                started = time.perf_counter()
                for _ in range(frames):
                    model.frame()
                model.read()
            except (RuntimeError, wgpu.GPUError) as exc:
                measured[name] = {"error": str(exc)[:200]}
                del built[name], best_ms[name]
                continue
            best_ms[name] = min(best_ms[name], (time.perf_counter() - started) / frames * 1000)
    if not built:
        raise RuntimeError(f"no backend draws this model: {measured}")
    measured.update({n: {"ms": round(ms, 2)} for n, ms in best_ms.items()})
    best = min(best_ms, key=best_ms.get)
    for name, model in built.items():
        if name != best:
            model.destroy()
    report = {"best": best, "measured": measured}
    saved = _choices(choices)
    saved[key] = report
    try:
        choices.parent.mkdir(parents=True, exist_ok=True)
        choices.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    except OSError:
        pass                                # not remembered: the next load measures again
    return built[best], report


def _checked(adapter, program, weights, options) -> Model:
    """A model built on `adapter`, refused if it does not draw the program's probe."""
    model = Model(_device(adapter), program, weights, **options)
    if "probe" in program:
        off = model.strays()
        if off > PROBE_LEVELS:
            model.destroy()
            raise RuntimeError(f"draws the probe {off:.1f} levels off")
    return model


def _gpus(backend: str | None) -> list:
    """The GPU adapters worth measuring: not the CPU one, nor OpenGL, whose compute is the
    least complete; any adapter if nothing else is there."""
    every = wgpu.gpu.enumerate_adapters_sync()
    gpus = [a for a in every if a.info["adapter_type"] != "CPU"
            and not a.info["backend_type"].startswith("OpenGL")]
    if backend:
        gpus = [a for a in every if a.info["backend_type"].lower() == backend.lower()]
        if not gpus:
            raise ValueError(f"no {backend} adapter here; there are "
                             f"{', '.join(sorted({a.info['backend_type'] for a in every}))}")
    return gpus or [wgpu.gpu.request_adapter_sync(power_preference="high-performance")]


def _names(adapters) -> list[str]:
    """Each adapter as its backend, GPU and the driver version where the backend reports one
    (backends put it in different fields), numbered when two are otherwise the same."""
    base = [" | ".join(str(a.info.get(k, "")) for k in (
        "backend_type", "device", "vendor_id", "device_id", "vendor", "description")) for a in adapters]
    return [f"{b} #{base[:i].count(b) + 1}" if base.count(b) > 1 else b for i, b in enumerate(base)]


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
