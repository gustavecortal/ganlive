"""Run a ganlive engine program (from `ganlive convert`) through wgpu-py: the same shaders and
dispatches `runner.mjs` runs in a browser, on Vulkan, Metal or DX12.

    model = Model.load("runs/engine/lichen")   # on this machine's fastest backend and plans
    model.set_latent(z); model.set_settings(k)
    model.frame()                              # submits one frame
    pixels = model.read()                      # (H, W, 4) uint8, or (3, H, W) float32 for f32
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import wgpu

from ganlive.checkpoints import MANIFEST, WEIGHTS
from ganlive.engine.codegen import FORMAT
from ganlive.engine.compile import compile_manifest
from ganlive.engine.probe import PROBE_LEVELS, probe_error
from ganlive.engine.stamp import code_hash
from ganlive.files import remember

USAGE = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
#: Lets a shader write straight into a buffer the host maps (`EngineStage.nv12_bytes`). wgpu
#: warns that every buffer might then live in host memory, but only those created mappable do.
MAPPABLE = "mappable-primary-buffers"
logging.getLogger("wgpu").addFilter(lambda record: "MAPPABLE_PRIMARY_BUFFERS" not in record.getMessage())


def default_device(fallback: bool = False):
    """The high-performance adapter's device (or, with `fallback`, the CPU one's)."""
    return _device(wgpu.gpu.request_adapter_sync(power_preference="high-performance",
                                                 force_fallback_adapter=fallback))


class BuildTimeout(RuntimeError):
    """A backend compiles this program slower than the time it was given."""


class Model:
    """One loaded program: its buffers, compiled pipelines and the steps of a frame.

    `pipelines` (shader code -> pipeline) shares compiled pipelines between models on one
    device. A build still compiling at `deadline` (a `time.perf_counter()` value) stops with
    `BuildTimeout` before its next shader: one shader's compile is not interrupted. A model
    that fails to build frees what it made."""

    def __init__(self, device, program: dict, weights: bytes, *, pipelines: dict | None = None,
                 deadline: float | None = None) -> None:
        if program.get("format") != FORMAT:
            raise ValueError(f"a {program.get('format')!r} program; this ganlive plays {FORMAT!r}")
        if len(weights) != program["buffers"]["P"]["size"]:
            raise ValueError(f"the weights are {len(weights)} bytes and the program expects "
                             f"{program['buffers']['P']['size']}: they come from different conversions")
        self.device, self.program = device, program
        self.height, self.width = program["height"], program["width"]
        self.buffers: dict[str, wgpu.GPUBuffer] = {}
        try:
            self._build(weights, pipelines, deadline)
        except BaseException:
            self.destroy()
            raise

    def _build(self, weights: bytes, pipelines: dict | None, deadline: float | None) -> None:
        device, program = self.device, self.program
        limit = min(device.limits["max-storage-buffer-binding-size"], device.limits["max-buffer-size"])
        for name, spec in program["buffers"].items():
            if spec["size"] > limit:
                raise ValueError(f"{name}: a {spec['size'] / 2**20:.0f} MiB buffer is over this "
                                 f"device's {limit / 2**20:.0f} MiB limit")
            buf = self.buffers[name] = device.create_buffer(size=spec["size"], usage=USAGE)
            init = spec.get("init")
            if init == "weights":
                device.queue.write_buffer(buf, 0, weights)
            elif init == "ones":
                device.queue.write_buffer(buf, 0, np.ones(spec["size"] // 4, np.float32))
            elif init == "words":
                device.queue.write_buffer(buf, 0, np.array(spec["words"], np.uint32))
        self.output = self.buffers["out"]
        pipelines = {} if pipelines is None else pipelines

        def compiled(spec):
            code = program["shaders"][spec["shader"]]
            if code not in pipelines:
                if deadline is not None and time.perf_counter() > deadline:
                    raise BuildTimeout(f"{device.adapter.info['backend_type']} compiles too slowly")
                pipelines[code] = device.create_compute_pipeline(layout="auto", compute={
                    "module": device.create_shader_module(code=code), "entry_point": "main"})
            pipeline = pipelines[code]
            bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {"binding": i, "resource": {"buffer": self.buffers[n], "offset": 0,
                                            "size": self.buffers[n].size}}
                for i, n in enumerate(spec["bind"])])
            return pipeline, bind, spec["groups"]

        self.steps = [compiled(s) for s in program["steps"]]
        encoder = device.create_command_encoder()          # once: the noise
        self._record(encoder, [compiled(s) for s in program["load"]])
        device.queue.submit([encoder.finish()])

    @classmethod
    def load(cls, folder, device=None, **options) -> Model:
        """The model in `folder`, checked against its probe, on `device`, or else on the
        fastest backend (`fastest`), each with the plans tuned for it (`ganlive tune`)."""
        manifest, weights = read_folder(folder)
        if device is None:
            return fastest(manifest, weights, **options)[0]
        return built(device, manifest, weights, tuned_plans(manifest, device.adapter), own_device=False)

    def destroy(self) -> None:
        """Free this model's buffers. Its device may be shared, so it stays."""
        for buf in self.buffers.values():
            buf.destroy()

    def set_latent(self, z) -> None:
        self.device.queue.write_buffer(self.buffers["Z"], 0, np.ascontiguousarray(z, np.float32))

    def set_settings(self, k) -> None:
        self.device.queue.write_buffer(self.buffers["K"], 0, np.ascontiguousarray(k, np.float32))

    def encode(self, encoder) -> None:
        """Records one frame into `encoder`."""
        self._record(encoder, self.steps)

    def frame(self) -> None:
        encoder = self.device.create_command_encoder()
        self.encode(encoder)
        self.device.queue.submit([encoder.finish()])

    def read(self) -> np.ndarray:
        """The last frame, waited for: (H, W, 4) uint8 RGBA (whichever order the program
        draws), or (3, H, W) float32 in [-1, 1]."""
        data = self.device.queue.read_buffer(self.output)
        if self.program["output"] == "f32":
            return np.frombuffer(data, np.float32).reshape(3, self.height, self.width)
        pixels = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 4)
        return pixels[..., [2, 1, 0, 3]] if self.program["output"] == "bgra8" else pixels

    def wait(self) -> None:
        """Until the frames submitted so far are drawn."""
        self.device.queue.read_buffer(self.output, 0, 4)

    def strays(self) -> float:
        """How far this backend's drawing of the program's probe is from PyTorch's (see
        `program.probe_error`). Leaves the settings at neutral."""
        probe = self.program["probe"]
        self.set_latent(np.asarray(probe["z"], np.float32))
        self.set_settings(np.ones(len(self.program["settings"]), np.float32))
        self.frame()
        return probe_error(probe, self.read())

    @staticmethod
    def _record(encoder, steps) -> None:
        compute = encoder.begin_compute_pass()
        for pipeline, bind, (x, y, z) in steps:
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, bind)
            compute.dispatch_workgroups(x, y, z)
        compute.end()


def read_folder(folder) -> tuple[dict, bytes]:
    """The manifest and weights of the engine model in `folder`."""
    folder = Path(folder)
    return json.loads((folder / MANIFEST).read_text(encoding="utf-8")), (folder / WEIGHTS).read_bytes()


def cache_dir() -> Path:
    """Where this user's machine-specific measurements are kept."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ganlive"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ganlive"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ganlive"


#: Backends whose compilers are fast are built first, so that they set the bar the others
#: must build within (wgpu's D3D12 compiles with FXC, minutes for a large StyleGAN2).
FIRST = ("Vulkan", "Metal")
#: Seconds another backend may take to build a program while `fastest` measures it.
BUDGET = 20.0
#: What a desktop program draws: the BGRA8 an SDL texture and `EngineStage` take.
OUTPUT = "bgra8"


def model_key(manifest: dict) -> str:
    """What `choices` keeps a model's measurements under: the manifest and the shader
    generator's code, so that a change to either measures and tunes again."""
    digest = hashlib.sha1(json.dumps(manifest, sort_keys=True).encode())
    digest.update(code_hash("ganlive.engine.compile").encode())
    return digest.hexdigest()[:12]


def adapter_name(adapter) -> str:
    """An adapter as its backend, GPU and driver, the name tuned plans are kept under: two
    identical cards share them."""
    return _names([adapter])[0]


def tuned_plans(manifest: dict, adapter, choices: Path | None = None) -> dict | None:
    """The plans `ganlive tune` found for `manifest` on `adapter`, if it did."""
    saved = _choices(choices or cache_dir() / "backends.json").get(model_key(manifest), {})
    return saved.get("plans", {}).get(adapter_name(adapter))


def built(device, manifest: dict, weights: bytes, plans: dict | None = None, *, own_device: bool,
          deadline: float | None = None) -> Model:
    """`manifest` built on `device` and checked against its probe (`checked`): with `plans`,
    or with the defaults if those no longer compile or draw right."""
    if plans:
        try:
            return checked(device, compile_manifest(manifest, plans, OUTPUT), weights,
                           own_device=False, deadline=deadline)
        except BuildTimeout:
            if own_device:
                device.destroy()
            raise
        except (ValueError, RuntimeError, wgpu.GPUError):
            pass                            # plans from an older generator: the defaults
    return checked(device, compile_manifest(manifest, None, OUTPUT), weights,
                   own_device=own_device, deadline=deadline)


def fastest(manifest: dict, weights: bytes, *, backend: str | None = None, frames: int = 20,
            choices: Path | None = None, budget: float = BUDGET) -> tuple[Model, dict]:
    """The model built on whichever GPU backend draws it right and fastest here, and what was
    measured.

    One card can differ between backends (on an Arc A770, wgpu-py's DX12 ran amber in 8.1 ms
    and its Vulkan in 9.4; DX12 compiled with DXC ran it in 114), and the order depends on the
    GPU and driver, so each backend that draws the program's probe is timed and the winner
    remembered in `choices`. The first backend in `FIRST` order builds without a limit, the
    others within `budget` seconds. `backend` ("vulkan", "d3d12", "metal", ...) uses that
    backend without measuring."""
    choices = choices or cache_dir() / "backends.json"
    adapters = sorted(_gpus(backend), key=lambda a: a.info["backend_type"] not in FIRST)
    names = _names(adapters)
    key = model_key(manifest)
    saved = _choices(choices)
    known = saved.get(key, {})
    plans = known.get("plans", {})

    def build(adapter, deadline=None):
        return built(_device(adapter), manifest, weights, plans.get(adapter_name(adapter)),
                     own_device=True, deadline=deadline)

    if backend or len(adapters) == 1:
        return build(adapters[0]), {"best": names[0]}
    if sorted(known.get("measured", ())) == sorted(names) and known.get("best") in names:
        try:
            return build(adapters[names.index(known["best"])]), known
        except (RuntimeError, ValueError, wgpu.GPUError):
            pass                            # the remembered backend fails today: measure again

    measured, made, slow = {}, {}, []
    for i, (name, adapter) in enumerate(zip(names, adapters, strict=True)):
        try:
            made[name] = build(adapter, time.perf_counter() + budget if i else None)
        except BuildTimeout:
            measured[name] = {"error": f"builds in over {budget:.0f} s"}
            slow.append((name, adapter))
        except (RuntimeError, ValueError, wgpu.GPUError) as exc:
            measured[name] = {"error": str(exc)[:200]}
    if not made and slow:                   # nothing else draws it: wait for a slow one
        name, adapter = slow.pop(0)
        try:
            made[name] = build(adapter)
            del measured[name]
        except (RuntimeError, ValueError, wgpu.GPUError) as exc:
            measured[name] = {"error": str(exc)[:200]}
    # Rounds alternate between backends and each keeps its best, so a moment when something
    # else holds the GPU cannot decide the choice. A backend that fails mid-way drops out.
    best_ms = dict.fromkeys(made, math.inf)
    for _ in range(3):
        for name in list(best_ms):
            model = made[name]
            try:
                model.frame()
                model.wait()
                started = time.perf_counter()
                for _ in range(frames):
                    model.frame()
                model.wait()
            except (RuntimeError, wgpu.GPUError) as exc:
                measured[name] = {"error": str(exc)[:200]}
                del made[name], best_ms[name]
                continue
            best_ms[name] = min(best_ms[name], (time.perf_counter() - started) / frames * 1000)
    if not made:
        raise RuntimeError(f"no backend draws this model: {measured}")
    measured.update({n: {"ms": round(ms, 2)} for n, ms in best_ms.items()})
    best = min(best_ms, key=best_ms.get)
    for name, model in made.items():
        if name != best:                    # each was built on a device of its own
            model.destroy()
            model.device.destroy()
    report = {"best": best, "measured": measured}
    # A failure may be passing (memory held elsewhere), so only a clean measurement is kept.
    # A slow compiler stays slow, and is not waited for again.
    if len(best_ms) + len(slow) == len(names):
        saved[key] = {**known, **report}
        remember(choices, json.dumps(saved, indent=2))
    return made[best], report


def checked(device, program, weights, *, own_device: bool, deadline: float | None = None,
            pipelines: dict | None = None) -> Model:
    """A model built on `device`, refused if it does not draw the program's probe. A device the
    model owns goes with it, and a shared one stays."""
    try:
        model = Model(device, program, weights, deadline=deadline, pipelines=pipelines)
    except BaseException:
        if own_device:
            device.destroy()
        raise
    off = model.strays() if "probe" in program else 0.0
    if off > PROBE_LEVELS:
        model.destroy()
        if own_device:
            device.destroy()
        raise RuntimeError(f"draws the probe {off:.1f} levels off on "
                           f"{device.adapter.info['backend_type']}")
    return model


def _gpus(backend: str | None) -> list:
    """The GPU adapters worth measuring: not the CPU one, nor OpenGL, whose compute is the
    least complete, or any adapter if nothing else is there."""
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
    """A device with the adapter's largest buffers: a 3072x2048 layer is over the defaults."""
    limits = {k: adapter.limits[k] for k in (
        "max-buffer-size", "max-storage-buffer-binding-size", "max-storage-buffers-per-shader-stage")}
    features = [MAPPABLE] if MAPPABLE in adapter.features else []
    return adapter.request_device_sync(required_limits=limits, required_features=features)
