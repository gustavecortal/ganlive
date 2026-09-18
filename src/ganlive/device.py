"""Accelerator abstraction."""

from __future__ import annotations

import contextlib
import functools
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

Backend = Literal["xpu", "cuda", "mps", "cpu"]


def detect_backend() -> Backend:
    """Pick the best available accelerator. ``GANLIVE_DEVICE`` overrides."""
    forced = os.environ.get("GANLIVE_DEVICE", "").strip().lower()
    if forced:
        if forced not in ("xpu", "cuda", "mps", "cpu"):
            raise ValueError(f"GANLIVE_DEVICE={forced!r} is not one of xpu/cuda/mps/cpu")
        return forced  # type: ignore[return-value]

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class DeviceCaps:
    """What the accelerator can actually do, measured rather than assumed."""

    backend: Backend
    name: str
    total_memory_gb: float
    has_fp64: bool
    supports_bf16: bool
    supports_gradscaler: bool
    autocast_dtype: torch.dtype | None
    weight_dtype: torch.dtype
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def device(self) -> torch.device:
        return torch.device(self.backend)

    @property
    def is_xpu(self) -> bool:
        return self.backend == "xpu"

    def summary(self) -> str:
        lines = [
            f"backend          : {self.backend}",
            f"device           : {self.name}",
            f"memory           : {self.total_memory_gb:.1f} GB",
            f"fp64             : {'yes' if self.has_fp64 else 'NO'}",
            f"bf16             : {'yes' if self.supports_bf16 else 'no'}",
            f"GradScaler (fp16): {'usable' if self.supports_gradscaler else 'UNUSABLE'}",
            f"autocast dtype   : {self.autocast_dtype}",
            f"weight dtype     : {self.weight_dtype}",
        ]
        lines += [f"note             : {n}" for n in self.notes]
        return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def get_caps() -> DeviceCaps:
    """Probe the active accelerator once and cache the result."""
    backend = detect_backend()
    notes: list[str] = []

    if backend == "xpu":
        p = torch.xpu.get_device_properties(0)
        has_fp64 = bool(getattr(p, "has_fp64", 0))
        bf16 = bool(torch.xpu.is_bf16_supported())
        if not has_fp64:
            notes.append(
                "No fp64 on this GPU -> torch.amp.GradScaler is unusable, so fp16 mixed "
                "precision is not an option. Training runs bf16 autocast (needs no scaler)."
            )
        if not bf16:
            notes.append("bf16 unsupported and fp16 unusable: falling back to fp32 (slow).")
        return DeviceCaps(
            backend=backend,
            name=p.name,
            total_memory_gb=p.total_memory / (1024**3),
            has_fp64=has_fp64,
            supports_bf16=bf16,
            supports_gradscaler=has_fp64,
            autocast_dtype=torch.bfloat16 if bf16 else None,
            weight_dtype=torch.bfloat16 if bf16 else torch.float32,
            notes=tuple(notes),
        )

    if backend == "cuda":
        p = torch.cuda.get_device_properties(0)
        bf16 = torch.cuda.is_bf16_supported()
        if not bf16:
            notes.append("Pre-Ampere GPU: bf16 unavailable, using fp16 + GradScaler.")
        return DeviceCaps(
            backend=backend,
            name=p.name,
            total_memory_gb=p.total_memory / (1024**3),
            has_fp64=True,
            supports_bf16=bf16,
            supports_gradscaler=True,
            autocast_dtype=torch.bfloat16 if bf16 else torch.float16,
            weight_dtype=torch.bfloat16 if bf16 else torch.float16,
            notes=tuple(notes),
        )

    if backend == "mps":
        notes.append("MPS: bf16 coverage is patchy; running fp32 without autocast for safety.")
        return DeviceCaps(
            backend=backend, name="Apple MPS", total_memory_gb=0.0, has_fp64=False,
            supports_bf16=False, supports_gradscaler=False,
            autocast_dtype=None, weight_dtype=torch.float32, notes=tuple(notes),
        )

    notes.append("CPU only. Fine for curation and metrics; training will be impractically slow.")
    return DeviceCaps(
        backend="cpu", name=platform.processor() or "cpu", total_memory_gb=0.0,
        has_fp64=True, supports_bf16=True, supports_gradscaler=False,
        autocast_dtype=None, weight_dtype=torch.float32, notes=tuple(notes),
    )


def _mod(name: str | None = None):
    """torch submodule for a named device (xpu, cuda, mps), or None for the cpu. `mps` has
    `synchronize` like the other two; the functions below `hasattr` for what it lacks."""
    b = (name or detect_backend()).split(":")[0]
    if b in ("xpu", "cuda", "mps"):
        return getattr(torch, b, None)
    return None


def streams(name: str | None = None):
    """The backend module that has `Stream`, `Event` and the `stream(...)` context -- xpu and
    cuda -- or None where frames cannot be queued on a second stream at all."""
    m = _mod(name)
    return m if m is not None and hasattr(m, "Stream") else None


def playback_dtype(device: str | torch.device) -> torch.dtype:
    """The precision the instrument *plays* a model in when the caller did not say.

    Half on every accelerator, single on the CPU, where a half convolution is emulated and
    runs slower than the float32 it emulates. A different question from `DeviceCaps`, which
    answers for training -- there `mps` is fp32 because bf16 autocast is patchy, which says
    nothing about inference in fp16."""
    return torch.float32 if str(device).split(":")[0] == "cpu" else torch.float16


_ALLOCATOR = ("memory_allocated", "memory_reserved", "max_memory_allocated",
              "max_memory_reserved")


def memory_report(name: str | None = None) -> dict[str, float] | None:
    """The allocator's numbers in GB -- `allocated_gb`, `max_reserved_gb`, ... -- for whichever
    of the four this backend keeps, or None where it keeps none."""
    m = _mod(name)
    out = {key.replace("memory_", "") + "_gb": round(getattr(m, key)() / 1e9, 3)
           for key in _ALLOCATOR if hasattr(m, key)}
    return out or None


def synchronize(name: str | None = None) -> None:
    """Wait for the device to finish. Pass the device a copy was issued to; see `_mod`."""
    m = _mod(name)
    if m is not None:
        m.synchronize()


def empty_cache() -> None:
    m = _mod()
    if m is not None:
        m.empty_cache()


def reset_peak_memory() -> None:
    m = _mod()
    if m is not None and hasattr(m, "reset_peak_memory_stats"):
        m.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    return (memory_report() or {}).get("max_allocated_gb", 0.0)


def autocast(enabled: bool = True):
    """Autocast context for the active backend in the only dtype that works."""
    caps = get_caps()
    if not enabled or caps.autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=caps.backend, dtype=caps.autocast_dtype)


def make_grad_scaler(enabled: bool = True):
    """A GradScaler when one is both needed and usable, else a no-op stand-in."""
    caps = get_caps()
    needs_scaling = caps.autocast_dtype is torch.float16
    if enabled and needs_scaling and caps.supports_gradscaler:
        return torch.amp.GradScaler(caps.backend)
    return _NullScaler()


class _NullScaler:
    """Duck-typed GradScaler that does nothing."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self, new_scale=None):
        return None

    def get_scale(self) -> float:
        return 1.0

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict) -> None:
        return None


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = _mod()
    if m is not None and hasattr(m, "manual_seed_all"):
        m.manual_seed_all(seed)


_HIGH_PRIORITY_CLASS = 0x00000080


def prioritise_gpu_feeder(pin_p_cores: bool = True, verbose: bool = True,
                          raise_priority: bool = True) -> dict:
    """Keep this process's GPU-submission thread on a performance core, at high priority."""
    import ctypes
    import os
    import sys

    result: dict[str, object] = {"platform": sys.platform, "priority": None,
                                 "affinity_mask": None, "note": None}
    if sys.platform != "win32":
        result["note"] = "not Windows; nothing to do"
        return result

    if raise_priority:
        try:
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.GetCurrentProcess.argtypes = []
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass.restype = ctypes.c_int
            if k32.SetPriorityClass(k32.GetCurrentProcess(), _HIGH_PRIORITY_CLASS):
                result["priority"] = "HIGH_PRIORITY_CLASS"
            else:
                err = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
                result["note"] = f"SetPriorityClass failed (GetLastError={err})"
        except Exception as exc:  # noqa: BLE001 - pragma: no cover, tuning is best-effort
            result["note"] = f"priority unchanged: {exc}"

    if pin_p_cores:
        try:
            import psutil  # optional: only needed to count *physical* cores

            # P-cores without asking Windows for a core's kind: with `p` hyperthreaded performance cores and
            # `e` efficiency cores, `logical = 2p + e` and `physical = p + e`, so `p = logical - physical`,
            # and Windows enumerates the P-core threads first.
            logical, physical = os.cpu_count() or 0, psutil.cpu_count(logical=False) or 0
            p_cores = (logical - physical) if logical > physical else 0
            fast = list(range(2 * p_cores))
            if 0 < len(fast) < logical:
                psutil.Process().cpu_affinity(fast)
                result["affinity_mask"] = hex((1 << (2 * p_cores)) - 1)
                result["cores"] = fast
            else:
                result["note"] = "not a hybrid CPU; affinity left alone"
        except Exception as exc:  # noqa: BLE001 - psutil's own errors are not enumerable
            result["note"] = f"affinity unchanged: {exc}"

    if verbose:
        cores = result.get("cores")
        said = (f"pinned to {len(cores)} P-core threads (0-{cores[-1]}) of {os.cpu_count()}"
                if cores else "left where the scheduler puts it")
        print(f"cpu: {said}"
              + (f", priority {result['priority']}" if result["priority"] else "")
              + (f"  ({result['note']})" if result["note"] else ""), flush=True)
    return result


def use_stable_inductor_cache(root: str = "data/cache/inductor") -> str:
    """Keep Inductor's compiled kernels somewhere that survives between sessions."""
    existing = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if existing and not os.path.basename(existing).startswith("torchinductor_"):
        return existing
    path = os.path.abspath(root)
    os.makedirs(path, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    return path


_LLVM_BIN = Path("C:/Program Files/LLVM/bin/clang.exe")


def host_c_compiler() -> str | None:
    """A host C compiler Triton can build launcher stubs with, or `None`."""
    from shutil import which

    for name in ("clang", "gcc", "cl"):
        found = which(name)
        if found:
            return found
    if _LLVM_BIN.exists():
        return str(_LLVM_BIN)
    from . import _msvc as msvc

    return which("cl") if msvc.activate(verbose=False) else None


def prepare_inductor(threads: int = 8, spawn_pool: bool = False) -> str:
    """Everything that has to happen on this machine before `torch.compile` is called."""
    from . import _msvc as msvc

    msvc.activate(verbose=False)
    use_parallel_inductor_compile(threads, spawn_pool)
    return use_stable_inductor_cache()


def use_parallel_inductor_compile(threads: int = 8, spawn_pool: bool = False) -> int:
    """Compile Inductor's kernels on several cores."""
    from torch._inductor import config as inductor_config

    if sys.platform == "win32":
        if not spawn_pool:
            return inductor_config.compile_threads or 1
        if compiler := host_c_compiler():
            os.environ.setdefault("CC", compiler)
        os.environ["TORCHINDUCTOR_WORKER_START"] = "spawn"
        inductor_config.worker_start_method = "spawn"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
        inductor_config.compile_threads = threads
        return threads
    existing = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if existing:
        threads = int(existing)
    else:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
    inductor_config.compile_threads = threads
    return threads


def cpu_generator(seed: int | None = None) -> torch.Generator:
    """A CPU generator for reproducible sampling."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    return g


def flat_norm(tensors) -> torch.Tensor | None:
    """The L2 norm over a list of tensors, as one 0-dim tensor left on their own device."""
    if not tensors:
        return None
    per = [torch.linalg.vector_norm(t.detach(), 2) for t in tensors]
    return torch.linalg.vector_norm(torch.stack(per), 2)


def clip_grads(params, max_norm: float):
    """Clip a parameter list in place. Returns the norm it had **before** clipping."""
    live = [p for p in params if p.grad is not None]
    if not live:
        return None
    total = flat_norm([p.grad for p in live])
    if max_norm:
        torch.nn.utils.clip_grads_with_norm_(live, max_norm, total)
    return total


def other_gpu_pythons() -> list[int]:
    """PIDs of `python.exe` processes that are not this one or one of its ancestors."""
    if sys.platform != "win32":
        return []
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "ForEach-Object { \"$($_.ProcessId):$($_.ParentProcessId)\" }"],
        capture_output=True, text=True).stdout.strip()
    parents: dict[int, int] = {}
    for line in out.splitlines():
        pid, _, ppid = line.strip().partition(":")
        if pid.isdigit() and ppid.isdigit():
            parents[int(pid)] = int(ppid)

    mine, seen = os.getpid(), set()
    while mine and mine not in seen:
        seen.add(mine)
        mine = parents.get(mine, 0)
    return sorted(set(parents) - seen)


def refuse_if_gpu_busy(what: str) -> bool:
    """False, with an explanation, while anything else holds the card."""
    others = other_gpu_pythons()
    if others:
        print(f"REFUSING {what}: python already running (pids "
              f"{', '.join(str(p) for p in others)}). Stop them and let the driver release.")
        return False
    return True


def host_ram_free_gb() -> float:
    """Free **host** RAM in GB, or NaN where it cannot be read."""
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1e9
    except Exception:  # noqa: BLE001 - no kernel32, or a ctypes ABI mismatch; fall through
        try:
            import psutil

            return psutil.virtual_memory().available / 1e9
        except Exception:  # noqa: BLE001 - free memory is a report, never a decision
            return float("nan")
