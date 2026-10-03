"""Which accelerator to run on, and the few per-backend calls the rest of the code needs.

Also process tuning for live play (priority and CPU affinity), and checks for other processes
holding the card.
"""

from __future__ import annotations

import contextlib
import os
import sys
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
    """The precision the instrument plays a model in when the caller did not say.

    Half on every accelerator; single on the CPU, where half-precision convolution is
    emulated and runs slower than float32."""
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


def peak_memory_gb() -> float:
    return (memory_report() or {}).get("max_allocated_gb", 0.0)


_HIGH_PRIORITY_CLASS = 0x00000080


def _raise_priority() -> str | None:
    """Put this process above the desktop's other work, in whatever the platform calls it."""
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetCurrentProcess.argtypes = []
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.restype = ctypes.c_int
        if k32.SetPriorityClass(k32.GetCurrentProcess(), _HIGH_PRIORITY_CLASS):
            return "HIGH_PRIORITY_CLASS"
        raise OSError(f"SetPriorityClass failed (GetLastError={ctypes.get_last_error()})")
    # POSIX: a lower nice number is higher priority, and going below 0 needs privileges.
    # -5 succeeds under `sudo` or a raised RLIMIT_NICE, and raises otherwise.
    os.nice(-5)
    return f"nice {os.nice(0)}"


def prioritise_gpu_feeder() -> None:
    """Raise this process's priority and, on a hybrid CPU, restrict the whole process to its
    performance cores, so the thread feeding the GPU is never parked on an efficiency core.

    Best-effort: whatever the platform refuses is left alone, and one line says what was done."""
    priority, note, cores = None, None, None
    try:
        priority = _raise_priority()
    except Exception as exc:  # noqa: BLE001 - tuning is best-effort everywhere
        note = f"priority unchanged: {exc}"

    try:
        import psutil

        # P-cores without asking the OS for a core's kind: with `p` hyperthreaded performance cores
        # and `e` efficiency cores, `logical = 2p + e` and `physical = p + e`, so `p = logical -
        # physical`, and both Windows and Linux enumerate the P-core threads first.
        logical, physical = os.cpu_count() or 0, psutil.cpu_count(logical=False) or 0
        p_cores = (logical - physical) if logical > physical else 0
        fast = list(range(2 * p_cores))
        if not hasattr(psutil.Process(), "cpu_affinity"):
            # macOS has no affinity API at all, hybrid silicon or not.
            note = f"no affinity control on {sys.platform}"
        elif 0 < len(fast) < logical:
            psutil.Process().cpu_affinity(fast)
            cores = fast
        else:
            note = "not a hybrid CPU; affinity left alone"
    except Exception as exc:  # noqa: BLE001 - psutil's own errors are not enumerable
        note = f"affinity unchanged: {exc}"

    said = (f"pinned to {len(cores)} P-core threads (0-{cores[-1]}) of {os.cpu_count()}"
            if cores else "left where the scheduler puts it")
    print(f"cpu: {said}"
          + (f", priority {priority}" if priority else "")
          + (f"  ({note})" if note else ""), flush=True)


def other_gpu_pythons() -> list[int] | None:
    """PIDs of python processes that are not this one or one of its ancestors.

    `None` when `psutil` is not installed -- which a caller must report rather than read as
    "nothing is running". Even with it this is "none found": another user's processes are
    invisible, so it is a courtesy, not a lock."""
    try:
        import psutil
    except ImportError:
        return None

    parents = {}
    for p in psutil.process_iter(["name", "ppid"]):
        # A process can exit between the listing and the read, and another user's is not ours
        # to see. Either way it is not a process we would ask about.
        with contextlib.suppress(psutil.Error):
            if (p.info["name"] or "").lower().startswith("python"):
                parents[p.pid] = p.info["ppid"]
    mine, seen = os.getpid(), set()
    while mine and mine not in seen:
        seen.add(mine)
        mine = parents.get(mine, 0)
    return sorted(set(parents) - seen)


def host_ram_free_gb() -> float:
    """Host RAM available right now, in GB, or `nan` where it cannot be asked."""
    try:
        import psutil
    except ImportError:
        return float("nan")
    return round(psutil.virtual_memory().available / 1e9, 2)


def refuse_if_gpu_busy(what: str) -> bool:
    """False, with an explanation, while anything else holds the card."""
    others = other_gpu_pythons()
    if others is None:
        print(f"cannot tell whether anything else holds the card (no psutil); "
              f"{what} is going ahead. `pip install psutil` to have this checked.")
        return True
    if others:
        print(f"REFUSING {what}: python already running (pids "
              f"{', '.join(str(p) for p in others)}). Stop them and let the driver release.")
        return False
    return True


