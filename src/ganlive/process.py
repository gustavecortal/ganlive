"""This process and the others on the machine: tuning for live play (priority and CPU
affinity), and checks for other processes holding the card. Torch-free."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys

_HIGH_PRIORITY_CLASS = 0x00000080


def _raise_priority() -> str | None:
    """Put this process above the desktop's other work, in whatever the platform calls it."""
    if sys.platform == "win32":
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


#: Words in a command line that mark a process that drives the card, where the platform does
#: not list a process's loaded libraries (macOS).
TORCH_WORDS = ("ganlive", "gantrain", "torch")


def _holds_torch(process) -> bool:
    """Whether a python process has PyTorch loaded: read off its memory maps where the platform
    lists them (Linux, Windows), off its command line where it does not (macOS). An editor's
    linter server is a python process too, and holds no card."""
    maps = getattr(process, "memory_maps", None)
    if maps is not None:
        return any("torch" in (m.path or "").lower() for m in maps())
    return any(word in " ".join(process.cmdline()).lower() for word in TORCH_WORDS)


def other_gpu_pythons() -> list[int] | None:
    """PIDs of python processes with PyTorch loaded that are not this one or one of its
    ancestors.

    `None` when `psutil` is not installed -- which a caller must report rather than read as
    "nothing is running". Even with it this is "none found": another user's processes are
    invisible, so it is a courtesy, not a lock."""
    try:
        import psutil
    except ImportError:
        return None

    parents, processes = {}, {}
    for p in psutil.process_iter(["name", "ppid"]):
        # A process can exit between the listing and the read, and another user's is not ours
        # to see. Either way it is not a process we would ask about.
        with contextlib.suppress(psutil.Error):
            if (p.info["name"] or "").lower().startswith("python"):
                parents[p.pid], processes[p.pid] = p.info["ppid"], p
    mine, seen = os.getpid(), set()
    while mine and mine not in seen:
        seen.add(mine)
        mine = parents.get(mine, 0)
    holding = []
    for pid in sorted(set(parents) - seen):
        with contextlib.suppress(psutil.Error):
            if _holds_torch(processes[pid]):
                holding.append(pid)
    return holding


def host_ram_free_gb() -> float:
    """Host RAM available right now, in GB, or `nan` where it cannot be asked."""
    try:
        import psutil
    except ImportError:
        return float("nan")
    return round(psutil.virtual_memory().available / 1e9, 2)


def refuse_if_gpu_busy(device: str, what: str) -> bool:
    """False, with an explanation, while anything else holds the card. The CPU is never refused."""
    if device.lower() == "cpu":
        return True
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
