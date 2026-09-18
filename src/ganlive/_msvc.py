"""Locate MSVC and import its environment, so Triton and Inductor can find `cl.exe`."""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) \
    / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"

_WANTED = {name.upper() for name in (
    "PATH", "INCLUDE", "LIB", "LIBPATH",
    "VCINSTALLDIR", "VCToolsInstallDir", "VCToolsVersion",
    "WindowsSdkDir", "WindowsSDKVersion", "WindowsSdkVerBinPath",
    "UCRTVersion", "UniversalCRTSdkDir", "VSINSTALLDIR",
)}


def find_vs_install() -> Path | None:
    """Path to a VS/BuildTools installation that includes the C++ toolset."""
    if not VSWHERE.exists():
        return None
    try:
        out = subprocess.run(
            [str(VSWHERE), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=120)
        path = out.stdout.strip().splitlines()
        return Path(path[0]) if path else None
    except Exception:  # noqa: BLE001 - absence is the normal case, not an error
        return None


def find_vcvars(install: Path | None = None) -> Path | None:
    install = install or find_vs_install()
    if not install:
        return None
    cand = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return cand if cand.exists() else None


def parse_set_output(text: str) -> dict[str, str]:
    """Pull the wanted variables out of `set` output, keyed upper-case."""
    env: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.upper() in _WANTED:
            env[key.upper()] = value
    return env


@functools.lru_cache(maxsize=1)
def _vcvars_environment() -> dict[str, str] | None:
    """Environment variables that `vcvars64.bat` sets, captured from a subshell."""
    vcvars = find_vcvars()
    if not vcvars:
        return None
    try:
        out = subprocess.run(
            f'call "{vcvars}" && set',
            shell=True, capture_output=True, text=True,
            errors="replace", timeout=300)
    except Exception:  # noqa: BLE001
        return None

    env = parse_set_output(out.stdout)
    if "PATH" not in env or "INCLUDE" not in env:
        return None
    return env


def activate(verbose: bool = True) -> bool:
    """Merge the MSVC environment into this process. True if `cl.exe` then runs."""
    import shutil

    if shutil.which("cl"):
        if verbose:
            print("  msvc: cl.exe already on PATH")
        return True

    env = _vcvars_environment()
    if not env:
        if verbose:
            install = find_vs_install()
            if install is None:
                print("  msvc: no Visual Studio C++ toolset found. Install with:\n"
                      "        winget install --id Microsoft.VisualStudio.2022.BuildTools "
                      '--override "--add Microsoft.VisualStudio.Workload.VCTools --quiet"')
            else:
                print(f"  msvc: found {install} but could not run vcvars64.bat")
        return False

    for key, value in env.items():
        if key.upper() == "PATH":
            os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")
        else:
            os.environ[key] = value

    ok = shutil.which("cl") is not None
    if verbose:
        print(f"  msvc: {'activated' if ok else 'vcvars ran but cl.exe still missing'}"
              f" ({env.get('VCTOOLSVERSION', '?')}, "
              f"SDK {env.get('WINDOWSSDKVERSION', '?').rstrip(chr(92))})")
    return ok


def status() -> dict:
    """Diagnostic snapshot, for reporting rather than control flow."""
    import shutil

    install = find_vs_install()
    return {
        "vswhere_present": VSWHERE.exists(),
        "vs_install": str(install) if install else None,
        "vcvars": str(find_vcvars(install)) if install else None,
        "cl_on_path": shutil.which("cl"),
        "cl_after_activate": None,
    }


if __name__ == "__main__":
    import json
    s = status()
    print(json.dumps(s, indent=2))
    print()
    ok = activate()
    import shutil
    print(f"\ncl.exe: {shutil.which('cl')}")
    if ok:
        r = subprocess.run(["cl"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr)
              else "(no version banner)")
