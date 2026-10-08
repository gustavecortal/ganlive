"""The commands `ganlive` dispatches to, one module each with `main(argv)`, and their shared setup.

Importing this package sets up the process before any tool's own imports run, because Python
runs a package's `__init__` before any module inside it.
"""

import argparse
import os
import sys

#: `sounddevice` reads this at import to enable ASIO, the only host API some multichannel USB
#: interfaces appear on. Ignored on other platforms.
os.environ.setdefault("SD_ENABLE_ASIO", "1")

# A Windows console defaults to cp1252, and some libraries print non-ASCII marks; replace what
# cannot be encoded rather than raise. Both streams, because a traceback goes to stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parser(name: str, doc: str) -> argparse.ArgumentParser:
    """The argument parser for `ganlive <name>`, with the tool's docstring as its description."""
    return argparse.ArgumentParser(prog=f"ganlive {name}", description=doc,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)


def add_backend(ap):
    """The `--backend` option every tool that plays a model takes."""
    return ap.add_argument("--backend", default=None, choices=("vulkan", "d3d12", "metal"),
                           help="the wgpu backend to play on. Default: the fastest this machine "
                                "measured for the model, remembered after its first load")


def add_device(ap, help: str = "default: whichever accelerator is there, else the CPU", **kw):
    """The `--device` option every tool that runs a generator takes. `kw` overrides the
    default and metavar for a tool whose devices are not torch's."""
    return ap.add_argument("--device", **{"default": None, "metavar": "xpu|cuda|mps|cpu",
                                          "help": help, **kw})
