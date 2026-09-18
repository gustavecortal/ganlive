"""`ganlive <command>` -- one entry point for everything in `ganlive.tools`."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "play": ("play", "play a model live from a MIDI controller"),
    "latency": ("latency", "measure this machine's frame time, drift included"),
    "wire": ("wire", "is the controller talking? notes, channels, audio"),
    "learn": ("learn", "watch a controller and report which note is which pad"),
    "doctor": ("doctor", "list audio devices and meter one, to find the right input"),
    "dials": ("dials", "derive a checkpoint's latent directions and save them beside it"),
    "import-stylegan2": ("import_stylegan2", "convert an NVIDIA StyleGAN2 pickle to a playable checkpoint"),
    "export-onnx": ("export_onnx", "export a checkpoint as an ONNX graph with its dials as inputs"),
    "adopt": ("adopt", "make any ONNX graph or Hub model playable: dial it, prove it, save it"),
    "shoot-strip": ("shoot_strip", "render the control strip to a PNG, headless"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    return "usage: ganlive <command> [options]\n\n" + "\n".join(lines) + \
           "\n\n`ganlive <command> --help` for a command's own options.\n"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    name = argv[0]
    if name not in COMMANDS:
        print(f"ganlive: no command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"ganlive.tools.{COMMANDS[name][0]}")
    return module.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
