"""A manifest compiled into a program, for whichever family made it."""

from __future__ import annotations

from ganlive.engine.program import compile_program
from ganlive.engine.program_stylegan2 import compile_stylegan2


def compile_manifest(manifest: dict, plans: dict | None = None, output: str = "rgba8") -> dict:
    """The program of `manifest`, carrying its probe and dials for the hosts. `plans` overrides
    how a FastGAN's convolutions split (see `program._plan`)."""
    if manifest.get("family") == "stylegan2":
        program = compile_stylegan2(manifest, output=output)
    else:
        program = compile_program(manifest, plans, output=output)
    for key in ("family", "probe", "dials", "stamp"):
        if key in manifest:
            program[key] = manifest[key]
    return program
