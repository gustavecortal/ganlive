"""A manifest compiled into a program, for whichever family made it."""

from __future__ import annotations

from ganlive.engine import program, program_stylegan2
from ganlive.engine.program import compile_program
from ganlive.engine.program_stylegan2 import compile_stylegan2


def compile_manifest(manifest: dict, plans: dict | None = None, output: str = "rgba8",
                     browser: bool = False) -> dict:
    """The program of `manifest`, carrying its probe and dials for the hosts. `plans` overrides
    how a FastGAN's convolutions split (see `program._plan`), and `browser` asks for the plans
    every browser plays well."""
    if manifest.get("family") == "stylegan2":
        program = compile_stylegan2(manifest, plans, output=output)
    else:
        program = compile_program(manifest, plans, output=output, browser=browser)
    for key in ("family", "probe", "dials", "stamp"):
        if key in manifest:
            program[key] = manifest[key]
    return program


def plan_choices(manifest: dict) -> dict[str, list[dict]]:
    """Each layer's plans worth trying (`ganlive tune`), by the name a plan is given under."""
    family = program_stylegan2 if manifest.get("family") == "stylegan2" else program
    return family.plan_choices(manifest)
