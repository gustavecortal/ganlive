"""A manifest compiled into a program, for whichever family made it."""

from __future__ import annotations

from ganlive.engine import fit, program, program_stylegan2
from ganlive.engine.player import browser_layout
from ganlive.engine.program import compile_program
from ganlive.engine.program_stylegan2 import compile_stylegan2

#: The WebGPU feature a program that computes in half precision needs.
F16 = "shader-f16"


def compile_manifest(manifest: dict, plans: dict | None = None, output: str = "rgba8",
                     browser: bool = False) -> dict:
    """The program of `manifest`, carrying its probe and dials for the hosts. `plans` overrides
    how a FastGAN's convolutions split (see `program._plan`), and `browser` asks for the plans
    every browser plays well, and for the dials it plays (`player.browser_layout`)."""
    if manifest.get("family") == "stylegan2":
        program = compile_stylegan2(manifest, plans, output=output)
    else:
        program = compile_program(manifest, plans, output=output, browser=browser)
    for key in ("family", "probe", "dials", "stamp"):
        if key in manifest:
            program[key] = manifest[key]
    # What a device must have to run it: half precision, for a plan that asks for it.
    if any(code.startswith("enable f16;") for code in program["shaders"]):
        program["features"] = [F16]
    if output != "f32":         # how a browser draws the pixels onto a canvas
        program["screen"] = fit.screen(output)
    if browser:                 # and the dials it plays them with
        program["knobs"], program["live"] = browser_layout(program)
    return program


def browser_programs(manifest: dict) -> tuple[dict, dict[str, dict]]:
    """The browser program of `manifest`, and its kernel variants by name, which it lists
    (`variants`) for a browser to time on its own card. A FastGAN has none."""
    program = compile_manifest(manifest, browser=True)
    variants = {} if manifest.get("family") != "stylegan2" else {
        name: compile_manifest(manifest, plans, browser=True)
        for name, plans in program_stylegan2.browser_variants(manifest).items() if plans}
    if variants:
        program["variants"] = list(variants)
    return program, variants


def quick_choices(manifest: dict) -> dict[str, list[dict]]:
    """The few plans a first launch tries on the layers that cost the most (`tune.quick`). A
    StyleGAN2 has none: `ganlive tune` tunes it."""
    return {} if manifest.get("family") == "stylegan2" else program.quick_choices(manifest)


def plan_choices(manifest: dict) -> dict[str, list[dict]]:
    """Each layer's plans worth trying (`ganlive tune`), by the name a plan is given under."""
    family = program_stylegan2 if manifest.get("family") == "stylegan2" else program
    return family.plan_choices(manifest)
