"""Find, on this machine, how each layer of a model runs fastest: for each layer in turn, try
the plans it can take (`plan_choices`) and keep one when a whole frame gets faster and the
probe still matches. The plans found are remembered per adapter, beside `fastest`'s choice of
backend, and every later load uses them (`runner.built`)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import wgpu

from ganlive.engine.compile import compile_manifest, plan_choices
from ganlive.engine.runner import OUTPUT, _device, checked, frame_ms, remember_plans


def tune(manifest: dict, weights: bytes, adapter, *, log: Callable[[str], None] = print,
         choices: Path | None = None) -> tuple[dict, float, float]:
    """The plans found for `adapter`, the frame time they give and the defaults' (ms), also
    remembered in `choices`."""
    device = _device(adapter)
    pipelines: dict = {}                 # each trial compiles only the shaders it changes
    seen: set[str] = set()               # options that compile to a program already timed

    def timed(plans: dict) -> float:
        program = compile_manifest(manifest, plans, OUTPUT)
        shape = json.dumps([program["shaders"], program["steps"], program["load"]])
        if shape in seen:
            return math.inf
        seen.add(shape)
        try:
            model = checked(device, program, weights, pipelines=pipelines)
        except RuntimeError:             # draws the probe wrong
            return math.inf
        try:
            return frame_ms(model)
        finally:
            model.destroy()

    plans: dict = {}
    start = best = timed(plans)
    log(f"defaults: {start:.2f} ms")
    for layer, options in plan_choices(manifest).items():
        for option in options:
            trial = {**plans, layer: option}
            try:
                ms = timed(trial)
            except (ValueError, wgpu.GPUError):
                continue                 # a plan this layer cannot take
            if ms < best * 0.995:
                plans, best = trial, ms
                log(f"  {layer} {json.dumps(option)}: {ms:.2f} ms")
    device.destroy()
    remember_plans(manifest, adapter, plans, choices)
    return plans, best, start
