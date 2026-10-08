"""Find, on this machine, how each layer of a model runs fastest: for each layer in turn, try
the plans it can take (`plan_choices`) and keep one when a whole frame gets faster and the
probe still matches. The plans found are remembered per adapter, beside `fastest`'s choice of
backend, and every later load uses them (`runner.built`)."""

from __future__ import annotations

import json
from collections.abc import Callable

import wgpu

from ganlive.engine.compile import compile_manifest, plan_choices, quick_choices
from ganlive.engine.runner import OUTPUT, checked, frame_ms


def tune(manifest: dict, weights: bytes, device, *, log: Callable[[str], None] = print,
         options: dict | None = None, frames: int = 15, base=None) -> tuple[dict, float, float, object]:
    """`(plans, ms, defaults' ms, model)`: the plans that draw `manifest` fastest on `device`,
    over `options` (layer -> plans, every layer's by default), timing `frames` frames a round,
    and the model built with them, which the caller owns. `base`, the model already built with
    the defaults, is timed rather than built again, and is the one returned or destroyed."""
    pipelines = getattr(base, "pipelines", None) or {}   # a trial compiles what it changes
    seen: set[str] = set()                               # programs already timed

    def built(plans: dict):
        program = compile_manifest(manifest, plans, OUTPUT)
        shape = json.dumps([program["shaders"], program["steps"], program["load"]])
        if shape in seen:
            return None
        seen.add(shape)
        return checked(device, program, weights, pipelines=pipelines)

    plans: dict = {}
    if base is None:
        best_model = built(plans)
    else:
        best_model = base
        defaults = compile_manifest(manifest, None, OUTPUT)
        seen.add(json.dumps([defaults["shaders"], defaults["steps"], defaults["load"]]))
    start = best = frame_ms(best_model, frames)
    log(f"defaults: {start:.2f} ms")
    for layer, tried in (options if options is not None else plan_choices(manifest)).items():
        for option in tried:
            trial = {**plans, layer: option}
            try:
                model = built(trial)
            except (ValueError, RuntimeError, wgpu.GPUError):
                continue                 # a plan this layer cannot take, or one drawn wrong
            if model is None:
                continue
            ms = frame_ms(model, frames)
            if ms < best * 0.995:
                best_model.destroy()
                plans, best, best_model = trial, ms, model
                log(f"  {layer} {json.dumps(option)}: {ms:.2f} ms")
            else:
                model.destroy()
    return plans, best, start, best_model


def quick(manifest: dict, weights: bytes, base) -> tuple[dict, float, float, object]:
    """`tune` over a model's heaviest layers alone, each at a few plans (`quick_choices`), in
    fewer frames, starting from `base`, the model as built: what a model's first launch on a
    GPU runs, in seconds rather than minutes."""
    return tune(manifest, weights, base.device, log=lambda _line: None,
                options=quick_choices(manifest), frames=8, base=base)
