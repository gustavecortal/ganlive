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

#: What a first launch tunes (`quick`): the layers that cost the most, at the direct tiles
#: chosen most often where `tune` was run, plain, with f32 weights or through workgroup memory.
QUICK_LAYERS = 3
QUICK_DIRECT = [{"by": by, "bx": bx, "oct": oct, **extra}
                for by, bx, oct in ((2, 2, 4), (2, 2, 8), (1, 2, 8))
                for extra in ({}, {"f32": True}, {"slm": True})]


def heaviest(manifest: dict, count: int) -> list[str]:
    """The `count` convolutions of `manifest` with the most multiply-adds a frame."""
    t = manifest["tensors"]

    def work(op):
        cout, h, w = t[op["out"]]
        return h * w * (4 if op["up"] else 9) * t[op["in"]][0] * 2 * cout

    convs = [op for op in manifest["ops"] if op["op"] == "conv"]
    return [op["out"] for op in sorted(convs, key=work, reverse=True)[:count]]


def quick(manifest: dict, weights: bytes, device, *, log: Callable[[str], None] = print,
          choices: Path | None = None) -> tuple[dict, float, float]:
    """`tune` over the heaviest layers alone, each at a few plans, in fewer frames: what a
    model's first launch on a GPU runs, in seconds rather than minutes."""
    layers = heaviest(manifest, QUICK_LAYERS)
    options = {layer: [o for o in found if o.get("gemm") or o in QUICK_DIRECT or len(o) == 3]
               for layer, found in plan_choices(manifest).items() if layer in layers}
    return tune(manifest, weights, device.adapter, log=log, choices=choices, device=device,
                options=options, frames=8, compare=False)


def tune(manifest: dict, weights: bytes, adapter, *, log: Callable[[str], None] = print,
         choices: Path | None = None, device=None, options: dict | None = None,
         frames: int = 15, compare: bool = True) -> tuple[dict, float, float]:
    """The plans found for `adapter`, the frame time they give and the defaults' (ms), also
    remembered in `choices`: over `options` (layer -> plans), every layer's by default, on
    `device` or a device of its own, timing `frames` frames a round. With `compare`, the
    backends are measured again at the next load (`remember_plans`)."""
    own = device is None
    device = _device(adapter) if own else device
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
            return frame_ms(model, frames)
        finally:
            model.destroy()

    plans: dict = {}
    start = best = timed(plans)
    log(f"defaults: {start:.2f} ms")
    for layer, tried in (options or plan_choices(manifest)).items():
        for option in tried:
            trial = {**plans, layer: option}
            try:
                ms = timed(trial)
            except (ValueError, wgpu.GPUError):
                continue                 # a plan this layer cannot take
            if ms < best * 0.995:
                plans, best = trial, ms
                log(f"  {layer} {json.dumps(option)}: {ms:.2f} ms")
    if own:
        device.destroy()
    remember_plans(manifest, adapter, plans, choices, compare)
    return plans, best, start
