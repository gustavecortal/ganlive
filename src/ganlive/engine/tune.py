"""Find, on this machine, how each layer of a model runs fastest: for each layer in turn, try
the plans it can take (`plan_choices`) and keep one when a whole frame gets faster and the
probe still matches. The plans found are remembered per adapter, beside `fastest`'s choice of
backend, and every later load uses them (`runner.built`)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import wgpu

from ganlive.engine import program, program_stylegan2
from ganlive.engine.compile import compile_manifest
from ganlive.engine.probe import PROBE_LEVELS
from ganlive.engine.runner import (
    OUTPUT,
    Model,
    _choices,
    _device,
    adapter_name,
    cache_dir,
    model_key,
)
from ganlive.files import remember


def plan_choices(manifest: dict) -> dict[str, list[dict]]:
    """Each layer's plans worth trying, by the name a plan is given under."""
    family = program_stylegan2 if manifest.get("family") == "stylegan2" else program
    return family.plan_choices(manifest)


def frame_ms(model: Model, frames: int = 15, rounds: int = 3) -> float:
    """The best of `rounds` mean frame times over `frames` frames."""
    best = float("inf")
    for _ in range(rounds):
        model.frame()
        model.wait()
        started = time.perf_counter()
        for _ in range(frames):
            model.frame()
        model.wait()
        best = min(best, (time.perf_counter() - started) / frames * 1000)
    return best


def tune(manifest: dict, weights: bytes, adapter, *, log: Callable[[str], None] = print,
         choices: Path | None = None) -> tuple[dict, float, float]:
    """The plans found for `adapter`, the frame time they give and the defaults' (ms), also
    remembered in `choices`."""
    device = _device(adapter)
    pipelines: dict = {}                 # each trial compiles only the shaders it changes

    def timed(plans: dict) -> float:
        model = Model(device, compile_manifest(manifest, plans, OUTPUT), weights, pipelines=pipelines)
        try:
            if "probe" in model.program and model.strays() > PROBE_LEVELS:
                return float("inf")
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
            except (ValueError, RuntimeError, wgpu.GPUError):
                continue                 # a plan this layer cannot take
            if ms < best * 0.995:
                plans, best = trial, ms
                log(f"  {layer} {json.dumps(option)}: {ms:.2f} ms")
    device.destroy()
    path = choices or cache_dir() / "backends.json"
    saved = _choices(path)
    entry = saved.setdefault(model_key(manifest), {})
    entry.setdefault("plans", {})[adapter_name(adapter)] = plans
    entry.pop("measured", None)          # backends compare again, each with its plans
    remember(path, json.dumps(saved, indent=2))
    return plans, best, start
