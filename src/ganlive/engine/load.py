"""Opening a model on the engine: an engine folder as it is, or a FastGAN or StyleGAN2
checkpoint through the conversion beside it, converted first when it is missing or out of date
(see `stamp`).

PyTorch is imported only when a conversion has to run.
"""

from __future__ import annotations

import json
from pathlib import Path

from ganlive.checkpoints import ENGINE_SUFFIX, MANIFEST, WEIGHTS, is_engine, slug_for
from ganlive.engine import stamp
from ganlive.engine.compile import compile_manifest
from ganlive.engine.player import EngineGenerator, PlayedDirections, dials_of
from ganlive.engine.runner import cache_dir, checked, fastest
from ganlive.levels import RANDOM_FLOOR


def converted(checkpoint, floor: float = RANDOM_FLOOR, grain: bool = True) -> tuple[Path, dict]:
    """The folder holding `checkpoint`'s current engine model, and its manifest. Beside the
    checkpoint, or in the user's cache when its folder cannot be written."""
    checkpoint = Path(checkpoint)
    beside = checkpoint.with_suffix(ENGINE_SUFFIX)
    want = stamp.stamp_for(checkpoint, floor, grain, known=stamp.manifest_stamp(beside))
    folders = (beside, cache_dir() / "engine" / f"{slug_for(checkpoint)}-{want['checkpoint'][:12]}")
    for folder in folders:
        if stamp.same(stamp.manifest_stamp(folder), want):
            return folder, json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    from ganlive.engine.convert import convert  # PyTorch, from here on

    for folder in folders:
        print(f"converting {checkpoint.name} for the engine (once) -> {folder}", flush=True)
        try:
            return folder, convert(checkpoint, folder, stamp=want, floor=floor, grain=grain)
        except OSError as exc:
            print(f"  cannot write there ({exc})", flush=True)
    raise RuntimeError(f"{checkpoint.name}: nowhere to write its engine model")


def open_model(path, gpu=None, floor: float = RANDOM_FLOOR,
               grain: bool = True) -> tuple[EngineGenerator, object, PlayedDirections | None]:
    """The model at `path` ready to play, with its dial layout and directions: on the wgpu
    device `gpu`, or else on this machine's fastest backend for it."""
    if is_engine(path):
        folder = Path(path)
        manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    else:
        folder, manifest = converted(path, floor, grain)
    program = compile_manifest(manifest)
    weights = (folder / WEIGHTS).read_bytes()
    if gpu is None:
        model, report = fastest(program, weights)
        print(f"engine: {report['best']}", flush=True)
    else:
        model = checked(gpu, program, weights, own_device=False)
    net = EngineGenerator(model)
    print(net.report(), flush=True)
    layout, directions = dials_of(program)
    if directions is not None:
        print(f"directions: {directions.report()}", flush=True)
    return net, layout, directions
