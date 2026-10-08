"""Opening a model on the engine: an engine folder as it is, or a FastGAN or StyleGAN2
checkpoint through the conversion beside it, converted first when it is missing or out of date
(see `stamp`).

PyTorch is imported only when a conversion has to run.
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path

from ganlive.checkpoints import ENGINE_SUFFIX, MANIFEST, WEIGHTS, is_engine, slug_for
from ganlive.engine import stamp
from ganlive.engine.player import EngineGenerator, PlayedDirections, dials_of
from ganlive.engine.runner import built, cache_dir, fastest, read_folder, tuned_plans
from ganlive.engine.tune import quick
from ganlive.levels import RANDOM_FLOOR


def read_manifest(folder) -> dict:
    """The manifest of the engine model in `folder`, parsed once while the file is unchanged."""
    path = Path(folder) / MANIFEST
    st = path.stat()
    return _parsed(str(path.resolve()), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=32)
def _parsed(path: str, _size: int, _mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stamp_in(folder) -> dict | None:
    """The stamp of the engine model in `folder`, or None if there is none to read."""
    try:
        return read_manifest(folder).get("stamp")
    except (OSError, ValueError):
        return None


def conversion_of(checkpoint) -> tuple[Path, dict] | None:
    """`(folder, manifest)` of a conversion of `checkpoint` as it is now, beside it or else in
    the cache, or None: one made from a file of this size and modification time, found
    without converting or hashing anything."""
    checkpoint = Path(checkpoint)
    try:
        st = checkpoint.stat()
    except OSError:
        return None

    def current(folder) -> bool:
        made = _stamp_in(folder) or {}
        return made.get("size") == st.st_size and made.get("mtime_ns") == st.st_mtime_ns

    beside = checkpoint.with_suffix(ENGINE_SUFFIX)
    if current(beside):
        return beside, read_manifest(beside)
    for folder in sorted((cache_dir() / "engine").glob(f"{slug_for(checkpoint)}-*")):
        if current(folder):
            return folder, read_manifest(folder)
    return None


def converted(checkpoint, floor: float = RANDOM_FLOOR, grain: bool = True) -> tuple[Path, dict]:
    """The folder holding `checkpoint`'s current engine model, and its manifest. Beside the
    checkpoint, or in the user's cache when its folder cannot be written."""
    checkpoint = Path(checkpoint)
    beside = checkpoint.with_suffix(ENGINE_SUFFIX)
    found = conversion_of(checkpoint)
    # A conversion of this very file lends its hash of it, so it is not hashed again.
    known = found[1].get("stamp") if found else _stamp_in(beside)
    want = stamp.stamp_for(checkpoint, floor, grain, known=known)
    folders = (beside, cache_dir() / "engine" / f"{slug_for(checkpoint)}-{want['checkpoint'][:12]}")
    for folder in ([found[0]] if found else []) + list(folders):
        if stamp.same(_stamp_in(folder), want):
            return folder, read_manifest(folder)
    from ganlive.engine.convert import convert  # PyTorch, from here on

    for folder in folders:
        print(f"converting {checkpoint.name} for the engine (once) -> {folder}", flush=True)
        try:
            return folder, convert(checkpoint, folder, stamp=want, floor=floor, grain=grain)
        except OSError as exc:
            print(f"  cannot write there ({exc})", flush=True)
    raise RuntimeError(f"{checkpoint.name}: nowhere to write its engine model")


def _tuned(model, manifest: dict, weights: bytes, path):
    """`model` rebuilt with the plans a quick tune finds for its GPU (`tune.quick`), the first
    time this model plays there. Remembered, so it happens once."""
    device = model.device
    started = time.perf_counter()
    print("  plans: a first load on this GPU, so its heaviest layers are tuned", flush=True)
    plans, best, start = quick(manifest, weights, device, log=lambda _line: None)
    print(f"  plans: {start:.2f} -> {best:.2f} ms a frame in {time.perf_counter() - started:.0f} s. "
          f"`ganlive tune {path}` searches every layer", flush=True)
    if not plans:
        return model
    model.destroy()
    return built(device, manifest, weights, plans)


def open_model(path, gpu=None, floor: float = RANDOM_FLOOR, grain: bool = True,
               backend: str | None = None,
               tune: bool = True) -> tuple[EngineGenerator, object, PlayedDirections | None]:
    """The model at `path` ready to play, with its dial layout and directions: on the wgpu
    device `gpu`, or else on `backend` or this machine's fastest backend for it, with the plans
    tuned for that device (`ganlive tune`), or tuned now if it has none and `tune` allows."""
    if is_engine(path):
        manifest, weights = read_folder(path)
    else:
        folder, manifest = converted(path, floor, grain)
        weights = (folder / WEIGHTS).read_bytes()
    if gpu is None:
        model, report = fastest(manifest, weights, backend=backend)
        print(f"engine: {report['best']}", flush=True)
    else:
        model = built(gpu, manifest, weights, tuned_plans(manifest, gpu.adapter))
    # A FastGAN on a GPU: a software adapter gains nothing from it, and StyleGAN2 has `tune`.
    adapter = model.device.adapter
    if (tune and manifest.get("family", "fastgan") == "fastgan"
            and adapter.info["adapter_type"] != "CPU" and tuned_plans(manifest, adapter) is None):
        model = _tuned(model, manifest, weights, path)
    net = EngineGenerator(model)
    print(net.report(), flush=True)
    layout, directions = dials_of(model.program)
    if directions is not None:
        print(f"directions: {directions.report()}", flush=True)
    return net, layout, directions
