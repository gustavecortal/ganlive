"""Opening a model on the engine: an engine folder as it is, or a FastGAN or StyleGAN2
checkpoint through the conversion beside it, converted first when it is missing or out of date
(see `stamp`).

PyTorch is imported only when a conversion has to run.
"""

from __future__ import annotations

import contextlib
import functools
import json
from pathlib import Path

from ganlive.checkpoints import ENGINE_SUFFIX, MANIFEST, WEIGHTS, is_engine, slug_for
from ganlive.engine import stamp
from ganlive.engine.player import EngineGenerator, PlayedDirections, dials_of
from ganlive.engine.runner import built, cache_dir, fastest, read_folder, tuned_plans
from ganlive.levels import RANDOM_FLOOR

#: What a missing PyTorch says: playing needs none, converting a checkpoint does.
NEEDS_TORCH = ("a checkpoint without its conversion needs PyTorch to read and convert: install "
               "PyTorch for your GPU (https://pytorch.org/get-started/locally/), or "
               "`pip install 'ganlive[convert]'`")


@contextlib.contextmanager
def needs_torch():
    """Say how to install PyTorch when what runs inside fails for the want of it."""
    try:
        yield
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ModuleNotFoundError(NEEDS_TORCH, name="torch") from exc


def read_manifest(folder) -> dict:
    """The manifest of the engine model in `folder`, parsed once while the file is unchanged."""
    path = Path(folder) / MANIFEST
    st = path.stat()
    return _parsed(str(path.resolve()), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=32)
def _parsed(path: str, _size: int, _mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conversion_of(checkpoint) -> dict | None:
    """The manifest of a conversion of `checkpoint` as it is now, beside it or in the cache, or
    None: one made from a file of this size and modification time, read without converting
    or hashing anything."""
    checkpoint = Path(checkpoint)
    try:
        st = checkpoint.stat()
    except OSError:
        return None
    cached = sorted((cache_dir() / "engine").glob(f"{slug_for(checkpoint)}-*"))
    for folder in (checkpoint.with_suffix(ENGINE_SUFFIX), *cached):
        try:
            manifest = read_manifest(folder)
        except (OSError, ValueError):
            continue
        made = manifest.get("stamp") or {}
        if made.get("size") == st.st_size and made.get("mtime_ns") == st.st_mtime_ns:
            return manifest
    return None


def converted(checkpoint, floor: float = RANDOM_FLOOR, grain: bool = True) -> tuple[Path, dict]:
    """The folder holding `checkpoint`'s current engine model, and its manifest. Beside the
    checkpoint, or in the user's cache when its folder cannot be written."""
    checkpoint = Path(checkpoint)
    beside = checkpoint.with_suffix(ENGINE_SUFFIX)
    want = stamp.stamp_for(checkpoint, floor, grain, known=stamp.manifest_stamp(beside))
    folders = (beside, cache_dir() / "engine" / f"{slug_for(checkpoint)}-{want['checkpoint'][:12]}")
    for folder in folders:
        if stamp.same(stamp.manifest_stamp(folder), want):
            return folder, read_manifest(folder)
    with needs_torch():
        from ganlive.engine.convert import convert  # PyTorch, from here on

    for folder in folders:
        print(f"converting {checkpoint.name} for the engine (once) -> {folder}", flush=True)
        try:
            return folder, convert(checkpoint, folder, stamp=want, floor=floor, grain=grain)
        except OSError as exc:
            print(f"  cannot write there ({exc})", flush=True)
    raise RuntimeError(f"{checkpoint.name}: nowhere to write its engine model")


def open_model(path, gpu=None, floor: float = RANDOM_FLOOR, grain: bool = True,
               backend: str | None = None) -> tuple[EngineGenerator, object, PlayedDirections | None]:
    """The model at `path` ready to play, with its dial layout and directions: on the wgpu
    device `gpu`, or else on `backend` or this machine's fastest backend for it, with the plans
    tuned for that device (`ganlive tune`)."""
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
    net = EngineGenerator(model)
    print(net.report(), flush=True)
    layout, directions = dials_of(model.program)
    if directions is not None:
        print(f"directions: {directions.report()}", flush=True)
    return net, layout, directions
