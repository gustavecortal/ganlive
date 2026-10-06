"""An ONNX file as this project reads and writes it.

The `ganlive.*` metadata a graph carries about its own dials, read and written here and
nowhere else, plus the few protobuf helpers every graph edit needs.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

from ganlive.models.common import Ladder

#: The prefix of every metadata key this project writes into a graph.
META = "ganlive."

#: The names of a playable graph's two inputs: the latent, and the settings vector.
LATENT_INPUT, SETTINGS_INPUT = "z", "k"


@dataclass(frozen=True)
class OnnxConfig:
    """What the bank needs to know about a model, read out of the graph rather than a sidecar."""

    nz: int
    ladder: Ladder


def dims(value) -> list[int]:
    """A graph value's declared shape. An unknown dimension reads as 0."""
    return [d.dim_value for d in value.type.tensor_type.shape.dim]


def producers(graph) -> dict[str, int]:
    """Where each tensor comes from: its name, to the index in `graph.node` of the node
    that writes it. Graph inputs and initializers have no producer."""
    return {out: i for i, node in enumerate(graph.node) for out in node.output}


def initializers(graph) -> dict:
    """The graph's stored tensors, by name."""
    return {t.name: t for t in graph.initializer}


def structure(path):
    """The graph, without loading a sibling `.onnx.data` of hundreds of megabytes. Enough for
    anything that reads shapes or metadata. Weights stored inside the file are still read."""
    import onnx

    return onnx.load(str(path), load_external_data=False)


def weights_file(path) -> Path:
    """Where a graph's external weights live: `<name>.onnx.data` beside it."""
    path = Path(path)
    return path.with_name(path.name + ".data")


def config_of(path) -> OnnxConfig:
    """Latent width and output size, from the graph's own declared shapes."""
    said = dials_of(path)
    height, width = said["size"]
    return OnnxConfig(nz=said["nz"], ladder=Ladder(width=width, height=height))


def dials_of(path) -> dict:
    """Everything the file says about its own dials, in one read.

    `settings` (names), `rests`, `curves` and `levels` per dial, `precision` (the verdicts
    keyed `backend/device`), and the graph's `nz` and output `size`."""
    path = Path(path)
    stat = path.stat()
    return _dials_cached(str(path), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=8)
def _dials_cached(path: str, _mtime: int, _size: int) -> dict:
    """The read itself, keyed on the file's identity so a rewritten file is read again."""
    model = structure(path)
    graph = model.graph
    raw = {e.key: e.value for e in model.metadata_props if e.value}
    names = raw.get(f"{META}settings", "")
    out = {"settings": names.split(",") if names else []}
    for key, empty in (("precision", {}), ("rests", []), ("curves", []), ("levels", [])):
        value = raw.get(f"{META}{key}")
        out[key] = json.loads(value) if value else empty
    out["nz"] = int(dims(graph.input[0])[-1])
    out["size"] = (int(dims(graph.output[0])[-2]), int(dims(graph.output[0])[-1]))
    return out


def name_settings(model, names, curves=None, levels=None, rests=None,
                  precision=None) -> None:
    """Write the dials into the graph's metadata, replacing any already there, so the file
    stands on its own."""
    keep = [e for e in model.metadata_props if not e.key.startswith(META)]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    write = {"settings": ",".join(names)}
    if curves is not None:
        write["curves"] = json.dumps([list(map(float, c)) for c in curves])
    if levels is not None:
        write["levels"] = json.dumps([round(float(x), 3) for x in levels])
    if rests is not None:
        write["rests"] = json.dumps([float(x) for x in rests])
    if precision:
        write["precision"] = json.dumps(precision)
    for key, value in write.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = f"{META}{key}", value
