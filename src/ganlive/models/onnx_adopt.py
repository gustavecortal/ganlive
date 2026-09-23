"""Giving a graph nobody here wrote a set of dials, and proving each one moves the picture.

Protobuf surgery: freeze every random draw so the same latent gives the same pixels, find the
resolution bands, insert a `Mul` on each fed from a new settings input, then measure what each
one bought and write all of it into the file's own metadata. The instrument that opens the
result afterwards knows nothing about the architecture.

The measuring half is `models/calibrate.py`; this is the half that edits a graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ganlive.models.calibrate import (
    TARGET_LEVELS,
    Adopted,
    Dial,
    Probe,
    _latent,
    calibrate,
    deterministic,
    levels,
)
from ganlive.models.onnx import _dims
from ganlive.models.onnx import dials_of as onnx_dials

#: The ops that make a graph give a different answer to the same question.
RANDOM = ("RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike")


#: The ops a generator ends in. A gain into one of these is the dial that washes the picture
#: out below 1 and hard-clips it above.
SQUASH = ("Tanh", "Sigmoid")


#: How far half precision may move the picture before it is refused, in mean 8-bit levels. This project's
#: own generator measures 0.100 there and has shipped in fp16 for months.
HALF_LEVELS = 1.0


def _shapes(model) -> dict[str, tuple[int, ...]]:
    """Every tensor's static shape, from shape inference. Unknown dims are dropped."""
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    out: dict[str, tuple[int, ...]] = {}
    for group in (inferred.graph.value_info, inferred.graph.input, inferred.graph.output):
        for value in group:
            dims = _dims(value)
            if dims and all(d > 0 for d in dims):
                out[value.name] = tuple(dims)
    return out


def _rewire(graph, old: str, new: str, skip) -> None:
    """Point every consumer of `old` at `new`, except the node doing the replacing."""
    for node in graph.node:
        if node is skip:
            continue
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
    for value in graph.output:
        if value.name == old:
            value.name = new


def _gate(graph, tensor: str, slot: int, feed: str, tag: str, at: int):
    """Insert `tensor * k[slot]` and rewire everything downstream onto it."""
    from onnx import helper, numpy_helper

    take = f"ganlive_take_{tag}"
    scaled = f"ganlive_scaled_{tag}"
    for name, value in ((f"{take}_start", [slot]), (f"{take}_end", [slot + 1])):
        graph.initializer.append(
            numpy_helper.from_array(np.array(value, np.int64), name))
    # No `axes` input: it is optional and the settings vector has one dimension, so the
    # default is the axis wanted. Every dial was storing its own identical copy of `[0]`.
    cut = helper.make_node("Slice", [feed, f"{take}_start", f"{take}_end"],
                           [take], name=take)
    mul = helper.make_node("Mul", [tensor, take], [scaled], name=scaled)
    _rewire(graph, tensor, scaled, mul)
    graph.node.insert(at, mul)
    graph.node.insert(at, cut)


def _freeze_random(graph, shapes, seed: int) -> list[tuple[str, tuple[int, ...]]]:
    """Replace every random op with a seeded constant. Returns `(tensor, shape)` per draw."""
    from onnx import numpy_helper

    rng = np.random.default_rng(seed)
    frozen: list[tuple[str, tuple[int, ...]]] = []
    drop = []
    for i, node in enumerate(graph.node):
        if node.op_type not in RANDOM:
            continue
        out = node.output[0]
        shape = shapes.get(out)
        if shape is None:
            raise RuntimeError(
                f"{node.op_type} {node.name or out} has no inferred shape, so its draw "
                f"cannot be frozen; the graph would stay non-deterministic and every dial "
                f"measured against it would be measuring noise")
        name = f"ganlive_noise_{i}"
        values = (rng.standard_normal(shape) if "Normal" in node.op_type
                  else rng.random(shape)).astype(np.float32)
        graph.initializer.append(numpy_helper.from_array(values, name))
        _rewire(graph, out, name, node)
        frozen.append((name, tuple(shape)))
        drop.append(node)
    for node in drop:
        graph.node.remove(node)
    return frozen


#: The smallest spatial side that counts as a resolution band.
MIN_BAND = 4


def _feeds(graph, target: str) -> set[str]:
    """Every tensor the graph's own output actually depends on."""
    produced = {out: node for node in graph.node for out in node.output}
    seen, edge = set(), [target]
    while edge:
        name = edge.pop()
        if name in seen:
            continue
        seen.add(name)
        node = produced.get(name)
        if node is not None:
            edge.extend(node.input)
    return seen


def _ladder(size) -> set[tuple[int, int]]:
    """The sizes a resolution ladder passes through: the output, halved until it is small."""
    out, h, w = set(), int(size[0]), int(size[1])
    while h >= MIN_BAND and w >= MIN_BAND:
        out.add((h, w))
        if h % 2 or w % 2:
            break
        h, w = h // 2, w // 2
    return out


def _bands(graph, shapes, size) -> dict[tuple[int, int], tuple[str, int]]:
    """The last tensor at each spatial size, as `{(h, w): (tensor, node index)}`."""
    consumed = {name for node in graph.node for name in node.input}
    mine = _feeds(graph, graph.output[0].name)
    rungs = _ladder(size) if size else None
    out: dict[tuple[int, int], tuple[str, int]] = {}
    for i, node in enumerate(graph.node):
        for name in node.output:
            shape = shapes.get(name)
            if shape is None or len(shape) != 4 or name not in consumed or name not in mine:
                continue
            h, w = int(shape[2]), int(shape[3])
            if min(h, w) < MIN_BAND or (rungs is not None and (h, w) not in rungs):
                continue
            out[(h, w)] = (name, i)
    return out


def _band_names(bands) -> list[tuple[tuple[int, int], str]]:
    """A name per band: its height, and its width too only when the height is ambiguous."""
    sizes = sorted(bands)
    heights = [h for h, _w in sizes]
    return [(size, f"gain_{size[0]}" if heights.count(size[0]) == 1
             else f"gain_{size[0]}x{size[1]}") for size in sizes]


#: Ops that can sit between a squash and the graph's output without being the picture: a denormalise, a
#: layout change, a cast to bytes.
AFTER_SQUASH = ("Mul", "Add", "Sub", "Div", "Clip", "Cast", "Transpose", "Reshape",
                "Squeeze", "Unsqueeze", "Identity", "Round")


def _squash(graph) -> str | None:
    """The input to the squash the graph's *first* output comes out of, if there is one."""
    produced = {out: node for node in graph.node for out in node.output}
    name = graph.output[0].name
    for _step in range(len(AFTER_SQUASH) + 4):
        node = produced.get(name)
        if node is None:
            return None
        if node.op_type in SQUASH:
            return node.input[0]
        if node.op_type not in AFTER_SQUASH:
            return None
        name = node.input[0]
    return None


def _with_initializers(graph, shapes) -> dict[str, tuple[int, ...]]:
    """The inferred shapes, plus the stored tensors' own, which need no inference at all."""
    out = dict(shapes)
    for tensor in graph.initializer:
        out.setdefault(tensor.name, tuple(tensor.dims))
    return out


def _constant(graph) -> set[str]:
    """Every tensor whose value does not depend on any graph input."""
    live = {value.name for value in graph.input}
    for node in graph.node:
        if live.intersection(node.input):
            live.update(node.output)
    produced = {out for node in graph.node for out in node.output}
    return (produced | {i.name for i in graph.initializer}) - live


def _baked_noise(graph, shapes, size) -> list[tuple[str, tuple[int, ...]]]:
    """Noise patterns that were frozen before the export, as `(tensor, shape)` per draw."""
    fixed = _constant(graph)
    mine = _feeds(graph, graph.output[0].name)
    out = []
    for node in graph.node:
        if node.op_type != "Add" or node.output[0] not in mine:
            continue
        for name, other in (node.input[0], node.input[1]), (node.input[1], node.input[0]):
            shape, feature = shapes.get(name), shapes.get(other)
            # **Two dimensions is enough, and requiring four found none of NVIDIA's.** StyleGAN2 registers
            # its noise as `(H, W)` and lets it broadcast over batch and channel; this project's own is `(1,
            # C, H, W)`.
            if (name in fixed and shape is not None and feature is not None
                    and len(shape) >= 2 and len(feature) == 4
                    and shape[-2:] == feature[-2:]
                    and min(shape[-2:]) >= MIN_BAND
                    and (not size or shape[-2] <= size[0])):
                out.append((name, tuple(shape)))
                break
    return out


def insert_dials(model, seed: int = 0, feed: str = "k") -> Adopted:
    """Freeze the randomness, add the dials, and give the graph the input that drives them."""
    from onnx import TensorProto, helper

    graph = model.graph
    shapes = _shapes(model)
    found = Adopted(nz=int(shapes[graph.input[0].name][-1]))
    out_shape = shapes.get(graph.output[0].name)
    if out_shape and len(out_shape) == 4:
        found.size = (int(out_shape[2]), int(out_shape[3]))

    frozen = _freeze_random(graph, shapes, seed)
    found.noise = len(frozen)
    if not frozen:
        # Nothing random left to freeze does not mean no noise: it may have been frozen
        # before the export, and then it is a stored pattern rather than a draw.
        frozen = _baked_noise(graph, _with_initializers(graph, shapes), found.size)
        found.noise, found.was_baked = len(frozen), True

    # One dial per band of noise, not one per draw: a generator injects at several points per
    # resolution, and a dial each would be a page of controls that move together.
    by_size: dict[int, list[str]] = {}
    for name, shape in frozen:
        by_size.setdefault(int(shape[-2]), []).append(name)

    bands = _bands(graph, shapes, found.size)
    found.bands = tuple(sorted(bands))
    named = _band_names(bands)

    # **No dial on the squash's input.** A gain there is a contrast curve on the finished picture,
    # not a way through the model: it is the class `zoom`, `chroma` and this project's own
    # `pre_tanh` were all deleted for, and measured on the shipping checkpoint `pre_tanh` was
    # also the weakest gate with its two ends one change sign-flipped. Excluded rather than
    # renamed, because on the first foreign model tried here that tensor *was* the last band's,
    # and letting `gain_<h>` claim it would keep the deleted dial under an honest-looking name.
    squash = _squash(graph)
    wanted: list[tuple[str, list[str]]] = []
    wanted += [(f"noise_{h}", by_size[h]) for h in sorted(by_size)]
    wanted += [(name, [bands[size][0]]) for size, name in named]

    # Seeded with the squash's input, so one loop decides which tensor a dial may claim: the
    # exclusion and the no-two-dials-on-one-tensor rule are the same rule. A band left with
    # nothing fresh drops out below, which is what the exclusion needs to happen anyway.
    claimed: set[str] = set() if squash is None else {squash}
    deduped: list[tuple[str, list[str]]] = []
    for name, tensors in wanted:
        fresh = [t for t in tensors if t not in claimed]
        if fresh:
            claimed.update(fresh)
            deduped.append((name, fresh))
    wanted = deduped

    # Later first. Each insertion shifts the nodes after it, so working down the graph means the positions
    # worked out before the first edit are still right for every edit after it.
    produced = {out: i for i, node in enumerate(graph.node) for out in node.output}
    order = sorted(((produced.get(t, -1), slot, t)
                    for slot, (_name, ts) in enumerate(wanted) for t in ts), reverse=True)
    for tag, (index, slot, tensor) in enumerate(order):
        _gate(graph, tensor, slot, feed, f"{wanted[slot][0]}_{tag}", index + 1)

    found.dials = [Dial(name=name, tensor=tensors[0]) for name, tensors in wanted]
    graph.input.append(helper.make_tensor_value_info(feed, TensorProto.FLOAT, [len(wanted)]))
    return found


def name_settings(model, names, curves=None, levels=None, rests=None,
                  precision=None) -> None:
    """Write the dials into the graph's metadata, so the file stands on its own."""
    keep = [e for e in model.metadata_props if not e.key.startswith("ganlive.")]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    write = {"ganlive.settings": ",".join(names)}
    if curves is not None:
        write["ganlive.curves"] = json.dumps([list(map(float, c)) for c in curves])
    if levels is not None:
        write["ganlive.levels"] = json.dumps([round(float(x), 3) for x in levels])
    if rests is not None:
        write["ganlive.rests"] = json.dumps([float(x) for x in rests])
    if precision:
        write["ganlive.precision"] = json.dumps(precision)
    for key, value in write.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value


def usable_precision(path, backend: str = "auto",
                     device: str = "") -> tuple[str, str, float]:
    """Whether this graph survives half precision, and on which backend and device."""
    from ganlive.models.onnx import config_of
    from ganlive.models.runtime import key_for, open_graph

    z = _latent(config_of(path).nz)
    runner = open_graph(path, backend=backend, device=device, precision="FP32")
    key = key_for(runner.backend, runner.asked)
    frames = {"FP32": np.asarray(runner.infer(z), np.float32)}
    # Dropped before the next one is built. Otherwise two compiled copies of a six-megapixel
    # graph are resident on the card at once, and this card has frozen the whole machine at a
    # 13.92 GB peak rather than raising out of memory.
    runner = open_graph(path, backend=runner.backend, device=runner.asked, precision="FP16")
    frames["FP16"] = np.asarray(runner.infer(z), np.float32)
    gap = levels(frames["FP16"], frames["FP32"])
    return key, ("FP16" if gap < HALF_LEVELS else "FP32"), gap


def adopt(source, out, seed: int = 0, target: float = TARGET_LEVELS,
          measure: bool = True, device: str = "cpu",
          re_adopt: bool = False) -> Adopted:
    """Read a graph, give it dials, prove them, and write it back out ready to play."""
    import onnx

    source, out = Path(source), Path(out)
    model = onnx.load(str(source))
    already = onnx_dials(source)["settings"]
    if already and not re_adopt:
        raise RuntimeError(
            f"{Path(source).name} already declares {len(already)} dial(s) "
            f"({', '.join(already[:4])}...). Adopting it again would insert a second "
            f"settings input and overwrite what is there -- and if those dials were tuned by "
            f"hand rather than derived, that tuning is not recoverable from the file. Pass "
            f"re_adopt=True if replacing them is what you want.")

    found = insert_dials(model, seed=seed)
    onnx.checker.check_model(model, full_check=False)
    # Named before it is written, so the intermediate file is already a model something can
    # open rather than a graph with an anonymous second input.
    name_settings(model, found.names)
    _write(model, out)

    if measure:
        # Precision first, because everything after it is measured through the runtime and a
        # runtime that has quietly broken the picture measures its own damage.
        cpu = device.lower() == "cpu"
        backend = "ort" if cpu else "auto"
        precision = "FP16"
        if not cpu:
            key, precision, gap = usable_precision(out, backend, device)
            print(f"half precision moves this graph {gap:.3f} 8-bit levels on {key}; "
                  f"measuring in {precision}", flush=True)
            found.precision = {key: precision}
        probe = Probe(out, device=device, precision=precision)
        drift = deterministic(probe)
        if drift > 0.01:
            raise RuntimeError(
                f"the graph still answers the same latent {drift:.2f} 8-bit levels apart "
                f"after freezing {found.noise} random draw(s). Something else in it is "
                f"non-deterministic, and every measurement below would be measuring that.")
        calibrate(probe, found, target=target, seed=seed)

    name_settings(model, found.names,
                  curves=[d.curve for d in found.dials],
                  levels=[d.moved for d in found.dials],
                  rests=[d.rest for d in found.dials],
                  precision=found.precision)
    _write(model, out)
    return found


def _write(model, out: Path) -> None:
    """Save the graph, with its weights beside it once they are too big to sit inside."""
    import onnx

    out.parent.mkdir(parents=True, exist_ok=True)
    big = model.ByteSize() > 1_800_000_000
    onnx.save(model, str(out), save_as_external_data=big,
              location=f"{out.name}.data" if big else None)
