"""Giving an ONNX graph nobody here wrote a set of dials, and proving each one moves the picture.

Freeze every random draw so the same latent gives the same pixels, find the resolution bands,
insert a `Mul` on each fed from a new settings input, measure what each buys (see
`calibrate`), and write all of it into the file's own metadata. Whatever opens
the result needs to know nothing about the architecture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ganlive.models import runtime
from ganlive.models.calibrate import (
    TARGET_LEVELS,
    Adopted,
    Dial,
    Probe,
    calibrate,
    deterministic,
)
from ganlive.models.common import host_latent
from ganlive.models.onnx_file import (
    dials_of,
    dims,
    initializers,
    name_settings,
    producers,
    weights_file,
)
from ganlive.pixels import levels

#: The ops that make a graph give a different answer to the same question.
RANDOM = ("RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike")

#: The ops a generator ends in. A gain into one of these only changes contrast, so no dial
#: goes there.
SQUASH = ("Tanh", "Sigmoid")

#: Ops that can sit between a squash and the graph's output without being the picture: a
#: denormalise, a layout change, a cast to bytes.
AFTER_SQUASH = ("Mul", "Add", "Sub", "Div", "Clip", "Cast", "Transpose", "Reshape",
                "Squeeze", "Unsqueeze", "Identity", "Round")

#: How far half precision may move the picture before it is refused, in mean 8-bit levels.
HALF_LEVELS = 1.0

#: The smallest spatial side that counts as a resolution band.
MIN_BAND = 4

#: Above this a graph's weights are saved beside it rather than inside: protobuf's limit is 2 GB.
INLINE_BYTES = 1_800_000_000


def _shapes(model) -> dict[str, tuple[int, ...]]:
    """Every tensor's static shape, from shape inference, with a symbolic batch read as 1.

    A tensor with any other unknown dimension is left out. The batch is the one dimension an
    exporter routinely leaves symbolic, and this plays one frame."""
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    out: dict[str, tuple[int, ...]] = {}
    for group in (inferred.graph.value_info, inferred.graph.input, inferred.graph.output):
        for value in group:
            shape = dims(value)
            if len(shape) >= 2 and shape[0] <= 0:
                shape[0] = 1
            if shape and all(d > 0 for d in shape):
                out[value.name] = tuple(shape)
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
    """Insert `tensor * k[slot]` at node index `at` and rewire everything downstream onto it."""
    from onnx import helper, numpy_helper

    take = f"ganlive_take_{tag}"
    scaled = f"ganlive_scaled_{tag}"
    for name, value in ((f"{take}_start", [slot]), (f"{take}_end", [slot + 1])):
        graph.initializer.append(
            numpy_helper.from_array(np.array(value, np.int64), name))
    # No `axes` input: the settings vector has one dimension, which is the default.
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


def _feeds(graph, produced, target: str) -> set[str]:
    """Every tensor `target` depends on."""
    seen, edge = set(), [target]
    while edge:
        name = edge.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in produced:
            edge.extend(graph.node[produced[name]].input)
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


def _bands(graph, shapes, size, mine) -> dict[tuple[int, int], tuple[str, int]]:
    """The last tensor at each spatial size the output depends on, as
    `{(h, w): (tensor, node index)}`."""
    consumed = {name for node in graph.node for name in node.input}
    rungs = _ladder(size) if any(size) else None
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


def _squash(graph) -> str | None:
    """The input to the squash the graph's first output comes out of, if there is one."""
    produced = producers(graph)
    name = graph.output[0].name
    for _step in range(len(AFTER_SQUASH) + 4):
        if name not in produced:
            return None
        node = graph.node[produced[name]]
        if node.op_type in SQUASH:
            return node.input[0]
        if node.op_type not in AFTER_SQUASH:
            return None
        name = node.input[0]
    return None


def _constant(graph) -> set[str]:
    """Every tensor whose value does not depend on any graph input."""
    live = {value.name for value in graph.input}
    for node in graph.node:
        if live.intersection(node.input):
            live.update(node.output)
    return (set(producers(graph)) | set(initializers(graph))) - live


def _baked_noise(graph, shapes, size, mine) -> list[tuple[str, tuple[int, ...]]]:
    """Noise patterns frozen before the export, as `(tensor, shape)` per pattern: a constant
    at least two-dimensional, added to a feature map of the same spatial size."""
    fixed = _constant(graph)
    shapes = {**{n: tuple(t.dims) for n, t in initializers(graph).items()}, **shapes}
    out = []
    for node in graph.node:
        if node.op_type != "Add" or node.output[0] not in mine:
            continue
        for name, other in (node.input[0], node.input[1]), (node.input[1], node.input[0]):
            shape, feature = shapes.get(name), shapes.get(other)
            # Two dimensions is enough: StyleGAN2 stores its noise as `(H, W)` and lets it
            # broadcast over batch and channel.
            if (name in fixed and shape is not None and feature is not None
                    and len(shape) >= 2 and len(feature) == 4
                    and shape[-2:] == feature[-2:]
                    and min(shape[-2:]) >= MIN_BAND
                    and (not any(size) or shape[-2] <= size[0])):
                out.append((name, tuple(shape)))
                break
    return out


def insert_dials(model, seed: int = 0, feed: str = "k") -> Adopted:
    """Freeze the randomness, add the dials, and give the graph the input that drives them."""
    from onnx import TensorProto, helper

    graph = model.graph
    halves = [t.name for t in graph.initializer if t.data_type == TensorProto.FLOAT16]
    if halves:
        raise RuntimeError(
            f"this graph stores {len(halves)} weight(s) in half precision. The dials and the "
            f"frozen noise are float32, which ONNX will not multiply into a float16 tensor. "
            f"Export it in float32: the runtime chooses half precision itself, per device, "
            f"once it has measured that the picture survives it.")
    shapes = _shapes(model)
    found = Adopted(nz=int(shapes[graph.input[0].name][-1]))
    out_shape = shapes.get(graph.output[0].name)
    if out_shape and len(out_shape) == 4:
        found.size = (int(out_shape[2]), int(out_shape[3]))

    frozen = _freeze_random(graph, shapes, seed)
    found.noise = len(frozen)
    mine = _feeds(graph, producers(graph), graph.output[0].name)
    if not frozen:
        # No random ops does not mean no noise: it may have been frozen before the export.
        frozen = _baked_noise(graph, shapes, found.size, mine)
        found.noise, found.was_baked = len(frozen), True

    # One dial per band of noise, not one per draw: a generator injects at several points per
    # resolution, and a dial each would be a page of controls that move together.
    by_size: dict[int, list[str]] = {}
    for name, shape in frozen:
        by_size.setdefault(int(shape[-2]), []).append(name)

    bands = _bands(graph, shapes, found.size, mine)
    found.bands = tuple(sorted(bands))
    wanted: list[tuple[str, list[str]]] = []
    wanted += [(f"noise_{h}", by_size[h]) for h in sorted(by_size)]
    wanted += [(name, [bands[size][0]]) for size, name in _band_names(bands)]

    # One dial per tensor, and none on the squash's input, where a gain is only a contrast
    # curve on the finished picture. A band left with no tensor of its own is dropped.
    squash = _squash(graph)
    claimed: set[str] = set() if squash is None else {squash}
    deduped: list[tuple[str, list[str]]] = []
    for name, tensors in wanted:
        fresh = [t for t in tensors if t not in claimed]
        if fresh:
            claimed.update(fresh)
            deduped.append((name, fresh))
    wanted = deduped

    # Last in the graph first, so each insertion leaves the positions of those still to come.
    produced = producers(graph)
    order = sorted(((produced.get(t, -1), slot, t)
                    for slot, (_name, ts) in enumerate(wanted) for t in ts), reverse=True)
    for tag, (index, slot, tensor) in enumerate(order):
        _gate(graph, tensor, slot, feed, f"{wanted[slot][0]}_{tag}", index + 1)

    found.dials = [Dial(name=name, tensor=tensors[0]) for name, tensors in wanted]
    graph.input.append(helper.make_tensor_value_info(feed, TensorProto.FLOAT, [len(wanted)]))
    return found


def usable_precision(path, backend: str, device: str) -> tuple[str, str, float, object]:
    """Whether this graph survives half precision on this backend and device: the key the
    verdict is filed under, the verdict, the gap, and the FP16 runner it was measured with."""
    z = host_latent(dials_of(path)["nz"])
    runner = runtime.open_graph(path, backend=backend, device=device, precision="FP32")
    key, backend, device = (runtime.key_for(runner.backend, runner.asked), runner.backend,
                            runner.asked)
    # A copy: the runner's output is a view into its own buffer, which is about to go.
    frames = {"FP32": np.array(runner.infer(z), np.float32)}
    # Dropped before the FP16 one is compiled, so two compiled copies of a large graph are
    # never resident on the card at once.
    del runner
    runner = runtime.open_graph(path, backend=backend, device=device, precision="FP16")
    frames["FP16"] = np.asarray(runner.infer(z), np.float32)
    gap = levels(frames["FP16"], frames["FP32"])
    return key, ("FP16" if gap < HALF_LEVELS else "FP32"), gap, runner


def adopt(source, out, seed: int = 0, target: float = TARGET_LEVELS,
          measure: bool = True, device: str = "cpu") -> Adopted:
    """Read a graph, give it dials, prove them, and write it back out ready to play."""
    import onnx

    source, out = Path(source), Path(out)
    model = onnx.load(str(source))
    already = dials_of(source)["settings"]
    if already:
        raise RuntimeError(
            f"{source.name} already declares {len(already)} dial(s) "
            f"({', '.join(already[:4])}...). Adopting it again would insert a second "
            f"settings input and overwrite what is there, and a hand-tuned dial cannot be "
            f"recovered from the file. Adopt the graph it was made from instead.")

    found = insert_dials(model, seed=seed)
    onnx.checker.check_model(model, full_check=False)
    # Named before it is written, so the intermediate file can already be opened.
    name_settings(model, found.names)
    _write(model, out)

    if measure:
        # Precision first: everything after it is measured through the runtime, and a runtime
        # that breaks the picture would be measuring its own damage.
        precision, runner = "FP16", None
        if device.lower() != "cpu":
            key, precision, gap, runner = usable_precision(out, *runtime.measuring_on(device))
            print(f"half precision moves this graph {gap:.3f} 8-bit levels on {key}; "
                  f"measuring in {precision}", flush=True)
            found.precision = {key: precision}
            if precision != "FP16":
                runner = None          # dropped before the FP32 one is compiled; see above
        # The dials are measured on the runner the verdict came from: the same backend and
        # device, and one compile fewer.
        probe = Probe(out, device=device, precision=precision, runner=runner)
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
    big = model.ByteSize() > INLINE_BYTES
    onnx.save(model, str(out), save_as_external_data=big,
              location=weights_file(out).name if big else None)
