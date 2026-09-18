"""Turning a graph nobody here wrote into something with dials on it."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The ops that make a graph give a different answer to the same question.
RANDOM = ("RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike")

#: The ops a generator ends in. A gain into one of these is the dial that washes the picture
#: out below 1 and hard-clips it above.
SQUASH = ("Tanh", "Sigmoid")

#: A derived dial moving less than this many mean 8-bit levels at full travel is not a
#: control. Imported rather than restated: it is the floor the latent directions already use,
#: for the same reason, and two copies of one threshold drift.
from ganlive.dials.derive import FLOOR_LEVELS  # noqa: E402
from ganlive.models.onnx import dials_of as onnx_dials  # noqa: E402

#: What a calibrated dial should buy at full travel, in mean 8-bit levels. Chosen so that
#: every dial on every model feels like the same amount of change under the hand, which is
#: the entire argument for calibrating rather than shipping a range.
TARGET_LEVELS = 25.0

#: How far half precision may move the picture before it is refused, in mean 8-bit levels. This project's
#: own generator measures 0.100 there and has shipped in fp16 for months.
HALF_LEVELS = 1.0

#: Where a gain dial is allowed to be searched.
GAIN_RANGE = (1e-4, 4.0)
NOISE_RANGE = (1e-3, 400.0)


@dataclass
class Dial:
    """One derived control: where it was inserted, and what it turned out to do."""

    name: str
    #: The tensor the `Multiply` was placed on, for the report and for debugging a surprise.
    tensor: str
    #: What to feed this slot at each of `len(curve)` evenly spaced dial positions from 0 to 1. Three points
    #: was not enough: the response is logarithmic where a gain meets an instance norm, and a straight line
    #: through three of them put three of five rendered frames at visibly the same picture.
    curve: tuple[float, ...] = (0.0, 1.0, 2.0)
    #: Where on its own travel this dial sits when nothing is touching it. 0.5 for a control
    #: that works both ways; 0.0 or 1.0 for one that works only one way.
    rest: float = 0.5
    #: Mean 8-bit levels at each end, once measured. `None` before that.
    levels: tuple[float, float] | None = None

    @property
    def moved(self) -> float:
        return 0.0 if self.levels is None else max(self.levels)


@dataclass
class Adopted:
    """What adoption found, for the report and for the tests to assert against."""

    dials: list[Dial] = field(default_factory=list)
    #: How many noise patterns there are, however they got that way.
    noise: int = 0
    #: Whether they were already constants when the graph arrived, rather than live draws.
    was_baked: bool = False
    #: The `(height, width)` of every resolution band found in the graph.
    bands: tuple[tuple[int, int], ...] = ()
    #: What precision this graph survives, per `backend/device` -- one runtime breaking it is
    #: not evidence about another's rounding. Empty until something measures a pair.
    precision: dict = field(default_factory=dict)
    nz: int = 0
    size: tuple[int, int] = (0, 0)
    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dials]

    @property
    def dropped(self) -> list[str]:
        """Dials that were driven and moved nothing. Derived, so it cannot disagree."""
        return [d.name for d in self.dials if d.moved < FLOOR_LEVELS]

    def report(self) -> str:
        live = ", ".join(f"{d.name} {d.moved:.1f}" for d in self.dials) or "none"
        noise = (f"{self.noise} noise pattern(s) already frozen in the file"
                 if self.was_baked else f"{self.noise} random draw(s) frozen")
        bands = ", ".join(f"{w}x{h}" for h, w in self.bands)
        said = ", ".join(f"{k} {v}" for k, v in self.precision.items()) or "unmeasured"
        out = (f"{self.nz} latent, {self.size[1]}x{self.size[0]}, {said}, {noise}, "
               f"bands {bands}\n"
               f"  dials (mean 8-bit levels at full travel): {live}")
        if self.dropped:
            out += f"\n  dropped as inert: {', '.join(self.dropped)}"
        return out


def _shapes(model) -> dict[str, tuple[int, ...]]:
    """Every tensor's static shape, from shape inference. Unknown dims are dropped."""
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    out: dict[str, tuple[int, ...]] = {}
    for group in (inferred.graph.value_info, inferred.graph.input, inferred.graph.output):
        for value in group:
            dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
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


def _latent(nz: int, seed: int = 0) -> np.ndarray:
    """The latent every measurement in this module is taken on.

    One spelling, because `usable_precision` decides fp16 against fp32 on it and `calibrate`
    then measures every dial on it: drawn differently in the two places, the precision verdict
    would belong to a picture no dial was ever calibrated against, with nothing to show it."""
    return np.random.default_rng(seed).standard_normal((1, nz)).astype(np.float32)


class _Probe:
    """The three things `calibrate` asks of a model, and nothing else about it."""

    nz: int
    settings: int

    def latent(self, seed: int = 0) -> np.ndarray:
        return _latent(self.nz, seed)

    def neutral(self) -> np.ndarray:
        return np.ones(self.settings, np.float32)

    def frame(self, z, k=None) -> np.ndarray:
        raise NotImplementedError


class Probe(_Probe):
    """The adopted graph, runnable on the host, so a dial can be asked what it does."""

    def __init__(self, model, device: str = "cpu", precision: str = "FP16") -> None:
        """`device` is `cpu` for ONNX Runtime, or an OpenVINO device name such as `GPU`."""
        from ganlive.models.runtime import open_graph

        cpu = device.lower() == "cpu"
        self.runner = open_graph(model, backend="ort" if cpu else "openvino",
                                 device="CPUExecutionProvider" if cpu else device,
                                 precision=precision)
        self.nz, self.settings = self.runner.nz, self.runner.settings

    def frame(self, z, k=None) -> np.ndarray:
        # A copy, not the view an OpenVINO request hands back: it dies at the next
        # submission, and every measurement here compares one frame with another taken later.
        return np.array(self.runner.infer(z, self.neutral() if k is None else k), np.float32)


def levels(a, b) -> float:
    """Mean absolute difference in 8-bit levels -- the unit every measurement here is in."""
    return float(np.abs(a - b).mean() * 127.5)


def deterministic(probe: Probe, seed: int = 0) -> float:
    """8-bit levels between two answers to the same question. Zero, or the graph is unusable."""
    z = probe.latent(seed)
    return levels(probe.frame(z), probe.frame(z))


def _sweep(probe: Probe, z, base, slot: int, limit: float,
           samples: int, stop: float | None = None) -> list[tuple[float, float]]:
    """What this slot does to the picture, logarithmically from rest out to `limit`.

    **It stops at `stop`, because nothing downstream can see past it.** `_knots` takes the
    *first* crossing of each target and the largest target is `stop`; `Dial.levels` is capped at
    it; the live-half test asks only for the smallest. So every sample after the first one at or
    above `stop` is a forward pass rendered and discarded, and dropping them changes no knot, no
    level and no verdict. A side that never gets there is untouched and still costs the lot.
    """
    # The first sample is `exp(0) == 1.0`, which is rest: the same frame `base` already is,
    # measuring 0.0 levels, which `_knots` can never choose. Rendering it was 22 wasted
    # forwards per adopt out of 531.
    out = [(1.0, 0.0)]
    k = probe.neutral()
    for value in np.exp(np.linspace(0.0, math.log(limit), samples))[1:]:
        k[slot] = value
        out.append((float(value), levels(probe.frame(z, k), base)))
        if stop is not None and out[-1][1] >= stop:
            break
    k[slot] = 1.0
    return out


def _knots(sweep, targets) -> list[float]:
    """The value that first reaches each target level, read off the sweep."""
    out = []
    for target in targets:
        value = sweep[-1][0]
        for (k0, l0), (k1, l1) in zip(sweep, sweep[1:], strict=False):
            if l1 >= target:
                share = 0.0 if l1 == l0 else (target - l0) / (l1 - l0)
                lo, hi = math.log(k0), math.log(k1)
                value = math.exp(lo + max(0.0, min(1.0, share)) * (hi - lo))
                break
        out.append(_sig(value))
    return out


def _sig(x: float, digits: int = 4) -> float:
    """A calibrated value, to four significant figures rather than four decimal places."""
    return 0.0 if x == 0 else float(f"{x:.{digits}g}")


class TorchProbe(_Probe):
    """The same contract as `Probe`, for a network that is a network rather than a graph."""

    def __init__(self, net, knobs, nz: int, device="cpu", dtype=None) -> None:
        self.net, self.knobs, self.nz, self.device = net, knobs, nz, device
        #: The dtype the latent arrives in *at play time*. Handing a compiled graph a latent
        #: of a different dtype than it was traced with recompiles the whole thing, which on
        #: a StyleGAN2 is a hundred seconds, in the middle of measuring it.
        self.dtype = dtype
        self.settings = len(getattr(knobs, "names", ()) or ())

    def frame(self, z, k=None) -> np.ndarray:
        import torch

        from ganlive.models.fastgan import first_image

        if self.settings:
            self.knobs.write[:] = self.neutral() if k is None else k
            self.knobs.commit()
        latent = torch.from_numpy(z).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            out = first_image(self.net(latent))
        return np.asarray(out.float().cpu().numpy(), np.float32)


def measured(probe, names, size=(0, 0), **kw) -> Adopted:
    """Calibrate a set of dials that something else installed, and report them as `Adopted`.

    The dial's name *is* its tensor here: the installer already put them where they go, and
    nothing downstream of this reads `tensor` on a dial it did not insert itself."""
    found = Adopted(dials=[Dial(name=n, tensor=n) for n in names], nz=probe.nz, size=size)
    return calibrate(probe, found, **kw)


def calibrate(probe: Probe, found: Adopted, target: float = TARGET_LEVELS,
              knots: int = 4, samples: int = 24, seed: int = 0) -> Adopted:
    """Measure every derived dial and give each the travel that buys the same change."""
    z = probe.latent(seed)
    base = probe.frame(z)
    steps = [target * (i + 1) / knots for i in range(knots)]
    for slot, dial in enumerate(found.dials):
        noise = dial.name.startswith("noise_")
        floor_v, ceiling = NOISE_RANGE if noise else GAIN_RANGE
        up = _sweep(probe, z, base, slot, ceiling, samples, stop=steps[-1])
        down = _sweep(probe, z, base, slot, floor_v, samples, stop=steps[-1])
        reached = (max(l for _k, l in down), max(l for _k, l in up))
        # What full travel actually buys, which is the target for a dial that gets there and
        # less for one that does not -- so a weak dial reads as weak rather than as its
        # headroom. `reached` is what the range limit could do; the dial stops at its knots.
        dial.levels = (round(min(reached[0], target), 3), round(min(reached[1], target), 3))

        # **A side that cannot buy one knot is not a side.** The floor asks whether the dial does anything
        # at all; a *half* has to be worth turning.
        live = (reached[0] >= steps[0], reached[1] >= steps[0])

        # **A dial with a dead half is a dial that lies about where it is.** Scaling a feature band *up*
        # buys 0.2 levels on this StyleGAN and scaling it down buys 91, because the instance norm downstream
        # divides a constant gain straight back out.
        if all(live):
            dial.curve = tuple(_knots(down, steps)[::-1]) + (1.0,) + tuple(_knots(up, steps))
            dial.rest = 0.5
        elif live[0]:
            dial.curve = tuple(_knots(down, steps)[::-1]) + (1.0,)
            dial.rest = 1.0
        else:
            dial.curve = (1.0,) + tuple(_knots(up, steps))
            dial.rest = 0.0
    return found


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
