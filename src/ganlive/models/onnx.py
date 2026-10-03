"""An ONNX generator that plays like a torch one: `net(z) -> [big, small]`."""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ganlive.models.common import Ladder
from ganlive.settings import Knobs


@dataclass(frozen=True)
class OnnxConfig:
    """What the bank needs to know about a model, read out of the graph rather than a sidecar."""

    nz: int
    ladder: Ladder


def config_of(path) -> OnnxConfig:
    """Latent width and output size, from the graph's own declared shapes."""
    said = dials_of(path)
    height, width = said["size"]
    return OnnxConfig(nz=said["nz"],
                      ladder=Ladder(width=width, height=height))


def _dims(value) -> list[int]:
    return [d.dim_value for d in value.type.tensor_type.shape.dim]


def dials_of(path) -> dict:
    """Everything the file says about its own dials, in one read."""
    path = Path(path)
    stat = path.stat()
    return _dials_cached(str(path), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=8)
def _dials_cached(path: str, _mtime: int, _size: int) -> dict:
    """The read itself, keyed on the file's identity so a rewrite is not served stale."""
    import onnx

    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    # The prefix graphs carry that were exported under this project's earlier name.
    raw = {e.key.replace("smallgen.", "ganlive."): e.value
           for e in model.metadata_props if e.value}
    names = raw.get("ganlive.settings", "")
    # The precision verdict is keyed by backend and device -- one runtime breaking a graph is
    # not evidence about another's rounding. A bare string is a file written before it was
    # keyed, and `runtime.precision_for` takes it as the answer for everything.
    said = raw.get("ganlive.precision", "")
    out = {"settings": names.split(",") if names else [],
           "precision": json.loads(said) if said.startswith("{") else said}
    for key in ("rests", "curves", "levels"):
        value = raw.get(f"ganlive.{key}")
        out[key] = json.loads(value) if value else []
    # The shapes come out of the same parse rather than a second one. Small enough to hold:
    # two ints and a pair, against the `ModelProto` itself, which is 141 MB and stays out.
    out["nz"] = int(_dims(graph.input[0])[-1])
    out["size"] = (int(_dims(graph.output[0])[-2]), int(_dims(graph.output[0])[-1]))
    return out


def settings_of(path) -> list[str]:
    """The names of the dials the graph takes. `dials_of` for anything more."""
    return dials_of(path)["settings"]


@functools.lru_cache(maxsize=1)
def _core():
    """One OpenVINO `Core` for the process."""
    import openvino as ov

    return ov.Core()


def _constant(output):
    """The value behind a graph edge, if it is a constant."""
    node = output.get_node()
    return node.get_data() if node.get_type_name() == "Constant" else None


def _squared(output):
    """`x`, if this edge is `x * x` or `x ** 2`."""
    node = output.get_node()
    kind = node.get_type_name()
    if kind == "Multiply" and node.input_value(0) == node.input_value(1):
        return node.input_value(0)
    if kind == "Power":
        exponent = _constant(node.input_value(1))
        if exponent is not None and exponent.size == 1 and float(exponent.flat[0]) == 2.0:
            return node.input_value(0)
    return None


def wide_norms(model) -> int:
    """Rewrite every `sqrt(sum(x * x) + eps)` so that half precision cannot overflow it.
    Returns how many were rewritten.

    **This is why an adopted StyleGAN2 played in FP32.** Demodulation takes the norm of each
    modulated filter, and on FFHQ-1024 the sum of squares under that root reaches 3.7e6 -- past
    FP16's 65504, so the frame came back `inf`, 59 8-bit levels off, and the adoption correctly
    refused half precision: 93.8 ms a frame. The root itself is only 1.9e3. `ReduceL2` computes
    it without ever holding the square, and the GPU plugin accumulates it wide, so the graph
    plays in FP16 at 13.1 ms, 0.32 levels from FP32 with the frame handed over in FP16.

    The epsilon goes back in exactly, as `hypot(norm, sqrt(eps))` spelled so neither term is
    squared at full size: `big * sqrt(1 + (small / big)**2)`. It also keeps an all-zero filter
    finite, which a plain `norm * sqrt(1 + eps / norm**2)` does not."""
    from openvino import opset13 as ops

    rewritten = 0
    for root in [op for op in model.get_ordered_ops() if op.get_type_name() == "Sqrt"]:
        inner, eps = root.input_value(0), None
        if inner.get_node().get_type_name() == "Add":
            add = inner.get_node()
            for i in (0, 1):
                value = _constant(add.input_value(1 - i))
                if value is not None and value.size == 1:
                    eps, inner = float(value.flat[0]), add.input_value(i)
                    break
        total = inner.get_node()
        if total.get_type_name() != "ReduceSum":
            continue
        x = _squared(total.input_value(0))
        if x is None or (eps is not None and eps < 0):
            continue
        norm = ops.reduce_l2(x, total.input_value(1), total.get_attributes()["keep_dims"])
        if eps:
            floor = ops.constant(np.full([1], np.sqrt(eps), np.float32))
            big, small = ops.maximum(norm, floor), ops.minimum(norm, floor)
            ratio = ops.divide(small, big)
            one = ops.constant(np.ones([1], np.float32))
            norm = ops.multiply(big, ops.sqrt(ops.add(one, ops.multiply(ratio, ratio))))
        for user in list(root.output(0).get_target_inputs()):
            user.replace_source_output(norm.output(0))
        rewritten += 1
    if rewritten:
        model.validate_nodes_and_infer_types()
    return rewritten


def compile_ov(path, ov_device: str = "GPU", precision: str = "FP16"):
    """One compiled OpenVINO model and the device's full name."""
    from openvino import Type
    from openvino.preprocess import PrePostProcessor

    core = _core()
    if ov_device not in core.available_devices:
        raise RuntimeError(
            f"{ov_device} is not among {core.available_devices}. OpenVINO raises here on "
            f"purpose; the ONNX Runtime path fails this over to the CPU without saying so, "
            f"which reads as a 650 ms model rather than as a missing device.")
    model = core.read_model(str(path))

    if precision == "FP16":
        # Before the verdict is taken as well as before play: `usable_precision` compiles
        # through here, so the half precision it measures is the half precision that plays.
        wide_norms(model)
        # The graph computes in f16 and would otherwise convert the frame back to f32 to hand
        # it over -- twice the bytes, for a pipeline that is f16 from here to the window.
        # Declaring f16 outputs took the frame from 29.5 ms to 19.5 at 3072x2048.
        ppp = PrePostProcessor(model)
        for i in range(len(model.outputs)):
            ppp.output(i).tensor().set_element_type(Type.f16)
        model = ppp.build()

    compiled = core.compile_model(
        model, ov_device,
        {"INFERENCE_PRECISION_HINT": "f16" if precision == "FP16" else "f32",
         "PERFORMANCE_HINT": "LATENCY"})
    return compiled, core.get_property(ov_device, "FULL_DEVICE_NAME")


class OnnxGenerator:
    """One compiled graph, called once per frame, with the frame handed back on `device`."""

    #: The walk hands over its host view rather than a device tensor; `infer` takes it as is.
    latent_on_host = True

    def __init__(self, path, device: str | None = None, backend: str = "auto",
                 accelerator: str = "", precision: str = "") -> None:
        """`device` is where the *frame* is handed back -- a torch device, or `cpu`. Default is
        whichever accelerator this machine has."""
        from ganlive.device import detect_backend
        from ganlive.models.runtime import open_graph

        self.path = Path(path)
        self.device = device = device or detect_backend()
        # **Touch the torch device before OpenVINO touches it, and this is not optional.** Both runtimes
        # drive the same card through Level Zero.
        if device and device != "cpu":
            from ganlive.device import synchronize

            torch.zeros(1, device=device)
            synchronize(device)
        # The precision is `open_graph`'s to decide, from what adoption measured on the backend it ends up
        # choosing.
        said = dials_of(self.path)
        self.runner = open_graph(self.path, backend=backend, device=accelerator,
                                 precision=precision)
        self.settings = said["settings"]
        self.steerable = self.runner.settings > 0
        if self.steerable and len(self.settings) != self.runner.settings:
            raise RuntimeError(
                f"the graph takes {self.runner.settings} settings and names "
                f"{len(self.settings)} of them; refusing to guess which dial is which")
        self.precision = self.runner.precision
        self.cfg = OnnxConfig(nz=self.runner.nz,
                              ladder=Ladder(width=self.runner.size[1],
                                            height=self.runner.size[0],))
        self.nz = self.cfg.nz
        self._z = np.zeros((1, self.nz), dtype=np.float32)
        # The generator owns its settings, so `net(z)` stays a one-argument call and nothing
        # above here learns there are two backends. On the host in f32, because that is what
        # the graph takes and where the values already live.
        self.knobs = Knobs(self.settings if self.steerable else [], "cpu", torch.float32)
        # The graph is handed `committed()`, never `vec`, so a commit that also copied into
        # `vec` was paying `copy_`'s GIL release for a tensor nothing reads. Fed from itself,
        # a commit is the host write alone.
        self.knobs.feed_from(self.knobs.vec)
        #: Where the runtime writes each frame, when that can be pinned memory: the upload to
        #: the card is then a DMA rather than a copy staged through a pinned buffer of the
        #: driver's -- 22.8 ms to 19.3 a frame at 3072x2048. `None` on the CPU, or where the
        #: runtime only hands back arrays of its own.
        self._landing = None
        if device != "cpu" and self.runner.land is not None:
            from ganlive.pixels import pinned

            host = pinned(self.runner.shapes[0], torch.from_numpy(
                np.zeros(0, self.runner.dtype)).dtype)
            if host.is_pinned():
                self.runner.land(host.numpy())
                self._landing = host

    def __call__(self, z) -> list[torch.Tensor]:
        """One frame, on `device`, as a torch generator would return it."""
        frame = self.infer(z)
        host = self._landing if self._landing is not None else torch.from_numpy(frame)
        # Blocking, so the upload has read the buffer before the next frame is written into it.
        out = host.to(self.device)
        # On the CPU `.to` is a no-op, and the runtime writes the next frame into this buffer.
        return [out.clone() if out.data_ptr() == host.data_ptr() else out]

    def infer(self, z) -> np.ndarray:
        """One frame as OpenVINO left it: a view into host-visible memory."""
        if isinstance(z, np.ndarray):
            self._z[:] = z.reshape(1, self.nz)
        else:
            # The graph takes f32 whatever it computes in, and the walk hands over f16 on the
            # card. One 1 KB download, against 37 MB coming the other way.
            self._z[:] = z.detach().to("cpu", torch.float32).reshape(1, self.nz).numpy()

        return self.runner.infer(self._z, self.knobs.committed() if self.steerable else None)

    def eval(self):
        return self

    def report(self) -> str:
        dials = (f"{len(self.settings)} settings" if self.steerable
                 else "settings frozen at export")
        return (f"onnx {self.path.name}: {self.runner.report()}, {self.precision}, {dials}, "
                f"frame handed back on {self.device}")
