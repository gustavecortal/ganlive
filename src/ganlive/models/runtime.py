"""Running an ONNX graph on whatever accelerator this machine has.

OpenVINO reaches Intel GPUs and NPUs; ONNX Runtime reaches everything else. `open_graph`
tries them in order and hands back a `Runner`, whichever one opened.
"""
from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ganlive.models.onnx_file import dials_of

#: ONNX Runtime's GPU providers, in the order `auto` tries them.
ORT_GPU = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
           "MIGraphXExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider")

#: What `auto` tries, in order: `(backend, device)`. A backend that is not installed, or a
#: device that is not present, is skipped rather than raising.
ORDER = (("openvino", "GPU"), ("openvino", "NPU"), ("ort", "gpu"),
         ("openvino", "CPU"), ("ort", "CPUExecutionProvider"))


class Unavailable(RuntimeError):
    """This backend cannot run this graph here. Tried in turn by `open_graph`."""


@dataclass
class Runner:
    """One compiled graph and what is known about it, whichever runtime compiled it."""

    backend: str
    #: The device, with the card's full name where the runtime gives one.
    device: str
    #: The device string that was asked for, which is what a precision verdict is filed under.
    asked: str
    nz: int
    #: `(height, width)` of the first output.
    size: tuple[int, int]
    #: How many settings the graph takes as its second input. Zero if it takes none.
    settings: int
    #: The shape of every output.
    shapes: tuple[tuple[int, ...], ...]
    #: The precision this graph runs in.
    precision: str
    _run: Callable[..., np.ndarray] = dataclasses.field(repr=False, compare=False)
    #: Hand the runtime a host array of the first output's shape to write every frame into,
    #: or `None` where it only returns arrays of its own. See `OnnxGenerator`.
    land: Callable[[np.ndarray], None] | None = dataclasses.field(default=None, repr=False,
                                                                  compare=False)
    #: The element type the first output arrives in.
    dtype: type = np.float32

    @property
    def outputs(self) -> int:
        return len(self.shapes)

    def infer(self, z, k=None) -> np.ndarray:
        """One frame, in host memory. `z` is `(1, nz)` float32; `k` is the settings vector."""
        return self._run(z, k)

    def report(self) -> str:
        return (f"{self.backend} on {self.device}: {self.size[1]}x{self.size[0]}, "
                f"latent {self.nz}, {self.settings} setting(s), {self.outputs} output(s)"
                + (f", {self.precision}" if self.precision else ""))


def open_graph(path, backend: str = "auto", device: str = "",
               precision: str = "") -> Runner:
    """The best runtime on this machine that will actually run this graph.

    `precision` empty means the one adoption measured for the backend and device opened."""
    path = Path(path)
    said = None if precision else dials_of(path)["precision"]
    if backend != "auto":
        return _open(path, backend, device,
                     precision or precision_for(said, backend, device))

    tried = []
    for name, want in ORDER:
        try:
            return _open(path, name, want,
                         precision or precision_for(said, name, want))
        except (Unavailable, ImportError) as exc:
            tried.append(f"{name}/{want}: {exc}")
        except Exception as exc:                    # noqa: BLE001  a backend refusing a graph
            tried.append(f"{name}/{want}: {type(exc).__name__}: {exc}")
    raise RuntimeError("nothing on this machine would run this graph.\n  "
                       + "\n  ".join(tried) + "\n\nwhat is here:\n  "
                       + survey().replace("\n", "\n  "))


def measuring_on(device: str) -> tuple[str, str]:
    """`adopt --device` as the `(backend, device)` every pass of an adoption opens.

    `cpu` is ONNX Runtime; anything else names an OpenVINO device. Every pass asks this, so
    the precision verdict and the dials are measured on the same runtime."""
    if device.lower() == "cpu":
        return "ort", "CPUExecutionProvider"
    return "openvino", device.upper()           # OpenVINO's device names are case-sensitive


def key_for(backend: str, device: str) -> str:
    """How a precision verdict is filed: the pair it was measured on."""
    return f"{backend}/{device}"


def precision_for(said: dict | None, backend: str, device: str,
                  default: str = "FP16") -> str:
    """What precision this graph was measured to survive on this backend and device.

    An unmeasured pair takes FP32 if any pair needed it, and `default` otherwise."""
    if not said:
        return default
    here = said.get(key_for(backend, device))
    if here:
        return here
    return "FP32" if any(v == "FP32" for v in said.values()) else default


def _open(path: Path, backend: str, device: str, precision: str) -> Runner:
    if backend == "openvino":
        return _openvino(path, device or "GPU", precision)
    if backend == "ort":
        return _ort(path, device or "gpu", precision)
    raise ValueError(f"{backend!r} is not a backend; have openvino, ort, auto")


@functools.lru_cache(maxsize=1)
def _core():
    """One OpenVINO `Core` for the process."""
    import openvino as ov

    return ov.Core()


def _constant(output):
    """The value behind an OpenVINO graph edge, if it is a constant."""
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

    StyleGAN2's demodulation takes the norm of each modulated filter. On FFHQ-1024 the sum
    of squares reaches 3.7e6, past FP16's 65504, while the root is only 1.9e3. `ReduceL2`
    computes the root without holding the square, and the GPU plugin accumulates it wide.

    The epsilon goes back in exactly, as `hypot(norm, sqrt(eps))` written so neither term is
    squared at full size: `big * sqrt(1 + (small / big)**2)`. That form also keeps an all-zero
    filter finite."""
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
        # Raised rather than left to OpenVINO, which could fall back to another device and
        # read as a slow model rather than a missing device.
        raise RuntimeError(f"{ov_device} is not among {core.available_devices}")
    model = core.read_model(str(path))

    if precision == "FP16":
        # The adoption's precision check compiles through here too, so the half precision it
        # measures is the half precision that plays.
        wide_norms(model)
        # Hand the frame over in f16 rather than converting it back to f32: half the bytes,
        # for a pipeline that is f16 from here to the window.
        ppp = PrePostProcessor(model)
        for i in range(len(model.outputs)):
            ppp.output(i).tensor().set_element_type(Type.f16)
        model = ppp.build()

    compiled = core.compile_model(
        model, ov_device,
        {"INFERENCE_PRECISION_HINT": "f16" if precision == "FP16" else "f32",
         "PERFORMANCE_HINT": "LATENCY"})
    return compiled, core.get_property(ov_device, "FULL_DEVICE_NAME")


def _openvino(path: Path, device: str, precision: str) -> Runner:
    """OpenVINO: what reaches an Intel GPU, and the fastest thing there."""
    try:
        compiled, full_name = compile_ov(path, device, precision)
    except ImportError as exc:
        raise Unavailable(f"openvino is not installed: {exc}") from exc
    except RuntimeError as exc:
        raise Unavailable(str(exc)) from exc
    request = compiled.create_infer_request()
    _n, _c, height, width = (d.get_length() for d in compiled.outputs[0].partial_shape)
    settings = (int(compiled.inputs[1].partial_shape[0].get_length())
                if len(compiled.inputs) > 1 else 0)

    def run(z, k=None):
        # `start_async` then `wait`, never `request.infer()`, which copies every output into
        # a fresh array. This leaves the frame in the request's own output tensor, already in
        # host-visible memory, so `.data` is a view. It is overwritten by the next submission.
        feed = {0: z}
        if settings:
            feed[1] = np.ones(settings, np.float32) if k is None else k
        request.start_async(feed)
        request.wait()
        return request.get_output_tensor(0).data

    def land(host):
        import openvino as ov

        request.set_output_tensor(0, ov.Tensor(host, shared_memory=True))

    return Runner(backend="openvino", device=f"{device} ({full_name})", asked=device,
                  nz=int(compiled.inputs[0].partial_shape[-1].get_length()),
                  size=(int(height), int(width)), settings=settings,
                  shapes=tuple(tuple(d.get_length() for d in out.partial_shape)
                               for out in compiled.outputs),
                  precision=precision, _run=run, land=land,
                  # `compile_ov` declares half-precision outputs for an FP16 graph.
                  dtype=np.float16 if precision == "FP16" else np.float32)


def _ort(path: Path, device: str, precision: str) -> Runner:
    """ONNX Runtime: what reaches everything that is not an Intel GPU."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise Unavailable(f"onnxruntime is not installed: {exc}") from exc

    have = ort.get_available_providers()
    if device == "gpu":
        wanted = next((p for p in ORT_GPU if p in have), None)
        if wanted is None:
            raise Unavailable(f"this onnxruntime build has no GPU provider, only {have}")
    else:
        wanted = device
        if wanted not in have:
            raise Unavailable(f"{wanted} is not in this onnxruntime build: {have}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=[wanted])

    # A provider that cannot load, or that refuses the graph, is only a warning on stderr,
    # and the session then runs on the CPU. Checked here so that is an error instead.
    got = session.get_providers()
    if wanted not in got:
        raise Unavailable(
            f"{wanted} did not take this graph and onnxruntime fell back to {got} without "
            f"raising. That is a large slowdown under the fast path's name, so it is refused "
            f"here; pass the provider by name to see the runtime's own error.")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    shape = outputs[0].shape
    settings = int(inputs[1].shape[0]) if len(inputs) > 1 else 0
    names = [i.name for i in inputs]
    # Only the first output: ONNX Runtime allocates a fresh array for every output it returns.
    fetch = [outputs[0].name]

    def run(z, k=None):
        feed = {names[0]: z}
        if settings:
            feed[names[1]] = np.ones(settings, np.float32) if k is None else k
        return session.run(fetch, feed)[0]

    return Runner(backend="ort", device=wanted, asked=wanted,
                  nz=int(inputs[0].shape[-1]),
                  size=(int(shape[-2]), int(shape[-1])), settings=settings,
                  shapes=tuple(tuple(o.shape) for o in outputs),
                  precision=precision, _run=run)


def survey() -> str:
    """What this machine could run a graph on. For a report, and for a bug report."""
    lines = []
    try:
        import openvino as ov

        lines.append(f"openvino {ov.__version__.split('-')[0]}: "
                     f"{', '.join(_core().available_devices)}")
    except ImportError:
        lines.append("openvino: not installed")
    try:
        import onnxruntime as ort

        have = ort.get_available_providers()
        gpu = [p for p in ORT_GPU if p in have]
        lines.append(f"onnxruntime {ort.__version__}: {', '.join(have)}"
                     + (f"  (GPU: {', '.join(gpu)})" if gpu else "  (no GPU provider)"))
    except ImportError:
        lines.append("onnxruntime: not installed")
    return "\n".join(lines)
