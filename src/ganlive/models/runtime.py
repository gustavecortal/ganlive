"""Running an ONNX graph on whatever accelerator this machine actually has."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    device: str
    #: The device string that was *asked for*, which is what a verdict is filed under. The
    #: `device` above carries the card's full name, which is not a key.
    asked: str
    nz: int
    #: `(height, width)` of the first output.
    size: tuple[int, int]
    #: How many settings the graph takes as its second input. Zero if it takes none.
    settings: int
    #: The shape of every output, so a benchmark does not have to reach past this to get them.
    shapes: tuple[tuple[int, ...], ...]
    #: The precision this graph is actually being run in, resolved from what was measured.
    precision: str
    _run: Callable[..., np.ndarray] = dataclasses.field(repr=False, compare=False)
    #: Hand the runtime a host array of the first output's shape to write every frame into,
    #: or `None` where it only returns its own. See `OnnxGenerator`, which lands frames in
    #: pinned memory so the upload to the card is a DMA and not a staged copy.
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
    """The best runtime on this machine that will actually run this graph."""
    path = Path(path)
    said = None if precision else _said(path)
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

    `cpu` is ONNX Runtime; anything else names an OpenVINO device. **Asked once, by every pass.**
    The precision check used to go through `auto` -- which ignores the device and takes the
    first backend on `ORDER` that opens -- while the dials were calibrated on OpenVINO. On a
    CUDA machine the verdict was measured on, and filed under, a runtime the dials never saw."""
    if device.lower() == "cpu":
        return "ort", "CPUExecutionProvider"
    return "openvino", device


def key_for(backend: str, device: str) -> str:
    """How a precision verdict is filed: the pair it was measured on."""
    return f"{backend}/{device}"


def _said(path) -> object:
    from ganlive.models.onnx import dials_of

    return dials_of(path)["precision"]


def precision_for(said, backend: str, device: str, default: str = "FP16") -> str:
    """What precision this graph was measured to survive on this backend and device."""
    if not said:
        return default
    if isinstance(said, str):                       # written before the verdicts were keyed
        return said
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


def _openvino(path: Path, device: str, precision: str) -> Runner:
    """OpenVINO, which is what reaches an Intel GPU and is the fastest thing here."""
    from ganlive.models.onnx import compile_ov

    try:
        compiled, full_name = compile_ov(path, device, precision)
    except ImportError as exc:
        raise Unavailable(f"openvino is not installed: {exc}") from exc
    except RuntimeError as exc:
        # `compile_ov` already refuses a device this machine does not have, and says it
        # better than a second copy of the check here did.
        raise Unavailable(str(exc)) from exc
    request = compiled.create_infer_request()
    _n, _c, height, width = (d.get_length() for d in compiled.outputs[0].partial_shape)
    settings = (int(compiled.inputs[1].partial_shape[0].get_length())
                if len(compiled.inputs) > 1 else 0)

    def run(z, k=None):
        # **Never `request.infer()`**: it builds a result dict, and building it copies every
        # output into a fresh array -- 7.4 ms of 19.8 at 3072x2048. `start_async` then `wait`
        # leaves the frame in the request's own output tensor, which the GPU already wrote
        # into host-visible memory, so `.data` is a view. It dies at the next submission.
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


def _ort(path: Path, device: str, precision: str = "") -> Runner:
    """ONNX Runtime, which is what reaches everything that is not an Intel GPU."""
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

    settings_for = ort.SessionOptions()
    settings_for.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), settings_for, providers=[wanted])

    # **The load-bearing check, and the reason this function exists.** A provider that cannot load, or that
    # refuses the graph, is a warning on stderr -- not an exception -- and the session then runs on the CPU
    # looking entirely healthy.
    got = session.get_providers()
    if wanted not in got:
        raise Unavailable(
            f"{wanted} did not take this graph and onnxruntime fell back to {got} without "
            f"raising. That is a 45x slowdown wearing the fast path's name, so it is refused "
            f"here; pass the provider by name to see the runtime's own error.")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    shape = outputs[0].shape
    settings = int(inputs[1].shape[0]) if len(inputs) > 1 else 0
    names = [i.name for i in inputs]
    # **`None` means every output, and every output means a fresh allocation.** ONNX Runtime hands back new
    # numpy arrays, so asking for all of them on this project's own adopted graph allocated and copied 18.87
    # MB a frame -- 2.55 ms of a 10 ms budget -- plus a second head nothing reads.
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
                     f"{', '.join(ov.Core().available_devices)}")
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
