"""Adopting a graph nobody here wrote: dials derived from it, and proved on it.

CPU only, like the rest of the suite. The fixture is a small generator built to have the
three things adoption keys on -- live random draws, a resolution ladder, and a final squash
-- and deliberately not built like this project's own generator, because a test that only
passes on the architecture the code was written for is not testing generality.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch
from torch import nn


class _Noisy(nn.Module):
    """A tiny generator with a live noise draw at every resolution, as a StyleGAN exports one."""

    def __init__(self, nz: int = 16, base: int = 4, rungs: int = 3, gain: float = 0.6) -> None:
        super().__init__()
        self.base = base
        self.stem = nn.Linear(nz, 32 * base * base)
        self.blocks = nn.ModuleList()
        width = 32
        for _ in range(rungs):
            self.blocks.append(nn.Conv2d(width, width, 3, padding=1))
        self.out = nn.Conv2d(width, 3, 1)
        self.weight = nn.Parameter(torch.full((rungs,), gain))

    def forward(self, z):
        x = self.stem(z).reshape(1, 32, self.base, self.base)
        for i, block in enumerate(self.blocks):
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            x = torch.relu(block(x))
            # A fresh draw per forward, which adoption has to freeze.
            x = x + self.weight[i] * torch.randn_like(x[:, :1])
        return torch.tanh(self.out(x))


def _export(tmp_path, net, nz, name="raw.onnx"):
    path = tmp_path / name
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise), torch.no_grad():
        torch.onnx.export(net.eval(), (torch.zeros(1, nz),), str(path), input_names=["z"],
                          output_names=["image"], opset_version=18, dynamo=True)
    return path


def test_a_graph_that_answers_the_same_latent_differently_is_made_to_stop(tmp_path):
    """A StyleGAN's noise injection survives export as live `RandomNormalLike` nodes, so the
    graph answers one latent differently every time. Every dial measured against that would
    be measuring the noise, and a still picture would not show it."""
    from ganlive.models.calibrate import Probe, deterministic
    from ganlive.models.onnx_adopt import adopt

    torch.manual_seed(0)
    raw = _export(tmp_path, _Noisy(), 16)
    assert deterministic(Probe(raw)) > 1.0, "the fixture is supposed to be non-deterministic"

    found = adopt(raw, tmp_path / "playable.onnx")
    assert found.noise == 3 and not found.was_baked, (
        f"froze {found.noise} draws, expected one per rung")
    assert deterministic(Probe(tmp_path / "playable.onnx")) == 0.0


def test_every_derived_dial_moves_the_picture_through_the_real_graph(tmp_path):
    """A dial derived from the graph's shape is a guess until it is driven."""
    from ganlive.models.calibrate import Probe
    from ganlive.models.onnx_adopt import adopt
    from ganlive.models.onnx_file import dials_of
    from ganlive.pixels import levels

    torch.manual_seed(1)
    out = tmp_path / "playable.onnx"
    found = adopt(_export(tmp_path, _Noisy(), 16), out)
    names = dials_of(out)["settings"]
    assert names == found.names and names, "the names have to travel inside the file"

    probe = Probe(out)
    z, k = probe.latent(3), probe.neutral()
    base = probe.frame(z)
    for slot, dial in enumerate(found.dials):
        driven = k.copy()
        driven[slot] = dial.curve[-1] if dial.rest < 0.5 else dial.curve[0]
        assert levels(probe.frame(z, driven), base) > 1.0, (
            f"{dial.name} exports as decoration")


def test_a_dial_buys_the_same_change_on_any_model_it_is_derived_from(tmp_path):
    """Calibration is the argument for deriving rather than shipping a range: the same gain
    does not buy the same effect on another model."""
    from ganlive.models.calibrate import TARGET_LEVELS, Probe
    from ganlive.models.onnx_adopt import adopt
    from ganlive.pixels import levels

    torch.manual_seed(2)
    out = tmp_path / "playable.onnx"
    found = adopt(_export(tmp_path, _Noisy(gain=0.2), 16), out)
    probe = Probe(out)
    z = probe.latent(0)
    base = probe.frame(z)
    for slot, dial in enumerate(found.dials):
        if dial.moved < TARGET_LEVELS - 0.5:
            continue                       # a dial that cannot get there is reported, not faked
        # Both ends, and the best of them: a two-sided dial can have one half that saturates
        # short, and the claim is about the end that goes furthest.
        got = []
        for end in (dial.curve[0], dial.curve[-1]):
            k = probe.neutral()
            k[slot] = end
            got.append(levels(probe.frame(z, k), base))
        assert max(got) == pytest.approx(TARGET_LEVELS, rel=0.35), (
            f"{dial.name} buys {max(got):.1f} levels at full travel, not {TARGET_LEVELS}")


def test_a_band_is_picture_shaped_and_not_merely_four_dimensional(tmp_path):
    """A style vector is `(1, C, 1, 1)` and a blur kernel is `(1, 1, 3, 3)`: four-dimensional,
    and not resolution bands."""
    from ganlive.models.onnx_adopt import MIN_BAND, adopt

    torch.manual_seed(3)
    found = adopt(_export(tmp_path, _Noisy(), 16), tmp_path / "playable.onnx")
    assert found.bands, "a convolutional generator has a ladder"
    assert all(min(h, w) >= MIN_BAND for h, w in found.bands), found.bands
    assert max(found.bands) == found.size, "the top band is the picture"


def test_one_dial_per_tensor_and_none_at_all_on_the_squash(tmp_path):
    """Two dials on one tensor are two controls that move together. And a gain into the
    output squash is only a contrast curve, so on a fixture whose top band is the squash's
    input that band gets no dial."""
    import onnx

    from ganlive.models.onnx_adopt import _squash, adopt

    torch.manual_seed(4)
    source = _export(tmp_path, _Noisy(), 16)
    found = adopt(source, tmp_path / "playable.onnx")
    tensors = [d.tensor for d in found.dials]
    assert len(tensors) == len(set(tensors)), tensors

    # Read off the graph the dials were derived from, not the rewritten one: the insertions
    # rename what feeds the squash, so the output graph cannot answer this.
    squash = _squash(onnx.load(str(source)).graph)
    assert squash is not None, "this fixture ends in a tanh"
    assert squash not in tensors, (
        f"{squash} feeds the output squash and still carries a dial")


def test_the_strip_reads_an_adopted_graph_without_knowing_the_architecture(tmp_path):
    """End to end: a file this code has never seen becomes a control surface."""
    from ganlive.dials import table as S
    from ganlive.models import onnx_adopt as A
    from ganlive.models.onnx_file import dials_of

    torch.manual_seed(5)
    out = tmp_path / "playable.onnx"
    found = A.adopt(_export(tmp_path, _Noisy(), 16), out)

    said = dials_of(out)
    layout = S.adopted(said["settings"], said["rests"], said["curves"], said["levels"])
    assert [t for t, _names in layout.groups] == ["MASTER", "MOTION", "LATENT", "MODEL"]
    assert dict(layout.groups)["MODEL"] == tuple(found.names)

    written: dict[str, float] = {}

    class _Knobs:
        index = {n: i for i, n in enumerate(found.names)}

        def set(self, name, value):
            written[name] = value

    class _Walk:
        amounts = ()

    surface = S.Surface(layout=layout)
    surface.apply(_Knobs(), _Walk())
    for dial in found.dials:
        assert written[dial.name] == pytest.approx(1.0), (
            f"{dial.name} does not sit at the trained value when nothing is touching it")

    # And driven to each end of its own travel.
    for position, end in ((0.0, 0), (1.0, -1)):
        for dial in found.dials:
            surface.set(dial.name, position)
        surface.apply(_Knobs(), _Walk())
        for dial in found.dials:
            assert written[dial.name] == pytest.approx(dial.curve[end], rel=1e-6), (
                f"{dial.name} at dial {position} is not what adoption measured")


def test_a_generator_is_identified_by_what_it_does_not_by_what_it_is_called():
    """`Generator`, `StyleGAN` and `SynthesisNetwork` are all plausible names for the thing
    that makes a picture, and only one of them is right in any given repository. One forward
    pass separates them and nothing else does."""
    from ganlive.models.foreign import probe

    class Critic(nn.Module):
        def forward(self, x):
            return x.mean()

    class Synthesis(nn.Module):
        """Takes a `w`, not a `z` -- and at the wrong width, returns nothing usable."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 3 * 16 * 16)

        def forward(self, w):
            return self.fc(w).reshape(1, 3, 16, 16)

    assert probe(Critic()) is None, "a critic returns a scalar, not a picture"
    assert probe(Synthesis(), widths=(512, 128)) is None, "no width it takes was offered"
    assert probe(Synthesis(), widths=(512, 64)) == (64, (16, 16))


def test_fetching_someone_elses_code_is_refused_unless_it_was_asked_for():
    """A repository that ships `model.py` is asking to have it imported and run. That is a
    decision for the person at the keyboard, and the flag is not defaulted anywhere."""
    from ganlive.models.foreign import from_hub

    with pytest.raises(PermissionError, match="trust-remote-code"):
        from_hub("someone/whatever")


def test_a_curve_is_the_response_it_was_measured_from(tmp_path):
    """A curve is any number of evenly spaced values, read piecewise-linearly."""
    from ganlive.dials.table import at, evenly

    points = evenly((0.001, 0.01, 0.1, 1.0))
    assert [p for p, _v in points] == [0.0, 1 / 3, 2 / 3, 1.0]
    assert at(points, 0.0) == pytest.approx(0.001)
    assert at(points, 1.0) == pytest.approx(1.0)
    assert at(points, 1 / 3) == pytest.approx(0.01)
    assert at(points, 0.5) == pytest.approx(0.055, rel=1e-3)
    assert at(points, -5.0) == pytest.approx(0.001), "clamped, like every other dial"
    assert at(points, 5.0) == pytest.approx(1.0)
    assert np.isfinite(at(((0.0, 2.0), (0.0, 3.0)), 0.0)), "a zero-width step must not divide"


def test_a_graph_runs_on_whatever_this_machine_has(tmp_path):
    """`auto` opens whichever runtime this machine has, and it computes the same frame."""
    from ganlive.models.common import host_latent
    from ganlive.models.runtime import open_graph, survey
    from ganlive.pixels import worst_levels

    torch.manual_seed(6)
    out = tmp_path / "playable.onnx"
    A_adopt(out, tmp_path)

    reference = open_graph(out, backend="ort", device="CPUExecutionProvider")
    chosen = open_graph(out)
    assert chosen.nz == reference.nz and chosen.size == reference.size
    assert chosen.settings == reference.settings

    z = host_latent(reference.nz)
    k = np.ones(reference.settings, np.float32)
    gap = worst_levels(torch.from_numpy(np.asarray(chosen.infer(z, k), np.float32)),
                       torch.from_numpy(np.asarray(reference.infer(z, k), np.float32)))
    assert gap < 1.0, f"{chosen.backend}/{chosen.device} differs from ONNX Runtime by {gap:.3f} levels"
    assert "onnxruntime" in survey()


def A_adopt(out, tmp_path):
    from ganlive.models.onnx_adopt import adopt

    return adopt(_export(tmp_path, _Noisy(), 16), out, measure=False)


def test_a_provider_that_is_not_there_is_refused_by_name(tmp_path):
    """Named, not guessed: a caller that asked for CUDA wants to know it is not here."""
    from ganlive.models.runtime import Unavailable, open_graph

    torch.manual_seed(7)
    out = tmp_path / "playable.onnx"
    A_adopt(out, tmp_path)
    with pytest.raises(Unavailable, match="CUDAExecutionProvider"):
        open_graph(out, backend="ort", device="CUDAExecutionProvider")


def test_a_provider_that_silently_became_the_cpu_is_an_error(tmp_path, monkeypatch):
    """A provider whose DLL will not load, or that refuses the graph, is only a warning on
    stderr, and the session runs on the CPU under the provider's name."""
    import onnxruntime as ort

    from ganlive.models import runtime as R

    torch.manual_seed(8)
    out = tmp_path / "playable.onnx"
    A_adopt(out, tmp_path)

    class Liar(ort.InferenceSession):
        def __init__(self, path, options, providers=None, provider_options=None):
            super().__init__(path, options, providers=["CPUExecutionProvider"])

        def get_providers(self):                    # what a fallen-back session really says
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", Liar)
    with pytest.raises(R.Unavailable, match="fell back"):
        R.open_graph(out, backend="ort", device="DmlExecutionProvider")


def test_config_of_reads_a_real_graph_through_the_cached_parse(tmp_path):
    from ganlive.models import onnx_file

    path = _export(tmp_path, _Noisy(), 16)
    cfg = onnx_file.config_of(path)

    assert cfg.nz == 16
    assert (cfg.ladder.height, cfg.ladder.width) == (32, 32)

    # The same numbers as the graph's own declared shapes, from the cached read.
    said = onnx_file.dials_of(path)
    assert said["nz"] == cfg.nz
    assert said["size"] == (cfg.ladder.height, cfg.ladder.width)


def test_the_precision_verdict_and_the_dials_are_measured_on_one_runtime(tmp_path, monkeypatch):
    """The precision verdict is measured, and filed, on the runtime the dials are calibrated
    on, and calibration reuses the FP16 runner the verdict came from."""
    import dataclasses

    from ganlive.models import runtime
    from ganlive.models.onnx_adopt import adopt

    real, opened = runtime.open_graph, []

    def recording(path, backend="auto", device="", precision=""):
        opened.append((backend, device, precision))
        # Run on this machine's CPU, reporting itself as what was asked for.
        got = real(path, backend="ort", device="CPUExecutionProvider", precision=precision)
        return dataclasses.replace(got, backend=backend, asked=device)

    monkeypatch.setattr(runtime, "open_graph", recording)
    torch.manual_seed(8)
    found = adopt(_export(tmp_path, _Noisy(), 16), tmp_path / "playable.onnx", device="GPU")
    assert {(b, d) for b, d, _p in opened} == {("openvino", "GPU")}, opened
    assert list(found.precision) == ["openvino/GPU"]
    assert [p for _b, _d, p in opened] == ["FP32", "FP16"], "calibration reuses the FP16 runner"


class _Demodulated(nn.Module):
    """The one reduction in a StyleGAN2 that breaks half precision: a filter's norm, where the
    sum of squares under the root is far past FP16's range while the root is not."""

    def forward(self, z):
        m = z.reshape(4, 64) * 300.0
        m = torch.cat([m, torch.zeros(1, 64)])          # a dead filter must stay finite too
        return m * (m.square().sum(dim=1, keepdim=True) + 1e-2).rsqrt()


def test_a_norm_is_taken_without_holding_its_square(tmp_path):
    """`wide_norms` is exact: the same frame in single precision, and no full-size square."""
    ov = pytest.importorskip("openvino")
    from ganlive.models.common import host_latent
    from ganlive.models.runtime import wide_norms

    path = _export(tmp_path, _Demodulated(), 256)
    core = ov.Core()
    z = host_latent(256)
    frames = []
    for rewrite in (False, True):
        model = core.read_model(str(path))
        if rewrite:
            assert wide_norms(model) == 1
            kinds = {op.get_type_name() for op in model.get_ordered_ops()}
            assert "ReduceL2" in kinds and "ReduceSum" not in kinds, kinds
        request = core.compile_model(model, "CPU").create_infer_request()
        request.infer({0: z})
        frames.append(np.array(request.get_output_tensor(0).data))
    assert np.isfinite(frames[1]).all()
    assert np.abs(frames[1] - frames[0]).max() < 1e-5


def test_a_frame_lands_in_the_buffer_it_was_given(tmp_path):
    """What lets an ONNX frame reach the card as a DMA: the runtime writes into our array."""
    pytest.importorskip("openvino")
    from ganlive.models.common import host_latent
    from ganlive.models.runtime import open_graph

    torch.manual_seed(6)
    out = tmp_path / "playable.onnx"
    A_adopt(out, tmp_path)
    runner = open_graph(out, backend="openvino", device="CPU", precision="FP32")
    host = np.zeros(runner.shapes[0], runner.dtype)
    runner.land(host)
    z = host_latent(runner.nz)
    frame = runner.infer(z, np.ones(runner.settings, np.float32))
    assert np.shares_memory(frame, host) and np.abs(host).max() > 0
