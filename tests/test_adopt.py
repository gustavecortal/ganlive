"""Adopting a graph nobody here wrote: dials derived from it, and proved on it.

CPU only, like the rest of the suite. The fixture is a small generator built to have the
three things adoption keys on -- live random draws, a resolution ladder, and a final squash
-- and deliberately not built like this project's own generator, so the tests check that
adoption works on an architecture the code was not written for.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import types

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from ganlive.clock import WalkConfig
from ganlive.curves import at, evenly
from ganlive.dials import table as S
from ganlive.models import onnx_file, runtime
from ganlive.models.calibrate import TARGET_LEVELS, Probe, deterministic
from ganlive.models.common import host_latent
from ganlive.models.foreign import from_hub, probe
from ganlive.models.onnx_adopt import MIN_BAND, _squash, adopt
from ganlive.models.runtime import Unavailable, open_graph, survey, wide_norms
from ganlive.pixels import levels, worst_levels
from tests.support import FakeSettings


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


def _export(folder, net, nz, name="raw.onnx"):
    path = folder / name
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise), torch.no_grad():
        torch.onnx.export(net.eval(), (torch.zeros(1, nz),), str(path), input_names=["z"],
                          output_names=["image"], opset_version=18, dynamo=True)
    return path


@pytest.fixture(scope="module")
def adopted(tmp_path_factory):
    """`_Noisy` exported raw and adopted with its dials measured, once for the module:
    `raw`, `out` and what `adopt` found. No test here writes to either file."""
    folder = tmp_path_factory.mktemp("adopted")
    torch.manual_seed(0)
    raw = _export(folder, _Noisy(), 16)
    out = folder / "playable.onnx"
    return types.SimpleNamespace(raw=raw, out=out, found=adopt(raw, out))


@pytest.fixture(scope="module")
def playable(tmp_path_factory):
    """`_Noisy` adopted without measuring its dials: a playable graph, cheaply."""
    folder = tmp_path_factory.mktemp("playable")
    torch.manual_seed(6)
    out = folder / "playable.onnx"
    adopt(_export(folder, _Noisy(), 16), out, measure=False)
    return out


def test_a_graph_that_answers_the_same_latent_differently_is_made_to_stop(adopted):
    """A StyleGAN's noise injection survives export as live `RandomNormalLike` nodes, so the
    graph answers one latent differently every time, and every dial measured against it would
    be measuring the noise."""
    assert deterministic(Probe(adopted.raw)) > 1.0, "the fixture is supposed to be random"
    assert adopted.found.noise == 3 and not adopted.found.was_baked, (
        f"froze {adopted.found.noise} draws, expected one per rung")
    assert deterministic(Probe(adopted.out)) == 0.0


def test_every_derived_dial_moves_the_picture_through_the_real_graph(adopted):
    """A dial derived from the graph's shape is a guess until it is driven."""
    found = adopted.found
    names = onnx_file.dials_of(adopted.out)["settings"]
    assert names == found.names and names, "the names have to travel inside the file"

    graph = Probe(adopted.out)
    z, k = graph.latent(3), graph.neutral()
    base = graph.frame(z)
    for slot, dial in enumerate(found.dials):
        driven = k.copy()
        driven[slot] = dial.curve[-1] if dial.rest < 0.5 else dial.curve[0]
        assert levels(graph.frame(z, driven), base) > 1.0, (
            f"{dial.name} exports as decoration")


def test_a_dial_buys_the_change_it_was_calibrated_to(adopted):
    """Calibration is the argument for deriving rather than shipping a range: the same gain
    does not buy the same effect on another model, so each dial is scaled to a stated one."""
    graph = Probe(adopted.out)
    z = graph.latent(0)
    base = graph.frame(z)
    for slot, dial in enumerate(adopted.found.dials):
        if dial.moved < TARGET_LEVELS - 0.5:
            continue                       # a dial that cannot get there is reported, not faked
        # Both ends, and the best of them: a two-sided dial can have one half that saturates
        # short, and the claim is about the end that goes furthest.
        got = []
        for end in (dial.curve[0], dial.curve[-1]):
            k = graph.neutral()
            k[slot] = end
            got.append(levels(graph.frame(z, k), base))
        assert max(got) == pytest.approx(TARGET_LEVELS, rel=0.35), (
            f"{dial.name} buys {max(got):.1f} levels at full travel, not {TARGET_LEVELS}")


def test_a_band_is_picture_shaped_and_not_merely_four_dimensional(adopted):
    """A style vector is `(1, C, 1, 1)` and a blur kernel is `(1, 1, 3, 3)`: four-dimensional,
    and not resolution bands."""
    found = adopted.found
    assert found.bands, "a convolutional generator has a ladder"
    assert all(min(h, w) >= MIN_BAND for h, w in found.bands), found.bands
    assert max(found.bands) == found.size, "the top band is the picture"


def test_one_dial_per_tensor_and_none_at_all_on_the_squash(adopted):
    """Two dials on one tensor are two controls that move together. And a gain into the
    output squash is only a contrast curve, so on a fixture whose top band is the squash's
    input that band gets no dial."""
    tensors = [d.tensor for d in adopted.found.dials]
    assert len(tensors) == len(set(tensors)), tensors

    # Read off the graph the dials were derived from, not the rewritten one: the insertions
    # rename what feeds the squash, so the output graph cannot answer this.
    squash = _squash(onnx.load(str(adopted.raw)).graph)
    assert squash is not None, "this fixture ends in a tanh"
    assert squash not in tensors, (
        f"{squash} feeds the output squash and still carries a dial")


def test_the_strip_reads_an_adopted_graph_without_knowing_the_architecture(adopted):
    """End to end: a file this code has never seen becomes a control surface."""
    found = adopted.found
    said = onnx_file.dials_of(adopted.out)
    layout = S.adopted(said["settings"], said["rests"], said["curves"], said["levels"])
    assert [t for t, _names in layout.groups] == ["MASTER", "MOTION", "LATENT", "MODEL"]
    assert dict(layout.groups)["MODEL"] == tuple(found.names)

    settings = FakeSettings(found.names)
    surface = S.Surface(layout=layout)
    surface.apply(settings, WalkConfig())
    for dial in found.dials:
        assert settings.written[dial.name] == pytest.approx(1.0), (
            f"{dial.name} does not sit at the trained value when nothing is touching it")

    # And driven to each end of its own travel.
    for position, end in ((0.0, 0), (1.0, -1)):
        for dial in found.dials:
            surface.set(dial.name, position)
        surface.apply(settings, WalkConfig())
        for dial in found.dials:
            assert settings.written[dial.name] == pytest.approx(dial.curve[end], rel=1e-6), (
                f"{dial.name} at dial {position} is not what adoption measured")


def test_config_of_reads_a_real_graph_through_the_cached_parse(adopted):
    cfg = onnx_file.config_of(adopted.raw)

    assert cfg.nz == 16
    assert (cfg.ladder.height, cfg.ladder.width) == (32, 32)

    # The same numbers as the graph's own declared shapes, from the cached read.
    said = onnx_file.dials_of(adopted.raw)
    assert said["nz"] == cfg.nz
    assert said["size"] == (cfg.ladder.height, cfg.ladder.width)


def test_the_precision_verdict_and_the_dials_are_measured_on_one_runtime(adopted, tmp_path,
                                                                          monkeypatch):
    """The precision verdict is measured, and filed, on the runtime the dials are calibrated
    on, and calibration reuses the FP16 runner the verdict came from."""
    real, opened = runtime.open_graph, []

    def recording(path, backend="auto", device="", precision=""):
        opened.append((backend, device, precision))
        # Run on this machine's CPU, reporting itself as what was asked for.
        got = real(path, backend="ort", device="CPUExecutionProvider", precision=precision)
        return dataclasses.replace(got, backend=backend, asked=device)

    monkeypatch.setattr(runtime, "open_graph", recording)
    found = adopt(adopted.raw, tmp_path / "playable.onnx", device="GPU")
    assert {(b, d) for b, d, _p in opened} == {("openvino", "GPU")}, opened
    assert list(found.precision) == ["openvino/GPU"]
    assert [p for _b, _d, p in opened] == ["FP32", "FP16"], "calibration reuses the FP16 runner"


def test_a_graph_runs_on_whatever_this_machine_has(playable):
    """`auto` opens whichever runtime this machine has, and it computes the same frame."""
    reference = open_graph(playable, backend="ort", device="CPUExecutionProvider")
    chosen = open_graph(playable)
    assert chosen.nz == reference.nz and chosen.size == reference.size
    assert chosen.width == reference.width

    z = host_latent(reference.nz)
    k = np.ones(reference.width, np.float32)
    gap = worst_levels(torch.from_numpy(np.asarray(chosen.infer(z, k), np.float32)),
                       torch.from_numpy(np.asarray(reference.infer(z, k), np.float32)))
    assert gap < 1.0, f"{chosen.backend}/{chosen.device} differs from ONNX Runtime by {gap:.3f} levels"
    assert "onnxruntime" in survey()


def test_a_provider_that_is_not_there_is_refused_by_name(playable):
    """Named, not guessed: a caller that asked for CUDA wants to know it is not here."""
    with pytest.raises(Unavailable, match="CUDAExecutionProvider"):
        open_graph(playable, backend="ort", device="CUDAExecutionProvider")


def test_a_provider_that_silently_became_the_cpu_is_an_error(playable, monkeypatch):
    """A provider whose DLL will not load, or that refuses the graph, is only a warning on
    stderr, and the session runs on the CPU under the provider's name."""
    class Liar(ort.InferenceSession):
        def __init__(self, path, options, providers=None, provider_options=None):
            super().__init__(path, options, providers=["CPUExecutionProvider"])

        def get_providers(self):                    # what a fallen-back session really says
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", Liar)
    with pytest.raises(Unavailable, match="fell back"):
        open_graph(playable, backend="ort", device="DmlExecutionProvider")


def test_a_frame_lands_in_the_buffer_it_was_given(playable):
    """What lets an ONNX frame reach the card as a DMA: the runtime writes into our array."""
    pytest.importorskip("openvino")
    runner = open_graph(playable, backend="openvino", device="CPU", precision="FP32")
    host = np.zeros(runner.shapes[0], runner.dtype)
    runner.land(host)
    z = host_latent(runner.nz)
    frame = runner.infer(z, np.ones(runner.width, np.float32))
    assert np.shares_memory(frame, host) and np.abs(host).max() > 0


def test_a_generator_is_identified_by_what_it_does_not_by_what_it_is_called():
    """`Generator`, `StyleGAN` and `SynthesisNetwork` are all plausible names for the thing
    that makes a picture, and only one of them is right in any given repository. One forward
    pass separates them and nothing else does."""
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
    with pytest.raises(PermissionError, match="trust-remote-code"):
        from_hub("someone/whatever")


def test_a_curve_is_the_response_it_was_measured_from():
    """A curve is any number of evenly spaced values, read piecewise-linearly."""
    points = evenly((0.001, 0.01, 0.1, 1.0))
    assert [p for p, _v in points] == [0.0, 1 / 3, 2 / 3, 1.0]
    assert at(points, 0.0) == pytest.approx(0.001)
    assert at(points, 1.0) == pytest.approx(1.0)
    assert at(points, 1 / 3) == pytest.approx(0.01)
    assert at(points, 0.5) == pytest.approx(0.055, rel=1e-3)
    assert at(points, -5.0) == pytest.approx(0.001), "clamped, like every other dial"
    assert at(points, 5.0) == pytest.approx(1.0)
    assert np.isfinite(at(((0.0, 2.0), (0.0, 3.0)), 0.0)), "a zero-width step must not divide"


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
