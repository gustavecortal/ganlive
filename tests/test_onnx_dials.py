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
    """A tiny generator with a live noise draw at every rung, as a StyleGAN exports one."""

    def __init__(self, nz: int = 16, base: int = 4, rungs: int = 3, gain: float = 0.6) -> None:
        super().__init__()
        self.nz, self.base, self.gain = nz, base, rungs
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
            # A fresh draw per forward: the thing that makes a graph answer the same
            # question differently, and the whole reason adoption freezes anything.
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
    """**The failure that is invisible in a still picture.** A StyleGAN's noise injection
    survives export as live `RandomNormalLike` nodes, so the graph boils on its own: the
    first foreign model tried here answered one latent 165 8-bit levels apart. Every dial
    measured against that would have been measuring the boiling, and the picture looks
    entirely like a working GAN the whole time.
    """
    from ganlive.dials.onnx_dials import Probe, adopt, deterministic

    torch.manual_seed(0)
    raw = _export(tmp_path, _Noisy(), 16)
    assert deterministic(Probe(raw)) > 1.0, "the fixture is supposed to be non-deterministic"

    found = adopt(raw, tmp_path / "playable.onnx")
    assert found.noise == 3 and not found.was_baked, (
        f"froze {found.noise} draws, expected one per rung")
    assert deterministic(Probe(tmp_path / "playable.onnx")) == 0.0


def test_every_derived_dial_moves_the_picture_through_the_real_graph(tmp_path):
    """The inert-knob failure, in the place it would now come from. A dial derived by shape
    is a guess until it is driven, and a guess on the strip is exactly what this project has
    paid for three times."""
    from ganlive.dials.onnx_dials import Probe, adopt, levels
    from ganlive.models.onnx import settings_of

    torch.manual_seed(1)
    out = tmp_path / "playable.onnx"
    found = adopt(_export(tmp_path, _Noisy(), 16), out)
    names = settings_of(out)
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
    """Calibration is the whole argument for deriving rather than shipping a range: the same
    gain does not buy the same effect on another model, and four grain dials shipped with a
    fixed range once felt like four different instruments."""
    from ganlive.dials.onnx_dials import TARGET_LEVELS, Probe, adopt, levels

    torch.manual_seed(2)
    out = tmp_path / "playable.onnx"
    found = adopt(_export(tmp_path, _Noisy(gain=0.2), 16), out)
    probe = Probe(out)
    z = probe.latent(0)
    base = probe.frame(z)
    for slot, dial in enumerate(found.dials):
        if dial.moved < TARGET_LEVELS - 0.5:
            continue                       # a dial that cannot get there is reported, not faked
        # Both ends, and the best of them. A two-sided dial can have one half that saturates
        # short -- a band gain under this fixture's tanh reaches 25 downward and 9.8 upward --
        # and the claim being made is about full travel, which is the end that goes furthest.
        got = []
        for end in (dial.curve[0], dial.curve[-1]):
            k = probe.neutral()
            k[slot] = end
            got.append(levels(probe.frame(z, k), base))
        assert max(got) == pytest.approx(TARGET_LEVELS, rel=0.35), (
            f"{dial.name} buys {max(got):.1f} levels at full travel, not {TARGET_LEVELS}")


def test_a_band_is_picture_shaped_and_not_merely_four_dimensional(tmp_path):
    """**A style vector is `(1, C, 1, 1)` and a blur kernel is `(1, 1, 3, 3)`.** Both passed
    a bare four-dimensional test on the first foreign model tried here and arrived on the
    strip as `gain_1` and `gain_3`, which are not resolutions and are not what those tensors
    are."""
    from ganlive.dials.onnx_dials import MIN_BAND, adopt

    torch.manual_seed(3)
    found = adopt(_export(tmp_path, _Noisy(), 16), tmp_path / "playable.onnx")
    assert found.bands, "a convolutional generator has a ladder"
    assert all(min(h, w) >= MIN_BAND for h, w in found.bands), found.bands
    assert max(found.bands) == found.size, "the top band is the picture"


def test_one_dial_per_tensor_and_none_at_all_on_the_squash(tmp_path):
    """Two dials on one tensor is two controls that move together, and nothing on the strip
    would say so. It happened on the first model tried: `gain_128` and `pre_tanh` were the
    same `Multiply`. `pre_tanh` is gone now -- a gain into the output squash is a contrast
    curve on the finished picture -- so on a fixture whose top band *is* the squash's input the
    answer is one fewer dial, not the same dial under the band's name."""
    import onnx

    from ganlive.dials.onnx_dials import _squash, adopt

    torch.manual_seed(4)
    source = _export(tmp_path, _Noisy(), 16)
    found = adopt(source, tmp_path / "playable.onnx")
    tensors = [d.tensor for d in found.dials]
    assert len(tensors) == len(set(tensors)), tensors
    assert "pre_tanh" not in found.names, "the squash dial was deleted, not renamed"

    # Read off the graph the dials were derived from, not the rewritten one: the insertions
    # rename what feeds the squash, so the output graph cannot answer this.
    squash = _squash(onnx.load(str(source)).graph)
    assert squash is not None, "this fixture ends in a tanh"
    assert squash not in tensors, (
        f"{squash} feeds the output squash and still carries a dial")


def test_the_strip_reads_an_adopted_graph_without_knowing_the_architecture(tmp_path):
    """The payoff, end to end: a file this code has never seen becomes a control surface."""
    from ganlive.dials import onnx_dials as A
    from ganlive.dials import table as S
    from ganlive.models.onnx import dials_of

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
        noise_gains = None

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
    """Three points was not enough. Where a gain feeds an instance norm the norm divides it
    straight back out, so nothing happens until the gain is near zero -- three of five
    rendered frames were visibly the same picture."""
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
    """The ONNX path was OpenVINO and nothing else, which reaches Intel GPUs and no others."""
    from ganlive.models.runtime import open_graph, survey

    torch.manual_seed(6)
    out = tmp_path / "playable.onnx"
    A_adopt(out, tmp_path)

    reference = open_graph(out, backend="ort", device="CPUExecutionProvider")
    chosen = open_graph(out)
    assert chosen.nz == reference.nz and chosen.size == reference.size
    assert chosen.settings == reference.settings

    z = np.random.default_rng(0).standard_normal((1, reference.nz)).astype(np.float32)
    k = np.ones(reference.settings, np.float32)
    gap = float(np.abs(np.asarray(chosen.infer(z, k), np.float32)
                       - np.asarray(reference.infer(z, k), np.float32)).max() * 127.5)
    assert gap < 1.0, f"{chosen.backend}/{chosen.device} is not the reference: {gap:.3f} levels"
    assert "onnxruntime" in survey()


def A_adopt(out, tmp_path):
    from ganlive.dials.onnx_dials import adopt

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
    """**The failure this project has paid for twice.** A provider whose DLL will not load,
    or that refuses the graph, is a warning on stderr -- not an exception -- and the session
    runs on the CPU looking healthy. `onnxruntime-openvino` did it at 650 ms and read as an
    ONNX verdict; DirectML does it to this project's own 1536x1024 exports, turning 3.27 ms
    into 146 without a word. A slow backend is a disappointment. A slow backend wearing a
    fast one's name is a wrong measurement.
    """
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


def test_onnx_config_reports_a_foreign_size_instead_of_crashing():
    """`onnx_model` reads sizes through `models.Ladder`, not a second copy of it."""
    from ganlive.models import fastgan as models
    from ganlive.models import onnx as onnx_model

    assert onnx_model.Ladder is models.Ladder

    # A size no generator here builds, which is the whole point of the backend.
    ladder = onnx_model.Ladder(width=1536, height=1024)
    assert (ladder.height, ladder.width) == (1024, 1536)

    cfg = onnx_model.OnnxConfig(nz=256, ladder=ladder)
    assert (cfg.ladder.width, cfg.ladder.height) == (1536, 1024)


def test_config_of_reads_a_real_graph_through_the_cached_parse(tmp_path):
    """The call that used to raise, against an actual file."""
    from ganlive.models import onnx as onnx_model

    path = _export(tmp_path, _Noisy(), 16)
    cfg = onnx_model.config_of(path)

    assert cfg.nz == 16
    assert (cfg.ladder.height, cfg.ladder.width) == (32, 32)

    # Same numbers as the graph's own declared shapes, and from the one cached read that
    # `dials_of` already performs rather than a second parse.
    said = onnx_model.dials_of(path)
    assert said["nz"] == cfg.nz
    assert said["size"] == (cfg.ladder.height, cfg.ladder.width)
