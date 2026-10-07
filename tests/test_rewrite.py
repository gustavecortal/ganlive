"""The export rewrites, and graph capture: each must produce the same picture.

The claim is mathematical identity, not bit identity: splitting one convolution into two
changes the order floating-point results accumulate in. `IDENTICAL` is far under one 8-bit
level, so any real change fails it and reassociation does not.
"""
from __future__ import annotations

import contextlib
import copy
import io
import pathlib
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from ganlive.dials import steer as K
from ganlive.models import capture as speedups
from ganlive.models.capture import capture
from ganlive.models.common import host_latent
from ganlive.models.fastgan import Generator, freeze_noise
from ganlive.models.fold import prepare_for_inference
from ganlive.models.onnx_file import dials_of, name_settings, structure
from ganlive.models.onnx_rewrite import (
    GatedPair,
    equivalent,
    settings_as_input,
    split_gated_convs,
)
from ganlive.pixels import levels
from ganlive.settings import Settings

#: Worst tolerable difference, in 8-bit levels. See the module note.
IDENTICAL = 1e-3


def _folded(nz: int = 32, seed: int = 0, gain: float = 2.0) -> tuple[nn.Module, int]:
    """A folded generator small enough for a test and loud enough to detect a change."""
    torch.manual_seed(seed)
    net = Generator(ngf=8, nz=nz, im_size=256, im_width=384)
    freeze_noise(net, seed=0)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(gain)
    prepared = prepare_for_inference(net, nz, "cpu", half=False)
    return prepared["net"].eval(), nz


def test_it_invents_no_parameters_and_drops_none():
    """Splitting a conv's output channels partitions its weights; it does not copy them."""
    net, _ = _folded()
    before = sum(p.numel() for p in net.parameters())
    split_gated_convs(net)
    assert sum(p.numel() for p in net.parameters()) == before


def test_every_gated_conv_is_rewritten_and_only_the_init_layer_keeps_its_split():
    """One GLU survives on purpose, and it is worth naming which."""
    net, _ = _folded()
    rewritten = split_gated_convs(net)
    assert rewritten > 0 and [m for m in net.modules() if isinstance(m, GatedPair)]

    survivors = [name for name, m in net.named_modules() if isinstance(m, nn.GLU)]
    assert all(name.startswith("init") for name in survivors), survivors


def test_a_gated_pair_is_exactly_conv_then_glu():
    """The unit claim, on one block, with no generator around it."""
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, 1, 1)
    reference = nn.Sequential(copy.deepcopy(conv), nn.GLU(dim=1)).eval()
    pair = GatedPair(conv).eval()

    x = torch.randn(2, 3, 16, 24)
    with torch.no_grad():
        assert torch.allclose(reference(x), pair(x), atol=1e-6)


def test_the_noise_coefficient_is_split_the_same_way_as_the_weights():
    """`FoldedNoise` carries one coefficient per channel, so it partitions with them, and the
    split net draws the same picture as the one it came from."""
    # Two nets from the same seed rather than a deepcopy: `spectral_norm` leaves the weights
    # non-leaf and `deepcopy` refuses them.
    before, _ = _folded()
    net, nz = _folded()
    split_gated_convs(net)
    pairs = [m for m in net.modules() if isinstance(m, GatedPair) and m.noise]
    assert pairs, "no noise-carrying block was rewritten"
    for pair in pairs:
        assert pair.coeff_value.shape == pair.coeff_gate.shape
        assert pair.coeff_value.shape[1] == pair.value.out_channels
    assert equivalent(before, net.eval(), nz, "cpu", probes=2) < IDENTICAL


def test_equivalent_reports_in_8_bit_levels_and_catches_a_real_difference():
    """The gate has to fail when something moves, or it is not a gate."""
    net, nz = _folded()
    other, _ = _folded()
    assert equivalent(net, other, nz, "cpu", probes=1) == 0.0, "same seed, same net"

    # Through `parameters()`, not `to_big.weight`: `to_big` is spectral-normalised, so its `.weight` is
    # computed on access and writing to it modifies a temporary that is thrown away.
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.mul_(1.5)
    assert equivalent(net, other, nz, "cpu", probes=1) > 0.5


@pytest.fixture(scope="module")
def steerable(tmp_path_factory):
    """A small generator exported with its settings as a second graph input, split as an
    export splits it: `(path, module, setting names)`. Exported once for the module; a test
    that edits the file works on a copy."""
    torch.manual_seed(4)
    net = Generator(ngf=16, nz=32, im_size=256, im_width=384).eval()
    for module in net.modules():                       # noise gains initialise at exactly
        if type(module).__name__ == "NoiseInjection":  # zero, which makes any noise dial
            with torch.no_grad():                      # vacuous
                module.weight.fill_(0.4)
    freeze_noise(net, seed=3)
    net = prepare_for_inference(net, 32, "cpu", half=False)["net"].eval()

    # Bring the picture off the rails of the tanh. A randomly built generator saturates its
    # final squash almost everywhere, and a clipped picture cannot be moved by any dial.
    probe = torch.from_numpy(host_latent(32, 0))
    seen: list[float] = []
    handle = net.to_big.register_forward_hook(lambda _m, _i, o: seen.append(float(o.std())))
    with torch.no_grad():
        net(probe)
        handle.remove()
        net.to_big.weight.mul_(0.4 / seen[0])
        if net.to_big.bias is not None:
            net.to_big.bias.mul_(0.4 / seen[0])
        clipped = (net(probe)[0].abs() > 0.99).float().mean().item()
    assert clipped < 0.05, f"{clipped:.2%} of the fixture is clipped; it can measure nothing"

    # A 256-tall ladder has no `se_512` and no `feat_512`; `install` takes what it has.
    settings = K.install(net, "cpu", torch.float32)
    steered = settings_as_input(net, settings).eval()
    split_gated_convs(steered.net)
    steered = steered.eval()

    path = tmp_path_factory.mktemp("steer") / "steer.onnx"
    args = (torch.zeros(1, 32), torch.ones(len(settings.names)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), torch.no_grad():
        torch.onnx.export(steered, args, str(path), input_names=["z", "k"],
                          output_names=["image0", "image1"], opset_version=18, dynamo=True)
    return path, steered, list(settings.names)


def test_every_exported_setting_moves_the_picture_exactly_as_the_module_does(steerable):
    """A view traces as a constant, so the settings become a second graph input.
    What must then be true is not that the nodes are there but that driving them changes the
    output exactly as driving the module does."""
    path, steered, names = steerable
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["z", "k"]

    z = host_latent(32, 0)
    neutral = np.ones(len(names), np.float32)
    zt = torch.from_numpy(z)

    def moved(k):
        graph = sess.run(None, {"z": z, "k": k})[0]
        with torch.no_grad():
            module = steered(zt, torch.from_numpy(k))[0].numpy()
        return graph, module

    base_graph, base_module = moved(neutral)

    live = 0
    for i, name in enumerate(names):
        k = neutral.copy()
        k[i] = 20.0 if name.startswith("noise.") else 1.8
        graph, module = moved(k)
        by_graph = levels(graph, base_graph)
        by_module = levels(module, base_module)
        assert by_graph == pytest.approx(by_module, rel=0.05, abs=0.5), (
            f"{name}: the module moves {by_module:.3f} 8-bit levels and the graph "
            f"{by_graph:.3f} -- the exported dial is not the dial")
        live += by_module >= 0.5
    assert live >= 3, f"only {live} of {len(names)} settings move this fixture at all"


def test_the_settings_names_travel_inside_the_graph(steerable, tmp_path):
    """A second input of shape `(n,)` says how many dials there are and nothing about which
    is which, so the names travel in the graph's own metadata."""
    path = shutil.copy(steerable[0], tmp_path / "steer.onnx")
    names = steerable[2]
    assert dials_of(path)["settings"] == [], "nothing writes them during a bare export"

    model = structure(path)
    name_settings(model, names)
    onnx.save(model, str(path))
    assert dials_of(path)["settings"] == names


def test_the_rewrite_refuses_to_leave_a_setting_behind():
    """It counts settings reached, not modules replaced: one noise setting drives several
    modules, so counting modules would accept a graph with a frozen dial in it."""
    torch.manual_seed(4)
    net = Generator(ngf=16, nz=32, im_size=256, im_width=384).eval()
    freeze_noise(net, seed=3)
    net = prepare_for_inference(net, 32, "cpu", half=False)["net"].eval()
    settings = K.install(net, "cpu", torch.float32)

    # Slots read by position, so a vector the modules do not read would name other dials.
    with pytest.raises(RuntimeError, match="another settings vector"):
        settings_as_input(net, Settings(settings.names, "cpu", torch.float32))

    settings.names.append("sle.se_2048")         # a setting no module can possibly serve
    settings.index["sle.se_2048"] = len(settings.names) - 1
    with pytest.raises(RuntimeError, match="sle.se_2048"):
        settings_as_input(net, settings)


class _Echo(nn.Module):
    """A generator that writes each frame into a buffer it keeps, as a captured graph does.

    It remembers the latent it was handed, so the stub backend below can replay it by
    running it again.
    """

    def __init__(self, nz: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(nz, 3 * 4 * 4)
        self.register_buffer("out", torch.zeros(1, 3, 4, 4))
        self.seen: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.seen = z            # the buffer itself, as a capture bakes in the address
        self.out.copy_(self.lin(z).reshape(1, 3, 4, 4))
        return self.out


def _backend(monkeypatch, replay) -> list[torch.Tensor]:
    """Give `torch.cpu` the two names `capture` looks for, so the gate can be driven here.

    Returns the host buffers `capture` allocates, in order -- the latent's first, then one per
    feed -- so a stub replay can do what a recorded one does: upload, then run."""
    hosts: list[torch.Tensor] = []

    class Graph:
        def replay(self) -> None:
            replay()

    @contextlib.contextmanager
    def graph(_g):
        yield

    def plain(shape, dtype):
        hosts.append(torch.zeros(shape, dtype=dtype))
        return hosts[-1]

    monkeypatch.setattr(torch.cpu, "XPUGraph", Graph, raising=False)
    monkeypatch.setattr(torch.cpu, "graph", graph, raising=False)
    monkeypatch.setattr(speedups, "pinned", plain)
    return hosts


def test_a_capture_that_paints_the_same_frame_whatever_the_latent_is_refused(monkeypatch):
    """A recording that held none of the forward replays fast and paints a still picture,
    and no exception would report it."""
    net = _Echo()
    _backend(monkeypatch, lambda: None)                  # replays nothing at all
    got, said = capture(net, 8, "cpu", torch.float32)

    assert got is net, "a capture that records nothing must not be handed back as one"
    assert "same frame whatever the latent" in said


def test_a_capture_that_does_not_reproduce_the_forward_is_refused(monkeypatch):
    """A replay has to reproduce the forward exactly."""
    net = _Echo()
    _backend(monkeypatch, lambda: net.out.copy_(torch.rand(1, 3, 4, 4) * 2 - 1))
    got, said = capture(net, 8, "cpu", torch.float32)

    assert got is net
    assert "differs from the forward" in said


def _taken(monkeypatch, feeds=()) -> tuple[nn.Module, object]:
    """A generator and the capture of it that the gate accepted. The stub replay does what a
    recorded one does: the uploads from the host buffers, then the forward."""
    net = _Echo()

    def replay():
        net.seen.copy_(hosts[0])
        for dev, host in zip(feeds, hosts[1:], strict=True):
            dev.copy_(host)
        net(net.seen)

    hosts = _backend(monkeypatch, replay)
    got, said = capture(net, 8, "cpu", torch.float32, feeds=feeds)
    assert got is not net and "captured" in said, said
    return net, got


def test_a_capture_that_reproduces_the_forward_is_taken_and_steers(monkeypatch):
    _net, got = _taken(monkeypatch)

    a = got(torch.randn(1, 8)).clone()
    b = got(torch.randn(1, 8)).clone()
    assert not torch.allclose(a, b), "a replay must follow the latent it was handed"
    assert torch.allclose(b, got.frame), "the frame returned is the buffer, not a copy"
    c = got(torch.randn(1, 8).numpy()).clone()
    assert not torch.allclose(b, c), "a host array steers it the same way"


def test_a_feed_is_uploaded_by_the_graph_and_written_on_the_host(monkeypatch):
    """The settings vector and the push buffer are uploaded by the recording from host
    buffers, so the frame path issues nothing between replays."""
    setting = torch.full((3,), 2.0)
    _net, got = _taken(monkeypatch, feeds=[setting])
    host = got.host_buffer(setting)
    assert torch.equal(host, torch.full((3,), 2.0)), "it starts holding what the card holds"
    host.fill_(5.0)
    got(torch.randn(1, 8))
    assert torch.equal(setting, torch.full((3,), 5.0)), "a host write reached the card by replay"
    with pytest.raises(KeyError, match="no upload"):
        got.host_buffer(torch.zeros(3))


def test_a_generator_that_is_not_a_module_is_never_captured():
    """An ONNX graph runs under its own runtime, so a recording of the torch stream would
    hold none of its work. No backend is stubbed: the refusal comes before the device is
    asked anything."""
    got, said = capture(lambda z: z, 8, "cpu", torch.float32)

    assert "not a torch module" in said, said
    assert callable(got)


def test_a_captured_generator_refuses_a_latent_it_was_not_recorded_for(monkeypatch):
    _net, got = _taken(monkeypatch)

    with pytest.raises(ValueError, match="captured for a"):
        got(torch.randn(4, 8))
    assert isinstance(got.lin, nn.Linear), "a stand-in must answer for the generator it wraps"


def test_export_takes_a_run_directory_as_every_other_command_does(tmp_path, monkeypatch):
    """`--checkpoint runs/my-run` means the run's newest checkpoint, as it does for `play`."""
    from ganlive.tools import export_onnx

    run = tmp_path / "my-run" / "checkpoints"
    run.mkdir(parents=True)
    for step in (1000, 2000):
        (run / f"{step:07d}.pt").write_bytes(b"")
    asked = []
    monkeypatch.setattr(export_onnx, "export",
                        lambda checkpoint, out, *_a, **_k: asked.append((checkpoint, out)) or {})
    monkeypatch.setattr(export_onnx, "write_json", lambda *_a: None)
    with contextlib.redirect_stdout(io.StringIO()):
        assert export_onnx.main(["--checkpoint", str(tmp_path / "my-run")]) == 0
    assert asked == [(run / "0002000.pt", pathlib.Path("runs/onnx/my-run-2000.onnx"))], asked
    with contextlib.redirect_stderr(io.StringIO()):
        assert export_onnx.main(["--checkpoint", str(tmp_path / "nothing")]) == 2


def test_an_exported_fastgan_plays_with_its_own_dials(steerable, tmp_path):
    """`export-onnx` names the settings and measures no curves; the FastGAN's own dials know
    what each setting takes, so the graph loads with them rather than with none."""
    from ganlive.families import _onnx_layout

    path = shutil.copy(steerable[0], tmp_path / "steer.onnx")
    model = structure(path)
    name_settings(model, steerable[2])
    onnx.save(model, str(path))
    layout = _onnx_layout(path)
    model_dials = {k.name for k in layout.knobs if k.group == "MODEL"}
    assert {"se_64", "se_128", "se_256", "noise"} <= model_dials
