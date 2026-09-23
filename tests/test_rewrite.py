"""The export rewrites must produce the same picture, or they are worthless.

A rewrite that is *nearly* right is worse than one that is plainly wrong: it survives every
check that looks at a single picture, and the model that ships is then not the model that was
trained.

**The claim is mathematical identity, not bit identity, and the difference is worth stating.**
Splitting one convolution into two changes the order floating-point results accumulate in, so
the outputs can differ in the last bits. Measured: exactly 0.0 on the shipping fp32
checkpoint, and 2.7e-05 of one 8-bit level on the toy fixture below. `IDENTICAL` is set four
orders of magnitude under a level -- tight enough that any real change fails it, loose enough
not to fail on reassociation.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ganlive.models.fastgan import Generator, freeze_noise
from ganlive.models.fold import prepare_for_inference
from ganlive.models.onnx_rewrite import GatedPair, equivalent, split_gated_convs

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
    prepared = prepare_for_inference(net, nz, "cpu", half=False, fold=True)
    return prepared["net"].eval(), nz


def test_the_rewrite_does_not_change_one_pixel():
    # Two nets from the same seed rather than a deepcopy: `spectral_norm` leaves the weights
    # non-leaf and `deepcopy` refuses them.
    before, _ = _folded()
    net, nz = _folded()
    assert split_gated_convs(net) > 0, "nothing was rewritten, so nothing is being tested"
    assert equivalent(before, net.eval(), nz, "cpu", probes=3) < IDENTICAL


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
    """`FoldedNoise` carries one coefficient per channel, so it partitions with them."""
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
    assert equivalent(net, net, nz, "cpu", probes=1) == 0.0

    # Through `parameters()`, not `to_big.weight`: `to_big` is spectral-normalised, so its `.weight` is
    # computed on access and writing to it modifies a temporary that is thrown away.
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.mul_(1.5)
    assert equivalent(net, other, nz, "cpu", probes=1) > 0.5

def _steerable(tmp_path, split=True):
    """A small generator exported with its settings as a second graph input."""
    import contextlib
    import io

    import numpy as np

    from ganlive.dials import steer as K
    from ganlive.models.fastgan import Generator, freeze_noise
    from ganlive.models.fold import prepare_for_inference
    from ganlive.models.onnx_rewrite import bank_the_knobs, split_gated_convs

    torch.manual_seed(4)
    net = Generator(ngf=16, nz=32, im_size=256, im_width=384).eval()
    for module in net.modules():                       # noise gains initialise at exactly
        if type(module).__name__ == "NoiseInjection":  # zero, which makes any noise dial
            with torch.no_grad():                      # vacuous
                module.weight.fill_(0.4)
    freeze_noise(net, seed=3)
    net = prepare_for_inference(net, 32, "cpu", half=False, fold=True)["net"].eval()

    # **Bring the picture off the rails of the tanh.** A randomly built generator drives its final squash to
    # 99.996% saturation, and a clipped picture cannot be moved by anything upstream: `pre_tanh` and two
    # noise rungs measured as *exactly* inert, which is the fixture speaking and not the mechanism.
    probe = torch.from_numpy(
        np.random.default_rng(0).standard_normal((1, 32)).astype("float32"))
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

    # A 256-tall ladder has no `se_512` and no `feat_512`; ask for what it has.
    knobs = K.install(net, "cpu", torch.float32, wanted=K.available(net))
    banked = bank_the_knobs(net, knobs).eval()
    if split:
        split_gated_convs(banked.net)
        banked = banked.eval()

    path = tmp_path / "steer.onnx"
    args = (torch.zeros(1, 32), torch.ones(len(knobs.names)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), torch.no_grad():
        torch.onnx.export(banked, args, str(path), input_names=["z", "k"],
                          output_names=["image0", "image1"], opset_version=18, dynamo=True)
    return path, banked, list(knobs.names)


def test_every_exported_setting_moves_the_picture_exactly_as_the_module_does(tmp_path):
    """**A view traces as a constant**, which is how an exported model came to arrive with six dials the strip
    had to draw dark. Banking them makes each one a slice of the graph's second input, and what must then
    be true is not that the nodes are there but that driving them changes the output. An exported dial that
    does nothing is the inert-knob failure with the evidence thrown away."""
    import numpy as np
    import onnxruntime as ort

    path, banked, names = _steerable(tmp_path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["z", "k"]

    z = np.random.default_rng(0).standard_normal((1, 32)).astype(np.float32)
    neutral = np.ones(len(names), np.float32)
    zt = torch.from_numpy(z)

    def moved(k):
        graph = sess.run(None, {"z": z, "k": k})[0]
        with torch.no_grad():
            module = banked(zt, torch.from_numpy(k))[0].numpy()
        return graph, module

    base_graph, base_module = moved(neutral)

    live = 0
    for i, name in enumerate(names):
        k = neutral.copy()
        k[i] = 20.0 if name.startswith("noise.") else 1.8
        graph, module = moved(k)
        by_graph = float(np.abs(graph - base_graph).mean() * 127.5)
        by_module = float(np.abs(module - base_module).mean() * 127.5)
        assert by_graph == pytest.approx(by_module, rel=0.05, abs=0.5), (
            f"{name}: the module moves {by_module:.3f} 8-bit levels and the graph "
            f"{by_graph:.3f} -- the exported dial is not the dial")
        live += by_module >= 0.5
    assert live >= 3, f"only {live} of {len(names)} settings move this fixture at all"


def test_the_settings_names_travel_inside_the_graph(tmp_path):
    """A second input of shape `(n,)` says how many dials there are and nothing about which
    is which. The argument for this format is that a model nobody here has the code for
    still arrives able to be played, so the names cannot live in a sidecar only."""
    import onnx

    from ganlive.models.onnx import settings_of

    path, _banked, names = _steerable(tmp_path)
    assert settings_of(path) == [], "nothing writes them during a bare export"

    model = onnx.load(str(path), load_external_data=False)
    entry = model.metadata_props.add()
    # The prefix a graph exported under this project's earlier name carries; both are read.
    entry.key, entry.value = "smallgen.settings", ",".join(names)
    onnx.save(model, str(path))
    assert settings_of(path) == names


def test_banking_refuses_to_leave_a_setting_behind(tmp_path):
    """It counts *slots reached*, not modules replaced -- one noise dial drives every
    injection at its rung, so there are more sites than settings and counting modules would
    have said yes to a graph with a frozen dial in it."""
    from ganlive.dials import steer as K
    from ganlive.models.fastgan import Generator, freeze_noise
    from ganlive.models.fold import prepare_for_inference
    from ganlive.models.onnx_rewrite import bank_the_knobs

    torch.manual_seed(4)
    net = Generator(ngf=16, nz=32, im_size=256, im_width=384).eval()
    freeze_noise(net, seed=3)
    net = prepare_for_inference(net, 32, "cpu", half=False, fold=True)["net"].eval()
    knobs = K.install(net, "cpu", torch.float32, wanted=K.available(net))

    banked = bank_the_knobs(net, knobs)
    assert len(banked.names) == len(knobs.names)

    knobs.names.append("sle.se_2048")            # a setting no module can possibly serve
    knobs.index["sle.se_2048"] = len(knobs.names) - 1
    with pytest.raises(RuntimeError, match="sle.se_2048"):
        bank_the_knobs(banked.net, knobs)


class _Echo(nn.Module):
    """A generator that writes each frame into a buffer it keeps, as a captured graph does.

    It also remembers the latent it was handed, so the stub backend below can "replay" by
    running it again -- which is the only part of a device graph a CPU test can model.
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
    import contextlib

    from ganlive.models import capture as speedups

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
    """**The one failure no exception reports.** A recording that held none of the forward
    replays faster than anything and paints a still picture, and every later measurement --
    the dial sweep included -- would go on reading that one frame and calling it healthy."""
    from ganlive.models.capture import capture

    net = _Echo()
    _backend(monkeypatch, lambda: None)                  # replays nothing at all
    got, said = capture(net, 8, "cpu", torch.float32)

    assert got is net, "a capture that records nothing must not be handed back as one"
    assert "same frame whatever the latent" in said


def test_a_capture_that_does_not_reproduce_the_forward_is_refused(monkeypatch):
    """Exact, not approximate -- the rule this module is written to."""
    from ganlive.models.capture import capture

    net = _Echo()
    _backend(monkeypatch, lambda: net.out.copy_(torch.rand(1, 3, 4, 4) * 2 - 1))
    got, said = capture(net, 8, "cpu", torch.float32)

    assert got is net
    assert "differs from the forward" in said


def _taken(monkeypatch, feeds=()) -> tuple[nn.Module, object]:
    """A generator and the capture of it that the gate accepted. The stub replay does what a
    recorded one does: the uploads from the host buffers, then the forward."""
    from ganlive.models.capture import capture

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
    """The settings vector and the push are read by the recording from host twins, so the frame
    path issues nothing between replays -- the invariant `Replay` carries the measurement for."""
    setting = torch.full((3,), 2.0)
    _net, got = _taken(monkeypatch, feeds=[setting])
    twin = got.twin(setting)
    assert torch.equal(twin, torch.full((3,), 2.0)), "the twin starts holding what the card holds"
    twin.fill_(5.0)
    got(torch.randn(1, 8))
    assert torch.equal(setting, torch.full((3,), 5.0)), "a host write reached the card by replay"
    with pytest.raises(KeyError, match="no upload"):
        got.twin(torch.zeros(3))


def test_a_generator_that_is_not_a_module_is_never_captured():
    """An adopted ONNX graph runs under its own runtime, so a recording of the torch stream
    would hold none of its work -- and would replay a still picture with nothing raised. No
    backend is stubbed here: the refusal has to come before anything is asked of the device."""
    from ganlive.models.capture import capture

    got, said = capture(lambda z: z, 8, "cpu", torch.float32)

    assert "not a torch module" in said, said
    assert callable(got)


def test_a_captured_generator_refuses_a_latent_it_was_not_recorded_for(monkeypatch):
    _net, got = _taken(monkeypatch)

    with pytest.raises(ValueError, match="captured for a"):
        got(torch.randn(4, 8))
    assert isinstance(got.lin, nn.Linear), "a stand-in must answer for the generator it wraps"
