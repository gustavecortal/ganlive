"""The StyleGAN2 port, checked without NVIDIA's code or a pickle.

Exactness against the original needs the original, and is checked by
`ganlive import-stylegan2`. What is checked here: the shapes NVIDIA's rules give are the
shapes this builds, a checkpoint loads under its own names, and the network captures in one
graph with no breaks.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from ganlive import bank
from ganlive.dials import steer as K
from ganlive.dials import table
from ganlive.families import LoadOptions, config_of, is_stylegan2
from ganlive.models import stylegan2 as S2
from ganlive.models.common import latent
from ganlive.pixels import levels
from tests.support import fastgan_stub_checkpoint, tiny_stylegan2_file
from tests.support import tiny_stylegan2 as tiny


def test_the_ladder_is_nvidias_ladder():
    """Resolutions, channel counts and `num_ws` follow their formulas, not a table."""
    cfg = S2.Config()                                       # their published FFHQ-1024
    assert cfg.resolutions == (4, 8, 16, 32, 64, 128, 256, 512, 1024)
    assert [cfg.channels(r) for r in cfg.resolutions] == [512] * 5 + [256, 128, 64, 32]
    assert cfg.num_ws == 18
    assert cfg.fp16_from == 128                             # the top four, as they ship it
    assert dataclasses.replace(cfg, num_fp16_res=0).fp16_from > cfg.img_resolution


def test_the_padding_is_computed_not_remembered():
    """`conv2d_resample`'s arithmetic for the one case every StyleGAN2 upsample uses."""
    assert S2._pads(kernel=3, up=2, taps=4, padding=1) == (0, 0, 1, 1, 1, 1)
    # A wider filter pushes the work into the FIR pass rather than the transposed convolution.
    assert S2._pads(kernel=3, up=2, taps=6, padding=1) == (0, 0, 2, 2, 2, 2)


def test_it_renders_the_size_it_was_asked_for():
    net = S2.Generator(tiny())
    out = net(torch.zeros(2, 16))
    assert out.shape == (2, 3, 32, 32)
    assert out.dtype is torch.float32


def _fused(layer, x, w):
    """The modulated convolution as NVIDIA's `fused_modconv=True` writes it, one image at a time:
    the styles folded into a copy of the weight, which is then demodulated."""
    styles = layer.affine(w)
    out = []
    for i in range(x.shape[0]):
        m = layer.weight * styles[i].reshape(1, -1, 1, 1)
        m = m * (m.square().sum(dim=[1, 2, 3], keepdim=True) + 1e-8).rsqrt()
        if layer.up > 1:
            y = F.conv_transpose2d(x[i:i + 1], m.transpose(0, 1), stride=layer.up,
                                   padding=layer.tpad)
            out.append(S2._fir(y, layer.upfir, layer.fpad))
        else:
            out.append(F.conv2d(x[i:i + 1], m, padding=layer.pad))
    return F.leaky_relu(torch.cat(out) + layer.bias.reshape(1, -1, 1, 1), 0.2) * S2.LRELU_GAIN


@pytest.mark.parametrize("up", [1, 2])
@pytest.mark.parametrize("frozen", [False, True])
def test_the_styles_scale_the_activations_and_it_is_the_same_convolution(up, frozen):
    """What plays scales the input channels by the styles and demodulates the output, where the
    original implementation modulates the weight: the same arithmetic in another order, per
    image, so a batch of two with different styles comes out as two separate frames would."""
    torch.manual_seed(0)
    layer = S2.SynthesisLayer(8, 6, w_dim=16, resolution=16 if up > 1 else 8, up=up).eval()
    with torch.no_grad():
        layer.bias.normal_()
        if frozen:
            layer.freeze()
            layer.affine.freeze()
        x, w = torch.randn(2, 8, 8, 8), torch.randn(2, 16)
        got, want = layer(x, w), _fused(layer, x, w)
    assert got.shape == want.shape
    assert (got - want).abs().max() < 1e-5


def test_a_frozen_net_is_the_net_it_was_frozen_from():
    """`load` freezes every derived weight, from a `state_dict` as a converted checkpoint
    arrives; the frame must not know it happened."""
    torch.manual_seed(1)
    cfg = tiny(num_fp16_res=0)
    made = S2.Generator(cfg)
    frozen = S2.load(cfg, made.state_dict())
    assert all(m.played is not None for m in frozen.modules()
               if isinstance(m, S2.SynthesisLayer))
    z = torch.randn(2, cfg.z_dim)
    with torch.no_grad():
        assert torch.equal(made(z), frozen(z))


def test_the_filters_are_not_in_the_file():
    """NVIDIA ships `resample_filter` buffers; they are derived here, and ignored on load."""
    cfg = tiny()
    state = S2.Generator(cfg).state_dict()
    assert not [k for k in state if k.endswith("resample_filter")]
    state["synthesis.b8.conv0.resample_filter"] = torch.zeros(4, 4)
    S2.load(cfg, state)                                     # accepted, because it is derived


def test_the_wrong_architecture_is_refused_by_name():
    state = S2.Generator(tiny()).state_dict()
    with pytest.raises(RuntimeError, match="not the architecture"):
        S2.load(tiny(img_resolution=64), state)


def test_a_half_precision_block_is_decided_when_it_is_built():
    """Which blocks run in half precision is fixed when the network is built."""
    net = S2.Generator(tiny(num_fp16_res=1))
    assert [b.half for b in net.blocks] == [False, False, False, True]
    assert net.blocks[-1].conv1.prenorm > 0 and net.blocks[0].conv1.prenorm == 0.0


def test_it_captures_in_one_graph():
    """The whole reason the file exists. NVIDIA's own model breaks the graph 38 times."""
    net = S2.Generator(tiny())
    explained = torch._dynamo.explain(net)(torch.zeros(1, 16))
    torch._dynamo.reset()
    assert explained.graph_break_count == 0, [str(r.reason) for r in explained.break_reasons]
    assert explained.graph_count == 1


def test_a_neutral_dial_is_the_network_as_trained():
    """Installing steering must not change what the checkpoint does when nothing is touched."""
    torch.manual_seed(0)
    cfg = tiny()
    net = S2.Generator(cfg)
    # NVIDIA initialises `noise_strength` at zero, so an untrained net has no noise to gain.
    for layers in S2.noise_sites(net).values():
        for layer in layers:
            with torch.no_grad():
                layer.noise_strength.fill_(0.3)

    z = torch.randn(1, cfg.z_dim)
    with torch.no_grad():
        before = net(z)
        settings = K.install_stylegan2(net, "cpu")
        settings.reset()
        after = net(z)
    assert torch.equal(before, after)

    settings.set("noise_16", 40.0)
    settings.commit()
    with torch.no_grad():
        moved = levels(net(z), before)
    assert moved > 1.0, f"a noise dial that reaches the model moved {moved:.3f} levels"


def test_one_dial_per_resolution_not_per_layer():
    """Both convolutions in a block are the same grain at the same scale."""
    net = S2.Generator(tiny())
    settings = K.install_stylegan2(net, "cpu")
    assert settings.names == ["w_coarse", "w_mid", "w_fine",
                              "noise_4", "noise_8", "noise_16", "noise_32"]
    assert sum(map(len, S2.noise_sites(net).values())) == 7   # one block has one conv, three two
    assert net.blocks[1].conv0.noise_gain is net.blocks[1].conv1.noise_gain

    # The style dials must be a view into the settings vector, not a copy, or `commit` never
    # reaches the module.
    settings.reset()
    settings.set("w_mid", 0.25)
    settings.commit()
    assert float(net.mapping.truncation[1]) == 0.25


def test_ganlive_opens_one(tmp_path, capsys):
    """The whole load path on the engine: converted beside the checkpoint, its dials installed,
    swept and equalised then, and a surface built from what was measured."""
    wgpu = pytest.importorskip("wgpu")
    from ganlive.engine.runner import default_device

    try:
        gpu = default_device(fallback=True)        # the CPU adapter: the card stays free
    except (RuntimeError, wgpu.GPUError):
        pytest.skip("no WebGPU CPU adapter")
    torch.manual_seed(0)
    cfg = tiny()
    net = S2.Generator(cfg)
    for layers in S2.noise_sites(net).values():
        for layer in layers:
            with torch.no_grad():
                layer.noise_strength.fill_(0.3)
    path = tmp_path / "tiny.pt"
    S2.save(path, cfg, net.state_dict())

    model = bank._prepare(path, gpu, LoadOptions())
    assert model.settings.names[:3] == ["w_coarse", "w_mid", "w_fine"]
    assert model.settings.names[3:] == ["noise_4", "noise_8", "noise_16", "noise_32"]
    block = [k for k in model.layout.knobs if k.group == "MODEL"]
    assert [k.name for k in block] == model.settings.names
    # Every dial the strip shows was driven and measured, which is the gate's whole point.
    assert all(k.measured is not None for k in block)
    assert model.dials_live, "nothing reached the model"
    assert "This is a StyleGAN" not in capsys.readouterr().out    # no per-architecture text


def test_without_a_measurement_it_shows_the_shared_blocks_and_nothing_else():
    """No measurement, no MODEL block -- and specifically not FastGAN's dials. A derived
    dial's curve is its measurement, so before the sweep there is nothing to draw."""
    layout = table.stylegan2()
    assert [k.name for k in layout.knobs if k.group == "MODEL"] == []
    assert {k.name for k in layout.knobs} >= {"reaction", "speed", "dir1"}


def test_the_affine_map_is_the_slice_each_block_is_actually_handed():
    """`affine_sites` states which `w` each affine reads. This drives the forward and checks."""
    cfg = tiny()
    net = S2.Generator(cfg).eval()
    seen: dict[int, int] = {}
    for _idx, tag, affine in S2.affine_sites(net):
        affine.register_forward_hook(
            lambda _m, inp, _out, tag=tag: seen.__setitem__(
                tag, int(inp[0][0].argmax())))

    # One `w` per index, so whichever vector an affine receives names itself.
    ws = torch.eye(cfg.num_ws, cfg.w_dim).unsqueeze(0)
    with torch.no_grad():
        x = img = None
        for i, block in enumerate(net.blocks):
            x, img = block(x, img, ws[:, max(2 * i - 1, 0):2 * i + 2])

    claimed = {tag: idx for idx, tag, _affine in S2.affine_sites(net)}
    assert seen == claimed, (
        f"`affine_sites` says {claimed} and the forward pass used {seen}; the style ranges "
        f"are cut with this map")
    # The overlap, spelled out, because it is the part that reads like an off-by-one.
    assert claimed["b4.torgb"] == claimed["b8.conv0"]


def test_style_bands_cut_the_affines_where_the_truncation_dials_do():
    """One basis per `BANDS` range means the same ranges the three `w` dials already use."""
    net = S2.Generator(tiny()).eval()
    bands = S2.style_bands(net)
    assert [name for name, _w in bands] == [name for name, _lo, _hi in S2.BANDS]
    assert all(w.shape[1] == net.cfg.w_dim for _n, w in bands)

    sites = S2.affine_sites(net)
    for (name, weight), (_n, lo, hi) in zip(bands, S2.BANDS, strict=True):
        want = sum(a.weight.shape[0] for idx, _tag, a in sites if lo <= idx < hi)
        assert weight.shape[0] == want, f"{name} stacked the wrong affines"
    assert sum(w.shape[0] for _n, w in bands) >= sum(a.weight.shape[0] for *_x, a in sites), (
        "every affine belongs to some range; one falling through would be a silent hole")


def test_a_range_with_no_affines_is_reported_empty_rather_than_dropped():
    """`w_fine` on a model of eight or fewer `w` vectors. It exists and it is empty."""
    net = S2.Generator(tiny(img_resolution=16)).eval()     # 6 `w` vectors, so nothing above 8
    bands = dict(S2.style_bands(net))
    assert net.cfg.num_ws == 6
    assert bands["w_fine"].shape == (0, net.cfg.w_dim)
    assert bands["w_coarse"].shape[0] and bands["w_mid"].shape[0]


def test_the_style_push_reaches_the_picture_and_the_latent_push_cannot():
    """Why a StyleGAN2's directions push `w`: the mapping's pixel norm cancels a `z` scaling."""
    net = S2.Generator(tiny()).eval().requires_grad_(False)
    z = latent(net.cfg.z_dim, 0, "cpu", torch.float32)
    with torch.no_grad():
        base = net(z)
        scaled = net(z * 3.0)
        net.mapping.push = torch.zeros(len(S2.BANDS), net.cfg.w_dim)
        still = net(z)
        net.mapping.push[1] = 2.0                            # the middle style range only
        pushed = net(z)

    assert torch.allclose(base, scaled, atol=1e-4), "the pixel norm should cancel a z scaling"
    assert torch.equal(base, still), "a zero push must be the network as trained, exactly"
    assert (pushed - base).abs().mean() > 1e-3, "a w push must change the picture"


def test_the_push_lands_after_the_truncation_and_not_before_it():
    """A direction dial must not get weaker because a different dial was turned down."""
    # 64px, so all three ranges are non-empty: `tiny()` alone has an empty `w_fine`.
    net = S2.Generator(tiny(img_resolution=64)).eval().requires_grad_(False)
    mapping = net.mapping
    mapping.w_avg.normal_()
    z = latent(net.cfg.z_dim, 1, "cpu", torch.float32)
    push = torch.randn(len(S2.BANDS), net.cfg.w_dim)
    wanted = mapping.bands @ push                            # `(num_ws, w_dim)`

    with torch.no_grad():
        for trunc in (1.0, 0.2, 0.0):
            mapping.truncation = torch.full((len(S2.BANDS),), trunc)
            mapping.push = torch.zeros_like(push)
            plain = mapping(z)
            mapping.push = push.clone()
            moved = mapping(z)
            assert torch.allclose(moved - plain, wanted, atol=1e-5), (
                f"at truncation {trunc} the push arrived scaled by the truncation")


def test_a_converted_file_is_a_generator_and_is_told_apart_from_one_of_ours(tmp_path):
    """Two kinds of checkpoint share the `.pt` suffix, and ganlive opens both. A
    converted one holds a generator and nothing else."""
    path = tmp_path / "converted.pt"
    cfg = tiny_stylegan2_file(path)
    assert set(torch.load(path, weights_only=True)) == {"format", "config", "state"}
    assert S2.from_file(path).cfg == cfg
    assert is_stylegan2(path)
    assert config_of(path).nz == cfg.z_dim
    assert config_of(path).ladder.height == cfg.img_resolution

    # Anything else is not one, including a file that is not a checkpoint at all.
    other = fastgan_stub_checkpoint(tmp_path / "ours.pt", im_size=256)
    assert not is_stylegan2(other)
    assert not is_stylegan2(tmp_path / "missing.pt")


def test_a_checkpoint_that_carries_more_than_a_generator_still_opens(tmp_path):
    """A trainer writes a discriminator beside it. That is its business; this half reads past it."""
    path = tmp_path / "pair.pt"
    cfg = tiny_stylegan2_file(path)
    blob = torch.load(path, weights_only=True)
    blob["d_config"] = {"img_resolution": 32}
    blob["d_state"] = {"b4.out.bias": torch.zeros(1)}
    torch.save(blob, path)

    assert S2.is_stylegan2(path), "an extra key is not a different format"
    assert S2.from_file(path).cfg == cfg
