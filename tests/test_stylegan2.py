"""The clean StyleGAN2, checked without NVIDIA's code and without a 382 MB pickle.

What cannot be checked here is exactness, because that needs the original to compare against.
It is checked against `ffhq.pkl` and `afhqcat.pkl` on the card:
0.000 8-bit levels in fp32, block by block, and bit-for-bit in fp16. What *can* be checked
here is everything that made those two runs possible -- that the shapes NVIDIA's rules give
are the shapes this builds, that a checkpoint loads under its own names, and that the whole
network still captures in one graph with no breaks, which is the entire point of the file.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from ganlive.models import stylegan2 as S2


def tiny(**over) -> S2.Config:
    """Small enough to run in a pre-commit loop, and the same shape as the real thing."""
    return dataclasses.replace(
        S2.Config(z_dim=16, w_dim=16, img_resolution=32, channel_base=128, channel_max=32,
                  num_layers=2, num_fp16_res=0), **over)


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


def test_weights_load_under_their_own_names():
    """A round trip through a `state_dict`, which is how a converted checkpoint arrives."""
    torch.manual_seed(0)
    cfg = tiny()
    made = S2.Generator(cfg)
    z = torch.randn(1, cfg.z_dim)
    with torch.no_grad():
        want = made(z)
        got = S2.load(cfg, made.state_dict())(z)
    assert torch.equal(want, got)


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
    """Which blocks run half is a constant of the checkpoint, which is why there is no branch."""
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
    from ganlive.dials import steer as K

    torch.manual_seed(0)
    cfg = tiny()
    net = S2.Generator(cfg)
    # NVIDIA initialises `noise_strength` at zero, so an untrained net has no noise to gain.
    for layer in (layer for block in net.blocks
                  for layer in (block.conv0, block.conv1) if layer is not None):
        with torch.no_grad():
            layer.noise_strength.fill_(0.3)

    z = torch.randn(1, cfg.z_dim)
    with torch.no_grad():
        before = net(z)
        knobs = K.install_stylegan2(net, "cpu")
        knobs.reset()
        after = net(z)
    assert torch.equal(before, after)

    knobs.set("noise_16", 40.0)
    knobs.commit()
    with torch.no_grad():
        moved = float((net(z) - before).abs().mean() * 127.5)
    assert moved > 1.0, f"a noise dial that reaches the model moved {moved:.3f} levels"


def test_one_dial_per_resolution_not_per_layer():
    """Both convolutions in a block are the same grain at the same scale."""
    from ganlive.dials import steer as K

    net = S2.Generator(tiny())
    knobs = K.install_stylegan2(net, "cpu")
    assert knobs.names == ["w_coarse", "w_mid", "w_fine",
                           "noise_4", "noise_8", "noise_16", "noise_32"]
    assert knobs.sites == {"noise": 7, "style": 3}   # one block has one convolution, three two
    assert net.blocks[1].conv0.knob is net.blocks[1].conv1.knob

    # **The style dials must be a view, not a copy.** `torch.cat` of the three would allocate,
    # and the module would hold something `commit` never reaches -- three bright dials moving
    # nothing, which is the failure this project keeps paying for. It happened here once.
    knobs.reset()
    knobs.set("w_mid", 0.25)
    knobs.commit()
    assert float(net.mapping.knob[1]) == 0.25


def test_a_converted_file_is_told_apart_from_one_of_ours(tmp_path):
    """Two kinds of checkpoint share the `.pt` suffix, and the instrument opens both."""
    from ganlive import bank

    cfg = tiny()
    path = tmp_path / "converted.pt"
    S2.save(path, cfg, S2.Generator(cfg).state_dict())
    assert bank.is_stylegan2(path)
    assert bank.config_of(path).nz == cfg.z_dim
    assert bank.config_of(path).ladder.height == cfg.img_resolution

    # Anything else is not one, including a file that is not a checkpoint at all.
    other = tmp_path / "ours.pt"
    torch.save({"config": {"im_size": 256}, "g_ema": {}}, other)
    assert not bank.is_stylegan2(other)
    assert not bank.is_stylegan2(tmp_path / "missing.pt")


def test_the_instrument_opens_one(tmp_path, capsys):
    """The whole load path: install, measure, equalise, and a surface built from the result."""
    from ganlive import bank

    torch.manual_seed(0)
    cfg = tiny()
    net = S2.Generator(cfg)
    for block in net.blocks:
        for layer in (block.conv0, block.conv1):
            if layer is not None:
                with torch.no_grad():
                    layer.noise_strength.fill_(0.3)
    path = tmp_path / "tiny.pt"
    S2.save(path, cfg, net.state_dict())

    model, _ = bank._prepare(path, "cpu", torch.float32, (None, None, None),
                            bank.LoadOptions(compile_net=False))
    assert model.knobs.names[:3] == ["w_coarse", "w_mid", "w_fine"]
    assert model.knobs.names[3:] == ["noise_4", "noise_8", "noise_16", "noise_32"]
    block = [k for k in model.layout.knobs if k.group == "MODEL"]
    assert [k.name for k in block] == model.knobs.names
    # Every dial the strip shows was driven and measured, which is the gate's whole point.
    assert all(k.measured is not None for k in block)
    assert model.dials_live, "nothing reached the model"
    assert "This is a StyleGAN" not in capsys.readouterr().out    # no per-architecture text

    # **`--stock-grain` is a question about grain strength, not about which dials exist.** It
    # used to skip this family's sweep too -- under the name `--no-calibrate`, which is what
    # made that read plausible -- and since a derived dial's curve *is* its measurement, the
    # strip came up with a spine and an empty MODEL block while all twelve dials sat live on
    # the model. Same surface either way; only `noise_gains` is at stake.
    bare, _ = bank._prepare(path, "cpu", torch.float32, (None, None, None),
                           bank.LoadOptions(compile_net=False, measure_grain=False))
    assert [k.name for k in bare.layout.knobs if k.group == "MODEL"] == model.knobs.names
    assert all(k.measured is not None
               for k in bare.layout.knobs if k.group == "MODEL")
    assert bare.dials_live == model.dials_live


def test_without_a_measurement_it_shows_the_spine_and_nothing_else(tmp_path):
    """No measurement, no MODEL block -- and specifically not this project's own dials.

    `--stock-grain` no longer reaches this: it asks for stock grain gains, and the sweep that
    builds a StyleGAN2's dials runs either way, or twelve live dials sit on the model with
    nothing on the strip to reach them. What lands here is a `Model` built without a sweep at
    all, where a spine is the only honest answer -- a derived dial's curve *is* its measurement,
    so there is nothing to draw."""
    from ganlive import bank
    from ganlive.dials import steer as K

    cfg = tiny()
    path = tmp_path / "tiny.pt"
    S2.save(path, cfg, S2.Generator(cfg).state_dict())
    knobs = K.install_stylegan2(S2.from_file(path), "cpu")

    layout = bank.layout_for(knobs, path)
    assert [k.name for k in layout.knobs if k.group == "MODEL"] == []
    assert {k.name for k in layout.knobs} >= {"reaction", "speed", "dir1"}
    # and without the dispatch it is the hand-tuned surface, whose MODEL block names five
    # gates and a grain band that live in this project's architecture and in no other.
    fallback = bank.layout_for(knobs, tmp_path / "ours.pt")
    assert [k.name for k in fallback.knobs if k.group == "MODEL"] == ["se_256", "se_512",
                                                                     "se_128", "se_64",
                                                                     "noise"]


def test_the_affine_map_is_the_slice_each_block_is_actually_handed():
    """`affine_sites` claims a structural map. This drives the forward and checks it."""
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
        f"are cut with this map, so a wrong entry silently factorises the wrong weights")
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
    """The seam, and the reason it has to be a seam at all."""
    net = S2.Generator(tiny()).eval().requires_grad_(False)
    z = torch.randn(1, net.z_dim, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        base = net(z)
        scaled = net(z * 3.0)
        net.mapping.push = torch.zeros(len(S2.BANDS), net.cfg.w_dim)
        still = net(z)
        net.mapping.push[1] = 2.0                            # the middle style range only
        pushed = net(z)

    assert torch.allclose(base, scaled, atol=1e-4), "the pixel norm should cancel a z scaling"
    assert torch.equal(base, still), "a zero push must be the network as trained, exactly"
    assert (pushed - base).abs().mean() > 1e-3, "a w push that changes nothing is not a seam"


def test_the_push_lands_after_the_truncation_and_not_before_it():
    """A direction dial must not get weaker because a different dial was turned down."""
    # 64px, so all three ranges are non-empty: `tiny()` alone has eight `w` vectors and an
    # empty `w_fine`, which would have made this pass by pushing nothing.
    net = S2.Generator(tiny(img_resolution=64)).eval().requires_grad_(False)
    mapping = net.mapping
    mapping.w_avg.normal_()
    z = torch.randn(1, net.z_dim, generator=torch.Generator().manual_seed(1))
    push = torch.randn(len(S2.BANDS), net.cfg.w_dim)
    wanted = mapping.bands @ push                            # `(num_ws, w_dim)`

    with torch.no_grad():
        for trunc in (1.0, 0.2, 0.0):
            mapping.knob = torch.full((len(S2.BANDS),), trunc)
            mapping.push = torch.zeros_like(push)
            plain = mapping(z)
            mapping.push = push.clone()
            moved = mapping(z)
            assert torch.allclose(moved - plain, wanted, atol=1e-5), (
                f"at truncation {trunc} the push arrived scaled; before the lerp it would be "
                f"multiplied by the truncation, and a dial whose strength depends on another "
                f"dial gets blamed on the model")

def test_a_saved_checkpoint_is_a_generator_and_nothing_else(tmp_path):
    """This project plays models, so a generator is the whole of what it writes."""
    cfg = tiny()
    path = tmp_path / "g.pt"
    S2.save(path, cfg, S2.Generator(cfg).state_dict())

    blob = torch.load(path, weights_only=True)
    assert set(blob) == {"format", "config", "state"}
    assert S2.is_stylegan2(path) and S2.from_file(path).cfg == cfg


def test_a_checkpoint_that_carries_more_than_a_generator_still_opens(tmp_path):
    """A trainer writes a discriminator beside it. That is its business; this half reads past it."""
    cfg = tiny()
    path = tmp_path / "pair.pt"
    S2.save(path, cfg, S2.Generator(cfg).state_dict())
    blob = torch.load(path, weights_only=True)
    blob["d_config"] = {"img_resolution": 32}
    blob["d_state"] = {"b4.out.bias": torch.zeros(1)}
    torch.save(blob, path)

    assert S2.is_stylegan2(path), "an extra key is not a different format"
    assert S2.from_file(path).cfg == cfg


def test_this_repository_has_no_discriminator_at_all():
    """It was 219 lines of training network in a package whose headline is playback.

    The two repositories do not rely on each other: the training half has its own converter
    and its own discriminator, and needs nothing written here."""
    for gone in ("Discriminator", "DConfig", "load_d", "d_config_from",
                 "discriminator_from_file", "DBlock", "Epilogue"):
        assert not hasattr(S2, gone), f"{gone} is training code"
