"""Which dials exist, where they come from, and that each one moves the picture."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from ganlive.dials.fastgan_dials import (
    DIALS,
    MODEL,
    NOISE_BANDS,
    NOISE_FALLBACK_GAIN,
    NOISE_RAMP,
    fastgan,
    noise_for,
)
from ganlive.dials.table import (
    LATENT,
    MASTER,
    MOTION,
    Surface,
    clamp01,
)
from ganlive.presets import Preset
from ganlive.walk import (
    SlerpWalk,
    WalkConfig,
)
from tests.support import (
    FakeKnobs,
    _applied,
    _panel,
    _step,
    _StubModel,
)


def _stub_net(gates=("se_64", "se_128", "se_256", "se_512"),
              rungs=("feat_8", "feat_32", "feat_128", "feat_512")):
    """A generator-shaped object with only what `install` looks at."""
    from torch import nn

    from ganlive.models.fastgan import SkipLayerExcitation
    from ganlive.models.fold import FoldedNoise

    net = nn.Module()
    for name in rungs:
        coeff = torch.ones(1, 2, 1, 1)
        net.add_module(name, nn.Sequential(FoldedNoise(coeff, torch.zeros(1, 1, 4, 4))))
    for name in gates:
        net.add_module(name, SkipLayerExcitation(2, 2))
    net.to_big = nn.Identity()
    net.init = nn.Identity()
    return net


def _ramped_to(gains):
    """`noise_for(1.0)` if every band carried `gains`. One definition, two callers."""
    return {name: 1.0 + (gains[name] - 1.0) * clamp01((1.0 - start) / NOISE_RAMP)
            for name, levels, start in NOISE_BANDS}


def test_the_documented_spread_table_matches_what_the_walk_does():
    """The table in `WalkConfig.spread` is load-bearing -- it is how a value gets chosen -- so
    it is checked rather than trusted. Tolerances are wide because these are means over a
    random sequence, not identities."""
    for spread, expected in ((0.05, 1.8), (0.10, 3.5), (0.20, 7.0), (0.35, 11.8),
                             (0.50, 16.0), (0.70, 20.2), (1.00, 22.5)):
        got = _step(SlerpWalk(256, "cpu", WalkConfig(spread=spread)), n=60)
        assert got == pytest.approx(expected, abs=0.6), (spread, got, expected)


def test_nothing_can_be_driven_off_its_scale():
    """Clamping at the dial is what replaced twenty-two chances to write a value the network
    has never seen. A shove far past the end has to stop at the end."""
    surface = Surface(layout=fastgan())
    surface.add("noise", 40.0)
    surface.add("se_256", -12.0)
    assert surface["noise"] == 1.0
    assert surface["se_256"] == 0.0


def test_the_groups_cover_every_dial_exactly_once():
    """The groups are what a hardware surface gets laid out from, so they have to partition."""
    assert set(MASTER) | set(MOTION) | set(LATENT) | set(MODEL) == set(DIALS)
    assert (len(MASTER) + len(MOTION) + len(LATENT) + len(MODEL)) == len(DIALS)


def test_a_dial_reads_out_in_its_own_units():
    """A slider labelled 0.25 says nothing; one labelled `4 beats` says everything. The
    readout comes from the same tables `apply` reads, so the two cannot disagree about which
    detent a position lands on."""
    from ganlive.dials.table import readout

    assert readout("speed", 0.0) == "8 beats"
    assert readout("speed", 0.25) == "4 beats"
    assert readout("speed", 0.75) == "1 beat"             # not "1 beats"
    assert readout("grid", 0.0) == "glide"
    assert readout("grid", 1.0) == "32 steps"
    assert readout("hold", 0.8) == "80% still"
    assert "breathing" in readout("spread", 0.05)
    assert "new scene" in readout("spread", 1.0)
    gates = fastgan()
    assert readout("se_256", 0.5, gates) == "as trained"
    assert readout("se_256", 1.0, gates) == "up 100%"
    assert readout("se_256", 0.25, gates) == "down 50%"
    assert readout("late", 0.5) == "spread evenly"
    assert readout("late", 0.9).startswith("arrives")
    assert readout("late", 0.1).startswith("leaves")

    for name in DIALS:
        assert readout(name, 0.37)


def test_the_readout_agrees_with_what_the_dial_actually_writes():
    """The two detented dials are the ones a readout can lie about, because the number on the
    dial and the value written are different things."""
    from ganlive.dials.table import readout

    for position in (0.0, 0.2, 0.25, 0.5, 0.7, 1.0):
        _written, cfg = _applied(speed=position, grid=position)
        assert readout("speed", position).startswith(f"{cfg.beats_per_segment:g}")
        assert readout("grid", position) == ("glide" if not cfg.step_grid
                                             else f"{cfg.step_grid} steps")


def test_the_order_between_the_four_kinds_of_writer_is_stated_on_the_surface():
    """Resting value, then a hand (which holds), then anything that `set`s (which a hand beats), then anything
    that `add`s (which lands on top)."""
    surface = Surface(layout=fastgan())
    surface.set_held({"noise": 0.4})
    surface.set("noise", 0.9)
    assert surface["noise"] == pytest.approx(0.4), "a set must not move a held dial"
    surface.set("se_256", 0.9)
    assert surface["se_256"] == pytest.approx(0.9), "and must move an unheld one"
    surface.add("noise", 0.2)
    assert surface["noise"] == pytest.approx(0.6), "an add lands on top of a hand"

    surface.set_held({})
    surface.set("noise", 0.9)
    assert surface["noise"] == pytest.approx(0.9)

    surface = Surface(layout=fastgan())
    assert surface["noise"] == pytest.approx(DIALS["noise"][0])
    surface.set("noise", 0.9)
    assert surface["noise"] == pytest.approx(0.9)


def test_the_top_of_the_hold_dial_says_what_the_walk_will_actually_do():
    """The label and the value written come from one function, so they cannot disagree. They
    did: the dial read `100% still` while `apply` wrote 0.95, at the top of the one dial whose
    whole point is a standstill."""
    from ganlive.dials.table import HOLD_MAX, readout

    _written, cfg = _applied(hold=1.0)
    assert cfg.hold == pytest.approx(HOLD_MAX)
    assert readout("hold", 1.0) == f"{HOLD_MAX * 100:.0f}% still"
    for position in (0.0, 0.3, 0.8, 0.95, 1.0):
        _written, cfg = _applied(hold=position)
        assert readout("hold", position) == f"{cfg.hold * 100:.0f}% still", position


def test_every_dial_is_accounted_for_on_a_fully_featured_model():
    """**The same defect wearing the other face.**"""
    from ganlive.bank import live_dials
    from ganlive.dials.fastgan_dials import SETTINGS_WRITTEN
    from ganlive.dials.table import DIRECTIONS
    from ganlive.settings import Knobs

    full = Knobs(sorted(SETTINGS_WRITTEN), "cpu", torch.float32)
    live = live_dials(full, range(DIRECTIONS), fastgan())    # `len()` is all it asks of them
    missing = set(DIALS) - live
    assert not missing, (
        f"{sorted(missing)} can be turned on the strip and reach nothing on any model; "
        f"give each one a Span, a noise band or a direction, or take it off the surface")


def test_directions_read_off_an_onnx_file_match_the_ones_read_off_the_weights(tmp_path):
    """**The strongest argument for ONNX as the format**, and it needs to be true rather than assumed: a
    generator nobody here has the training code for still arrives with its own ranked latent axes, because
    the first weight that consumes `z` is in the file."""
    import torch.nn.utils as U

    from ganlive.dials.derive import sefa, sefa_onnx

    nz = 12

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = U.spectral_norm(torch.nn.ConvTranspose2d(nz, 8, 4, 1, 0, bias=False))

        def forward(self, z):
            return self.up(z.reshape(-1, nz, 1, 1))

    net = Tiny().eval()
    path = tmp_path / "tiny.onnx"
    with torch.no_grad():
        torch.onnx.export(net, (torch.zeros(1, nz),), str(path),
                          input_names=["z"], output_names=["image0"], dynamo=True)

    off_weights = sefa(net, nz)
    off_file = sefa_onnx(path, nz)

    assert len(off_file) == len(off_weights) == nz
    cos = (off_file.basis * off_weights.basis).sum(1).abs()
    assert float(cos.min()) > 0.999, f"the two readers disagree: {cos}"


def test_an_onnx_model_offers_no_settings_it_cannot_write(tmp_path):
    """`knobs.install` swaps steerable modules into a torch module tree and an ONNX graph has
    none, so the MODEL block is empty. Six bright dials writing into nothing is the failure
    this project keeps paying for. An empty `Knobs` accepts the writes and drops them, and
    `live_dials` reads the same empty index to stop them being drawn."""
    from ganlive.bank import live_dials
    from ganlive.dials.fastgan_dials import MODEL
    from ganlive.dials.table import MOTION, Surface
    from ganlive.settings import Knobs

    knobs = Knobs([], "cpu", torch.float32)
    live = live_dials(knobs, None, fastgan())

    assert not (set(MODEL) & live), "an ONNX model has none of these"
    assert set(MOTION) <= live, "and all of these, on every model"
    assert not any(name.startswith("dir") for name in live), "no basis, no direction dials"

    Surface(layout=fastgan()).apply(knobs, WalkConfig())            # every write accepted and dropped
    knobs.commit()


def test_a_direction_dial_moves_the_latent_and_by_exactly_the_amount_asked():
    """**Without this, deleting `SlerpWalk._offset` outright fails nothing.**"""

    from ganlive.dials.table import DIRECTION_RANGE, Surface
    from ganlive.walk import SlerpWalk, WalkConfig

    rows = np.eye(4, 16, dtype=np.float32)
    cfg = WalkConfig(directions=rows)
    walk = SlerpWalk(16, "cpu", cfg)
    surface = Surface(layout=fastgan())

    surface.apply(FakeKnobs(), cfg)
    at_rest = walk.latent(0.5).clone()

    surface.set("dir1", 1.0)
    surface.apply(FakeKnobs(), cfg)
    pushed = walk.latent(0.5)

    delta = (pushed - at_rest).flatten().numpy()
    assert delta[0] == pytest.approx(DIRECTION_RANGE, abs=1e-3), "dir1 must push its own axis"
    assert np.allclose(delta[1:], 0.0, atol=1e-3), "and must not disturb the others"


def test_a_w_push_goes_to_the_seam_and_leaves_the_latent_alone():
    """The other half of the same mechanism, for the family that steers `w`."""

    from ganlive.dials.table import DIRECTION_RANGE, Surface
    from ganlive.walk import SlerpWalk, WalkConfig

    #: Two ranges of four, so a row is only ever non-zero in its own half.
    rows = np.zeros((2, 8), dtype=np.float32)
    rows[0, 1] = 1.0
    rows[1, 5] = 1.0
    seam = torch.zeros(2, 4)
    cfg = WalkConfig(directions=rows, push_into=seam)
    walk = SlerpWalk(16, "cpu", cfg)
    surface = Surface(layout=fastgan())

    surface.apply(FakeKnobs(), cfg)
    at_rest = walk.latent(0.5).clone()
    assert float(seam.abs().max()) == 0.0, "nothing turned, nothing pushed"

    surface.set("dir2", 1.0)
    surface.apply(FakeKnobs(), cfg)
    pushed = walk.latent(0.5)

    assert torch.equal(pushed, at_rest), (
        "a `w` push reached the latent as well; on this family the mapping's pixel norm "
        "would then renormalise the walk's own vector around it")
    assert seam[1, 1] == pytest.approx(DIRECTION_RANGE, abs=1e-3), "dir2 pushes its own axis"
    assert float(seam[0].abs().max()) == 0.0, "and only inside its own style range"

    surface.set("dir2", 0.5)
    surface.apply(FakeKnobs(), cfg)
    walk.latent(0.5)
    assert float(seam.abs().max()) == 0.0, (
        "the seam kept the last push after the dial came back to rest, so the model stays "
        "steered by a dial the strip draws as centred")


def test_a_direction_reads_out_as_a_push_rather_than_as_a_trained_value():
    """`as trained` is meaningless for a direction: there is no trained value to be at."""
    from ganlive.dials.table import DIRECTION_RANGE, readout

    assert readout("dir1", 0.5) == "centred"
    assert readout("dir1", 1.0) == f"+{DIRECTION_RANGE:.2f}"
    assert readout("dir1", 0.0) == f"-{DIRECTION_RANGE:.2f}"


def test_a_two_sided_dial_says_it_is_at_rest_in_its_own_words():
    """Two defects, both found by rendering the strip and looking at it rather than by code."""
    from ganlive.dials.fastgan_dials import DIALS, POLES
    from ganlive.dials.table import POLES as SPINE_POLES
    from ganlive.dials.table import readout

    expected = {"se_64": "as trained", "se_128": "as trained", "se_512": "as trained",
                "reaction": "as written", "noise": "off"}
    gates = fastgan()
    for name, word in expected.items():
        assert readout(name, DIALS[name][0], gates) == word, name
    poles = {**SPINE_POLES, **POLES}
    assert set(expected) <= set(poles), "a dial with poles needs a word for its rest"
    assert all(len(p) == 3 for p in poles.values()), poles


def test_every_walk_config_field_is_either_written_each_frame_or_in_the_cache_key():
    """The cache key deleted an "invalidate" rule; this stops a new field re-creating it."""
    import dataclasses

    from ganlive.dials.table import Surface

    names = {f.name for f in dataclasses.fields(WalkConfig)}
    defaults = {name: getattr(WalkConfig(), name) for name in names}

    written_each_frame = set()
    for position in (0.0, 1.0):
        probe = WalkConfig()
        Surface({name: position for name in DIALS}).apply(FakeKnobs(), probe)
        written_each_frame |= {n for n in names if getattr(probe, n) != defaults[n]}

    keyed = set(SlerpWalk.SEED_FIELDS)
    # `directions` is a property of the loaded weights, not of a frame: `bank.walk` attaches it once when a
    # model is prepared, and nothing on the surface writes it.
    excluded = {"spread", "directions", "latent_on_host", "push_into"}

    missing = names - written_each_frame - keyed - excluded
    assert not missing, (
        f"{missing} is in WalkConfig and nothing has decided what it is: either Surface.apply "
        f"writes it every frame, or SlerpWalk.SEED_FIELDS must include it, or it belongs in "
        f"this test's exclusion list with a reason")
    assert not (keyed & written_each_frame), "a field cannot be both"


def test_the_grit_ladder_the_tests_read_is_the_one_the_frame_loop_runs():
    """`apply` inlines the ladder for speed and `noise_for` states it for the tests, so the six noise
    assertions in this file read a function the render loop never calls."""
    from ganlive.dials.fastgan_dials import NOISE_BANDS, noise_for

    for position in (0.0, 0.1, 0.25, 0.5, 0.72, 0.9, 1.0):
        written, _cfg = _applied(noise=position)
        for band, gain in noise_for(position):
            assert written[band] == pytest.approx(gain), (position, band)
        assert {b for b, _f, _s in NOISE_BANDS} <= set(written), "a band stopped being written"


def test_a_smaller_generator_installs_the_rungs_it_has_and_no_others():
    """A 256-pixel FastGAN has no 512 rungs, and it is a supported size. It used to be
    refused outright, so the whole family below 512 could not be played at all.

    The dials it does not have cannot reach the frame loop: `Knobs.set` drops a name it
    does not carry, and `live_dials` reads the same index to draw the dial dark."""
    from ganlive.dials.steer import install

    knobs = install(_stub_net(gates=("se_64", "se_128", "se_256")), "cpu", torch.float32)
    assert "sle.se_512" not in knobs.index
    assert "sle.se_256" in knobs.index
    knobs.set("sle.se_512", 2.0)          # dropped, not raised: no KeyError mid-performance


def test_an_explicitly_named_setting_the_model_lacks_is_still_refused():
    """A caller with a list -- the ONNX export names its graph inputs -- has to be told,
    because a silently dropped name there is a dial that is not in the exported graph."""
    from ganlive.dials.steer import install

    with pytest.raises(RuntimeError, match="sle.se_512"):
        install(_stub_net(gates=("se_64", "se_128")), "cpu", torch.float32,
                wanted={"sle.se_64", "sle.se_512"})


def test_only_the_settings_something_can_write_are_installed():
    """Thirteen of twenty-two knobs used to be installed with nothing able to write them, and
    ten of those wrapped a rung in a module that multiplies its whole feature map by a live
    tensor holding a constant 1.0, on every frame. `verify` then reported that all thirteen
    moved the picture, which is a pass that says nothing about whether the instrument works."""
    from ganlive.dials.fastgan_dials import SETTINGS_WRITTEN
    from ganlive.dials.steer import install

    net = _stub_net()
    before = net.to_big
    knobs = install(net, "cpu", torch.float32)
    assert set(knobs.names) == set(SETTINGS_WRITTEN)
    assert not any(n.startswith("feat_gain") for n in knobs.names), knobs.names
    # The `pre_tanh` dial is gone from the strip, so nothing writes that setting and the output
    # block is left alone rather than wrapped in a multiply by a constant 1.0 on every frame.
    assert "pre_tanh" not in knobs.names
    assert net.to_big is before


def test_the_settings_vector_is_staged_through_a_ring_like_the_walk_is():
    """A transfer from pinned memory is asynchronous, so a single host buffer can be
    overwritten by the next frame's writes while its copy is still in flight -- half of one
    frame's control state and half of the next. The walk already had a ring for this."""
    from ganlive.settings import Knobs

    knobs = Knobs(["a", "b"], "cpu", torch.float32)
    assert knobs.STAGING >= 2
    seen = []
    for value in (0.25, 0.5, 0.75, 1.0):
        knobs.set("a", value)
        seen.append(knobs.write.ctypes.data)
        knobs.commit()
        assert knobs.write[knobs.index["a"]] == pytest.approx(value)
        assert knobs.write[knobs.index["b"]] == pytest.approx(1.0)
    assert len(set(seen)) == knobs.STAGING, "consecutive frames must not share a buffer"


def test_an_uncalibrated_model_still_plays_exactly_as_it_used_to():
    """`NOISE_BANDS` states an EFFECT now, not a gain, so something has to supply the gain. Until
    a model is calibrated that is the table measured on `gv-2048-ft` -- and it has to be the
    same numbers, because a saved setting is dial positions and would otherwise render
    differently than it did when it was saved."""
    historical = {"noise.feat_512": 30.0, "noise.feat_128": 9.0,
                  "noise.feat_32": 8.0, "noise.feat_8": 60.0}
    assert NOISE_FALLBACK_GAIN == historical, "this changes how every saved setting sounds"

    assert dict(noise_for(1.0)) == pytest.approx(_ramped_to(historical))
    assert dict(noise_for(1.0))["noise.feat_8"] < historical["noise.feat_8"], (
        "the coarsest band is deliberately not fully reachable from the dial")
    assert all(v == pytest.approx(1.0) for v in dict(noise_for(0.0)).values()), "rest is 1.0"


def test_a_calibrated_model_uses_its_own_gains_rather_than_the_table():
    """The whole point: the same dial position must buy the same GRAIN on any model, which
    means a different gain on each. Measured across four checkpoints the fixed gain ran 2.53
    to 58.50 levels on feat_512 -- a 23x spread -- so a table cannot serve them all."""
    mine = {"noise.feat_512": 242.7, "noise.feat_128": 16.0,
            "noise.feat_32": 16.0, "noise.feat_8": 719.9}
    fell_back, used_mine = dict(noise_for(1.0)), dict(noise_for(1.0, mine))
    for name in mine:
        assert used_mine[name] > fell_back[name], f"{name} ignored the calibration"
    assert used_mine == pytest.approx(_ramped_to(mine))

    partial = dict(noise_for(1.0, {"noise.feat_8": 500.0}))
    assert partial["noise.feat_8"] > fell_back["noise.feat_8"]
    assert partial["noise.feat_512"] == pytest.approx(fell_back["noise.feat_512"])


def test_the_calibration_finds_the_gain_that_buys_the_levels_asked_for():
    """The solver, against a response whose answer is known, so this needs no card."""

    import torch

    from ganlive.dials import steer as K
    from ganlive.dials.fastgan_dials import NOISE_BANDS

    class StubKnobs:
        """Just enough of `Knobs`: a name index and a value per setting."""
        def __init__(self):
            self.index = {name: i for i, (name, _t, _s) in enumerate(NOISE_BANDS)}
            self.commits = 0
            self.reset()

        def reset(self):
            self.values = dict.fromkeys(self.index, 1.0)
            self.commit()

        def set(self, name, value):
            self.values[name] = value

        def commit(self):
            self.commits += 1

    SHARPNESS = {"noise.feat_512": 9.0, "noise.feat_128": 21.0,
                 "noise.feat_32": 30.0, "noise.feat_8": 12.0}

    class StubNet:
        def __init__(self, knobs):
            self.knobs = knobs

        def __call__(self, _z):
            total = sum(SHARPNESS[n] * math.log(max(v, 1.0))
                        for n, v in self.knobs.values.items())
            return torch.full((1, 3, 4, 4), -1.0 + 2.0 * total / 255.0)

    knobs = StubKnobs()
    got = K.calibrate_noise(StubNet(knobs), knobs, nz=8, device="cpu",
                           dtype=torch.float32, probes=24)

    for name, target, _start in NOISE_BANDS:
        want = math.exp(target / SHARPNESS[name])
        assert got[name] == pytest.approx(want, rel=0.02), (name, got[name], want)

    assert all(v == pytest.approx(1.0) for v in knobs.values.values()), knobs.values


def test_the_settings_vector_is_not_resent_when_no_dial_moved():
    """**Thirty-two bytes that measured 1.6 ms.** `Tensor.copy_` drops the GIL, and taking it
    back from a window thread that is uploading a texture and presenting costs up to one switch
    interval: 1.64 ms median on the played loop, against 0.03 with no window open. A frame on
    which no dial that writes the model moved was paying that to send the card the numbers it
    already had -- and only the MODEL block writes this vector at all, so a preset driving the
    motion and direction dials holds it still while the picture moves."""
    import torch

    from ganlive.settings import Knobs

    k = Knobs(["a", "b"], "cpu", torch.float32)
    k.commit()
    assert k.skipped == 1, "the vector is built at the neutral, so the first commit sends none"

    k.set("a", 2.0)
    k.commit()
    assert k.skipped == 1 and float(k.vec[0]) == 2.0, "a change still reaches the card"

    k.commit()
    k.set("a", 2.0)
    k.commit()
    assert k.skipped == 3, "and the same value written again is still the same value"

    k.set("b", 0.5)
    k.commit()
    assert k.skipped == 3 and [float(v) for v in k.vec] == [2.0, 0.5]
    k.reset()
    assert [float(v) for v in k.vec] == [1.0, 1.0], "a reset is a change, and must reach it"


def test_use_model_relayouts_at_startup_not_only_on_a_switch():
    """The first model gets its own dials, the same as the fourth."""
    import dataclasses

    from ganlive.dials import table as S
    from ganlive.presets import PresetRunner

    @dataclasses.dataclass
    class _Model:
        layout: object
        dials_live: frozenset

    foreign = S.adopted(("gain_128",), (0.5,), ((0.0, 0.1), (1.0, 2.0)), (30.0,))
    runner = PresetRunner(Preset(name="t", blurb=""), {}, 60.0)
    # The spine, not this project's own table: a runner with no model yet has the dials
    # every model has, and `adopt` brings the loaded one's.
    assert "se_256" not in runner.surface.layout
    assert "reaction" in runner.surface.layout and "dir1" in runner.surface.layout

    runner.use_model(_Model(layout=foreign, dials_live=frozenset({"gain_128"})))

    assert runner.surface.layout == foreign
    assert "gain_128" in runner.surface.layout
    assert "se_256" not in runner.surface.layout, (
        "a FastGAN dial survived onto a foreign model's surface, so the write path and the "
        "strip disagree about which dials exist")

    # And the shape that caused it: a bare set of names carries no layout, so accepting one
    # could only ever half-apply the model. It is refused by name now rather than quietly
    # doing less than the call reads as doing.
    with pytest.raises(TypeError, match="takes the Model"):
        runner.use_model(frozenset({"gain_128"}))


def test_every_load_setting_reaches_every_model_the_bank_ever_loads():
    """**The bank grows after launch.** The shelf loads a model mid-session through `Bank.add`,
    so a setting `build` took and `add` did not would change the instrument under the hand
    with nothing on the strip to explain it. `exact` was exactly that: `build` accepted it and
    `Bank.add` called `_prepare` without it, so a StyleGAN2 added from the shelf ran half
    precision however it had been asked for. One object carries them all now, and this asserts
    the structure rather than the list -- a seventh setting cannot be forgotten the way the
    fourth was."""
    import inspect

    from ganlive import bank as R
    from ganlive.dials.derive import RANDOM_FLOOR

    assert R.LoadOptions().direction_floor == RANDOM_FLOOR, (
        "the default has to be the measured one, not a second opinion about it")
    assert dataclasses.is_dataclass(R.LoadOptions) and R.LoadOptions.__dataclass_params__.frozen, (
        "a mutable Load would let one model's settings follow the next one's")

    # Every load path takes the whole object, so none of them can take a subset of it.
    # By annotation rather than by name: the parameter was called `load` until it shadowed
    # `fastgan.load` inside `_prepare_fastgan` and broke every FastGAN load.
    for fn in (R._prepare, R._prepare_onnx, R._prepare_stylegan2, R._prepare_fastgan):
        taken = [p for p in inspect.signature(fn).parameters.values()
                 if "LoadOptions" in str(p.annotation)]
        assert taken, fn.__name__

    # The rig carries it, and `add` -- the shelf's path -- hands on that same object.
    assert [f.name for f in dataclasses.fields(R.Bank) if f.type == "LoadOptions"] == ["options"]
    assert "self.options" in inspect.getsource(R.Bank.add), (
        "a model loaded from the shelf mid-session would get stock settings")

    # And no setting has been left behind as a loose parameter on the way down. Read off the
    # dataclass rather than listed, so an eighth setting is covered the day it is added and a
    # renamed one cannot leave this assertion guarding a name nothing uses.
    stale = {f.name for f in dataclasses.fields(R.LoadOptions)}
    for fn in (R._prepare, R._prepare_onnx, R._prepare_stylegan2, R._prepare_fastgan):
        left = stale & set(inspect.signature(fn).parameters)
        assert not left, f"{fn.__name__} still takes {sorted(left)} beside the Load"

    # The floor reaches `rank` as the relative bar, which is the only thing that spends it.
    assert "relative=floor" in inspect.getsource(R.directions_for)


def test_a_direction_dial_turns_proportionally_to_what_it_changes():
    """**Pushing proportionally does not turn proportionally.** Laid out straight, a direction
    dial gave 23.6% of its change in the first eighth of its travel and 10% in the last
    quarter, measured over 96 curves on two checkpoints -- so most of the dial was crowded into
    its first half and the top of it did almost nothing under the hand. The table warps the
    push so equal turns are equal amounts of visible change, exactly as `spread` already does.
    """
    from ganlive.dials.table import DIRECTION_POINTS, DIRECTION_RANGE, DIRECTION_RESPONSE, at

    assert at(DIRECTION_POINTS, 0.5) == 0.0, "rest is the middle and pushes nothing"

    # Symmetric about rest: the same turn either way is the same push the other way.
    for x in (0.05, 0.2, 0.35, 0.47, 0.5):
        assert at(DIRECTION_POINTS, 0.5 + x) == pytest.approx(-at(DIRECTION_POINTS, 0.5 - x))

    # Monotone, or the dial doubles back on itself under the hand.
    pushes = [at(DIRECTION_POINTS, i / 64) for i in range(65)]
    assert all(b >= a for a, b in zip(pushes, pushes[1:], strict=False)), pushes

    # Warped the right way round. This is the one claim the derivation below does not make:
    # a straight layout would put half a turn at half the push, and half a turn is worth 71%
    # of the change, so it has to land well under.
    assert at(DIRECTION_POINTS, 0.75) / DIRECTION_RANGE < 0.5, (
        "the table is warped the wrong way; this is worse than straight")

    # The measurement is stated once and the points are derived from it, so the two spellings
    # of the same curve cannot drift apart. The ends come along as the last row.
    for share, push in DIRECTION_RESPONSE:
        assert at(DIRECTION_POINTS, 0.5 + 0.5 * share) == pytest.approx(
            DIRECTION_RANGE * push, abs=1e-9), share


def test_a_direction_dial_is_labelled_from_the_rows_that_survived_the_gate():
    """The strip's half of the same failure: the notes come as data, and are not derived."""
    import ganlive.dials.table as S

    ranges = (("w_coarse", "the 4, 8 and 16 pixel stages"),
              ("w_mid", "the 16, 32 and 64 pixel stages"),
              ("w_mid", "the 16, 32 and 64 pixel stages"))
    notes = S.w_direction_notes(ranges)
    assert len(notes) == 3
    assert "1 of 1 in this model's w_coarse range" in notes[0]
    assert "1 of 2 in this model's w_mid range" in notes[1]
    assert "2 of 2 in this model's w_mid range" in notes[2]
    assert all(S.DIRECTION_TAIL in n for n in notes), "said once, used in both blurbs"

    layout = S.stylegan2(ranges=ranges)
    blurbs = [layout.get(name).blurb for name in S.DIRECTION_DIALS]
    assert blurbs[:3] == list(notes)
    assert all("w_coarse" not in b and "w_mid" not in b for b in blurbs[3:]), (
        "a dial past the end of the measured basis borrowed a range it is not in")
    assert all("principal latent direction" in b for b in blurbs[3:]), (
        "and it should say the generic thing rather than raise, which it once did")

    # No ranges at all -- an uncalibrated load, or a model whose directions did not derive.
    assert all("principal latent direction" in S.stylegan2().get(n).blurb
               for n in S.DIRECTION_DIALS)


def test_a_style_range_dial_says_it_is_truncation_and_not_just_that_it_exists():
    """**The control this family is known for read as "a setting this graph declares".**

    `w_coarse`, `w_mid` and `w_fine` are truncation per style range -- the three headline dials
    on every converted StyleGAN2 -- and they fell through `DERIVED_BLURB` to its catch-all while
    the grain underneath them was described in full. Only StyleGAN2 ever spells a dial `w_*`:
    `adopt` names every dial it derives `gain_*` or `noise_*`, so the family word is safe."""
    import ganlive.dials.table as S

    names = ("w_coarse", "w_mid", "w_fine", "noise_128")
    layout = S.stylegan2(names, (0.5,) * 4, ((0.1, 1.0, 2.0),) * 4, (30.0,) * 4)
    said = {n: layout[n].blurb for n in names}

    for name, word in (("w_coarse", "coarse"), ("w_mid", "mid"), ("w_fine", "fine")):
        assert "truncation" in said[name], said[name]
        assert f"{word} styles" in said[name], "and which range of layers it is"
    assert "the network's own noise" in said["noise_128"], "the grain is unchanged"
    assert "declares" not in " ".join(said.values()), "nothing here falls through any more"

    # An adopted graph is untouched: `w_` cannot reach one, and its own dials still read right.
    plain = S.adopted(("gain_512", "noise_8"), (0.5, 0.0), ((),) * 2, (20.0, 9.0))
    assert "hands upward" in plain["gain_512"].blurb
    assert "the network's own noise" in plain["noise_8"].blurb


def test_the_gate_measures_every_writing_dial_through_the_path_a_hand_takes():
    """A dial the surface writes into a setting the model demodulates straight back out has to
    read as dead at load, not on stage; and the measurement has to go through `Knob.writes`,
    since that is the path a hand takes."""
    import torch

    from ganlive import bank as R
    from ganlive.dials import steer as K
    from ganlive.dials.fastgan_dials import SETTINGS_WRITTEN, fastgan

    knobs = K.Knobs(sorted(SETTINGS_WRITTEN), "cpu", torch.float32)

    class _Net(torch.nn.Module):
        def forward(self, z):
            # Only one gate reaches the picture; every other setting is divided back out.
            return torch.tanh(torch.zeros(1, 3, 8, 8) + (knobs.view("sle.se_256") - 1.0))

    layout = R.measure_dials(_Net(), knobs, fastgan(), 4, "cpu", torch.float32)
    assert layout["se_256"].measured > 1.0
    assert layout["se_512"].measured == 0.0
    assert layout["noise"].measured == 0.0
    assert layout["speed"].measured is None, "a dial that writes nothing is not judged"
    assert float(knobs.committed()[0]) == 1.0, "left at the trained neutral"

    live = R.live_dials(knobs, None, layout)
    assert "se_256" in live
    assert {"se_512", "noise", "se_128"}.isdisjoint(live)
    assert {"speed", "grid", "reaction"} <= live


def test_a_measured_dial_says_so_under_the_hand_whatever_family_it_is():
    from dataclasses import replace

    from ganlive.dials.fastgan_dials import fastgan
    from ganlive.dials.table import Layout

    layout = Layout(tuple(replace(k, measured=12.4) if k.name == "se_256" else k
                          for k in fastgan().knobs))
    panel = _panel(DIALS)
    panel.bank.current = _StubModel(dials_live=frozenset(DIALS), layout=layout)
    assert "12 8-bit levels" in panel._measured("se_256")
    assert panel._measured("speed") == ""
    assert panel._measured("grid") == "", "a motion dial has no measurement to report"
