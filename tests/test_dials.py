"""Which dials exist, where they come from, and that each one moves the picture."""

from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import pytest
import torch
import torch.nn.utils as U
from torch import nn

from ganlive import bank as R
from ganlive import families as FA
from ganlive.clock import WalkConfig
from ganlive.curves import at, clamp01
from ganlive.dials import steer as K
from ganlive.dials import table as S
from ganlive.dials.derive import sefa, sefa_onnx
from ganlive.dials.fastgan_dials import (
    DIALS,
    MODEL,
    NOISE_BANDS,
    NOISE_FALLBACK_GAIN,
    NOISE_RAMP,
    POLES,
    SETTINGS_WRITTEN,
    SPANS,
    fastgan,
    noise_for,
)
from ganlive.dials.gate import directions_for, live_dials, measure_dials
from ganlive.dials.steer import install
from ganlive.dials.table import (
    DIRECTION_POINTS,
    DIRECTION_RANGE,
    DIRECTION_RESPONSE,
    DIRECTIONS,
    GRID_STEPS,
    HOLD_MAX,
    LATENT,
    MASTER,
    MOTION,
    SPEED_BEATS,
    SPREAD_TABLE,
    Layout,
    Surface,
    readout,
    spread_for,
)
from ganlive.models.fastgan import SkipLayerExcitation
from ganlive.models.fold import FoldedNoise
from ganlive.models.steerable import Slot
from ganlive.pixels import RANDOM_FLOOR
from ganlive.presets import Preset, PresetRunner
from ganlive.settings import Settings
from ganlive.tools import latency
from ganlive.walk import SlerpWalk
from tests.support import (
    FakeSettings,
    _applied,
    _panel,
    _step,
    _StubModel,
)


def _stub_net(gates=("se_64", "se_128", "se_256", "se_512"),
              rungs=("feat_8", "feat_32", "feat_128", "feat_512")):
    """A generator-shaped object with only what `install` looks at."""
    net = nn.Module()
    for name in rungs:
        coeff = torch.ones(1, 2, 1, 1)
        net.add_module(name, nn.Sequential(FoldedNoise(coeff, torch.zeros(1, 1, 4, 4))))
    for name in gates:
        net.add_module(name, SkipLayerExcitation(2, 2, (4, 4)))
    net.to_big = nn.Identity()
    net.init = nn.Identity()
    return net


def _ramped_to(gains):
    """`noise_for(1.0)` if every band carried `gains`."""
    return {name: 1.0 + (gains[name] - 1.0) * clamp01((1.0 - start) / NOISE_RAMP)
            for name, levels, start in NOISE_BANDS}


def test_every_dial_changes_something():
    """A dial that writes nothing gives a clean-looking null result, so each one is moved to
    both ends of its travel and something downstream has to differ."""
    inert = []
    for name in DIALS:
        if name in MASTER:
            continue
        lo_state, lo_walk = _applied(**{name: 0.0})
        hi_state, hi_walk = _applied(**{name: 1.0})
        if lo_state == hi_state and vars(lo_walk) == vars(hi_walk):
            inert.append(name)
    assert not inert, f"dials that write nothing: {inert}"
    written = set(_applied()[0])
    assert {s.target for s in SPANS} <= written, written


def test_a_dial_at_rest_leaves_the_trained_neutral_alone():
    """Resting values have to reproduce the network as trained, or every preset starts
    off-centre and nothing is comparable to anything."""
    state, _walk = _applied()
    for name in ("sle.se_256", "sle.se_512", "sle.se_128", "sle.se_64"):
        assert state[name] == pytest.approx(1.0, abs=1e-9), name


def test_the_gate_dials_are_two_sided_around_their_resting_value():
    """The gates rest mid-travel because the trained value is there and both ways are useful,
    which is not true of `noise` or `se_64`."""
    for dial, setting in (("se_256", "sle.se_256"), ("se_512", "sle.se_512"),
                          ("se_128", "sle.se_128")):
        lo = _applied(**{dial: 0.0})[0][setting]
        hi = _applied(**{dial: 1.0})[0][setting]
        assert lo < 1.0 < hi, (dial, lo, hi)


def test_speed_only_ever_picks_a_musical_length():
    """A move taking an unmusical number of beats would arrive between beats."""
    for i in range(21):
        assert _applied(speed=i / 20)[1].beats_per_segment in SPEED_BEATS


def test_every_quarter_turn_of_speed_halves_the_time():
    got = [_applied(speed=x)[1].beats_per_segment for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert got == [8.0, 4.0, 2.0, 1.0, 0.5], got


def test_the_grid_dial_only_picks_whole_subdivisions():
    for i in range(21):
        assert _applied(grid=i / 20)[1].step_grid in GRID_STEPS


def test_spread_is_laid_out_so_equal_turns_are_equal_amounts_of_change():
    """The underlying number is strongly compressive at the top, so a dial mapped straight
    onto it would be dead over most of its travel. The dial is laid out against the measured
    distance instead."""
    # The measured table read the other way round: spread in, distance moved out.
    steps = [at(SPREAD_TABLE, spread_for(x / 10)) for x in range(11)]
    gaps = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    assert max(gaps) - min(gaps) < 0.05, gaps
    assert steps[0] == pytest.approx(0.0, abs=1e-6)
    assert steps[-1] == pytest.approx(SPREAD_TABLE[-1][1], abs=0.1)


def test_the_documented_spread_table_matches_what_the_walk_does():
    """`SPREAD_TABLE` is how a spread value gets chosen, so it is checked against the walk.
    Tolerances are wide because these are means over a random sequence."""
    for spread, expected in ((0.05, 1.8), (0.10, 3.5), (0.20, 7.0), (0.35, 11.8),
                             (0.50, 16.0), (0.70, 20.2), (1.00, 22.5)):
        got = _step(SlerpWalk(256, "cpu", WalkConfig(spread=spread)), n=60)
        assert got == pytest.approx(expected, abs=0.6), (spread, got, expected)


def test_grit_walks_up_the_bands_instead_of_raising_them_together():
    """One dial over several injection points, finest first. Raised together they would
    saturate early and the top half of the dial would do nothing."""
    fine, coarse = NOISE_BANDS[0][0], NOISE_BANDS[-1][0]
    quarter, full = dict(noise_for(0.25)), dict(noise_for(1.0))
    assert quarter[fine] > 1.5, "the finest band should already be in at a quarter turn"
    assert quarter[coarse] == pytest.approx(1.0), "the coarsest should not be, yet"
    assert full[coarse] > 1.5
    for band, _full, _start in NOISE_BANDS:
        xs = [dict(noise_for(i / 20))[band] for i in range(21)]
        assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:], strict=False)), band


def test_the_grit_ladder_the_tests_read_is_the_one_the_frame_loop_runs():
    """`apply` inlines the ladder for speed and `noise_for` states it for the tests, so the
    two are checked against each other."""
    for position in (0.0, 0.1, 0.25, 0.5, 0.72, 0.9, 1.0):
        written, _cfg = _applied(noise=position)
        for band, gain in noise_for(position):
            assert written[band] == pytest.approx(gain), (position, band)
        assert {b for b, _f, _s in NOISE_BANDS} <= set(written), "a band stopped being written"


def test_nothing_can_be_driven_off_its_scale():
    """A shove far past the end of a dial stops at the end."""
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
    """`4 beats` says what a slider at 0.25 does. The readout comes from the same tables
    `apply` reads, so the two cannot disagree about which detent a position lands on."""
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
    for position in (0.0, 0.2, 0.25, 0.5, 0.7, 1.0):
        _written, cfg = _applied(speed=position, grid=position)
        assert readout("speed", position).startswith(f"{cfg.beats_per_segment:g}")
        assert readout("grid", position) == ("glide" if not cfg.step_grid
                                             else f"{cfg.step_grid} steps")


def test_the_top_of_the_hold_dial_says_what_the_walk_will_actually_do():
    """The label and the value written come from one function, so they cannot disagree, at
    the top of the travel included."""
    assert _applied(hold=1.0)[1].hold == pytest.approx(HOLD_MAX)
    for position in (0.0, 0.3, 0.8, 0.95, 1.0):
        _written, cfg = _applied(hold=position)
        assert readout("hold", position) == f"{cfg.hold * 100:.0f}% still", position


def test_the_order_between_the_four_kinds_of_writer_is_stated_on_the_surface():
    """Resting value, then a hand (which holds), then anything that `set`s (which a hand
    beats), then anything that `add`s (which lands on top)."""
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


def test_every_dial_is_accounted_for_on_a_fully_featured_model():
    """A dial that reaches nothing on any model is a control that cannot be played."""
    full = Settings(sorted(SETTINGS_WRITTEN), "cpu", torch.float32)
    live = live_dials(full, range(DIRECTIONS), fastgan())    # `len()` is all it asks of them
    missing = set(DIALS) - live
    assert not missing, (
        f"{sorted(missing)} can be turned on the strip and reach nothing on any model; "
        f"give each one a Span, a noise band or a direction, or take it off the surface")


def test_directions_read_off_an_onnx_file_match_the_ones_read_off_the_weights(tmp_path):
    """A generator nobody here has the training code for still arrives with its own ranked
    latent axes, because the first weight that consumes `z` is in the file."""
    nz = 12

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = U.spectral_norm(nn.ConvTranspose2d(nz, 8, 4, 1, 0, bias=False))

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


def test_an_onnx_model_offers_no_settings_it_cannot_write():
    """`steer.install` swaps steerable modules into a torch module tree and an ONNX graph has
    none, so the MODEL block is empty. An empty `Settings` accepts the writes and drops them,
    and `live_dials` reads the same empty index to stop them being drawn."""
    settings = Settings([], "cpu", torch.float32)
    live = live_dials(settings, None, fastgan())

    assert not (set(MODEL) & live), "an ONNX model has none of these"
    assert set(MOTION) <= live, "and all of these, on every model"
    assert not any(name.startswith("dir") for name in live), "no basis, no direction dials"

    Surface(layout=fastgan()).apply(settings, WalkConfig())   # every write accepted and dropped
    settings.commit()


def test_a_direction_dial_moves_the_latent_and_by_exactly_the_amount_asked():
    rows = np.eye(4, 16, dtype=np.float32)
    cfg = WalkConfig(directions=rows)
    walk = SlerpWalk(16, "cpu", cfg)
    surface = Surface(layout=fastgan())

    surface.apply(FakeSettings(), cfg)
    at_rest = walk.latent(0.5).clone()

    surface.set("dir1", 1.0)
    surface.apply(FakeSettings(), cfg)
    pushed = walk.latent(0.5)

    delta = (pushed - at_rest).flatten().numpy()
    assert delta[0] == pytest.approx(DIRECTION_RANGE, abs=1e-3), "dir1 must push its own axis"
    assert np.allclose(delta[1:], 0.0, atol=1e-3), "and must not disturb the others"


def test_a_w_push_goes_to_the_push_buffer_and_leaves_the_latent_alone():
    """The same mechanism, for the family that steers `w`."""
    # Two ranges of four, so a row is only ever non-zero in its own half.
    rows = np.zeros((2, 8), dtype=np.float32)
    rows[0, 1] = 1.0
    rows[1, 5] = 1.0
    push = torch.zeros(2, 4)
    cfg = WalkConfig(directions=rows, push_into=push)
    walk = SlerpWalk(16, "cpu", cfg)
    surface = Surface(layout=fastgan())

    surface.apply(FakeSettings(), cfg)
    at_rest = walk.latent(0.5).clone()
    assert float(push.abs().max()) == 0.0, "nothing turned, nothing pushed"

    surface.set("dir2", 1.0)
    surface.apply(FakeSettings(), cfg)
    pushed = walk.latent(0.5)

    assert torch.equal(pushed, at_rest), (
        "a `w` push reached the latent as well; on this family the mapping's pixel norm "
        "would then renormalise the walk's own vector around it")
    assert push[1, 1] == pytest.approx(DIRECTION_RANGE, abs=1e-3), "dir2 pushes its own axis"
    assert float(push[0].abs().max()) == 0.0, "and only inside its own style range"

    surface.set("dir2", 0.5)
    surface.apply(FakeSettings(), cfg)
    walk.latent(0.5)
    assert float(push.abs().max()) == 0.0, (
        "the push buffer kept the last push after the dial came back to rest, so the model "
        "stays steered by a dial the strip draws as centred")


def test_a_direction_reads_out_as_a_push_rather_than_as_a_trained_value():
    """`as trained` is meaningless for a direction: there is no trained value to be at."""
    assert readout("dir1", 0.5) == "centred"
    assert readout("dir1", 1.0) == f"+{DIRECTION_RANGE:.2f}"
    assert readout("dir1", 0.0) == f"-{DIRECTION_RANGE:.2f}"


def test_a_two_sided_dial_says_it_is_at_rest_in_its_own_words():
    expected = {"se_64": "as trained", "se_128": "as trained", "se_512": "as trained",
                "reaction": "as written", "noise": "off"}
    gates = fastgan()
    for name, word in expected.items():
        assert readout(name, DIALS[name][0], gates) == word, name
    poles = {**S.POLES, **POLES}
    assert set(expected) <= set(poles), "a dial with poles needs a word for its rest"
    assert all(len(p) == 3 for p in poles.values()), poles


def test_every_walk_config_field_is_either_written_each_frame_or_in_the_cache_key():
    """A new `WalkConfig` field has to be one or the other, or the walk caches a stale value."""
    names = {f.name for f in dataclasses.fields(WalkConfig)}
    defaults = {name: getattr(WalkConfig(), name) for name in names}

    written_each_frame = set()
    for position in (0.0, 1.0):
        probe = WalkConfig()
        Surface({name: position for name in DIALS}).apply(FakeSettings(), probe)
        written_each_frame |= {n for n in names if getattr(probe, n) != defaults[n]}

    keyed = set(SlerpWalk.SEED_FIELDS)
    # Properties of the loaded model, not of a frame: `Bank` sets them on a switch, and
    # nothing on the surface writes them.
    excluded = {"spread", "directions", "latent_on_host", "push_into"}

    missing = names - written_each_frame - keyed - excluded
    assert not missing, (
        f"{missing} is in WalkConfig and nothing has decided what it is: either Surface.apply "
        f"writes it every frame, or SlerpWalk.SEED_FIELDS must include it, or it belongs in "
        f"this test's exclusion list with a reason")
    assert not (keyed & written_each_frame), "a field cannot be both"


def test_a_smaller_generator_installs_the_rungs_it_has_and_no_others():
    """A 256-pixel FastGAN has no 512 rungs, and it is a supported size. The dials it does not
    have cannot reach the frame loop: `Settings.set` drops a name it does not carry, and
    `live_dials` reads the same index to draw the dial dark."""
    settings = install(_stub_net(gates=("se_64", "se_128", "se_256")), "cpu", torch.float32)
    assert "sle.se_512" not in settings.index
    assert "sle.se_256" in settings.index
    settings.set("sle.se_512", 2.0)          # dropped, not raised: no KeyError mid-performance


def test_a_steerable_module_reads_the_whole_vector_and_not_a_view_of_its_slot():
    """A compiled MPS kernel reads a half-precision view at an odd offset from the wrong
    element, so every module slices its slot inside its own forward."""
    net = _stub_net()
    settings = install(net, "cpu", torch.float16)
    modules = [m for m in net.modules() if isinstance(m, Slot)]
    assert len(modules) == 8
    assert all(m.holder is settings for m in modules)
    assert sorted(m.index for m in modules) == list(range(8))


def test_only_the_settings_something_can_write_are_installed():
    """A setting no dial writes would wrap a layer in a multiply by a constant 1.0 on every
    frame, and the gate would report it as live."""
    net = _stub_net()
    before = net.to_big
    settings = install(net, "cpu", torch.float32)
    assert set(settings.names) == set(SETTINGS_WRITTEN)
    assert not any(n.startswith("feat_gain") for n in settings.names), settings.names
    assert net.to_big is before, "the output block is left alone"


def test_an_uncalibrated_model_still_plays_its_saved_settings_the_same():
    """`NOISE_BANDS` states an effect, so an uncalibrated model needs a fallback gain; it has
    to be the measured table, because a saved setting is dial positions and would otherwise
    render differently than when it was saved."""
    fallback = {"noise.feat_512": 30.0, "noise.feat_128": 9.0,
                "noise.feat_32": 8.0, "noise.feat_8": 60.0}
    assert NOISE_FALLBACK_GAIN == fallback, "this changes how every saved setting looks"

    assert dict(noise_for(1.0)) == pytest.approx(_ramped_to(fallback))
    assert dict(noise_for(1.0))["noise.feat_8"] < fallback["noise.feat_8"], (
        "the coarsest band is deliberately not fully reachable from the dial")
    assert all(v == pytest.approx(1.0) for v in dict(noise_for(0.0)).values()), "rest is 1.0"


def test_a_calibrated_model_uses_its_own_gains_rather_than_the_table():
    """The same dial position must buy the same grain on any model, which means a different
    gain on each; one fixed gain varies more than twentyfold across checkpoints."""
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
    class StubSettings:
        """Just enough of `Settings`: a name index and a value per setting."""
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
        def __init__(self, settings):
            self.settings = settings

        def __call__(self, _z):
            total = sum(SHARPNESS[n] * math.log(max(v, 1.0))
                        for n, v in self.settings.values.items())
            return torch.full((1, 3, 4, 4), -1.0 + 2.0 * total / 255.0)

    settings = StubSettings()
    got = K.calibrate_noise(StubNet(settings), settings, nz=8, device="cpu",
                            dtype=torch.float32, probes=24)

    for name, target, _start in NOISE_BANDS:
        want = math.exp(target / SHARPNESS[name])
        assert got[name] == pytest.approx(want, rel=0.02), (name, got[name], want)

    assert all(v == pytest.approx(1.0) for v in settings.values.values()), settings.values


def test_use_model_relayouts_at_startup_not_only_on_a_switch():
    """The first model gets its own dials, the same as the fourth."""
    foreign = S.adopted(("gain_128",), (0.5,), ((0.1, 1.0, 2.0),), (30.0,))
    runner = PresetRunner(Preset(name="t", blurb=""), {}, 60.0)
    # A runner with no model yet has the dials every model has, not this project's own.
    assert "se_256" not in runner.surface.layout
    assert "reaction" in runner.surface.layout and "dir1" in runner.surface.layout

    runner.use_model(_StubModel(layout=foreign, dials_live=frozenset({"gain_128"})))

    assert runner.surface.layout == foreign
    assert "gain_128" in runner.surface.layout
    assert "se_256" not in runner.surface.layout, (
        "a FastGAN dial survived onto a foreign model's surface, so the write path and the "
        "strip disagree about which dials exist")


def test_every_load_setting_reaches_every_model_the_bank_ever_loads():
    """The shelf loads a model mid-session through `Bank.add`, so a setting `build` took and
    `add` did not would change ganlive under the hand. One object carries them all,
    and this asserts the structure rather than the list."""
    LoadOptions = FA.LoadOptions
    assert LoadOptions().direction_floor == RANDOM_FLOOR, (
        "the default has to be the measured one, not a second opinion about it")
    assert dataclasses.is_dataclass(LoadOptions) and LoadOptions.__dataclass_params__.frozen, (
        "mutable options would let one model's settings follow the next one's")

    # Every load path takes the whole object, so none of them can take a subset of it.
    loaders = (R._prepare, FA._prepare_onnx, FA._prepare_stylegan2, FA._prepare_fastgan)
    for fn in loaders:
        taken = [p for p in inspect.signature(fn).parameters.values()
                 if "LoadOptions" in str(p.annotation)]
        assert taken, fn.__name__

    # The bank carries it, and `add` -- the shelf's path -- hands on that same object.
    assert [f.name for f in dataclasses.fields(R.Bank) if f.type == "LoadOptions"] == ["options"]
    assert "self.options" in inspect.getsource(R.Bank.add), (
        "a model loaded from the shelf mid-session would get stock settings")

    # And no setting is also a loose parameter on the way down.
    loose = {f.name for f in dataclasses.fields(LoadOptions)}
    for fn in loaders:
        left = loose & set(inspect.signature(fn).parameters)
        assert not left, f"{fn.__name__} still takes {sorted(left)} beside the options"

    # The floor reaches `rank` as the relative bar, which is the only thing that spends it.
    assert "relative=floor" in inspect.getsource(directions_for)


def test_a_direction_dial_turns_proportionally_to_what_it_changes():
    """Pushing proportionally does not change the picture proportionally: most of a straight
    dial's change would be crowded into its first half. The table warps the push so equal
    turns are equal amounts of visible change, as `spread` does."""
    assert at(DIRECTION_POINTS, 0.5) == 0.0, "rest is the middle and pushes nothing"

    # Symmetric about rest: the same turn either way is the same push the other way.
    for x in (0.05, 0.2, 0.35, 0.47, 0.5):
        assert at(DIRECTION_POINTS, 0.5 + x) == pytest.approx(-at(DIRECTION_POINTS, 0.5 - x))

    # Monotone, or the dial doubles back on itself under the hand.
    pushes = [at(DIRECTION_POINTS, i / 64) for i in range(65)]
    assert all(b >= a for a, b in zip(pushes, pushes[1:], strict=False)), pushes

    # Warped the right way round: half a turn is worth more than half the change, so it has
    # to land well under half the push.
    assert at(DIRECTION_POINTS, 0.75) / DIRECTION_RANGE < 0.5, (
        "the table is warped the wrong way; this is worse than straight")

    # The measurement is stated once and the points are derived from it.
    for share, push in DIRECTION_RESPONSE:
        assert at(DIRECTION_POINTS, 0.5 + 0.5 * share) == pytest.approx(
            DIRECTION_RANGE * push, abs=1e-9), share


def test_a_direction_dial_is_labelled_from_the_rows_that_survived_the_gate():
    """The notes come as data, from the rows that survived, not derived from the dial index."""
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
        "and it should say the generic thing rather than raise")

    # No ranges at all -- an uncalibrated load, or a model whose directions did not derive.
    assert all("principal latent direction" in S.stylegan2().get(n).blurb
               for n in S.DIRECTION_DIALS)


def test_a_style_range_dial_says_it_is_truncation_and_not_just_that_it_exists():
    """`w_coarse`, `w_mid` and `w_fine` are truncation per style range, the headline dials on
    every converted StyleGAN2. Only StyleGAN2 spells a dial `w_*`: `adopt` names every dial it
    derives `gain_*` or `noise_*`, so the family word is safe."""
    names = ("w_coarse", "w_mid", "w_fine", "noise_128")
    layout = S.stylegan2(names, (0.5,) * 4, ((0.1, 1.0, 2.0),) * 4, (30.0,) * 4)
    said = {n: layout[n].blurb for n in names}

    for name, word in (("w_coarse", "coarse"), ("w_mid", "mid"), ("w_fine", "fine")):
        assert "truncation" in said[name], said[name]
        assert f"{word} styles" in said[name], "and which range of layers it is"
    assert "the network's own noise" in said["noise_128"], "the grain is unchanged"
    assert "declares" not in " ".join(said.values()), "nothing here falls through to the default"

    # An adopted graph's own dials still read right.
    plain = S.adopted(("gain_512", "noise_8"), (0.5, 0.0), ((),) * 2, (20.0, 9.0))
    assert "hands upward" in plain["gain_512"].blurb
    assert "the network's own noise" in plain["noise_8"].blurb


@pytest.mark.parametrize("captured", [False, True], ids=["module", "captured"])
def test_the_gate_measures_every_writing_dial_through_the_path_a_hand_takes(captured):
    """A dial whose setting the model divides straight back out has to read as dead at load,
    measured through `Knob.writes` since that is the path a hand takes. A captured generator
    replays into one output buffer, so the frame at rest must be copied before a dial moves."""
    settings = Settings(sorted(SETTINGS_WRITTEN), "cpu", torch.float32)
    out = torch.zeros(1, 3, 8, 8)

    class _Net(nn.Module):
        def forward(self, z):
            # Only one gate reaches the picture; every other setting is divided back out.
            frame = torch.tanh(torch.zeros(1, 3, 8, 8) + (settings.view("sle.se_256") - 1.0))
            return out.copy_(frame) if captured else frame

    layout = measure_dials(_Net(), settings, fastgan(), 4, "cpu", torch.float32)
    assert layout["se_256"].measured > 1.0
    assert layout["se_512"].measured == 0.0
    assert layout["noise"].measured == 0.0
    assert layout["speed"].measured is None, "a dial that writes nothing is not judged"
    assert float(settings.committed()[0]) == 1.0, "left at the trained neutral"

    live = live_dials(settings, None, layout)
    assert "se_256" in live
    assert {"se_512", "noise", "se_128"}.isdisjoint(live)
    assert {"speed", "grid", "reaction"} <= live


def test_a_measured_dial_says_so_under_the_hand_whatever_family_it_is():
    layout = Layout(tuple(dataclasses.replace(k, measured=12.4) if k.name == "se_256" else k
                          for k in fastgan().knobs))
    panel = _panel(DIALS)
    panel.bank.current = _StubModel(dials_live=frozenset(DIALS), layout=layout)
    assert "12 8-bit levels" in panel._measured("se_256")
    assert panel._measured("speed") == ""
    assert panel._measured("grid") == "", "a motion dial has no measurement to report"


def test_the_worst_case_setting_turns_on_every_dial_that_costs_anything():
    """The frame budget is priced against one setting, so a dial missing from it is a
    per-frame cost nobody has measured."""
    worst = latency.worst_case()
    driven = {name for name, value in worst.dials.items() if value != DIALS[name][0]}
    driven |= {i.dial for i in worst.impulses} | {m.dial for m in worst.macros}

    wanted = {s.dial for s in SPANS} | set(LATENT) | {"noise"}
    missing = wanted - driven
    assert not missing, (
        f"{sorted(missing)} cost something every frame and the worst case leaves them at "
        f"rest, so the frame-budget worst case is not the worst case")

    # Given a model's own layout it prices that model's dials, not this project's.
    foreign = S.stylegan2(("w_coarse", "noise_32"), (0.5, 0.0),
                          ((0.3, 1.0, 2.0), (0.0, 1.5, 3.0)), (25.0, 25.0))
    theirs = latency.worst_case(foreign)
    assert {"w_coarse", "noise_32"} <= set(theirs.dials)
    assert not {s.dial for s in SPANS} & set(theirs.dials), (
        "the budget named this project's gates at a model that does not have them, and "
        "`use_model` dropped every one")
