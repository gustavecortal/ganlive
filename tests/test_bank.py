"""Turning checkpoints into playable models, and switching between them."""

from __future__ import annotations

import math
import pathlib
import types

import numpy as np
import pytest
import torch

from ganlive.dials.table import (
    DIALS,
    GRID_STEPS,
    LATENT,
    MASTER,
    NOISE_BANDS,
    SPANS,
    SPEED_BEATS,
    SPREAD_TABLE,
    noise_for,
    spread_for,
)
from ganlive.frame import FrameStage
from ganlive.walk import (
    BEATS_PER_BAR,
    MusicalClock,
    SlerpWalk,
    WalkConfig,
    _ease_in,
    _ease_out,
    _smoothstep,
    shape,
)
from tests.support import (
    _applied,
    _panel,
    _pulses,
    _StubModel,
)


def _displacement(spread):
    """Read the measured table the other way round: spread in, distance moved out."""
    for (s0, d0), (s1, d1) in zip(SPREAD_TABLE, SPREAD_TABLE[1:], strict=False):
        if spread <= s1:
            return d0 + (d1 - d0) * (spread - s0) / (s1 - s0)
    return SPREAD_TABLE[-1][1]


def _stage_input(height=8, width=12, small=4):
    """A big picture and the small second one, distinguishable by their values."""
    big = torch.zeros(1, 3, height, width)
    big[:, :, :, : width // 2] = 1.0                     # left half bright, right half dark
    return [big, torch.full((1, 3, small, small * 3 // 2), 0.5)]


def _fake_run(root, name, step, width, height, nz=256):
    """A run directory holding one checkpoint that `config_of` can read. Config only -- the
    weights are what makes a real one 674 MB, and nothing here builds a generator."""
    import torch

    folder = root / name / "checkpoints"
    folder.mkdir(parents=True)
    path = folder / f"{step:07d}.pt"
    torch.save({"config": {"nz": nz, "im_size": height, "im_width": width}, "g_ema": {}}, path)
    return path


def test_the_internal_clock_counts_beats_at_the_stated_tempo():
    c = MusicalClock(120.0)
    for _ in range(60):
        c.advance(1 / 60)                                 # one second
    assert c.beats == pytest.approx(2.0, abs=1e-6)        # 120 BPM is 2 beats a second


def test_beats_are_counted_from_pulses_not_integrated_from_the_estimate():
    """An integrated tempo estimate accumulates its own error; a pulse count cannot."""
    c = MusicalClock()
    now = 0.0
    period = 60.0 / (128.0 * MusicalClock.PPQN)
    for i in range(MusicalClock.PPQN * 8):                # eight beats
        now += period * (1.0 + 0.2 * math.sin(i))         # jittery transport
        c.on_pulse(now)
    assert c.beats == pytest.approx(8.0, abs=1e-9)        # exact despite the jitter


def test_the_tempo_estimate_converges_on_the_real_tempo():
    c = MusicalClock(60.0)
    now, period = 0.0, 60.0 / (140.0 * MusicalClock.PPQN)
    for _ in range(MusicalClock.PPQN * 8):
        now += period
        c.on_pulse(now)
    assert c.bpm == pytest.approx(140.0, rel=0.02)


def test_stop_freezes_the_internal_clock():
    c = MusicalClock(120.0)
    c.advance(1.0)
    c.on_stop()
    c.advance(1.0)
    assert c.beats == pytest.approx(2.0)
    c.on_continue()
    c.advance(1.0)
    assert c.beats == pytest.approx(4.0)


def test_bar_phase_is_zero_on_the_downbeat():
    c = MusicalClock(120.0)
    assert c.bar_phase == pytest.approx(0.0)
    c.advance(1.0)                                        # two beats at 120 BPM
    assert c.bar_phase == pytest.approx(0.5)
    c.advance(1.0)
    assert c.bar_phase == pytest.approx(0.0, abs=1e-9)


def test_a_small_spread_keeps_the_session_in_one_neighbourhood():
    """With `home_every` off there is one home forever, so every target stays near it."""
    w = SlerpWalk(256, "cpu", WalkConfig(spread=0.1, home_every=0))
    home = w.home_for(0)
    assert all(torch.equal(w.home_for(k), home) for k in (1, 5, 50))
    far = max(float((w.seed_for(k) - home).norm()) for k in range(40))
    assert far < 6.0, far


def test_home_every_moves_the_neighbourhood_on_schedule():
    w = SlerpWalk(256, "cpu", WalkConfig(spread=0.1, home_every=16))
    assert torch.equal(w.home_for(0), w.home_for(15))
    assert not torch.equal(w.home_for(0), w.home_for(16))
    assert torch.equal(w.home_for(16), w.home_for(31))


def test_a_tempo_change_does_not_jump_the_picture():
    """The retiming must be a change of *rate*, not a discontinuity."""
    c = MusicalClock(128.0)
    w = SlerpWalk(64, "cpu", WalkConfig(beats_per_segment=BEATS_PER_BAR, spread=0.25))
    steps, prev, t = [], None, 0.0
    for bpm, secs in ((128.0, 4.0), (160.0, 4.0)):
        period = 60.0 / (bpm * MusicalClock.PPQN)
        for _ in range(int(secs / period)):
            t += period
            c.on_pulse(t)
            z = w.latent(c.beats)
            if prev is not None:
                steps.append(float((z - prev).norm()))
            prev = z
    assert max(steps) < 3.0 * (sum(steps) / len(steps)), (max(steps), sum(steps) / len(steps))


def test_transport_jitter_cannot_reach_the_position():
    """USB delivery jitter perturbs pulse *intervals*, so it perturbs the tempo estimate. The
    pulse *count* is unaffected, which is why position is taken from the count."""
    import random

    c = MusicalClock(130.0)
    _pulses(c, 130.0, 20.0, jitter=0.002, rng=random.Random(7))
    assert c.beats == pytest.approx(20 * 130 / 60, abs=1e-9)


def test_bars_get_shorter_as_the_tempo_rises():
    """The visible consequence: segment boundaries must crowd together during a ramp."""
    c = MusicalClock(120.0)
    w = SlerpWalk(64, "cpu", WalkConfig(beats_per_segment=BEATS_PER_BAR))
    marks, t = [], 0.0
    while t < 8.0:
        bpm = 120.0 + 25.0 * (t / 8.0)                    # a hand on the encoder
        t += 60.0 / (bpm * MusicalClock.PPQN)
        k = w.segment_index
        w.latent(c.beats)
        c.on_pulse(t)
        if w.segment_index != k:
            marks.append(t)
    gaps = [b - a for a, b in zip(marks, marks[1:], strict=False)]
    assert len(gaps) >= 3, marks
    assert gaps == sorted(gaps, reverse=True), gaps       # strictly shrinking


def test_start_is_what_aligns_the_video_bar_to_the_music_bar():
    """MIDI clock carries tempo but not bar position, so the origin has to come from Start."""
    c = MusicalClock(130.0)
    _pulses(c, 130.0, 1.7)                                # joined mid-pattern
    assert c.beats > 0.0
    c.on_start()                                          # the Rytm's play button
    assert c.beats == pytest.approx(0.0)
    w = SlerpWalk(64, "cpu", WalkConfig(beats_per_segment=BEATS_PER_BAR))
    assert torch.allclose(w.latent(c.beats), w.seed_for(0).view(1, -1), atol=1e-6)


@pytest.mark.parametrize("hold", [0.0, 0.3, 0.8, 0.95])
@pytest.mark.parametrize("when", [0.0, 0.5, 1.0])
def test_the_move_still_starts_at_one_seed_and_ends_at_the_next(hold, when):
    """Whatever the two motion dials are set to, the segment must still begin exactly on its
    own seed and finish on the next one -- that is what keeps arrival on the beat."""
    assert shape(0.0, hold, when) == pytest.approx(0.0, abs=1e-9)
    assert shape(1.0, hold, when) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("hold", [0.0, 0.4, 0.9])
@pytest.mark.parametrize("when", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_picture_never_goes_backwards(hold, when):
    """A curve that dipped would run the morph in reverse for part of the bar."""
    xs = [shape(i / 200, hold, when) for i in range(201)]
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:], strict=False)), (hold, when)


def test_hold_actually_stands_still_for_the_fraction_it_says():
    """`hold` is the dial that was asked for by name, so it is measured rather than trusted:
    at 0.8 with the standstill at the front, four fifths of the bar is one frozen image."""
    frozen = [shape(u / 100, 0.8, 1.0) for u in range(80)]
    assert max(frozen) == pytest.approx(0.0, abs=1e-12)
    assert shape(0.98, 0.8, 1.0) > 0.5                    # and then it moves, fast
    assert shape(0.5, 0.8, 0.0) == pytest.approx(1.0, abs=1e-9)


def test_no_standstill_means_the_picture_is_always_moving():
    """The control for the test above."""
    xs = [shape(u / 50, 0.0, 0.5) for u in range(1, 50)]
    assert all(b > a for a, b in zip(xs, xs[1:], strict=False))


def test_when_decides_whether_the_move_leaves_the_beat_or_arrives_on_it():
    """The fork measurement could not settle, now a dial. At 0 the picture is already moving as
    the beat lands; at 1 it has not started yet and finishes at speed."""
    eps = 1e-4
    leaving = shape(eps, 0.0, 0.0) / eps
    posing = shape(eps, 0.0, 0.5) / eps
    arriving = (1.0 - shape(1.0 - eps, 0.0, 1.0)) / eps
    assert leaving > 2.0, leaving
    assert posing < 0.01, posing
    assert arriving > 2.0, arriving


def test_the_three_curves_this_replaced_are_still_reachable():
    """`when` has to be a superset of the menu of named curves it replaced, or work done
    before the change cannot be reproduced. That is the whole justification for deleting the
    menu rather than keeping both mechanisms."""
    for u in [i / 20 for i in range(21)]:
        assert shape(u, 0.0, 0.5) == pytest.approx(_smoothstep(u), abs=1e-12)
        assert shape(u, 0.0, 0.0) == pytest.approx(_ease_out(u), abs=1e-12)
        assert shape(u, 0.0, 1.0) == pytest.approx(_ease_in(u), abs=1e-12)


def test_every_dial_changes_something():
    """**The guard this project keeps needing.** Three arms were once lost to a knob that did nothing, and a
    dial that writes no parameter gives a clean-looking null result. So each one is moved to both ends of
    its travel and something downstream has to differ."""
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
    """The gates rest in the middle of their travel on purpose: the trained value is in the
    middle and both directions are useful, which is not true of `noise` or `se_64`."""
    for dial, knob in (("se_256", "sle.se_256"), ("se_512", "sle.se_512"),
                       ("se_128", "sle.se_128")):
        lo = _applied(**{dial: 0.0})[0][knob]
        hi = _applied(**{dial: 1.0})[0][knob]
        assert lo < 1.0 < hi, (dial, lo, hi)


def test_speed_only_ever_picks_a_musical_length():
    """A move taking an unmusical number of beats would arrive between beats, which is the one
    thing the whole beat-locked design exists to prevent."""
    for i in range(21):
        assert _applied(speed=i / 20)[1].beats_per_segment in SPEED_BEATS


def test_every_quarter_turn_of_speed_halves_the_time():
    """The documented behaviour of the dial, checked rather than trusted."""
    got = [_applied(speed=x)[1].beats_per_segment for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert got == [8.0, 4.0, 2.0, 1.0, 0.5], got


def test_range_is_laid_out_so_equal_turns_are_equal_amounts_of_change():
    """The underlying number is strongly compressive at the top -- its last third buys a tenth
    of what its first third does -- so a dial mapped straight onto it would be dead over most
    of its travel. The dial is defined against the measured distance instead."""
    steps = [_displacement(spread_for(x / 10)) for x in range(11)]
    gaps = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    assert max(gaps) - min(gaps) < 0.05, gaps
    assert steps[0] == pytest.approx(0.0, abs=1e-6)
    assert steps[-1] == pytest.approx(SPREAD_TABLE[-1][1], abs=0.1)


def test_grit_walks_up_the_bands_instead_of_raising_them_together():
    """One dial over several injection points, finest mark first. Raised together they would
    saturate early and the top half of the dial would do nothing."""
    fine, coarse = NOISE_BANDS[0][0], NOISE_BANDS[-1][0]
    quarter, full = dict(noise_for(0.25)), dict(noise_for(1.0))
    assert quarter[fine] > 1.5, "the finest band should already be in at a quarter turn"
    assert quarter[coarse] == pytest.approx(1.0), "the coarsest should not be, yet"
    assert full[coarse] > 1.5
    for band, _full, _start in NOISE_BANDS:
        xs = [dict(noise_for(i / 20))[band] for i in range(21)]
        assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:], strict=False)), band


def test_the_grid_dial_only_picks_whole_subdivisions():
    for i in range(21):
        assert _applied(grid=i / 20)[1].step_grid in GRID_STEPS


def test_the_stage_shrinks_to_the_size_being_shown():
    stage = FrameStage(4, 6)
    assert tuple(stage.step(_stage_input())[0].shape[-2:]) == (4, 6)


def test_an_exported_graph_is_a_model_on_the_shelf_and_names_itself_apart():
    """An export lives in one flat folder and is named `<run>-<step>.onnx`, so the folder
    says nothing about it and the stem says everything. It must never share a row with the
    checkpoint it came from, which is what the trailing word is for."""
    from ganlive.bank import is_onnx, label_for, run_step, slug_for

    graph = pathlib.Path("runs/onnx/gv-warm-lr3-0078000.onnx")
    check = pathlib.Path("runs/gv-warm-lr3/checkpoints/0078000.pt")

    assert is_onnx(graph) and not is_onnx(check)
    assert run_step(graph) == ("gv-warm-lr3", "78000 onnx")
    assert label_for(graph) == "gv-warm-lr3 78000 onnx"
    assert label_for(check) == "gv-warm-lr3 78000"
    assert slug_for(graph) != slug_for(check)


def test_the_shelf_lists_every_graph_but_only_the_newest_checkpoint(tmp_path):
    """A run holds many checkpoints and the newest is the one anybody means, so it is one
    row. Exports are flat files and every one is a different model, so that folder is as
    many rows as it has files. The shelf used to assume a folder was a model."""
    import types

    from ganlive.bank import Shelf

    run = tmp_path / "a-run" / "checkpoints"
    run.mkdir(parents=True)
    for step in ("0001000.pt", "0002000.pt"):
        (run / step).write_bytes(b"")
    graphs = tmp_path / "onnx"
    graphs.mkdir()
    for name in ("a-run-0002000.onnx", "b-run-0009000.onnx"):
        (graphs / name).write_bytes(b"")

    shelf = Shelf.__new__(Shelf)
    shelf.root = tmp_path
    shelf.bank = types.SimpleNamespace(models=[])
    names = [p.name for p in shelf._models()]
    assert names == ["0002000.pt", "a-run-0002000.onnx", "b-run-0009000.onnx"], names


def test_the_directions_show_their_ranking_without_being_grabbed_one_at_a_time():
    """**Eight dials that look identical and differ by a factor of five.**"""
    panel = _panel(DIALS, levels=(100.0, 50.0, 25.0))
    assert panel._strengths() == pytest.approx((1.0, 0.5, 0.25))

    assert _panel(DIALS)._strengths() == (), "no basis, no ladder to draw"
    assert _panel()._strengths() == (), "and no model at all is not a crash"


def test_a_dark_dial_says_why_in_the_model_s_terms_not_the_interface_s():
    """A column of dashes with no reason is worse than no column. The group heading only
    covers a block that is dark entirely -- five dark directions under three live ones say
    nothing about themselves at all."""
    panel = _panel({"dir1"}, levels=(90.0,))

    why = panel._reason("dir4")
    assert "1 latent direction" in why, why
    assert "moved nothing" in why, "a dropped direction is not a missing feature"

    assert "architecture" in panel._reason("se_256")
    assert panel._reason("se_256") != why, "two different reasons, not one phrase reused"


def test_switching_models_repoints_the_directions_at_the_new_one():
    """The walk is built once, before the loop; the model changes under it."""

    from ganlive.bank import Bank
    from ganlive.walk import WalkConfig

    first = np.eye(2, 8, dtype=np.float32)
    second = np.full((2, 8), 0.5, dtype=np.float32)
    models = [_StubModel(rows=first), _StubModel(rows=second)]
    bank = Bank(models=models, stage=FrameStage(64, 96), device="cpu", width=96, height=64)

    cfg = WalkConfig()
    bank.walk(cfg, dtype=torch.float32)
    assert cfg.directions is first

    bank.use(1)
    assert cfg.directions is second, "the walk still holds the previous model's basis"


def test_the_compiled_window_conversion_agrees_with_the_eager_one():
    """`FrameStage.rgb` runs a compiled kernel when the bank gives it one and an eager chain otherwise, so
    every test in this file exercises the fallback and the card runs the other. Two paths to one answer is
    the drift this whole file was extracted to prevent."""
    from ganlive.models.fastgan import denormalise
    from ganlive.models.graph import to_rgb

    x = torch.linspace(-1.2, 1.2, 3 * 8 * 12).reshape(1, 3, 8, 12)
    was = (denormalise(x.float()) * 255).round().clamp_(0, 255).to(torch.uint8)
    was = was.permute(0, 2, 3, 1)[0].contiguous()
    assert torch.equal(to_rgb(x), was)
    assert to_rgb(x).shape == (8, 12, 3)
    assert to_rgb(torch.full((1, 3, 2, 2), 5.0)).min() == 255
    assert to_rgb(torch.full((1, 3, 2, 2), -5.0)).max() == 0


def test_the_staging_rings_are_kept_per_shape_so_a_switch_does_not_lose_them():
    """A bank of two native sizes sends two shapes to the same destination."""
    stage = FrameStage(4, 6, device="cpu")
    assert stage.rgb_bytes(torch.zeros(1, 3, 4, 6)).shape == (4, 6, 3)
    rings = dict(stage._rings)

    stage.resize(2, 3)
    assert stage.rgb_bytes(torch.zeros(1, 3, 2, 3)).shape == (2, 3, 3)
    assert len(stage._rings) == len(rings) + 1, "the second shape replaced the first's ring"

    stage.resize(4, 6)
    stage.rgb_bytes(torch.zeros(1, 3, 4, 6))
    assert len(stage._rings) == len(rings) + 1, "switching back allocated a third ring"
    assert set(stage.pinned()) == {"rgb"}, "one entry per destination, not per shape"


def test_a_stage_with_no_compiled_conversion_still_produces_a_frame():
    """The fallback is what every test and every machine without the card runs."""
    stage = FrameStage(4, 6)
    out = torch.zeros(1, 3, 4, 6)
    assert stage.rgb_bytes(out).shape == (4, 6, 3)
    assert stage.rgb_bytes(out).dtype.name == "uint8"


def test_a_checkpoint_path_may_name_a_file_or_a_run(tmp_path):
    """Naming a run is how someone asks for its newest checkpoint; naming the file is how they
    ask for an older step. Nothing found is an error rather than an empty list -- a tool that
    loads no models and reports it is ready is the failure this subsystem exists to refuse."""
    from ganlive.bank import checkpoint_for, label_for

    run = tmp_path / "gv-test" / "checkpoints"
    run.mkdir(parents=True)
    for step in (2000, 12000, 8000):
        (run / f"{step:07d}.pt").write_bytes(b"")

    assert checkpoint_for(tmp_path / "gv-test") == run / "0012000.pt"
    assert checkpoint_for(run) == run / "0012000.pt"
    assert checkpoint_for(run / "0008000.pt") == run / "0008000.pt"
    assert label_for(run / "0008000.pt") == "gv-test 8000"
    with pytest.raises(FileNotFoundError):
        checkpoint_for(tmp_path / "nothing-here")


def test_the_shelf_lists_what_is_on_disk_and_every_readable_model_can_join(tmp_path):
    """The picker's whole content: what is playing, what is a click away, and what its shape is -- which is
    now a *note* on the row and no longer a refusal."""
    import types

    from ganlive.bank import Bank, Shelf

    here = _fake_run(tmp_path, "gv-here", 82000, 1536, 1024)
    _fake_run(tmp_path, "gv-bigger", 72000, 3072, 2048)      # another native size
    _fake_run(tmp_path, "gv-square", 6900, 512, 512)         # 1:1 against 3:2
    _fake_run(tmp_path, "gv-narrow", 1000, 1536, 1024, nz=128)   # another latent width
    (tmp_path / "not-a-run").mkdir()                          # no checkpoint: not listed

    loaded = types.SimpleNamespace(path=here, name="gv-here 82000", cfg=types.SimpleNamespace(
        nz=256, ladder=types.SimpleNamespace(width=1536, height=1024)))
    bank = Bank(models=[loaded], stage=FrameStage(1024, 1536, device="cpu"), device="cpu",
              width=1536, height=1024)
    shelf = Shelf(bank, tmp_path)

    by_name = {e.name: e for e in shelf.entries()}
    assert set(by_name) == {"gv-here 82000", "gv-bigger 72000", "gv-square 6900",
                            "gv-narrow 1000"}, sorted(by_name)
    assert shelf.entries()[0].loaded, "what is already playing belongs at the top"
    assert shelf.entries() is shelf.entries(), "the listing is scanned once, not per repaint"

    assert not any(e.why for e in shelf.entries()), "nothing readable is refused any more"
    assert by_name["gv-square 6900"].note == "512x512 z256"
    assert by_name["gv-narrow 1000"].note == "1536x1024 z128"
    assert not by_name["gv-here 82000"].note, "what is loaded is described by its own strip"

    shelf.request(by_name["gv-square 6900"])
    assert shelf.pending == str(by_name["gv-square 6900"].path), (
        "a square model must be loadable into a 3:2 bank")

    shelf.pending = None
    shelf.request(by_name["gv-here 82000"])
    assert shelf.pending is None, "one already loaded is a switch, not a load"
    shelf.request(by_name["gv-bigger 72000"])
    assert shelf.pending == str(by_name["gv-bigger 72000"].path)

    shelf.bank = types.SimpleNamespace(add=lambda p: (_ for _ in ()).throw(ValueError("nope")))
    assert shelf.service() is None
    assert "nope" in shelf.note
    assert shelf.pending is None, "a serviced request is taken, whether or not it worked"


def test_every_model_is_shown_at_its_own_native_size_not_the_banks_smallest():
    """Each model carries its own frame size, and loading another one cannot change it."""
    import types

    from ganlive.bank import frame_size

    def cfg(w, h):
        return types.SimpleNamespace(ladder=types.SimpleNamespace(width=w, height=h))

    small, big = cfg(1536, 1024), cfg(3072, 2048)
    assert frame_size(big, 0) == (2048, 3072)
    assert frame_size(small, 0) == (1024, 1536)

    # The screen is the only thing that may reduce one, and it reduces each on its own terms.
    assert frame_size(big, None, (2560, 1440)) == (1440, 2160)
    assert frame_size(small, None, (2560, 1440)) == (1024, 1536)

    # A square model keeps its square in a bank whose other member is 3:2. Under `bank_size`
    # this pair could not be held at all: one size for two aspect ratios is a stretch.
    assert frame_size(cfg(1024, 1024), None, (2560, 1440)) == (1024, 1024)


def test_a_switch_moves_the_stage_to_the_incoming_models_size():
    """The stage follows the model, and loading a second one leaves the first alone."""
    import types

    from ganlive.bank import Bank

    class Net(torch.nn.Module):
        def __init__(self, w, h):
            super().__init__()
            self.w, self.h = w, h

        def forward(self, z):
            return [torch.zeros(1, 3, self.h, self.w), torch.zeros(1, 3, self.h // 4,
                                                                   self.w // 4)]

    def model(w, h):
        return types.SimpleNamespace(
            net=Net(w, h), graphs=0, rows=None, push=None, path=pathlib.Path("m.pt"),
            cfg=types.SimpleNamespace(nz=8, ladder=types.SimpleNamespace(width=w, height=h)))

    big, small = model(48, 32), model(24, 16)
    r = Bank(models=[big, small], stage=FrameStage(32, 48, device="cpu"), device="cpu",
            width=48, height=32, dtype=torch.float32)

    assert (r.height, r.width) == (32, 48)
    assert r.stage.rgb_bytes(r.stage.step(big.net(None))).shape == (32, 48, 3)

    r.use(1)
    assert (r.height, r.width) == (16, 24), "the stage did not follow the switch"
    assert r.stage.rgb_bytes(r.stage.step(small.net(None))).shape == (16, 24, 3)

    r.use(0)
    assert (r.height, r.width) == (32, 48), "switching back left the big model downscaled"


def test_a_bank_may_mix_latent_widths_and_aspect_ratios_and_refuses_only_a_duplicate():
    """What two checkpoints must share to play in one window: nothing."""
    import types

    from ganlive import bank as R

    def cfg(nz, w, h):
        return types.SimpleNamespace(nz=nz, ladder=types.SimpleNamespace(width=w, height=h))

    def m(path, nz, w, h):
        return types.SimpleNamespace(path=pathlib.Path(path), name=path, cfg=cfg(nz, w, h))

    incoming = cfg(512, 1024, 1024)                    # square, nz 512: refused outright once
    bank = [m("runs/gv/checkpoints/0072000.pt", 256, 3072, 2048)]
    was, R.config_of = R.config_of, lambda _p: incoming
    try:
        assert R.admit(bank, pathlib.Path("runs/stylegan2/ffhq.pt")) is None

        with pytest.raises(ValueError, match="already in this bank"):
            R.admit(bank, pathlib.Path("runs/gv/checkpoints/0072000.pt"))
    finally:
        R.config_of = was

    assert R.admit([], pathlib.Path("runs/anything.pt")) is None


def test_the_frame_report_says_which_model_the_slow_frames_belong_to():
    """**One aggregate over a bank cannot be read.** A played 338s session switched models and
    then loaded a StyleGAN2 from the shelf, and reported 53.8 fps with 25.7% of frames over
    budget as a single number -- which is either one expensive model or every model getting
    slower, and the report could not tell them apart."""
    from ganlive.tools import play as live

    fast = [8.0] * 100
    slow = [20.0] * 40 + [8.0] * 60
    lines = live.per_model_lines({"gv-warm-lr3 78000": fast, "stylegan2 ffhq": slow}, 16.67)

    assert len(lines) == 3, lines
    assert lines[0].split() == ["per", "model", "frames", "median", "p95", "over"]
    # First played, first named -- the order the session happened in.
    assert "gv-warm-lr3 78000" in lines[1] and "stylegan2 ffhq" in lines[2]
    assert lines[1].endswith("   0.0%"), "the cheap model carried none of it"
    assert lines[2].endswith("  40.0%"), "and the expensive one carried all of it"
    # The names are the widest thing on the line, so the columns must be measured, not assumed.
    assert len({len(line) for line in lines}) == 1, lines

    # One model played is the ordinary case and has nothing to attribute.
    assert live.per_model_lines({"gv-2048-ft 72000": fast}, 16.67) == []
    assert live.per_model_lines({}, 16.67) == []


def test_auto_height_is_what_the_screen_can_actually_draw():
    """A 2560x1440 screen draws a 3072x2048 frame at 2160x1440, so producing native sends 2.03x
    the pixels anyone can see across the bus twice. Measured on the live loop: 18.54 ms against
    12.17, and frames over budget falling from 74.8% to 0.3%."""
    from ganlive.bank import fit_height

    assert fit_height(2048, 3072, None, (2560, 1440)) == 1440
    assert fit_height(2048, 3072, None, (7680, 4320)) == 2048
    assert fit_height(2048, 3072, None, (1445, 4320)) == 962
    assert fit_height(2048, 3072, None, (1445, 4320)) % 2 == 0


def test_every_height_it_can_return_is_a_legal_nv12_frame():
    """NV12 has half-resolution chroma planes, so an odd height is not a frame. Until `auto` chose the
    number unattended only an explicit odd `--height` could produce one."""
    from ganlive.bank import fit_height

    for want in (1, 3, 101, 1023, 1439, 2047, 9999):
        assert fit_height(2048, 3072, want) % 2 == 0, want
    for width in range(1000, 4000, 37):
        got = fit_height(2048, 3072, None, (width, 1440))
        assert got % 2 == 0 and 2 <= got <= 2048, (width, got)


def test_a_screen_that_cannot_be_asked_falls_back_to_native_rather_than_a_guess():
    from ganlive.bank import fit_height

    for screen in (None, (), (0, 1440), (2560, 0)):
        assert fit_height(2048, 3072, None, screen) == 2048, screen
    assert fit_height(2048, 3072, 0) == 2048           # native, explicitly
    assert fit_height(2048, 3072, 1024) == 1024


def test_the_three_words_a_height_can_be():
    """`None` for auto travels all the way to `fit_height` rather than becoming a negative
    sentinel three separate places have to know how to read."""
    import argparse

    import pytest

    from ganlive.bank import parse_height

    assert parse_height("auto") is None
    assert parse_height("native") == 0
    assert parse_height("1024") == 1024
    with pytest.raises(argparse.ArgumentTypeError, match="auto, native"):
        parse_height("tall")


def test_the_worst_case_setting_turns_on_every_dial_that_costs_anything():
    """**The frame budget is priced against one setting**, so a dial missing from it is a per-frame cost
    nobody has measured -- which is what happened once when a stage dial was added and left out of the
    hand-written list."""

    from ganlive.dials.table import DIALS, SPANS
    from ganlive.tools import latency

    worst = latency.WORST
    driven = {name for name, value in worst.dials.items() if value != DIALS[name][0]}
    driven |= {i.dial for i in worst.impulses} | {m.dial for m in worst.macros}

    wanted = {s.dial for s in SPANS} | set(LATENT) | {"noise"}
    missing = wanted - driven
    assert not missing, (
        f"{sorted(missing)} cost something every frame and the worst case leaves them at "
        f"rest, so the frame-budget worst case is not the worst case")


def test_a_model_switch_moves_where_the_latent_goes_with_it():
    """A bank can hold both kinds, so the flag cannot be decided once at build time. **The
    generator says**, not the file suffix: an ONNX graph and a captured torch graph both read
    the latent from the host, and a compiled module from the card."""
    import pathlib

    from ganlive.bank import Bank
    from ganlive.walk import WalkConfig

    models = [_StubModel(path=pathlib.Path("runs/a.onnx"),
                         net=types.SimpleNamespace(latent_on_host=True)),
              _StubModel(path=pathlib.Path("runs/b.pt"))]
    bank = Bank(models=models, stage=FrameStage(64, 96), device="cpu", width=96, height=64)
    cfg = WalkConfig()
    bank._walk_cfgs = [cfg]

    bank.index = 0
    bank._rewire()
    assert cfg.latent_on_host is True, "a generator that takes the latent on the host is obeyed"

    bank.index = 1
    bank._rewire()
    assert cfg.latent_on_host is False, (
        "switching to a checkpoint left the walk handing a numpy array to a torch generator")


def test_both_stylegan2_layout_builders_offer_the_same_spine(tmp_path):
    """There are two of them -- the swept one `_prepare_stylegan2` builds, and the bare one a
    caller with no measurement gets -- and a dial that appears on one and not the other is a
    surface that changes under the hand for reasons the player cannot see."""
    import torch

    from ganlive import bank as R
    from ganlive.dials import table as S
    from ganlive.models import stylegan2 as S2

    cfg = S2.Config(z_dim=16, w_dim=16, img_resolution=32, channel_base=128,
                    channel_max=32, num_layers=2, num_fp16_res=0)
    converted = tmp_path / "converted.pt"
    S2.save(converted, cfg, S2.Generator(cfg).state_dict())
    ours = tmp_path / "ours.pt"
    torch.save({"g_ema": {}, "config": {"nz": 256, "im_size": 512}}, ours)

    bare = R.layout_for(None, converted)
    # The calibrated shape: dials with measured travel, as `_prepare_stylegan2` builds it.
    # `curves` are values at even spacing, one tuple per dial -- `onnx_dials.Dial.curve`.
    swept = S.stylegan2(("w_coarse", "noise_32"), (0.5, 0.0),
                        ((0.3, 1.0, 2.0), (0.0, 1.5, 3.0)), (25.0, 25.0))

    spine = [n for n in bare if bare[n].group != "MODEL"]
    assert spine == [n for n in swept if swept[n].group != "MODEL"]
    assert [n for n in bare if bare[n].group == "MODEL"] == []
    assert [n for n in swept if swept[n].group == "MODEL"] == ["w_coarse", "noise_32"], (
        "the calibrated builder must still carry the model's own dials")

    mine = R.layout_for(None, ours)
    assert [n for n in mine if mine[n].group != "MODEL"] == spine, (
        "the spine is the spine on every family, which is the whole argument for having one")


def test_the_instrument_opens_a_fastgan_checkpoint(tmp_path):
    """The whole FastGAN load path: rebuild, freeze the noise, install, measure, lay out.

    There was no test that took a `.pt` FastGAN from disk to a playable `Model`, so when
    `sample.load_generator` became `fastgan.load` and collided with `_prepare_fastgan`'s
    `load: LoadOptions` parameter, 372 tests passed and `ganlive play` raised on every
    FastGAN checkpoint it was given."""
    from ganlive import bank as R
    from ganlive.models.fastgan import Generator

    torch.manual_seed(0)          # the dial gate measures the picture; random weights vary
    cfg = dict(nz=16, ngf=8, im_size=256, im_width=None)
    net = Generator(**cfg)
    path = tmp_path / "tiny.pt"
    torch.save({"g_ema": net.state_dict(), "config": cfg}, path)

    model, _ = R._prepare(path, "cpu", torch.float32, (None, None, None),
                          R.LoadOptions(compile_net=False, capture=False,
                                        measure_grain=False))
    assert model.cfg.nz == 16 and model.cfg.ladder.height == 256
    assert model.knobs.names, "no dials were installed on the generator"
    # A 256-pixel generator has no 512 rungs, so the dials that write them are offered and
    # drawn dark rather than installed. Which of the rest survive is a measurement -- these
    # weights are random -- so the claim here is structural, not a list.
    assert "sle.se_512" not in model.knobs.index
    assert "se_512" not in model.dials_live
    assert {"se_64", "se_128", "se_256"} & model.dials_live, "no gate reached the model"
    assert {"reaction", "speed", "spread"} <= model.dials_live, "the spine is always live"
