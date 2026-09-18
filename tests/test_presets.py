"""A preset: the rules connecting what the drums do to what the picture does."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ganlive.dials.table import (
    DIALS,
)
from ganlive.presets import Impulse, Macro, Preset  # noqa: E402

NZ = 32
STILL = Preset(
    name="still",
    blurb="test fixture",
)
BREATHE = Preset(
    name="breathe",
    blurb="test fixture",
    dials={"spread": 0.12, "speed": 0.25},
    macros=[
        Macro("density", "spread", 2.5, 9.5, 0.10, 0.62, glide=1.6),
        Macro("density", "se_512", 2.5, 9.5, 0.40, 0.68, glide=1.2),
        Macro("density", "se_128", 2.5, 9.5, 0.42, 0.62, glide=1.4),
    ],
)
PULSE = Preset(
    name="pulse",
    blurb="test fixture",
    dials={"spread": 0.24},
    impulses=[
        Impulse("*", "dir1", amount=0.22, decay=0.13, velocity=0.0),
        Impulse("*", "noise", amount=0.16, decay=0.10, velocity=0.0),
    ],
)
VOICES = Preset(
    name="voices",
    blurb="test fixture",
    dials={"spread": 0.22},
    impulses=[
        Impulse("BD", "se_128", amount=0.34, decay=0.17, velocity=0.8),
        Impulse("CP", "se_512", amount=0.32, decay=0.16),
        Impulse("SD", "se_512", amount=0.24, decay=0.12),
        Impulse("OH", "se_256", amount=-0.30, decay=0.30),
        Impulse("CH", "noise", amount=0.10, decay=0.07, velocity=0.9),
        Impulse("CY", "noise", amount=0.30, decay=0.90),
        Impulse("LT", "dir1", amount=0.20, decay=0.22),
        Impulse("MT", "dir2", amount=0.18, decay=0.20),
        Impulse("HT", "dir3", amount=0.16, decay=0.18),
    ],
)
RELEASE = Preset(
    name="release",
    blurb="test fixture",
    dials={"hold": 0.78, "late": 0.15, "spread": 0.42, "speed": 0.5},
    impulses=[
        # A short attack, because these move where the picture IS rather than how fast it is
        # going, and shoving one instantly is a visible step. 60 ms is under four frames.
        Impulse("BD", "hold", amount=-0.62, decay=0.34, velocity=0.6, attack=0.06),
        Impulse("CP", "spread", amount=0.28, decay=0.45, attack=0.06),
        Impulse("OH", "late", amount=0.25, decay=0.30, attack=0.06),
    ],
    macros=[Macro("density", "speed", 2.5, 9.5, 0.30, 0.62, glide=1.8)],
)
FULL = Preset(
    name="full",
    blurb="test fixture",
    dials={"hold": 0.45, "late": 0.35, "spread": 0.20},
    impulses=list(VOICES.impulses) + [
        Impulse("BD", "hold", amount=-0.34, decay=0.30, velocity=0.6, attack=0.06),
        Impulse("RS", "dir1", amount=0.20, decay=0.20),
    ],
    macros=[
        Macro("density", "spread", 2.5, 9.5, 0.14, 0.55, glide=1.6),
        Macro("density", "hold", 2.5, 9.5, 0.62, 0.20, glide=1.8),
    ],
)
FIXTURES = {p.name: p for p in (STILL, BREATHE, PULSE, VOICES, RELEASE, FULL)}

from tests.support import FakeKnobs  # noqa: E402


def test_a_hit_cannot_push_a_dial_off_its_scale():
    """The clamp is at the dial, once, instead of at every parameter -- so an impulse with a
    wildly wrong amount is a setting that does nothing extra, not a broken picture."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Preset, PresetRunner

    preset = Preset(name="t", blurb="", dials={"noise": 0.5},
                  impulses=[Impulse("BD", "noise", amount=9.0, decay=0.5, velocity=0.0)])
    runner = PresetRunner(preset, INDEX, 60.0)
    since = [1e6] * len(INDEX)
    since[INDEX["BD"]] = 0.0
    runner.apply(since, {}, FakeKnobs())
    assert runner.surface["noise"] == 1.0


def test_the_runner_counts_which_dials_were_played():
    """Whether a dial earns its place on the strip is a count, printed at the end of a run:
    frames it moved, time a hand held it, how far it got from rest. A dial nothing touched
    is named as such rather than left off the list."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Preset, PresetRunner

    preset = Preset(name="t", blurb="",
                  impulses=[Impulse("BD", "noise", amount=0.5, decay=0.5, velocity=0.0)])
    runner = PresetRunner(preset, INDEX, 60.0)
    knobs, quiet = FakeKnobs(), [1e6] * len(INDEX)
    hit = list(quiet)
    hit[INDEX["BD"]] = 0.0
    runner.hold("hand", {"reaction": 0.5})
    runner.apply(quiet, {}, knobs)               # first frame: nothing to compare against
    for since in (hit, hit, quiet):
        runner.apply(since, {}, knobs)
    frames, held, far = runner.usage["noise"]
    assert frames >= 2 and held == 0 and far == pytest.approx(0.5)
    assert runner.usage["reaction"][1] == 4
    report = runner.usage_report()
    assert report[0].startswith("noise")
    assert "in hand" not in report[0]
    assert "never moved" in report[-1] and "speed" in report[-1] and "reaction" in report[-1]


def test_a_hit_on_any_track_reaches_a_star_rule():
    """One rule for the whole kit is the commonest thing to want and should not need twelve."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Preset, PresetRunner

    preset = Preset(name="t", blurb="",
                  impulses=[Impulse("*", "noise", amount=0.5, decay=0.5, velocity=0.0)])
    runner = PresetRunner(preset, INDEX, 60.0)
    knobs, since = FakeKnobs(), [1e6] * len(INDEX)
    runner.apply(since, {}, knobs)
    assert runner.surface["noise"] == pytest.approx(0.0)
    since[INDEX["CH"]] = 0.0                              # a hat, which no rule names
    runner.observe([(INDEX["CH"], 0.5, 0.0)])
    runner.apply(since, {}, knobs)
    assert runner.surface["noise"] > 0.4


def test_a_whole_kit_rule_decays_on_audio_time_like_every_other_rule():
    """`*` rules used to count their own seconds, `+= 1/fps` per frame, while every named rule read the
    extractor's array. The two came apart exactly when frames ran late -- and at the documented 21.31 ms
    against a 16.7 ms period that is 1.3x fast, on `pulse`, which is the one preset built entirely from `*`
    rules."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Preset, PresetRunner

    preset = Preset(name="t", blurb="",
                  impulses=[Impulse("*", "noise", amount=0.5, decay=0.2, velocity=0.0)])
    seen = {}
    for real_dt in (1 / 60, 1 / 30):                       # on time, then every frame late
        runner = PresetRunner(preset, INDEX, 60.0)
        since = [1e6] * len(INDEX)
        since[INDEX["BD"]] = 0.0
        runner.observe([(INDEX["BD"], 0.5, 0.0)])
        for _ in range(6):
            runner.apply(since, {}, FakeKnobs())
            since = [s + real_dt for s in since]           # the audio thread's own clock
        seen[real_dt] = runner.surface["noise"]

    assert seen[1 / 30] < seen[1 / 60] * 0.75, seen


def test_a_hand_on_a_dial_sets_where_the_drums_push_from():
    """`hands` is the seam a physical encoder writes to, and it is applied with the resting
    values rather than over the top of the rules -- so turning an encoder moves where the
    reaction happens instead of cancelling it."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Preset, PresetRunner

    preset = Preset(name="t", blurb="", dials={"noise": 0.1},
                  impulses=[Impulse("BD", "noise", amount=0.2, decay=0.5, velocity=0.0)])
    runner = PresetRunner(preset, INDEX, 60.0)
    knobs = FakeKnobs()
    since = [1e6] * len(INDEX)
    since[INDEX["BD"]] = 0.0

    runner.apply(since, {}, knobs)
    without = runner.surface["noise"]
    runner.hold("test", {**runner.held_by("test"), "noise": 0.6})
    runner.apply(since, {}, knobs)
    assert runner.surface["noise"] == pytest.approx(without + 0.5, abs=1e-6)


def test_reaction_scales_how_hard_hits_land_without_touching_the_arrangement():
    """The one control to reach for when a performance is too much or too little."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Macro, Preset, PresetRunner

    preset = Preset(
        name="t", blurb="",
        impulses=[Impulse("BD", "noise", amount=0.4, decay=0.5, velocity=0.0)],
        macros=[Macro("density", "dir1", 0.0, 10.0, 0.2, 0.8, glide=0.0)])
    since = [1e6] * len(INDEX)
    since[INDEX["BD"]] = 0.0
    seen = {}
    for reaction in (0.0, 0.5, 1.0):
        runner = PresetRunner(preset, INDEX, 60.0)
        runner.hold("test", {**runner.held_by("test"), "reaction": reaction})
        runner.apply(since, {"density": 10.0}, FakeKnobs())
        seen[reaction] = (runner.surface["noise"], runner.surface["dir1"])

    hit_at_rest = seen[0.5][0]
    assert seen[0.0][0] == pytest.approx(0.0), "at 0 the hits must not land at all"
    assert seen[1.0][0] == pytest.approx(hit_at_rest * 2, abs=1e-6), "1 is twice as hard"
    assert seen[0.0][1] == seen[0.5][1] == seen[1.0][1] == pytest.approx(0.8)


def test_at_zero_reaction_any_setting_behaves_like_the_structural_one():
    """A useful property that falls out rather than being built: turn it off and the drumming
    stops reaching the picture, leaving only the arrangement."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    since = [0.0] * len(INDEX)
    out = {}
    for name in ("full", "voices"):
        runner = PresetRunner(FIXTURES[name], INDEX, 60.0)
        runner.hold("test", {**runner.held_by("test"), "reaction": 0.0})
        runner.apply(since, {"density": 6.0}, FakeKnobs())
        out[name] = dict(runner.surface.values)
    for name, values in out.items():
        for dial in ("se_512", "se_256", "noise"):
            assert values[dial] == pytest.approx(DIALS[dial][0]), (name, dial)


def test_a_hand_beats_the_slow_rule_for_that_dial_and_only_that_dial():
    """A slow rule SETS, so without this a slider on any dial a macro drives would be overwritten a
    microsecond later and the control would look broken."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Macro, Preset, PresetRunner

    preset = Preset(name="t", blurb="",
                  macros=[Macro("density", "spread", 0.0, 10.0, 0.1, 0.9, glide=0.0),
                          Macro("density", "dir1", 0.0, 10.0, 0.1, 0.9, glide=0.0)],
                  impulses=[Impulse("BD", "spread", amount=0.2, decay=0.5, velocity=0.0)])
    since = [1e6] * len(INDEX)
    runner = PresetRunner(preset, INDEX, 60.0)
    runner.apply(since, {"density": 10.0}, FakeKnobs())
    assert runner.surface["spread"] == pytest.approx(0.9), "the rule drives it when free"

    runner.hold("test", {"spread": 0.25})
    runner.apply(since, {"density": 10.0}, FakeKnobs())
    assert runner.surface["spread"] == pytest.approx(0.25), "the hand must win"
    assert runner.surface["dir1"] == pytest.approx(0.9), "and only over its own dial"

    since[INDEX["BD"]] = 0.0
    runner.apply(since, {"density": 10.0}, FakeKnobs())
    assert runner.surface["spread"] > 0.25, "a hit still pushes up from where the hand set it"


def test_letting_go_returns_the_dial_to_where_the_slow_rule_has_reached():
    """The filter keeps running under a hand. Freezing it instead would make releasing a
    slider glide the dial from wherever the rule was when the hand arrived, which is a move
    nothing asked for and would read as the slider being sticky."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Macro, Preset, PresetRunner

    preset = Preset(name="t", blurb="",
                  macros=[Macro("density", "dir1", 0.0, 10.0, 0.0, 1.0, glide=0.2)])
    since = [1e6] * len(INDEX)
    runner = PresetRunner(preset, INDEX, 60.0)
    runner.hold("test", {"dir1": 0.1})
    for _ in range(120):                                  # two seconds under a hand
        runner.apply(since, {"density": 10.0}, FakeKnobs())
    assert runner.surface["dir1"] == pytest.approx(0.1)
    runner.hold("test", {})
    runner.apply(since, {"density": 10.0}, FakeKnobs())
    assert runner.surface["dir1"] > 0.9, "the rule should have gone on running underneath"


def test_switching_patch_keeps_the_objects_the_loop_and_the_walk_hold():
    """The console changes setting while the loop runs. The loop holds the runner and the walk
    holds `walk_cfg`, so rebuilding either would leave something driving an object nothing
    reads -- silently, which is the failure this project keeps paying for."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    cfg = runner.walk_cfg
    runner.hold("test", {"noise": 0.5})
    runner.load(FIXTURES["release"])

    assert runner.walk_cfg is cfg, "the walk would stop seeing every motion dial"
    assert runner.preset.name == "release"
    assert runner.hands == {"noise": 0.5}, "a hand on a control survives a change of setting"
    runner.apply([1e6] * len(INDEX), {"density": 0.0}, FakeKnobs())
    assert runner.surface["hold"] == pytest.approx(FIXTURES["release"].dials["hold"])


def test_a_setting_naming_a_dial_that_no_longer_exists_says_so():
    """**Silence here cost the player both of his saved takes.**"""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import Impulse, Macro, Preset, PresetRunner

    old = Preset(name="from-before-the-rename", blurb="",
                dials={"warmth": 0.8, "hold": 0.4},
                impulses=[Impulse("BD", "grit", amount=0.3, decay=0.2),
                          Impulse("SD", "noise", amount=0.2, decay=0.2)],
                macros=[Macro("density", "tint", 2.5, 9.5, 0.1, 0.6, glide=1.0)])
    runner = PresetRunner(old, INDEX, 60.0)

    assert "warmth (no such dial)" in runner.dropped
    assert "BD->grit (no such dial)" in runner.dropped
    assert any("tint" in line for line in runner.dropped), runner.dropped
    assert len(runner._impulses) == 1, "the rule that still names a real dial must survive"
    assert not runner._macros, "a macro on a dead dial is dropped, not silently applied"


def test_two_sources_can_hold_different_dials_without_dropping_each_other():
    """`set_hands` documented that it took one writer and that an encoder arriving as a second
    would need the merge moved inside. Two writers already existed: the console read-merged-wrote
    at the call site and the sweep renderer replaced the whole dict, which would have dropped
    every console-held dial the moment they met."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold("console", {"noise": 0.6})
    runner.hold("encoder", {"se_256": 0.2})
    assert runner.hands == {"noise": 0.6, "se_256": 0.2}

    runner.hold("encoder", {"se_256": 0.2, "noise": 0.9})
    assert runner.hands["noise"] == pytest.approx(0.9)
    runner.free("encoder", "noise")
    assert runner.hands["noise"] == pytest.approx(0.6), "the console never let go of it"
    runner.free("console")
    assert runner.hands == {"se_256": 0.2}


def test_a_writer_rebinds_the_dict_the_loop_reads_rather_than_editing_it():
    """The render loop iterates `hands` while the window thread writes it. Inserting into a
    dict that is being iterated is a `RuntimeError` that could only ever fire mid-performance,
    so every writer has to rebind -- one level up as well, or the merge itself is the race."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold("a", {"noise": 0.5})
    before = runner.hands
    runner.hold("b", {"se_256": 0.5})
    assert runner.hands is not before, "the loop's dict must never be edited under it"
    assert before == {"noise": 0.5}, "and the one it already read must not change"


def test_a_rule_wired_past_the_end_of_the_kit_is_reported_not_silently_skipped():
    """`--layout tracks` against an eight-input machine puts CY and CB past the end, which used
    to be a `continue` inside the frame loop sixty times a second with nothing said."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    full = PresetRunner(FIXTURES["voices"], INDEX, 60.0, channels=12)
    assert not full.dropped

    narrow = PresetRunner(FIXTURES["voices"], INDEX, 60.0, channels=8)
    assert narrow.dropped, "CY sits on channel 10 and this kit has eight"
    assert any("CY" in line for line in narrow.dropped), narrow.dropped
    assert all("CY" not in imp.track for imp, _ch in narrow._impulses)
    narrow.apply(np.full(8, 1e6, dtype=np.float32), {"density": 4.0},
                 FakeKnobs())


def test_a_drum_can_be_wired_to_a_dial_while_it_runs():
    """The thing this whole layer was heading toward: which drum drives which dial was data on
    the preset already, and what was missing was a way to change it without editing a file."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    assert runner.routing() == {}

    assert runner.route("BD", "noise") is True
    assert [(i.track, ch) for i, ch in runner.routing()["noise"]] == [("BD", INDEX["BD"])]
    since = np.full(len(INDEX), 1e6, dtype=np.float32)
    since[INDEX["BD"]] = 0.0
    runner.apply(since, {}, FakeKnobs())
    assert runner.surface["noise"] > DIALS["noise"][0]

    assert runner.route("BD", "noise") is False, "the same call unwires it"
    assert "noise" not in runner.routing()


def test_how_hard_a_drum_pushes_a_dial_can_be_set_without_rewiring_it():
    """Control a dial from a drum, but *not too much*. The amount was on the
    rule already and only the interface could not reach it -- so the failure this guards is a
    grid that can only wire at one fixed strength."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import AMOUNT_MAX, Impulse, PresetRunner

    bd = INDEX["BD"]
    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    assert runner.amount_on("noise", bd) is None, "nothing is wired yet"
    assert runner.set_amount_on("noise", bd, 0.5) is None, "an unwired cell is not created"

    runner.route("BD", "noise")
    assert runner.amount_on("noise", bd) == 0.3

    assert runner.set_amount_on("noise", bd, 0.08) == 0.08
    assert [(i.track, i.amount) for i, _ch in runner.routing()["noise"]] == [("BD", 0.08)]

    def push(dial, amount):
        r = PresetRunner(FIXTURES["still"], INDEX, 60.0)
        r.route("BD", dial)
        r.set_amount_on(dial, INDEX["BD"], amount)
        since = np.full(len(INDEX), 1e6, dtype=np.float32)
        since[INDEX["BD"]] = 0.0
        r.apply(since, {}, FakeKnobs())
        return r.surface[dial] - DIALS[dial][0]

    gentle, hard = push("noise", 0.08), push("noise", 0.8)
    assert 0 < gentle < hard, (gentle, hard)

    assert DIALS["se_128"][0] == 0.5, "this test needs a dial that rests mid-travel"
    assert push("se_128", -0.25) < 0 < push("se_128", 0.25)

    assert runner.set_amount_on("noise", bd, 87.0) == AMOUNT_MAX
    assert runner.set_amount_on("noise", bd, -87.0) == -AMOUNT_MAX
    assert Impulse("BD", "noise", amount=9.0).amount == AMOUNT_MAX


def test_the_strength_that_is_drawn_and_the_strength_that_is_edited_are_one_wire():
    """A routing cell is a CHANNEL, and under the voices layout two drums share one. Reading
    the loudest rule on the channel while writing to whichever track a display listed first
    let the bar sit still while the wheel moved something else -- a control that looks dead."""
    from ganlive.control.tracks import channel_map
    from ganlive.presets import Preset, PresetRunner

    voices = channel_map("voices")
    assert voices["RS"] == voices["CP"], "this test needs two drums on one channel"
    channel = voices["RS"]

    preset = Preset(**{**FIXTURES["still"].__dict__, "impulses": []})
    runner = PresetRunner(preset, voices, 60.0)
    runner.route("RS", "se_512")
    runner.route("CP", "se_512")
    runner.set_amount_on("se_512", channel, 0.12)
    runner.set_amount_on("se_512", channel, 0.44)

    assert runner.amount_on("se_512", channel) == 0.44
    assert sorted(i.amount for i, _ch in runner.routing()["se_512"]) == [0.44, 0.44]


def test_a_click_in_the_grid_wires_the_whole_column_and_a_second_click_clears_it():
    """**The column is a channel, not a drum.** RS and CP share one pair of outputs, so the
    grid draws them as one cell and reads its strength back per channel -- but the click wired
    only the first of the two. A cell lit by CP could not be cleared by clicking it: the click
    added an RS rule beside it, and clicking again took that one away and left the light on.
    Four of the twelve tracks could not be wired from the grid at all."""
    from ganlive.control.tracks import channel_map
    from ganlive.presets import Preset, PresetRunner

    voices = channel_map("voices")
    channel = voices["RS"]
    assert voices["CP"] == channel, "this test needs two drums on one channel"

    runner = PresetRunner(Preset(**{**FIXTURES["still"].__dict__, "impulses": []}), voices, 60.0)
    runner.route("CP", "se_512", amount=0.5)          # a saved setting wired the second of them

    assert runner.wire(["RS", "CP"], "se_512") is False, "the cell is lit, so a click clears it"
    assert runner.amount_on("se_512", channel) is None, "and the light goes out"

    assert runner.wire(["RS", "CP"], "se_512") is True
    assert sorted(i.track for i, _ch in runner.routing()["se_512"]) == ["CP", "RS"], \
        "both drums on the channel fire the cell the grid drew"


def test_changing_a_strength_does_not_restart_the_slow_rules():
    """`load` blanks `_macro_state` so a new setting does not glide down from the old one's
    smoothed values. A wheel drag is dozens of notches, and reloading on each one restarted
    every slow rule's glide -- a move nobody asked for, from the event thread, while the render
    thread was inside `apply`."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["full"], INDEX, 60.0)
    since = np.full(len(INDEX), 1e6, dtype=np.float32)
    runner.apply(since, {"density": 0.8, "energy": 0.5, "active": 0.4},
                 FakeKnobs())
    glide = dict(runner._macro_state)
    assert glide, "this test needs a preset whose slow rules have started smoothing"

    runner.route("BD", "noise")                      # creating a wire DOES reload, and resets
    runner.apply(since, {"density": 0.8, "energy": 0.5, "active": 0.4},
                 FakeKnobs())
    glide = dict(runner._macro_state)

    runner.set_amount_on("noise", INDEX["BD"], 0.5)
    assert runner._macro_state == glide, "adjusting a strength must not restart the glide"

    assert runner.amount_on("noise", INDEX["BD"]) == 0.5


def test_a_strength_set_by_hand_survives_being_saved_and_read_back():
    """A setting is saved as JSON and picked up next time. A strength that did not round-trip
    would silently revert to the default push, which looks like the interface forgetting."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner, from_dict, to_dict

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.route("BD", "noise")
    runner.set_amount_on("noise", INDEX["BD"], -0.11)

    back = from_dict(to_dict(runner.preset))
    wire = [i for i in back.impulses if i.track == "BD" and i.dial == "noise"]
    assert len(wire) == 1 and wire[0].amount == -0.11, back.impulses


def test_wiring_a_drum_by_hand_does_not_edit_the_setting_it_came_from():
    """A `Preset` outlives the runner holding it -- the library keeps every one it loaded.
    Editing it in place would mean tabbing away and back did not undo a hand-made rule, with
    nothing to say why."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    before = list(FIXTURES["voices"].impulses)
    runner = PresetRunner(FIXTURES["voices"], INDEX, 60.0)
    runner.route("BD", "se_64")
    assert FIXTURES["voices"].impulses == before, "the loaded setting must be untouched"
    assert runner.preset.impulses != before

    runner.load(FIXTURES["voices"])
    assert "se_64" not in runner.routing()


def test_the_default_push_points_away_from_where_the_dial_is_parked():
    """A dial parked near the top of its travel has nowhere to go upward, so a rule that pushes
    it up does nothing visible from where it already is."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["release"], INDEX, 60.0)
    assert runner._base["hold"] > 0.6, "the fixture needs a dial parked high"
    runner.route("CP", "hold")
    pushed = [i for i in runner.preset.impulses if i.track == "CP" and i.dial == "hold"]
    assert pushed and pushed[0].amount < 0, "a dial parked high gets pushed down"

    runner.route("CP", "noise")
    up = [i for i in runner.preset.impulses if i.track == "CP" and i.dial == "noise"]
    assert up and up[0].amount > 0, "one parked at the bottom gets pushed up"
    assert pushed[0].attack > 0 and up[0].attack == 0


def test_writes_from_two_threads_do_not_lose_a_source():
    """The first version argued a lost update would last one frame because every writer rewrites
    its own key sixty times a second. No writer that exists does: the console writes only when
    the mouse moves, so a dropped dial would have been dropped for good."""
    import threading

    from ganlive.control.tracks import INDEX
    from ganlive.presets import PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    done = threading.Barrier(3)

    def writer(source, dial):
        done.wait()
        for i in range(400):
            runner.hold(source, {dial: (i % 100) / 100.0})

    threads = [threading.Thread(target=writer, args=(s, d), daemon=True)
               for s, d in (("a", "noise"), ("b", "se_256"))]
    for t in threads:
        t.start()
    done.wait()
    for t in threads:
        t.join(timeout=20)

    assert "noise" in runner.hands and "se_256" in runner.hands, runner.hands


def test_every_shipped_setting_survives_being_written_down_and_read_back():
    from ganlive.presets import from_dict, to_dict

    for name, preset in FIXTURES.items():
        again = from_dict(to_dict(preset))
        assert again == preset, name


def test_a_saved_setting_is_in_the_rotation_the_next_time_it_starts(tmp_path):
    from dataclasses import replace

    from ganlive.control.tracks import INDEX
    from ganlive.presets import Library, PresetRunner

    runner = PresetRunner(FIXTURES["full"], INDEX, 60.0)
    runner.hold("console", {"noise": 0.62, "hold": 0.81})
    runner.route("BD", "se_256")

    library = Library(tmp_path)
    shipped = list(library.names)
    found = replace(runner.preset, dials={**runner.preset.dials, **runner.held_by("console")})
    path = library.save(found)
    assert path.exists()
    assert library.current.name == path.stem
    assert library.names[:len(shipped)] == shipped

    fresh = Library(tmp_path)
    assert not fresh.broken, fresh.broken
    assert path.stem in fresh.names
    kept = fresh.presets[fresh.names.index(path.stem)]
    assert kept.dials["noise"] == 0.62 and kept.dials["hold"] == 0.81
    assert ("BD", "se_256") in [(i.track, i.dial) for i in kept.impulses]


def test_saving_never_overwrites_a_shipped_setting_and_the_second_press_does(tmp_path):
    from dataclasses import replace

    from ganlive.presets import Library

    library = Library(tmp_path)
    first = library.save(FIXTURES["full"])
    assert first.stem != "full", "a shipped setting was edited from the strip"
    again = library.save(replace(library.current, dials={"noise": 0.4}))
    assert again == first, "the second press started a new file instead of keeping the setting"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_a_hand_edited_setting_with_a_typo_says_which_key_rather_than_dropping_it(tmp_path):

    import pytest

    from ganlive.presets import Library, from_dict

    with pytest.raises(KeyError, match="dails"):
        from_dict({"name": "x", "blurb": "y", "dails": {}})
    with pytest.raises(KeyError, match="ammount"):
        from_dict({"name": "x", "blurb": "y",
                   "impulses": [{"track": "BD", "dial": "noise", "ammount": 0.2}]})

    (tmp_path / "wrong.json").write_text(json.dumps({"name": "w", "nope": 1}), encoding="utf-8")
    library = Library(tmp_path)
    assert "wrong.json" in library.broken[0], library.broken
    assert "wrong" not in library.names


def test_the_hand_positions_are_not_read_as_a_setting(tmp_path):
    """`Positions` writes into the folder `Library` scans by extension, so every launch
    reported the hand positions as a setting that would not load -- one line of known noise
    in the one place a genuinely broken setting announces itself."""
    from ganlive.control.tracks import INDEX
    from ganlive.presets import POSITIONS_NAME, Library, Positions, PresetRunner

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold("hand", {"noise": 0.6})
    positions = Positions(tmp_path / POSITIONS_NAME)
    positions.stash("gv-2048-ft-72000", runner, "hand", ["noise"])

    library = Library(tmp_path)
    assert library.broken == [], library.broken
    assert POSITIONS_NAME.removesuffix(".json") not in library.names


def test_a_saved_setting_says_where_it_came_from_rather_than_wearing_another_blurb(tmp_path):
    """The blurb is what the strip prints when you tab onto a setting. Carrying `full`'s
    sentence across onto a setting whose rules have since been rewired by hand is a caption
    describing something else, on the one line that exists to say what this is."""
    from dataclasses import replace

    from ganlive.presets import Library

    library = Library(tmp_path)
    library.save(FIXTURES["full"])
    assert library.current.blurb == "Found at the controls, from full."
    assert library.current.impulses == FIXTURES["full"].impulses

    library.save(replace(library.current, dials={"noise": 0.4}))
    assert library.current.blurb == "Found at the controls, from full."
