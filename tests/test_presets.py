"""A preset: the rules connecting what the drums do to what the picture does."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import numpy as np
import pytest

from ganlive.control.kit import INDEX, channel_map
from ganlive.dials.fastgan_dials import DIALS
from ganlive.presets import (
    AMOUNT_MAX,
    POSITIONS_NAME,
    Impulse,
    Library,
    Macro,
    Positions,
    Preset,
    from_dict,
    to_dict,
)
from tests.support import FIXTURES, FakeSettings, _runner, since


def test_a_hit_cannot_push_a_dial_off_its_scale():
    """Dials are clamped to their range, so an impulse with a wildly wrong amount saturates the
    dial rather than breaking the picture."""
    preset = Preset(name="t", blurb="", dials={"noise": 0.5},
                  impulses=[Impulse("BD", "noise", amount=9.0, decay=0.5, velocity=0.0)])
    runner = _runner(preset)
    runner.apply(since("BD"), {}, FakeSettings())
    assert runner.surface["noise"] == 1.0


def test_the_runner_counts_which_dials_were_played():
    """The end-of-run usage report counts, per dial, frames moved, time held by a holder
    (mouse, knobs, pads) and distance from rest; untouched dials are listed as never moved."""
    preset = Preset(name="t", blurb="",
                  impulses=[Impulse("BD", "noise", amount=0.5, decay=0.5, velocity=0.0)])
    runner = _runner(preset)
    settings, quiet, hit = FakeSettings(), since(), since("BD")
    runner.hold("hand", {"reaction": 0.5})
    runner.apply(quiet, {}, settings)               # first frame: nothing to compare against
    for ages in (hit, hit, quiet):
        runner.apply(ages, {}, settings)
    frames, held, far = runner.usage["noise"]
    assert frames >= 2 and held == 0 and far == pytest.approx(0.5)
    assert runner.usage["reaction"][1] == 4
    report = runner.usage_report()
    assert report[0].startswith("noise")
    assert "in hand" not in report[0]
    assert "never moved" in report[-1] and "speed" in report[-1] and "reaction" in report[-1]


def test_a_hit_on_any_track_reaches_a_star_rule():
    """A `*` rule fires on a hit from any track, so one rule covers the whole kit."""
    preset = Preset(name="t", blurb="",
                  impulses=[Impulse("*", "noise", amount=0.5, decay=0.5, velocity=0.0)])
    runner = _runner(preset)
    settings = FakeSettings()
    runner.apply(since(), {}, settings)
    assert runner.surface["noise"] == pytest.approx(0.0)
    runner.observe([(INDEX["CH"], 0.5, 0.0)])
    runner.apply(since("CH"), {}, settings)              # a hat, which no rule names
    assert runner.surface["noise"] > 0.4


def test_a_whole_kit_rule_decays_on_audio_time_like_every_other_rule():
    """A `*` rule decays on the audio thread's time since the hit, like named rules, not by
    counting frames; so when frames run late, the decay still keeps real time."""
    preset = Preset(name="t", blurb="",
                  impulses=[Impulse("*", "noise", amount=0.5, decay=0.2, velocity=0.0)])
    seen = {}
    for real_dt in (1 / 60, 1 / 30):                       # on time, then every frame late
        runner = _runner(preset)
        ages = since("BD")
        runner.observe([(INDEX["BD"], 0.5, 0.0)])
        for _ in range(6):
            runner.apply(ages, {}, FakeSettings())
            ages = [s + real_dt for s in ages]             # the audio thread's own clock
        seen[real_dt] = runner.surface["noise"]

    assert seen[1 / 30] < seen[1 / 60] * 0.75, seen


def test_a_hand_on_a_dial_sets_where_the_drums_push_from():
    """A held value replaces the dial's resting value, and hits still push from there: moving a
    knob shifts where the reaction starts instead of cancelling it."""
    preset = Preset(name="t", blurb="", dials={"noise": 0.1},
                  impulses=[Impulse("BD", "noise", amount=0.2, decay=0.5, velocity=0.0)])
    runner = _runner(preset)
    settings, hit = FakeSettings(), since("BD")

    runner.apply(hit, {}, settings)
    without = runner.surface["noise"]
    runner.hold("test", {**runner.held_by("test"), "noise": 0.6})
    runner.apply(hit, {}, settings)
    assert runner.surface["noise"] == pytest.approx(without + 0.5, abs=1e-6)


def test_reaction_scales_how_hard_hits_land_without_touching_the_arrangement():
    """`reaction` scales hit impulses (0 = none, 1 = double) and leaves slow macro rules alone."""
    preset = Preset(
        name="t", blurb="",
        impulses=[Impulse("BD", "noise", amount=0.4, decay=0.5, velocity=0.0)],
        macros=[Macro("density", "dir1", 0.0, 10.0, 0.2, 0.8, glide=0.0)])
    seen = {}
    for reaction in (0.0, 0.5, 1.0):
        runner = _runner(preset)
        runner.hold("test", {**runner.held_by("test"), "reaction": reaction})
        runner.apply(since("BD"), {"density": 10.0}, FakeSettings())
        seen[reaction] = (runner.surface["noise"], runner.surface["dir1"])

    hit_at_rest = seen[0.5][0]
    assert seen[0.0][0] == pytest.approx(0.0), "at 0 the hits must not land at all"
    assert seen[1.0][0] == pytest.approx(hit_at_rest * 2, abs=1e-6), "1 is twice as hard"
    assert seen[0.0][1] == seen[0.5][1] == seen[1.0][1] == pytest.approx(0.8)


def test_at_zero_reaction_any_setting_behaves_like_the_structural_one():
    """At zero reaction the drumming stops reaching the picture: with every track hitting, the
    texture dials of any preset stay at their resting values."""
    everything = since(*INDEX)
    out = {}
    for name in ("full", "voices"):
        runner = _runner(name)
        runner.hold("test", {**runner.held_by("test"), "reaction": 0.0})
        runner.apply(everything, {"density": 6.0}, FakeSettings())
        out[name] = dict(runner.surface.values)
    for name, values in out.items():
        for dial in ("se_512", "se_256", "noise"):
            assert values[dial] == pytest.approx(DIALS[dial][0]), (name, dial)


def test_a_hand_beats_the_slow_rule_for_that_dial_and_only_that_dial():
    """A held value overrides a slow rule (which sets its dial every frame) on that dial only;
    otherwise the macro would overwrite the slider at once."""
    preset = Preset(name="t", blurb="",
                  macros=[Macro("density", "spread", 0.0, 10.0, 0.1, 0.9, glide=0.0),
                          Macro("density", "dir1", 0.0, 10.0, 0.1, 0.9, glide=0.0)],
                  impulses=[Impulse("BD", "spread", amount=0.2, decay=0.5, velocity=0.0)])
    runner = _runner(preset)
    runner.apply(since(), {"density": 10.0}, FakeSettings())
    assert runner.surface["spread"] == pytest.approx(0.9), "the rule drives it when free"

    runner.hold("test", {"spread": 0.25})
    runner.apply(since(), {"density": 10.0}, FakeSettings())
    assert runner.surface["spread"] == pytest.approx(0.25), "the hand must win"
    assert runner.surface["dir1"] == pytest.approx(0.9), "and only over its own dial"

    runner.apply(since("BD"), {"density": 10.0}, FakeSettings())
    assert runner.surface["spread"] > 0.25, "a hit still pushes up from where the hand set it"


def test_letting_go_returns_the_dial_to_where_the_slow_rule_has_reached():
    """A slow rule keeps gliding while its dial is held, so on release the dial jumps to where
    the rule is now rather than gliding from where it was when the hold began."""
    preset = Preset(name="t", blurb="",
                  macros=[Macro("density", "dir1", 0.0, 10.0, 0.0, 1.0, glide=0.2)])
    runner = _runner(preset)
    runner.hold("test", {"dir1": 0.1})
    for _ in range(120):                                  # two seconds held
        runner.apply(since(), {"density": 10.0}, FakeSettings())
    assert runner.surface["dir1"] == pytest.approx(0.1)
    runner.hold("test", {})
    runner.apply(since(), {"density": 10.0}, FakeSettings())
    assert runner.surface["dir1"] > 0.9, "the rule should have gone on running underneath"


def test_switching_patch_keeps_the_objects_the_loop_and_the_walk_hold():
    """Loading a preset mid-run updates the runner and its `walk_cfg` in place, since the loop
    and the walk hold references to them; held dials survive the change."""
    runner = _runner()
    cfg = runner.walk_cfg
    runner.hold("test", {"noise": 0.5})
    runner.load(FIXTURES["release"])

    assert runner.walk_cfg is cfg, "the walk would stop seeing every motion dial"
    assert runner.preset.name == "release"
    assert runner.hands == {"noise": 0.5}, "a hand on a control survives a change of setting"
    runner.apply(since(), {"density": 0.0}, FakeSettings())
    assert runner.surface["hold"] == pytest.approx(FIXTURES["release"].dials["hold"])


def test_a_setting_naming_a_dial_that_no_longer_exists_says_so():
    """Preset entries naming a dial that does not exist are dropped and listed in
    `runner.dropped`, while rules on real dials are kept."""
    old = Preset(name="old", blurb="",
                dials={"warmth": 0.8, "hold": 0.4},
                impulses=[Impulse("BD", "grit", amount=0.3, decay=0.2),
                          Impulse("SD", "noise", amount=0.2, decay=0.2)],
                macros=[Macro("density", "tint", 2.5, 9.5, 0.1, 0.6, glide=1.0)])
    runner = _runner(old)

    assert "warmth (no such dial)" in runner.dropped
    assert "BD->grit (no such dial)" in runner.dropped
    assert any("tint" in line for line in runner.dropped), runner.dropped
    assert len(runner._impulses) == 1, "the rule that still names a real dial must survive"
    assert not runner._macros, "a macro on a dead dial is dropped, not silently applied"


def test_two_sources_can_hold_different_dials_without_dropping_each_other():
    """Each holder (mouse, knobs, pads) holds dials under its own source name; the latest hold
    wins a shared dial, and freeing one source leaves the others' holds in place."""

    runner = _runner()
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
    """The render loop iterates `hands` while other threads write it, so each write builds a new
    dict and rebinds it; editing in place could raise `RuntimeError` mid-iteration."""

    runner = _runner()
    runner.hold("a", {"noise": 0.5})
    before = runner.hands
    runner.hold("b", {"se_256": 0.5})
    assert runner.hands is not before, "the loop's dict must never be edited under it"
    assert before == {"noise": 0.5}, "and the one it already read must not change"


def test_a_rule_wired_past_the_end_of_the_kit_is_reported_not_silently_skipped():
    """A rule on a channel the input does not have (e.g. CY on an eight-channel input) is
    dropped and reported when the preset loads."""
    full = _runner(FIXTURES["voices"], channels=12)
    assert not full.dropped

    narrow = _runner(FIXTURES["voices"], channels=8)
    assert narrow.dropped, "CY sits on channel 10 and this kit has eight"
    assert any("CY" in line for line in narrow.dropped), narrow.dropped
    assert all("CY" not in imp.track for imp, _ch in narrow._impulses)
    narrow.apply(np.full(8, 1e6, dtype=np.float32), {"density": 4.0},
                 FakeSettings())


def test_a_drum_can_be_wired_to_a_dial_while_it_runs():
    """`wire` toggles a drum-to-dial rule at runtime: the first call wires it and the hit then
    moves the dial, the second call unwires it."""
    runner = _runner()
    assert runner.routing() == {}

    assert runner.wire(["BD"], "noise") is True
    assert [(i.track, ch) for i, ch in runner.routing()["noise"]] == [("BD", INDEX["BD"])]
    runner.apply(since("BD"), {}, FakeSettings())
    assert runner.surface["noise"] > DIALS["noise"][0]

    assert runner.wire(["BD"], "noise") is False, "the same call unwires it"
    assert "noise" not in runner.routing()


def test_how_hard_a_drum_pushes_a_dial_can_be_set_without_rewiring_it():
    """A wired rule's amount can be changed (and is clamped to `AMOUNT_MAX`) without rewiring;
    a larger amount pushes further, and a negative one pushes down."""
    bd = INDEX["BD"]
    runner = _runner()
    assert runner.amount_on("noise", bd) is None, "nothing is wired yet"
    assert runner.set_amount_on("noise", bd, 0.5) is None, "an unwired cell is not created"

    runner.wire(["BD"], "noise")
    assert runner.amount_on("noise", bd) == 0.3

    assert runner.set_amount_on("noise", bd, 0.08) == 0.08
    assert [(i.track, i.amount) for i, _ch in runner.routing()["noise"]] == [("BD", 0.08)]

    def push(dial, amount):
        r = _runner()
        r.wire(["BD"], dial)
        r.set_amount_on(dial, INDEX["BD"], amount)
        r.apply(since("BD"), {}, FakeSettings())
        return r.surface[dial] - DIALS[dial][0]

    gentle, hard = push("noise", 0.08), push("noise", 0.8)
    assert 0 < gentle < hard, (gentle, hard)

    assert DIALS["se_128"][0] == 0.5, "this test needs a dial that rests mid-travel"
    assert push("se_128", -0.25) < 0 < push("se_128", 0.25)

    assert runner.set_amount_on("noise", bd, 87.0) == AMOUNT_MAX
    assert runner.set_amount_on("noise", bd, -87.0) == -AMOUNT_MAX
    assert Impulse("BD", "noise", amount=9.0).amount == AMOUNT_MAX


def test_the_strength_that_is_drawn_and_the_strength_that_is_edited_are_one_wire():
    """A routing cell is a channel, which two drums share under the voices layout. Setting its
    strength sets every rule on the channel, so the value shown is the value edited."""
    voices = channel_map("voices")
    assert voices["RS"] == voices["CP"], "this test needs two drums on one channel"
    channel = voices["RS"]

    runner = _runner("still", channel_of=voices)
    runner.wire(["RS"], "se_512")
    runner.wire(["CP"], "se_512")
    runner.set_amount_on("se_512", channel, 0.12)
    runner.set_amount_on("se_512", channel, 0.44)

    assert runner.amount_on("se_512", channel) == 0.44
    assert sorted(i.amount for i, _ch in runner.routing()["se_512"]) == [0.44, 0.44]


def test_a_click_in_the_grid_wires_the_whole_column_and_a_second_click_clears_it():
    """A grid column is a channel, not a drum: clicking a cell shared by RS and CP wires both,
    and clicking a lit cell clears every rule on it, whichever drum lit it."""
    voices = channel_map("voices")
    channel = voices["RS"]
    assert voices["CP"] == channel, "this test needs two drums on one channel"

    runner = _runner("still", channel_of=voices)
    runner.wire(["CP"], "se_512", amount=0.5)          # a saved setting wired the second of them

    assert runner.wire(["RS", "CP"], "se_512") is False, "the cell is lit, so a click clears it"
    assert runner.amount_on("se_512", channel) is None, "and the light goes out"

    assert runner.wire(["RS", "CP"], "se_512") is True
    assert sorted(i.track for i, _ch in runner.routing()["se_512"]) == ["CP", "RS"], \
        "both drums on the channel fire the cell the grid drew"


def test_changing_a_strength_does_not_restart_the_slow_rules():
    """Adjusting a wire's strength edits it in place without reloading, so the slow rules'
    smoothed state (which `load` resets) is untouched by a scroll-wheel drag."""
    runner = _runner("full")
    quiet = since()
    runner.apply(quiet, {"density": 0.8, "energy": 0.5, "active": 0.4},
                 FakeSettings())
    glide = dict(runner._macro_state)
    assert glide, "this test needs a preset whose slow rules have started smoothing"

    runner.wire(["BD"], "noise")                      # creating a wire DOES reload, and resets
    runner.apply(quiet, {"density": 0.8, "energy": 0.5, "active": 0.4},
                 FakeSettings())
    glide = dict(runner._macro_state)

    runner.set_amount_on("noise", INDEX["BD"], 0.5)
    assert runner._macro_state == glide, "adjusting a strength must not restart the glide"

    assert runner.amount_on("noise", INDEX["BD"]) == 0.5


def test_a_strength_set_by_hand_survives_being_saved_and_read_back():
    """A strength set at runtime round-trips through the preset's JSON form, rather than reverting
    to the default push on the next load."""
    runner = _runner()
    runner.wire(["BD"], "noise")
    runner.set_amount_on("noise", INDEX["BD"], -0.11)

    back = from_dict(to_dict(runner.preset))
    wire = [i for i in back.impulses if i.track == "BD" and i.dial == "noise"]
    assert len(wire) == 1 and wire[0].amount == -0.11, back.impulses


def test_wiring_a_drum_by_hand_does_not_edit_the_setting_it_came_from():
    """Wiring edits the runner's copy, not the library's `Preset`, so reloading the preset
    restores its original rules."""

    before = list(FIXTURES["voices"].impulses)
    runner = _runner("voices")
    runner.wire(["BD"], "se_64")
    assert FIXTURES["voices"].impulses == before, "the loaded setting must be untouched"
    assert runner.preset.impulses != before

    runner.load(FIXTURES["voices"])
    assert "se_64" not in runner.routing()


def test_the_default_push_points_away_from_where_the_dial_is_parked():
    """A new wire pushes a dial parked high downward and one parked low upward, so the push is
    always visible."""

    runner = _runner("release")
    assert runner._base["hold"] > 0.6, "the fixture needs a dial parked high"
    runner.wire(["CP"], "hold")
    pushed = [i for i in runner.preset.impulses if i.track == "CP" and i.dial == "hold"]
    assert pushed and pushed[0].amount < 0, "a dial parked high gets pushed down"

    runner.wire(["CP"], "noise")
    up = [i for i in runner.preset.impulses if i.track == "CP" and i.dial == "noise"]
    assert up and up[0].amount > 0, "one parked at the bottom gets pushed up"
    assert pushed[0].attack > 0 and up[0].attack == 0


def test_writes_from_two_threads_do_not_lose_a_source():
    """Concurrent holds from two threads must not lose either source; writers only write on
    change (e.g. when the mouse moves), so a lost update would stay lost."""


    runner = _runner()
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
    for name, preset in FIXTURES.items():
        again = from_dict(to_dict(preset))
        assert again == preset, name


def test_a_saved_setting_is_in_the_rotation_the_next_time_it_starts(tmp_path):
    runner = _runner("full")
    runner.hold("console", {"noise": 0.62, "hold": 0.81})
    runner.wire(["BD"], "se_256")

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
    library = Library(tmp_path)
    first = library.save(FIXTURES["full"])
    assert first.stem != "full", "a shipped setting was edited from the strip"
    again = library.save(replace(library.current, dials={"noise": 0.4}))
    assert again == first, "the second press started a new file instead of keeping the setting"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_a_hand_edited_setting_with_a_typo_says_which_key_rather_than_dropping_it(tmp_path):

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
    """The saved held-dial positions file lives in the preset folder, and `Library` must not
    list it as a preset or report it as broken."""
    runner = _runner()
    runner.hold("hand", {"noise": 0.6})
    positions = Positions(tmp_path / POSITIONS_NAME)
    positions.stash("gv-2048-ft-72000", runner, "hand", ["noise"])

    library = Library(tmp_path)
    assert library.broken == [], library.broken
    assert POSITIONS_NAME.removesuffix(".json") not in library.names


def test_a_saved_setting_says_where_it_came_from_rather_than_wearing_another_blurb(tmp_path):
    """A saved preset's blurb (shown on the strip) names the preset it was derived from instead
    of copying that preset's description."""
    library = Library(tmp_path)
    library.save(FIXTURES["full"])
    assert library.current.blurb == "Found at the controls, from full."
    assert library.current.impulses == FIXTURES["full"].impulses

    library.save(replace(library.current, dials={"noise": 0.4}))
    assert library.current.blurb == "Found at the controls, from full."


def test_a_hand_written_preset_needs_neither_name_nor_blurb(tmp_path):
    """The library names a preset after its file, so a file that leaves both out still loads."""
    import json

    from ganlive.presets import Library

    (tmp_path / "groove.json").write_text(json.dumps({"dials": {"speed": 0.5}}), encoding="utf-8")
    library = Library(tmp_path)
    assert not library.broken, library.broken
    assert library.select("groove").dials == {"speed": 0.5}
