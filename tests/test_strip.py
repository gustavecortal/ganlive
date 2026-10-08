"""The strip of sliders beside the picture, and the window holding both."""

from __future__ import annotations

import dataclasses
import inspect
import math
import threading

import numpy as np
import pygame
import pytest
import torch

from ganlive import strip as console
from ganlive import window
from ganlive.clock import WalkConfig, position
from ganlive.control.kit import INDEX, channel_map
from ganlive.control.midi import EncoderMap, PressureMap, parse_controls, parse_pressure
from ganlive.dials import fastgan_dials as _fastgan
from ganlive.dials import table as _surface
from ganlive.dials.fastgan_dials import DIALS, fastgan
from ganlive.dials.table import adopted, per_model
from ganlive.pixels import to_bgra, to_rgb
from ganlive.presets import DEFAULT, Positions
from ganlive.strip import (
    DESC_H,
    MODEL_ROW,
    PAD,
    PRIORITY,
    SANS,
    SOURCE,
    STATUS_H,
    TEXT_CACHE,
    WIDTH,
    DialPanel,
    blocks,
    driven_by,
    floor_height,
    grid_cell,
    grid_columns,
    layout,
    lights,
    model_at,
    model_rows,
    rows,
    scope_curve,
    track_span,
    wrap,
)
from ganlive.tools import play as live
from ganlive.walk import SlerpWalk
from tests import fuzz_surface
from tests.support import (
    FIXTURES,
    OVERBRIDGE,
    FakeSettings,
    _panel,
    _runner,
    _StubModel,
    dummy_display,
    headless_renderer,
    since,
    stub_bank,
)

#: The most crowded surface here -- this project's own FastGAN -- as the geometry these
#: layout tests measure against.
GROUPS = fastgan().groups
MIN_H = floor_height(GROUPS)


def _travelled(walk, beats):
    """Where along its arc the walk really is at `beats`, recovered from the latent it returns."""
    k = position(walk.cfg, beats)[0]
    z0 = walk.seed_for(k).astype(np.float64)
    z1 = walk.seed_for(k + 1).astype(np.float64)
    got = walk.latent(beats).reshape(-1).astype(np.float64)
    a, b = np.linalg.lstsq(np.stack([z0, z1], axis=1), got, rcond=None)[0]
    n0, n1 = z0 / np.linalg.norm(z0), z1 / np.linalg.norm(z1)
    omega = float(np.arccos(np.clip(float(n0 @ n1), -1.0, 1.0)))
    return float(np.arctan2(b * math.sin(omega), a + b * math.cos(omega)) / omega)


def test_every_dial_gets_a_row_and_no_row_gets_two():
    """The strip is built from `surface`, never from a second list of names. A dial added
    there has to appear here without anyone remembering to add it -- the alternative is a
    control that exists, does something, and cannot be reached."""
    named = [name for _title, names in fastgan().groups for name in names]
    assert [label for label, _y, _h in rows(900, GROUPS)] == named


def test_the_rows_stay_clickable_on_a_short_window_and_stop_sprawling_on_a_tall_one():
    """A row too short to hit is a control that cannot be turned, and one that grows without
    limit is twenty-one sliders spread down a 1440-pixel screen."""

    for height in (420, 700, 1080, 2160):
        found = [(y, h) for _l, y, h in rows(height, GROUPS)]
        assert all(18 <= h <= 34 for _y, h in found), height
        assert all(b >= a + h for (a, h), (b, _) in zip(found, found[1:], strict=False))


def test_a_click_lands_on_the_dial_it_looks_like_it_lands_on():
    """Hit-testing and drawing read the same track geometry. Two definitions would let the
    pointer disagree with the picture by a few pixels for ever."""

    w, h = 372, 900
    panel = DialPanel(_runner())
    panel._size, panel._rows = (w, h), rows(h, GROUPS)
    left, span = track_span(w)
    for kind, label, top, tall in layout(h, GROUPS):
        if kind != "dial":
            continue
        y = top + tall // 2
        assert panel._hit(left, y) == (label, pytest.approx(0.0))
        assert panel._hit(left + span, y) == (label, pytest.approx(1.0))
        assert panel._hit(left + span // 2, y)[1] == pytest.approx(0.5, abs=0.01)
        assert panel._hit(2, y) == (label, pytest.approx(0.0))
    heads = [(y, hgt) for kind, _l, y, hgt in layout(h, GROUPS) if kind == "head"]
    assert panel._hit(left, heads[0][0] + 2) is None


def test_the_panel_holds_what_it_is_set_to_until_it_lets_go():
    runner = _runner()
    panel = DialPanel(runner)
    panel.set("noise", 0.4)
    assert runner.hands == {"noise": 0.4}
    panel.set("se_64", 2.0)                       # clamped at the dial, like everything else
    assert runner.hands == {"noise": 0.4, "se_64": 1.0}

    panel.release("noise")
    assert runner.hands == {"se_64": 1.0}
    panel.release()
    assert runner.hands == {}


def test_the_panel_reports_only_what_is_off_its_resting_value():
    """`p` prints something to paste into a preset, and a preset that restated all fifteen
    resting values would say nothing about what was actually found."""
    runner = _runner()
    panel = DialPanel(runner)
    panel.set("hold", 0.72)
    runner.apply(since(), {}, FakeSettings())
    assert panel.held() == {"hold": 0.72}


def test_a_shared_voice_gets_one_light_and_says_so():
    """Under the shared-voice layout four pairs of drums arrive on one channel and genuinely
    cannot be told apart. Twelve lights would show a separation the hardware does not have."""
    per_track = DialPanel(_runner())
    assert ["/".join(names) for _ch, names in per_track.kit] == list(INDEX)

    shared = DialPanel(_runner(channel_of=channel_map("voices")))
    labels = ["/".join(names) for _ch, names in shared.kit]
    assert len(labels) == 8, labels
    assert "CH/OH" in labels and "MT/HT" in labels
    channels = [ch for ch, _names in shared.kit]
    assert channels == sorted(set(channels)), "one light per channel, in the machine's order"


def test_the_lights_fit_the_strip_whatever_the_kit_is():
    """A layout discovered with --meter is neither of the two built-in ones, so the row has to
    size itself rather than assume twelve."""

    assert lights(372, 900, 0) == []
    for count in (1, 5, 8, 12, 16):
        boxes = lights(372, 900, count)
        assert len(boxes) == count
        assert all(x >= PAD for x, _y, _w, _h in boxes)
        assert all(x + w <= 372 - PAD + 2 for x, _y, w, _h in boxes), count
        assert all(y >= 900 - STATUS_H for _x, y, _w, _h in boxes)
        assert all(b >= a + w for (a, _, w, _hh), (b, *_) in zip(boxes, boxes[1:],
                                                                strict=False))


def test_what_p_prints_is_what_you_set_not_what_the_patch_already_rested_at():
    """`p` writes down a discovery: what a hand set, not the preset's own resting values and
    not a decaying hit sampled from the live values."""
    runner = _runner("release")
    panel = DialPanel(runner)
    runner.apply(since(), {"density": 5.0}, FakeSettings())
    assert panel.held() == {}, "a preset's own resting values are not discoveries"

    panel.set("noise", 0.61)
    ringing = since()
    ringing[INDEX["BD"]] = 0.01                           # a kick still ringing on `hold`
    runner.apply(ringing, {"density": 5.0}, FakeSettings())
    assert runner.surface["hold"] != pytest.approx(FIXTURES["release"].dials["hold"]), (
        "the fixture needs the hit to actually be moving something")
    assert panel.held() == {"noise": 0.61}


def test_dragging_a_slider_does_not_repaint_the_strip():
    """A 1000 Hz mouse delivers about sixteen moves a frame. What a drag has to show is the bar
    and the marker, and both are rectangles drawn every frame, so a drag must not repaint the
    text texture."""

    panel = DialPanel(_runner())
    panel.set("spread", 0.2)                               # a grab: the held text changes
    assert panel._dirty
    panel._dirty = False
    for i in range(20):                                   # the drag itself
        panel.set("spread", 0.2 + i * 0.01)
    assert not panel._dirty, "a drag must not repaint"
    panel.release("spread")
    assert panel._dirty, "letting go changes the held colour and the count"


def test_a_dial_the_model_does_not_have_is_drawn_dark_and_cannot_be_grabbed():
    """An inert control that looks live is the failure the whole load path exists to prevent."""

    panel = _panel({"reaction", "speed", "spread", "hold", "late", "grid", "dir1", "dir2"})
    panel._size = (400, 900)
    panel._rows = rows(900, GROUPS)

    assert {"dir1", "dir2"} <= panel.live_dials()
    assert "dir3" not in panel.live_dials(), "a direction the model does not have"
    assert "se_256" not in panel.live_dials(), "nor a gate this architecture does not have"

    live_row = next(top for name, top, _t in panel._rows if name == "dir1")
    dead_row = next(top for name, top, _t in panel._rows if name == "dir3")
    assert panel._hit(200, live_row + 4) is not None
    assert panel._hit(200, dead_row + 4) is None, "dragging a dead dial must move nothing"


def test_the_routing_grid_will_not_wire_a_drum_to_a_dial_that_cannot_fire():
    """`route` raises on a dead dial, and the grid runs on the window's thread."""
    panel = _panel({"dir1", "speed"})
    panel._pg = pygame
    panel._columns = [(0, 20)]
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0))

    panel._grid_gesture(ev, "se_256", 0)               # must not raise
    assert not panel.runner.routing(), "and must not have wired anything"

    panel._grid_gesture(ev, "speed", 0)
    assert "speed" in panel.runner.routing(), "a live dial still wires"


def test_the_strip_lays_out_the_loaded_model_s_dials_and_not_the_departed_one_s():
    """The rows follow the layout, not only the window size: two models can have the same
    number of rows, and a switch between them must still lay the strip out again.

    Driven through the real `draw` under SDL's dummy driver, because the decision is there."""

    ours = _fastgan.fastgan()
    names = ("w_coarse", "w_mid", "w_fine", "noise_4", "noise_512")
    other = _surface.stylegan2(names, (0.5,) * 5, ((0.1, 1.0, 2.0),) * 5, (25.0,) * 5, ())
    assert len(ours.knobs) == len(other.knobs), (
        "this test needs two layouts the window cannot tell apart by height")

    runner = _runner()
    current = _StubModel(layout=ours, dials_live=frozenset(ours))
    holder = stub_bank(current)
    runner.use_model(current)

    with headless_renderer((WIDTH + 200, 900), "rows") as renderer:
        panel = DialPanel(runner, bank=holder)
        panel.attach(renderer)
        strip = (0, 0, WIDTH, 900)

        def frame():
            """One, exactly as `Display` draws it -- clear, draw, present."""
            renderer.draw_color = (0, 0, 0, 255)
            renderer.clear()
            panel.draw(renderer, strip)
            renderer.present()

        frame()
        assert {n for n, _t, _h in panel._rows} == set(ours), "the model it opened on"

        # `play.switch_model`'s two statements, in its order.
        holder.current = _StubModel(layout=other, dials_live=frozenset(other))
        frame()                                  # the window's thread, in the gap between them
        assert {n for n, _t, _h in panel._rows} == set(ours), (
            "in the gap the strip draws the outgoing model whole, not its values under the "
            "incoming one's layout")
        runner.use_model(holder.current)
        frame()

    assert {n for n, _t, _h in panel._rows} == set(other), (
        "the rows must follow the model even when the window has no reason to move")
    for name in names:
        assert name in panel.dials, "and the incoming model's own dials must be reachable"


def test_a_direction_says_what_it_measured_on_this_model():
    """`rank` measures every direction at load and orders them by what they do. That number
    is what separates a leading direction from a tail one, so the description shows it."""
    panel = _panel(DIALS, levels=(88.0, 41.0))

    assert "88 8-bit levels" in panel._measured("dir1")
    assert "ranked 1 of 2" in panel._measured("dir1")
    assert "41 8-bit levels" in panel._measured("dir2")
    assert panel._measured("dir3") == "", "a direction this model does not have says nothing"
    assert panel._measured("spread") == "", "and nothing else claims a measurement"


def test_with_no_bank_every_dial_is_live():
    """Offline tools build a panel without a bank, and nothing there says any dial is dark."""

    panel = DialPanel(_runner())
    assert panel.live_dials() is None, "nothing has said, so nothing is dark"


def test_the_pickers_rows_are_geometry_not_a_cache_left_by_the_last_repaint():
    """Drawing and hit-testing must not be able to disagree about which row a click is on."""

    floor = PAD + 10 * MODEL_ROW

    fits, scrolling = model_rows(floor, 6)
    assert (fits, scrolling) == (10, False)

    fits, scrolling = model_rows(floor, 40)
    assert (fits, scrolling) == (9, True)

    assert model_at(PAD, floor, 6) == 0
    assert model_at(PAD + MODEL_ROW, floor, 6) == 1
    assert model_at(PAD + MODEL_ROW * 9 + 3, floor, 6) == 9
    assert model_at(PAD - 1, floor, 6) is None
    assert model_at(PAD + MODEL_ROW * 10, floor, 6) is None
    assert model_at(PAD + MODEL_ROW * 9, floor, 40) is None


def test_the_scope_draws_the_curve_the_walk_actually_travels():
    """The motion dials change nothing about a single frame, so the scope is where they show.
    It reads the walk's own shaping, so it cannot drift from what the walk does."""
    cfg = WalkConfig(beats_per_segment=4.0, hold=0.55, when=0.8, step_grid=0, base_seed=3)
    walk = SlerpWalk(64, cfg)
    drawn = scope_curve(cfg, samples=33)

    for i in range(1, 32):
        beats = (i / 32) * cfg.beats_per_segment
        assert _travelled(walk, beats) == pytest.approx(position(cfg, beats)[2], abs=2e-3), beats
        assert drawn[i] == pytest.approx(position(cfg, beats * 0.9999)[2], abs=2e-3)

    stepped = WalkConfig(beats_per_segment=4.0, step_grid=4, base_seed=3)
    walk = SlerpWalk(64, stepped)
    assert _travelled(walk, 0.4) == pytest.approx(_travelled(walk, 0.9), abs=2e-3)
    assert _travelled(walk, 0.4) != pytest.approx(_travelled(walk, 1.4), abs=2e-3)
    assert len(set(round(t, 6) for t in scope_curve(stepped, samples=40))) <= 5


def test_the_standstill_and_where_it_sits_are_both_visible_on_the_scope():
    """`hold` is the flat part of the curve and `late` is where the flat part sits. If the
    scope cannot separate those two it is not showing what the dials do."""

    early = scope_curve(WalkConfig(hold=0.6, when=0.0), samples=41)
    late = scope_curve(WalkConfig(hold=0.6, when=1.0), samples=41)

    def flat_at(curve, first_half):
        half = curve[:len(curve) // 2] if first_half else curve[len(curve) // 2:]
        return max(half) - min(half) < 1e-6

    assert flat_at(late, first_half=True), "late means it waits, so the start is the standstill"
    assert flat_at(early, first_half=False), "early means it moves first and then holds"
    assert not flat_at(late, first_half=False)
    assert not flat_at(early, first_half=True)


def test_the_strip_says_which_drum_drives_which_dial_from_the_patch_itself():
    """Watching a dial move says nothing about who moved it, and the two together are the whole
    point. Read off the preset so it cannot go stale when a rule changes."""
    def wires(entry):
        return [(imp.track, ch, imp.amount) for imp, ch in entry[0]]

    got = driven_by(_runner("voices"))
    assert wires(got["se_128"]) == [("BD", INDEX["BD"], 0.34)]
    assert wires(got["se_512"]) == [("CP", INDEX["CP"], 0.32), ("SD", INDEX["SD"], 0.24)]
    assert "reaction" not in got, "nothing drives the master in this preset"

    slow = driven_by(_runner("breathe"))
    assert slow["spread"] == ([], ["density"]), "a slow rule is not a hit"

    both = driven_by(_runner("full"))
    assert wires(both["hold"]) == [("BD", INDEX["BD"], -0.34)], "a pull keeps its sign"
    assert both["hold"][1] == ["density"]

    kit = driven_by(_runner("pulse"))
    assert wires(kit["dir1"]) == [("*", -1, 0.22)], kit

    narrow = driven_by(_runner("voices", channels=8))
    assert all(ch < 8 for wired, _slow in narrow.values() for _imp, ch in wired), narrow
    assert "noise" not in narrow, narrow
    assert "se_128" in narrow, "the kick is still inside the channel range"


def test_the_window_actually_opens_with_the_strip_beside_it():
    """A real window opens, takes a frame, and its thread ends when it is closed."""

    threads = threading.active_count()
    with dummy_display():

        panel = _panel(DIALS, levels=(100.0, 50.0))
        display = window.Display((64, 96), title="test", overlay=panel,
                                         fullscreen=False)
        display.publish(np.zeros((64, 96, 4), dtype=np.uint8))
        display.close()
    assert threading.active_count() <= threads + 1, "the window thread outlived the window"


def test_the_window_can_draw_on_the_callers_own_thread():
    """macOS keeps a window and its events on the main thread, so there the window draws each
    frame as it is published; a frame it cannot draw stops the session as the thread does."""
    with dummy_display():
        display = window.Display((64, 96), title="inline", overlay=_panel(DIALS),
                                 fullscreen=False, threaded=False)
        assert display.wants, "an inline window is always ready for a frame"
        display.publish(np.zeros((64, 96, 4), dtype=np.uint8))
        assert not display.stopped
        display.publish(np.zeros((3, 5, 1), dtype=np.uint8))      # no picture at all
        assert display.stopped, "a frame that cannot be drawn must stop the session"
        display.close()


def test_every_routing_column_label_fits_the_column_it_names():
    """The columns share what is left of the strip after the dial names, and a kit that puts two
    drums on one channel labels them `MT/HT`. Centring a label wider than its column pushes it
    into the next one: `MT/HT CH/OH CY/CB` rendered as a single run of characters."""

    with headless_renderer((WIDTH, 900), "fit") as renderer:
        panel = DialPanel(_runner(DEFAULT, channel_of=channel_map("voices")))
        panel.attach(renderer)
        columns = grid_columns(WIDTH, len(panel.kit))
        labels = ["/".join(names) for _channel, names in panel.kit]
        too_wide = [(label, panel._micro.size(label)[0], cw)
                    for label, (_cx, cw) in zip(labels, columns, strict=True)
                    if panel._micro.size(label)[0] > cw]

    assert not too_wide, f"label, width, column: {too_wide}"


def test_a_dial_the_surface_does_not_carry_is_drawn_dark_rather_than_raised():
    """The strip draws `bank.current.layout` and reads values off `runner.surface`, which only
    agree after `PresetRunner.use_model`. A caller that skips it gets dark rows, not a
    `KeyError` on the window's thread; and a window thread that does die stops the session."""

    # A bank holding a StyleGAN2's dials behind a runner still holding the FastGAN's -- exactly
    # the state a caller that skips `use_model` is in.
    names = ("w_coarse", "w_mid", "w_fine")
    stub = _StubModel(layout=_surface.stylegan2(names, (0.5,) * 3, ((),) * 3, (25.0,) * 3),
                      dials_live=frozenset({"w_coarse"}))
    runner = _runner()
    foreign = set(stub.layout.rests) - set(runner.surface.values)
    assert foreign, "this test needs the two layouts to actually disagree"

    with headless_renderer((WIDTH, 900), "dark") as renderer:
        panel = DialPanel(runner, bank=stub_bank(stub))
        panel.attach(renderer)
        panel._resize(WIDTH, 900)
        panel._paint()

    source = inspect.getsource(window.Display._guarded)
    assert "self.stopped = True" in source and "except Exception" in source, (
        "a window thread that dies has to stop the session; a frozen window that still "
        "reports 60 fps is the one failure this whole path exists to prevent")


def test_a_thousand_gestures_across_model_switches_break_nothing():
    """Model switches mid-gesture, carrying the focus, the drag and the mode across, through
    the real paint path. Seeded, so a fault is reproducible with
    `python -m tests.fuzz_surface --seeds 1`."""

    size = (fuzz_surface.WIDTH + 200, max(fuzz_surface.HEIGHTS))
    with headless_renderer(size, "fuzz") as renderer:
        faults, reached = fuzz_surface.run(renderer, 500, seed=0)

    assert not faults, fuzz_surface.report(faults) or [f[5] for f in faults]
    # A fuzzer that never reaches the state is a fuzzer that proves nothing, and this is the
    # state: the strip describing a dial the incoming model does not have.
    assert reached["stale focus"], dict(reached)
    assert reached["into routing"] and reached["into models"], dict(reached)


def test_the_window_opens_no_shorter_than_the_strip_needs():
    """The floor is a property of the window, not of how it happened to open: it is
    resizable, so a floor applied once at construction is not one."""

    panel = _panel(DIALS)
    tall_screen = window.window_size(3072, 2048, 2560, 1440, panel)
    short_screen = window.window_size(3072, 2048, 1280, 400, panel)

    assert tall_screen[0] > WIDTH, "the strip sits beside the picture, not over it"
    assert tall_screen[1] >= MIN_H, tall_screen
    assert short_screen[1] <= 400, "and never taller than the display it has to fit on"


def test_every_block_along_the_bottom_fits_without_overlapping_the_rows():
    """Four blocks are stacked over twenty-one rows, and a window short enough to squeeze them is
    the case where a control silently stops being clickable."""

    for height in (600, MIN_H - 1, MIN_H, 700, 900, 1200, 1440):
        found = blocks(height, GROUPS)
        tops = [top for top, _tall in found.values()]
        assert tops == sorted(tops), found

        last = layout(height, GROUPS)[-1]
        assert last[2] + last[3] <= min(tops), height
        assert len(rows(height, GROUPS)) == len(DIALS)

        assert min(tops) - (last[2] + last[3]) <= PAD, height
        if height >= MIN_H:
            assert max(top + tall for top, tall in found.values()) <= height, height


def test_the_help_line_only_names_keys_the_strip_itself_handles():
    """The help line names the strip's own keys, and every one it names is handled."""
    panel = DialPanel(_runner(),
                      actions={a: (lambda *a_: None) for a in console.ACTIONS},
                      shelf=object(), encoders=EncoderMap({}))
    panel.attach(None)
    panel._size = (400, 900)
    offered = " ".join(panel._status_lines()[3:])
    strip = (0, 0, 400, 900)

    pygame.init()
    for spelling, label, action in console.HELP:
        named = f"{' '.join(spelling)} {label}"
        assert named in offered, (named, offered)
        for key in spelling:
            ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.key.key_code(key), mod=0)
            assert panel.handle(ev, strip) is (action is not None), (key, action)

    for key in (pygame.K_ESCAPE, pygame.K_f, pygame.K_n, pygame.K_LEFT):
        ev = pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)
        assert panel.handle(ev, strip) is False, key


def test_a_key_whose_action_was_not_supplied_is_neither_offered_nor_swallowed():
    """A key whose action the host did not supply is neither listed nor taken: `play` supplies
    no model action for a single loaded model, so `[ ]` must not appear."""
    bare = DialPanel(_runner())
    bare.attach(None)
    bare._size = (400, 900)
    offered = " ".join(bare._status_lines()[3:])
    strip = (0, 0, 400, 900)
    pygame.init()
    for spelling, label, action in console.HELP:
        if action in (None, console.MINE):
            continue
        assert f"{' '.join(spelling)} {label}" not in offered, (spelling, offered)
        for key in spelling:
            ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.key.key_code(key), mod=0)
            assert bare.handle(ev, strip) is False, action
    assert "r free" in offered and "g route" in offered

    with pytest.raises(KeyError, match="recrd"):
        DialPanel(_runner(), actions={"recrd": lambda: None})


def test_the_window_conversion_is_the_same_picture_in_the_texture_s_own_order():
    """A window is handed BGRA because a streaming texture is ARGB8888 and anything else makes
    SDL convert every pixel on the display thread. This pins the channel order: a swapped red
    and blue still looks like a working picture."""

    out = torch.linspace(-1, 1, 3 * 8 * 6).reshape(1, 3, 8, 6)
    rgb = to_rgb(out).numpy()
    bgra = to_bgra(out).numpy()

    assert rgb.shape == (8, 6, 3) and bgra.shape == (8, 6, 4)
    assert np.array_equal(bgra[:, :, 0], rgb[:, :, 2]), "blue first"
    assert np.array_equal(bgra[:, :, 1], rgb[:, :, 1]), "then green"
    assert np.array_equal(bgra[:, :, 2], rgb[:, :, 0]), "then red"
    assert (bgra[:, :, 3] == 255).all(), "alpha is opaque"


def test_the_description_is_wrapped_by_measuring_it_rather_than_counting_characters():
    """Every line of a dial's description fits the strip, measured with the font that draws it."""

    pygame.font.init()
    font = pygame.font.SysFont(SANS, 13)
    width = 400

    for name, (_rest, text) in DIALS.items():
        lines = wrap(text, font, width, 5)
        assert lines, name
        for line in lines:
            assert font.size(line)[0] <= width, (name, line, font.size(line)[0])

    long = DIALS["noise"][1]
    assert wrap(long, font, width, 2)[-1].endswith("...")
    assert not wrap("two words", font, width, 4)[-1].endswith("...")
    assert wrap("two words", font, width, 4) == ["two words"]

    broken = wrap("a" * 200, font, 60, 4)
    assert broken and all(font.size(line)[0] <= 60 for line in broken), broken


def test_the_routing_grid_maps_a_click_to_one_drum_and_one_dial():
    """Fifteen rows by however many channels the kit has. One column per CHANNEL, not per
    track, for the same reason the lights are: a shared voice cannot be told apart."""

    height, count = 900, 12
    columns = grid_columns(WIDTH, count)
    laid_out = rows(height, GROUPS)
    assert len(columns) == count
    assert all(w > 0 for _x, w in columns)
    assert all(a[0] + a[1] <= b[0] for a, b in zip(columns, columns[1:], strict=False)), columns
    assert columns[-1][0] + columns[-1][1] <= WIDTH

    for name, top, tall in laid_out:
        for i, (cx, cw) in enumerate(columns):
            got = grid_cell(cx + cw // 2, top + tall // 2, laid_out, columns)
            assert got == (name, i), (name, i, got)
    assert grid_cell(PAD, laid_out[0][1] + 4, laid_out, columns) is None


def test_the_strip_does_not_rasterise_the_same_line_twice():
    """`font.render` rasterises every glyph on every call, on the window's thread, and a
    repaint asks for the same few dozen lines as the last one."""

    with headless_renderer((WIDTH, 400), "text") as renderer:
        panel = DialPanel(_runner())
        panel.attach(renderer)

        once = panel._say(panel._small, "se_256", (1, 2, 3))
        assert panel._say(panel._small, "se_256", (1, 2, 3)) is once, "kept, not drawn again"
        assert panel._say(panel._small, "se_256", (9, 9, 9)) is not once, (
            "a dial goes bright when it is grabbed, and that is a different line")
        assert panel._say(panel._tiny, "se_256", (1, 2, 3)) is not once, "so is another font"

        lines = panel._lines("a b c d e f g h", panel._body, 60, 3)
        assert panel._lines("a b c d e f g h", panel._body, 60, 3) is lines
        assert lines == wrap("a b c d e f g h", panel._body, 60, 3), "and is what `wrap` says"

        panel._said = dict.fromkeys(range(TEXT_CACHE), None)
        panel._say(panel._small, "after the bound", (1, 2, 3))
        assert len(panel._said) == 1, "bounded by starting again, which costs one repaint"


def test_a_hand_on_the_strip_beats_a_knob_parked_on_the_same_dial():
    """Reaching for a control on screen means to override whatever the hardware is parked on,
    and that is a decision rather than a consequence of who moved first."""
    runner = _runner()
    knobs = EncoderMap(parse_controls("16=noise"))
    knobs.apply(runner, 0, 16, 127)
    runner.hold(SOURCE, {"noise": 0.25}, PRIORITY)
    assert runner.hands["noise"] == pytest.approx(0.25)

    runner.free(SOURCE, "noise")
    assert runner.hands["noise"] == pytest.approx(1.0)


def test_the_channel_map_is_remembered_so_it_stops_living_in_a_document(tmp_path, monkeypatch):
    """A missing `--map` falls back to a guessed layout, which credits every onset to the
    wrong drum when the machine's audio does not start at channel 0. A map given once is
    remembered, and the guess says it is one."""
    monkeypatch.setattr(live, "CHANNEL_MAP", tmp_path / "channels.txt")

    guessed, note = live.resolve_map("", "tracks")
    assert guessed["BD"] == 0, "the default layout starts the kit at 0"
    assert "GUESS" in note, note

    given, note = live.resolve_map(OVERBRIDGE, "tracks")
    assert given["BD"] == 2 and given["CH"] == given["OH"] == 8
    assert live.CHANNEL_MAP.read_text(encoding="utf-8").strip() == OVERBRIDGE

    recalled, note = live.resolve_map("", "tracks")
    assert recalled == given, "the remembered map did not come back the same"
    assert "GUESS" not in note, note

    live.CHANNEL_MAP.write_text("BD=nonsense", encoding="utf-8")
    fell_back, note = live.resolve_map("", "tracks")
    assert fell_back["BD"] == 0 and "GUESS" in note, note


def test_no_status_line_runs_off_the_edge_of_the_strip():
    """Every status line fits the strip in every mode, including the longest: all three
    holders on a dial at once."""

    runner = _runner()
    runner.hold(console.SOURCE, {"dir1": 0.5}, console.PRIORITY)
    EncoderMap({(-1, 35): "noise"}).apply(runner, 13, 35, 90)
    PressureMap(parse_pressure("BD=se_256")).apply(runner, 13, INDEX["BD"], 110)
    assert len(set(runner.hands_from.values())) == 3, "the longest line needs all three holders"

    panel = DialPanel(runner, actions={a: (lambda *a_: None) for a in console.ACTIONS},
                      shelf=object())
    panel.attach(None)
    panel._size = (WIDTH, 900)
    room = WIDTH - 2 * console.PAD

    for mode in (console.MODE_DIALS, console.MODE_ROUTING, console.MODE_MODELS):
        panel.mode = mode
        for line in panel._status_lines():
            assert panel._small.size(line)[0] <= room, (
                f"in {mode}: {line!r} is {panel._small.size(line)[0]}px in a "
                f"{room}px strip")


def test_no_dial_description_is_cut_off_by_the_block_that_shows_it():
    """A dial whose description outgrows its block would lose its last sentence silently."""
    pygame.font.init()
    body = pygame.font.SysFont(SANS, 13)
    room, most = WIDTH - 2 * PAD, (DESC_H - 22) // 16

    over = {name: len(wrap(prose, body, room, 99)) for name, (_rest, prose) in DIALS.items()
            if len(wrap(prose, body, room, 99)) > most}
    assert not over, f"cut off in a {most}-line block: {over}"


def test_the_strip_offers_l_only_when_there_are_knobs_to_learn():
    assert "learn" not in console.ACTIONS, "built in, like m; not an action the caller supplies"
    plain = _panel(DIALS)
    encoders = EncoderMap({})
    panel = _panel(DIALS)
    panel.encoders = encoders
    for p in (plain, panel):
        p._pg = pygame
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l, mod=0)

    assert plain.handle(ev, (0, 0, 10, 10)) is False, "no knobs, no key"
    assert panel.handle(ev, (0, 0, 10, 10)) is True
    assert encoders.learning == panel._focus
    assert panel.handle(ev, (0, 0, 10, 10)) is True
    assert encoders.learning is None, "a second press disarms"


def test_a_model_s_own_dials_come_back_where_the_hand_left_them(tmp_path):
    """The shared blocks stay under the hand; the MODEL block and the directions mean something
    different on every model, so they go with it and come back with it."""
    names = per_model(fastgan())
    assert {"se_256", "noise", "dir1", "dir8"} <= set(names)
    assert {"speed", "grid", "reaction"}.isdisjoint(names)

    runner = _runner()
    runner.hold(SOURCE, {"se_256": 0.9, "dir1": 0.1, "speed": 0.7}, PRIORITY)
    store = Positions(tmp_path / "positions.json")
    assert store.stash("a-1", runner, SOURCE, names) == {"se_256": 0.9, "dir1": 0.1}
    assert runner.hands == {"speed": 0.7}

    fresh = Positions(tmp_path / "positions.json")
    assert fresh.recall("b-2", runner, SOURCE, PRIORITY, names) == {}
    assert fresh.recall("a-1", runner, SOURCE, PRIORITY, names) == {"se_256": 0.9, "dir1": 0.1}
    assert runner.hands["se_256"] == pytest.approx(0.9)
    assert runner.hands["speed"] == pytest.approx(0.7), "and the shared blocks were not touched"
    assert fresh.recall("a-1", runner, SOURCE, PRIORITY, ["dir1"]) == {"dir1": 0.1}, (
        "only onto dials the new model has")

    broken = tmp_path / "bad.json"
    broken.write_text("{", "utf-8")
    assert Positions(broken).trouble, "an unreadable file is reported, not raised"


def test_the_drum_lights_sit_under_the_loaded_model_s_rows_not_the_default_ones():
    """The lights sit below the loaded model's rows, however many MODEL dials it has."""
    names = [f"gain_{h}" for h in (4, 8, 16, 32, 64, 128, 256, 512, 1024)] + [
        f"noise_{h}" for h in (8, 32, 128, 512)]
    many = adopted(names, [0.5] * 13, [(0.5, 1.0, 2.0)] * 13, [25.0] * 13).groups
    for height in (1000, 1200, 1400):
        _name, top, tall = rows(height, many)[-1]
        boxes = lights(424, blocks(height, many)["lights"][0], 12)
        assert all(y == blocks(height, many)["lights"][0] + 6 for _x, y, _w, _h in boxes)
        assert all(y >= top + tall for _x, y, _w, _h in boxes), (height, boxes[0], top + tall)


def test_a_tall_window_gives_its_spare_height_to_the_description():

    at_floor = blocks(MIN_H, GROUPS)["desc"][1]
    assert at_floor == DESC_H
    assert blocks(1400, GROUPS)["desc"][1] == 2 * DESC_H, "capped at twice, the rest stays empty"
    assert DESC_H < blocks(MIN_H + 60, GROUPS)["desc"][1] < 2 * DESC_H
    for height in (MIN_H, MIN_H + 60, 1400):
        found = blocks(height, GROUPS)
        assert max(top + tall for top, tall in found.values()) <= height


def test_the_directions_show_their_ranking_without_being_grabbed_one_at_a_time():
    """Eight direction dials look identical and can differ by a factor of five."""
    panel = _panel(DIALS, levels=(100.0, 50.0, 25.0))
    assert panel._strengths() == pytest.approx((1.0, 0.5, 0.25))

    assert _panel(DIALS)._strengths() == (), "no basis, no ladder to draw"
    assert _panel()._strengths() == (), "and no model at all is not a crash"


def test_a_dark_dial_says_why_in_the_model_s_terms_not_the_interface_s():
    """A dark column with no reason is worse than no column. The group heading only covers a
    block that is dark entirely, so each dark direction says why itself."""
    panel = _panel({"dir1"}, levels=(90.0,))

    why = panel._reason("dir4")
    assert "1 latent direction" in why, why
    assert "random direction" in why, "a dropped direction is not a missing feature"

    assert "architecture" in panel._reason("se_256")
    assert panel._reason("se_256") != why, "two different reasons, not one phrase reused"


def test_the_described_dial_survives_a_model_that_does_not_have_it():
    """A switch can retire the dial the hand was last on, and every reader of the focus looks
    the dial up in the layout, on the window's thread, one frame after the switch."""
    panel = _panel(DIALS)
    panel._focus = "dir5"
    assert panel.focus == "dir5", "a dial the model has is left alone"

    panel.bank.current = dataclasses.replace(panel.bank.current,
                                             layout=_fastgan.fastgan(directions=4))
    assert "dir5" not in panel.dials, "this test needs the switch to retire the focused dial"

    assert panel.focus in panel.dials
    panel._colour(panel.focus)
    assert panel.dials[panel.focus].blurb


def test_the_model_keys_work_on_a_layout_whose_brackets_need_altgr():
    """On AZERTY `]` is AltGr plus a key whose own keycode is not a bracket: the typed
    character or the key's position must still switch models."""
    import types

    import pygame

    from ganlive.strip import model_step

    def ev(key, unicode="", scancode=0):
        return types.SimpleNamespace(key=key, unicode=unicode, scancode=scancode)

    assert model_step(pygame, ev(pygame.K_RIGHTBRACKET)) == 1
    assert model_step(pygame, ev(pygame.K_5, unicode="[")) == -1           # AltGr+5 on AZERTY
    assert model_step(pygame, ev(pygame.K_DOLLAR, scancode=48)) == 1      # the key right of `^`
    assert model_step(pygame, ev(pygame.K_g, unicode="g")) == 0
