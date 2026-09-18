"""The latent walk: where the picture goes, measured in beats."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ganlive.dials.fastgan_dials import DIALS, SPANS
from ganlive.dials.table import (
    LATENT,
    MASTER,
    MOTION,
)
from ganlive.walk import (
    BEATS_PER_BAR,
    MusicalClock,
    SlerpWalk,
    WalkConfig,
)
from tests.support import NZ, _step, walk


def test_the_latent_depends_only_on_the_musical_position():
    """Two walks that reached the same beat by different routes must agree."""
    a, b = walk(), walk()
    for beat in [x * 0.25 for x in range(41)]:           # a fine sweep, 10 beats
        a.latent(beat)
    for beat in [0.0, 3.7, 10.0]:                        # a coarse one, same destination
        b.latent(beat)
    assert torch.allclose(a.latent(10.0), b.latent(10.0), atol=0)


def test_rewinding_returns_the_picture_that_was_there():
    """Looping a pattern must loop the visual, so going back to bar 2 must be bit-identical."""
    w = walk()
    first = w.latent(8.0).clone()
    for beat in [12.0, 20.0, 33.5, 60.0]:                 # wander off
        w.latent(beat)
    assert torch.equal(w.latent(8.0), first)


def test_a_frame_rate_change_does_not_change_the_path():
    """Sampling the same musical span at 30 and at 90 fps must visit the same latents."""
    slow, fast = walk(), walk()
    bpm = 130.0
    for i in range(1, 31):                                # 1 s at 30 fps
        slow.latent(i / 30 * bpm / 60)
    for i in range(1, 91):                                # 1 s at 90 fps
        fast.latent(i / 90 * bpm / 60)
    assert torch.allclose(slow.latent(bpm / 60), fast.latent(bpm / 60), atol=0)


@pytest.mark.parametrize("beats_per_segment", [1.0, 2.0, 4.0, 8.0])
def test_a_segment_boundary_lands_exactly_on_a_seed(beats_per_segment):
    """Arrival is on the bar line by construction, which is what makes quantisation free."""
    w = walk(beats_per_segment=beats_per_segment)
    for k in range(4):
        beat = k * beats_per_segment
        assert torch.allclose(w.latent(beat), w.seed_for(k).view(1, -1), atol=1e-6), (
            f"segment {k} does not start on its own seed")


def test_the_bar_line_is_a_bar_line_at_any_tempo():
    """Segments are counted in beats, so the boundary is at the same *musical* place whatever
    the tempo -- and at a different wall-clock place, which is the entire point."""
    w = walk(beats_per_segment=BEATS_PER_BAR)
    assert w.seconds_per_segment(120.0) == pytest.approx(2.0)
    assert w.seconds_per_segment(140.0) == pytest.approx(1.714, abs=1e-3)
    assert 60 / w.seconds_per_segment(140.0) > 60 / w.seconds_per_segment(120.0)


def test_a_higher_tempo_changes_seed_more_often():
    """The property that was actually asked for, asserted end to end through the clock."""
    seen = {}
    for bpm in (100.0, 150.0):
        clock = MusicalClock(bpm)
        w = walk(beats_per_segment=BEATS_PER_BAR)
        indices = set()
        for _ in range(600):                              # 10 s at 60 fps
            clock.advance(1 / 60)
            w.latent(clock.beats)
            indices.add(w.segment_index)
        seen[bpm] = len(indices)
    assert seen[150.0] > seen[100.0], seen
    assert seen[150.0] == pytest.approx(seen[100.0] * 1.5, abs=1.5)


def test_the_step_grid_holds_the_image_between_subdivisions():
    """`grid` is a look -- stepped, machine-like motion -- so it has to actually hold."""
    w = walk(beats_per_segment=BEATS_PER_BAR, step_grid=16)
    inside = [w.latent(1.0 + d).clone() for d in (0.0, 0.02, 0.04, 0.06)]
    for other in inside[1:]:
        assert torch.equal(inside[0], other)
    assert not torch.equal(inside[0], w.latent(1.30))     # a later step is a different place


def test_without_a_step_grid_the_motion_is_continuous():
    """The control for the test above: the same positions must differ when the grid is off."""
    w = walk(beats_per_segment=BEATS_PER_BAR, step_grid=0)
    assert not torch.equal(w.latent(1.0), w.latent(1.05))


def test_the_seed_for_a_segment_ignores_how_it_was_reached():
    """Seeds come from a counter-seeded generator rather than an advancing stream, so that
    jumping to bar 33 gives the same picture as arriving there."""
    a, b = walk(), walk()
    for k in range(20):
        a.seed_for(k)
    assert torch.equal(a.seed_for(33), b.seed_for(33))


def test_loop_segments_repeats_the_sequence():
    w = walk(loop_segments=8)
    assert torch.equal(w.seed_for(0), w.seed_for(8))
    assert torch.equal(w.seed_for(3), w.seed_for(11))
    assert not torch.equal(w.seed_for(0), w.seed_for(1))


def test_different_base_seeds_give_different_walks():
    assert not torch.equal(walk(base_seed=0).seed_for(5), walk(base_seed=1).seed_for(5))


def test_a_seed_has_the_norm_the_generator_was_trained_on():
    """Targets are plain Gaussian draws, so their length must sit near sqrt(nz). A walk whose
    endpoints drifted off that shell would spend the whole path out of distribution, which is
    the failure the straight-line path was rejected for."""
    w = SlerpWalk(256, "cpu", WalkConfig())
    norms = [float(w.seed_for(k).norm()) for k in range(16)]
    assert all(0.85 * 16.0 < n < 1.15 * 16.0 for n in norms), norms


def test_song_position_jumps_the_clock():
    """A Song Position Pointer is how a rewind reaches us, and the walk is only rewindable if
    the clock is."""
    c = MusicalClock()
    c.on_song_position(16)                                # 16 sixteenths is four beats
    assert c.beats == pytest.approx(4.0)
    c.on_song_position(0)
    assert c.beats == pytest.approx(0.0)


def test_spread_controls_reach_independently_of_rate():
    """Rate and reach are separate knobs and the walk must not conflate them."""
    steps = [_step(SlerpWalk(256, "cpu", WalkConfig(spread=s)))
             for s in (0.05, 0.2, 0.5, 1.0)]
    assert steps == sorted(steps), steps
    assert steps[0] < 3.0, "spread 0.05 should stay inside the 'hold' band"
    assert steps[-1] > 16.0, "spread 1.0 should be a scene replacement"


def test_spread_keeps_targets_on_the_shell():
    """A target off the training shell would be out of distribution before the walk even
    starts, which is the failure the great circle exists to avoid."""
    for spread in (0.05, 0.3, 0.7, 1.0):
        w = SlerpWalk(256, "cpu", WalkConfig(spread=spread))
        norms = [float(w.seed_for(k).norm()) for k in range(20)]
        assert all(12.0 < n < 20.0 for n in norms), (spread, min(norms), max(norms))


def test_spread_does_not_break_purity():
    """Everything above still has to hold once targets are drawn around a home."""
    a, b = walk(spread=0.25, home_every=8), walk(spread=0.25, home_every=8)
    for k in range(30):
        a.seed_for(k)
    assert torch.equal(a.seed_for(21), b.seed_for(21))
    assert torch.equal(a.latent(9.5), b.latent(9.5))


def test_spread_one_is_exactly_the_raw_draw():
    """The default has to be bit-identical to drawing, so the old behaviour is recoverable."""
    w = SlerpWalk(256, "cpu", WalkConfig(spread=1.0))
    # `_draw` is the walk's own width, `seed_for` is the loaded model's: the sequence is
    # drawn once, above every model, and each reads the front of it.
    assert torch.equal(w.seed_for(7), w._draw(7)[:256])


def test_the_walk_uses_the_motion_dials_when_they_are_set():
    """The config fields have to actually reach the walk, or the dial is inert."""
    held = SlerpWalk(NZ, "cpu", WalkConfig(beats_per_segment=4.0, hold=0.8, when=1.0))
    assert torch.equal(held.latent(0.0), held.latent(2.0))       # frozen through the standstill
    assert not torch.equal(held.latent(0.0), held.latent(3.9))   # then it moves


def test_the_recorder_can_ask_for_frames_and_still_get_musical_positions():
    """The recorder counts frames; everything here counts beats. The adapter is what lets the
    existing frame loop become an instrument without being edited."""
    from ganlive.walk import BeatDriver

    clock = MusicalClock(120.0)
    w = walk(beats_per_segment=BEATS_PER_BAR)
    driver = BeatDriver(w, clock, fps=60.0)
    for step in range(120):                               # two seconds at 120 BPM
        z = driver.next_z(step)
    assert tuple(z.shape) == (1, NZ)
    assert clock.beats == pytest.approx(4.0, abs=1e-6)    # two seconds is four beats


def test_midi_clock_takes_the_position_away_from_the_frame_count():
    """Free-running, a dropped frame slows the music with it, which is right for recording to
    a file. Under real clock the position comes from counted pulses, so a dropped frame skips
    further along the path instead -- which is what is wanted on stage."""
    from ganlive.walk import BeatDriver

    clock = MusicalClock(120.0)
    driver = BeatDriver(walk(), clock, fps=60.0)
    for _ in range(MusicalClock.PPQN * 8):
        clock.on_pulse()
    before = clock.beats
    for step in range(60):
        driver.next_z(step)
    assert clock.beats == before, "frame counting must not move a clock the Rytm is driving"


def test_a_new_seed_sequence_is_seen_at_once_rather_than_at_the_end_of_the_bar():
    """The endpoint cache keys on what the endpoints actually depend on, not just on `k`."""
    where = 8.0 * 4
    for field, value in (("base_seed", 99), ("loop_segments", 3), ("home_every", 2)):
        w = walk(base_seed=1, beats_per_segment=8.0, spread=0.4)
        before = w.latent(where).clone()
        assert torch.allclose(w.latent(where), before), "same position, same latent"
        setattr(w.cfg, field, value)
        assert not torch.allclose(w.latent(where), before), field

    w = walk(base_seed=1, beats_per_segment=8.0, spread=0.4)
    before = w.latent(2.0).clone()
    w.cfg.spread = 0.9
    assert torch.allclose(w.latent(2.0), before), "spread must wait for the segment to end"
    assert not torch.allclose(w.latent(10.0), before)


def test_two_latent_widths_read_the_same_seed_sequence():
    """**The claim that lets a bank hold two latent widths at once.**"""
    from ganlive.walk import SlerpWalk, WalkConfig

    narrow = SlerpWalk(256, "cpu", WalkConfig(base_seed=7), dtype=torch.float32)
    wide = SlerpWalk(512, "cpu", WalkConfig(base_seed=7), dtype=torch.float32)

    for k in (0, 1, 5):
        torch.testing.assert_close(narrow.seed_for(k), wide.seed_for(k)[:256],
                                   rtol=0, atol=0)

    for beats in (0.0, 4.0, 8.0):                       # a seed lands on every 4th beat
        a, b = narrow.latent(beats).reshape(-1), wide.latent(beats).reshape(-1)
        assert a.shape == (256,) and b.shape == (512,)
        torch.testing.assert_close(a, b[:256], rtol=0, atol=1e-6)

    mid = (narrow.latent(2.0).reshape(-1) - wide.latent(2.0).reshape(-1)[:256]).abs().max()
    assert mid < 0.05, f"the two widths are walking different neighbourhoods: {mid}"


def test_a_direction_basis_of_the_wrong_width_is_skipped_rather_than_crashing():
    """**This killed a live set, and the good version of it is a missing push.**"""

    from ganlive.walk import SlerpWalk, WalkConfig

    cfg = WalkConfig(directions=np.eye(4, 256, dtype=np.float32), amounts=(1.0, 0, 0, 0))
    walk = SlerpWalk(256, "cpu", cfg, dtype=torch.float32)
    assert walk._offset() is not None, "a matching basis must still push"

    walk.retarget(512)                       # the basis is now the previous model's
    assert walk._offset() is None, "a 256-wide push was folded into a 512-wide latent"
    assert walk.latent(1.5).shape == (1, 512), "and the frame still has to be produced"

    # The `w` seam is exempt: its width is the mapping's and never `nz`.
    cfg.push_into = torch.zeros(3, 256)
    assert walk._offset() is not None, "a w-space basis was refused for not matching nz"


def test_retargeting_a_walk_changes_its_width_and_keeps_its_place():
    """A switch hands the incoming model the position the outgoing one was at."""
    from ganlive.walk import SlerpWalk, WalkConfig

    walk = SlerpWalk(256, "cpu", WalkConfig(base_seed=3), dtype=torch.float32)
    reference = SlerpWalk(512, "cpu", WalkConfig(base_seed=3), dtype=torch.float32)

    walk.latent(5.5)                                    # load a segment, so there is a cache
    walk.retarget(512)
    after = walk.latent(5.5).reshape(-1)

    assert walk.nz == 512 and after.shape == (512,)
    torch.testing.assert_close(after, reference.latent(5.5).reshape(-1), rtol=0, atol=0), (
        "the walk did not land where a walk of the new width would be at this beat")
    assert walk.phase(5.5) == reference.phase(5.5), "the switch moved the beat"

    assert walk._scratch.shape == (512,), "the offset scratch is still the old width"
    assert all(b.shape == (1, 512) for b in walk._staging), "the staging ring is stale"

    walk.retarget(512)                                  # idempotent: a repaint is not a switch
    assert walk.nz == 512


def test_the_offset_is_rebuilt_when_the_basis_changes_under_a_held_dial():
    """A model switch does not move the dials, so the amounts alone cannot be the cache key."""

    from ganlive.walk import SlerpWalk, WalkConfig

    first = np.eye(2, 8, dtype=np.float32)
    cfg = WalkConfig(directions=first, amounts=(1.0, 0.0))
    walk = SlerpWalk(8, "cpu", cfg)
    before = walk.latent(0.5).clone()

    cfg.directions = np.zeros((2, 8), dtype=np.float32)      # a different model's basis
    after = walk.latent(0.5)
    assert not torch.equal(before, after), (
        "the latent did not change when the basis did, so a stale offset is being reused")


def test_a_walk_with_no_directions_still_plays():
    """A model whose first layer cannot be factorised must not take the instrument down."""
    from ganlive.walk import SlerpWalk, WalkConfig

    cfg = WalkConfig(directions=None, amounts=(1.0, -1.0))
    walk = SlerpWalk(8, "cpu", cfg)
    assert walk.latent(0.5).shape == (1, 8)


def test_the_cache_key_actually_reads_every_field_it_claims_to():
    """`_key` builds a literal tuple for speed, so it can drift from `SEED_FIELDS`, which is
    what the partition test above trusts. Changing each named field must change the key."""
    w = walk()
    base = w._key(3)
    for name in SlerpWalk.SEED_FIELDS:
        setattr(w.cfg, name, getattr(w.cfg, name) + 7)
        assert w._key(3) != base, f"_key ignores {name}, which SEED_FIELDS says it reads"
        setattr(w.cfg, name, getattr(w.cfg, name) - 7)
    assert w._key(3) == base
    assert w._key(4) != base, "and the segment index itself"


def test_the_two_routes_a_dial_can_be_written_by_do_not_overlap_or_invent():
    """That a dial is *written* is `test_every_dial_changes_something`'s job, and it does it behaviourally,
    which is stronger. This is the pair of facts that one cannot see: a dial written by both routes would
    have two owners, and a span naming a dial that does not exist would be a destination nothing can ever
    reach."""
    # Three routes now, not two. `noise` walks a ladder of bands and the `dir` dials are
    # written as one vector onto the walk, so neither can be a `Span`.
    by_table = set(MASTER) | set(MOTION) | {"noise"} | set(LATENT)
    spanned = {s.dial for s in SPANS}
    assert not (spanned & by_table), "a dial cannot be written by both routes"
    assert spanned | by_table == set(DIALS), "a dial is wired to neither route"


def test_the_walk_hands_an_onnx_graph_its_latent_on_the_host():
    """Same numbers, without the round trip to the card and back."""

    from ganlive.walk import SlerpWalk, WalkConfig

    assert WalkConfig().latent_on_host is False, "the torch path is the default"

    cfg = WalkConfig(base_seed=11)
    walk = SlerpWalk(nz=32, device="cpu", config=cfg, dtype=torch.float16)

    for beats in (0.0, 1.4, 3.7, 8.2):
        cfg.latent_on_host = False
        on_card = walk.latent(beats)
        cfg.latent_on_host = True
        on_host = walk.latent(beats)

        assert isinstance(on_card, torch.Tensor)
        assert isinstance(on_host, np.ndarray), "the graph is handed the host view itself"
        assert np.array_equal(np.asarray(on_card).reshape(-1), on_host), (
            f"the two hand-over paths disagree at beat {beats}, so one of them is not the "
            f"latent the other measured")
