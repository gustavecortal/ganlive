"""Tests for the audio-reactive control layer's load-bearing claims.

`trajectory.py` justifies itself with four properties that all follow from one decision -- the
latent is a pure function of musical position -- and every one of them is the kind of thing
that silently stops being true. A walk that drifts a few milliseconds off the bar still looks
fine in isolation; it only reads as wrong beside a kick drum, which is not a thing a test suite
can watch. So the properties are asserted directly.

CPU only, no checkpoint, no GPU: the whole module is arithmetic on 256-vectors, so the tests
stay usable while a training run has the card.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import sys
import time
import types

import numpy as np
import pytest
import torch

from ganlive.timing import stat_ms
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

NZ = 32


def walk(**kw) -> SlerpWalk:
    return SlerpWalk(NZ, "cpu", WalkConfig(**kw))


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


def test_the_walk_stays_on_the_shell_between_seeds():
    """The great circle's whole justification: no midpoint sag. The straight line measured
    10.5 against endpoints of 15.2, and the frame there had 36% of their contrast."""
    w = SlerpWalk(256, "cpu", WalkConfig(beats_per_segment=4.0))
    ends = min(float(w.seed_for(0).norm()), float(w.seed_for(1).norm()))
    mids = [float(w.latent(t).norm()) for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]]
    assert min(mids) >= ends * 0.98, (min(mids), ends)


def test_the_internal_clock_counts_beats_at_the_stated_tempo():
    c = MusicalClock(120.0)
    for _ in range(60):
        c.advance(1 / 60)                                 # one second
    assert c.beats == pytest.approx(2.0, abs=1e-6)        # 120 BPM is 2 beats a second


def test_midi_pulses_take_over_from_the_internal_clock():
    """The fallback must become invisible the moment real clock arrives, so a driver can call
    `advance` unconditionally without checking who is in charge."""
    c = MusicalClock(120.0)
    c.advance(1.0)
    assert c.source == "internal"
    for _ in range(MusicalClock.PPQN):
        c.on_pulse()
    assert c.source == "midi"
    assert c.beats == pytest.approx(1.0)                  # 24 pulses is one quarter note
    c.advance(10.0)                                       # must now be ignored
    assert c.beats == pytest.approx(1.0)


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


def test_song_position_jumps_the_clock():
    """A Song Position Pointer is how a rewind reaches us, and the walk is only rewindable if
    the clock is."""
    c = MusicalClock()
    c.on_song_position(16)                                # 16 sixteenths is four beats
    assert c.beats == pytest.approx(4.0)
    c.on_song_position(0)
    assert c.beats == pytest.approx(0.0)


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


def _step(w: SlerpWalk, n: int = 40) -> float:
    """Mean displacement between consecutive targets -- how far one segment actually travels."""
    return sum(float((w.seed_for(k) - w.seed_for(k + 1)).norm()) for k in range(n)) / n


def test_spread_controls_reach_independently_of_rate():
    """Rate and reach are separate knobs and the walk must not conflate them."""
    steps = [_step(SlerpWalk(256, "cpu", WalkConfig(spread=s)))
             for s in (0.05, 0.2, 0.5, 1.0)]
    assert steps == sorted(steps), steps
    assert steps[0] < 3.0, "spread 0.05 should stay inside the 'hold' band"
    assert steps[-1] > 16.0, "spread 1.0 should be a scene replacement"


def test_the_documented_spread_table_matches_what_the_walk_does():
    """The table in `WalkConfig.spread` is load-bearing -- it is how a value gets chosen -- so
    it is checked rather than trusted. Tolerances are wide because these are means over a
    random sequence, not identities."""
    for spread, expected in ((0.05, 1.8), (0.10, 3.5), (0.20, 7.0), (0.35, 11.8),
                             (0.50, 16.0), (0.70, 20.2), (1.00, 22.5)):
        got = _step(SlerpWalk(256, "cpu", WalkConfig(spread=spread)), n=60)
        assert got == pytest.approx(expected, abs=0.6), (spread, got, expected)


def test_spread_keeps_targets_on_the_shell():
    """A target off the training shell would be out of distribution before the walk even
    starts, which is the failure the great circle exists to avoid."""
    for spread in (0.05, 0.3, 0.7, 1.0):
        w = SlerpWalk(256, "cpu", WalkConfig(spread=spread))
        norms = [float(w.seed_for(k).norm()) for k in range(20)]
        assert all(12.0 < n < 20.0 for n in norms), (spread, min(norms), max(norms))


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


def _pulses(clock, bpm, seconds, t0=0.0, jitter=0.0, rng=None):
    """Feed `seconds` of MIDI clock at `bpm`. Returns the timestamp reached."""
    period = 60.0 / (bpm * MusicalClock.PPQN)
    t = t0
    for _ in range(int(seconds / period)):
        t += period
        clock.on_pulse(t + (rng.uniform(-jitter, jitter) if rng else 0.0))
    return t


def test_changing_tempo_mid_performance_moves_the_position_immediately():
    """Turning the tempo encoder must retime the video, and exactly."""
    c = MusicalClock(128.0)
    t = _pulses(c, 128.0, 4 * 4 * 60 / 128)
    _pulses(c, 160.0, 4 * 4 * 60 / 160, t0=t)
    assert c.beats == pytest.approx(32.0, abs=1e-9)


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


from ganlive.dials import table as _surface  # noqa: E402
from ganlive.dials.table import (
    DIALS,
    GRID_STEPS,
    NOISE_BANDS,
    NOISE_FALLBACK_GAIN,
    NOISE_RAMP,
    LATENT,
    MASTER,
    MODEL,
    MOTION,
    SPANS,
    SPEED_BEATS,
    SPREAD_TABLE,
    Surface,
    clamp01,
    noise_for,
    spread_for,
)


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


def test_the_walk_uses_the_motion_dials_when_they_are_set():
    """The config fields have to actually reach the walk, or the dial is inert."""
    held = SlerpWalk(NZ, "cpu", WalkConfig(beats_per_segment=4.0, hold=0.8, when=1.0))
    assert torch.equal(held.latent(0.0), held.latent(2.0))       # frozen through the standstill
    assert not torch.equal(held.latent(0.0), held.latent(3.9))   # then it moves


class FakeKnobs:
    """Enough of `Knobs` to record what a dial writes, with no card and no checkpoint."""

    def __init__(self, names=None):
        from ganlive.dials.table import SETTINGS_WRITTEN

        self.index = {n: i for i, n in enumerate(SETTINGS_WRITTEN if names is None else names)}
        self.written = {}
        self.noise_gains = None

    def set(self, name, value):
        if name not in self.index:
            raise KeyError(f"{name!r} is not a setting this generator has; "
                           f"have {sorted(self.index)}")
        self.written[name] = value

    def commit(self):
        pass


def _applied(**dials):
    """One set of dial values, applied to both of the surface's destinations."""
    surface = Surface(dials)
    knobs, walk = FakeKnobs(), WalkConfig()
    surface.apply(knobs, walk)
    return (dict(knobs.written), walk)


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


def test_nothing_can_be_driven_off_its_scale():
    """Clamping at the dial is what replaced twenty-two chances to write a value the network
    has never seen. A shove far past the end has to stop at the end."""
    surface = Surface()
    surface.add("noise", 40.0)
    surface.add("se_256", -12.0)
    assert surface["noise"] == 1.0
    assert surface["se_256"] == 0.0


def test_speed_only_ever_picks_a_musical_length():
    """A move taking an unmusical number of beats would arrive between beats, which is the one
    thing the whole beat-locked design exists to prevent."""
    for i in range(21):
        assert _applied(speed=i / 20)[1].beats_per_segment in SPEED_BEATS


def test_every_quarter_turn_of_speed_halves_the_time():
    """The documented behaviour of the dial, checked rather than trusted."""
    got = [_applied(speed=x)[1].beats_per_segment for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert got == [8.0, 4.0, 2.0, 1.0, 0.5], got


def _displacement(spread):
    """Read the measured table the other way round: spread in, distance moved out."""
    for (s0, d0), (s1, d1) in zip(SPREAD_TABLE, SPREAD_TABLE[1:], strict=False):
        if spread <= s1:
            return d0 + (d1 - d0) * (spread - s0) / (s1 - s0)
    return SPREAD_TABLE[-1][1]


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


def test_the_groups_cover_every_dial_exactly_once():
    """The groups are what a hardware surface gets laid out from, so they have to partition."""
    assert set(MASTER) | set(MOTION) | set(LATENT) | set(MODEL) == set(DIALS)
    assert (len(MASTER) + len(MOTION) + len(LATENT) + len(MODEL)) == len(DIALS)


from ganlive.presets import Impulse, Macro, Preset  # noqa: E402

# --- preset fixtures -------------------------------------------------------------------
#
# These were the eight presets the package used to ship. They are test data now: the player
# asked for the interface to open at the dials' resting values and be wired by hand, so the
# library holds `DEFAULT` and whatever was saved at the controls. The rules below still
# exercise every shape a rule can take -- per-track, kit-wide `*`, negative amounts, macros
# -- which is what these tests were always really about.

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
# --------------------------------------------------------------------------------------


def test_a_hit_cannot_push_a_dial_off_its_scale():
    """The clamp is at the dial, once, instead of at every parameter -- so an impulse with a
    wildly wrong amount is a setting that does nothing extra, not a broken picture."""
    from ganlive.presets import Impulse, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import Impulse, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import Impulse, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import Impulse, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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


def _stage_input(height=8, width=12, small=4):
    """A big picture and the small second one, distinguishable by their values."""
    big = torch.zeros(1, 3, height, width)
    big[:, :, :, : width // 2] = 1.0                     # left half bright, right half dark
    return [big, torch.full((1, 3, small, small * 3 // 2), 0.5)]


def test_the_stage_shrinks_to_the_size_being_shown():
    stage = FrameStage(4, 6)
    assert tuple(stage.step(_stage_input())[0].shape[-2:]) == (4, 6)












def test_hit_detection_is_scored_against_the_simulator_rather_than_asserted():
    """The reason the stand-in Rytm returns its event list at all."""
    from ganlive.control.features import offline, score_onsets
    from ganlive.control.tracks import INDEX, MachineSim

    for seed in (1, 3, 7):
        take = MachineSim(bpm=130.0, seed=seed).render(bars=8, tail=0.5)
        feats = offline(take.stems, take.samplerate, fps=60.0)
        got = score_onsets(feats["onsets"], 60.0, take.events, INDEX)
        assert got["precision"] == 1.0, (seed, got)
        assert got["recall"] > 0.98, (seed, got)
        assert got["abs_error_ms"] < 4.0, (seed, got)


def test_the_ground_truth_drops_hits_that_were_silenced_before_they_sounded():
    """Two tracks sharing one voice on the same step means the second erases the first, so
    listing the first as a hit would make the ground truth claim something nothing can hear --
    which once scored the detector at 90% on closed hats while it was completely correct."""
    from ganlive.control.tracks import MachineSim, Section, _s

    both = [Section("clash", 1, {"CH": _s("x..............."),
                                 "OH": _s("x...............")})]
    take = MachineSim(bpm=130.0, seed=0, sections=both).render(bars=1, tail=0.2)
    names = [name for _t, name, _v in take.events]
    assert names == ["OH"], names


def test_a_hand_on_a_dial_sets_where_the_drums_push_from():
    """`hands` is the seam a physical encoder writes to, and it is applied with the resting
    values rather than over the top of the rules -- so turning an encoder moves where the
    reaction happens instead of cancelling it."""
    from ganlive.presets import Impulse, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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


def test_the_preflight_tool_and_the_stand_in_agree_on_the_drums():
    """A preset tuned against the stand-in has to address the same drums the hardware tool
    labels, or the channel map discovered with one is read with the other's names."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    from ganlive.tools import doctor as rytm_preflight

    from ganlive.control.tracks import TRACKS

    assert rytm_preflight.TRACKS == TRACKS


def test_the_pessimistic_eight_channel_case_still_finds_the_hits():
    """If the MKI turns out to expose voices rather than tracks, closed and open hat arrive on
    one channel. A preset has to survive that, so the loss it costs is measured rather than
    assumed -- and the loss is real hardware behaviour, not a detection failure."""
    from ganlive.control.features import offline, score_onsets
    from ganlive.control.tracks import VOICE_GROUPS, MachineSim, channel_map

    take = MachineSim(bpm=130.0, seed=3).render(bars=8, tail=0.5)
    channel_of = channel_map("voices")
    voices = take.stems_for(channel_of)
    assert voices.shape[0] == len(VOICE_GROUPS) == 8
    feats = offline(voices, take.samplerate, fps=60.0)
    got = score_onsets(feats["onsets"], 60.0, take.events, channel_of)
    assert got["precision"] == 1.0, got
    assert 0.80 < got["recall"] < 1.0, got                # the choke, not the detector


def test_the_five_midi_messages_that_matter_reach_the_clock():
    """Each one answers a different question, and getting any of them wrong produces video
    that still looks smooth. They are checked as pure arithmetic, with no hardware."""
    from ganlive.control import midi

    clock = MusicalClock(120.0)
    for _ in range(MusicalClock.PPQN * 2):
        assert midi.dispatch(clock, midi.CLOCK) == "clock"
    assert clock.beats == pytest.approx(2.0)
    assert clock.source == "midi"

    assert midi.dispatch(clock, midi.START) == "start"
    assert clock.beats == pytest.approx(0.0), "Start is what puts the video's bar on the music's"

    assert midi.dispatch(clock, midi.STOP) == "stop"
    assert not clock.running
    assert midi.dispatch(clock, midi.CONTINUE) == "continue"
    assert clock.running

    assert midi.dispatch(clock, midi.SONG_POSITION, 16, 0) == "song_position"
    assert clock.beats == pytest.approx(4.0)
    assert midi.dispatch(clock, midi.SONG_POSITION, 0, 1) == "song_position"
    assert clock.beats == pytest.approx(128 / 4)


def test_anything_else_on_the_wire_is_ignored():
    """A Rytm sends note-ons and controller moves down the same cable. None of them may
    disturb the position."""
    from ganlive.control import midi

    clock = MusicalClock(120.0)
    for _ in range(MusicalClock.PPQN):
        midi.dispatch(clock, midi.CLOCK)
    before = clock.beats
    assert midi.dispatch(clock, 0x90, 60, 100) == "note_on"
    assert midi.dispatch(clock, 0x9F, 60, 100) == "note_on", "any MIDI channel"
    assert midi.dispatch(clock, 0x90, 60, 0) == "note_off", "velocity 0 is a release"
    for status in (0x80, 0xE0, 0xF0, 0xFE, 0xF1):
        assert midi.dispatch(clock, status, 60, 100) is None
    for channel in range(16):
        assert midi.dispatch(clock, midi.CONTROL_CHANGE + channel, 60, 100) == "control_change"
    assert clock.beats == before


def test_the_reader_hands_pressure_on_and_not_just_notes_and_knobs():
    """**The seam the other pressure tests cannot reach.** `dispatch` naming a byte and
    `PressureMap` acting on one are both covered, and neither would have caught the reader
    having no branch to carry it between them -- which is exactly the state the aftertouch
    stream was in: classified nowhere, delivered nowhere, and silent about both."""
    from ganlive.control.midi import AFTERTOUCH_POLY, CONTROL_CHANGE, NOTE_ON, ClockReader

    got = {"pressure": [], "note": [], "control": []}
    reader = ClockReader(MusicalClock(120.0),
                         on_pressure=lambda c, n, v: got["pressure"].append((c, n, v)),
                         on_note=lambda c, n, v: got["note"].append((c, n, v)),
                         on_control=lambda c, n, v, _top: got["control"].append((c, n, v)))

    class _Port:
        """One burst, then nothing -- the shape `run`'s drain loop reads."""

        def __init__(self, events):
            self._events = list(events)

        def poll(self):
            return bool(self._events)

        def read(self, _n):
            out, self._events = self._events, []
            return [[e, 0] for e in out]

    port = _Port([[AFTERTOUCH_POLY + 13, 3, 90, 0], [NOTE_ON + 13, 5, 100, 0],
                  [CONTROL_CHANGE + 13, 35, 64, 0]])
    reader.open_ports = lambda: [("fake", port)]
    reader.start()
    for _ in range(200):                       # the thread polls; do not sleep a fixed guess
        if got["pressure"] and got["note"] and got["control"]:
            break
        time.sleep(0.005)
    reader.stop_flag = True
    reader.join(2.0)

    assert got["pressure"] == [(13, 3, 90)], got
    assert got["note"] == [(13, 5, 100)] and got["control"] == [(13, 35, 64)]
    assert reader.counts.get("aftertouch_poly") == 1, reader.counts


def test_a_reader_with_no_ports_says_so_rather_than_failing():
    """No MIDI is a real outcome -- the Rytm's USB setting is a choice between Overbridge and
    MIDI, and they may not coexist. It has to degrade to the internal tempo, loudly."""
    from ganlive.control.midi import ClockReader

    reader = ClockReader(MusicalClock(120.0), port_match="nothing-matches-this")
    assert reader.open_ports() == []
    said = reader.describe()
    assert said.startswith("MIDI:"), said
    assert "ignoring the machine" in said or "no MIDI" in said, said


def test_the_two_channel_layouts_are_what_the_hardware_might_give():
    """One channel per track is the optimistic case; one per analog voice is what an MKI
    advertising ten inputs most likely means."""
    from ganlive.control.tracks import TRACKS, VOICE_GROUPS, channel_map

    per_track = channel_map("tracks")
    assert len(set(per_track.values())) == len(TRACKS) == 12

    per_voice = channel_map("voices")
    assert len(set(per_voice.values())) == len(VOICE_GROUPS) == 8
    assert per_voice["CH"] == per_voice["OH"]
    assert per_voice["RS"] == per_voice["CP"]
    assert per_voice["CY"] == per_voice["CB"]
    assert per_voice["BD"] != per_voice["SD"]


def test_a_typo_in_a_discovered_layout_is_refused():
    """A wrong map is a machine where the kick drives what the hat should, and nothing
    anywhere reports a problem -- so a name that is not a real track has to raise."""
    from ganlive.control.tracks import parse_channel_map

    assert parse_channel_map("BD=0, ch=3") == {"BD": 0, "CH": 3}
    with pytest.raises(ValueError, match="unknown track"):
        parse_channel_map("KICK=0")


def test_reaction_scales_how_hard_hits_land_without_touching_the_arrangement():
    """The one control to reach for when a performance is too much or too little."""
    from ganlive.presets import Impulse, Macro, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

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


def test_every_dial_gets_a_row_and_no_row_gets_two():
    """The strip is built from `surface`, never from a second list of names. A dial added
    there has to appear here without anyone remembering to add it -- the alternative is a
    control that exists, does something, and cannot be reached."""
    from ganlive.strip import GROUPS, rows

    named = [name for _title, names in GROUPS for name in names]
    assert [label for label, _y, _h in rows(900, GROUPS)] == named


def test_the_rows_stay_clickable_on_a_short_window_and_stop_sprawling_on_a_tall_one():
    """A row too short to hit is a control that cannot be turned, and one that grows without
    limit is twenty-one sliders spread down a 1440-pixel screen."""
    from ganlive.strip import GROUPS
    from ganlive.strip import rows as dial_rows

    for height in (420, 700, 1080, 2160):
        rows = [(y, h) for _l, y, h in dial_rows(height, GROUPS)]
        assert all(18 <= h <= 34 for _y, h in rows), height
        assert all(b >= a + h for (a, h), (b, _) in zip(rows, rows[1:], strict=False))


def test_a_click_lands_on_the_dial_it_looks_like_it_lands_on():
    """Hit-testing and drawing read the same track geometry. Two definitions would let the
    pointer disagree with the picture by a few pixels for ever."""
    from ganlive.strip import GROUPS, hit, layout, track_span

    w, h = 372, 900
    left, span = track_span(w)
    for kind, label, top, tall in layout(h, GROUPS):
        if kind != "dial":
            continue
        y = top + tall // 2
        assert hit(left, y, w, h, GROUPS) == (label, pytest.approx(0.0))
        assert hit(left + span, y, w, h, GROUPS) == (label, pytest.approx(1.0))
        assert hit(left + span // 2, y, w, h, GROUPS)[1] == pytest.approx(0.5, abs=0.01)
        assert hit(2, y, w, h, GROUPS) == (label, pytest.approx(0.0))
    heads = [(y, hgt) for kind, _l, y, hgt in layout(h, GROUPS) if kind == "head"]
    assert hit(left, heads[0][0] + 2, w, h, GROUPS) is None


def test_a_hand_beats_the_slow_rule_for_that_dial_and_only_that_dial():
    """A slow rule SETS, so without this a slider on any dial a macro drives would be overwritten a
    microsecond later and the control would look broken."""
    from ganlive.presets import Impulse, Macro, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import Macro, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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


def test_the_panel_publishes_a_new_dict_rather_than_editing_the_loops_one():
    """The seam between the mouse and the render loop, and the reason it needs no lock."""
    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    panel = DialPanel(runner)
    first = runner.hands
    panel.set("noise", 0.4)
    assert runner.hands == {"noise": 0.4}
    assert runner.hands is not first, "the loop's own dict must never be written into"

    second = runner.hands
    panel.set("se_64", 2.0)                       # clamped at the dial, like everything else
    assert runner.hands == {"noise": 0.4, "se_64": 1.0}
    assert second == {"noise": 0.4}, "a dict already handed over must not change afterwards"

    panel.release("noise")
    assert runner.hands == {"se_64": 1.0}
    panel.release()
    assert runner.hands == {}


def test_the_panel_reports_only_what_is_off_its_resting_value():
    """`p` prints something to paste into a preset, and a preset that restated all fifteen
    resting values would say nothing about what was actually found."""
    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    panel = DialPanel(runner)
    panel.set("hold", 0.72)
    runner.apply([1e6] * len(INDEX), {}, FakeKnobs())
    assert panel.settings() == {"hold": 0.72}


def test_switching_patch_keeps_the_objects_the_loop_and_the_walk_hold():
    """The console changes setting while the loop runs. The loop holds the runner and the walk
    holds `walk_cfg`, so rebuilding either would leave something driving an object nothing
    reads -- silently, which is the failure this project keeps paying for."""
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    cfg = runner.walk_cfg
    runner.hold("test", {"noise": 0.5})
    runner.load(FIXTURES["release"])

    assert runner.walk_cfg is cfg, "the walk would stop seeing every motion dial"
    assert runner.preset.name == "release"
    assert runner.hands == {"noise": 0.5}, "a hand on a control survives a change of setting"
    runner.apply([1e6] * len(INDEX), {"density": 0.0}, FakeKnobs())
    assert runner.surface["hold"] == pytest.approx(FIXTURES["release"].dials["hold"])


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
    assert readout("se_256", 0.5) == "as trained"
    assert readout("se_256", 1.0) == "up 100%"
    assert readout("se_256", 0.25) == "down 50%"
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


def test_building_a_walk_pays_every_first_call_cost_up_front():
    """A deferred import in a frame loop is 1.93 seconds of a performance's first bar."""
    import sys

    name = "ganlive.models.fastgan"
    built = SlerpWalk(16, "cpu", WalkConfig(spread=0.3))
    saved = sys.modules[name]
    try:
        sys.modules[name] = None                          # any import of it now raises
        assert built.seed_for(3) is not None, "nothing in the loop may need the import again"
        assert built.latent(9.0) is not None
    finally:
        sys.modules[name] = saved


def test_a_shared_voice_gets_one_light_and_says_so():
    """Under the shared-voice layout four pairs of drums arrive on one channel and genuinely
    cannot be told apart. Twelve lights would show a separation the hardware does not have."""
    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX, channel_map

    per_track = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0))
    assert [label for _ch, label in per_track.kit] == list(INDEX)

    shared = DialPanel(PresetRunner(FIXTURES["still"], channel_map("voices"), 60.0))
    labels = [label for _ch, label in shared.kit]
    assert len(labels) == 8, labels
    assert "CH/OH" in labels and "MT/HT" in labels
    channels = [ch for ch, _label in shared.kit]
    assert channels == sorted(set(channels)), "one light per channel, in the machine's order"


def test_the_lights_fit_the_strip_whatever_the_kit_is():
    """A layout discovered with --meter is neither of the two built-in ones, so the row has to
    size itself rather than assume twelve."""
    from ganlive.strip import PAD, STATUS_H, lights

    assert lights(372, 900, 0) == []
    for count in (1, 5, 8, 12, 16):
        boxes = lights(372, 900, count)
        assert len(boxes) == count
        assert all(x >= PAD for x, _y, _w, _h in boxes)
        assert all(x + w <= 372 - PAD + 2 for x, _y, w, _h in boxes), count
        assert all(y >= 900 - STATUS_H for _x, y, _w, _h in boxes)
        assert all(b >= a + w for (a, _, w, _hh), (b, *_) in zip(boxes, boxes[1:],
                                                                strict=False))


def test_the_order_between_the_four_kinds_of_writer_is_stated_on_the_surface():
    """Resting value, then a hand (which holds), then anything that `set`s (which a hand beats), then anything
    that `add`s (which lands on top)."""
    surface = Surface()
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

    surface = Surface()
    assert surface["noise"] == pytest.approx(DIALS["noise"][0])
    surface.set("noise", 0.9)
    assert surface["noise"] == pytest.approx(0.9)


def test_what_p_prints_is_what_you_set_not_what_the_patch_already_rested_at():
    """`p` exists to write down a discovery. Comparing against the module's resting values
    printed `release`'s own four dials straight back as if they had been found, and sampling
    the live values baked a decaying hit into the number."""
    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["release"], INDEX, 60.0)
    panel = DialPanel(runner)
    runner.apply([1e6] * len(INDEX), {"density": 5.0}, FakeKnobs())
    assert panel.settings() == {}, "a preset's own resting values are not discoveries"

    panel.set("noise", 0.61)
    since = [1e6] * len(INDEX)
    since[INDEX["BD"]] = 0.01                             # a kick still ringing on `hold`
    runner.apply(since, {"density": 5.0}, FakeKnobs())
    assert runner.surface["hold"] != pytest.approx(FIXTURES["release"].dials["hold"]), (
        "the fixture needs the hit to actually be moving something")
    assert panel.settings() == {"noise": 0.61}


def test_dragging_a_slider_does_not_repaint_the_strip():
    """A 1000 Hz mouse delivers about sixteen moves a frame. Repainting on each one rebuilt
    eighteen text runs and re-uploaded 1.79 MB at 60 Hz for the length of every drag -- 1.38 ms
    against 0.14 on the display thread, during the one activity the panel exists for. What a
    drag has to show is the bar and the marker, and both are rectangles drawn every frame."""
    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    panel = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0))
    panel.set("spread", 0.2)                               # a grab: the held text changes
    assert panel._dirty
    panel._dirty = False
    for i in range(20):                                   # the drag itself
        panel.set("spread", 0.2 + i * 0.01)
    assert not panel._dirty, "a drag must not repaint"
    panel.release("spread")
    assert panel._dirty, "letting go changes the held colour and the count"


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


def test_a_setting_naming_a_dial_that_no_longer_exists_says_so():
    """**Silence here cost the player both of his saved takes.**"""
    from ganlive.presets import Impulse, Macro, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

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


@dataclasses.dataclass
class _StubModel:
    """Just enough of `bank.Model` for the bank to be switched between."""

    rows: object = None
    directions: object = None
    dials_live: frozenset = frozenset()
    name: str = "stub"
    #: Not `None`: `bank.Model.path` has no default, so production always has one, and
    #: `_rewire` asks it which backend is loaded. A stub without one made that a crash.
    path: object = dataclasses.field(default_factory=lambda: pathlib.Path("stub.pt"))
    net: object = None
    #: A `ladder` beside the `nz`, because `_rewire` now moves the *stage* to the incoming
    #: model's native size as well as the walk to its latent width -- a bank no longer has
    #: one size. Fourth field this stub has grown for that reason; see the note below.
    cfg: object = dataclasses.field(default_factory=lambda: types.SimpleNamespace(
        nz=8, ladder=types.SimpleNamespace(width=96, height=64)))
    knobs: object = None
    graphs: int = 0
    compile_s: float = 0.0
    layout: object = dataclasses.field(default_factory=lambda: _surface.fastgan())
    #: `None` is what `bank.Model` defaults it to, and it is what every family except a converted StyleGAN2
    #: carries.
    push: object = None


def test_every_dial_is_accounted_for_on_a_fully_featured_model():
    """**The same defect wearing the other face.**"""
    from ganlive.dials.steer import Knobs
    from ganlive.bank import live_dials
    from ganlive.dials.table import DIRECTIONS, SETTINGS_WRITTEN

    full = Knobs(sorted(SETTINGS_WRITTEN), "cpu", torch.float32)
    live = live_dials(full, directions=range(DIRECTIONS))    # `len()` is all it asks of them
    missing = set(DIALS) - live
    assert not missing, (
        f"{sorted(missing)} can be turned on the strip and reach nothing on any model; "
        f"give each one a Span, a noise band or a direction, or take it off the surface")


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
    from ganlive.dials.steer import Knobs
    from ganlive.bank import live_dials
    from ganlive.dials.table import MODEL, MOTION, Surface

    knobs = Knobs([], "cpu", torch.float32)
    live = live_dials(knobs, directions=None)

    assert not (set(MODEL) & live), "an ONNX model has none of these"
    assert set(MOTION) <= live, "and all of these, on every model"
    assert not any(name.startswith("dir") for name in live), "no basis, no direction dials"

    Surface().apply(knobs, WalkConfig())            # every write accepted and dropped
    knobs.commit()


def test_a_dial_the_model_does_not_have_is_drawn_dark_and_cannot_be_grabbed():
    """**The failure this project keeps paying for, in the one place it is still visible.**"""
    import types

    from ganlive.strip import GROUPS, DialPanel
    from ganlive.strip import rows as console_rows
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    live = frozenset({"reaction", "speed", "spread", "hold", "late", "grid",
                      "dir1", "dir2"})
    panel = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0),
                      bank=types.SimpleNamespace(current=_StubModel(dials_live=live),
                                                models=[1], index=0, name="stub"))
    panel._size = (400, 900)
    panel._rows = console_rows(900, GROUPS)

    assert {"dir1", "dir2"} <= panel.live_dials()
    assert "dir3" not in panel.live_dials(), "a direction the model does not have"
    assert "se_256" not in panel.live_dials(), "nor a gate this architecture does not have"

    live_row = next(top for name, top, _t in panel._rows if name == "dir1")
    dead_row = next(top for name, top, _t in panel._rows if name == "dir3")
    assert panel._hit(200, live_row + 4) is not None
    assert panel._hit(200, dead_row + 4) is None, "dragging a dead dial must move nothing"


def _panel(dials_live=None, levels=None):
    """A strip with a stub rig behind it, for the things that are facts about a model."""
    import types

    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    dirs = None if levels is None else types.SimpleNamespace(levels=levels)
    bank = None if dials_live is None else types.SimpleNamespace(
        current=_StubModel(dials_live=frozenset(dials_live), directions=dirs),
        models=[1], index=0, name="stub")
    return DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0), bank=bank)


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


def test_the_routing_grid_will_not_wire_a_drum_to_a_dial_that_cannot_fire():
    """`route` raises on a dead dial, and the grid runs on the window's thread -- so the
    guard that was added for the mouse on a slider had to reach the other way in too."""
    import pygame

    panel = _panel({"dir1", "speed"})
    panel._pg = pygame
    panel._columns = [(0, 20)]
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0))

    panel._grid_gesture(ev, "se_256", 0)               # must not raise
    assert not panel.runner.routing(), "and must not have wired anything"

    panel._grid_gesture(ev, "speed", 0)
    assert "speed" in panel.runner.routing(), "a live dial still wires"


def test_the_described_dial_survives_a_model_that_does_not_have_it():
    """A switch retires dials: one checkpoint keeps 4 of its 8 directions and a StyleGAN2 has no
    `se_*` at all. The name a hand was last on outlived them, and every reader of it looks the
    dial up in the layout -- which raises `KeyError`, on the window's thread, one frame after
    the switch, with nothing on screen to say which dial it was."""
    panel = _panel(DIALS)
    panel._focus = "dir5"
    assert panel.focus == "dir5", "a dial the model has is left alone"

    panel.bank.current = dataclasses.replace(panel.bank.current,
                                            layout=_surface.fastgan(directions=4))
    assert "dir5" not in panel.dials, "this test needs the switch to retire the focused dial"

    assert panel.focus in panel.dials
    panel._colour(panel.focus)                        # the two lookups that used to raise
    assert panel.dials[panel.focus].blurb


def test_the_strip_lays_out_the_loaded_model_s_dials_and_not_the_departed_one_s():
    """**The rows are a function of the layout, and the cache was keyed on the window.**

    Only a resize rebuilt them, and the window is only ever made taller -- so a switch to a
    model with the same number of rows, or fewer, left the strip drawing the outgoing model's
    names, every one of them dark, with no row at all for the model now playing. Ours and a
    converted StyleGAN2 both come to nineteen rows, so this is the ordinary case rather than an
    edge of one, and it never raised: it just quietly offered the wrong instrument.

    Driven through the real `draw` under SDL's dummy driver, because the decision is there."""
    import os
    import types

    import pygame

    from ganlive.strip import WIDTH, DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    fastgan = _surface.fastgan()
    names = ("w_coarse", "w_mid", "w_fine", "noise_4", "noise_512")
    other = _surface.stylegan2(names, (0.5,) * 5, ((0.1, 1.0, 2.0),) * 5, (25.0,) * 5, ())
    assert len(fastgan.knobs) == len(other.knobs), (
        "this test needs two layouts the window cannot tell apart by height")

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    current = _StubModel(layout=fastgan, dials_live=frozenset(fastgan))
    holder = types.SimpleNamespace(current=current, models=[1], index=0, name="stub")
    runner.use_model(current)

    before = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        from pygame._sdl2.video import Renderer, Window

        pygame.init()
        renderer = Renderer(Window("rows", size=(WIDTH + 200, 900)), vsync=False)
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
        assert {n for n, _t, _h in panel._rows} == set(fastgan), "the model it opened on"

        # `rytm_live.switch_model`'s two statements, in its order.
        holder.current = _StubModel(layout=other, dials_live=frozenset(other))
        frame()                                  # the window's thread, in the gap between them
        runner.use_model(holder.current)
        frame()
    finally:
        pygame.display.quit()
        if before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = before

    assert {n for n, _t, _h in panel._rows} == set(other), (
        "the rows must follow the model even when the window has no reason to move")
    for name in names:
        assert name in panel.dials, "and the incoming model's own dials must be reachable"


def test_a_direction_says_what_it_measured_on_this_model():
    """`rank` measures every direction at load and re-orders them by what they do, and that
    number used to be computed and dropped on the floor. It is the only thing separating a
    leading direction from a tail one at the controls, so it belongs where the player is
    already looking: the description under the dial they just grabbed."""
    import types

    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    dirs = types.SimpleNamespace(levels=(88.0, 41.0))
    panel = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0),
                      bank=types.SimpleNamespace(
                          current=_StubModel(dials_live=frozenset(DIALS), directions=dirs),
                          models=[1], index=0, name="stub"))

    assert "88 8-bit levels" in panel._measured("dir1")
    assert "ranked 1 of 2" in panel._measured("dir1")
    assert "41 8-bit levels" in panel._measured("dir2")
    assert panel._measured("dir3") == "", "a direction this model does not have says nothing"
    assert panel._measured("spread") == "", "and nothing else claims a measurement"


def test_with_no_rig_every_dial_is_live():
    """Every offline tool builds a panel without one, and a strip of dark dials would be a"""
    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    panel = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0))
    assert panel.live_dials() is None, "nothing has said, so nothing is dark"


def test_switching_models_repoints_the_directions_at_the_new_one():
    """The walk is built once, before the loop; the model changes under it."""
    import numpy as np

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
    import numpy as np

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


def test_a_direction_dial_moves_the_latent_and_by_exactly_the_amount_asked():
    """**Without this, deleting `SlerpWalk._offset` outright fails nothing.**"""
    import numpy as np

    from ganlive.dials.table import DIRECTION_RANGE, Surface
    from ganlive.walk import SlerpWalk, WalkConfig

    rows = np.eye(4, 16, dtype=np.float32)
    cfg = WalkConfig(directions=rows)
    walk = SlerpWalk(16, "cpu", cfg)
    surface = Surface()

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
    import numpy as np

    from ganlive.dials.table import DIRECTION_RANGE, Surface
    from ganlive.walk import SlerpWalk, WalkConfig

    #: Two ranges of four, so a row is only ever non-zero in its own half.
    rows = np.zeros((2, 8), dtype=np.float32)
    rows[0, 1] = 1.0
    rows[1, 5] = 1.0
    seam = torch.zeros(2, 4)
    cfg = WalkConfig(directions=rows, push_into=seam)
    walk = SlerpWalk(16, "cpu", cfg)
    surface = Surface()

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


def test_the_offset_is_rebuilt_when_the_basis_changes_under_a_held_dial():
    """A model switch does not move the dials, so the amounts alone cannot be the cache key."""
    import numpy as np

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


def test_a_direction_reads_out_as_a_push_rather_than_as_a_trained_value():
    """`as trained` is meaningless for a direction: there is no trained value to be at."""
    from ganlive.dials.table import DIRECTION_RANGE, readout

    assert readout("dir1", 0.5) == "centred"
    assert readout("dir1", 1.0) == f"+{DIRECTION_RANGE:.2f}"
    assert readout("dir1", 0.0) == f"-{DIRECTION_RANGE:.2f}"


def test_a_two_sided_dial_says_it_is_at_rest_in_its_own_words():
    """Two defects, both found by rendering the strip and looking at it rather than by code."""
    from ganlive.dials.table import DIALS, POLES, readout

    expected = {"se_64": "as trained", "se_128": "as trained", "se_512": "as trained",
                "reaction": "as written", "noise": "off"}
    for name, word in expected.items():
        assert readout(name, DIALS[name][0]) == word, name
    assert set(expected) <= set(POLES), "a dial with poles needs a word for its rest"
    assert all(len(p) == 3 for p in POLES.values()), POLES


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


def test_the_grit_ladder_the_tests_read_is_the_one_the_frame_loop_runs():
    """`apply` inlines the ladder for speed and `noise_for` states it for the tests, so the six noise
    assertions in this file read a function the render loop never calls."""
    from ganlive.dials.table import NOISE_BANDS, noise_for

    for position in (0.0, 0.1, 0.25, 0.5, 0.72, 0.9, 1.0):
        written, _cfg = _applied(noise=position)
        for band, gain in noise_for(position):
            assert written[band] == pytest.approx(gain), (position, band)
        assert {b for b, _f, _s in NOISE_BANDS} <= set(written), "a band stopped being written"


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


def test_the_stage_does_no_work_at_all_when_the_model_is_already_the_right_size():
    """The frame path's most valuable branch, now that the size follows the model."""
    stage = FrameStage(4, 6, device="cpu")
    exact = torch.zeros(1, 3, 4, 6)
    assert stage.step(exact) is exact, "the frame was resampled to the size it already is"

    # Down through `area`, up through `bilinear`. The branch is not pedantry: `area` on the
    # way up is nearest-neighbour, and that case was unreachable while a bank was sized to
    # its smallest member -- it became reachable the moment the stage started moving.
    stage.resize(2, 3)
    assert stage.step(exact).shape == (1, 3, 2, 3)
    stage.resize(8, 12)
    ramp = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6).repeat(1, 3, 1, 1)
    up = stage.step(ramp)
    assert up.shape == (1, 3, 8, 12)
    assert len(up.unique()) > len(ramp.unique()), "an enlargement that only repeated pixels"


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


def test_the_stand_in_produces_whatever_layout_the_map_asks_for():
    """The one command meant to rehearse a discovered map used to hand it per-track audio."""
    import numpy as np

    from ganlive.control.tracks import (
        INDEX,
        TRACKS,
        MachineSim,
        channel_map,
        parse_channel_map,
    )

    take = MachineSim(bpm=130.0, seed=2).render(bars=2, tail=0.2)

    per_track = take.stems_for(channel_map("tracks"))
    assert per_track.shape == take.stems.shape
    assert np.allclose(per_track, take.stems)

    found = parse_channel_map("BD=0,SD=1,CH=2,OH=2")
    mixed = take.stems_for(found)
    assert mixed.shape[0] == 3, "sized by the channels the map addresses"
    assert np.allclose(mixed[0], take.stems[INDEX["BD"]])
    assert np.allclose(mixed[2], take.stems[INDEX["CH"]] + take.stems[INDEX["OH"]])
    mapped = sum(take.stems[INDEX[track]] for track in found)
    assert np.allclose(mixed.sum(axis=0), mapped)

    for layout, n in (("tracks", len(TRACKS)), ("voices", 8)):
        assert take.stems_for(channel_map(layout)).shape[0] == n


def test_the_stand_in_feeder_refuses_audio_shorter_than_one_block():
    """The wrap-around skips a short tail by `continue`, so a take shorter than one block spins
    the thread forever without ever pushing a sample -- silently, as a run that detects no hits
    and looks merely quiet. Both copies of this class had it before there was one class."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.control.tracks import StemFeeder

    ex = FeatureExtractor(2, 48000)
    with pytest.raises(ValueError, match="shorter than one"):
        StemFeeder(ex, np.zeros((2, 100), dtype=np.float32), 48000, 256)
    assert StemFeeder(ex, np.zeros((2, 512), dtype=np.float32), 48000, 256) is not None


def test_the_feeder_reaches_the_extractor_at_something_like_wall_clock_rate():
    """One class now, used by both the live tool and the timing harness. It used to be two,
    differing only in whether being late was counted -- so they could drift apart and only one
    of them could have noticed."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.control.tracks import MachineSim, StemFeeder

    take = MachineSim(bpm=130.0, seed=4).render(bars=2, tail=0.2)
    ex = FeatureExtractor(take.stems.shape[0], take.samplerate)
    feeder = StemFeeder(ex, take.stems, take.samplerate, 256)
    feeder.start()
    time.sleep(0.35)
    feeder.stop()

    expected = 0.35 / (256 / take.samplerate)
    assert 0.4 * expected < feeder.calls < 1.6 * expected, (feeder.calls, expected)
    assert ex.pending, "the extractor should have seen hits"
    assert feeder.push_ms, "the bookkeeping is always on now, not only in the timing harness"

    pushes = list(feeder.push_ms)
    steady = stat_ms(pushes[1:])
    assert steady["p95"] < 5.33, (steady, pushes[0])
    assert pushes[0] < 40.0, pushes[0]


def test_swing_and_humanising_move_the_hits_and_the_detector_still_finds_them():
    """Two simulator settings that nothing turned. Rather than delete them, they are the
    pessimistic case the detector should be scored against: real drumming is not on the grid,
    and a detector tuned only against a perfect one is tuned against a machine that does not
    exist. This also makes two branches in the innermost render loop reachable.
    """
    from ganlive.control.features import offline, score_onsets
    from ganlive.control.tracks import INDEX, MachineSim

    straight = MachineSim(bpm=130.0, seed=6).render(bars=4, tail=0.4)
    loose = MachineSim(bpm=130.0, seed=6, swing=0.18, humanise_ms=6.0).render(bars=4, tail=0.4)

    assert len(straight.events) == len(loose.events), "same pattern, different timing"
    moved = [abs(a - b) for a, b in zip(sorted(t for t, _n, _v in straight.events),
                                        sorted(t for t, _n, _v in loose.events), strict=True)]
    assert max(moved) > 0.01, "the settings must actually move something"

    feats = offline(loose.stems, loose.samplerate, fps=60.0)
    got = score_onsets(feats["onsets"], 60.0, loose.events, INDEX)
    assert got["recall"] > 0.95, got
    assert got["precision"] > 0.95, got


def test_no_voice_rings_longer_than_the_choke_reaches():
    """The choke silences a shared voice forward from the trig, and it stops at `LONGEST_VOICE_S` rather than
    at the end of the record -- which was averaging half a multi-megabyte row per trig, 625 MB written
    where 62 could be non-zero."""
    from ganlive.control.tracks import LONGEST_VOICE_S, TRACKS, _voice

    longest = {}
    for track in TRACKS:
        longest[track] = max(_voice(track, 1.0, 48000, np.random.default_rng(s)).shape[0]
                             for s in range(6)) / 48000
    worst = max(longest.values())
    assert worst <= LONGEST_VOICE_S, (
        f"{max(longest, key=longest.get)} rings for {worst:.3f}s, past the "
        f"{LONGEST_VOICE_S}s the choke reaches")
    assert LONGEST_VOICE_S < worst * 2.0, "the bound has drifted far past what it guards"


def test_a_choked_trig_leaves_no_tail_behind_it():
    """The property the bound exists to preserve, asserted on the audio rather than on the
    arithmetic: a closed hat on top of a ringing open hat silences it from that sample on."""
    from ganlive.control.tracks import INDEX, MachineSim

    take = MachineSim(bpm=130.0, seed=3).render(bars=8, tail=0.5)
    sr = take.samplerate
    oh = take.stems[INDEX["OH"]]
    at = {name: sorted(int(t * sr) for t, n, _v in take.events if n == name)
          for name in ("CH", "OH")}
    assert at["CH"] and at["OH"], "the fixture needs both hats playing"

    checked = 0
    for start in at["CH"]:
        nxt = next((o for o in at["OH"] if o > start), len(oh))
        window = oh[start:min(nxt, start + int(1.6 * sr))]
        if window.size < sr // 50:                        # too short to say anything
            continue
        checked += 1
        assert np.abs(window).max() < 1e-6, (
            f"an open hat is still sounding {window.size / sr:.3f}s after a closed hat "
            f"took the voice at sample {start}")
    assert checked, "no closed hat landed on a ringing open hat; the fixture proves nothing"


def _stub_net(gates=("se_64", "se_128", "se_256", "se_512"),
              rungs=("feat_8", "feat_32", "feat_128", "feat_512")):
    """A generator-shaped object with only what `install` looks at."""
    from torch import nn

    from ganlive.models.fastgan import SkipLayerExcitation
    from ganlive.models.graph import FoldedNoise

    net = nn.Module()
    for name in rungs:
        coeff = torch.ones(1, 2, 1, 1)
        net.add_module(name, nn.Sequential(FoldedNoise(coeff, torch.zeros(1, 1, 4, 4))))
    for name in gates:
        net.add_module(name, SkipLayerExcitation(2, 2))
    net.to_big = nn.Identity()
    net.init = nn.Identity()
    return net


def test_a_checkpoint_missing_a_setting_the_dials_write_is_refused_at_set_up():
    """A 1024-tall generator has no `feat_2048` rung, and the surface writes its names as
    literals. Without this the mismatch surfaces as a `KeyError` out of `Knobs.set` -- inside
    the frame loop, mid-performance, rather than before a note is played."""
    from ganlive.dials.steer import install

    with pytest.raises(RuntimeError, match="sle.se_512"):
        install(_stub_net(gates=("se_64", "se_128", "se_256")), "cpu", torch.float32)


def test_only_the_settings_something_can_write_are_installed():
    """Thirteen of twenty-two knobs used to be installed with nothing able to write them, and
    ten of those wrapped a rung in a module that multiplies its whole feature map by a live
    tensor holding a constant 1.0, on every frame. `verify` then reported that all thirteen
    moved the picture, which is a pass that says nothing about whether the instrument works."""
    from ganlive.dials.steer import install
    from ganlive.dials.table import SETTINGS_WRITTEN

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
    from ganlive.dials.steer import Knobs

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


def _fake_run(root, name, step, width, height, nz=256):
    """A run directory holding one checkpoint that `config_of` can read. Config only -- the
    weights are what makes a real one 674 MB, and nothing here builds a generator."""
    import torch

    folder = root / name / "checkpoints"
    folder.mkdir(parents=True)
    path = folder / f"{step:07d}.pt"
    torch.save({"config": {"nz": nz, "im_size": height, "im_width": width}, "g_ema": {}}, path)
    return path


def test_the_pickers_rows_are_geometry_not_a_cache_left_by_the_last_repaint():
    """Drawing and hit-testing must not be able to disagree about which row a click is on."""
    from ganlive.strip import MODEL_ROW, PAD, model_at, model_rows

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


def test_two_sources_can_hold_different_dials_without_dropping_each_other():
    """`set_hands` documented that it took one writer and that an encoder arriving as a second
    would need the merge moved inside. Two writers already existed: the console read-merged-wrote
    at the call site and the sweep renderer replaced the whole dict, which would have dropped
    every console-held dial the moment they met."""
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold("a", {"noise": 0.5})
    before = runner.hands
    runner.hold("b", {"se_256": 0.5})
    assert runner.hands is not before, "the loop's dict must never be edited under it"
    assert before == {"noise": 0.5}, "and the one it already read must not change"


def test_a_rule_wired_past_the_end_of_the_kit_is_reported_not_silently_skipped():
    """`--layout tracks` against an eight-input machine puts CY and CB past the end, which used
    to be a `continue` inside the frame loop sixty times a second with nothing said."""
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    full = PresetRunner(FIXTURES["voices"], INDEX, 60.0, channels=12)
    assert not full.dropped

    narrow = PresetRunner(FIXTURES["voices"], INDEX, 60.0, channels=8)
    assert narrow.dropped, "CY sits on channel 10 and this kit has eight"
    assert any("CY" in line for line in narrow.dropped), narrow.dropped
    assert all("CY" not in imp.track for imp, _ch in narrow._impulses)
    narrow.apply(np.full(8, 1e6, dtype=np.float32), {"density": 4.0},
                 FakeKnobs())


def _travelled(walk, beats):
    """Where along its arc the walk really is at `beats`, recovered from the latent it returns."""
    k, _u = walk.phase(beats)
    z0 = walk.seed_for(k).numpy().astype(np.float64)
    z1 = walk.seed_for(k + 1).numpy().astype(np.float64)
    got = walk.latent(beats).numpy().reshape(-1).astype(np.float64)
    a, b = np.linalg.lstsq(np.stack([z0, z1], axis=1), got, rcond=None)[0]
    n0, n1 = z0 / np.linalg.norm(z0), z1 / np.linalg.norm(z1)
    omega = float(np.arccos(np.clip(float(n0 @ n1), -1.0, 1.0)))
    return float(np.arctan2(b * math.sin(omega), a + b * math.cos(omega)) / omega)


def test_the_scope_draws_the_curve_the_walk_actually_travels():
    """The four motion dials change nothing about a single frame by construction, so a strip without this
    showed four controls whose entire behaviour was off-screen. It has to read the walk's own shaping -- a
    display with its own copy would drift from the walk and be believed, and this is the one display whose
    whole job is to be believed about that."""
    from ganlive.strip import scope_curve
    from ganlive.walk import SlerpWalk, WalkConfig, position

    cfg = WalkConfig(beats_per_segment=4.0, hold=0.55, when=0.8, step_grid=0, base_seed=3)
    walk = SlerpWalk(64, "cpu", cfg)
    drawn = scope_curve(cfg, samples=33)

    for i in range(1, 32):
        beats = (i / 32) * cfg.beats_per_segment
        assert _travelled(walk, beats) == pytest.approx(position(cfg, beats)[2], abs=2e-3), beats
        assert drawn[i] == pytest.approx(position(cfg, beats * 0.9999)[2], abs=2e-3)

    stepped = WalkConfig(beats_per_segment=4.0, step_grid=4, base_seed=3)
    walk = SlerpWalk(64, "cpu", stepped)
    assert _travelled(walk, 0.4) == pytest.approx(_travelled(walk, 0.9), abs=2e-3)
    assert _travelled(walk, 0.4) != pytest.approx(_travelled(walk, 1.4), abs=2e-3)
    assert len(set(round(t, 6) for t in scope_curve(stepped, samples=40))) <= 5


def test_the_standstill_and_where_it_sits_are_both_visible_on_the_scope():
    """`hold` is the flat part of the curve and `late` is where the flat part sits. If the
    scope cannot separate those two it is not showing what the dials do."""
    from ganlive.strip import scope_curve
    from ganlive.walk import WalkConfig

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
    from ganlive.strip import driven_by
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    def wires(entry):
        return [(imp.track, ch, imp.amount) for imp, ch in entry[0]]

    got = driven_by(PresetRunner(FIXTURES["voices"], INDEX, 60.0))
    assert wires(got["se_128"]) == [("BD", INDEX["BD"], 0.34)]
    assert wires(got["se_512"]) == [("CP", INDEX["CP"], 0.32), ("SD", INDEX["SD"], 0.24)]
    assert "reaction" not in got, "nothing drives the master in this preset"

    slow = driven_by(PresetRunner(FIXTURES["breathe"], INDEX, 60.0))
    assert slow["spread"] == ([], ["density"]), "a slow rule is not a hit"

    both = driven_by(PresetRunner(FIXTURES["full"], INDEX, 60.0))
    assert wires(both["hold"]) == [("BD", INDEX["BD"], -0.34)], "a pull keeps its sign"
    assert both["hold"][1] == ["density"]

    kit = driven_by(PresetRunner(FIXTURES["pulse"], INDEX, 60.0))
    assert wires(kit["dir1"]) == [("*", -1, 0.22)], kit

    narrow = driven_by(PresetRunner(FIXTURES["voices"], INDEX, 60.0, channels=8))
    assert all(ch < 8 for wired, _slow in narrow.values() for _imp, ch in wired), narrow
    assert "noise" not in narrow, narrow
    assert "se_128" in narrow, "the kick is still inside the channel range"


def test_the_window_actually_opens_with_the_strip_beside_it():
    """**Nothing opened a real window, and a one-line change to how one opens shipped.**"""
    import os
    import sys
    import threading

    import pygame

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    before = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    threads = threading.active_count()
    try:
        from ganlive import window as realtime_video

        panel = _panel(DIALS, levels=(100.0, 50.0))
        display = realtime_video.Display((64, 96), title="test", overlay=panel,
                                         fullscreen=False)
        display.publish(np.zeros((64, 96, 4), dtype=np.uint8))
        display.close()
    finally:
        pygame.display.quit()
        if before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = before
    assert threading.active_count() <= threads + 1, "the window thread outlived the window"


def test_every_routing_column_label_fits_the_column_it_names():
    """The columns share what is left of the strip after the dial names, and a kit that puts two
    drums on one channel labels them `MT/HT`. Centring a label wider than its column pushes it
    into the next one: `MT/HT CH/OH CY/CB` rendered as a single run of characters."""
    import os
    import sys

    import pygame

    from ganlive.strip import WIDTH, DialPanel, grid_columns
    from ganlive.presets import DEFAULT, PresetRunner
    from ganlive.control.tracks import channel_map

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    before = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        from pygame._sdl2.video import Renderer, Window

        pygame.init()
        panel = DialPanel(PresetRunner(DEFAULT, channel_map("voices"), 60.0))
        panel.attach(Renderer(Window("fit", size=(WIDTH, 900)), vsync=False))
        columns = grid_columns(WIDTH, len(panel.kit))
        too_wide = [(label, panel._micro.size(label)[0], cw)
                    for (_channel, label), (_cx, cw) in zip(panel.kit, columns, strict=True)
                    if panel._micro.size(label)[0] > cw]
    finally:
        pygame.display.quit()
        if before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = before

    assert not too_wide, f"label, width, column: {too_wide}"


def test_a_dial_the_surface_does_not_carry_is_drawn_dark_rather_than_raised():
    """**The window thread died on the first frame and the frame loop reported healthy times.**

    The strip draws `bank.current.layout` and reads its values off `runner.surface`, and those
    are only the same set after `PresetRunner.use_model`. The latency harness never made that
    call, so its first paint of a converted StyleGAN2 raised `KeyError: 'w_coarse'` on the
    window's thread -- which stopped the picture while the loop went on timing frames nobody
    could see and printed FITS at the end of them.

    Two things had to change and both are asserted here: a value the surface does not carry
    means the dial is unreachable, which the strip already knows how to draw; and `Display`
    stops the session when its thread dies instead of leaving a frozen window up."""
    import inspect
    import os
    import sys
    import types

    import pygame

    from ganlive.dials import table as S
    from ganlive.strip import WIDTH, DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    # A rig holding a StyleGAN2's dials behind a runner still holding the FastGAN's -- exactly
    # the state a caller that skips `use_model` is in.
    names = ("w_coarse", "w_mid", "w_fine")
    stub = _StubModel(layout=S.stylegan2(names, (0.5,) * 3, ((),) * 3, (25.0,) * 3),
                      dials_live=frozenset({"w_coarse"}))
    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    foreign = set(stub.layout.rests) - set(runner.surface.values)
    assert foreign, "this test needs the two layouts to actually disagree"

    before = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    try:
        from pygame._sdl2.video import Renderer, Window

        pygame.init()
        panel = DialPanel(runner, bank=types.SimpleNamespace(current=stub, models=[1], index=0,
                                                            name="stub"))
        panel.attach(Renderer(Window("dark", size=(WIDTH, 900)), vsync=False))
        panel._resize(WIDTH, 900)
        panel._paint(panel.live())                  # used to raise KeyError here
    finally:
        pygame.display.quit()
        if before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = before

    from ganlive import window as realtime_video

    source = inspect.getsource(realtime_video.Display._run)
    assert "self.stopped = True" in source and "except Exception" in source, (
        "a window thread that dies has to stop the session; a frozen window that still "
        "reports 60 fps is the one failure this whole path exists to prevent")


def test_a_thousand_gestures_across_model_switches_break_nothing():
    """**The defect that shipped was found by a hand, not by a test.** Hover a direction,
    press `m`, and the paint one frame later looked the dial up in a layout that no longer
    had it. No test covered it because every test drives one layout, and the crash needs two.

    `rytm_fuzz` drives the real paint path under SDL's dummy driver, switching models
    mid-gesture and carrying the focus and the drag across. Seeded, so a fault here is
    reproducible with `scripts/rytm_fuzz.py --seeds 1`."""
    import os
    import sys

    import pygame

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    before = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        import test_fuzz_surface as rytm_fuzz
        from pygame._sdl2.video import Renderer, Window

        pygame.init()
        window = Window("fuzz", size=(rytm_fuzz.WIDTH + 200, max(rytm_fuzz.HEIGHTS)))
        faults, reached = rytm_fuzz.run(Renderer(window, vsync=False), 500, seed=0)
    finally:
        pygame.display.quit()
        if before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = before

    assert not faults, rytm_fuzz.report(faults) or [f[5] for f in faults]
    # A fuzzer that never reaches the state is a fuzzer that proves nothing, and this is the
    # state: the strip describing a dial the incoming model does not have.
    assert reached["stale focus"], dict(reached)
    assert reached["into routing"] and reached["into models"], dict(reached)


def test_the_window_opens_no_shorter_than_the_strip_needs():
    """The floor is a property of the window, not of how it happened to open: it is
    resizable, so a floor applied once at construction is not one."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from ganlive import window as realtime_video

    from ganlive.strip import MIN_H, WIDTH

    panel = _panel(DIALS)
    tall_screen = realtime_video.window_size(3072, 2048, 2560, 1440, panel)
    short_screen = realtime_video.window_size(3072, 2048, 1280, 400, panel)

    assert tall_screen[0] > WIDTH, "the strip sits beside the picture, not over it"
    assert tall_screen[1] >= MIN_H, tall_screen
    assert short_screen[1] <= 400, "and never taller than the display it has to fit on"


def test_every_block_along_the_bottom_fits_without_overlapping_the_rows():
    """Four blocks are stacked over twenty-one rows, and a window short enough to squeeze them is
    the case where a control silently stops being clickable."""
    from ganlive.strip import GROUPS, MIN_H, PAD, blocks, layout, rows

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
    """It used to list the window's keys as well, and had gone stale against them -- an overlay
    advertising a host binding it does not own is a claim it cannot keep."""
    import pygame

    from ganlive import strip as console
    from ganlive.strip import DialPanel
    from ganlive.control.midi import EncoderMap
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    panel = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0),
                      actions={a: (lambda *a_: None) for a in console.ACTIONS},
                      shelf=object(), encoders=EncoderMap({}))
    panel._pg = pygame
    pygame.font.init()
    panel._small = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 13)
    panel._size = (400, 900)
    offered = " ".join(panel._status_lines()[3:])
    strip = (0, 0, 400, 900)

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
    """`[ ] model` was printed unconditionally while the live tool supplies no model action for
    a single loaded model, so the strip advertised a key that did nothing. Both halves of that
    now come from one table, and the two have to move together."""
    import pygame

    from ganlive import strip as console
    from ganlive.strip import DialPanel
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    bare = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0))
    bare._pg = pygame
    pygame.font.init()
    bare._small = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 13)
    bare._size = (400, 900)
    offered = " ".join(bare._status_lines()[3:])
    strip = (0, 0, 400, 900)
    for spelling, label, action in console.HELP:
        if action in (None, console.MINE):
            continue
        assert f"{' '.join(spelling)} {label}" not in offered, (spelling, offered)
        for key in spelling:
            ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.key.key_code(key), mod=0)
            assert bare.handle(ev, strip) is False, action
    assert "r free" in offered and "g route" in offered and "p print" in offered

    import pytest

    with pytest.raises(KeyError, match="recrd"):
        DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0), actions={"recrd": lambda: None})


def test_the_window_conversion_is_the_same_picture_in_the_texture_s_own_order():
    """A window is handed BGRA because a streaming texture is ARGB8888 and anything else makes
    SDL convert every pixel on the display thread. That is only a speed change if the bytes are
    the same picture, so this pins the channel order rather than trusting the name -- a swapped
    red and blue is the one bug here that still looks like a working picture."""
    from ganlive.models.graph import to_bgra, to_rgb

    out = torch.linspace(-1, 1, 3 * 8 * 6).reshape(1, 3, 8, 6)
    rgb = to_rgb(out).numpy()
    bgra = to_bgra(out).numpy()

    assert rgb.shape == (8, 6, 3) and bgra.shape == (8, 6, 4)
    assert np.array_equal(bgra[:, :, 0], rgb[:, :, 2]), "blue first"
    assert np.array_equal(bgra[:, :, 1], rgb[:, :, 1]), "then green"
    assert np.array_equal(bgra[:, :, 2], rgb[:, :, 0]), "then red"
    assert (bgra[:, :, 3] == 255).all(), "alpha is opaque"


def test_a_stage_reports_which_conversions_are_compiled_rather_than_implying_it():
    """All three fall back to an eager chain that still produces the right picture several
    times slower. A silent fallback to a slow path is how this project loses arms."""
    from ganlive.models.graph import to_bgra

    bare = FrameStage(8, 12)
    assert bare.compiled == {"yuv": False, "rgb": False, "bgra": False}
    frame = bare.step(torch.zeros(1, 3, 8, 12))
    assert bare.bgra_bytes(frame).shape == (8, 12, 4), "the fallback still produces a frame"

    marked = FrameStage(8, 12, to_bgra=to_bgra)
    assert marked.compiled["bgra"] is True


def test_a_midi_port_filter_that_matched_nothing_says_so():
    """A typo in `--midi-port` and a machine with no MIDI at all produced the same message,
    and that message describes the failure this whole design fears most: the picture runs at
    its own tempo and looks entirely plausible while ignoring the drummer. Two different
    problems must not share one symptom when one of them is a typo."""
    from ganlive.control.midi import ClockReader
    from ganlive.walk import MusicalClock

    empty = ClockReader(MusicalClock(), "rytmm")
    assert "no input ports at all" in empty.describe()

    missed = ClockReader(MusicalClock(), "rytmm")
    missed.rejected = ["Elektron Analog Rytm MKI", "loopMIDI Port"]
    said = missed.describe()
    assert "rytmm" in said and "Elektron Analog Rytm MKI" in said
    assert "matched none of" in said

    listening = ClockReader(MusicalClock(), "rytm")
    listening.ports = ["Elektron Analog Rytm MKI"]
    listening.rejected = ["loopMIDI Port"]
    assert "listening to Elektron" in listening.describe()
    assert "ignoring loopMIDI Port" in listening.describe()


def test_the_preflight_certifies_the_dispatch_the_live_tool_actually_runs():
    """The tool exists to answer whether the clock survives Overbridge, on the one question
    where a wrong answer looks exactly like a right one. It used to re-declare the status bytes
    and re-implement the dispatch, so a green preflight was evidence about code that would not
    run."""
    import importlib.util
    from pathlib import Path

    from ganlive.control import midi
    from ganlive.walk import MusicalClock

    path = Path(__file__).resolve().parents[1] / "src" / "ganlive" / "tools" / "doctor.py"
    spec = importlib.util.spec_from_file_location("doctor_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.dispatch is midi.dispatch, "it must run the shipping dispatch, not its own"
    for name in ("CLOCK", "START", "CONTINUE", "STOP", "SPP", "SONG_POSITION"):
        assert not hasattr(mod, name), f"{name} is a second copy of a midi.py constant"

    listener = mod._MidiListener.__new__(mod._MidiListener)
    listener.clock_state = MusicalClock()
    listener.clock, listener.transport, listener.notes = 0, [], {}
    listener.first_clock = listener.last_clock = None
    assert midi.dispatch(listener.clock_state, midi.CLOCK, now=0.0) == "clock"
    assert listener.clock_state.source == "midi"


def test_the_description_is_wrapped_by_measuring_it_rather_than_counting_characters():
    """A guessed column count ran the text off the right-hand edge and cut its last line off
    mid-sentence -- on the one block whose whole job is to say what a control does, and the
    only defect here that nothing but looking at it would have found."""
    import pygame

    from ganlive.strip import wrap

    pygame.font.init()
    font = pygame.font.SysFont("segoeui,dejavusans,arial", 13)
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


def test_a_drum_can_be_wired_to_a_dial_while_it_runs():
    """The thing this whole layer was heading toward: which drum drives which dial was data on
    the preset already, and what was missing was a way to change it without editing a file."""
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import AMOUNT_MAX, Impulse, PresetRunner
    from ganlive.control.tracks import INDEX

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


def _ramped_to(gains):
    """`noise_for(1.0)` if every band carried `gains`. One definition, two callers."""
    return {name: 1.0 + (gains[name] - 1.0) * clamp01((1.0 - start) / NOISE_RAMP)
            for name, _levels, start in NOISE_BANDS}


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
    import math

    import torch

    from ganlive.dials import steer as K
    from ganlive.dials.table import NOISE_BANDS

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


def test_midi_notes_drive_the_same_features_audio_does():
    """`NoteFeatures` is a drop-in for `FeatureExtractor`: the frame loop reads exactly `n`,
    `since`, `drain()` and `features()`, and nothing downstream may know which it has."""
    from ganlive.control.features import FeatureExtractor, NoteFeatures

    audio = FeatureExtractor(12, 48000)
    notes = NoteFeatures(12)
    for name in ("n", "since", "drain", "features"):
        assert hasattr(notes, name), name
    assert set(notes.features()) == set(audio.features()), "a macro naming a source would break"

    notes.tick(0.0)
    notes.on_note(14, 0, 100, when=0.0)      # BD, hard
    notes.on_note(14, 9, 40, when=0.0)       # OH, soft
    got = notes.drain()
    assert sorted((i, round(v, 2)) for i, v, _ago in got) == [(0, 0.79), (9, 0.31)]
    assert notes.drain() == [], "draining twice must not replay a hit"

    notes.tick(0.25)
    assert notes.since[0] == pytest.approx(0.25, abs=1e-3)
    assert notes.since[9] == pytest.approx(0.25, abs=1e-3)
    assert notes.since[1] > 100, "a track that never fired is long ago, not zero"


def test_midi_separates_the_drums_that_share_an_analog_voice():
    """The whole reason this exists. On an MKI, CH and OH share one voice and arrive on one
    audio channel, so no audio rule can ever tell them apart -- and BT and LT are not streamed
    at all for want of USB bandwidth. Each still has its own note."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.tracks import INDEX, VOICE_GROUPS

    shared = [g for g in VOICE_GROUPS if len(g) > 1]
    assert shared, "this test is about the pairs; there must be some"

    notes = NoteFeatures(12)
    notes.tick(0.0)
    for pair in shared:
        first, second = pair[0], pair[1]
        notes.on_note(14, INDEX[first], 127, when=0.0)
        hits = {i for i, _v, _a in notes.drain()}
        assert hits == {INDEX[first]}, f"{first} must not also report {second}"
        assert notes.since[INDEX[second]] > 100, f"{second} did not fire and must not read so"
        notes.since[INDEX[first]] = 1e6


def test_the_note_source_reports_density_and_does_not_leave_energy_inert():
    """All six macros that ship read `density`, so it is the one that must be right. `energy`
    is a different quantity here -- how hard the pads are struck, not how loud the room is --
    and the failure to guard against is it silently reading 0.0 forever."""
    from ganlive.control.features import NoteFeatures

    notes = NoteFeatures(12)
    window = notes.cfg.energy_window
    notes.tick(0.0)
    for k in range(6):
        notes.on_note(14, k % 12, 127, when=k * 0.05)
    notes.tick(0.30)
    assert notes.density == pytest.approx(6 / window, rel=1e-6)
    assert notes.energy > 0.9, "six hits at full velocity is not zero energy"
    assert notes.active == pytest.approx(6 / 12)

    notes.tick(0.30 + window + 0.01)
    assert notes.density == 0.0
    assert notes.energy == 0.0
    assert notes.active == 0.0


def test_a_quiet_send_loses_its_quietest_drums_first_and_preflight_says_so():
    """`FeatureConfig.floor` is an ABSOLUTE level, measured against the stand-in Rytm rather than against
    anybody's Overbridge gain staging. This pins how much room it actually has, because `rytm_preflight`'s
    `HEADROOM` is a claim about this number and nothing else checks it."""
    from ganlive.control.features import FeatureConfig, FeatureExtractor
    from ganlive.control.tracks import INDEX, MachineSim

    sr, block, tol = 48000, 256, 0.030
    render = MachineSim(bpm=130, samplerate=sr, seed=0).render(bars=2)
    pcm = render.stems.astype(np.float32)
    truth = [(t, INDEX[name]) for t, name, _v in render.events]
    assert truth, "the arrangement must actually trig something"

    def recall(gain):
        ex = FeatureExtractor(pcm.shape[0], sr)
        found = []
        for i in range(0, pcm.shape[1] - block, block):
            ex.push(pcm[:, i:i + block] * gain)
            for ch, _v, ago in ex.drain():
                found.append((ch, (i + block) / sr - ago))
        left, hit = sorted(truth), 0
        for ch, t in sorted(found, key=lambda e: e[1]):
            for k, (tt, tch) in enumerate(left):
                if tch == ch and abs(tt - t) <= tol:
                    hit += 1
                    left.pop(k)
                    break
        return hit / len(truth)

    assert recall(1.0) == 1.0
    assert recall(0.1) == 1.0

    quietest = min(np.abs(pcm[ch]).max() for _t, ch in truth)
    assert recall(0.002) < 0.6, "if this passes, the floor no longer binds and HEADROOM is moot"
    assert quietest > FeatureConfig().floor * 8, (
        "the quietest voice must clear preflight's HEADROOM at unity, or the check it prints "
        "would flag a healthy machine")


def test_the_strength_that_is_drawn_and_the_strength_that_is_edited_are_one_wire():
    """A routing cell is a CHANNEL, and under the voices layout two drums share one. Reading
    the loudest rule on the channel while writing to whichever track a display listed first
    let the bar sit still while the wheel moved something else -- a control that looks dead."""
    from ganlive.presets import Preset, PresetRunner
    from ganlive.control.tracks import channel_map

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
    from ganlive.presets import Preset, PresetRunner
    from ganlive.control.tracks import channel_map

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
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import PresetRunner, from_dict, to_dict
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

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
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["release"], INDEX, 60.0)
    assert runner._base["hold"] > 0.6, "the fixture needs a dial parked high"
    runner.route("CP", "hold")
    pushed = [i for i in runner.preset.impulses if i.track == "CP" and i.dial == "hold"]
    assert pushed and pushed[0].amount < 0, "a dial parked high gets pushed down"

    runner.route("CP", "noise")
    up = [i for i in runner.preset.impulses if i.track == "CP" and i.dial == "noise"]
    assert up and up[0].amount > 0, "one parked at the bottom gets pushed up"
    assert pushed[0].attack > 0 and up[0].attack == 0


def test_the_routing_grid_maps_a_click_to_one_drum_and_one_dial():
    """Fifteen rows by however many channels the kit has. One column per CHANNEL, not per
    track, for the same reason the lights are: a shared voice cannot be told apart."""
    from ganlive.strip import GROUPS, PAD, WIDTH, grid_cell, grid_columns, rows

    height, count = 900, 12
    columns = grid_columns(WIDTH, count)
    assert len(columns) == count
    assert all(w > 0 for _x, w in columns)
    assert all(a[0] + a[1] <= b[0] for a, b in zip(columns, columns[1:], strict=False)), columns
    assert columns[-1][0] + columns[-1][1] <= WIDTH

    for name, top, tall in rows(height, GROUPS):
        for i, (cx, cw) in enumerate(columns):
            got = grid_cell(cx + cw // 2, top + tall // 2, WIDTH, height, count, GROUPS)
            assert got == (name, i), (name, i, got)
    assert grid_cell(PAD, rows(height, GROUPS)[0][1] + 4, WIDTH, height, count, GROUPS) is None


def test_the_settings_vector_is_not_resent_when_no_dial_moved():
    """**Thirty-two bytes that measured 1.6 ms.** `Tensor.copy_` drops the GIL, and taking it
    back from a window thread that is uploading a texture and presenting costs up to one switch
    interval: 1.64 ms median on the played loop, against 0.03 with no window open. A frame on
    which no dial that writes the model moved was paying that to send the card the numbers it
    already had -- and only the MODEL block writes this vector at all, so a preset driving the
    motion and direction dials holds it still while the picture moves."""
    import torch

    from ganlive.dials.steer import Knobs

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


def test_the_strip_does_not_rasterise_the_same_line_twice():
    """`font.render` rasterises every glyph on every call, and a repaint asked for sixty-five
    of them -- the dial names, the group headings, the drum labels and the whole key help, in
    the same colour as the repaint before. That runs on the window's thread at `TEXT_HZ`, one
    frame in eight at 60 fps, holding the GIL the frame loop wants back."""
    import os

    import pygame

    from ganlive.strip import TEXT_CACHE, WIDTH, DialPanel, wrap
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    before = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        from pygame._sdl2.video import Renderer, Window

        pygame.init()
        panel = DialPanel(PresetRunner(FIXTURES["still"], INDEX, 60.0))
        panel.attach(Renderer(Window("text", size=(WIDTH, 400)), vsync=False))

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
    finally:
        pygame.display.quit()
        if before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = before


def test_a_dial_says_which_knob_is_on_it_and_not_only_which_drum():
    """**The routing grid shows the drums; the knobs had nothing.** A learned binding was
    visible for exactly as long as the learn, and afterwards the only way to find out which
    knob moved a dial was to turn all of them and watch the strip."""
    from ganlive.control.midi import EncoderMap, control_name, nrpn_number, parse_controls

    knobs = EncoderMap(parse_controls("16=noise,2:17=se_256,n1.3=dir1,3:16=noise"))
    assert knobs.where("se_256") == "2:17"
    assert knobs.where("dir1") == control_name(-1, nrpn_number(1, 3)) == "n1.3"
    assert knobs.where("noise") == "16 3:16", "both, because both move it"
    assert knobs.where("spread") == "", "and a dial no knob is on says nothing"

    # The spelling is the flag's own, so what the strip shows can be typed back in.
    assert parse_controls("2:17=se_256") == {(1, 17): "se_256"}


def test_a_hand_on_the_strip_outranks_an_encoder_parked_on_the_same_dial():
    """Which writer wins used to be dict insertion order -- whoever touched any dial first this
    session, an arbitrary fact about the past rather than a decision."""
    from ganlive.strip import PRIORITY, SOURCE
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold("encoder", {"noise": 0.2})
    runner.hold(SOURCE, {"noise": 0.8}, PRIORITY)
    assert runner.hands["noise"] == pytest.approx(0.8), runner.hands

    runner.free(SOURCE, "noise")
    assert runner.hands["noise"] == pytest.approx(0.2)


def test_writes_from_two_threads_do_not_lose_a_source():
    """The first version argued a lost update would last one frame because every writer rewrites
    its own key sixty times a second. No writer that exists does: the console writes only when
    the mouse moves, so a dropped dial would have been dropped for good."""
    import threading

    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

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


def test_a_knob_holds_a_dial_through_the_same_seam_a_hand_does():
    """The last piece of "the Rytm controls the knobs": no hole is cut in the frame loop for
    the hardware, and everything the strip already shows about a held dial shows an encoder's
    holds for free."""
    from ganlive.control.midi import EncoderMap, parse_controls
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    knobs = EncoderMap(parse_controls("16=noise, 2:17=se_256"))

    assert knobs.apply(runner, 5, 16, 0) == "noise"
    assert runner.hands["noise"] == pytest.approx(0.0)
    assert knobs.apply(runner, 5, 16, 127) == "noise"
    assert runner.hands["noise"] == pytest.approx(1.0)

    assert knobs.apply(runner, 5, 17, 127) is None
    assert "se_256" not in runner.hands
    assert knobs.apply(runner, 1, 17, 64) == "se_256"
    assert runner.hands["se_256"] == pytest.approx(64 / 127)

    assert set(runner.hands) == {"noise", "se_256"}

    assert knobs.unmapped == {(5, 17): 1}

    knobs.release(runner, "noise")
    assert set(runner.hands) == {"se_256"}
    knobs.release(runner)
    assert runner.hands == {}


def test_a_hand_on_the_strip_beats_a_knob_parked_on_the_same_dial():
    """Reaching for a control on screen means to override whatever the hardware is parked on,
    and that is a decision rather than a consequence of who moved first."""
    from ganlive.strip import PRIORITY, SOURCE
    from ganlive.control.midi import EncoderMap, parse_controls
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    knobs = EncoderMap(parse_controls("16=noise"))
    knobs.apply(runner, 0, 16, 127)
    runner.hold(SOURCE, {"noise": 0.25}, PRIORITY)
    assert runner.hands["noise"] == pytest.approx(0.25)

    runner.free(SOURCE, "noise")
    assert runner.hands["noise"] == pytest.approx(1.0)


def test_a_knob_mapping_names_any_dial_and_is_checked_against_the_model_that_plays():
    """**Which dials exist is the loaded model's business, and this parses before one opens.**

    Checking a mapping against this project's own `DIALS` refused every converted StyleGAN2's
    whole MODEL block from `--cc` and `--pressure` -- and did worse than refuse to a learned
    one. `l` on `w_fine` binds and `flush` writes `1:16=w_fine` in the flag's own words; the
    next launch read it back, raised here, and `rytm_live.remembered` swallowed it. The learn
    was undone and the only trace was one line about a file that "does not parse"."""
    from ganlive.control.midi import EncoderMap, format_controls, parse_controls
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    assert parse_controls("") == {}
    assert parse_controls("16=noise") == {(-1, 16): "noise"}
    assert parse_controls(" 3:22 = se_64 ") == {(2, 22): "se_64"}
    assert parse_controls("16=w_fine") == {(-1, 16): "w_fine"}, (
        "a converted StyleGAN2's own dial, which this used to refuse outright")
    with pytest.raises(ValueError):
        parse_controls("16=")               # a mapping with no dial at all is still a typo

    # The whole round trip a learn takes: bound on the strip, written down, read back.
    knobs = EncoderMap({})
    knobs.learning = "w_fine"
    knobs.apply(PresetRunner(FIXTURES["still"], INDEX, 60.0), 0, 16, 64)
    assert parse_controls(format_controls(knobs.controls)) == knobs.controls

    # And the check that used to live here, moved to where the model is known.
    sg2 = _surface.stylegan2(("w_fine",), (0.5,), ((),), (25.0,))
    assert not EncoderMap(parse_controls("16=w_fine")).unreachable(sg2)
    stray = EncoderMap(parse_controls("16=se_256,17=punch")).unreachable(sg2)
    assert "se_256" in stray and "punch" in stray, (
        "a dial this model does not have, and a name no model ever will, read the same here "
        "and are both worth saying")
    assert not EncoderMap({}).unreachable(sg2), "nothing wired is nothing to report"


def test_what_p_prints_can_be_pasted_back_and_gives_the_same_setting():
    """A discovery that cannot be written down ends when the window closes. `p` used to print
    the held dials alone, which was right while a session could only find values -- the grid can
    change which drum drives which dial now, so it prints the rules too, from the runner's own
    copy, which is what was playing rather than what the file said."""
    from ganlive.strip import DialPanel
    from ganlive.presets import Impulse, Macro, Preset, PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["release"], INDEX, 60.0)
    panel = DialPanel(runner)
    panel.set("noise", 0.42)
    runner.route("CH", "se_64")

    text = panel.as_patch()
    scope = {"Preset": Preset, "Impulse": Impulse, "Macro": Macro}
    exec(text, scope)                                       # noqa: S102 - that is the test
    back = scope["RELEASE"]

    assert back.dials["noise"] == pytest.approx(0.42), "a discovery survives the round trip"
    assert back.dials["hold"] == pytest.approx(FIXTURES["release"].dials["hold"])
    wired = {(i.track, i.dial) for i in back.impulses}
    assert ("CH", "se_64") in wired, "and so does a rule made on the grid"
    assert ("BD", "hold") in wired, "without losing the ones that were already there"
    assert len(back.macros) == len(FIXTURES["release"].macros)

    again = PresetRunner(back, INDEX, 60.0)
    since = np.full(len(INDEX), 1e6, dtype=np.float32)
    a, b = FakeKnobs(), FakeKnobs()
    runner.apply(since, {"density": 5.0}, a)
    again.apply(since, {"density": 5.0}, b)
    assert a.written == pytest.approx(b.written)
    assert runner.surface.values == pytest.approx(again.surface.values)


def _pngs(folder):
    import hashlib
    from pathlib import Path

    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).glob("*.png"))}


def test_every_shipped_setting_survives_being_written_down_and_read_back():
    from ganlive.presets import from_dict, to_dict

    for name, preset in FIXTURES.items():
        again = from_dict(to_dict(preset))
        assert again == preset, name


def test_a_saved_setting_is_in_the_rotation_the_next_time_it_starts(tmp_path):
    from dataclasses import replace

    from ganlive.presets import Library, PresetRunner
    from ganlive.control.tracks import INDEX

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
    import json

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
    from ganlive.presets import POSITIONS_NAME, Library, PresetRunner, Positions
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold("hand", {"noise": 0.6})
    positions = Positions(tmp_path / POSITIONS_NAME)
    positions.stash("gv-2048-ft-72000", runner, "hand", ["noise"])

    library = Library(tmp_path)
    assert library.broken == [], library.broken
    assert POSITIONS_NAME.removesuffix(".json") not in library.names


def test_a_take_drops_frames_rather_than_making_the_picture_wait():
    """x264 does not encode six megapixels at 60 fps here, so a blocking put would pace the
    whole instrument to the encoder. No thread is started: this is the queue policy alone."""
    import numpy as np

    from ganlive.record.video import Recorder

    rec = Recorder("unused.mp4", 8, 8, 60.0, realtime=True, depth=2)
    plane = np.zeros((12, 8), dtype=np.uint8)
    assert rec.offer(plane) is True
    assert rec.offer(plane) is True
    assert rec.offer(plane) is False, "a full queue blocked instead of dropping"
    assert rec.dropped == 1 and rec.offered == 3

    slow = Recorder("unused.mp4", 8, 8, 60.0, depth=2)
    assert [slow.offer(plane) for _ in range(2)] == [True, True]
    assert slow.dropped == 0


def test_a_dropped_frame_costs_the_take_a_held_frame_not_its_length():
    """With timestamps counted rather than clocked, dropped frames make the file play back
    faster than it was played -- fifteen minutes of performance as eleven minutes of video,
    which is the kind of wrong that gets believed."""
    import time

    import numpy as np

    from ganlive.record.video import Recorder

    rec = Recorder("unused.mp4", 8, 8, 60.0, realtime=True, depth=64)
    plane = np.zeros((12, 8), dtype=np.uint8)
    rec.offer(plane)
    time.sleep(0.1)                        # six frames of wall clock at 60 fps
    rec.offer(plane)
    stamps = [rec._q.get()[0] for _ in range(2)]
    assert stamps[0] == 0
    assert stamps[1] >= 4, f"a 100 ms gap became {stamps[1]} frames"

    counted = Recorder("unused.mp4", 8, 8, 60.0, depth=64)
    counted.offer(plane)
    time.sleep(0.05)
    counted.offer(plane)
    assert [counted._q.get()[0] for _ in range(2)] == [0, 1]


def test_a_take_is_a_playable_file_with_a_trailer(tmp_path):
    import numpy as np

    from ganlive.record.video import Recorder

    out = tmp_path / "take.mp4"
    rec = Recorder(out, 128, 64, 30.0, "libx264", realtime=True, depth=8).start()
    rng = np.random.default_rng(0)
    for _ in range(20):
        rec.offer(np.ascontiguousarray(rng.integers(0, 255, (96, 128), dtype=np.uint8)))
    report = rec.stop()
    assert "error" not in report, report
    assert report["frames"] == 20 - report["dropped"], report
    assert report["offered"] == 20, report
    assert out.exists() and out.stat().st_size > 0

    import av

    with av.open(str(out)) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded == report["frames"], (decoded, report)


def test_a_still_is_its_own_copy_rather_than_a_buffer_a_later_frame_overwrites():
    """A PNG is a few hundred milliseconds of zlib on another thread and every ring here is
    three deep, so a capture that borrowed one would be written half from a later picture."""
    import torch

    from ganlive.frame import FrameStage

    stage = FrameStage(8, 8, device="cpu")
    first = stage.rgb_still(torch.full((1, 3, 8, 8), -1.0))
    second = stage.rgb_still(torch.full((1, 3, 8, 8), 1.0))
    assert first.max() == 0 and second.min() == 255, (first.max(), second.min())


def test_a_still_is_written_by_the_time_its_thread_is_joined(tmp_path):
    import numpy as np

    from ganlive.record.video import save_still

    rgb = np.zeros((16, 24, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    path = tmp_path / "shot.png"
    save_still(path, rgb).join(timeout=20.0)

    from PIL import Image

    with Image.open(path) as img:
        assert img.size == (24, 16)
        assert img.convert("RGB").getpixel((0, 0)) == (200, 0, 0)


def test_serial_names_do_not_collide_and_sort_in_the_order_they_were_made(tmp_path):
    from ganlive.record.video import next_path

    made = []
    for _ in range(3):
        path = next_path(tmp_path, "take", ".mp4")
        path.write_bytes(b"")
        made.append(path)
    assert [p.name for p in made] == ["take-01.mp4", "take-02.mp4", "take-03.mp4"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [p.name for p in made]


def test_the_stand_in_can_be_heard_and_measured_from_the_same_sample_index():
    """Two clocks over one recording would agree at the start of a set and be half a bar apart
    by the end of it, so the picture would answer a kick that had already gone."""
    import numpy as np

    from ganlive.control.features import FeatureExtractor
    from ganlive.control.tracks import MonitorFeeder

    stems = np.zeros((2, 1000), dtype=np.float32)
    stems[0, 500] = 1.0
    mix = np.zeros((2, 1000), dtype=np.float32)
    mix[:, 500] = 0.5
    feeder = MonitorFeeder(FeatureExtractor(2, 48000), stems, mix, 48000, 256)

    out = np.zeros((256, 2), dtype=np.float32)
    heard, at = [], []
    for _ in range(3):
        at.append(feeder.at)
        feeder._block(out, 256, None, None)
        heard.append(float(np.abs(out).max()))
    assert at == [0, 256, 512]
    assert heard[1] == 0.5 and heard[0] == 0.0
    assert feeder.calls == 3


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


def test_the_encoder_can_be_made_to_open_before_the_clock_starts(tmp_path):
    """An encoder is not opened by `add_stream` -- it opens on the first frame it is given, and
    `av1_qsv` takes about a second and a half to do it while holding the GIL, so the thread
    producing frames stops wherever it happens to be. That is why the frame timings
    under-reported it by five: it lands between two frames as often as inside one. `drain` is
    how a caller pays it before the clock starts."""
    import numpy as np

    from ganlive.record.video import Recorder

    rec = Recorder(tmp_path / "warm.mp4", 64, 64, 30.0, "libx264",
                   realtime=True, depth=4).start()
    plane = np.zeros((96, 64), dtype=np.uint8)
    assert rec.offer(plane) is True
    assert rec.drain(timeout=30.0) is True, "the writer never caught up"
    assert rec.written == 1, rec.written

    rec.offer(plane)
    assert rec.drain(timeout=30.0) is True
    report = rec.stop()
    assert report["frames"] == 2 and report["offered"] == 2, report


def test_both_sources_share_one_track_space_and_the_pairs_come_apart():
    """**Neither source alone covers a live set on this machine.** The sequencer sends no notes
    at all and playing its trigs is how the instrument is used; the pads send an exact note per
    drum and are the only thing that can separate the four pairs sharing an analog voice. So the
    index space is the NOTE's, the coarse one maps into it, and an onset on a shared voice fires
    both of its tracks -- which is what the audio genuinely says and all it can say."""
    from ganlive.control.features import BothFeatures, FeatureExtractor
    from ganlive.control.tracks import INDEX, parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
    both = BothFeatures(FeatureExtractor(10, 48000), tracks)
    assert both.n == 12, "the space is the channel map's ten, not the twelve tracks"
    assert both.channel_of() == dict(INDEX)

    both.audio.pending.append((8, 0.8, 0.0))
    both.tick(1.0)
    assert sorted(i for i, _v, _a in both.drain()) == sorted([INDEX["CH"], INDEX["OH"]])

    both.on_note(13, INDEX["OH"], 100, when=2.0)
    both.tick(2.0)
    assert [i for i, _v, _a in both.drain()] == [INDEX["OH"]]
    both.tick(2.5)
    assert both.since[INDEX["CH"]] > both.since[INDEX["OH"]], "the voice's other track fired"
    assert float(both.since[INDEX["OH"]]) == pytest.approx(0.5, abs=1e-4)


def test_the_channel_map_is_remembered_so_it_stops_living_in_a_document(tmp_path):
    """**A missing --map is silent and looks exactly like a routing bug.** `rytm_map.py` discovers which drum
    arrives on which Overbridge channel, prints it, and forgets it, so the working line lived in
    docs/RESUME.md and had to be retyped. Launched without it the tool falls back to
    `channel_map("tracks")` -- BD=0, SD=1, RS=2 -- while Overbridge starts the kit at 2, so every onset is
    credited to the wrong drum and the four channels carrying a pair light two tracks for one hit. That is
    what it looks like from the front, and it cost a 467-second session."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("rytm_live_for_test",
                                                  pathlib.Path("src/ganlive/tools/play.py"))
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    live.CHANNELS = tmp_path / "channels.txt"

    overbridge = "BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9"

    guessed, note = live.resolve_map("", "tracks")
    assert guessed["BD"] == 0, "the default layout starts the kit at 0"
    assert "GUESS" in note, note

    given, note = live.resolve_map(overbridge, "tracks")
    assert given["BD"] == 2 and given["CH"] == given["OH"] == 8
    assert live.CHANNELS.read_text(encoding="utf-8").strip() == overbridge

    recalled, note = live.resolve_map("", "tracks")
    assert recalled == given, "the remembered map did not come back the same"
    assert "GUESS" not in note, note

    live.CHANNELS.write_text("BD=nonsense", encoding="utf-8")
    fell_back, note = live.resolve_map("", "tracks")
    assert fell_back["BD"] == 0 and "GUESS" in note, note


def test_the_frame_report_says_which_model_the_slow_frames_belong_to():
    """**One aggregate over a bank cannot be read.** A played 338s session switched models and
    then loaded a StyleGAN2 from the shelf, and reported 53.8 fps with 25.7% of frames over
    budget as a single number -- which is either one expensive model or every model getting
    slower, and the report could not tell them apart."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("rytm_live_for_report",
                                                  pathlib.Path("src/ganlive/tools/play.py"))
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)

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


def test_a_handler_that_raises_drops_one_message_and_not_every_message_after_it():
    """**This is how a live set goes deaf halfway through and nothing says so.** `run` had no guard: a handler
    raising unwound the thread, `run` returned, and every clock, note, knob and pad after that moment was
    gone for the rest of the session. The end-of-run counts then show whatever arrived before the fault, so
    a run that died at minute two reads exactly like a machine that was never sending -- and the fix it
    reports, CLOCK SEND = ON, is the wrong one."""
    from ganlive.control.midi import NOTE_ON, ClockReader, MusicalClock

    seen = []

    def explodes(_channel, note, _velocity):
        seen.append(note)
        if note == 2:
            raise ValueError("boom")

    reader = ClockReader(MusicalClock(), on_note=explodes)

    class _OneBurst:
        """One burst, then it stops the reader -- `run` polls until `stop_flag`."""

        def __init__(self, events):
            self._events = list(events)

        def poll(self):
            if not self._events:
                reader.stop_flag = True
                return False
            return True

        def read(self, _n):
            out, self._events = self._events, []
            return [[e, 0] for e in out]

    reader.open_ports = lambda: [
        ("fake", _OneBurst([[NOTE_ON, 1, 100], [NOTE_ON, 2, 100], [NOTE_ON, 3, 100]]))]
    reader.run()

    assert seen == [1, 2, 3], "the messages after the raising one never arrived"
    assert reader.counts.get("note_on") == 3, "counting stopped at the fault"
    assert reader.trouble().startswith("1 MIDI message"), reader.trouble()
    assert "boom" in reader.trouble(), reader.trouble()


def test_every_source_answers_everything_the_end_of_run_report_asks_it():
    """The report reached for `.density` and `.energy`, which only the two single-source extractors have.
    `BothFeatures` is the DEFAULT -- `--triggers both` -- so the line raised at the end of every
    performance, after the playing was over, and took the whole report with it: the tracks that played, the
    hit counts, the diagnosis of what to fix. The one run that needed it most was the one where nothing
    arrived."""
    from ganlive.control.features import BothFeatures, FeatureExtractor, NoteFeatures
    from ganlive.control.tracks import parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
    sources = {
        "audio": FeatureExtractor(10, 48000),
        "midi": NoteFeatures(12),
        "both": BothFeatures(FeatureExtractor(10, 48000), tracks),
    }
    for kind, src in sources.items():
        assert isinstance(src.played(), int), kind
        assert isinstance(src.n, int), kind
        got = src.features()
        assert set(got) == {"density", "energy", "active"}, f"{kind} answered {sorted(got)}"
        assert all(isinstance(v, float) for v in got.values()), kind
        src.channel_of()                      # may be empty; must not raise
        assert len(list(src.hits)) == src.n, kind


def test_a_pad_that_is_heard_as_well_as_read_counts_once():
    """A pad struck by hand arrives twice -- the note over USB MIDI, early, and the sound
    through Overbridge a few milliseconds later. Counting both would push every dial a rule
    drives twice as far for a hand hit as for a sequencer trig, which is a difference nobody
    asked for and nothing would report."""
    from ganlive.control.features import PAIRED_S, BothFeatures, FeatureExtractor
    from ganlive.control.tracks import INDEX, parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
    both = BothFeatures(FeatureExtractor(10, 48000), tracks)

    both.on_note(13, INDEX["CH"], 100, when=5.0)          # the note, first
    both.audio.pending.append((8, 0.8, 0.0))              # ...and its sound, just after
    both.tick(5.0 + PAIRED_S * 0.5)
    assert [i for i, _v, _a in both.drain()] == [INDEX["CH"]], "the same hit was counted twice"

    both.audio.pending.append((8, 0.8, 0.0))
    both.tick(5.0 + PAIRED_S * 2.0)
    assert sorted(i for i, _v, _a in both.drain()) == sorted([INDEX["CH"], INDEX["OH"]])


def test_the_combined_source_answers_everything_a_source_is_asked():
    """The frame loop reads `n`, `since`, `drain()` and `features()`; the guide wants `tap` and
    `sr`; the summary wants `played()`, `hits` and `heard()`. A wrapper that answered most of
    them would fail at the one call site that used the rest, and several of those run once at
    the end of a performance -- the worst possible time to learn about it."""
    from ganlive.control.features import BothFeatures, FeatureExtractor
    from ganlive.control.tracks import parse_channel_map

    tracks = parse_channel_map("BD=2,SD=3,RS=4,CP=4,BT=5,LT=6,MT=7,HT=7,CH=8,OH=8,CY=9,CB=9")
    audio = FeatureExtractor(10, 48000)
    both = BothFeatures(audio, tracks)

    for name in ("n", "since", "hits", "sr", "tap"):
        assert hasattr(both, name), name
    for name in ("drain", "features", "played", "heard", "channel_of", "push", "tick",
                 "on_note"):
        assert callable(getattr(both, name, None)), name
    assert set(both.features()) == {"density", "energy", "active"}
    assert both.sr == 48000 and len(both.since) == 12 and len(both.hits) == 12

    both.tap = print
    assert audio.tap is print, "the tap landed on the wrapper instead of the audio"
    both.push(np.zeros((10, 256), dtype=np.float32))
    assert both.heard() > 0.0, "heard() did not reach the audio half"


def test_pressure_is_classified_at_all_which_it_previously_was_not():
    """`dispatch` had no branch for polyphonic aftertouch, so the one continuous gesture the
    machine has could never reach anything downstream. Measured on the wire at 122 messages a
    minute from ordinary playing, and dropped every one of them."""
    from ganlive.control.midi import AFTERTOUCH_POLY, dispatch
    from ganlive.walk import MusicalClock

    clock = MusicalClock()
    assert dispatch(clock, AFTERTOUCH_POLY, 3, 90) == "aftertouch_poly"
    assert dispatch(clock, AFTERTOUCH_POLY + 0x0D, 3, 90) == "aftertouch_poly", "any channel"
    assert clock.beats == 0.0


def test_a_pad_leaned_on_holds_a_dial_and_gives_it_back_when_released():
    """**The one real difference from a knob**, and why this is a subclass rather than a flag:
    a knob parked at zero means zero, a pad nobody is touching means nothing. Everything else
    is `EncoderMap`'s code and is not copied."""
    from ganlive.control.midi import EncoderMap, PressureMap, parse_pressure
    from ganlive.presets import Preset, PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(Preset(name="t", blurb="", dials={"noise": 0.1, "se_256": 0.1}), INDEX,
                         60.0)
    pads = PressureMap(parse_pressure("BD=noise,SD=se_256"))
    assert pads.controls == {(-1, INDEX["BD"]): "noise", (-1, INDEX["SD"]): "se_256"}

    assert pads.apply(runner, 13, INDEX["BD"], 127) == "noise"
    assert runner.held_by(pads.source) == {"noise": 1.0}
    pads.apply(runner, 13, INDEX["BD"], 64)
    assert runner.held_by(pads.source)["noise"] == pytest.approx(64 / 127)
    assert pads.apply(runner, 13, INDEX["BD"], 0) == "noise"
    assert runner.held_by(pads.source) == {}
    assert pads.seen == 3, "held is current state and cannot say the machine ever spoke"

    pads.apply(runner, 13, INDEX["CH"], 100)
    assert pads.unmapped == {(13, INDEX["CH"]): 1}

    knobs = EncoderMap({(-1, 35): "noise", (-1, 36): "se_256"})
    assert knobs.source != pads.source, "a source cannot outrank itself"
    knobs.apply(runner, 13, 35, 20)
    knobs.apply(runner, 13, 36, 127)
    pads.apply(runner, 13, INDEX["BD"], 127)
    assert runner.hands["noise"] == 1.0, "the parked knob won a dial being actively squeezed"
    assert runner.hands["se_256"] == 1.0, "the squeeze wiped a knob it was not touching"
    pads.apply(runner, 13, INDEX["BD"], 0)
    assert runner.hands["noise"] == pytest.approx(20 / 127)


def test_a_pressure_wiring_refuses_a_name_that_is_not_a_track():
    """A track typo is a pad that can never fire, and no model can make it one. A dial typo is
    reported against the loaded model instead -- see the knob mapping's own test for why."""
    from ganlive.control.midi import parse_pressure
    from ganlive.control.tracks import INDEX

    assert parse_pressure("BD=w_fine") == {(-1, INDEX["BD"]): "w_fine"}
    with pytest.raises(ValueError, match="unknown track"):
        parse_pressure("XX=noise")
    with pytest.raises(ValueError):
        parse_pressure("BD=")


def test_a_track_channel_map_is_written_one_based_and_read_zero_based():
    """A person writes the channel the machine's screen shows; the comparison happens against
    `status & 0x0F`. `midi.parse_controls` resolves the same mismatch the same way."""
    from ganlive.control.tracks import INDEX, TRACKS, parse_track_channels

    whole = parse_track_channels("1-12")
    assert whole == {i: i for i in range(len(TRACKS))}, whole
    assert parse_track_channels("1=BD,2=SD") == {0: INDEX["BD"], 1: INDEX["SD"]}
    with pytest.raises(ValueError, match="11 channels"):
        parse_track_channels("1-11")
    with pytest.raises(ValueError, match="already BD"):
        parse_track_channels("3=BD,3=SD")


def test_the_track_a_note_is_comes_from_the_channel_when_the_machine_sends_that_way():
    """**The machine has two output modes and they identify a track differently.** On AUTO CH
    the kit shares one channel and the note says which track -- 0 to 11, what the pads send. On
    TRACK CH each track has its own channel and the note carries the trig's PITCH, 12 to 60,
    which names no track at all. Reading the note in that mode indexes the kit with a pitch."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.tracks import parse_track_channels

    auto = NoteFeatures(12)
    auto.on_note(13, 5, 100)                       # pad: channel 14 on the wire, note 5
    assert auto.hits[5] == 1 and auto.hits.sum() == 1

    auto.on_note(13, 47, 100)
    assert auto.hits.sum() == 1, "a pitch was read as a track index"

    per_track = NoteFeatures(12, channels=parse_track_channels("1-12"))
    per_track.on_note(5, 47, 100)                  # track 6, pitch 47 -- the pitch is not read
    per_track.on_note(5, 12, 64)
    assert per_track.hits[5] == 2 and per_track.hits.sum() == 2
    per_track.on_note(15, 40, 100)
    assert per_track.hits.sum() == 2


def test_the_pair_source_claims_a_voice_by_the_same_rule_the_notes_use():
    """`BothFeatures` had its own copy of `note - base_note`. It would have kept working while
    `NoteFeatures` learned about channels, so the pad-beats-onset rule would have gone on
    reading pitches and every trig would have been counted twice."""
    from ganlive.control.features import PAIRED_S, BothFeatures, FeatureExtractor
    from ganlive.control.tracks import INDEX, parse_channel_map, parse_track_channels

    tracks = parse_channel_map("BD=0,SD=1,RS=2,CP=2,BT=3,LT=3,MT=4,HT=4,CH=5,OH=5,CY=6,CB=6")
    audio = FeatureExtractor(7, 48000)
    both = BothFeatures(audio, tracks, channels=parse_track_channels("1-12"))

    assert both.on_note(7, 53, 110, when=1.0) == INDEX["HT"]
    assert both.hits[INDEX["HT"]] == 1

    audio.pending.append((4, 0.8, 0.0))
    both.tick(1.0 + PAIRED_S * 0.5)
    assert both.hits[INDEX["HT"]] == 1 and both.hits[INDEX["MT"]] == 0


def test_a_note_that_names_no_track_is_counted_rather_than_dropped():
    """**Both ways of setting the mode wrongly are otherwise silent, and one of them looks
    like a working run.** With the map omitted on TRACK CH every pitch falls out of spread and
    the wire reads as dead. With it supplied on AUTO CH the whole kit shares one channel, so
    all twelve drums resolve to ONE index -- `played()` is 1, the hit watchdog stays quiet, and
    the picture runs with one drum driving what twelve should."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.tracks import parse_track_channels

    missing = NoteFeatures(12)                     # AUTO CH reader given TRACK CH pitches
    for note in (47, 53, 60):
        assert missing.on_note(5, note, 100) is None
    assert missing.played() == 0 and missing.unresolved == 3
    assert missing.unclaimed == {(5, 47): 1, (5, 53): 1, (5, 60): 1}

    for note in range(200):
        missing.on_note(9, note + 64, 100)
    assert len(missing.unclaimed) == 32 and missing.unresolved == 203

    wired = NoteFeatures(12, channels=parse_track_channels("1-12"))
    assert wired.on_note(0, 47, 100) == 0 and wired.unresolved == 0


def test_the_two_output_modes_are_told_apart_by_the_traffic_itself():
    """The flag was very nearly the only way to know the mode. It is an override: the modes do
    not overlap in what they put on the wire, so a few seconds of it settles the question --
    except for the one genuinely ambiguous case, which must not be guessed."""
    from ganlive.control.tracks import output_mode

    assert output_mode({13: {0, 1, 5, 11}}) == "auto"           # pads, one channel, low notes
    assert output_mode({0: {47}, 1: {53}, 8: {36}}) == "track"  # pitches across many channels
    assert output_mode({}) is None
    assert output_mode({4: {47, 53}}) is None
    assert output_mode({13: {0, 5, 11}, 0: {47}, 1: {53}, 7: {36}}) == "mixed"


def test_a_channel_map_and_a_note_base_are_both_live_because_the_machine_uses_both():
    """**Read off the MKI manual, not guessed, and it overturned a guard added an hour before.**
    `TRK SEND MIDI` sends on the track's channel (11.8); `OUTPUT CH` selects auto or track for
    the PADS and knobs (13.4.2). Two settings, two sources: per-track channels with OUTPUT CH
    left on AUTO puts sequencer pitches on channels 1-12 and pad notes on the auto channel
    simultaneously, so a reader that insisted on one rule would drop half the kit."""
    from ganlive.control.features import NoteFeatures
    from ganlive.control.tracks import parse_track_channels

    both = NoteFeatures(12, channels=parse_track_channels("1-12"))
    assert both.on_note(6, 53, 100) == 6, "a mapped channel is the track, whatever the pitch"
    assert both.on_note(13, 9, 100) == 9, "the auto channel still names a track by its note"
    assert both.unresolved == 0
    assert both.on_note(13, 47, 100) is None
    assert both.unresolved == 1


class _TapSource:
    """The shape `Guide.listen_to` reads: a samplerate, a channel count and a free tap slot."""

    tap = None

    def __init__(self, n=10, sr=48000):
        self.n, self.sr = n, sr


def _guide_blocks(guide, count, channels=10, frames=256, level=0.5):
    """Push `count` blocks of the shape an ASIO callback hands over."""
    block = np.full((channels, frames), level, dtype=np.float32)
    for _ in range(count):
        guide.push(block)


def _drained(guide):
    """Wait for the writer thread to catch up, and hand the guide back."""
    for _ in range(500):
        if guide._q.empty():
            break
        time.sleep(0.002)
    return guide


def test_a_guide_is_the_length_that_was_played_and_cannot_clip(tmp_path):
    """**The one distortion that matters here is clipping**, because flattened peaks are
    exactly what a waveform match reads. Every channel is in [-1, 1] so their mean is too, and
    that is why this sums by averaging rather than adding: ten channels at full scale is the
    case where the obvious version writes a square wave and the file still looks fine."""
    import wave

    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=10))
    guide.start(tmp_path / "take-01.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    _guide_blocks(guide, 20, level=1.0)                   # every channel pinned at full scale
    _guide_blocks(guide, 20, level=-1.0)                  # ...and at the other end of it
    report = _drained(guide).stop()

    with wave.open(str(tmp_path / "take-01.wav")) as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 48000
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    assert len(pcm) == 40 * 256, f"{len(pcm)} samples for 40 blocks of 256"
    assert (pcm.max(), pcm.min()) == (32767, -32767), "the mean of ten full-scale channels " \
        "did not land on full scale, or wrapped past it"
    assert report["audio"]["seconds"] == round(40 * 256 / 48000, 3)
    assert report["audio"]["peak"] == 1.0


def test_the_guide_header_is_written_once_and_not_per_block(tmp_path):
    """`wave.writeframes` re-presets the RIFF header on EVERY call -- tell, seek, four bytes,
    seek, four bytes, seek back -- and each seek flushes the buffer, so a 512-byte block
    becomes a write syscall. Measured over one minute of audio: 12.26 us a block against 0.97
    for `writeframesraw`, which is what made the writer fall a queue behind and drop 28 blocks
    of a fourteen-second take. What `close` must still do is fix the header up, and that is
    what this pins -- the frame count on disk, read back by a reader that was not told it."""
    import wave

    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=2))
    guide.start(tmp_path / "take-07.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    _guide_blocks(guide, 9, channels=2)
    _drained(guide)
    wav = tmp_path / "take-07.wav"
    handed = 9 * 256 * 2
    assert wav.stat().st_size < handed // 2, (
        f"{wav.stat().st_size} bytes of {handed} on disk mid-take -- the header is being "
        f"patched per block again, and every preset flushes the buffer")
    guide.stop()
    with wave.open(str(wav)) as f:
        assert f.getnframes() == 9 * 256, "close did not fix the header up"


def test_the_audio_thread_drops_a_block_rather_than_waiting_for_the_disk(tmp_path):
    """A guide that made the sound card wait would cost dropped audio, and a dropped audio
    block is a missed HIT in the extractor rather than merely a gap in a file. No writer is
    started here: this is the queue policy alone, exactly as the frame recorder's is tested."""
    from ganlive.record.sync import Guide

    guide = Guide(depth=3)
    guide.attached = True                                 # ...but never started, so no writer
    guide._path = tmp_path / "take-01.mp4"
    _guide_blocks(guide, 5, channels=4)

    assert guide.blocks == 3, "the queue took more than it has room for"
    assert guide.dropped == 2, "a full queue blocked instead of dropping"
    assert guide.position == 5 * 256, "the stream position stopped at the dropped block"


def test_a_take_with_no_audio_still_gets_the_bar_lines(tmp_path):
    """`--triggers midi` and `--no-audio` open no input stream at all, and their takes still
    have a tempo, a start beat and a bar grid -- which alone places one in a DAW. Refusing to
    write the sidecar because half of it is unknown would take that away for nothing."""
    from ganlive.record.sync import Guide

    guide = Guide()                                       # never attached: no stream was open
    armed = guide.start(tmp_path / "take-02.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    assert "no audio input" in armed, armed
    side = guide.sidecar_path
    for beat in (0.0, 3.9, 4.0, 7.99, 8.0):
        guide.mark(beat)
    report = guide.stop()

    assert not (tmp_path / "take-02.wav").exists(), "an empty guide was written anyway"
    assert side.name == "take-02.json" and json.loads(side.read_text()) == report
    assert report["audio"] is None, "a run with no input claimed one"
    assert [m["beat"] for m in report["marks"]] == [4.0, 8.0]
    assert report["marks"][0]["audio_s"] is None
    assert report["drift_ms"] is None, "drift was reported with only one clock to read"


def test_the_sidecar_owns_its_own_field_names(tmp_path):
    """**Every field the spec asks for is a parameter of `start`, not a key in a dict handed
    over.** They were splatted at the top level for a while, so a caller with a key called
    `marks` or `audio` would have overwritten the sidecar's own and nothing would have said
    so -- and a reader written against the file would then be reading the call site."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.start(tmp_path / "take-08.mp4", bpm=99.6, beat=0.5417, beat_source="midi",
                about={"marks": "not the real ones", "model": "gv-warm-lr3"})
    report = guide.stop({"frames": 841})

    assert report["take"] == "take-08.mp4", "an absolute path where a name is enough"
    assert (report["bpm"], report["beat"], report["beat_source"]) == (99.6, 0.5417, "midi")
    assert report["marks"] == [], "a caller's key displaced the sidecar's own"
    assert report["about"]["marks"] == "not the real ones"
    assert report["video"] == {"frames": 841}
    assert report["started"][:2] == "20" and "T" in report["started"], report["started"]


def test_the_bar_lines_are_the_machine_s_and_not_the_take_s_own(tmp_path):
    """A mark exists to name a bar line the DAW also has. Counting four beats from wherever
    `v` happened to be pressed would put every mark a fraction of a bar off the grid, which is
    a beat map that is wrong everywhere and looks right."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.start(tmp_path / "take-03.mp4", bpm=130.0, beat=53.29, beat_source="midi")
    for beat in (53.29, 55.9, 56.1, 60.0, 63.5, 64.2):
        guide.mark(beat)
    assert [m["beat"] for m in guide.report()["marks"]] == [56.1, 60.0, 64.2]


def test_the_sidecar_anchors_the_take_on_the_audio_stream(tmp_path):
    """**Where the guide's first sample sits on the stream is the writer's answer, not the
    render thread's.** Reading the counter at `start` races the audio thread -- it may be
    mid-block, so the position read is one block out either way -- and it is the first block
    that actually lands in the file the sidecar has to describe. Two takes in one session are
    placed relative to each other by this number alone, with no waveform anywhere."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=4))
    _guide_blocks(guide, 7, channels=4)                   # played before `v` was pressed
    assert guide.position == 7 * 256 and guide.blocks == 0, "kept audio with no take running"

    guide.position += 256
    guide.start(tmp_path / "take-04.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    guide._q.put_nowait((7 * 256, time.perf_counter(),
                         np.zeros((4, 256), dtype=np.float32)))
    guide.blocks += 1
    _guide_blocks(guide, 3, channels=4)
    report = _drained(guide).stop()
    assert report["audio"]["start_sample"] == 7 * 256, \
        "the anchor was read from the counter on the render thread, one block out"
    assert report["audio"]["offset_ms"] > 0.0, report["audio"]
    assert not guide._q.qsize()


def test_a_mark_reads_both_clocks_at_one_instant_so_drift_is_measurable(tmp_path):
    """The frame clock is the system clock and the guide's is the sound card's crystal. Over a
    long take they part, and the video and its guide slide apart with them -- which is
    invisible in either file on its own. Both are read inside one `mark` for that reason."""
    from ganlive.record.sync import Guide

    guide = Guide()
    guide.listen_to(_TapSource(n=2))
    guide.start(tmp_path / "take-05.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    _guide_blocks(guide, 188, channels=2)                 # ~1.003 s of audio delivered
    _drained(guide).mark(4.0)
    report = guide.stop()

    mark = report["marks"][0]
    assert mark["audio_s"] == round(188 * 256 / 48000, 4), mark
    offset = report["audio"]["offset_ms"] / 1000.0
    assert report["drift_ms"] == round(
        (mark["video_s"] - mark["audio_s"] - offset) * 1000, 3)
    assert report["drift_ms"] < -500, report["drift_ms"]


def test_the_guide_hears_whatever_pushes_the_extractor(tmp_path):
    """**The tap is on the extractor, not on the sound card.** Three things push blocks into
    it -- the live tool's ASIO callback, `StemFeeder` and `MonitorFeeder` -- so a hook beside
    one of them is two copies waiting to be written and a `--simulate` path that quietly does
    not do it. This drives it through the stand-in, which is the only path testable here."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.record.sync import Guide

    ex = FeatureExtractor(4, 48000)
    guide = Guide()
    guide.listen_to(ex)
    assert (guide.sr, guide.channels) == (48000, 4), "the extractor's own numbers were not read"
    guide.start(tmp_path / "take-06.mp4", bpm=130.0, beat=0.0, beat_source="internal")
    ex.push(np.zeros((256, 4), dtype=np.float32))
    ex.push(np.zeros((4, 256), dtype=np.float32))
    report = _drained(guide).stop()
    assert report["audio"]["blocks"] == 2 and guide.position == 512
    assert report["audio"]["peak"] == 0.0
    assert "SILENT" in guide.describe(report), guide.describe(report)


def test_a_second_guide_is_refused_rather_than_displacing_the_first(tmp_path):
    """`FeatureExtractor.tap` holds one consumer. A second attach used to overwrite it, which
    leaves the first reporting on a stream that no longer reaches it -- a fault that counts
    itself as fine, and this project's signature failure."""
    from ganlive.control.features import FeatureExtractor
    from ganlive.record.sync import Guide

    ex = FeatureExtractor(2, 48000)
    Guide().listen_to(ex)
    with pytest.raises(RuntimeError, match="already has a tap"):
        Guide().listen_to(ex)


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


def test_a_deferred_block_waits_once_and_never_hands_out_a_buffer_that_has_not_landed():
    """Every download waits for the card on its own, so pulling one picture to two destinations
    flushed the whole pipeline twice a frame -- and the second flush waited on work it had only
    just queued. Measured while recording: 1.71 ms at 3072x2048, 1.14 at 2160x1440."""
    import torch

    from ganlive.frame import FrameStage

    stage = FrameStage(8, 8, device="cpu")
    waits = []
    stage.sync = lambda: waits.append(len(waits))

    frame = stage.step(torch.zeros(1, 3, 8, 8))
    with stage.deferred():
        a = stage.bgra_bytes(frame)
        b = stage.nv12_bytes(frame, "take", 4)
    assert len(waits) == 1, "two destinations should cost one wait, not two"
    assert a.shape == (8, 8, 4) and b.shape == (12, 8)

    with stage.deferred():
        pass
    assert len(waits) == 1, "an empty block waited for the card"

    stage.bgra_bytes(frame)
    assert stage._wait is True


def test_a_deferred_block_still_waits_when_its_body_raises():
    """A copy left in flight because the body raised would be read by whatever ran next."""
    import pytest
    import torch

    from ganlive.frame import FrameStage

    stage = FrameStage(8, 8, device="cpu")
    waits = []
    stage.sync = lambda: waits.append(1)
    frame = stage.step(torch.zeros(1, 3, 8, 8))

    with pytest.raises(RuntimeError), stage.deferred():
        stage.bgra_bytes(frame)
        raise RuntimeError("mid-frame")
    assert len(waits) == 1 and stage._wait is True and stage._pending is False


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


def test_the_worst_case_setting_turns_on_every_dial_that_costs_anything():
    """**The frame budget is priced against one setting**, so a dial missing from it is a per-frame cost
    nobody has measured -- which is what happened once when a stage dial was added and left out of the
    hand-written list."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    from ganlive.tools import latency as rytm_latency

    from ganlive.dials.table import DIALS, LATENT, SPANS

    worst = rytm_latency.WORST
    driven = {name for name, value in worst.dials.items() if value != DIALS[name][0]}
    driven |= {i.dial for i in worst.impulses} | {m.dial for m in worst.macros}

    wanted = {s.dial for s in SPANS} | set(LATENT) | {"noise"}
    missing = wanted - driven
    assert not missing, (
        f"{sorted(missing)} cost something every frame and the worst case leaves them at "
        f"rest, so the frame-budget worst case is not the worst case")


def test_no_status_line_runs_off_the_edge_of_the_strip():
    """**A status line you cannot read is not one**, which is what `_status_lines` says about the key list --
    and the key list was the only part measured. The preset line beside it was short enough to get away with
    until it started naming which hands are holding a dial, at which point `drag to set` ran off the right-
    hand edge and nothing said so."""
    import pygame

    from ganlive import strip as console
    from ganlive.strip import WIDTH, DialPanel
    from ganlive.control.midi import EncoderMap, PressureMap, parse_pressure
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold(console.SOURCE, {"dir1": 0.5}, console.PRIORITY)
    EncoderMap({(-1, 35): "noise"}).apply(runner, 13, 35, 90)
    PressureMap(parse_pressure("BD=se_256")).apply(runner, 13, INDEX["BD"], 110)
    assert len(set(runner.hands_from.values())) == 3, "the longest line needs all three hands"

    panel = DialPanel(runner, actions={a: (lambda *a_: None) for a in console.ACTIONS},
                      shelf=object())
    panel._pg = pygame
    pygame.font.init()
    panel._small = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 13)
    panel._size = (WIDTH, 900)
    room = WIDTH - 2 * console.PAD

    for mode in (console.MODE_DIALS, console.MODE_ROUTING, console.MODE_MODELS):
        panel.mode = mode
        for line in panel._status_lines():
            assert panel._small.size(line)[0] <= room, (
                f"in {mode}: {line!r} is {panel._small.size(line)[0]}px in a "
                f"{room}px strip")


def test_no_dial_description_is_cut_off_by_the_block_that_shows_it():
    """**The one block whose whole job is to say what a control does**, and a dial whose prose outgrows it
    loses its last sentence with nothing to say so -- silently, and only for the one dial somebody happened
    to write at length. `chroma` arrived needing seven lines in a block that holds five, and the truncation
    was visible only by rendering the strip."""
    import pygame

    from ganlive.strip import DESC_H, PAD, WIDTH, wrap
    from ganlive.dials.table import DIALS

    pygame.font.init()
    body = pygame.font.SysFont("segoeui,dejavusans,arial", 13)
    room, most = WIDTH - 2 * PAD, (DESC_H - 22) // 16

    over = {name: len(wrap(prose, body, room, 99)) for name, (_rest, prose) in DIALS.items()
            if len(wrap(prose, body, room, 99)) > most}
    assert not over, f"cut off in a {most}-line block: {over}"


def test_use_model_relayouts_at_startup_not_only_on_a_switch():
    """The first model gets its own dials, the same as the fourth."""
    import dataclasses

    from ganlive.dials import table as S
    from ganlive.presets import Preset, PresetRunner

    @dataclasses.dataclass
    class _Model:
        layout: object
        dials_live: frozenset

    foreign = S.adopted(("gain_128",), (0.5,), ((0.0, 0.1), (1.0, 2.0)), (30.0,))
    runner = PresetRunner(Preset(name="t", blurb=""), {}, 60.0)
    assert "se_256" in runner.surface.layout, "the default surface is ours"

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


def test_the_walk_hands_an_onnx_graph_its_latent_on_the_host():
    """Same numbers, without the round trip to the card and back."""
    import numpy as np

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


def test_every_load_setting_reaches_every_model_the_bank_ever_loads():
    """**The bank grows after launch.** The shelf loads a model mid-session through `Bank.add`,
    so a setting `build` took and `add` did not would change the instrument under the hand
    with nothing on the strip to explain it. `exact` was exactly that: `build` accepted it and
    `Bank.add` called `_prepare` without it, so a StyleGAN2 added from the shelf ran half
    precision however it had been asked for. One object carries them all now, and this asserts
    the structure rather than the list -- a seventh setting cannot be forgotten the way the
    fourth was."""
    import inspect

    from ganlive.dials.derive import RANDOM_FLOOR
    from ganlive import bank as R

    assert R.LoadOptions().direction_floor == RANDOM_FLOOR, (
        "the default has to be the measured one, not a second opinion about it")
    assert dataclasses.is_dataclass(R.LoadOptions) and R.LoadOptions.__dataclass_params__.frozen, (
        "a mutable Load would let one model's settings follow the next one's")

    # Every load path takes the whole object, so none of them can take a subset of it.
    for fn in (R._prepare, R._prepare_onnx, R._prepare_stylegan2, R._prepare_fastgan):
        assert "load" in inspect.signature(fn).parameters, fn.__name__

    # The rig carries it, and `add` -- the shelf's path -- hands on that same object.
    assert [f.name for f in dataclasses.fields(R.Bank) if f.type == "LoadOptions"] == ["load"]
    assert "self.load" in inspect.getsource(R.Bank.add), (
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


# --- the load gate, learn, NRPN and per-model positions ------------------------------------


def test_the_gate_measures_every_writing_dial_through_the_path_a_hand_takes():
    """A dial the surface writes into a setting the model demodulates straight back out has to
    read as dead at load, not on stage; and the measurement has to go through `Knob.writes`,
    since that is the path a hand takes."""
    import torch

    from ganlive.dials import steer as K
    from ganlive import bank as R
    from ganlive.dials.table import SETTINGS_WRITTEN, fastgan

    knobs = K.Knobs(sorted(SETTINGS_WRITTEN), "cpu", torch.float32)

    class _Net(torch.nn.Module):
        def forward(self, z):
            # Only one gate reaches the picture; every other setting is divided back out.
            return torch.tanh(torch.zeros(1, 3, 8, 8) + (knobs.view("sle.se_256") - 1.0))

    layout = K.verify(_Net(), knobs, fastgan(), 4, "cpu", torch.float32)
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

    from ganlive.dials.table import Layout, fastgan

    layout = Layout(tuple(replace(k, measured=12.4) if k.name == "se_256" else k
                          for k in fastgan().knobs))
    panel = _panel(DIALS)
    panel.bank.current = _StubModel(dials_live=frozenset(DIALS), layout=layout)
    assert "12 8-bit levels" in panel._measured("se_256")
    assert panel._measured("speed") == ""
    assert panel._measured("grid") == "", "a motion dial has no measurement to report"


def test_nrpn_arrives_as_one_fourteen_bit_control():
    """Four CCs in, one control out, with the low byte as an update rather than a wait."""
    from ganlive.control.midi import DATA_LSB, DATA_MSB, NRPN_LSB, NRPN_MSB, Nrpn, nrpn_number

    n = Nrpn()
    assert n.feed(0, NRPN_MSB, 1) is None
    assert n.feed(0, NRPN_LSB, 3) is None
    assert n.feed(0, DATA_MSB, 64) == (nrpn_number(1, 3), 64 << 7, 16383)
    assert n.feed(0, DATA_LSB, 5) == (nrpn_number(1, 3), 64 << 7 | 5, 16383)
    assert n.feed(1, DATA_MSB, 9) == (6, 9, 127), "data entry with no parameter named is a plain CC"
    assert n.feed(0, 17, 9) is None, "an ordinary CC is not the state machine's business"


def test_the_reader_routes_nrpn_and_plain_ccs_to_the_same_handler():
    from ganlive.control.midi import CONTROL_CHANGE, ClockReader, nrpn_number

    got = []
    reader = ClockReader(MusicalClock(120.0), on_control=lambda *a: got.append(a))
    for cc, value in ((99, 1), (98, 3), (6, 64), (38, 5), (17, 100)):
        reader._handle([CONTROL_CHANGE + 2, cc, value, 0], 0.0)
    reader._handle([CONTROL_CHANGE + 5, 6, 50, 0], 0.0)     # data entry with no NRPN armed
    assert got == [(2, nrpn_number(1, 3), 64 << 7, 16383),
                   (2, nrpn_number(1, 3), 64 << 7 | 5, 16383),
                   (2, 17, 100, 127),
                   (5, 6, 50, 127)]


def test_the_knob_map_reads_nrpns_and_writes_itself_back_in_the_same_words():
    from ganlive.control.midi import format_controls, nrpn_number, number_name, parse_controls

    assert parse_controls("n1.3=dir1") == {(-1, nrpn_number(1, 3)): "dir1"}
    assert parse_controls("2:N1.3=dir1") == {(1, nrpn_number(1, 3)): "dir1"}
    assert number_name(nrpn_number(1, 3)) == "n1.3"
    assert number_name(17) == "17"
    controls = parse_controls("16=noise,2:17=se_256,n1.3=dir1")
    assert parse_controls(format_controls(controls)) == controls


def test_learn_binds_the_next_control_to_the_focused_dial_and_writes_it_down(tmp_path):
    """Click a dial, turn a knob: the pair is the map now, on disk, in `--cc`'s own words."""
    from ganlive.control.midi import EncoderMap, parse_controls
    from ganlive.presets import PresetRunner
    from ganlive.control.tracks import INDEX

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    saved = tmp_path / "cc.txt"
    knobs = EncoderMap({(-1, 16): "noise"}, remember=saved)

    knobs.learning = "se_256"
    assert knobs.apply(runner, 1, 17, 64) == "se_256", "bound and applied in the same turn"
    assert knobs.learning is None
    assert knobs.controls == {(-1, 16): "noise", (1, 17): "se_256"}
    assert not saved.exists(), "the reader's thread never touches the disk"
    knobs.flush()
    assert parse_controls(saved.read_text("utf-8")) == knobs.controls
    assert knobs.learned == ["2:17=se_256"]

    knobs.learning = "noise"
    knobs.apply(runner, 3, 20, 10)
    assert (-1, 16) not in knobs.controls, "a dial has one knob; relearning moves it"
    assert knobs.controls[(3, 20)] == "noise"

    knobs.learning = "dir1"
    knobs.learning = None
    assert knobs.apply(runner, 0, 99, 1) is None, "disarmed, so an unmapped control stays unmapped"

    assert knobs.apply(runner, 1, 17, 8191, top=16383) == "se_256"
    assert runner.hands["se_256"] == pytest.approx(0.5, abs=1e-3), "14-bit scales by its own top"


def test_the_strip_offers_l_only_when_there_are_knobs_to_learn():
    import pygame

    from ganlive import strip as console
    from ganlive.control.midi import EncoderMap

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
    """The spine is shared and stays under the hand; the MODEL block and the directions mean
    something different on every model, so they go with it and come back with it."""
    from ganlive.strip import PRIORITY, SOURCE
    from ganlive.presets import PresetRunner, Positions
    from ganlive.control.tracks import INDEX
    from ganlive.dials.table import fastgan, per_model

    names = per_model(fastgan())
    assert {"se_256", "noise", "dir1", "dir8"} <= set(names)
    assert {"speed", "grid", "reaction"}.isdisjoint(names)

    runner = PresetRunner(FIXTURES["still"], INDEX, 60.0)
    runner.hold(SOURCE, {"se_256": 0.9, "dir1": 0.1, "speed": 0.7}, PRIORITY)
    store = Positions(tmp_path / "positions.json")
    assert store.stash("a-1", runner, SOURCE, names) == {"se_256": 0.9, "dir1": 0.1}
    assert runner.hands == {"speed": 0.7}

    fresh = Positions(tmp_path / "positions.json")
    assert fresh.recall("b-2", runner, SOURCE, PRIORITY, names) == {}
    assert fresh.recall("a-1", runner, SOURCE, PRIORITY, names) == {"se_256": 0.9, "dir1": 0.1}
    assert runner.hands["se_256"] == pytest.approx(0.9)
    assert runner.hands["speed"] == pytest.approx(0.7), "and the spine was never touched"
    assert fresh.recall("a-1", runner, SOURCE, PRIORITY, ["dir1"]) == {"dir1": 0.1}, (
        "only onto dials the new model has")

    broken = tmp_path / "bad.json"
    broken.write_text("{", "utf-8")
    assert Positions(broken).trouble, "an unreadable file is reported, not raised"


def test_the_drum_lights_sit_under_the_loaded_model_s_rows_not_the_default_ones():
    """Seen in a picture, not in code: a model with thirteen MODEL dials had the light row drawn
    over its last dial, because the lights were laid out against the default layout."""
    from ganlive.strip import blocks, lights, rows
    from ganlive.dials.table import adopted

    names = [f"gain_{h}" for h in (4, 8, 16, 32, 64, 128, 256, 512, 1024)] + [
        f"noise_{h}" for h in (8, 32, 128, 512)]
    many = adopted(names, [0.5] * 13, [(0.5, 1.0, 2.0)] * 13, [25.0] * 13).groups
    for height in (1000, 1200, 1400):
        _name, top, tall = rows(height, many)[-1]
        boxes = lights(424, blocks(height, many)["lights"][0], 12)
        assert all(y == blocks(height, many)["lights"][0] + 6 for _x, y, _w, _h in boxes)
        assert all(y >= top + tall for _x, y, _w, _h in boxes), (height, boxes[0], top + tall)


def test_a_tall_window_gives_its_spare_height_to_the_description():
    from ganlive.strip import DESC_H, GROUPS, MIN_H, blocks

    at_floor = blocks(MIN_H, GROUPS)["desc"][1]
    assert at_floor == DESC_H
    assert blocks(1400, GROUPS)["desc"][1] == 2 * DESC_H, "capped at twice, the rest stays empty"
    assert DESC_H < blocks(MIN_H + 60, GROUPS)["desc"][1] < 2 * DESC_H
    for height in (MIN_H, MIN_H + 60, 1400):
        found = blocks(height, GROUPS)
        assert max(top + tall for top, tall in found.values()) <= height
