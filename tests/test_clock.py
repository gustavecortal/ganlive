"""The musical clock, and the shape of one move between two seeds."""

from __future__ import annotations

import random

import pytest

from ganlive.clock import MusicalClock, _ease_in, _ease_out, _smoothstep, shape
from tests.support import _pulses


def test_the_internal_clock_counts_beats_at_the_stated_tempo():
    c = MusicalClock(120.0)
    for _ in range(60):
        c.advance(1 / 60)                                 # one second
    assert c.beats == pytest.approx(2.0, abs=1e-6)        # 120 BPM is 2 beats a second


def test_beats_are_counted_from_pulses_so_jitter_cannot_reach_the_position():
    """USB delivery jitter perturbs pulse intervals, and so the tempo estimate; an integrated
    estimate would carry that error into the position. The pulse count does not."""
    c = MusicalClock(130.0)
    _pulses(c, 130.0, 20.0, jitter=0.002, rng=random.Random(7))
    assert c.beats == pytest.approx(20 * 130 / 60, abs=1e-9)


def test_the_tempo_estimate_converges_on_the_real_tempo():
    c = MusicalClock(60.0)
    now, period = 0.0, 60.0 / (140.0 * MusicalClock.PPQN)
    for _ in range(MusicalClock.PPQN * 8):
        now += period
        c.on_pulse(now)
    assert c.bpm == pytest.approx(140.0, rel=0.02)


def test_midi_pulses_take_over_from_the_internal_clock():
    """Once pulses arrive the position is theirs, so a driver can call `advance` every frame
    without checking who is in charge, and a dropped frame skips ahead instead of slowing."""
    c = MusicalClock(120.0)
    c.advance(1.0)
    assert c.source == "internal"
    for _ in range(MusicalClock.PPQN):
        c.on_pulse()
    assert c.source == "midi"
    assert c.beats == pytest.approx(1.0)                  # 24 pulses is one quarter note
    c.advance(10.0)
    assert c.beats == pytest.approx(1.0), "frame counting moved a clock the machine drives"


def test_stop_freezes_the_internal_clock():
    c = MusicalClock(120.0)
    c.advance(1.0)
    c.on_stop()
    c.advance(1.0)
    assert c.beats == pytest.approx(2.0)
    c.on_continue()
    c.advance(1.0)
    assert c.beats == pytest.approx(4.0)


@pytest.mark.parametrize("hold", [0.0, 0.3, 0.8, 0.95])
@pytest.mark.parametrize("when", [0.0, 0.5, 1.0])
def test_the_move_still_starts_at_one_seed_and_ends_at_the_next(hold, when):
    """Whatever the two motion dials are set to, the segment begins exactly on its own seed
    and finishes on the next one, which is what keeps arrival on the beat."""
    assert shape(0.0, hold, when) == pytest.approx(0.0, abs=1e-9)
    assert shape(1.0, hold, when) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("hold", [0.0, 0.4, 0.9])
@pytest.mark.parametrize("when", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_picture_never_goes_backwards(hold, when):
    """A curve that dipped would run the morph in reverse for part of the bar."""
    xs = [shape(i / 200, hold, when) for i in range(201)]
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:], strict=False)), (hold, when)


def test_hold_actually_stands_still_for_the_fraction_it_says():
    """At 0.8 with the standstill at the front, four fifths of the bar is one frozen image."""
    frozen = [shape(u / 100, 0.8, 1.0) for u in range(80)]
    assert max(frozen) == pytest.approx(0.0, abs=1e-12)
    assert shape(0.98, 0.8, 1.0) > 0.5                    # and then it moves, fast
    assert shape(0.5, 0.8, 0.0) == pytest.approx(1.0, abs=1e-9)


def test_no_standstill_means_the_picture_is_always_moving():
    """The control for the test above."""
    xs = [shape(u / 50, 0.0, 0.5) for u in range(1, 50)]
    assert all(b > a for a, b in zip(xs, xs[1:], strict=False))


def test_when_decides_whether_the_move_leaves_the_beat_or_arrives_on_it():
    """At 0 the picture is already moving as the beat lands; at 1 it has not started yet and
    finishes at speed."""
    eps = 1e-4
    leaving = shape(eps, 0.0, 0.0) / eps
    posing = shape(eps, 0.0, 0.5) / eps
    arriving = (1.0 - shape(1.0 - eps, 0.0, 1.0)) / eps
    assert leaving > 2.0, leaving
    assert posing < 0.01, posing
    assert arriving > 2.0, arriving


def test_the_three_classic_easing_curves_are_points_on_when():
    """Ease-out, smoothstep and ease-in are `when` at 0, 0.5 and 1, so one dial covers them."""
    for u in [i / 20 for i in range(21)]:
        assert shape(u, 0.0, 0.5) == pytest.approx(_smoothstep(u), abs=1e-12)
        assert shape(u, 0.0, 0.0) == pytest.approx(_ease_out(u), abs=1e-12)
        assert shape(u, 0.0, 1.0) == pytest.approx(_ease_in(u), abs=1e-12)
