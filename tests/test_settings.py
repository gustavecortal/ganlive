"""The settings vector: how a frame's dial writes reach the card, and when they do not need to."""

from __future__ import annotations

import pytest
import torch

from ganlive.settings import Settings


def test_the_settings_vector_is_staged_through_a_ring_like_the_walk_is():
    """A copy from pinned memory is asynchronous, so a single host buffer could be overwritten
    by the next frame's writes while its copy is still in flight."""
    settings = Settings(["a", "b"], "cpu", torch.float32)
    assert settings.STAGING >= 2
    seen = []
    for value in (0.25, 0.5, 0.75, 1.0):
        settings.set("a", value)
        seen.append(settings.write.ctypes.data)
        settings.commit()
        assert settings.write[settings.index["a"]] == pytest.approx(value)
        assert settings.write[settings.index["b"]] == pytest.approx(1.0)
    assert len(set(seen)) == settings.STAGING, "consecutive frames must not share a buffer"


def test_the_settings_vector_is_not_resent_when_no_dial_moved():
    """The copy is tiny, but it releases the GIL, and getting it back from a busy window
    thread is what costs; at rest nothing moves, so nothing is sent."""
    k = Settings(["a", "b"], "cpu", torch.float32)
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
