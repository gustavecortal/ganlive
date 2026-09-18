"""The knob vector fed from a graph's host twin, and the stage's own queue.

Both exist for one measured reason: an eager op on the generator's queue between two graph
replays makes every later replay slower, without bound. See `speedups.Replay`."""
from __future__ import annotations

import numpy as np
import torch

from ganlive.frame import FrameStage
from ganlive.dials.steer import Knobs


def test_a_fed_knob_vector_commits_to_the_host_and_sends_nothing():
    knobs = Knobs(["a", "b", "c"], "cpu", torch.float32)
    host = torch.ones(3)
    knobs.feed_from(host)
    knobs.set("b", 4.0)
    knobs.commit()
    assert np.array_equal(host.numpy(), [1.0, 4.0, 1.0]), "the commit landed in the twin"
    assert torch.equal(knobs.vec, torch.ones(3)), "and nothing was sent to the card itself"
    assert np.array_equal(knobs.committed(), host.numpy())
    skipped = knobs.skipped
    knobs.commit()
    assert knobs.skipped == skipped + 1, "an unchanged frame still sends nothing"


def test_the_stage_has_no_side_queue_where_there_are_no_streams():
    stage = FrameStage(4, 6, device="cpu")
    with stage.aside():
        pass                                        # a null context, not a raise
    out = stage.step(torch.zeros(1, 3, 8, 12))
    assert tuple(out.shape[-2:]) == (4, 6)
