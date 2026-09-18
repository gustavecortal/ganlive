"""The frame stage: what crosses the bus, and in which colour order."""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest
import torch

from ganlive.dials import table as _surface  # noqa: E402
from ganlive.dials.steer import Knobs
from ganlive.dials.table import (
    DIALS,
)
from ganlive.frame import FrameStage
from ganlive.walk import (
    SlerpWalk,
    WalkConfig,
)
from tests.support import _panel


def test_the_walk_stays_on_the_shell_between_seeds():
    """The great circle's whole justification: no midpoint sag. The straight line measured
    10.5 against endpoints of 15.2, and the frame there had 36% of their contrast."""
    w = SlerpWalk(256, "cpu", WalkConfig(beats_per_segment=4.0))
    ends = min(float(w.seed_for(0).norm()), float(w.seed_for(1).norm()))
    mids = [float(w.latent(t).norm()) for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]]
    assert min(mids) >= ends * 0.98, (min(mids), ends)


def test_building_a_walk_pays_every_first_call_cost_up_front():
    """A deferred import in a frame loop is 1.93 seconds of a performance's first bar."""

    name = "ganlive.models.fastgan"
    built = SlerpWalk(16, "cpu", WalkConfig(spread=0.3))
    saved = sys.modules[name]
    try:
        sys.modules[name] = None                          # any import of it now raises
        assert built.seed_for(3) is not None, "nothing in the loop may need the import again"
        assert built.latent(9.0) is not None
    finally:
        sys.modules[name] = saved


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


def test_a_still_is_its_own_copy_rather_than_a_buffer_a_later_frame_overwrites():
    """A PNG is a few hundred milliseconds of zlib on another thread and every ring here is
    three deep, so a capture that borrowed one would be written half from a later picture."""
    import torch

    from ganlive.frame import FrameStage

    stage = FrameStage(8, 8, device="cpu")
    first = stage.rgb_still(torch.full((1, 3, 8, 8), -1.0))
    second = stage.rgb_still(torch.full((1, 3, 8, 8), 1.0))
    assert first.max() == 0 and second.min() == 255, (first.max(), second.min())


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

