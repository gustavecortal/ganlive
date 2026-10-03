"""The frame stage: what crosses the bus, and in which colour order."""

from __future__ import annotations

import contextlib

import pytest
import torch

from ganlive.frame import FrameStage
from ganlive.models.common import denormalise
from ganlive.pixels import to_bgra, to_rgb


def _waiting_stage():
    """An 8x8 stage on the CPU whose waits for the card are counted rather than made."""
    stage = FrameStage(8, 8, device="cpu")
    waits = []
    stage.sync = lambda: waits.append(1)
    return stage, waits


def test_the_stage_does_no_work_at_all_when_the_model_is_already_the_right_size():
    """The frame path's most valuable branch, since the size follows the model."""
    stage = FrameStage(4, 6, device="cpu")
    exact = torch.zeros(1, 3, 4, 6)
    assert stage.step(exact) is exact, "the frame was resampled to the size it already is"

    # Down through `area`, up through `bilinear`: `area` on the way up is nearest-neighbour.
    stage.resize(2, 3)
    assert stage.step(exact).shape == (1, 3, 2, 3)
    stage.resize(8, 12)
    ramp = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6).repeat(1, 3, 1, 1)
    up = stage.step(ramp)
    assert up.shape == (1, 3, 8, 12)
    assert len(up.unique()) > len(ramp.unique()), "an enlargement that only repeated pixels"


def test_the_stage_shrinks_to_its_size_and_has_no_side_queue_without_streams():
    stage = FrameStage(4, 6, device="cpu")
    with stage.aside():
        pass                                        # a null context, not a raise
    out = stage.step(torch.zeros(1, 3, 8, 12))
    assert tuple(out.shape[-2:]) == (4, 6)


def test_a_stage_reports_which_conversions_are_compiled_rather_than_implying_it():
    """All three fall back to an eager chain that still produces the right picture several
    times slower, so the fallback has to be visible."""
    bare = FrameStage(8, 12)
    assert bare.compiled == {"yuv": False, "rgb": False, "bgra": False}
    frame = bare.step(torch.zeros(1, 3, 8, 12))
    assert bare.bgra_bytes(frame).shape == (8, 12, 4), "the fallback still produces a frame"
    assert bare.rgb_bytes(frame).dtype.name == "uint8"

    marked = FrameStage(8, 12, to_bgra=to_bgra)
    assert marked.compiled["bgra"] is True


def test_the_rgb_conversion_is_the_rounded_clamped_picture():
    """The card runs a compiled kernel and every test here runs the eager chain, so the eager
    one is pinned to the plain arithmetic."""
    x = torch.linspace(-1.2, 1.2, 3 * 8 * 12).reshape(1, 3, 8, 12)
    want = (denormalise(x.float()) * 255).round().clamp_(0, 255).to(torch.uint8)
    want = want.permute(0, 2, 3, 1)[0].contiguous()
    assert torch.equal(to_rgb(x), want)
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


def test_a_still_is_its_own_copy_rather_than_a_buffer_a_later_frame_overwrites():
    """A PNG is written on another thread and every ring here is three deep, so a still that
    borrowed one would be written half from a later picture."""
    stage = FrameStage(8, 8, device="cpu")
    first = stage.rgb_still(torch.full((1, 3, 8, 8), -1.0))
    second = stage.rgb_still(torch.full((1, 3, 8, 8), 1.0))
    assert first.max() == 0 and second.min() == 255, (first.max(), second.min())


def test_a_deferred_block_waits_once_and_never_hands_out_a_buffer_that_has_not_landed():
    """Pulling one picture to two destinations waits for the card once, not once each."""
    stage, waits = _waiting_stage()
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
    stage, waits = _waiting_stage()
    frame = stage.step(torch.zeros(1, 3, 8, 8))

    with pytest.raises(RuntimeError), stage.deferred():
        stage.bgra_bytes(frame)
        raise RuntimeError("mid-frame")
    assert len(waits) == 1 and stage._wait is True and stage._pending is False


def test_a_handoff_where_there_is_no_second_queue_is_the_deferred_wait():
    stage, waits = _waiting_stage()
    with stage.handoff() as sent:
        stage.bgra_bytes(stage.step(torch.zeros(1, 3, 8, 8)))
    assert waits == [1]
    sent.ticket.wait()
    stage.release()


class _Queue:
    """A second device queue that only writes down what was asked of it."""

    def __init__(self, log):
        self.log = log

    def wait_stream(self, _other):
        self.log.append("fence")


def _queued_stage(log):
    """A stage with a second queue, on a machine that has none: the events are bookkeeping."""
    class Event:
        def __init__(self):
            self.name = f"event{sum(1 for x in log if x.startswith('record'))}"

        def record(self, _queue):
            log.append(f"record {self.name}")

        def synchronize(self):
            log.append(f"wait {self.name}")

    class Streams:
        pass

    streams = Streams()
    streams.Event = Event
    streams.stream = lambda _q: contextlib.nullcontext()
    streams.current_stream = lambda: None
    stage = FrameStage(8, 8, device="cpu")
    stage._streams, stage._side = streams, _Queue(log)
    stage.sync = lambda: log.append("sync the card")
    return stage


def test_a_handed_off_frame_waits_for_its_own_copies_and_not_for_the_card():
    """What lets the next frame's generation run while this one downloads: the handoff leaves
    an event to wait on, where `deferred` waits for the whole device, generator included."""
    log = []
    stage = _queued_stage(log)
    frame = stage.step(torch.zeros(1, 3, 8, 8))
    with stage.handoff() as sent:
        stage.bgra_bytes(frame)
    assert "sync the card" not in log, log
    # Read before copied: the conversion's event is recorded ahead of the download's.
    assert log == ["fence", "record event0", "record event1"], log

    stage.release()                  # the generator may write its buffer again
    sent.ticket.wait()               # the bytes may be read
    sent.ticket.wait()               # and a second wait is free
    assert log[3:] == ["wait event0", "wait event1"], log
    stage.release()
    assert len(log) == 5, "a release with nothing converted since still waited"
