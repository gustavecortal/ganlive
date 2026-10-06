"""ganlive on any machine: another card, a Mac, or no card at all.

The device and precision defaults, and the two compile failures that degrade to eager instead
of ending the load."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from ganlive import bank
from ganlive import device as dev
from ganlive.dials import steer as K
from ganlive.families import open_stylegan2
from ganlive.frame import FrameStage
from ganlive.models import capture as speedups
from ganlive.models import stylegan2 as S2


def test_half_on_a_card_and_single_on_the_cpu():
    assert dev.playback_dtype("cpu") is torch.float32
    for card in ("xpu", "cuda", "mps", "xpu:0", torch.device("cuda:1")):
        assert dev.playback_dtype(card) is torch.float16


def test_the_mps_module_is_a_backend_too():
    """`synchronize` on a Mac has to wait, so the lookup must not return None there."""
    assert dev._mod("mps") is getattr(torch, "mps", None)
    assert dev._mod("cpu") is None


def test_no_memory_report_where_there_is_no_allocator():
    assert dev.memory_report("cpu") is None


def test_the_stage_and_the_bank_default_to_the_detected_backend():
    assert FrameStage(4, 6).device == dev.detect_backend()
    r = bank.Bank(models=[], stage=FrameStage(4, 6, device="cpu"), device="cpu")
    assert r.dtype is torch.float32, "a cpu bank built without a dtype must not play in half"


def test_a_compile_that_fails_plays_eager_and_says_so(monkeypatch, capsys):
    def refuse(*_a, **_k):
        raise RuntimeError("no host compiler\nsecond line nobody needs")

    monkeypatch.setattr(torch, "compile", refuse)
    net = torch.nn.Identity()
    got, graphs, _secs = speedups.compile_and_count(net, 8, "cpu", torch.float32)
    assert got is net and graphs == 0
    assert "running eager -- no host compiler" in capsys.readouterr().out


def test_conversions_that_fail_to_compile_fall_back_to_eager(capsys):
    def refuse(_frame):
        raise RuntimeError("Inductor cannot build this")

    stage = FrameStage(4, 6, to_bgra=refuse, device="cpu")
    assert stage.compiled["bgra"]
    assert stage.warm(torch.zeros(1, 3, 4, 6)) == 0
    assert not any(stage.compiled.values())
    assert "conversions: running eager" in capsys.readouterr().out


def test_a_single_precision_session_has_no_half_block():
    """`exact` or not: on the CPU the file's own four half blocks would be the slow path."""
    assert S2.half_from_for(torch.float32, exact=False) == S2.SINGLE_EVERYWHERE
    assert S2.half_from_for(torch.float32, exact=True) == S2.SINGLE_EVERYWHERE
    assert S2.half_from_for(torch.float16, exact=False) == S2.HALF_EVERYWHERE
    assert S2.half_from_for(torch.float16, exact=True) is None


def test_the_opener_takes_the_devices_precision_when_not_told(monkeypatch):
    asked = []
    monkeypatch.setattr(S2, "from_file",
                        lambda path, device, half_from=None: asked.append(half_from) or
                        SimpleNamespace(mapping=SimpleNamespace(push=None)))
    monkeypatch.setattr(K, "install_stylegan2", lambda net, device: None)
    monkeypatch.setattr(S2, "style_bands", lambda net: [])
    open_stylegan2("x.pt", "cpu")
    assert asked == [S2.SINGLE_EVERYWHERE]
