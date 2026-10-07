"""ganlive on any machine: another card, a Mac, or no card at all.

The device and precision defaults, and the two compile failures that degrade to eager instead
of ending the load."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ganlive import bank
from ganlive import device as dev
from ganlive.dials import steer as K
from ganlive.families import open_stylegan2
from ganlive.frame import FrameStage
from ganlive.models import capture as speedups
from ganlive.models import fastgan
from ganlive.models import stylegan2 as S2
from ganlive.settings import Settings


def test_half_on_a_card_and_single_on_the_cpu():
    assert dev.playback_dtype("cpu") is torch.float32
    for card in ("xpu", "cuda", "mps", "xpu:0", torch.device("cuda:1")):
        assert dev.playback_dtype(card) is torch.float16


def test_the_mps_module_is_a_backend_too():
    """`synchronize` on a Mac has to wait, so the lookup must not return None there."""
    assert dev._mod("mps") is getattr(torch, "mps", None)
    assert dev._mod("cpu") is None


@pytest.mark.parametrize("size", [(4, 6), (8, 12), (5, 7), (4, 4)])
def test_the_matrix_pool_averages_the_windows_the_kernel_does(size):
    """MPS refuses the 3:2 base block's 4x6 into 4x4, so a Mac plays a FastGAN through this."""
    x = torch.randn(2, 8, *size)
    pooled = fastgan.MatrixPool(size, 4)(x)
    assert torch.allclose(pooled, torch.nn.AdaptiveAvgPool2d(4)(x), atol=1e-6)


def _refuse_adaptive_pools(monkeypatch):
    """Make this CPU answer an adaptive pool the way MPS answers a 4x6 input."""
    def refuse(*_a, **_k):
        raise RuntimeError("Adaptive pool MPS: input sizes must be divisible by output sizes")

    monkeypatch.setattr(torch.nn.functional, "adaptive_avg_pool2d", refuse)


def test_a_generator_plays_where_adaptive_pools_are_refused(monkeypatch):
    """Every gate pools through `MatrixPool`, so a backend that refuses adaptive pools at
    these sizes still draws the picture."""
    _refuse_adaptive_pools(monkeypatch)
    net = fastgan.Generator(ngf=8, nz=16, im_size=256, im_width=384).eval()
    with torch.no_grad():
        assert net(torch.randn(1, 16))[0].shape == (1, 3, 256, 384)


def test_a_half_settings_vector_hands_out_no_views():
    """A compiled MPS kernel reads a half view at an odd offset from the wrong element."""
    with pytest.raises(TypeError, match="no views"):
        Settings(["a", "b"], "cpu", torch.float16).view("b")
    assert Settings(["a", "b"], "cpu", torch.float32).view("b").shape == (1,)


def test_a_backend_that_refuses_area_still_shrinks_the_frame(monkeypatch, capsys):
    stage = FrameStage(9, 15, device="cpu")
    frame = torch.rand(1, 3, 40, 60) * 2 - 1
    area = stage.step(frame)
    _refuse_adaptive_pools(monkeypatch)
    shrunk = stage.step(frame)
    assert shrunk.shape == area.shape == (1, 3, 9, 15)
    assert (shrunk - area).abs().mean() < 0.05, "a different filter, not a different picture"
    assert capsys.readouterr().out.count("resize: antialiased bilinear") == 1
    stage.step(frame)
    assert "resize" not in capsys.readouterr().out, "said once, not every frame"


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


def test_an_editors_python_is_not_taken_for_a_job_on_the_card():
    """A linter server is a python process with no card; a ganlive run is one with one."""
    def process(cmdline, maps=None):
        out = SimpleNamespace(cmdline=lambda: cmdline)
        if maps is not None:
            out.memory_maps = lambda: [SimpleNamespace(path=p) for p in maps]
        return out

    lint = ["python", "/x/.vscode/extensions/ms-python.flake8/bundled/tool/lsp_server.py"]
    assert not dev._holds_torch(process(lint))
    assert dev._holds_torch(process(["python", "-m", "ganlive", "play"]))
    assert dev._holds_torch(process(lint, maps=["/venv/lib/torch/lib/libtorch_cpu.so"]))
    assert not dev._holds_torch(process(["python", "train.py"], maps=["/usr/lib/libc.so"]))
