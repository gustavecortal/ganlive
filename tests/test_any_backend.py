"""ganlive on any machine: another card, a Mac, or no card at all.

The device and precision defaults, and the two compile failures that degrade to eager instead
of ending the load."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ganlive import device as dev
from ganlive import process
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


def test_a_single_precision_session_has_no_half_block():
    """`exact` or not: on the CPU the file's own four half blocks would be the slow path."""
    assert S2.half_from_for(torch.float32, exact=False) == S2.SINGLE_EVERYWHERE
    assert S2.half_from_for(torch.float32, exact=True) == S2.SINGLE_EVERYWHERE
    assert S2.half_from_for(torch.float16, exact=False) == S2.HALF_EVERYWHERE
    assert S2.half_from_for(torch.float16, exact=True) is None


def test_an_editors_python_is_not_taken_for_a_job_on_the_card():
    """A linter server is a python process with no card; a ganlive run is one with one."""
    def a_process(cmdline, maps=None):
        out = SimpleNamespace(cmdline=lambda: cmdline)
        if maps is not None:
            out.memory_maps = lambda: [SimpleNamespace(path=p) for p in maps]
        return out

    lint = ["python", "/x/.vscode/extensions/ms-python.flake8/bundled/tool/lsp_server.py"]
    assert not process._holds_card(a_process(lint))
    assert process._holds_card(a_process(["python", "-m", "ganlive", "play"]))
    assert process._holds_card(a_process(lint, maps=["/venv/lib/torch/lib/libtorch_cpu.so"]))
    assert not process._holds_card(a_process(["python", "train.py"], maps=["/usr/lib/libc.so"]))
    assert process._holds_card(a_process(lint, maps=["C:/venv/Lib/site-packages/wgpu/resources/wgpu_native-release.dll"])), (
        "a player draws through wgpu, without torch")
