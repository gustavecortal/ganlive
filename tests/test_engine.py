"""A converted FastGAN draws, through wgpu, the picture PyTorch draws: the promise `ganlive
convert` makes. Runs on any adapter wgpu finds, the CPU one included."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ganlive.engine.convert import manifest_of, prepared, probe
from ganlive.engine.program import PROBE_LEVELS, compile_program
from ganlive.models.fastgan import Config, Generator

wgpu = pytest.importorskip("wgpu")
from ganlive.engine.runner import Model, default_device  # noqa: E402


@pytest.fixture(scope="module")
def device():
    """The CPU adapter where there is one, so the suite leaves the card to whatever is
    playing; any adapter otherwise."""
    for fallback in (True, False):
        try:
            return default_device(fallback)
        except (RuntimeError, wgpu.GPUError):           # this kind of adapter is missing
            continue
    pytest.skip("no WebGPU adapter")


def _prepared(nz: int = 32):
    """A small 384x256 FastGAN with live noise, prepared as `convert` prepares it, its RGB
    layer scaled so that its tanh does not saturate (a random generator's does)."""
    torch.manual_seed(4)
    cfg = Config(nz=nz, ngf=16, im_size=256, im_width=384)
    net = Generator(**cfg.generator_kwargs).requires_grad_(False)
    for module in net.modules():
        if type(module).__name__ == "NoiseInjection":  # noise gains start at exactly zero
            module.weight.fill_(0.4)
        if isinstance(module, torch.nn.BatchNorm2d):   # statistics from the batches below,
            module.momentum = None                     # as training leaves them
    with torch.no_grad():                              # untrained statistics let activations
        for _ in range(4):                             # grow without bound
            net(torch.randn(8, nz))
    net.eval()
    steerable, cfg, names = prepared((net, cfg))
    to_big = steerable.net.to_big                      # a plain conv once prepared
    seen: list[float] = []
    handle = to_big.register_forward_hook(lambda _m, _i, o: seen.append(float(o.std())))
    with torch.no_grad():
        steerable(torch.randn(1, nz), torch.ones(len(names)))
        handle.remove()
        to_big.weight.mul_(0.4 / seen[0])
    return steerable, cfg, names


def test_a_converted_fastgan_draws_what_pytorch_draws_at_every_setting(device):
    """With the noise the engine makes on the GPU itself, as a converted model plays."""
    steerable, cfg, names = _prepared()
    manifest, blob, _ = manifest_of(steerable, cfg, names)
    program = compile_program(manifest)
    program["probe"] = probe(steerable, cfg, names)
    model = Model(device, program, bytes(blob.data), output="f32")
    assert model.strays() < PROBE_LEVELS                # what a backend must pass to be used
    rng = np.random.default_rng(0)
    for k in (np.ones(len(names)), np.linspace(0.3, 1.7, len(names))):
        z = rng.standard_normal((1, cfg.nz)).astype(np.float32)
        k = k.astype(np.float32)
        model.set_latent(z)
        model.set_settings(k)
        model.frame()
        with torch.no_grad():
            ref = steerable(torch.from_numpy(z), torch.from_numpy(k))[0][0].numpy()
        assert (np.abs(ref) > 0.99).mean() < 0.05, "the fixture is clipped and can measure nothing"
        levels = np.abs(model.read() - ref) * 127.5
        # fp16 weights and activations: most pixels within a level, none far off.
        assert levels.mean() < 0.5 and np.percentile(levels, 99.9) < 4, (levels.mean(), levels.max())
