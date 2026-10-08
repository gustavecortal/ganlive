"""A converted FastGAN or StyleGAN2 draws, through wgpu, the picture PyTorch draws: the
promise `ganlive convert` makes. Runs on any adapter wgpu finds, the CPU one included."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ganlive.engine.convert import build, prepared, stylegan2_manifest, stylegan2_net
from ganlive.engine.program import LEVEL, PROBE_LEVELS
from ganlive.engine.program_stylegan2 import compile_stylegan2
from ganlive.models import stylegan2 as S2
from ganlive.models.common import host_latent
from ganlive.models.fastgan import Config, Generator
from tests.support import tiny_stylegan2

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


NZ = 32


def _prepared():
    """A small 384x256 FastGAN with live noise, prepared as `convert` prepares it, its RGB
    layer scaled so that its tanh does not saturate (a random generator's does)."""
    torch.manual_seed(4)
    cfg = Config(nz=NZ, ngf=16, im_size=256, im_width=384)
    net = Generator(**cfg.generator_kwargs).requires_grad_(False)
    for module in net.modules():
        if type(module).__name__ == "NoiseInjection":  # noise gains start at exactly zero
            module.weight.fill_(0.4)
        if isinstance(module, torch.nn.BatchNorm2d):   # statistics from the batches below,
            module.momentum = None                     # as training leaves them
    with torch.no_grad():                              # untrained statistics let activations
        for _ in range(4):                             # grow without bound
            net(torch.randn(8, NZ))
    net.eval()
    steerable, cfg, names = prepared((net, cfg))
    to_big = steerable.net.to_big                      # a plain conv once prepared
    seen: list[float] = []
    handle = to_big.register_forward_hook(lambda _m, _i, o: seen.append(float(o.std())))
    with torch.no_grad():
        steerable(torch.randn(1, NZ), torch.ones(len(names)))
        handle.remove()
        to_big.weight.mul_(0.4 / seen[0])
    return steerable, cfg, names


def test_a_converted_fastgan_draws_what_pytorch_draws_at_every_setting(device):
    """With the noise the engine makes on the GPU itself, as a converted model plays."""
    steerable, cfg, names = _prepared()
    program, blob = build(steerable, cfg, names, output="f32")
    model = Model(device, program, bytes(blob.data))
    assert model.strays() < PROBE_LEVELS                # what a backend must pass to be used
    for seed, k in enumerate((np.ones(len(names)), np.linspace(0.3, 1.7, len(names)))):
        z = host_latent(cfg.nz, seed=seed + 1)
        k = k.astype(np.float32)
        model.set_latent(z)
        model.set_settings(k)
        model.frame()
        with torch.no_grad():
            ref = steerable(torch.from_numpy(z), torch.from_numpy(k))[0][0].numpy()
        assert (np.abs(ref) > 0.99).mean() < 0.05, "the fixture is clipped and can measure nothing"
        levels = np.abs(model.read() - ref) * LEVEL
        # fp16 weights and activations: most pixels within a level, none far off.
        assert levels.mean() < 0.5 and np.percentile(levels, 99.9) < 4, (levels.mean(), levels.max())


def test_a_converted_stylegan2_draws_what_pytorch_draws_at_every_setting(device, tmp_path):
    """Mapping, truncation, modulated and transposed convolutions, blur and the colour skip."""
    torch.manual_seed(0)
    cfg = tiny_stylegan2()
    fresh = S2.Generator(cfg)
    for layers in S2.noise_sites(fresh).values():
        for layer in layers:
            layer.noise_strength.data.fill_(0.3)        # trained noise is never off
    S2.save(tmp_path / "tiny.pt", cfg, fresh.state_dict())
    net, settings, _push = stylegan2_net(tmp_path / "tiny.pt")
    manifest, blob = stylegan2_manifest(net, list(settings.names))
    model = Model(device, compile_stylegan2(manifest, output="f32"), bytes(blob.data))
    for seed, k in enumerate((np.ones(len(settings.names)), np.linspace(0.5, 1.5, len(settings.names)))):
        z = host_latent(cfg.z_dim, seed=seed + 1)
        settings.write[:] = k
        settings.commit()
        model.set_latent(z)
        model.set_settings(k.astype(np.float32))
        model.frame()
        with torch.no_grad():
            ref = net(torch.from_numpy(z))[0].numpy()
        levels = np.abs(model.read() - ref) * LEVEL
        assert levels.mean() < 0.5 and np.percentile(levels, 99.9) < 4, (levels.mean(), levels.max())
