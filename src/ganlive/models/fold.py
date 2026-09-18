"""Inference-only rewrites of a trained generator. Exact, not approximate.

Spectral norm baked into the weight, BatchNorm folded into the convolution feeding it, and the
frozen noise precomputed -- all of it measured at 0.00000 8-bit levels against the net it
replaces. `onnx_rewrite.py` is the other kind of rewrite: the ones made on the way out to ONNX.
"""

from __future__ import annotations

import torch
from torch import nn

from ganlive.models.fastgan import NoiseInjection
from ganlive.pixels import compiled_to_bgra, compiled_to_nv12, compiled_to_rgb, to_bgra, to_nv12, to_rgb


def remove_spectral_norm(net: nn.Module) -> int:
    """Bake every spectral-norm reparametrisation into a plain weight, in place."""
    removed = 0
    for module in net.modules():
        try:
            nn.utils.remove_spectral_norm(module)
            removed += 1
        except (ValueError, RuntimeError):
            continue  # no spectral_norm on this module
    return removed


class FoldedNoise(nn.Module):
    """What remains of `conv -> NoiseInjection -> BatchNorm` once the norm is folded away."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff, self.noise)


def _bn_affine(bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """`(scale, shift)` with `BN(z) == scale*z + shift` per channel, computed in float32."""
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    return scale, bn.bias.float() - bn.running_mean.float() * scale


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """`conv` followed by an eval `bn`, as one convolution."""
    scale, shift = _bn_affine(bn)
    dt = conv.weight.dtype
    weight = (conv.weight.float() * scale.reshape(-1, 1, 1, 1)).to(dt)
    bias = (shift if conv.bias is None else conv.bias.float() * scale + shift).to(dt)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode, device=conv.weight.device, dtype=dt)
    fused.weight = nn.Parameter(weight.detach(), requires_grad=False)
    fused.bias = nn.Parameter(bias.detach(), requires_grad=False)
    return fused


def fold_norms(net: nn.Module) -> dict[str, int]:
    """Fold every eval-time BatchNorm into the convolution feeding it. Returns what changed."""
    counts = {"conv_bn": 0, "conv_noise_bn": 0, "skipped_unfrozen": 0}
    for parent in net.modules():
        if not isinstance(parent, nn.Sequential):
            continue
        items, out, i = list(parent), [], 0
        while i < len(items):
            a = items[i]
            b = items[i + 1] if i + 1 < len(items) else None
            c = items[i + 2] if i + 2 < len(items) else None
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                out.append(_fuse(a, b))
                counts["conv_bn"] += 1
                i += 2
            elif (isinstance(a, nn.Conv2d) and isinstance(b, NoiseInjection)
                  and isinstance(c, nn.BatchNorm2d)):
                if b.frozen is None:
                    counts["skipped_unfrozen"] += 1
                    out.append(a)
                    i += 1
                    continue
                scale, _ = _bn_affine(c)
                out.append(_fuse(a, c))
                out.append(FoldedNoise((scale * b.weight).reshape(1, -1, 1, 1).float(),
                                       b.frozen.float()))
                counts["conv_noise_bn"] += 1
                i += 3
            else:
                out.append(a)
                i += 1
        if len(out) != len(items):
            parent._modules.clear()
            for j, m in enumerate(out):
                parent._modules[str(j)] = m
    return counts


def fold_free_noise(net: nn.Module) -> int:
    """Precompute `weight * frozen` for injections no norm fold absorbed."""
    n = 0
    for m in net.modules():
        if isinstance(m, NoiseInjection) and m.freeze and m.frozen is not None:
            const = (m.weight.detach() * m.frozen).detach()
            m.forward = (lambda c: (lambda x: x + c.to(x.dtype)))(const)
            n += 1
    return n


# ORDER MATTERS BOTH WAYS. The fold must come AFTER a forward pass -- it reads running statistics, so
# folding a cold net silently folds nothing in the blocks that matter most -- and BEFORE compile.
def prepare_for_inference(net: nn.Module, nz: int, device, *, half: bool = True,
                          fold: bool = True, compile_yuv: bool = True) -> dict:
    """Fold, cast, and hand back the colour conversion to use. Reports what was applied.

    **It does not compile.** It took a `compile_net` flag that every caller passed `False` --
    the bank compiles through `bank._compiled`, which is the one place that knows what this
    load asked for, and the export must not compile at all. The flag's only effect was to make
    this module import `models.capture`, which is a dependency on the graph recorder from a
    module that does nothing but rewrite weights."""
    with torch.no_grad():
        net(torch.zeros(1, nz, device=device))       # draws the lazy frozen patterns
    report: dict = {"folded": {}, "half": half}
    # **Bake the spectral norm, which was being recomputed on every frame.** It is a pre-forward hook, so
    # `weight = weight_orig / sigma` ran once per forward for every module still carrying one -- 15.4 M
    # parameters read and written for nothing, plus the power-iteration matmuls, at 61.6 MB of traffic a
    # frame in fp16. In eval nothing updates `u` and `v`, so sigma is a constant and baking it is exact:
    # measured at 0.00000 8-bit levels over three latents in fp32. Before the fold, not after.
    report["spectral_removed"] = remove_spectral_norm(net)
    if fold:
        report["folded"] = fold_norms(net)
        report["folded"]["free_noise"] = fold_free_noise(net)
    if half:
        net = net.to(torch.float16).to(memory_format=torch.channels_last)
    report["net"] = net
    report["yuv"] = to_nv12
    report["rgb"] = to_rgb
    report["bgra"] = to_bgra
    if compile_yuv:
        report["yuv"] = compiled_to_nv12()
        report["rgb"] = compiled_to_rgb()
        report["bgra"] = compiled_to_bgra()
    return report
