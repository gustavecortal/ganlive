"""Inference-only rewrites of a trained FastGAN. Exact, not approximate.

Spectral norm baked into the weight, BatchNorm folded into the convolution feeding it, and the
frozen noise precomputed: each measures 0.00000 8-bit levels against the net it replaces.
`onnx_rewrite.py` holds the rewrites made only on the way out to ONNX.
"""

from __future__ import annotations

import torch
from torch import nn

from ganlive.models.fastgan import NoiseInjection
from ganlive.models.surgery import rewrite_sequential


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


def _norm_rule(a, b, c):
    """`conv -> bn`, and `conv -> noise -> bn`, as one folded convolution plus what is left."""
    if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
        return [_fuse(a, b)], 2, "conv_bn"
    if not (isinstance(a, nn.Conv2d) and isinstance(b, NoiseInjection)
            and isinstance(c, nn.BatchNorm2d)):
        return None
    if b.frozen is None:
        # Nothing to precompute yet, so the run is left whole and only the conv consumed.
        return [a], 1, "skipped_unfrozen"
    scale, _ = _bn_affine(c)
    return ([_fuse(a, c),
             FoldedNoise((scale * b.weight).reshape(1, -1, 1, 1).float(), b.frozen.float())],
            3, "conv_noise_bn")


def fold_norms(net: nn.Module) -> dict[str, int]:
    """Fold every eval-time BatchNorm into the convolution feeding it. Returns what changed."""
    counts = rewrite_sequential(net, _norm_rule)
    return {name: counts[name] for name in ("conv_bn", "conv_noise_bn", "skipped_unfrozen")}


def fold_free_noise(net: nn.Module) -> int:
    """Precompute `weight * frozen` for injections no norm fold absorbed."""
    n = 0
    for m in net.modules():
        if isinstance(m, NoiseInjection) and m.freeze and m.frozen is not None:
            const = (m.weight.detach() * m.frozen).detach()
            m.forward = (lambda c: (lambda x: x + c.to(x.dtype)))(const)
            n += 1
    return n


def prepare_for_inference(net: nn.Module, nz: int, device, *, half: bool = True,
                          fold: bool = True) -> dict:
    """Bake, fold and cast a FastGAN for play. Reports what was applied, and the net under
    `"net"`. It does not compile.

    The order matters: the fold reads the frozen noise patterns, which the first forward
    draws, and it has to come before anything compiles the net. `fold` is ignored and kept
    only for callers that still pass it; the net is always folded."""
    del fold
    with torch.no_grad():
        net(torch.zeros(1, nz, device=device))       # draws the lazy frozen patterns
    report: dict = {"half": half}
    # In eval nothing updates spectral norm's power iteration, so sigma is a constant and
    # baking it is exact. It has to come before the fold.
    report["spectral_removed"] = remove_spectral_norm(net)
    report["folded"] = fold_norms(net)
    report["folded"]["free_noise"] = fold_free_noise(net)
    if half:
        net = net.to(torch.float16).to(memory_format=torch.channels_last)
    report["net"] = net
    return report
