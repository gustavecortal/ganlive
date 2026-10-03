"""Rewrites made to a FastGAN on its way out to ONNX, and nowhere else.

`split_gated_convs` replaces each `conv -> GLU` with two half-width convolutions, so the
runtime never copies a tensor to split it. `settings_as_input` turns the live settings into a
second graph input, `forward(z, k)`, because a tensor view would trace as a constant.
"""
from __future__ import annotations

import torch
from torch import nn

from ganlive.models.fold import FoldedNoise
from ganlive.models.steerable import SteerableNoise, SteerableSLE
from ganlive.models.surgery import rewrite_sequential
from ganlive.pixels import worst_levels


class GatedPair(nn.Module):
    """`conv -> [folded noise] -> GLU`, with the halving done on the weights."""

    def __init__(self, conv: nn.Conv2d, noise=None) -> None:
        super().__init__()
        half = conv.out_channels // 2
        self.value = _slice_conv(conv, 0, half)
        self.gate = _slice_conv(conv, half, conv.out_channels)
        self.noise = noise is not None
        # A noise gain read from the settings input survives the split: both coefficient
        # halves are scaled by the same slice of it.
        self.gain = noise.gain if isinstance(noise, ExportedNoise) else None
        if noise is not None:
            self.register_buffer("coeff_value", noise.coeff[:, :half].clone())
            self.register_buffer("coeff_gate", noise.coeff[:, half:].clone())
            self.register_buffer("pattern", noise.noise.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.value(x), self.gate(x)
        if self.noise:
            gain = 1.0 if self.gain is None else self.gain()
            value = torch.addcmul(value, self.coeff_value * gain, self.pattern)
            gate = torch.addcmul(gate, self.coeff_gate * gain, self.pattern)
        return value * torch.sigmoid(gate)


def _slice_conv(conv: nn.Conv2d, lo: int, hi: int) -> nn.Conv2d:
    """A convolution producing only output channels `[lo, hi)` of `conv`."""
    out = nn.Conv2d(conv.in_channels, hi - lo, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, bias=conv.bias is not None,
                    padding_mode=conv.padding_mode,
                    device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        out.weight.copy_(conv.weight[lo:hi])
        if conv.bias is not None:
            out.bias.copy_(conv.bias[lo:hi])
    out.requires_grad_(False)
    return out


def _gated_rule(a, b, c):
    """`conv -> GLU`, with a folded noise between them or without, as one `GatedPair`."""
    if not isinstance(a, nn.Conv2d):
        return None
    if isinstance(b, nn.GLU) and b.dim == 1:
        return [GatedPair(a)], 2, "split"
    if (isinstance(b, (FoldedNoise, ExportedNoise))
            and isinstance(c, nn.GLU) and c.dim == 1):
        return [GatedPair(a, b)], 3, "split"
    return None


def split_gated_convs(net: nn.Module) -> int:
    """Replace every `conv [-> folded noise] -> GLU` run with a `GatedPair`. Returns the count."""
    return rewrite_sequential(net, _gated_rule)["split"]


def equivalent(before: nn.Module, after: nn.Module, nz: int, device="cpu",
               probes: int = 3, settings=None) -> float:
    """Worst 8-bit level difference between two generators over a few latents. With
    `settings`, both take them as a second argument, scaled differently per latent."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    worst = 0.0
    with torch.no_grad():
        for i in range(probes):
            z = torch.randn(1, nz, generator=generator).to(device)
            if settings is None:
                a, b = before(z)[0], after(z)[0]
            else:
                k = settings if i == 0 else settings * (1.0 + 0.35 * (i % 3))
                a, b = before(z, k)[0], after(z, k)[0]
            worst = max(worst, worst_levels(a, b))
    return worst


class SettingsVector(nn.Module):
    """The settings vector, for the duration of one forward."""

    vec: torch.Tensor | None = None

    def __deepcopy__(self, memo):
        """A fresh, empty vector. After a forward `vec` holds a non-leaf tensor, which torch
        refuses to copy. Every module in the copy shares this one, through `memo`."""
        fresh = SettingsVector()
        memo[id(self)] = fresh
        return fresh


class ExportedSLE(nn.Module):
    """`SteerableSLE` reading its blend from the settings vector instead of from a view."""

    def __init__(self, gate: nn.Module, settings: SettingsVector, index: int) -> None:
        super().__init__()
        self.gate, self.settings, self.index = gate, settings, index

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        blend = self.settings.vec[self.index:self.index + 1].reshape(1, 1, 1, 1)
        return high * (1.0 + blend * (self.gate(low) - 1.0))


class ExportedNoise(nn.Module):
    """`SteerableNoise` reading its gain from the settings vector instead of from a view."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor,
                 settings: SettingsVector, index: int) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)
        self.settings, self.index = settings, index

    def gain(self) -> torch.Tensor:
        return self.settings.vec[self.index:self.index + 1].reshape(1, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff * self.gain(), self.noise)


class Steerable(nn.Module):
    """The generator plus its settings, as a two-input graph: `forward(z, k)`."""

    def __init__(self, net: nn.Module, settings: SettingsVector) -> None:
        super().__init__()
        self.net, self.settings = net, settings

    def forward(self, z: torch.Tensor, k: torch.Tensor):
        self.settings.vec = k
        return self.net(z)


def settings_as_input(net: nn.Module, settings) -> Steerable:
    """Rewrite a net with installed `Settings` so its settings come from a second argument."""
    vector, reached = SettingsVector(), set()
    for parent in list(net.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, SteerableSLE):
                index = settings.index[f"sle.{child_name}"]
                setattr(parent, child_name, ExportedSLE(child.gate, vector, index))
            elif isinstance(child, SteerableNoise):
                index = _slot_of(settings, child.gain)
                setattr(parent, child_name,
                        ExportedNoise(child.coeff, child.noise, vector, index))
            else:
                continue
            reached.add(index)
    # Settings reached, not modules replaced: one noise setting drives several modules.
    missing = [n for i, n in enumerate(settings.names) if i not in reached]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} reached no module, so {len(missing)} of "
            f"{len(settings.names)} settings would export as constants and the exported model "
            f"would have dials that do nothing")
    return Steerable(net, vector)


def _slot_of(settings, view: torch.Tensor) -> int:
    """Which slot of the settings vector a module's view points at."""
    for i, name in enumerate(settings.names):
        if settings.view(name).data_ptr() == view.data_ptr():
            return i
    raise RuntimeError("a steerable module holds a view into no known setting")
