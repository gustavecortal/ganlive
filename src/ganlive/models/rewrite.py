"""Graph rewrites applied on the way OUT to ONNX, and nowhere else."""
from __future__ import annotations

import torch
from torch import nn

from ganlive.models.graph import FoldedNoise


class GatedPair(nn.Module):
    """`conv -> [folded noise] -> GLU`, with the halving done on the WEIGHTS."""

    def __init__(self, conv: nn.Conv2d, noise=None) -> None:
        super().__init__()
        half = conv.out_channels // 2
        self.value = _slice_conv(conv, 0, half)
        self.gate = _slice_conv(conv, half, conv.out_channels)
        self.noise = noise is not None
        # A banked rung stays live through the split: the coefficient halves are constants
        # either way, and the scalar that multiplies them is still a slice of input 1. A
        # noise dial that went dead here would be the inert knob the split exists to avoid.
        self.rung = noise.rung if isinstance(noise, ExportedNoise) else None
        if noise is not None:
            self.register_buffer("coeff_value", noise.coeff[:, :half].clone())
            self.register_buffer("coeff_gate", noise.coeff[:, half:].clone())
            self.register_buffer("pattern", noise.noise.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.value(x), self.gate(x)
        if self.noise:
            rung = 1.0 if self.rung is None else self.rung()
            value = torch.addcmul(value, self.coeff_value * rung, self.pattern)
            gate = torch.addcmul(gate, self.coeff_gate * rung, self.pattern)
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


def split_gated_convs(net: nn.Module) -> int:
    """Replace every `conv [-> folded noise] -> GLU` run with a `GatedPair`. Returns the count."""
    replaced = 0
    for parent in net.modules():
        if not isinstance(parent, nn.Sequential):
            continue
        items, out, i = list(parent), [], 0
        while i < len(items):
            a = items[i]
            b = items[i + 1] if i + 1 < len(items) else None
            c = items[i + 2] if i + 2 < len(items) else None
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.GLU) and b.dim == 1:
                out.append(GatedPair(a))
                replaced += 1
                i += 2
            elif (isinstance(a, nn.Conv2d) and isinstance(b, (FoldedNoise, ExportedNoise))
                  and isinstance(c, nn.GLU) and c.dim == 1):
                out.append(GatedPair(a, b))
                replaced += 1
                i += 3
            else:
                out.append(a)
                i += 1
        if len(out) != len(items):
            parent._modules.clear()
            for j, module in enumerate(out):
                parent._modules[str(j)] = module
    return replaced


def equivalent(before: nn.Module, after: nn.Module, nz: int, device="cpu",
               probes: int = 3, seed: int = 0, settings=None) -> float:
    """Worst 8-bit level difference between two generators over a few latents."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    worst = 0.0
    with torch.no_grad():
        for i in range(probes):
            z = torch.randn(1, nz, generator=generator).to(device)
            if settings is None:
                a, b = before(z)[0], after(z)[0]
            else:
                k = settings if i == 0 else settings * (1.0 + 0.35 * (i % 3))
                a, b = before(z, k)[0], after(z, k)[0]
            worst = max(worst, float((a - b).abs().max() * 127.5))
    return worst


class SettingsVector(nn.Module):
    """The settings vector, for the duration of one forward."""

    vec: torch.Tensor | None = None

    def __deepcopy__(self, memo):
        """A fresh, empty bank. After any forward `vec` holds a non-leaf tensor, which torch
        refuses to copy -- so `equivalent()` taking a reference copy would raise, and only
        the order things happen in today keeps it from doing so. The copy's modules all
        share this one, because `memo` hands every reference the same object."""
        fresh = SettingsVector()
        memo[id(self)] = fresh
        return fresh


class ExportedSLE(nn.Module):
    """`ExportedSLE` reading its blend from the bank instead of from a view."""

    def __init__(self, gate: nn.Module, bank: SettingsVector, index: int) -> None:
        super().__init__()
        self.gate, self.bank, self.index = gate, bank, index

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        blend = self.bank.vec[self.index:self.index + 1].reshape(1, 1, 1, 1)
        return high * (1.0 + blend * (self.gate(low) - 1.0))


class ExportedNoise(nn.Module):
    """`ExportedNoise` reading its rung from the bank."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor,
                 bank: SettingsVector, index: int) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)
        self.bank, self.index = bank, index

    def rung(self) -> torch.Tensor:
        return self.bank.vec[self.index:self.index + 1].reshape(1, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff * self.rung(), self.noise)


class Steerable(nn.Module):
    """The generator plus its settings, as a two-input graph: `forward(z, k)`."""

    def __init__(self, net: nn.Module, bank: SettingsVector, names: list[str]) -> None:
        super().__init__()
        self.net, self.bank, self.names = net, bank, list(names)

    def forward(self, z: torch.Tensor, k: torch.Tensor):
        self.bank.vec = k
        return self.net(z)


def bank_the_knobs(net: nn.Module, knobs) -> Steerable:
    """Rewrite an installed net so its settings come from a second forward argument."""
    from ganlive.dials.steer import SteerableNoise, SteerableSLE

    bank, reached, sites = SettingsVector(), set(), 0
    for parent in list(net.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, SteerableSLE):
                index = knobs.index[f"sle.{child_name}"]
                setattr(parent, child_name, ExportedSLE(child.gate, bank, index))
            elif isinstance(child, SteerableNoise):
                index = _slot_of(knobs, child.rung)
                setattr(parent, child_name,
                        ExportedNoise(child.coeff, child.noise, bank, index))
            else:
                continue
            reached.add(index)
            sites += 1
    # **Slots reached, not modules replaced.** One noise dial drives every injection at its
    # rung, so there are more sites than settings and counting modules says nothing about
    # whether a dial got left behind.
    missing = [n for i, n in enumerate(knobs.names) if i not in reached]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} reached no module, so {len(missing)} of "
            f"{len(knobs.names)} settings would export as constants and the exported model "
            f"would have dials that do nothing")
    return Steerable(net, bank, knobs.names)


def _slot_of(knobs, view: torch.Tensor) -> int:
    """Which slot of the settings vector a module's view points at."""
    for i, name in enumerate(knobs.names):
        if knobs.view(name).data_ptr() == view.data_ptr():
            return i
    raise RuntimeError("a steerable module holds a view into no known setting")
