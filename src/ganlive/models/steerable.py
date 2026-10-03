"""The two modules a live FastGAN dial is made of, grafted into a built generator.

Each holds a view into the settings vector, never a float: `torch.compile` guards on scalar
values and would rebuild the graph every time a dial moved.
"""

from __future__ import annotations

import torch
from torch import nn


class SteerableNoise(nn.Module):
    """`FoldedNoise` with a live gain: `x + (coeff * gain) * noise`."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor,
                 gain: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)
        # A plain attribute, not a buffer: `.to()` would give each module its own copy of a
        # buffer, and the settings vector would no longer reach it.
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff * self.gain, self.noise)


class SteerableSLE(nn.Module):
    """`SkipLayerExcitation` with the gate blended toward identity: `high * (1 + b*(g-1))`."""

    def __init__(self, gate: nn.Module, blend: torch.Tensor) -> None:
        super().__init__()
        self.gate = gate
        self.blend = blend

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * (1.0 + self.blend * (self.gate(low) - 1.0))
