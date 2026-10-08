"""The two modules a live FastGAN dial is made of, grafted into a built generator.

Each reads its slot of the settings vector out of a holder, never a float: `torch.compile`
guards on scalar values and would rebuild the graph every time a dial moved. Nor a view of the
slot: a compiled MPS kernel reads a half-precision view at an odd offset from the wrong element,
so the slot is sliced inside the forward, which every backend compiles correctly. The holder is
the live `Settings`, or the converter's second input (`rewrite.settings_as_input`).
"""

from __future__ import annotations

import torch
from torch import nn


class Slot(nn.Module):
    """One slot of whatever settings vector `holder.vec` is at the time of the forward."""

    def __init__(self, holder, index: int) -> None:
        super().__init__()
        # Plain attributes: the holder is shared by every module, and is not theirs to move.
        self.holder, self.index = holder, index

    def value(self) -> torch.Tensor:
        return self.holder.vec[self.index:self.index + 1].reshape(1, 1, 1, 1)


class SteerableNoise(Slot):
    """`FoldedNoise` with a live gain: `x + (coeff * gain) * noise`."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor, holder, index: int) -> None:
        super().__init__(holder, index)
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff * self.value(), self.noise)


class SteerableSLE(Slot):
    """`SkipLayerExcitation` with the gate blended toward identity: `high * (1 + b*(g-1))`."""

    def __init__(self, gate: nn.Module, holder, index: int) -> None:
        super().__init__(holder, index)
        self.gate = gate

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * (1.0 + self.value() * (self.gate(low) - 1.0))
