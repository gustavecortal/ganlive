"""The two modules a live dial is actually made of, grafted into a built generator.

They are `nn.Module`s that replace a trained one in the tree, so they are model code, and
they lived in `dials.steer` -- which meant `models.onnx_rewrite` had to import upward, out of
`models` and into `dials`, to name the classes it rewrites. That import was written inside a
function to keep the cycle out of the importer's way, which is the usual sign that a thing is
one layer off from where it belongs.

What each holds is a **view into the settings vector**, never a float: `torch.compile` guards
on scalar values and would rebuild the graph every time a dial moved.
"""

from __future__ import annotations

import torch
from torch import nn


class SteerableNoise(nn.Module):
    """`FoldedNoise` with a live gain: `x + (coeff * rung) * noise`."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor,
                 rung: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)
        # A plain attribute, NOT a buffer: register_buffer + .to() hands each module its
        # own detached copy, leaving the caller holding a tensor that steers nothing.
        self.rung = rung

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff * self.rung, self.noise)


class SteerableSLE(nn.Module):
    """`SkipLayerExcitation` with the gate blended toward identity: `high * (1 + b*(g-1))`."""

    def __init__(self, gate: nn.Module, blend: torch.Tensor) -> None:
        super().__init__()
        self.gate = gate
        self.blend = blend

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * (1.0 + self.blend * (self.gate(low) - 1.0))
