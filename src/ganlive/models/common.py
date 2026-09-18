"""What every generator family here shares: its output geometry, and its output."""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Every ladder ends on a block this many pixels tall, whatever family built it.
BASE = 4


@dataclass(frozen=True)
class Ladder:
    """The (height, width) a generator climbs to -- square or not.

    Geometry only. Whether a *particular* family can build a given size is that family's
    question, asked where it builds: `fastgan.check_buildable` is the one that has an
    answer, and it used to live here behind a flag every other family passed to turn it off."""

    height: int
    width: int

    @classmethod
    def of(cls, height: int, width: int | None = None) -> Ladder:
        """`width=None` means square, which is what every existing checkpoint is."""
        return cls(height, height if width is None else width)

    @property
    def base_width(self) -> int:
        """Width of the `BASE`-high bottom rung. 4 when square, 6 at 3:2."""
        return BASE * self.width // self.height

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def at(self, rung: int) -> tuple[int, int]:
        """The (height, width) of the ladder `rung` rows tall."""
        return rung, rung * self.base_width // BASE


def denormalise(x: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1, 1] -> [0, 1], ready to save as an image."""
    return x.float().add(1).mul(0.5).clamp(0, 1)


def first_image(out):
    """The image a generator returned, whichever calling convention it uses."""
    return out[0] if isinstance(out, (list, tuple)) else out


def latent(nz: int, seed: int, device, dtype) -> torch.Tensor:
    """One latent, drawn on the host so the picture does not depend on the card it ran on.

    Here rather than in `dials.derive`, where it started: `models.capture` needs a seeded
    latent to check a replay against the forward, and was importing it -- by its private name
    -- from a module two layers above it. Drawing a probe latent is not a fact about dials."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, nz, generator=generator).to(device=device, dtype=dtype)
