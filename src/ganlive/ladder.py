"""The geometry every generator family shares: the (height, width) it climbs to. Torch-free, so
the engine and the shelf can read a model's size without loading a generator."""

from __future__ import annotations

from dataclasses import dataclass

#: Every ladder ends on a block this many pixels tall, whatever family built it.
BASE = 4


@dataclass(frozen=True)
class Ladder:
    """The (height, width) a generator climbs to, square or not.

    Geometry only. Whether a particular family can build a given size is that family's
    question: see `fastgan.check_buildable`."""

    height: int
    width: int

    @classmethod
    def of(cls, height: int, width: int | None = None) -> Ladder:
        """`width=None` means square."""
        return cls(height, height if width is None else width)

    @property
    def base_width(self) -> int:
        """Width of the `BASE`-high bottom rung. 4 when square, 6 at 3:2."""
        return BASE * self.width // self.height

    def at(self, rung: int) -> tuple[int, int]:
        """The (height, width) of the ladder `rung` rows tall."""
        return rung, rung * self.base_width // BASE
