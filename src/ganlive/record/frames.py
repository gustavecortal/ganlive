"""A take's frames, from the time each is due to the time it is drawn.

While recording, the frame loop works out the drums, the dials and the beat at every frame of the
take, at that frame's own time (`Take.due`), and logs what drawing it needs (`Take.keep`). It
draws the log in order while the card keeps up, so each picture is drawn once, and a card too
slow for the take draws what it had no time for after the take is stopped. Every take therefore
has every frame, evenly spaced, on any card. The browser's take (`web/take.mjs`) works the same
way.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

#: Frames a block of the log holds.
BLOCK = 256


@dataclass
class Frame:
    """What drawing a frame again exactly needs: the generator, its `(1, nz)` latent, its
    settings, and a StyleGAN2's `w` push (None for a model steered through its latent)."""

    index: int
    net: object
    latent: np.ndarray
    settings: np.ndarray
    push: np.ndarray | None


@dataclass
class _Block:
    """Frames of one generator side by side, a row each: latent, settings, push."""

    net: object
    sizes: tuple[int, int]
    push_shape: tuple | None
    first: int
    data: np.ndarray
    start: int = 0
    end: int = 0

    def frame(self, i: int) -> Frame:
        row = self.data[i]
        nz, ns = self.sizes
        push = None if self.push_shape is None else row[nz + ns:].reshape(self.push_shape)
        return Frame(self.first + i, self.net, row[:nz].reshape(1, nz), row[nz:nz + ns], push)


@dataclass
class Take:
    """The frames of a take starting at `start` (seconds, `time.perf_counter`), `fps` a second:
    when each is due, and those logged and not drawn yet (`behind`), about 1 KB each for a
    FastGAN."""

    fps: float
    start: float
    #: The last frame logged.
    slot: int = -1
    #: The newest frame logged: the one the window shows.
    latest: Frame | None = None
    behind: int = 0
    #: Set once the take is stopped and only its log is left to draw.
    stopped: bool = False
    _blocks: deque = field(default_factory=deque)

    def due(self, now: float) -> float | None:
        """The time of the take's next frame if it is due by `now`: up to half a frame early,
        so each pass of a loop at the take's rate makes one."""
        at = self.start + (self.slot + 1) / self.fps
        return at if at <= now + 0.5 / self.fps else None

    def keep(self, net, latent, settings, push) -> Frame:
        """Log the take's next frame."""
        self.slot += 1
        parts = [np.asarray(latent, np.float32).reshape(-1), np.asarray(settings, np.float32)]
        if push is not None:
            parts.append(np.asarray(push, np.float32).reshape(-1))
        width = sum(p.size for p in parts)
        block = self._blocks[-1] if self._blocks else None
        if (block is None or block.net is not net or block.data.shape[1] != width
                or block.end == BLOCK):
            block = _Block(net, (parts[0].size, parts[1].size),
                           None if push is None else tuple(np.shape(push)), self.slot,
                           np.empty((BLOCK, width), np.float32))
            self._blocks.append(block)
        np.concatenate(parts, out=block.data[block.end])
        block.end += 1
        self.behind += 1
        self.latest = block.frame(block.end - 1)
        return self.latest

    def oldest(self) -> Frame:
        """The oldest frame not drawn yet."""
        block = self._blocks[0]
        return block.frame(block.start)

    def shift(self) -> None:
        """The oldest frame has been drawn."""
        block = self._blocks[0]
        block.start += 1
        self.behind -= 1
        if block.start == block.end and (block.end == BLOCK or len(self._blocks) > 1):
            self._blocks.popleft()
