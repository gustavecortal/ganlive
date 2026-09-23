"""The generator's whole control state as one vector, and getting it to the card cheaply.

Plumbing, not dials: it knows how many settings there are and what they are called, and
nothing about what any of them means. It lived in `dials.steer`, which made
`models.onnx.OnnxGenerator` -- which owns one, and reads it on the frame path -- import
upward out of `models` into `dials`. That was the last edge of its kind, and with it gone
`models` depends on nothing above itself.
"""

from __future__ import annotations

import numpy as np
import torch

from ganlive.pixels import pinned


class Knobs:
    """The generator's whole control state as one vector, plus the names to address it by."""

    STAGING = 3

    def __init__(self, names, device, dtype) -> None:
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.vec = torch.ones(len(self.names), device=device, dtype=dtype)
        self.host = [pinned((len(self.names),), dtype).fill_(1.0) for _ in range(self.STAGING)]
        self._writes = [buf.numpy() for buf in self.host]
        self.pinned = all(buf.is_pinned() for buf in self.host)
        self._slot = 0
        self.write = self._writes[0]
        #: What the card holds. Starts at the neutral the vector is built holding, so a frame
        #: that moves nothing sends nothing -- see `commit`. Once `feed_from` is called it *is*
        #: the buffer the card reads, and the ring below stops being the sender.
        self._sent = np.ones(len(self.names), dtype=self._writes[0].dtype)
        self._fed = False
        self.skipped = 0
        self.sites: dict[str, int] = {}

    def view(self, name: str) -> torch.Tensor:
        """The slice of the vector a module should hold. A view, so `commit` reaches it."""
        i = self.index[name]
        return self.vec[i:i + 1]

    def span(self, names) -> torch.Tensor:
        """Several adjacent settings as one view, for a module that wants them as a vector."""
        first = self.index[names[0]]
        if [self.index[n] for n in names] != list(range(first, first + len(names))):
            raise ValueError(
                f"{list(names)} are not adjacent in {self.names}, so they cannot be one view. "
                f"Order them together when the vector is built.")
        return self.vec[first:first + len(names)]

    def set(self, name: str, value: float) -> None:
        """Write one setting. A name this model does not have is accepted and dropped."""
        i = self.index.get(name)
        if i is not None:
            self.write[i] = value

    def reset(self) -> None:
        """Back to the trained neutral: every setting 1.0, the identity for all of them."""
        self.write[:] = 1.0
        self.commit()

    def committed(self) -> object:
        """The values last committed, as a host array."""
        return self._sent

    def feed_from(self, host: torch.Tensor) -> None:
        """From now on the card reads `vec` from `host` on every replay -- a captured graph
        uploads it itself, see `models.capture.capture` -- so a commit is a host write into it and
        nothing is sent. `host` already holds what the card holds."""
        self._sent, self._fed = host.numpy(), True

    def commit(self) -> None:
        """Send this frame's settings to the card -- unless the card already has them.

        **The copy is thirty-two bytes and it measured 1.6 ms.** `Tensor.copy_` drops the GIL,
        and taking it back from a window thread that is uploading a texture and presenting
        costs up to one switch interval: 5 ms by default, 1.64 median and 3.38 at p95 on the
        played loop, against 0.03 with no window open. A frame on which no dial moved was
        paying all of that for a copy of the numbers already there, and at rest no dial moves
        -- every dial on its own rest and nothing wired is what the instrument starts on.
        Comparing the eight floats first is about a microsecond against that."""
        if np.array_equal(self.write, self._sent):
            self.skipped += 1
            return
        np.copyto(self._sent, self.write)
        if self._fed:
            return                      # the graph reads `_sent` itself on the next replay
        self.vec.copy_(self.host[self._slot], non_blocking=True)
        nxt = (self._slot + 1) % self.STAGING
        self._writes[nxt][:] = self.write
        self._slot = nxt
        self.write = self._writes[nxt]
