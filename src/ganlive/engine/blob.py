"""The weight file a conversion writes: fp16 arrays, addressed by their offset."""

from __future__ import annotations

import numpy as np


class Blob:
    """The weight file being written: fp16 arrays, each starting and ending on a 4-byte word."""

    def __init__(self) -> None:
        self.data = bytearray()

    def put(self, a) -> int:
        """Append an array; return its offset in 16-bit elements."""
        at = len(self.data) // 2
        self.data += np.asarray(a, dtype="<f2").tobytes()
        self.data += bytes(-len(self.data) % 4)
        return at

    def conv(self, conv) -> int:
        w = conv.weight.detach().float().numpy()            # [cout][cin][3][3]
        return self.put(w.reshape(w.shape[0], w.shape[1], 9).transpose(1, 2, 0))
