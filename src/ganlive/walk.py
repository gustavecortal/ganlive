"""The latent walk: a great-circle path through seeded latents, addressed by beat.

`SlerpWalk` turns a musical position into the generator's latent. The clock and the timing
of a move (`MusicalClock`, `shape`, `position`, `WalkConfig`) live in `clock`, which is
torch-free, and are re-exported here.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ganlive.clock import (  # noqa: F401  re-exported
    BEATS_PER_BAR,
    MusicalClock,
    WalkConfig,
    _ease_in,
    _ease_out,
    _smoothstep,
    position,
    shape,
)


def _angle(z0: np.ndarray, z1: np.ndarray) -> float:
    """The angle between two latents, in radians."""
    n0 = z0 / np.linalg.norm(z0)
    n1 = z1 / np.linalg.norm(z1)
    return float(np.arccos(np.clip(float((n0 * n1).sum()), -1.0, 1.0)))


def _weights(omega: float, t: float) -> tuple[float, float]:
    """`(a, b)` such that `a * z0 + b * z1` is the point `t` of the way along the great circle
    between two latents `omega` apart. A straight line when they are nearly parallel, where the
    arc formula divides by almost zero."""
    if abs(omega) < 1e-6:
        return 1.0 - t, t
    so = math.sin(omega)
    return math.sin((1.0 - t) * omega) / so, math.sin(t * omega) / so


class SlerpWalk:
    """Great-circle walk over a deterministic seed sequence, addressed by musical position.

    Segment `k` runs from seed `k` to seed `k + 1`, and each seed is a pure function of `k` and
    the config, so any beat can be jumped to and a rewind returns the same picture."""

    #: How many host buffers the latent rotates through, so a buffer is not overwritten while
    #: an earlier frame's copy to the device may still be reading it.
    STAGING = 3

    #: The width every seed is drawn at, before any model sees it. A model reads the first `nz`
    #: entries, so models of different latent widths in one bank walk the same sequence.
    CANON = 4096

    #: Every config field that chooses WHICH seed a segment lands on. The endpoint cache is keyed
    #: on these, so a change is seen at once. `spread` is not one: it changes how far, not which.
    SEED_FIELDS = ("base_seed", "loop_segments", "home_every")

    def __init__(self, nz: int, device, config: WalkConfig | None = None,
                 dtype=torch.float32) -> None:
        self.nz, self.device, self.dtype = nz, device, dtype
        self.cfg = config or WalkConfig()
        self._k: tuple | None = None
        self._z1 = None
        self._omega = 0.0
        self._home: tuple | None = None
        self.segment_index = 0
        self._offset_key: tuple | None = None
        self._offset_rows = None
        self._offset_vec = None
        #: The offset array last handed to a `w` seam, by identity. `_offset` returns the same
        #: array while the amounts hold still, which keeps a host-to-device copy off the frame path.
        self._pushed = None
        self._slot = 0
        self._staging, self._views = self._staging_ring()
        self._pair = None
        self._coef = np.zeros(2, dtype=np.float32)
        self._scratch = np.zeros(nz, dtype=np.float32)

    def _staging_ring(self):
        """The pinned host ring the latent is written into, at the current width."""
        from ganlive.pixels import pinned

        bufs = [pinned((1, self.nz), self.dtype) for _ in range(self.STAGING)]
        return bufs, [buf.numpy().reshape(-1) for buf in bufs]

    def retarget(self, nz: int) -> None:
        """Hand this walk to a generator of a different latent width, mid-set."""
        nz = int(nz)
        if nz == self.nz:
            return
        self.nz = nz
        self._offset_key = self._offset_rows = self._offset_vec = None
        self._scratch = np.zeros(nz, dtype=np.float32)
        self._staging, self._views = self._staging_ring()
        # The cached endpoints are the old width's. The position is not cached: segment and
        # phase are read back from the beat, so the walk resumes where it was.
        self._k = self._z1 = self._pair = None

    @property
    def canon(self) -> int:
        """The width the walk itself runs at, above every model that reads it."""
        return max(self.CANON, self.nz)

    def _draw(self, k: int, salt: int = 0) -> torch.Tensor:
        """A Gaussian draw that depends only on `(base_seed, salt, k)`. Stays on the host.

        Drawn with torch's generator, so a seed gives the same picture it always has."""
        g = torch.Generator(device="cpu").manual_seed(
            (self.cfg.base_seed * 1_000_003 + salt * 7_919 + k) & 0x7FFF_FFFF)
        return torch.randn(self.canon, generator=g)

    def home_for(self, k: int) -> torch.Tensor:
        """The neighbourhood segment `k` is drawn around. Salted so `home_0` is not `seed_0`."""
        return torch.from_numpy(self._home_canon(k)[:self.nz].copy())

    def _home_canon(self, k: int) -> np.ndarray:
        """The home at the walk's own width. Every `home_every` segments share one, so the last
        draw is kept and only redrawn when the block or the seed changes."""
        block = k // self.cfg.home_every if self.cfg.home_every else 0
        key = (self.cfg.base_seed, block, self.canon)
        if self._home is None or self._home[0] != key:
            self._home = (key, self._draw(block, salt=1).numpy())
        return self._home[1]

    def seed_for(self, k: int) -> torch.Tensor:
        """Segment `k`'s target latent, at the loaded model's width."""
        return torch.from_numpy(self._seed_canon(k)[:self.nz])

    def _seed_canon(self, k: int) -> np.ndarray:
        """The same target at the walk's own width. A pure function of `k` and the config.

        `spread` below 1 pulls the raw draw toward its home along the great circle between
        them, so successive seeds stay in one neighbourhood."""
        if self.cfg.loop_segments:
            k %= self.cfg.loop_segments
        target = self._draw(k).numpy()
        if self.cfg.spread >= 1.0:
            return target
        home = self._home_canon(k)
        a, b = _weights(_angle(home, target), max(0.0, self.cfg.spread))
        return (np.float32(a) * home + np.float32(b) * target).astype(np.float32)

    def _key(self, k: int) -> tuple:
        """What the cached endpoints for segment `k` depend on."""
        return (k, *(getattr(self.cfg, name) for name in self.SEED_FIELDS))

    def _load(self, k: int) -> None:
        """Cache the segment's endpoints and the angle between them."""
        key = self._key(k)
        if self._k == key:
            return
        # Walking forward, this segment's start is the last one's end.
        z0 = self._z1 if self._k == self._key(k - 1) and self._z1 is not None else None
        if z0 is None:
            z0 = self._seed_canon(k)[:self.nz].astype(np.float32)
        z1 = self._seed_canon(k + 1)[:self.nz].astype(np.float32)
        self._z1 = z1
        self._pair = np.stack([z0, z1])
        self._omega = _angle(z0, z1)
        self._k = key
        self.segment_index = k

    def latent(self, beats: float) -> torch.Tensor:
        """The `(1, nz)` input for this musical position, on `device`.

        A host array instead when `cfg.latent_on_host` says the generator reads it there."""
        k, _u, t = position(self.cfg, beats)
        self._load(k)
        self._coef[0], self._coef[1] = _weights(self._omega, t)

        self._slot = (self._slot + 1) % self.STAGING
        view = self._views[self._slot]
        offset = self._offset()
        if self.cfg.push_into is not None:
            # A `w` push never touches the latent: the mapping opens with a pixel norm that
            # cancels any scaling of `z`, so the push is applied after the mapping instead.
            self._hand_over(offset)
            offset = None
        if view.dtype == np.float32:
            np.dot(self._coef, self._pair, out=view)
            if offset is not None:
                view += offset
        else:
            np.dot(self._coef, self._pair, out=self._scratch)
            if offset is not None:
                self._scratch += offset
            view[:] = self._scratch
        if self.cfg.latent_on_host:
            return view
        return self._staging[self._slot].to(self.device, non_blocking=True)

    def _hand_over(self, offset) -> None:
        """Put a `w` push where the generator reads it, and only when it changed.

        Remembered together with the tensor it went into: a model switch points `push_into` at
        another model's seam, and switching back must still clear or rewrite the first one."""
        into = self.cfg.push_into
        if self._pushed is not None and self._pushed[0] is offset and self._pushed[1] is into:
            return
        if into.device.type == "cpu":
            # A captured generator reads the push from this host buffer: a numpy write, with
            # no torch small-tensor overhead and nothing issued to the card.
            view = into.numpy().reshape(-1)
            view[:] = 0.0 if offset is None else offset.reshape(-1)
        elif offset is None:
            into.zero_()
        else:
            into.copy_(torch.from_numpy(offset).reshape(into.shape))
        self._pushed = (offset, into)

    def _offset(self):
        """The summed direction push, or `None` when every dial is at rest."""
        rows = self.cfg.directions
        amounts = self.cfg.amounts
        if rows is None or not any(amounts):
            self._offset_key = self._offset_rows = self._offset_vec = None
            return None
        # A push folded into the latent has to be the latent's width. A mismatch means the walk
        # is between two models mid-switch, and the push is skipped for that frame.
        if self.cfg.push_into is None and getattr(rows, "shape", (0,))[-1] != self.nz:
            return None
        # Keyed on the basis as well as the amounts: a model switch changes the basis without
        # moving any dial.
        amounts = tuple(amounts)
        if amounts != self._offset_key or rows is not self._offset_rows:
            n = min(len(amounts), len(rows))
            self._offset_vec = np.asarray(amounts[:n], dtype=np.float32) @ rows[:n]
            self._offset_key, self._offset_rows = amounts, rows
        return self._offset_vec
