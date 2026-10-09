"""The probe: what PyTorch draws for a fixed latent, kept as block means, which a backend must
match before it is used."""

from __future__ import annotations

import numpy as np

from ganlive.levels import LEVEL

#: A probe is the picture's mean over this grid of blocks, coarse enough that fp16 rounding
#: averages out and fine enough that a backend drawing the wrong picture cannot match it.
PROBE_GRID = (16, 24)
#: How far a backend's probe may stray, in 8-bit levels averaged over the blocks.
PROBE_LEVELS = 2.0


def box_means(image: np.ndarray) -> np.ndarray:
    """Block means in [-1, 1], (3, 16, 24), of a (3, H, W) float picture or of (H, W, 4)
    RGBA8 pixels, the latter summed as integers with no full-size float copy."""
    gh, gw = PROBE_GRID
    pixels = image.dtype == np.uint8
    h, w = image.shape[:2] if pixels else image.shape[1:]
    # Block i spans rows [i*h/gh, (i+1)*h/gh), so any size divides into the grid.
    rows, cols = (np.arange(n) * size // n for n, size in ((gh, h), (gw, w)))
    area = np.outer(np.diff(rows, append=h), np.diff(cols, append=w))
    if pixels and h % gh == 0 and w % gw == 0:
        # Equal blocks, as every engine model's are: summed by rows of blocks first, ten
        # times faster than `reduceat` on a 3072x2048 frame.
        bands = image.reshape(gh, h // gh, -1).sum(axis=1, dtype=np.uint32)
        sums = bands.reshape(gh, gw, w // gw, -1).sum(axis=2)[..., :3]
        return (sums.transpose(2, 0, 1) / area) / LEVEL - 1.0
    if pixels:
        sums = np.add.reduceat(np.add.reduceat(image[..., :3], rows, axis=0, dtype=np.uint64),
                               cols, axis=1)
        return (sums.transpose(2, 0, 1) / area) / LEVEL - 1.0
    sums = np.add.reduceat(np.add.reduceat(image, rows, axis=1, dtype=np.float64), cols, axis=2)
    return sums / area


def probe_error(probe: dict, image: np.ndarray) -> float:
    """How far a drawing of the probe's latent is from PyTorch's, in 8-bit levels averaged over
    the blocks: what a backend must keep under PROBE_LEVELS to be used."""
    return float(np.abs(box_means(image).ravel() - np.asarray(probe["means"])).mean() * LEVEL)
