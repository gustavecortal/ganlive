"""Small numeric helpers for dial curves: clamping to [0, 1] and reading a piecewise-linear curve.

No imports, so the control layer and the tools can use them without pulling in torch.
"""
from __future__ import annotations


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def evenly(values) -> tuple[tuple[float, float], ...]:
    """A curve given as values at even spacing, as `(position, value)` pairs."""
    values = tuple(values)
    last = max(1, len(values) - 1)
    return tuple((i / last, float(v)) for i, v in enumerate(values))


def at(points, x: float) -> float:
    """A dial's value at position `x`, walking the line between its measured points."""
    # By index rather than `zip(points, points[1:])`, which allocates a tuple on every call;
    # this runs once per dial per frame.
    x = clamp01(x)
    for i in range(1, len(points)):
        x1, v1 = points[i]
        if x <= x1:
            x0, v0 = points[i - 1]
            return v0 if x1 == x0 else v0 + (v1 - v0) * (x - x0) / (x1 - x0)
    return points[-1][1]
