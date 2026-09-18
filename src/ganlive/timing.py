"""How a run is summarised: a median, and the slope a median hides."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def stat_ms(samples, budget_ms: float | None = None) -> dict:
    """The summary every timing loop here reports."""
    xs = sorted(float(x) for x in samples)
    if not xs:
        return {"median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "n": 0}
    out = {"median": round(statistics.median(xs), 3),
           "p95": round(xs[min(len(xs) - 1, int(len(xs) * 0.95))], 3),
           "max": round(xs[-1], 3),
           "mean": round(statistics.fmean(xs), 3),
           "n": len(xs)}
    if budget_ms is not None:
        out["over_budget"] = sum(1 for x in xs if x > budget_ms)
    return out


def drift_ms(samples, seconds: float) -> dict:
    """The first fifth of a run against the last, as a slope in ms per second.

    Beside `stat_ms`, because a median hides a slope: a graph replay was gaining
    0.066 ms every second and every 15 s median read fine. `samples` in run order."""
    xs = [float(x) for x in samples]
    fifth = max(1, len(xs) // 5)
    first, last = statistics.median(xs[:fifth]), statistics.median(xs[-fifth:])
    return {"first_ms": round(first, 3), "last_ms": round(last, 3),
            "ms_per_s": round((last - first) / max(seconds * 0.8, 1e-9), 4)}


def write_metrics(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
