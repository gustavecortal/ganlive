"""Pin the whole suite to the CPU.

Not a convenience: a suite running on the card would compete with whatever else is using it.
`device.detect_backend` reads this variable, so every path that resolves a device on its own
lands on the CPU.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _force_cpu():
    """Session-wide, automatic, and not overridable by a test -- that is the point."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GANLIVE_DEVICE", "cpu")
        yield
