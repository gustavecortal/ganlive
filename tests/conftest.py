"""Pin the whole suite to the CPU.

Not a convenience: a suite that silently becomes a second GPU job went off once during
development. `device.detect_backend` reads this variable, so every path that resolves a
device on its own lands on the CPU.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _force_cpu():
    """Session-wide, automatic, and not overridable by a test -- that is the point."""
    previous = os.environ.get("GANLIVE_DEVICE")
    os.environ["GANLIVE_DEVICE"] = "cpu"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GANLIVE_DEVICE", None)
        else:
            os.environ["GANLIVE_DEVICE"] = previous
