"""Pin the whole suite to the CPU.

Not a convenience: this machine has one GPU and runs long training jobs on it, and a
suite that silently becomes a second GPU job went off once during development. Anything
reaching `device.get_caps()` would otherwise pick the accelerator. `get_caps` is cached,
so the cache is cleared after the variable is set and again on the way out.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _force_cpu():
    """Session-wide, automatic, and not overridable by a test -- that is the point."""
    from ganlive import device as dev

    previous = os.environ.get("GANLIVE_DEVICE")
    os.environ["GANLIVE_DEVICE"] = "cpu"
    dev.get_caps.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GANLIVE_DEVICE", None)
        else:
            os.environ["GANLIVE_DEVICE"] = previous
        dev.get_caps.cache_clear()
