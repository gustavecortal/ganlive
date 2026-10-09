"""Pin the whole suite to the CPU.

Not a convenience: a suite running on the card would compete with whatever else is using it.
`device.detect_backend` reads this variable, so every path that resolves a device on its own
lands on the CPU. The machine's measurements (`runner.cache_dir`) go to a folder of the suite's
own, so a run neither reads nor rewrites the user's.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _force_cpu(tmp_path_factory):
    """Session-wide, automatic, and not overridable by a test -- that is the point."""
    cache = tmp_path_factory.mktemp("cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GANLIVE_DEVICE", "cpu")
        for name in ("LOCALAPPDATA", "XDG_CACHE_HOME"):
            mp.setenv(name, str(cache))
        mp.setenv("HOME", str(cache))           # macOS keeps it under ~/Library/Caches
        yield
