"""Small file helpers: numbering new files, writing settings and JSON, and file sizes.

No imports from the rest of the package, so any layer can use them.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Where ganlive keeps what it remembers between runs.
SETTINGS = Path("runs/ganlive/settings")
#: The drum-to-audio-channel map, in `--map`'s own words: written by `ganlive doctor --learn`
#: or by `play --map`, read by `play` when no `--map` is given.
CHANNEL_MAP = SETTINGS / "channels.txt"


def next_path(folder: Path, stem: str, suffix: str) -> Path:
    """`folder/stem-01.suffix`, at the lowest number not already taken."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # One directory listing rather than a stat per candidate: this can run inside a frame.
    # The number right after the stem, so a take renamed after it (`take-07-hits-...`) keeps it.
    taken = {p.stem[len(stem) + 1:].split("-", 1)[0] for p in folder.glob(f"{stem}-*{suffix}")}
    used = {int(t) for t in taken if t.isdigit()}
    n = next((i for i in range(1, 1000) if i not in used), None)
    if n is None:
        raise FileExistsError(f"a thousand {stem} files in {folder}")
    return folder / f"{stem}-{n:02d}{suffix}"


def write_json(path: str | Path, payload, indent: int = 2) -> None:
    """Write `payload` as JSON, creating the folder."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def remember(path: Path, text: str) -> str:
    """Write a settings file, creating its folder. Returns what went wrong, or "".

    Never raises an `OSError`: a setting that cannot be saved is reported, not fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return f"{path}: {exc}"
    return ""


def size_mb(path: str | Path) -> float:
    """The file's size in MB, or 0 if it does not exist."""
    p = Path(path)
    return p.stat().st_size / 1e6 if p.exists() else 0.0
