"""Where a file goes, and getting it written. Two functions nothing else here depends on.

Both were living inside something larger and being imported back out of it, upward:
`next_path` sat in `record/video.py`, so `presets.py` -- a rules module -- imported the video
recorder to name a settings file; and `remember` sat in `presets.py`, so `control/midi.py`
imported the preset library, and with it the recorder and the walk, to write a line of text
when a knob is learned. That one is `wire`'s whole cost for importing eight status bytes.

Neither has anything to do with presets or with video.
"""

from __future__ import annotations

from pathlib import Path


def next_path(folder: Path, stem: str, suffix: str) -> Path:
    """`folder/stem-01.suffix`, at the lowest number not already taken."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # One directory read, not one stat per candidate. This is called from the frame
    # loop, where a session that has saved 200 stills paid 200 sequential syscalls
    # inside a single frame.
    taken = {p.stem.rsplit("-", 1)[-1] for p in folder.glob(f"{stem}-*{suffix}")}
    used = {int(t) for t in taken if t.isdigit()}
    n = next((i for i in range(1, 1000) if i not in used), None)
    if n is None:
        raise FileExistsError(f"a thousand {stem} files in {folder}")
    return folder / f"{stem}-{n:02d}{suffix}"


def remember(path: Path, text: str) -> str:
    """Write a settings file, creating its folder. Returns what went wrong, or ""."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:                                                  # noqa: BLE001
        return f"{path}: {exc}"
    return ""
