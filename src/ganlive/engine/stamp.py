"""Whether a checkpoint's engine model is current, answered without loading PyTorch.

A conversion is stamped with what it was made from: the converter, as the bytes of every
ganlive module it imports, and the checkpoint, as its SHA-1. A model whose stamp matches is
played as it is. Any change to either converts it again at its next load.
"""

from __future__ import annotations

import ast
import functools
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]           # src/ganlive


def _sources(module: str, seen: set[str]) -> None:
    """Every ganlive module `module` imports, transitively, by dotted name."""
    if module in seen:
        return
    path = PACKAGE.parent / Path(*module.split("."))
    file = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    if not file.is_file():
        return
    seen.add(module)
    for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for name in names:
            if name.startswith("ganlive"):
                _sources(name, seen)


def made_by() -> str:
    """The converter's identity (`code_hash`)."""
    return code_hash("ganlive.engine.convert")


@functools.cache
def code_hash(module: str) -> str:
    """A hash of every ganlive module `module` imports, itself included. Read from the files,
    not from what is loaded, so that every tool computes the same one."""
    seen: set[str] = set()
    _sources(module, seen)
    digest = hashlib.sha1()
    for module in sorted(seen):
        path = PACKAGE.parent / Path(*module.split("."))
        file = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        digest.update(module.encode() + b"\0" + file.read_bytes())
    return digest.hexdigest()[:12]


def stamp_for(checkpoint, floor: float, grain: bool, known: dict | None = None) -> dict:
    """What a conversion of `checkpoint` with these options is stamped with. `known`, a stamp
    already on disk, saves hashing the checkpoint again when its size and modification time
    are the ones it records."""
    st = Path(checkpoint).stat()
    if known and known.get("size") == st.st_size and known.get("mtime_ns") == st.st_mtime_ns:
        digest = known["checkpoint"]
    else:
        sha = hashlib.sha1()
        with open(checkpoint, "rb") as f:            # in pieces: a checkpoint is 134 MB
            for piece in iter(lambda: f.read(1 << 22), b""):
                sha.update(piece)
        digest = sha.hexdigest()
    return {"made_by": made_by(), "checkpoint": digest, "floor": floor, "grain": grain,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def same(a: dict | None, b: dict) -> bool:
    """Whether two stamps describe the same conversion. Size and time only speed the check."""
    keys = ("made_by", "checkpoint", "floor", "grain")
    return a is not None and all(a.get(k) == b[k] for k in keys)
