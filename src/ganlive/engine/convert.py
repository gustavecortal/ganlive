"""Convert a FastGAN or a converted StyleGAN2 checkpoint into an engine model: a folder with
`weights.bin` (fp16) and `manifest.json` (the ops, the probe and the measured dials). A load
compiles the manifest into shaders (`engine.compile`), so improving the shaders converts
nothing again. Needs PyTorch, and playing the result does not.
"""

from __future__ import annotations

import os
from pathlib import Path

from ganlive.checkpoints import MANIFEST, WEIGHTS
from ganlive.engine import convert_fastgan, convert_stylegan2
from ganlive.engine import stamp as stamps
from ganlive.files import write_json
from ganlive.levels import RANDOM_FLOOR
from ganlive.models import stylegan2 as S2


def convert(checkpoint, out: Path, device=None, *, floor: float = RANDOM_FLOOR,
            grain: bool = True, stamp: dict | None = None) -> dict:
    """Write the engine model of `checkpoint`, its dials measured on `device`, into the folder
    `out`. Returns the manifest, stamped with what it was made from (`stamp.stamp_for`) so a
    load can tell it is current."""
    family = convert_stylegan2 if S2.is_stylegan2(checkpoint) else convert_fastgan
    manifest, blob = family.measure(checkpoint, device, floor=floor, grain=grain)
    manifest["stamp"] = stamp or stamps.stamp_for(checkpoint, floor, grain)
    # The weights first and the manifest last, each renamed into place whole: a conversion cut
    # short leaves no manifest that a load would take for a finished one.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{WEIGHTS}.part").write_bytes(blob.data)
    os.replace(out / f"{WEIGHTS}.part", out / WEIGHTS)
    write_json(out / f"{MANIFEST}.part", manifest, indent=None)
    os.replace(out / f"{MANIFEST}.part", out / MANIFEST)
    return manifest
