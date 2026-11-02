"""The weights a browser downloads: each weight matrix of the fp16 blob as int8, with one f32
scale per output channel (symmetric), and the rest of the blob as it is. The player expands
them back into the fp16 blob as they arrive (`runner.mjs`), so the shaders are unchanged and
the download is 40 to 50% smaller.

The 8-bit file walks the blob in order. A stretch outside every matrix is copied as fp16
bytes. A matrix is its scales (f32), then its values (int8, row by row), padded to 4 bytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ganlive.checkpoints import MANIFEST, PROGRAM, WEIGHTS
from ganlive.files import write_json

#: The 8-bit weights beside weights.bin, which `program["int8"]` describes.
WEIGHTS8 = "weights8.bin"

#: Matrices smaller than this stay fp16: the few bytes saved are not worth an error.
SMALLEST = 4096


def matrices(manifest: dict) -> list[list[int]]:
    """`[offset, rows, cols, axis]` of each weight matrix in the blob, in blob order: the
    offset in fp16 elements, and `axis` 1 when the output channels are the columns, 0 when
    they are the rows."""
    found = []

    def add(at, rows, cols, axis):
        if rows * cols >= SMALLEST:
            found.append([int(at), int(rows), int(cols), axis])

    if manifest.get("family") == "stylegan2":
        # The mapping network stays fp16: at 8 bits its error moves every style, and FFHQ's
        # faces visibly change.
        for op in manifest["ops"]:
            if op["op"] in ("conv", "torgb"):
                add(op["affine"], op["cin"], manifest["w_dim"], 0)
            if op["op"] == "conv":
                add(op["weight"], op["cin"] * 9, op["cout"], 1)
    else:
        tensors = manifest["tensors"]
        for op in manifest["ops"]:
            if op["op"] == "init":
                c, h, w = tensors[op["out"]]
                add(op["w"], 2 * c * h * w, manifest["nz"], 0)
            elif op["op"] == "sle":
                add(op["fc1"], tensors[op["low"]][0] * 16, op["ch"], 1)
                add(op["fc2"], op["ch"], op["ch"], 1)
            elif op["op"] == "conv":
                cin, cout = tensors[op["in"]][0], tensors[op["out"]][0]
                add(op["wv"], cin * 9, cout, 1)
                add(op["wg"], cin * 9, cout, 1)
    found.sort()
    for (at, rows, cols, _), after in zip(found, found[1:] + [[manifest["bytes"] // 2]], strict=True):
        assert at + rows * cols <= after[0], f"the matrix at {at} overlaps what follows it"
    return found


def quantize(blob: bytes, table: list[list[int]]) -> bytes:
    """The 8-bit file of the fp16 `blob`, its matrices those of `table` (`matrices`)."""
    half = np.frombuffer(blob, "<f2")
    out, at = bytearray(), 0
    for start, rows, cols, axis in table:
        out += half[at:start].tobytes()
        w = half[start:start + rows * cols].astype(np.float32).reshape(rows, cols)
        peak = np.abs(w).max(axis=1 - axis, keepdims=True)
        scale = np.where(peak > 0, peak / 127, 1).astype(np.float32)
        out += scale.astype("<f4").tobytes()
        out += np.clip(np.rint(w / scale), -127, 127).astype(np.int8).tobytes()
        out += bytes(-len(out) % 4)
        at = start + rows * cols
    out += half[at:].tobytes()
    return bytes(out)


def expand(data: bytes, table: list[list[int]], size: int) -> bytes:
    """The fp16 blob of `size` bytes that the 8-bit `data` stands for, as `runner.mjs` makes it."""
    half, src, at = np.zeros(size // 2, "<f2"), 0, 0

    def raw(n):
        nonlocal src
        half[at:at + n] = np.frombuffer(data, "<f2", n, src)
        src += 2 * n

    for start, rows, cols, axis in table:
        raw(start - at)
        k = cols if axis else rows
        scale = np.frombuffer(data, "<f4", k, src)
        q = np.frombuffer(data, np.int8, rows * cols, src + 4 * k).reshape(rows, cols)
        half[start:start + rows * cols] = (q * (scale if axis else scale[:, None])).astype("<f2").ravel()
        src += 4 * k + rows * cols
        src += -src % 4
        at = start + rows * cols
    raw(size // 2 - at)
    assert src == len(data), "the 8-bit weights do not match their table"
    return half.tobytes()


def add_int8(folder, program: dict | None = None) -> dict:
    """Write the 8-bit weights of the engine model in `folder` and name them in its program,
    which a browser then downloads instead of weights.bin. Returns the program."""
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    if program is None:
        program = json.loads((folder / PROGRAM).read_text(encoding="utf-8"))
    table = matrices(manifest)
    (folder / WEIGHTS8).write_bytes(quantize((folder / WEIGHTS).read_bytes(), table))
    program["int8"] = {"file": WEIGHTS8, "matrices": table}
    write_json(folder / PROGRAM, program, indent=None)
    return program


if __name__ == "__main__":      # python -m ganlive.engine.int8 <engine folder>...: converted models
    for name in sys.argv[1:]:
        add_int8(name)
        print(f"{name}: {(Path(name) / WEIGHTS8).stat().st_size / 1e6:.1f} MB of 8-bit weights")
