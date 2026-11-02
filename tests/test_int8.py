"""The browser's 8-bit weights (engine/int8.py) expand in JavaScript (web/int8.mjs) to the very
fp16 blob that Python expands them to, whatever sizes the download arrives in."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ganlive.engine import int8

NODE = shutil.which("node")
WEB = Path(__file__).parents[1] / "src/ganlive/web/int8.mjs"

# A blob of fp16 values, some stretches matrices (output channels by columns or by rows, one of
# an odd size), the others left as they are.
TABLE = [[6, 64, 72, 1], [4614, 33, 129, 0], [8872, 513, 9, 1], [13490, 1, 4097, 0]]


def blob() -> bytes:
    rng = np.random.default_rng(5)
    size = 17600
    half = (rng.standard_normal(size) * rng.choice([1e-3, 0.05, 2.0], size)).astype("<f2")
    half[8872:8872 + 513 * 9].reshape(513, 9)[:, 0] = 0      # a channel of zeros
    return half.tobytes()


def test_8_bit_weights_are_about_half_and_close():
    data = blob()
    small = int8.quantize(data, TABLE)
    assert len(small) < 0.6 * len(data)
    back = np.frombuffer(int8.expand(small, TABLE, len(data)), "<f2").astype(np.float32)
    was = np.frombuffer(data, "<f2").astype(np.float32)
    inside = np.zeros(len(was), bool)
    for at, rows, cols, axis in TABLE:
        inside[at:at + rows * cols] = True
        w = was[at:at + rows * cols].reshape(rows, cols)
        step = np.abs(w).max(axis=1 - axis, keepdims=True) / 127
        error = np.abs(back[at:at + rows * cols].reshape(rows, cols) - w)
        assert (error <= step / 2 + np.abs(w) * 2.0**-10 + 1e-7).all()
    assert (back[~inside] == was[~inside]).all()


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
@pytest.mark.parametrize("chunk", [1, 7, 4096, 1 << 20])
def test_the_browser_expands_8_bit_weights_as_python_does(tmp_path, chunk):
    data = blob()
    (tmp_path / "w8").write_bytes(int8.quantize(data, TABLE))
    script = f"""
import {{ readFileSync, writeFileSync }} from "node:fs";
import {{ expander }} from {json.dumps(WEB.as_uri())};
const data = readFileSync({json.dumps(str(tmp_path / "w8"))});
const out = new ArrayBuffer({len(data)});
const expand = expander({json.dumps(TABLE)}, {len(data)}, out);
if (expand.bytes !== data.length) throw new Error(`${{expand.bytes}} bytes expected, ${{data.length}} made`);
for (let at = 0; at < data.length; at += {chunk}) expand.push(data.subarray(at, at + {chunk}));
expand.done();
writeFileSync({json.dumps(str(tmp_path / "out"))}, new Uint8Array(out));
"""
    run = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "out").read_bytes() == int8.expand((tmp_path / "w8").read_bytes(), TABLE, len(data))
