"""Convert a FastGAN checkpoint into an engine model, which plays through WebGPU on any GPU and in
a browser, without PyTorch.

    ganlive convert runs/lichen/lichen.pt            ->  runs/engine/lichen/

The folder holds `weights.bin` (fp16) and `program.json`: every compute shader of a frame, the
buffers they use, and a probe of the picture PyTorch draws, which a backend must match before
it is trusted. Converting needs PyTorch, on the CPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ganlive.checkpoints import checkpoint_for, slug_for
from ganlive.files import size_mb
from ganlive.tools import parser


def main(argv=None) -> int:
    ap = parser("convert", __doc__)
    ap.add_argument("checkpoint", type=Path, help="a FastGAN checkpoint, or a run folder for its newest")
    ap.add_argument("--out", type=Path, default=None,
                    help="destination folder. Default: runs/engine/<model>")
    args = ap.parse_args(argv)
    try:
        checkpoint = checkpoint_for(args.checkpoint)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    from ganlive.engine.convert import convert  # PyTorch loads only once there is work

    out = args.out or Path("runs/engine") / slug_for(checkpoint)
    started = time.perf_counter()
    try:
        program = convert(checkpoint, out)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{checkpoint.name} -> {out}: {program['width']}x{program['height']}, "
          f"{len(program['steps'])} steps, {size_mb(out / 'weights.bin'):.1f} MB of weights, "
          f"in {time.perf_counter() - started:.1f} s")
    return 0
