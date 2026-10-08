"""Convert a FastGAN or StyleGAN2 checkpoint into an engine model, which plays through WebGPU on
any GPU and in a browser, without PyTorch.

    ganlive convert runs/lichen/lichen.pt            ->  runs/engine/lichen/

The folder holds `weights.bin` (fp16) and `manifest.json`: the model's layers, its dials
(measured here, drawn by the engine on this machine's GPU), and a probe of the picture PyTorch
draws, which a backend must match before it is trusted. `program.json` holds the compute shaders compiled from it, for the
browser runner. Converting needs PyTorch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ganlive.checkpoints import PROGRAM, WEIGHTS, checkpoint_for, slug_for
from ganlive.engine.compile import compile_manifest
from ganlive.engine.player import browser_dials
from ganlive.files import size_mb, write_json
from ganlive.tools import add_device, parser


def main(argv=None) -> int:
    ap = parser("convert", __doc__)
    ap.add_argument("checkpoint", type=Path, help="a checkpoint, or a run folder for its newest")
    ap.add_argument("--out", type=Path, default=None,
                    help="destination folder. Default: runs/engine/<model>")
    add_device(ap, help="measure the dials with PyTorch on this device rather than on the engine")
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
        manifest = convert(checkpoint, out, args.device)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    program = compile_manifest(manifest, browser=True)
    program["knobs"] = browser_dials(program)
    write_json(out / PROGRAM, program, indent=None)
    print(f"{checkpoint.name} -> {out}: {program['width']}x{program['height']}, "
          f"{len(program['steps'])} steps, {size_mb(out / WEIGHTS):.1f} MB of weights, "
          f"{len((program['dials']['directions'] or {}).get('basis', []))} directions, "
          f"in {time.perf_counter() - started:.1f} s")
    return 0
