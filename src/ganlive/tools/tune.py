"""Find how each layer of a model runs fastest on this machine's GPU, and remember it: every
later load of the model uses what was found.

    ganlive tune runs/lichen/lichen.pt                 # on the backend ganlive plays it on
    ganlive tune runs/engine/lichen --backend d3d12

Each layer tries its plans in turn (matrix product or direct convolution, tile sizes,
splits), and keeps the one that makes a whole frame faster while the picture stays right.
It takes a few minutes, longer on a backend whose shader compiler is slow.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ganlive.checkpoints import is_engine
from ganlive.engine.load import converted
from ganlive.engine.runner import _device, fastest, read_folder, remember_plans
from ganlive.engine.tune import tune
from ganlive.tools import add_backend, parser


def main(argv=None) -> int:
    ap = parser("tune", __doc__)
    ap.add_argument("model", type=Path, help="an engine model folder or a checkpoint")
    add_backend(ap)
    args = ap.parse_args(argv)
    folder = args.model if is_engine(args.model) else converted(args.model)[0]
    manifest, weights = read_folder(folder)
    try:
        model, report = fastest(manifest, weights, backend=args.backend)
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    adapter = model.device.adapter
    model.destroy()
    model.device.destroy()
    print(f"tuning {folder.name} on {report['best']}", flush=True)
    device = _device(adapter)
    plans, best, start, model = tune(manifest, weights, device,
                                     log=lambda line: print(line, flush=True))
    model.destroy()
    device.destroy()
    remember_plans(manifest, adapter, plans)
    print(f"{start:.2f} -> {best:.2f} ms a frame, {len(plans)} layers changed. Remembered.")
    return 0
