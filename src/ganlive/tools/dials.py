"""Derive a checkpoint's latent directions from the whole generator and save them beside it.

    ganlive dials runs/stylegan2/ffhq.pt

At load, directions come from SeFa: the SVD of the first affine each style range meets. That
affine is a poor proxy for the synthesis network behind it, and reading the Jacobian of the
whole generator instead gives stronger dials (see `derive.active_banded`). It takes around
100 seconds, too long for a load, so it runs here and writes `<checkpoint>.directions.pt`,
which the next load picks up.

What is saved is a proposal: every basis is still measured at load, so a stale file cannot
put a dead dial on the strip.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ganlive.dials import derive as D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ganlive dials", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="a converted StyleGAN2 `.pt`")
    ap.add_argument("--device", default=None, metavar="xpu|cuda|mps|cpu",
                    help="default: whichever accelerator is there, else the CPU")
    ap.add_argument("--count", type=int, default=D.CANDIDATES, metavar="N",
                    help=f"candidates per style range, before measurement picks among them at "
                         f"load. Default {D.CANDIDATES}.")
    ap.add_argument("--latents", type=int, default=1, metavar="N",
                    help="latents the metric is averaged over. One is usually enough: "
                         "different latents pick different members of the same set of strong "
                         "directions. Each costs another two passes per push-buffer entry, "
                         "per range.")
    args = ap.parse_args(argv)

    from ganlive import device as dev
    from ganlive.bank import open_stylegan2
    from ganlive.models import stylegan2 as S2

    if not S2.is_stylegan2(args.checkpoint):
        print(f"{args.checkpoint} is not a converted StyleGAN2, and the banded derivation has "
              f"no style ranges to read on anything else", file=sys.stderr)
        return 2
    device = args.device or dev.detect_backend()
    if device.lower() != "cpu" and not dev.refuse_if_gpu_busy("derivation"):
        return 1

    # The precision the instrument would play this model in on this device.
    dtype = dev.playback_dtype(device)
    net, _knobs, push, bands = open_stylegan2(args.checkpoint, device, dtype=dtype)
    names = [name for name, _weight in bands]
    print(f"{args.checkpoint.name}: {len(names)} style ranges, "
          f"{2 * push.shape[1] * args.latents} passes each", flush=True)

    started = time.perf_counter()
    found = D.active_banded(net, names, (args.count,) * len(names), net.cfg.z_dim,
                            device, dtype, into=push, seeds=args.latents)
    how = f"{args.count}/band over {args.latents} latent(s) at eps {D.PROBE_EPS:g}"
    where = D.save(found, args.checkpoint, net, how)
    print(f"{found.report()}\nwritten to {where} in {time.perf_counter() - started:.1f}s")
    return 0

