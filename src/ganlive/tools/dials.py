"""Derive a checkpoint's latent directions from the whole generator and save them beside it.

    ganlive dials runs/stylegan2/ffhq.pt

At load, directions come from SeFa: the SVD of the first affine each style range meets. That
affine is a poor proxy for the synthesis network behind it -- reading the Jacobian of the
whole generator instead took FFHQ-1024's strongest fine dial from 52 8-bit levels to 76. It
costs about 100 seconds, which is too long for a load, so it runs here and writes
`<checkpoint>.directions.pt`. The next load picks that up, or falls back to SeFa.

What is saved is a proposal, not a verdict: every basis still goes through the same
measurement at load, so a stale file cannot put a dead dial on the strip.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ganlive.dials import derive as D  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="a converted StyleGAN2 `.pt`")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--count", type=int, default=D.CANDIDATES, metavar="N",
                    help=f"candidates per style range, before measurement picks among them at "
                         f"load. Default {D.CANDIDATES}.")
    ap.add_argument("--latents", type=int, default=1, metavar="N",
                    help="latents the metric is averaged over. One is enough for the verdict: "
                         "over disjoint sets of four the basis itself agrees at only 0.38 to "
                         "0.76, yet every subset tried keeps the fine band between 67 and 77 "
                         "levels, because there are more strong directions than the strip can "
                         "show. Each costs another 1,024 passes a range.")
    args = ap.parse_args(argv)

    from ganlive import device as dev
    from ganlive.bank import open_stylegan2
    from ganlive.models import stylegan2 as S2

    if not S2.is_stylegan2(args.checkpoint):
        print(f"{args.checkpoint} is not a converted StyleGAN2, and the banded derivation has "
              f"no style ranges to read on anything else", file=sys.stderr)
        return 2
    if args.device.lower() != "cpu" and not dev.refuse_if_gpu_busy("derivation"):
        return 1

    # The same precision the instrument would open this model in on this device, or the basis
    # is derived on a model that is not the one that plays.
    dtype = dev.playback_dtype(args.device)
    net, _knobs, push, bands = open_stylegan2(args.checkpoint, args.device, dtype=dtype)
    names = [name for name, _weight in bands]
    print(f"{args.checkpoint.name}: {len(names)} style ranges, "
          f"{2 * push.shape[1] * args.latents} passes each", flush=True)

    started = time.perf_counter()
    found = D.active_banded(net, names, (args.count,) * len(names), net.cfg.z_dim,
                            args.device, dtype, into=push, seeds=args.latents)
    how = f"{args.count}/band over {args.latents} latent(s) at eps {D.PROBE_EPS:g}"
    where = D.save(found, args.checkpoint, net, how)
    print(f"{found.report()}\nwritten to {where} in {time.perf_counter() - started:.1f}s")
    return 0

