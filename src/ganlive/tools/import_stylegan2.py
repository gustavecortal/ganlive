"""Convert an NVIDIA StyleGAN2-ADA pickle into a checkpoint that runs without NVIDIA's code.

    git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch
    curl -L -o ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    ganlive import-stylegan2 ffhq.pkl --repo stylegan2-ada-pytorch

Runs once, offline, and needs their repository importable -- opening the pickle *is* running
their code, because `torch_utils.persistence` re-executes each class's pickled source. What it
writes needs nothing but torch.

The check this rests on runs by default, and `--no-check` skips it: the same latent through
both generators in full precision, reported in 8-bit levels. It measures 0.000 on `ffhq.pkl`
and `afhqcat.pkl`. Anything else means the conversion is wrong -- fp32 against fp32 either
agrees or does not.

**A generator is what this writes.** A discriminator in the pickle is ignored: this project
plays models, and the only thing a discriminator is for is adversarial finetuning, which
happens elsewhere. A trainer that wants one converts the pickle with its own importer.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from ganlive.models import stylegan2 as S2

#: A `half_from` above any resolution, so every block runs fp32. The comparison against the
#: original is the one thing here that must not have a rounding argument available to it.
FULL_PRECISION = 1 << 30


def open_pickle(pkl: Path, repo: Path):
    """Their loader, with the one thing this card cannot survive taken out of the result."""
    sys.path.insert(0, str(repo))
    try:
        import dnnlib
        import legacy
    except ImportError as exc:
        raise SystemExit(f"--repo {repo} is not a checkout of NVlabs/stylegan2-ada-pytorch "
                         f"({exc}). Opening one of their pickles runs their code; there is "
                         f"no way around it, which is why this is a one-time conversion.") from exc
    import numpy as np

    with dnnlib.util.open_url(str(pkl)) as f:
        blob = legacy.load_network_pkl(f)
    G = blob["G_ema"].eval()
    # Their persistence layer restores numpy scalars where the class annotates plain numbers,
    # and those reach arithmetic this card cannot do.
    for module in G.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, np.floating):
                setattr(module, name, float(value))
            elif isinstance(value, np.integer):
                setattr(module, name, int(value))
    return G


def check(G, cfg: S2.Config, state: dict, seed: int) -> tuple[float, float]:
    """Both networks, one latent, full precision on the host. The unit is 8-bit levels."""
    import numpy as np
    import torch

    z = torch.from_numpy(np.random.default_rng(seed)
                         .standard_normal((1, cfg.z_dim)).astype(np.float32))
    # `half_from` and not `num_fp16_res=0`: one spelling of "run this in full precision",
    # and it is the one both configs share.
    ours = S2.load(dataclasses.replace(cfg, half_from=FULL_PRECISION), state)
    with torch.no_grad():
        gap = (ours(z) - G(z, None, noise_mode="const", force_fp32=True)).abs() * 127.5
    return float(gap.mean()), float(gap.max())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ganlive import-stylegan2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pickle", type=Path, help="an NVIDIA StyleGAN2-ADA .pkl")
    p.add_argument("--repo", type=Path, default=Path("stylegan2-ada-pytorch"),
                   help="a checkout of NVlabs/stylegan2-ada-pytorch")
    p.add_argument("--out", type=Path, default=None, help="default: runs/stylegan2/<name>.pt")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--no-check", action="store_true",
                   help="skip the comparison against the original. Do not.")
    args = p.parse_args(argv)

    if not args.pickle.exists():
        raise SystemExit(f"{args.pickle} is not there.")
    out = args.out or Path("runs/stylegan2") / f"{args.pickle.stem}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    G = open_pickle(args.pickle, args.repo)
    cfg = S2.config_from(G)
    state = G.state_dict()
    print(f"{args.pickle.name}: {cfg.img_resolution}px, {cfg.img_channels} channels, "
          f"{sum(v.numel() for v in state.values()) / 1e6:.2f}M parameters, "
          f"half precision from {cfg.fp16_from}px up", flush=True)

    S2.load(cfg, state)                       # refuses here if the weights are not this shape
    if not args.no_check:
        mean, worst = check(G, cfg, state, args.seed)
        print(f"against the original, in fp32: {mean:.6f} mean {worst:.6f} max 8-bit levels",
              flush=True)
        if worst > 0.0:
            raise SystemExit("the conversion is not exact, so it is wrong. Nothing written.")

    S2.save(out, cfg, state)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)\n"
          f"  ganlive.models.stylegan2.from_file({str(out)!r}, device) opens it with no other "
          f"code in the room.", flush=True)
    return 0

