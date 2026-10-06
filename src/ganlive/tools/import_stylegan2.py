"""Convert an NVIDIA StyleGAN2-ADA pickle into a checkpoint that runs without NVIDIA's code.

    git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch
    curl -L -o ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    ganlive import-stylegan2 ffhq.pkl --repo stylegan2-ada-pytorch

Runs once, offline, and needs their repository importable: opening the pickle runs their
code, because `torch_utils.persistence` re-executes each class's pickled source. What it
writes needs nothing but torch.

By default it checks the conversion (`--no-check` skips it): the same latent through both
generators in full precision. The two compute the modulated convolution in a different order,
so they differ by float rounding, about 0.0001 8-bit levels; a wrong weight differs by whole
levels, so anything past `TOLERANCE` is refused.

Only the generator is written; a discriminator in the pickle is ignored.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

from ganlive.files import size_mb
from ganlive.models import stylegan2 as S2
from ganlive.models.common import host_latent
from ganlive.pixels import levels, worst_levels
from ganlive.tools import parser

#: The largest fp32 difference from NVIDIA's generator, in 8-bit levels, that is still rounding.
TOLERANCE = 0.1


def open_pickle(pkl: Path, repo: Path):
    """Their loader, with numpy scalars in the result turned into Python numbers."""
    sys.path.insert(0, str(repo))
    try:
        import dnnlib
        import legacy
    except ImportError as exc:
        raise SystemExit(f"--repo {repo} is not a checkout of NVlabs/stylegan2-ada-pytorch "
                         f"({exc}). Opening one of their pickles runs their code; there is "
                         f"no way around it, which is why this is a one-time conversion.") from exc

    with dnnlib.util.open_url(str(pkl)) as f:
        blob = legacy.load_network_pkl(f)
    G = blob["G_ema"].eval()
    # Their persistence layer restores numpy scalars where the class annotates plain numbers,
    # and numpy float64 scalars reach arithmetic some devices cannot do.
    for module in G.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, np.floating):
                setattr(module, name, float(value))
            elif isinstance(value, np.integer):
                setattr(module, name, int(value))
    return G


def check(G, cfg: S2.Config, state: dict, seed: int) -> tuple[float, float]:
    """Both networks, one latent, full precision on the host: the mean and the largest
    difference, in 8-bit levels."""
    z = torch.from_numpy(host_latent(cfg.z_dim, seed))
    ours = S2.load(dataclasses.replace(cfg, half_from=S2.SINGLE_EVERYWHERE), state)
    with torch.no_grad():
        a, b = ours(z), G(z, None, noise_mode="const", force_fp32=True)
    return levels(a, b), worst_levels(a, b)


def main(argv=None) -> int:
    p = parser("import-stylegan2", __doc__)
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
        if worst > TOLERANCE:
            raise SystemExit(f"the conversion differs by more than {TOLERANCE} levels, so it is "
                             f"wrong. Nothing written.")

    S2.save(out, cfg, state)
    print(f"wrote {out} ({size_mb(out):.0f} MB)\n"
          f"  ganlive.models.stylegan2.from_file({str(out)!r}, device) opens it with no other "
          f"code in the room.", flush=True)
    return 0
