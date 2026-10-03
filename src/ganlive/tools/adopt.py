"""Take a generator from anywhere and make it playable: fetch, export, dial, prove, save.

    ganlive adopt hf:someone/some-gan --trust-remote-code
    ganlive adopt some/model.onnx
    ganlive play --checkpoint runs/onnx/<what it printed>

What comes out is a file that stands on its own: the graph, its dials, where each rests,
what each takes at every point of its travel, and how many 8-bit levels each was measured to
be worth. The instrument reads all of that out of the file, so the measuring happens here,
once, rather than at every load.

It refuses rather than guessing: a repository whose code it was not told to import, a module
that never turned a latent into a picture, and a graph that stays non-deterministic after its
random draws have been frozen.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ganlive import device as dev
from ganlive.checkpoints import is_onnx
from ganlive.models import foreign as F
from ganlive.models import onnx_adopt as A
from ganlive.models.onnx_file import weights_file

HUB = "hf:"


def slug(source: str) -> str:
    """A filename for what came from where. `hf:a/b` becomes `a-b`."""
    text = source.removeprefix(HUB) if source.startswith(HUB) else Path(source).stem
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-")


def graph_for(source: str, out_dir: Path, trust: bool, opset: int) -> tuple[Path, str, bool]:
    """The ONNX file to adopt, fetching and exporting it if that is what the source needs."""
    if not source.startswith(HUB):
        path = Path(source)
        if is_onnx(path):
            return path, f"{path.name}, as it arrived", False
        raise SystemExit(f"{source} is not a .onnx. A checkpoint of this project's own "
                         f"architecture goes through `ganlive export-onnx`, which folds it "
                         f"and bakes its spectral norm first.")

    repo = source.removeprefix(HUB)
    inside = F.graph_in(repo)
    if inside is not None:
        from huggingface_hub import hf_hub_download

        print(f"{repo} carries {inside}; no export needed", flush=True)
        return Path(hf_hub_download(repo, inside)), f"{repo}:{inside}", False

    print(f"{repo} carries no ONNX; loading its own model code", flush=True)
    fetched = F.from_hub(repo, trust=trust)
    print(f"  {fetched.report()}", flush=True)
    raw = out_dir / f"{slug(source)}-raw.onnx"
    F.export(fetched, raw, opset=opset)
    return raw, f"{repo} via {fetched.which}", True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ganlive adopt", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a .onnx file, or hf:<owner>/<repo>")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the playable graph (default runs/onnx/<name>.onnx)")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/onnx"))
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="import and run the model code the repository ships. Required for "
                         "any repo that is not already ONNX, and not defaulted anywhere.")
    ap.add_argument("--seed", type=int, default=0,
                    help="the seed the graph's own random draws are frozen at")
    ap.add_argument("--target-levels", type=float, default=A.TARGET_LEVELS,
                    help="mean 8-bit levels a dial should buy at full travel, on every model")
    ap.add_argument("--device", default="cpu",
                    help="where to measure the dials: cpu (ONNX Runtime), or an OpenVINO "
                         "device such as GPU. A six-megapixel generator needs the card -- "
                         "adoption is about six hundred forward passes.")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the un-dialled export beside the playable one")
    args = ap.parse_args(argv)

    # Never two GPU jobs at once: adoption compiles the graph twice and renders hundreds of
    # frames.
    if args.device.lower() != "cpu" and not dev.refuse_if_gpu_busy("adoption"):
        return 1
    free = dev.host_ram_free_gb()
    if free == free and free < 3.0:
        print(f"only {free:.1f} GB of host RAM free; a large graph parses into several times "
              f"its size on disk", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw, came_from, made = graph_for(args.source, args.out_dir, args.trust_remote_code,
                                     args.opset)
    out = args.out or args.out_dir / f"{slug(args.source)}.onnx"

    print(f"adopting {came_from}", flush=True)
    found = A.adopt(raw, out, seed=args.seed, target=args.target_levels,
                    device=args.device)
    print(found.report(), flush=True)
    for dial in found.dials:
        print(f"  {dial.name:<12} rest {dial.rest:.1f}  "
              + " ".join(f"{v:.4g}" for v in dial.curve), flush=True)

    if made and raw != out and not args.keep_raw:
        raw.unlink(missing_ok=True)
        weights_file(raw).unlink(missing_ok=True)
    print(f"\n{out}  ({out.stat().st_size / 1e6:.0f} MB, "
          f"{time.perf_counter() - started:.0f}s)", flush=True)
    print(f"play it:  ganlive play --console "
          f"--checkpoint {out}", flush=True)
    return 0

