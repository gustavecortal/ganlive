"""Export a trained FastGAN to ONNX, so a runtime other than PyTorch can play it.

The net is exported as it plays: BatchNorm folded, noise frozen, and its dials as a second
graph input `k`, named in the graph's metadata. fp32 on the way out; the runtime chooses half
precision itself. Weights land in a sibling `.onnx.data` file; both files travel together.

The dynamo exporter, because the TorchScript one refuses any non-square model.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

from ganlive.dials import steer as K
from ganlive.models.fastgan import freeze_noise, load
from ganlive.models.fold import prepare_for_inference
from ganlive.models.onnx_file import initializers, name_settings, structure, weights_file
from ganlive.models.onnx_rewrite import bank_the_knobs, equivalent, split_gated_convs
from ganlive.pixels import EXACT_LEVELS
from ganlive.timing import write_metrics


def export(checkpoint: Path, out: Path, opset: int, noise_seed: int,
           split_glu: bool = True) -> dict:
    """Write `checkpoint` as an ONNX graph at `out`. Returns a report of what was done."""
    net, cfg = load(checkpoint, "cpu")
    freeze_noise(net, seed=noise_seed)
    report = prepare_for_inference(net, cfg.nz, "cpu", half=False)
    # The dials become a second graph input: a view into a settings tensor traces as a constant.
    knobs = K.install(report["net"].eval(), "cpu", torch.float32)
    net = bank_the_knobs(report["net"], knobs).eval()
    names = list(knobs.names)
    args = (torch.zeros(1, cfg.nz), torch.ones(len(names)))

    split, drift = 0, 0.0
    if split_glu:
        original = copy.deepcopy(net)
        split = split_gated_convs(net.net)
        net = net.eval()
        drift = equivalent(original, net, cfg.nz, "cpu", probes=3, settings=args[1])
        if drift >= EXACT_LEVELS:
            raise RuntimeError(
                f"the GLU rewrite moved the picture by {drift:.3f} 8-bit levels; refusing to "
                f"export a graph that is not the model")

    with torch.no_grad():
        outs = net(*args)
    shapes = [tuple(o.shape) for o in outs]

    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            net, args, str(out),
            input_names=["z", "k"],
            output_names=[f"image{i}" for i in range(len(outs))],
            opset_version=opset,
            dynamo=True,
        )
    seconds = time.perf_counter() - started
    _settings_survived(out, names)
    _name_the_settings(out, names)

    data = weights_file(out)
    weights_mb = round(data.stat().st_size / 1e6, 1) if data.exists() else 0.0

    return {"checkpoint": str(checkpoint), "onnx": str(out),
            "nz": cfg.nz, "native": [cfg.im_size, cfg.im_width],
            "outputs": [list(s) for s in shapes],
            "folded": report["folded"],
            "split_glu": split, "split_glu_drift_levels": round(drift, 6),
            "settings": names,
            "export_s": round(seconds, 1),
            "graph_mb": round(out.stat().st_size / 1e6, 2),
            "weights_mb": weights_mb}


def _settings_survived(path: Path, names: list[str]) -> None:
    """Refuse a graph where a dial traced as a constant."""
    graph = structure(path).graph
    if len(graph.input) < 2:
        raise RuntimeError("the exported graph takes no settings input at all")
    fed = graph.input[1].name
    initial = initializers(graph)
    consumed = {name for node in graph.node for name in node.input}
    reached = {_slot(node, initial) for node in graph.node
               if node.op_type == "Slice" and fed in node.input
               and any(out in consumed for out in node.output)}
    missing = [n for i, n in enumerate(names) if i not in reached]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} did not survive the trace: {len(reached)} of "
            f"{len(names)} settings reach the graph. The rest are baked into it as "
            f"constants, and the model would load with dials that cannot move a pixel.")


def _slot(node, initial) -> int:
    """Which slot of the settings vector a `Slice` takes: its `starts` input, as a number."""
    import onnx

    starts = node.input[1]
    if starts not in initial:
        raise RuntimeError(f"{node.name}: a settings slice with a computed start")
    return int(onnx.numpy_helper.to_array(initial[starts]).reshape(-1)[0])


def _name_the_settings(path: Path, names: list[str]) -> None:
    """Write the settings' names into the graph, so the file stands on its own."""
    import onnx

    model = structure(path)
    name_settings(model, names)
    onnx.save(model, str(path), save_as_external_data=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ganlive export-onnx", description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Destination .onnx. Defaults to runs/onnx/<run>-<step>.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--no-split-glu", action="store_true",
                    help="Export the GLU as a tensor split, as the trained module computes it, "
                         "rather than as two half-width convolutions. Slower to run.")
    args = ap.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"no such checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2
    out = args.out
    if out is None:
        from ganlive.checkpoints import slug_for

        out = Path("runs/onnx") / f"{slug_for(args.checkpoint)}.onnx"

    print(f"exporting {args.checkpoint} -> {out}", flush=True)
    report = export(args.checkpoint, out, args.opset, args.noise_seed,
                    split_glu=not args.no_split_glu)
    print(json.dumps(report, indent=2))
    write_metrics(out.with_suffix(".json"), report)
    return 0
