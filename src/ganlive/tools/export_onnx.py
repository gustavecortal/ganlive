"""Export a trained generator to ONNX, so a runtime other than PyTorch can play it.

The dials become graph inputs, so a runtime with no module tree can still steer the model.

Exported after the BatchNorm folding and noise freezing, because that is the net that
actually ships. fp32 on the way out: OpenVINO converts to fp16 itself at load.

The dynamo exporter, not the TorchScript one, which refuses any non-square model --
`adaptive_avg_pool2d` onto a size that is not a factor of the input.

Weights land in a sibling `.onnx.data` file; both files must travel together.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The dynamo exporter prints a check mark when it succeeds, and a Windows console is cp1252
# by default, so the export dies with a UnicodeEncodeError *after* doing all of the work.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from ganlive.dials import steer as K  # noqa: E402
from ganlive.models.fastgan import load, pin_noise  # noqa: E402
from ganlive.models.graph import prepare_for_inference  # noqa: E402
from ganlive.models.rewrite import (  # noqa: E402
    bank_the_knobs,
    equivalent,
    split_gated_convs,
)


def export(checkpoint: Path, out: Path, opset: int, noise_seed: int,
           split_glu: bool = True, steerable: bool = True) -> dict:
    import copy

    net, cfg = load(checkpoint, "cpu")
    pin_noise(net, noise_seed)
    report = prepare_for_inference(net, cfg.nz, "cpu", half=False, fold=True,
                                   compile_yuv=False, compile_net=False)
    net = report["net"].eval()

    # **The settings, as a second graph input.** `knobs.install` hands each steerable module a view into one
    # tensor, and a view traces as a *constant* -- which is why an exported model used to arrive with an
    # empty MODEL block and the strip drew six dials dark.
    names: list[str] = []
    if steerable:
        knobs = K.install(net, "cpu", torch.float32)
        net = bank_the_knobs(net, knobs).eval()
        names = list(knobs.names)
    k = torch.ones(len(names)) if steerable else None
    args = (torch.zeros(1, cfg.nz),) + ((k,) if steerable else ())

    split, drift = 0, 0.0
    if split_glu:
        reference = copy.deepcopy(net)
        split = split_gated_convs(net.net if steerable else net)
        net = net.eval()
        drift = equivalent(reference, net, cfg.nz, "cpu", probes=3, settings=k)
        if drift >= 0.5:
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
            input_names=["z"] + (["k"] if steerable else []),
            output_names=[f"image{i}" for i in range(len(outs))],
            opset_version=opset,
            dynamo=True,
        )
    seconds = time.perf_counter() - started
    if steerable:
        _settings_survived(out, names)
        _name_the_settings(out, names)

    data = out.with_suffix(out.suffix + ".data")
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
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    if len(graph.input) < 2:
        raise RuntimeError("the exported graph takes no settings input at all")
    fed = graph.input[1].name
    consumed = {name for node in graph.node for name in node.input}
    reached = {int(_only(node, graph)) for node in graph.node
               if node.op_type == "Slice" and fed in node.input
               and any(out in consumed for out in node.output)}
    missing = [n for i, n in enumerate(names) if i not in reached]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} did not survive the trace: {len(reached)} of "
            f"{len(names)} settings reach the graph. The rest are baked into it as "
            f"constants, and the model would load with dials that cannot move a pixel.")


def _only(node, graph) -> int:
    """Which slot a `Slice` of the settings vector takes. Its `starts` input, as a number."""
    import onnx

    initial = {i.name: i for i in graph.initializer}
    starts = node.input[1]
    if starts not in initial:
        raise RuntimeError(f"{node.name}: a settings slice with a computed start")
    return int(onnx.numpy_helper.to_array(initial[starts]).reshape(-1)[0])


def _name_the_settings(path: Path, names: list[str]) -> None:
    """Write the settings' names into the graph, so the file stands on its own."""
    import onnx

    from ganlive.dials.onnx_dials import name_settings

    model = onnx.load(str(path), load_external_data=False)
    name_settings(model, names)
    onnx.save(model, str(path), save_as_external_data=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Destination .onnx. Defaults to runs/onnx/<run>-<step>.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--frozen-settings", action="store_true",
                    help="export the dials as constants at their trained neutral. The graph "
                         "then takes a latent only, and the strip draws its MODEL block dark "
                         "-- which is what every export before this one did.")
    ap.add_argument("--no-split-glu", action="store_true",
                    help="Export the GLU as a tensor split, which is what the "
                         "trained module does and what OpenVINO spends 37%% of "
                         "its layer time copying.")
    args = ap.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"no such checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2
    out = args.out
    if out is None:
        run = args.checkpoint.parent.parent.name
        out = Path("runs/onnx") / f"{run}-{args.checkpoint.stem}.onnx"

    print(f"exporting {args.checkpoint} -> {out}", flush=True)
    report = export(args.checkpoint, out, args.opset, args.noise_seed,
                    split_glu=not args.no_split_glu,
                    steerable=not args.frozen_settings)
    print(json.dumps(report, indent=2))
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0

