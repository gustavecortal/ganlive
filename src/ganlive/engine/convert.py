"""Convert a FastGAN checkpoint into an engine model: a folder with `weights.bin` (fp16) and
`program.json` (shaders and dispatches, see `program.py`). Needs PyTorch, on CPU; playing the
result does not.

The net is prepared as `ganlive export-onnx` prepares it: spectral norm baked, BatchNorm folded,
noise frozen, dials as a settings vector, each GLU split into value and gate convolutions.

Weight layouts (fp16, little-endian; every array starts on a 4-byte boundary):
  conv weights  [cin][3*3][cout]   value and gate halves separately
  init weight   [2*ch*cells][nz]   row = channel * cells + cell, cells = 4*6
  sle fc1       [ch_low*16][ch]    fc2 [ch][ch]
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from ganlive.dials import steer
from ganlive.engine.program import box_means, compile_program, noise_seed, seeded_noise
from ganlive.models.fastgan import freeze_noise, load
from ganlive.models.fold import _bn_affine, prepare_for_inference
from ganlive.models.onnx_rewrite import (
    GatedPair,
    equivalent,
    settings_as_input,
    split_gated_convs,
)
from ganlive.models.steerable import SteerableSLE
from ganlive.pixels import EXACT_LEVELS


class Blob:
    """The weight file being written: fp16 arrays, each at a 4-byte boundary."""

    def __init__(self) -> None:
        self.data = bytearray()

    def put(self, a) -> int:
        """Append an array; return its offset in 16-bit elements."""
        while len(self.data) % 4:
            self.data.append(0)
        at = len(self.data) // 2
        self.data.extend(np.asarray(a, dtype=np.float32).astype("<f2").tobytes())
        return at

    def conv(self, conv) -> int:
        w = conv.weight.detach().float().numpy()            # [cout][cin][3][3]
        return self.put(w.reshape(w.shape[0], w.shape[1], 9).transpose(1, 2, 0))


def prepared(source, noise_seed: int = 0):
    """The FastGAN as the engine runs it, its config, and its settings' names. `source` is a
    checkpoint path, or a `(Generator, Config)` already loaded."""
    net, cfg = load(source, "cpu") if isinstance(source, (str, Path)) else source
    # What the engine has ops for: a sub-pixel block's conv is not a GatedPair (PixelShuffle
    # sits between it and its GLU), a mapping network has no op, and init reads z as vec4s.
    if cfg.pixelshuffle_from or cfg.mapping_depth or cfg.nz % 4:
        raise ValueError(
            f"the engine plays nearest-upsample FastGANs without a "
            f"mapping network and with nz a multiple of 4; this one has pixelshuffle_from="
            f"{cfg.pixelshuffle_from}, mapping_depth={cfg.mapping_depth}, nz={cfg.nz}")
    freeze_noise(net, seed=noise_seed)
    report = prepare_for_inference(net, cfg.nz, "cpu", half=False)
    settings = steer.install(report["net"].eval(), "cpu", torch.float32)
    steerable = settings_as_input(report["net"], settings).eval()
    original = copy.deepcopy(steerable)
    split_gated_convs(steerable.net)
    neutral = torch.ones(len(settings.names))
    drift = equivalent(original, steerable.eval(), cfg.nz, "cpu", probes=3, settings=neutral)
    if drift >= EXACT_LEVELS:
        raise RuntimeError(f"the GLU split moved the picture by {drift:.3f} 8-bit levels")
    return steerable, cfg, list(settings.names)


def manifest_of(steerable, cfg, names: list[str]) -> tuple[dict, Blob, dict]:
    """The ops in run order, the weights they read, and the noise by layer.

    The noise becomes the engine's own, seeded per layer and made on the GPU at load (see
    `program.NOISE`), and is written into `steerable` too, so that it draws what the engine
    draws: no host ships the 25 MB patterns of a 3072x2048 layer."""
    g, blob = steerable.net, Blob()
    ops, tensors, patterns = [], {}, {}

    def tensor(name, c, h, w):
        tensors[name] = [c, h, w]
        return name

    # init: ConvTranspose(nz -> 2ch, kernel 4x6) on a 1x1 input is a dense layer, BN, GLU.
    init = g.init.main
    w = init[0].weight.detach().float().numpy()          # [nz][2ch][4][6]
    scale, shift = _bn_affine(init[1])
    h0, w0 = cfg.ladder.at(4)
    ops.append({"op": "init", "out": tensor("f4", w.shape[1] // 2, h0, w0),
                "w": blob.put(w.reshape(cfg.nz, -1).T), "scale": blob.put(scale),
                "shift": blob.put(shift)})

    def sle(name, low):
        m = getattr(g, name)
        assert isinstance(m, SteerableSLE)
        pool, fc1, _, fc2, _ = m.gate
        ch = fc1.weight.shape[0]
        w1 = fc1.weight.detach().float().numpy()         # [ch][c_low][4][4]
        ops.append({"op": "sle", "name": name, "low": low, "slot": m.index, "ch": ch,
                    "rows": blob.put(pool.rows.numpy()), "cols": blob.put(pool.cols.numpy()),
                    "fc1": blob.put(w1.reshape(ch, -1).T),
                    "fc2": blob.put(fc2.weight.detach().float().numpy().reshape(ch, ch).T)})
        return name

    def block(name, src, scale_by=None):
        _, h, wd = tensors[src]
        convs = [m for m in getattr(g, name) if isinstance(m, GatedPair)]
        for i, pair in enumerate(convs):
            up = i == 0                                  # a block's first conv upsamples
            out = name if i == len(convs) - 1 else f"{name}.{i}"
            if up:
                h, wd = h * 2, wd * 2
            op = {"op": "conv", "in": src, "out": tensor(out, pair.value.out_channels, h, wd),
                  "up": up, "wv": blob.conv(pair.value), "wg": blob.conv(pair.gate),
                  "bv": blob.put(pair.value.bias.detach()), "bg": blob.put(pair.gate.bias.detach())}
            if pair.noise:
                key = f"{name}.{i}"
                noise = seeded_noise(h * wd, noise_seed(key))
                pair.pattern.copy_(torch.from_numpy(noise).reshape(pair.pattern.shape))
                patterns[key] = noise.reshape(h, wd)
                # `gain` is the bound `value` method of the slot the dial reads.
                op.update(noise=key, cv=blob.put(pair.coeff_value.reshape(-1)),
                          cg=blob.put(pair.coeff_gate.reshape(-1)),
                          slot=None if pair.gain is None else pair.gain.__self__.index)
            ops.append(op)
            src = out
        if scale_by:
            ops[-1]["scale"] = scale_by
        return src

    # The order of Generator.forward; each SLE gate is computed once its low input exists,
    # and applied as a channel scale by the last convolution of the block it gates.
    f8 = block("feat_8", "f4")
    f16 = block("feat_16", f8)
    f32 = block("feat_32", f16)
    f64 = block("feat_64", f32, sle("se_64", "f4"))
    f128 = block("feat_128", f64, sle("se_128", f8))
    feat = block("feat_256", f128, sle("se_256", f16))
    if cfg.ladder.height > 256:
        feat = block("feat_512", feat, sle("se_512", f32))
    if cfg.ladder.height > 512:
        feat = block("feat_1024", feat)
    if cfg.ladder.height > 1024:
        feat = block("feat_2048", feat)
    ops.append({"op": "rgb", "in": feat, "w": blob.conv(g.to_big)})
    while len(blob.data) % 4:                          # GPU uploads come in whole words
        blob.data.append(0)
    _, h, wd = tensors[feat]
    manifest = {"nz": cfg.nz, "height": h, "width": wd, "settings": names,
                "bytes": len(blob.data), "tensors": tensors, "ops": ops,
                "patterns": {k: list(v.shape) for k, v in patterns.items()}}
    return manifest, blob, patterns


def convert(checkpoint, out: Path, plans: dict | None = None) -> dict:
    """Write the engine model for `checkpoint` into the folder `out`. Returns the program."""
    steerable, cfg, names = prepared(checkpoint)
    manifest, blob, _ = manifest_of(steerable, cfg, names)
    program = compile_program(manifest, plans)
    program["probe"] = probe(steerable, cfg, names)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(blob.data)
    (out / "program.json").write_text(json.dumps(program), encoding="utf-8")
    return program


def probe(steerable, cfg, names: list[str]) -> dict:
    """What PyTorch draws for a fixed latent at neutral settings, as block means: a host checks
    a backend against it before trusting the backend's timings."""
    z = np.random.default_rng(0).standard_normal((1, cfg.nz)).astype(np.float32)
    with torch.no_grad():
        image = steerable(torch.from_numpy(z), torch.ones(len(names)))[0][0].numpy()
    return {"z": z[0].tolist(), "means": np.round(box_means(image), 5).ravel().tolist()}
