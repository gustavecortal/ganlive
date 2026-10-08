"""Convert a FastGAN checkpoint into an engine model: a folder with `weights.bin` (fp16) and
`program.json` (shaders and dispatches, see `program.py`). Needs PyTorch, on CPU; playing the
result does not.

The net is prepared as `ganlive export-onnx` prepares it: spectral norm baked, BatchNorm folded,
dials as a settings vector, each GLU split into value and gate convolutions. Its noise is the
engine's own (`program.seeded_noise`), so that PyTorch and the engine draw the same picture.

Weight layouts (fp16, little-endian; every array starts on a 4-byte boundary):
  conv weights  [cin][3*3][cout]   value and gate halves separately
  init weight   [2*ch*cells][nz]   row = channel * cells + cell, cells = 4*6
  sle fc1       [ch_low*16][ch]    fc2 [ch][ch]
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import torch

from ganlive.checkpoints import PROGRAM, WEIGHTS
from ganlive.device import detect_backend
from ganlive.dials import fastgan_dials, steer
from ganlive.dials.gate import directions_for, measure_dials
from ganlive.engine import stamp as stamps
from ganlive.engine.program import box_means, compile_program, noise_seed, seeded_noise
from ganlive.files import write_json
from ganlive.levels import EXACT_LEVELS, RANDOM_FLOOR
from ganlive.models.common import host_latent
from ganlive.models.fastgan import freeze_noise, load
from ganlive.models.fold import prepare_for_inference
from ganlive.models.onnx_rewrite import (
    GatedPair,
    equivalent,
    settings_as_input,
    split_gated_convs,
)
from ganlive.models.steerable import SteerableSLE
from ganlive.settings import Settings


class Blob:
    """The weight file being written: fp16 arrays, each starting and ending on a 4-byte word."""

    def __init__(self) -> None:
        self.data = bytearray()

    def put(self, a) -> int:
        """Append an array; return its offset in 16-bit elements."""
        at = len(self.data) // 2
        self.data += np.asarray(a, dtype="<f2").tobytes()
        self.data += bytes(-len(self.data) % 4)
        return at

    def conv(self, conv) -> int:
        w = conv.weight.detach().float().numpy()            # [cout][cin][3][3]
        return self.put(w.reshape(w.shape[0], w.shape[1], 9).transpose(1, 2, 0))


def noisy_pairs(net):
    """`(layer, GatedPair)` for every convolution with noise, named as the program names it:
    the block, then the convolution's place in it."""
    for block, seq in net.named_children():
        if isinstance(seq, torch.nn.Sequential):
            pairs = [m for m in seq if isinstance(m, GatedPair)]
            yield from ((f"{block}.{i}", pair) for i, pair in enumerate(pairs) if pair.noise)


def prepared(source):
    """The FastGAN as the engine runs it, its config, and its settings' names. `source` is a
    checkpoint path, or a `(Generator, Config)` already loaded."""
    net, cfg = load(source, "cpu") if isinstance(source, (str, Path)) else source
    # What the engine has ops for: a sub-pixel block's conv is not a GatedPair (PixelShuffle
    # sits between it and its GLU), a mapping network has no op, and init reads z as vec4s.
    if cfg.pixelshuffle_from or cfg.mapping_depth or cfg.nz % 4:
        raise ValueError(
            f"the engine plays nearest-upsample FastGANs without a mapping network and with "
            f"nz a multiple of 4; this one has pixelshuffle_from={cfg.pixelshuffle_from}, "
            f"mapping_depth={cfg.mapping_depth}, nz={cfg.nz}")
    freeze_noise(net, seed=0)               # the fold needs a pattern; the engine's replaces it
    report = prepare_for_inference(net, cfg.nz, "cpu", half=False)
    settings = steer.install(report["net"].eval(), "cpu", torch.float32)
    steerable = settings_as_input(report["net"], settings).eval()
    original = copy.deepcopy(steerable)
    split_gated_convs(steerable.net)
    # The split rewrites weights, not arithmetic, so one latent proves it.
    neutral = torch.ones(len(settings.names))
    drift = equivalent(original, steerable.eval(), cfg.nz, "cpu", probes=1, settings=neutral)
    if drift >= EXACT_LEVELS:
        raise RuntimeError(f"the GLU split moved the picture by {drift:.3f} 8-bit levels")
    for layer, pair in noisy_pairs(steerable.net):
        noise = seeded_noise(pair.pattern.numel(), noise_seed(layer))
        pair.pattern.copy_(torch.from_numpy(noise).reshape(pair.pattern.shape))
    return steerable, cfg, list(settings.names)


def manifest_of(steerable, cfg, names: list[str]) -> tuple[dict, Blob]:
    """The ops in run order and the weights they read."""
    g, blob = steerable.net, Blob()
    ops, tensors = [], {}

    def tensor(name, c, h, w):
        tensors[name] = [c, h, w]
        return name

    # init: a ConvTranspose(nz -> 2ch, kernel 4x6, BatchNorm folded in) on a 1x1 input is a
    # dense layer with a bias, then GLU.
    init = g.init.main[0]
    w = init.weight.detach().float().numpy()             # [nz][2ch][4][6]
    h0, w0 = cfg.ladder.at(4)
    ops.append({"op": "init", "out": tensor("f4", w.shape[1] // 2, h0, w0),
                "w": blob.put(w.reshape(cfg.nz, -1).T), "bias": blob.put(init.bias.detach())})

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
                op.update(noise=f"{name}.{i}", slot=pair.slot,
                          cv=blob.put(pair.coeff_value.reshape(-1)),
                          cg=blob.put(pair.coeff_gate.reshape(-1)))
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
    _, h, wd = tensors[feat]
    manifest = {"nz": cfg.nz, "height": h, "width": wd, "settings": names,
                "bytes": len(blob.data), "tensors": tensors, "ops": ops}
    return manifest, blob


def probe(steerable, cfg, names: list[str]) -> dict:
    """What PyTorch draws for a fixed latent at neutral settings, as block means: a host checks
    a backend against it before trusting the backend's timings."""
    z = host_latent(cfg.nz)
    with torch.no_grad():
        image = steerable(torch.from_numpy(z), torch.ones(len(names)))[0][0].numpy()
    return {"z": z[0].tolist(), "means": np.round(box_means(image), 5).ravel().tolist()}


class Driven(torch.nn.Module):
    """The steerable net as ganlive's dial measurements drive a played FastGAN: `net(z)`, its
    settings read from the `Settings` they write."""

    def __init__(self, steerable, settings: Settings) -> None:
        super().__init__()
        self.steerable, self.settings = steerable, settings

    def forward(self, z):
        return self.steerable(z, self.settings.vec)


def measured_dials(steerable, cfg, names: list[str], device=None, checkpoint=None, *,
                   floor: float = RANDOM_FLOOR, grain: bool = True) -> dict:
    """The dials of this net, measured once here so that playing measures nothing: the noise
    gains each band needs (the stock ones without `grain`), the latent directions that beat a
    random one `floor` times over (ranked), and how far each MODEL dial moves the picture,
    which decides which dials are live. On `device`, the fastest PyTorch has here unless
    given, with the engine's own noise."""
    device = device or detect_backend()
    net = Driven(copy.deepcopy(steerable).to(device),
                 Settings(names, device, torch.float32)).eval()
    with torch.no_grad():
        gains = (steer.calibrate_noise(net, net.settings, cfg.nz, device, torch.float32)
                 if grain else None)
        layout = fastgan_dials.fastgan(noise_gains=gains)
        found = directions_for(net, cfg.nz, device, torch.float32, path=checkpoint, floor=floor)
        layout = measure_dials(net, net.settings, layout, cfg.nz, device, torch.float32)
    directions = None if found is None else {
        "basis": np.round(found.basis.float().cpu().numpy(), 6).tolist(),
        "levels": [round(v, 2) for v in found.levels or ()], "report": found.report()}
    return {"noise_gains": gains, "directions": directions,
            "measured": {k.name: k.measured for k in layout.knobs if k.measured is not None}}


def build(steerable, cfg, names: list[str], output: str = "rgba8", dials=None) -> tuple[dict, Blob]:
    """The program, probe and `dials` (from `measured_dials`) included, and the weights."""
    manifest, blob = manifest_of(steerable, cfg, names)
    program = compile_program(manifest, output=output)
    program["probe"] = probe(steerable, cfg, names)
    if dials is not None:
        program["dials"] = dials
    return program, blob


def convert(checkpoint, out: Path, device=None, *, floor: float = RANDOM_FLOOR,
            grain: bool = True, stamp: dict | None = None) -> dict:
    """Write the engine model for `checkpoint`, its dials measured on `device` (see
    `measured_dials` for `floor` and `grain`), into the folder `out`. Returns the program,
    stamped with what it was made from (`stamp.stamp_for`) so a load can tell it is current."""
    steerable, cfg, names = prepared(checkpoint)
    dials = measured_dials(steerable, cfg, names, device, checkpoint=checkpoint,
                           floor=floor, grain=grain)
    program, blob = build(steerable, cfg, names, dials=dials)
    program["stamp"] = stamp or stamps.stamp_for(checkpoint, floor, grain)
    # The weights first and the program last, each renamed into place whole: a conversion cut
    # short leaves no program that a load would take for a finished one.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{WEIGHTS}.part").write_bytes(blob.data)
    os.replace(out / f"{WEIGHTS}.part", out / WEIGHTS)
    write_json(out / f"{PROGRAM}.part", program, indent=None)
    os.replace(out / f"{PROGRAM}.part", out / PROGRAM)
    return program
