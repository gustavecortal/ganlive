"""A converted StyleGAN2 for the engine: its ops with every gain folded in, the demodulation
energies, the trained noise, its probe, and its dials measured (w-space directions per style
range, and each truncation and noise dial's swept curve)."""

from __future__ import annotations

import numpy as np
import torch

from ganlive.device import detect_backend
from ganlive.dials import derive, steer
from ganlive.dials.gate import directions_for
from ganlive.engine.blob import Blob
from ganlive.engine.probe import box_means
from ganlive.levels import RANDOM_FLOOR
from ganlive.models import calibrate
from ganlive.models import stylegan2 as S2
from ganlive.models.common import host_latent


def stylegan2_net(checkpoint, device="cpu"):
    """A converted StyleGAN2 in full precision with its dials installed: `(net, settings)`."""
    net = S2.from_file(checkpoint, device, half_from=S2.SINGLE_EVERYWHERE)
    return net, steer.install_stylegan2(net, device)


def stylegan2_manifest(net, names: list[str]) -> tuple[dict, Blob]:
    """The ops of a StyleGAN2 in run order and the weights they read, every gain folded in."""
    cfg, blob = net.cfg, Blob()
    if tuple(cfg.taps) != S2.TAPS or cfg.img_channels != 3 or cfg.c_dim:
        raise ValueError(f"the engine plays unconditional RGB StyleGAN2s with the [1, 3, 3, 1] "
                         f"resample filter. This one has {cfg.img_channels} channel(s), filter "
                         f"{list(cfg.taps)} and {cfg.c_dim} class dimensions")

    def put(t) -> int:
        return blob.put(t.detach().float().cpu().numpy())

    mapping = [{"n_in": layer.in_features, "n_out": layer.out_features,
                "w": put(layer.weight * layer.weight_gain), "b": put(layer.bias * layer.bias_gain)}
               for layer in net.mapping.layers]
    bands = [name for name, _lo, _hi in S2.BANDS]
    ws_band = [next(i for i, (_n, lo, hi) in enumerate(S2.BANDS) if lo <= k < hi)
               for k in range(cfg.num_ws)]
    clamp = None if cfg.conv_clamp is None else float(cfg.conv_clamp)

    def affine(dense, gain=1.0) -> dict:
        return {"affine": put(dense.weight * (dense.weight_gain * gain)),
                "affine_b": put(dense.bias * (dense.bias_gain * gain))}

    ops = []
    for i, block in enumerate(net.blocks):
        base, res = max(2 * i - 1, 0), block.res
        ch = cfg.channels(res)
        if block.conv0 is None:
            const = block.const.detach().float()                   # [ch][4][4]
            # Packed as the features are: channel pairs per pixel.
            ops.append({"op": "const", "cout": ch, "h": res, "w": res,
                        "data": put(const.reshape(ch // 2, 2, res * res).transpose(1, 2))})
            layers = [(block.conv1, base, "conv1")]
        else:
            layers = [(block.conv0, base, "conv0"), (block.conv1, base + 1, "conv1")]
        for layer, ws, tag in layers:
            weight = layer.weight.detach().float()                 # [out][in][3][3]
            ops.append({
                "op": "conv", "name": f"b{res}.{tag}", "up": layer.up > 1,
                "cin": weight.shape[1], "cout": weight.shape[0], "h": res, "w": res, "ws": ws,
                **affine(layer.affine),
                # One layout for both kinds: a transposed convolution reads `weight[o][i]`
                # at tap (ky, kx) as the plain one does.
                "weight": put(weight.reshape(weight.shape[0], weight.shape[1], 9).permute(1, 2, 0)),
                "energy": put(weight.square().sum(dim=[2, 3])),
                "bias": put(layer.bias), "noise": put(layer.noise_const),
                "strength": float(layer.noise_strength), "slot": names.index(f"noise_{res}"),
                "clamp": clamp})
        rgb = block.torgb
        ops.append({"op": "torgb", "name": f"b{res}.torgb", "cin": ch, "cout": 3, "h": res,
                    "w": res, "ws": base + (1 if block.conv0 is None else 2),
                    **affine(rgb.affine, rgb.weight_gain),
                    "weight": put(rgb.weight.detach().float().reshape(3, ch).T),
                    "bias": put(rgb.bias), "clamp": clamp})
    manifest = {"family": "stylegan2", "nz": cfg.z_dim, "w_dim": cfg.w_dim,
                "num_ws": cfg.num_ws, "height": cfg.img_resolution, "width": cfg.img_resolution,
                "settings": names, "mapping": mapping, "w_avg": put(net.mapping.w_avg),
                "ws_band": ws_band, "band_slots": [names.index(b) for b in bands],
                "ops": ops, "bytes": len(blob.data)}
    return manifest, blob


def stylegan2_probe(net, settings) -> dict:
    """What PyTorch draws for a fixed latent at neutral settings, as block means."""
    z = host_latent(net.cfg.z_dim)
    settings.reset()
    with torch.no_grad():
        image = net(torch.from_numpy(z).to(net.mapping.w_avg.device))[0].float().cpu().numpy()
    # Clamped as the engine's picture is: a StyleGAN2 has no tanh, and its highlights overshoot.
    image = np.clip(image, -1.0, 1.0)
    return {"z": z[0].tolist(), "means": np.round(box_means(image), 5).ravel().tolist()}


def stylegan2_dials(net, settings, checkpoint, device, *, floor: float = RANDOM_FLOOR) -> dict:
    """A StyleGAN2's dials: w-space directions per style range through the push buffer, ranked
    against random ones, and each MODEL dial's swept curve."""
    push = net.mapping.push
    bands = S2.style_bands(net)
    with torch.no_grad():
        found = directions_for(
            net, net.cfg.z_dim, device, torch.float32, into=push, floor=floor, path=checkpoint,
            read=lambda _net, nz: derive.sefa_banded(bands, (derive.CANDIDATES,) * len(bands), nz))
        swept = calibrate.measured(calibrate.TorchProbe(net, settings, net.cfg.z_dim, device,
                                                        torch.float32),
                                   settings.names, size=(net.cfg.img_resolution,) * 2)
    ranges = [] if found is None or not found.ranges else list(found.ranges)
    directions = None if found is None else {
        "basis": np.round(found.basis.float().cpu().numpy(), 6).tolist(),
        "levels": [round(v, 2) for v in found.levels or ()], "report": found.report(),
        "push_shape": list(found.push_shape) if found.push_shape else None, "ranges": ranges}
    return {"directions": directions,
            "swept": {"names": list(swept.names), "rests": [d.rest for d in swept.dials],
                      "curves": [list(d.curve) for d in swept.dials],
                      "levels": [d.moved for d in swept.dials],
                      "ranges": [[n, S2.BAND_PIXELS[n]] for n in ranges]}}


def measure(checkpoint, device=None, *, floor: float = RANDOM_FLOOR,
            grain: bool = True) -> tuple[dict, Blob]:
    """The manifest of `checkpoint`, probe and dials included, and its weights, from one net
    built on `device`. `grain` is a FastGAN question and is ignored."""
    device = device or detect_backend()
    net, settings = stylegan2_net(checkpoint, device)
    manifest, blob = stylegan2_manifest(net, list(settings.names))
    manifest["probe"] = stylegan2_probe(net, settings)
    manifest["dials"] = stylegan2_dials(net, settings, checkpoint, device, floor=floor)
    return manifest, blob
