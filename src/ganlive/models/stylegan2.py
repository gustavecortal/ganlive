"""StyleGAN2, written so that `torch.compile` captures it in one graph.

Loads NVIDIA's StyleGAN2-ADA weights under their own names, once `ganlive import-stylegan2`
has converted the pickle, and matches their generator to float rounding.
"""
from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LRELU_SLOPE = 0.2
LRELU_GAIN = math.sqrt(2.0)
#: Every StyleGAN2-ADA checkpoint published uses this resample filter.
TAPS = (1.0, 3.0, 3.0, 1.0)
#: The three style ranges StyleGAN is described in, as `(name, first w, end w)`. The bounds
#: are absolute rather than proportional because they are about resolution.
BANDS = (("w_coarse", 0, 4), ("w_mid", 4, 8), ("w_fine", 8, 1 << 30))

#: Which pixel stages each range drives, as a phrase for a dial's description.
BAND_PIXELS = {"w_coarse": "the 4, 8 and 16 pixel stages",
               "w_mid": "the 16, 32 and 64 pixel stages",
               "w_fine": "everything from 64 pixels up to native"}
#: Written into every converted file, so ganlive can tell it from a FastGAN `.pt`.
FORMAT = "ganlive-stylegan2/1"

#: Config fields that describe how to play a checkpoint rather than what is in it. `save`
#: drops them.
NOT_SAVED = ("half_from",)


def fir(taps=TAPS) -> torch.Tensor:
    """`upfirdn2d.setup_filter`: the separable taps as a normalised 2-D kernel."""
    f = torch.tensor(taps, dtype=torch.float32)
    f = f.ger(f)
    return f / f.sum()


@dataclass(frozen=True)
class Config:
    """What a checkpoint says about itself. Every field but the last is in NVIDIA's
    `init_kwargs`; `half_from` is a decision about how to *play* it, and is never written."""

    z_dim: int = 512
    c_dim: int = 0
    w_dim: int = 512
    img_resolution: int = 1024
    img_channels: int = 3
    channel_base: int = 32768
    channel_max: int = 512
    num_fp16_res: int = 4
    conv_clamp: float | None = 256.0
    num_layers: int = 8
    lr_multiplier: float = 0.01
    taps: tuple[float, ...] = TAPS
    #: The lowest resolution to run in half precision, overriding NVIDIA's rule below. `None`
    #: is the checkpoint as they shipped it.
    half_from: int | None = None

    @property
    def resolutions(self) -> tuple[int, ...]:
        top = int(math.log2(self.img_resolution))
        return tuple(2 ** i for i in range(2, top + 1))

    def channels(self, res: int) -> int:
        """NVIDIA's channel rule."""
        return min(self.channel_base // res, self.channel_max)

    @property
    def fp16_from(self) -> int:
        """The lowest resolution to run in half precision -- NVIDIA's rule, or `half_from`."""
        if self.half_from is not None:
            return self.half_from
        return max(2 ** (int(math.log2(self.img_resolution)) + 1 - self.num_fp16_res), 8)

    @property
    def num_ws(self) -> int:
        return 2 * len(self.resolutions)

    # The two names the bank asks every family's config for.
    @property
    def nz(self) -> int:
        return self.z_dim

    @property
    def ladder(self):
        from ganlive.models.common import Ladder

        return Ladder(height=self.img_resolution, width=self.img_resolution)


def _open(path) -> dict:
    """A converted checkpoint, checked, with its weights left on disk until touched."""
    blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if blob.get("format") != FORMAT:
        raise RuntimeError(f"{path} is not a {FORMAT} checkpoint. "
                           f"`ganlive import-stylegan2` writes them.")
    return blob


def _rebuild(raw: dict, half_from: int | None = None) -> Config:
    """A stored config back into its dataclass."""
    return Config(**{**raw, "taps": tuple(raw["taps"]), "half_from": half_from})


def config_of(path) -> Config:
    """What a converted checkpoint says about itself, without loading its weights."""
    return _rebuild(_open(path)["config"])


def config_from(G) -> Config:
    """Read a loaded NVIDIA `Generator`'s own `init_kwargs`. Needs their repo importable."""
    k, m, s = dict(G.init_kwargs), dict(G.mapping.init_kwargs), dict(G.synthesis.init_kwargs)
    return Config(
        z_dim=k["z_dim"], c_dim=k["c_dim"], w_dim=k["w_dim"],
        img_resolution=k["img_resolution"], img_channels=k["img_channels"],
        channel_base=s.get("channel_base", 32768), channel_max=s.get("channel_max", 512),
        num_fp16_res=s.get("num_fp16_res", 0), conv_clamp=s.get("conv_clamp"),
        num_layers=m.get("num_layers", 8), lr_multiplier=m.get("lr_multiplier", 0.01),
        taps=tuple(float(t) for t in s.get("resample_filter", TAPS)))


class Dense(nn.Linear):
    """NVIDIA's `FullyConnectedLayer`. The gains stay outside the stored weight, so their
    weights load as they are."""

    def __init__(self, n_in: int, n_out: int, activation: str = "linear",
                 lr_multiplier: float = 1.0, bias_init: float = 0.0) -> None:
        super().__init__(n_in, n_out, bias=True)
        with torch.no_grad():
            self.weight.copy_(torch.randn(n_out, n_in) / lr_multiplier)
            self.bias.fill_(float(bias_init))
        self.weight_gain = float(lr_multiplier / math.sqrt(n_in))
        self.bias_gain = float(lr_multiplier)
        self.lrelu = activation == "lrelu"
        #: `weight * weight_gain` and `bias * bias_gain`, once `freeze` has run. `None`
        #: scales them every forward, for a net built without `load`.
        self.register_buffer("scaled", None, persistent=False)
        self.register_buffer("scaled_bias", None, persistent=False)

    @torch.no_grad()
    def freeze(self) -> None:
        """Scale the weights once, rather than in every forward of the captured graph."""
        self.scaled, self.scaled_bias = self._scaled()

    def _scaled(self) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.bias if self.bias_gain == 1.0 else self.bias * self.bias_gain
        return self.weight * self.weight_gain, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = (self.scaled, self.scaled_bias) if self.scaled is not None else self._scaled()
        if not self.lrelu:
            return torch.addmm(b.unsqueeze(0), x, w.t())
        return F.leaky_relu(x.matmul(w.t()) + b, LRELU_SLOPE) * LRELU_GAIN


def _pads(kernel: int, up: int, taps: int, padding: int) -> tuple[int, ...]:
    """`conv2d_resample`'s padding arithmetic, done once at build time instead of per frame."""
    p = [padding] * 4
    p[0] += (taps + up - 1) // 2
    p[1] += (taps - up) // 2
    p[2] += (taps + up - 1) // 2
    p[3] += (taps - up) // 2
    p[0] -= kernel - 1
    p[1] -= kernel - up
    p[2] -= kernel - 1
    p[3] -= kernel - up
    tx = max(min(-p[0], -p[1]), 0)
    ty = max(min(-p[2], -p[3]), 0)
    return ty, tx, p[0] + tx, p[1] + tx, p[2] + ty, p[3] + ty


def _fir(x: torch.Tensor, f: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """`upfirdn2d` with the zero-insertion already done and the filter already prepared."""
    x0, x1, y0, y1 = pad
    if x0 == x1 >= 0 and y0 == y1 >= 0:
        # The convolution's own zero padding, rather than `F.pad` writing a padded copy of the
        # feature map first. Every StyleGAN2 upsample is this case; see `_pads`.
        return F.conv2d(x, f, padding=(y0, x0), groups=x.shape[1])
    x = F.pad(x, [max(x0, 0), max(x1, 0), max(y0, 0), max(y1, 0)])
    if min(x0, x1, y0, y1) < 0:
        x = x[:, :, max(-y0, 0):x.shape[2] - max(-y1, 0),
              max(-x0, 0):x.shape[3] - max(-x1, 0)]
    return F.conv2d(x, f, groups=x.shape[1])


class SynthesisLayer(nn.Module):
    """One modulated convolution, its noise, its bias and its activation."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int, resolution: int, up: int = 1,
                 kernel: int = 3, conv_clamp: float | None = None, half: bool = False,
                 taps: tuple[float, ...] = TAPS) -> None:
        super().__init__()
        self.affine = Dense(w_dim, in_ch, bias_init=1.0)
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.register_buffer("noise_const", torch.randn(resolution, resolution))
        self.noise_strength = nn.Parameter(torch.zeros([]))
        #: A live gain on this layer's noise, or `None` for the network as trained.
        self.noise_gain = None

        self.up, self.half = up, half
        # NVIDIA renormalises the weight and the styles before a half-precision modulation so
        # neither overflows. Demodulation divides any such scaling back out, so it changes no
        # result.
        self.prenorm = float(1.0 / math.sqrt(in_ch * kernel * kernel)) if half else 0.0
        #: The weight as every frame convolves with it, and its squared sum per `(out, in)`
        #: pair, once `freeze` has run. `None` derives both every forward, for a net built
        #: without `load`.
        self.register_buffer("played", None, persistent=False)
        self.register_buffer("energy", None, persistent=False)
        self.pad = kernel // 2
        self.clamp = None if conv_clamp is None else float(conv_clamp)
        if up > 1:
            ty, tx, *rest = _pads(kernel, up, len(taps), self.pad)
            self.tpad, self.fpad = (ty, tx), tuple(rest)
            f = (fir(taps) * float(up * up)).flip([0, 1]).to(self.dtype)
            self.register_buffer("upfir", f[None, None].repeat(out_ch, 1, 1, 1),
                                 persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        """What this layer computes in."""
        return torch.float16 if self.half else torch.float32

    @torch.no_grad()
    def freeze(self) -> None:
        """Derive the played weight once, rather than in every forward of the captured graph."""
        self.played, self.energy = self._weights()

    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The weight the convolution reads, in the precision it runs in, and `energy`, the
        squared weight summed over the kernel: what demodulation needs of it, `(out, in)`. A
        half-precision weight is renormalised first; see `prenorm`."""
        weight = self.weight
        if self.half:
            weight = weight * (self.prenorm / weight.norm(float("inf"), dim=[1, 2, 3],
                                                          keepdim=True))
        energy = weight.square().sum(dim=[2, 3])
        if self.up > 1:
            weight = weight.transpose(0, 1)             # `conv_transpose2d` wants `(in, out)`
        return weight.to(self.dtype).contiguous(), energy

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        styles = self.affine(w)
        if self.half:
            styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)
        weight, energy = ((self.played, self.energy) if self.played is not None
                          else self._weights())

        # The styles scale the activations, not the weight: NVIDIA's `fused_modconv=False`,
        # the same arithmetic in another order. The input channels are scaled before a plain
        # convolution and the output channels demodulated after it, so the weight stays a
        # constant and is never copied per frame.
        # Both scalings stay in float32 and round once, into the dtype the convolution reads;
        # rounding each to half first moves the frame measurably further from full precision.
        dcoefs = (styles.square() @ energy.t() + 1e-8).rsqrt()
        x = (x * styles[:, :, None, None]).to(self.dtype)
        if self.up > 1:
            x = F.conv_transpose2d(x, weight, stride=self.up, padding=self.tpad)
            x = _fir(x, self.upfir, self.fpad)
        else:
            x = F.conv2d(x, weight, padding=self.pad)
        x = x * dcoefs[:, :, None, None]

        # The two scalars multiply each other, not the pattern, so a live gain costs one
        # broadcast rather than two.
        strength = (self.noise_strength if self.noise_gain is None
                    else self.noise_strength * self.noise_gain)
        x = x.add_(self.noise_const * strength)
        x = F.leaky_relu(x + self.bias.to(x.dtype).reshape(1, -1, 1, 1), LRELU_SLOPE)
        x = x * LRELU_GAIN
        x = x if self.clamp is None else x.clamp(-self.clamp, self.clamp)
        return x.to(self.dtype)


class ToRGB(nn.Module):
    """The 1x1 modulated convolution that turns a block's features into colour."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int,
                 conv_clamp: float | None = None) -> None:
        super().__init__()
        self.affine = Dense(w_dim, in_ch, bias_init=1.0)
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.weight_gain = float(1.0 / math.sqrt(in_ch))
        self.clamp = conv_clamp

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        styles = self.affine(w) * self.weight_gain
        weight = self.weight
        n, out_ch, in_ch = x.shape[0], *weight.shape[:2]
        m = (weight.unsqueeze(0) * styles.reshape(n, 1, -1, 1, 1)).to(x.dtype)
        x = F.conv2d(x.reshape(1, -1, *x.shape[2:]),
                     m.reshape(n * out_ch, in_ch, 1, 1), groups=n).reshape(n, out_ch,
                                                                          *x.shape[2:])
        x = x + self.bias.to(x.dtype).reshape(1, -1, 1, 1)
        return x if self.clamp is None else x.clamp(-self.clamp, self.clamp)


class Block(nn.Module):
    """One resolution: up to two modulated convolutions and the colour it contributes."""

    def __init__(self, in_ch: int, out_ch: int, res: int, cfg: Config, half: bool) -> None:
        super().__init__()
        self.res, self.half = res, half
        common = dict(w_dim=cfg.w_dim, resolution=res, conv_clamp=cfg.conv_clamp,
                      half=half, taps=cfg.taps)
        if in_ch == 0:
            self.const = nn.Parameter(torch.randn(out_ch, res, res))
            self.conv0 = None
        else:
            self.conv0 = SynthesisLayer(in_ch, out_ch, up=2, **common)
        self.conv1 = SynthesisLayer(out_ch, out_ch, **common)
        self.torgb = ToRGB(out_ch, cfg.img_channels, cfg.w_dim, cfg.conv_clamp)
        if in_ch != 0:
            f = (fir(cfg.taps) * 4.0).flip([0, 1])
            self.register_buffer("skipfir", f[None, None].repeat(cfg.img_channels, 1, 1, 1),
                                 persistent=False)
            taps = len(cfg.taps)
            self.skippad = ((taps + 1) // 2, (taps - 2) // 2) * 2

    @property
    def dtype(self) -> torch.dtype:
        """What this block's features are computed in."""
        return torch.float16 if self.half else torch.float32

    def forward(self, x, img, ws) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conv0 is None:
            x = self.const.to(self.dtype).unsqueeze(0).repeat(ws.shape[0], 1, 1, 1)
            x = self.conv1(x, ws[:, 0])
        else:
            x = self.conv1(self.conv0(x.to(self.dtype), ws[:, 0]), ws[:, 1])
            # The colour so far is carried up alongside the features, in full precision.
            img = _fir(_stuff(img, 2), self.skipfir, self.skippad)
        y = self.torgb(x, ws[:, -1]).to(torch.float32)
        return x, y if img is None else img.add_(y)


def _stuff(x: torch.Tensor, up: int) -> torch.Tensor:
    """Insert `up - 1` zeros after every sample, which is the up half of `upfirdn2d`."""
    n, c, h, w = x.shape
    x = F.pad(x.reshape(n, c, h, 1, w, 1), [0, up - 1, 0, 0, 0, up - 1])
    return x.reshape(n, c, h * up, w * up)


class Mapping(nn.Module):
    """`z` to `w`, and the average `w` that truncation pulls toward."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.num_ws = cfg.num_ws
        #: A live truncation per style range, or `None` for the network as trained. A plain
        #: attribute, set before `torch.compile`, like `SynthesisLayer.noise_gain`.
        self.truncation = None
        #: The push buffer: a live offset per style range, `(len(BANDS), w_dim)`, written by
        #: the direction dials. `None` for the network as trained.
        self.push = None
        band = torch.zeros(cfg.num_ws, len(BANDS))
        for i, (_name, lo, hi) in enumerate(BANDS):
            band[lo:min(hi, cfg.num_ws), i] = 1.0
        # `bands @ per_range` spreads one value per style range over every `w`, in one op.
        self.register_buffer("bands", band, persistent=False)
        for i in range(cfg.num_layers):
            n_in = cfg.z_dim if i == 0 else cfg.w_dim
            setattr(self, f"fc{i}", Dense(n_in, cfg.w_dim, activation="lrelu",
                                          lr_multiplier=cfg.lr_multiplier))
        self.layers = [getattr(self, f"fc{i}") for i in range(cfg.num_layers)]
        self.register_buffer("w_avg", torch.zeros(cfg.w_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.to(torch.float32)
        x = x * (x.square().mean(dim=1, keepdim=True) + 1e-8).rsqrt()
        for layer in self.layers:
            x = layer(x)
        x = x.unsqueeze(1).repeat(1, self.num_ws, 1)
        # Truncation: below 1 the style is pulled toward the average `w` and the picture
        # becomes more typical; above it, less.
        if self.truncation is not None:
            x = self.w_avg.lerp(x, (self.bands @ self.truncation).reshape(1, -1, 1))
        # After the truncation, so a direction dial does not weaken when truncation is
        # turned down.
        if self.push is not None:
            x = x + (self.bands @ self.push).unsqueeze(0)
        return x


class Generator(nn.Module):
    """A StyleGAN2 generator that captures in one graph and loads NVIDIA's weights as they are."""

    def __init__(self, cfg: Config = Config()) -> None:
        super().__init__()
        self.cfg = cfg
        self.mapping = Mapping(cfg)
        self.synthesis = nn.Module()
        blocks = []
        for i, res in enumerate(cfg.resolutions):
            block = Block(cfg.channels(res // 2) if i else 0, cfg.channels(res), res, cfg,
                          half=res >= cfg.fp16_from)
            setattr(self.synthesis, f"b{res}", block)
            blocks.append(block)
        self.blocks = blocks

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ws = self.mapping(z)
        x = img = None
        for i, block in enumerate(self.blocks):
            # Block `i` reads `ws[2i-1 : 2i+2]`; its last `w` is the next block's first, which
            # is the overlap NVIDIA's `w_idx += num_conv` produces and not an off-by-one.
            x, img = block(x, img, ws[:, max(2 * i - 1, 0):2 * i + 2])
        return img


def noise_sites(net: Generator) -> dict[str, list[SynthesisLayer]]:
    """The steerable noise, one dial per resolution rather than one per layer."""
    out: dict[str, list[SynthesisLayer]] = {}
    for block in net.blocks:
        layers = [layer for layer in (block.conv0, block.conv1) if layer is not None]
        out[f"noise_{block.res}"] = layers
    return out


def affine_sites(net: Generator) -> list[tuple[int, str, Dense]]:
    """Every style affine, with the index of the `w` vector it consumes."""
    out: list[tuple[int, str, Dense]] = []
    for i, block in enumerate(net.blocks):
        base = max(2 * i - 1, 0)
        first = 0 if block.conv0 is None else 1
        for offset, tag in ((0, "conv0"), (first, "conv1"), (first + 1, "torgb")):
            layer = getattr(block, tag, None)
            if layer is not None:
                out.append((base + offset, f"b{block.res}.{tag}", layer.affine))
    return out


def style_bands(net: Generator) -> list[tuple[str, torch.Tensor]]:
    """Per `BANDS` range, the style affine weights that range's `w` vectors feed, stacked."""
    sites = affine_sites(net)
    out = []
    for name, lo, hi in BANDS:
        rows = [affine.weight.detach() for idx, _tag, affine in sites if lo <= idx < hi]
        # `weight_gain` is the same scalar on every affine, and a uniform scaling does not
        # change a basis of unit vectors, so it is not applied.
        out.append((name, torch.cat(rows, 0) if rows
                    else torch.zeros(0, net.mapping.w_avg.shape[0])))
    return out


def load(cfg: Config, state: dict, device="cpu") -> Generator:
    """Build from a checkpoint's own description and load its weights under their own names."""
    net = Generator(cfg).eval().requires_grad_(False)
    missing, unexpected = net.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("resample_filter")]
    if missing or unexpected:
        raise RuntimeError(
            f"this is not the architecture the weights describe: {len(missing)} parameter(s) "
            f"the network wants and the file has not ({missing[:4]}), {len(unexpected)} the "
            f"file has and the network does not ({unexpected[:4]}).")
    net = net.to(device)
    for module in net.modules():
        if isinstance(module, (SynthesisLayer, Dense)):
            module.freeze()
    return net


def save(path, cfg: Config, state: dict) -> None:
    """Write a generator checkpoint that this file alone can open; see `ganlive
    import-stylegan2`. `from_file` ignores any other keys a checkpoint carries."""
    blob = {"format": FORMAT,
            "config": {k: v for k, v in asdict(cfg).items() if k not in NOT_SAVED},
            "state": {k: v for k, v in state.items() if not k.endswith("resample_filter")}}
    torch.save(blob, path)


def is_stylegan2(path) -> bool:
    """Whether this `.pt` is one of these, without building anything. Cached on the file's
    identity, because the bank asks several times per model."""
    try:
        stat = Path(path).stat()
    except OSError:
        return False
    return _is_stylegan2_cached(str(path), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=16)
def _is_stylegan2_cached(path: str, _mtime: int, _size: int) -> bool:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:                                                        # noqa: BLE001
        return False
    return isinstance(blob, dict) and blob.get("format") == FORMAT


#: Every block NVIDIA's rule would ever allow in half precision. It floors at 8, so the
#: 4-pixel block stays fp32 whatever is asked for.
HALF_EVERYWHERE = 8
#: No block in half precision, whatever the checkpoint asked for.
SINGLE_EVERYWHERE = 1 << 30


def half_from_for(dtype: torch.dtype, exact: bool = False) -> int | None:
    """Where the half-precision ladder starts for a session that plays in `dtype`.

    Half everywhere on a card unless `exact` asks for the file's own rule: 11% faster on
    FFHQ-1024, for 0.26 mean 8-bit levels. Nowhere in a single-precision session (the CPU;
    see `device.playback_dtype`), where half precision is the slow path."""
    if dtype is not torch.float16:
        return SINGLE_EVERYWHERE
    return None if exact else HALF_EVERYWHERE


def from_file(path, device="cpu", half_from: int | None = None) -> Generator:
    """Open what `save` wrote, on any machine, with no other code in the room."""
    blob = _open(path)
    return load(_rebuild(blob["config"], half_from), blob["state"], device)
