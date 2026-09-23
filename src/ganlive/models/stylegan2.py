"""StyleGAN2 synthesis, written so that `torch.compile` can capture it in one graph."""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LRELU_SLOPE = 0.2
LRELU_GAIN = math.sqrt(2.0)
#: What each branch of a residual is scaled by so that adding the two keeps unit variance.
SQRT_HALF = math.sqrt(0.5)
#: Every StyleGAN2-ADA checkpoint published uses this resample filter.
TAPS = (1.0, 3.0, 3.0, 1.0)
#: The three style ranges every StyleGAN paper and every user of one talks in. The bounds are absolute
#: rather than proportional because they are about *resolution*.
BANDS = (("w_coarse", 0, 4), ("w_mid", 4, 8), ("w_fine", 8, 1 << 30))

#: Which pixel stages each range actually drives, as a phrase anything showing a person a dial can use.
BAND_PIXELS = {"w_coarse": "the 4, 8 and 16 pixel stages",
               "w_mid": "the 16, 32 and 64 pixel stages",
               "w_fine": "everything from 64 pixels up to native"}
#: Written into every converted file. A StyleGAN2 and one of this project's own checkpoints
#: are both `.pt`, and the instrument has to tell them apart before it opens either.
FORMAT = "ganlive-stylegan2/1"
#: The tag written under this project's earlier name. Read, never written, so a
#: checkpoint converted before the rename still opens.
FORMATS = (FORMAT, "smallgen-stylegan2/1")


#: Config fields that describe how to *play* a checkpoint rather than what is in it. `save` drops them, so a
#: converted file is byte-identical whether or not one was in force.
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
    #: The lowest resolution to run in half precision, overriding NVIDIA's rule below. `None` is the
    #: checkpoint as they shipped it.
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

    # The two names `bank` asks every checkpoint for, whatever kind of file it came from.
    # Spelled here rather than wrapped in a third config class, because they are the same two
    # numbers this already holds under the names NVIDIA gave them.
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
    if blob.get("format") not in FORMATS:
        raise RuntimeError(f"{path} is not a {FORMAT} checkpoint. "
                           f"`ganlive import-stylegan2` writes them.")
    return blob


def _rebuild(raw: dict, half_from: int | None = None) -> Config:
    """A stored config back into its dataclass."""
    return Config(**{**raw, "taps": tuple(raw["taps"]), "half_from": half_from})


def config_of(path) -> Config:
    """What a converted checkpoint says about itself, without loading 133 MB of weights."""
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
    """`FullyConnectedLayer`. The gains stay outside the weight so the numbers match."""

    def __init__(self, n_in: int, n_out: int, activation: str = "linear",
                 lr_multiplier: float = 1.0, bias_init: float = 0.0) -> None:
        super().__init__(n_in, n_out, bias=True)
        with torch.no_grad():
            self.weight.copy_(torch.randn(n_out, n_in) / lr_multiplier)
            self.bias.fill_(float(bias_init))
        self.weight_gain = float(lr_multiplier / math.sqrt(n_in))
        self.bias_gain = float(lr_multiplier)
        self.lrelu = activation == "lrelu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_gain
        b = self.bias
        if self.bias_gain != 1.0:
            b = b * self.bias_gain
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
        self.knob = None

        self.up, self.half = up, half
        dtype = torch.float16 if half else torch.float32
        # NVIDIA renormalises the weight and the styles before a half-precision modulation so
        # neither overflows. Demodulation divides any such scaling straight back out, so this
        # buys range and changes no result -- and it runs only where it is needed.
        self.prenorm = float(1.0 / math.sqrt(in_ch * kernel * kernel)) if half else 0.0
        #: `prenorm` over the weight's per-channel inf-norm, once the weights are final -- see
        #: `freeze_scale`. `None` recomputes it every forward, for a net built without `load`.
        self.register_buffer("wscale", None, persistent=False)
        self.pad = kernel // 2
        self.clamp = None if conv_clamp is None else float(conv_clamp)
        if up > 1:
            ty, tx, *rest = _pads(kernel, up, len(taps), self.pad)
            self.tpad, self.fpad = (ty, tx), tuple(rest)
            f = (fir(taps) * float(up * up)).flip([0, 1]).to(dtype)
            self.register_buffer("upfir", f[None, None].repeat(out_ch, 1, 1, 1),
                                 persistent=False)

    @torch.no_grad()
    def freeze_scale(self) -> None:
        """Compute the half-precision weight renormalisation once. It is a reduction over a
        constant weight, and inside the captured graph it ran every frame: every conv weight
        but the 4-pixel block's read a second time, ~85 MB a frame on FFHQ-1024."""
        if self.half:
            self.wscale = self.prenorm / self.weight.norm(float("inf"), dim=[1, 2, 3],
                                                          keepdim=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        styles = self.affine(w)
        weight = self.weight
        if self.half:
            scale = self.wscale
            if scale is None:
                scale = self.prenorm / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True)
            weight = weight * scale
            styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)

        # `weight` and not `self.weight`: with a low-rank adapter attached, `.weight` is a parametrisation
        # that recomputes `B @ A` and materialises the whole tensor on every access --
        # `torch.nn.utils.parametrize` caches nothing outside a `cached()` block.
        n, out_ch, in_ch, kh, kw = x.shape[0], *weight.shape
        m = weight.unsqueeze(0) * styles.reshape(n, 1, -1, 1, 1)
        dcoefs = (m.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        m = (m * dcoefs.reshape(n, -1, 1, 1, 1)).to(x.dtype)

        # One image per group, which for the one frame this renders is one group and a plain
        # convolution: every reshape here is a view.
        x = x.reshape(1, -1, *x.shape[2:])
        if self.up > 1:
            m = m.transpose(1, 2).reshape(n * in_ch, out_ch, kh, kw)
            x = F.conv_transpose2d(x, m, stride=self.up, padding=self.tpad, groups=n)
            x = _fir(x, self.upfir.repeat(n, 1, 1, 1) if n > 1 else self.upfir, self.fpad)
        else:
            x = F.conv2d(x, m.reshape(n * out_ch, in_ch, kh, kw), padding=self.pad, groups=n)
        x = x.reshape(n, -1, *x.shape[2:])

        # The two scalars multiply each other rather than the pattern, so a live gain costs one broadcast
        # and not two.
        strength = self.noise_strength if self.knob is None else self.noise_strength * self.knob
        x = x.add_(self.noise_const * strength)
        x = F.leaky_relu(x + self.bias.to(x.dtype).reshape(1, -1, 1, 1), LRELU_SLOPE)
        x = x * LRELU_GAIN
        return x if self.clamp is None else x.clamp(-self.clamp, self.clamp)


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
        weight = self.weight                     # once: see the note in `SynthesisLayer`
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
        self.dtype = torch.float16 if half else torch.float32
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
        #: A live truncation per style range, or `None` for the network as trained. Same rule
        #: as `SynthesisLayer.knob`: a plain attribute, set before `torch.compile`.
        self.knob = None
        #: A live offset per style range -- `(len(BANDS), w_dim)`, the direction dials' push.
        self.push = None
        band = torch.zeros(cfg.num_ws, len(BANDS))
        for i, (_name, lo, hi) in enumerate(BANDS):
            band[lo:min(hi, cfg.num_ws), i] = 1.0
        # `[num_ws, 3] @ [3]` is a fifty-four element matmul that turns three dials into a
        # truncation for every layer. Cheaper to say than to special-case three slices, and it
        # is one op in the graph rather than three writes into a tensor.
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
        # **Truncation is the control this family is known for, and it is host arithmetic.** Below 1 the
        # style is pulled toward the average the checkpoint was trained around and the picture becomes more
        # typical; above it, less.
        if self.knob is not None:
            x = self.w_avg.lerp(x, (self.bands @ self.knob).reshape(1, -1, 1))
        # **After the truncation and not before it.** Both orders are defensible on paper; only one of them
        # makes an instrument.
        if self.push is not None:
            x = x + (self.bands @ self.push).unsqueeze(0)
        return x


class Generator(nn.Module):
    """A StyleGAN2 generator that captures in one graph and loads NVIDIA's weights as they are."""

    def __init__(self, cfg: Config = Config()) -> None:
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_dim
        self.mapping = Mapping(cfg)
        self.synthesis = nn.Module()
        blocks = []
        for i, res in enumerate(cfg.resolutions):
            block = Block(cfg.channels(res // 2) if i else 0, cfg.channels(res), res, cfg,
                          half=res >= cfg.fp16_from)
            setattr(self.synthesis, f"b{res}", block)
            blocks.append(block)
        self.blocks = blocks

    def forward(self, z: torch.Tensor, c=None) -> torch.Tensor:
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
        # `weight_gain` is `1 / sqrt(w_dim)` for every one of these -- the same scalar on
        # every affine -- and a basis of unit vectors cannot be moved by a uniform scaling,
        # so it is deliberately not applied here.
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
        if isinstance(module, SynthesisLayer):
            module.freeze_scale()
    return net


def save(path, cfg: Config, state: dict) -> None:
    """Write a checkpoint that this file alone can open. See `ganlive import-stylegan2`.

    A generator, because a generator is the whole of what this project runs. A checkpoint
    that carries a discriminator alongside -- a trainer's, say -- still opens here: the extra
    keys are simply not read."""
    from dataclasses import asdict

    blob = {"format": FORMAT,
            "config": {k: v for k, v in asdict(cfg).items() if k not in NOT_SAVED},
            "state": {k: v for k, v in state.items() if not k.endswith("resample_filter")}}
    torch.save(blob, path)


def is_stylegan2(path) -> bool:
    """Whether this `.pt` is one of these, without building anything.

    Cached on the file's identity, like `onnx.dials_of`: the bank asks four times for
    every model it loads -- the dispatcher, the layout, the capture gate, the shelf --
    and each ask was opening the checkpoint again to read one string."""
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
    return isinstance(blob, dict) and blob.get("format") in FORMATS


#: Every block NVIDIA's own rule would ever allow in half precision -- their formula floors at 8, so the
#: 4-pixel block stays fp32 whatever is asked for.
HALF_EVERYWHERE = 8
#: No block in half precision, whatever the checkpoint asked for.
SINGLE_EVERYWHERE = 1 << 30


def half_from_for(dtype: torch.dtype, exact: bool = False) -> int | None:
    """Where the half-precision ladder starts for a session that plays in `dtype`.

    Half everywhere on a card unless `exact` asks for the file's own rule: 16.76 ms to 14.87
    on FFHQ-1024 for 0.26 mean 8-bit levels. Nowhere in a single-precision session -- the CPU,
    see `device.playback_dtype` -- exact or not, since the file's own four half blocks would be
    the slow path there."""
    if dtype is not torch.float16:
        return SINGLE_EVERYWHERE
    return None if exact else HALF_EVERYWHERE


def from_file(path, device="cpu", half_from: int | None = None) -> Generator:
    """Open what `save` wrote, on any machine, with no other code in the room."""
    blob = _open(path)
    return load(_rebuild(blob["config"], half_from), blob["state"], device)
