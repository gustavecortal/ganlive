"""The FastGAN generator, in plain PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import spectral_norm

G_WIDTH = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125,
           2048: 0.125}

SUPPORTED_SIZES = (256, 512, 1024, 2048)

BASE = 4

def _memory_format(x: torch.Tensor) -> torch.memory_format:
    if x.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    return torch.contiguous_format


@dataclass(frozen=True)
class Ladder:
    """The (height, width) the generator climbs — square or not."""

    height: int
    width: int
    #: Whether this size has to be one *this project's* generator could build.
    built_here: bool = field(default=True, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.built_here:
            return
        if self.height not in SUPPORTED_SIZES:
            raise ValueError(f"height must be one of {SUPPORTED_SIZES}, got {self.height}")
        step = self.height // BASE
        if self.width <= 0 or self.width % step:
            raise ValueError(
                f"width must be a positive multiple of {step} for a {self.height}-tall "
                f"ladder, so that every rung down to the {BASE}-high base block is a "
                f"whole number of pixels. Got {self.width}; the nearest legal values "
                f"are {self.width // step * step} and {(self.width // step + 1) * step}.")

    @classmethod
    def of(cls, height: int, width: int | None = None) -> Ladder:
        """`width=None` means square, which is what every existing checkpoint is."""
        return cls(height, height if width is None else width)

    @property
    def base_width(self) -> int:
        """Width of the `BASE`-high bottom rung. 4 when square, 6 at 3:2."""
        return BASE * self.width // self.height

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def at(self, rung: int) -> tuple[int, int]:
        """The (height, width) of the ladder `rung` rows tall."""
        return rung, rung * self.base_width // BASE


def _widths(table: dict[int, float], base: int) -> dict[int, int]:
    return {k: int(v * base) for k, v in table.items()}


def conv(*args, spectral: bool = True, **kwargs) -> nn.Module:
    """Spectrally normalised convolution."""
    c = nn.Conv2d(*args, **kwargs)
    return spectral_norm(c) if spectral else c


def conv_transpose(*args, **kwargs) -> nn.Module:
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def gated() -> nn.Module:
    """`nn.GLU` over the channel axis: `a * sigmoid(b)`, halving channels."""
    return nn.GLU(dim=1)


class NoiseInjection(nn.Module):
    """Per-pixel Gaussian noise with a learned scalar gain, StyleGAN-style."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.freeze = False
        self.frozen: torch.Tensor | None = None
        self.generator: torch.Generator | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if self.freeze:
            if self.frozen is None or self.frozen.shape[-2:] != (h, w):
                if self.generator is None:
                    self.frozen = torch.randn(1, 1, h, w, device=x.device, dtype=x.dtype)
                else:
                    self.frozen = torch.randn(
                        1, 1, h, w, generator=self.generator).to(x.device, x.dtype)
            return x + self.weight * self.frozen.to(dtype=x.dtype)
        noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class SkipLayerExcitation(nn.Module):
    """Channel-wise gate: a low-res feature map modulates a high-res one."""

    def __init__(self, ch_low: int, ch_high: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv(ch_low, ch_high, 4, 1, 0, bias=False),
            nn.SiLU(),  # x * sigmoid(x); "Swish" in the paper
            conv(ch_high, ch_high, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * self.gate(low)


def pixel_norm(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Put every latent on the same shell, so the MLP is not also spending capacity"""
    return z / z.square().mean(dim=1, keepdim=True).add(eps).sqrt()


class MappingNetwork(nn.Module):
    """StyleGAN1's `z -> w` MLP. The untested candidate for the popping."""

    def __init__(self, nz: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"mapping depth must be at least 1, got {depth}")
        layers: list[nn.Module] = []
        for _ in range(depth):
            linear = nn.Linear(nz, nz)
            nn.init.kaiming_normal_(linear.weight, a=0.2, nonlinearity="leaky_relu")
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.LeakyReLU(0.2, inplace=True)]
        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(pixel_norm(z))


class InitLayer(nn.Module):
    """z -> the base feature map, via a transposed conv from a 1x1 "image"."""

    def __init__(self, nz: int, ch: int, shape: tuple[int, int]) -> None:
        super().__init__()
        self.main = nn.Sequential(
            conv_transpose(nz, ch * 2, shape, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 2),
            gated(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z.view(z.shape[0], -1, 1, 1))


def _upsample_conv(ch_in: int, ch_mid: int, subpixel: bool) -> list[nn.Module]:
    """The 2x-upsampling front of a block: resample and convolve, in either order."""
    if subpixel:
        return [conv(ch_in, ch_mid * 4, 3, 1, 1, bias=False), nn.PixelShuffle(2)]
    return [nn.Upsample(scale_factor=2, mode="nearest"),
            conv(ch_in, ch_mid, 3, 1, 1, bias=False)]


def up_block(ch_in: int, ch_out: int, subpixel: bool = False) -> nn.Sequential:
    """Cheap 2x upsample block: resample + conv + BN + GLU."""
    return nn.Sequential(
        *_upsample_conv(ch_in, ch_out * 2, subpixel),
        nn.BatchNorm2d(ch_out * 2),
        gated(),
    )


def up_block_comp(ch_in: int, ch_out: int, subpixel: bool = False) -> nn.Sequential:
    """The expensive variant: two convs, each with noise injection."""
    return nn.Sequential(
        *_upsample_conv(ch_in, ch_out * 2, subpixel),
        NoiseInjection(),
        nn.BatchNorm2d(ch_out * 2),
        gated(),
        conv(ch_out, ch_out * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        nn.BatchNorm2d(ch_out * 2),
        gated(),
    )


class Generator(nn.Module):
    """Outputs two scales: the full-resolution image and a 128-high version."""

    def __init__(self, ngf: int = 64, nz: int = 256, nc: int = 3, im_size: int = 512,
                 im_width: int | None = None, pixelshuffle_from: int = 0,
                 mapping_depth: int = 0) -> None:
        super().__init__()
        self.ladder = Ladder.of(im_size, im_width)
        w = _widths(G_WIDTH, ngf)
        self.mapping = MappingNetwork(nz, mapping_depth) if mapping_depth else None

        def sub(rung: int) -> bool:
            """Whether the block *producing* `rung` uses a sub-pixel convolution."""
            return bool(pixelshuffle_from) and rung >= pixelshuffle_from

        self.init = InitLayer(nz, w[4], self.ladder.at(BASE))
        self.feat_8 = up_block_comp(w[4], w[8], sub(8))
        self.feat_16 = up_block(w[8], w[16], sub(16))
        self.feat_32 = up_block_comp(w[16], w[32], sub(32))
        self.feat_64 = up_block(w[32], w[64], sub(64))
        self.feat_128 = up_block_comp(w[64], w[128], sub(128))
        self.feat_256 = up_block(w[128], w[256], sub(256))

        self.se_64 = SkipLayerExcitation(w[4], w[64])
        self.se_128 = SkipLayerExcitation(w[8], w[128])
        self.se_256 = SkipLayerExcitation(w[16], w[256])

        self.to_small = conv(w[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv(w[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = up_block_comp(w[256], w[512], sub(512))
            self.se_512 = SkipLayerExcitation(w[32], w[512])
        if im_size > 512:
            self.feat_1024 = up_block(w[512], w[1024], sub(1024))
        if im_size > 1024:
            self.feat_2048 = up_block_comp(w[1024], w[2048], sub(2048))

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        """`z -> images`, through the mapping network if there is one."""
        return self.synthesise(self.mapping(z) if self.mapping is not None else z)

    def synthesise(self, w: torch.Tensor) -> list[torch.Tensor]:
        """`w -> images`, skipping the mapping network. The synthesis half alone."""
        f4 = self.init(w)
        f8 = self.feat_8(f4)
        f16 = self.feat_16(f8)
        f32 = self.feat_32(f16)
        f64 = self.se_64(f4, self.feat_64(f32))
        f128 = self.se_128(f8, self.feat_128(f64))
        feat = self.se_256(f16, self.feat_256(f128))

        if self.ladder.height > 256:
            feat = self.se_512(f32, self.feat_512(feat))
        if self.ladder.height > 512:
            feat = self.feat_1024(feat)
        if self.ladder.height > 1024:
            feat = self.feat_2048(feat)

        return [torch.tanh(self.to_big(feat)), torch.tanh(self.to_small(f128))]


def freeze_noise(net: nn.Module, freeze: bool = True, *,
                 seed: int | None = None) -> nn.Module:
    """Pin every `NoiseInjection` layer, so the same `z` gives the same pixels."""
    generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    for m in net.modules():
        if isinstance(m, NoiseInjection):
            m.freeze = freeze
            m.generator = generator
            m.frozen = None
    return net


def denormalise(x: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1, 1] -> [0, 1], ready to save as an image."""
    return x.float().add(1).mul(0.5).clamp(0, 1)


def first_image(out):
    """The image a generator returned, whichever calling convention it uses."""
    return out[0] if isinstance(out, (list, tuple)) else out




@dataclass(frozen=True)
class Config:
    """The architecture a FastGAN checkpoint was trained with -- all a player needs of it."""

    nz: int = 256
    ngf: int = 64
    im_size: int = 512
    im_width: int | None = None
    pixelshuffle_from: int = 0
    mapping_depth: int = 0

    @property
    def ladder(self) -> Ladder:
        return Ladder.of(self.im_size, self.im_width)

    @property
    def generator_kwargs(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _config_from(saved: dict) -> Config:
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in saved.items() if k in known})


def _strip_compile_prefix(state: dict) -> dict:
    """Drop `_orig_mod.` from every key that has it, so compiled checkpoints still load."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k.removeprefix(prefix): v for k, v in state.items()}


def config_of(checkpoint: str | Path) -> Config:
    """What a checkpoint was trained with, without building the generator."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    return _config_from(ckpt.get("config", {}))


def load(checkpoint: str | Path, device=None) -> tuple[Generator, Config]:
    """Rebuild the EMA generator and the architecture it was trained with."""
    from ganlive.device import detect_backend

    target = torch.device(device) if device is not None else torch.device(detect_backend())
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = _config_from(ckpt.get("config", {}))
    net = Generator(**cfg.generator_kwargs).to(target)
    net.load_state_dict(_strip_compile_prefix(ckpt["g_ema"]))
    net.eval().requires_grad_(False)
    return net, cfg


def pin_noise(net: Generator, seed: int) -> None:
    """Freeze the noise layers to a pattern that depends only on `seed`."""
    freeze_noise(net, seed=seed)
