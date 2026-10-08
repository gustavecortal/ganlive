"""The rewrites a FastGAN goes through on its way to the engine: each must draw the same picture.

The claim is mathematical identity, not bit identity: splitting one convolution into two
changes the order floating-point results accumulate in. `IDENTICAL` is far under one 8-bit
level, so any real change fails it and reassociation does not.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ganlive.dials import steer as K
from ganlive.models.fastgan import Generator, freeze_noise
from ganlive.models.fold import prepare_for_inference
from ganlive.models.rewrite import (
    GatedPair,
    equivalent,
    settings_as_input,
    split_gated_convs,
)
from ganlive.settings import Settings

#: Worst tolerable difference, in 8-bit levels. See the module note.
IDENTICAL = 1e-3


def _folded(nz: int = 32, seed: int = 0, gain: float = 2.0) -> tuple[nn.Module, int]:
    """A folded generator small enough for a test and loud enough to detect a change."""
    torch.manual_seed(seed)
    net = Generator(ngf=8, nz=nz, im_size=256, im_width=384)
    freeze_noise(net, seed=0)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(gain)
    prepared = prepare_for_inference(net, nz, "cpu", half=False)
    return prepared["net"].eval(), nz


def test_it_invents_no_parameters_and_drops_none():
    """Splitting a conv's output channels partitions its weights; it does not copy them."""
    net, _ = _folded()
    before = sum(p.numel() for p in net.parameters())
    split_gated_convs(net)
    assert sum(p.numel() for p in net.parameters()) == before


def test_every_gated_conv_is_rewritten_and_only_the_init_layer_keeps_its_split():
    """One GLU survives on purpose, and it is worth naming which."""
    net, _ = _folded()
    rewritten = split_gated_convs(net)
    assert rewritten > 0 and [m for m in net.modules() if isinstance(m, GatedPair)]

    survivors = [name for name, m in net.named_modules() if isinstance(m, nn.GLU)]
    assert all(name.startswith("init") for name in survivors), survivors


def test_a_gated_pair_is_exactly_conv_then_glu():
    """The unit claim, on one block, with no generator around it."""
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, 1, 1)
    reference = nn.Sequential(copy.deepcopy(conv), nn.GLU(dim=1)).eval()
    pair = GatedPair(conv).eval()

    x = torch.randn(2, 3, 16, 24)
    with torch.no_grad():
        assert torch.allclose(reference(x), pair(x), atol=1e-6)


def test_the_noise_coefficient_is_split_the_same_way_as_the_weights():
    """`FoldedNoise` carries one coefficient per channel, so it partitions with them, and the
    split net draws the same picture as the one it came from."""
    # Two nets from the same seed rather than a deepcopy: `spectral_norm` leaves the weights
    # non-leaf and `deepcopy` refuses them.
    before, _ = _folded()
    net, nz = _folded()
    split_gated_convs(net)
    pairs = [m for m in net.modules() if isinstance(m, GatedPair) and m.noise]
    assert pairs, "no noise-carrying block was rewritten"
    for pair in pairs:
        assert pair.coeff_value.shape == pair.coeff_gate.shape
        assert pair.coeff_value.shape[1] == pair.value.out_channels
    assert equivalent(before, net.eval(), nz, "cpu", probes=2) < IDENTICAL


def test_equivalent_reports_in_8_bit_levels_and_catches_a_real_difference():
    """The gate has to fail when something moves, or it is not a gate."""
    net, nz = _folded()
    other, _ = _folded()
    assert equivalent(net, other, nz, "cpu", probes=1) == 0.0, "same seed, same net"

    # Through `parameters()`, not `to_big.weight`: `to_big` is spectral-normalised, so its `.weight` is
    # computed on access and writing to it modifies a temporary that is thrown away.
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.mul_(1.5)
    assert equivalent(net, other, nz, "cpu", probes=1) > 0.5


def test_the_rewrite_refuses_to_leave_a_setting_behind():
    """It counts settings reached, not modules replaced: one noise setting drives several
    modules, so counting modules would accept a graph with a frozen dial in it."""
    torch.manual_seed(4)
    net = Generator(ngf=16, nz=32, im_size=256, im_width=384).eval()
    freeze_noise(net, seed=3)
    net = prepare_for_inference(net, 32, "cpu", half=False)["net"].eval()
    settings = K.install(net, "cpu", torch.float32)

    # Slots read by position, so a vector the modules do not read would name other dials.
    with pytest.raises(RuntimeError, match="another settings vector"):
        settings_as_input(net, Settings(settings.names, "cpu", torch.float32))

    settings.names.append("sle.se_2048")         # a setting no module can possibly serve
    settings.index["sle.se_2048"] = len(settings.names) - 1
    with pytest.raises(RuntimeError, match="sle.se_2048"):
        settings_as_input(net, settings)
