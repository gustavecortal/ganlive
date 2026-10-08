"""What every shader generator shares: the program format, the weight helpers, WGSL templates
and binding declarations, dispatch shapes, and the Builder that collects a program."""

from __future__ import annotations

import math
from string import Template

FORMAT = "ganlive-engine/1"

WG = 8                      # direct-convolution workgroups are WG x WG threads
GEMM_TARGET = 256           # workgroups a matrix product aims to fill, by splitting its sum

# w1 / w4: one / four fp16 weights at element offset `e` of the weight blob P.
HELPERS = """
fn w1(e: u32) -> f32 { return unpack2x16float(P[e >> 1u])[e & 1u]; }
fn w4(e: u32) -> vec4f { let i = e >> 1u; return vec4f(unpack2x16float(P[i]), unpack2x16float(P[i + 1u])); }
fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn sigmoid4(x: vec4f) -> vec4f { return 1.0 / (1.0 + exp(-x)); }
"""


def linear(n: int, size: int = 64) -> tuple[list[int], str]:
    """The workgroups for `n` invocations, `size` a group, in rows of at most 65535 groups (the
    per-dimension limit), and the WGSL expression for an invocation's index among the `n`."""
    groups = math.ceil(n / size)
    return [min(groups, 65535), math.ceil(groups / 65535), 1], f"(id.y * {65535 * size}u + id.x)"


def split(tiles: int, steps: int) -> int:
    """How many slices a matrix product's sum is split into, so that `tiles` workgroups fill the
    GPU: doubling while that leaves at least 4 of the `steps` of 16 in each slice."""
    S = 1
    while tiles * S < GEMM_TARGET and steps % (S * 2) == 0 and steps // (S * 2) >= 4:
        S *= 2
    return S


def wgsl(text: str, **values) -> str:
    """WGSL is full of braces, so shaders are templates with `${name}` holes."""
    return Template(text).substitute(values, HELPERS=HELPERS)


def storage(entries) -> str:
    """`@binding` declarations for (name, type) pairs in order; `Y` is the one written."""
    return "\n".join(
        f"@group(0) @binding({i}) var<storage, {'read_write' if name == 'Y' else 'read'}> "
        f"{name}: {kind};" for i, (name, kind) in enumerate(entries))


class Builder:
    """Collects shaders (deduplicated), buffers and steps while the ops are compiled."""

    def __init__(self, manifest: dict, plans: dict) -> None:
        self.m = manifest
        self.overrides = plans
        self.shaders: list[str] = []
        self.index: dict[str, int] = {}
        self.buffers: dict[str, dict] = {}
        self.steps: list[dict] = []          # every frame
        self.load: list[dict] = []           # once, after the buffers are made

    def buffer(self, name: str, size: int, **spec) -> str:
        self.buffers[name] = {"size": max(16, math.ceil(size / 4) * 4), **spec}
        return name

    def shader(self, code: str) -> int:
        if code not in self.index:
            self.index[code] = len(self.shaders)
            self.shaders.append(code)
        return self.index[code]

    def step(self, name, code, bind, groups, plan=None, *, load=False) -> None:
        step = {"name": name, "shader": self.shader(code), "bind": list(bind),
                "groups": [int(g) for g in groups]}
        if plan:
            step["plan"] = plan
        (self.load if load else self.steps).append(step)

    def shape(self, tensor: str) -> list[int]:
        return self.m["tensors"][tensor]
