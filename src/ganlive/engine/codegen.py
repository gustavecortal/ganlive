"""What every shader generator shares: the program format, the weight helpers, WGSL templates
and binding declarations, dispatch shapes, and the Builder that collects a program."""

from __future__ import annotations

import math
from string import Template

FORMAT = "ganlive-engine/1"

WG = 8                      # direct-convolution workgroups are WG x WG threads

#: The most workgroup memory, in 32-bit words, one shader of a desktop program declares, and
#: of a browser program. Firefox zero-fills a workgroup array in one statement its compiler
#: expands word by word, so its compile time climbs with the words: at 1296 rather than 2592,
#: lichen loads in 4.8 s there rather than 7.0, and its frame is 10.0 ms rather than 10.7.
SHARED_WORDS = 3072
BROWSER_SHARED_WORDS = 1296

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


def wgsl(text: str, **values) -> str:
    """WGSL is full of braces, so shaders are templates with `${name}` holes."""
    return Template(text).substitute(values, HELPERS=HELPERS)


def storage(entries) -> str:
    """`@binding` declarations for (name, type) pairs in order, `Y` being the one written."""
    return "\n".join(
        f"@group(0) @binding({i}) var<storage, {'read_write' if name == 'Y' else 'read'}> "
        f"{name}: {kind};" for i, (name, kind) in enumerate(entries))


def bindings(entries) -> tuple[str, list[str]]:
    """The declarations and the buffers of (name, type, buffer) bindings, in order."""
    return storage([(n, k) for n, k, _ in entries]), [buf for *_, buf in entries]


class Builder:
    """Collects shaders (deduplicated), buffers and steps while the ops are compiled."""

    def __init__(self, manifest: dict, plans: dict, browser: bool = False) -> None:
        self.m = manifest
        self.overrides = plans
        #: Whether the program is for a browser, whose defaults differ (see `program._plan`).
        self.browser = browser
        #: The workgroup memory one shader may declare, in words.
        self.shared = BROWSER_SHARED_WORDS if browser else SHARED_WORDS
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

    def step(self, name, code, bind, groups, *, load=False) -> None:
        step = {"name": name, "shader": self.shader(code), "bind": list(bind),
                "groups": [int(g) for g in groups]}
        (self.load if load else self.steps).append(step)

    def shape(self, tensor: str) -> list[int]:
        return self.m["tensors"][tensor]
