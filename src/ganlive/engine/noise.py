"""The noise every engine model plays: seeded per layer, made on the GPU at load by `NOISE`, and
computed the same way on the host by `seeded_noise`, so a conversion can give PyTorch the noise
the engine draws."""

from __future__ import annotations

import numpy as np

# Standard normal noise made on the GPU (a PCG hash per element, Box-Muller), so no host
# ships or computes it. U holds the element count and the seed.
NOISE = """
@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> U: array<u32>;
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn unit(v: u32) -> f32 { return (f32(pcg(v) >> 8u) + 0.5) / 16777216.0; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.y * 16776960u + id.x;          // rows of 65535 workgroups of 256
  if (i >= U[0]) { return; }
  let a = unit((2u * i) ^ U[1]); let b = unit((2u * i + 1u) ^ U[1]);
  Y[i] = sqrt(-2.0 * log(a)) * cos(6.2831853 * b);
}"""


def seeded_noise(n: int, seed: int) -> np.ndarray:
    """What `NOISE` writes into an n-element buffer, computed on the host: the noise every
    engine model plays, so that `convert` can give PyTorch the same."""
    i = np.arange(n, dtype=np.uint32)

    def pcg(v):
        s = v * np.uint32(747796405) + np.uint32(2891336453)
        w = ((s >> ((s >> np.uint32(28)) + np.uint32(4))) ^ s) * np.uint32(277803737)
        return (w >> np.uint32(22)) ^ w

    def unit(v):
        return ((pcg(v) >> np.uint32(8)).astype(np.float32) + np.float32(0.5)) / np.float32(16777216.0)

    a = unit((np.uint32(2) * i) ^ np.uint32(seed))
    b = unit((np.uint32(2) * i + np.uint32(1)) ^ np.uint32(seed))
    return (np.sqrt(np.float32(-2.0) * np.log(a)) * np.cos(np.float32(6.2831853) * b)).astype(np.float32)



def noise_seed(text: str) -> int:
    """FNV-1a, so each noise layer gets its own seed from its name."""
    h = 2166136261
    for ch in text:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h
