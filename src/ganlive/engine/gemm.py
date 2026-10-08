"""The matrix product both families run their small convolutions as: A [M x K] (the input,
gathered on the fly) by one or more B [K x N] (weights) sharing A, tiled through workgroup
memory, the sum split into S slices so that even a small map fills the GPU. A second pass,
the caller's, adds the slices and finishes the layer.

Each workgroup is 8x8 threads, and each thread holds `rm` rows by 4 * `rn` columns of every
B, reading 4 rows of A and 4 columns of a B per workgroup-memory read.
"""

from __future__ import annotations

import math

from ganlive.engine.codegen import linear, wgsl

TK = 16                     # the sum advances 16 rows of B at a time
GEMM_MAX = 6144             # maps up to this many pixels run as a product by default
GEMM_TARGET = 256           # workgroups a product aims to fill, by splitting its sum


def split(tiles: int, steps: int) -> int:
    """How many slices a product's sum is split into, so that `tiles` workgroups fill the GPU:
    doubling while that leaves at least 4 of the `steps` of 16 in each slice."""
    S = 1
    while tiles * S < GEMM_TARGET and steps % (S * 2) == 0 and steps // (S * 2) >= 4:
        S *= 2
    return S


def matmul(*, M: int, N: int, K: int, S: int, rm: int, rn: int, mats: tuple[str, ...], decl: str,
           gather: str, bload: str, parts: int = 1, finish=None) -> tuple[str, list[int]]:
    """The shader and workgroups of one product.

    `decl` declares the bindings and helpers, Y being the partial sums (array<vec4f>).
    `gather` sets `v0` and `v1` (vec4f, zero) to A's rows `m4` .. `m4` + 3 at columns `k`
    (even) and `k` + 1, which callers order so that one input read serves both.
    `bload` sets `B<mat>[e]` for each of `mats` to row `k` of that B at columns `nn` .. `nn` + 3.
    Both may read `par`: the product runs `parts` independent products of M rows along z, each
    in S slices, and writes slice partial sums of rows m to
    Y[(wg.z * M + m) * (len(mats) * N / 4) + i * N / 4 + n / 4] for B number i. With one
    slice, `finish` may instead be given each B's sum of row `m`, columns `c` .. `c` + 3, and
    return the WGSL that finishes the layer there, so that no second pass is needed."""
    TM, TN = 8 * rm, 32 * rn
    if N % TN or K % (TK * S) or rm % 4:
        raise ValueError(f"a {M}x{K} by {K}x{N} product does not tile as {TM}x{TN} in {S} slices")
    rows = [(r, i) for r in range(rm // 4) for i in range(4)]
    cols = range(rn)
    width = len(mats) * N // 4
    accs = " ".join(f"var a{x}{r}{i}{c} = vec4f(0.0);" for x in mats for r, i in rows for c in cols)
    reads = " ".join([f"let a{r} = As[kk * {TM // 4}u + lid.y + {8 * r}u];" for r in range(rm // 4)]
                     + [f"let b{x}{c} = B{x}[kk * {TN // 4}u + lid.x + {8 * c}u];" for x in mats for c in cols])
    fma = " ".join(f"a{x}{r}{i}{c} += a{r}.{'xyzw'[i]} * b{x}{c};" for x in mats for r, i in rows for c in cols)
    def stored(r, i, c):
        if finish is not None and S == 1:
            return (f"    let c = n0 + (lid.x + {8 * c}u) * 4u;\n"
                    + finish({x: f"a{x}{r}{i}{c}" for x in mats}))
        return (f"    let row = (wg.z * {M}u + m) * {width}u + n0 / 4u + lid.x + {8 * c}u;\n    "
                + " ".join(f"Y[row + {j * N // 4}u] = a{x}{r}{i}{c};" for j, x in enumerate(mats)))

    store = "\n".join(f"  {{ let m = m0 + (lid.y + {8 * r}u) * 4u + {i}u; if (m < {M}u) {{\n"
                      + stored(r, i, c) + " } }" for r, i in rows for c in cols)
    shared = "\n".join(f"var<workgroup> B{x}: array<vec4f, {TK * TN // 4}>;" for x in mats)
    code = wgsl(decl + """
var<workgroup> As: array<vec4f, ${tkm4}>;
${shared}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32,
        @builtin(local_invocation_id) lid: vec3u) {
  let m0 = wg.x * ${TM}u; let n0 = wg.y * ${TN}u; let s = wg.z % ${S}u; let par = wg.z / ${S}u;
  ${accs}
  for (var k0 = s * ${KS}u; k0 < (s + 1u) * ${KS}u; k0 += ${TK}u) {
    for (var r = 0u; r < ${aloads}u; r++) {         // A, 4 rows by 2 columns at a time
      let e = li + r * 64u; let kk = 2u * (e / ${tm4}u); let k = k0 + kk;
      let m4 = m0 + (e % ${tm4}u) * 4u;
      var v0 = vec4f(0.0); var v1 = vec4f(0.0);
${gather}
      As[kk * ${tm4}u + e % ${tm4}u] = v0; As[(kk + 1u) * ${tm4}u + e % ${tm4}u] = v1;
    }
    for (var r = 0u; r < ${bloads}u; r++) {
      let e = li + r * 64u; let k = k0 + e / ${tn4}u; let nn = n0 + (e % ${tn4}u) * 4u;
${bload}
    }
    workgroupBarrier();
    for (var kk = 0u; kk < ${TK}u; kk++) {
      ${reads}
      ${fma}
    }
    workgroupBarrier();
  }
${store}
}""", tkm4=TK * TM // 4, shared=shared, TM=TM, TN=TN, TK=TK, S=S, KS=K // S, accs=accs,
                aloads=TK * TM // 8 // 64, tm4=TM // 4, gather=gather, bloads=TK * TN // 4 // 64,
                tn4=TN // 4, bload=bload, reads=reads, fma=fma, store=store)
    return code, [math.ceil(M / TM), N // TN, S * parts]


def sum_slices(*, rows: int, M: int, N: int, S: int, mats: tuple[str, ...], decl: str, where: str,
               finish) -> tuple[str, list[int]]:
    """The second pass of a product split in S slices, X being its partial sums: for each of
    `rows` rows `r` and group of columns `c` .. `c` + 3, `where` sets `m`, the product's row,
    and `first`, its first slice, and `finish` is given each B's total."""
    n = rows * (N // 4)
    groups, index = linear(n)
    width = len(mats) * N // 4
    code = decl + f"""
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
  let i = {index};
  if (i >= {n}u) {{ return; }}
  let r = i % {rows}u; let c = (i / {rows}u) * 4u;
  {where}
  {" ".join(f"var s{x} = vec4f(0.0);" for x in mats)}
  for (var s = first; s < first + {S}u; s++) {{
    let row = (s * {M}u + m) * {width}u + c / 4u;
    {" ".join(f"s{x} += X[row + {j * N // 4}u];" for j, x in enumerate(mats))}
  }}
{finish({x: f"s{x}" for x in mats})}
}}"""
    return code, groups
