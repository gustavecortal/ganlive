"""The direct 3x3 convolution both families run their larger maps with: each thread BY x BX
output pixels and `oct` channels of every weight set, so that every weight read serves all of
the thread's pixels. A plan may add "f32" (weights made f32 at load, read without converting)
and "slm" (the workgroup reads the input block it needs into workgroup memory once, rather
than each thread reading its neighbours from the buffer)."""

from __future__ import annotations

import math
from collections.abc import Callable

from ganlive.engine.codegen import WG, Builder, bindings, linear, storage, wgsl

#: The direct plans worth trying on a new machine: the tiles that do not spill registers, with
#: f32 weights and workgroup memory or without.
CHOICES = [{"by": by, "bx": bx, "oct": oct, **extra}
           for by, bx, oct in ((2, 2, 8), (2, 2, 4), (1, 2, 8), (1, 4, 4), (2, 4, 4))
           for extra in ({}, {"f32": True}, {"slm": True}, {"f32": True, "slm": True})]


def check(plan: dict, name: str, cout: int, h: int, w: int, keys=("by", "bx", "oct", "f32", "slm")) -> dict:
    """`plan` if its tile fits `cout` channels of an h x w map, refused otherwise."""
    if (not set(plan) <= set(keys) or plan["oct"] % 4 or cout % plan["oct"]
            or h % plan["by"] or w % plan["bx"]):
        raise ValueError(f"{name}: {plan} does not fit {cout} channels at {h}x{w}")
    return plan


def conv3x3(b: Builder, *, step: str, cin: int, cout: int, h: int, w: int, plan: dict,
            mats: dict[str, int], entries: list[tuple[str, str, str]],
            finish: Callable[[dict[str, str]], str], scale: str | None = None,
            helpers: str = "") -> None:
    """Add the conv as `step`. `mats` maps each weight set's name to its offset in P, laid out
    [cin][3*3][cout]. `entries` are the bindings (name, type, buffer), X being the input and Y
    the output. `finish` is given each set's sum of channels c..c+3 at pixel `px` and returns
    the WGSL that stores them; `scale`, a vec2f expression of `c2`, scales an input pair."""
    BY, BX, oct = plan["by"], plan["bx"], plan["oct"]
    nv, f32w, tiled = oct // 4, plan.get("f32", False), plan.get("slm", False)
    pix = [(dy, dx) for dy in range(BY) for dx in range(BX)]
    grid = [(r, c) for r in range(-1, BY + 1) for c in range(-1, BX + 1)]
    each = [(p, j) for p in range(len(pix)) for j in range(nv)]

    def nb(r, c):
        return f"n{r + 1}_{c + 1}"

    RY, RX = WG * BY + 2, WG * BX + 2
    CH = max(1, min(cin // 2, 3072 // (RY * RX)))      # channel pairs a tiled block holds
    while (cin // 2) % CH:
        CH -= 1
    body = [f"  let oy = id.y * {BY}u; let ox = id.x * {BX}u;"]
    if tiled:
        body += [f"  let y0 = i32(wg.y * {WG * BY}u) - 1; let x0 = i32(wg.x * {WG * BX}u) - 1;",
                 f"  let ly = lid.y * {BY}u + 1u; let lx = lid.x * {BX}u + 1u;"]
    else:
        body += [f"  if (oy >= {h}u || ox >= {w}u) {{ return; }}",
                 "  let by = i32(oy); let bx = i32(ox);"]
        for r, c in grid:
            body += [f"  let v{nb(r, c)} = by + {r} >= 0 && by + {r} < {h} && bx + {c} >= 0 && bx + {c} < {w};",
                     f"  let i{nb(r, c)} = select(0u, u32(by + {r}) * {w}u + u32(bx + {c}), v{nb(r, c)});"]
    body.append(f"  let co = id.z * {oct}u;")
    body += [f"  var {x}{p}_{j} = vec4f(0.0);" for p, j in each for x in mats]
    if tiled:
        body += [f"  for (var c0 = 0u; c0 < {cin // 2}u; c0 += {CH}u) {{",
                 "  workgroupBarrier();",
                 f"  for (var e = li; e < {CH * RY * RX}u; e += {WG * WG}u) {{",
                 f"    let y = y0 + i32((e % {RY * RX}u) / {RX}u); let x = x0 + i32(e % {RX}u);",
                 f"    F[e] = select(0u, X[(c0 + e / {RY * RX}u) * {h * w}u + u32(y) * {w}u + u32(x)],",
                 f"                  y >= 0 && y < {h} && x >= 0 && x < {w});",
                 "  }",
                 "  workgroupBarrier();",
                 f"  for (var c2 = c0; c2 < c0 + {CH}u; c2++) {{",
                 f"    let base = (c2 - c0) * {RY * RX}u;"]
    else:
        body += [f"  for (var c2 = 0u; c2 < {cin // 2}u; c2++) {{",
                 f"    let base = c2 * {h * w}u;"]
    body.append(f"    let wq = c2 * {9 * cout // 4}u + co / 4u;" if f32w else
                f"    let we = c2 * {18 * cout}u + co; let wodd = we + {9 * cout}u;")
    sv = ""
    if scale:
        body.append(f"    let sv = {scale};")
        sv = " * sv"
    for r, c in grid:
        if tiled:
            dy = f"+ {r}" if r >= 0 else f"- {-r}u"
            dx = f"+ {c}" if c >= 0 else f"- {-c}u"
            read = f"unpack2x16float(F[base + (ly {dy}) * {RX}u + lx {dx}])"
        else:
            read = f"select(vec2f(0.0), unpack2x16float(X[base + i{nb(r, c)}]), v{nb(r, c)})"
        body.append(f"    let {nb(r, c)} = {read}{sv};")
    names = list(mats)
    for t in range(9):
        ky, kx = divmod(t, 3)
        for j in range(nv):
            if f32w:
                e = f"{2 * len(names)}u * (wq + {t * cout // 4 + j}u)"
                loads = " ".join(f"let {x}e = W[{e} + {2 * i}u]; let {x}o = W[{e} + {2 * i + 1}u];"
                                 for i, x in enumerate(names))
            else:
                o = f"{t * cout + 4 * j}u"
                loads = " ".join(f"let {x}e = w4({mats[x]}u + we + {o}); let {x}o = w4({mats[x]}u + wodd + {o});"
                                 for x in names)
            body.append(f"    {{ {loads}")
            for p, (dy, dx) in enumerate(pix):
                n = nb(dy + ky - 1, dx + kx - 1)
                body.append("      " + " ".join(f"{x}{p}_{j} += {n}.x * {x}e + {n}.y * {x}o;" for x in names))
            body.append("    }")
    body.append("  }")
    if tiled:
        body += ["  }", f"  if (oy >= {h}u || ox >= {w}u) {{ return; }}"]
    for p, j in each:
        dy, dx = pix[p]
        body.append(f"  {{ let px = (oy + {dy}u) * {w}u + ox + {dx}u; let c = co + {4 * j}u;")
        body.append(finish({x: f"{x}{p}_{j}" for x in names}) + " }")
    if f32w:
        entries = [*entries, ("W", "array<vec4f>", f32_weights(b, step, mats, cin, cout))]
    shared = f"var<workgroup> F: array<u32, {CH * RY * RX}>;" if tiled else ""
    decl, bufs = bindings(entries)
    code = wgsl(decl + """
${HELPERS}
${helpers}
${shared}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(workgroup_id) wg: vec3u,
        @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
${body}
}""", WG=WG, body="\n".join(body), shared=shared, helpers=helpers)
    b.step(step, code, bufs, [math.ceil(w / BX / WG), math.ceil(h / BY / WG), cout // oct])


def f32_weights(b: Builder, key: str, mats: dict[str, int], cin: int, cout: int) -> str:
    """A 3x3 conv's weight sets made f32 at load, laid out [channel pair][tap][4 output
    channels] as, for each set, a vec4f for the even and one for the odd input channel: a
    thread reads them without converting. Faster on some backends (Vulkan on an Arc A770),
    slower on others (its D3D12)."""
    name = f"f32.{key}"
    q4 = cout // 4
    n = cin // 2 * 9 * q4
    k = 2 * len(mats)
    b.buffer(name, n * k * 16)
    groups, index = linear(n)
    stores = "\n".join(f"  Y[{k}u * i + {2 * a}u] = w4({off}u + e); Y[{k}u * i + {2 * a + 1}u] = w4({off}u + e + {9 * cout}u);"
                       for a, off in enumerate(mats.values()))
    b.step(name, wgsl(storage([("P", "array<u32>"), ("Y", "array<vec4f>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = ${index};
  if (i >= ${n}u) { return; }
  let e = (i / ${q4}u / 9u) * ${even}u + (i / ${q4}u % 9u) * ${cout}u + (i % ${q4}u) * 4u;
${stores}
}""", index=index, n=n, q4=q4, even=18 * cout, cout=cout, stores=stores), ["P", name], groups, load=True)
    return name
