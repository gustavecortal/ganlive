"""Compile a converted model into a program: WebGPU compute shaders, the buffers they use, and
the dispatches of one frame. Every host runs the same program (`runner.py` through wgpu-py,
`runner.mjs` in a browser), so the shaders exist once, here.

A model is a manifest (ops in run order, tensor shapes, offsets into one fp16 weight blob)
and the blob. Activations are fp16 channel pairs packed in u32 (channel 2i in the low half),
so no shader needs the shader-f16 feature, and every sum accumulates in f32.

FastGAN ops, one shader each, generated for their exact shapes so that loops unroll:
  init  z -> dense -> BatchNorm -> GLU, the 4x6 base map
  conv  [nearest x2 upsample] -> 3x3 value and gate convolutions -> bias -> noise -> GLU
        -> [SLE channel scale]. Small maps run as a tiled matrix product split across
        workgroups, larger ones as a direct convolution of a few pixels per thread.
  sle   4x4 adaptive pool -> 4x4 conv -> SiLU -> 1x1 conv -> sigmoid, blended by its dial
  rgb   3x3 convolution to RGB -> tanh, as packed RGBA8 for display or f32 planes for checks
"""

from __future__ import annotations

import math

from ganlive.engine import direct
from ganlive.engine.codegen import FORMAT, HELPERS, WG, Builder, bindings, linear, storage, wgsl
from ganlive.engine.direct import conv3x3
from ganlive.engine.gemm import GEMM_MAX, matmul, split, sum_slices
from ganlive.engine.noise import NOISE, noise_seed


def compile_program(manifest: dict, plans: dict | None = None, output: str = "rgba8") -> dict:
    """The program for `manifest`, drawing packed RGBA8 pixels for a browser canvas, BGRA8
    (`output="bgra8"`) for the desktop app, or float planes in [-1, 1] (`"f32"`) for checks.
    `plans` overrides how conv layers run, by name (see `_plan`)."""
    b = Builder(manifest, plans or {})
    m = manifest
    b.buffer("P", m["bytes"], init="weights")
    b.buffer("Z", m["nz"] * 4)
    b.buffer("K", len(m["settings"]) * 4, init="ones")
    convs = [op for op in m["ops"] if op["op"] == "conv"]
    for op in convs:
        if op.get("noise"):
            _noise(b, op["noise"], *b.shape(op["out"])[1:])

    plans = {op["out"]: _plan(b, op) for op in convs}
    scratch = 0
    for op in convs:
        if plans[op["out"]].get("S", 1) > 1:    # one slice: value and gate for every pixel
            cout, ho, wo = b.shape(op["out"])
            scratch = max(scratch, plans[op["out"]]["S"] * ho * wo * 2 * cout * 4)
    b.buffer("scratch", scratch)
    tensors = _allocate(b)

    for op in m["ops"]:
        kind = op["op"]
        if kind == "init":
            _init(b, op, tensors)
        elif kind == "sle":
            _sle(b, op, tensors)
        elif kind == "conv":
            plan = plans[op["out"]]
            (_gemm if plan.get("gemm") else _upconv if op["up"] else _conv)(b, op, tensors, plan)
        elif kind == "rgb":
            _rgb(b, op, tensors, output)
        else:
            raise ValueError(f"unknown op {kind}")
    if "out" not in b.buffers:
        raise ValueError("the manifest has no rgb op, so the program would draw nothing")

    return {"format": FORMAT, "nz": m["nz"], "height": m["height"], "width": m["width"],
            "settings": m["settings"], "output": output, "shaders": b.shaders,
            "buffers": b.buffers, "load": b.load, "steps": b.steps}


# -- planning -------------------------------------------------------------------------------

def _plan(b: Builder, op: dict) -> dict:
    """How a conv layer runs. An upsampling layer always runs as four 2x2 convs on its input,
    one per output parity (`_parity_weights`). A layer whose input (for upsampling) or output
    has at most GEMM_MAX pixels runs as a matrix product: {"gemm": True, "rm", "rn", "S"};
    a larger one directly, each thread a tile of pixels: {"by", "bx", "oct"}, and for a plain
    conv optionally "f32" (weights made f32 at load) and "slm" (inputs through workgroup
    memory). The defaults were measured on an Arc A770, and `plans` overrides them by layer."""
    cout, ho, wo = b.shape(op["out"])
    cin = b.shape(op["in"])[0]
    up = op["up"]
    rows = ho * wo // 4 if up else ho * wo
    steps = cin * (4 if up else 9) // 16          # a matrix product's sum, in steps of 16
    gemm_fits = cout % 32 == 0 and cin % 16 == 0

    forced = b.overrides.get(op["out"])
    if (forced or {}).get("gemm") or (not forced and rows <= GEMM_MAX and gemm_fits):
        forced = forced or {}               # `matmul` refuses what does not tile
        rm, rn = forced.get("rm", 4), forced.get("rn", 1)
        tiles = math.ceil(rows / (8 * rm)) * (cout // (32 * rn)) * (4 if up else 1)
        return {"gemm": True, "rm": rm, "rn": rn, "S": forced.get("S") or split(tiles, steps)}
    if up:                                  # a thread tiles input pixels
        return direct.check(dict(forced or {"by": 1, "bx": 2, "oct": 4}), op["out"], cout,
                            ho // 2, wo // 2, keys=("by", "bx", "oct"))
    return direct.check(dict(forced or {"by": 2, "bx": 2, "oct": 4}), op["out"], cout, ho, wo)


def plan_choices(manifest: dict) -> dict[str, list[dict]]:
    """Each conv's plans worth trying on a new machine (see `_plan`; `tune` keeps those that fit
    and help): as a matrix product of 4 or 8 rows a thread and various splits where the layer
    tiles and is small enough, and directly at the tiles that do not spill registers."""
    out = {}
    for op in manifest["ops"]:
        if op["op"] != "conv":
            continue
        cout, ho, wo = manifest["tensors"][op["out"]]
        cin = manifest["tensors"][op["in"]][0]
        rows = ho * wo // 4 if op["up"] else ho * wo
        options = []
        if cout % 32 == 0 and cin % 16 == 0 and rows <= 4 * GEMM_MAX:
            options += [{"gemm": True, "rm": rm, "rn": 1, **({"S": S} if S else {})}
                        for rm in (4, 8) for S in (None, 1, 2, 4, 8, 16)]
        if op["up"]:
            options += [{"by": by, "bx": bx, "oct": oct} for by, bx, oct in
                        ((1, 2, 4), (1, 4, 4), (2, 2, 4), (1, 2, 8), (2, 1, 4), (1, 1, 8))]
        else:
            options += direct.CHOICES
        out[op["out"]] = options
    return out


def _allocate(b: Builder) -> dict[str, str]:
    """Tensor -> buffer name. One buffer per tensor would hold every activation at once (255 MiB
    for lichen); instead a tensor takes a free buffer once its last reader has run, best fit."""
    ops = b.m["ops"]
    last = {}
    for i, op in enumerate(ops):
        for t in (op.get("in"), op.get("low")):
            if t:
                last[t] = i
    size = {n: c // 2 * h * w * 4 for n, (c, h, w) in b.m["tensors"].items()}
    free: list[str] = []
    placed: dict[str, str] = {}
    for i, op in enumerate(ops):
        out = op.get("out")
        if out and out not in placed:
            fits = sorted((n for n in free if b.buffers[n]["size"] >= size[out]),
                          key=lambda n: b.buffers[n]["size"])
            if fits:
                placed[out] = fits[0]
                free.remove(fits[0])
            else:
                placed[out] = b.buffer(f"t{len(placed)}", size[out])
        free += [placed[t] for t, end in last.items() if end == i]
    return placed


# -- ops ------------------------------------------------------------------------------------

def _init(b: Builder, op: dict, tensors: dict) -> None:
    ch, h, w = b.shape(op["out"])
    cells, nz = h * w, b.m["nz"]
    n = ch // 2 * cells
    code = wgsl(storage([("P", "array<u32>"), ("Z", "array<vec4f>"), ("Y", "array<u32>")]) + """
${HELPERS}
fn row(r: u32) -> f32 {
  var s = vec4f(0.0);
  for (var i = 0u; i < ${nz4}u; i++) { s += w4(${w}u + r * ${nz}u + 4u * i) * Z[i]; }
  return s.x + s.y + s.z + s.w;
}
fn glu(c: u32, cell: u32) -> f32 {
  let v = row(c * ${cells}u + cell) + w1(${bias}u + c);
  let g = row((c + ${ch}u) * ${cells}u + cell) + w1(${bias}u + c + ${ch}u);
  return v * sigmoid(g);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${n}u) { return; }
  let pair = id.x / ${cells}u; let cell = id.x % ${cells}u;
  Y[id.x] = pack2x16float(vec2f(glu(2u * pair, cell), glu(2u * pair + 1u, cell)));
}""", nz4=nz // 4, nz=nz, w=op["w"], cells=cells, bias=op["bias"], ch=ch, n=n)
    b.step("init", code, ["P", "Z", tensors[op["out"]]], [math.ceil(n / 64), 1, 1])


def _params(b: Builder, name: str, *words: int) -> str:
    """A small buffer of a step's sizes and offsets, which a shader shared between layers reads
    as `U`: one compile serves every layer (a browser compiles each shader in turn)."""
    return b.buffer(name, 4 * len(words), init="words", words=[int(w) for w in words])


# The SLE gate's four small steps, one shader each for every gate. U: see `_sle`.
SLE_POOL = wgsl(storage([("P", "array<u32>"), ("X", "array<u32>"), ("Y", "array<f32>"),
                         ("U", "array<u32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = U[0]; let h = U[1]; let w = U[2]; let rows = U[3]; let cols = U[4];
  if (id.x >= n) { return; }
  let c = id.x / 16u; let i = (id.x % 16u) / 4u; let j = id.x % 4u;
  var s = 0.0;
  for (var y = 0u; y < h; y++) {
    let r = w1(rows + i * h + y);
    if (r == 0.0) { continue; }
    for (var x = 0u; x < w; x++) {
      let v = unpack2x16float(X[(c / 2u) * h * w + y * w + x])[c & 1u];
      s += r * v * w1(cols + x * 4u + j);
    }
  }
  Y[id.x] = s;
}""")
SLE_FC1 = wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>"),
                        ("U", "array<u32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  // Thread = output channel, so neighbouring threads read neighbouring weights. Workgroup
  // row y sums one 256-term part of the n inputs, which the SiLU step adds up.
  let n = U[0]; let ch = U[1]; let fc1 = U[2];
  if (id.x >= ch) { return; }
  var s = 0.0;
  let end = min(n, (id.y + 1u) * 256u);
  for (var i = id.y * 256u; i < end; i++) { s += w1(fc1 + i * ch + id.x) * X[i]; }
  Y[id.y * ch + id.x] = s;
}""")
SLE_SILU = wgsl(storage([("X", "array<f32>"), ("Y", "array<f32>"), ("U", "array<u32>")]) + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ch = U[1]; let parts = U[3];
  if (id.x >= ch) { return; }
  var t = 0.0;
  for (var q = 0u; q < parts; q++) { t += X[q * ch + id.x]; }
  Y[id.x] = t / (1.0 + exp(-t));
}""")
SLE_FC2 = wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>"),
                        ("K", "array<f32>"), ("U", "array<u32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ch = U[1]; let fc2 = U[4]; let slot = U[5];
  if (id.x >= ch) { return; }
  var s = 0.0;
  for (var i = 0u; i < ch; i++) { s += w1(fc2 + i * ch + id.x) * X[i]; }
  Y[id.x] = 1.0 + K[slot] * (sigmoid(s) - 1.0);
}""")


def _sle(b: Builder, op: dict, tensors: dict) -> None:
    cl, h, w = b.shape(op["low"])
    ch, slot, n, name = op["ch"], op["slot"], cl * 16, op["name"]
    pooled = b.buffer(f"{name}.pooled", n * 4)
    scale = b.buffer(f"{name}.scale", ch * 4)
    parts = math.ceil(n / 256)
    hidden = b.buffer(f"{name}.hidden", parts * ch * 4)
    silu = b.buffer(f"{name}.silu", ch * 4)
    pool = _params(b, f"{name}.pool.u", n, h, w, op["rows"], op["cols"])
    fc = _params(b, f"{name}.fc.u", n, ch, op["fc1"], parts, op["fc2"], slot)
    b.step(f"{name}.pool", SLE_POOL, ["P", tensors[op["low"]], pooled, pool], [math.ceil(n / 64), 1, 1])
    b.step(f"{name}.fc1", SLE_FC1, ["P", pooled, hidden, fc], [math.ceil(ch / 64), parts, 1])
    b.step(f"{name}.silu", SLE_SILU, [hidden, silu, fc], [math.ceil(ch / 64), 1, 1])
    b.step(f"{name}.fc2", SLE_FC2, ["P", silu, scale, "K", fc], [math.ceil(ch / 64), 1, 1])


def _entries(op: dict, tensors: dict, source: str, kind: str = "array<u32>") -> list:
    """A conv's bindings: P, its input, Y, then what its epilogue reads. "auto" layouts drop
    bindings a shader never reads, so only those are declared."""
    entries = [("P", "array<u32>", "P"), ("X", kind, source), ("Y", "array<u32>", tensors[op["out"]])]
    if op.get("noise"):
        entries.append(("N", "array<f32>", f"noise.{op['noise']}"))
        if op.get("slot") is not None:
            entries.append(("K", "array<f32>", "K"))
    if op.get("scale"):
        entries.append(("S", "array<f32>", f"{op['scale']}.scale"))
    return entries


def _epilogue(op: dict, v: str, g: str, plane: int, indent: str) -> str:
    """Bias, noise, GLU, SLE scale and the packed store of channels c..c+3 at pixel `px`,
    given the value and gate sums: the one place a conv's semantics live."""
    gain = "1.0" if op.get("slot") is None else f"K[{op['slot']}u]"
    lines = [f"var v = {v} + w4({op['bv']}u + c); var g = {g} + w4({op['bg']}u + c);"]
    if op.get("noise"):
        lines.append(f"let nz = N[px] * {gain}; v += w4({op['cv']}u + c) * nz; "
                     f"g += w4({op['cg']}u + c) * nz;")
    lines.append("var o = v * sigmoid4(g);")
    if op.get("scale"):
        lines.append("o *= vec4f(S[c], S[c + 1u], S[c + 2u], S[c + 3u]);")
    lines += [f"Y[(c / 2u) * {plane}u + px] = pack2x16float(o.xy);",
              f"Y[(c / 2u + 1u) * {plane}u + px] = pack2x16float(o.zw);"]
    return "\n".join(indent + line for line in lines)


def _gemm(b: Builder, op: dict, tensors: dict, plan: dict) -> None:
    """[pixels x (cin*taps)] by [(cin*taps) x cout] for value and gate (`gemm.matmul`). An
    upsampling layer's product (see `_parity_weights`) runs over the input's pixels for each
    of the four output parities, 4 taps a channel, its weights the combined f32 ones. Split
    in slices, a second pass adds them and runs the epilogue."""
    cout, ho, wo = b.shape(op["out"])
    cin, hi, wi = b.shape(op["in"])
    parity, S = op["up"], plan["S"]
    if parity:
        M, taps, gw, gh = hi * wi, 4, wi, hi
        offsets = "let dy = i32(par >> 1u) - 1 + i32(t >> 1u); let dx = i32(par & 1u) - 1 + i32(t & 1u);"
        bload = (f"      let ci = k % {cin}u; let at = 4u * (((par * {cin // 2}u + ci / 2u) * 4u + k / {cin}u) * "
                 f"{cout // 4}u + nn / 4u) + (ci & 1u); Bv[e] = W[at]; Bg[e] = W[at + 2u];")
        weights = [("W", "array<vec4f>", _parity_weights(b, op, cin, cout))]
        at = f"let px = (2u * (m / {wi}u) + par / 2u) * {wo}u + 2u * (m % {wi}u) + par % 2u;"
        where = (f"let px = r; let par = (px / {wo}u % 2u) * 2u + px % 2u; "
                 f"let m = (px / {wo}u / 2u) * {wi}u + (px % {wo}u) / 2u; let first = par * {S}u;")
    else:
        M, taps, gw, gh = ho * wo, 9, wo, ho
        offsets = "let dy = i32(t / 3u) - 1; let dx = i32(t % 3u) - 1;"
        bload = (f"      let w = ((k % {cin}u) * 9u + k / {cin}u) * {cout}u + nn; "
                 f"Bv[e] = w4({op['wv']}u + w); Bg[e] = w4({op['wg']}u + w);")
        weights = [("P", "array<u32>", "P")]
        at, where = "let px = m;", "let px = r; let m = r; let first = 0u;"
    # The sum runs tap-major: k = tap * cin + channel, so k and k + 1 share an input read.
    gather = (f"      let ci = k % {cin}u; let t = k / {cin}u; {offsets}\n"
              f"      let base = (ci / 2u) * {hi * wi}u;\n" + "\n".join(
                  f"      {{ let m = m4 + {i}u; let ry = i32(m / {gw}u) + dy; let rx = i32(m % {gw}u) + dx;\n"
                  f"        if (m < {M}u && ry >= 0 && ry < {gh} && rx >= 0 && rx < {gw}) {{\n"
                  f"          let x = unpack2x16float(X[base + u32(ry) * {wi}u + u32(rx)]); "
                  f"v0[{i}] = x.x; v1[{i}] = x.y; }} }}"
                  for i in range(4)))
    product = dict(M=M, N=cout, K=cin * taps, S=S, rm=plan["rm"], rn=plan["rn"], mats=("v", "g"),
                   gather=gather, bload=bload, parts=4 if parity else 1)
    if S == 1:                  # one slice: the product finishes the layer itself
        decl, bufs = bindings(_entries(op, tensors, tensors[op["in"]]) + weights[parity - 1:])
        code, groups = matmul(**product, decl=decl + HELPERS, finish=lambda acc: (
            f"    {at}\n" + _epilogue(op, acc["v"], acc["g"], ho * wo, "    ")))
        b.step(op["out"], code, bufs, groups)
        return
    decl, bufs = bindings([("X", "array<u32>", tensors[op["in"]]), ("Y", "array<vec4f>", "scratch"), *weights])
    code, groups = matmul(**product, decl=decl + ("" if parity else HELPERS))
    b.step(f"{op['out']}.mm", code, bufs, groups)
    decl, bufs = bindings(_entries(op, tensors, "scratch", "array<vec4f>"))
    code, groups = sum_slices(rows=ho * wo, M=M, N=cout, S=S, mats=("v", "g"), decl=decl + HELPERS,
                              where=where, finish=lambda acc: _epilogue(op, acc["v"], acc["g"], ho * wo, "  "))
    b.step(op["out"], code, bufs, groups)


def _conv(b: Builder, op: dict, tensors: dict, plan: dict) -> None:
    """A plain conv at one size, directly (`direct.conv3x3`): value and gate together."""
    cout, ho, wo = b.shape(op["out"])
    conv3x3(b, step=op["out"], cin=b.shape(op["in"])[0], cout=cout, h=ho, w=wo, plan=plan,
            mats={"v": op["wv"], "g": op["wg"]}, entries=_entries(op, tensors, tensors[op["in"]]),
            finish=lambda acc: _epilogue(op, acc["v"], acc["g"], ho * wo, "    "))


# A nearest-2x-upsampling 3x3 conv's weights as four 2x2 convs, made at load (see
# `_parity_weights`). U: n, cout / 4, cin / 2, the value and gate offsets in P.
PARITY = wgsl(storage([("P", "array<u32>"), ("Y", "array<vec4f>"), ("U", "array<u32>")]) + """
${HELPERS}
fn sum(base: u32, cout: u32, i: i32, j: i32, ry: i32, rx: i32) -> vec4f {
  var s = vec4f(0.0);
  for (var ky = 0; ky < 3; ky++) {
    if (((i + ky - 1) >> 1u) + 1 - i != ry) { continue; }
    for (var kx = 0; kx < 3; kx++) {
      if (((j + kx - 1) >> 1u) + 1 - j != rx) { continue; }
      s += w4(base + u32(ky * 3 + kx) * cout);
    }
  }
  return s;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = ${index};
  let total = U[0]; let q4 = U[1]; let c2s = U[2]; let wv = U[3]; let wg = U[4];
  if (n >= total) { return; }
  let cout = 4u * q4;
  let co4 = n % q4; let q = (n / q4) % 4u; let c2 = (n / (4u * q4)) % c2s;
  let p = n / (c2s * 4u * q4);
  let i = i32(p >> 1u); let j = i32(p & 1u); let ry = i32(q >> 1u); let rx = i32(q & 1u);
  let e = c2 * 18u * cout + co4 * 4u;
  Y[4u * n] = sum(wv + e, cout, i, j, ry, rx);
  Y[4u * n + 1u] = sum(wv + e + 9u * cout, cout, i, j, ry, rx);
  Y[4u * n + 2u] = sum(wg + e, cout, i, j, ry, rx);
  Y[4u * n + 3u] = sum(wg + e + 9u * cout, cout, i, j, ry, rx);
}""", index=linear(1)[1])


def _parity_weights(b: Builder, op: dict, cin: int, cout: int) -> str:
    """The weights of a nearest-2x-upsampling 3x3 conv as four 2x2 convs on its input, one
    per output parity, made at load in f32: output (2a+i, 2b+j) reads input rows a+i-1 and
    a+i, each through the sum of the 3x3 taps that land on it. Laid out [parity][channel
    pair][tap][4 output channels] as four vec4f: value even, value odd, gate even, gate odd."""
    name = f"parity.{op['out']}"
    n = 4 * (cin // 2) * 4 * (cout // 4)
    b.buffer(name, n * 64)
    words = _params(b, f"{name}.u", n, cout // 4, cin // 2, op["wv"], op["wg"])
    b.step(name, PARITY, ["P", name, words], linear(n)[0], load=True)
    return name


def _upconv(b: Builder, op: dict, tensors: dict, plan: dict) -> None:
    """A nearest-2x-upsampling conv as four 2x2 convs on its input (`_parity_weights`): 4
    taps an output instead of 9. Each thread holds BY x BX input positions and `oct` channels
    and computes both column parities at once, one row parity after the other, so that it
    writes whole runs of 2BX pixels: a 2BY x 2BX block in all."""
    cout, ho, wo = b.shape(op["out"])
    cin, hi, wi = b.shape(op["in"])
    BY, BX, oct = plan["by"], plan["bx"], plan["oct"]
    nv, q4 = oct // 4, cout // 4
    pix = [(u, v, j) for u in range(BY) for v in range(BX) for j in range(2)]
    grid = [(r, c) for r in range(BY + 1) for c in range(BX + 2)]
    each = [(k, n) for k in range(len(pix)) for n in range(nv)]
    body = [f"  let a0 = id.y * {BY}u; let b0 = id.x * {BX}u;",
            f"  if (a0 >= {hi}u || b0 >= {wi}u) {{ return; }}",
            f"  let co = id.z * {oct}u; let bx = i32(b0) - 1;"]
    for c in range(BX + 2):
        body.append(f"  let vc{c} = bx + {c} >= 0 && bx + {c} < {wi};")
    body.append("  for (var i = 0u; i < 2u; i++) {")
    body.append("    let by = i32(a0 + i) - 1;")
    for r, c in grid:
        body += [f"    let v{r}{c} = vc{c} && by + {r} >= 0 && by + {r} < {hi};",
                 f"    let i{r}{c} = select(0u, u32(by + {r}) * {wi}u + u32(bx + {c}), v{r}{c});"]
    body += [f"    var av{k}_{n} = vec4f(0.0); var ag{k}_{n} = vec4f(0.0);" for k, n in each]
    body += [f"    for (var c2 = 0u; c2 < {cin // 2}u; c2++) {{",
             f"      let base = c2 * {hi * wi}u;",
             f"      let w0 = ((2u * i * {cin // 2}u + c2) * 4u) * {q4}u + co / 4u;",
             f"      let w1 = w0 + {cin // 2 * 4 * q4}u;"]
    body += [f"      let n{r}{c} = select(vec2f(0.0), unpack2x16float(X[base + i{r}{c}]), v{r}{c});"
             for r, c in grid]
    for j in range(2):
        for q in range(4):
            ry, rx = divmod(q, 2)
            for n in range(nv):
                e = f"4u * (w{j} + {q * q4 + n}u)"
                body.append(f"      {{ let a = W[{e}]; let b = W[{e} + 1u]; let c = W[{e} + 2u]; let d = W[{e} + 3u];")
                for k, (u, v, jj) in enumerate(pix):
                    if jj != j:
                        continue
                    x = f"n{u + ry}{v + rx + j}"
                    body.append(f"        av{k}_{n} += {x}.x * a + {x}.y * b; ag{k}_{n} += {x}.x * c + {x}.y * d;")
                body.append("      }")
    body.append("    }")
    for k, n in each:
        u, v, j = pix[k]
        body.append(f"    {{ let px = (2u * (a0 + {u}u) + i) * {wo}u + 2u * (b0 + {v}u) + {j}u; let c = co + {4 * n}u;")
        body.append(_epilogue(op, f"av{k}_{n}", f"ag{k}_{n}", ho * wo, "      ") + " }")
    body.append("  }")
    decl, bind = bindings(_entries(op, tensors, tensors[op["in"]])
                          + [("W", "array<vec4f>", _parity_weights(b, op, cin, cout))])
    code = wgsl(decl + """
${HELPERS}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
${body}
}""", WG=WG, body="\n".join(body))
    b.step(op["out"], code, bind, [math.ceil(wi / BX / WG), math.ceil(hi / BY / WG), cout // oct])


def _rgb(b: Builder, op: dict, tensors: dict, fmt: str) -> None:
    """The last step: packed RGBA8 for display, or f32 planes. A workgroup draws a 16x16
    block, reading the 18x18 block of features it needs into workgroup memory once."""
    cin, h, w = b.shape(op["in"])
    T, R = 16, 18
    acc = []
    for c2 in range(cin // 2):
        for t in range(9):
            e = op["w"] + (2 * c2 * 9 + t) * 3
            o = e + 27
            acc.append(
                f"  {{ let n = unpack2x16float(F[{c2 * R * R}u + (ly + {t // 3}u) * {R}u + lx + {t % 3}u]);\n"
                f"    s += n.x * vec3f(w1({e}u), w1({e + 1}u), w1({e + 2}u))\n"
                f"       + n.y * vec3f(w1({o}u), w1({o + 1}u), w1({o + 2}u)); }}")
    if fmt == "f32":
        kind, size = "f32", 3 * h * w * 4
        store = f"Y[px] = s.x; Y[{h * w}u + px] = s.y; Y[{2 * h * w}u + px] = s.z;"
    else:
        kind, size = "u32", h * w * 4
        rgb = "s.zyx" if fmt == "bgra8" else "s"
        store = f"Y[px] = pack4x8unorm(vec4f({rgb} * 0.5 + 0.5, 1.0));"
    code = wgsl(storage([("P", "array<u32>"), ("X", "array<u32>"), ("Y", f"array<{kind}>")]) + """
${HELPERS}
var<workgroup> F: array<u32, ${tile}>;
@compute @workgroup_size(${T}, ${T}, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let y0 = i32(wg.y * ${T}u) - 1; let x0 = i32(wg.x * ${T}u) - 1;
  for (var e = li; e < ${tile}u; e += ${TT}u) {
    let c2 = e / ${RR}u; let r = (e % ${RR}u) / ${R}u; let c = e % ${R}u;
    let y = y0 + i32(r); let x = x0 + i32(c);
    var v = 0u;
    if (y >= 0 && y < ${h} && x >= 0 && x < ${w}) { v = X[c2 * ${hw}u + u32(y) * ${w}u + u32(x)]; }
    F[e] = v;
  }
  workgroupBarrier();
  let ly = lid.y; let lx = lid.x;
  let oy = wg.y * ${T}u + ly; let ox = wg.x * ${T}u + lx;
  if (oy >= ${h}u || ox >= ${w}u) { return; }
  let px = oy * ${w}u + ox;
  var s = vec3f(0.0);
${acc}
  s = tanh(clamp(s, vec3f(-10.0), vec3f(10.0)));
  ${store}
}""", h=h, w=w, hw=h * w, T=T, R=R, RR=R * R, TT=T * T, tile=cin // 2 * R * R,
                acc="\n".join(acc), store=store)
    b.buffer("out", size)
    b.step("rgb", code, ["P", tensors[op["in"]], "out"], [math.ceil(w / T), math.ceil(h / T), 1])


def _noise(b: Builder, key: str, h: int, w: int) -> None:
    """A noise layer's buffer, filled once at load by `NOISE` with the layer's own seed."""
    n = h * w
    b.buffer(f"noise.{key}", n * 4)
    b.buffer(f"noise.{key}.n", 16, init="words", words=[n, noise_seed(key)])
    b.step(f"noise.{key}", NOISE, [f"noise.{key}", f"noise.{key}.n"], linear(n, 256)[0],
           load=True)
