"""Compile a converted model into a program: WebGPU compute shaders, the buffers they use, and
the dispatches of one frame. Every host runs the same program (`runner.py` through wgpu-py,
`runner.mjs` in a browser), so the shaders exist once, here.

A model is a manifest (ops in run order, tensor shapes, offsets into one fp16 weight blob)
and the blob. Activations are fp16 channel pairs packed in u32 (channel 2i in the low half),
so no shader needs the shader-f16 feature; every sum accumulates in f32.

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

from ganlive.engine.codegen import FORMAT, HELPERS, WG, Builder, linear, split, storage, wgsl
from ganlive.engine.direct import conv3x3
from ganlive.engine.gemm import matmul
from ganlive.engine.noise import NOISE, noise_seed

GEMM_MAX = 6144             # maps up to this many pixels run as a matrix product


def compile_program(manifest: dict, plans: dict | None = None, output: str = "rgba8") -> dict:
    """The program for `manifest`, drawing packed RGBA8 pixels for display or, with
    `output="f32"`, float planes in [-1, 1] for checks. `plans` overrides how conv layers
    split, by name: {"gemm": True} or {"by": 2, "bx": 4, "oct": 4}."""
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
        if plans[op["out"]].get("gemm"):        # one slice: value and gate for every pixel
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
    memory). The defaults were measured on an Arc A770; `plans` overrides them by layer."""
    cout, ho, wo = b.shape(op["out"])
    cin = b.shape(op["in"])[0]
    up = op["up"]
    rows = ho * wo // 4 if up else ho * wo
    steps = cin * (4 if up else 9) // 16          # a matrix product's sum, in steps of 16
    gemm_fits = cout % 32 == 0 and cin % 16 == 0

    forced = b.overrides.get(op["out"])
    if (forced or {}).get("gemm") or (not forced and rows <= GEMM_MAX and gemm_fits):
        forced = forced or {}
        rm, rn = forced.get("rm", 4), forced.get("rn", 1)
        if not gemm_fits or cout % (32 * rn) or rm % 4:
            raise ValueError(f"{op['out']}: {cin}->{cout} channels do not tile as a matrix "
                             f"product of {8 * rm}x{32 * rn}")
        tiles = math.ceil(rows / (8 * rm)) * (cout // (32 * rn)) * (4 if up else 1)
        S = forced.get("S") or split(tiles, steps)
        if steps % S:
            raise ValueError(f"{op['out']}: a sum of {steps} steps does not split {S} ways")
        return {"gemm": True, "rm": rm, "rn": rn, "S": S}
    plan = dict(forced or {"by": 1, "bx": 2, "oct": 4} if up else forced or {"by": 2, "bx": 2, "oct": 4})
    known = {"by", "bx", "oct"} | (set() if up else {"f32", "slm"})
    if (not set(plan) <= known or plan["oct"] % 4 or cout % plan["oct"]
            or (not up and (ho % plan["by"] or wo % plan["bx"]))):
        raise ValueError(f"{op['out']}: {plan} does not fit {cout} channels at {ho}x{wo}")
    return plan


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


def _sle(b: Builder, op: dict, tensors: dict) -> None:
    cl, h, w = b.shape(op["low"])
    ch, slot, n, name = op["ch"], op["slot"], cl * 16, op["name"]
    pooled = b.buffer(f"{name}.pooled", n * 4)
    scale = b.buffer(f"{name}.scale", ch * 4)
    b.step(f"{name}.pool", wgsl(storage([("P", "array<u32>"), ("X", "array<u32>"), ("Y", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${n}u) { return; }
  let c = id.x / 16u; let i = (id.x % 16u) / 4u; let j = id.x % 4u;
  var s = 0.0;
  for (var y = 0u; y < ${h}u; y++) {
    let r = w1(${rows}u + i * ${h}u + y);
    if (r == 0.0) { continue; }
    for (var x = 0u; x < ${w}u; x++) {
      let v = unpack2x16float(X[(c / 2u) * ${hw}u + y * ${w}u + x])[c & 1u];
      s += r * v * w1(${cols}u + x * 4u + j);
    }
  }
  Y[id.x] = s;
}""", n=n, h=h, w=w, hw=h * w, rows=op["rows"], cols=op["cols"]),
           ["P", tensors[op["low"]], pooled], [math.ceil(n / 64), 1, 1])
    parts = math.ceil(n / 256)
    hidden = b.buffer(f"{name}.hidden", parts * ch * 4)
    b.step(f"{name}.fc1", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  // Thread = output channel, so neighbouring threads read neighbouring weights; workgroup row
  // y sums one 256-term part of the ${n}, which fc2 adds up.
  if (id.x >= ${ch}u) { return; }
  var s = 0.0;
  let end = min(${n}u, (id.y + 1u) * 256u);
  for (var i = id.y * 256u; i < end; i++) { s += w1(${fc1}u + i * ${ch}u + id.x) * X[i]; }
  Y[id.y * ${ch}u + id.x] = s;
}""", n=n, fc1=op["fc1"], ch=ch), ["P", pooled, hidden], [math.ceil(ch / 64), parts, 1])
    silu = b.buffer(f"{name}.silu", ch * 4)
    b.step(f"{name}.silu", wgsl(storage([("X", "array<f32>"), ("Y", "array<f32>")]) + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${ch}u) { return; }
  var t = 0.0;
  for (var q = 0u; q < ${parts}u; q++) { t += X[q * ${ch}u + id.x]; }
  Y[id.x] = t / (1.0 + exp(-t));
}""", ch=ch, parts=parts), [hidden, silu], [math.ceil(ch / 64), 1, 1])
    b.step(f"{name}.fc2", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>"),
                                        ("K", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${ch}u) { return; }
  var s = 0.0;
  for (var i = 0u; i < ${ch}u; i++) { s += w1(${fc2}u + i * ${ch}u + id.x) * X[i]; }
  Y[id.x] = 1.0 + K[${slot}u] * (sigmoid(s) - 1.0);
}""", ch=ch, fc2=op["fc2"], slot=slot), ["P", silu, scale, "K"], [math.ceil(ch / 64), 1, 1])


def _bindings(op: dict, tensors: dict, source: str, kind: str = "array<u32>", extra=()):
    """A conv's binding declarations and buffers (see `_entries`)."""
    entries = _entries(op, tensors, source, kind) + list(extra)
    return storage([(n, k) for n, k, _ in entries]), [buf for *_, buf in entries]


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
    of the four output parities, 4 taps a channel, its weights the combined f32 ones. A
    second pass adds the slices and runs the epilogue."""
    cout, ho, wo = b.shape(op["out"])
    cin, hi, wi = b.shape(op["in"])
    parity = op["up"]
    M, taps = (hi * wi, 4) if parity else (ho * wo, 9)
    gw, gh = (wi, hi) if parity else (wo, ho)
    offsets = ("let dy = i32(par >> 1u) - 1 + i32(t >> 1u); let dx = i32(par & 1u) - 1 + i32(t & 1u);"
               if parity else "let dy = i32(t / 3u) - 1; let dx = i32(t % 3u) - 1;")
    # The sum runs tap-major: k = tap * cin + channel, so k and k + 1 share an input read.
    gather = (f"      let ci = k % {cin}u; let t = k / {cin}u; {offsets}\n"
              f"      let base = (ci / 2u) * {hi * wi}u;\n" + "\n".join(
                  f"      {{ let m = m4 + {i}u; let ry = i32(m / {gw}u) + dy; let rx = i32(m % {gw}u) + dx;\n"
                  f"        if (m < {M}u && ry >= 0 && ry < {gh} && rx >= 0 && rx < {gw}) {{\n"
                  f"          let x = unpack2x16float(X[base + u32(ry) * {wi}u + u32(rx)]); "
                  f"v0[{i}] = x.x; v1[{i}] = x.y; }} }}"
                  for i in range(4)))
    if parity:                  # reads the combined weights, not P
        bload = (f"      let ci = k % {cin}u; let at = 4u * (((par * {cin // 2}u + ci / 2u) * 4u + k / {cin}u) * "
                 f"{cout // 4}u + nn / 4u) + (ci & 1u); Bv[e] = W[at]; Bg[e] = W[at + 2u];")
        decl = storage([("X", "array<u32>"), ("Y", "array<vec4f>"), ("W", "array<vec4f>")])
        bind = [tensors[op["in"]], "scratch", _parity_weights(b, op, cin, cout)]
    else:
        bload = (f"      let w = ((k % {cin}u) * 9u + k / {cin}u) * {cout}u + nn; "
                 f"Bv[e] = w4({op['wv']}u + w); Bg[e] = w4({op['wg']}u + w);")
        decl = storage([("P", "array<u32>"), ("X", "array<u32>"), ("Y", "array<vec4f>")]) + "\n" + HELPERS
        bind = ["P", tensors[op["in"]], "scratch"]
    S = plan["S"]
    if S == 1:                  # one slice: the product finishes the layer itself
        entries = _entries(op, tensors, tensors[op["in"]])
        if parity:
            entries.append(("W", "array<vec4f>", bind[-1]))
        at = (f"let px = (2u * (m / {wi}u) + par / 2u) * {wo}u + 2u * (m % {wi}u) + par % 2u;"
              if parity else "let px = m;")
        code, groups = matmul(
            M=M, N=cout, K=cin * taps, S=1, rm=plan["rm"], rn=plan["rn"], mats=("v", "g"),
            decl=storage([(n, k) for n, k, _ in entries]) + "\n" + HELPERS, gather=gather,
            bload=bload, parts=4 if parity else 1,
            finish=lambda acc: f"    {at}\n" + _epilogue(op, acc["v"], acc["g"], ho * wo, "    "))
        b.step(op["out"], code, [buf for *_, buf in entries], groups, plan)
        return
    code, groups = matmul(M=M, N=cout, K=cin * taps, S=S, rm=plan["rm"], rn=plan["rn"],
                          mats=("v", "g"), decl=decl, gather=gather, bload=bload,
                          parts=4 if parity else 1)
    b.step(f"{op['out']}.mm", code, bind, groups)

    decl, bind = _bindings(op, tensors, "scratch", "array<vec4f>")
    n = ho * wo * (cout // 4)
    groups, index = linear(n)
    if parity:
        where = (f"let par = (px / {wo}u % 2u) * 2u + px % 2u; "
                 f"let m = (px / {wo}u / 2u) * {wi}u + (px % {wo}u) / 2u; let first = par * {S}u;")
    else:
        where = "let m = px; let first = 0u;"
    code = wgsl(decl + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = ${index};
  if (i >= ${n}u) { return; }
  let px = i % ${plane}u; let c = (i / ${plane}u) * 4u;
  ${where}
  var sv = vec4f(0.0); var sg = vec4f(0.0);
  for (var s = first; s < first + ${S}u; s++) {
    let row = (s * ${M}u + m) * ${c2}u + c / 4u;
    sv += X[row]; sg += X[row + ${c4}u];
  }
${epilogue}
}""", n=n, index=index, M=M, S=S, c2=2 * cout // 4, c4=cout // 4, plane=ho * wo, where=where,
                epilogue=_epilogue(op, "sv", "sg", ho * wo, "  "))
    b.step(op["out"], code, bind, groups, plan)


def _conv(b: Builder, op: dict, tensors: dict, plan: dict) -> None:
    """A plain conv at one size, directly (`direct.conv3x3`): value and gate together."""
    cout, ho, wo = b.shape(op["out"])
    conv3x3(b, step=op["out"], cin=b.shape(op["in"])[0], cout=cout, h=ho, w=wo, plan=plan,
            mats={"v": op["wv"], "g": op["wg"]}, entries=_entries(op, tensors, tensors[op["in"]]),
            finish=lambda acc: _epilogue(op, acc["v"], acc["g"], ho * wo, "    "))


def _parity_weights(b: Builder, op: dict, cin: int, cout: int) -> str:
    """The weights of a nearest-2x-upsampling 3x3 conv as four 2x2 convs on its input, one
    per output parity, made at load in f32: output (2a+i, 2b+j) reads input rows a+i-1 and
    a+i, each through the sum of the 3x3 taps that land on it. Laid out [parity][channel
    pair][tap][4 output channels] as four vec4f: value even, value odd, gate even, gate odd."""
    name = f"parity.{op['out']}"
    q4 = cout // 4
    n = 4 * (cin // 2) * 4 * q4
    b.buffer(name, n * 64)
    groups, index = linear(n)
    b.step(name, wgsl(storage([("P", "array<u32>"), ("Y", "array<vec4f>")]) + """
${HELPERS}
fn sum(base: u32, i: i32, j: i32, ry: i32, rx: i32) -> vec4f {
  var s = vec4f(0.0);
  for (var ky = 0; ky < 3; ky++) {
    if (((i + ky - 1) >> 1u) + 1 - i != ry) { continue; }
    for (var kx = 0; kx < 3; kx++) {
      if (((j + kx - 1) >> 1u) + 1 - j != rx) { continue; }
      s += w4(base + u32(ky * 3 + kx) * ${cout}u);
    }
  }
  return s;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = ${index};
  if (n >= ${n}u) { return; }
  let co4 = n % ${q4}u; let q = (n / ${q4}u) % 4u; let c2 = (n / ${q16}u) % ${c2s}u;
  let p = n / ${pq}u;
  let i = i32(p >> 1u); let j = i32(p & 1u); let ry = i32(q >> 1u); let rx = i32(q & 1u);
  let e = c2 * ${even}u + co4 * 4u;
  Y[4u * n] = sum(${wv}u + e, i, j, ry, rx);
  Y[4u * n + 1u] = sum(${wv}u + e + ${odd}u, i, j, ry, rx);
  Y[4u * n + 2u] = sum(${wg}u + e, i, j, ry, rx);
  Y[4u * n + 3u] = sum(${wg}u + e + ${odd}u, i, j, ry, rx);
}""", index=index, n=n, q4=q4, q16=4 * q4, c2s=cin // 2, pq=(cin // 2) * 4 * q4,
                         even=18 * cout, odd=9 * cout, cout=cout, wv=op["wv"], wg=op["wg"]),
           ["P", name], groups, load=True)
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
    decl, bind = _bindings(op, tensors, tensors[op["in"]],
                           extra=[("W", "array<vec4f>", _parity_weights(b, op, cin, cout))])
    code = wgsl(decl + """
${HELPERS}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
${body}
}""", WG=WG, body="\n".join(body))
    b.step(op["out"], code, bind,
           [math.ceil(wi / BX / WG), math.ceil(hi / BY / WG), cout // oct], plan)


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
        store = "Y[px] = pack4x8unorm(vec4f(s * 0.5 + 0.5, 1.0));"
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
