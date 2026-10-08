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
from string import Template

from ganlive.engine.noise import NOISE, noise_seed

FORMAT = "ganlive-engine/1"

WG = 8                      # direct-convolution workgroups are WG x WG threads
GEMM_MAX = 6144             # maps up to this many pixels run as a matrix product
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
            (_gemm if plan.get("gemm") else _conv)(b, op, tensors, plan)
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
    """How a conv layer is split. Explicit overrides must fit; the defaults were measured on
    an Arc A770: maps up to GEMM_MAX pixels as a matrix product, upsampling layers at 2x4
    pixels and 4 channels a thread, the others at 2x2 and 8. Larger tiles spill registers and
    run 5-10x slower."""
    cout, ho, wo = b.shape(op["out"])
    cin = b.shape(op["in"])[0]
    M = ho * wo
    gemm_fits = cout % 32 == 0 and M % 32 == 0 and cin % 16 == 0

    def tile_fits(t):
        return (t["oct"] % 4 == 0 and cout % t["oct"] == 0 and ho % t["by"] == 0
                and wo % t["bx"] == 0 and (not op["up"] or (t["by"] % 2 == 0 and t["bx"] % 2 == 0)))

    forced = b.overrides.get(op["out"])
    if (forced or {}).get("gemm") or (not forced and M <= GEMM_MAX and gemm_fits):
        if not gemm_fits:
            raise ValueError(f"{op['out']}: {cin}->{cout} channels at {ho}x{wo} do not tile "
                             f"as a matrix product")
        return {"gemm": True, "S": split((M // 32) * (cout // 32), cin * 9 // 16)}
    if forced:
        if not tile_fits(forced):
            raise ValueError(f"{op['out']}: a {forced['by']}x{forced['bx']} tile of "
                             f"{forced['oct']} channels does not fit {cout} at {ho}x{wo}")
        return dict(forced)
    preferred = {"by": 2, "bx": 4, "oct": 4} if op["up"] else {"by": 2, "bx": 2, "oct": 8}
    for t in (preferred, {"by": 2, "bx": 2, "oct": 4}):
        if tile_fits(t):
            return t
    raise ValueError(f"{op['out']}: no tile fits {cout} channels at {ho}x{wo}")


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
    hidden = b.buffer(f"{name}.hidden", ch * 4)
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
    b.step(f"{name}.fc1", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>")]) + """
${HELPERS}
var<workgroup> part: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  // One workgroup per output channel; its 64 threads split the ${n}-term sum.
  var s = 0.0;
  for (var i = li; i < ${n}u; i += 64u) { s += w1(${fc1}u + i * ${ch}u + wg.x) * X[i]; }
  part[li] = s;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (li < half) { part[li] += part[li + half]; }
    workgroupBarrier();
  }
  if (li == 0u) { let t = part[0]; Y[wg.x] = t * sigmoid(t); }
}""", n=n, fc1=op["fc1"], ch=ch), ["P", pooled, hidden], [ch, 1, 1])
    b.step(f"{name}.fc2", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>"),
                                        ("K", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${ch}u) { return; }
  var s = 0.0;
  for (var i = 0u; i < ${ch}u; i++) { s += w1(${fc2}u + i * ${ch}u + id.x) * X[i]; }
  Y[id.x] = 1.0 + K[${slot}u] * (sigmoid(s) - 1.0);
}""", ch=ch, fc2=op["fc2"], slot=slot), ["P", hidden, scale, "K"], [math.ceil(ch / 64), 1, 1])


def _bindings(op: dict, tensors: dict, source: str, kind: str = "array<u32>"):
    """A conv's bindings: P, its input, Y, then what its epilogue reads. "auto" layouts drop
    bindings a shader never reads, so only those are declared."""
    entries = [("P", "array<u32>", "P"), ("X", kind, source), ("Y", "array<u32>", tensors[op["out"]])]
    if op.get("noise"):
        entries.append(("N", "array<f32>", f"noise.{op['noise']}"))
        if op.get("slot") is not None:
            entries.append(("K", "array<f32>", "K"))
    if op.get("scale"):
        entries.append(("S", "array<f32>", f"{op['scale']}.scale"))
    return storage([(n, k) for n, k, _ in entries]), [buf for *_, buf in entries]


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
    """[pixels x (cin*9)] by [(cin*9) x cout] for value and gate, tiled through workgroup
    memory, the sum split into S slices so that even a 12x8 map fills the GPU. A second pass
    adds the slices and runs the epilogue."""
    cout, ho, wo = b.shape(op["out"])
    cin, hi, wi = b.shape(op["in"])
    S = plan["S"]
    M, KS, TM, TN, TK = ho * wo, cin * 9 // S, 32, 32, 16
    up = " >> 1" if op["up"] else ""
    quad = range(4)
    accs = " ".join(f"var av{i} = vec4f(0.0); var ag{i} = vec4f(0.0);" for i in quad)
    a = ", ".join(f"As[kk * {TM}u + lid.y * 4u + {i}u]" for i in quad)
    fma = " ".join(f"av{i} += a.{c} * bv; ag{i} += a.{c} * bg;" for i, c in zip(quad, "xyzw", strict=True))
    store = "\n".join(
        f"  {{ let row = (s * {M}u + m0 + lid.y * 4u + {i}u) * {2 * cout // 4}u + (n0 / 4u) + lid.x;\n"
        f"    Y[row] = av{i}; Y[row + {cout // 4}u] = ag{i}; }}" for i in quad)
    code = wgsl(storage([("P", "array<u32>"), ("X", "array<u32>"), ("Y", "array<vec4f>")]) + """
${HELPERS}
var<workgroup> As: array<f32, ${tkm}>;
var<workgroup> Bv: array<vec4f, ${tkn4}>;
var<workgroup> Bg: array<vec4f, ${tkn4}>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32,
        @builtin(local_invocation_id) lid: vec3u) {
  let m0 = wg.x * ${TM}u; let n0 = wg.y * ${TN}u; let s = wg.z;
  ${accs}
  for (var k0 = s * ${KS}u; k0 < (s + 1u) * ${KS}u; k0 += ${TK}u) {
    for (var r = 0u; r < ${aloads}u; r++) {         // im2col, on the fly
      let e = li + r * 64u; let kk = e / ${TM}u; let mm = e % ${TM}u;
      let k = k0 + kk; let ci = k / 9u; let t = k % 9u; let m = m0 + mm;
      let ry = i32(m / ${wo}u) + i32(t / 3u) - 1; let rx = i32(m % ${wo}u) + i32(t % 3u) - 1;
      var v = 0.0;
      if (ry >= 0 && ry < ${ho} && rx >= 0 && rx < ${wo}) {
        v = unpack2x16float(X[(ci / 2u) * ${hwi}u + u32(ry${up}) * ${wi}u + u32(rx${up})])[ci & 1u];
      }
      As[e] = v;
    }
    for (var r = 0u; r < ${bloads}u; r++) {
      let e = li + r * 64u; let kk = e / ${tn4}u; let nn = (e % ${tn4}u) * 4u;
      let w = (k0 + kk) * ${cout}u + n0 + nn;
      Bv[e] = w4(${wv}u + w); Bg[e] = w4(${wg}u + w);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < ${TK}u; kk++) {
      let a = vec4f(${a});
      let bv = Bv[kk * ${tn4}u + lid.x]; let bg = Bg[kk * ${tn4}u + lid.x];
      ${fma}
    }
    workgroupBarrier();
  }
${store}
}""", tkm=TK * TM, tkn4=TK * TN // 4, TM=TM, TN=TN, TK=TK, KS=KS, accs=accs,
                aloads=TK * TM // 64, wo=wo, ho=ho, hwi=hi * wi, up=up, wi=wi,
                bloads=TK * TN // 4 // 64, tn4=TN // 4, cout=cout, wv=op["wv"], wg=op["wg"],
                a=a, fma=fma, store=store)
    b.step(f"{op['out']}.mm", code, ["P", tensors[op["in"]], "scratch"], [M // TM, cout // TN, S])

    decl, bind = _bindings(op, tensors, "scratch", "array<vec4f>")
    n = M * (cout // 4)
    groups, index = linear(n)
    code = wgsl(decl + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = ${index};
  if (i >= ${n}u) { return; }
  let px = i % ${M}u; let c = (i / ${M}u) * 4u;
  var sv = vec4f(0.0); var sg = vec4f(0.0);
  for (var s = 0u; s < ${S}u; s++) {
    let row = (s * ${M}u + px) * ${c2}u + c / 4u;
    sv += X[row]; sg += X[row + ${c4}u];
  }
${epilogue}
}""", n=n, index=index, M=M, S=S, c2=2 * cout // 4, c4=cout // 4,
                epilogue=_epilogue(op, "sv", "sg", M, "  "))
    b.step(op["out"], code, bind, groups, plan)


def _conv(b: Builder, op: dict, tensors: dict, plan: dict) -> None:
    """A direct convolution, each thread BY x BX output pixels and `oct` channels: with
    nearest upsampling, a 2x2 block of outputs reads only a 3x3 block of inputs, and every
    weight read serves all of the thread's pixels."""
    cout, ho, wo = b.shape(op["out"])
    cin, hi, wi = b.shape(op["in"])
    BY, BX, oct = plan["by"], plan["bx"], plan["oct"]
    nv, up = oct // 4, op["up"]
    pix = [(dy, dx) for dy in range(BY) for dx in range(BX)]

    def at(d, k):
        """Which input row (or column), relative to the thread's, tap k of offset d reads."""
        return (d + k - 1) // 2 if up else d + k - 1

    def span(n):
        return sorted({at(d, k) for d in range(n) for k in range(3)})

    rows, cols = span(BY), span(BX)

    def nb(r, c):
        return f"n{r + 1}_{c + 1}"

    grid = [(r, c) for r in rows for c in cols]
    each = [(p, j, d) for p, d in enumerate(pix) for j in range(nv)]
    half = " / 2u" if up else ""
    body = [f"  let oy = id.y * {BY}u; let ox = id.x * {BX}u;",
            f"  if (oy >= {ho}u || ox >= {wo}u) {{ return; }}",
            f"  let by = i32(oy{half}); let bx = i32(ox{half});"]
    for r, c in grid:
        body += [f"  let v{nb(r, c)} = by + {r} >= 0 && by + {r} < {hi} && bx + {c} >= 0 && bx + {c} < {wi};",
                 f"  let i{nb(r, c)} = select(0u, u32(by + {r}) * {wi}u + u32(bx + {c}), v{nb(r, c)});"]
    body.append(f"  let co = id.z * {oct}u;")
    body += [f"  var av{p}_{j} = vec4f(0.0); var ag{p}_{j} = vec4f(0.0);" for p, j, _ in each]
    body += [f"  for (var c2 = 0u; c2 < {cin // 2}u; c2++) {{",
             f"    let base = c2 * {hi * wi}u;",
             f"    let we = c2 * {18 * cout}u + co; let wodd = we + {9 * cout}u;"]
    body += [f"    let {nb(r, c)} = select(vec2f(0.0), unpack2x16float(X[base + i{nb(r, c)}]), v{nb(r, c)});"
             for r, c in grid]
    for t in range(9):
        ky, kx = divmod(t, 3)
        for j in range(nv):
            o = f"{t * cout + 4 * j}u"
            body += [f"    {{ let a = w4({op['wv']}u + we + {o}); let b = w4({op['wv']}u + wodd + {o});",
                     f"      let c = w4({op['wg']}u + we + {o}); let d = w4({op['wg']}u + wodd + {o});"]
            for p, (dy, dx) in enumerate(pix):
                n = nb(at(dy, ky), at(dx, kx))
                body.append(f"      av{p}_{j} += {n}.x * a + {n}.y * b; ag{p}_{j} += {n}.x * c + {n}.y * d;")
            body.append("    }")
    body.append("  }")
    for p, j, (dy, dx) in each:
        body.append(f"  {{ let px = (oy + {dy}u) * {wo}u + ox + {dx}u; let c = co + {4 * j}u;")
        body.append(_epilogue(op, f"av{p}_{j}", f"ag{p}_{j}", ho * wo, "    ") + " }")
    decl, bind = _bindings(op, tensors, tensors[op["in"]])
    code = wgsl(decl + """
${HELPERS}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
${body}
}""", WG=WG, body="\n".join(body))
    b.step(op["out"], code, bind,
           [math.ceil(wo / BX / WG), math.ceil(ho / BY / WG), cout // oct], plan)


def _rgb(b: Builder, op: dict, tensors: dict, fmt: str) -> None:
    """The last step: packed RGBA8 for display, or f32 planes."""
    cin, h, w = b.shape(op["in"])
    acc = []
    for c2 in range(cin // 2):
        for t in range(9):
            e = op["w"] + (2 * c2 * 9 + t) * 3
            o = e + 27
            acc.append(
                f"  {{ let n = tap({c2 * h * w}u, oy + {t // 3 - 1}, ox + {t % 3 - 1});\n"
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
fn tap(base: u32, y: i32, x: i32) -> vec2f {
  if (y < 0 || y >= ${h} || x < 0 || x >= ${w}) { return vec2f(0.0); }
  return unpack2x16float(X[base + u32(y) * ${w}u + u32(x)]);
}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${w}u || id.y >= ${h}u) { return; }
  let ox = i32(id.x); let oy = i32(id.y); let px = id.y * ${w}u + id.x;
  var s = vec3f(0.0);
${acc}
  s = tanh(clamp(s, vec3f(-10.0), vec3f(10.0)));
  ${store}
}""", h=h, w=w, WG=WG, acc="\n".join(acc), store=store)
    b.buffer("out", size)
    b.step("rgb", code, ["P", tensors[op["in"]], "out"], [math.ceil(w / WG), math.ceil(h / WG), 1])


def _noise(b: Builder, key: str, h: int, w: int) -> None:
    """A noise layer's buffer, filled once at load by `NOISE` with the layer's own seed."""
    n = h * w
    b.buffer(f"noise.{key}", n * 4)
    b.buffer(f"noise.{key}.n", 16, init="words", words=[n, noise_seed(key)])
    b.step(f"noise.{key}", NOISE, [f"noise.{key}", f"noise.{key}.n"], linear(n, 256)[0],
           load=True)
