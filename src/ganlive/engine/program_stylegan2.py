"""The program of a StyleGAN2, in the same form as a FastGAN's (see `program.py`): the shaders
of one frame, the buffers they use and the dispatches, for any host to run.

One frame, in order:
  map     z -> pixel norm -> eight dense layers -> w
  ws      w for every layer: truncated toward the average w by its style range's dial, then
          pushed by the direction dials (the push buffer, one row per range)
  styles  per modulated layer: its affine of its w, and the demodulation coefficients
  conv    3x3 modulated convolution (inputs scaled by the styles), demodulated, noise, bias,
          leaky ReLU, clamp. An upsampling layer is a stride-2 transposed convolution, then the
          4x4 blur, as NVIDIA's `conv2d_resample` does it
  torgb   1x1 modulated convolution to colour, added to the colour so far blurred up 2x

Features are fp16 channel pairs packed in u32, as in a FastGAN. The colour, the styles and every
sum are f32.
"""

from __future__ import annotations

import math

from ganlive.engine import direct
from ganlive.engine.codegen import FORMAT, HELPERS, WG, Builder, bindings, linear, storage, wgsl
from ganlive.engine.direct import conv3x3
from ganlive.engine.gemm import GEMM_MAX, TK, matmul, split, sum_slices

LRELU = """
fn lrelu4(x: vec4f) -> vec4f { return select(x * 0.2, x, x >= vec4f(0.0)) * 1.41421356; }
"""

#: NVIDIA's resample filter, [1, 3, 3, 1] outer itself, normalised and times 4 for a 2x up.
TAPS = (1, 3, 3, 1)
BLUR = [[a * b / 16.0 for b in TAPS] for a in TAPS]


def compile_stylegan2(manifest: dict, plans: dict | None = None, output: str = "rgba8") -> dict:
    """The program for a StyleGAN2 manifest (from `convert.stylegan2_manifest`). `plans`
    overrides how a matrix product tiles, by its step name: {"rm": 8, "S": 4}."""
    b = Builder(manifest, plans or {})
    m = manifest
    nz, wd, n_ws = m["nz"], m["w_dim"], m["num_ws"]
    b.buffer("P", m["bytes"], init="weights")
    b.buffer("Z", nz * 4)
    b.buffer("K", len(m["settings"]) * 4, init="ones")
    b.buffer("push", len(m["band_slots"]) * wd * 4)
    b.buffer("h0", max(nz, wd) * 4)
    b.buffer("h1", wd * 4)
    b.buffer("WS", n_ws * wd * 4)
    width = max(op.get("cin", 0) for op in m["ops"])
    b.buffer("S", width * 4)
    b.buffer("D", max(op.get("cout", 0) for op in m["ops"]) * 4)
    feat = max(op["cout"] // 2 * op["h"] * op["w"] * 4 for op in m["ops"] if op["op"] != "torgb")
    b.buffer("x0", feat)
    b.buffer("x1", feat)
    b.buffer("T", max([op["cout"] // 2 * (op["h"] + 1) * (op["w"] + 1) * 4
                       for op in m["ops"] if op.get("up")], default=16))
    b.buffer("scratch", max([_scratch(b, op) for op in m["ops"] if op["op"] == "conv"], default=16))
    side = m["height"] * m["width"] * 3 * 4
    b.buffer("img0", side)
    b.buffer("img1", side)

    _ws(b, _mapping(b))
    src, img = None, None
    for op in m["ops"]:
        if op["op"] == "const":
            _const(b, op, "x0")
            src = "x0"
        elif op["op"] == "conv":
            dst = "x1" if src == "x0" else "x0"
            _styles(b, op)
            if op["up"]:
                _upconv(b, op, src, dst)
            elif _as_product(b.overrides, op):
                _mm(b, op, src, dst, _plain(op))
            else:
                _conv(b, op, src, dst)
            src = dst
        elif op["op"] == "torgb":
            dst = "img1" if img == "img0" else "img0"
            _styles(b, op)
            _torgb(b, op, src, img, dst)
            img = dst
    _out(b, img, output)
    return {"format": FORMAT, "family": "stylegan2", "nz": nz, "height": m["height"],
            "width": m["width"], "settings": m["settings"], "output": output,
            "push_shape": [len(m["band_slots"]), wd], "shaders": b.shaders,
            "buffers": b.buffers, "load": b.load, "steps": b.steps}


def _mapping(b: Builder) -> str:
    m = b.m
    nz = m["nz"]
    b.step("map.norm", wgsl(storage([("Z", "array<f32>"), ("Y", "array<f32>")]) + """
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var s = 0.0;
  for (var i = li; i < ${nz}u; i += 256u) { s += Z[i] * Z[i]; }
  part[li] = s;
  workgroupBarrier();
  for (var half = 128u; half > 0u; half >>= 1u) {
    if (li < half) { part[li] += part[li + half]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(part[0] / f32(${nz}u) + 1e-8);
  for (var i = li; i < ${nz}u; i += 256u) { Y[i] = Z[i] * r; }
}""", nz=nz), ["Z", "h0"], [1, 1, 1])
    src, dst = "h0", "h1"
    for i, layer in enumerate(m["mapping"]):
        b.step(f"map.fc{i}", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${n_out}u) { return; }
  var s = 0.0;
  for (var k = 0u; k < ${n_in}u; k++) { s += w1(${w}u + id.x * ${n_in}u + k) * X[k]; }
  s += w1(${bias}u + id.x);
  Y[id.x] = select(s * 0.2, s, s >= 0.0) * 1.41421356;
}""", n_in=layer["n_in"], n_out=layer["n_out"], w=layer["w"], bias=layer["b"]),
               ["P", src, dst], [math.ceil(layer["n_out"] / 64), 1, 1])
        src, dst = dst, src
    return src


def _ws(b: Builder, mapped: str) -> None:
    """Every layer's w: `w_avg + t (w - w_avg) + push`, t and push its style range's."""
    m = b.m
    wd, n_ws = m["w_dim"], m["num_ws"]
    bands = ", ".join(f"{band}u" for band in m["ws_band"])
    slots = ", ".join(f"{slot}u" for slot in m["band_slots"])
    b.step("ws", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("K", "array<f32>"),
                               ("U", "array<f32>"), ("Y", "array<f32>")]) + """
${HELPERS}
const BAND = array<u32, ${n_ws}>(${bands});
const SLOT = array<u32, ${n_bands}>(${slots});
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${total}u) { return; }
  let i = id.x / ${wd}u; let c = id.x % ${wd}u; let band = BAND[i];
  let avg = w1(${avg}u + c);
  Y[id.x] = avg + K[SLOT[band]] * (X[c] - avg) + U[band * ${wd}u + c];
}""", n_ws=n_ws, bands=bands, n_bands=len(m["band_slots"]), slots=slots, total=n_ws * wd,
                   wd=wd, avg=m["w_avg"]),
           ["P", mapped, "K", "push", "WS"], [math.ceil(n_ws * wd / 64), 1, 1])


def _styles(b: Builder, op: dict) -> None:
    """The layer's styles `A w + b` into S and, for a demodulated layer, `rsqrt(E s^2)` into D."""
    wd, cin = b.m["w_dim"], op["cin"]
    b.step(f"{op['name']}.styles", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${cin}u) { return; }
  var s = 0.0;
  for (var k = 0u; k < ${wd}u; k++) { s += w1(${a}u + id.x * ${wd}u + k) * X[${ws}u + k]; }
  Y[id.x] = s + w1(${ab}u + id.x);
}""", cin=cin, wd=wd, a=op["affine"], ab=op["affine_b"], ws=op["ws"] * wd),
           ["P", "WS", "S"], [math.ceil(cin / 64), 1, 1])
    if op["op"] != "conv":
        return
    b.step(f"{op['name']}.demod", wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${cout}u) { return; }
  var s = 0.0;
  for (var i = 0u; i < ${cin}u; i++) { s += w1(${e}u + id.x * ${cin}u + i) * X[i] * X[i]; }
  Y[id.x] = inverseSqrt(s + 1e-8);
}""", cout=op["cout"], cin=cin, e=op["energy"]), ["P", "S", "D"], [math.ceil(op["cout"] / 64), 1, 1])


def _const(b: Builder, op: dict, dst: str) -> None:
    n = op["cout"] // 2 * op["h"] * op["w"]
    b.step("const", wgsl(storage([("P", "array<u32>"), ("Y", "array<u32>")]) + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${n}u) { return; }
  Y[id.x] = P[${w}u / 2u + id.x];
}""", n=n, w=op["data"]), ["P", dst], [math.ceil(n / 64), 1, 1])


def _epilogue(op: dict, acc: str, plane: int) -> list[str]:
    """Noise, bias, activation, clamp and the packed store of channels c..c+3 at `px`, given
    their demodulated sums."""
    gain = f"K[{op['slot']}u]"
    clamp = op["clamp"]
    return [
        f"let nz = w1({op['noise']}u + px) * {op['strength']} * {gain};",
        f"var o = lrelu4({acc} + nz + w4({op['bias']}u + c));",
        f"o = clamp(o, vec4f({-clamp}), vec4f({clamp}));" if clamp else "",
        f"Y[(c / 2u) * {plane}u + px] = pack2x16float(o.xy);",
        f"Y[(c / 2u + 1u) * {plane}u + px] = pack2x16float(o.zw);"]


def _channels(op: dict) -> None:
    """Refuse channel counts the direct paths cannot split: pairs in, groups of 4 out."""
    if op["cin"] % 2 or op["cout"] % 4:
        raise ValueError(f"{op['name']}: {op['cin']} -> {op['cout']} channels. The engine plays "
                         f"even input and multiple-of-4 output channel counts")


def _conv(b: Builder, op: dict, src: str, dst: str) -> None:
    """A modulated 3x3 convolution at one size, directly (`direct.conv3x3`): the inputs scaled
    by the styles as they are read, the sums demodulated. A plan may override the tile by the
    layer's name."""
    _channels(op)
    cout, h, w = op["cout"], op["h"], op["w"]
    side = 2 if h % 2 == 0 else 1
    plan = direct.check({"by": side, "bx": side, "oct": 8 if cout % 8 == 0 else 4,
                         **_override(b, op["name"])}, op["name"], cout, h, w)
    entries = [("P", "array<u32>", "P"), ("X", "array<u32>", src), ("Y", "array<u32>", dst),
               ("S", "array<f32>", "S"), ("D", "array<f32>", "D"), ("K", "array<f32>", "K")]

    def finish(acc: dict) -> str:
        lines = ["let d = vec4f(D[c], D[c + 1u], D[c + 2u], D[c + 3u]);",
                 *_epilogue(op, f"{acc['w']} * d", h * w)]
        return "\n".join("    " + line for line in lines if line)

    conv3x3(b, step=op["name"], cin=op["cin"], cout=cout, h=h, w=w, plan=plan,
            mats={"w": op["weight"]}, entries=entries, finish=finish,
            scale="vec2f(S[2u * c2], S[2u * c2 + 1u])", helpers=LRELU)


def _tiles(op: dict, M: int) -> bool:
    """Whether a matrix product of this layer tiles: whole channel tiles and a sum in TK steps."""
    return op["cout"] % 32 == 0 and op["cin"] % TK == 0 and M > 0


def _mm_plan(b: Builder, op: dict, g: dict) -> dict:
    """How one product tiles: rows a thread (`rm`), column groups (`rn`) and slices (`S`), by
    default so that even a 4x4 map fills the GPU."""
    plan = {"rm": 4, "rn": 1, **_override(b, g["name"])}
    steps = op["cin"] * len(g["taps"]) // TK
    plan.setdefault("S", split(math.ceil(g["M"] / (8 * plan["rm"])) * (op["cout"] // (32 * plan["rn"])), steps))
    return plan


def _plain(op: dict) -> dict:
    """A 3x3 convolution at one size as a matrix product: output pixel m reads (y+ky-1, x+kx-1)."""
    h, w = op["h"], op["w"]
    return {"M": h * w, "taps": list(range(9)), "hi": h, "wi": w, "name": op["name"],
            "at": f"let y = i32(m / {w}u) + i32(t / 3u) - 1; let x = i32(m % {w}u) + i32(t % 3u) - 1;",
            "store": None}


def _parities(op: dict) -> list[dict]:
    """The four output parities of a stride-2 transposed 3x3 convolution, each a matrix product
    over the blocks of that parity: (2a+i, 2b+j) reads (a + dy, b + dx) through tap (ky, kx)."""
    hi, wi = op["h"] // 2, op["w"] // 2
    th, tw = op["h"] + 1, op["w"] + 1
    out = []
    for i in range(2):
        for j in range(2):
            rows, cols = (hi + 1 if i == 0 else hi), (wi + 1 if j == 0 else wi)
            taps, dys, dxs = [], [], []
            for ky in ((0, 2) if i == 0 else (1,)):
                for kx in ((0, 2) if j == 0 else (1,)):
                    taps.append(ky * 3 + kx)
                    dys.append((i - ky) // 2)
                    dxs.append((j - kx) // 2)
            n = len(taps)
            dy = ", ".join(f"{d}" for d in dys)
            dx = ", ".join(f"{d}" for d in dxs)
            out.append({
                "M": rows * cols, "taps": taps, "hi": hi, "wi": wi, "name": f"{op['name']}.t{i}{j}",
                "at": (f"let DY = array<i32, {n}>({dy}); let DX = array<i32, {n}>({dx});\n"
                       f"      let y = i32(m / {cols}u) + DY[q]; let x = i32(m % {cols}u) + DX[q];"),
                "store": f"(2u * (m / {cols}u) + {i}u) * {tw}u + 2u * (m % {cols}u) + {j}u",
                "plane": th * tw})
    return out


def _override(b: Builder, name: str) -> dict:
    """The plan given for step `name`, without the choice of kernel ("gemm")."""
    return {k: v for k, v in b.overrides.get(name, {}).items() if k != "gemm"}


def _as_product(overrides: dict, op: dict) -> bool:
    """Whether a plain conv runs as a matrix product: up to GEMM_MAX pixels, or when a plan
    says {"gemm": True} for it, by name."""
    forced = overrides.get(op["name"], {}).get("gemm")
    M = op["h"] * op["w"]
    return _tiles(op, M) and (M <= GEMM_MAX if forced is None else forced)


def plan_choices(manifest: dict) -> dict[str, list[dict]]:
    """Each step's plans worth trying on a new machine. A matrix product (by its name) tries
    rows and columns a thread and splits, a direct conv (by its layer's name) the tiles that
    do not spill registers, with f32 weights and workgroup memory or without."""
    out = {}
    for op in manifest["ops"]:
        if op["op"] != "conv":
            continue
        if op["up"]:
            products = _parities(op) if _tiles(op, 1) else []
        else:
            products = [_plain(op)] if _as_product({}, op) else []
        for g in products:
            out[g["name"]] = [{"rn": 2}, {"rm": 8}, {"rm": 8, "rn": 2}, {"S": 1}, {"S": 2}, {"S": 4},
                              {"rn": 2, "S": 2}]
        if not op["up"] and not products:
            out[op["name"]] = direct.CHOICES
    return out


def _scratch(b: Builder, op: dict) -> int:
    """The partial sums the largest matrix product of this layer keeps."""
    if op["up"]:
        products = _parities(op)
    elif _as_product(b.overrides, op):
        products = [_plain(op)]
    else:
        return 16
    # A product of one slice finishes its layer itself and keeps no partial sums.
    return max([S * g["M"] * op["cout"] * 4 for g in products
                if (S := _mm_plan(b, op, g)["S"]) > 1], default=16)


def _mm(b: Builder, op: dict, src: str, dst: str, g: dict) -> None:
    """`[pixels x (cin*taps)]` by `[(cin*taps) x cout]`, the inputs scaled by the styles as they
    are read, tiled through workgroup memory and the sum split into slices. A second pass adds
    the slices, demodulates, and either finishes the layer (noise, bias, activation, clamp into
    `dst`) or, for a parity of a transposed convolution, stores into T for the blur."""
    cin, cout = op["cin"], op["cout"]
    M, taps = g["M"], g["taps"]
    nt = len(taps)
    plan = _mm_plan(b, op, g)
    S = plan["S"]
    tap = f"const TAP = array<u32, {nt}>({', '.join(f'{t}u' for t in taps)});\n"
    # The sum runs tap-major: k = q * cin + channel, so k and k + 1 share an input read.
    gather = (f"      let ci = k % {cin}u; let q = k / {cin}u; let t = TAP[q];\n"
              f"      let sc = vec2f(S[ci], S[ci + 1u]); let base = (ci / 2u) * {g['hi'] * g['wi']}u;\n"
              + "\n".join(
                  f"      {{ let m = m4 + {i}u; {g['at']}\n"
                  f"        if (m < {M}u && y >= 0 && y < {g['hi']} && x >= 0 && x < {g['wi']}) {{\n"
                  f"          let v = unpack2x16float(X[base + u32(y) * {g['wi']}u + u32(x)]) * sc; "
                  f"v0[{i}] = v.x; v1[{i}] = v.y; }} }}"
                  for i in range(4)))
    bload = (f"      let row = (k % {cin}u) * 9u + TAP[k / {cin}u]; "
             f"Bw[e] = w4({op['weight']}u + row * {cout}u + nn);")
    # What finishes the layer from the sum of channels c..c+3 at row m, demodulated: the
    # epilogue into `dst`, or, for a parity of a transposed convolution, a store into T.
    if g["store"] is None:
        tail = "\n".join(["let px = m;", *(line for line in _epilogue(op, "v", M) if line)])
        out, extra, helpers = dst, [("K", "array<f32>", "K")], LRELU
    else:
        plane = g["plane"]
        tail = (f"let px = {g['store']};\n"
                f"Y[(c / 2u) * {plane}u + px] = pack2x16float(v.xy);\n"
                f"Y[(c / 2u + 1u) * {plane}u + px] = pack2x16float(v.zw);")
        out, extra, helpers = "T", [], ""

    def finish(acc: dict) -> str:
        return "    " + (f"var v = {acc['w']} * vec4f(D[c], D[c + 1u], D[c + 2u], D[c + 3u]);\n"
                         + tail).replace("\n", "\n    ")

    product = dict(M=M, N=cout, K=cin * nt, S=S, rm=plan["rm"], rn=plan["rn"], mats=("w",),
                   gather=gather, bload=bload)
    ends = [("Y", "array<u32>", out), ("D", "array<f32>", "D"), *extra]
    if S == 1:                  # one slice: the product finishes the layer itself
        decl, bufs = bindings([("P", "array<u32>", "P"), ("X", "array<u32>", src), *ends,
                               ("S", "array<f32>", "S")])
        code, groups = matmul(**product, decl=decl + HELPERS + helpers + tap, finish=finish)
        b.step(g["name"], code, bufs, groups)
        return
    decl, bufs = bindings([("P", "array<u32>", "P"), ("X", "array<u32>", src),
                           ("Y", "array<vec4f>", "scratch"), ("S", "array<f32>", "S")])
    code, groups = matmul(**product, decl=decl + HELPERS + tap)
    b.step(f"{g['name']}.mm", code, bufs, groups)
    # The store into T reads no weights, so P is bound only for the epilogue.
    reads_p = [("P", "array<u32>", "P")] if g["store"] is None else []
    decl, bufs = bindings([*reads_p, ("X", "array<vec4f>", "scratch"), *ends])
    code, groups = sum_slices(rows=M, M=M, N=cout, S=S, mats=("w",),
                              decl=decl + (HELPERS if reads_p else "") + helpers,
                              where="let m = r; let first = 0u;", finish=finish)
    b.step(g["name"], code, bufs, groups)


def _upconv(b: Builder, op: dict, src: str, dst: str) -> None:
    """2x up: the stride-2 transposed convolution, as a matrix product per output parity when
    the layer tiles (or directly), into T, then the 4x4 blur with the epilogue into `dst`."""
    if _tiles(op, 1):
        for g in _parities(op):
            _mm(b, op, src, "T", g)
        _blur(b, op, dst)
        return
    _upconv_direct(b, op, src, dst)


def _upconv_direct(b: Builder, op: dict, src: str, dst: str) -> None:
    """2x up: the stride-2 transposed 3x3 convolution of the styled input, demodulated, into T
    (2h+1 square), then the 4x4 blur with the epilogue into `dst` (2h square).

    Output (2a+i, 2b+j) of the transposed convolution reads input (a, b) through tap
    (i, j), and (a-1, b) or (a, b-1) through taps (i+2, j) and (i, j+2) where i, j are 0: one
    thread for the four outputs of block (a, b) reads a 2x2 block of input, every tap once."""
    _channels(op)
    cin, cout, h, w = op["cin"], op["cout"], op["h"], op["w"]
    hi, wi = h // 2, w // 2
    th, tw = h + 1, w + 1
    oct = 8 if cout % 8 == 0 else 4
    nv = oct // 4
    # Output parity (i, j) -> the (input dy, dx offset, tap) pairs it sums.
    terms = {}
    for i in range(2):
        for j in range(2):
            got = []
            for ky in ((0, 2) if i == 0 else (1,)):
                for kx in ((0, 2) if j == 0 else (1,)):
                    got.append(((i - ky) // 2, (j - kx) // 2, ky * 3 + kx))
            terms[(i, j)] = got
    body = ["  let a = id.y; let bb = id.x;",
            f"  if (a > {hi}u || bb > {wi}u) {{ return; }}",
            f"  let co = id.z * {oct}u;"]
    for dy in (-1, 0):
        for dx in (-1, 0):
            name = f"m{dy + 1}{dx + 1}"
            body += [f"  let v{name} = i32(a) + {dy} >= 0 && i32(a) + {dy} < {hi} && i32(bb) + {dx} >= 0 && i32(bb) + {dx} < {wi};",
                     f"  let i{name} = select(0u, u32(i32(a) + {dy}) * {wi}u + u32(i32(bb) + {dx}), v{name});"]
    body += [f"  var q{i}{j}_{k} = vec4f(0.0);" for i in range(2) for j in range(2) for k in range(nv)]
    body += [f"  for (var c2 = 0u; c2 < {cin // 2}u; c2++) {{",
             f"    let base = c2 * {hi * wi}u; let sv = vec2f(S[2u * c2], S[2u * c2 + 1u]);",
             f"    let we = c2 * {18 * cout}u + co; let wodd = we + {9 * cout}u;"]
    for dy in (-1, 0):
        for dx in (-1, 0):
            name = f"m{dy + 1}{dx + 1}"
            body.append(f"    let {name} = select(vec2f(0.0), unpack2x16float(X[base + i{name}]), v{name}) * sv;")
    for (i, j), got in terms.items():
        for dy, dx, t in got:
            n = f"m{dy + 1}{dx + 1}"
            for k in range(nv):
                o = f"{t * cout + 4 * k}u"
                body.append(f"    q{i}{j}_{k} += {n}.x * w4({op['weight']}u + we + {o}) + {n}.y * w4({op['weight']}u + wodd + {o});")
    body.append("  }")
    for i in range(2):
        for j in range(2):
            body.append(f"  if (2u * a + {i}u < {th}u && 2u * bb + {j}u < {tw}u) {{")
            body.append(f"    let px = (2u * a + {i}u) * {tw}u + 2u * bb + {j}u;")
            for k in range(nv):
                body.append(f"    {{ let c = co + {4 * k}u; let o = q{i}{j}_{k} * vec4f(D[c], D[c + 1u], D[c + 2u], D[c + 3u]);")
                body.append(f"      Y[(c / 2u) * {th * tw}u + px] = pack2x16float(o.xy);")
                body.append(f"      Y[(c / 2u + 1u) * {th * tw}u + px] = pack2x16float(o.zw); }}")
            body.append("  }")
    code = wgsl(storage([("P", "array<u32>"), ("X", "array<u32>"), ("Y", "array<u32>"),
                         ("S", "array<f32>"), ("D", "array<f32>")]) + """
${HELPERS}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
${body}
}""", WG=WG, body="\n".join(body))
    b.step(f"{op['name']}.t", code, ["P", src, "T", "S", "D"],
           [math.ceil((wi + 1) / WG), math.ceil((hi + 1) / WG), cout // oct])
    _blur(b, op, dst)


def _blur(b: Builder, op: dict, dst: str) -> None:
    """T (2h+1 square) through NVIDIA's 4x4 blur with padding 1, then the epilogue, into `dst`."""
    h, w = op["h"], op["w"]
    th, tw = h + 1, w + 1
    taps = []
    for ky in range(4):
        for kx in range(4):
            taps.append(f"    acc += {BLUR[ky][kx]} * t(c2, oy + {ky}u - 1u, ox + {kx}u - 1u);")
    body = ["  let ox = id.x; let oy = id.y; let c2 = id.z;",
            f"  if (ox >= {w}u || oy >= {h}u) {{ return; }}",
            "  var acc = vec2f(0.0);"] + taps + [
            f"  let px = oy * {w}u + ox; let c = 2u * c2;",
            f"  let nz = w1({op['noise']}u + px) * {op['strength']} * K[{op['slot']}u];",
            f"  var o = lrelu4(vec4f(acc + nz + vec2f(w1({op['bias']}u + c), w1({op['bias']}u + c + 1u)), 0.0, 0.0)).xy;",
            f"  o = clamp(o, vec2f({-op['clamp']}), vec2f({op['clamp']}));" if op["clamp"] else "",
            f"  Y[c2 * {h * w}u + px] = pack2x16float(o);"]
    code = wgsl(storage([("P", "array<u32>"), ("X", "array<u32>"), ("Y", "array<u32>"), ("K", "array<f32>")]) + """
${HELPERS}
${LRELU}
fn t(c2: u32, y: u32, x: u32) -> vec2f {
  if (y >= ${th}u || x >= ${tw}u) { return vec2f(0.0); }
  return unpack2x16float(X[c2 * ${tt}u + y * ${tw}u + x]);
}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
${body}
}""", LRELU=LRELU, WG=WG, th=th, tw=tw, tt=th * tw, body="\n".join(line for line in body if line))
    b.step(op["name"], code, ["P", "T", dst, "K"], [math.ceil(w / WG), math.ceil(h / WG), op["cout"] // 2])


def _torgb(b: Builder, op: dict, src: str, prev: str | None, dst: str) -> None:
    """Colour from this block's features (`S` holds its styles, gain folded in), plus the colour
    so far, blurred up 2x when there is any."""
    cin, h, w = op["cin"], op["h"], op["w"]
    plane = h * w
    acc = []
    for c2 in range(cin // 2):
        e = op["weight"] + 2 * c2 * 3
        acc.append(f"  {{ let n = unpack2x16float(X[{c2 * plane}u + px]) * vec2f(S[{2 * c2}u], S[{2 * c2 + 1}u]);\n"
                   f"    y += n.x * vec3f(w1({e}u), w1({e + 1}u), w1({e + 2}u))"
                   f" + n.y * vec3f(w1({e + 3}u), w1({e + 4}u), w1({e + 5}u)); }}")
    clamp = op["clamp"]
    up = []
    if prev is not None:
        hp, wp = h // 2, w // 2
        # Stuffed (zeros between samples) and blurred with padding 2 before, 1 after: output
        # (oy, ox) reads stuffed (oy + ky - 2, ox + kx - 2), a sample where both are even.
        for ky in range(4):
            for kx in range(4):
                up.append(f"  {{ let sy = i32(oy) + {ky - 2}; let sx = i32(ox) + {kx - 2};\n"
                          f"    if (sy >= 0 && sx >= 0 && sy % 2 == 0 && sx % 2 == 0 && sy / 2 < {hp} && sx / 2 < {wp}) {{\n"
                          f"      let q = u32(sy / 2) * {wp}u + u32(sx / 2);\n"
                          f"      y += {BLUR[ky][kx]} * vec3f(I[q], I[{hp * wp}u + q], I[{2 * hp * wp}u + q]); }} }}")
    names = [("P", "array<u32>"), ("X", "array<u32>"), ("Y", "array<f32>"), ("S", "array<f32>")]
    binds = ["P", src, dst, "S"]
    if prev is not None:
        names.append(("I", "array<f32>"))
        binds.append(prev)
    code = wgsl(storage(names) + """
${HELPERS}
@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ox = id.x; let oy = id.y;
  if (ox >= ${w}u || oy >= ${h}u) { return; }
  let px = oy * ${w}u + ox;
  var y = vec3f(w1(${bias}u), w1(${bias}u + 1u), w1(${bias}u + 2u));
${acc}
${clamp}
${up}
  Y[px] = y.x; Y[${plane}u + px] = y.y; Y[${plane2}u + px] = y.z;
}""", WG=WG, w=w, h=h, bias=op["bias"], acc="\n".join(acc), plane=plane, plane2=2 * plane,
                clamp=f"  y = clamp(y, vec3f({-clamp}), vec3f({clamp}));" if clamp else "",
                up="\n".join(up))
    b.step(op["name"], code, binds, [math.ceil(w / WG), math.ceil(h / WG), 1])


def _out(b: Builder, img: str, output: str) -> None:
    h, w = b.m["height"], b.m["width"]
    plane = h * w
    if output == "f32":
        b.buffer("out", 3 * plane * 4)
        store = f"Y[px] = I[px]; Y[{plane}u + px] = I[{plane}u + px]; Y[{2 * plane}u + px] = I[{2 * plane}u + px];"
        kind = "f32"
    else:
        b.buffer("out", plane * 4)
        store = (f"let s = clamp(vec3f(I[px], I[{plane}u + px], I[{2 * plane}u + px]), vec3f(-1.0), vec3f(1.0));\n"
                 f"  Y[px] = pack4x8unorm(vec4f({'s.zyx' if output == 'bgra8' else 's'} * 0.5 + 0.5, 1.0));")
        kind = "u32"
    groups, index = linear(plane, 256)
    b.step("out", wgsl(storage([("I", "array<f32>"), ("Y", f"array<{kind}>")]) + """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let px = ${index};
  if (px >= ${plane}u) { return; }
  ${store}
}""", plane=plane, store=store, index=index), [img, "out"], groups)


