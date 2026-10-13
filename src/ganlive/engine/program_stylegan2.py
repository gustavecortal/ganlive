"""The program of a StyleGAN2, in the same form as a FastGAN's (see `program.py`): the shaders
of one frame, the buffers they use and the dispatches, for any host to run.

One frame, in order:
  map     z -> pixel norm -> eight dense layers -> w
  ws      w for every layer: truncated toward the average w by its style range's dial, then
          pushed by the direction dials (the push buffer, one row per range)
  styles  every modulated layer's affine of its w, then every conv's demodulation
          coefficients: two dispatches for the frame
  conv    3x3 modulated convolution (inputs scaled by the styles), demodulated, noise, bias,
          leaky ReLU, clamp: directly, as a matrix product on a small map, or through Winograd
          F(2x2, 3x3) on a mid-sized one. An upsampling layer is a stride-2 transposed
          convolution, then the 4x4 blur, as NVIDIA's `conv2d_resample` does it
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

#: The 2x2 blocks of the maps a plain conv runs through Winograd by default (`_as_wino`): from
#: 16x16 to 128x128 on an A770, where it took FFHQ's 3x3 layers there from 7.5 ms to 3.9. On
#: larger maps its partial sums cost more to write and read than the products it saves.
WINO_BLOCKS = (64, 4096)

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
    # Every modulated layer's styles, and every conv's demodulation, side by side: a layer
    # reads its own from `s` and `d`, so two dispatches make them all.
    ops, s_at, d_at = [], 0, 0
    for op in m["ops"]:
        if op["op"] in ("conv", "torgb"):
            op = {**op, "s": s_at, "d": d_at}
            s_at += op["cin"]
            d_at += op["cout"] if op["op"] == "conv" else 0
        ops.append(op)
    b.buffer("S", s_at * 4)
    b.buffer("D", d_at * 4)
    feat = max(op["cout"] // 2 * op["h"] * op["w"] * 4 for op in m["ops"] if op["op"] != "torgb")
    b.buffer("x0", feat)
    b.buffer("x1", feat)
    b.buffer("T", max(_t_bytes(b, op) for op in m["ops"]))
    b.buffer("scratch", max([_scratch(b, op) for op in m["ops"] if op["op"] == "conv"], default=16))
    side = m["height"] * m["width"] * 3 * 4
    b.buffer("img0", side)
    b.buffer("img1", side)

    _ws(b, _mapping(b))
    _styles(b, [op for op in ops if "s" in op])
    src, img = None, None
    for op in ops:
        if op["op"] == "const":
            _const(b, op, "x0")
            src = "x0"
        elif op["op"] == "conv":
            dst = "x1" if src == "x0" else "x0"
            if op["up"]:
                _upconv(b, op, src, dst)
            elif _as_wino(b.overrides, op):
                _wino(b, op, src, dst)
            elif _as_product(b.overrides, op):
                _mm(b, op, src, dst, _plain(op))
            else:
                _conv(b, op, src, dst)
            src = dst
        elif op["op"] == "torgb":
            dst = "img1" if img == "img0" else "img0"
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
        _rows(b, f"map.fc{i}", [{"rows": layer["n_out"], "k": layer["n_in"], "w": layer["w"],
                                 "b": layer["b"], "x": 0}],
              src, dst, "let v = s + w1(B[L] + i); Y[r] = select(v * 0.2, v, v >= 0.0) * 1.41421356;")
        src, dst = dst, src
    return src


#: Threads that share one row of a matrix-vector product, and rows a workgroup takes.
ROW_THREADS, ROW_GROUP = 64, 4


def _rows(b: Builder, name: str, parts: list[dict], src: str, dst: str, store: str,
          square: bool = False) -> None:
    """Matrix-vector products in one dispatch, `ROW_THREADS` threads summing each row: part L
    has `rows` rows of `k` weights at `w` in P, read against X from `x` (squared with
    `square`), its rows stored one after the other into Y. `store` writes row `r` (the
    part's row `i`) from its sum `s`, and may read B[L], the part's `b`."""
    starts = [0]
    for p in parts:
        starts.append(starts[-1] + p["rows"])
    total, n = starts[-1], len(parts)

    def table(values) -> str:
        return f"array<u32, {len(values)}>({', '.join(f'{v}u' for v in values)})"

    x = "X[X0[L] + k] * X[X0[L] + k]" if square else "X[X0[L] + k]"
    groups, index = linear(math.ceil(total / ROW_GROUP), 1)
    b.step(name, wgsl(storage([("P", "array<u32>"), ("X", "array<f32>"), ("Y", "array<f32>")]) + """
${HELPERS}
const START = ${starts};
const W = ${w}; const B = ${bias}; const K = ${k}; const X0 = ${x0};
var<workgroup> part: array<f32, ${size}>;
@compute @workgroup_size(${size})
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
  let r = ${index} * ${group}u + li / ${threads}u; let t = li % ${threads}u;
  var L = 0u;
  while (L + 1u < ${n}u && START[L + 1u] <= r) { L++; }
  let i = r - START[L];
  var s = 0.0;
  if (r < ${total}u) {
    for (var k = t; k < K[L]; k += ${threads}u) { s += w1(W[L] + i * K[L] + k) * ${x}; }
  }
  part[li] = s;
  workgroupBarrier();
  for (var half = ${half}u; half > 0u; half >>= 1u) {
    if (t < half) { part[li] += part[li + half]; }
    workgroupBarrier();
  }
  if (r < ${total}u && t == 0u) { s = part[li]; ${store} }
}""", starts=table(starts), w=table([p["w"] for p in parts]),
                      bias=table([p.get("b", 0) for p in parts]), k=table([p["k"] for p in parts]),
                      x0=table([p["x"] for p in parts]), size=ROW_THREADS * ROW_GROUP,
                      group=ROW_GROUP, threads=ROW_THREADS, half=ROW_THREADS // 2, n=n,
                      total=total, x=x, index=index, store=store),
           ["P", src, dst], groups)


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


def _styles(b: Builder, ops: list[dict]) -> None:
    """Every modulated layer's styles `A w + b` into S at its `s`, then every conv's
    demodulation `rsqrt(E s^2)` into D at its `d`: two dispatches for the whole frame."""
    wd = b.m["w_dim"]
    _rows(b, "styles", [{"rows": op["cin"], "k": wd, "w": op["affine"], "b": op["affine_b"],
                         "x": op["ws"] * wd} for op in ops],
          "WS", "S", "Y[r] = s + w1(B[L] + i);")
    _rows(b, "demod", [{"rows": op["cout"], "k": op["cin"], "w": op["energy"], "x": op["s"]}
                       for op in ops if op["op"] == "conv"],
          "S", "D", "Y[r] = inverseSqrt(s + 1e-8);", square=True)


def _sv(op: dict, pair: str) -> str:
    """The styles of input channels 2 `pair` and 2 `pair` + 1, from S."""
    return f"vec2f(S[{op['s']}u + 2u * {pair}], S[{op['s']}u + 2u * {pair} + 1u])"


def _dv(op: dict) -> str:
    """The demodulation of output channels c..c+3, from D."""
    d = op["d"]
    return f"vec4f(D[{d}u + c], D[{d}u + c + 1u], D[{d}u + c + 2u], D[{d}u + c + 3u])"


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
    # Weights made f32 at load by default: on an A770 they took FFHQ's frame from 26.4 to
    # 24.7 ms in Edge, and from 25.4 to 23.5 ms on Vulkan. A plan says which weights it reads.
    plan = _override(b, op["name"]) or {"f32": True}
    plan = direct.check({"by": side, "bx": side, "oct": 8 if cout % 8 == 0 else 4, **plan},
                        op["name"], cout, h, w)
    entries = [("P", "array<u32>", "P"), ("X", "array<u32>", src), ("Y", "array<u32>", dst),
               ("S", "array<f32>", "S"), ("D", "array<f32>", "D"), ("K", "array<f32>", "K")]

    def finish(acc: dict) -> str:
        lines = [f"let d = {_dv(op)};", *_epilogue(op, f"{acc['w']} * d", h * w)]
        return "\n".join("    " + line for line in lines if line)

    conv3x3(b, step=op["name"], cin=op["cin"], cout=cout, h=h, w=w, plan=plan,
            mats={"w": op["weight"]}, entries=entries, finish=finish,
            scale=_sv(op, "c2"), helpers=LRELU)


def _tiles(op: dict, M: int) -> bool:
    """Whether a matrix product of this layer tiles: whole channel tiles and a sum in TK steps."""
    return op["cout"] % 32 == 0 and op["cin"] % TK == 0 and M > 0


def _mm_plan(b: Builder, op: dict, g: dict, parts: int = 1) -> dict:
    """How one product (or `parts` alike) tiles: rows a thread (`rm`), column groups (`rn`)
    and slices (`S`), by default so that even a 4x4 map fills the GPU."""
    plan = {"rm": 4, "rn": 1, **_override(b, g["name"])}
    steps = op["cin"] * len(g["taps"]) // TK
    tiles = math.ceil(g["M"] / (8 * plan["rm"])) * (op["cout"] // (32 * plan["rn"])) * parts
    plan.setdefault("S", split(tiles, steps))
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
    """The plan given for step `name`, without the choice of kernel ("gemm", "wino")."""
    return {k: v for k, v in b.overrides.get(name, {}).items() if k not in ("gemm", "wino")}


# Winograd F(2x2, 3x3) (Lavin and Gray, 2016): a 2x2 block of a 3x3 convolution from its 4x4
# input patch d and the kernel g is A^T [(G g G^T) * (B^T d B)] A, 16 products where the direct
# sum takes 36. The weights do not change between frames (the styles scale the input, the
# demodulation the output), so G g G^T is made once, at load.
WINO_B = ((1, 0, -1, 0), (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, 0, -1))      # B^T
WINO_G = ((1, 0, 0), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0, 0, 1))
WINO_A = ((1, 1, 1, 0), (0, 1, -1, -1))                                    # A^T


def _wino_fits(op: dict) -> bool:
    """Whether a plain conv can run through Winograd: 2x2 blocks and whole product tiles."""
    return op["h"] % 2 == 0 and op["w"] % 2 == 0 and _tiles(op, 1)


def _as_wino(overrides: dict, op: dict) -> bool:
    """Whether a plain conv runs through Winograd: where it fits and has `WINO_BLOCKS`
    blocks, or when a plan says {"wino": True} for it, by name."""
    forced = overrides.get(op["name"], {}).get("wino")
    lo, hi = WINO_BLOCKS
    return _wino_fits(op) and (lo <= _blocks(op) <= hi if forced is None else forced)


def _combine(rows, cols, value) -> str:
    """The WGSL sum of `rows[a] * value(i, j) * cols[b]` over i, j, its zero terms left out."""
    terms = []
    for i, ra in enumerate(rows):
        for j, cb in enumerate(cols):
            k = ra * cb
            if k:
                terms.append(("" if k == 1 else "-" if k == -1 else f"{k} * ") + value(i, j))
    return " + ".join(terms).replace("+ -", "- ")


def _blocks(op: dict) -> int:
    """The 2x2 blocks of a layer's map, which Winograd draws one at a time."""
    return op["h"] * op["w"] // 4


def _wino_plan(b: Builder, op: dict) -> dict:
    """How the 16 products [blocks x cin] by [cin x cout] tile (`_mm_plan`)."""
    return _mm_plan(b, op, {"name": f"{op['name']}.mm", "M": _blocks(op), "taps": [0]}, parts=16)


def _wino(b: Builder, op: dict, src: str, dst: str) -> None:
    """A modulated 3x3 convolution at one size through Winograd F(2x2, 3x3): the styled input's
    patches transformed into T, 16 matrix products [blocks x cin] by [cin x cout] into the
    scratch, then each 2x2 block transformed back, demodulated and finished into `dst`."""
    cin, cout, h, w = op["cin"], op["cout"], op["h"], op["w"]
    bw = w // 2
    blocks = _blocks(op)
    name = op["name"]
    # Once: the kernel of each (input, output pair) as its 16 values, f16 pairs [16][cin][cout/2].
    v = b.buffer(f"wino.{name}", 16 * cin * cout * 2)
    g = "\n".join(f"  let g{t} = unpack2x16float(P[(e + {t * cout}u) >> 1u]);" for t in range(9))
    stores = "\n".join(
        f"  Y[{p * cin * cout // 2}u + i] = pack2x16float("
        + _combine(WINO_G[p // 4], WINO_G[p % 4], lambda y, x: f"g{y * 3 + x}") + ");"
        for p in range(16))
    n = cin * cout // 2
    groups, index = linear(n)
    b.step(f"{name}.wino", wgsl(storage([("P", "array<u32>"), ("Y", "array<u32>")]) + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = ${index};
  if (i >= ${n}u) { return; }
  let ci = i / ${half}u; let co = 2u * (i % ${half}u);
  let e = ${weight}u + ci * ${taps}u + co;
${g}
${stores}
}""", index=index, n=n, half=cout // 2, weight=op["weight"], taps=9 * cout, g=g, stores=stores),
           ["P", v], groups, load=True)

    # Each frame: the 4x4 patch of every block, B^T d B, as f16 pairs [16][cin/2][blocks]. The
    # styles scale it as the products read it, in f32, so that what is stored stays as small as
    # the features.
    plane, cells = cin // 2 * blocks, h * w
    reads = "\n".join(
        f"  let v{y}{x} = y0 + {y} >= 0 && y0 + {y} < {h} && x0 + {x} >= 0 && x0 + {x} < {w};\n"
        f"  let d{y}{x} = select(vec2f(0.0), unpack2x16float(X[base + select(0u, u32(y0 + {y}) * {w}u"
        f" + u32(x0 + {x}), v{y}{x})]), v{y}{x});"
        for y in range(4) for x in range(4))
    stores = "\n".join(
        f"  Y[{p * plane}u + i] = pack2x16float("
        + _combine(WINO_B[p // 4], WINO_B[p % 4], lambda y, x: f"d{y}{x}") + ");"
        for p in range(16))
    groups, index = linear(plane)
    b.step(f"{name}.in", wgsl(storage([("X", "array<u32>"), ("Y", "array<u32>")]) + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = ${index};
  if (i >= ${n}u) { return; }
  let c2 = i / ${blocks}u; let blk = i % ${blocks}u;
  let y0 = 2 * i32(blk / ${bw}u) - 1; let x0 = 2 * i32(blk % ${bw}u) - 1;
  let base = c2 * ${cells}u;
${reads}
${stores}
}""", index=index, n=plane, blocks=blocks, bw=bw, cells=cells, reads=reads, stores=stores),
           [src, "T"], groups)

    # The 16 products, `par` being which: A [blocks x cin] from T, B [cin x cout] from the kernels.
    plan = _wino_plan(b, op)
    S = plan["S"]
    gather = (f"      let base = (par * {cin // 2}u + k / 2u) * {blocks}u; let sv = {_sv(op, 'k / 2u')};\n"
              + "\n".join(f"      {{ let m = m4 + {i}u; if (m < {blocks}u) {{ let x = unpack2x16float(X[base + m]) * sv; "
                          f"v0[{i}] = x.x; v1[{i}] = x.y; }} }}" for i in range(4)))
    bload = (f"      let q = ((par * {cin}u + k) * {cout}u + nn) / 2u; "
             f"Bw[e] = vec4f(unpack2x16float(V[q]), unpack2x16float(V[q + 1u]));")
    decl, bufs = bindings([("X", "array<u32>", "T"), ("Y", "array<vec4f>", "scratch"),
                           ("V", "array<u32>", v), ("S", "array<f32>", "S")])
    code, groups = matmul(M=blocks, N=cout, K=cin, S=S, rm=plan["rm"], rn=plan["rn"], mats=("w",),
                          decl=decl, gather=gather, bload=bload, parts=16)
    b.step(f"{name}.mm", code, bufs, groups)

    # Back: each block's 16 sums (over the slices), A^T m A, demodulated and finished.
    sums = "\n".join(
        f"  var m{p} = vec4f(0.0); for (var s = 0u; s < {S}u; s++) "
        f"{{ m{p} += X[(({p * S}u + s) * {blocks}u + blk) * {cout // 4}u + c / 4u]; }}"
        for p in range(16))
    outs = []
    for i in range(2):
        for j in range(2):
            out = _combine(WINO_A[i], WINO_A[j], lambda a, c: f"m{a * 4 + c}")
            lines = [f"let px = (2u * (blk / {bw}u) + {i}u) * {w}u + 2u * (blk % {bw}u) + {j}u;",
                     *_epilogue(op, f"({out}) * d", h * w)]
            outs.append("  { " + " ".join(line for line in lines if line) + " }")
    n = blocks * cout // 4
    groups, index = linear(n)
    b.step(name, wgsl(storage([("P", "array<u32>"), ("X", "array<vec4f>"), ("Y", "array<u32>"),
                               ("D", "array<f32>"), ("K", "array<f32>")]) + """
${HELPERS}
${LRELU}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = ${index};
  if (i >= ${n}u) { return; }
  let c = (i % ${q4}u) * 4u; let blk = i / ${q4}u;
  let d = ${dv};
${sums}
${outs}
}""", LRELU=LRELU, index=index, n=n, q4=cout // 4, dv=_dv(op), sums=sums, outs="\n".join(outs)),
           ["P", "scratch", dst, "D", "K"], groups)


def _t_bytes(b: Builder, op: dict) -> int:
    """What an op keeps in T: an upsampling conv's output before its blur, or a Winograd
    conv's transformed input."""
    if op.get("up"):
        return op["cout"] // 2 * (op["h"] + 1) * (op["w"] + 1) * 4
    if op["op"] == "conv" and _as_wino(b.overrides, op):
        return 16 * op["cin"] // 2 * _blocks(op) * 4
    return 16


def _as_product(overrides: dict, op: dict) -> bool:
    """Whether a plain conv runs as a matrix product: up to GEMM_MAX pixels, or when a plan
    says {"gemm": True} for it, by name."""
    forced = overrides.get(op["name"], {}).get("gemm")
    M = op["h"] * op["w"]
    return _tiles(op, M) and (M <= GEMM_MAX if forced is None else forced)


#: The plans a matrix product tries: rows and columns a thread, and splits.
PRODUCT_CHOICES = [{"rn": 2}, {"rm": 8}, {"rm": 8, "rn": 2}, {"S": 1}, {"S": 2}, {"S": 4},
                   {"rn": 2, "S": 2}]
#: The most blocks `ganlive tune` tries Winograd on. On FFHQ's 256x256 layer (128 channels) its
#: buffers reach the 128 MiB a binding gets by default. A wider layer outgrows it sooner, and
#: tune skips a trial that fails to build.
WINO_TRIED = 16384


def plan_choices(manifest: dict) -> dict[str, list[dict]]:
    """Each step's plans worth trying on a new machine. A matrix product (by its name) tries
    rows and columns a thread and splits, a direct conv (by its layer's name) the tiles that
    do not spill registers, with f32 weights and workgroup memory or without, and a plain conv
    that Winograd fits tries it on or off."""
    out = {}
    for op in manifest["ops"]:
        if op["op"] != "conv":
            continue
        if op["up"]:
            for g in _parities(op) if _tiles(op, 1) else []:
                out[g["name"]] = PRODUCT_CHOICES
            continue
        # A plain conv's kernel and its plans share the layer's name: Winograd, or else a
        # product or the direct conv, each with its plans.
        others = PRODUCT_CHOICES if _as_product({}, op) else direct.CHOICES
        if _as_wino({}, op):
            off = [{"wino": False, **o} for o in others]
            out[op["name"]] = off if {} in others else [{"wino": False}, *off]
            out[f"{op['name']}.mm"] = PRODUCT_CHOICES
        elif _wino_fits(op) and _blocks(op) <= WINO_TRIED:
            out[op["name"]] = [{"wino": True}, *others]
        else:
            out[op["name"]] = others
    return out


def _scratch(b: Builder, op: dict) -> int:
    """The partial sums this layer's largest matrix product keeps. Winograd's 16 always keep them."""
    if op["up"]:
        products = _parities(op)
    elif _as_wino(b.overrides, op):
        return 16 * _wino_plan(b, op)["S"] * _blocks(op) * op["cout"] * 4
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
              f"      let sc = {_sv(op, 'ci / 2u')}; let base = (ci / 2u) * {g['hi'] * g['wi']}u;\n"
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
        return "    " + (f"var v = {acc['w']} * {_dv(op)};\n"
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
             f"    let base = c2 * {hi * wi}u; let sv = {_sv(op, 'c2')};",
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
                body.append(f"    {{ let c = co + {4 * k}u; let o = q{i}{j}_{k} * {_dv(op)};")
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
    """T (2h+1 square) through NVIDIA's 4x4 blur with padding 1, then the epilogue, into `dst`.
    A thread draws 2x2 pixels of a channel pair from the 5x5 samples they share."""
    h, w = op["h"], op["w"]
    th, tw = h + 1, w + 1
    body = ["  let ox = 2u * id.x; let oy = 2u * id.y; let c2 = id.z;",
            f"  if (ox >= {w}u || oy >= {h}u) {{ return; }}",
            "  let c = 2u * c2;",
            f"  let bias = vec2f(w1({op['bias']}u + c), w1({op['bias']}u + c + 1u));"]
    body += [f"  let r{y}{x} = t(c2, oy + {y}u - 1u, ox + {x}u - 1u);" for y in range(5) for x in range(5)]
    for dy in range(2):
        for dx in range(2):
            taps = " + ".join(f"{BLUR[ky][kx]} * r{dy + ky}{dx + kx}" for ky in range(4) for kx in range(4))
            body += [f"  {{ let px = (oy + {dy}u) * {w}u + ox + {dx}u;",
                     f"    let nz = w1({op['noise']}u + px) * {op['strength']} * K[{op['slot']}u];",
                     f"    var o = lrelu4(vec4f({taps} + nz + bias, 0.0, 0.0)).xy;",
                     f"    o = clamp(o, vec2f({-op['clamp']}), vec2f({op['clamp']}));" if op["clamp"] else "",
                     f"    Y[c2 * {h * w}u + px] = pack2x16float(o); }}"]
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
    b.step(op["name"], code, ["P", "T", dst, "K"],
           [math.ceil(w / 2 / WG), math.ceil(h / 2 / WG), op["cout"] // 2])


def _torgb(b: Builder, op: dict, src: str, prev: str | None, dst: str) -> None:
    """Colour from this block's features (`S` holds its styles, gain folded in), plus the colour
    so far, blurred up 2x when there is any."""
    cin, h, w = op["cin"], op["h"], op["w"]
    plane = h * w
    # A loop over channel pairs: unrolled, a 512-channel block took Edge seconds to compile.
    acc = (f"  for (var c2 = 0u; c2 < {cin // 2}u; c2++) {{\n"
           f"    let n = unpack2x16float(X[c2 * {plane}u + px]) * {_sv(op, 'c2')};\n"
           f"    let e = {op['weight']}u + 6u * c2;\n"
           f"    y += n.x * vec3f(w1(e), w1(e + 1u), w1(e + 2u)) + n.y * vec3f(w1(e + 3u), w1(e + 4u), w1(e + 5u));\n"
           f"  }}")
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
}""", WG=WG, w=w, h=h, bias=op["bias"], acc=acc, plane=plane, plane2=2 * plane,
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


