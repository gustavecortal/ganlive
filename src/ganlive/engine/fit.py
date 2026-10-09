"""The one way a frame is drawn at another size, as a WGSL function every host shares: a take's
frame (`screen.RESIZE`), the desktop window (`present`) and a browser canvas (`SCREEN`, which a
program carries for `runner.mjs`). Shrinking averages the frame pixels each output pixel covers,
the windows of torch's `area`, and growing is bilinear.
"""

from __future__ import annotations

#: `fit(j, out, src, size)`: output pixel `j` of an `out`-pixel picture showing the rectangle
#: `src` (x, y, w, h) of a frame `size` pixels big, as `px(y, x)` unpacks the frame's words.
#: The windows are found in integers, so a frame shrunk by a whole factor averages exactly
#: that many pixels.
FIT = """
fn fit(j: vec2u, out: vec2u, src: vec4u, size: vec2u) -> vec4f {
  if (out.x <= src.z && out.y <= src.w) {
    let lo = src.xy + j * src.zw / out;
    let hi = src.xy + ((j + 1u) * src.zw + out - 1u) / out;
    var c = vec4f(0.0);
    for (var y = lo.y; y < hi.y; y++) { for (var x = lo.x; x < hi.x; x++) { c += px(y, x); } }
    return c / f32((hi.y - lo.y) * (hi.x - lo.x));
  }
  let f = clamp(vec2f(src.xy) + (vec2f(j) + 0.5) * vec2f(src.zw) / vec2f(out) - 0.5,
                vec2f(0.0), vec2f(size - 1u));
  let a = vec2u(f); let b = min(a + 1u, size - 1u); let t = f - floor(f);
  return mix(mix(px(a.y, a.x), px(a.y, b.x), t.x), mix(px(b.y, a.x), px(b.y, b.x), t.x), t.y);
}
"""

#: One triangle over the target, whatever its size and format, in a render pass.
COVER = """
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  return vec4f(f32((i << 1u) & 2u) * 2.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
"""


def screen(output: str) -> str:
    """A program's frame drawn over a whole canvas: the render shader `runner.mjs` draws with.
    X is the frame (packed `output`, rgba8 or bgra8), V its size and the canvas's."""
    order = "zyx" if output == "bgra8" else "xyz"
    return f"""
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<uniform> V: vec4u;        // frame width, height, canvas width, height
fn px(y: u32, x: u32) -> vec4f {{ return unpack4x8unorm(X[y * V.x + x]); }}
{FIT}{COVER}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {{
  return vec4f(fit(vec2u(pos.xy), V.zw, vec4u(0u, 0u, V.xy), V.xy).{order}, 1.0);
}}"""
