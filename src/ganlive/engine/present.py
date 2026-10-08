"""Drawing an engine frame straight into the window, from the GPU that made it.

`Screen` presents through a wgpu surface on the SDL window's native handle: the model's frame
is scaled to the window by a fragment shader, beside the strip, which arrives as a small
texture. Nothing goes through host memory, which a download and an SDL texture upload of the
same frame would cost twice over (25 MB a frame for a 3072x2048 model).
"""

from __future__ import annotations

import ctypes
import os
import sys

import numpy as np
import wgpu

# One triangle over the window. A pixel is the strip's, the picture's or black. Shrinking
# averages the source pixels a window pixel covers (as `screen.RESIZE` does), growing is
# bilinear. X is the frame as packed BGRA8, which unpacks to (b, g, r, a).
SHADER = """
struct View { dst: vec4f, src: vec4f, strip: vec4f, size: vec4f }   // size: frame w, h, strip w, h
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var S: texture_2d<f32>;
@group(0) @binding(2) var<uniform> V: View;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  return vec4f(f32((i << 1u) & 2u) * 2.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
fn px(y: u32, x: u32) -> vec3f { return unpack4x8unorm(X[y * u32(V.size.x) + x]).zyx; }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = pos.xy;
  if (p.x >= V.strip.x && p.x < V.strip.x + V.strip.z && p.y < V.strip.y + V.strip.w) {
    let q = vec2u((p - V.strip.xy) * V.size.zw / V.strip.zw);
    return vec4f(textureLoad(S, min(q, vec2u(V.size.zw) - 1u), 0).xyz, 1.0);
  }
  if (any(p < V.dst.xy) || any(p >= V.dst.xy + V.dst.zw)) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let W = u32(V.size.x); let H = u32(V.size.y);
  let s = V.src.zw / V.dst.zw;                       // frame pixels per window pixel
  if (s.x >= 1.0 && s.y >= 1.0) {
    let lo = V.src.xy + (p - 0.5 - V.dst.xy) * s;
    let x0 = u32(lo.x); let y0 = u32(lo.y);
    let x1 = min(W, max(x0 + 1u, u32(ceil(lo.x + s.x)))); let y1 = min(H, max(y0 + 1u, u32(ceil(lo.y + s.y))));
    var c = vec3f(0.0);
    for (var y = y0; y < y1; y++) { for (var x = x0; x < x1; x++) { c += px(y, x); } }
    return vec4f(c / f32((y1 - y0) * (x1 - x0)), 1.0);
  }
  let f = clamp(V.src.xy + (p - V.dst.xy) * s - 0.5, vec2f(0.0), vec2f(f32(W - 1u), f32(H - 1u)));
  let x0 = u32(f.x); let y0 = u32(f.y); let x1 = min(x0 + 1u, W - 1u); let y1 = min(y0 + 1u, H - 1u);
  let t = f - floor(f);
  return vec4f(mix(mix(px(y0, x0), px(y0, x1), t.x), mix(px(y1, x0), px(y1, x1), t.x), t.y), 1.0);
}
"""



class _WMInfo(ctypes.Structure):
    """SDL_SysWMinfo: SDL's version, the windowing system, and its handles."""
    _fields_ = [("major", ctypes.c_uint8), ("minor", ctypes.c_uint8), ("patch", ctypes.c_uint8),
                ("subsystem", ctypes.c_int), ("info", ctypes.c_void_p * 8)]


def _sdl():
    """The SDL library pygame itself uses. On Windows its own DLL, beside pygame. Elsewhere
    pygame's video module, whose symbols include those of the SDL it links."""
    import pygame
    import pygame._sdl2.video

    if sys.platform == "win32":
        return ctypes.CDLL(os.path.join(os.path.dirname(pygame.__file__), "SDL2.dll"))
    return ctypes.CDLL(pygame._sdl2.video.__file__)


def native_window(window) -> tuple[dict, object]:
    """The native handle of a pygame `Window`, as wgpu takes it, and SDL itself."""
    import pygame

    sdl = _sdl()
    sdl.SDL_GetWindowFromID.restype = ctypes.c_void_p
    sdl.SDL_GetWindowFromID.argtypes = [ctypes.c_uint32]
    sdl.SDL_GetWindowWMInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_WMInfo)]
    ptr = sdl.SDL_GetWindowFromID(window.id)
    info = _WMInfo()
    info.major, info.minor, info.patch = pygame.get_sdl_version()
    if not ptr or not sdl.SDL_GetWindowWMInfo(ptr, ctypes.byref(info)):
        raise RuntimeError("SDL gives no native handle for this window")
    # SDL_SYSWM_TYPE: 1 Windows, 2 X11, 4 Cocoa, 6 Wayland.
    handles = {1: lambda i: {"platform": "windows", "window": i[0]},
               2: lambda i: {"platform": "x11", "display": i[0], "window": i[1]},
               4: lambda i: {"platform": "cocoa", "window": i[0]},
               6: lambda i: {"platform": "wayland", "display": i[0], "window": i[1]}}
    if info.subsystem not in handles:
        raise RuntimeError(f"no wgpu surface for SDL window system {info.subsystem}")
    return {"method": "screen", "vsync": False, **handles[info.subsystem](info.info)}, (sdl, ptr)


class Screen:
    """A window's surface on `device`: `present` draws one engine frame and the strip into it."""

    def __init__(self, window, device) -> None:
        handle, (self._sdl, self._ptr) = native_window(window)
        self.device = device
        self.context = wgpu.gpu.get_canvas_context(handle)
        self._size = self.pixels(window)
        self.context.set_physical_size(*self._size)
        # The surface's own format, not its sRGB view: the shader writes the frame's values
        # as they are. Whichever channel order it has, the format stores the colour right.
        self.format = self.context.get_preferred_format(device.adapter).removesuffix("-srgb")
        self.context.configure(device=device, format=self.format,
                               usage=wgpu.TextureUsage.RENDER_ATTACHMENT, alpha_mode="opaque")
        module = device.create_shader_module(code=SHADER)
        self._pipeline = device.create_render_pipeline(
            layout="auto", vertex={"module": module, "entry_point": "vs"},
            fragment={"module": module, "entry_point": "fs", "targets": [{"format": self.format}]},
            primitive={"topology": "triangle-list"})
        self._view = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._strip = self._texture(1, 1)
        self._binds: dict = {}

    def pixels(self, window) -> tuple[int, int]:
        """The window's size in pixels, which a high-density screen makes larger than its size."""
        w, h = ctypes.c_int(), ctypes.c_int()
        try:
            self._sdl.SDL_GetWindowSizeInPixels(ctypes.c_void_p(self._ptr), ctypes.byref(w), ctypes.byref(h))
        except AttributeError:                   # an SDL before 2.26
            return window.size
        return max(1, w.value), max(1, h.value)

    def _texture(self, w: int, h: int):
        return self.device.create_texture(size=(w, h, 1), format="bgra8unorm",
                                          usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)

    def upload_strip(self, pixels, pitch: int, w: int, h: int) -> None:
        """The strip's latest picture: `pixels`, rows of `pitch` bytes of BGRX."""
        if (self._strip.width, self._strip.height) != (w, h):
            self._strip.destroy()
            self._strip = self._texture(w, h)
            self._binds.clear()
        self.device.queue.write_texture({"texture": self._strip}, pixels,
                                        {"bytes_per_row": pitch, "rows_per_image": h}, (w, h, 1))

    def present(self, frame, size: tuple[int, int], dst, src, strip) -> None:
        """Draw `frame` (a `screen.Stepped`), its `src` rect (x, y, w, h in frame pixels) into
        `dst`, and the strip into `strip`, both rects in the pixels of a window `size` big. A
        window with no frame to draw into (minimised, covered) skips it."""
        pixels = self._size
        if size and size != pixels:
            self.context.set_physical_size(*size)
            self._size = pixels = size
        view = np.array([*dst, *src, *strip, frame.width, frame.height,
                         self._strip.width, self._strip.height], np.float32)
        self.device.queue.write_buffer(self._view, 0, view)
        key = (id(frame.buffer), id(self._strip))
        bind = self._binds.get(key)
        if bind is None:
            if len(self._binds) > 8:
                self._binds.clear()
            bind = self._binds[key] = self.device.create_bind_group(
                layout=self._pipeline.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": {"buffer": frame.buffer, "offset": 0, "size": frame.buffer.size}},
                    {"binding": 1, "resource": self._strip.create_view()},
                    {"binding": 2, "resource": {"buffer": self._view, "offset": 0, "size": 64}}])
        try:
            target = self.context.get_current_texture()
        except wgpu.DrawCancelled:
            return              # minimised, covered or being resized: no frame to draw into
        encoder = self.device.create_command_encoder()
        draw = encoder.begin_render_pass(color_attachments=[{
            "view": target.create_view(), "load_op": "clear", "store_op": "store",
            "clear_value": (0, 0, 0, 1)}])
        draw.set_pipeline(self._pipeline)
        draw.set_bind_group(0, bind)
        draw.draw(3)
        draw.end()
        self.device.queue.submit([encoder.finish()])
        self.context.present()
