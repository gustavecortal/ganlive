// Runs a ganlive engine program (program.json, from `ganlive convert`) on a WebGPU device.
// The shaders and the dispatches of a frame come from the program; this file only creates
// buffers and pipelines and records frames. runner.py does the same through wgpu-py.
//
//   const model = await loadModel(device, program, await fetch("weights.bin"));
//   model.setLatent(z); model.setSettings(k);
//   const enc = device.createCommandEncoder(); model.encode(enc); device.queue.submit([enc.finish()]);
//   // model.output: packed RGBA8 pixels, height x width (f32 planes for an "f32" program)

const FORMAT = "ganlive-engine/1";

/**
 * `weights`: an ArrayBuffer, or a fetch Response streamed straight into GPU memory
 * (`onProgress(bytes)` reports it). `cache`: a Map of compiled pipelines to share between loads.
 * `onStage(name)` is told when "download" and then "compile" begin.
 */
export async function loadModel(device, program, weights, { cache = new Map(), onProgress, onStage } = {}) {
  if (program.format !== FORMAT) throw new Error(`a ${program.format} program; this runner plays ${FORMAT}`);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const buffers = {};
  const destroy = () => Object.values(buffers).forEach((b) => b.destroy());

  device.pushErrorScope("validation");
  try {
    onStage?.("download");
    for (const [name, spec] of Object.entries(program.buffers)) {
      if (spec.size > limit) {
        throw new Error(`${name}: a ${Math.round(spec.size / 2 ** 20)} MiB buffer is over this ` +
          `device's ${Math.round(limit / 2 ** 20)} MiB limit; request higher limits or use a smaller model`);
      }
      const streamed = spec.init === "weights" && !(weights instanceof ArrayBuffer);
      const b = buffers[name] = device.createBuffer({ size: spec.size, usage, mappedAtCreation: streamed });
      if (spec.init === "weights") await upload(b, spec.size, weights);
      else if (spec.init === "ones") device.queue.writeBuffer(b, 0, new Float32Array(spec.size / 4).fill(1));
      else if (spec.init === "words") device.queue.writeBuffer(b, 0, new Uint32Array(spec.words));
    }
    onStage?.("compile");
    // Every shader compiles in parallel; identical sources share one compile.
    const pipeline = (index) => {
      const code = program.shaders[index];
      if (!cache.has(code)) {
        const p = device.createComputePipelineAsync({ layout: "auto",
          compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
        p.catch(() => cache.delete(code));
        cache.set(code, p);
      }
      return cache.get(code);
    };
    const compiled = async (specs) => {
      const pipelines = await Promise.all(specs.map((s) => pipeline(s.shader)));
      return specs.map((s, i) => ({ name: s.name, groups: s.groups, pipeline: pipelines[i],
        bind: device.createBindGroup({ layout: pipelines[i].getBindGroupLayout(0),
          entries: s.bind.map((n, j) => ({ binding: j, resource: { buffer: buffers[n] } })) }) }));
    };
    const [load, steps] = await Promise.all([compiled(program.load), compiled(program.steps)]);
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);

    const run = (pass, s) => {
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bind);
      pass.dispatchWorkgroups(s.groups[0], s.groups[1], s.groups[2]);
    };
    const once = device.createCommandEncoder();     // the noise
    const pass = once.beginComputePass();
    for (const s of load) run(pass, s);
    pass.end();
    device.queue.submit([once.finish()]);

    return {
      output: buffers.out, height: program.height, width: program.width, format: program.output,
      steps, destroy,
      setLatent: (z) => device.queue.writeBuffer(buffers.Z, 0, z),
      setSettings: (k) => device.queue.writeBuffer(buffers.K, 0, k),
      /** Records one frame. With `timestamps` (a GPUQuerySet), one timed pass per step,
       *  writing queries from 2 * `first` on, so that one set can time many frames. */
      encode(encoder, timestamps, first = 0) {
        if (!timestamps) {
          const frame = encoder.beginComputePass();
          for (const s of steps) run(frame, s);
          frame.end();
          return;
        }
        steps.forEach((s, i) => {
          const timed = encoder.beginComputePass({ timestampWrites: { querySet: timestamps,
            beginningOfPassWriteIndex: 2 * (first + i), endOfPassWriteIndex: 2 * (first + i) + 1 } });
          run(timed, s);
          timed.end();
        });
      },
    };
  } catch (e) {
    await device.popErrorScope().catch(() => {});    // gone already if it was popped above
    destroy();                                       // a failed load leaks nothing
    throw e;
  }

  async function upload(b, size, source) {
    if (source instanceof ArrayBuffer) {
      if (source.byteLength !== size) throw new Error(`weights: ${source.byteLength} bytes, the program expects ${size}`);
      device.queue.writeBuffer(b, 0, source);
      return;
    }
    // A Response: streamed into the buffer, mapped at creation, so no copy of it sits in JS.
    const view = new Uint8Array(b.getMappedRange());
    let at = 0;
    for (const reader = source.body.getReader(); ;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (at + value.byteLength > size) throw new Error(`weights: more than the ${size} bytes the program expects`);
      view.set(value, at);
      at += value.byteLength;
      onProgress?.(at);
    }
    b.unmap();
    if (at !== size) throw new Error(`weights: got ${at} bytes, the program expects ${size}`);
  }
}

// A frame onto a canvas of any size and format, as ganlive's desktop window draws one
// (engine/present.py): shrinking averages the frame pixels a canvas pixel covers, growing is
// bilinear. Drawn rather than copied, so the canvas takes the format its browser prefers and
// the size it is shown at.
const SCREEN = (order) => `
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<uniform> V: vec4f;      // frame width, height, canvas width, height
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  return vec4f(f32((i << 1u) & 2u) * 2.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
fn px(y: u32, x: u32) -> vec3f { return unpack4x8unorm(X[y * u32(V.x) + x]).${order}; }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let W = u32(V.x); let H = u32(V.y);
  let s = V.xy / V.zw;                               // frame pixels per canvas pixel
  if (s.x >= 1.0 && s.y >= 1.0) {
    let lo = (pos.xy - 0.5) * s;
    let x0 = u32(lo.x); let y0 = u32(lo.y);
    let x1 = min(W, max(x0 + 1u, u32(ceil(lo.x + s.x)))); let y1 = min(H, max(y0 + 1u, u32(ceil(lo.y + s.y))));
    var c = vec3f(0.0);
    for (var y = y0; y < y1; y++) { for (var x = x0; x < x1; x++) { c += px(y, x); } }
    return vec4f(c / f32((y1 - y0) * (x1 - x0)), 1.0);
  }
  let f = clamp(pos.xy * s - 0.5, vec2f(0.0), V.xy - 1.0);
  let x0 = u32(f.x); let y0 = u32(f.y); let x1 = min(x0 + 1u, W - 1u); let y1 = min(y0 + 1u, H - 1u);
  let t = f - floor(f);
  return vec4f(mix(mix(px(y0, x0), px(y0, x1), t.x), mix(px(y1, x0), px(y1, x1), t.x), t.y), 1.0);
}`;

/** Draws `model`'s frame onto canvas textures of `format`: `draw(encoder, texture)`. */
export function screen(device, model, format) {
  if (model.format === "f32") throw new Error("an f32 program draws planes, not pixels");
  const module = device.createShaderModule({ code: SCREEN(model.format === "bgra8" ? "zyx" : "xyz") });
  const pipeline = device.createRenderPipeline({ layout: "auto",
    vertex: { module, entryPoint: "vs" }, fragment: { module, entryPoint: "fs", targets: [{ format }] } });
  const view = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: model.output } }, { binding: 1, resource: { buffer: view } }] });
  let size = "";
  return {
    draw(encoder, texture) {
      if (size !== `${texture.width}x${texture.height}`) {
        size = `${texture.width}x${texture.height}`;
        device.queue.writeBuffer(view, 0, new Float32Array([model.width, model.height, texture.width, texture.height]));
      }
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(),
        loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
    },
  };
}
