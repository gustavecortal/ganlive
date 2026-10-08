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
 */
export async function loadModel(device, program, weights, { cache = new Map(), onProgress } = {}) {
  if (program.format !== FORMAT) throw new Error(`a ${program.format} program; this runner plays ${FORMAT}`);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const buffers = {};
  const destroy = () => Object.values(buffers).forEach((b) => b.destroy());

  device.pushErrorScope("validation");
  try {
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
      output: buffers.out, height: program.height, width: program.width, steps, destroy,
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
