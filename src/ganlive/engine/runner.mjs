// Runs a ganlive engine program (program.json, from `ganlive convert`) on a WebGPU device.
// The shaders and the dispatches of a frame come from the program; this file only creates
// buffers and pipelines and records frames. runner.py does the same through wgpu-py.
//
//   const model = await loadModel(device, program, await fetch("weights.bin"));
//   model.setLatent(z); model.setSettings(k);
//   const enc = device.createCommandEncoder(); model.encode(enc); device.queue.submit([enc.finish()]);
//   // model.output: packed RGBA8 pixels, height x width (or f32 planes with output: "f32")

/**
 * `weights`: an ArrayBuffer, or a fetch Response streamed straight into GPU memory
 * (`onProgress(bytes)` reports it). Options: `output` "rgba8" or "f32"; `noise` {layer: Float32Array}
 * to use instead of the seeded noise made on the GPU; `cache` a Map of compiled pipelines.
 */
export async function loadModel(device, program, weights, options = {}) {
  const { output = "rgba8", noise = {}, cache = new Map(), onProgress } = options;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const owned = [];
  const destroy = () => owned.forEach((b) => b.destroy());
  const create = (size, mapped = false) => {
    if (size > limit) {
      throw new Error(`a ${Math.round(size / 2 ** 20)} MiB buffer is over this device's ` +
        `${Math.round(limit / 2 ** 20)} MiB limit; request higher limits or use a smaller model`);
    }
    const b = device.createBuffer({ size, usage, mappedAtCreation: mapped });
    owned.push(b);
    return b;
  };

  device.pushErrorScope("validation");
  try {
    const { buffer: outName, size: outSize, step: last } = program.outputs[output];
    const buffers = { [outName]: create(outSize) };
    for (const [name, spec] of Object.entries(program.buffers)) {
      if (spec.init === "weights") buffers[name] = await upload(spec.size, weights);
      else buffers[name] = create(spec.size);
      if (spec.init === "ones") device.queue.writeBuffer(buffers[name], 0, new Float32Array(spec.size / 4).fill(1));
      const layer = name.startsWith("noise.") && name.slice(6);
      if (spec.init === "noise") {
        if (noise[layer]) device.queue.writeBuffer(buffers[name], 0, noise[layer]);
        else seed(buffers[name], spec.size / 4, spec.seed);
      }
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
    const specs = [...program.steps, last];
    const pipelines = await Promise.all(specs.map((s) => pipeline(s.shader)));
    const steps = specs.map((s, i) => ({ name: s.name, plan: s.plan, groups: s.groups, pipeline: pipelines[i],
      bind: device.createBindGroup({ layout: pipelines[i].getBindGroupLayout(0),
        entries: s.bind.map((n, j) => ({ binding: j, resource: { buffer: buffers[n] } })) }) }));
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);

    const run = (pass, s) => {
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bind);
      pass.dispatchWorkgroups(s.groups[0], s.groups[1], s.groups[2]);
    };
    return {
      output: buffers[outName], height: program.height, width: program.width, steps, destroy,
      setLatent: (z) => device.queue.writeBuffer(buffers.Z, 0, z),
      setSettings: (k) => device.queue.writeBuffer(buffers.K, 0, k),
      /** Records one frame. With `timestamps` (a GPUQuerySet), one timed pass per step. */
      encode(encoder, timestamps) {
        if (!timestamps) {
          const pass = encoder.beginComputePass();
          for (const s of steps) run(pass, s);
          pass.end();
          return;
        }
        steps.forEach((s, i) => {
          const pass = encoder.beginComputePass({ timestampWrites: { querySet: timestamps,
            beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
          run(pass, s);
          pass.end();
        });
      },
    };
  } catch (e) {
    await device.popErrorScope().catch(() => {});    // gone already if it was popped above
    destroy();                                       // a failed load leaks nothing
    throw e;
  }

  async function upload(size, source) {
    if (source instanceof ArrayBuffer) {
      const b = create(size);
      device.queue.writeBuffer(b, 0, source);
      return b;
    }
    // A Response: stream it into a buffer mapped at creation, so no copy of it sits in JS.
    const b = create(size, true);
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
    if (at !== Math.ceil(size / 4) * 4 && at !== size) throw new Error(`weights: got ${at} bytes, the program expects ${size}`);
    return b;
  }

  function seed(buffer, n, value) {
    const code = program.noise.replace("${n}", String(n)).replace(/\$\{seed\}/g, String(value));
    const p = device.createComputePipeline({ layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    const groups = Math.ceil(n / 256);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }] }));
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    pass.end();
    device.queue.submit([enc.finish()]);
  }
}
