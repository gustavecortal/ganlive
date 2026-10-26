import { test } from "node:test";
import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import { loadModel } from "./runner.mjs";

globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4 };
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const program = {
  format: "ganlive-engine/1", width: 1, height: 1,
  buffers: { P: { size: 4, init: "weights" }, out: { size: 4 } },
  shaders: ["shared", "frame"],
  load: [{ name: "init", shader: 0, bind: ["P", "out"], groups: [1, 1, 1] }],
  steps: [0, 1].map((shader) => ({ name: "draw", shader, bind: ["P", "out"], groups: [1, 1, 1] })),
};

function setup() {
  const buffers = [], pipelines = [], bindings = [], submitted = [], stages = [], cache = new Map();
  let controller, popped = 0;
  const weights = new Response(new ReadableStream({ start(c) { controller = c; } }));
  const device = {
    limits: { maxStorageBufferBindingSize: 1024, maxBufferSize: 1024 },
    pushErrorScope() {},
    async popErrorScope() { popped++; return null; },
    createShaderModule: (spec) => spec,
    createComputePipelineAsync(spec) {
      const pending = deferred();
      pipelines.push({ ...pending, spec });
      return pending.promise;
    },
    createBuffer({ size }) {
      const data = new ArrayBuffer(size);
      const buffer = { destroyed: false, getMappedRange: () => data, unmap() {},
        destroy() { this.destroyed = true; } };
      buffers.push(buffer);
      return buffer;
    },
    createBindGroup(spec) {
      assert.ok(spec.entries.every((e) => e.resource.buffer && !e.resource.buffer.destroyed));
      bindings.push(spec);
      return spec;
    },
    createCommandEncoder() {
      const dispatches = [];
      return {
        beginComputePass: () => ({ setPipeline() {}, setBindGroup() {},
          dispatchWorkgroups: (...groups) => dispatches.push(groups), end() {} }),
        finish: () => dispatches,
      };
    },
    queue: { writeBuffer() {}, submit: (commands) => submitted.push(commands) },
  };
  return { device, buffers, pipelines, bindings, submitted, stages, cache, weights,
    get popped() { return popped; },
    finishDownload() { controller.enqueue(new Uint8Array([1, 2, 3, 4])); controller.close(); },
    failDownload() { controller.error(new Error("offline")); },
    finishCompilation() { for (const p of pipelines) p.resolve({ getBindGroupLayout: () => ({}) }); },
    load: () => loadModel(device, program, weights, { cache, onStage: (s) => stages.push(s) }),
  };
}

test("compilation overlaps the weight stream, and drawing waits for both", async () => {
  const s = setup(), loading = s.load();
  assert.equal(s.pipelines.length, 2, "identical sources compile once");
  assert.deepEqual(s.stages, ["download"]);
  assert.equal(s.bindings.length, 0);
  s.finishDownload();
  await setImmediate();
  assert.deepEqual(s.stages, ["download", "compile"]);
  assert.equal(s.submitted.length, 0, "download alone cannot start the model");
  s.finishCompilation();
  const model = await loading;
  assert.equal(s.bindings.length, 3);
  assert.equal(s.submitted.length, 1);
  const frame = s.device.createCommandEncoder();
  model.encode(frame);
  assert.equal(frame.finish().length, 2);
  model.destroy();
  assert.ok(s.buffers.every((b) => b.destroyed));
});

test("a compile failure during download is handled and releases the buffers", async () => {
  const s = setup(), loading = s.load();
  const rejected = assert.rejects(loading, /bad shader/);
  s.pipelines[0].reject(new Error("bad shader"));
  await setImmediate(); // an early rejection must not become an unhandled promise
  s.finishDownload();
  s.finishCompilation();
  await rejected;
  assert.equal(s.cache.has("shared"), false);
  assert.ok(s.buffers.every((b) => b.destroyed));
  assert.equal(s.popped, 1);
  assert.equal(s.submitted.length, 0);
});

test("a broken weight stream releases buffers while compilation is pending", async () => {
  const s = setup(), loading = s.load();
  const rejected = assert.rejects(loading, /offline/);
  s.failDownload();
  await rejected;
  assert.ok(s.buffers.every((b) => b.destroyed));
  s.pipelines[0].reject(new Error("late compiler failure"));
  s.finishCompilation();
  await setImmediate();
  assert.equal(s.popped, 1);
  assert.equal(s.submitted.length, 0);
});
