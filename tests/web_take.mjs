// tests/test_web.py's take: a walk played across two models, with the hand on a direction and
// the settings changing, logged as a take records it and read back. Prints the frames logged,
// the frames read, and the frames whose latent or settings came back different.

import { walkConfig } from "../src/ganlive/web/clock.mjs";
import { readLog, Recording } from "../src/ganlive/web/take.mjs";
import { compose, SlerpWalk } from "../src/ganlive/web/walk.mjs";

const nz = 16;
const rows = [1, 2, 3].map((r) => Float32Array.from({ length: nz }, (_, i) => Math.sin(i * r)));
const models = ["lichen", "amber"].map((name) => ({ name, width: 8, height: 8, directions: rows }));
const cfg = walkConfig({ beatsPerSegment: 1, spread: 0.4, directions: rows, amounts: [0, 0, 0] });
const walk = new SlerpWalk(nz, cfg);
const settings = { sent: new Float32Array([1, 1]) };
const take = new Recording();
const drawn = [];
for (let i = 0; i < 400; i++) {
  const m = models[i < 250 ? 0 : 1];
  cfg.amounts = [i > 100 ? Math.sin(i / 30) : 0, 0.5, 0];
  cfg.spread = i < 200 ? 0.4 : 0.9;
  if (i % 50 === 0) settings.sent = Float32Array.from([i / 400, 1]);
  const z = walk.latent(i * 0.05);
  take.log(m, walk, settings);
  drawn.push([m.name, z, settings.sent.slice()]);
}
const { log, frames } = await take.stop();
let read = 0;
let model;
const wrong = [];
for (const item of readLog(log)) {
  if (item.model) {
    model = models.find((m) => m.name === item.model);
    continue;
  }
  const z = compose(item.z0, item.a, item.z1, item.b, model.directions, item.pushes);
  const [name, latent, k] = drawn[read];
  if (name !== model.name || z.some((v, j) => v !== latent[j]) || item.settings.some((v, j) => v !== k[j]))
    wrong.push(read);
  read++;
}
process.stdout.write(JSON.stringify({ frames, read, wrong, bytes: log.length }));
