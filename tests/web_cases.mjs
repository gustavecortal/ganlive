// The web player's side of tests/test_web.py: reads cases as JSON on stdin, runs them through
// src/ganlive/web, and prints what came out as JSON.

import { readFileSync } from "node:fs";
import { position, walkConfig } from "../src/ganlive/web/clock.mjs";
import { Layout, readout, Settings } from "../src/ganlive/web/dials.mjs";
import { parseNotes, parseTrackChannels } from "../src/ganlive/web/kit.mjs";
import { EncoderMap, formatControls, NoteFeatures, Nrpn, parseControls, parsePressure, PressureMap } from "../src/ganlive/web/midi.mjs";
import { fromDict, PresetRunner, toDict } from "../src/ganlive/web/presets.mjs";
import { seededNoise, SlerpWalk } from "../src/ganlive/web/walk.mjs";

const cases = JSON.parse(readFileSync(0, "utf8"));
const layout = new Layout(cases.knobs);
const out = {};

out.readouts = cases.readouts.map(([name, v]) => readout(name, v, layout));

out.positions = cases.positions.map(([cfg, beats]) => position(walkConfig(cfg), beats));

out.noise = cases.noise.map(([n, seed]) => [...seededNoise(n, seed)]);

out.walk = cases.walk.map(([nz, cfg, rows, amounts, beats]) => {
  const walk = new SlerpWalk(nz, walkConfig({ ...cfg, directions: rows.map((r) => Float32Array.from(r)), amounts }));
  return beats.map((b) => [...walk.latent(b)]);
});

// A preset played: note-ons and knobs in time, the surface and settings read each frame.
out.played = (() => {
  const p = cases.played;
  const features = new NoteFeatures(12, null, parseNotes(p.notes));
  const runner = new PresetRunner(fromDict(p.preset), features.channelOf(), p.fps, layout);
  runner.useModel({ layout, live: cases.live });
  const settings = new Settings(cases.settings);
  const knobs = new EncoderMap(parseControls(p.cc));
  const pads = new PressureMap(parsePressure(p.pressure, parseNotes(p.notes)));
  const frames = [];
  for (const [now, events] of p.frames) {
    for (const [kind, ...args] of events) {
      if (kind === "note") features.onNote(args[0], args[1], args[2], args[3]);
      else if (kind === "cc") knobs.apply(runner, ...args);
      else if (kind === "pad") pads.apply(runner, ...args);
      else if (kind === "hold") runner.hold("console", args[0], 10);
      else if (kind === "wire") runner.wire(args[0], args[1]);
    }
    features.tick(now);
    runner.observe(features.drain());
    runner.apply(features.since, features.features(), settings);
    const w = runner.walk;
    frames.push({ values: { ...runner.surface.values }, settings: [...settings.sent],
      walk: [w.beatsPerSegment, w.spread, w.hold, w.when, w.stepGrid, w.amounts] });
  }
  return { frames, preset: toDict(runner.preset), dropped: runner.dropped };
})();

out.midi = (() => {
  const nrpn = new Nrpn();
  const fed = cases.nrpn.map(([ch, cc, v]) => nrpn.feed(ch, cc, v));
  const learn = new EncoderMap(parseControls("16=noise"));
  learn.learning = "dir1";
  const runner = { hold() {}, free() {} };
  learn.apply(runner, 1, 74, 64);
  return {
    nrpn: fed,
    controls: formatControls(parseControls(cases.controls)),
    learned: formatControls(learn.controls),
    notes: parseNotes(cases.notes),
    channels: parseTrackChannels(cases.channels),
  };
})();

process.stdout.write(JSON.stringify(out));
