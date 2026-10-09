"""The web player (src/ganlive/web) plays a dial, a beat, a walk, a preset and MIDI as the
desktop does: the same cases through both, compared (web_cases.mjs runs the JavaScript)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ganlive.clock import WalkConfig, position
from ganlive.control.features import NoteFeatures
from ganlive.control.kit import parse_notes, parse_track_channels
from ganlive.control.midi import (
    EncoderMap,
    Nrpn,
    PressureMap,
    format_controls,
    parse_controls,
    parse_pressure,
)
from ganlive.dials import fastgan_dials, table
from ganlive.engine.noise import seeded_noise
from ganlive.engine.player import HostSettings, browser_layout, dials_of
from ganlive.presets import PresetRunner, from_dict, to_dict
from ganlive.walk import SlerpWalk

NODE = shutil.which("node")
NZ = 8
ROWS = np.linalg.qr(np.random.default_rng(3).standard_normal((NZ, 3)))[0].T.astype(np.float32)
PROGRAM = {"settings": list(fastgan_dials.SETTINGS_WRITTEN),
           "dials": {"directions": {"basis": ROWS.tolist(), "levels": [30.0, 20.0, 12.0],
                                    "report": ""}, "measured": {}}}
PLAYED = {
    "notes": "36=BD,38=SD,42=CH", "fps": 60.0, "cc": "16=noise,2:n1.3=dir1", "pressure": "SD=se_128",
    "preset": {"dials": {"spread": 0.6, "speed": 0.5},
               "impulses": [{"track": "BD", "dial": "dir2", "amount": 0.4},
                            {"track": "*", "dial": "hold", "amount": -0.2, "attack": 0.05}],
               "macros": [{"source": "density", "dial": "late", "src_lo": 0, "src_hi": 4,
                           "out_lo": 0.2, "out_hi": 0.9}]},
    "frames": [[0.0, []],
               [0.016, [["note", 9, 36, 100, 0.010]]],
               [0.033, [["cc", 0, 16, 90, 127], ["note", 9, 42, 64, 0.030]]],
               [0.050, [["cc", 1, 131, 9000, 16383], ["pad", 0, 38, 70, 127]]],
               [0.066, [["hold", {"se_256": 0.8, "noise": 0.1}], ["pad", 0, 38, 0, 127]]],
               [0.300, [["wire", ["SD"], "spread"], ["note", 9, 38, 127, 0.290]]],
               [0.400, []]],
}


def web(cases: dict) -> dict:
    run = subprocess.run([NODE, str(Path(__file__).with_name("web_cases.mjs"))],
                         input=json.dumps(cases), capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def played(layout, live) -> dict:
    """`PLAYED` through the desktop's own classes."""
    notes = parse_notes(PLAYED["notes"])
    features = NoteFeatures(12, notes=notes)
    runner = PresetRunner(from_dict(PLAYED["preset"]), features.channel_of(), PLAYED["fps"],
                          channels=12, layout=layout)
    runner.use_model(SimpleNamespace(layout=layout, dials_live=live))
    settings = HostSettings(PROGRAM["settings"])
    knobs = EncoderMap(parse_controls(PLAYED["cc"]))
    pads = PressureMap(parse_pressure(PLAYED["pressure"], notes))
    frames = []
    for now, events in PLAYED["frames"]:
        for kind, *args in events:
            if kind == "note":
                features.on_note(*args)
            elif kind == "cc":
                knobs.apply(runner, *args)
            elif kind == "pad":
                pads.apply(runner, *args)
            elif kind == "hold":
                runner.hold("console", args[0], 10)
            elif kind == "wire":
                runner.wire(args[0], args[1])
        features.tick(now)
        runner.observe(features.drain())
        runner.apply(features.since, features.features(), settings)
        w = runner.walk_cfg
        frames.append({"values": dict(runner.surface.values),
                       "settings": settings.committed().tolist(),
                       "walk": [w.beats_per_segment, w.spread, w.hold, w.when, w.step_grid,
                                list(w.amounts)]})
    return {"frames": frames, "preset": to_dict(runner.preset), "dropped": runner.dropped}


def close(a, b, tol=1e-5) -> bool:
    """Equal, numbers within `tol`, at any depth."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol * max(1.0, abs(a))
    return a == b


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
def test_the_web_player_plays_as_the_desktop_does():
    knobs, live = browser_layout(PROGRAM)
    layout, _ = dials_of(PROGRAM)
    rng = np.random.default_rng(7)
    readouts = [(k.name, float(v)) for k in layout.knobs for v in [0.0, 1.0, k.rest, *rng.random(12)]]
    positions = [({"beatsPerSegment": bps, "hold": hold, "when": when, "stepGrid": grid}, float(b))
                 for bps, hold, when, grid in ((4.0, 0.0, 0.5, 0), (1.0, 0.6, 0.9, 8), (0.5, 0.3, 0.1, 16))
                 for b in rng.random(5) * 9]
    walks = [({"spread": 0.35, "homeEvery": 2, "baseSeed": 5}, [0.5, 0.0, -1.25], [0.0, 3.7, 9.2, 13.0]),
             ({"spread": 1.0, "loopSegments": 3, "baseSeed": 2, "beatsPerSegment": 2.0}, [], [1.0, 7.5])]
    cases = {
        "knobs": knobs, "live": live, "settings": PROGRAM["settings"],
        "readouts": readouts,
        "positions": positions,
        "noise": [[16, 12345], [16, 4294967295]],
        "walk": [[NZ, cfg, ROWS.tolist(), amounts, beats] for cfg, amounts, beats in walks],
        "played": PLAYED,
        "nrpn": [[0, 99, 1], [0, 98, 3], [0, 6, 64], [0, 38, 5], [1, 6, 9], [0, 7, 100]],
        "controls": "n1.3=dir1,16=noise,3:17=se_256", "notes": "36=BD,38=SD", "channels": "1-12",
    }
    got = web(cases)

    assert got["readouts"] == [table.readout(n, v, layout) for n, v in readouts]
    for (cfg, beats), js in zip(positions, got["positions"], strict=True):
        py = WalkConfig(beats_per_segment=cfg["beatsPerSegment"], hold=cfg["hold"], when=cfg["when"],
                        step_grid=cfg["stepGrid"])
        assert close(list(position(py, beats)), js)
    for (n, seed), js in zip(cases["noise"], got["noise"], strict=True):
        assert np.allclose(seeded_noise(n, seed), js, atol=1e-5)
    for (cfg, amounts, beats), js in zip(walks, got["walk"], strict=True):
        py = WalkConfig(spread=cfg["spread"], home_every=cfg.get("homeEvery", 0),
                        base_seed=cfg["baseSeed"], loop_segments=cfg.get("loopSegments", 0),
                        beats_per_segment=cfg.get("beatsPerSegment", 4.0),
                        directions=ROWS, amounts=tuple(amounts))
        walk = SlerpWalk(NZ, py)
        for b, z in zip(beats, js, strict=True):
            assert np.allclose(walk.latent(b).reshape(-1), z, atol=1e-4)
    assert close(played(layout, frozenset(live)), got["played"])

    controls = parse_controls(cases["controls"])
    learn = EncoderMap(parse_controls("16=noise"))
    learn.learning = "dir1"
    learn.apply(SimpleNamespace(hold=lambda *a: None), 1, 74, 64)
    nrpn = Nrpn()
    assert got["midi"] == {
        "nrpn": [None if r is None else list(r) for r in (nrpn.feed(*e) for e in cases["nrpn"])],
        "controls": format_controls(controls),
        "learned": format_controls(learn.controls),
        "notes": {str(k): v for k, v in parse_notes(cases["notes"]).items()},
        "channels": {str(k): v for k, v in parse_track_channels(cases["channels"]).items()},
    }


def test_web_take_colour_is_the_desktop_take_colour():
    """A browser take's I420 (take.mjs) and the desktop's NV12 (engine/screen.py) convert
    colour with the same maths, so a take looks the same from either."""
    from ganlive.engine.screen import NV12

    js = (Path(__file__).parents[1] / "src/ganlive/web/take.mjs").read_text(encoding="utf-8")
    for line in ("const KR = 0.2126; const KB = 0.0722; const KG = 1.0 - 0.2126 - 0.0722;",
                 "* 2.0 - 1.0;", "* 109.5 + 125.5), 0.0, 255.0));",
                 "(c.b - l) * (224.0 / (4.0 * (1.0 - KB)))", "(c.r - l) * (224.0 / (4.0 * (1.0 - KR)))",
                 "return u32(clamp(round(v + 128.0), 0.0, 255.0));"):
        assert line in NV12 and line in js, line
