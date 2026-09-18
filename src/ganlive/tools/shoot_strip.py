"""The strip as a picture, headless, in every mode -- the check no test performs.

    ganlive shoot-strip runs/my-run --out runs/strip

Four layout defects so far have been invisible in code and obvious in a PNG. Two traps, both
paid for once: draw through the renderer rather than the strip's own surface, or the routing
cells, dial bars and drum lights are all missing; and build the bank for real, or a stubbed
`knobs.index` darkens exactly the dials you already assumed it would.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Before pygame is imported, or SDL opens a window on his screen.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
from pygame._sdl2.video import Renderer, Window  # noqa: E402

from ganlive import bank  # noqa: E402
from ganlive.control.midi import EncoderMap  # noqa: E402
from ganlive.control.tracks import INDEX, channel_map  # noqa: E402
from ganlive.presets import DEFAULT, PresetRunner  # noqa: E402
from ganlive.strip import PRIORITY, SOURCE, WIDTH, DialPanel  # noqa: E402

MODES = ("dials", "routing", "models")


def dressed(runner, model, knobs=None) -> None:
    """A strip mid-performance rather than at rest: hands down, drums wired, a knob parked.

    At rest every bar sits on its own rest and the routing grid is empty, which is the one
    state the layout is easiest to get right -- and the state a picture proves least about.
    """
    live = [n for n in model.layout if n in model.dials_live]
    if not live:
        return
    runner.hold(SOURCE, {live[1]: 0.72, live[-1]: 0.34}, PRIORITY)
    for track, dial in (("BD", live[-1]), ("CH", live[-2]), ("SD", live[1])):
        try:
            runner.wire([track], dial)
        except KeyError:
            pass                      # a dial this model cannot write; the refusal is the point
    if knobs is not None:
        # The description line names the knob on a dial as well as the drums, and a strip with
        # no map behind it is the one state that cannot show it.
        knobs.controls = {(-1, 16): live[-1], (1, 17): live[1]}


def shoot(panel, renderer, tall: int, out: Path, name: str) -> Path:
    strip = (0, 0, WIDTH, tall)
    panel._dirty = True
    renderer.draw_color = (0, 0, 0, 255)
    renderer.clear()
    panel.draw(renderer, strip)
    shot = pygame.Surface((WIDTH, tall))
    shot.blit(renderer.to_surface(), (0, 0), pygame.Rect(0, 0, WIDTH, tall))
    path = out / name
    pygame.image.save(shot, str(path))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ganlive shoot-strip", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="+", type=Path,
                    help="a checkpoint, a run directory, or an exported graph. Several loads a "
                         "bank, and the strip is shot on each")
    ap.add_argument("--out", type=Path, default=Path("runs/ganlive/strip"))
    ap.add_argument("--device", default=None, help="default: whichever accelerator is there")
    ap.add_argument("--heights", type=int, nargs="*", default=None,
                    help="window heights. The default is the layout's own floor and 1200: the "
                         "floor is where a block runs off the bottom, and nothing enforces it "
                         "for a window that is dragged shorter")
    ap.add_argument("--layout", default="voices", help="which kit map the routing grid draws")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    r = bank.build(args.checkpoint, args.device)
    runner = PresetRunner(DEFAULT, channel_map(args.layout) or INDEX, 60.0)
    shelf = bank.Shelf(r, Path("runs"))

    pygame.init()
    window = Window("ganlive strip", size=(WIDTH + 40, 1500))
    renderer = Renderer(window, vsync=False)
    since = np.full(runner.channels, 1e6, dtype=np.float32)

    knobs = EncoderMap({})
    panel = None
    for index in range(len(r.models)):
        model = r.use(index)
        runner.use_model(model)
        # **One strip, switched underneath, because that is what the instrument is.** Building a
        # fresh `DialPanel` per model lays it out from scratch every time, so it is always right
        # -- and the pictures then prove nothing about a switch. They were taken that way while
        # the rows were a function of the layout and the cache was keyed on the window alone.
        if panel is None:
            panel = DialPanel(runner, bank=r, shelf=shelf, encoders=knobs,
                              actions={"preset": lambda d: None, "save": lambda f: None,
                                       "record": lambda: None, "still": lambda: None,
                                       "model": lambda d: None})
            panel.attach(renderer)
        dressed(runner, model, knobs)
        # What `_grid_gesture` does after it wires one. Without it the grid draws the rules the
        # panel was built with, and a picture of an empty grid reads as a mode that does not work.
        panel.reload()
        runner.apply(since, {"density": 0.6, "energy": 0.45, "active": 0.3}, model.knobs)

        heights = args.heights or [panel.floor_height(), 1200]
        slug = bank.slug_for(model.path)
        for mode in MODES:
            panel.mode = mode
            for tall in heights:
                print(f"  {shoot(panel, renderer, tall, args.out, f'{slug}-{mode}-{tall}.png')}",
                      flush=True)
    print(f"\n{len(r.models) * len(MODES) * len(heights)} shots in {args.out}")
    return 0

