"""Render the control strip to PNGs, headless, in every mode and for every model in a bank.

A developer script, for layout checks and README screenshots:

    python scripts/shoot_strip.py runs/my-run --out runs/strip

Layout defects are easier to see in a picture than in code. The shots draw through the real
renderer, since the bars, routing cells and lights are drawn there rather than into the strip's
own surface, and use a real bank, so the dark dials are the model's own.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ganlive.checkpoints import slug_for
from ganlive.control.kit import INDEX, channel_map
from ganlive.control.midi import EncoderMap
from ganlive.presets import DEFAULT, PresetRunner
from ganlive.strip import PRIORITY, SOURCE, WIDTH, DialPanel
from ganlive.tools import add_device

MODES = ("dials", "routing", "models")


def dressed(runner, model, knobs=None) -> None:
    """A strip mid-performance rather than at rest: dials held, drums wired, a knob parked.
    At rest every bar sits on its rest and the grid is empty, which proves the least."""
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
        # So the description line has a knob to name as well as the drums.
        knobs.controls = {(-1, 16): live[-1], (1, 17): live[1]}


def shoot(panel, renderer, tall: int, out: Path, name: str) -> Path:
    """Draw the strip `tall` pixels high and save it as `out/name`."""
    import pygame

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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="+", type=Path,
                    help="a checkpoint, a run directory, or an exported graph. Several loads a "
                         "bank, and the strip is shot on each")
    ap.add_argument("--out", type=Path, default=Path("runs/ganlive/strip"))
    add_device(ap)
    ap.add_argument("--heights", type=int, nargs="*", default=None,
                    help="window heights. The default is the layout's own floor and 1200: the "
                         "floor is where a block runs off the bottom, and nothing enforces it "
                         "for a window that is dragged shorter")
    ap.add_argument("--layout", default="voices", help="which kit map the routing grid draws")
    args = ap.parse_args(argv)

    # Before pygame is imported, so SDL opens no window on the screen.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame
    from pygame._sdl2.video import Renderer, Window

    from ganlive import bank

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
        # One strip, switched underneath as ganlive does, so the shots also show
        # that a switch lays the strip out again.
        if panel is None:
            panel = DialPanel(runner, bank=r, shelf=shelf, encoders=knobs,
                              actions={"preset": lambda d: None, "save": lambda f: None,
                                       "record": lambda: None, "still": lambda: None,
                                       "model": lambda delta=0, to=None: None})
            panel.attach(renderer)
        dressed(runner, model, knobs)
        # Re-read the rules `dressed` wired, as `_grid_gesture` does after a click.
        panel.reload()
        runner.apply(since, {"density": 0.6, "energy": 0.45, "active": 0.3}, model.settings)

        heights = args.heights or [panel.floor_height(), 1200]
        slug = slug_for(model.path)
        for mode in MODES:
            panel.mode = mode
            for tall in heights:
                print(f"  {shoot(panel, renderer, tall, args.out, f'{slug}-{mode}-{tall}.png')}",
                      flush=True)
    print(f"\n{len(r.models) * len(MODES) * len(heights)} shots in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
