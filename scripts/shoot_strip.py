"""Render the control strip to PNGs, headless, in every mode and for every model in a bank.

A developer script, for layout checks and README screenshots:

    python scripts/shoot_strip.py runs/my-run --out runs/strip

Layout defects are easier to see in a picture than in code. The shots are the strip the window
draws (`DialPanel.compose`), and use a real bank, so the dark dials are the model's own.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ganlive.checkpoints import slug_for
from ganlive.control.features import NEVER
from ganlive.control.kit import INDEX, channel_map
from ganlive.control.midi import EncoderMap
from ganlive.presets import DEFAULT, PresetRunner
from ganlive.strip import (
    MODE_DIALS,
    MODE_MODELS,
    MODE_ROUTING,
    PRIORITY,
    SOURCE,
    WIDTH,
    DialPanel,
)
from ganlive.tools import add_backend

MODES = (MODE_DIALS, MODE_ROUTING, MODE_MODELS)


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


def shoot(panel, tall: int, out: Path, name: str) -> Path:
    """Compose the strip `tall` pixels high and save it as `out/name`."""
    import pygame

    panel._dirty = True
    path = out / name
    pygame.image.save(panel.compose(WIDTH, tall), str(path))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="+", type=Path,
                    help="a checkpoint, a run directory, or an engine model. Several loads a "
                         "bank, and the strip is shot on each")
    ap.add_argument("--out", type=Path, default=Path("runs/ganlive/strip"))
    add_backend(ap)
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

    from ganlive import bank
    from ganlive.families import LoadOptions

    args.out.mkdir(parents=True, exist_ok=True)
    r = bank.build(args.checkpoint, options=LoadOptions(backend=args.backend))
    runner = PresetRunner(DEFAULT, channel_map(args.layout) or INDEX, 60.0)
    shelf = bank.Shelf(r, Path("runs"))

    pygame.display.init()
    since = np.full(runner.channels, NEVER, dtype=np.float32)

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
            panel.attach()
        dressed(runner, model, knobs)
        # Re-read the rules `dressed` wired, as `_grid_gesture` does after a click.
        panel.reload()
        runner.apply(since, {"density": 0.6, "energy": 0.45, "active": 0.3}, model.settings)

        heights = args.heights or [panel.floor_height(), 1200]
        slug = slug_for(model.path)
        for mode in MODES:
            panel.mode = mode
            for tall in heights:
                print(f"  {shoot(panel, tall, args.out, f'{slug}-{mode}-{tall}.png')}",
                      flush=True)
    print(f"\n{len(r.models) * len(MODES) * len(heights)} shots in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
