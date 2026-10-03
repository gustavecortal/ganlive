"""Drive the strip the way a pair of hands does, headless, and see what falls over.

A real window under SDL's dummy driver, so the real renderer, the real fonts and the real
paint path all run -- only the screen is missing. Every event goes through `DialPanel.handle`
exactly as `Display` delivers it, and in the order `Display` delivers it: paint, present,
then read what arrived. The defects worth finding here live in the gap between a gesture that
changes state and the paint that reads it back a frame later, and a harness that reorders
those two finds faults the window cannot produce.

**Model switches are the point.** A switch retires dials -- one checkpoint keeps four of its
eight directions, the next keeps three, a converted StyleGAN2 has no SLE gate at all -- while
the focus, the drag and the mode carry across. That is the state that crashed the window.

    pytest tests/test_fuzz_surface.py -q
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import os
import pathlib
import random
import sys
import traceback
import types

# Before pygame is imported, or SDL takes the real driver and opens a window on his screen.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from pygame._sdl2.video import Renderer, Window  # noqa: E402

from ganlive.control.kit import INDEX  # noqa: E402
from ganlive.dials import table as S  # noqa: E402
from ganlive.dials.fastgan_dials import fastgan  # noqa: E402
from ganlive.presets import DEFAULT, PresetRunner  # noqa: E402
from ganlive.strip import WIDTH, DialPanel, floor_height  # noqa: E402

#: The layouts a bank can hold at once, including the awkward ones: a single direction, and a
#: family whose MODEL block shares no dial name with this one.
LAYOUTS = {
    "fastgan 8 dirs": fastgan(),
    "fastgan 4 dirs": fastgan(directions=4),
    "fastgan 1 dir": fastgan(directions=1),
    "stylegan2": S.stylegan2(),
}
#: What a model reports as live. `none` is the model whose every dial measured under the floor;
#: `stale names` is a set naming dials the layout beside it does not have, which is what a bank
#: holding two families looks like if the two ever get crossed.
LIVE = {
    "all": None,
    "none": frozenset(),
    "directions only": frozenset({"dir1", "dir2"}),
    "stale names": frozenset({"se_128", "dir7", "reaction", "speed"}),
}
#: Every key the strip claims, plus three it does not, plus tab and return.
KEYS = "gmlspvtr123 \t\r"
#: Including one below the floor the window is supposed to enforce, because it is resizable.
#: The shortest window FastGAN's own surface is whole in -- the most crowded layout
#: here -- and one shorter than that, which is the case the floor exists to catch.
MIN_H = floor_height(fastgan().groups)
HEIGHTS = (MIN_H - 120, MIN_H, 900, 1440)


@dataclasses.dataclass
class Entry:
    """One row of the picker, as `bank.Shelf` hands them over."""

    name: str
    path: pathlib.Path
    loaded: bool = False
    why: str = ""
    note: str = "on disk"


class Shelf:
    """The picker's list, without a disk under it."""

    def __init__(self, count: int = 19) -> None:
        self._entries = [Entry(f"model-{i}", pathlib.Path(f"m{i}.pt"), loaded=i < 2,
                               why="different latent width" if i == 5 else "")
                         for i in range(count)]
        self.pending = None
        self.note = ""

    def entries(self):
        return self._entries

    def request(self, entry) -> None:
        self.pending = str(entry.path)


class Knobs:
    """Takes any write and keeps the last of each, like the real settings vector."""

    def __init__(self) -> None:
        self.index, self.values = {}, {}

    def set(self, name: str, value: float) -> None:
        self.values[name] = value

    def commit(self) -> None:
        pass

    def view(self, name: str):
        return None


def stub_model(layout, live):
    """One prepared model, as `bank.Model` presents itself to the strip."""
    return types.SimpleNamespace(
        layout=layout, dials_live=frozenset(layout) if live is None else live,
        directions=types.SimpleNamespace(levels=(100.0, 50.0, 25.0)),
        path=pathlib.Path("stub.pt"), name="stub")


def switch(bank, runner, layout, live, between=None):
    """Play a different model behind whatever strip is already drawing this one.

    **`play.switch_model`'s two statements, and the gap between them.** Building a fresh
    `DialPanel` per switch -- which is what this harness used to do -- cannot reach the state a
    switch leaves: one strip that is already laid out, for the model that has just left. A new
    strip lays itself out on its first paint and is therefore always right, so the defect where
    the rows are a function of the layout and the cache was keyed on the window alone read as
    health here for as long as the harness existed.

    `between` is called after `Bank.use` and before `PresetRunner.use_model`, because those two
    run on the frame loop's thread and nothing holds the window's thread off between them: for
    one paint the bank names the incoming model's dials and the surface still carries the
    outgoing model's values."""
    bank.current = stub_model(layout, live)
    if between is not None:
        between()
    runner.use_model(bank.current)


def build(layout, live, shelf):
    """A strip with a stub model behind it, wired the way `ganlive play` wires the real one."""
    runner = PresetRunner(DEFAULT, INDEX, 60.0, layout=fastgan())
    bank = types.SimpleNamespace(current=None, models=[1, 2], index=0, name="stub",
                                index_of=lambda path: 0)
    switch(bank, runner, layout, live)
    panel = DialPanel(runner, bank=bank, shelf=shelf,
                      actions={"preset": lambda delta: None, "save": lambda found: None,
                               "record": lambda: None, "still": lambda: None,
                               "model": lambda delta=0, to=None: None})
    return panel


def gesture(rng, w: int, h: int):
    """One plausible thing a hand does: a click with its release, a wheel, a move, a key.

    Presses are paired with releases because an unpaired press leaves `_drag` set, and a strip
    that only ever saw unpaired presses would never exercise the release path at all. Positions
    run past the edges on purpose -- the strip is handed events from the whole window.
    """
    roll = rng.random()
    pos = (rng.randrange(-20, w + 20), rng.randrange(-20, h + 20))
    if roll < 0.42:
        return [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=rng.choice((1, 2, 3)),
                                   pos=pos),
                pygame.event.Event(pygame.MOUSEMOTION,
                                   pos=(pos[0] + rng.randrange(-80, 80),
                                        pos[1] + rng.randrange(-40, 40))),
                pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos)]
    if roll < 0.60:
        return [pygame.event.Event(pygame.MOUSEWHEEL, y=rng.choice((-3, -1, 1, 3)), x=0,
                                   pos=pos)]
    if roll < 0.75:
        return [pygame.event.Event(pygame.MOUSEMOTION, pos=pos)]
    key = rng.choice(KEYS)
    return [pygame.event.Event(pygame.KEYDOWN, key=pygame.key.key_code(key), mod=0,
                               unicode=key)]


def check(panel) -> None:
    """What has to be true of the strip after every paint, whatever gesture preceded it.

    **A defect does not have to raise to be one.** When the rows stopped being the loaded
    model's dials nothing crashed: the window drew the departed model's names, every one of
    them dark, and offered no row at all for the model you had just switched to. A harness that
    only catches tracebacks reads that as a clean run, which is what this one did for as long as
    it existed. Raised rather than returned so it lands in the same report as a crash."""
    rows = {name for name, _top, _tall in panel._rows}
    theirs = set(panel.dials)
    if rows != theirs:
        raise AssertionError(
            f"the strip's rows are not the loaded model's dials: drawing {sorted(rows - theirs)}"
            f" which it does not have, missing {sorted(theirs - rows)} which it does")


def _fault(where, mode, height, event, seen):
    """Where the traceback that is on the stack ended, or None if that spot is already known."""
    spot = traceback.extract_tb(sys.exc_info()[2])[-1]
    full = traceback.format_exc()
    line = full.strip().splitlines()[-1]
    key = (line, spot.filename, spot.lineno)
    if key in seen:
        return None
    seen.add(key)
    return (where, mode, height, event, spot, line, full)


def run(renderer, rounds: int, seed: int, switch_odds: float = 0.06):
    """Returns `(faults, states reached)`. A fault is one distinct place a traceback ended."""
    shelf, rng = Shelf(), random.Random(seed)
    faults, seen, reached = [], set(), collections.Counter()
    names = list(itertools.product(LAYOUTS, LIVE))
    panel, where = None, ""

    for _ in range(rounds):
        height = HEIGHTS[rng.randrange(len(HEIGHTS))]
        strip = (0, 0, WIDTH, height)

        def paint(at=strip):
            """One frame, exactly as `Display` draws it.

            `panel` is read late on purpose: it is rebuilt below, after this is defined
            and before this is called, so a default argument would paint the old one."""
            renderer.draw_color = (0, 0, 0, 255)
            renderer.clear()
            panel.draw(renderer, at)                             # noqa: B023
            renderer.present()
            check(panel)                                         # noqa: B023

        if panel is None:
            lname, vname = names[rng.randrange(len(names))]
            panel = build(LAYOUTS[lname], LIVE[vname], shelf)
            panel.attach(renderer)
            where = f"{lname}, {vname} live"
        elif rng.random() < switch_odds:
            lname, vname = names[rng.randrange(len(names))]
            was = len(panel.dials)
            # Half the switches paint in the gap between the two statements, because the window
            # can; the other half do not, so both orderings are reached. See `switch`.
            try:
                switch(panel.bank, panel.runner, LAYOUTS[lname], LIVE[vname],
                       between=paint if rng.random() < 0.5 else None)
            except Exception:                     # noqa: BLE001  a paint mid-switch may raise
                found = _fault(where, panel.mode, height, "a paint mid-switch", seen)
                if found is not None:
                    faults.append(found)
                panel = None
                continue
            # The strip is the same object, so the focus, the drag, the mode AND the rows it
            # laid out for the model that just left all carry across on their own.
            reached["stale focus" if panel._focus not in panel.dials else "live focus"] += 1
            reached[f"into {panel.mode}"] += 1
            reached["same row count" if was == len(panel.dials) else "row count moved"] += 1
            where = f"{lname}, {vname} live"
        for event in gesture(rng, WIDTH, height):
            try:
                paint()
                panel.handle(event, strip)
            except Exception:                     # noqa: BLE001  catching them is the job
                found = _fault(where, panel.mode, height, event, seen)
                if found is not None:
                    faults.append(found)
                panel = None                      # start clean rather than pile on one fault
                break
        else:
            # A frame's worth of the preset writing through, so a gesture that sets a dial is
            # followed by the thing that reads every dial back.
            panel.runner.apply(np.full(len(INDEX), 1e6, dtype=np.float32),
                               {"density": 0.4, "energy": 0.3, "active": 0.2}, Knobs())
    return faults, reached


def report(faults) -> None:
    for where, mode, height, event, spot, line, full in faults:
        print(f"--- {line}")
        print(f"    {pathlib.Path(spot.filename).name}:{spot.lineno}  {spot.line}")
        print(f"    model {where}, mode {mode!r}, window {height}px")
        print(f"    event {event}")
        print("    " + "\n    ".join(full.strip().splitlines()[-6:-1]))
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=1200, help="gestures per seed")
    ap.add_argument("--seeds", type=int, default=4,
                    help="how many runs, seeded 0 upward. Fixed, so a fault is reproducible")
    args = ap.parse_args()

    pygame.init()
    window = Window("rytm fuzz", size=(WIDTH + 200, max(HEIGHTS)))
    renderer = Renderer(window, vsync=False)

    total = 0
    for seed in range(args.seeds):
        faults, reached = run(renderer, args.rounds, seed)
        states = ", ".join(f"{k} {v}" for k, v in sorted(reached.items()))
        print(f"seed {seed}: {args.rounds} rounds, {len(faults)} distinct fault(s)")
        print(f"  reached: {states}", flush=True)
        report(faults)
        total += len(faults)

    print(f"\n{args.seeds * args.rounds} gestures, {total} distinct fault(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
