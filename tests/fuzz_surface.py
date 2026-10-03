"""Drive the strip the way a pair of hands does, headless, and see what falls over.

A real SDL renderer on the dummy video driver, so the real fonts and paint path run. Every
event goes through `DialPanel.handle` in the order `Display` delivers it: paint, present, then
the events that arrived. Model switches happen mid-gesture, because a switch retires dials while
the focus, the drag and the mode carry across.

`test_strip` runs one seed of this. For a longer run:

    python -m tests.fuzz_surface --rounds 1200 --seeds 4
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import pathlib
import random
import sys
import traceback
import types

import numpy as np
import pygame

from ganlive.control.kit import INDEX
from ganlive.dials import table as S
from ganlive.dials.fastgan_dials import fastgan
from ganlive.presets import DEFAULT, PresetRunner
from ganlive.strip import WIDTH, DialPanel, floor_height
from tests.support import FakeKnobs, _StubModel, headless_renderer, stub_bank

#: The layouts a bank can hold at once, including the awkward ones: a single direction, and a
#: family whose MODEL block shares no dial name with this one.
LAYOUTS = {
    "fastgan 8 dirs": fastgan(),
    "fastgan 4 dirs": fastgan(directions=4),
    "fastgan 1 dir": fastgan(directions=1),
    "stylegan2": S.stylegan2(),
}
#: What a model reports as live. `none` is a model whose every dial measured under the floor;
#: `stale names` names dials the layout beside it does not have.
LIVE = {
    "all": None,
    "none": frozenset(),
    "directions only": frozenset({"dir1", "dir2"}),
    "stale names": frozenset({"se_128", "dir7", "reaction", "speed"}),
}
#: Every key the strip claims, plus three it does not, plus tab and return.
KEYS = "gmlspvtr123 \t\r"
#: The floor for the most crowded layout here, and one window shorter than it: the window is
#: resizable, so the strip has to survive being drawn below its floor.
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
                               why="unreadable" if i == 5 else "")
                         for i in range(count)]
        self.pending = None
        self.note = ""

    def entries(self):
        return self._entries

    def request(self, entry) -> None:
        self.pending = str(entry.path)


def stub_model(layout, live):
    """One prepared model, as `bank.Model` presents itself to the strip."""
    return _StubModel(layout=layout, dials_live=frozenset(layout) if live is None else live,
                      directions=types.SimpleNamespace(levels=(100.0, 50.0, 25.0)))


def switch(bank, runner, layout, live, between=None):
    """Play a different model behind a strip that is already drawing the last one.

    These are `play`'s two statements -- `Bank.use`, then `PresetRunner.use_model` -- and
    `between` runs in the gap, because the window's thread may paint there: for one paint the
    bank names the incoming model's dials while the surface still carries the outgoing one's."""
    bank.current = stub_model(layout, live)
    if between is not None:
        between()
    runner.use_model(bank.current)


def build(layout, live, shelf):
    """A strip with a stub model behind it, wired the way `ganlive play` wires the real one."""
    runner = PresetRunner(DEFAULT, INDEX, 60.0, layout=fastgan())
    bank = stub_bank(None, models=[1, 2], index_of=lambda path: 0)
    switch(bank, runner, layout, live)
    return DialPanel(runner, bank=bank, shelf=shelf,
                     actions={"preset": lambda delta: None, "save": lambda found: None,
                              "record": lambda: None, "still": lambda: None,
                              "model": lambda delta=0, to=None: None})


def gesture(rng, w: int, h: int):
    """One plausible thing a hand does: a click with its release, a wheel, a move, a key.

    Presses are paired with releases so the release path runs. Positions run past the edges
    on purpose: the strip is handed events from the whole window."""
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
    """What has to hold after every paint: the strip's rows are the loaded model's dials.

    A wrong strip need not raise, so this is checked rather than left to a traceback."""
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
            """One frame, exactly as `Display` draws it. Reads `panel` late, on purpose: it is
            rebuilt below after this is defined."""
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
            # Half the switches paint in the gap between the two statements; see `switch`.
            try:
                switch(panel.bank, panel.runner, LAYOUTS[lname], LIVE[vname],
                       between=paint if rng.random() < 0.5 else None)
            except Exception:                     # noqa: BLE001  a paint mid-switch may raise
                found = _fault(where, panel.mode, height, "a paint mid-switch", seen)
                if found is not None:
                    faults.append(found)
                panel = None
                continue
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
            # A frame of the preset writing through, so whatever a gesture set is read back.
            panel.runner.apply(np.full(len(INDEX), 1e6, dtype=np.float32),
                               {"density": 0.4, "energy": 0.3, "active": 0.2}, FakeKnobs())
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

    total = 0
    with headless_renderer((WIDTH + 200, max(HEIGHTS)), "fuzz") as renderer:
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
