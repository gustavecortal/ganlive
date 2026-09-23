"""A strip of sliders beside the picture, one per dial, turned with the mouse while it runs."""
from __future__ import annotations

import time
from dataclasses import replace

from ganlive.control.kit import by_channel
from ganlive.dials.table import clamp01, direction_index, readout
from ganlive.presets import AMOUNT_MAX
from ganlive.walk import position

PAD = 12
DOT_W = 10
NAME_W = 78
VALUE_W = 104
HEAD_H = 22
#: Where a row's bar starts: after the driver dot and the name.
TRACK_LEFT = PAD + DOT_W + NAME_W
ROW_MIN, ROW_MAX = 18, 32
WIDTH = 424

SCOPE_H = 78
DESC_H = 112
LIGHTS_H = 34
#: Status block: three fixed lines (status, model, what the mouse does) and the key help.
FIXED_LINES = 3
KEY_LINES = 3
LINE_PITCH = 15
LINES_H = 8 + (FIXED_LINES + KEY_LINES) * LINE_PITCH
STATUS_H = SCOPE_H + DESC_H + LIGHTS_H + LINES_H

#: The shortest window the strip is whole in, for the layout it is drawing. A model with
#: more dials than ours needs a taller window, and the number was a constant.
def floor_height(groups) -> int:
    return PAD + len(groups) * HEAD_H + _dial_count(groups) * ROW_MIN + STATUS_H


def _dial_count(groups) -> int:
    return sum(len(names) for _title, names in groups)


WHEEL_STEP = 0.02

CELL_H = 12

LIGHT_TAIL = 0.35

PANEL = (20, 25, 29)
TRACK = (36, 44, 50)
TEXT = (201, 209, 214)
FAINT = (108, 120, 128)
HAND = (243, 246, 248, 255)
DOT_DARK = (44, 54, 60, 255)
#: A dial this model does not have. Dim enough to read as "not a control here" at a glance,
#: legible enough that the name can still be read.
DEAD = (62, 70, 76)
#: A routing cell on a dial that cannot be wired. `DEAD` is a *text* colour -- dimmer than TEXT
#: where it is read, but brighter than `TRACK`, the cell outline it was reused for. That made
#: the rows a click is refused on the brightest thing in the grid.
DEAD_CELL = (28, 34, 39)
#: The measured-strength underline beneath a direction's track. Dim: it is a fact about the
#: dial, not a value, and it must not compete with the bar that shows where the dial is.
STRENGTH = (84, 116, 92)
REC = (226, 74, 74, 255)
#: One colour per group. LATENT is the block that is identical on every model, so it
#: gets its own; MODEL is the per-architecture block and keeps the old LOOK colour.
GROUP_COLOUR = {"MASTER": (233, 196, 74), "MOTION": (74, 170, 178),
                "LATENT": (146, 200, 120), "MODEL": (228, 87, 46)}

TEXT_HZ = 8.0

#: How many rendered lines and wrapped paragraphs the strip keeps before starting again. A
#: bound rather than an eviction policy: almost every line a repaint asks for is the same one
#: it asked for last time, and the few that are not -- the readouts, the status -- are cheap to
#: draw again after a clear. Each kept line is a small surface, so this is a few megabytes.
TEXT_CACHE = 2048

SOURCE = "console"
PRIORITY = 10

HAND_WORDS = {SOURCE: "hand", "encoder": "knob", "pressure": "pad"}

MODE_DIALS, MODE_ROUTING, MODE_MODELS = "dials", "routing", "models"

MODEL_ROW = 17


def model_rows(floor: int, count: int) -> tuple[int, bool]:
    """`(how many rows fit above floor, whether the list has to scroll)`."""
    fits = max(1, (floor - PAD) // MODEL_ROW)
    scrolling = count > fits
    return (max(1, fits - 1) if scrolling else fits), scrolling


def model_at(y: int, floor: int, count: int) -> int | None:
    """Which visible row a click at `y` is on, as an offset into the shown slice."""
    fits, _scrolling = model_rows(floor, count)
    row = (y - PAD) // MODEL_ROW
    return row if 0 <= row < fits else None

MINE = "panel"
NEEDS_SHELF = "shelf"
NEEDS_ENCODERS = "encoders"

HELP = ((("r",), "free", MINE), (("s",), "save", "save"), (("v",), "rec", "record"),
        (("c",), "still", "still"), (("tab",), "preset", "preset"),
        (("[", "]"), "model", "model"), (("m",), "load", NEEDS_SHELF),
        (("l",), "learn", NEEDS_ENCODERS),
        (("g",), "route", MINE), (("escape",), "quit", None))

BUILT_IN = (None, MINE, NEEDS_SHELF, NEEDS_ENCODERS)
ACTIONS = frozenset(a for _keys, _label, a in HELP if a not in BUILT_IN)

def _breakable(words, font, width: int):
    """Split any word wider than the strip, so a line can always be made to fit."""
    for word in words:
        while font.size(word)[0] > width and len(word) > 1:
            cut = len(word)
            while cut > 1 and font.size(word[:cut])[0] > width:
                cut -= 1
            yield word[:cut]
            word = word[cut:]
        yield word


def wrap(text: str, font, width: int, max_lines: int) -> list[str]:
    """Break `text` to fit `width` pixels, measured with the font that will draw it."""
    lines, line, dropped = [], "", False
    for word in _breakable(text.split(), font, width):
        trial = f"{line} {word}".strip()
        if line and font.size(trial)[0] > width:
            lines.append(line)
            line = word
            if len(lines) == max_lines:
                dropped = True
                break
        else:
            line = trial
    if not dropped and line:
        lines.append(line)
    if dropped and lines:
        last = lines[-1]
        while last and font.size(last + " ...")[0] > width:
            last = last.rsplit(" ", 1)[0] if " " in last else ""
        lines[-1] = (last + " ...").strip()
    return lines[:max_lines]


def _row_height(height: int, groups) -> int:
    """How tall one dial row is: the height on offer shared out, between a floor and a ceiling."""
    avail = height - 2 * PAD - STATUS_H - len(groups) * HEAD_H
    return max(ROW_MIN, min(ROW_MAX, avail // max(1, _dial_count(groups))))


def layout(height: int, groups) -> list[tuple[str, str, int, int]]:
    """`(kind, label, y, height)` for every row, top to bottom. Pure; no window needed."""
    row = _row_height(height, groups)
    out, y = [], PAD
    for title, names in groups:
        out.append(("head", title, y, HEAD_H))
        y += HEAD_H
        for name in names:
            out.append(("dial", name, y, row))
            y += row
    return out


def rows(height: int, groups) -> list[tuple[str, int, int]]:
    """`(dial name, y, height)` for the rows that are controls, skipping the group headings."""
    return [(label, top, tall)
            for kind, label, top, tall in layout(height, groups) if kind == "dial"]


def blocks(height: int, groups) -> dict[str, tuple[int, int]]:
    """`(top, height)` of each block below the dials, so drawing and hit-testing cannot disagree."""
    rows_end = (PAD + len(groups) * HEAD_H
                + _dial_count(groups) * _row_height(height, groups))
    top = max(rows_end, min(height - STATUS_H, rows_end + PAD))
    # A tall window's spare height goes to the description, up to twice its floor: the rows stop
    # growing at ROW_MAX, and a direction's blurb ends with the one measured fact about it.
    desc = DESC_H + max(0, min(DESC_H, height - top - STATUS_H))
    out = {}
    for name, tall in (("scope", SCOPE_H), ("desc", desc),
                       ("lights", LIGHTS_H), ("lines", LINES_H)):
        out[name] = (top, tall)
        top += tall
    return out


def lights(width: int, top: int, count: int) -> list[tuple[int, int, int, int]]:
    """`(x, y, w, h)` for each drum's light, in the block whose top is `top`."""
    if count <= 0:
        return []
    top += 6
    span = (width - 2 * PAD) / count
    w = max(6, int(span) - 4)
    return [(PAD + round(i * span), top, w, 10) for i in range(count)]


def grid_columns(width: int, count: int) -> list[tuple[int, int]]:
    """`(x, w)` for each kit channel's column in the routing view."""
    span = (width - TRACK_LEFT - PAD) / max(1, count)
    return [(TRACK_LEFT + round(i * span), max(8, int(span) - 4)) for i in range(count)]


def cell_box(cx: int, cw: int, top: int, tall: int) -> tuple[int, int, int, int]:
    """`(x, y, w, h)` of one routing cell, relative to the strip."""
    return cx + 2, top + (tall - CELL_H) // 2, cw - 4, CELL_H


def grid_cell(x: int, y: int, width: int, height: int, count: int,
              groups) -> tuple[str, int] | None:
    """`(dial, column)` under a point in the routing view, or None."""
    for label, top, tall in rows(height, groups):
        if not (top <= y < top + tall):
            continue
        for i, (cx, cw) in enumerate(grid_columns(width, count)):
            if cx <= x < cx + cw:
                return label, i
    return None


def track_span(width: int) -> tuple[int, int]:
    """`(left, width)` of the bar inside a row, used by drawing and hit-testing alike."""
    return TRACK_LEFT, max(24, width - TRACK_LEFT - VALUE_W - PAD)


def value_at(x: int, width: int) -> float:
    """What a horizontal position in the strip asks a dial to be."""
    left, span = track_span(width)
    return clamp01((x - left) / span)


def row_at(y: int, laid_out) -> str | None:
    """Which dial a vertical position is on, given rows from `rows()` or already laid out."""
    return next((label for label, top, tall in laid_out if top <= y < top + tall), None)


def hit(x: int, y: int, width: int, height: int, groups) -> tuple[str, float] | None:
    """Which dial is under a point in the strip, and what value that point asks for."""
    label = row_at(y, rows(height, groups))
    return None if label is None else (label, value_at(x, width))


def scope_box(width: int, top: int, tall: int) -> tuple[int, int, int, int]:
    """`(x0, x1, baseline, inner)` for the scope's plot area."""
    return PAD, width - PAD - 4, top + 20, tall - 26


def lit(since, channel: int) -> float:
    """How brightly a drum's light burns, from how long ago it last played."""
    if channel >= len(since):
        return 0.0
    return max(0.0, 1.0 - since[channel] / LIGHT_TAIL)


def scope_curve(cfg, samples: int = 56) -> list[float]:
    """The shaped position at `samples` points across one segment, for drawing."""
    span = max(cfg.beats_per_segment, 1e-6)
    return [position(cfg, (i / (samples - 1)) * span * 0.9999)[2] for i in range(samples)]


def _wired_amounts(drivers) -> dict[str, dict[int, tuple[float, bool]]]:
    """`dial -> {channel: (fraction of full scale, is a push)}` for the routing view."""
    out: dict[str, dict[int, tuple[float, bool]]] = {}
    for dial, (wired, _slow) in drivers.items():
        cell: dict[int, tuple[float, bool]] = {}
        for imp, ch in wired:
            fraction = min(1.0, abs(imp.amount) / AMOUNT_MAX)
            if fraction >= cell.get(ch, (0.0, True))[0]:
                cell[ch] = (fraction, imp.amount >= 0)
        out[dial] = cell
    return out


def driven_by(runner) -> dict[str, tuple[list[tuple[object, int]], list[str]]]:
    """`dial -> ([(rule, channel)] whose hits push it, measurements that set it)`."""
    out: dict[str, tuple[list[tuple[object, int]], list[str]]] = {}
    for dial, wired in runner.routing().items():
        out.setdefault(dial, ([], []))[0].extend(wired)
    for macro in runner.preset.macros:
        _hits, slow = out.setdefault(macro.dial, ([], []))
        if macro.source not in slow:
            slow.append(macro.source)
    return out


class DialPanel:
    """The strip: one row per dial, the mouse, and the one dict the render loop reads."""

    width = WIDTH

    def __init__(self, runner, actions=None, extractor=None, bank=None, shelf=None,
                 encoders=None) -> None:
        self.runner = runner
        self.status = ""
        self.recording = False
        self.beats = 0.0
        self.extractor = extractor
        self.bank = bank
        self.shelf = shelf
        #: The machine's knobs, for `l`: the next one turned takes the focused dial.
        self.encoders = encoders
        self._shelf_rows: list[tuple[object, int, int]] = []
        self._shelf_top = 0
        shared = by_channel(runner.channel_of)
        self.kit = [(channel, "/".join(names)) for channel, names in sorted(shared.items())]
        self.tracks_on = [names for _channel, names in sorted(shared.items())]
        self.mode = MODE_DIALS
        self.actions = dict(actions or {})
        unknown = set(self.actions) - ACTIONS
        if unknown:
            raise KeyError(f"{', '.join(sorted(unknown))} is not a key this strip offers; "
                           f"have {', '.join(sorted(ACTIONS))}")
        self._drag: str | None = None
        self._focus = next(iter(self.dials))
        self._was_held: set[str] = set()
        self._learns_seen = 0
        self._size = (0, 0)
        self._track = (0, 0)
        self._surf = self._tex = self._pg = self._ren = None
        self._font = self._small = self._tiny = self._body = self._micro = None
        self._cells: list[tuple[str, str, int, int]] = []
        self._rows: list[tuple[str, int, int]] = []
        #: The layout the rows below were laid out for. See `draw`.
        self._laid: tuple = ()
        self._lights: list[tuple[int, int, int, int]] = []
        self._columns: list[tuple[int, int]] = []
        self._blocks: dict[str, tuple[int, int]] = {}
        #: `dial -> group colour`, filled on demand by `_colour` and cleared on a relayout.
        self._colours: dict[str, tuple[int, int, int, int]] = {}
        self._dirty = True
        self._last_paint = 0.0
        self._said: dict = {}
        self._wrapped: dict = {}
        self.reload()

    def _say(self, font, text: str, colour):
        """One rendered line, kept.

        **`font.render` rasterises every glyph on every call**, and a repaint asks for
        sixty-five of them -- every dial name, every group heading, every drum label and the
        whole key help, in the same colour as the repaint before it. That runs at `TEXT_HZ` on
        the window's thread, which on a 60 fps loop is one frame in eight, holding the GIL the
        frame loop is waiting to take back: the strip costs +0.18 ms on the median frame and
        +1.79 at p95, and p95 is what the budget is judged on."""
        key = (font, text, colour)
        got = self._said.get(key)
        if got is None:
            if len(self._said) >= TEXT_CACHE:
                self._said.clear()
            got = self._said[key] = font.render(text, True, colour)
        return got

    def _lines(self, text: str, font, width: int, max_lines: int) -> list[str]:
        """`wrap`, kept. It measures with the font -- ninety-one `font.size` calls a repaint --
        and what it measures is the focused dial's paragraph and the key help, neither of which
        changes between repaints unless a hand moves."""
        key = (text, font, width, max_lines)
        got = self._wrapped.get(key)
        if got is None:
            if len(self._wrapped) >= TEXT_CACHE:
                self._wrapped.clear()
            got = self._wrapped[key] = wrap(text, font, width, max_lines)
        return got


    def _model(self):
        """The model the strip draws: **the runner's**, not the bank's.

        `Bank.use` and `PresetRunner.use_model` are consecutive statements on the frame loop's
        thread, and the window paints between them. Reading the layout off the bank and the
        values off the runner drew half of each for that frame. The runner takes its model as
        the last thing a switch does, so one reference says which model this frame is. The
        bank answers only before the runner has been handed one."""
        model = self.runner.model
        if model is None and self.bank is not None:
            model = self.bank.current
        return model

    def live_dials(self) -> frozenset | None:
        """The dials that reach the loaded model, or `None` when there is nothing to ask."""
        model = self._model()
        return None if model is None else model.dials_live

    @staticmethod
    def dead(name: str, live, live_dials) -> bool:
        """Whether a row is drawn dark rather than as a control.

        **Both paint passes ask this, and they had disagreed.** The texture pass tested the
        runner's surface as well as the model's live set; the per-frame pass tested only the
        live set, and then read `live[name]` for a row the surface did not carry. A caller that
        hands the runner a layout without a model can still put the two apart."""
        return name not in live or (live_dials is not None and name not in live_dials)

    @property
    def dials(self):
        """The loaded model's dials. **Not a module constant any more**: an adopted graph brings its own MODEL
        block, with its own count, and the strip's whole geometry -- row height, block tops, the shortest
        window it is whole in -- is a function of how many rows there are."""
        model = self._model()
        return (self.runner.surface if model is None else model).layout

    @property
    def focus(self) -> str:
        """The dial being described, always one the loaded model has.

        A switch retires dials -- one checkpoint keeps 4 of its 8 directions where the next
        keeps all 8, a StyleGAN2 has no `se_*` -- and the name a hand was last on outlives them.
        Every reader of it looks the dial up in the layout, which raises, on the window's
        thread, one frame after the switch."""
        if self._focus not in self.dials:
            self._focus = next(iter(self.dials))
        return self._focus

    def _colour(self, name: str) -> tuple[int, int, int, int]:
        """This dial's group colour, kept.

        `self.dials` is a property that walks the bank to the loaded model's layout, and this
        is asked once per drawn row and twice on a row with a driver dot -- twenty-odd chains
        and twenty-odd freshly built 4-tuples per frame, for a mapping that changes only when
        the model does. Cleared in `_resize`, which is what runs when it changes."""
        got = self._colours.get(name)
        if got is None:
            got = self._colours[name] = (*GROUP_COLOUR[self.dials[name].group], 255)
        return got

    def _directions(self):
        model = self._model()
        return None if model is None else model.directions

    def _right(self, surf, img, y: int) -> None:
        """Blit against the strip's right-hand edge."""
        surf.blit(img, (self._size[0] - PAD - img.get_width(), y))

    def floor_height(self) -> int:
        """The shortest window this strip is whole in. Asked for by whoever opens one."""
        return floor_height(self.dials.groups)

    def set(self, name: str, value: float) -> None:
        """Put a dial somewhere and hold it there."""
        held = self.runner.held_by(SOURCE)
        self._dirty |= name not in held
        self._focus = name
        self.runner.hold(SOURCE, {**held, name: clamp01(value)}, PRIORITY)

    def release(self, name: str | None = None) -> None:
        """Let go of one dial, or of all of them, so the preset and the drums have it back."""
        held = self.runner.held_by(SOURCE)
        self._dirty |= bool(held) if name is None else name in held
        self.runner.free(SOURCE, name)

    def live(self, layout=None) -> dict[str, float]:
        """Where every dial actually is this frame, once the preset and the rules have moved it.

        `layout` is the one `draw` already resolved, so the property is not walked again."""
        values = self.runner.surface.values
        names = self.dials if layout is None else layout
        return {name: values[name] for name in names if name in values}

    def settings(self) -> dict[str, float]:
        """What your hands are holding, as a line to paste into a preset."""
        return {name: round(value, 3)
                for name, value in sorted(self.runner.held_by(SOURCE).items())}

    def preset_now(self):
        """The whole setting as it stands at the controls: the preset with your hands folded in."""
        preset = self.runner.preset
        return replace(preset, dials={**preset.dials, **self.settings()})

    def reload(self, repaint: bool = True) -> None:
        """The preset changed underneath us: re-read what drives what, and repaint."""
        self._drivers = driven_by(self.runner)
        self._wired = _wired_amounts(self._drivers)
        self._dirty = self._dirty or repaint


    def attach(self, renderer) -> None:
        """Called once by `Display`, on the thread that owns the window."""
        import pygame

        pygame.font.init()
        self._pg = pygame
        self._ren = renderer
        self._font = pygame.font.SysFont("segoeui,dejavusans,arial", 15)
        self._body = pygame.font.SysFont("segoeui,dejavusans,arial", 13)
        self._small = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 13)
        self._tiny = pygame.font.SysFont("segoeui,dejavusans,arial", 12, bold=True)
        #: For a label that has to fit a column rather than a line of its own.
        self._micro = pygame.font.SysFont("segoeui,dejavusans,arial", 10, bold=True)
        # Both caches are keyed by the font object, so a second `attach` would otherwise keep
        # lines drawn with fonts this panel no longer holds.
        self._said.clear()
        self._wrapped.clear()

    def draw(self, ren, strip) -> None:
        x0, y0, w, h = strip
        # **Keyed on the layout as well as the window.** The rows are the loaded model's dials,
        # and a switch changes which dials those are without touching the window's size. The
        # only thing that rebuilt them was a resize, and the window is only ever made TALLER --
        # so a switch to a model with the same number of rows, or fewer, left the strip drawing
        # the outgoing model's names, every one of them dark, with none of the incoming model's
        # on it at all. Ours and a converted StyleGAN2 both come to nineteen rows.
        # Resolved once and passed down. It is a property that walks `bank.current.layout`,
        # and this function and the two below it asked for it about twenty-eight times a frame.
        layout = self.dials
        if (w, h) != self._size or layout.groups != self._laid:
            self._resize(w, h)
        live = self.live(layout)
        self._follow_the_hand(layout)
        now = time.perf_counter()
        if self._dirty or now - self._last_paint >= 1.0 / TEXT_HZ:
            self._paint(live)
            self._tex.update(self._surf)
            self._dirty = False
            self._last_paint = now
        self._tex.draw(dstrect=strip)

        rect = self._pg.Rect
        left, span = self._track
        hands = self.runner.hands
        since = getattr(self.extractor, "since", None)
        since = [] if since is None else since.tolist()

        if self.recording:
            ren.draw_color = REC
            ren.fill_rect(rect(x0 + self._size[0] - PAD - 9, y0 + PAD - 1, 9, 9))

        if self.mode == MODE_MODELS:
            self._draw_lights(ren, x0, y0, since)
            return

        if self.mode == MODE_ROUTING:
            self._draw_routing(ren, x0, y0, since)
            self._draw_scope(ren, x0, y0)
            self._draw_lights(ren, x0, y0, since)
            return

        live_dials = self.live_dials()
        since_any = min(since) if since else 1e6
        for name, top, tall in self._rows:
            if self.dead(name, live, live_dials):
                continue
            ty = y0 + top + (tall - 8) // 2
            ren.draw_color = self._colour(name)
            ren.fill_rect(rect(x0 + left, ty, max(2, round(span * live[name])), 8))
            held = hands.get(name)
            if held is not None:
                ren.draw_color = HAND
                ren.fill_rect(rect(x0 + left + min(span - 2, round(span * held)) - 1,
                                   ty - 3, 3, 14))
            drivers = self._drivers.get(name)
            if drivers is not None:
                ren.draw_color = self._driver_colour(name, drivers, since, since_any)
                ren.fill_rect(rect(x0 + PAD, ty + 1, 6, 6))

        self._draw_scope(ren, x0, y0)

        self._draw_lights(ren, x0, y0, since)

    def _follow_the_hand(self, layout) -> None:
        """Describe whatever was just grabbed, whoever grabbed it."""
        held = self.runner.hands
        grabbed = [n for n in layout if n in held and n not in self._was_held]
        self._was_held = set(held)
        if grabbed:
            self._focus = grabbed[0]
            self._dirty = True
        # A learn completes on the MIDI thread; the status line has to notice.
        if self.encoders is not None and self.encoders.version != self._learns_seen:
            self._learns_seen = self.encoders.version
            self.encoders.flush()                # here, on the window's thread, not the reader's
            self._dirty = True

    def _draw_lights(self, ren, x0, y0, since):
        """Which drum just played. Drawn every frame rather than painted into the texture."""
        rect = self._pg.Rect
        for (channel, _label), (lx, ly, lw, lh) in zip(self.kit, self._lights, strict=True):
            glow = lit(since, channel)
            ren.draw_color = (round(36 + 192 * glow), round(44 + 43 * glow),
                              round(50 - 4 * glow), 255)
            ren.fill_rect(rect(x0 + lx, y0 + ly, lw, lh))

    def _grid_gesture(self, ev, dial: str, column: int) -> None:
        """Click wires, wheel sets how hard, right click reverses. One tail for all three."""
        pg = self._pg
        live = self.live_dials()
        if live is not None and dial not in live:
            return          # `route` raises on a dead dial, and this is the window's thread
        wheel = ev.type == pg.MOUSEWHEEL
        channel = self.kit[column][0]
        if not wheel and ev.button == 1:
            # The whole column, not its first track: it is one column because those drums share
            # one channel, and the cell beside it is read back per channel.
            self.runner.wire(self.tracks_on[column], dial)
            self.reload()
            return
        now = self.runner.amount_on(dial, channel)
        if now is None:
            return
        if wheel:
            new = now + ev.y * WHEEL_STEP
        elif ev.button == 3:
            new = -now
        else:
            return
        self.runner.set_amount_on(dial, channel, new)
        self.reload(repaint=not wheel)

    def _draw_routing(self, ren, x0, y0, since):
        """Only the cells that carry a rule, lit by how recently their drum fired."""
        rect = self._pg.Rect
        columns = self._columns
        for name, top, tall in self._rows:
            channels = self._wired.get(name)
            if not channels:
                continue
            any_hit = channels.get(-1)
            r, g, b, _ = self._colour(name)
            for i, (channel, _label) in enumerate(self.kit):
                wire = channels.get(channel, any_hit)
                if wire is None:
                    continue
                fraction, push = wire
                k = 0.45 + 0.55 * lit(since, channel)
                ren.draw_color = (round(r * k), round(g * k), round(b * k), 255)
                cx, cy, cw, ch = cell_box(*columns[i], top, tall)
                half = cw // 2
                span = max(1, round(half * fraction))
                mid = x0 + cx + half
                ren.fill_rect(rect(mid if push else mid - span, y0 + cy, span, ch))

    def _driver_colour(self, name, drivers, since, since_any: float):
        """`since_any` is `min(since)`, taken once by the caller: an any-hit rule asked for it
        per driven row, which is a reduction over the kit for each dot in the column."""
        hits, slow = drivers
        glow = 0.0
        for _imp, channel in hits:
            if channel < 0:
                glow = max(glow, 1.0 - since_any / LIGHT_TAIL)
            else:
                glow = max(glow, lit(since, channel))
        glow = max(0.35 if slow else 0.0, min(1.0, glow))
        r, g, b, _ = self._colour(name)
        return (round(DOT_DARK[0] + (r - DOT_DARK[0]) * glow),
                round(DOT_DARK[1] + (g - DOT_DARK[1]) * glow),
                round(DOT_DARK[2] + (b - DOT_DARK[2]) * glow), 255)

    def _draw_scope(self, ren, x0, y0):
        """The dot travelling the curve. The curve itself is in the texture behind it."""
        top, tall = self._blocks["scope"]
        x_lo, x_hi, base, inner = scope_box(self._size[0], top, tall)
        _k, u, t = position(self.runner.walk_cfg, self.beats)
        px = x_lo + round((x_hi - x_lo) * u)
        py = base + round(inner * (1.0 - t))
        ren.draw_color = HAND
        ren.fill_rect(self._pg.Rect(x0 + px, y0 + py - 2, 5, 5))

    def _resize(self, w: int, h: int) -> None:
        """Rebuild the strip's own texture at a new size."""
        from pygame._sdl2.video import Texture

        self._surf = self._pg.Surface((w, h))
        self._tex = Texture(self._ren, (w, h), streaming=True)
        self._size = (w, h)
        self._track = track_span(w)
        groups = self._laid = self.dials.groups
        # The incoming model's dials are not the outgoing one's, and a name may move group.
        self._colours.clear()
        self._cells = layout(h, groups)
        self._rows = [(label, top, tall) for kind, label, top, tall in self._cells
                      if kind == "dial"]
        self._blocks = blocks(h, groups)
        self._lights = lights(w, self._blocks["lights"][0], len(self.kit))
        self._columns = grid_columns(w, len(self.kit))
        self._dirty = True

    def _paint(self, live: dict[str, float]) -> None:
        """Everything that is not moving: names, numbers, empty tracks, resting ticks."""
        pg, surf = self._pg, self._surf
        w, h = self._size
        surf.fill(PANEL)
        pg.draw.line(surf, TRACK, (0, 0), (0, h))
        left, span = self._track
        hands = self.runner.hands
        live_dials = self.live_dials()
        strengths = self._strengths()

        if self.mode == MODE_MODELS:
            self._paint_models(surf)
        shown = self.dials
        blocks_of = dict(shown.groups)
        for kind, label, top, tall in (self._cells if self.mode != MODE_MODELS else ()):
            if kind == "head":
                surf.blit(self._say(self._tiny, label, GROUP_COLOUR[label]),
                          (PAD, top + tall - 15))
                # Said once, at the top of the block, rather than eleven times down the
                # right-hand column. An ONNX model darkens six rows at a stroke.
                names = blocks_of[label]
                if live_dials is not None and not (set(names) & live_dials):
                    self._right(surf, self._say(self._tiny, "none on this model", DEAD),
                                top + tall - 15)
                continue
            mid = top + tall // 2
            # **A dial the surface does not carry is drawn dark, not raised.** A caller that
            # skips `PresetRunner.use_model` -- the latency harness did -- used to die here with
            # `KeyError: 'w_coarse'` on the window thread, which stops the picture and leaves
            # the frame loop reporting healthy times into a frozen window. `dead` carries the
            # rest of it, including why the other pass has to ask the same question.
            dead = self.dead(label, live, live_dials)
            colour = (DEAD if dead else
                      TEXT if label == self.focus or label in hands else FAINT)
            surf.blit(self._say(self._font, label, colour), (PAD + DOT_W, mid - 9))
            if self.mode == MODE_ROUTING:
                for column in self._columns:
                    pg.draw.rect(surf, DEAD_CELL if dead else TRACK,
                                 cell_box(*column, top, tall), 1)
                continue
            if dead:
                self._right(surf, self._say(self._small, "—", DEAD), mid - 7)
                continue
            pg.draw.rect(surf, TRACK, (left, mid - 4, span, 8))
            tick = left + round(span * shown[label].rest)
            pg.draw.line(surf, FAINT, (tick, mid - 7), (tick, mid + 6))
            i = direction_index(label)
            if i is not None and i < len(strengths):
                pg.draw.rect(surf, STRENGTH,
                             (left, mid + 6, max(1, round(span * strengths[i])), 2))
            self._right(surf, self._say(self._small, readout(label, live[label], shown),
                                          TEXT if label in hands else FAINT), mid - 7)

        if self.mode == MODE_ROUTING:
            self._paint_grid_header(surf)
        if self.mode != MODE_MODELS:
            self._paint_scope(surf, w)
            self._paint_description(surf, w)

        top, _tall = self._blocks["lights"]
        pg.draw.line(surf, TRACK, (0, top), (w, top))
        for (_channel, label), (lx, ly, lw, lh) in zip(self.kit, self._lights, strict=True):
            img = self._say(self._tiny, label, FAINT)
            surf.blit(img, (lx + max(0, (lw - img.get_width()) // 2), ly + lh + 2))

        top, _tall = self._blocks["lines"]
        for i, line in enumerate(self._status_lines()):
            surf.blit(self._say(self._small, line, FAINT), (PAD, top + 4 + i * LINE_PITCH))

    def _floor(self) -> int:
        """Where the picker's list stops: the top of the drum lights."""
        return self._blocks["lights"][0] - PAD

    def _paint_models(self, surf) -> None:
        """Every model on disk: what is playing, what is loaded, and what could be."""
        top = PAD
        entries = self.shelf.entries()
        fits, scrolling = model_rows(self._floor(), len(entries))
        self._shelf_top = max(0, min(self._shelf_top, max(0, len(entries) - fits)))
        self._shelf_rows = entries[self._shelf_top:self._shelf_top + fits]
        playing = self.bank.current.path if self.bank is not None else None

        for entry in self._shelf_rows:
            if entry.path == playing:
                colour, note = HAND[:3], "playing"
            elif entry.loaded:
                colour, note = TEXT, "loaded"
            elif entry.why:
                colour, note = TRACK, entry.why
            else:
                colour, note = FAINT, entry.note
            surf.blit(self._say(self._small, entry.name, colour), (PAD, top + 1))
            if note:
                self._right(surf, self._say(self._small, note, colour), top + 1)
            top += MODEL_ROW

        if scrolling:
            more = (f"{self._shelf_top + 1}-{self._shelf_top + fits} of {len(entries)} "
                    f"· wheel to scroll")
            surf.blit(self._say(self._small, more, TRACK), (PAD, top + 2))

    def _paint_grid_header(self, surf) -> None:
        """One column label per kit channel, above the first group heading.

        In the narrower font when the name of a shared channel does not fit its column: the
        columns share what is left after the dial names, and centring a label wider than its
        column pushes it into the next one. `MT/HT CH/OH CY/CB` ran together as one word."""
        for (_channel, label), (cx, cw) in zip(self.kit, self._columns, strict=True):
            font = self._tiny if self._tiny.size(label)[0] <= cw else self._micro
            img = self._say(font, label, FAINT)
            surf.blit(img, (cx + max(0, (cw - img.get_width()) // 2), 2))

    def _paint_scope(self, surf, w: int) -> None:
        """One segment of the walk's own position curve, plus where the bar lines are."""
        pg = self._pg
        top, tall = self._blocks["scope"]
        cfg = self.runner.walk_cfg
        pg.draw.line(surf, TRACK, (0, top), (w, top))
        surf.blit(self._say(self._tiny, "MOVE", GROUP_COLOUR["MOTION"]), (PAD, top + 3))
        note = (f"{cfg.beats_per_segment:g} beats  ·  {cfg.hold * 100:.0f}% still"
                + (f"  ·  {cfg.step_grid} steps" if cfg.step_grid else ""))
        self._right(surf, self._say(self._small, note, FAINT), top + 3)

        x0, x1, base, inner = scope_box(w, top, tall)
        pg.draw.line(surf, TRACK, (x0, base + inner), (x1, base + inner))
        pg.draw.line(surf, TRACK, (x0, base), (x1, base))
        for beat in range(1, int(cfg.beats_per_segment) or 1):
            bx = x0 + round((x1 - x0) * beat / cfg.beats_per_segment)
            pg.draw.line(surf, TRACK, (bx, base), (bx, base + inner))

        curve = scope_curve(cfg)
        points = [(x0 + round((x1 - x0) * i / (len(curve) - 1)),
                   base + round(inner * (1.0 - t))) for i, t in enumerate(curve)]
        pg.draw.lines(surf, GROUP_COLOUR["MOTION"], False, points)

    def _paint_description(self, surf, w: int) -> None:
        """What the dial under the pointer actually does, in its own words."""
        pg = self._pg
        top, tall = self._blocks["desc"]
        pg.draw.line(surf, TRACK, (0, top), (w, top))
        name = self.focus
        live = self.live_dials()
        dead = live is not None and name not in live
        # The heading carries the signal; the paragraph under it is meant to be read, so it
        # keeps the ordinary weight. A dim heading over dim body reads as a rendering fault.
        head = self._say(self._tiny, name.upper(),
                         DEAD if dead else self._colour(name)[:3])
        surf.blit(head, (PAD, top + 4))
        # Everything wired to this dial, on one line: the knob that holds it, the measurements
        # that set it, and the drums that push it. The knob was the missing third -- the routing
        # grid shows the drums and a learned binding vanished the moment its learn ended.
        #
        # **In that order, because the line is elided and the tail is what goes.** The drums are
        # already on the strip twice over -- the coloured dot at the head of the row, and a lit
        # cell each in the routing grid -- while the knob and the slow rule are written nowhere
        # else at all. Elided against what is free BESIDE THE HEADING: four drums and a slow
        # rule already ran the line straight through `NOISE`, and the heading is the part that
        # says which dial any of this belongs to.
        hits, slow = self._drivers.get(name, ((), ()))
        knob = "" if self.encoders is None else self.encoders.where(name)
        who = " ".join(([f"cc {knob}"] if knob else []) + [f"~{s}" for s in slow]
                       + [f"{imp.track} {imp.amount:+.2f}" for imp, _ch in hits])
        fitted = self._lines(who, self._small, w - 2 * PAD - head.get_width() - 8, 1) if who else []
        if fitted:
            self._right(surf, self._say(self._small, fitted[0], FAINT), top + 4)

        # A row of dashes with no reason is worse than no row. The group heading only covers a block
        # that is dead entirely.
        text = (self._reason(name) if dead
                else self.dials[name].blurb + self._measured(name))
        for i, line in enumerate(self._lines(text, self._body, w - 2 * PAD, (tall - 22) // 16)):
            surf.blit(self._say(self._body, line, FAINT), (PAD, top + 20 + i * 16))

    def _strengths(self) -> tuple[float, ...]:
        """Each direction's measured effect as a fraction of the strongest, or `()`."""
        dirs = self._directions()
        levels = None if dirs is None else dirs.levels
        if not levels:
            return ()
        top = max(levels) or 1.0
        return tuple(level / top for level in levels)

    def _reason(self, name: str) -> str:
        """Why this dial is dark on the loaded model, in terms of the model rather than the
        interface."""
        if direction_index(name) is not None:
            dirs = self._directions()
            have = len(dirs.levels) if dirs is not None and dirs.levels else 0
            return (f"Not on this model. Its first weight gave {have} latent direction(s) "
                    f"that survived to the picture; the rest moved nothing and were dropped "
                    f"rather than left here as a dial that does not act.")
        knob = self.dials.get(name)
        measured = None if knob is None else knob.measured
        if measured is not None:
            return (f"This model has it and it does nothing. Driven to the far end of its "
                    f"travel when the graph was adopted, it moved the picture "
                    f"{measured:.2f} 8-bit levels -- so it is here rather than deleted, and "
                    f"dark rather than offered. A gain feeding a modulated convolution is "
                    f"the usual reason: the demodulation divides it straight back out.")
        return ("Not on this model. This is a setting inside one architecture, and the "
                "loaded generator does not have it -- a graph exported without its dials "
                "has none of them, because they are swapped into a module tree it does "
                "not have.")

    def _measured(self, name: str) -> str:
        """What this model's own load pass found out about a dial, in the dial's own words."""
        i = direction_index(name)
        if i is not None:
            dirs = self._directions()
            levels = None if dirs is None else dirs.levels
            if not levels or i >= len(levels):
                return ""
            level, rank = levels[i], f", ranked {i + 1} of {len(levels)}"
        else:
            knob = self.dials.get(name)
            if knob is None or knob.measured is None or not knob.writes:
                return ""
            level, rank = knob.measured, ""
        return f" Measured on this model: {level:.0f} 8-bit levels at full travel{rank}."

    def offers(self, action: str | None) -> bool:
        """Whether this strip acts on the key whose `HELP` row names `action`.

        **One test, because the help line and the handler have to agree.** They each decided
        it for themselves -- `_status_lines` from a four-clause comprehension, `handle` from
        the same conditions written out again down its `KEYDOWN` chain -- and a key listed and
        not handled is a lie, while a key handled and not listed cannot be found."""
        if action in (None, MINE):
            return True                       # the strip itself, or the window, always acts
        if action == NEEDS_SHELF:
            return self.shelf is not None
        if action == NEEDS_ENCODERS:
            return self.encoders is not None
        return self.actions.get(action) is not None

    def _status_lines(self) -> list[str]:
        """Short lines rather than long ones, so a long model name cannot push the rest off the panel."""
        model = "no model"
        if self.bank is not None:
            model = self.bank.name
            if len(self.bank.models) > 1:
                model = f"model {self.bank.index + 1}/{len(self.bank.models)} · {model}"
        keys = [f"{' '.join(spelling)} {label}" for spelling, label, action in HELP
                if self.offers(action)]
        hands = sorted({HAND_WORDS.get(s, s) for s in self.runner.hands_from.values()})
        holding = f"held by {', '.join(hands)}" if hands else "nothing held"
        learning = None if self.encoders is None else self.encoders.learning
        # What the mouse does here, which changes with the mode and only ever named the dials one.
        # The whole line, not a suffix: appending to the dial line ran it off the right-hand edge.
        third = {MODE_ROUTING: "click wires · wheel how hard · right-click flips",
                 MODE_MODELS: "click plays a loaded model, or loads one"}.get(
                     self.mode, f"LEARN {learning}: turn a knob · l cancels" if learning
                     else f"{self.runner.preset.name} · {holding} · drag to set")
        lines = [self.status, model, third]
        width = max(1, self._size[0] - 2 * PAD)
        return lines + [line.removeprefix("· ")
                        for line in self._lines(" · ".join(keys), self._small, width, KEY_LINES)]

    def _hit(self, x: int, y: int):
        """`hit`, against the rows this panel has already laid out."""
        label = row_at(y, self._rows)
        live = self.live_dials()
        if label is None or (live is not None and label not in live):
            return None                              # a dead dial does not take the mouse
        return label, value_at(x, self._size[0])


    def _choose(self, y: int) -> None:
        """A click in the picker: play a model already loaded, or ask for one that is not."""
        row = model_at(y, self._floor(), len(self.shelf.entries()))
        if row is None or row >= len(self._shelf_rows):
            return
        entry = self._shelf_rows[row]
        act = self.actions.get("model")
        where = self.bank.index_of(entry.path) if self.bank is not None else None
        if entry.loaded:
            if act is not None and where is not None:
                act(where - self.bank.index)
        else:
            self.shelf.request(entry)
        self._dirty = True

    def handle(self, ev, strip) -> bool:
        """One SDL event. True means `Display` should not also act on it."""
        pg = self._pg
        if pg is None:
            return False
        x0, y0, w, h = strip

        if ev.type == pg.MOUSEBUTTONUP:
            was, self._drag = self._drag, None
            return was is not None
        if ev.type in (pg.MOUSEBUTTONDOWN, pg.MOUSEMOTION, pg.MOUSEWHEEL):
            pos = getattr(ev, "pos", None)
            if pos is None:
                pos = pg.mouse.get_pos()
            x, y = pos[0] - x0, pos[1] - y0
            if ev.type == pg.MOUSEMOTION:
                if self.mode in (MODE_ROUTING, MODE_MODELS):
                    return False
                if self._drag is None:
                    if not (0 <= x < w and 0 <= y < h):
                        return False
                    found = self._hit(x, y)
                    if found is not None:
                        self._focus = found[0]
                    return False
                self.set(self._drag, value_at(x, w))
                return True
            if not (0 <= x < w and 0 <= y < h):
                return False
            if self.mode == MODE_MODELS:
                if ev.type == pg.MOUSEWHEEL:
                    self._shelf_top = max(0, self._shelf_top - ev.y)
                    self._dirty = True
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    self._choose(y)
                return True
            if self.mode == MODE_ROUTING:
                cell = grid_cell(x, y, w, h, len(self.kit), self.dials.groups)
                if cell is not None:
                    dial, column = cell
                    self._focus = dial
                    self._grid_gesture(ev, dial, column)
                return True
            found = self._hit(x, y)
            if found is None:
                return True                       # inside the strip, but not on a control
            name, value = found
            if ev.type == pg.MOUSEWHEEL:
                now = self.runner.hands.get(name, self.runner.surface[name])
                self.set(name, now + ev.y * WHEEL_STEP)
            elif ev.button == 1:
                self._drag = name
                self.set(name, value)
            elif ev.button == 3:
                self.release(name)
            return True

        if ev.type == pg.KEYDOWN:
            # Every arm asks `offers` the same question the key help asks, so a key can never
            # be listed and unhandled, or handled and unlisted. `act` is only reached once
            # `offers` has said the action is there.
            act = self.actions.get
            if ev.key == pg.K_r:
                self.release()
            elif ev.key == pg.K_g:
                self.mode = MODE_DIALS if self.mode == MODE_ROUTING else MODE_ROUTING
                self._dirty = True
            elif ev.key == pg.K_m and self.offers(NEEDS_SHELF):
                self.mode = MODE_DIALS if self.mode == MODE_MODELS else MODE_MODELS
                self._dirty = True
            elif ev.key == pg.K_l and self.offers(NEEDS_ENCODERS):
                self.encoders.learning = None if self.encoders.learning else self.focus
                self._dirty = True
            elif ev.key == pg.K_s and self.offers("save"):
                act("save")(self.preset_now())
                self._dirty = True
            elif ev.key == pg.K_v and self.offers("record"):
                act("record")()
                self._dirty = True
            elif ev.key == pg.K_c and self.offers("still"):
                act("still")()
            elif ev.key == pg.K_TAB and self.offers("preset"):
                act("preset")(-1 if ev.mod & pg.KMOD_SHIFT else 1)
                self.reload()
            elif ev.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET) and self.offers("model"):
                act("model")(1 if ev.key == pg.K_RIGHTBRACKET else -1)
                self._dirty = True
            else:
                return False
            return True
        return False
