"""The control strip beside the picture: one slider per dial, a routing grid, a model picker,
and the walk's scope, drawn with SDL and turned with the mouse while the picture runs.
"""
from __future__ import annotations

import time
from dataclasses import replace

from ganlive.clock import position
from ganlive.control.kit import by_channel
from ganlive.control.midi import EncoderMap, PressureMap
from ganlive.curves import clamp01
from ganlive.dials.table import DIRECTIONS, direction_index, readout
from ganlive.presets import AMOUNT_MAX

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

#: Font families, first found wins: Windows, Linux, macOS, then anything.
SANS = "segoeui,dejavusans,helveticaneue,arial"
MONO = "consolas,dejavusansmono,menlo,couriernew"


def floor_height(groups) -> int:
    """The shortest window the strip is whole in, for this layout's groups."""
    return PAD + len(groups) * HEAD_H + _dial_count(groups) * ROW_MIN + STATUS_H


def _dial_count(groups) -> int:
    return sum(len(names) for _title, names in groups)


WHEEL_STEP = 0.02

CELL_H = 12

#: Seconds a drum's light takes to fade after a hit.
LIGHT_TAIL = 0.35

PANEL = (20, 25, 29)
TRACK = (36, 44, 50)
TEXT = (201, 209, 214)
FAINT = (108, 120, 128)
HAND = (243, 246, 248, 255)
DOT_DARK = (44, 54, 60, 255)
#: A dial this model does not have: dim, but the name stays legible.
DEAD = (62, 70, 76)
#: The outline of a routing cell that cannot be wired; darker than `TRACK`, the live outline.
DEAD_CELL = (28, 34, 39)
#: The measured-strength underline beneath a direction's track. Dim, so it does not compete
#: with the bar that shows where the dial is.
STRENGTH = (84, 116, 92)
REC = (226, 74, 74, 255)
#: One colour per group.
GROUP_COLOUR = {"MASTER": (233, 196, 74), "MOTION": (74, 170, 178),
                "LATENT": (146, 200, 120), "MODEL": (228, 87, 46)}

#: How often the strip's text texture is repainted when nothing forces it.
TEXT_HZ = 8.0

#: How many rendered lines and wrapped paragraphs the strip keeps before starting again. Almost
#: every line a repaint asks for is the one it asked for last time.
TEXT_CACHE = 2048

#: The holder name and priority the strip holds dials under; see `PresetRunner.hold`.
SOURCE = "console"
PRIORITY = 10

HAND_WORDS = {SOURCE: "hand", EncoderMap.SOURCE: "knob", PressureMap.SOURCE: "pad"}

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


#: What a key in `HELP` needs before the strip offers it: nothing (`MINE`), a shelf, the
#: machine's encoders, or else an action of that name passed in by the caller.
MINE = "panel"
NEEDS_SHELF = "shelf"
NEEDS_ENCODERS = "encoders"

#: `(keys, label, what it needs)`. One table for the help line and the key handler.
HELP = ((("r",), "free", MINE), (("s",), "save", "save"), (("v",), "rec", "record"),
        (("c",), "still", "still"), (("tab",), "preset", "preset"),
        (("[", "]"), "model", "model"), (("m",), "load", NEEDS_SHELF),
        (("l",), "learn", NEEDS_ENCODERS),
        (("g",), "route", MINE), (("escape",), "quit", None))

BUILT_IN = (None, MINE, NEEDS_SHELF, NEEDS_ENCODERS)
#: The actions a caller may pass to `DialPanel`.
ACTIONS = frozenset(a for _keys, _label, a in HELP if a not in BUILT_IN)


def model_step(pg, ev) -> int:
    """+1 for `]`, -1 for `[`, else 0, on any keyboard layout.

    By typed character, by US keycode, or by the two keys right of `P`: on AZERTY and other
    layouts the brackets need AltGr, and SDL then reports neither bracket keycode."""
    by_char = {"]": 1, "[": -1}.get(getattr(ev, "unicode", "") or "")
    if by_char:
        return by_char
    if ev.key in (pg.K_RIGHTBRACKET, pg.K_LEFTBRACKET):
        return 1 if ev.key == pg.K_RIGHTBRACKET else -1
    scancode = getattr(ev, "scancode", None)
    return {48: 1, 47: -1}.get(scancode, 0)    # SDL_SCANCODE_RIGHTBRACKET / LEFTBRACKET


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
    """Break `text` to fit `width` pixels, measured with the font that will draw it. A text
    that needs more than `max_lines` ends in `...`."""
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
    """`(kind, label, y, height)` for every row, top to bottom; `kind` is `head` or `dial`."""
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
    """`(top, height)` of each block below the dials, for drawing and hit-testing alike."""
    rows_end = (PAD + len(groups) * HEAD_H
                + _dial_count(groups) * _row_height(height, groups))
    top = max(rows_end, min(height - STATUS_H, rows_end + PAD))
    # A tall window's spare height goes to the description, up to twice its floor.
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


def grid_cell(x: int, y: int, laid_out, columns) -> tuple[str, int] | None:
    """`(dial, column)` under a point in the routing view, given `rows()` and `grid_columns()`."""
    label = row_at(y, laid_out)
    if label is None:
        return None
    return next(((label, i) for i, (cx, cw) in enumerate(columns) if cx <= x < cx + cw), None)


def track_span(width: int) -> tuple[int, int]:
    """`(left, width)` of the bar inside a row, used by drawing and hit-testing alike."""
    return TRACK_LEFT, max(24, width - TRACK_LEFT - VALUE_W - PAD)


def value_at(x: int, width: int) -> float:
    """What a horizontal position in the strip asks a dial to be."""
    left, span = track_span(width)
    return clamp01((x - left) / span)


def row_at(y: int, laid_out) -> str | None:
    """Which dial a vertical position is on, given rows from `rows()`."""
    return next((label for label, top, tall in laid_out if top <= y < top + tall), None)


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
    """`dial -> ([(rule, channel)] whose hits push it, measurements that set it)`.
    Channel -1 means any hit."""
    out: dict[str, tuple[list[tuple[object, int]], list[str]]] = {}
    for dial, wired in runner.routing().items():
        out.setdefault(dial, ([], []))[0].extend(wired)
    for macro in runner.preset.macros:
        _hits, slow = out.setdefault(macro.dial, ([], []))
        if macro.source not in slow:
            slow.append(macro.source)
    return out


class DialPanel:
    """The strip: one row per dial of the playing model, and the mouse and keys that turn them.

    `runner` is the `PresetRunner` it reads and holds dials on. `actions` maps the names in
    `ACTIONS` to callbacks for keys the host implements; `bank`, `shelf` and `encoders` are
    optional and enable the model switch, the picker and knob learning."""

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
        #: `(channel, track names)` per kit channel: one light and one routing column each.
        self.kit = sorted(by_channel(runner.channel_of).items())
        self.mode = MODE_DIALS
        self.actions = dict(actions or {})
        unknown = set(self.actions) - ACTIONS
        if unknown:
            raise KeyError(f"{', '.join(sorted(unknown))} is not a key this strip offers; "
                           f"have {', '.join(sorted(ACTIONS))}")
        self._drag: str | None = None
        self._model_seen = runner.model
        self._focus = next(iter(self.dials))
        self._hands_seen = None
        self._was_held: set[str] = set()
        self._learns_seen = 0
        self._size = (0, 0)
        self._track = (0, 0)
        self._surf = self._tex = self._pg = self._ren = None
        self._font = self._small = self._tiny = self._body = self._micro = None
        self._cells: list[tuple[str, str, int, int]] = []
        self._rows: list[tuple[str, int, int]] = []
        #: The groups `_rows` were laid out for; a switch to another layout lays them out again.
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

    @staticmethod
    def _kept(cache: dict, key, make):
        """`cache[key]`, made on a miss. Cleared when full rather than evicting."""
        got = cache.get(key)
        if got is None:
            if len(cache) >= TEXT_CACHE:
                cache.clear()
            got = cache[key] = make()
        return got

    def _say(self, font, text: str, colour):
        """One rendered line, kept: `font.render` rasterises every glyph on every call, on the
        window's thread, holding the GIL the frame loop wants."""
        return self._kept(self._said, (font, text, colour),
                          lambda: font.render(text, True, colour))

    def _lines(self, text: str, font, width: int, max_lines: int) -> list[str]:
        """`wrap`, kept: it measures every word with the font."""
        return self._kept(self._wrapped, (text, font, width, max_lines),
                          lambda: wrap(text, font, width, max_lines))

    def _model(self):
        """The model the strip draws: the runner's, falling back to the bank's.

        The runner's, because a switch sets the bank first and the runner last, and the window
        may paint between the two; the runner's model always matches the runner's values."""
        model = self.runner.model
        if model is None and self.bank is not None:
            model = self.bank.current
        return model

    def live_dials(self) -> frozenset | None:
        """The dials that reach the loaded model, or None when there is nothing to ask."""
        model = self._model()
        return None if model is None else model.dials_live

    def _dark(self, name: str) -> bool:
        """Whether the loaded model says this dial reaches nothing."""
        live = self.live_dials()
        return live is not None and name not in live

    def _dead(self, name: str) -> bool:
        """Whether a row is drawn dark rather than as a control: the model lacks it, or the
        runner's surface does not carry it yet (a caller that skipped `use_model`)."""
        return name not in self.runner.surface.values or self._dark(name)

    @property
    def dials(self):
        """The loaded model's dial layout. The strip's geometry follows it."""
        model = self._model()
        return (self.runner.surface if model is None else model).layout

    @property
    def focus(self) -> str:
        """The dial being described: the last one touched, if the loaded model still has it."""
        if self._focus not in self.dials:
            self._focus = next(iter(self.dials))
        return self._focus

    def _colour(self, name: str) -> tuple[int, int, int, int]:
        """This dial's group colour, kept until the next relayout."""
        got = self._colours.get(name)
        if got is None:
            got = self._colours[name] = (*GROUP_COLOUR[self.dials[name].group], 255)
        return got

    def _levels(self) -> tuple[float, ...]:
        """Each surviving direction's measured effect, in rank order, or `()`."""
        model = self._model()
        dirs = None if model is None else model.directions
        return tuple(dirs.levels) if dirs is not None and dirs.levels else ()

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

    def held(self) -> dict[str, float]:
        """What the holders are holding -- the strip and the machine's knobs -- as dial values
        to fold into a preset. The strip wins where both hold one dial."""
        held = {} if self.encoders is None else self.runner.held_by(self.encoders.SOURCE)
        held.update(self.runner.held_by(SOURCE))
        return {name: round(value, 3) for name, value in sorted(held.items())}

    def preset_now(self):
        """The whole preset as it stands at the controls: the preset with the holders folded in."""
        preset = self.runner.preset
        return replace(preset, dials={**preset.dials, **self.held()})

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
        self._font = pygame.font.SysFont(SANS, 15)
        self._body = pygame.font.SysFont(SANS, 13)
        self._small = pygame.font.SysFont(MONO, 13)
        self._tiny = pygame.font.SysFont(SANS, 12, bold=True)
        #: For a label that has to fit a column rather than a line of its own.
        self._micro = pygame.font.SysFont(SANS, 10, bold=True)
        # Both caches are keyed by font object, so lines drawn with old fonts are dropped.
        self._said.clear()
        self._wrapped.clear()

    def draw(self, ren, strip) -> None:
        """One frame of the strip into `strip`, a `(x, y, w, h)` rect of the window.

        The text is a texture repainted at `TEXT_HZ` or when something changes; the bars, the
        held markers, the lights and the scope dot are rectangles drawn every frame."""
        x0, y0, w, h = strip
        # Relaid out on a new window size or a new layout: a switch can change the dials
        # without changing the window.
        layout = self.dials
        if (w, h) != self._size or layout.groups != self._laid:
            self._resize(w, h)
        # A switch re-resolves which rules fire on the incoming model; the grid has to follow.
        if self.runner.model is not self._model_seen:
            self._model_seen = self.runner.model
            self.reload()
        self._follow_the_hand(layout)
        now = time.perf_counter()
        if self._dirty or now - self._last_paint >= 1.0 / TEXT_HZ:
            self._paint()
            self._tex.update(self._surf)
            self._dirty = False
            self._last_paint = now
        self._tex.draw(dstrect=strip)

        rect = self._pg.Rect
        left, span = self._track
        hands = self.runner.hands
        values = self.runner.surface.values
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

        for name, top, tall in self._rows:
            if self._dead(name):
                continue
            ty = y0 + top + (tall - 8) // 2
            ren.draw_color = self._colour(name)
            ren.fill_rect(rect(x0 + left, ty, max(2, round(span * values[name])), 8))
            held = hands.get(name)
            if held is not None:
                ren.draw_color = HAND
                ren.fill_rect(rect(x0 + left + min(span - 2, round(span * held)) - 1,
                                   ty - 3, 3, 14))
            drivers = self._drivers.get(name)
            if drivers is not None:
                ren.draw_color = self._driver_colour(name, drivers, since)
                ren.fill_rect(rect(x0 + PAD, ty + 1, 6, 6))

        self._draw_scope(ren, x0, y0)
        self._draw_lights(ren, x0, y0, since)

    def _follow_the_hand(self, layout) -> None:
        """Describe whatever was just grabbed, by whichever holder grabbed it."""
        # A learn completes on the MIDI thread; flushed here, on the window's thread.
        if self.encoders is not None and self.encoders.version != self._learns_seen:
            self._learns_seen = self.encoders.version
            self.encoders.flush()
            self._dirty = True
        held = self.runner.hands
        # The runner publishes a new dict on every change, so the same one means no change.
        if held is self._hands_seen:
            return
        self._hands_seen = held
        grabbed = [n for n in layout if n in held and n not in self._was_held]
        self._was_held = set(held)
        if grabbed:
            self._focus = grabbed[0]
            self._dirty = True

    def _draw_lights(self, ren, x0, y0, since):
        """Which drum just played. Drawn every frame rather than painted into the texture."""
        rect = self._pg.Rect
        for (channel, _names), (lx, ly, lw, lh) in zip(self.kit, self._lights, strict=True):
            glow = lit(since, channel)
            ren.draw_color = (round(36 + 192 * glow), round(44 + 43 * glow),
                              round(50 - 4 * glow), 255)
            ren.fill_rect(rect(x0 + lx, y0 + ly, lw, lh))

    def _grid_gesture(self, ev, dial: str, column: int) -> None:
        """Click wires, wheel sets how hard, right click reverses."""
        pg = self._pg
        if self._dark(dial):
            return          # `route` raises on a dead dial, and this is the window's thread
        wheel = ev.type == pg.MOUSEWHEEL
        channel, tracks = self.kit[column]
        if not wheel and ev.button == 1:
            # Every track on the channel: they cannot be told apart once they arrive.
            self.runner.wire(tracks, dial)
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
            for i, (channel, _names) in enumerate(self.kit):
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

    def _driver_colour(self, name, drivers, since):
        """The driver dot's colour: the group colour, as bright as the most recent hit on any
        drum wired to it. A dial set by a slow measurement never goes fully dark."""
        hits, slow = drivers
        glow = 0.0
        for _imp, channel in hits:
            for ch in (range(len(since)) if channel < 0 else (channel,)):
                glow = max(glow, lit(since, ch))
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
        """Rebuild the strip's own texture and geometry for a new size or layout."""
        from pygame._sdl2.video import Texture

        self._surf = self._pg.Surface((w, h))
        self._tex = Texture(self._ren, (w, h), streaming=True)
        self._size = (w, h)
        self._track = track_span(w)
        groups = self._laid = self.dials.groups
        # The incoming model's dials are not the outgoing one's, and a name may move group.
        self._colours.clear()
        self._cells = layout(h, groups)
        self._rows = rows(h, groups)
        self._blocks = blocks(h, groups)
        self._lights = lights(w, self._blocks["lights"][0], len(self.kit))
        self._columns = grid_columns(w, len(self.kit))
        self._dirty = True

    def _paint(self) -> None:
        """Everything that is not moving: names, numbers, empty tracks, resting ticks."""
        pg, surf = self._pg, self._surf
        w, h = self._size
        surf.fill(PANEL)
        pg.draw.line(surf, TRACK, (0, 0), (0, h))
        left, span = self._track
        hands = self.runner.hands
        values = self.runner.surface.values
        live_dials = self.live_dials()
        strengths = self._strengths()
        focus = self.focus

        if self.mode == MODE_MODELS:
            self._paint_models(surf)
        shown = self.dials
        blocks_of = dict(shown.groups)
        for kind, label, top, tall in (self._cells if self.mode != MODE_MODELS else ()):
            if kind == "head":
                surf.blit(self._say(self._tiny, label, GROUP_COLOUR[label]),
                          (PAD, top + tall - 15))
                # Said once on the heading when the whole block is dark.
                names = blocks_of[label]
                if live_dials is not None and not (set(names) & live_dials):
                    self._right(surf, self._say(self._tiny, "none on this model", DEAD),
                                top + tall - 15)
                continue
            mid = top + tall // 2
            dead = self._dead(label)
            colour = (DEAD if dead else
                      TEXT if label == focus or label in hands else FAINT)
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
            self._right(surf, self._say(self._small, readout(label, values[label], shown),
                                        TEXT if label in hands else FAINT), mid - 7)

        if self.mode == MODE_ROUTING:
            self._paint_grid_header(surf)
        if self.mode != MODE_MODELS:
            self._paint_scope(surf, w)
            self._paint_description(surf, w, focus)

        top, _tall = self._blocks["lights"]
        pg.draw.line(surf, TRACK, (0, top), (w, top))
        for (_channel, names), (lx, ly, lw, lh) in zip(self.kit, self._lights, strict=True):
            img = self._say(self._tiny, "/".join(names), FAINT)
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
        """One label per kit channel above the routing columns, in the narrower font when a
        shared channel's name (`MT/HT`) is wider than its column."""
        for (_channel, names), (cx, cw) in zip(self.kit, self._columns, strict=True):
            label = "/".join(names)
            font = self._tiny if self._tiny.size(label)[0] <= cw else self._micro
            img = self._say(font, label, FAINT)
            surf.blit(img, (cx + max(0, (cw - img.get_width()) // 2), 2))

    def _paint_scope(self, surf, w: int) -> None:
        """One segment of the walk's own position curve, plus where the beat lines are."""
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

    def _paint_description(self, surf, w: int, name: str) -> None:
        """The focused dial: its name, what is wired to it, and what it does."""
        pg = self._pg
        top, tall = self._blocks["desc"]
        pg.draw.line(surf, TRACK, (0, top), (w, top))
        dark = self._dark(name)
        head = self._say(self._tiny, name.upper(),
                         DEAD if dark else self._colour(name)[:3])
        surf.blit(head, (PAD, top + 4))
        # Everything wired to this dial, on one line beside the heading: the knob that holds
        # it, the measurements that set it, then the drums. Elided from the end, so the drums
        # (also shown by the dot and the routing grid) are what is cut.
        hits, slow = self._drivers.get(name, ((), ()))
        knob = "" if self.encoders is None else self.encoders.where(name)
        who = " ".join(([f"cc {knob}"] if knob else []) + [f"~{s}" for s in slow]
                       + [f"{imp.track} {imp.amount:+.2f}" for imp, _ch in hits])
        fitted = self._lines(who, self._small, w - 2 * PAD - head.get_width() - 8, 1) if who else []
        if fitted:
            self._right(surf, self._say(self._small, fitted[0], FAINT), top + 4)

        text = (self._reason(name) if dark
                else self.dials[name].blurb + self._measured(name))
        for i, line in enumerate(self._lines(text, self._body, w - 2 * PAD, (tall - 22) // 16)):
            surf.blit(self._say(self._body, line, FAINT), (PAD, top + 20 + i * 16))

    def _strengths(self) -> tuple[float, ...]:
        """Each direction's measured effect as a fraction of the strongest, or `()`."""
        levels = self._levels()
        if not levels:
            return ()
        top = max(levels) or 1.0
        return tuple(level / top for level in levels)

    def _reason(self, name: str) -> str:
        """Why this dial is dark on the loaded model, in terms of the model."""
        if direction_index(name) is not None:
            have = len(self._levels())
            return (f"Not on this model. {have} latent direction(s) earned a dial at load; "
                    f"the others did not beat a random direction by the --direction-floor "
                    f"margin, or fell beyond the {DIRECTIONS} the strip shows.")
        knob = self.dials.get(name)
        model = self._model()
        index = getattr(getattr(model, "settings", None), "index", None) or ()
        reaches = knob is not None and any(w.setting in index for w in knob.writes)
        if reaches and knob.measured is not None:
            return (f"This model has it, but driven to either end of its travel at load it "
                    f"moved the picture only {knob.measured:.2f} 8-bit levels, too little to "
                    f"count as a control, so it is drawn dark rather than offered.")
        return ("Not on this model. This is a setting inside one architecture, and the "
                "loaded generator does not have it.")

    def _measured(self, name: str) -> str:
        """What this model's own load pass found out about a dial, in the dial's own words."""
        i = direction_index(name)
        if i is not None:
            levels = self._levels()
            if i >= len(levels):
                return ""
            level, rank = levels[i], f", ranked {i + 1} of {len(levels)}"
        else:
            knob = self.dials.get(name)
            if knob is None or knob.measured is None or not knob.writes:
                return ""
            level, rank = knob.measured, ""
        return f" Measured on this model: {level:.0f} 8-bit levels at full travel{rank}."

    def offers(self, action: str | None) -> bool:
        """Whether this strip acts on the key whose `HELP` row names `action`. The help line
        and the key handler both ask, so they cannot disagree."""
        if action in (None, MINE):
            return True                       # the strip itself, or the window, always acts
        if action == NEEDS_SHELF:
            return self.shelf is not None
        if action == NEEDS_ENCODERS:
            return self.encoders is not None
        return self.actions.get(action) is not None

    def _status_lines(self) -> list[str]:
        """Status, model, what the mouse does, then the key help. Short lines, so a long model
        name cannot push the rest off the strip."""
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
        third = {MODE_ROUTING: "click wires · wheel how hard · right-click flips",
                 MODE_MODELS: "click plays a loaded model, or loads one"}.get(
                     self.mode, f"LEARN {learning}: turn a knob · l cancels" if learning
                     else f"{self.runner.preset.name} · {holding} · drag to set")
        lines = [self.status, model, third]
        width = max(1, self._size[0] - 2 * PAD)
        return lines + [line.removeprefix("· ")
                        for line in self._lines(" · ".join(keys), self._small, width, KEY_LINES)]

    def _hit(self, x: int, y: int):
        """`(dial, value)` under a point in the strip, or None. A dark dial takes no mouse."""
        label = row_at(y, self._rows)
        if label is None or self._dark(label):
            return None
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
                act(to=where)
        else:
            self.shelf.request(entry)
        self._dirty = True

    def handle(self, ev, strip) -> bool:
        """One SDL event. True means `Display` should not also act on it."""
        pg = self._pg
        if pg is None:
            return False
        if ev.type == pg.MOUSEBUTTONUP:
            was, self._drag = self._drag, None
            return was is not None
        if ev.type in (pg.MOUSEBUTTONDOWN, pg.MOUSEMOTION, pg.MOUSEWHEEL):
            return self._mouse(ev, strip)
        if ev.type == pg.KEYDOWN:
            return self._key(ev)
        return False

    def _mouse(self, ev, strip) -> bool:
        pg = self._pg
        x0, y0, w, h = strip
        pos = getattr(ev, "pos", None)
        if pos is None:
            pos = pg.mouse.get_pos()
        x, y = pos[0] - x0, pos[1] - y0
        inside = 0 <= x < w and 0 <= y < h
        if ev.type == pg.MOUSEMOTION:
            if self.mode in (MODE_ROUTING, MODE_MODELS):
                return False
            if self._drag is not None:
                self.set(self._drag, value_at(x, w))
                return True
            found = self._hit(x, y) if inside else None
            if found is not None:
                self._focus = found[0]          # hovering describes, without taking the event
            return False
        if not inside:
            return False
        if self.mode == MODE_MODELS:
            if ev.type == pg.MOUSEWHEEL:
                self._shelf_top = max(0, self._shelf_top - ev.y)
                self._dirty = True
            elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                self._choose(y)
            return True
        if self.mode == MODE_ROUTING:
            cell = grid_cell(x, y, self._rows, self._columns)
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

    def _key(self, ev) -> bool:
        """A key from `HELP`. Each arm asks `offers`, as the help line does."""
        pg = self._pg
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
        elif (step := model_step(pg, ev)) and self.offers("model"):
            act("model")(step)
            self._dirty = True
        else:
            return False
        return True
