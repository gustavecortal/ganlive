"""The control surface: the dials the drums, and your hands, actually turn."""
from __future__ import annotations

import collections
from dataclasses import dataclass, replace

#: How many principal latent directions to expose. Eight, because the Rytm gives eight
#: encoders on a page -- so the directions are one page and everything else is another.
DIRECTIONS = 8

#: The furthest a direction dial pushes, in units of that direction's own length -- of the
#: latent, or of `w`. Not of the picture: what that buys is measured per model.
DIRECTION_RANGE = 2.5

#: **What a turn buys, measured** -- the fraction of full push that delivers each even share of
#: a direction's total change. Pushing proportionally does not turn proportionally: over 96
#: curves (8 directions x 6 latents x 2 checkpoints, both signs averaged) an eighth of the push
#: already buys 23.6% of the change and half of it buys 71.1%, so laid out straight the last
#: quarter of the dial is worth a tenth of the picture and the first eighth is worth a quarter.
#: The shape is the same for every direction on both checkpoints, which is what makes one table
#: honest; `spread` carries the same treatment for the same reason. Nine pushes at even shares
#: of the change, read through `evenly` rather than written as pairs, so the share column cannot
#: be mistyped into a dial that doubles back on itself.
DIRECTION_PUSH = (0.0, 0.0662, 0.1342, 0.2160, 0.3078, 0.4147, 0.5465, 0.7166, 1.0)


def evenly(values) -> tuple[tuple[float, float], ...]:
    """A curve given as values at even spacing, as `(position, value)` pairs."""
    values = tuple(values)
    last = max(1, len(values) - 1)
    return tuple((i / last, float(v)) for i, v in enumerate(values))


DIRECTION_RESPONSE = evenly(DIRECTION_PUSH)


def _direction_points() -> tuple[tuple[float, float], ...]:
    """A direction dial's travel: rests centred, pushes `DIRECTION_RANGE` either way, spaced
    so equal turns are equal amounts of visible change rather than equal amounts of push."""
    out = [(0.5 - 0.5 * share, -DIRECTION_RANGE * push)
           for share, push in reversed(DIRECTION_RESPONSE[1:])]
    out += [(0.5 + 0.5 * share, DIRECTION_RANGE * push)
            for share, push in DIRECTION_RESPONSE]
    return tuple(out)


DIRECTION_POINTS = _direction_points()

#: The half of a direction dial's description that is true whatever space the basis lives in.
#: Said once, because it appears both in the generic blurb and in the per-range one.
DIRECTION_TAIL = ("Rests in the middle and travels both ways. What it does is measured on the "
                  "model you have loaded, and is for you to name.")

#: The spine: what every model offers, whatever it is. One architecture's own
#: dials live with that architecture -- see `fastgan_dials`. A graph that declares its
#: own reads them out of the file instead; see `adopted` below.
DIALS: dict[str, tuple[float, str]] = {
    "reaction": (0.50, "how hard the picture answers individual hits. 0.5 is as written, 0 "
                       "ignores every hit so only the arrangement moves the picture, 1 is "
                       "twice as hard. This mutes the drumming, not the structure."),

    "speed": (0.25, "how often the picture sets off for a new place. Every quarter turn "
                    "halves the time: 0 is one move every eight beats, 0.25 is one bar, "
                    "1.0 is one every half beat. Always a musical value, so arrival is "
                    "always on a beat."),
    "spread": (0.30, "how far each move goes -- the walk's slerp fraction from home toward a "
                     "fresh draw. Near 0 it is one picture breathing; near 1 every move is a "
                     "different scene. Laid out so equal turns are equal amounts of visible "
                     "change, which the underlying number is not."),
    "hold": (0.00, "how much of each move is spent completely still. 0 is always gliding, "
                   "0.8 stands still for four fifths of the bar and then travels."),
    "late": (0.50, "how late in the bar the movement happens. 0 leaves the beat already "
                   "moving and settles, 1 waits and arrives exactly on the next beat. Only "
                   "visible against a beat, with the picture running."),
    "grid": (0.00, "snap the movement to steps instead of gliding. 0 is off, then quarter "
                   "notes, eighths, sixteenths, thirty-seconds."),

    # No claim about the ranking here: loading re-orders by measured effect, and the strip appends
    # that measurement to this paragraph.
    **{f"dir{i + 1}": (0.50,
        f"principal latent direction {i + 1}, from the SVD of the first weight that consumes "
        f"z. {DIRECTION_TAIL}")
       for i in range(DIRECTIONS)},
}




MASTER = ("reaction",)
MOTION = ("speed", "spread", "hold", "late", "grid")
#: The direction dials, in order, so nothing has to spell `dir{i+1}` or slice `name[3:]`.
DIRECTION_DIALS = tuple(f"dir{i + 1}" for i in range(DIRECTIONS))
#: The LATENT block, which since `z_scale` went is exactly those. Two names because one is what
#: the dials are and the other is where they are drawn, and the block could take another dial.
LATENT = DIRECTION_DIALS
SPEED_BEATS = (8.0, 4.0, 2.0, 1.0, 0.5)

GRID_STEPS = (0, 4, 8, 16, 32)

SPREAD_TABLE = ((0.00, 0.0), (0.05, 1.8), (0.10, 3.5), (0.20, 7.0), (0.35, 11.8),
                (0.50, 16.0), (0.70, 20.2), (1.00, 22.5))
SPREAD_MAX = SPREAD_TABLE[-1][1]
#: The same table as a curve the dial can be read through, so the walk between measured points
#: is `at`'s one loop rather than a second copy of it. The distance is the position here,
#: because the dial is laid out against what it buys and not against the number underneath.
SPREAD_POINTS = tuple((far / SPREAD_MAX, near) for near, far in SPREAD_TABLE)


def direction_index(name: str) -> int | None:
    """Which direction a dial drives, counting from 0, or `None` if it drives none."""
    try:
        return DIRECTION_DIALS.index(name)
    except ValueError:
        return None


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _detent(x: float, values):
    return values[min(len(values) - 1, max(0, round(x * (len(values) - 1))))]


def spread_for(x: float) -> float:
    """`spread` dial to the walk's own spread, through the measured table."""
    return at(SPREAD_POINTS, x)


RANGE_WORDS = ((3.5, "breathing"), (16.0, "altering"))

HOLD_MAX = 0.95

#: `(below, above, at rest)` for the readout. A bare 0.62 cannot say which way a dial is
#: heading. The resting word is per dial: `reaction` never touches the weights, so there is
#: nothing trained about where it sits, and `noise` rests at off.
POLES = {"reaction": ("calmer", "harder", "as written")}


def hold_for(x: float) -> float:
    """`hold` dial to the fraction of a segment the picture spends still."""
    return min(HOLD_MAX, clamp01(x))


def readout(name: str, value: float, layout=None) -> str:
    """What a dial's position means in its own units, for anything that shows one to a person."""
    v = clamp01(value)
    knob = None if layout is None else layout.get(name)
    if name == "speed":
        beats = _detent(v, SPEED_BEATS)
        return f"{beats:g} beat" + ("" if beats == 1 else "s")
    if name == "grid":
        steps = _detent(v, GRID_STEPS)
        return "glide" if not steps else f"{steps} steps"
    if name == "spread":
        far = v * SPREAD_MAX
        word = next((w for edge, w in RANGE_WORDS if far < edge), "new scene")
        return f"{far:.1f} {word}"
    if name == "hold":
        return f"{hold_for(v) * 100:.0f}% still"
    if name == "late":
        if abs(v - 0.5) < 0.02:
            return "spread evenly"
        return f"{'arrives' if v > 0.5 else 'leaves'} {abs(v - 0.5) * 200:.0f}%"
    if direction_index(name) is not None:
        # A direction has no trained value to be "as" -- it has a push, so say the push.
        push = at(DIRECTION_POINTS, v)
        return "centred" if abs(push) < 0.02 else f"{push:+.2f}"
    poles = (knob.poles if knob is not None else None) or POLES.get(name)
    if poles is not None:
        # Measured from this dial's own rest, not from the middle: a one-sided gate rests at 0,
        # so a fixed 0.5 pivot had it reading "down 100%" while sitting where it starts.
        rest = knob.rest if knob is not None else DIALS[name][0]
        span = max(rest, 1.0 - rest)
        away = (v - rest) / span if span else 0.0
        return (poles[2] if abs(away) < 0.02
                else f"{poles[away > 0]} {abs(away) * 100:.0f}%")
    return f"{v:.2f}"


class Surface:
    """The dial values, and the one method that turns them into everything downstream."""

    def __init__(self, values: dict[str, float] | None = None, layout=None) -> None:
        # The spine, not one architecture's table: a surface with no model yet has the
        # dials every model has. `PresetRunner.adopt` relayouts the moment one loads.
        self.layout = layout if layout is not None else Layout(tuple(spine()))
        self.values = dict(self.layout.rests)
        self.held: frozenset[str] = frozenset()
        for name, value in (values or {}).items():
            self.set(name, value)

    def relayout(self, layout) -> None:
        """Take a different model's dials, keeping every value the two have in common."""
        self.layout = layout
        self.values = {name: self.values.get(name, rest)
                       for name, rest in layout.rests.items()}


    def set_held(self, values) -> None:
        """Put these dials where they are asked and stop `set` moving them."""
        self.held = frozenset(values)
        for name, value in values.items():
            if name in self.values:
                self.values[name] = clamp01(value)

    def set(self, name: str, value: float) -> None:
        if name in self.values and name not in self.held:
            self.values[name] = clamp01(value)

    def add(self, name: str, value: float) -> None:
        if name in self.values:
            self.values[name] = clamp01(self.values[name] + value)


    def __getitem__(self, name: str) -> float:
        return self.values[name]


    # Written once per frame, so a frame is never drawn against a half-written control
    # state. Writer order is resting value, then a held hand, then anything automated.
    def apply(self, knobs, walk_cfg) -> None:
        """Write every dial through to its two destinations: the network, and the walk."""
        v = self.values

        walk_cfg.beats_per_segment = _detent(v["speed"], SPEED_BEATS)
        walk_cfg.spread = spread_for(v["spread"])
        walk_cfg.hold = hold_for(v["hold"])
        walk_cfg.when = clamp01(v["late"])
        walk_cfg.step_grid = _detent(v["grid"], GRID_STEPS)

        # Every remaining dial through the layout, because which settings exist and what each takes
        # at each point of a turn is a property of the loaded model.
        for knob in self.layout.knobs:
            for write in knob.writes:
                knobs.set(write.setting, at(write.points, v[knob.name]))

        walk_cfg.amounts = tuple(at(DIRECTION_POINTS, v[name]) if name in v else 0.0
                                 for name in DIRECTION_DIALS)



# The strip, per model.

def at(points, x: float) -> float:
    """A dial's value at position `x`, walking the line between its measured points."""
    # By index rather than `zip(points, points[1:])`: that slice allocates a fresh tuple on
    # every call, and this runs once per dial per frame. Free at three points; the direction
    # dials' measured table is seventeen, and there are eight of them.
    x = clamp01(x)
    for i in range(1, len(points)):
        x1, v1 = points[i]
        if x <= x1:
            x0, v0 = points[i - 1]
            return v0 if x1 == x0 else v0 + (v1 - v0) * (x - x0) / (x1 - x0)
    return points[-1][1]


@dataclass(frozen=True)
class Write:
    """One dial writing one setting of the model, through its own measured travel."""

    setting: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Knob:
    """One dial on the strip: where it rests, what it says, and what it drives."""

    name: str
    rest: float
    blurb: str
    group: str
    #: Into the model's settings vector. Empty for a dial that never touches the graph.
    writes: tuple[Write, ...] = ()
    #: `(below, above, at rest)` for the readout, when this dial has a direction to report.
    poles: tuple[str, str, str] | None = None
    #: Mean 8-bit levels this dial was measured to buy at full travel, or `None` when nothing
    #: measured it. Read, not decoration: `bank.live_dials` draws a dial that measured nothing dark.
    measured: float | None = None


@dataclass(frozen=True)
class Layout:
    """The dials one loaded model offers, in the order they are drawn."""

    knobs: tuple[Knob, ...]

    def __post_init__(self) -> None:
        """Every view of this layout, once. It is frozen, so none of them can go stale."""
        by_name = {k.name: k for k in self.knobs}
        blocks: dict[str, list[str]] = {}
        for knob in self.knobs:
            blocks.setdefault(knob.group, []).append(knob.name)
        object.__setattr__(self, "_by_name", by_name)
        object.__setattr__(self, "_groups",
                           tuple((title, tuple(names)) for title, names in blocks.items()))
        object.__setattr__(self, "_rests", {k.name: k.rest for k in self.knobs})

    def __getitem__(self, name: str) -> Knob:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self.knobs)

    def get(self, name: str, default=None):
        return self._by_name.get(name, default)

    @property
    def rests(self) -> dict[str, float]:
        return self._rests

    @property
    def groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """`(title, names)` per block, in first-seen order, skipping a block with no dials."""
        return self._groups


def spine(directions: int = DIRECTIONS) -> list[Knob]:
    """MASTER, MOTION and LATENT: host arithmetic, identical on every model ever loaded."""
    out = [Knob("reaction", *DIALS["reaction"], group="MASTER", poles=POLES["reaction"])]
    out += [Knob(name, *DIALS[name], group="MOTION") for name in MOTION]
    out += [Knob(name, *DIALS[name], group="LATENT") for name in DIRECTION_DIALS[:directions]]
    return out


def stylegan2(names=(), rests=(), curves=(), levels=(), ranges=(),
              directions: int = DIRECTIONS) -> Layout:
    """A converted StyleGAN2's surface: `adopted`, saying on the dials themselves that its
    directions are style directions and not latent ones."""
    notes = w_direction_notes(ranges)
    out = []
    for knob in adopted(names, rests, curves, levels, directions).knobs:
        i = direction_index(knob.name)
        if i is not None and i < len(notes):
            knob = replace(knob, blurb=notes[i])
        out.append(knob)
    return Layout(tuple(out))


def w_direction_notes(ranges) -> tuple[str, ...]:
    """One blurb per direction dial, saying which style range that dial factorises."""
    total = collections.Counter(name for name, _pixels in ranges)
    seen: collections.Counter = collections.Counter()
    out = []
    for name, pixels in ranges:
        seen[name] += 1
        out.append(f"style direction {seen[name]} of {total[name]} in this model's {name} "
                   f"range -- {pixels} -- from the SVD of the affines those style vectors "
                   f"feed. {DIRECTION_TAIL}")
    return tuple(out)


#: What a derived dial's description says. The mechanism is known -- a gain on one band, or on
#: the network's own noise at one scale -- and what it looks like is a property of the model.
DERIVED_BLURB = {
    "noise": "the network's own noise at the {h}-tall stage, frozen at load and brought in "
             "as this rises. Derived from the graph and measured on it.",
    "gain": "the gain on everything the {h}-tall stage hands upward. Derived from the graph "
            "and measured on it.",
    # **The control this family is known for, and it read as "a setting this graph declares".**
    # `w_coarse`, `w_mid` and `w_fine` are truncation per style range -- the three headline dials
    # on every converted StyleGAN2, unnamed on the strip while the grain beneath them was
    # described in full. Only that family ever spells a dial `w_*`: an adopted graph's are named
    # by `adopt`, which only ever writes `gain_*` and `noise_*`.
    "w": "how far this model's {h} styles are pulled toward the average `w` it was trained "
         "around -- truncation, for that range of layers alone. Below its rest the picture is "
         "more typical of what the model was trained on and less extreme; above it, less "
         "typical. Derived from the graph and measured on it.",
    # No `pre_tanh` entry: `adopt` no longer puts a dial on the squash's input, for the reason
    # written beside `fastgan_dials.SPANS` -- a gain there is a contrast curve, not a way through.
}


def adopted(names, rests, curves, levels, directions: int = DIRECTIONS) -> Layout:
    """The layout of a graph nobody here wrote, read out of the graph."""
    knobs = spine(directions)
    for i, name in enumerate(names):
        family = name.split("_")[0] if name.startswith(("noise_", "gain_", "w_")) else name
        height = name.partition("_")[2]
        blurb = DERIVED_BLURB.get(family, "a setting this graph declares.")
        rest = float(rests[i]) if i < len(rests) else 0.5
        level = float(levels[i]) if i < len(levels) else None
        knobs.append(Knob(
            name, rest, blurb.format(h=height), group="MODEL",
            writes=(Write(name, evenly(curves[i])),) if i < len(curves) and len(curves[i]) else (),
            # Every derived dial rests at what the model was trained to do, whichever end of its travel
            # that is, because rest is always a gain of 1.0. Reading "off" there says the opposite.
            poles=("down", "up", "as trained"), measured=level))
    return Layout(tuple(knobs))


def per_model(layout) -> tuple[str, ...]:
    """The dials whose meaning belongs to one model: its MODEL block and its directions."""
    return tuple(k.name for k in layout.knobs
                 if k.group == "MODEL" or direction_index(k.name) is not None)
