"""The control surface: the dials, what each one does, and how a model's dials are laid out.

The shared blocks (MASTER, MOTION, LATENT) are host arithmetic and the same on every model.
A model adds a MODEL block: FastGAN's from `fastgan_dials`, any other's read out of what was
measured on it (`adopted`, `stylegan2`).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, replace

from ganlive.curves import at, clamp01, evenly

#: How many latent directions to expose: one page of eight encoders on the drum machine.
DIRECTIONS = 8

#: The furthest a direction dial pushes, in units of that direction's own length (of the
#: latent, or of `w`). What that buys in the picture is measured per model.
DIRECTION_RANGE = 2.5

#: The fraction of full push that buys each even share of a direction's total change,
#: measured over 96 curves on two checkpoints. Pushing proportionally does not turn
#: proportionally: an eighth of the push already buys about a quarter of the change. The
#: shape was the same for every direction, so one table serves them all.
DIRECTION_PUSH = (0.0, 0.0662, 0.1342, 0.2160, 0.3078, 0.4147, 0.5465, 0.7166, 1.0)

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
DIRECTION_TAIL = ("Rests in the middle and travels both ways. What it does is measured on the "
                  "model you have loaded, and is for you to name.")

#: The shared dials, as `{name: (rest, description)}`.
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

    # Loading re-orders the directions by measured effect, and the strip appends that
    # measurement to this description.
    **{f"dir{i + 1}": (0.50,
        f"principal latent direction {i + 1}, from the SVD of the first weight that consumes "
        f"z. {DIRECTION_TAIL}")
       for i in range(DIRECTIONS)},
}

MASTER = ("reaction",)
MOTION = ("speed", "spread", "hold", "late", "grid")
#: The direction dials, in order.
DIRECTION_DIALS = tuple(f"dir{i + 1}" for i in range(DIRECTIONS))
#: The LATENT block.
LATENT = DIRECTION_DIALS
SPEED_BEATS = (8.0, 4.0, 2.0, 1.0, 0.5)

GRID_STEPS = (0, 4, 8, 16, 32)

#: The walk's spread against how far a move goes, measured: `(spread, distance)`.
SPREAD_TABLE = ((0.00, 0.0), (0.05, 1.8), (0.10, 3.5), (0.20, 7.0), (0.35, 11.8),
                (0.50, 16.0), (0.70, 20.2), (1.00, 22.5))
SPREAD_MAX = SPREAD_TABLE[-1][1]
#: The same table as a curve, with the distance as the dial position: the dial is laid out
#: against what it buys, not against the number underneath.
SPREAD_POINTS = tuple((far / SPREAD_MAX, near) for near, far in SPREAD_TABLE)


def direction_index(name: str) -> int | None:
    """Which direction a dial drives, counting from 0, or `None` if it drives none."""
    try:
        return DIRECTION_DIALS.index(name)
    except ValueError:
        return None


def _detent(x: float, values):
    return values[min(len(values) - 1, max(0, round(x * (len(values) - 1))))]


def spread_for(x: float) -> float:
    """`spread` dial to the walk's own spread, through the measured table."""
    return at(SPREAD_POINTS, x)


RANGE_WORDS = ((3.5, "breathing"), (16.0, "altering"))

HOLD_MAX = 0.95

#: `(below, above, at rest)` for the readout, so it can say which way a dial is heading.
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
        # A direction has no trained value; say the push.
        push = at(DIRECTION_POINTS, v)
        return "centred" if abs(push) < 0.02 else f"{push:+.2f}"
    poles = (knob.poles if knob is not None else None) or POLES.get(name)
    if poles is not None:
        # Measured from this dial's own rest, which for a one-sided dial is an end.
        rest = knob.rest if knob is not None else DIALS[name][0]
        span = max(rest, 1.0 - rest)
        away = (v - rest) / span if span else 0.0
        return (poles[2] if abs(away) < 0.02
                else f"{poles[away > 0]} {abs(away) * 100:.0f}%")
    return f"{v:.2f}"


class Surface:
    """The dial values, and the one method that turns them into everything downstream."""

    def __init__(self, values: dict[str, float] | None = None, layout=None) -> None:
        # With no model yet, the dials every model has.
        self.layout = layout if layout is not None else Layout(tuple(shared_knobs()))
        self.values = dict(self.layout.rests)
        self.held: frozenset[str] = frozenset()
        for name, value in (values or {}).items():
            self.set(name, value)

    def relayout(self, layout) -> None:
        """Take a different model's dials, keeping every value the two have in common."""
        self.layout = layout
        self.values = {name: self.values.get(name, rest)
                       for name, rest in layout.rests.items()}

    def set_held(self, values, names: frozenset[str] | None = None) -> None:
        """Put these dials where they are asked and stop `set` moving them. `names` is
        `frozenset(values)`, for a caller that holds the same dials frame after frame."""
        self.held = frozenset(values) if names is None else names
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

    def apply(self, knobs, walk_cfg) -> None:
        """Write every dial through to its two destinations: the network, and the walk.

        Called once per frame, so a frame is never drawn against a half-written state."""
        v = self.values

        walk_cfg.beats_per_segment = _detent(v["speed"], SPEED_BEATS)
        walk_cfg.spread = spread_for(v["spread"])
        walk_cfg.hold = hold_for(v["hold"])
        walk_cfg.when = clamp01(v["late"])
        walk_cfg.step_grid = _detent(v["grid"], GRID_STEPS)

        # Which settings exist, and what each takes at each point of a turn, is the loaded
        # model's layout.
        for knob in self.layout.knobs:
            for write in knob.writes:
                knobs.set(write.setting, at(write.points, v[knob.name]))

        walk_cfg.amounts = tuple(at(DIRECTION_POINTS, v[name]) if name in v else 0.0
                                 for name in DIRECTION_DIALS)


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
    #: measured it. A dial that measured nothing is drawn dark.
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


def shared_knobs(directions: int = DIRECTIONS) -> list[Knob]:
    """The MASTER, MOTION and LATENT blocks: the same on every model."""
    out = [Knob("reaction", *DIALS["reaction"], group="MASTER", poles=POLES["reaction"])]
    out += [Knob(name, *DIALS[name], group="MOTION") for name in MOTION]
    out += [Knob(name, *DIALS[name], group="LATENT") for name in DIRECTION_DIALS[:directions]]
    return out


def stylegan2(names=(), rests=(), curves=(), levels=(), ranges=(),
              directions: int = DIRECTIONS) -> Layout:
    """A converted StyleGAN2's layout: `adopted`, with each direction dial saying which
    style range it comes from."""
    notes = w_direction_notes(ranges)
    out = []
    for knob in adopted(names, rests, curves, levels, directions).knobs:
        i = direction_index(knob.name)
        if i is not None and i < len(notes):
            knob = replace(knob, blurb=notes[i])
        out.append(knob)
    return Layout(tuple(out))


def w_direction_notes(ranges) -> tuple[str, ...]:
    """One description per direction dial, from its `(range name, pixel stages)`."""
    total = collections.Counter(name for name, _pixels in ranges)
    seen: collections.Counter = collections.Counter()
    out = []
    for name, pixels in ranges:
        seen[name] += 1
        out.append(f"style direction {seen[name]} of {total[name]} in this model's {name} "
                   f"range -- {pixels} -- from the SVD of the affines those style vectors "
                   f"feed. {DIRECTION_TAIL}")
    return tuple(out)


#: A derived dial's description, by the prefix of its name. The mechanism is known; what it
#: looks like is a property of the model.
DERIVED_BLURB = {
    "noise": "the network's own noise at the {h}-tall stage, frozen at load and brought in "
             "as this rises. Derived from the graph and measured on it.",
    "gain": "the gain on everything the {h}-tall stage hands upward. Derived from the graph "
            "and measured on it.",
    # A StyleGAN2's truncation per style range: only that family names a dial `w_*`.
    "w": "how far this model's {h} styles are pulled toward the average `w` it was trained "
         "around -- truncation, for that range of layers alone. Below its rest the picture is "
         "more typical of what the model was trained on and less extreme; above it, less "
         "typical. Derived from the graph and measured on it.",
}


def adopted(names, rests, curves, levels, directions: int = DIRECTIONS) -> Layout:
    """The layout of a model whose dials were measured rather than written by hand: one
    rest, curve and level per name, as `calibrate` produced them."""
    knobs = shared_knobs(directions)
    for name, rest, curve, level in zip(names, rests, curves, levels, strict=True):
        family = name.split("_")[0] if name.startswith(("noise_", "gain_", "w_")) else name
        blurb = DERIVED_BLURB.get(family, "a setting this graph declares.")
        knobs.append(Knob(
            name, float(rest), blurb.format(h=name.partition("_")[2]), group="MODEL",
            writes=(Write(name, evenly(curve)),) if len(curve) else (),
            # Every derived dial rests at a gain of 1.0, which is the model as trained.
            poles=("down", "up", "as trained"), measured=float(level)))
    return Layout(tuple(knobs))


def per_model(layout) -> tuple[str, ...]:
    """The dials whose meaning belongs to one model: its MODEL block and its directions."""
    return tuple(k.name for k in layout.knobs
                 if k.group == "MODEL" or direction_index(k.name) is not None)
