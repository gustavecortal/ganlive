"""A preset: the rules connecting what the drums do to what the picture does."""
from __future__ import annotations

import json
import math
import threading
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path

from ganlive.dials.table import MOTION, Surface, clamp01
from ganlive.files import next_path, remember
from ganlive.walk import WalkConfig

AMOUNT_MAX = 1.0


def clamp_amount(x: float) -> float:
    """Hold a push inside `+/-AMOUNT_MAX`, keeping its sign."""
    return max(-AMOUNT_MAX, min(AMOUNT_MAX, x))


@dataclass
class Impulse:
    """One drum hit pushing one dial, and letting go."""

    track: str
    dial: str
    amount: float
    decay: float = 0.25
    velocity: float = 0.7
    attack: float = 0.0

    def __post_init__(self) -> None:
        self.amount = clamp_amount(self.amount)

    def value(self, since: float, velocity: float) -> float:
        if since >= self.decay * 6.0:
            return 0.0
        shape = math.exp(-since / max(self.decay, 1e-4))
        if self.attack > 0.0 and since < self.attack:
            shape *= since / self.attack
        return self.amount * shape * ((1.0 - self.velocity) + self.velocity * velocity)


@dataclass
class Macro:
    """A slow move: one measurement of the audio drives one dial over a range."""

    source: str
    dial: str
    src_lo: float
    src_hi: float
    out_lo: float
    out_hi: float
    glide: float = 0.8

    def map(self, value: float) -> float:
        if self.src_hi == self.src_lo:
            return self.out_lo
        u = (value - self.src_lo) / (self.src_hi - self.src_lo)
        return self.out_lo + clamp01(u) * (self.out_hi - self.out_lo)


@dataclass
class Preset:
    """One complete performance setting."""

    name: str
    blurb: str
    dials: dict[str, float] = field(default_factory=dict)
    impulses: list[Impulse] = field(default_factory=list)
    macros: list[Macro] = field(default_factory=list)
    base_seed: int = 5
    loop_segments: int = 0
    home_every: int = 0


RULE_KINDS = {"impulses": Impulse, "macros": Macro}


def _spare(obj, skip: tuple[str, ...] = ()) -> dict:
    """Every field of a dataclass that is not simply its default."""
    out = {}
    for f in fields(obj):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        if f.default is not MISSING and value == f.default:
            continue
        if f.default_factory is not MISSING and value == f.default_factory():
            continue
        out[f.name] = value
    return out


def to_dict(preset: Preset) -> dict:
    """A preset as plain data: only what differs from a bare one, rules included."""
    out = _spare(preset, skip=tuple(RULE_KINDS))
    for name in RULE_KINDS:
        rules = getattr(preset, name)
        if rules:
            out[name] = [_spare(rule) for rule in rules]
    return out


def from_dict(data: dict) -> Preset:
    """The other direction. Anything unrecognised is refused rather than dropped."""
    kwargs: dict = {}
    known = {f.name for f in fields(Preset)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"{key!r} is not part of a setting; have {', '.join(sorted(known))}")
        cls = RULE_KINDS.get(key)
        if cls is None:
            kwargs[key] = value
            continue
        takes = {f.name for f in fields(cls)}
        rules = []
        for rule in value:
            unknown = set(rule) - takes
            if unknown:
                raise KeyError(f"{', '.join(sorted(unknown))} is not part of a "
                               f"{cls.__name__.lower()}; have {', '.join(sorted(takes))}")
            rules.append(cls(**rule))
        kwargs[key] = rules
    return Preset(**kwargs)


class PresetRunner:
    """Turns one frame of audio measurements into dial values, then applies them."""

    def __init__(self, preset: Preset, channel_of: dict[str, int], fps: float,
                 channels: int = 0, layout=None) -> None:
        self.channel_of = channel_of
        self.fps = fps
        self.channels = int(channels) or max(channel_of.values(), default=-1) + 1
        #: `None` is the spine -- the dials every model has. `adopt` replaces it with the
        #: loaded model's the moment one arrives, which is before any frame is drawn.
        self.surface = Surface(layout=layout)
        self.walk_cfg = WalkConfig()

        self._sources: dict[str, dict[str, float]] = {}
        self._rank: dict[str, int] = {}
        self._writing = threading.Lock()
        self.hands: dict[str, float] = {}
        self.hands_from: dict[str, str] = {}

        self._macro_state: dict[str, float] = {}
        self._velocity: dict[int, float] = {}
        #: Dials this model can actually write, from `bank.live_dials`. Empty means no model
        #: has said -- every offline tool -- and then everything is playable.
        self.live: frozenset[str] = frozenset()
        #: The model `use_model` last finished switching to, or `None` before the first. Written
        #: last, so a reader on another thread sees the old model whole or the new one whole.
        self.model = None
        #: Per dial: frames it moved, frames a hand held it, furthest it got from rest.
        self.usage: dict[str, list[float]] = {}
        self._last: dict[str, float] = {}
        self.load(preset)

    def playable(self, name: str) -> bool:
        """Whether writing this dial would reach the loaded model."""
        return not self.live or name in self.live

    def use_model(self, model) -> None:
        """The model changed; take its dials, re-decide what is writable, re-read the rules."""
        if not hasattr(model, "layout"):
            raise TypeError(
                f"use_model takes the Model, not {type(model).__name__} -- the layout comes "
                f"with it. This signature used to accept a bare set of live names too, and "
                f"that arm skipped the relayout: the surface kept the dials it was built "
                f"with while the strip drew the loaded model's, and the two agreed again "
                f"only after a switch away and back.")
        live = frozenset(model.dials_live)
        if live != self.live or model.layout != self.surface.layout:
            self.live = live
            self.surface.relayout(model.layout)
            self._last.clear()
            self.load(self.preset)                # re-reports the rules this model cannot run
            self._replace(dict(self._sources))   # and lets go of any hand on a dead dial
        self.model = model

    def load(self, preset: Preset) -> None:
        """Swap in a different preset without rebuilding anything that addresses this runner."""
        self.preset = replace(preset, dials=dict(preset.dials),
                             impulses=list(preset.impulses), macros=list(preset.macros))
        preset = self.preset
        base = dict(self.surface.layout.rests)
        impulses, dropped = [], []
        # A dial that does not exist is reported, not filtered: this used to check the track only, so
        # a setting saved before a rename loaded as `still` wearing its old name, silently.
        known = self.surface.layout
        for name, value in preset.dials.items():
            if name in known:
                base[name] = clamp01(value)
            else:
                dropped.append(f"{name} (no such dial)")
        for imp in preset.impulses:
            if imp.dial not in known:
                dropped.append(f"{imp.track}->{imp.dial} (no such dial)")
                continue
            if not self.playable(imp.dial):
                dropped.append(f"{imp.track}->{imp.dial} (not on this model)")
                continue
            if imp.track == "*":
                impulses.append((imp, -1))
                continue
            channel = self.channel_of.get(imp.track)
            if channel is None:
                dropped.append(f"{imp.track}->{imp.dial} (no such track in this kit)")
            elif channel >= self.channels:
                dropped.append(
                    f"{imp.track}->{imp.dial} (channel {channel}, kit has {self.channels})")
            else:
                impulses.append((imp, channel))
        macros = []
        for mac in preset.macros:
            if mac.dial not in known:
                dropped.append(f"{mac.source}->{mac.dial} (no such dial)")
            elif not self.playable(mac.dial):
                dropped.append(f"{mac.source}->{mac.dial} (not on this model)")
            else:
                macros.append((mac, 1.0 - math.exp(-1.0 / max(mac.glide * self.fps, 1e-6))))
        self._base, self._impulses, self._macros, self.dropped = base, impulses, macros, dropped
        self._macro_state = {}
        self.walk_cfg.base_seed = preset.base_seed
        self.walk_cfg.loop_segments = preset.loop_segments
        self.walk_cfg.home_every = preset.home_every


    def route(self, track: str, dial: str, amount: float | None = None) -> bool:
        """Wire a drum to a dial, or unwire it. Returns True if the rule now exists."""
        return self.wire([track], dial, amount)

    def wire(self, tracks, dial: str, amount: float | None = None) -> bool:
        """Wire a whole kit channel to a dial, or unwire it. Returns True if it now fires.

        **Every track at once, because a channel is what fires.** A kit puts RS and CP on one
        pair of Overbridge outputs, and downstream nothing can tell them apart: the rule is
        stored per track but resolved to a channel. Toggling only the first of a shared pair
        left a cell that was lit by the second and could not be cleared by clicking it, and
        wrote a track name into the saved preset that was not the one under the pointer."""
        tracks = list(tracks)
        unknown = [t for t in tracks if t not in self.channel_of]
        if dial not in self.surface.layout or unknown or not tracks:
            raise KeyError(f"cannot route {'/'.join(tracks) or '<nothing>'!r} to {dial!r}")
        if not self.playable(dial):
            raise KeyError(f"{dial!r} is not on the loaded model, so wiring a drum to it "
                           f"would be a rule that fires and does nothing")
        on = set(tracks)
        kept = [imp for imp in self.preset.impulses
                if not (imp.track in on and imp.dial == dial)]
        added = len(kept) == len(self.preset.impulses)
        if added:
            if amount is None:
                amount = -0.3 if self._base.get(dial, 0.0) > 0.6 else 0.3
            attack = 0.06 if dial in MOTION else 0.0
            kept += [Impulse(track, dial, amount=amount, attack=attack) for track in tracks]
        self.preset.impulses = kept
        self.load(self.preset)
        return added

    def routing(self) -> dict[str, list[tuple[Impulse, int]]]:
        """`dial -> [(rule, channel)]` for rules that will fire. `-1` is any hit."""
        out: dict[str, list[tuple[Impulse, int]]] = {}
        for imp, channel in self._impulses:
            out.setdefault(imp.dial, []).append((imp, channel))
        return out

    def _rules_on(self, dial: str, channel: int) -> list[Impulse]:
        """Every resolved rule that pushes `dial` when `channel` fires."""
        return [imp for imp, ch in self._impulses if imp.dial == dial and ch == channel]

    def amount_on(self, dial: str, channel: int) -> float | None:
        """The strength at `(dial, channel)`, or None if nothing is wired there."""
        return max((imp.amount for imp in self._rules_on(dial, channel)), key=abs, default=None)

    def set_amount_on(self, dial: str, channel: int, amount: float) -> float | None:
        """Set how hard a cell pushes. Returns the new amount, or None if nothing is wired."""
        rules = self._rules_on(dial, channel)
        if not rules:
            return None
        amount = clamp_amount(amount)
        for imp in rules:
            imp.amount = amount
        return amount


    def hold(self, source: str, values: dict[str, float], priority: int = 0) -> None:
        """`source` is now holding exactly these dials. Anything it held before is let go."""
        with self._writing:
            self._rank[source] = priority
            self._replace({**self._sources, source: dict(values)})

    def held_by(self, source: str) -> dict[str, float]:
        """What one source is holding. A copy, so a caller cannot edit the live set by hand."""
        return dict(self._sources.get(source, {}))

    def free(self, source: str, name: str | None = None) -> None:
        """Let go of one dial for one source, or of everything that source holds."""
        with self._writing:
            held = self._sources.get(source)
            if not held:
                return
            keep = {} if name is None else {k: v for k, v in held.items() if k != name}
            self._replace({**self._sources, source: keep})

    def _replace(self, sources: dict[str, dict[str, float]]) -> None:
        """Swap in a whole new set of sources and the flat dict the loop reads."""
        merged: dict[str, float] = {}
        owner: dict[str, str] = {}
        order = {n: i for i, n in enumerate(sources)}
        for name in sorted(sources, key=lambda n: (self._rank.get(n, 0), order[n])):
            values = {k: v for k, v in sources[name].items() if self.playable(k)}
            merged.update(values)
            owner.update(dict.fromkeys(values, name))
        self._sources = sources
        self.hands = merged
        self.hands_from = owner

    def observe(self, onsets) -> None:
        """Record what arrived this frame."""
        for ch, vel, _ago in onsets:
            self._velocity[ch] = clamp01(vel)

    def apply(self, since, features: dict, knobs) -> None:
        """Write this frame's whole control state."""
        surface = self.surface
        surface.values.update(self._base)
        surface.set_held(self.hands)

        for macro, k in self._macros:
            target = macro.map(float(features.get(macro.source, 0.0)))
            cur = self._macro_state.get(macro.dial, target)
            cur += (target - cur) * k
            self._macro_state[macro.dial] = cur
            surface.set(macro.dial, cur)

        reaction = surface["reaction"] * 2.0
        elapsed_by_channel = since.tolist() if hasattr(since, "tolist") else list(since)
        since_any = min(elapsed_by_channel) if elapsed_by_channel else 1e6
        for imp, ch in self._impulses:
            if ch < 0:
                elapsed, velocity = since_any, 1.0
            else:
                elapsed, velocity = elapsed_by_channel[ch], self._velocity.get(ch, 1.0)
            add = imp.value(elapsed, velocity) * reaction
            if add:
                surface.add(imp.dial, add)

        surface.apply(knobs, self.walk_cfg)
        knobs.commit()
        self._tally()

    def _tally(self) -> None:
        """Count this frame against every dial, so the end of a run can say which were played."""
        rests, held, last = self.surface.layout.rests, self.hands, self._last
        for name, value in self.surface.values.items():
            use = self.usage.get(name)
            if use is None:
                use = self.usage[name] = [0, 0, 0.0]
            if name in last and value != last[name]:
                use[0] += 1
            if name in held:
                use[1] += 1
            use[2] = max(use[2], abs(value - rests.get(name, value)))
            # Written back in the loop that already visits every dial, not copied whole a
            # frame. `use_model` clears it, so a dial the new model lacks leaves no stale value.
            last[name] = value

    def usage_report(self) -> list[str]:
        """One line per dial that moved, most-moved first, then the dials nothing touched."""
        moved = sorted((n for n, u in self.usage.items() if u[0]), key=lambda n: -self.usage[n][0])
        out = []
        for name in moved:
            frames, held, far = self.usage[name]
            hand = f", in hand {held / self.fps:.1f} s" if held else ""
            out.append(f"{name:<10} moved {frames / self.fps:6.1f} s, "
                       f"up to {far:.2f} from rest{hand}")
        still = [n for n, u in self.usage.items() if not u[0]]
        if still:
            out.append(f"never moved: {', '.join(still)}")
        return out


#: What the interface starts on: every dial at its resting value and nothing wired. The eight
#: shipped presets are gone -- the routing grid takes seconds to fill in, and what a setting is
#: worth is whether it was found at these controls.
DEFAULT = Preset(
    name="default",
    blurb="Every dial at rest, nothing routed. Wire the drums up in the routing grid and "
          "save what you find.",
)


#: `Positions` writes this into the same folder the settings live in, and `Library` scans that
#: folder by extension. One definition, because the two have to agree or the hand positions load
#: as a setting: they did, and every launch printed a "settings that would not load" line for a
#: file that was never one -- which is where a genuinely broken setting would have hidden.
POSITIONS_NAME = "positions.json"


class Library:
    """The settings a performance can move between, including the ones you found yourself."""

    def __init__(self, folder) -> None:
        self.folder = Path(folder)
        self.broken: list[str] = []
        self.presets = [DEFAULT] + self._found()
        self.shipped = {DEFAULT.name}
        self.index = 0

    def _found(self) -> list[Preset]:
        """Every setting saved in the folder, named by its file rather than by its contents."""
        out = []
        for path in sorted(self.folder.glob("*.json")) if self.folder.is_dir() else []:
            if path.name == POSITIONS_NAME:
                continue
            try:
                out.append(replace(from_dict(json.loads(path.read_text("utf-8"))),
                                   name=path.stem))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.broken.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return out

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.presets]

    @property
    def current(self) -> Preset:
        return self.presets[self.index]

    def step(self, delta: int) -> Preset:
        """Next or previous setting. Wraps, so a rotation has no ends to fall off."""
        self.index = (self.index + delta) % len(self.presets)
        return self.current

    def select(self, name: str) -> Preset:
        self.index = self.names.index(name)
        return self.current

    def save(self, preset: Preset) -> Path:
        """Write this setting out, and make it the current one."""
        path = self.folder / f"{preset.name}.json"
        if preset.name in self.shipped or not path.exists():
            path = next_path(self.folder, "take", ".json")
            preset = replace(preset, blurb=f"Found at the controls, from {preset.name}.")
        preset = replace(preset, name=path.stem)
        path.write_text(json.dumps(to_dict(preset), indent=2), encoding="utf-8")
        if preset.name in self.names:
            self.index = self.names.index(preset.name)
            self.presets[self.index] = preset
        else:
            self.presets.append(preset)
            self.index = len(self.presets) - 1
        return path


class Positions:
    """Where your hands left each model's own dials, so a switch lands where you left it.

    Hands only: a dial nobody is holding follows the preset, and the preset is shared. Keyed by
    `bank.slug_for`, written on every switch and at the end, read back the next time."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.data: dict[str, dict[str, float]] = {}
        self.trouble = ""
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError) as exc:                                # noqa: BLE001
            self.trouble = f"{self.path} was ignored: {exc}"

    def stash(self, key: str, runner: PresetRunner, source: str, names) -> dict[str, float]:
        """Remember what `source` holds on `names`, let go of them, and write it down."""
        names = set(names)
        held = {n: round(v, 4) for n, v in runner.held_by(source).items() if n in names}
        if held or key in self.data:
            self.data[key] = held
            self._save()
        for name in held:
            runner.free(source, name)
        return held

    def recall(self, key: str, runner: PresetRunner, source: str, priority: int,
               names) -> dict[str, float]:
        """Put `source`'s hands back where they were on this model, on dials it still has."""
        names = set(names)
        found = {n: clamp01(float(v)) for n, v in self.data.get(key, {}).items() if n in names}
        if found:
            runner.hold(source, {**runner.held_by(source), **found}, priority)
        return found

    def _save(self) -> None:
        self.trouble = remember(self.path, json.dumps(self.data, indent=1, sort_keys=True))
