"""A bank of loaded models, one playing at a time, and the shelf of models on disk that can join it.

`build` turns checkpoint paths into a `Bank`. Every model plays on the engine, all on the one
wgpu device the first model found fastest. How each kind of file becomes an engine model is in
`families`, and file naming is in `checkpoints`.
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ganlive.checkpoints import (
    admit,
    checkpoint_for,
    checkpoints_in,
    index_of,
    is_engine,
    is_published_engine,
    label_for,
)
from ganlive.clock import WalkConfig
from ganlive.dials.table import live_dials
from ganlive.engine.screen import EngineStage
from ganlive.families import LoadOptions, config_of, family_of
from ganlive.walk import SlerpWalk
from ganlive.window import fit_height


def frame_size(cfg, want: int | None, screen=None) -> tuple[int, int]:
    """`(height, width)` one model's frames are produced at: its native size, fitted by
    `window.fit_height`, with the width rounded to keep the aspect and stay even."""
    lad = cfg.ladder
    h = fit_height(lad.height, lad.width, want, screen)
    return h, max(2, round(lad.width * h / lad.height) // 2 * 2)


@dataclass
class Model:
    """One prepared generator and the settings vector that steers it."""

    path: Path
    net: object
    cfg: object
    settings: object
    #: The dials this model offers.
    layout: object
    graphs: int = 0
    compile_s: float = 0.0
    #: The measured `dials.derive.Directions`, or None when they could not be derived.
    directions: object = None
    #: The same basis as the `(n, nz)` float32 array the walk adds. Held here so the walk's
    #: cache, keyed on the array's identity, survives a switch away and back.
    rows: object = None
    #: Every dial that provably reaches this model. The strip draws the rest dark.
    dials_live: frozenset = frozenset()
    #: The `(bands, w_dim)` tensor the walk writes a StyleGAN2's `w` push into, or None on
    #: families that steer `z`. On a captured model, the host buffer the graph reads from.
    push: object = None

    def __post_init__(self) -> None:
        if self.rows is None and self.directions is not None:
            self.rows = np.asarray(self.directions.basis, np.float32)

    @functools.cached_property
    def name(self) -> str:
        """Cached: it is read on the frame path and never changes."""
        return label_for(self.path)


def _prepare(path, gpu, options: LoadOptions) -> Model:
    """One model made ready to play on the engine, on the wgpu device `gpu` (the bank's), or
    on this machine's fastest for it when the bank has none yet. Its dials were measured when
    it was converted, so which of them are live is read off the program, not measured."""
    family = family_of(path)
    started = time.perf_counter()
    got = family.prepare(path, gpu, None, options)
    return Model(path=Path(path), net=got.net, cfg=got.cfg, settings=got.settings,
                 layout=got.layout, compile_s=time.perf_counter() - started,
                 directions=got.directions, push=got.push,
                 dials_live=live_dials(got.settings, got.directions, got.layout))


def _warm(stage: EngineStage, model: Model, size) -> int:
    """One warm-up frame through the whole after-generator path, at the size it will play."""
    stage.resize(*size)
    probe = model.net(np.zeros((1, model.cfg.nz), np.float32))
    return stage.warm(stage.step(probe))


@dataclass
class Bank:
    """The models, the work that happens after any of them, and which one is playing."""

    models: list[Model]
    stage: EngineStage
    #: Where the walk keeps its latent: the host, which is where the engine reads it.
    device: str
    index: int = 0
    #: How every model in this bank was prepared, including the ones added later.
    options: LoadOptions = field(default_factory=LoadOptions)
    #: The precision the bank plays in. None is the device's own; see `device.playback_dtype`.
    dtype: object = None
    #: The height asked for: None fits the screen, 0 is native. See `window.fit_height`.
    height_want: int | None = 0
    screen: tuple | None = None
    #: Every walk built by `walk()`, re-pointed at the new model's directions by `use`.
    _walks: list = field(default_factory=list)
    compile_s: float = 0.0
    graphs: int = 0

    def __post_init__(self) -> None:
        if self.dtype is None:
            self.dtype = torch.float32

    @property
    def gpu(self):
        """The wgpu device every model of this bank plays on."""
        return self.stage.device

    def sync(self) -> None:
        """Wait until everything queued for the card is done."""
        self.stage.sync()

    @property
    def current(self) -> Model:
        return self.models[self.index]

    @property
    def cfg(self):
        """The playing model's config."""
        return self.current.cfg

    @property
    def name(self) -> str:
        return self.current.name

    @property
    def width(self) -> int:
        """The width frames are produced at now: the stage's."""
        return self.stage.width

    @property
    def height(self) -> int:
        return self.stage.height

    def size_of(self, model: Model) -> tuple[int, int]:
        """`(height, width)` this model's frames are produced at in this session."""
        return frame_size(model.cfg, self.height_want, self.screen)

    def use(self, index: int) -> Model:
        """Play a different model. Wraps, so stepping past the end comes back round."""
        self.index = index % len(self.models)
        self._rewire()
        return self.current

    def _rewire(self) -> None:
        """Point the stage and every walk at the playing model: its frame size, its latent
        width and directions, and where it takes the latent and the push."""
        model = self.current
        # An ONNX graph and a captured one read the latent from the host; a compiled module
        # from the card.
        on_host = bool(getattr(model.net, "latent_on_host", False))
        self.stage.resize(*self.size_of(model))
        for walk in self._walks:
            walk.cfg.directions = model.rows
            walk.cfg.latent_on_host = on_host
            walk.cfg.push_into = model.push
            walk.retarget(model.cfg.nz)

    def index_of(self, path) -> int | None:
        """Where a checkpoint sits in this bank, or None."""
        return index_of(self.models, path)

    def add(self, target) -> Model:
        """Load one more model into this bank and return it. Does not switch to it."""
        path = checkpoint_for(Path(target))
        admit(self.models, path)
        playing = (self.height, self.width)
        model = _prepare(path, self.gpu, self.options)
        try:
            warmed = _warm(self.stage, model, self.size_of(model))
        finally:
            self.stage.resize(*playing)
        # Joined only once it has run, so a model whose warm-up raised is not left in the bank.
        self.models.append(model)
        self.graphs += model.graphs + warmed
        self.compile_s += model.compile_s
        return model

    def report(self) -> str:
        """One line naming every path that could silently have fallen back to a slow one."""
        live = [k for k, on in self.stage.compiled.items() if on] or ["none"]
        models = ", ".join(f"{m.name} ({m.cfg.ladder.width}x{m.cfg.ladder.height}, "
                           f"{m.compile_s:.0f}s)" for m in self.models)
        pinned = {"settings": self.current.settings.pinned,
                  **(self.stage.pinned() or {"frames": "after the first one"})}
        return (f"{self.width}x{self.height} out of {self.current.cfg.ladder.width}x"
                f"{self.current.cfg.ladder.height}, {self.graphs} graph(s) in "
                f"{self.compile_s:.0f}s, "
                f"{len(self.current.settings.names)} settings, "
                f"compiled conversions: {'+'.join(live)}, "
                f"pinned: {pinned}\n  models: {models}")

    def walk(self, config=None, dtype=None):
        """A walk wired to the playing model's directions, kept wired across switches."""
        config = config if config is not None else WalkConfig()
        walk = SlerpWalk(self.cfg.nz, self.device, config, dtype=dtype or self.dtype)
        self._walks.append(walk)
        self._rewire()
        return walk


@dataclass
class Shelved:
    """One model on disk, as the picker shows it."""

    name: str
    path: Path
    loaded: bool
    #: Why this row cannot be loaded: an unreadable file.
    why: str = ""
    #: What it is, as `1024x1024 z512`, shown before it is loaded.
    note: str = ""


class Shelf:
    """Every model on disk, which of them are loaded, and the one waiting to be."""

    def __init__(self, bank: Bank, root="runs") -> None:
        self.bank = bank
        self.root = Path(root)
        self.pending: str | None = None
        self.note = ""
        self._cfgs: dict[Path, object] = {}
        self._listing: list[Shelved] | None = None

    def _config(self, path: Path):
        if path not in self._cfgs:
            self._cfgs[path] = config_of(path)
        return self._cfgs[path]

    def entries(self) -> list[Shelved]:
        """Every model under `root`, scanned once.

        Call it before the window opens: reading a config unpickles the file and holds the
        GIL, which on the window's thread would stall the picture."""
        if self._listing is None:
            self._listing = self._scan()
        return self._listing

    def _scan(self) -> list[Shelved]:
        """Loaded models first, then the rest newest first. Only an unreadable file is refused."""
        here = {m.path for m in self.bank.models}
        out = []
        for path in self._models():
            why, note = "", ""
            if path in here:
                pass
            else:
                try:
                    cfg = self._config(path)
                except ValueError:
                    why = "not playable"          # a file of the wrong kind, such as an adapter
                except Exception as exc:          # noqa: BLE001  a broken file is not a crash
                    why = f"unreadable: {type(exc).__name__}"
                else:
                    note = f"{cfg.ladder.width}x{cfg.ladder.height} z{cfg.nz}"
            out.append(Shelved(name=label_for(path), path=path, loaded=path in here,
                               why=why, note=note))
        return sorted(out, key=lambda s: (not s.loaded, -s.path.stat().st_mtime))

    def _models(self) -> list[Path]:
        """Every model under `root`: each engine model, the newest checkpoint of each training
        run, and every checkpoint of a flat folder. Empty when there is no `root`."""
        if not self.root.is_dir():
            return []
        found: list[Path] = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if is_engine(folder):
                if is_published_engine(folder):
                    found.append(folder)
                continue
            # Inside `runs/engine/`. A checkpoint's own conversion is listed as the checkpoint.
            found += sorted(p for p in folder.iterdir() if is_published_engine(p))
            history = folder / "checkpoints"
            found += checkpoints_in(history)[-1:] if history.is_dir() else checkpoints_in(folder)
        return found

    def request(self, entry: Shelved) -> None:
        """Ask for a model, from whichever thread the click arrived on."""
        if entry.loaded or entry.why:
            return
        self.pending = str(entry.path)

    def service(self):
        """Do the waiting load, on the thread that owns the card. Returns the new model or None."""
        want, self.pending = self.pending, None
        if want is None:
            return None
        try:
            model = self.bank.add(want)
        except Exception as exc:  # noqa: BLE001  reported, never raised into the frame loop
            self.note = f"{label_for(want)}: {exc}"
            return None
        self._listing = self._scan()
        self.note = f"loaded {model.name}, {model.compile_s:.1f}s to compile"
        return model


def build(checkpoints: list, device: str | None = None, height: int | None = 0, dtype=None,
          screen=None, options: LoadOptions | None = None) -> Bank:
    """Load and prepare every checkpoint on the engine, warm the stage at each one's size, and
    return the bank playing the first. The first model picks the wgpu device (its fastest
    backend here) and the others join it. `device` and `dtype` are unused: the engine chooses its
    own backend and plays in fp16. `height` is as `window.parse_height` returns."""
    options = options or LoadOptions()
    paths = [checkpoint_for(Path(t)) for t in checkpoints]

    t_all = time.perf_counter()
    models = [_prepare(paths[0], None, options)]
    gpu = models[0].net.model.device
    for path in paths[1:]:
        admit(models, path)
        models.append(_prepare(path, gpu, options))

    out_height, out_width = frame_size(models[0].cfg, height, screen)
    stage = EngineStage(gpu, out_height, out_width)

    # Each at its own size, `models[0]` last, so the stage is left at the size that plays.
    warmed = sum(_warm(stage, m, frame_size(m.cfg, height, screen)) for m in reversed(models))
    stage.resize(out_height, out_width)

    return Bank(models=models, stage=stage, device="cpu",
                graphs=sum(m.graphs for m in models) + warmed,
                compile_s=time.perf_counter() - t_all, dtype=torch.float32,
                height_want=height, screen=screen, options=options)
