"""Everything needed to turn a checkpoint -- or several -- into something a preset can drive."""
from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from ganlive.dials import fastgan_dials as F
from ganlive.dials import steer as K
from ganlive.dials import table as S
from ganlive.dials.derive import FLOOR_LEVELS, RANDOM_FLOOR
from ganlive.frame import FrameStage


@dataclass(frozen=True)
class LoadOptions:
    """Everything that decides how a checkpoint becomes a playable model.

    One object rather than six parameters through five signatures, because **the bank grows
    after launch**. The shelf loads a model mid-session through `Bank.add`, and a setting that
    reached `build` but not `add` changes the instrument under the hand with nothing on the
    strip to explain it. `exact` was exactly that -- taken by `build`, never passed on, so a
    StyleGAN2 added from the shelf ran half precision however it had been asked for. Carried
    on the `Bank`, the invariant is structural instead of remembered once per setting."""

    compile_net: bool = True
    #: Record the compiled forward as one device graph and replay it, instead of asking for a
    #: hundred-odd kernels a frame from Python. Exact -- 0.0000 8-bit levels against the
    #: compiled net -- and worth 9.11 ms to 8.07 on a 2048px FastGAN, 14.80 to 11.57 on a StyleGAN2.
    capture: bool = True
    noise_seed: int = 0
    #: Measure each model's grain gains at load rather than using the table measured on
    #: one checkpoint. Only the grain: the dial sweep and the dead-dial gate always run. Named
    #: for what it does after `--no-calibrate` was read as "show me none of this model's
    #: controls" and hid a whole StyleGAN2's MODEL block.
    measure_grain: bool = True
    #: Run a converted StyleGAN2 in the precision its own file declares.
    exact: bool = False
    #: Times what a random direction of the same length moves, for a derived direction to earn
    #: a dial. See `directions.RANDOM_FLOOR` for why it is relative.
    direction_floor: float = RANDOM_FLOOR


#: The one suffix that means "a graph, not a checkpoint". Everything that branches on the
#: backend branches on this, so there is one spelling of the question.
ONNX = ".onnx"


def is_onnx(path) -> bool:
    return Path(path).suffix.lower() == ONNX


def run_step(path) -> tuple[str, str]:
    """A checkpoint's identity as the pair `(run, step)` -- not as a spelling of it."""
    path = Path(path)
    if is_onnx(path):
        # An export is named `<run>-<step>.onnx` in one flat folder, so its parent says nothing.
        # `onnx` stays in the step so an export and its checkpoint are never two names for one row.
        run, _dash, step = path.stem.rpartition("-")
        return run or path.stem, f"{step.lstrip('0') or step} onnx"
    run = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    return run, path.stem.lstrip("0") or path.stem


def label_for(path: Path) -> str:
    """A checkpoint's short name, as `run step` -- `my-run 72000`."""
    return "{} {}".format(*run_step(path))


def slug_for(path) -> str:
    """The same identity as a filename and a JSON key -- `my-run-72000`."""
    return "{}-{}".format(*run_step(path))


def index_of(models, path) -> int | None:
    """Where a checkpoint sits in a bank, or None."""
    path = Path(path)
    return next((i for i, m in enumerate(models) if m.path == path), None)


def is_stylegan2(path) -> bool:
    """Whether this is a converted StyleGAN2. Three kinds of file, and two of them `.pt`."""
    from ganlive.models.stylegan2 import is_stylegan2 as said

    return not is_onnx(path) and said(path)


def config_of(path):
    """A model's latent width and output size, whichever kind of file it is."""
    return family_of(path).config_of(path)


def admit(models, path) -> None:
    """Refuse, with a `ValueError` saying why, a checkpoint that may not join this bank."""
    if models and index_of(models, path) is not None:
        raise ValueError(f"{label_for(path)} is already in this bank")


def frame_size(cfg, want: int | None, screen=None) -> tuple[int, int]:
    """What one model's frames are produced at: its own native size, fitted to the screen."""
    lad = cfg.ladder
    h = fit_height(lad.height, lad.width, want, screen)
    return h, max(2, round(lad.width * h / lad.height) // 2 * 2)


def _conversions():
    """The three host conversions, compiled. They are functions of a frame, not of a net."""
    from ganlive.pixels import compiled_to_bgra, compiled_to_nv12, compiled_to_rgb

    return compiled_to_nv12(), compiled_to_rgb(), compiled_to_bgra()


def _onnx_config(path):
    from ganlive.models.onnx import config_of as said

    return said(path)


def _stylegan2_config(path):
    from ganlive.models.stylegan2 import config_of as said

    return said(path)


def _fastgan_config(path):
    from ganlive.models.fastgan import config_of as said

    return said(path)


def _prepare_onnx(path, device, dtype, conversions, options: LoadOptions):
    """One exported graph made ready to play, with whatever settings it carries."""
    from ganlive.dials import derive as D
    from ganlive.models.onnx import OnnxGenerator

    net = OnnxGenerator(path, device=device)
    print(net.report(), flush=True)
    if options.measure_grain and net.steerable:
        net.knobs.noise_gains = K.calibrate_noise(net, net.knobs, net.cfg.nz, device, dtype)
    found = directions_for(net, net.cfg.nz, device, dtype, floor=options.direction_floor,
                           path=path,
                           read=lambda _net, nz: D.sefa_onnx(path, nz, count=D.CANDIDATES))
    return dict(net=net, cfg=net.cfg, knobs=net.knobs, directions=found), conversions


def open_stylegan2(path, device, exact: bool = False, dtype=None):
    """A converted StyleGAN2 with its seam installed, and the three things read off its tree.

    Shared with `ganlive dials`, which has to open the model the same way the instrument
    does or it derives a basis for a model that is not the one that plays. `dtype` is the
    precision the session plays in, defaulting to the device's; see `S2.half_from_for`."""
    from ganlive.device import playback_dtype
    from ganlive.models import stylegan2 as S2

    net = S2.from_file(path, device,
                       half_from=S2.half_from_for(dtype or playback_dtype(device), exact))
    knobs = K.install_stylegan2(net, device)
    # Read off the uncompiled module tree, before the compile that hides it. Every measurement
    # runs after, against the graph that will actually play.
    return net, knobs, net.mapping.push, S2.style_bands(net)


def _compiled(net, nz: int, device, dtype, options: LoadOptions):
    """The net compiled if this load asks for it, and what that cost. Shared by the families
    that own an `nn.Module`, because the three lines had been written out in both and `LoadOptions`'s
    own docstring names `exact` as a setting that reached one path and not the other."""
    from ganlive.models.capture import compile_and_count

    if not options.compile_net:
        return net, 0, 0.0
    return compile_and_count(net, nz, device, dtype)


def _prepare_stylegan2(path, device, dtype, conversions, options: LoadOptions):
    """A converted StyleGAN2 made ready to play, dials measured rather than remembered."""
    from ganlive.dials import derive as D
    from ganlive.dials import onnx_dials as A
    from ganlive.models import stylegan2 as S2

    net, knobs, push, bands = open_stylegan2(path, device, exact=options.exact, dtype=dtype)
    cfg = net.cfg
    net, graphs, secs = _compiled(net, cfg.nz, device, dtype, options)

    # Directions before layout: `rank` drops what dies before the output, so which style range a
    # surviving dial belongs to -- and the strip labels by range -- is only known after this call.
    found_dirs = directions_for(
        net, cfg.nz, device, dtype, into=push, floor=options.direction_floor, path=path,
        # `CANDIDATES` is per band, so the pool is that many from each of the three.
        read=lambda _net, nz: D.sefa_banded(bands, (D.CANDIDATES,) * len(bands), nz))
    ranges = () if found_dirs is None or not found_dirs.ranges else tuple(
        (name, S2.BAND_PIXELS[name]) for name in found_dirs.ranges)

    # **Not gated on `measure_grain`.** This sweep is where a StyleGAN2's dials come from at
    # all -- a derived dial's curve *is* its measurement -- so skipping it left the strip with a
    # spine and an empty MODEL block while twelve working dials sat on the model unreachable.
    # That gate asks for stock grain gains, which is a question about strength on a family that
    # has `noise_gains`; it never meant "show me none of this model's controls".
    found = A.measured(A.TorchProbe(net, knobs, cfg.nz, device, dtype), knobs.names,
                       size=(cfg.ladder.height, cfg.ladder.width))
    print(found.report(), flush=True)
    layout = S.stylegan2(found.names, [d.rest for d in found.dials],
                         [d.curve for d in found.dials],
                         [d.moved for d in found.dials], ranges)
    return dict(net=net, cfg=cfg, knobs=knobs, graphs=graphs, compile_s=secs, layout=layout,
                push=push, directions=found_dirs), conversions


def _prepare_fastgan(path, device, dtype, conversions, options: LoadOptions):
    """This project's own generator made ready to play."""
    from ganlive.models.fastgan import freeze_noise, load
    from ganlive.models.fold import prepare_for_inference

    net, cfg = load(path, device)
    freeze_noise(net, seed=options.noise_seed)
    prep = prepare_for_inference(net, cfg.nz, device, half=dtype is torch.float16, fold=True,
                                 compile_yuv=conversions is None, compile_net=False)
    if conversions is None:
        conversions = (prep["yuv"], prep["rgb"], prep["bgra"])
    net = prep["net"]

    knobs = K.install(net, device, dtype)
    net, graphs, secs = _compiled(net, cfg.nz, device, dtype, options)
    if options.measure_grain:
        knobs.noise_gains = K.calibrate_noise(net, knobs, cfg.nz, device, dtype)
    return (dict(net=net, cfg=cfg, knobs=knobs, graphs=graphs, compile_s=secs,
                 directions=directions_for(net, cfg.nz, device, dtype, path=path,
                                           floor=options.direction_floor)),
            conversions)


@dataclass(frozen=True)
class Family:
    """One kind of generator file, and everything the bank ever asks about it.

    Five separate places used to re-ask "is this ONNX? is this a StyleGAN2?" -- the config
    reader, the dispatcher, the capture gate, the layout, and the shelf -- so a fourth format
    meant five edits and a disagreement between any two of them was a silent bug. It is one
    record and one lookup now."""

    name: str
    #: Is this file mine? Asked in `FAMILIES` order, so the last may simply say yes.
    owns: object
    #: Latent width and output size, without building the generator.
    config_of: object
    #: `(what the `Model` needs, the conversions this compiled)`.
    prepare: object
    #: The dials it offers, given a path and whatever knobs are already installed.
    layout: object
    #: Whether its forward can be recorded as one device graph.
    capturable: bool = True


def _prepare(path, device, dtype, conversions, options: LoadOptions | None = None):
    """One model made ready to play, its dials verified, and the conversions the bank shares.

    Each family returns what differs -- the `Model` fields it knows -- and the conversions it
    compiled, if it compiled any; the `Model` is built once, here."""
    options = options or LoadOptions()
    family = family_of(path)
    found, conversions = family.prepare(path, device, dtype, conversions, options)
    # Last, because every sweep above reads the module tree or holds two frames side by side to
    # difference them, and a capture offers one output buffer and bakes in the addresses it
    # recorded. Before the gate below, though, so the gate runs *through* the capture: the
    # graph that plays is the graph whose dials were checked.
    # Asked of the family rather than left for `capture` to refuse. It would refuse, but an
    # ONNX load would then carry a line reporting the outcome of a question it can never be
    # asked -- its graph runs under its own runtime, so a recording of the torch stream would
    # hold none of its work.
    if options.capture and family.capturable:
        from ganlive.models.capture import Replay, capture

        # Both torch families build a `K.Knobs`; the push is the StyleGAN2 family's alone.
        knobs, push = found["knobs"], found.get("push")
        feeds = [knobs.vec] + ([push] if push is not None else [])
        found["net"], said = capture(found["net"], found["cfg"].nz, device, dtype, feeds=feeds)
        print(f"graph: {said}", flush=True)
        if isinstance(found["net"], Replay):
            # From here the card reads its settings and its push from the graph's own host
            # buffers, so a dial write and a walk step are host writes; see `Replay`.
            knobs.feed_from(found["net"].twin(knobs.vec))
            if push is not None:
                found["push"] = found["net"].twin(push)
    # Also not gated: this is the gate that draws a dial that reaches nothing dark rather than
    # offering it, and an inert knob that looks live is the one failure this whole path exists
    # to prevent. It drives each unmeasured MODEL dial to both ends -- ten forwards on ours.
    model = verified(Model(path=Path(path), **found), device, dtype)
    return model, conversions or _conversions()


def verified(model: Model, device, dtype) -> Model:
    """The same model with every MODEL dial measured through the path a hand takes.

    The one gate every family passes: a dial that reaches nothing is drawn dark rather than
    offered, whether it was hand-tuned here, swept at load, or read out of a foreign file."""
    layout = K.verify(model.net, model.knobs, model.layout, model.cfg.nz, device, dtype)
    live = live_dials(model.knobs, model.directions, layout)
    writing = [k for k in layout.knobs if k.writes]
    dark = [k for k in writing if k.name not in live]
    print(f"verified: {len(writing) - len(dark)} of {len(writing)} model dial(s) move the "
          f"picture at full travel"
          + ("; dark: " + ", ".join(f"{k.name} {k.measured:.2f}" for k in dark)
             if dark else "") + f" (floor {FLOOR_LEVELS:g} 8-bit levels)", flush=True)
    return replace(model, layout=layout, dials_live=live)


def layout_for(knobs, path):
    """The dials this model offers: its own if it declares them, ours if it is one of ours.

    **Ours is the special case, not the fallback.** Anything foreign with no measurement gets
    the spine and an empty MODEL block, because a derived dial's curve *is* its measurement and
    there is nothing honest to draw without one. Handing it this project's `se_*` gates instead
    would put five dials from one architecture on the strip of another -- drawn dark by
    `verified`, but still named there, still taking the space, and still implying the model has
    something it does not."""
    return family_of(path).layout(knobs, path)


def _onnx_layout(_knobs, path):
    from ganlive.models.onnx import dials_of

    said = dials_of(path)
    # An export with no settings baked in: a plain graph, nothing to steer inside it.
    return (S.adopted(said["settings"], said["rests"], said["curves"], said["levels"])
            if said["curves"] else S.adopted((), (), (), ()))


def _stylegan2_layout(_knobs, _path):
    # Reached by a caller holding a path and no sweep; the bank always measures. `S.stylegan2`
    # rather than `S.adopted` so what that family says on its dials is said in one place.
    return S.stylegan2()


def _fastgan_layout(knobs, _path):
    return F.fastgan(noise_gains=getattr(knobs, "noise_gains", None))


#: Order matters: the suffix is decisive, then the file's own format tag, then what is left.
#: A FastGAN checkpoint says nothing about itself that the others do not, so it is the tail.
FAMILIES = (
    Family("onnx", is_onnx, _onnx_config, _prepare_onnx, _onnx_layout, capturable=False),
    Family("stylegan2", is_stylegan2, _stylegan2_config, _prepare_stylegan2, _stylegan2_layout),
    Family("fastgan", lambda _path: True, _fastgan_config, _prepare_fastgan, _fastgan_layout),
)


def family_of(path) -> Family:
    """Which of `FAMILIES` this file belongs to. Never `None`: the last one takes anything."""
    return next(f for f in FAMILIES if f.owns(path))


def live_dials(knobs, directions, layout=None) -> frozenset:
    """Which dials actually reach this model. Derived, never listed."""
    layout = layout if layout is not None else F.fastgan()
    have = set(getattr(knobs, "index", ()) or ())
    count = 0 if directions is None else len(directions)
    live = set()
    for knob in layout.knobs:
        index = S.direction_index(knob.name)
        if index is not None:
            if index < count:
                live.add(knob.name)
            continue
        # `any`, not `all`: a dial that reaches three of its four bands is still a dial. And
        # reaching the model is not moving the picture -- seven of NVIDIA's FFHQ-1024 band gains
        # measure 0.000 levels, because a modulated convolution demodulates a constant gain back
        # out. One spelling of the floor for both kinds of dial: written twice, a change to it
        # would have left a writing dial and a plain one disagreeing about the same threshold.
        reaches = not knob.writes or any(write.setting in have for write in knob.writes)
        if reaches and (knob.measured is None or knob.measured >= FLOOR_LEVELS):
            live.add(knob.name)
    return frozenset(live)


def directions_for(net, nz: int, device, dtype, read=None, into=None,
                   floor: float = RANDOM_FLOOR, path=None):
    """This model's principal latent directions, measured and ranked, at load.

    A basis derived beside the checkpoint wins over the family's own proposal when there is
    one -- see `ganlive dials`, which reads the whole generator's Jacobian rather than its
    first affine and is far too slow to run with the picture stopped. Here rather than in one
    family's `read`, because nothing about a cached proposal is StyleGAN2's business."""
    from ganlive.dials import derive as D
    from ganlive.dials.onnx_dials import TARGET_LEVELS
    from ganlive.dials.table import DIRECTION_RANGE, DIRECTIONS

    # A pool, not a shortlist: the eigenvalue order is a poor selector in W-space (see
    # `D.CANDIDATES`), so `rank` is handed several times what the strip can show and picks by
    # measurement. `read` sizes its own pool, being the only thing that knows its band count.
    read = read or (lambda net, nz: D.sefa(net, nz, count=D.CANDIDATES))
    try:
        cached = None if path is None else D.saved(
            path, nz, None if into is None else tuple(into.shape), net)
        found = cached if cached is not None else read(net, nz)
    except (ValueError, RuntimeError, KeyError) as exc:                  # noqa: BLE001
        print(f"no latent directions for this model: {exc}", flush=True)
        return None
    # Cheaply first, so the dear passes below run over a shortlist and not the whole pool.
    found = D.shortlist(net, found, device, dtype, DIRECTION_RANGE, into=into,
                        keep=2 * DIRECTIONS)
    # A no-op where "one unit along a unit row" already means something -- `equalise` asks that
    # itself, being handed the basis the answer is a property of. Target is what every derived
    # MODEL dial is equalised to.
    found = D.equalise(net, found, device, dtype, amount=DIRECTION_RANGE,
                       target=TARGET_LEVELS, into=into)
    found = D.rank(net, found, device, dtype, amount=DIRECTION_RANGE, into=into,
                   relative=floor, keep_best=DIRECTIONS)
    if not len(found):
        print(f"no latent direction on this model beats a random one: {found.report()}",
              flush=True)
        return None
    print(f"directions: {found.report()}", flush=True)
    return found


# Warm at the REAL size. A small dummy builds a second graph on the first real frame --
# 381 ms, inside the loop, with nothing to name it.
def _warm(stage: FrameStage, model: Model, device, dtype, size=None) -> int:
    """One warm-up frame through the whole after-generator path. Returns graphs built."""
    if size is not None:
        stage.resize(*size)
    with torch.no_grad():
        probe = model.net(torch.zeros(1, model.cfg.nz, device=device, dtype=dtype))
        return stage.warm(stage.step(probe))


def parse_height(text: str) -> int | None:
    """`auto`, `native`, or a number of pixels -- the three things a height argument can mean."""
    import argparse

    if text == "auto":
        return None
    if text == "native":
        return 0
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a height; use auto, native, or a number of pixels") from None


def screen_size():
    """The desktop's size in pixels, or None if it cannot be asked.

    Beside `fit_height`, which is the only thing that spends it. It lived in `tools.play` while
    the latency harness built its bank without a screen, so the harness sized frames natively
    and the tool sized them to the display -- and at 3072x2048 that is 25 MB a frame across the
    bus instead of 12, which is most of what the window costs."""
    import sys

    # Named for the platform rather than discovered by letting `ctypes.windll` raise: SDL
    # answers everywhere, and asking it first elsewhere saves an exception per call.
    asks = (_win32_screen, _sdl_screen) if sys.platform == "win32" else (_sdl_screen,)
    for ask in asks:
        try:
            size = ask()
        except Exception:  # noqa: BLE001 -- no desktop at all, or no SDL: try the next
            continue
        if min(size) > 0:
            return size
    return None


def _win32_screen():
    """Windows only. Asked before SDL there because it needs no video subsystem started."""
    import ctypes

    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _sdl_screen():
    """Anywhere else: SDL is there for the window anyway, and its init is idempotent."""
    import pygame

    pygame.display.init()
    info = pygame.display.Info()
    return info.current_w, info.current_h


def fit_height(native_h: int, native_w: int, want: int | None, screen=None) -> int:
    """What the frame should be *produced* at, given what a caller asked for."""
    if want is None:
        if not screen or min(screen) <= 0:
            return native_h
        sw, sh = screen
        want = min(sh, round(native_h * sw / native_w))
    return max(2, min(want, native_h)) // 2 * 2 if want else native_h


def checkpoint_for(target: Path) -> Path:
    """The one model a path means: a file, a run whose newest checkpoint is wanted, or a folder of exported
    graphs, whose newest is wanted the same way."""
    target = Path(target)
    if target.is_file():
        return target
    inner = target / "checkpoints"
    folder = inner if inner.is_dir() else target
    found = sorted(folder.glob("*.pt")) + sorted(folder.glob(f"*{ONNX}"))
    if not found:
        raise FileNotFoundError(f"no checkpoint or exported graph at {target}")
    return found[-1]


@dataclass
class Model:
    """One prepared generator and the settings vector that steers it."""

    path: Path
    net: object
    cfg: object
    knobs: object
    graphs: int = 0
    compile_s: float = 0.0
    #: The measured `gan.directions.Directions`, or `None` when they could not be derived.
    directions: object = None
    #: The same basis as the `(n, nz)` float32 array the walk adds. Held here rather than
    #: converted per switch, because `SlerpWalk._offset` caches on the array's identity.
    rows: object = None
    #: Every dial that provably reaches this model. The strip draws the rest dark.
    dials_live: frozenset = frozenset()
    #: The dials this model offers at all -- ours by hand, a foreign graph's out of the file.
    layout: object = None
    #: The `(bands, w_dim)` tensor the walk writes this generator's `w` push into, or `None` on
    #: families that steer `z`, where the walk folds the push into the latent instead. The
    #: mapping's own device tensor on an uncaptured model; on a captured one, the pinned host
    #: twin the graph uploads from every replay.
    push: object = None

    def __post_init__(self) -> None:
        """Both derived fields, here rather than in a helper with eight positional arguments."""
        if self.rows is None and self.directions is not None:
            self.rows = self.directions.basis.numpy()
        if self.layout is None:
            self.layout = layout_for(self.knobs, self.path)
        if not self.dials_live:
            self.dials_live = live_dials(self.knobs, self.directions, self.layout)

    @functools.cached_property
    def name(self) -> str:
        """Cached, because it is read on the frame path. `label_for` walks the path twice, asks
        `is_onnx` about the suffix and formats a string -- 9 us, every frame, for an answer that
        is fixed the moment the model is built."""
        return label_for(self.path)


@dataclass
class Bank:
    """The models, the work that happens after any of them, and which one is playing."""

    models: list[Model]
    stage: FrameStage
    device: str
    width: int
    height: int
    index: int = 0
    #: How every model in this bank was prepared, including the ones added later.
    options: LoadOptions = field(default_factory=LoadOptions)
    #: The precision the bank plays in. `None` is the device's own; see `device.playback_dtype`.
    dtype: object = None
    height_want: int | None = 0
    screen: tuple | None = None
    #: Every walk config built by `walk()`, so `use` can re-point them at the new model's
    #: directions. A list: the latency harness builds its own walk beside the instrument's.
    _walk_cfgs: list = field(default_factory=list)
    #: The walks themselves, for the one thing that lives on the walk and not its config: the
    #: latent width. One config may serve two walks, and both have to be retargeted.
    _walks: list = field(default_factory=list)
    compile_s: float = 0.0
    graphs: int = 0

    def __post_init__(self) -> None:
        if self.dtype is None:
            from ganlive.device import playback_dtype

            self.dtype = playback_dtype(self.device)

    @property
    def current(self) -> Model:
        return self.models[self.index]

    @property
    def cfg(self):
        """The *playing* model's config, not the first one loaded."""
        return self.current.cfg

    @property
    def nz(self) -> int:
        return self.cfg.nz

    @property
    def name(self) -> str:
        return self.current.name

    def size_of(self, model: Model) -> tuple[int, int]:
        """What this model's frames are produced at, under this session's height setting."""
        return frame_size(model.cfg, self.height_want, self.screen)

    def use(self, index: int) -> Model:
        """Play a different model. Wraps, so stepping past the end comes back round."""
        self.index = index % len(self.models)
        # The directions belong to the weights, so they move with them: a switch used to leave the
        # walk pushing along the previous generator's basis, silently.
        self._rewire()
        return self.current

    def _rewire(self) -> None:
        """Point everything that belongs to the loaded model at the loaded model: the walk's width and
        directions, where it puts the latent, and the size frames come out at."""
        model = self.current
        # The generator says where it takes the latent: an ONNX graph and a captured one both
        # read it from the host, a compiled module from the card.
        on_host = bool(getattr(model.net, "latent_on_host", False))
        self.height, self.width = self.size_of(model)
        self.stage.resize(self.height, self.width)
        for cfg in self._walk_cfgs:
            cfg.directions = model.rows
            cfg.latent_on_host = on_host
            cfg.push_into = model.push
        for walk in self._walks:
            walk.retarget(model.cfg.nz)

    def index_of(self, path) -> int | None:
        """Where a checkpoint sits in this bank, or None."""
        return index_of(self.models, path)

    def add(self, target) -> Model:
        """Load one more model into this bank and return it. Does **not** switch to it."""
        path = checkpoint_for(Path(target))
        admit(self.models, path)
        stage = self.stage
        model, _ = _prepare(path, self.device, self.dtype,
                            (stage.to_yuv, stage.to_rgb, stage.to_bgra), self.options)
        self.models.append(model)
        self.graphs += model.graphs
        self.compile_s += model.compile_s
        self.graphs += _warm(stage, model, self.device, self.dtype, self.size_of(model))
        stage.resize(self.height, self.width)
        return model

    def report(self) -> str:
        """One line naming every path that could silently have fallen back to a slow one."""
        live = [k for k, on in self.stage.compiled.items() if on] or ["none"]
        models = ", ".join(f"{m.name} ({m.cfg.ladder.width}x{m.cfg.ladder.height}, "
                           f"{m.compile_s:.0f}s)" for m in self.models)
        pinned = {"knobs": self.current.knobs.pinned,
                  **(self.stage.pinned() or {"frames": "after the first one"})}
        return (f"{self.width}x{self.height} out of {self.current.cfg.ladder.width}x"
                f"{self.current.cfg.ladder.height}, {self.graphs} graph(s) in "
                f"{self.compile_s:.0f}s, "
                f"{len(self.current.knobs.names)} settings, "
                f"compiled conversions: {'+'.join(live)}, "
                f"pinned: {pinned}\n  models: {models}")

    def walk(self, config=None, dtype=None):
        """A walk wired to this generator, with this generator's own latent directions."""
        from ganlive.walk import SlerpWalk, WalkConfig

        config = config if config is not None else WalkConfig()
        # Attached here: a walk built without them has eight dials that move nothing. Remembered, so
        # `use` can re-point it when the model changes under the walk.
        if config not in self._walk_cfgs:
            self._walk_cfgs.append(config)
        # The latent in the precision the bank plays in, unless asked otherwise.
        walk = SlerpWalk(self.cfg.nz, self.device, config, dtype=dtype or self.dtype)
        self._walks.append(walk)
        self._rewire()
        return walk


@dataclass
class Shelved:
    """One run on disk, as the picker needs to show it."""

    name: str
    path: Path
    loaded: bool
    #: Why this row cannot be loaded at all. Unreadable files only, now that no model is
    #: refused for its shape.
    why: str = ""
    #: What it is -- `1024x1024 z512`. Shown on a row that can be loaded, so the size and
    #: the latent width are visible *before* the switch rather than inferred after it.
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

    def count(self) -> int:
        """How many models are on disk, without opening any of them."""
        return len(self._models())

    def entries(self) -> list[Shelved]:
        """Every run under `root` that has a checkpoint. Scanned once; see the class note."""
        if self._listing is None:
            self._listing = self._scan()
        return self._listing

    def _scan(self) -> list[Shelved]:
        """Every model on disk. Only a file that cannot be read is disqualified."""
        here = {m.path for m in self.bank.models}
        out = []
        for path in self._models():
            why, note = "", ""
            if path not in here:
                try:
                    cfg = self._config(path)
                except Exception as exc:          # noqa: BLE001  a broken file is not a crash
                    why = f"unreadable: {type(exc).__name__}"
                else:
                    note = f"{cfg.ladder.width}x{cfg.ladder.height} z{cfg.nz}"
            out.append(Shelved(name=label_for(path), path=path, loaded=path in here,
                               why=why, note=note))
        return sorted(out, key=lambda s: (not s.loaded, -s.path.stat().st_mtime))

    def _models(self) -> list[Path]:
        """Every model file under `root`: one per run, and one per exported graph.

        **Empty, not a crash, when there is no `root`.** Playing a checkpoint from anywhere
        else is the ordinary first run -- nothing creates `runs/` until something is imported
        into it -- and this raised out of `play`'s first screenful and out of the picker."""
        if not self.root.is_dir():
            return []
        found: list[Path] = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            # Every graph, plus the newest checkpoint. Both, not one or the other: a folder
            # holding an export beside its checkpoints used to show only the exports.
            found += sorted(folder.glob(f"*{ONNX}"))
            try:
                newest = checkpoint_for(folder)
            except (FileNotFoundError, OSError):
                continue                          # not a run, or nothing saved yet
            if not is_onnx(newest):
                found.append(newest)
        return found

    def request(self, entry: Shelved) -> None:
        """Ask for a model, from whichever thread the click arrived on."""
        if entry.loaded or entry.why:
            return
        self.pending = str(entry.path)

    def service(self):
        """Do the waiting load, on the thread that owns the card. Returns the new model."""
        want, self.pending = self.pending, None
        if want is None:
            return None
        try:
            model = self.bank.add(want)
        except Exception as exc:                  # noqa: BLE001  reported, never raised at the
            self.note = f"{label_for(want)}: {exc}"                 # frame loop
            return None
        self._listing = None
        self.note = f"loaded {model.name} in {model.compile_s:.1f}s"
        return model


def build(checkpoint, device: str | None = None, height: int | None = 0, dtype=None,
          screen=None, options: LoadOptions | None = None) -> Bank:
    """Load, prepare, install the dials' settings, compile -- in that order, for each model.

    `device` and `dtype` default to whichever accelerator this machine has and the precision
    it plays fastest in -- see `device.playback_dtype`. Given, they are taken as they are."""
    import time

    from ganlive.device import detect_backend, playback_dtype

    device = device or detect_backend()
    dtype = dtype or playback_dtype(device)
    options = options or LoadOptions()
    targets = [checkpoint] if isinstance(checkpoint, (str, Path)) else list(checkpoint)
    paths = [checkpoint_for(Path(t)) for t in targets]

    models: list[Model] = []
    conversions = None
    t_all = time.perf_counter()
    for path in paths:
        admit(models, path)
        model, conversions = _prepare(path, device, dtype, conversions, options)
        models.append(model)

    out_height, out_width = frame_size(models[0].cfg, height, screen)
    stage = FrameStage(out_height, out_width, conversions[0], to_rgb=conversions[1],
                       to_bgra=conversions[2], device=device)

    # Each at its own size, then the stage left on the model that will play -- `models[0]`,
    # which is why it is warmed last rather than first.
    warmed = sum(_warm(stage, m, device, dtype, frame_size(m.cfg, height, screen))
                 for m in reversed(models))
    stage.resize(out_height, out_width)

    return Bank(models=models, stage=stage, device=device, width=out_width, height=out_height,
               graphs=sum(m.graphs for m in models) + warmed,
               compile_s=time.perf_counter() - t_all, dtype=dtype,
               height_want=height, screen=screen, options=options)
