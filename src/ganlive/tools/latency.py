"""Does the audio-reactive loop fit in a frame? Measured, with the sound actually running.

Three loops, each timed for `--seconds`:
  generation only  the generator and the recorder's frame conversion, nothing else;
  recorded         the whole control loop with the stand-in kit playing, ending where a
                   recorded frame ends;
  played           (`--window`) the same, ending where a played frame ends: the real window
                   open and the strip drawn beside it.
The verdict is the played loop's when it ran, else the recorded loop's.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from ganlive import bank
from ganlive.control.features import FeatureExtractor
from ganlive.control.kit import INDEX
from ganlive.control.simulate import MachineSim, StemFeeder
from ganlive.dials.fastgan_dials import fastgan
from ganlive.engine.screen import Stepped
from ganlive.families import LoadOptions
from ganlive.files import write_json
from ganlive.presets import Impulse, Preset, PresetRunner
from ganlive.strip import DialPanel
from ganlive.timing import drift_ms, stat_ms
from ganlive.tools import add_backend, parser
from ganlive.window import Display, parse_height, screen_size


def _far(rest: float) -> float:
    """The end of a dial's travel furthest from where it rests.

    Up for a dial resting in the middle: `reaction` rests at 0.5, and at 0 it would scale
    every impulse to nothing."""
    return 1.0 if rest <= 0.5 else 0.0


def worst_case(layout=None) -> Preset:
    """Every dial off its rest and every drum wired. A frame budget, not a preset.

    Built from the loaded model's own layout when there is one, so every dial named is one
    the model has. Without one it uses this project's FastGAN surface, the most crowded here.
    Every dial gets an impulse, cycling through the kit, pushing back toward the middle so
    the push never clamps into a value that does not change."""
    rests = {k.name: k.rest for k in (layout or fastgan()).knobs}
    return Preset(
        name="worst-case",
        blurb="Every dial away from rest and every track wired. A frame budget, not a preset.",
        dials={name: _far(rest) for name, rest in rests.items()},
        impulses=[Impulse(track, dial, decay=0.2,
                          amount=-0.3 if _far(rests[dial]) > 0.5 else 0.3)
                  for dial, track in zip(rests, itertools.cycle(INDEX), strict=False)],
    )


def latent_for(model) -> np.ndarray:
    """A fixed latent, a host array as the walk hands it over."""
    return np.random.default_rng(0).standard_normal((1, model.cfg.nz)).astype(np.float32)


def _time_frames(frames: int, step) -> list[float]:
    """Milliseconds each of `frames` calls of `step()` took."""
    ms = []
    for _ in range(frames):
        t0 = time.perf_counter()
        step()
        ms.append((time.perf_counter() - t0) * 1000)
    return ms


def _stage_table(title: str, stats: dict[str, dict]) -> None:
    """Print median, p95 and max per stage, one row each."""
    print(f"   {title:22} {'median':>8} {'p95':>8} {'max':>8}")
    for name, s in stats.items():
        print(f"   {name:22} {s['median']:8.3f} {s['p95']:8.3f} {s['max']:8.3f}")


def played(r, runner, ex, walk, model, args, take, pcm, period_ms) -> tuple[dict, dict]:
    """The played loop: the control loop, finishing where a played frame finishes.

    The recorded loop ends at `nv12_bytes`, with no window on screen. A played frame ends
    with a `publish`, which presents the frame and wakes the window thread to paint the strip
    and take the events. That work holds the GIL and lands on this thread's next frame, so
    it has to be measured with the window really open."""
    panel = DialPanel(runner, actions={}, extractor=ex, bank=r)
    display = Display((r.height, r.width), r.gpu, title="ganlive - latency", overlay=panel,
                      fullscreen=False)
    stages: dict[str, list] = {"hit detection": [], "rules": [], "walk": [],
                              "issue": [], "window": [], "publish": []}
    total_ms: list[float] = []
    # A feeder of its own: the recorded loop's has finished, and the drums must be playing.
    feeder = StemFeeder(ex, pcm, take.samplerate, args.blocksize)
    feeder.start()
    time.sleep(0.25)
    try:
        for _ in range(8):                                  # the window's first texture
            model.net(walk.latent(0.0))
            display.publish(Stepped.native(model.net))
        r.sync()
        for f in range(int(args.seconds * args.fps)):
            t0 = time.perf_counter()
            runner.observe(ex.drain())
            t1 = time.perf_counter()
            runner.apply(ex.since, ex.features(), model.settings)
            t2 = time.perf_counter()
            z = walk.latent(f / args.fps * take.bpm / 60.0)
            t3 = time.perf_counter()
            model.net(z)
            t4 = time.perf_counter()
            frame = Stepped.native(model.net)
            t5 = time.perf_counter()
            display.publish(frame)
            t6 = time.perf_counter()
            panel.beats = f / args.fps * take.bpm / 60.0
            for name, dt in (("hit detection", t1 - t0), ("rules", t2 - t1),
                             ("walk", t3 - t2), ("issue", t4 - t3),
                             ("window", t5 - t4), ("publish", t6 - t5)):
                stages[name].append(dt * 1000)
            total_ms.append((t6 - t0) * 1000)
    finally:
        feeder.stop()
        display.close()
    out = stat_ms(total_ms, period_ms)
    print(f"played           window open, strip drawn     "
          f"{out['median']:6.2f} ms median  {out['p95']:6.2f} p95", flush=True)
    per = {k: stat_ms(v, period_ms) for k, v in stages.items()}
    print()
    _stage_table("part of the played frame", per)
    return out, per


def main(argv=None) -> int:
    ap = parser("latency", __doc__)
    ap.add_argument("--checkpoint", type=Path, action="append", metavar="PATH", required=True,
                    help="repeatable. Several models are loaded into one bank, as `play` "
                         "loads them, and every one of them is timed: a bank holds "
                         "them all resident at once, and each has its own frame time. The "
                         "full loops run on the first. Required.")
    add_backend(ap)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--height", type=parse_height, default=0,
                    metavar="auto|native|PIXELS",
                    help="what the window is sent, as `ganlive play` takes it. 'native' (the "
                         "default here) is the generator's own size; 'auto' is what `play` "
                         "uses -- the largest this screen can draw the frame at")
    ap.add_argument("--blocksize", type=int, default=256)
    ap.add_argument("--channels", type=int, default=12,
                    help="audio channels the stand-in kit is played on: 12 is one per track, "
                         "8 one per shared voice")
    ap.add_argument("--out", type=Path, default=Path("runs/ganlive/latency.json"))
    ap.add_argument("--window", action="store_true",
                    help="also time the played loop: the real window open and the real strip "
                         "drawn beside it, finishing at `publish` rather "
                         "than at the recorder's `nv12_bytes`. The window thread's work lands "
                         "on the frame thread, so only this loop shows its cost. Needs a "
                         "screen.")
    args = ap.parse_args(argv)

    take = MachineSim(bpm=130.0, seed=1).render(bars=8)
    pcm = take.stems[:args.channels]
    print(f"audio: {pcm.shape[0]} ch, {take.seconds:.1f}s, blocksize {args.blocksize} "
          f"({args.blocksize / take.samplerate * 1000:.2f} ms)", flush=True)

    r = bank.build(args.checkpoint, height=args.height,
                   screen=screen_size(),
                   options=LoadOptions(backend=args.backend))
    print(f"generator: {r.report()}", flush=True)

    total = int(args.seconds * args.fps)
    period_ms = 1000.0 / args.fps
    results: dict = {"fps": args.fps, "height": r.height, "width": r.width,
                     "channels": pcm.shape[0], "blocksize": args.blocksize,
                     "preset": "worst-case", "native": [r.cfg.ladder.width, r.cfg.ladder.height],
                     "bank": [m.name for m in r.models]}

    model = r.current
    z_fixed = latent_for(model)
    r.stage.nv12_bytes(r.stage.step(model.net(z_fixed)))
    r.sync()

    ex = FeatureExtractor(pcm.shape[0], take.samplerate)
    runner = PresetRunner(worst_case(model.layout), INDEX, float(args.fps),
                          channels=pcm.shape[0])
    runner.use_model(model)
    walk = r.walk(runner.walk_cfg)

    runner.apply(ex.since, ex.features(), model.settings)
    ms = _time_frames(total, lambda: r.stage.nv12_bytes(r.stage.step(model.net(z_fixed))))
    gen = results["generation_only"] = stat_ms(ms, period_ms)
    print(f"generation only  no sound, no control layer   "
          f"{gen['median']:6.2f} ms median  {gen['p95']:6.2f} p95", flush=True)
    drift = results["generation_drift"] = drift_ms(ms, args.seconds)
    print(f"   drift: {drift['first_ms']:.2f} ms in the first fifth of the run, "
          f"{drift['last_ms']:.2f} in the last ({drift['ms_per_s']:+.3f} ms/s)", flush=True)
    stages = {"hit detection": [], "rules": [], "walk": [], "generate and finish": [],
              "total": []}

    feeder = StemFeeder(ex, pcm, take.samplerate, args.blocksize)
    feeder.start()
    time.sleep(0.25)                                    # let it settle

    late = 0
    start = time.perf_counter()
    for f in range(total):
        t0 = time.perf_counter()
        runner.observe(ex.drain())
        t1 = time.perf_counter()
        runner.apply(ex.since, ex.features(), model.settings)
        t2 = time.perf_counter()
        z = walk.latent(f / args.fps * take.bpm / 60.0)
        t3 = time.perf_counter()
        r.stage.nv12_bytes(r.stage.step(model.net(z)))
        t4 = time.perf_counter()

        stages["hit detection"].append((t1 - t0) * 1000)
        stages["rules"].append((t2 - t1) * 1000)
        stages["walk"].append((t3 - t2) * 1000)
        stages["generate and finish"].append((t4 - t3) * 1000)
        stages["total"].append((t4 - t0) * 1000)
        if (t4 - start) > (f + 1) / args.fps:
            late += 1
    feeder.stop()

    total_ms = stages.pop("total")
    recorded = results["recorded"] = stat_ms(total_ms, period_ms)
    recorded["frames_late"] = late
    recorded["frames_late_pct"] = round(late / total * 100, 1)
    results["stages"] = {k: stat_ms(v, period_ms) for k, v in stages.items()}
    push = stat_ms(feeder.push_ms)
    results["audio_thread"] = {
        "callbacks": feeder.calls,
        "push_median_ms": push["median"],
        "push_p95_ms": push["p95"],
        "push_budget_ms": round(args.blocksize / take.samplerate * 1000, 2),
        "callbacks_late": feeder.late,
    }

    a, b = gen["median"], recorded["median"]
    results["control_layer_cost_ms"] = round(b - a, 2)
    results["control_layer_cost_pct"] = round((b - a) / a * 100, 1)

    print(f"recorded         sound thread running         "
          f"{b:6.2f} ms median  {recorded['p95']:6.2f} p95   "
          f"({late}/{total} frames late)", flush=True)
    print(f"\n   control layer costs {b - a:+.2f} ms ({(b - a) / a * 100:+.1f}%) "
          f"against generation only\n", flush=True)
    _stage_table("part of the loop", results["stages"])

    if args.window:
        results["played"], results["stages_played"] = played(r, runner, ex, walk, model,
                                                             args, take, pcm, period_ms)

    # Every model in the bank, at its own size, through `r.use`, so this prices the switch as
    # well as the model. After the full loops, so the first model is not read on a warmer card.
    if len(r.models) > 1:
        per: dict[str, dict] = {}
        was = r.index
        for i, m in enumerate(r.models):
            r.use(i)
            z = latent_for(m)

            def step(m=m, z=z):
                r.stage.nv12_bytes(r.stage.step(m.net(z)))

            for _ in range(8):                          # the first frames after a switch
                step()
            r.sync()
            per[m.name] = stat_ms(_time_frames(total, step), period_ms)
            per[m.name]["size"] = [r.width, r.height]
        r.use(was)
        results["per_model"] = per
        print(f"\n   {'model in the bank':34} {'size':>11} {'median':>8} {'p95':>8} "
              f"{'fps':>6}")
        for name, s in per.items():
            print(f"   {name:34} {s['size'][0]:5d}x{s['size'][1]:<5d} {s['median']:8.2f} "
                  f"{s['p95']:8.2f} {1000.0 / s['median']:6.1f}")


    at = results["audio_thread"]
    print(f"\n   sound callback: {at['push_median_ms']:.3f} ms median, "
          f"{at['push_p95_ms']:.3f} p95, against a {at['push_budget_ms']:.2f} ms budget "
          f"({at['push_p95_ms'] / at['push_budget_ms'] * 100:.1f}% used), "
          f"{at['callbacks_late']} late of {at['callbacks']} callbacks")

    # The verdict belongs to the loop that is played; the recorded loop is a lower bound on it.
    results["verdict_loop"] = "played" if "played" in results else "recorded"
    judged = results[results["verdict_loop"]]
    mid = judged["median"]
    verdict = ("FITS" if judged["p95"] < period_ms else
               "MEDIAN FITS, SLOWEST 1 IN 20 DOES NOT" if mid < period_ms else "DOES NOT FIT")
    results["verdict"] = verdict
    results["headroom_x"] = round(period_ms / mid, 2)
    print(f"\n   {args.fps} fps budget is {period_ms:.2f} ms -> {verdict} on the "
          f"{results['verdict_loop']} loop, {period_ms / mid:.2f}x headroom on the median")

    write_json(args.out, results)
    print(f"\nwrote {args.out}")
    return 0
