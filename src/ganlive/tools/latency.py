"""Does the audio-reactive loop fit in a frame? Measured, with the sound actually running."""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch

from ganlive import bank
from ganlive import device as dev
from ganlive.control.features import FeatureExtractor
from ganlive.control.tracks import INDEX, MachineSim, StemFeeder
from ganlive.dials.table import DIALS
from ganlive.presets import Impulse, Preset, PresetRunner
from ganlive.timing import drift_ms, stat_ms, write_metrics


def _far(rest: float) -> float:
    """The end of a dial's travel furthest from where it rests.

    **Up, for a dial that rests in the middle.** `reaction` rests at 0.50, and `rest < 0.5` put
    it at 0.0 -- where `PresetRunner.apply` multiplies every impulse by `reaction * 2.0` and gets
    nothing. The frame-budget worst case had its whole drum layer muted: every rule below was
    resolved, counted and then worth zero, so the budget was measured on a loop in which no drum
    moved a dial."""
    return 1.0 if rest <= 0.5 else 0.0


#: The frame-budget worst case, here rather than in the library because it is not something to
#: perform with: every dial off its rest and every drum wired. Built from `DIALS`, so a new dial
#: joins it by existing.
WORST = Preset(
    name="worst-case",
    blurb="Every dial away from rest and every track wired. A frame budget, not a setting.",
    dials={name: _far(rest) for name, (rest, _) in DIALS.items()},
    # Every dial, cycling the kit, rather than `zip(INDEX, DIALS)` -- which stopped at the
    # twelfth name and so wired no MODEL dial at all, leaving the settings vector untouched for
    # the whole run and the host-to-device copy out of the budget it is supposed to bound. And
    # pushed back toward the middle, because a dial parked at the end of its travel clamps:
    # a rule that clamps writes the same number it wrote last frame.
    impulses=[Impulse(track, dial, decay=0.2,
                      amount=-0.3 if _far(DIALS[dial][0]) > 0.5 else 0.3)
              for dial, track in zip(DIALS, itertools.cycle(INDEX), strict=False)],
)


def latent_for(model, r):
    """A fixed latent where this generator takes it: on the host for a graph that reads it
    there, as the walk hands it over; on the card otherwise. Handing a captured generator a
    device tensor would download it every frame -- a stall the played loop never pays."""
    z = torch.randn(1, model.cfg.nz).to(r.dtype)
    return z.numpy() if getattr(model.net, "latent_on_host", False) else z.to(r.device)


def played(r, runner, ex, walk, model, args, take, pcm, period_ms) -> tuple[dict, dict]:
    """Arm C: the same loop, finishing where a played frame finishes.

    Arms A and B end at `nv12_bytes`, which is the recorder's path and runs with no window on
    the screen. What a player's frame ends with is `bgra_bytes` and a `publish` that wakes the
    window thread, and that thread then uploads a texture, redraws the strip and presents --
    three C calls that hold the GIL and land on this thread's next frame. The gap is not small:
    a played FFHQ-1024 session read 17.9 ms a frame against this harness's 15.3.
    """
    from ganlive.strip import DialPanel
    from ganlive.window import Display

    panel = DialPanel(runner, actions={}, extractor=ex, rig=r)
    display = Display((r.height, r.width), title="ganlive - latency", overlay=panel,
                      fullscreen=False)
    to_window = r.stage.bgra_bytes if Display.PIXELS == "bgra" else r.stage.rgb_bytes
    stages: dict[str, list] = {"hit detection": [], "rules": [], "walk": [],
                              "issue": [], "window": [], "publish": []}
    total_ms: list[float] = []
    # Its own feeder: arm B drank the take, and a silent control layer is not the played loop.
    feeder = StemFeeder(ex, pcm, take.samplerate, args.blocksize)
    feeder.start()
    time.sleep(0.25)
    try:
        for _ in range(8):                                  # the window's first texture
            display.publish(to_window(r.stage.step(model.net(walk.latent(0.0)))))
        dev.synchronize()
        for f in range(int(args.seconds * args.fps)):
            t0 = time.perf_counter()
            runner.observe(ex.drain())
            t1 = time.perf_counter()
            runner.apply(ex.since, ex.features(), model.knobs)
            t2 = time.perf_counter()
            z = walk.latent(f / args.fps * take.bpm / 60.0)
            t3 = time.perf_counter()
            frame = r.stage.step(model.net(z))
            t4 = time.perf_counter()
            shown = to_window(frame)
            t5 = time.perf_counter()
            display.publish(shown)
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
    print(f"C  everything, window open, strip drawn        "
          f"{out['median']:6.2f} ms median  {out['p95']:6.2f} p95", flush=True)
    per = {k: stat_ms(v, period_ms) for k, v in stages.items()}
    print(f"\n   {'part of the played frame':22} {'median':>8} {'p95':>8} {'max':>8}")
    for name, s in per.items():
        print(f"   {name:22} {s['median']:8.3f} {s['p95']:8.3f} {s['max']:8.3f}")
    return out, per


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, action="append", metavar="PATH", required=True,
                    help="repeatable. Several models are loaded into one bank, exactly as "
                         "the instrument loads them, and **every one of them is timed** -- "
                         "which is the only way to answer what a bank costs, since holding "
                         "three networks compiled and resident at once is VRAM the frame "
                         "budget otherwise owns, and each model's own frame time is a "
                         "different number. Arm A and the full loop run on the first. "
                         "Required.")
    ap.add_argument("--device", default=None, help="default: whichever accelerator is there")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=60)
    # The same three answers the instrument takes, because arm C claims to be the played loop
    # and the size the frame crosses the bus at is the largest dial on that loop.
    ap.add_argument("--height", type=bank.parse_height, default=0,
                    metavar="auto|native|PIXELS",
                    help="what the window is sent. 'native' (the default here, for comparison "
                         "with every earlier reading in NOTES) is the generator's own size; "
                         "'auto' is what `ganlive play` uses -- the largest this screen can draw "
                         "the frame at, which at 3072x2048 is 12 MB a frame across the bus "
                         "instead of 25")
    ap.add_argument("--blocksize", type=int, default=256)
    ap.add_argument("--channels", type=int, default=12,
                    help="12 is the optimistic per-track case; 8 is the MKI voice case, which "
                         "its USB 2.0 Full Speed bandwidth may also force")
    ap.add_argument("--out", type=Path, default=Path("runs/ganlive/latency.json"))
    ap.add_argument("--window", action="store_true",
                    help="also run arm C: the loop as it is actually played, with the real "
                         "window open and the real strip drawn beside it, finishing at "
                         "`bgra_bytes` and `publish` rather than at the recorder's "
                         "`nv12_bytes`. **Arms A and B do not measure the played loop.** A "
                         "session that reported a quarter of its frames over budget had no "
                         "arm here that could reproduce it, because the window and the strip "
                         "are where that quarter lives: the window thread's texture upload "
                         "and present are C calls that hold the GIL, and they land on the "
                         "frame thread's next host work. Needs a screen.")
    ap.add_argument("--no-capture", dest="capture", action="store_false",
                    help="load without recording each compiled forward as one device graph. "
                         "The control arm for that change: it is the largest single number "
                         "this harness has reported, and quoting it needs a reading from the "
                         "same command on the same day rather than one out of an older file")
    args = ap.parse_args(argv)

    take = MachineSim(bpm=130.0, seed=1).render(bars=8)
    pcm = take.stems[:args.channels]
    print(f"audio: {pcm.shape[0]} ch, {take.seconds:.1f}s, blocksize {args.blocksize} "
          f"({args.blocksize / take.samplerate * 1000:.2f} ms)", flush=True)

    # The screen only matters to `--height auto`, and passing it always means the two tools
    # resolve a height the same way rather than nearly the same way.
    r = bank.build(args.checkpoint, args.device, height=args.height,
                  screen=bank.screen_size(), capture=args.capture)
    print(f"generator: {r.report()}", flush=True)

    total = int(args.seconds * args.fps)
    period_ms = 1000.0 / args.fps
    results: dict = {"fps": args.fps, "height": r.height, "width": r.width,
                     "channels": pcm.shape[0], "blocksize": args.blocksize,
                     "preset": WORST.name, "native": [r.cfg.ladder.width, r.cfg.ladder.height],
                     "bank": [m.name for m in r.models], "capture": args.capture}

    model = r.current
    z_fixed = latent_for(model, r)
    r.stage.nv12_bytes(r.stage.step(model.net(z_fixed)))
    dev.synchronize()

    ex = FeatureExtractor(pcm.shape[0], take.samplerate)
    runner = PresetRunner(WORST, INDEX, float(args.fps), channels=pcm.shape[0])
    # The call the live loop makes at startup, and this harness did not. `PresetRunner` is built
    # holding the FastGAN layout, so without it the control layer spends every frame writing
    # dial names a converted StyleGAN2 does not have -- a worst-case budget measured against
    # writes that land nowhere, and a strip drawing one model's dials from another's values.
    runner.use_model(model)
    walk = r.walk(runner.walk_cfg)

    runner.apply(ex.since, ex.features(), model.knobs)
    ms = []
    for _ in range(total):
        t0 = time.perf_counter()
        r.stage.nv12_bytes(r.stage.step(model.net(z_fixed)))
        ms.append((time.perf_counter() - t0) * 1000)
    results["A_generation_only"] = stat_ms(ms, period_ms)
    print(f"A  generation only, no sound, no control layer   "
          f"{results['A_generation_only']['median']:6.2f} ms median  "
          f"{results['A_generation_only']['p95']:6.2f} p95", flush=True)
    drift = results["A_drift"] = drift_ms(ms, args.seconds)
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
        runner.apply(ex.since, ex.features(), model.knobs)
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
    results["B_full_loop"] = stat_ms(total_ms, period_ms)
    results["B_full_loop"]["frames_late"] = late
    results["B_full_loop"]["frames_late_pct"] = round(late / total * 100, 1)
    results["stages"] = {k: stat_ms(v, period_ms) for k, v in stages.items()}
    push = stat_ms(feeder.push_ms)
    results["audio_thread"] = {
        "callbacks": feeder.calls,
        "push_median_ms": push["median"],
        "push_p95_ms": push["p95"],
        "push_budget_ms": round(args.blocksize / take.samplerate * 1000, 2),
        "callbacks_late": feeder.late,
    }

    a = results["A_generation_only"]["median"]
    b = results["B_full_loop"]["median"]
    results["control_layer_cost_ms"] = round(b - a, 2)
    results["control_layer_cost_pct"] = round((b - a) / a * 100, 1)

    print(f"B  everything, sound thread running             "
          f"{b:6.2f} ms median  {results['B_full_loop']['p95']:6.2f} p95   "
          f"({late}/{total} frames late)", flush=True)
    print(f"\n   control layer costs {b - a:+.2f} ms ({(b - a) / a * 100:+.1f}%) "
          f"against arm A\n", flush=True)
    print(f"   {'part of the loop':22} {'median':>8} {'p95':>8} {'max':>8}")
    for name in ("hit detection", "rules", "walk", "generate and finish"):
        s = results["stages"][name]
        print(f"   {name:22} {s['median']:8.3f} {s['p95']:8.3f} {s['max']:8.3f}")

    if args.window:
        results["C_played"], results["stages_played"] = played(r, runner, ex, walk, model,
                                                               args, take, pcm, period_ms)

    # Every model in the bank, at its own size. Timed after the full loop so the first model's
    # numbers above are not read on a warmer card. Each goes through the path a frame takes, and
    # switching is `r.use`, so this prices the switch as well as the model.
    if len(r.models) > 1:
        per: dict[str, dict] = {}
        was = r.index
        for i, m in enumerate(r.models):
            r.use(i)
            z = latent_for(m, r)
            for _ in range(8):                          # the first frame after a switch
                r.stage.nv12_bytes(r.stage.step(m.net(z)))
            dev.synchronize()
            each = []
            for _ in range(total):
                t0 = time.perf_counter()
                r.stage.nv12_bytes(r.stage.step(m.net(z)))
                each.append((time.perf_counter() - t0) * 1000)
            per[m.name] = stat_ms(each, period_ms)
            per[m.name]["size"] = [r.width, r.height]
        r.use(was)
        results["per_model"] = per
        print(f"\n   {'model in the bank':34} {'size':>11} {'median':>8} {'p95':>8} "
              f"{'fps':>6}")
        for name, s in per.items():
            print(f"   {name:34} {s['size'][0]:5d}x{s['size'][1]:<5d} {s['median']:8.2f} "
                  f"{s['p95']:8.2f} {1000.0 / s['median']:6.1f}")

    vram = dev.memory_report(r.device)
    if vram:
        results["vram"] = vram
        print(f"\n   memory: {vram.get('max_allocated_gb', float('nan')):.2f} GB peak in use, "
              f"{vram.get('max_reserved_gb', float('nan')):.2f} GB peak held")

    at = results["audio_thread"]
    print(f"\n   sound callback: {at['push_median_ms']:.3f} ms median, "
          f"{at['push_p95_ms']:.3f} p95, against a {at['push_budget_ms']:.2f} ms budget "
          f"({at['push_p95_ms'] / at['push_budget_ms'] * 100:.1f}% used), "
          f"{at['callbacks_late']} late of {at['callbacks']} callbacks")

    # **The verdict belongs to the loop that is played.** Arm B ends at the recorder's
    # `nv12_bytes` with nothing on the screen; where arm C ran, it is the frame a hand sees and
    # B is a lower bound on it.
    judged = results.get("C_played") or results["B_full_loop"]
    results["verdict_arm"] = "C_played" if "C_played" in results else "B_full_loop"
    mid = judged["median"]
    verdict = ("FITS" if judged["p95"] < period_ms else
               "MEDIAN FITS, SLOWEST 1 IN 20 DOES NOT" if mid < period_ms else "DOES NOT FIT")
    results["verdict"] = verdict
    results["headroom_x"] = round(period_ms / mid, 2)
    print(f"\n   {args.fps} fps budget is {period_ms:.2f} ms -> {verdict} on arm "
          f"{results['verdict_arm'][0]}, {period_ms / mid:.2f}x headroom on the median")

    write_metrics(args.out, results)
    print(f"\nwrote {args.out}")
    return 0

