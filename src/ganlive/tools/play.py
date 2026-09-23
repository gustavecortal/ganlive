"""Play a GAN's latent space live, from a MIDI controller."""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import torch

from ganlive import bank
from ganlive import device as dev
from ganlive.control.audio import NoAudioDevice, pick_input
from ganlive.control.features import (
    BothFeatures,
    FeatureExtractor,
    NoteFeatures,
)
from ganlive.control.kit import (
    TRACKS,
    by_channel,
    channel_map,
    output_mode,
    parse_channel_map,
    parse_notes,
    parse_track_channels,
)
from ganlive.control.machine import profile
from ganlive.control.midi import (
    ClockReader,
    EncoderMap,
    PressureMap,
    parse_controls,
    parse_pressure,
)
from ganlive.control.simulate import MachineSim, MonitorFeeder, StemFeeder
from ganlive.dials.table import per_model
from ganlive.files import next_path, remember
from ganlive.presets import POSITIONS_NAME, Library, Positions, PresetRunner
from ganlive.record import video
from ganlive.record.sync import Guide
from ganlive.strip import PRIORITY as HAND_PRIORITY
from ganlive.strip import SOURCE as HAND
from ganlive.timing import stat_ms
from ganlive.walk import MusicalClock

OUT = Path("runs/ganlive")
SETTINGS, TAKES, STILLS = OUT / "settings", OUT / "takes", OUT / "stills"

CHANNELS = SETTINGS / "channels.txt"
#: The knob map `l` writes, in `--cc`'s own words; `--cc` on the command line replaces it.
CC_MAP = SETTINGS / "cc.txt"
POSITIONS = SETTINGS / POSITIONS_NAME


def remembered(path: Path, parse, label: str):
    """`(parsed, note)` from a settings file, or None if there is none or it does not parse."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return parse(text), f"{label}: {text}  -- from {path}"
    except Exception as exc:                                                # noqa: BLE001
        print(f"{path} does not parse and was ignored: {exc}", flush=True)
        return None


def report_dropped(runner) -> None:
    """Say which of the preset's rules this kit and model cannot run, if any."""
    if runner.dropped:
        print(f"  dropped {len(runner.dropped)} rule(s): {'; '.join(runner.dropped)}",
              flush=True)


def report_unreachable(wired, layout) -> None:
    """Say which knobs and pads name a dial this model does not have.

    The other half of `report_dropped`, and said in the same place: a rule wired to a missing
    dial and a knob parked on one are the same fact about the loaded model, and a knob that
    moves nothing reads exactly like a knob that is not arriving."""
    for one in wired:
        line = "" if one is None else one.unreachable(layout)
        if line:
            print(f"  {line}", flush=True)


def resolve_controls(explicit: str) -> tuple[dict, str]:
    """The knob map, and one line saying where it came from."""
    if explicit:
        return parse_controls(explicit), f"encoders: --cc {explicit}"
    return remembered(CC_MAP, parse_controls, "encoders") or (
        {}, f"encoders: none wired; press l on a dial and turn a knob (saved to {CC_MAP})")


def resolve_map(explicit: str, layout: str) -> tuple[dict[str, int], str]:
    """The audio channel map, and one line saying where it came from."""
    if explicit:
        tracks = parse_channel_map(explicit)           # parse first: never save an unusable map
        trouble = remember(CHANNELS, explicit)
        if trouble:
            return tracks, f"map: {explicit} (not remembered: {trouble})"
        return tracks, f"map: {explicit}  -- remembered in {CHANNELS}"
    found = remembered(CHANNELS, parse_channel_map, "map")
    if found is not None:
        return found
    return channel_map(layout), (
        f"map: --layout {layout}, a GUESS. Overbridge does not start the kit at channel 0, so "
        f"every hit may be credited to the wrong drum and a shared channel lights both of its "
        f"tracks. Run `ganlive learn`, then pass --map once and it is remembered.")

SAMPLE_CAP = 110_000

TAKE_DEPTH = 2


def per_model_lines(by_model, budget_ms: float) -> list[str]:
    """One row per model played, or nothing at all when only one was.

    **An aggregate over a bank is unattributable.** A 338s session that started on one FastGAN
    and loaded a StyleGAN2 from the shelf reported 53.8 fps with 25.7% of frames over budget,
    and no line in the report could say whether that was one slow model or the whole path
    slowing down. Named in the order they were first played, because that is the order the
    session happened in."""
    if len(by_model) < 2:
        return []
    wide = max(map(len, (*by_model, "per model")))
    out = [f"  {'per model':<{wide}} {'frames':>7} {'median':>8} {'p95':>8} {'over':>7}"]
    for name, took_ms in by_model.items():
        each = stat_ms(took_ms, budget_ms)
        out.append(f"  {name:<{wide}} {each['n']:7d} {each['median']:8.2f} "
                   f"{each['p95']:8.2f} {each['over_budget'] / each['n'] * 100:6.1f}%")
    return out


def open_audio(args, extractor, device, info, channels):
    """Build the stream. **It is NOT started** -- `main` starts it last of all, and must."""
    import sounddevice as sd

    dropped = [0]

    def callback(indata, _frames, _t, status):
        if status:
            dropped[0] += 1
        extractor.push(indata)

    stream = sd.InputStream(device=device, channels=channels, samplerate=extractor.sr,
                            blocksize=args.blocksize, dtype="float32", callback=callback)
    print(f"audio: {info['name']}, {channels} ch at {extractor.sr} Hz, "
          f"blocksize {args.blocksize} "
          f"({args.blocksize / extractor.sr * 1000:.2f} ms); starts with the picture",
          flush=True)
    return stream, dropped


def _parser() -> argparse.ArgumentParser:
    """Every option, in the groups `--help` prints them in."""
    ap = argparse.ArgumentParser(prog="ganlive play", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    what = ap.add_argument_group("what to play")
    what.add_argument("--checkpoint", type=Path, action="append", metavar="PATH", required=True,
                      help="a checkpoint, a run directory for its newest, or an exported .onnx. "
                           "Repeatable: `[` and `]` switch between them on the beat, and they "
                           "may differ in latent width, native size and aspect. An ONNX graph "
                           "plays with the motion and latent dials only")
    what.add_argument("--runs", type=Path, default=Path("runs"),
                      help="where `m` looks for more models to load")
    what.add_argument("--no-shelf", action="store_true",
                      help="no model picker, and no start-up scan of --runs")
    what.add_argument("--preset", default="default",
                      help=f"which setting to start on. Saved with `s` into {SETTINGS}")

    how = ap.add_argument_group("how it runs")
    how.add_argument("--device", default=None, metavar="xpu|cuda|mps|cpu",
                     help="default: whichever accelerator is there, else the CPU")
    how.add_argument("--height", type=bank.parse_height, default=None,
                     metavar="auto|native|PIXELS",
                     help="what the window is sent -- the generator always runs at its native "
                          "size. The largest thing on the loop: `auto` shrinks on the card to "
                          "the most this screen can draw, `native` sends every pixel. A "
                          "recording follows this")
    how.add_argument("--fps", type=int, default=60)
    how.add_argument("--seconds", type=float, default=0.0, help="0 runs until stopped")
    how.add_argument("--cpu", default="fast", choices=("fast", "all"),
                     help="`fast` pins to the performance cores of a hybrid CPU, which is worth "
                          "roughly a third of the frame rate because the loop is submission "
                          "bound. `all` leaves the scheduler alone")
    how.add_argument("--no-compile-net", dest="compile_net", action="store_false",
                     help="skip the ~45 s per-model compile. Slower frames; do not quote timings")
    how.add_argument("--no-capture", dest="capture", action="store_false",
                     help="do not record the forward as one device graph. The capture is exact "
                          "and is checked against the forward at every load")
    how.add_argument("--exact", action="store_true",
                     help="run a converted StyleGAN2 in the precision its own file describes, "
                          "rather than half wherever NVIDIA's rule allows. ~13%% slower, for "
                          "when a frame is being compared against the original")
    how.add_argument("--headless", action="store_true",
                     help="no window: run the loop and report the timing")
    how.add_argument("--console", action="store_true",
                     help="a slider per dial beside the picture, turned with the mouse")

    dials = ap.add_argument_group("which dials appear")
    dials.add_argument("--direction-floor", type=float, default=bank.RANDOM_FLOOR, metavar="X",
                       help="how many times a random direction of the same length a derived "
                            "direction must move the picture to earn a dial. Default "
                            f"{bank.RANDOM_FLOOR:g}. A taste control: lower is not simply more, "
                            "because a collapse to a flat field also measures as a large change")
    dials.add_argument("--stock-grain", dest="measure_grain", action="store_false",
                       help="do not measure each model's noise gains at load (~0.5 s a model). "
                            "The same gain buys very different grain on different models, so "
                            "with this off the noise dial means something different on each")

    play = ap.add_argument_group("what plays it")
    play.add_argument("--triggers", choices=("auto", "both", "midi", "audio"), default="auto",
                      help="where the hits come from. `auto` is both when an audio input opens "
                           "and midi alone when none does")
    play.add_argument("--midi-port", default="",
                      help="substring of the MIDI input to listen to; empty means all of them")
    play.add_argument("--notes", default="0", metavar="FIRST|MAP",
                      help="which note is which track, when the kit shares one channel. A first "
                           "note for pads that send consecutive notes -- 0 is a Rytm's twelve -- "
                           'or a map for a kit whose do not: "36=BD,38=SD,42=CH,46=OH". Tracks '
                           "are named or indexed 0 to 11, here and in --pressure")
    play.add_argument("--midi-channels", default="",
                      help='identify a track by MIDI channel instead of note, as "1-12" or '
                           '"1=BD,2=SD". Which is right is a setting on the machine')
    play.add_argument("--cc", default="",
                      help='wire knobs to dials, as "16=noise,2:17=se_256,n1.3=dir1" -- a CC '
                           "number or an NRPN as n<msb>.<lsb>, optionally a channel before a "
                           "colon. Or press `l` on a dial and turn the knob: the pair is "
                           f"learned and written to {CC_MAP}. NRPN is 14-bit against a CC's 7. "
                           "Any dial the loaded model has is addressable by name")
    play.add_argument("--pressure", default="",
                      help='lean on a pad to hold a dial, as "BD=noise,SD=dir1", through '
                           "polyphonic aftertouch. A squeeze that ends gives the dial back")
    play.add_argument("--no-midi", dest="midi", action="store_false")
    play.add_argument("--simulate", action="store_true",
                      help="play a stand-in drum machine instead of hardware")
    play.add_argument("--bpm", type=float, default=130.0,
                      help="fallback tempo, used only while no MIDI clock is arriving")

    sound = ap.add_argument_group("audio in")
    sound.add_argument("--audio-name", default=None, metavar="TEXT",
                       help="find the input by a word in its name, on any host API. The default "
                            "finds a Rytm, which Overbridge exposes on ASIO only")
    sound.add_argument("--audio-device", type=int, default=None)
    sound.add_argument("--channels", type=int, default=0)
    sound.add_argument("--samplerate", type=int, default=48000)
    sound.add_argument("--blocksize", type=int, default=256)
    sound.add_argument("--layout", default="tracks", choices=("tracks", "voices"),
                       help="one channel per track, or one per shared analog voice")
    sound.add_argument("--map", default="",
                       help='an explicit channel layout, e.g. "BD=0,SD=1,CH=6,OH=6"')
    sound.add_argument("--no-audio", dest="audio", action="store_false",
                       help="no sound source: the picture walks and the sliders drive it")
    sound.add_argument("--monitor", action="store_true",
                       help="with --simulate, send the stand-in to the speakers too, so the "
                            "picture can be judged against an audible beat")

    out = ap.add_argument_group("getting a take out")
    out.add_argument("--record", action="store_true", help="start recording immediately")
    out.add_argument("--codec", default=video.DEFAULT_CODEC,
                     help="what `v` records with. The default uses the card's media engine; a "
                          "software encoder at this size costs most of the frame rate")
    out.add_argument("--no-guide", dest="guide", action="store_false",
                     help="no guide track or sidecar beside a take. They exist to line the "
                          "video up against a DAW multitrack afterwards")
    return ap


def silence_words(use_notes: bool, use_audio: bool, machine=None) -> tuple[str, str]:
    """What to call a silence, and what to check, given where the hits were meant to come from.

    `machine` supplies the words for "what to switch on", which is the one part of this that
    really is per-controller -- see `control.machine`."""
    from ganlive.control.machine import GENERIC

    m = machine or GENERIC
    if use_notes and use_audio:
        return ("both", f"No notes means the pads are not transmitting -- on {m.name}, "
                        f"{m.says('notes')}. No sound means the channel map or the send "
                        f"levels; `ganlive doctor --meter` reports those.")
    if use_notes:
        return ("notes", f"No note-on named a track: check --notes and --midi-channels "
                         f"against the unresolved-notes line below, and on {m.name}, "
                         f"{m.says('notes')}. `--triggers both` adds the sound back as a "
                         f"second source.")
    return ("audio", "Check the channel map and the send levels; "
                     "`ganlive doctor --meter` reports both.")

def report_unheard(extractor, kind: str, fix: str, notes, note_channels, grace_s: float) -> None:
    """Once, a few seconds in: is anything actually arriving, and if not, what is wrong."""
    if not extractor.played():
        print(f"  NO HITS YET after {grace_s:g}s over {kind}. The picture "
              f"is running on the clock and ignoring the drums. {fix}",
              flush=True)
    unresolved = getattr(extractor, "unresolved", 0)
    if unresolved:
        seen = getattr(extractor, "unclaimed", {})
        by_channel = {}
        for ch, note in seen:
            by_channel.setdefault(ch, set()).add(note)
        mode = output_mode(by_channel, notes)
        if mode:
            hint = (f" -- that traffic is {mode.upper()} CH; "
                    + ("drop --midi-channels" if mode == "auto"
                       else "pass --midi-channels 1-12"))
        else:
            hint = f" on channels {sorted(c + 1 for c in by_channel)}"
        # Or not a Rytm at all: a kit whose notes are not the ones read.
        unknown = sorted(n for ns in by_channel.values() for n in ns
                         if n not in notes)
        if unknown and not note_channels:
            hint += (f"; notes {unknown} name no track -- --notes {unknown[0]} "
                     f"reads a kit that is consecutive from there, or a map "
                     f"like --notes 36=BD,38=SD names each")
        print(f"  {unresolved} NOTE-ONS NAMED NO TRACK{hint}", flush=True)

def report_frames(frames, wall, ms, total, late, by_model, dropped, source, args, period) -> None:
    """Frame time against the budget, and the drift a median hides."""
    stats = stat_ms(ms, period * 1000)
    print(f"\n{frames} frames in {wall:.1f}s = {frames / wall:.1f} fps"
          + (f"   (timings over the last {len(ms)} frames)" if len(ms) < frames else ""))
    # More than one, because the last frame of a healthy run and the deadline land together and
    # a single hiccup anywhere in the run decides which of the two goes first.
    if total and total - frames > 1:
        print(f"  short       {total - frames} of {total} frames never happened: the run ended "
              f"on its {args.seconds:g}s, and {frames / wall:.1f} fps is what this model held")
    print(f"  frame       {stats['median']:6.2f} ms median   {stats['p95']:6.2f} p95   "
          f"{stats['max']:6.2f} worst")
    print(f"  budget      {period * 1000:6.2f} ms at {args.fps} fps -> "
          f"{period * 1000 / stats['median']:.2f}x headroom on the median")
    print(f"  over budget {stats['over_budget']} of {frames} "
          f"({stats['over_budget'] / frames * 100:.1f}%) took longer than one frame")
    print(f"  behind      {late} of {frames} ({late / frames * 100:.1f}%) arrived after "
          f"their slot on an absolute schedule -- drift, not slow frames, and once the "
          f"loop is behind it stays there")
    for line in per_model_lines(by_model, period * 1000):
        print(line)
    if dropped[0]:
        print(f"  SOUND DROPOUTS: {dropped[0]}. Raise --blocksize.")
    if getattr(source, "late", 0):
        print(f"  monitor    {source.late} underrun(s) of {source.calls} blocks; "
              f"push {stat_ms(source.push_ms)['median']:.2f} ms median against "
              f"{args.blocksize / args.samplerate * 1000:.2f}")

def report_clock(clock, reader, args, pressure, encoders, runner, machine) -> None:
    """Where the beat came from, what was wired to what, and what each dial was worth."""
    src = "MIDI" if clock.source == "midi" else f"internal, {args.bpm:g} BPM"
    print(f"  beat        {src}, {clock.beats:.2f} beats, {clock.bpm:.1f} BPM")
    for wired in (pressure, encoders):
        for line in ([] if wired is None else wired.report()):
            print(f"  {line}", flush=True)
    for line in runner.usage_report():
        print(f"  {line}", flush=True)
    if reader is not None:
        trouble = reader.trouble()
        if trouble:
            print(f"  {trouble}")
        counts = reader.counts or {}
        if not counts.get("clock"):
            print(f"  NO MIDI CLOCK ARRIVED. The picture ran at its own tempo. On "
                  f"{machine.name}, {machine.says('clock')}.")
        elif not counts.get("start"):
            print(f"  clock but no transport: the tempo was right and the bar position was "
                  f"whatever it happened to be. On {machine.name}, "
                  f"{machine.says('transport')}.")
        else:
            print(f"  midi        {counts}")

def report_drums(extractor, tracks, kind: str, fix: str, hears, heard0, wall) -> None:
    """Which drums reached the picture, which were silent, and which arrived unclaimed."""
    names = by_channel(extractor.channel_of() or tracks)
    played = extractor.played()
    if not played:
        print(f"  NO HITS ARRIVED over {kind}. The tempo was right and no drum moved the "
              f"picture. {fix}")
    else:
        # Against the kit, not the width of the stream: twelve tracks share eight channels
        # here and the stream carries ten, so "8 of 10" read as two silent drums every run.
        f = extractor.features()
        print(f"  {kind:<11} {played} of {len(names)} kit channel(s) carried drums; "
              f"density {f['density']:.2f}/s, energy {f['energy']:.2f}")
    # Only channels a track maps to. The stream is wider than the kit -- the map here starts
    # at 2, leaving the machine's main outs unclaimed -- and listing those as silent named a
    # fault that cannot exist, because a rule names a track and no track points at them.
    counts = [(int(n), "/".join(sorted(names[i])))
              for i, n in enumerate(extractor.hits) if i in names]
    heard_from = " ".join(f"{label} {n}" for n, label in counts if n)
    quiet = " ".join(label for n, label in counts if not n)
    print(f"  reached      {heard_from or 'nothing'}")
    if quiet:
        print(f"  silent       {quiet}  -- any rule wired to these can never fire")
    # The other half of the same question, and the one that says the map is wrong: a channel
    # carrying drums that no track claims.
    stray = [f"ch{i}" for i, n in enumerate(extractor.hits) if i not in names and n]
    if stray:
        print(f"  unclaimed    {' '.join(stray)}  -- drums arrived on these and no track is "
              f"mapped to them, so the map is missing a row and those hits reach nothing")
    loudest = getattr(extractor, "loudest", None)
    if quiet and loudest is not None:
        levels, floor = loudest(), extractor.floor
        for label in quiet.split():
            channel = tracks.get(label.split("/")[0])
            if channel is None or channel >= len(levels):
                continue
            peak = levels[channel]
            why = ("NOTHING reached this channel -- send level, --map, or the machine"
                   if peak < floor else
                   "loud enough, so the DETECTOR rejected it -- floor or onset_ratio")
            print(f"    {label:<9} ch{channel} peak {peak:.4f} vs floor {floor:.4f}"
                  f"  -- {why}")
    if hears is not None:
        heard = hears() - heard0
        print(f"  delivered   {heard:.1f}s of audio over a {wall:.1f}s run "
              f"({heard / wall * 100:.0f}%)"
              + ("" if heard > 0.9 * wall else
                 "  <- THE CARD STOPPED DELIVERING. The picture went on running in "
                 "perfect tempo while nothing reached it."))


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    if args.console and args.headless:
        print("--console needs a window; drop --headless")
        return 1
    if args.monitor and not args.simulate:
        print("--monitor plays the stand-in, so it needs --simulate. With a real Rytm the "
              "machine is already making the sound.")
        return 1
    tracks, map_note = resolve_map(args.map, args.layout)
    # What to tell the user to go and switch on, in their own machine's words. Matched on
    # what they already typed rather than on a flag of its own; see `control.machine`.
    machine = profile(args.midi_port, args.audio_name or "")
    # Before anything opens the card: the affinity is for this process's own submission thread, and the
    # compile that follows is the first thing to use it.
    if args.cpu == "fast":
        dev.prioritise_gpu_feeder()

    note_channels = parse_track_channels(args.midi_channels) if args.midi_channels else None
    notes = parse_notes(args.notes)
    library = Library(SETTINGS)
    if library.broken:
        print("settings that would not load: " + "; ".join(library.broken), flush=True)
    use_notes = args.triggers in ("auto", "both", "midi") and not args.simulate and args.midi
    use_audio = args.triggers in ("auto", "both", "audio") and args.audio
    picked = None
    if use_audio and not args.simulate:
        # Asked for now rather than where the stream is built, so `auto` can fall back to the
        # notes alone when nothing opens -- a machine with no ASIO and no Rytm used to end
        # here with a traceback -- while an explicit `audio` or `both` still refuses.
        try:
            import sounddevice as sd

            picked = (pick_input(sd, args.audio_device, channels=args.channels)
                      if args.audio_name is None else
                      pick_input(sd, args.audio_device, args.audio_name,
                                 channels=args.channels, hostapi=None))
        except (NoAudioDevice, ImportError, OSError) as exc:
            if args.triggers != "auto":
                raise
            print(f"audio: none -- {exc}\n  Triggers come from MIDI notes alone; --triggers "
                  f"audio to insist on the sound.", flush=True)
            use_audio = False
    reactive = use_audio or use_notes
    wanted = args.preset
    if wanted not in library.names:
        print(f"unknown setting {wanted!r}; have {', '.join(library.names)}")
        return 1
    preset = library.select(wanted)

    dropped = [0]
    card = None
    if use_notes and not use_audio:
        extractor = NoteFeatures(len(TRACKS), channels=note_channels, notes=notes)
        source = None
        if note_channels:
            print(f"triggers: MIDI notes only, {len(TRACKS)} tracks, the CHANNEL names the "
                  f"track. Sequencer trigs arrive per track, exactly, with no channel map and "
                  f"no send levels -- and the four shared analog voices come apart.",
                  flush=True)
        else:
            print(f"triggers: MIDI notes only, {len(TRACKS)} tracks, one per pad. No channel "
                  f"map and no send levels. Sequencer trigs reach this path only where TRK "
                  f"SEND MIDI is on, which sends on the track's own channel -- add "
                  f"--midi-channels 1-12 to read them. --triggers both adds them back through "
                  f"the sound instead.", flush=True)
    elif not use_audio:
        extractor = FeatureExtractor(max(tracks.values(), default=11) + 1, args.samplerate)
        source = None
        print("audio: none. The walk runs and the sliders drive it.", flush=True)
    elif args.simulate:
        take = MachineSim(bpm=args.bpm, seed=1).render(bars=16, tail=0.5)
        stems = take.stems_for(tracks)
        channels = stems.shape[0]
        extractor = FeatureExtractor(channels, take.samplerate)
        source, heard = None, ""
        if args.monitor:
            source = MonitorFeeder(extractor, stems, take.mix, take.samplerate, args.blocksize)
            try:
                source.start()
                heard = ", through the speakers"
            except Exception as exc:                            # noqa: BLE001
                print(f"monitor: no output device ({type(exc).__name__}: {exc}); "
                      "running silent", flush=True)
                source = None
        if source is None:
            source = StemFeeder(extractor, stems, take.samplerate, args.blocksize)
            source.start()
        print(f"audio: the stand-in Rytm, {channels} ch "
              f"({'shared voices' if channels < 12 else 'one per track'}) "
              f"at {take.samplerate} Hz{heard}", flush=True)
    else:
        device, info, channels = picked
        heard_by = FeatureExtractor(channels, args.samplerate)
        stream, dropped = open_audio(args, heard_by, device, info, channels)
        source = card = stream
        extractor = (BothFeatures(heard_by, tracks, channels=note_channels, notes=notes)
                     if use_notes else heard_by)
        if use_notes:
            by = ("their MIDI channel" if note_channels else
                  f"their MIDI note, {len(TRACKS)} tracks in one note space")
            print(f"triggers: BOTH. Notes name a track by {by}; audio fills in anything that "
                  f"sends none, and an onset on a voice a note has just claimed is dropped "
                  f"rather than counted twice.", flush=True)

    if use_audio:
        print(map_note, flush=True)

    hears = extractor.heard if source is not None and use_audio else None
    ticks = getattr(extractor, "tick", None) if use_notes else None
    guide = Guide()
    if source is not None and args.guide:
        guide.listen_to(extractor)

    clock = MusicalClock(args.bpm)
    reader = encoders = pressure = None
    if args.pressure:
        pressure = PressureMap(parse_pressure(args.pressure, notes), machine=machine)
        print(f"pressure: {len(pressure.controls)} pad(s) wired -> "
              f"{', '.join(sorted(set(pressure.controls.values())))}", flush=True)
    if args.midi:
        controls, cc_note = resolve_controls(args.cc)
        encoders = EncoderMap(controls, remember=CC_MAP, machine=machine)
        print(cc_note, flush=True)
    positions = Positions(POSITIONS)
    if positions.trouble:
        print(positions.trouble, flush=True)

    screen = bank.screen_size()
    r = bank.build(args.checkpoint, args.device,
                  height=args.height, screen=screen,
                  options=bank.LoadOptions(compile_net=args.compile_net, capture=args.capture,
                                measure_grain=args.measure_grain, exact=args.exact,
                                direction_floor=args.direction_floor))
    print(f"generator: {r.report()}", flush=True)
    if args.height is None and r.height < r.cfg.ladder.height:
        print(f"  fitted to the {screen[0]}x{screen[1]} screen -- a take will be this size too; "
              f"--height native to record at {r.cfg.ladder.width}x{r.cfg.ladder.height}",
              flush=True)

    shelf = None if args.no_shelf else bank.Shelf(r, args.runs)
    if shelf is not None:
        print(f"models: {len(shelf.entries())} on disk under {args.runs}", flush=True)

    runner = PresetRunner(preset, extractor.channel_of() or tracks, float(args.fps),
                         channels=extractor.n)
    # Which dials this generator can actually write. Told to the runner rather than looked up,
    # because every input meets there and one refusing while the others write is an inert knob.
    runner.use_model(r.current)
    if args.midi:
        reader = ClockReader(
            clock, args.midi_port,
            on_control=(lambda ch, n, v, top: encoders.apply(runner, ch, n, v, top))
            if encoders is not None else None,
            on_pressure=(lambda ch, pad, v: pressure.apply(runner, ch, pad, v))
            if pressure is not None else None,
            on_note=extractor.on_note if use_notes else None)
        reader.start()
        time.sleep(0.2)
        print(reader.describe(), flush=True)
    report_dropped(runner)
    report_unreachable((encoders, pressure), r.current.layout)
    walk = r.walk(runner.walk_cfg)

    # A model's own dials go with it and come back with it; the spine stays under the hand.
    def stash(model):
        positions.stash(bank.slug_for(model.path), runner, HAND, per_model(model.layout))

    def recall(model) -> str:
        back = positions.recall(bank.slug_for(model.path), runner, HAND, HAND_PRIORITY,
                                per_model(model.layout))
        return f"  hands back on {', '.join(sorted(back))}" if back else ""

    recalled = recall(r.current)
    if recalled:
        print(recalled.strip() + " from last time", flush=True)

    def switch_patch(delta):
        """Next or previous setting, in place. Called from the window's thread."""
        runner.load(library.step(delta))
        report_dropped(runner)
        print(f"preset: {runner.preset.name} -- {runner.preset.blurb}", flush=True)

    def switch_model(delta):
        """Play a different loaded generator. One integer assignment; see `Bank.current`."""
        want = r.models[(r.index + delta) % len(r.models)]
        # A take is one size for its whole length, and models in a bank no longer are. The encoder
        # was opened at the outgoing model's shape; refused as the shelf refuses a load mid-take.
        if rec is not None and r.size_of(want) != (r.height, r.width):
            print(f"not while a take is running -- {want.name} is "
                  f"{'x'.join(str(n) for n in reversed(r.size_of(want)))} and this take is "
                  f"{r.width}x{r.height}; press v to stop it first", flush=True)
            return
        was = (r.width, r.height, r.cfg.nz)
        stash(r.current)
        m = r.use(r.index + delta)
        runner.use_model(m)
        report_dropped(runner)
        report_unreachable((encoders, pressure), m.layout)
        back = recall(m)
        now = (r.width, r.height, r.cfg.nz)
        # Named, because both are visible and neither is a fault. A different size moves the window;
        # a different latent width moves the picture, since the walk hands over a position, not a vector.
        moved = "" if now == was else (
            f"  [{was[0]}x{was[1]} z{was[2]} -> {now[0]}x{now[1]} z{now[2]}"
            + ("" if was[2] == now[2] else "; same walk, new latent") + "]")
        print(f"model: {m.name}  ({r.index + 1} of {len(r.models)}){moved}{back}", flush=True)

    #: A model change asked for from the window's thread, waiting for the frame's own.
    pending_model = [0]

    def ask_model(delta):
        """Record a model change from whichever thread the key or the click arrived on."""
        pending_model[0] += delta

    def save_setting(found):
        """Write down what is at the controls, and carry on playing it."""
        path = library.save(found)
        runner.load(library.current)
        print(f"saved {path}  --  '{library.current.name}' is in the tab rotation now",
              flush=True)

    asked = {"still": False, "record": False}

    def ask(what):
        def request():
            asked[what] = True
        return request

    display = panel = None
    if not args.headless:
        from ganlive.window import Display

        if args.console:
            from ganlive.strip import DialPanel

            actions = {"preset": switch_patch, "save": save_setting,
                       "record": ask("record"), "still": ask("still")}
            if len(r.models) > 1 or shelf is not None:
                actions["model"] = ask_model
            panel = DialPanel(runner, actions=actions, extractor=extractor, bank=r, shelf=shelf,
                              encoders=encoders)
        display = Display((r.height, r.width), title=f"ganlive - {preset.name}",
                          overlay=panel, fullscreen=not args.console)
        print("window open. Esc or Q to stop.", flush=True)

    to_window = r.stage.bgra_bytes

    first = r.stage.step(r.current.net(walk.latent(0.0)))
    to_window(first)
    dev.synchronize()

    rec = None
    stills: list = []

    def toggle_take(at=None):
        """Start or stop recording. Called by the frame loop, never by the window; see `asked`."""
        nonlocal rec
        if rec is not None:
            report = rec.stop()
            print(f"take: {report}  staging {r.stage.pinned()}", flush=True)
            if guide.running:
                print(f"  sync: {guide.describe(guide.stop(report))}", flush=True)
            rec = None
        else:
            path = next_path(TAKES, "take", ".mp4")
            rec = video.Recorder(path, r.width, r.height, args.fps, args.codec,
                                 realtime=True, depth=TAKE_DEPTH).start()
            print(f"recording {r.width}x{r.height} to {path} with {args.codec}", flush=True)
            if args.guide:
                armed = guide.start(path, bpm=clock.bpm, beat=clock.beats,
                                    beat_source=clock.source, at=at,
                                    about={"fps": float(args.fps), "width": r.width,
                                           "height": r.height, "model": r.current.name,
                                           "preset": runner.preset.name})
                print(f"  sync: {armed}", flush=True)
        if panel is not None:
            panel.recording = rec is not None

    if args.record:
        toggle_take()
        rec.offer(r.stage.nv12_bytes(first, "take", TAKE_DEPTH + 2))
        if not rec.drain():
            print("  the encoder did not start; the take may be short", flush=True)

    print(f"host staging: {r.stage.pinned()}", flush=True)

    if card is not None:
        card.start()

    period = 1.0 / args.fps
    total = int(args.seconds * args.fps) if args.seconds else 0
    ms: deque[float] = deque(maxlen=SAMPLE_CAP)
    # The status line wants the last 60 frames four times a second, and `list(ms)[-60:]` copies
    # the whole deque to read its tail -- O(110,000) an hour in, on the frame loop's thread.
    recent: deque[float] = deque(maxlen=60)
    # Same cap per model, so a long stretch on one cannot starve the others. Why at all:
    # `per_model_lines`.
    by_model: dict[str, deque] = {}
    late, frames = 0, 0
    heard0 = hears() if hears is not None else 0.0
    start = time.perf_counter()
    print(f"\nplaying '{preset.name}' at {args.fps} fps"
          + (f" for {args.seconds:g}s" if total else " -- stop it with the window"),
          flush=True)

    HIT_GRACE_S = 6.0
    kind, fix = silence_words(use_notes, use_audio, machine)
    hits_checked = False
    t_start = time.perf_counter()
    # **Seconds, not frames-that-would-have-fitted.** `total` is what `--seconds` buys at the
    # asked-for rate, and a model that cannot hold that rate used to run past the end instead of
    # stopping: `--seconds 12` on a StyleGAN2-1024 through ONNX ran for 68. The deadline bounds
    # it, and a run cut short says so rather than reading as a crash.
    deadline = t_start + args.seconds if args.seconds else 0.0
    try:
        with torch.no_grad():
            while ((not total or frames < total)
                   and (not deadline or time.perf_counter() < deadline)):
                if shelf is not None and shelf.pending is not None:
                    if rec is not None:
                        shelf.pending = None
                        print("not while a take is running -- press v to stop it first",
                              flush=True)
                    else:
                        print(f"loading {shelf.pending} -- the picture stops until it is ready",
                              flush=True)
                        got = shelf.service()
                        print(f"  {shelf.note}", flush=True)
                        if got is not None:
                            switch_model(r.index_of(got.path) - r.index)

                # Here, and not in the handler that asked for it: see `ask_model`.
                if pending_model[0]:
                    delta, pending_model[0] = pending_model[0], 0
                    switch_model(delta)

                t0 = time.perf_counter()
                model = r.current
                if ticks is not None:
                    ticks(t0)
                if not hits_checked and reactive and t0 - t_start > HIT_GRACE_S:
                    hits_checked = True
                    report_unheard(extractor, kind, fix, notes, note_channels, HIT_GRACE_S)
                runner.observe(extractor.drain())
                runner.apply(extractor.since, extractor.features(), model.knobs)
                clock.advance(period)
                guide.mark(clock.beats)
                out = model.net(walk.latent(clock.beats))
                frame = r.stage.step(out)
                if asked["record"]:
                    asked["record"] = False
                    toggle_take(t0)
                with r.stage.deferred():
                    shown = (to_window(frame)
                             if display is not None and display.wants else None)
                    taped = None
                    if rec is not None:
                        if rec.wants:
                            taped = r.stage.nv12_bytes(frame, "take", TAKE_DEPTH + 2)
                        else:
                            rec.skip()
                if shown is not None:
                    display.publish(shown)
                if taped is not None:
                    rec.offer(taped)
                if asked["still"]:
                    asked["still"] = False
                    shot = next_path(STILLS, model.name.replace(" ", "-"), ".png")
                    stills.append(video.save_still(shot, r.stage.rgb_still(frame)))
                    print(f"still: {shot}", flush=True)
                took = (time.perf_counter() - t0) * 1000
                ms.append(took)
                recent.append(took)
                if model.name not in by_model:
                    by_model[model.name] = deque(maxlen=SAMPLE_CAP)
                by_model[model.name].append(took)
                frames += 1
                if panel is not None:
                    panel.beats = clock.beats
                if panel is not None and frames % 15 == 0:
                    live = time.perf_counter() - start
                    share = 1.0 if hears is None else (hears() - heard0) / max(live, 1e-6)
                    deaf = "" if share > 0.95 else f" · DEAF {share:.0%}"
                    panel.status = (f"{frames / live:.0f} fps · "
                                    f"{stat_ms(recent)['median']:.1f} ms · "
                                    f"{clock.bpm:.0f} bpm {clock.source}"
                                    + (f" · REC {rec.seconds:.0f}s {rec.dropped} dropped"
                                       if rec is not None else "") + deaf)

                slack = start + frames * period - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    late += 1
                if display is not None and display.stopped:
                    print("stopped from the window", flush=True)
                    break
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        stash(r.current)
        if positions.trouble:
            print(positions.trouble, flush=True)
        if encoders is not None:
            encoders.flush()
        if rec is not None:
            toggle_take()
        for writer in stills:
            writer.join(timeout=10.0)
        if source is not None:                            # --no-audio has nothing to stop
            source.stop()
            source.close()
        if reader is not None:
            reader.stop_flag = True
        if display is not None:
            display.close()

    wall = time.perf_counter() - start
    if not ms:
        print("no frames")
        return 1
    report_frames(frames, wall, ms, total, late, by_model, dropped, source,
                  args, period)
    report_clock(clock, reader, args, pressure, encoders, runner, machine)
    if reactive:
        report_drums(extractor, tracks, kind, fix, hears, heard0, wall)
    peak = dev.peak_memory_gb()
    if peak:
        print(f"  memory      {peak:.2f} GB peak")
    return 0

