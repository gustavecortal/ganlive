"""Play a GAN's latent space live, from a drum machine's MIDI and audio.

    ganlive play --checkpoint runs/my-run --console
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from pathlib import Path

from ganlive.checkpoints import slug_for
from ganlive.clock import MusicalClock
from ganlive.control.audio import NoAudioDevice, input_stream, pick_input, require_sounddevice
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
from ganlive.files import CHANNEL_MAP, SETTINGS, next_path, remember
from ganlive.presets import POSITIONS_NAME, Library, Positions, PresetRunner
from ganlive.record import video
from ganlive.record.sync import Guide
from ganlive.strip import PRIORITY as HAND_PRIORITY
from ganlive.strip import SOURCE as HAND
from ganlive.strip import DialPanel
from ganlive.timing import stat_ms
from ganlive.tools import add_device, parser
from ganlive.window import Display, parse_height, screen_size

OUT = Path("runs/ganlive")
TAKES, STILLS = OUT / "takes", OUT / "stills"

#: The knob map `l` writes, in `--cc`'s own words; `--cc` on the command line replaces it.
CC_MAP = SETTINGS / "cc.txt"
POSITIONS = SETTINGS / POSITIONS_NAME

#: Frame times kept for the end-of-run report: about half an hour at 60 fps.
SAMPLE_CAP = 110_000

#: Frames the recorder may queue before it starts skipping.
TAKE_DEPTH = 2
#: Host buffers a take's frames cycle through: the recorder's queue, the frame being encoded,
#: and the one still downloading behind the next frame's generation.
TAKE_RING = TAKE_DEPTH + 3

#: Seconds into a reactive run before saying that no hits have arrived.
HIT_GRACE_S = 6.0


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
    """Say which knobs and pads name a dial this model does not have: a knob that moves
    nothing reads exactly like a knob whose messages are not arriving."""
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
    """The audio channel map, and one line saying where it came from. A map given with
    `--map` is remembered for next time."""
    if explicit:
        tracks = parse_channel_map(explicit)           # parse first: never save an unusable map
        trouble = remember(CHANNEL_MAP, explicit)
        if trouble:
            return tracks, f"map: {explicit} (not remembered: {trouble})"
        return tracks, f"map: {explicit}  -- remembered in {CHANNEL_MAP}"
    found = remembered(CHANNEL_MAP, parse_channel_map, "map")
    if found is not None:
        return found
    return channel_map(layout), (
        f"map: --layout {layout}, a GUESS. A machine's USB audio need not start the kit at "
        f"channel 0, so every hit may be credited to the wrong drum and a shared channel "
        f"lights both of its tracks. Run `ganlive doctor --learn`, then pass --map once and "
        f"it is remembered.")


def per_model_lines(by_model, budget_ms: float) -> list[str]:
    """One timing row per model played, in the order first played; nothing when only one was.
    An aggregate over a bank cannot say which model the slow frames belong to."""
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
    """`(stream, dropouts)` for the audio input. The stream is not started: `main` starts it
    last, once the picture is ready, so no audio queues up behind the compile."""
    sd = require_sounddevice()
    dropped = [0]

    def callback(indata, _frames, _t, status):
        if status:
            dropped[0] += 1
        extractor.push(indata)

    stream = input_stream(sd, device, channels, extractor.sr, args.blocksize, callback)
    print(f"audio: {info['name']}, {channels} ch at {extractor.sr} Hz, "
          f"blocksize {args.blocksize} "
          f"({args.blocksize / extractor.sr * 1000:.2f} ms); starts with the picture",
          flush=True)
    return stream, dropped


def _parser():
    """Every option, in the groups `--help` prints them in."""
    ap = parser("play", __doc__)

    what = ap.add_argument_group("what to play")
    what.add_argument("--checkpoint", type=Path, action="append", metavar="PATH", required=True,
                      help="a checkpoint, a run directory for its newest, or an exported .onnx. "
                           "Repeatable: `[` and `]` switch between them on the next frame, and "
                           "they may differ in latent width, native size and aspect")
    what.add_argument("--runs", type=Path, default=Path("runs"),
                      help="where `m` looks for more models to load")
    what.add_argument("--no-shelf", action="store_true",
                      help="no model picker, and no start-up scan of --runs")
    what.add_argument("--preset", default="default",
                      help=f"which setting to start on. Saved with `s` into {SETTINGS}")

    how = ap.add_argument_group("how it runs")
    add_device(how)
    how.add_argument("--height", type=parse_height, default=None,
                     metavar="auto|native|PIXELS",
                     help="what the window is sent -- the generator always runs at its native "
                          "size. The largest cost on the loop: `auto` shrinks on the card to "
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
    how.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                     help="finish every frame before starting the next. By default a frame that "
                          "has missed its slot leaves its download running behind the next "
                          "frame's generation and is shown once that is under way; a frame on "
                          "time is shown at once either way")
    how.add_argument("--headless", action="store_true",
                     help="no window: run the loop and report the timing")
    how.add_argument("--console", action="store_true",
                     help="a slider per dial beside the picture, turned with the mouse")

    dials = ap.add_argument_group("which dials appear")
    dials.add_argument("--direction-floor", type=float, default=None, metavar="X",
                       help="how many times a random direction of the same length a derived "
                            "direction must move the picture to earn a dial. Default: 2 "
                            "(pixels.RANDOM_FLOOR). A taste control: lower is not simply more, "
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


def silence_words(use_notes: bool, use_audio: bool, machine) -> tuple[str, str]:
    """`(what to call the trigger source, what to check)` for a run where no hits arrive.
    `machine` is the `control.machine` profile whose words say what to switch on."""
    m = machine
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
        notes_on = {}
        for ch, note in seen:
            notes_on.setdefault(ch, set()).add(note)
        mode = output_mode(notes_on, notes)
        if mode:
            hint = (f" -- that traffic is {mode.upper()} CH; "
                    + ("drop --midi-channels" if mode == "auto"
                       else "pass --midi-channels 1-12"))
        else:
            hint = f" on channels {sorted(c + 1 for c in notes_on)}"
        # Or a kit whose notes are not the ones being read.
        unknown = sorted(n for ns in notes_on.values() for n in ns
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
    # More than one, because the last frame and the deadline can land together.
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
        # Counted against the kit's channels, not the stream's width, which may be wider.
        f = extractor.features()
        print(f"  {kind:<11} {played} of {len(names)} kit channel(s) carried drums; "
              f"density {f['density']:.2f}/s, energy {f['energy']:.2f}")
    # Only channels a track maps to: a stream channel no track points at cannot be "silent".
    counts = [(int(n), "/".join(sorted(names[i])))
              for i, n in enumerate(extractor.hits) if i in names]
    heard_from = " ".join(f"{label} {n}" for n, label in counts if n)
    quiet = " ".join(label for n, label in counts if not n)
    print(f"  reached      {heard_from or 'nothing'}")
    if quiet:
        print(f"  silent       {quiet}  -- any rule wired to these can never fire")
    # The other half, which says the map is wrong: a channel carrying drums no track claims.
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


def pick_audio(args):
    """`(device, info, channels)` for the audio input, or None when `auto` finds none.
    An explicit `--triggers audio` or `both` raises instead."""
    kwargs = {"channels": args.channels, "samplerate": args.samplerate}
    if args.audio_name is not None:
        kwargs.update(pattern=args.audio_name, hostapi=None)
    try:
        import sounddevice as sd

        return pick_input(sd, args.audio_device, **kwargs)
    except (NoAudioDevice, ImportError, OSError) as exc:
        if args.triggers != "auto":
            raise
        print(f"audio: none -- {exc}\n  Triggers come from MIDI notes alone; --triggers "
              f"audio to insist on the sound.", flush=True)
        return None


def open_triggers(args, tracks, use_notes, use_audio, picked, note_channels, notes):
    """Where the hits come from: `(extractor, source, card, dropouts)`.

    `source` is whatever has to be stopped at the end, `card` the audio stream `main` starts
    last, and `dropouts` a one-item list the audio callback counts into."""
    dropped = [0]
    if use_notes and not use_audio:
        extractor = NoteFeatures(len(TRACKS), channels=note_channels, notes=notes)
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
        return extractor, None, None, dropped

    if not use_audio:
        extractor = FeatureExtractor(max(tracks.values(), default=11) + 1, args.samplerate)
        print("audio: none. The walk runs and the sliders drive it.", flush=True)
        return extractor, None, None, dropped

    if args.simulate:
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
        print(f"audio: the stand-in drum machine, {channels} ch "
              f"({'shared voices' if channels < 12 else 'one per track'}) "
              f"at {take.samplerate} Hz{heard}", flush=True)
        return extractor, source, None, dropped

    device, info, channels = picked
    heard_by = FeatureExtractor(channels, args.samplerate)
    stream, dropped = open_audio(args, heard_by, device, info, channels)
    if not use_notes:
        return heard_by, stream, stream, dropped
    by = ("their MIDI channel" if note_channels else
          f"their MIDI note, {len(TRACKS)} tracks in one note space")
    print(f"triggers: BOTH. Notes name a track by {by}; audio fills in anything that "
          f"sends none, and an onset on a voice a note has just claimed is dropped "
          f"rather than counted twice.", flush=True)
    extractor = BothFeatures(heard_by, tracks, channels=note_channels, notes=notes)
    return extractor, stream, stream, dropped


class Requests:
    """What the window's thread asked for, waiting for the frame loop to act on it.

    Keys and clicks arrive on the window's thread, but recording, stills and model switches
    touch the card, so the loop takes them at the top of its next frame."""

    def __init__(self, bank) -> None:
        self.bank = bank
        self.still = False
        self.record = False
        #: The model index asked for, or None.
        self.model: int | None = None

    def ask_still(self) -> None:
        self.still = True

    def ask_record(self) -> None:
        self.record = True

    def ask_model(self, delta: int = 0, to: int | None = None) -> None:
        """A key steps from wherever the last unserved request left off; a picker click
        names its model outright, so two clicks during a stall land on the second."""
        base = self.bank.index if self.model is None else self.model
        self.model = (base + delta if to is None else to) % len(self.bank.models)

    def take(self, name: str):
        """The request's value, cleared."""
        value = getattr(self, name)
        setattr(self, name, None if name == "model" else False)
        return value


def serve_models(shelf, requests, r, recording: bool, switch_model) -> None:
    """At the top of a frame: do a load the picker asked for, then any model switch."""
    if shelf is not None and shelf.pending is not None:
        if recording:
            shelf.pending = None
            print("not while a take is running -- press v to stop it first", flush=True)
        else:
            print(f"loading {shelf.pending} -- the picture stops until it is ready",
                  flush=True)
            got = shelf.service()
            print(f"  {shelf.note}", flush=True)
            if got is not None:
                switch_model(r.index_of(got.path) - r.index)
    want = requests.take("model")
    if want is not None and want != r.index:
        switch_model(want - r.index)


def status_line(frames: int, elapsed: float, recent, clock, rec, share: float) -> str:
    """The strip's first line: frame rate, frame time, tempo, the take, and audio delivery."""
    deaf = "" if share > 0.95 else f" · DEAF {share:.0%}"
    return (f"{frames / elapsed:.0f} fps · "
            f"{statistics.median(recent):.1f} ms · "
            f"{clock.bpm:.0f} bpm {clock.source}"
            + (f" · REC {rec.seconds:.0f}s {rec.dropped} dropped" if rec is not None else "")
            + deaf)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    if args.console and args.headless:
        print("--console needs a window; drop --headless")
        return 1
    if args.monitor and not args.simulate:
        print("--monitor plays the stand-in, so it needs --simulate. A real machine is "
              "already making the sound.")
        return 1

    # Imported here, after the arguments parse, so `--help` and a typo answer at once.
    import torch

    from ganlive import bank
    from ganlive import device as dev
    from ganlive.families import LoadOptions

    tracks, map_note = resolve_map(args.map, args.layout)
    # What to tell the user to switch on, in their own machine's words; see `control.machine`.
    machine = profile(args.midi_port, args.audio_name or "")
    # Before anything opens the card: the affinity is for this process's submission thread,
    # and the compile that follows is the first thing to use it.
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
        # Picked now, so `auto` can fall back to notes alone when no input opens.
        picked = pick_audio(args)
        use_audio = picked is not None
    reactive = use_audio or use_notes
    if args.preset not in library.names:
        print(f"unknown setting {args.preset!r}; have {', '.join(library.names)}")
        return 1
    preset = library.select(args.preset)

    extractor, source, card, dropped = open_triggers(args, tracks, use_notes, use_audio,
                                                     picked, note_channels, notes)
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

    screen = screen_size()
    floor = {} if args.direction_floor is None else {"direction_floor": args.direction_floor}
    r = bank.build(args.checkpoint, args.device,
                   height=args.height, screen=screen,
                   options=LoadOptions(compile_net=args.compile_net, capture=args.capture,
                                       measure_grain=args.measure_grain, exact=args.exact,
                                       **floor))
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
    # Every input writes dials through the runner, so it is the one told which dials this
    # generator has.
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

    # A model's own dials are stashed with it and recalled with it; the shared blocks stay put.
    def stash(model):
        positions.stash(slug_for(model.path), runner, HAND, per_model(model.layout))

    def recall(model) -> str:
        back = positions.recall(slug_for(model.path), runner, HAND, HAND_PRIORITY,
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
        """Play a different loaded generator. Called by the frame loop."""
        land()
        want = r.models[(r.index + delta) % len(r.models)]
        # A take is one size for its whole length, and models in a bank need not be.
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
        # A different size moves the window; a different latent width moves the picture,
        # since the walk hands over a position, not a vector.
        moved = "" if now == was else (
            f"  [{was[0]}x{was[1]} z{was[2]} -> {now[0]}x{now[1]} z{now[2]}"
            + ("" if was[2] == now[2] else "; same walk, new latent") + "]")
        print(f"model: {m.name}  ({r.index + 1} of {len(r.models)}){moved}{back}", flush=True)

    requests = Requests(r)

    def save_setting(found):
        """Write down what is at the controls, and carry on playing it."""
        path = library.save(found)
        runner.load(library.current)
        print(f"saved {path}  --  '{library.current.name}' is in the tab rotation now",
              flush=True)

    display = panel = None
    if not args.headless:
        if args.console:
            actions = {"preset": switch_patch, "save": save_setting,
                       "record": requests.ask_record, "still": requests.ask_still}
            if len(r.models) > 1 or shelf is not None:
                actions["model"] = requests.ask_model
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
    #: The last frame's `(ticket, shown, taped)` while its downloads are still running behind
    #: the next frame's generation. None once it has been put up.
    in_flight = None

    def land():
        """Show and tape the frame left in flight, once its downloads have finished."""
        nonlocal in_flight
        if in_flight is None:
            return
        (ticket, shown, taped), in_flight = in_flight, None
        ticket.wait()
        if shown is not None:
            display.publish(shown)
        if taped is not None and rec is not None:
            rec.offer(taped)

    def toggle_take(at=None):
        """Start or stop recording. Called by the frame loop, never by the window."""
        nonlocal rec
        land()                  # the frame in flight belongs to the take it was taped for
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
        rec.offer(r.stage.nv12_bytes(first, "take", TAKE_RING))
        if not rec.drain():
            print("  the encoder did not start; the take may be short", flush=True)

    print(f"host staging: {r.stage.pinned()}", flush=True)

    if card is not None:
        card.start()

    period = 1.0 / args.fps
    total = int(args.seconds * args.fps) if args.seconds else 0
    ms: deque[float] = deque(maxlen=SAMPLE_CAP)
    #: The last second's frame times, for the status line.
    recent: deque[float] = deque(maxlen=60)
    #: Frame times per model, each capped like `ms`; see `per_model_lines`.
    by_model: dict[str, deque] = {}
    late, frames = 0, 0
    heard0 = hears() if hears is not None else 0.0
    kind, fix = silence_words(use_notes, use_audio, machine)
    hits_checked = False
    start = time.perf_counter()
    # A deadline as well as a frame count, so a model that cannot hold the asked-for rate
    # still stops after `--seconds`.
    deadline = start + args.seconds if args.seconds else 0.0
    print(f"\nplaying '{preset.name}' at {args.fps} fps"
          + (f" for {args.seconds:g}s" if total else " -- stop it with the window"),
          flush=True)
    try:
        with torch.no_grad():
            while ((not total or frames < total)
                   and (not deadline or time.perf_counter() < deadline)):
                serve_models(shelf, requests, r, rec is not None, switch_model)

                t0 = time.perf_counter()
                # Before anything writes this frame's inputs: a captured generator reads its
                # latent and settings from host buffers, and writes every frame into one
                # buffer the last frame's conversions may still be reading.
                r.stage.release()
                model = r.current
                if ticks is not None:
                    ticks(t0)
                if not hits_checked and reactive and t0 - start > HIT_GRACE_S:
                    hits_checked = True
                    report_unheard(extractor, kind, fix, notes, note_channels, HIT_GRACE_S)
                runner.observe(extractor.drain())
                runner.apply(extractor.since, extractor.features(), model.knobs)
                clock.advance(period)
                guide.mark(clock.beats)
                out = model.net(walk.latent(clock.beats))
                frame = r.stage.step(out)
                if requests.take("record"):
                    toggle_take(t0)
                with r.stage.handoff() as sent:
                    shown = (to_window(frame)
                             if display is not None and display.wants else None)
                    taped = None
                    if rec is not None:
                        if rec.wants:
                            taped = r.stage.nv12_bytes(frame, "take", TAKE_RING)
                        else:
                            rec.skip()
                # The last frame's, whose downloads ran while this one was being generated.
                land()
                in_flight = (sent.ticket, shown, taped)
                if requests.take("still"):
                    shot = next_path(STILLS, model.name.replace(" ", "-"), ".png")
                    stills.append(video.save_still(shot, r.stage.rgb_still(frame)))
                    print(f"still: {shot}", flush=True)
                # A frame on time goes up now. Overlapping its download with the next frame
                # costs a frame of lag, worth paying only on a frame that missed its slot.
                if not args.pipeline or time.perf_counter() < start + (frames + 1) * period:
                    land()
                now = time.perf_counter()
                took = (now - t0) * 1000
                ms.append(took)
                recent.append(took)
                if model.name not in by_model:
                    by_model[model.name] = deque(maxlen=SAMPLE_CAP)
                by_model[model.name].append(took)
                frames += 1
                if panel is not None:
                    panel.beats = clock.beats
                    if frames % 15 == 0:
                        elapsed = now - start
                        share = 1.0 if hears is None else (hears() - heard0) / max(elapsed, 1e-6)
                        panel.status = status_line(frames, elapsed, recent, clock, rec, share)

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
        land()
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
