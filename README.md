# ganlive

**Play a GAN like an instrument.**

<p align="center">
  <img src="docs/demo.webp" alt="ganlive playing NVIDIA's FFHQ StyleGAN2 against its built-in drum machine: a direction pushed by hand, the routing grid lighting up with the drums, a switch to the AFHQ cat model and back" width="860">
</p>

ganlive turns a trained image generator into something you perform with. The picture walks
through the model's latent space in time with your drum machine: every move is measured in
beats, every kick and snare can push a dial, and the knobs on your controller turn the rest.
It runs at 60 fps and beyond on one GPU, and it works with no hardware at all.

- **Any GAN.** NVIDIA's StyleGAN2 weights unchanged, the FastGAN this project was built
  around, or any generator as an ONNX graph — including one fetched from the Hugging Face Hub.
- **Dials found, not configured.** Each model's controls are discovered from its weights when
  it loads, driven to both ends, and measured. A dial that does not visibly move the picture is
  never offered as one. There is no per-model configuration file.
- **In time with the music.** The walk follows your drum machine's MIDI clock. Drums arrive as
  MIDI notes, as audio, or both, and a routing grid wires any drum to any dial.
- **Any machine.** Windows, macOS or Linux; CUDA, Intel XPU, Apple MPS, or the CPU, detected
  rather than configured. The forward pass is compiled and replayed as one device graph where
  the backend allows it.
- **Takes you can use.** Record the performance to video with a guide track that lines it up
  with your multitrack in a DAW, or grab a full-resolution still.

## Contents

[Quick start](#quick-start) · [The interface](#the-interface) · [Playing with hardware](#playing-with-hardware) ·
[Models](#models) · [How the dials are found](#how-the-dials-are-found) · [Recording](#recording) ·
[Speed](#speed) · [Install](#install) · [Commands](#commands) · [Glossary](#glossary) ·
[Code layout](#code-layout) · [Development](#development) · [Related work](#related-work)

## Quick start

No controller, no drum machine. This converts NVIDIA's public FFHQ model and plays it against a
built-in drum machine, with sliders you turn with the mouse.

```bash
# 1. Install (pick the torch build for your GPU; see Install below)
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[audio,record]"

# 2. Get a model: NVIDIA's StyleGAN2 FFHQ, converted once
git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch
curl -L -o ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
ganlive import-stylegan2 ffhq.pkl --repo stylegan2-ada-pytorch

# 3. Play it
ganlive play --checkpoint runs/stylegan2/ffhq.pt --console --simulate --monitor
```

- `--console` opens the dial strip beside the picture.
- `--simulate` plays a twelve-voice drum pattern into the same path a real machine would drive.
- `--monitor` sends that pattern to your speakers, so there is a beat to judge the picture against.

The first launch compiles each model for a minute or two; later launches reuse the compiled
graphs and open in under half a minute. Press `Esc` to stop. Drop `--simulate` and `--monitor` once a controller is plugged in.

## The interface

The window is the picture with the **strip** on its right: one slider per dial, grouped in
blocks, with a description of whichever dial you last touched.

| Block | Dials | Same on every model? |
|---|---|---|
| `MASTER` | `reaction` — how hard the picture answers individual hits | yes |
| `MOTION` | `speed` `spread` `hold` `late` `grid` — where the walk goes and when, in beats | yes |
| `LATENT` | `dir1` … `dir8` — the model's strongest directions, ranked by measured effect | names yes, meaning per model |
| `MODEL` | this architecture's own controls: truncation per style range, noise per resolution, layer gains | per model |

`MOTION` is the musical part:

- `speed` — how often the picture sets off somewhere new: one move every 8 beats down to every half beat.
- `spread` — how far each move goes, from one picture breathing to a new scene every time.
- `hold` — how much of each move is spent standing still.
- `late` — whether the movement leaves on the beat or arrives on the next one.
- `grid` — glide, or step in quarter, eighth, sixteenth or thirty-second notes.

**Mouse:** drag a slider to hold it, wheel to nudge it, right-click to let it go back to the
preset and the drums.

**Keys** (with `--console`):

| Key | Does |
|---|---|
| `g` | routing grid: click a cell to wire a drum to a dial, wheel to set how hard, right-click to reverse it |
| `l` | learn: the next knob you turn on the controller takes the dial you last touched |
| `[` `]` | previous / next loaded model |
| `m` | model picker: everything under `--runs`; click to load or switch |
| `tab` | next preset (`shift+tab` previous) |
| `s` | save what is at the controls as a preset |
| `r` | let go of every dial held with the mouse |
| `v` | start / stop recording |
| `c` | save a still |
| `n` | show native pixels, cropped (arrow keys pan) |
| `f` | fullscreen |
| `Esc` / `q` | quit |

Without `--console` the picture opens fullscreen and only `n`, `f`, the arrows and `Esc` apply.

A dial drawn dark is one this model does not have, or one that was measured to do nothing on
it; the description says which.

## Playing with hardware

Plug in a controller and drop `--simulate`. ganlive picks the accelerator, the MIDI ports and
an audio input itself.

**Drums.** The vocabulary is twelve tracks — `BD SD RS CP BT LT MT HT CH OH CY CB` — named or
indexed 0–11. Tell ganlive which note is which drum:

```bash
ganlive play --checkpoint runs/stylegan2/ffhq.pt --console --notes 36=BD,38=SD,42=CH,46=OH   # a General MIDI kit
ganlive play --checkpoint runs/stylegan2/ffhq.pt --console --notes 0                         # pads on consecutive notes from 0
```

Use `--midi-channels` instead when your machine sends each track on its own MIDI channel. If it
also sends audio with one channel per drum, `--triggers both` reads hits from the sound as
well (`ganlive doctor --learn` maps which channel is which drum).

**Knobs.** Press `l` on a dial and turn a knob, or declare the wiring:

```bash
--cc "16=w_fine,2:17=noise_64,n1.3=dir1"    # CC 16; CC 17 on channel 2; NRPN 1.3 (14-bit)
```

Learned knobs are remembered in `runs/ganlive/settings/cc.txt`.

**Pads.** `--pressure "BD=dir1"` lets you lean on a pad to hold a dial, using polyphonic aftertouch.

**Tempo.** With MIDI clock arriving, the walk follows it, and Start and Stop on the machine
start and stop the picture. Without clock it runs at `--bpm`.

**When a drum is not moving the picture,** ask three questions in order:

```bash
ganlive doctor              # is the hardware there? lists every audio input and MIDI port
ganlive doctor --listen     # is it sending? counts notes, clock and transport as they arrive
ganlive doctor --learn      # is it landing where ganlive thinks? maps audio channels to drums
```

`ganlive doctor --drive` starts the machine's sequencer from the computer and checks that its
steps send notes, and `--meter` shows live levels per audio channel. The defaults suit an
Elektron Analog Rytm over Overbridge (Elektron's USB audio), the machine this was built with;
every one of them is a flag.

## Models

Three families play, side by side in one session. A 1024×1024 StyleGAN2 with a 512-wide latent
can sit next to a 3072×2048 FastGAN with a 256-wide one; repeat `--checkpoint` and switch with
`[` and `]`.

| Family | Get one with | Notes |
|---|---|---|
| **StyleGAN2** | `ganlive import-stylegan2 model.pkl --repo stylegan2-ada-pytorch` | NVIDIA's weights, read by this project's own synthesis network, so their CUDA kernels are not needed and it runs on any GPU. StyleGAN3 is not supported. |
| **FastGAN** | download the published checkpoint below, or train one with [smallgen](https://github.com/gustavecortal/smallgen) | The architecture this project grew up with. Plays at 3072×2048. |
| **Any ONNX graph** | `ganlive adopt model.onnx` or `ganlive adopt hf:owner/repo` | Needs `.[onnx]` (or `.[onnx-intel]` on an Intel GPU), plus `.[hub]` for `hf:` sources. |

**Adopting a model you did not write.** `adopt` fetches a generator, exports it to ONNX if it
is not already, freezes its random draws, finds its dials, measures what each is worth and
writes all of that into the graph's metadata. The file that comes out plays on its own:

```bash
ganlive adopt hf:someone/some-gan --trust-remote-code    # runs the repo's own code: only for repos you trust
ganlive play --checkpoint runs/onnx/some-gan.onnx --console
```

It refuses rather than guesses in three places: code it has not been allowed to run, a module
that never produced a picture, and a graph that is still non-deterministic after its draws are
frozen.

**The published FastGAN.** `gv-2048-ft`, 3072×2048, is on the Hugging Face Hub at
[gustavecortal/ganlive-fastgan-3072](https://huggingface.co/gustavecortal/ganlive-fastgan-3072):

```bash
huggingface-cli download gustavecortal/ganlive-fastgan-3072 --local-dir runs/gv-2048-ft
ganlive play --checkpoint runs/gv-2048-ft --console --simulate
```

**Stronger directions, offline.** `ganlive dials runs/stylegan2/ffhq.pt` spends about 100
seconds finding directions from the whole generator rather than from its first layer, and
saves them beside the checkpoint for every later load to use.

**Exporting.** `ganlive export-onnx --checkpoint runs/my-run` writes a FastGAN checkpoint as an
ONNX graph with its dials as an input, for runtimes other than PyTorch.

## How the dials are found

Every measurement in ganlive is in **8-bit levels**: the mean absolute difference between two
frames, on the 0–255 scale of the picture you see.

- **Directions** (`dir1`…`dir8`) come from [SeFa](https://arxiv.org/abs/2007.06600): the
  singular vectors of the first weight the latent meets. For a StyleGAN2, that is the style
  affines of each style range (coarse, mid, fine), so the directions live in `w`. Each
  candidate is pushed and measured. It earns a dial only if it moves the picture
  `--direction-floor` times more than a random direction of the same length (2× by default),
  and the survivors are ranked by effect.
- **Model dials** are found by walking the generator's graph: truncation per style range, the
  network's own noise at each resolution, the gain on each resolution stage. Each one is driven
  to both ends and kept only if it moves the picture by at least 1 level.
- **Travel** is calibrated: a dial's curve is laid out so that equal turns give roughly equal
  amounts of visible change, aiming for about 25 levels at full travel on every model.
- **Rest** is where the model behaves as trained. A dial rests there, and its readout says
  which way and how far you have moved it.

## Recording

`v` starts and stops a take, written to `runs/ganlive/` with the best hardware encoder this
machine opens (QSV, NVENC, VideoToolbox or AMF, falling back to x264). `--record` starts one
immediately. Beside each take, ganlive writes a mono guide track from the audio input and a
JSON beat map with a mark at every bar line. Drop both into your DAW and the take lines up with
the multitrack. `c` saves the current frame as a full-resolution PNG.

## Speed

Measured with `ganlive latency` on an Intel Arc A770 (torch 2.13+xpu), with the sound thread
running. Each row is one model's frame, from latent to a frame ready for the window:

| Model | Native size | Frame | fps |
|---|---|---|---|
| FFHQ StyleGAN2 | 1024×1024 | 10.7 ms | 94 |
| FastGAN `gv-2048-ft` (shown at 1620×1080) | 3072×2048 | 8.8 ms | 114 |
| FFHQ StyleGAN2 adopted as ONNX, OpenVINO FP16 | 1024×1024 | 14.3 ms | 70 |

With the window open and the strip drawn, the whole played loop on FFHQ took 12.4 ms (13.8 ms
at p95), inside a 60 fps budget with a third to spare.

Only that machine has been measured. `ganlive latency --checkpoint ...` measures yours in a few
minutes. Beside the median it prints a drift line, because a median hides a slope: a frame time
that creeps up by a fraction of a millisecond every second reads fine in every short window.

## Install

ganlive needs Python 3.10–3.13 and a PyTorch build for your hardware, installed first:

| Hardware | torch index |
|---|---|
| NVIDIA | `--index-url https://download.pytorch.org/whl/cu128` |
| Intel Arc / Core Ultra | `--index-url https://download.pytorch.org/whl/xpu` |
| AMD (Linux) | `--index-url https://download.pytorch.org/whl/rocm6.3` |
| Apple silicon, or CPU only | no `--index-url` |

Then `uv pip install -e ".[audio,record]"`, adding the extras you want:

| Extra | Adds |
|---|---|
| `audio` | audio input, for drums heard rather than sent as MIDI, and for the guide track |
| `record` | video recording |
| `onnx` | playing and adopting ONNX graphs (ONNX Runtime) |
| `onnx-intel` | the same through OpenVINO, the fastest path on Intel GPUs |
| `hub` | `adopt hf:...` |
| `all` | everything above except `onnx-intel` |
| `dev` | the test suite and the linter |

On Windows, audio interfaces that expose many channels only through ASIO are found
automatically.

## Commands

| Command | What it does |
|---|---|
| `ganlive play` | play one or more models live |
| `ganlive doctor` | check audio and MIDI: list, `--meter`, `--listen`, `--drive`, `--learn` |
| `ganlive latency` | measure this machine's frame time, drift included |
| `ganlive import-stylegan2` | convert an NVIDIA StyleGAN2-ADA pickle |
| `ganlive adopt` | make any ONNX graph or Hub generator playable |
| `ganlive dials` | find stronger directions offline and save them beside the checkpoint |
| `ganlive export-onnx` | export a FastGAN checkpoint as ONNX with its dials as an input |

`ganlive <command> --help` lists each command's options.

## Glossary

| Term | Meaning |
|---|---|
| **8-bit level** | the unit of every measurement: mean absolute pixel difference on the 0–255 scale |
| **dial** | one 0–1 control on the strip or a knob, with a rest position and a calibrated curve |
| **setting** | one number the generator reads every frame, such as a gain; dials write settings |
| **latent `z`** | the random vector a generator starts from |
| **style `w`** | StyleGAN's mapped latent; StyleGAN2 directions are pushes in `w` |
| **style range** | StyleGAN2's layers grouped by resolution — coarse, mid, fine — each with its own directions and truncation |
| **truncation** | pulling `w` toward the average; lower is more typical, higher more unusual |
| **walk** | the path through latent space, made of moves timed in beats |
| **track** | one of the twelve drum names, `BD` to `CB` |
| **impulse** | a drum hit pushing a dial and letting go |
| **preset** | dial positions plus the rules wiring drums and audio to dials |
| **bank** | the models loaded in one session |
| **adopt** | give a foreign generator measured dials, saved inside its ONNX file |
| **capture** | recording the compiled forward pass as one device graph, replayed every frame |
| **take** / **still** | a recorded video / a saved frame |

## Code layout

| Path | What lives there |
|---|---|
| `models/` | the three model families, graph capture, and the ONNX side: export, adoption, calibration, runtimes |
| `dials/` | what a dial is (`table`), finding directions (`derive`), the handles on a loaded generator (`steer`), and the measurement that decides which dials are offered (`gate`) |
| `control/` | MIDI in, audio in, the twelve-track drum vocabulary, and the built-in drum machine |
| `record/` | video, stills, and the guide track |
| `families.py`, `checkpoints.py`, `bank.py` | recognising a model file, preparing it, and holding the loaded models |
| `clock.py`, `walk.py` | musical time, and the walk through latent space measured in beats |
| `presets.py` | the rules connecting what the drums do to what the picture does |
| `strip.py`, `window.py`, `frame.py` | the sliders, the window, and what crosses the bus each frame |
| `tools/`, `cli.py` | one module per `ganlive` command |

Imports only ever point down this stack, from small torch-free modules (`curves`, `clock`,
`files`) through `models/` and `dials/` up to `tools/`; `tests/test_layout.py` enforces it.

## Development

```bash
uv pip install -e ".[all,dev]"
GANLIVE_DEVICE=cpu python -m pytest tests -q    # runs on the CPU, no hardware needed
ruff check src tests scripts
```

`python scripts/shoot_strip.py <checkpoint> --out shots/` renders the strip headless in each
mode, which is the quickest way to check a layout change.

## Related work

- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) (Karras et al., 2020), whose weights ganlive plays unchanged.
- [FastGAN](https://arxiv.org/abs/2101.04775) (Liu et al., 2021), the architecture behind the FastGAN family.
- [SeFa](https://arxiv.org/abs/2007.06600) (Shen & Zhou, 2021), closed-form latent directions.
- [Autolume-Live](https://www.metacreation.net/projects/autolume-automating-live-music-visualisation-technical-report)
  (Kraasch & Pasquier, 2022), an audio-reactive StyleGAN VJ system. ganlive differs in taking
  any generator, deriving and measuring its dials, and locking the walk to MIDI clock.

## Licence

MIT. Model weights keep their own licences: NVIDIA's StyleGAN2 models are under the
[NVIDIA Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt),
and models fetched with `adopt` under whatever their authors chose.
