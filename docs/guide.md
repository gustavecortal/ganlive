# ganlive user guide

[Back to the README](../README.md)

Load a model to get dials for its latent space. Use them with your mouse or MIDI controller.

[Models](#models) · [Controls](#the-interface) · [MIDI and audio](#playing-with-hardware) ·
[Recording](#recording) · [Performance](#speed) · [Installation](#install) ·
[Development](#development)

Commands assume an active project environment. Otherwise, prefix `ganlive` with
`uv run --no-sync` from the project directory.

## Models

Pass a checkpoint or run directory with `--checkpoint`.
Repeat the option to load several models, then switch with `[` and `]`.
Models can have different resolutions and latent dimensions.

### StyleGAN2

Convert NVIDIA StyleGAN2-ADA weights once, using a checkout of its repository.
Install the dependencies needed to read NVIDIA's model file:

```bash
uv pip install requests click setuptools
ganlive import-stylegan2 model.pkl --repo stylegan2-ada-pytorch
ganlive play --checkpoint runs/stylegan2/model.pt --console
```

See the [README](../README.md#quick-start) for a complete example.
The converted model runs without NVIDIA's custom CUDA kernels.

### FastGAN

Load a checkpoint trained with [smallgen](https://github.com/gustavecortal/smallgen),
or download a published one, [lichen 3072](https://huggingface.co/gustavecortal/ganlive-lichen-3072)
or the lighter [amber 1536](https://huggingface.co/gustavecortal/ganlive-amber-1536):

```bash
uv pip install huggingface_hub
hf download gustavecortal/ganlive-lichen-3072 --local-dir runs/lichen-3072
ganlive play --checkpoint runs/lichen-3072 --console
```

### Other generators

Install the `onnx` extra and import a compatible ONNX generator:

```bash
ganlive adopt model.onnx --out runs/onnx/model.onnx
ganlive play --checkpoint runs/onnx/model.onnx --console
```

For a Hugging Face repository, install `hub` and use:

```bash
ganlive adopt hf:owner/repo --out runs/onnx/model.onnx
```

If the repository requires its own Python code, add `--trust-remote-code` only for code
you trust. Import needs a generator that accepts a latent and returns an image.
Models with extra inputs or unsupported operations may need a custom ONNX export.

Import creates and calibrates controls, then saves them with the graph for later use.

### Additional tools

| Command | Use |
|---|---|
| `ganlive dials runs/stylegan2/model.pt` | Find directions using the whole generator and save them for later launches |
| `ganlive export-onnx --checkpoint runs/my-run` | Export a FastGAN checkpoint with its controls |

## The interface

Launch with `--console` to show dials beside the image. Drag or scroll to adjust a dial.
Right-click to return it to the preset and MIDI control.

| Group | Controls |
|---|---|
| `MASTER` | Strength of the response to each hit |
| `MOTION` | Speed, distance, and timing of movement |
| `LATENT` | Directions through the model's latent space |
| `MODEL` | Model-specific controls such as noise and truncation |

Motion dials control how the image moves:

| Dial | Effect |
|---|---|
| `speed` | Time between moves, from 8 beats to half a beat |
| `spread` | Distance travelled per move |
| `hold` | Time spent still during a move |
| `late` | Leave on the beat or arrive on the next beat |
| `grid` | Smooth movement or steps on subdivisions of the beat |

Dark dials are inactive for the current model.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `g` | Open the MIDI routing grid |
| `l` | Map the next MIDI knob to the last dial touched |
| `[` / `]` | Previous / next loaded model |
| `m` | Browse models under `--runs` |
| `Tab` / `Shift+Tab` | Next / previous preset |
| `s` | Save a preset |
| `r` | Release all mouse-controlled dials |
| `v` | Start / stop video recording |
| `c` | Save an image |
| `n` | Show native pixels, with arrow keys to pan |
| `f` | Toggle fullscreen |
| `Esc` / `q` | Quit |

Without `--console`, the image opens fullscreen with `n`, `f`, arrow keys, and `Esc`.

## Playing with hardware

Connect your MIDI device and launch without `--no-midi` or `--simulate`.
ganlive listens to all MIDI inputs by default. Use `--midi-port` to select one.

### Notes

Map note numbers to tracks, then press `g` to connect tracks to dials.
Click a grid cell to connect it, scroll to adjust the strength, and right-click to reverse it.

```bash
ganlive play --checkpoint runs/stylegan2/ffhq.pt --console --no-audio --notes 36=BD,38=SD,42=CH,46=OH
```

Track names are `BD SD RS CP BT LT MT HT CH OH CY CB`, or indices 0–11.
These are routing labels. You can map any note numbers to them.

For twelve consecutive notes, use `--notes 0`, replacing `0` with the first note.
If tracks use separate MIDI channels, use `--midi-channels "1=BD,2=SD"` instead.

### Knobs and pressure

Touch a dial, press `l`, and turn a MIDI knob. Mappings are saved in
`runs/ganlive/settings/cc.txt`.

You can also pass mappings with `--cc`:

| Mapping | Meaning |
|---|---|
| `16=dir1` | CC 16 controls `dir1` |
| `2:17=dir1` | CC 17 on channel 2 controls `dir1` |
| `n1.3=dir1` | NRPN 1.3 controls `dir1` |

Use `--pressure "BD=dir1"` to control a dial with polyphonic aftertouch.

### Tempo and audio

MIDI clock sets the tempo. MIDI Start and Stop control movement.
Without clock, `--bpm` sets the tempo.

With the `audio` extra, use `--triggers audio` for audio hits or `--triggers both`
for MIDI and audio. Select an input with `--audio-name` or `--audio-device`.
Run `ganlive doctor --learn` to map audio channels to tracks.

To try music-driven visuals without hardware, install the `audio` extra and add
`--simulate --monitor`. This plays the built-in drum pattern through your speakers.

### Troubleshooting

| Command | Check |
|---|---|
| `ganlive doctor` | Available MIDI ports and audio inputs |
| `ganlive doctor --listen` | Incoming notes, clock, and transport |
| `ganlive doctor --meter` | Audio levels by channel |
| `ganlive doctor --learn` | Audio channel mapping |
| `ganlive doctor --drive` | Start a sequencer and check its notes |

Audio defaults target an Elektron Analog Rytm through Overbridge.
Adjust the input and channel mapping for other devices.

## How the dials are found

ganlive derives latent directions from model weights using
[SeFa](https://arxiv.org/abs/2007.06600), then ranks them by their effect on the image.
For StyleGAN2, directions act in its mapped latent space, `w`.

Directions must produce more change than a random direction of the same length.
`--direction-floor` sets this threshold, with a default of 2.
Controls for noise, truncation, and layer gains depend on the model.

Dial ranges are calibrated so that similar movements produce roughly similar visual changes.
Controls with no measured effect are inactive.

## Recording

Press `v` to start or stop recording, or launch with `--record`.
Videos use an available hardware encoder, with software fallback.

Press `c` to save a full-resolution PNG. Files are saved under `runs/ganlive/`.

Recordings follow the display resolution. Use `--height native` to record at the
model's full resolution. When audio input is available, the guide track and beat map
help align video with a DAW recording. Use `--no-guide` to disable them.

## Speed

Measure your hardware with:

```bash
ganlive latency --checkpoint runs/stylegan2/ffhq.pt
```

Reported measurements on a $350 Intel Arc A770 with PyTorch 2.13+xpu:

| Model | Native resolution | Frame time | FPS |
|---|---|---|---|
| StyleGAN2 FFHQ | 1024×1024 | 10.7 ms | 94 |
| FastGAN, displayed at 1620×1080 | 3072×2048 | 8.8 ms | 114 |
| StyleGAN2 FFHQ via ONNX, OpenVINO FP16 | 1024×1024 | 14.3 ms | 70 |

These measure generation and frame preparation. With the window and controls,
the FFHQ loop took 12.4 ms, with 13.8 ms at the 95th percentile.
Results depend on your model, hardware, and runtime.

## Install

Follow the [README quick start](../README.md#quick-start).
Install [PyTorch for your hardware](https://pytorch.org/get-started/locally/) first.

ganlive detects CUDA, Intel XPU, Apple MPS, or CPU.
AMD GPUs use the ROCm PyTorch build on Linux.

Add optional features with `uv pip install -e ".[EXTRA]"`:

| Extra | Feature |
|---|---|
| `audio` | Audio input and recording guide tracks |
| `record` | Video recording |
| `onnx` | ONNX import and playback |
| `onnx-intel` | ONNX through OpenVINO for Intel GPUs |
| `hub` | Hugging Face imports, including ONNX dependencies |
| `all` | All features except `onnx-intel` |
| `dev` | Development dependencies |

Combine extras as `".[audio,record]"`.
Run `ganlive <command> --help` for all options.

## Development

```bash
uv pip install -e ".[dev]"
GANLIVE_DEVICE=cpu python -m pytest tests -q
ruff check src tests scripts
```

Render the controls without a window:

```bash
python scripts/shoot_strip.py <checkpoint> --out shots/
```

Source code is under `src/ganlive/`:

| Path | Purpose |
|---|---|
| `models/`, `dials/` | Model loading and automatic controls |
| `control/` | MIDI, audio, and simulated drums |
| `record/` | Video, images, and recording alignment |
| `strip.py`, `window.py` | Interface |
| `tools/`, `cli.py` | Commands |

## Related work

- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), Karras et al., 2020.
- [FastGAN](https://arxiv.org/abs/2101.04775), Liu et al., 2021.
- [SeFa](https://arxiv.org/abs/2007.06600), Shen and Zhou, 2021.
- [Autolume-Live](https://www.metacreation.net/projects/autolume-automating-live-music-visualisation-technical-report),
  Kraasch and Pasquier, 2022.

## License

[MIT](../LICENSE). Model weights retain their original licenses.
