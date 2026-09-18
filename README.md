# ganlive

Explore a GAN's latent space in real time, on one GPU, with a MIDI controller.

The generator runs live at 60 fps. Its knobs are **derived from the weights at load** — no
per-architecture config file, no hand-written dial list — and every one is driven to both ends and
measured before it is offered, so a dial that does nothing is never shown.

- **Any accelerator.** CUDA, Intel XPU, Apple MPS, or the CPU. Detected, not configured.
- **Any GAN.** This project's FastGAN, NVIDIA's StyleGAN2 weights unchanged, or any ONNX graph.
- **Any controller.** Anything that sends MIDI notes and CCs. `l` on a dial, turn a knob, bound.

## Install

    uv venv
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv/bin/python -e ".[audio,record,onnx]"

Swap `cu128` for your build — `xpu`, `rocm6.3`, or drop `--index-url` entirely for CPU and
Apple silicon. The extras are optional: `audio` for audio-driven triggers, `record` for video,
`onnx` for the third model family (`onnx-intel` instead, on an Intel GPU), `hub` for `adopt`.

## Play

    ganlive play --checkpoint runs/my-model/checkpoints/0072000.pt --console

It picks the accelerator, the MIDI ports and an audio input on its own. `--checkpoint` is
repeatable and **any model can join any bank**: a StyleGAN2 at 1024×1024 with a 512-wide latent
plays beside a FastGAN at 3072×2048 with a 256-wide one, switched with `[` and `]` on the beat.

    ganlive                     # the command list
    ganlive wire                # is the controller talking?
    ganlive latency --checkpoint ...   # what this machine does, drift included
    ganlive adopt hf:someone/some-gan  # dial an unseen model and save it playable

## The dials

Two blocks never change, whatever is loaded, so muscle memory survives a model switch:

| | |
|---|---|
| `MOTION` | `speed` `spread` `hold` `late` `grid` — where the walk goes, in beats |
| `LATENT` | `dir1`…`dir8` — principal directions, from the SVD of the first weight that eats `z` |
| `MODEL` | this architecture's own gates, discovered and verified at load |

`MODEL` is the only per-model block. Its dials are found by walking the graph, driven to both
ends, and kept only if they move the picture more than a random direction of the same length
does — the floor is `--direction-floor`.

## Wiring a controller

The twelve tracks are the vocabulary: `BD SD RS CP BT LT MT HT CH OH CY CB`, named or indexed
0–11. Tell it which note is which pad:

    --notes 36=BD,38=SD,42=CH,46=OH      # a General MIDI kit
    --notes 0                            # pads sending consecutive notes from 0

Knobs are learned (`l` on a dial, then turn) or declared: `--cc "16=noise,2:17=se_256,n1.3=dir1"`
— a CC number, or an NRPN as `n<msb>.<lsb>`, which is 14-bit against a CC's 7.

An **Analog Rytm** is the machine this was built against and its defaults suit it: pads on notes
0–11, stems over Overbridge on ASIO. `ganlive wire --drive` starts its sequencer and certifies
the notes against audio. Nothing else assumes it.

## Speed

On the reference machine — Intel Arc A770, torch 2.13+xpu — a FastGAN at 3072×2048 shown at
1620×1080 runs **9.4 ms a frame as played**, window open, strip drawn, sound thread running, and
flat over a run. FFHQ-1024 StyleGAN2 is 12.25 ms generating.

**Only that machine has been measured.** `ganlive latency` reports yours in a few minutes, and
prints a drift line beside the median, because a median hides a slope: a graph replay here was
gaining 0.066 ms every second and every 15-second median read fine.

The forward is compiled and then recorded as one device graph where the backend allows (XPU,
CUDA); elsewhere it runs eager and says so. The capture is checked against the forward at every
load and is exact to 0.0000 8-bit levels.

## Getting models in

    ganlive import-stylegan2 ffhq.pkl --repo stylegan2-ada-pytorch
    ganlive export-onnx --checkpoint runs/my-model/checkpoints/0072000.pt

    ganlive adopt hf:someone/some-gan --trust-remote-code

StyleGAN2 is read through this project's own synthesis network, which loads NVIDIA's weights
unchanged and compiles as one graph — their CUDA kernels are not needed and do not build on
non-NVIDIA hardware. **StyleGAN3 is not supported**: it needs `affine_grid_generator`, which
ONNX does not have.

`adopt` is the general path: it fetches a generator, exports it, finds its dials, measures
what each is worth, and writes all of that into the graph — so the instrument that opens it
afterwards knows nothing about the architecture. It refuses rather than guesses in three
places: code it will not import unasked, a module that never produced a picture, and a graph
still non-deterministic after its random draws are frozen. Needs `.[hub]`.

FastGAN checkpoints come from [smallgen](https://github.com/gustavecortal/smallgen), which is
the training half of this project.

## Layout

| | |
|---|---|
| `models/` | the three families, and the graph capture that makes them fast |
| `dials/` | where dials come from: SVD on the weights, the verification gate, the table |
| `control/` | MIDI in, audio in, and the track vocabulary both speak |
| `bank.py` | turning a checkpoint into something playable, and switching between several |
| `walk.py` | the latent walk, measured in beats rather than frames |
| `strip.py` | the sliders beside the picture |
| `tools/` | one module per `ganlive` subcommand |

Tests run on the CPU and need no hardware:

    GANLIVE_DEVICE=cpu python -m pytest tests -q

## Licence

MIT.
