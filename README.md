# ganlive

**Explore GAN latent spaces in real time.**

Load a trained GAN and change its generated images with your mouse or MIDI controller.
ganlive automatically creates dials for latent directions and model controls, keeping those
that produce a visible change.

<p align="center">
  <img src="docs/demo.webp" alt="Exploring GAN images with dials, MIDI routing, and live model switching" width="860">
</p>

- **Automatic dials.** Discover controls from the model, without writing a configuration.
- **Mouse or MIDI.** Turn dials by hand, map controller knobs, or use notes to drive movement.
- **Live visuals.** Switch models, follow MIDI clock, and save images or videos.

## Quick start

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Runs on Windows, macOS, and Linux with NVIDIA, AMD, Intel, or Apple GPUs through supported
backends; CPU fallback is available. Frame rate depends on the model and hardware.

### 1. Install

```bash
git clone https://github.com/gustavecortal/ganlive.git
cd ganlive
uv venv --python 3.12
```

Install the [PyTorch build for your hardware](https://pytorch.org/get-started/locally/)
using `uv pip install`, then install ganlive:

```bash
uv pip install -e ".[record]"
```

See the [installation reference](docs/guide.md#install) for GPU builds and optional features.

### 2. Load a model

This example uses NVIDIA's pretrained StyleGAN2 face model:

```bash
git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch
curl -L -o ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
uv run --no-sync ganlive import-stylegan2 ffhq.pkl --repo stylegan2-ada-pytorch
```

### 3. Explore

```bash
uv run --no-sync ganlive play --checkpoint runs/stylegan2/ffhq.pt --console --no-audio --no-midi
```

Drag the dials beside the image to explore. Right-click a dial to release it.
The first launch may take a minute or two to compile the model.

## Controls

| Action | Control |
|---|---|
| Adjust a dial | Drag or scroll |
| Connect a MIDI knob | Touch a dial, press `l`, then turn the knob |
| Route MIDI notes to dials | Press `g` to open the routing grid |
| Choose a model | `m` |
| Change preset | `Tab` |
| Save image / record video | `c` / `v` |
| Quit | `Esc` |

For MIDI, remove `--no-midi` from the launch command. Map any note numbers to named tracks,
then connect those tracks to dials in the routing grid:

```bash
uv run --no-sync ganlive play --checkpoint runs/stylegan2/ffhq.pt --console --no-audio --notes 36=BD,38=SD,42=CH
```

MIDI clock synchronizes movement to your device's tempo.
[More controls and hardware setup →](docs/guide.md#playing-with-hardware)

## Bring your own GAN

| Model | How to load it |
|---|---|
| **StyleGAN2** | Convert NVIDIA StyleGAN2-ADA `.pkl` weights with `ganlive import-stylegan2` |
| **FastGAN** | Load a checkpoint trained with [smallgen](https://github.com/gustavecortal/smallgen), or try the [published model](https://huggingface.co/gustavecortal/ganlive-fastgan-3072) |
| **Other GANs** | Import a compatible ONNX generator with `ganlive adopt model.onnx`, or a supported Hub repository with `ganlive adopt hf:owner/repo` |

ONNX import requires the `onnx` extra; Hub import also requires `hub`.
Compatibility depends on the model's inputs and supported operations. StyleGAN3 is not supported.

[Model import and examples →](docs/guide.md#models)

## Learn more

[User guide](docs/guide.md) · [How dials are found](docs/guide.md#how-the-dials-are-found) ·
[Performance](docs/guide.md#speed) · [Development](docs/guide.md#development) ·
[Related work](docs/guide.md#related-work)

## License

[MIT](LICENSE). Model weights retain their original licenses.
