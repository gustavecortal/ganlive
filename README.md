# ganlive

[ganlive](https://gustavecortal.com/ganlive/): real-time latent space exploration of GANs trained on [my photographs](https://huggingface.co/collections/gustavecortal/ganlive-6ac56a61006a78b3e1ff5406), played by hand or by a drum machine over MIDI. This repository converts a GAN for the browser and plays it on the desktop.

<p align="center">
  <a href="https://gustavecortal.com/ganlive/"><img src="docs/demo.webp" width="640" alt="ganlive playing lichen, a FastGAN trained on my photographs: dragging a dial moves the picture along one latent direction"></a>
</p>

## Play in your browser

Open [the player](https://gustavecortal.com/ganlive/) in a current Chrome, Edge, Firefox or Safari. Drag a dial to move the picture along its direction, and double-click it to return it to the preset. To play from a drum machine, open the Drums tab and press Connect MIDI.

## Bring your own GAN

Convert a FastGAN or StyleGAN2 checkpoint once, with Python 3.10 to 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/), then upload it to the Hugging Face Hub:

```bash
git clone https://github.com/gustavecortal/ganlive.git
cd ganlive
uv venv --python 3.12
uv pip install -e ".[convert]" huggingface_hub
uv run --no-sync ganlive convert path/to/checkpoint.pt
uv run --no-sync hf auth login
uv run --no-sync hf upload <you>/<model> runs/engine/<name> engine
```

Then play it at `https://gustavecortal.com/ganlive/?model=https://huggingface.co/<you>/<model>`. To train a FastGAN on your own photographs, see [gantrain](https://github.com/gustavecortal/gantrain).

## How it works

Converting compiles the generator into WebGPU compute shaders with half-precision weights. It also finds the dials: eight directions in the latent space, from [closed-form factorization](https://arxiv.org/abs/2007.06600), kept only if they visibly change the picture. Between dial moves, the latent drifts from one random point to the next, one step per bar. Over MIDI, a drum machine sets the tempo, and each hit pushes the dials routed to it.

The [user guide](docs/guide.md) covers the desktop player and the development setup.

## License

The code is under the [MIT license](LICENSE). Model weights keep their original licenses.
