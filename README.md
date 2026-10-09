# ganlive

[ganlive](https://gustavecortal.com/ganlive/) explores a GAN's latent space in real time, with dials derived automatically and played by hand, with MIDI, or from drums. [Models trained on my photographs](https://huggingface.co/collections/gustavecortal/ganlive-6ac56a61006a78b3e1ff5406) run at over 100 fps on a $350 GPU.

<p align="center">
  <a href="docs/demo.mp4"><img src="docs/demo.webp" alt="Real-time GAN latent-space exploration with automatically derived dials: ganlive-lichen, a FastGAN trained on my photographs, with a dial turned by hand and drums wired to dials in the routing grid" width="860"></a>
</p>

## Play in your browser

Open [the player](https://gustavecortal.com/ganlive/) in a browser with WebGPU, such as a
current Chrome, Edge, Firefox or Safari. Nothing to install: the model downloads once and runs
on your graphics card, whatever its make.

Drag a dial to hold it, and let go to return it to the preset. To play from a drum machine or a
controller, open the Drums tab and press Connect MIDI (Chrome, Edge and Firefox support Web
MIDI). Its clock sets the tempo, and the routing grid sends each drum to the dials it pushes.

Record saves an MP4 of the picture at your screen's size, at 60, 30 or 20 fps, whichever your
card keeps up with. Its file name gives the first and last drum hit in seconds of the video, to
line up the audio in an editor.

## Bring your own GAN

Convert a FastGAN or StyleGAN2 checkpoint once, with Python 3.10–3.13 and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then upload it to the
Hugging Face Hub:

```bash
git clone https://github.com/gustavecortal/ganlive.git
cd ganlive
uv venv --python 3.12
uv pip install -e ".[convert]" huggingface_hub
uv run --no-sync ganlive convert path/to/checkpoint.pt
uv run --no-sync hf auth login
uv run --no-sync hf upload <you>/<model> runs/engine/<name> engine
```

Then play it at `https://gustavecortal.com/ganlive/?model=https://huggingface.co/<you>/<model>`.
Converting measures the model's dials, and NVIDIA's StyleGAN2-ADA `.pkl` files are
[imported](docs/guide.md#stylegan2) first.

## Learn more

- The [user guide](docs/guide.md) covers the desktop app, which plays models from your disk,
  reacts to audio input, and records with a guide track.
- [How the dials are found](docs/guide.md#how-the-dials-are-found)
- [Development](docs/guide.md#development) and [related work](docs/guide.md#related-work)
- [Citation](CITATION.cff)

## License

[MIT](LICENSE). Model weights retain their original licenses.
