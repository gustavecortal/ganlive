"""Getting a generator from somewhere else, and finding out what it is by asking it."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

#: Latent widths to try when a config does not say. In the order they are common; the probe
#: stops at the first that produces a picture, and a generator that takes none of them is
#: refused by name rather than guessed at.
WIDTHS = (512, 256, 128, 100, 64, 32)

#: Keys a repo's `config.json` uses for the latent width. Read first, probed second.
WIDTH_KEYS = ("z_dim", "latent_dim", "nz", "noise_dim", "latent_size", "z_dimension")

TRUST = ("this repository ships its own model code, and loading it means importing and "
         "running Python written by somebody else on your machine. Pass "
         "--trust-remote-code if that is what you want.")


@dataclass
class Fetched:
    """A generator from elsewhere, and what probing it established."""

    net: object
    nz: int
    size: tuple[int, int]
    source: str
    which: str

    def report(self) -> str:
        params = sum(p.numel() for p in self.net.parameters()) / 1e6
        return (f"{self.source}: {self.which}, {params:.1f} M parameters, latent {self.nz}, "
                f"{self.size[1]}x{self.size[0]}")


# No `token` parameter anywhere here. It was threaded through all five of these functions,
# twelve mentions, and no caller in the repo ever passed one -- `tools/adopt.py` has no flag
# for it. `huggingface_hub` reads `HF_TOKEN` from the environment itself, so a private repo
# still works; what the parameter bought was a ternary and five wider signatures.
def files_of(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo)


def graph_in(repo: str) -> str | None:
    """An ONNX file already in the repo, if there is one. Then none of the rest is needed."""
    return next((f for f in files_of(repo) if f.endswith(".onnx")), None)


def _config(repo: str) -> dict:
    import json

    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, "config.json")
    except Exception:                                    # noqa: BLE001  no config is normal
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _import_code(repo: str) -> list[object]:
    """Every Python module the repo ships, imported. This is the part that trusts."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, allow_patterns=["*.py", "*.json"]))
    names = sorted(f.stem for f in root.glob("*.py"))
    if not names:
        return []

    shadowed = {n: sys.modules[n] for n in names if n in sys.modules}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    out, mine = [], []
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            mine.append(name)
            try:
                spec.loader.exec_module(module)
                out.append(module)
            except Exception as exc:              # noqa: BLE001  a helper may not import here
                print(f"  {name}.py: {type(exc).__name__}: {exc}", flush=True)
    finally:
        sys.path.remove(str(root))
        for name in mine:
            sys.modules.pop(name, None)
        sys.modules.update(shadowed)
    return out


def _candidates(modules) -> list[type]:
    """Classes that could be a generator: a module that knows how to load its own weights."""
    from torch import nn

    seen: dict[str, type] = {}
    for module in modules:
        for name, obj in vars(module).items():
            if (isinstance(obj, type) and issubclass(obj, nn.Module)
                    and hasattr(obj, "from_pretrained") and name not in seen):
                seen[name] = obj
    return list(seen.values())


def _picture(out) -> tuple[int, int] | None:
    """The size of the image a forward returned, or `None` if it did not return one."""
    while isinstance(out, (list, tuple)) and out:
        out = out[0]
    if not isinstance(out, torch.Tensor) or out.dim() != 4 or out.shape[1] not in (1, 3, 4):
        return None
    return (int(out.shape[2]), int(out.shape[3]))


def probe(net, widths=WIDTHS) -> tuple[int, tuple[int, int]] | None:
    """The latent width this module takes and the size it draws, by handing it latents."""
    net.eval()
    for nz in widths:
        try:
            with torch.no_grad():
                size = _picture(net(torch.zeros(1, nz)))
        except Exception:                                # noqa: BLE001  wrong shape, wrong class
            continue
        if size is not None and min(size) >= 8:
            return nz, size
    return None


def from_hub(repo: str, trust: bool = False) -> Fetched:
    """One generator out of a Hub repository, identified by what it does.

    No `revision`: it took one and passed it to none of the three calls that fetch, so
    pinning a commit downloaded the branch tip and said nothing. Wiring it through means
    threading it into `_import_code`, `_config` and each `from_pretrained`, and nothing here
    asks for it yet -- an argument that is not honoured is worse than one that is absent."""
    if not trust:
        raise PermissionError(f"{repo}: {TRUST}")
    modules = _import_code(repo)
    if not modules:
        raise RuntimeError(
            f"{repo} ships no model code, so there is nothing here that knows how to build "
            f"the network its weights belong to. Install the library it came from and export "
            f"to ONNX with that, or find a repository that carries a .onnx.")

    config = _config(repo)
    widths = tuple(dict.fromkeys(
        [int(config[k]) for k in WIDTH_KEYS if isinstance(config.get(k), int)] + list(WIDTHS)))

    found = []
    for cls in _candidates(modules):
        try:
            net = cls.from_pretrained(repo)
        except Exception:                                # noqa: BLE001  most classes are not it
            continue
        got = probe(net, widths)
        if got is not None:
            found.append((cls.__name__, net, got))

    if not found:
        names = ", ".join(c.__name__ for c in _candidates(modules)) or "none"
        raise RuntimeError(
            f"{repo}: nothing in it turned a latent into a picture. Tried {names} at widths "
            f"{widths}. It may be conditional, may take a `w` rather than a `z`, or may need "
            f"arguments this cannot guess.")
    if len(found) > 1:
        # Smallest wins: a wrapper that holds the generator answers too, and so does the
        # generator inside it. The inner one is the model; the outer one is the trainer.
        found.sort(key=lambda f: sum(p.numel() for p in f[1].parameters()))
    which, net, (nz, size) = found[0]
    return Fetched(net=net.eval(), nz=nz, size=size, source=repo, which=which)


def export(fetched: Fetched, out: Path, opset: int = 18) -> Path:
    """The fetched generator as an ONNX graph, ready for `adopt`."""
    import contextlib
    import io

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise), \
            torch.no_grad():
        torch.onnx.export(fetched.net, (torch.zeros(1, fetched.nz),), str(out),
                          input_names=["z"], output_names=["image"],
                          opset_version=opset, dynamo=True)
    return out
