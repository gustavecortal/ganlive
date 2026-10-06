"""Model files on disk: what a path means, what a checkpoint is called, and which may join a bank.

Torch-free, so a tool can name and find checkpoints without loading a generator.
"""
from __future__ import annotations

from pathlib import Path

#: The one suffix that means "an exported graph, not a checkpoint".
ONNX = ".onnx"
#: The suffix of the derived directions saved beside a checkpoint (`dials.derive.cache_path`).
CACHE_SUFFIX = ".directions.pt"


def is_onnx(path) -> bool:
    return Path(path).suffix.lower() == ONNX


def run_step(path) -> tuple[str, str]:
    """A checkpoint's identity as the pair `(run, step)`. A file named after its run, as a
    published model is (`lichen/lichen.pt`), has no step: its run says everything."""
    path = Path(path)
    if is_onnx(path):
        # Exports sit in one flat folder as `<run>-<step>.onnx`, so the parent says nothing;
        # a graph with no step in its name (`lichen.onnx`, an adopted one) is its stem.
        # `onnx` stays in the step so an export and its checkpoint never share a name.
        run, _dash, step = path.stem.rpartition("-")
        if run and step.isdigit():
            return run, f"{int(step)} onnx"
        return path.stem, "onnx"
    run = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    if path.stem == run:
        return run, ""
    return run, path.stem.lstrip("0") or path.stem


def label_for(path) -> str:
    """A checkpoint's short name, as `run step`: `my-run 72000`."""
    return " ".join(part for part in run_step(path) if part)


def slug_for(path) -> str:
    """The same identity as a filename and a JSON key: `my-run-72000`."""
    return "-".join(part for part in run_step(path) if part)


def index_of(models, path) -> int | None:
    """Where a checkpoint sits in a list of models, or None."""
    path = Path(path).resolve()
    return next((i for i, m in enumerate(models) if m.path.resolve() == path), None)


def admit(models, path) -> None:
    """Raise `ValueError` if this checkpoint may not join these models. Only a duplicate is
    refused: models in one bank may differ in latent width, size and aspect."""
    if index_of(models, path) is not None:
        raise ValueError(f"{label_for(path)} is already in this bank")


def checkpoints_in(folder) -> list[Path]:
    """Every checkpoint in one folder, sorted by name, leaving out the dial caches beside them."""
    return sorted(p for p in Path(folder).glob("*.pt") if not p.name.endswith(CACHE_SUFFIX))


def checkpoint_for(target) -> Path:
    """The one model a path means: a file; a run, whose last checkpoint by name is wanted; or a
    folder of exported graphs, whose last by name is wanted the same way."""
    target = Path(target)
    if target.is_file():
        return target
    inner = target / "checkpoints"
    folder = inner if inner.is_dir() else target
    found = checkpoints_in(folder) or sorted(folder.glob(f"*{ONNX}"))
    if not found:
        raise FileNotFoundError(f"no checkpoint or exported graph at {target}")
    return found[-1]
