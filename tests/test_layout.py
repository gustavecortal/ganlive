"""Facts about the package itself: that it imports, that its layers hold, and that its commands
answer."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import pathlib
import re
import subprocess
import sys

from ganlive import cli

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ganlive"
REPO = SRC.parents[1]

#: Each module's layer. A module may import from its own layer or a lower one, never a higher
#: one, so a reader can learn the package bottom up. A package's entry covers its modules.
LAYERS = {
    # Torch-free basics, and the device calls everything else asks.
    "ganlive": 0, "ganlive.curves": 0, "ganlive.clock": 0, "ganlive.files": 0,
    "ganlive.timing": 0, "ganlive.device": 0, "ganlive.checkpoints": 0,
    # Frames off the card, and the settings vector on it.
    "ganlive.pixels": 1, "ganlive.settings": 1,
    "ganlive.models": 2,
    "ganlive.dials": 3,
    # The WGSL engine: shaders generated from a converted model, and the hosts that run them.
    "ganlive.engine": 3,
    # What plays the dials: the walk, the rules, the controllers, the recorders.
    "ganlive.walk": 4, "ganlive.presets": 4, "ganlive.control": 4, "ganlive.record": 4,
    "ganlive.frame": 5, "ganlive.families": 5,
    "ganlive.strip": 6, "ganlive.window": 6,
    "ganlive.bank": 7,
    "ganlive.tools": 8, "ganlive.cli": 8,
}

#: Optional or heavy third-party packages that may be imported inside a function, so that a
#: machine without them still runs everything that does not need them.
DEFERRABLE = {"onnx", "onnxruntime", "openvino", "sounddevice", "av", "huggingface_hub",
              "psutil", "pygame"}

#: Every other import made inside a function, and why it is not at the top.
DEFERRED = {
    # Loading the compiler costs start-up time; only a compile needs it.
    ("ganlive.frame", "torch._dynamo.utils"): "torch",
    ("ganlive.models.capture", "torch._dynamo.utils"): "torch",
    # `play` answers `--help` and a bad argument without loading torch.
    ("ganlive.tools.play", "torch"): "torch",
    ("ganlive.tools.play", "ganlive.bank"): "torch",
    ("ganlive.tools.play", "ganlive.device"): "torch",
    ("ganlive.tools.play", "ganlive.families"): "torch",
    # Recording is torch-free until a take actually starts.
    ("ganlive.record.video", "ganlive.pixels"): "torch",
    # The converted-StyleGAN2 file opens with no other module of this package.
    ("ganlive.models.stylegan2", "ganlive.models.common"): "self-contained",
    # NVIDIA's own code, from the checkout named on the command line.
    ("ganlive.tools.import_stylegan2", "dnnlib"): "on --repo",
    ("ganlive.tools.import_stylegan2", "legacy"): "on --repo",
}

#: Modules that must import without torch: the hardware checks run before any model does.
TORCH_FREE = ("ganlive.clock", "ganlive.control.midi", "ganlive.tools.doctor", "ganlive.tools.play")


def _modules():
    """`(name, path, tree)` for every module in the package."""
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        yield name, path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(name: str, path: pathlib.Path, tree):
    """`(lineno, imported module, deferred)` for every import in a module, nested ones too.

    `from ganlive.dials import derive` names the module `ganlive.dials.derive`."""
    top = {id(n) for stmt in tree.body if not isinstance(stmt, (ast.FunctionDef,
                                                               ast.AsyncFunctionDef,
                                                               ast.ClassDef))
           for n in ast.walk(stmt)}
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")[:len(package.split(".")) - node.level + 1]
                base = ".".join(parts + ([base] if base else []))
            targets = [f"{base}.{a.name}" if base.startswith("ganlive")
                       and _is_module(f"{base}.{a.name}") else base for a in node.names]
        else:
            continue
        for target in dict.fromkeys(targets):
            yield node.lineno, target, id(node) not in top


def _is_module(dotted: str) -> bool:
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError):
        return False


def _layer(module: str) -> int | None:
    while module:
        if module in LAYERS:
            return LAYERS[module]
        module = module.rpartition(".")[0]
    return None


def test_every_module_has_a_layer():
    assert not [name for name, _p, _t in _modules() if _layer(name) is None]


def test_no_module_imports_from_a_layer_above_its_own():
    """Deferred imports count: a function-level import is still a dependency."""
    upward = []
    for name, path, tree in _modules():
        mine = _layer(name)
        for line, target, _deferred in _imports(name, path, tree):
            theirs = _layer(target) if target.startswith("ganlive") else None
            if theirs is not None and theirs > mine:
                upward.append(f"{name}:{line} (layer {mine}) imports {target} (layer {theirs})")
    assert not upward, "\n".join(upward)


def test_imports_inside_functions_are_only_the_listed_ones():
    """An import moved into a function hides a dependency and its failure until the call.

    Allowed for optional packages, and for the few listed in `DEFERRED`."""
    stray, used = [], set()
    for name, path, tree in _modules():
        for line, target, deferred in _imports(name, path, tree):
            if not deferred or target.split(".")[0] in DEFERRABLE:
                continue
            if (name, target) in DEFERRED:
                used.add((name, target))
            else:
                stray.append(f"{name}:{line} imports {target} inside a function")
    assert not stray, "\n".join(stray)
    assert not set(DEFERRED) - used, f"listed but not deferred: {set(DEFERRED) - used}"


def test_the_hardware_checks_import_without_torch():
    """In a fresh interpreter, since this one has torch loaded already."""
    script = ("import importlib, sys\n"
              f"for name in {TORCH_FREE!r}:\n"
              "    importlib.import_module(name)\n"
              "    if 'torch' in sys.modules:\n"
              "        sys.exit(name + ' imports torch')\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_every_ganlive_import_resolves_including_the_deferred_ones():
    """A deferred import is only executed when its function runs, so a stale one passes
    every other test until then."""
    missing = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("ganlive"):
                continue
            where = f"{path.relative_to(SRC)}:{node.lineno}"
            try:
                module = importlib.import_module(node.module)
            except Exception as exc:  # noqa: BLE001 -- the failure is the finding
                missing.append(f"{where} {node.module}: {exc}")
                continue
            for alias in node.names:
                if hasattr(module, alias.name):
                    continue
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except Exception:  # noqa: BLE001
                    missing.append(f"{where} {node.module}.{alias.name}")
    assert not missing, "\n".join(missing)


def test_every_command_parses_and_answers_help():
    """`ganlive <command> --help` is the first thing a new user runs. One interpreter for all
    of them: each would otherwise pay for its own torch import."""
    script = ("from ganlive import cli\n"
              f"for name in {list(cli.COMMANDS)!r}:\n"
              "    try:\n"
              "        cli.main([name, '--help'])\n"
              "    except SystemExit as exc:\n"
              "        if exc.code not in (0, None):\n"
              "            raise SystemExit(f'{name} --help exited {exc.code}')\n"
              "    else:\n"
              "        raise SystemExit(f'{name} --help did not stop at the help')\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=180, env={**os.environ, "GANLIVE_DEVICE": "cpu"})
    assert done.returncode == 0, done.stderr[-2000:]
    for name in cli.COMMANDS:
        assert f"ganlive {name}" in done.stdout, f"{name} printed no usage"


def test_an_unknown_command_is_refused_with_the_list(capsys):
    assert cli.main(["nope"]) == 2
    err = capsys.readouterr().err
    assert "no command 'nope'" in err and "play" in err


def test_every_file_the_text_points_at_exists():
    """Help text and error messages send people to files; those files must be here. Paths
    into a virtualenv are machine-specific and point nowhere on anyone else's."""
    named = re.compile(r"\b(scripts/[\w.-]+\.py|[\w-]+\.md)\b")
    found = []
    # Not this file, which spells out what it looks for.
    tests = (p for p in REPO.joinpath("tests").glob("*.py") if p.name != pathlib.Path(__file__).name)
    for path in sorted(SRC.rglob("*.py")) + sorted(tests):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            where = f"{path.name}:{number}"
            found += [f"{where}: {m}" for m in named.findall(line) if not (REPO / m).exists()]
            if ".venv/" in line or "\\Scripts\\" in line:
                found.append(f"{where}: a virtualenv path")
    assert not found, "\n".join(found)


def _ganlive_names(tree, module_name: str) -> dict[str, str]:
    """What each bare name in this file refers to, for the names that come from `ganlive`.

    `from ganlive import bank` and `from ganlive.strip import DialPanel` both land here, as
    `bank -> ganlive.bank` and `DialPanel -> ganlive.strip.DialPanel`. Relative imports are
    resolved against the file's own package."""
    out: dict[str, str] = {}
    package = module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ganlive"):
                    out[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:                        # `from ..models import x`
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
            if not base.startswith("ganlive"):
                continue
            for alias in node.names:
                out[alias.asname or alias.name] = f"{base}.{alias.name}"
    return out


def _resolve(node, names: dict[str, str]) -> str | None:
    """The dotted `ganlive` path a call target or attribute names, if it names one."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in names:
        return None
    return ".".join([names[node.id], *reversed(parts)])


def _lookup(dotted: str):
    """The object a dotted path names, or None if nothing of that name exists."""
    try:
        return importlib.import_module(dotted)
    except ImportError:
        pass
    module, _dot, attr = dotted.rpartition(".")
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        return None


def _calls_and_attributes():
    """Every call and every attribute reference in `src/` that names something in `ganlive`."""
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC.parent)
        module_name = ".".join(rel.with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = _ganlive_names(tree, module_name)
        if not names:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield path, node, _resolve(node.func, names), True
            elif isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Store):
                yield path, node, _resolve(node, names), False


def test_every_keyword_argument_names_a_parameter_that_exists():
    """A renamed parameter whose callers were not renamed with it raises `TypeError` only when
    the call runs, and `--help` never reaches the body that makes it."""
    wrong = []
    for path, node, dotted, is_call in _calls_and_attributes():
        if not is_call or dotted is None or not node.keywords:
            continue
        target = _lookup(dotted)
        if target is None or not callable(target):
            continue
        try:
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in params:
                wrong.append(f"{path.relative_to(SRC)}:{node.lineno} "
                             f"{dotted}({keyword.arg}=...) -- it takes "
                             f"{', '.join(params)}")
    assert not wrong, "keyword arguments that name no parameter:\n  " + "\n  ".join(wrong)


def test_every_attribute_reached_through_a_module_exists():
    """`module.name` for a name the module no longer has fails only when that line runs."""
    missing = []
    for path, node, dotted, is_call in _calls_and_attributes():
        if is_call or dotted is None:
            continue
        module, _dot, attr = dotted.rpartition(".")
        try:
            owner = importlib.import_module(module)
        except ImportError:
            continue                    # not a module path; the import test covers those
        if not hasattr(owner, attr):
            missing.append(f"{path.relative_to(SRC)}:{node.lineno} {dotted}")
    assert not missing, "attributes that do not exist:\n  " + "\n  ".join(sorted(set(missing)))
