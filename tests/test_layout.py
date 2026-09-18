"""Facts about the package itself: that it imports, and that its commands answer."""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ganlive"


def test_every_ganlive_import_resolves_including_the_lazy_ones():
    """Most imports here are inside functions, to keep start-up off the first bar.

    A stale one then survives every import check and fails at load, which is exactly
    how three of them reached the end of a package rename."""
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
    """`ganlive <command> --help` is the only contract a new user meets first."""
    from ganlive.cli import COMMANDS

    assert len(COMMANDS) >= 5, COMMANDS
    env = {**os.environ, "GANLIVE_DEVICE": "cpu"}
    for name, (module, _) in COMMANDS.items():
        source = SRC / "tools" / f"{module}.py"
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        done = subprocess.run([sys.executable, "-m", "ganlive.cli", name, "--help"],
                              capture_output=True, text=True, timeout=180, env=env)
        assert done.returncode == 0, f"{name} --help failed: {done.stderr[-2000:]}"


def test_an_unknown_command_is_refused_with_the_list():
    done = subprocess.run([sys.executable, "-m", "ganlive.cli", "nope"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2
    assert "no command 'nope'" in done.stderr and "play" in done.stderr


def test_nothing_points_at_a_file_or_tool_this_repository_does_not_have():
    """Error messages and help text send people places. Those places must exist.

    The split out of `smallgen` left nine `scripts/rytm_*.py` paths in text a user reads
    only when something has already gone wrong -- and seven dead `sys.path.insert` calls in
    the suite, which this missed for looking only at `src/`."""
    gone = ("scripts/", "rytm_live", "rytm_map", "rytm_preflight", "rytm_sysex",
            "rytm_wire", "rytm_fuzz", "realtime_video", "onnx_bench", "NOTES.md",
            "PLAN.md", "smallgen", ".venv/Scripts", "\\Scripts", "in NOTES", "See NOTES")
    #: The one place the old name is deliberate: graphs and checkpoints written before the
    #: rename still carry it, and both readers accept either spelling.
    allowed = "before this project was named"
    found = []
    here = pathlib.Path(__file__).resolve().parent
    # not this file: it names every stale spelling in order to look for them
    for path in sorted(SRC.rglob("*.py")) + sorted(
            p for p in here.glob("*.py") if p.name != pathlib.Path(__file__).name):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if allowed in "\n".join(lines[max(0, number - 4):number + 1]):
                continue
            found += [f"{path.name}:{number}: {line.strip()[:90]}"
                      for stale in gone if stale in line]
    assert not found, "\n".join(found)
