"""Turning checkpoints into playable models, the shelf of models on disk, and switching between
them."""

from __future__ import annotations

import argparse
import pathlib
import types

import numpy as np
import pytest
import torch

from ganlive import bank as R
from ganlive.bank import Bank, Shelf, frame_size
from ganlive.checkpoints import admit, checkpoint_for, is_onnx, label_for, run_step, slug_for
from ganlive.clock import WalkConfig
from ganlive.dials import fastgan_dials
from ganlive.dials import table as S
from ganlive.families import FAMILIES, LoadOptions, family_of
from ganlive.frame import FrameStage
from ganlive.models.fastgan import Generator
from ganlive.tools import play as live
from ganlive.window import fit_height, parse_height
from tests.support import (
    _StubModel,
    fastgan_stub_checkpoint,
    stub_cfg,
)


def _fake_run(root, name, step, width, height, nz=256):
    """A run directory holding one config-only checkpoint."""
    folder = root / name / "checkpoints"
    folder.mkdir(parents=True)
    return fastgan_stub_checkpoint(folder / f"{step:07d}.pt", nz=nz, im_size=height,
                                   im_width=width)


def _no_models():
    """Just enough of a bank for a shelf that has loaded nothing."""
    return types.SimpleNamespace(models=[])


def test_an_exported_graph_is_a_model_on_the_shelf_and_names_itself_apart():
    """An export lives in one flat folder as `<run>-<step>.onnx`, so the stem says everything.
    The trailing word keeps it from sharing a row with the checkpoint it came from."""
    graph = pathlib.Path("runs/onnx/gv-warm-lr3-0078000.onnx")
    check = pathlib.Path("runs/gv-warm-lr3/checkpoints/0078000.pt")

    assert is_onnx(graph) and not is_onnx(check)
    assert run_step(graph) == ("gv-warm-lr3", "78000 onnx")
    assert label_for(graph) == "gv-warm-lr3 78000 onnx"
    assert label_for(check) == "gv-warm-lr3 78000"
    assert slug_for(graph) != slug_for(check)


def test_the_shelf_lists_every_graph_but_only_the_newest_checkpoint(tmp_path):
    """A run holds many checkpoints and the newest is the one anybody means, so it is one row.
    Every export is a different model, so that folder is as many rows as it has files."""
    run = tmp_path / "a-run" / "checkpoints"
    run.mkdir(parents=True)
    for step in ("0001000.pt", "0002000.pt"):
        (run / step).write_bytes(b"")
    graphs = tmp_path / "onnx"
    graphs.mkdir()
    for name in ("a-run-0002000.onnx", "b-run-0009000.onnx"):
        (graphs / name).write_bytes(b"")

    names = [p.name for p in Shelf(_no_models(), tmp_path)._models()]
    assert names == ["0002000.pt", "a-run-0002000.onnx", "b-run-0009000.onnx"], names


def test_the_shelf_is_empty_rather_than_a_crash_without_a_runs_directory(tmp_path):
    """Playing a checkpoint that lives anywhere else is the ordinary first run, and nothing
    creates `runs/` until a model is imported into it."""
    assert Shelf(_no_models(), tmp_path / "never-made").entries() == []


def test_a_flat_folder_of_conversions_offers_every_one_and_no_dial_cache(tmp_path):
    """`import-stylegan2` writes each conversion flat into one folder, and a `.directions.pt`
    cache beside a checkpoint is not a model."""
    flat = tmp_path / "stylegan2"
    flat.mkdir()
    for name in ("afhq", "ffhq", "metfaces"):
        fastgan_stub_checkpoint(flat / f"{name}.pt", nz=256, im_size=64, im_width=64)
    (flat / "ffhq.directions.pt").write_bytes(b"not a model")
    found = sorted(p.name for p in Shelf(_no_models(), tmp_path)._models())
    assert found == ["afhq.pt", "ffhq.pt", "metfaces.pt"], found


def test_the_shelf_lists_what_is_on_disk_and_every_readable_model_can_join(tmp_path):
    """The picker's whole content: what is playing, what is a click away, and its shape as a
    note on the row. Only an unreadable file is refused."""
    here = _fake_run(tmp_path, "gv-here", 82000, 1536, 1024)
    _fake_run(tmp_path, "gv-bigger", 72000, 3072, 2048)      # another native size
    _fake_run(tmp_path, "gv-square", 6900, 512, 512)         # 1:1 against 3:2
    _fake_run(tmp_path, "gv-narrow", 1000, 1536, 1024, nz=128)   # another latent width
    (tmp_path / "not-a-run").mkdir()                          # no checkpoint: not listed

    loaded = _StubModel(path=here, name="gv-here 82000", cfg=stub_cfg(256, 1536, 1024))
    bank = Bank(models=[loaded], stage=FrameStage(1024, 1536, device="cpu"), device="cpu")
    shelf = Shelf(bank, tmp_path)

    by_name = {e.name: e for e in shelf.entries()}
    assert set(by_name) == {"gv-here 82000", "gv-bigger 72000", "gv-square 6900",
                            "gv-narrow 1000"}, sorted(by_name)
    assert shelf.entries()[0].loaded, "what is already playing belongs at the top"
    assert shelf.entries() is shelf.entries(), "the listing is scanned once, not per repaint"

    assert not any(e.why for e in shelf.entries()), "nothing readable is refused"
    assert by_name["gv-square 6900"].note == "512x512 z256"
    assert by_name["gv-narrow 1000"].note == "1536x1024 z128"
    assert not by_name["gv-here 82000"].note, "what is loaded is described by its own strip"

    shelf.request(by_name["gv-square 6900"])
    assert shelf.pending == str(by_name["gv-square 6900"].path), (
        "a square model must be loadable into a 3:2 bank")

    shelf.pending = None
    shelf.request(by_name["gv-here 82000"])
    assert shelf.pending is None, "one already loaded is a switch, not a load"
    shelf.request(by_name["gv-bigger 72000"])
    assert shelf.pending == str(by_name["gv-bigger 72000"].path)

    shelf.bank = types.SimpleNamespace(add=lambda p: (_ for _ in ()).throw(ValueError("nope")))
    assert shelf.service() is None
    assert "nope" in shelf.note
    assert shelf.pending is None, "a serviced request is taken, whether or not it worked"


def test_a_checkpoint_path_may_name_a_file_or_a_run(tmp_path):
    """Naming a run asks for its newest checkpoint; naming a file asks for that step. Nothing
    found is an error rather than an empty list, so a tool never starts with no model."""
    run = tmp_path / "gv-test" / "checkpoints"
    run.mkdir(parents=True)
    for step in (2000, 12000, 8000):
        (run / f"{step:07d}.pt").write_bytes(b"")

    assert checkpoint_for(tmp_path / "gv-test") == run / "0012000.pt"
    assert checkpoint_for(run) == run / "0012000.pt"
    assert checkpoint_for(run / "0008000.pt") == run / "0008000.pt"
    assert label_for(run / "0008000.pt") == "gv-test 8000"
    with pytest.raises(FileNotFoundError):
        checkpoint_for(tmp_path / "nothing-here")


def test_a_bank_may_mix_latent_widths_and_aspect_ratios_and_refuses_only_a_duplicate():
    """What two checkpoints must share to play in one window: nothing."""
    def m(path, nz, w, h):
        return types.SimpleNamespace(path=pathlib.Path(path), name=path, cfg=stub_cfg(nz, w, h))

    models = [m("runs/gv/checkpoints/0072000.pt", 256, 3072, 2048)]
    # A square model with another latent width joins a 3:2 bank.
    assert admit(models, pathlib.Path("runs/stylegan2/ffhq.pt")) is None

    with pytest.raises(ValueError, match="already in this bank"):
        admit(models, pathlib.Path("runs/gv/checkpoints/0072000.pt"))

    assert admit([], pathlib.Path("runs/anything.pt")) is None


def test_one_record_answers_every_question_about_a_model_file():
    """The config reader, the loader, the capture gate, the layout and the shelf all ask the
    same `Family`, so a new format is one entry and no two places can disagree."""
    assert [f.name for f in FAMILIES] == ["onnx", "stylegan2", "fastgan"], (
        "order is load-bearing: suffix, then the file's own tag, then whatever is left")
    assert family_of(pathlib.Path("anything.onnx")).name == "onnx"
    assert family_of(pathlib.Path("no-such-file.pt")).name == "fastgan", (
        "the tail takes anything, so this never returns None")

    for family in FAMILIES:
        for field in ("owns", "config_of", "prepare"):
            assert callable(getattr(family, field)), f"{family.name}.{field}"
    assert [f.name for f in FAMILIES if not f.capturable] == ["onnx"], (
        "an ONNX graph runs under its own runtime, so a torch-stream recording holds nothing")


def test_both_stylegan2_layout_builders_offer_the_same_shared_blocks():
    """The swept layout `_prepare_stylegan2` builds and the bare one, before any measurement,
    must offer the same shared dials, or the surface changes under the hand for reasons the
    player cannot see."""
    bare = S.stylegan2()
    # `curves` are values at even spacing, one tuple per dial, as `calibrate.Dial.curve`.
    swept = S.stylegan2(("w_coarse", "noise_32"), (0.5, 0.0),
                        ((0.3, 1.0, 2.0), (0.0, 1.5, 3.0)), (25.0, 25.0))

    shared = [n for n in bare if bare[n].group != "MODEL"]
    assert shared == [n for n in swept if swept[n].group != "MODEL"]
    assert [n for n in bare if bare[n].group == "MODEL"] == []
    assert [n for n in swept if swept[n].group == "MODEL"] == ["w_coarse", "noise_32"], (
        "the calibrated builder must still carry the model's own dials")

    mine = fastgan_dials.fastgan()
    assert [n for n in mine if mine[n].group != "MODEL"] == shared, (
        "the shared blocks are the same on every family")


def test_ganlive_opens_a_fastgan_checkpoint(tmp_path):
    """The whole FastGAN load path, from a `.pt` on disk to a playable `Model`: rebuild,
    freeze the noise, install, measure, lay out."""
    torch.manual_seed(0)          # the dial gate measures the picture; random weights vary
    cfg = dict(nz=16, ngf=8, im_size=256, im_width=None)
    net = Generator(**cfg)
    path = tmp_path / "tiny.pt"
    torch.save({"g_ema": net.state_dict(), "config": cfg}, path)

    model = R._prepare(path, "cpu", torch.float32,
                       LoadOptions(compile_net=False, capture=False, measure_grain=False))
    assert model.cfg.nz == 16 and model.cfg.ladder.height == 256
    assert model.settings.names, "no dials were installed on the generator"
    # A 256-pixel generator has no 512 rungs, so the dials that write them are offered and
    # drawn dark rather than installed. Which of the rest survive is a measurement -- these
    # weights are random -- so the claim here is structural, not a list.
    assert "sle.se_512" not in model.settings.index
    assert "se_512" not in model.dials_live
    assert {"se_64", "se_128", "se_256"} & model.dials_live, "no gate reached the model"
    assert {"reaction", "speed", "spread"} <= model.dials_live, "the shared dials are live"


def test_switching_models_repoints_the_directions_at_the_new_one():
    """The walk is built once, before the loop; the model changes under it."""
    first = np.eye(2, 8, dtype=np.float32)
    second = np.full((2, 8), 0.5, dtype=np.float32)
    models = [_StubModel(rows=first), _StubModel(rows=second)]
    bank = Bank(models=models, stage=FrameStage(64, 96), device="cpu")

    cfg = WalkConfig()
    bank.walk(cfg, dtype=torch.float32)
    assert cfg.directions is first

    bank.use(1)
    assert cfg.directions is second, "the walk still holds the previous model's basis"


def test_a_model_switch_moves_where_the_latent_goes_with_it():
    """A bank can hold both kinds, so the generator says where it takes the latent: an ONNX
    graph and a captured torch graph read it from the host, a compiled module from the card."""
    models = [_StubModel(path=pathlib.Path("runs/a.onnx"),
                         net=types.SimpleNamespace(latent_on_host=True)),
              _StubModel(path=pathlib.Path("runs/b.pt"))]
    bank = Bank(models=models, stage=FrameStage(64, 96), device="cpu")
    cfg = WalkConfig()
    bank.walk(cfg, dtype=torch.float32)

    bank.index = 0
    bank._rewire()
    assert cfg.latent_on_host is True, "a generator that takes the latent on the host is obeyed"

    bank.index = 1
    bank._rewire()
    assert cfg.latent_on_host is False, (
        "switching to a checkpoint left the walk handing a numpy array to a torch generator")


def test_every_model_is_shown_at_its_own_native_size_not_the_banks_smallest():
    """Each model carries its own frame size, and loading another one cannot change it."""
    def cfg(w, h):
        return stub_cfg(0, w, h)

    small, big = cfg(1536, 1024), cfg(3072, 2048)
    assert frame_size(big, 0) == (2048, 3072)
    assert frame_size(small, 0) == (1024, 1536)

    # The screen is the only thing that may reduce one, and it reduces each on its own terms.
    assert frame_size(big, None, (2560, 1440)) == (1440, 2160)
    assert frame_size(small, None, (2560, 1440)) == (1024, 1536)

    # A square model keeps its square in a bank whose other member is 3:2.
    assert frame_size(cfg(1024, 1024), None, (2560, 1440)) == (1024, 1024)


def test_a_switch_moves_the_stage_to_the_incoming_models_size():
    """The stage follows the model, and loading a second one leaves the first alone."""
    class Net(torch.nn.Module):
        def __init__(self, w, h):
            super().__init__()
            self.w, self.h = w, h

        def forward(self, z):
            return [torch.zeros(1, 3, self.h, self.w), torch.zeros(1, 3, self.h // 4,
                                                                   self.w // 4)]

    def model(w, h):
        return _StubModel(net=Net(w, h), path=pathlib.Path("m.pt"), cfg=stub_cfg(8, w, h))

    big, small = model(48, 32), model(24, 16)
    r = Bank(models=[big, small], stage=FrameStage(32, 48, device="cpu"), device="cpu",
             dtype=torch.float32)

    assert (r.height, r.width) == (32, 48)
    assert r.stage.bgra_bytes(r.stage.step(big.net(None))).shape == (32, 48, 4)

    r.use(1)
    assert (r.height, r.width) == (16, 24), "the stage did not follow the switch"
    assert r.stage.bgra_bytes(r.stage.step(small.net(None))).shape == (16, 24, 4)

    r.use(0)
    assert (r.height, r.width) == (32, 48), "switching back left the big model downscaled"


def test_the_frame_report_says_which_model_the_slow_frames_belong_to():
    """One aggregate over a bank cannot say whether one model is expensive or every model got
    slower, so the report splits it per model."""
    fast = [8.0] * 100
    slow = [20.0] * 40 + [8.0] * 60
    lines = live.per_model_lines({"gv-warm-lr3 78000": fast, "stylegan2 ffhq": slow}, 16.67)

    assert len(lines) == 3, lines
    assert lines[0].split() == ["per", "model", "frames", "median", "p95", "over"]
    # First played, first named -- the order the session happened in.
    assert "gv-warm-lr3 78000" in lines[1] and "stylegan2 ffhq" in lines[2]
    assert lines[1].endswith("   0.0%"), "the cheap model carried none of it"
    assert lines[2].endswith("  40.0%"), "and the expensive one carried all of it"
    # The names are the widest thing on the line, so the columns must be measured, not assumed.
    assert len({len(line) for line in lines}) == 1, lines

    # One model played is the ordinary case and has nothing to attribute.
    assert live.per_model_lines({"gv-2048-ft 72000": fast}, 16.67) == []
    assert live.per_model_lines({}, 16.67) == []


def test_auto_height_is_what_the_screen_can_actually_draw():
    """A 2560x1440 screen draws a 3072x2048 frame at 2160x1440, so producing native would send
    twice the pixels anyone can see across the bus."""
    assert fit_height(2048, 3072, None, (2560, 1440)) == 1440
    assert fit_height(2048, 3072, None, (7680, 4320)) == 2048
    assert fit_height(2048, 3072, None, (1445, 4320)) == 962
    assert fit_height(2048, 3072, None, (1445, 4320)) % 2 == 0


def test_every_height_it_can_return_is_a_legal_nv12_frame():
    """NV12 has half-resolution chroma planes, so an odd height is not a frame."""
    for want in (1, 3, 101, 1023, 1439, 2047, 9999):
        assert fit_height(2048, 3072, want) % 2 == 0, want
    for width in range(1000, 4000, 37):
        got = fit_height(2048, 3072, None, (width, 1440))
        assert got % 2 == 0 and 2 <= got <= 2048, (width, got)


def test_a_screen_that_cannot_be_asked_falls_back_to_native_rather_than_a_guess():
    for screen in (None, (), (0, 1440), (2560, 0)):
        assert fit_height(2048, 3072, None, screen) == 2048, screen
    assert fit_height(2048, 3072, 0) == 2048           # native, explicitly
    assert fit_height(2048, 3072, 1024) == 1024


def test_the_three_words_a_height_can_be():
    """`None` for auto travels all the way to `fit_height` rather than becoming a sentinel
    every reader has to know."""
    assert parse_height("auto") is None
    assert parse_height("native") == 0
    assert parse_height("1024") == 1024
    with pytest.raises(argparse.ArgumentTypeError, match="auto, native"):
        parse_height("tall")
