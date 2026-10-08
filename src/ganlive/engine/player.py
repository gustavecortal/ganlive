"""An engine model as the play loop drives it: `net(z)` draws a frame on the GPU, the model owns
its settings, and its dials come out of the program, measured when it was converted.

PyTorch-free: what a played FastGAN needs from `ganlive.settings`, `dials.derive` and
`models.common`, without the tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ganlive.dials import fastgan_dials, table
from ganlive.engine.runner import Model
from ganlive.ladder import Ladder


@dataclass(frozen=True)
class EngineConfig:
    """What the bank and the shelf ask a model's config: its latent width and its size."""

    nz: int
    ladder: Ladder

    @classmethod
    def of(cls, program: dict) -> EngineConfig:
        return cls(program["nz"], Ladder(program["height"], program["width"]))


class HostSettings:
    """The settings vector on the host, with `ganlive.settings.Settings`' interface: values are
    written (`set`), then `commit`ted, and the engine uploads what was committed when it
    changed (`changed`). Every setting is a multiplier whose neutral is 1.0."""


    def __init__(self, names) -> None:
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.write = np.ones(len(self.names), np.float32)
        self._sent = np.ones(len(self.names), np.float32)
        #: Committed and not yet uploaded. True at first, so the first frame uploads.
        self.changed = True

    def set(self, name: str, value: float) -> None:
        """Write one setting. A name this model does not have is accepted and dropped."""
        i = self.index.get(name)
        if i is not None:
            self.write[i] = value

    def reset(self) -> None:
        self.write[:] = 1.0
        self.commit()

    def commit(self) -> None:
        if not np.array_equal(self.write, self._sent):
            np.copyto(self._sent, self.write)
            self.changed = True

    def committed(self) -> np.ndarray:
        return self._sent


class PlayedDirections:
    """A model's latent directions as measured at conversion: the rows the walk adds, and what
    each moves, for the strip."""

    def __init__(self, found: dict) -> None:
        self.basis = np.asarray(found["basis"], np.float32)
        self.levels = tuple(found["levels"])
        self._report = found["report"]

    def __len__(self) -> int:
        return len(self.basis)

    def report(self) -> str:
        return self._report


class EngineGenerator:
    """One engine model, called once per frame. Returns itself: the picture is in
    `model.output` on the GPU until the next call, and `screen.EngineStage` reads it there."""

    #: The walk hands over its host view, which is what the engine uploads.

    def __init__(self, model: Model) -> None:
        self.model = model
        self.cfg = EngineConfig.of(model.program)
        self.settings = HostSettings(model.program["settings"])
        #: A StyleGAN2's `w` push, one row per style range, which the walk writes. None for a
        #: model steered through its latent.
        shape = model.program.get("push_shape")
        self.push = None if shape is None else np.zeros(shape, np.float32)
        self._pushed = None if shape is None else self.push.copy()

    def __call__(self, z) -> EngineGenerator:
        self.model.set_latent(np.asarray(z, np.float32).reshape(1, self.cfg.nz))
        if self.settings.changed:
            self.model.set_settings(self.settings.committed())
            self.settings.changed = False
        if self.push is not None and not np.array_equal(self.push, self._pushed):
            self.model.device.queue.write_buffer(self.model.buffers["push"], 0, self.push)
            np.copyto(self._pushed, self.push)
        self.model.frame()
        return self

    def report(self) -> str:
        info = self.model.device.adapter.info
        return (f"engine on {info['backend_type']} {info['device']}: {self.cfg.ladder.width}x"
                f"{self.cfg.ladder.height}, {len(self.model.steps)} steps")


def dials_of(program: dict) -> tuple[table.Layout, PlayedDirections | None]:
    """The layout a converted model offers, with each MODEL dial's measured travel, and its
    directions. A program converted without dials offers the stock tables, unmeasured."""
    dials = program.get("dials") or {}
    if program.get("family") == "stylegan2":
        swept = dials.get("swept")
        layout = table.stylegan2() if swept is None else table.stylegan2(
            swept["names"], swept["rests"], [tuple(c) for c in swept["curves"]],
            swept["levels"], tuple(tuple(r) for r in swept["ranges"]))
    else:
        layout = fastgan_dials.fastgan(noise_gains=dials.get("noise_gains"))
    measured = dials.get("measured", {})
    layout = table.Layout(tuple(replace(k, measured=measured.get(k.name, k.measured))
                                for k in layout.knobs))
    found = dials.get("directions")
    return layout, None if not found else PlayedDirections(found)
