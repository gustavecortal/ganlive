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


@dataclass(frozen=True)
class Ladder:
    height: int
    width: int


@dataclass(frozen=True)
class EngineConfig:
    """What the bank and the shelf ask a model's config: its latent width and its size."""

    nz: int
    ladder: Ladder


class HostSettings:
    """The settings vector on the host, with `ganlive.settings.Settings`' interface: values are
    written (`set`), then `commit`ted, and the engine uploads what was committed when it
    changed. Every setting is a multiplier whose neutral is 1.0."""

    pinned = False

    def __init__(self, names) -> None:
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.write = np.ones(len(self.names), np.float32)
        self._sent = np.ones(len(self.names), np.float32)
        self.skipped = 0

    def set(self, name: str, value: float) -> None:
        """Write one setting. A name this model does not have is accepted and dropped."""
        i = self.index.get(name)
        if i is not None:
            self.write[i] = value

    def reset(self) -> None:
        self.write[:] = 1.0
        self.commit()

    def commit(self) -> None:
        if np.array_equal(self.write, self._sent):
            self.skipped += 1
            return
        np.copyto(self._sent, self.write)

    def committed(self) -> np.ndarray:
        return self._sent


class PlayedDirections:
    """A model's latent directions as measured at conversion: the rows the walk adds, and what
    each moves, for the strip."""

    space = "z"
    ranges = None

    def __init__(self, found: dict) -> None:
        self.basis = np.asarray(found["basis"], np.float32)
        self.levels = tuple(found["levels"])
        self._report = found["report"]

    def __len__(self) -> int:
        return len(self.basis)

    def report(self) -> str:
        return self._report


class EngineGenerator:
    """One engine model, called once per frame. Returns itself as the frame: the picture is in
    `model.output` on the GPU until the next call, and `screen.EngineStage` reads it there."""

    #: The walk hands over its host view, which is what the engine uploads.
    latent_on_host = True

    def __init__(self, model: Model) -> None:
        self.model = model
        program = model.program
        self.cfg = EngineConfig(program["nz"], Ladder(program["height"], program["width"]))
        self.nz = self.cfg.nz
        self.settings = HostSettings(program["settings"])
        self._uploaded: np.ndarray | None = None

    def __call__(self, z) -> list[EngineGenerator]:
        self.model.set_latent(np.asarray(z, np.float32).reshape(1, self.nz))
        k = self.settings.committed()
        if self._uploaded is None or not np.array_equal(k, self._uploaded):
            self.model.set_settings(k)
            self._uploaded = k.copy()
        self.model.frame()
        return [self]

    def eval(self) -> EngineGenerator:
        return self

    def report(self) -> str:
        info = self.model.device.adapter.info
        return (f"engine on {info['backend_type']} {info['device']}: {self.cfg.ladder.width}x"
                f"{self.cfg.ladder.height}, {len(self.model.steps)} steps")


def dials_of(program: dict) -> tuple[table.Layout, PlayedDirections | None]:
    """The layout a converted FastGAN offers, with each MODEL dial's measured travel, and its
    directions. A program converted without dials offers the stock tables, unmeasured."""
    dials = program.get("dials") or {}
    layout = fastgan_dials.fastgan(noise_gains=dials.get("noise_gains"))
    measured = dials.get("measured", {})
    layout = table.Layout(tuple(replace(k, measured=measured.get(k.name, k.measured))
                                for k in layout.knobs))
    found = dials.get("directions")
    return layout, None if not found else PlayedDirections(found)
