"""An ONNX generator that plays like a torch one: `net(z) -> [frame]`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ganlive.device import detect_backend, synchronize
from ganlive.models.common import Ladder
from ganlive.models.onnx_file import OnnxConfig, dials_of
from ganlive.models.runtime import open_graph
from ganlive.pixels import pinned
from ganlive.settings import Settings


class OnnxGenerator:
    """One compiled graph, called once per frame, with the frame handed back on `device`."""

    #: The walk hands over its host view rather than a device tensor; `infer` takes it as is.
    latent_on_host = True

    def __init__(self, path, device: str | None = None) -> None:
        """`device` is where the frame is handed back: a torch device, or `cpu`. Default is
        whichever accelerator this machine has."""
        self.path = Path(path)
        self.device = device = device or detect_backend()
        # Touch the torch device before OpenVINO does: both runtimes drive the same card
        # through Level Zero, and torch has to initialise it first.
        if device and device != "cpu":
            torch.zeros(1, device=device)
            synchronize(device)
        self.runner = open_graph(self.path)
        self.settings = dials_of(self.path)["settings"]
        if len(self.settings) != self.runner.settings:
            raise RuntimeError(
                f"the graph takes {self.runner.settings} settings and names "
                f"{len(self.settings)} of them; refusing to guess which dial is which")
        self.precision = self.runner.precision
        self.cfg = OnnxConfig(nz=self.runner.nz,
                              ladder=Ladder(width=self.runner.size[1],
                                            height=self.runner.size[0]))
        self.nz = self.cfg.nz
        self._z = np.zeros((1, self.nz), dtype=np.float32)
        # The generator owns its settings, so `net(z)` stays a one-argument call. On the host
        # in f32, because that is what the graph takes.
        self.knobs = Settings(self.settings, "cpu", torch.float32)
        # The graph reads `committed()`, never `vec`, so a commit only has to write the host.
        self.knobs.feed_from(self.knobs.vec)
        #: Pinned memory the runtime writes each frame into, so the upload to the card is a
        #: DMA rather than a copy staged through a driver buffer. `None` on the CPU, or where
        #: the runtime only hands back arrays of its own.
        self._landing = None
        if device != "cpu" and self.runner.land is not None:
            host = pinned(self.runner.shapes[0], torch.from_numpy(
                np.zeros(0, self.runner.dtype)).dtype)
            if host.is_pinned():
                self.runner.land(host.numpy())
                self._landing = host

    def __call__(self, z) -> list[torch.Tensor]:
        """One frame, on `device`, as a torch generator would return it."""
        frame = self.infer(z)
        host = self._landing if self._landing is not None else torch.from_numpy(frame)
        # Blocking, so the upload has read the buffer before the next frame is written into it.
        out = host.to(self.device)
        # On the CPU `.to` is a no-op, and the runtime writes the next frame into this buffer.
        return [out.clone() if out.data_ptr() == host.data_ptr() else out]

    def infer(self, z) -> np.ndarray:
        """One frame as the runtime left it: a view into host-visible memory."""
        if isinstance(z, np.ndarray):
            self._z[:] = z.reshape(1, self.nz)
        else:
            # The graph takes f32 whatever it computes in.
            self._z[:] = z.detach().to("cpu", torch.float32).reshape(1, self.nz).numpy()
        return self.runner.infer(self._z, self.knobs.committed())

    def eval(self):
        return self

    def report(self) -> str:
        return (f"onnx {self.path.name}: {self.runner.report()}, {self.precision}, "
                f"{len(self.settings)} settings, frame handed back on {self.device}")
