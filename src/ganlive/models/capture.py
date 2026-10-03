"""Compiling a prepared generator, and recording it as one device graph to replay per frame.

A recording removes the cost of submitting a frame, not of computing it. The invariants
that make it safe are on `Replay`.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ganlive.models.common import first_image
from ganlive.models.common import latent as probe_latent
from ganlive.pixels import EXACT_LEVELS, FLOOR_LEVELS, levels, pinned

#: Forwards run before a compile or a recording is trusted.
WARMUP = 3


def compile_and_count(net, nz: int, device,
                      dtype=torch.float16) -> tuple[object, int, float]:
    """Compile a generator, warm it up, and report how many graphs came out and how long it
    took.

    Never fatal: where Inductor fails (no host compiler, a backend it cannot generate code
    for) the generator comes back in eager, with `0` graphs and a line saying so."""
    from torch._dynamo.utils import counters

    before = counters["frames"]["ok"]
    t0 = time.perf_counter()
    try:
        compiled = torch.compile(net, dynamic=False)
        with torch.no_grad():
            for _ in range(WARMUP):
                compiled(torch.zeros(1, nz, device=device, dtype=dtype))
    except Exception as exc:  # noqa: BLE001 -- Inductor's failures are not enumerable; eager answers every one
        print(f"compile: running eager -- {str(exc).splitlines()[0][:120]}", flush=True)
        return net, 0, time.perf_counter() - t0
    return compiled, counters["frames"]["ok"] - before, time.perf_counter() - t0


class Replay:
    """A generator recorded as one device graph, called exactly like the generator.

    A replayed frame matches the compiled net exactly. Three invariants:

    - **Nothing is issued on the generator's queue between two replays.** The latent and
      every `capture(feeds=)` tensor are uploaded by the graph itself from pinned host
      buffers this object owns, so a frame's inputs are host writes. On XPU an eager op
      between two replays makes every later replay slower, without bound.
    - **The frame returned is the same tensor every time.** Nothing may hold it across a
      frame boundary; copy it to keep one.
    - **The latent arrives on the host** (`latent_on_host`). A device tensor is still
      accepted, as a download, for the checks run at load.
    """

    latent_on_host = True

    def __init__(self, net, graph, latent: torch.Tensor, frame, host: torch.Tensor,
                 host_buffers) -> None:
        self.net, self.graph, self.latent, self.frame = net, graph, latent, frame
        #: The pinned host buffer the latent is read from, and its numpy view. The frame path
        #: writes the view: numpy keeps the GIL, where `Tensor.copy_` releases it and the
        #: window thread may hold on to it.
        self.host, self._view = host, host.numpy().reshape(-1)
        #: Each recorded feed's host buffer, by the identity of the device tensor it feeds.
        self.host_buffers = {id(dev): buf for dev, buf in host_buffers}

    def __getattr__(self, name):
        """Anything else asked of this is asked of the generator it stands in for."""
        return getattr(self.__dict__["net"], name)

    def host_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
        """The host buffer the graph uploads `tensor` from. A write there reaches the next
        replay."""
        try:
            return self.host_buffers[id(tensor)]
        except KeyError:
            raise KeyError("no upload of that tensor was recorded; pass it in "
                           "`capture(feeds=)`") from None

    def __call__(self, z):
        if (z.size if isinstance(z, np.ndarray) else z.numel()) != self._view.size:
            raise ValueError(
                f"this generator was captured for a {tuple(self.host.shape)} latent and was "
                f"handed {tuple(z.shape)}. A captured graph has one shape; pass "
                f"`capture=False` in the `LoadOptions` to drive it at another.")
        if isinstance(z, np.ndarray):
            self._view[:] = z.reshape(-1)               # the frame path; see `_view`
        else:
            self.host.copy_(z.reshape(self.host.shape))  # a download, for the checks at load
        self.graph.replay()
        return self.frame


def capture(net, nz: int, device, dtype=torch.float16, feeds=()):
    """Record a prepared generator as one device graph. Returns `(callable, what happened)`.

    `feeds` are the device tensors the frame path writes between forwards, such as the
    settings vector or a push buffer. Each is recorded as an upload from a pinned host buffer
    at the head of the graph, so the caller writes that buffer (`Replay.host_buffer`) and
    issues nothing.

    Run it after every measurement that reads the module tree or holds frames side by side:
    a recording has one output buffer.

    Never fatal: a backend without graph capture, a generator that is not a torch module (an
    ONNX graph runs under its own runtime), or a forward that cannot be recorded gives back
    the generator it was given and says why. A recording that was made is then replayed on
    two latents, and kept only if it reproduces the forward and gives a different frame for a
    different latent: a recording that holds none of the work replays fast and paints a still
    picture, which nothing else would report."""
    if not isinstance(net, nn.Module):
        return net, "not captured: this generator is not a torch module"
    graphs = getattr(torch, str(device).split(":")[0], None)
    # Each backend names its own class (`XPUGraph`, `CUDAGraph`), and both hand it to a
    # `graph(...)` context manager of the same shape.
    kind = getattr(graphs, "XPUGraph", None) or getattr(graphs, "CUDAGraph", None)
    if kind is None or not hasattr(graphs, "graph"):
        return net, "not captured: no graph capture on this device"
    latent = torch.zeros(1, nz, device=device, dtype=dtype)
    host = pinned((1, nz), dtype)
    buffers = [(t, pinned(t.shape, t.dtype)) for t in feeds]
    # Seeded, so the check below cannot flake, and so it does not disturb the global stream.
    probes = [probe_latent(nz, seed, device, dtype) for seed in (0, 1)]

    def upload() -> None:
        latent.copy_(host, non_blocking=True)
        for dev, buf in buffers:
            dev.copy_(buf, non_blocking=True)

    try:
        with torch.no_grad():
            for dev, buf in buffers:
                buf.copy_(dev)              # each host buffer starts holding what the card holds
            for _ in range(WARMUP):
                upload()
                net(latent)
            # Taken before the recording, because afterwards the net writes into the graph's
            # own memory pool.
            host.copy_(probes[0])
            upload()
            want = first_image(net(latent)).clone()
            graph = kind()
            with graphs.graph(graph):
                upload()
                frame = net(latent)
            played = Replay(net, graph, latent, frame, host, buffers)
            same = levels(first_image(played(probes[0])), want)
            moved = levels(first_image(played(probes[1])), want)
    except (RuntimeError, NotImplementedError, AttributeError) as exc:
        return net, f"not captured: {str(exc).splitlines()[0][:120]}"
    if same > EXACT_LEVELS:
        return net, f"not captured: the replay differs from the forward by {same:.3f} 8-bit levels"
    if moved <= FLOOR_LEVELS:
        return net, "not captured: the replay paints the same frame whatever the latent"
    return played, (f"captured, exact to {same:.4f} 8-bit levels, {1 + len(buffers)} upload(s) "
                    f"recorded")
