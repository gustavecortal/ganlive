"""Recording a prepared generator as one device graph, and replaying it.

It removes submission, not work. The invariants that make it safe are on `Replay`; the one
that bites hardest is that nothing may be issued on the generator's queue between two replays.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ganlive.pixels import EXACT_LEVELS, FLOOR_LEVELS, levels, pinned


def compile_and_count(net, nz: int, device, dtype=torch.float16,
                      warmup: int = 3) -> tuple[object, int, float]:
    """Compile a generator, warm it up, and report how many graphs came out.

    Never fatal, on the same rule as `capture`: a machine without a host compiler, or a backend
    Inductor's codegen is still wrong on, gets the generator back in eager with `0` graphs and
    a line saying so, rather than no picture at all."""
    from torch._dynamo.utils import counters

    before = counters["frames"]["ok"]
    t0 = time.perf_counter()
    try:
        compiled = torch.compile(net, dynamic=False)
        with torch.no_grad():
            for _ in range(warmup):
                compiled(torch.zeros(1, nz, device=device, dtype=dtype))
    except Exception as exc:  # noqa: BLE001 -- Inductor's failures are not enumerable; eager answers every one
        print(f"compile: running eager -- {str(exc).splitlines()[0][:120]}", flush=True)
        return net, 0, time.perf_counter() - t0
    return compiled, counters["frames"]["ok"] - before, time.perf_counter() - t0


class Replay:
    """A generator recorded as one device graph, called exactly like the generator.

    It removes submission, not work: the card finishes a frame long before Python has
    finished asking for it, and a capture does the asking once, at load. Exact, not
    approximate -- a replayed frame matches the compiled net to 0.0000 8-bit levels.

    Three invariants, structural rather than remembered:

    - **Nothing is issued on the generator's queue between two replays.** The latent and
      every `capture(feeds=)` tensor is read by the graph from a pinned host buffer this
      object owns, so a frame's inputs are host writes. On XPU an eager op between two
      replays makes every later replay slower, without bound.
    - **The frame returned is the same tensor every time.** Nothing may hold it across a
      frame boundary; copy through `.float()` or `.cpu()` to keep one.
    - **The latent arrives on the host.** `latent_on_host` asks the walk for its own host
      view. A device tensor is still accepted, as a download, for the gates run at load.
    """

    latent_on_host = True

    def __init__(self, net, graph, latent: torch.Tensor, frame, host: torch.Tensor,
                 twins) -> None:
        self.net, self.graph, self.latent, self.frame = net, graph, latent, frame
        #: The pinned host buffer the latent is read from, and its numpy view -- the frame path
        #: writes the view, because `Tensor.copy_` releases the GIL and a window thread takes
        #: it: 9.37 ms to 10.75 on the played loop, twice, for that one line. numpy holds it.
        self.host, self._view = host, host.numpy().reshape(-1)
        #: Each recorded feed's host twin, by the identity of the device tensor it feeds.
        self.twins = {id(dev): twin for dev, twin in twins}

    def __getattr__(self, name):
        """Anything else asked of this, asked of the generator -- it is a stand-in for one."""
        return getattr(self.__dict__["net"], name)

    def twin(self, tensor: torch.Tensor) -> torch.Tensor:
        """The host buffer the graph reads `tensor` from. Writes there reach the next replay."""
        try:
            return self.twins[id(tensor)]
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
            self.host.copy_(z.reshape(self.host.shape))  # a download, for the probes at load
        self.graph.replay()
        return self.frame


def capture(net, nz: int, device, dtype=torch.float16, warmup: int = 3, feeds=()):
    """Record a prepared generator as one device graph. Returns `(callable, what happened)`.

    `feeds` are the device tensors the frame path writes between forwards -- the settings
    vector, a `w` push. Each is recorded as an upload from a pinned host twin at the head of
    the graph, so the caller writes the twin (`Replay.twin`) and issues nothing.

    **After every measurement, before the gate.** The sweeps that derive a model's dials read
    its module tree and hold frames side by side to difference them, and a captured graph
    offers one output buffer -- so it is recorded once those have run, and the dead-dial gate
    then runs through the capture rather than around it.

    Never fatal, and never taken on trust. A backend without graph capture, a generator that
    is not a module -- an adopted ONNX graph runs under its own runtime, so a recording of the
    torch stream would hold none of its work -- or a forward that cannot be recorded hands back
    the generator it was given and says so on the load line. And a recording that *was* made is
    then **replayed against the answer taken before it**, on two latents: it has to reproduce
    the compiled net's own frame, and it has to give a different frame for a different latent.
    A capture that recorded nothing replays fast and paints a still picture, which is the one
    failure mode of this that no exception reports and no later measurement would question."""
    from ganlive.models.common import first_image
    from ganlive.models.common import latent as probe_latent

    if not isinstance(net, nn.Module):
        return net, "not captured: this generator is not a torch module"
    graphs = getattr(torch, str(device).split(":")[0], None)
    # Each backend names its own class -- `XPUGraph` here, `CUDAGraph` on the other one -- and
    # both hand it to a `graph(...)` context manager of the same shape.
    kind = getattr(graphs, "XPUGraph", None) or getattr(graphs, "CUDAGraph", None)
    if kind is None or not hasattr(graphs, "graph"):
        return net, "not captured: no graph capture on this device"
    latent = torch.zeros(1, nz, device=device, dtype=dtype)
    host = pinned((1, nz), dtype)
    twins = [(t, pinned(t.shape, t.dtype)) for t in feeds]
    # Seeded, so the verdict below cannot flake on two latents that happened to be alike, and
    # so taking it does not disturb the global stream the frozen noise was drawn from.
    probes = [probe_latent(nz, seed, device, dtype) for seed in (0, 1)]

    def upload() -> None:
        latent.copy_(host, non_blocking=True)
        for dev, twin in twins:
            dev.copy_(twin, non_blocking=True)

    try:
        with torch.no_grad():
            for dev, twin in twins:
                twin.copy_(dev)             # the twin starts holding what the card holds
            for _ in range(warmup):
                upload()
                net(latent)
            # Taken before the capture: afterwards this net writes into the graph's own pool.
            # Kept in the frame's own precision -- `levels` accumulates in float32 -- so a
            # 3072x2048 model spends 38 MB here rather than 151 MB at the moment the graph's
            # pool is being reserved.
            host.copy_(probes[0])
            upload()
            want = first_image(net(latent)).clone()
            graph = kind()
            with graphs.graph(graph):
                upload()
                frame = net(latent)
            played = Replay(net, graph, latent, frame, host, twins)
            same = levels(first_image(played(probes[0])), want)
            moved = levels(first_image(played(probes[1])), want)
    except (RuntimeError, NotImplementedError, AttributeError) as exc:
        return net, f"not captured: {str(exc).splitlines()[0][:120]}"
    if same > EXACT_LEVELS:
        return net, f"not captured: the replay differs from the forward by {same:.3f} 8-bit levels"
    # A different threshold because it is a different question -- not "has the rewrite drifted"
    # but "is this a control at all", which is the floor a derived dial has to clear.
    if moved <= FLOOR_LEVELS:
        return net, "not captured: the replay paints the same frame whatever the latent"
    return played, (f"captured, exact to {same:.4f} 8-bit levels, {1 + len(twins)} upload(s) "
                    f"recorded")
