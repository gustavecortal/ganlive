"""Inference-only rewrites of a trained generator. Exact, not approximate."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ganlive.models.fastgan import NoiseInjection


def remove_spectral_norm(net: nn.Module) -> int:
    """Bake every spectral-norm reparametrisation into a plain weight, in place."""
    removed = 0
    for module in net.modules():
        try:
            nn.utils.remove_spectral_norm(module)
            removed += 1
        except (ValueError, RuntimeError):
            continue  # no spectral_norm on this module
    return removed


_KR, _KB = 0.2126, 0.0722
_KG = 1 - _KR - _KB

#: What "exact, not approximate" is allowed to mean for a rewrite that should be bit-identical.
#: The capture measured 0.0000 on both families; loose enough to survive a driver that
#: reassociates something, tight enough that a still picture -- 77 levels -- cannot pass.
EXACT_LEVELS = 0.5


def _levels(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean difference between two frames in [-1, 1], in the 8-bit levels this repo judges by.

    `adopt.levels` is the same quantity for numpy arrays and carries the note about the unit;
    this one stays in torch and on the card. Scaling **after** the reduction rather than before
    it, and accumulating in float32, keeps a full-resolution frame from costing three more
    tensors of its own size -- 226 MB at 3072x2048 -- to answer one scalar question. The
    subtraction of two near-identical halves is exact, so nothing is lost by it."""
    return (a - b).abs().mean(dtype=torch.float32).item() * 127.5


class FoldedNoise(nn.Module):
    """What remains of `conv -> NoiseInjection -> BatchNorm` once the norm is folded away."""

    def __init__(self, coeff: torch.Tensor, noise: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff)
        self.register_buffer("noise", noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(x, self.coeff, self.noise)


def _bn_affine(bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """`(scale, shift)` with `BN(z) == scale*z + shift` per channel, computed in float32."""
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    return scale, bn.bias.float() - bn.running_mean.float() * scale


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """`conv` followed by an eval `bn`, as one convolution."""
    scale, shift = _bn_affine(bn)
    dt = conv.weight.dtype
    weight = (conv.weight.float() * scale.reshape(-1, 1, 1, 1)).to(dt)
    bias = (shift if conv.bias is None else conv.bias.float() * scale + shift).to(dt)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode, device=conv.weight.device, dtype=dt)
    fused.weight = nn.Parameter(weight.detach(), requires_grad=False)
    fused.bias = nn.Parameter(bias.detach(), requires_grad=False)
    return fused


def fold_norms(net: nn.Module) -> dict[str, int]:
    """Fold every eval-time BatchNorm into the convolution feeding it. Returns what changed."""
    counts = {"conv_bn": 0, "conv_noise_bn": 0, "skipped_unfrozen": 0}
    for parent in net.modules():
        if not isinstance(parent, nn.Sequential):
            continue
        items, out, i = list(parent), [], 0
        while i < len(items):
            a = items[i]
            b = items[i + 1] if i + 1 < len(items) else None
            c = items[i + 2] if i + 2 < len(items) else None
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                out.append(_fuse(a, b))
                counts["conv_bn"] += 1
                i += 2
            elif (isinstance(a, nn.Conv2d) and isinstance(b, NoiseInjection)
                  and isinstance(c, nn.BatchNorm2d)):
                if b.frozen is None:
                    counts["skipped_unfrozen"] += 1
                    out.append(a)
                    i += 1
                    continue
                scale, _ = _bn_affine(c)
                out.append(_fuse(a, c))
                out.append(FoldedNoise((scale * b.weight).reshape(1, -1, 1, 1).float(),
                                       b.frozen.float()))
                counts["conv_noise_bn"] += 1
                i += 3
            else:
                out.append(a)
                i += 1
        if len(out) != len(items):
            parent._modules.clear()
            for j, m in enumerate(out):
                parent._modules[str(j)] = m
    return counts


def fold_free_noise(net: nn.Module) -> int:
    """Precompute `weight * frozen` for injections no norm fold absorbed."""
    n = 0
    for m in net.modules():
        if isinstance(m, NoiseInjection) and m.freeze and m.frozen is not None:
            const = (m.weight.detach() * m.frozen).detach()
            m.forward = (lambda c: (lambda x: x + c.to(x.dtype)))(const)
            n += 1
    return n


def compiled_to_nv12():
    """`to_nv12` through TorchInductor, which is a 5.8x on that function alone."""
    return torch.compile(to_nv12, dynamic=False)


def to_rgb(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the `(H, W, 3)` uint8 bytes a window blits, on the GPU."""
    x = out.add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1)[0].contiguous()


def compiled_to_rgb():
    """`to_rgb` through TorchInductor: 1.335 ms eager against 0.522 at 1536x1024."""
    return torch.compile(to_rgb, dynamic=False)


def to_bgra(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the `(H, W, 4)` uint8 bytes an SDL texture already is."""
    x = out.add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    b, g, r = x[:, 2], x[:, 1], x[:, 0]
    return torch.stack([b, g, r, torch.full_like(b, 255)], dim=-1)[0].contiguous()


def compiled_to_bgra():
    """`to_bgra` through TorchInductor. Same `dynamic=False` argument as the other two."""
    return torch.compile(to_bgra, dynamic=False)


# ORDER MATTERS BOTH WAYS. The fold must come AFTER a forward pass -- it reads running statistics, so
# folding a cold net silently folds nothing in the blocks that matter most -- and BEFORE compile.
def prepare_for_inference(net: nn.Module, nz: int, device, *, half: bool = True,
                          fold: bool = True, compile_yuv: bool = True,
                          compile_net: bool = False, spectral: bool = True) -> dict:
    """Fold, cast, and hand back the colour conversion to use. Reports what was applied."""
    with torch.no_grad():
        net(torch.zeros(1, nz, device=device))       # draws the lazy frozen patterns
    report: dict = {"folded": {}, "half": half, "graphs": 0}
    # **Bake the spectral norm, which was being recomputed on every frame.** It is a pre-forward hook, so
    # `weight = weight_orig / sigma` ran once per forward for every module still carrying one -- 15.4 M
    # parameters read and written for nothing, plus the power-iteration matmuls, at 61.6 MB of traffic a
    # frame in fp16. In eval nothing updates `u` and `v`, so sigma is a constant and baking it is exact:
    # measured at 0.00000 8-bit levels over three latents in fp32. Before the fold, not after.
    if spectral:
        report["spectral_removed"] = remove_spectral_norm(net)
    if fold:
        report["folded"] = fold_norms(net)
        report["folded"]["free_noise"] = fold_free_noise(net)
    if half:
        net = net.to(torch.float16).to(memory_format=torch.channels_last)
    if compile_net:
        net, graphs, secs = compile_and_count(
            net, nz, device, torch.float16 if half else torch.float32)
        report["net_compile_s"] = secs
        report["net_graphs"] = graphs
    report["net"] = net
    report["yuv"] = to_nv12
    report["rgb"] = to_rgb
    report["bgra"] = to_bgra
    if compile_yuv:
        report["yuv"] = compiled_to_nv12()
        report["rgb"] = compiled_to_rgb()
        report["bgra"] = compiled_to_bgra()
    return report


def warm(fns, probe: torch.Tensor) -> int:
    """Compile the conversions now, **on a frame the real shape**, and report the graph count."""
    from torch._dynamo.utils import counters

    before = counters["frames"]["ok"]
    with torch.no_grad():
        for fn in fns:
            fn(probe)
    return counters["frames"]["ok"] - before


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


def _pinned(shape, dtype) -> torch.Tensor:
    """A host buffer the card reads directly, or a plain one where pinning is refused -- the
    recording then fails on the pageable copy and says so there, rather than here."""
    try:
        return torch.zeros(shape, dtype=dtype, pin_memory=True)
    except (RuntimeError, NotImplementedError):
        return torch.zeros(shape, dtype=dtype)


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
    from ganlive.dials.derive import FLOOR_LEVELS, _latent
    from ganlive.models.fastgan import first_image

    if not isinstance(net, nn.Module):
        return net, "not captured: this generator is not a torch module"
    graphs = getattr(torch, str(device).split(":")[0], None)
    # Each backend names its own class -- `XPUGraph` here, `CUDAGraph` on the other one -- and
    # both hand it to a `graph(...)` context manager of the same shape.
    kind = getattr(graphs, "XPUGraph", None) or getattr(graphs, "CUDAGraph", None)
    if kind is None or not hasattr(graphs, "graph"):
        return net, "not captured: no graph capture on this device"
    latent = torch.zeros(1, nz, device=device, dtype=dtype)
    host = _pinned((1, nz), dtype)
    twins = [(t, _pinned(t.shape, t.dtype)) for t in feeds]
    # Seeded, so the verdict below cannot flake on two latents that happened to be alike, and
    # so taking it does not disturb the global stream the frozen noise was drawn from.
    probes = [_latent(nz, seed, device, dtype) for seed in (0, 1)]

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
            # Kept in the frame's own precision -- `_levels` accumulates in float32 -- so a
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
            same = _levels(first_image(played(probes[0])), want)
            moved = _levels(first_image(played(probes[1])), want)
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


class PinnedRing:
    """A ring of pinned host buffers to copy device frames into, instead of `Tensor.cpu()`."""

    def __init__(self, depth: int, device: str, mode: str = "pinned"):
        self.depth, self._device, self._mode = max(2, depth), device, mode
        self._buf: list = []
        self._key = None
        self._n = 0
        self.pinned = False
        self.odd = 0

    def _alloc(self, src) -> None:
        if self._mode == "pinned":
            self._buf = [_pinned(src.shape, src.dtype) for _ in range(self.depth)]
        else:
            self._buf = [torch.empty(src.shape, dtype=src.dtype) for _ in range(self.depth)]
        self.pinned = all(b.is_pinned() for b in self._buf)
        self._key = (tuple(src.shape), src.dtype)

    def take(self, src, wait: bool = True):
        """Copy `src` to the host and return a numpy view of the buffer it landed in."""
        if self._mode == "cpu":
            return src.cpu().numpy()            # the original path, kept as the baseline arm
        if self._key is None:
            self._alloc(src)
        if (tuple(src.shape), src.dtype) != self._key:
            self.odd += 1
            return src.cpu().numpy()            # odd shape, e.g. a short final batch
        dst = self._buf[self._n % len(self._buf)]
        self._n += 1
        dst.copy_(src, non_blocking=True)
        if wait:
            self.sync()
        return dst.numpy()

    def sync(self) -> None:
        """Wait for everything queued on **this ring's** device, copies included."""
        from ganlive.device import synchronize

        synchronize(self._device)


def nv12_plane_views(frame, height: int, width: int):
    """numpy views onto an nv12 frame's own buffers, honouring each plane's line size."""
    return [np.frombuffer(p, dtype=np.uint8).reshape(rows, p.line_size)[:, :width]
            for p, rows in ((frame.planes[0], height), (frame.planes[1], height // 2))]


def to_nv12(out: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> the encoder's `(H*3/2, W)` uint8 plane stack, on the GPU.

    NV12: the Y plane, then the chroma interleaved as UVUVUV rows."""
    y, u, v = _yuv_planes(out)
    h, w = out.shape[-2:]
    chroma = torch.stack([u, v], dim=-1).reshape(-1)     # U,V,U,V... in row order
    return torch.cat([y.reshape(-1), chroma]).reshape(h * 3 // 2, w)


def _yuv_planes(out: torch.Tensor):
    """The shared arithmetic: `(y, u, v)` as uint8, chroma at half resolution."""
    o = out.float().clamp(-1, 1)
    r, g, b = o[:, 0], o[:, 1], o[:, 2]
    luma = _KR * r + _KG * g + _KB * b
    y = (luma * 109.5 + 125.5).round().clamp(0, 255).to(torch.uint8)   # 219/2, 219/2 + 16

    pooled = F.avg_pool2d(o, 2)                      # one call for all three planes
    pr, pg, pb = pooled[:, 0], pooled[:, 1], pooled[:, 2]
    pluma = _KR * pr + _KG * pg + _KB * pb
    u = ((pb - pluma) * (224 / (4 * (1 - _KB))) + 128).round().clamp(0, 255).to(torch.uint8)
    v = ((pr - pluma) * (224 / (4 * (1 - _KR))) + 128).round().clamp(0, 255).to(torch.uint8)
    return y, u, v
