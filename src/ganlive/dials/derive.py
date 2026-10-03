"""Latent directions for the direction dials: derived, then measured.

A basis is proposed from the weights (`sefa`, `sefa_banded`, `sefa_onnx`) or from the whole
generator's Jacobian (`active_banded`), then `shortlist`, `equalise` and `rank` measure it on
the model that plays and keep only rows that move the picture more than a random direction.
A `z` basis is added to the latent; a `w` basis is written to a StyleGAN2's push buffer.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, replace

import torch
from torch import nn

from ganlive.models.common import first_image, latent
from ganlive.models.onnx_file import initializers, producers, structure
from ganlive.pixels import FLOOR_LEVELS, LEVEL, RANDOM_FLOOR

#: The layer types that can be the first thing a latent meets.
CONSUMERS = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)


@dataclass(frozen=True)
class Directions:
    """An ordered basis for the latent, and where it was read from."""

    #: One row per direction, unit length. By singular value from `sefa`, by measured
    #: effect after `rank`.
    basis: torch.Tensor
    #: Singular value per row. A flat spectrum means the tail directions are arbitrary.
    strength: torch.Tensor
    #: The module the basis came from, for the report.
    source: str
    #: What that module's weight looked like, so a surprise is visible rather than assumed.
    shape: tuple[int, ...]
    #: The generator's latent width, set only for a `w` basis, whose rows are not latents.
    z_dim: int | None = None
    #: The shape a row takes in the push buffer, `(ranges, width)`, or `None` for a `z` basis.
    push_shape: tuple[int, int] | None = None
    #: Which style range each row came out of, in row order.
    ranges: tuple[str, ...] | None = None
    #: What each kept row moves turning up and turning down, the larger first (see `orient`).
    halves: tuple[tuple[float, float], ...] | None = None
    #: The bar: what a *random* unit direction of the same length moves, pooled over the bands.
    random_levels: float | None = None
    #: The same bar per style range, because the ranges are not equally sensitive. `None`
    #: without ranges.
    random_by_range: tuple[tuple[str, float], ...] | None = None
    #: The push every level here was measured at. The ratio to random changes with it, so
    #: the report states it.
    amount: float | None = None
    #: Rows `rank` dropped, so the report can say how many of the derived set survived.
    dropped: int = 0

    def __len__(self) -> int:
        return int(self.basis.shape[0])

    @property
    def space(self) -> str:
        """`"z"` if these push the latent, `"w"` if they push the style after the mapping."""
        return "z" if self.z_dim is None else "w"

    @property
    def row_bands(self) -> tuple[str | None, ...]:
        """The band each row came from. A basis with no ranges has one anonymous band."""
        return self.ranges or (None,) * len(self)

    @property
    def band_names(self) -> tuple[str | None, ...]:
        """The bands in the ladder's order, not the alphabet's, which puts `w_fine` first."""
        return band_names(self.row_bands)

    @property
    def levels(self) -> tuple[float, ...] | None:
        """Mean 8-bit levels each row moves the picture, read off `halves` through `middle`;
        `None` until `rank` has measured it."""
        if self.halves is None:
            return None
        return tuple(middle([u for u, _ in self.halves], [d for _, d in self.halves]))

    @property
    def nz(self) -> int:
        return self.z_dim if self.z_dim is not None else int(self.basis.shape[1])

    def spread(self) -> float:
        """Largest singular value over smallest. 1.0 means the ranking carries no signal."""
        return float(self.strength.max() / self.strength.min().clamp_min(1e-12))

    def report(self) -> str:
        s = self.strength
        head = f"{len(self)} {self.space}-directions from {self.source} {self.shape}"
        if self.dropped:
            head = (f"{len(self)} of {len(self) + self.dropped} {self.space}-directions kept "
                    f"from {self.source} {self.shape}")
        if not len(self):
            if self.baseline is None:
                return head
            return (f"{head}; a random direction moves {self.baseline} "
                    f"and none of these beat it {RANDOM_FLOOR:g}x")
        # max and min, not first and last: `rank` re-orders the rows by measured effect.
        out = head + f"; strength {s.max():.3g} to {s.min():.3g}, spread {self.spread():.1f}x"
        if self.levels:
            out += f"; measured {', '.join(f'{lv:.0f}' for lv in self.levels)} 8-bit levels"
        if self.baseline is not None:
            out += f" against {self.baseline} for a random direction"
        if self.amount is not None:
            out += f", at a push of {self.amount:g} each way"
        if self.halves:
            worst = min(self.halves, key=lambda h: h[1] / max(h[0], 1e-9))
            out += (f"; the most lopsided moves {worst[0]:.0f} turning up and "
                    f"{worst[1]:.0f} turning down")
        return out

    @property
    def baseline(self) -> str | None:
        """What a random direction moves, per range where the ranges differ -- the number each
        row was actually judged against. `None` if none was measured."""
        if self.random_by_range:
            return ", ".join(f"{name} {level:.1f}" for name, level in self.random_by_range)
        if self.random_levels is None:
            return None
        return f"{self.random_levels:.1f} 8-bit levels"


def pushed(net: nn.Module, z: torch.Tensor, push, into, push_shape=None) -> torch.Tensor:
    """The picture with `push` applied: added to the latent, or written to the push buffer
    `into`. `push=None` is the model at rest. The push buffer is left cleared, so no later
    frame carries a push it was not given."""
    if into is None:
        return first_image(net(z if push is None else z + push.unsqueeze(0)))
    if push is None:
        into.zero_()
    else:
        into.copy_(push.reshape(push_shape or into.shape))
    try:
        return first_image(net(z))
    finally:
        into.zero_()


def first_consumer(net: nn.Module, nz: int) -> tuple[str, nn.Module]:
    """The first module that takes `nz` inputs. Raises rather than guessing."""
    for name, module in net.named_modules():
        if not isinstance(module, CONSUMERS):
            continue
        width = (module.in_features if isinstance(module, nn.Linear)
                 else module.in_channels)
        if width == nz:
            return name, module
    raise ValueError(
        f"no Linear/Conv2d/ConvTranspose2d in this generator takes {nz} inputs, so there is "
        f"no first z-consumer to factorise. Types seen: "
        f"{sorted({type(m).__name__ for m in net.modules()})}")


def sefa(net: nn.Module, nz: int, count: int | None = None) -> Directions:
    """SeFa: closed-form factorisation of the first `z` consumer's weight."""
    name, module = first_consumer(net, nz)
    weight = module.weight.detach()
    transposed = isinstance(module, nn.ConvTranspose2d)
    return _factorise(_rows(weight, transposed), name, tuple(weight.shape), count)


def _rows(w: torch.Tensor, transposed: bool) -> torch.Tensor:
    """A weight as `(nz, everything else)`: one row per latent dimension."""
    if transposed:
        return w.reshape(w.shape[0], -1)
    return w.reshape(w.shape[0], w.shape[1], -1).permute(1, 0, 2).reshape(w.shape[1], -1)


def _factorise(rows: torch.Tensor, name: str, shape, count) -> Directions:
    """The factorisation itself, shared by the torch and the ONNX readers."""
    # `eigh` on the small `(nz, nz)` Gram rather than an SVD of the wide weight: same vectors.
    rows = rows.float().cpu()
    # Columns to unit length first, as the original implementation does, so every output
    # unit votes once instead of the loudest voting for everything.
    rows = rows / rows.norm(dim=0, keepdim=True).clamp_min(1e-12)
    basis, s = _top(rows @ rows.T, count)
    return Directions(basis=basis, strength=s, source=name, shape=tuple(shape))


def _top(gram: torch.Tensor, count: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """The leading eigenvectors as unit rows, strongest first, with their singular values.
    Used for SeFa's Gram of a weight and for the image metric alike."""
    values, vectors = torch.linalg.eigh(gram.float())
    order = torch.argsort(values, descending=True)          # eigh returns ascending
    if count is not None:
        order = order[:count]
    return vectors[:, order].T.contiguous(), values[order].clamp_min(0.0).sqrt()


def _laid_out(per_band, seats: int, width: int, z_dim: int, source: str) -> Directions:
    """Rows from several bands as one wide basis, each row live in exactly one band.

    `per_band` is `(slot, name, rows, strengths)` per non-empty band; `slot` is which row of
    the push buffer the band's rows write."""
    placed, strengths, came_from, sources = [], [], [], []
    for slot, name, rows, s in per_band:
        for row, one in zip(rows, s, strict=True):
            seat = torch.zeros(seats, width)
            seat[slot] = row
            placed.append(seat.reshape(-1))
            strengths.append(one)
            came_from.append(name)
        sources.append(f"{name}x{len(rows)}")
    if not placed:
        raise ValueError(f"no {source} carry any weight, so there is nothing to factorise")
    return Directions(basis=torch.stack(placed), strength=torch.stack(strengths),
                      source=f"{source} ({', '.join(sources)})", shape=(seats, width),
                      z_dim=z_dim, push_shape=(seats, width), ranges=tuple(came_from))


def sefa_banded(bands, counts, z_dim: int) -> Directions:
    """SeFa per style range of a StyleGAN2, laid out as rows of one wide basis. `bands` is
    `(name, stacked affine weights)` per range, as `stylegan2.style_bands` gives them."""
    width = max(int(w.shape[1]) for _name, w in bands)
    per_band = []
    for slot, ((name, weight), count) in enumerate(zip(bands, counts, strict=True)):
        if weight.shape[0] == 0 or not count:
            continue
        one = _factorise(_rows(weight, transposed=False), name, tuple(weight.shape), count)
        per_band.append((slot, name, one.basis, one.strength))
    return _laid_out(per_band, len(bands), width, z_dim, "style affines")


#: The difference image is average-pooled by this before the Gram, so the metric is
#: `(PJ)^T(PJ)` for a pooling operator `P`. It also keeps the differences small enough to hold.
PROBE_POOL = 8

#: The step the Jacobian is read at. The error is second-order in it until fp32 cancellation
#: takes over; at 0.25 it is about 1e-3 relative, far below the differences being ranked.
PROBE_EPS = 0.25


def metric(net: nn.Module, nz: int, device, dtype, into=None, slot: int = 0,
           eps: float = PROBE_EPS, seeds: int = 1, pool: int = PROBE_POOL) -> torch.Tensor:
    """`E[J^T J]` where `J` is the whole generator's Jacobian, for push row `slot`.

    Also called the active subspace, or the Riemannian metric of the image manifold; SeFa is
    this matrix for a generator one layer deep.

    No backward pass: the symmetric difference `d_i = f(x + eps e_i) - f(x - eps e_i)` is
    `2 eps J e_i`, so the Gram of the difference images is `4 eps^2 J^T J`. That costs
    `2 * width` forwards per latent and stores no activations.
    """
    width = nz if into is None else int(into.shape[1])
    # One flat push vector, reused: as wide as the latent, or as the whole push buffer.
    step = torch.zeros(width if into is None else into.numel(), device=device, dtype=dtype)
    seat = 0 if into is None else slot * width
    total = torch.zeros(width, width, dtype=torch.float64)

    def side(z, i: int, sign: float) -> torch.Tensor:
        """One probe's picture, pooled before the difference: pooling is linear, and the two
        sides then never coexist at full resolution."""
        step[seat + i] = sign
        got = pushed(net, z, step, into)
        step[seat + i] = 0.0
        return torch.nn.functional.avg_pool2d(
            got.reshape(1, -1, *got.shape[-2:]).float(), pool).flatten()

    with torch.no_grad():
        for k in range(seeds):
            z = latent(nz, k, device, dtype)
            # Written into one buffer rather than stacked from a list, which would hold a
            # second copy on the card.
            rows = None
            for i in range(width):
                d = side(z, i, eps) - side(z, i, -eps)
                if rows is None:
                    rows = torch.empty(width, len(d), device=d.device, dtype=d.dtype)
                rows[i] = d
            # Widened to fp64 on the host: not every card has fp64, and the widening is exact.
            total += (rows @ rows.T).cpu().double() / (4 * eps * eps)
            del rows                                  # before the next seed's passes
    return total / seeds


def active_banded(net: nn.Module, names, counts, z_dim: int, device, dtype, into,
                  seeds: int = 1, eps: float = PROBE_EPS) -> Directions:
    """`sefa_banded`'s layout, with each band's basis read off the whole generator's
    Jacobian (`metric`) instead of the first affine.

    Worth its cost in `w` space, where the style affine is a poor proxy for the synthesis
    network behind it: on FFHQ-1024 it takes the strongest fine dial from 52 8-bit levels to
    76. In `z` space SeFa does as well. One latent is enough: different latents pick
    different members of the same set of strong directions.
    """
    # The band layout is the push buffer's shape.
    seats, width = int(into.shape[0]), int(into.shape[1])
    per_band = [(slot, name, *_top(metric(net, z_dim, device, dtype, into=into, slot=slot,
                                          seeds=seeds, eps=eps), count))
                for slot, (name, count) in enumerate(zip(names, counts, strict=True)) if count]
    return _laid_out(per_band, seats, width, z_dim, "image metric")


#: What a derived basis is saved as, beside the checkpoint it belongs to. Only the proposal is
#: stored: `shortlist`, `equalise` and `rank` still run on it at load.
CACHE_SUFFIX = ".directions.pt"


def cache_path(checkpoint) -> pathlib.Path:
    return pathlib.Path(checkpoint).with_suffix(CACHE_SUFFIX)


def fingerprint(net: nn.Module) -> str:
    """Enough of a model's weights to tell it from another with the same filename, such as a
    fine-tune written over its own checkpoint: the sums of the first four float tensors."""
    state = getattr(net, "state_dict", None)
    if state is None:
        return ""                    # a graph, not a module: nothing to read, so nothing to check
    seen = []
    for _name, value in sorted(state().items()):
        if value.is_floating_point() and value.numel() > 1:
            # Accumulated in fp32 rather than cast to it, which would copy the whole tensor.
            seen.append(float(value.detach().sum(dtype=torch.float32)))
        if len(seen) == 4:
            break
    return ",".join(f"{v:.6g}" for v in seen)


def save(dirs: Directions, checkpoint, net: nn.Module, how: str = "") -> pathlib.Path:
    """Write the proposal beside its checkpoint, with `how` it was derived. Measured fields
    are not kept."""
    path = cache_path(checkpoint)
    torch.save({"basis": dirs.basis, "strength": dirs.strength, "source": dirs.source,
                "shape": list(dirs.shape), "z_dim": dirs.z_dim, "how": how,
                "of": fingerprint(net),
                "push_shape": list(dirs.push_shape or ()), "ranges": list(dirs.ranges or ())},
               path)
    return path


def saved(checkpoint, z_dim: int, push_shape, net: nn.Module) -> Directions | None:
    """The cached proposal, or `None` if there is none, or it is not this model's.

    How it was derived is appended to its `source`, so the load line names it."""
    path = cache_path(checkpoint)
    if not path.exists():
        return None
    try:
        got = torch.load(path, weights_only=True)
        back = Directions(basis=got["basis"], strength=got["strength"],
                          source=got["source"] + (f", {got['how']}" if got["how"] else ""),
                          shape=tuple(got["shape"]), z_dim=got["z_dim"],
                          push_shape=tuple(got["push_shape"]) or None,
                          ranges=tuple(got["ranges"]) or None)
        made_from = got["of"]
    except Exception as exc:                                         # noqa: BLE001
        print(f"ignoring {path.name}: it cannot be read ({type(exc).__name__}), so the "
              f"directions are derived again", flush=True)
        return None
    # `nz` rather than `z_dim`: a z-space basis leaves `z_dim` unset.
    if back.nz != z_dim or back.push_shape != push_shape:
        print(f"ignoring {path.name}: it holds a {back.nz}-wide {back.space}-basis for a push "
              f"buffer {back.push_shape}, and this model wants {z_dim} and {push_shape}",
              flush=True)
        return None
    mine = fingerprint(net)
    if mine and made_from and made_from != mine:
        print(f"ignoring {path.name}: it was derived from different weights under this name. "
              f"Run `ganlive dials` again if this checkpoint has been retrained.", flush=True)
        return None
    return back


#: Ops that move a latent about without mixing its dimensions. Walking through them is what
#: lets the reader start at the graph input and still find the first real consumer.
PASSTHROUGH = ("Reshape", "Squeeze", "Unsqueeze", "Identity", "Flatten", "Cast")

#: Ops that scale a weight without changing what it spans, such as spectral norm's
#: `weight / sigma` in an export.
RESCALE = ("Div", "Mul")

#: Ops the latent may pass through on its way to the first layer with a weight.
NORMALISE = ("Pow", "Sqrt", "ReduceMean", "ReduceSum", "ReduceL2", "Add", "Sub", "Div", "Mul",
             "Reciprocal", "Neg", "Expand", "Constant", "ConstantOfShape", "Shape",
             "LpNormalization", "InstanceNormalization", "MeanVarianceNormalization")


def sefa_onnx(path, nz: int, count: int | None = None) -> Directions:
    """The same factorisation, read off an ONNX file with no torch model anywhere."""
    import onnx

    # The structure, then the one tensor needed, rather than every weight in the file.
    graph = structure(path).graph
    name, weight, op = _first_onnx_consumer(graph, initializers(graph), producers(graph))
    if weight.HasField("data_location") and weight.data_location == onnx.TensorProto.EXTERNAL:
        onnx.external_data_helper.load_external_data_for_tensor(
            weight, str(pathlib.Path(path).parent))
        weight.data_location = onnx.TensorProto.DEFAULT   # or `to_array` looks it up again

    w = torch.from_numpy(onnx.numpy_helper.to_array(weight).copy())
    if op in ("Conv", "ConvTranspose"):
        rows = _rows(w, transposed=op == "ConvTranspose")
    else:
        # Gemm and MatMul are two-dimensional; the latent axis is whichever one is `nz`.
        rows = w if w.shape[0] == nz else w.T
    if rows.shape[0] != nz:
        raise ValueError(f"{name} is a {op} of shape {tuple(w.shape)}, whose latent axis is "
                         f"not {nz} wide; refusing to factorise the wrong matrix")
    return _factorise(rows, name, tuple(w.shape), count)


#: The four ops that can be the first affine consumer of a latent, in any exported generator.
AFFINE = ("Conv", "ConvTranspose", "Gemm", "MatMul")


def _first_onnx_consumer(graph, initial, producer):
    """`(name, weight initializer, op type)` for the first node that really consumes `z`.
    The latent's axis of the weight is checked by `sefa_onnx`."""
    live = {graph.input[0].name}
    for node in graph.node:
        if not live.intersection(node.input):
            continue
        if node.op_type in PASSTHROUGH + NORMALISE:
            live.update(node.output)
            continue
        if node.op_type not in AFFINE:
            raise ValueError(
                f"the latent reaches a {node.op_type} before it reaches anything with a "
                f"weight, so there is no first consumer to factorise")
        for name in node.input:
            if name in live:
                continue
            found = _initializer(graph, name, initial, producer)
            if found is not None:
                return node.name or node.op_type, found, node.op_type
        raise ValueError(f"{node.op_type} consumes the latent but none of its other inputs "
                         f"resolve to a stored weight")
    raise ValueError(f"nothing in this graph consumes {graph.input[0].name}")


def _initializer(graph, name: str, initial, producer, depth: int = 8):
    """The stored weight behind a name, seeing through the scaling spectral norm leaves."""
    for _ in range(depth):
        if name in initial:
            return initial[name]
        if name not in producer:
            return None
        node = graph.node[producer[name]]
        if node.op_type not in RESCALE + PASSTHROUGH + ("Transpose",):
            return None
        name = node.input[0]
    return None


def split(count: int, bands: int) -> tuple[int, ...]:
    """`count` directions shared over `bands` style ranges, the remainder to the later ones."""
    base, extra = divmod(count, bands)
    return tuple(base + (i >= bands - extra) for i in range(bands))


def middle(up: list[float], down: list[float]) -> list[float]:
    """What a dial is worth over the travel it has: the mean of its two halves. `equalise`
    scales on it and `rank` scores with it."""
    return [(u + d) / 2 for u, d in zip(up, down, strict=True)]


def band_names(row_bands) -> tuple[str | None, ...]:
    """The bands a row list covers, in first-seen order (the ladder's), not the alphabet's.
    The order matters: `split` gives the remainder to the later bands."""
    return tuple(dict.fromkeys(row_bands))


def _take(dirs: Directions, rows: list[int], **measured) -> Directions:
    """`dirs` narrowed to `rows`, with every row-aligned field taken along. `measured` sets
    fields a caller computed fresh."""
    fields = dict(basis=dirs.basis[rows], strength=dirs.strength[rows],
                  ranges=None if dirs.ranges is None
                  else tuple(dirs.ranges[i] for i in rows),
                  halves=None if dirs.halves is None
                  else tuple(dirs.halves[i] for i in rows))
    return replace(dirs, **(fields | measured))


def share(order: list[int], row_bands, keep: int) -> list[int]:
    """Take from `order`, already in priority order, until each band has had its share of
    `keep`. The caller decides what "best" means; `shortlist` and `rank` share this one rule."""
    names = band_names(row_bands)
    room = dict(zip(names, split(keep, len(names)), strict=True))
    taken = []
    for i in order:
        if room[row_bands[i]] > 0:
            room[row_bands[i]] -= 1
            taken.append(i)
    return taken


def equalise(net: nn.Module, dirs: Directions, device, dtype, amount: float,
             target: float, into=None) -> Directions:
    """Scale a `w` basis so its median direction moves `target` 8-bit levels at full travel,
    measured over both halves and `SEEDS` latents. Fewer latents leave the scale noisy.

    A `z` basis is returned untouched: a unit row is already the unit the latent was
    trained on.
    """
    if dirs.space == "z":
        return dirs
    levels = sorted(middle(*travel(net, dirs, device, dtype, amount, into=into)))
    mid = levels[len(levels) // 2]
    if mid <= 0.0:
        return dirs
    return replace(dirs, basis=dirs.basis * (target / mid))


#: Latents a level is averaged over. One latent alone can read half or double the mean.
SEEDS = 4

#: Random directions the bar averages, per band: the smallest count at which the keep-or-drop
#: verdict stopped changing. Each is measured one way only; `random_like` draws half of them
#: negative, which is sound because a random direction has no preferred sign.
RANDOM_PROBES = 8


def travel(net: nn.Module, dirs: Directions, device, dtype, amount: float,
           into=None, seeds: int = SEEDS) -> tuple[list[float], list[float]]:
    """What each row moves at `+amount` and at `-amount`. A direction dial travels both ways,
    and an eigenvector's sign is arbitrary, so both halves are measured."""
    return tuple(_measure(net, dirs, device, dtype, amount, 0, into, seeds,
                          signs=(1.0, -1.0)))


def orient(dirs: Directions, up: list[float], down: list[float]) -> tuple:
    """Turn every row so its stronger half is the one reached turning the dial up. Returns
    the turned basis and the halves, stronger first."""
    flip = torch.tensor([-1.0 if d > u else 1.0 for u, d in zip(up, down, strict=True)])
    basis = dirs.basis * flip.reshape(-1, *([1] * (dirs.basis.dim() - 1)))
    return (replace(dirs, basis=basis),
            [max(u, d) for u, d in zip(up, down, strict=True)],
            [min(u, d) for u, d in zip(up, down, strict=True)])


def shortlist(net: nn.Module, dirs: Directions, device, dtype, amount: float,
              into=None, keep: int = 16) -> Directions:
    """Cut a wide candidate pool down to `keep` rows worth measuring properly, shared over
    the bands. One latent and one sign: cheap, since everything after this costs `SEEDS`
    latents and two renders a row, and it only has to avoid dropping a row that would win.
    """
    if len(dirs) <= keep:
        return dirs
    levels = verify(net, dirs, device, dtype=dtype, amount=amount, into=into, seeds=1)
    order = sorted(range(len(levels)), key=lambda i: -levels[i])
    take = share(order, dirs.row_bands, keep)
    take.sort()                              # the pool's own order, so `rank` still sorts it
    return _take(dirs, take)


def random_like(dirs: Directions, seed: int = 1, count: int = RANDOM_PROBES) -> Directions:
    """Random rows with the same support and length as `dirs`' rows, `count` per band: the
    bands are not equally sensitive, so each gets its own bar. Half are drawn negative, so a
    one-sided measurement of them is a bar over both signs, like the candidates'.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = dirs.row_bands
    rows, names = [], []
    for name in dirs.band_names:
        here = [row for row, group in zip(dirs.basis, groups, strict=True) if group == name]
        for i in range(count):
            template = here[i % len(here)]
            noise = torch.randn(template.shape, generator=generator) * (template != 0)
            sign = -1.0 if i % 2 else 1.0
            rows.append(sign * noise / noise.norm().clamp_min(1e-12) * template.norm())
            names.append(name)
    # Clear the measurement, which `levels` is read from.
    return replace(dirs, basis=torch.stack(rows), strength=torch.ones(len(rows)),
                   ranges=None if dirs.ranges is None else tuple(names), halves=None)


#: Candidates offered per band, before measurement picks among them. In `w` space the
#: eigenvalue order is a poor selector: a row seventh by eigenvalue can be the strongest
#: measured.
CANDIDATES = 16


def rank(net: nn.Module, dirs: Directions, device, dtype, amount: float,
         floor: float = FLOOR_LEVELS, into=None, relative: float = RANDOM_FLOOR,
         keep_best: int | None = None) -> Directions:
    """Re-order a basis by what it actually does, and drop the rows that do nothing -- or
    that do no more than a random direction of the same length would.

    `keep_best` caps how many survive, shared over the bands the way `split` shares them, so a
    wide candidate pool can be offered without handing the player forty encoders."""
    dirs, up, down = orient(dirs, *travel(net, dirs, device, dtype, amount, into=into))
    levels = middle(up, down)
    means: dict[str | None, float] = {}
    if relative > 0.0 and len(dirs):
        # One-sided: `random_like` has drawn half its rows negative. See `RANDOM_PROBES`.
        probe = random_like(dirs)
        probe_levels = verify(net, probe, device, dtype=dtype, amount=amount, into=into)
        per_band: dict[str | None, list[float]] = {}
        for name, level in zip(probe.row_bands, probe_levels, strict=True):
            per_band.setdefault(name, []).append(level)
        means = {name: sum(got) / len(got) for name, got in per_band.items()}
    # Each band against its own bar; `floor` stays the absolute one, in levels.
    bars = {name: max(floor, relative * mean) for name, mean in means.items()}
    # Sorted inside each band, not across them, so a band's dials sit together.
    bands, rows = dirs.band_names, dirs.row_bands
    order = sorted(range(len(levels)), key=lambda i: (bands.index(rows[i]), -levels[i]))
    keep = [i for i in order if levels[i] >= bars.get(rows[i], floor)]
    if keep_best is not None:                   # `order` is already band, then best first
        keep = share(keep, rows, keep_best)
    return _take(dirs, keep,
                 halves=tuple((round(up[i], 2), round(down[i], 2)) for i in keep),
                 random_levels=None if not means
                 else round(sum(means.values()) / len(means), 2),
                 random_by_range=None if not means or not dirs.ranges
                 else tuple((name, round(mean, 2)) for name, mean in means.items()),
                 amount=amount,
                 dropped=len(dirs) - len(keep))


def verify(net: nn.Module, dirs: Directions, device, dtype=torch.float16,
           amount: float = 2.0, seed: int = 0, into=None, seeds: int = SEEDS) -> list[float]:
    """Mean 8-bit levels each direction moves the picture, averaged over `seeds` latents
    drawn from `seed` on. The inert-direction detector."""
    return _measure(net, dirs, device, dtype, amount, seed, into, seeds, signs=(1.0,))[0]


def _measure(net: nn.Module, dirs: Directions, device, dtype, amount: float, seed: int,
             into, seeds: int, signs: tuple[float, ...]) -> list[list[float]]:
    """One list of levels per sign in `signs`, over the same latents. The signs share each
    latent's image at rest."""
    if (into is None) != (dirs.space == "z"):
        raise ValueError(
            f"a {dirs.space}-space basis {'needs' if into is None else 'has no use for'} a "
            f"push buffer; `into` was {'not ' if into is None else ''}given")
    # On the push buffer's device, so a push is not a blocking upload per forward.
    basis = (dirs.basis.to(device=into.device) if into is not None
             else dirs.basis.to(device=device, dtype=dtype))
    # Each reading stays on the card until every forward is queued, and is read back once:
    # a `.item()` per forward would stall the queue.
    got = torch.empty(seeds, len(signs), len(basis), dtype=torch.float32, device=basis.device)
    with torch.no_grad():
        for k in range(seeds):
            z = latent(dirs.nz, seed + k, device, dtype)
            base = pushed(net, z, None, into, dirs.push_shape)
            for s, sign in enumerate(signs):
                for i, row in enumerate(basis):
                    moved = pushed(net, z, (sign * amount) * row, into, dirs.push_shape)
                    got[k, s, i] = (moved - base).abs().mean(dtype=torch.float32)
    vals = got.cpu().tolist()
    totals = [[0.0] * len(basis) for _ in signs]
    for k in range(seeds):
        for s in range(len(signs)):
            for i in range(len(basis)):
                totals[s][i] += vals[k][s][i] * LEVEL
    return [[t / seeds for t in row] for row in totals]
