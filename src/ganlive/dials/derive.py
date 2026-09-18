"""Latent directions read out of the weights, with no sampling and no labels."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, replace

import torch
from torch import nn

from ganlive.models.fastgan import first_image

#: The layer types that can be the first thing a latent meets.
CONSUMERS = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)


@dataclass(frozen=True)
class Directions:
    """An ordered basis for the latent, and where it was read from."""

    #: `(n, nz)`, unit rows. By singular value from `sefa`, by measured effect after `rank`.
    basis: torch.Tensor
    #: Singular value per row. A flat spectrum means the tail directions are arbitrary.
    strength: torch.Tensor
    #: The module the basis came from, for the report.
    source: str
    #: What that module's weight looked like, so a surprise is visible rather than assumed.
    shape: tuple[int, ...]
    #: The generator's latent width, set only when these rows are **not** latents: a `w` basis
    #: has to say, because nothing else then knows how wide a latent to draw.
    z_dim: int | None = None
    #: The shape a row takes at the seam -- `(ranges, width)` -- or `None` for a `z` basis.
    push_shape: tuple[int, int] | None = None
    #: Which style range each row came out of, in row order.
    ranges: tuple[str, ...] | None = None
    #: What each kept row moves turning up and turning down. `orient` has been through them, so
    #: the first is always the larger; the ratio is how lopsided the dial is.
    halves: tuple[tuple[float, float], ...] | None = None
    #: The bar: what a *random* unit direction of the same length moves, pooled over the bands.
    random_levels: float | None = None
    #: The same bar per style range, because the ranges are not equally sensitive -- on FFHQ
    #: 6.2, 4.3 and 6.7 levels. `None` without ranges, where the pooled bar *is* the band's.
    random_by_range: tuple[tuple[str, float], ...] | None = None
    #: The push every level here was measured at. **The ratio to random is not scale-invariant**
    #: -- four FFHQ directions read 4.31x at an eighth of the strip's travel and 1.56x at eight
    #: times it -- so a report that omits the amplitude has not said anything.
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
    def bands(self) -> int:
        return 1 if self.push_shape is None else self.push_shape[0]

    @property
    def row_bands(self) -> tuple[str | None, ...]:
        """The band each row came from. **A basis with no ranges is a basis with one anonymous
        band** -- said once here, because five re-spellings of it had drifted apart."""
        return self.ranges or (None,) * len(self)

    @property
    def band_names(self) -> tuple[str | None, ...]:
        """The bands in the ladder's order, not the alphabet's, which puts `w_fine` first."""
        return band_names(self.row_bands)

    @property
    def levels(self) -> tuple[float, ...] | None:
        """Mean 8-bit levels each row moves the picture; `None` until `rank` has measured it.

        **Derived, not stored.** It is `middle` of the two halves, and `middle` says in its own
        docstring that it is a definition rather than an expression -- keeping the mean beside
        the halves it is the mean of put that definition back in two places, one of them a
        column that could be written out of step with the other."""
        if self.halves is None:
            return None
        return tuple(middle([u for u, _ in self.halves], [d for _, d in self.halves]))

    @property
    def intrinsic_scale(self) -> bool:
        """Whether "one unit along a unit row" already means something on this model."""
        return self.z_dim is None

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


def _latent(nz: int, seed: int, device, dtype) -> torch.Tensor:
    """One latent, drawn on the host so the picture does not depend on the card it ran on."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, nz, generator=generator).to(device=device, dtype=dtype)


def pushed(net: nn.Module, z: torch.Tensor, push, into, push_shape=None) -> torch.Tensor:
    """The picture with `push` applied -- added to the latent, or written to the seam.

    `push=None` is the model at rest, and **the seam is cleared for it**: zeroed once outside a
    loop instead, only the first reference was at rest and every later one carried the previous
    row's push, so three of four readings measured one direction against another. That lesson
    is one line here rather than one line in each caller that renders.
    """
    if into is None:
        return first_image(net(z if push is None else z + push.unsqueeze(0)))
    if push is None:
        into.zero_()
    else:
        into.copy_(push.reshape(push_shape or into.shape))
    return first_image(net(z))


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
    """Closed-form factorisation of the first `z` consumer's weight."""
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
    """The SVD itself, shared by the torch and the ONNX readers so they cannot disagree."""
    # `eigh` on the `(nz, nz)` Gram, not an SVD of the `(nz, 49152)` weight: same `U`.
    rows = rows.float().cpu()
    # Columns to unit length first, as the reference does: every output unit then votes once,
    # instead of the loudest voting for everything. On FFHQ's affines it moves directions 6-8.
    rows = rows / rows.norm(dim=0, keepdim=True).clamp_min(1e-12)
    basis, s = _top(rows @ rows.T, count)
    return Directions(basis=basis, strength=s, source=name, shape=tuple(shape))


def _top(gram: torch.Tensor, count: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """The leading eigenvectors as unit rows, strongest first, with their singular values.

    One spelling for both derivations: SeFa's Gram of a weight and the image metric are the
    same kind of matrix, and "leading" has to mean the same thing in each or the two bases
    cannot be compared."""
    values, vectors = torch.linalg.eigh(gram.float())
    order = torch.argsort(values, descending=True)          # eigh returns ascending
    if count is not None:
        order = order[:count]
    return vectors[:, order].T.contiguous(), values[order].clamp_min(0.0).sqrt()


def _laid_out(per_band, seats: int, width: int, z_dim: int, source: str) -> Directions:
    """Rows from several bands as one wide basis, each row live in exactly one band.

    `per_band` is `(slot, name, rows, strengths)`, empty bands simply left out -- the slot is
    carried rather than implied by position, so a caller never has to pad. Shared because a row
    placed in the wrong seat is a dial that steers another band's layers, which nothing
    downstream can see."""
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


def sefa_banded(bands, counts, z_dim: int, source: str = "style affines") -> Directions:
    """One factorisation per style range, laid out as rows of a single wide basis."""
    width = max(int(w.shape[1]) for _name, w in bands)
    per_band = []
    for slot, ((name, weight), count) in enumerate(zip(bands, counts, strict=True)):
        if weight.shape[0] == 0 or not count:
            continue
        one = _factorise(_rows(weight, transposed=False), name, tuple(weight.shape), count)
        per_band.append((slot, name, one.basis, one.strength))
    return _laid_out(per_band, len(bands), width, z_dim, source)


#: The difference image is average-pooled by this before the Gram. That makes the metric
#: `(PJ)^T(PJ)` for a pooling operator `P` -- a stated choice, and the one every perceptual
#: metric makes as well. It also keeps one band's differences at 151 MB instead of 9.7 GB.
PROBE_POOL = 8

#: The step the Jacobian is read at. The estimator is exact to floating point on a linear map
#: and second-order on a nonlinear one, so this is a bias/noise trade: measured against
#: autograd's own Jacobian the relative error falls as `eps^2` to 7e-6 at 0.01 and then rises
#: again as fp32 cancellation takes over, reaching 5.9e-4 at 1e-4. At 0.25 it is 1.3e-3, three
#: orders below the differences this is used to rank.
PROBE_EPS = 0.25


def metric(net: nn.Module, nz: int, device, dtype, into=None, slot: int = 0,
           eps: float = PROBE_EPS, seeds: int = 1, pool: int = PROBE_POOL) -> torch.Tensor:
    """`E[J^T J]` where `J` is the whole generator's Jacobian, not one layer's weight.

    Uncertainty quantification calls this the active subspace and Wang & Ponce call it the
    Riemannian metric of the image manifold; for a vector output they are the same matrix, and
    SeFa is this matrix for a generator one layer deep.

    **No backward pass.** With `v_i = e_i` the symmetric difference
    `d_i = f(x + eps e_i) - f(x - eps e_i)` is `2 eps J e_i`, so `d_i . d_j = 4 eps^2 C_ij`:
    the Gram of the difference images *is* the metric. That costs `2 * width` forwards per
    latent and stores no activations, on a card that has frozen this machine once.
    """
    width = nz if into is None else int(into.shape[1])
    # One flat push vector, reused: as wide as the latent, or as the whole seam.
    step = torch.zeros(width if into is None else into.numel(), device=device, dtype=dtype)
    seat = 0 if into is None else slot * width
    total = torch.zeros(width, width, dtype=torch.float64)

    def side(z, i: int, sign: float) -> torch.Tensor:
        """One probe's picture, pooled on the way out.

        **Pooled before the difference, not after.** Average pooling is linear, so
        `pool(a) - pool(b)` is `pool(a - b)`, and doing it here means the two sides never
        coexist at full resolution: 0.6 MB held instead of 57, and the widening to fp32 is
        paid on 1/64th of the pixels."""
        step[seat + i] = sign
        got = pushed(net, z, step, into)
        step[seat + i] = 0.0
        return torch.nn.functional.avg_pool2d(
            got.reshape(1, -1, *got.shape[-2:]).float(), pool).flatten()

    with torch.no_grad():
        for k in range(seeds):
            z = _latent(nz, k, device, dtype)
            # Written into one buffer rather than stacked from a list: `torch.stack` would
            # allocate and copy a second 151 MB on the card, doubling this function's peak.
            rows = None
            for i in range(width):
                d = side(z, i, eps) - side(z, i, -eps)
                if rows is None:
                    rows = torch.empty(width, len(d), device=d.device, dtype=d.dtype)
                rows[i] = d
            # Widened after the copy, not before: this card has no fp64, so `.double()` on it
            # is emulated at best, and the transfer is half the size in fp32. fp32 to fp64 is
            # exact, so where it happens cannot change the number.
            total += (rows @ rows.T).cpu().double() / (4 * eps * eps)
            del rows                                  # before the next seed's 1,024 passes
    if into is not None:
        into.zero_()
    return total / seeds


def active_banded(net: nn.Module, names, counts, z_dim: int, device, dtype, into,
                  seeds: int = 1, eps: float = PROBE_EPS,
                  source: str = "image metric") -> Directions:
    """`sefa_banded`'s basis, read off the whole generator instead of the first affine.

    **Worth the price only in W.** On a z-space FastGAN the first layer is nearly the whole
    story and SeFa wins: 8 directions over the bar against this basis's 6. In W it is the other
    way round, because the style affine is a poor proxy for the synthesis network behind it --
    on FFHQ-1024 this takes `w_fine`'s strongest dial from 52 8-bit levels to 76, `w_mid`'s
    three from 17/16/14 to 20/20/17, and ties in `w_coarse`.

    One latent is enough for the verdict even though it is not enough for the basis: over
    disjoint sets of four latents the top-8 subspace agrees at only 0.38 to 0.76, but every
    subset tried keeps `w_fine` between 67 and 77 levels. There are more strong directions than
    the strip can show and different latents pick different members of the same set.
    """
    # The seam is the layout: this derivation never opens a weight, so taking the band widths
    # from the thing the rows are pushed through is both shorter and impossible to disagree with.
    seats, width = int(into.shape[0]), int(into.shape[1])
    per_band = [(slot, name, *_top(metric(net, z_dim, device, dtype, into=into, slot=slot,
                                          seeds=seeds, eps=eps), count))
                for slot, (name, count) in enumerate(zip(names, counts, strict=True)) if count]
    return _laid_out(per_band, seats, width, z_dim, source)


#: What a derived basis is saved as, beside the checkpoint it belongs to. **A cache, not a
#: shortcut past the gate**: what is stored is the proposal, and `shortlist`, `equalise` and
#: `rank` still run on it at load, so a stale or wrong file cannot put a dead dial on the strip.
CACHE_SUFFIX = ".directions.pt"


def cache_path(checkpoint) -> pathlib.Path:
    return pathlib.Path(checkpoint).with_suffix(CACHE_SUFFIX)


def fingerprint(net: nn.Module) -> str:
    """Enough of a model's weights to tell it from another one with the same filename.

    A cache is keyed by a path, and a path is not an identity: a fine-tune written back over
    its own checkpoint keeps the same name, the same latent width and the same seam, so a shape
    check waves the stale basis through. Four tensors' sums are not a hash of the file, and do
    not need to be -- they only have to move when the weights move."""
    state = getattr(net, "state_dict", None)
    if state is None:
        return ""                    # a graph, not a module: nothing to read, so nothing to check
    seen = []
    for _name, value in sorted(state().items()):
        if value.is_floating_point() and value.numel() > 1:
            # Accumulated in fp32 rather than cast to it: `.float()` first materialises a
            # second copy of a whole conv weight, on the load path, to produce one number.
            seen.append(float(value.detach().sum(dtype=torch.float32)))
        if len(seen) == 4:
            break
    return ",".join(f"{v:.6g}" for v in seen)


def save(dirs: Directions, checkpoint, net: nn.Module, how: str = "") -> pathlib.Path:
    """Write the proposal beside its checkpoint. Measured fields are deliberately not kept."""
    path = cache_path(checkpoint)
    torch.save({"basis": dirs.basis, "strength": dirs.strength, "source": dirs.source,
                "shape": list(dirs.shape), "z_dim": dirs.z_dim, "how": how,
                "of": fingerprint(net),
                "push_shape": list(dirs.push_shape or ()), "ranges": list(dirs.ranges or ())},
               path)
    return path


def saved(checkpoint, z_dim: int, push_shape, net: nn.Module) -> Directions | None:
    """The cached proposal, or `None` if there is none, or it is not this model's.

    **It says what it read.** The settings a basis was derived at travel with it and land in
    `source`, so the load line names them; a basis derived at a tenth of the step is otherwise
    indistinguishable from a good one, and the report would say only "image metric"."""
    path = cache_path(checkpoint)
    if not path.exists():
        return None
    got = torch.load(path, weights_only=True)
    back = Directions(basis=got["basis"], strength=got["strength"],
                      source=got["source"] + (f", {got['how']}" if got.get("how") else ""),
                      shape=tuple(got["shape"]), z_dim=got["z_dim"],
                      push_shape=tuple(got["push_shape"]) or None,
                      ranges=tuple(got["ranges"]) or None)
    # `nz` rather than `z_dim`: a z-space basis leaves `z_dim` unset and is as wide as the
    # latent, so comparing the stored field would refuse every one of them.
    if back.nz != z_dim or back.push_shape != push_shape:
        print(f"ignoring {path.name}: it holds a {back.nz}-wide {back.space}-basis for a seam "
              f"{back.push_shape}, and this model wants {z_dim} and {push_shape}", flush=True)
        return None
    mine = fingerprint(net)
    if mine and got.get("of") and got["of"] != mine:
        print(f"ignoring {path.name}: it was derived from different weights under this name. "
              f"Run `ganlive dials` again if this checkpoint has been retrained.", flush=True)
        return None
    return back


#: Ops that move a latent about without mixing its dimensions. Walking through them is what
#: lets the reader start at the graph input and still find the first real consumer.
PASSTHROUGH = ("Reshape", "Squeeze", "Unsqueeze", "Identity", "Flatten", "Cast")

#: Ops that scale a weight without changing what it spans -- spectral norm's `weight / dot`
#: survives export as one of these, so the reader has to see through it to the initializer.
RESCALE = ("Div", "Mul")

#: Ops the latent may pass through on its way to the first layer with a weight.
NORMALISE = ("Pow", "Sqrt", "ReduceMean", "ReduceSum", "ReduceL2", "Add", "Sub", "Div", "Mul",
             "Reciprocal", "Neg", "Expand", "Constant", "ConstantOfShape", "Shape",
             "LpNormalization", "InstanceNormalization", "MeanVarianceNormalization")


def sefa_onnx(path, nz: int, count: int | None = None) -> Directions:
    """The same factorisation, read off an ONNX file with no torch model anywhere."""
    import onnx

    # Structure first, then one tensor: `load_external_data=True` reads all 191 MB of the
    # sibling `.onnx.data` to use one, resident twice while `core.read_model` holds it too.
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initial = {i.name: i for i in graph.initializer}
    producer = {out: node for node in graph.node for out in node.output}

    name, weight, op = _first_onnx_consumer(graph, initial, producer, nz)
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


def _first_onnx_consumer(graph, initial, producer, nz: int):
    """`(name, weight initializer, op type)` for the first node that really consumes `z`."""
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
            found = _initializer(name, initial, producer)
            if found is not None:
                return node.name or node.op_type, found, node.op_type
        raise ValueError(f"{node.op_type} consumes the latent but none of its other inputs "
                         f"resolve to a stored weight")
    raise ValueError(f"nothing in this graph consumes {graph.input[0].name}")


def _initializer(name: str, initial, producer, depth: int = 8):
    """The stored weight behind a name, seeing through the scaling spectral norm leaves."""
    for _ in range(depth):
        if name in initial:
            return initial[name]
        node = producer.get(name)
        if node is None or node.op_type not in RESCALE + PASSTHROUGH + ("Transpose",):
            return None
        name = node.input[0]
    return None


def split(count: int, bands: int) -> tuple[int, ...]:
    """`count` directions shared over `bands` style ranges, the remainder to the later ones."""
    base, extra = divmod(count, bands)
    return tuple(base + (i >= bands - extra) for i in range(bands))


def middle(up: list[float], down: list[float]) -> list[float]:
    """What a dial is worth over the travel it has: the mean of its two halves.

    A definition, not an expression: `equalise` scales on it and `rank` scores with it, so a
    change to it (a min, say) has to be a change in one place."""
    return [(u + d) / 2 for u, d in zip(up, down, strict=True)]


def band_names(row_bands) -> tuple[str | None, ...]:
    """The bands a row list covers, in the ladder's order rather than the alphabet's.

    A function as well as a `Directions` property because `share` holds the raw row list and
    not the basis, and the *order* is load-bearing: `split` gives the spare seat to the later
    bands, so a second spelling here would hand it to a different band than the report says."""
    return tuple(dict.fromkeys(row_bands))


def _take(dirs: Directions, rows: list[int], **measured) -> Directions:
    """`dirs` narrowed to `rows`, with every row-aligned field taken along.

    **One place that knows which fields are row-aligned.** Two callers built this by hand and
    had already disagreed about it; the class has grown those fields one at a time, and a row
    kept beside another row's band label is a dial that steers the wrong layers, which nothing
    downstream can see. `measured` is for what a caller computes fresh rather than subsets."""
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
    """Scale a whole basis so its median direction moves `target` 8-bit levels at full travel.

    Over both halves: a lopsided row scaled by its strong half alone gets a dial whose
    useful range is the top of one side.

    The four latents cost 132 forward passes and are load-bearing. The ranking survives
    fewer -- the bar moves with the probes -- but the strength does not: over disjoint
    draws the scale spreads 3.37x on one latent, 1.59x on two and 1.18x on four, and the
    same dial then ships at 55 levels or at 76. That is the number the hand feels.

    A basis whose scale already means something is returned untouched, and that test lives
    here rather than in the caller: rescaling a `z` basis whose rows are already the unit
    the model was trained on gives a working strip of the wrong strength, silently.
    """
    if dirs.intrinsic_scale:
        return dirs
    levels = sorted(middle(*travel(net, dirs, device, dtype, amount, into=into)))
    mid = levels[len(levels) // 2]
    if mid <= 0.0:
        return dirs
    return replace(dirs, basis=dirs.basis * (target / mid))


#: A direction moving less than this many 8-bit levels at full travel is not a control.
FLOOR_LEVELS = 1.0

#: Times what a random unit direction of the same length moves -- on the same latents, in the
#: pipeline that plays -- for a direction to be a control rather than a walk. **Relative, and
#: deliberately so**: `gv-warm-lr3` moves 70 levels along its best direction against this
#: model's 44 for its worst kept one, and still keeps fewer, because everything on that
#: checkpoint moves the picture that hard. See NOTES for how 2.5 became 2.0.
RANDOM_FLOOR = 2.0

#: Latents a level is averaged over. One is not a measurement: the same random direction reads
#: 18 levels on one latent and 36 on the next, and `dir1` 46 on one and 132 on another.
SEEDS = 4

#: Random directions the baseline averages, **per band**. Chosen by the smallest count at which
#: the *verdict* stops moving, not by the baseline's value: over four seeds FFHQ is unchanged
#: at 4, 6, 8, 12 and 16, while the FastGAN moves at every one of them because with a wide pool
#: its middle rows sit on the bar -- a fact about those rows, not about this constant.
#:
#: **A side is a render, not a free sample.** Eight probes both ways costs 200 forward passes
#: against sixteen one way at 196 -- the same price. What halves the work is drawing the sign
#: instead of rendering it (`random_like` negates half the rows, sound because a probe is
#: isotropic) and measuring one-sided for 100. The baseline was 14.0s of a 47s StyleGAN2 load.
RANDOM_PROBES = 8


def travel(net: nn.Module, dirs: Directions, device, dtype, amount: float,
           into=None, seeds: int = SEEDS) -> tuple[list[float], list[float]]:
    """What each row moves at `+amount` **and** at `-amount`.

    The strip's travel is symmetric, so the gate has to be. And the sign is arbitrary: `eigh`
    returns an eigenvector, not a ray, so which half got measured was whatever LAPACK handed
    back. Measured, `gv-2048-ft` ships two half-dials -- `dir2` at 48.7 levels one way and 31.1
    the other, `dir3` at 47.0 and 34.5.
    """
    up, down = _measure(net, dirs, device, dtype, amount, into=into, seeds=seeds,
                        signs=(1.0, -1.0))
    return up, down


def orient(dirs: Directions, up: list[float], down: list[float]) -> tuple:
    """Turn every row so its stronger half is the one the encoder reaches turning up.

    Free, and without it whether a dial's big move is clockwise is LAPACK's sign convention.
    On FFHQ four of seven shipping dials point the wrong way, `dir7` worst at 23.3 up, 38.6 down.
    """
    from dataclasses import replace

    flip = torch.tensor([-1.0 if d > u else 1.0 for u, d in zip(up, down, strict=True)])
    basis = dirs.basis * flip.reshape(-1, *([1] * (dirs.basis.dim() - 1)))
    return (replace(dirs, basis=basis),
            [max(u, d) for u, d in zip(up, down, strict=True)],
            [min(u, d) for u, d in zip(up, down, strict=True)])


def shortlist(net: nn.Module, dirs: Directions, device, dtype, amount: float,
              into=None, keep: int = 16) -> Directions:
    """Cut a wide candidate pool down to something worth measuring properly.

    **A pool is cheap to propose and dear to judge**: everything after this costs four latents
    and two renders a row, and running it over forty-eight candidates took a StyleGAN2's load
    from 40 to 72 seconds, on a shelf change with the picture stopped. So one latent and one
    direction here -- not a verdict, it only has to avoid throwing away a row that would have
    won -- and `keep` leaves room above what the strip can show.
    """
    if len(dirs) <= keep:
        return dirs
    levels = verify(net, dirs, device, dtype=dtype, amount=amount, into=into, seeds=1)
    order = sorted(range(len(levels)), key=lambda i: -levels[i])
    take = share(order, dirs.row_bands, keep)
    take.sort()                              # the pool's own order, so `rank` still sorts it
    return _take(dirs, take)


def random_like(dirs: Directions, seed: int = 1, count: int = RANDOM_PROBES) -> Directions:
    """Random unit rows with the same support as `dirs`' rows -- `count` of them per band.

    **Per band**: the bands are not equally sensitive (on FFHQ 6.2, 4.3 and 6.7 levels), so one
    pooled mean asked a `w_mid` dial to clear a bar half again its own band's.

    **Half the rows are drawn negative**, which is where the two-sidedness of the bar comes
    from: a candidate is scored over both halves, so its bar has to be too, and a probe is
    isotropic so the sign is a draw rather than a second render each.
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
    # `halves`, not `levels`: the measured pair is what is stored and the mean is read off it,
    # so clearing the pair is what stops a probe carrying a candidate's measurement.
    return replace(dirs, basis=torch.stack(rows), strength=torch.ones(len(rows)),
                   ranges=None if dirs.ranges is None else tuple(names), halves=None)


#: Candidates offered to `rank` per band, before measurement picks among them. **The eigenvalue
#: ordering is a poor selector in W-space** and `rank` was never given enough to choose from:
#: over four disjoint sets of latents `w_fine`'s seventh eigenvector reads 7.98, 9.74, 9.81 and
#: 8.04 times random -- strongest in every set -- while the three that shipped read 2.4x, 2.9x
#: and 4.7x. In z-space the ordering is sound, so this costs that family only load time.
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
        # One-sided, because `random_like` has already drawn half its rows negative: the bar
        # is over both signs like the candidates it judges, and the sign cost a draw instead
        # of a render. See `RANDOM_PROBES`.
        probe = random_like(dirs)
        probe_levels = verify(net, probe, device, dtype=dtype, amount=amount, into=into)
        per_band: dict[str | None, list[float]] = {}
        for name, level in zip(probe.row_bands, probe_levels, strict=True):
            per_band.setdefault(name, []).append(level)
        means = {name: sum(got) / len(got) for name, got in per_band.items()}
    # Each band against its own bar; `floor` stays the absolute one, in levels.
    bars = {name: max(floor, relative * mean) for name, mean in means.items()}
    # Sorted inside each band, not across them: the ranges are different questions, and
    # interleaving them by strength gives the player a page where no two neighbours belong.
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


def _measure(net: nn.Module, dirs: Directions, device, dtype=torch.float16,
             amount: float = 2.0, seed: int = 0, into=None, seeds: int = SEEDS,
             signs: tuple[float, ...] = (1.0,)) -> list[list[float]]:
    """One list of levels per sign in `signs`, over the same latents and the same references.

    **The signs share the reference image** -- the model at rest at one latent is the same
    picture whichever way the row is then pushed -- which is why this is one function and not
    two calls to `verify`.
    """
    if (into is None) != (dirs.space == "z"):
        raise ValueError(
            f"a {dirs.space}-space basis {'needs' if into is None else 'has no use for'} a "
            f"seam to push through; `into` was {'not ' if into is None else ''}given")
    basis = dirs.basis if into is not None else dirs.basis.to(device=device, dtype=dtype)
    totals = [[0.0] * len(basis) for _ in signs]

    with torch.no_grad():
        for k in range(seeds):
            z = _latent(dirs.nz, seed + k, device, dtype)
            base = pushed(net, z, None, into, dirs.push_shape).float()
            for s, sign in enumerate(signs):
                for i, row in enumerate(basis):
                    moved = pushed(net, z, (sign * amount) * row, into, dirs.push_shape)
                    totals[s][i] += float((moved.float() - base).abs().mean() * 127.5)
        if into is not None:
            into.zero_()
    return [[t / seeds for t in got] for got in totals]
