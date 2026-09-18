"""What SeFa must get right, on a generator small enough to build in a test.

Getting it wrong gives directions that look principled and behave like noise -- the first
version returned the right singular vectors, orthonormal, ranked, and in the wrong space. Two
things hold on any weights and are asserted here: the basis is orthonormal *in the latent*, and
`dir 0` maximises the first layer's response. What a direction does to a finished picture is a
property of a *trained* model and is measured on the checkpoint, not here.
"""
from __future__ import annotations

import pytest
import torch

from ganlive.dials.derive import first_consumer, sefa, verify
from ganlive.models.fastgan import Generator


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return Generator(ngf=8, nz=32, im_size=256, im_width=384).eval()


def test_the_basis_spans_the_latent_and_not_the_layer_output(net):
    """**The left singular vectors, not the right ones.**"""
    d = sefa(net, 32)
    assert d.nz == 32, f"a direction must be a latent, got width {d.nz}"
    assert len(d) == 32, "full_matrices=False gives one direction per latent dimension"

    gram = d.basis @ d.basis.T
    assert torch.allclose(gram, torch.eye(32), atol=1e-4), "the basis must be orthonormal"
    assert torch.allclose(d.basis.norm(dim=1), torch.ones(32), atol=1e-5), "unit rows"


def test_the_directions_are_ranked_by_the_strength_they_report(net):
    d = sefa(net, 32)
    s = d.strength.tolist()
    assert all(a >= b for a, b in zip(s, s[1:], strict=False)), f"not descending: {s}"
    assert d.spread() >= 1.0


def test_count_takes_the_strongest_and_keeps_them_paired_with_their_strength(net):
    everything, few = sefa(net, 32), sefa(net, 32, count=4)
    assert len(few) == 4
    assert torch.equal(few.basis, everything.basis[:4])
    assert torch.equal(few.strength, everything.strength[:4])


def test_the_first_consumer_is_found_by_width_not_by_name(net):
    """It has to work on a generator whose layers are named nothing like this one's."""
    name, module = first_consumer(net, 32)
    assert name == "init.main.0"
    assert isinstance(module, torch.nn.ConvTranspose2d)

    with pytest.raises(ValueError, match="no Linear/Conv2d/ConvTranspose2d"):
        first_consumer(net, 7)


def test_a_mapping_network_is_found_instead_when_there_is_one():
    """With `mapping_depth`, z meets a Linear first and that is the weight to factorise."""
    torch.manual_seed(0)
    mapped = Generator(ngf=8, nz=32, im_size=256, im_width=384, mapping_depth=2).eval()
    name, module = first_consumer(mapped, 32)
    assert isinstance(module, torch.nn.Linear), f"{name} is {type(module).__name__}"
    assert name.startswith("mapping")
    assert sefa(mapped, 32).nz == 32


def test_every_derived_direction_actually_moves_the_picture(net):
    """The inert-knob detector, applied to directions."""
    d = sefa(net, 32, count=6)
    levels = verify(net, d, "cpu", dtype=torch.float32, amount=2.0)
    assert len(levels) == 6
    assert all(lv > 0.5 for lv in levels), f"an inert direction is in here: {levels}"


def test_the_leading_direction_maximises_the_first_layers_response(net):
    """The property SeFa actually guarantees, and the only one true on *any* weights."""
    name, module = first_consumer(net, 32)
    weight = module.weight.detach()
    rows = weight.reshape(weight.shape[0], -1).float()      # ConvTranspose2d: (nz, ...)

    rows = rows / rows.norm(dim=0, keepdim=True)              # the reference's column norm

    d = sefa(net, 32)
    lead = (d.basis[0] @ rows).norm()
    assert torch.allclose(lead, d.strength[0], atol=1e-4), (
        f"dir 0's response {lead:.4f} should be the top singular value {d.strength[0]:.4f}")

    generator = torch.Generator(device="cpu").manual_seed(99)
    probes = torch.randn(64, 32, generator=generator)
    probes = probes / probes.norm(dim=1, keepdim=True)
    best_random = (probes @ rows).norm(dim=1).max()
    assert lead >= best_random, (
        f"dir 0 responds {lead:.4f} at {name} but a random direction reached "
        f"{best_random:.4f}; the basis is not the maximiser it claims to be")


def test_the_basis_is_a_property_of_the_weights_and_not_of_a_sample(net):
    """No sampling anywhere: two calls must agree exactly, on any machine, forever."""
    assert torch.equal(sefa(net, 32).basis, sefa(net, 32).basis)


def _sg2(**over):
    """A StyleGAN2 small enough for a test, with all three style ranges non-empty."""
    import dataclasses

    from ganlive.models import stylegan2 as S2

    cfg = dataclasses.replace(
        S2.Config(z_dim=16, w_dim=16, img_resolution=64, channel_base=128, channel_max=32,
                  num_layers=2, num_fp16_res=0), **over)
    torch.manual_seed(3)
    return S2.Generator(cfg).eval().requires_grad_(False)


def _banded(count: int = 8, **over):
    """A StyleGAN2, a banded basis over its style ranges, and the seam zeroed ready to push.

    `z_dim=net.z_dim` and a `push` shaped like `push_shape` are what a test gets wrong in
    silence, so they are written once."""
    import torch as _torch

    from ganlive.dials.derive import sefa_banded, split
    from ganlive.models import stylegan2 as S2

    net = _sg2(**over)
    dirs = sefa_banded(S2.style_bands(net), split(count, len(S2.BANDS)), z_dim=net.z_dim)
    net.mapping.push = _torch.zeros(dirs.push_shape)
    return net, dirs


def _live_band(row, push_shape) -> list[int]:
    """Which seats of the seam a row writes to. One is the whole point; two is a bug."""
    return (row.reshape(push_shape).abs().sum(dim=1) > 0).nonzero().flatten().tolist()


def test_the_gram_of_symmetric_differences_is_the_jacobian_metric():
    """**The claim `metric` rests on, checked where the answer is known.**

    On a linear map the Jacobian is the matrix, so `J^T J` is available in closed form and the
    estimator has to reproduce it to floating point. On a nonlinear one it is second-order, and
    that is what sets `PROBE_EPS`: the error falls as `eps^2` until fp32 cancellation takes
    over and it rises again."""
    from ganlive.dials.derive import metric

    torch.manual_seed(0)
    nz, out = 12, 40
    w = torch.randn(out, nz)

    class Linear(torch.nn.Module):
        def forward(self, z):
            return (w @ z.reshape(-1)).reshape(1, 1, 8, 5)

    got = metric(Linear(), nz, "cpu", torch.float32, seeds=1, pool=1)
    exact = (w.T @ w).double()
    assert (got - exact).abs().max() / exact.abs().max() < 1e-4, (
        "the Gram of symmetric differences is not J^T J, so every direction read off it is "
        "answering a question nobody asked")


def test_a_metric_basis_lands_in_bands_like_the_factorised_one():
    """The two derivations differ in where the basis comes from and in nothing else: same
    layout, same unit rows, same one-band-per-row, or `rank` and the strip disagree."""
    from ganlive.dials.derive import active_banded, split

    net, sefa = _banded(6)
    got = active_banded(net, sefa.band_names, split(6, len(sefa.band_names)), net.z_dim,
                        "cpu", torch.float32, into=net.mapping.push, seeds=1)

    assert len(got) == len(sefa) and got.ranges == sefa.ranges
    assert got.push_shape == sefa.push_shape and got.z_dim == sefa.z_dim
    assert torch.allclose(got.basis.norm(dim=1), torch.ones(len(got)), atol=1e-4)
    # The same seat as the factorised basis's row for the same band, which is the invariant
    # `_laid_out` exists to hold -- and reads it off that basis rather than restating the map.
    for mine, theirs in zip(got.basis, sefa.basis, strict=True):
        assert _live_band(mine, got.push_shape) == _live_band(theirs, sefa.push_shape)
    assert float(net.mapping.push.abs().max()) == 0.0, "the seam was left pushed"


def test_a_saved_basis_comes_back_and_one_for_another_model_does_not(tmp_path):
    """**The cache is a proposal, so the only thing it must never do is fit the wrong model.**

    It carries no measured levels -- those belong to the load that measures them -- and a file
    whose seam or latent width disagrees is refused rather than reshaped."""
    from ganlive.dials.derive import cache_path, rank, save, saved

    net, dirs = _banded()
    kept = rank(net, dirs, "cpu", torch.float32, amount=2.0, relative=0.0, floor=0.0,
                into=net.mapping.push)
    checkpoint = tmp_path / "model.pt"
    assert save(kept, checkpoint, net, "8/band over 1 latent") == cache_path(checkpoint)

    back = saved(checkpoint, kept.z_dim, kept.push_shape, net)
    assert torch.equal(back.basis, kept.basis) and back.ranges == kept.ranges
    assert back.levels is None, "a cached proposal must not carry a measurement it did not make"
    assert "8/band over 1 latent" in back.report(), (
        "what it was derived at has to reach the report, or a basis read at the wrong settings "
        "is indistinguishable from a good one")

    assert saved(checkpoint, kept.z_dim + 1, kept.push_shape, net) is None
    assert saved(checkpoint, kept.z_dim, (99, 99), net) is None
    assert saved(tmp_path / "nothing.pt", kept.z_dim, kept.push_shape, net) is None

    # **A path is not an identity.** A fine-tune written back over its own name keeps the width
    # and the seam, so only the weights say the cached basis is no longer this model's.
    moved = _sg2()
    with torch.no_grad():
        for p in moved.parameters():
            p.add_(0.5)
    assert saved(checkpoint, kept.z_dim, kept.push_shape, moved) is None


def test_a_banded_basis_is_zero_outside_its_own_range():
    """What lets one matrix-vector product on the host still produce the whole push."""
    from ganlive.dials.derive import sefa_banded, split
    from ganlive.models import stylegan2 as S2

    net = _sg2()
    counts = split(8, len(S2.BANDS))
    assert counts == (2, 3, 3)
    d = sefa_banded(S2.style_bands(net), counts, z_dim=net.z_dim)

    assert d.space == "w" and d.bands == len(S2.BANDS) and d.nz == net.z_dim
    assert len(d) == 8
    assert d.push_shape == (len(S2.BANDS), net.cfg.w_dim)
    for i, row in enumerate(d.basis):
        live = _live_band(row, d.push_shape)
        assert live == [0 if i < 2 else (1 if i < 5 else 2)], (
            f"row {i} carries weight in ranges {live}; a banded row that leaks into another "
            f"range makes one dial two dials")
        assert abs(float(row.norm()) - 1.0) < 1e-5, "unit rows, in the space they live in"


def test_a_w_basis_and_a_latent_basis_each_refuse_the_others_seam():
    """The two failure modes are opposite, so neither is allowed to happen quietly."""
    from ganlive.dials.derive import sefa, sefa_banded, split, verify
    from ganlive.models import stylegan2 as S2

    net = _sg2()
    w = sefa_banded(S2.style_bands(net), split(8, len(S2.BANDS)), z_dim=net.z_dim)
    z = sefa(net, net.z_dim, count=4)

    net.mapping.push = torch.zeros(w.push_shape)
    with pytest.raises(ValueError, match="needs"):
        verify(net, w, "cpu", dtype=torch.float32, amount=1.0)
    with pytest.raises(ValueError, match="no use for"):
        verify(net, z, "cpu", dtype=torch.float32, amount=1.0, into=net.mapping.push)

    levels = verify(net, w, "cpu", dtype=torch.float32, amount=2.0, into=net.mapping.push)
    assert len(levels) == 8 and all(x > 0 for x in levels)
    assert float(net.mapping.push.abs().max()) == 0.0, (
        "the seam was left pushed after measuring, so every later frame carries the last "
        "direction the gate happened to try")


def test_ranking_a_banded_basis_keeps_each_range_together():
    """Sorted inside a range, never across: three ranges are three different questions."""
    from ganlive.dials.derive import rank

    net, d = _banded()
    ranked = rank(net, d, "cpu", torch.float32, amount=3.0, floor=0.0, relative=0.0,
                  into=net.mapping.push)

    bands = [int(row.reshape(d.push_shape).abs().sum(dim=1).argmax()) for row in ranked.basis]
    assert bands == sorted(bands), f"the ranges came back interleaved: {bands}"
    assert ranked.bands == d.bands and ranked.z_dim == d.z_dim, "carried through the copy"
    for lo in range(len(bands)):
        run = [ranked.levels[i] for i in range(len(bands)) if bands[i] == bands[lo]]
        assert run == sorted(run, reverse=True), "inside a range it is strongest first"


def test_equalising_gives_a_w_basis_the_size_a_latent_basis_gets_for_free():
    """`w` has no natural scale, so one scalar for the whole basis buys it a stated one."""
    from ganlive.dials.derive import equalise, verify

    net, d = _banded()
    scaled = equalise(net, d, "cpu", torch.float32, amount=2.0, target=12.0,
                      into=net.mapping.push)

    levels = sorted(verify(net, scaled, "cpu", torch.float32, amount=2.0,
                           into=net.mapping.push))
    assert abs(levels[len(levels) // 2] - 12.0) < 3.0, (       # a quarter: the response is not linear in the push
        f"the median direction landed at {levels[len(levels) // 2]:.1f} levels, not 12")
    ratio = scaled.basis[0].norm() / d.basis[0].norm()
    assert torch.allclose(scaled.basis.norm(dim=1) / d.basis.norm(dim=1),
                          ratio.expand(len(d)), atol=1e-5), (
        "one scalar for the whole basis, so the ranking the spectrum earned is still visible")


def test_the_range_a_row_came_from_survives_the_drop_that_reorders_the_rows():
    """**The label has to ride on the basis, because the basis is what gets shortened.**"""
    from ganlive.dials.derive import rank
    from ganlive.models import stylegan2 as S2

    net, d = _banded()
    assert d.ranges == ("w_coarse",) * 2 + ("w_mid",) * 3 + ("w_fine",) * 3

    ranked = rank(net, d, "cpu", torch.float32, amount=3.0, floor=0.0, relative=0.0,
                  into=net.mapping.push)
    order = [name for name, _lo, _hi in S2.BANDS]
    assert [order.index(n) for n in ranked.ranges] == sorted(
        order.index(n) for n in ranked.ranges), (
        "the ranges came back out of ladder order; sorting on the *name* puts w_fine between "
        "w_coarse and w_mid, which is alphabetical and is not the ladder")

    # A floor high enough to kill some directions. The absolute one and nothing else: on an
    # untrained generator the relative bar drops all eight, and a partial drop is the point.
    levels = dict(zip(ranked.ranges, ranked.levels, strict=True))
    cut = rank(net, d, "cpu", torch.float32, amount=3.0, relative=0.0,
               floor=sorted(ranked.levels)[3], into=net.mapping.push)
    assert 0 < len(cut) < len(ranked), f"nothing was dropped, so this proves nothing: {levels}"
    for row, name in zip(cut.basis, cut.ranges, strict=True):
        slot = int(row.reshape(cut.push_shape).abs().sum(dim=1).argmax())
        assert name == S2.BANDS[slot][0], (
            f"a surviving row from {S2.BANDS[slot][0]} is labelled {name}; every dial after "
            f"the first dropped one would say the wrong thing on the strip")


def test_a_direction_that_does_what_a_random_one_does_is_dropped(net):
    """The gate is relative: on an untrained generator no direction is special, so most of a
    derived set does what a random unit vector does. An empty set still reports."""
    from ganlive.dials.derive import RANDOM_FLOOR, rank

    d = sefa(net, 32, count=8)
    kept = rank(net, d, "cpu", torch.float32, amount=2.0)
    assert kept.random_levels is not None and kept.random_levels > 0
    assert kept.dropped == 8 - len(kept)
    assert all(lv >= RANDOM_FLOOR * kept.random_levels for lv in kept.levels)
    assert "random direction" in kept.report()

    everything = rank(net, d, "cpu", torch.float32, amount=2.0, relative=0.0)
    assert len(everything) == 8 and everything.random_levels is None

    nothing = rank(net, d, "cpu", torch.float32, amount=2.0, relative=1e9)
    assert len(nothing) == 0 and nothing.dropped == 8
    assert "none of these beat it" in nothing.report()


def test_the_seam_is_empty_before_every_reference_image_and_not_just_the_first():
    """**Three of four readings were measured against another direction.** The seam was zeroed
    outside the latent loop, so only the first reference was the model at rest; every later one
    carried the previous row's push, and the error moved with the candidate set's size."""
    from ganlive.dials.derive import verify

    net, d = _banded()

    seen = []
    forward = net.mapping.forward
    net.mapping.forward = lambda z, truncation=1.0: (
        seen.append(float(net.mapping.push.abs().max())) or forward(z, truncation))
    try:
        verify(net, d, "cpu", dtype=torch.float32, amount=2.0, seeds=4,
               into=net.mapping.push)
    finally:
        net.mapping.forward = forward

    # One reference plus one render per row, per latent. Every reference must see an empty seam.
    per_latent = 1 + len(d)
    references = [seen[k * per_latent] for k in range(4)]
    assert references == [0.0] * 4, (
        f"a reference image was rendered with {max(references)} still in the seam; the levels "
        f"measured after it are one direction against another, not against the model at rest")


def test_every_band_gets_its_own_random_probes_however_long_the_basis_is():
    """The baseline has to live where the candidate lives. `random_like` cycled one flat list
    of templates, so a basis longer than `RANDOM_PROBES` never reached its later bands and
    those rows were judged against the sensitivity of the bands above them."""
    from ganlive.dials.derive import RANDOM_PROBES, random_like, sefa_banded, split
    from ganlive.models import stylegan2 as S2

    net = _sg2()
    names = [name for name, _lo, _hi in S2.BANDS]
    # Deliberately longer than the probe count -- the shape that broke it.
    long = sefa_banded(S2.style_bands(net), split(3 * RANDOM_PROBES + 6, len(S2.BANDS)),
                       z_dim=net.z_dim)
    probes = random_like(long)

    assert len(probes) == RANDOM_PROBES * len(names)
    assert sorted(set(probes.ranges)) == sorted(names), (
        f"only {sorted(set(probes.ranges))} got probes; the missing band's rows would be "
        f"held to another band's bar")
    for slot, name in enumerate(names):
        assert probes.ranges.count(name) == RANDOM_PROBES
        for row, group in zip(probes.basis, probes.ranges, strict=True):
            if group != name:
                continue
            live = row.reshape(probes.push_shape)
            assert float(live[slot].abs().sum()) > 0.0, "a probe must write its own band"
            assert float(live.abs().sum() - live[slot].abs().sum()) == 0.0, (
                "a probe that writes outside its band measures the wrong band's sensitivity")


def test_each_band_is_held_to_its_own_bar_and_the_report_says_what_they_were():
    """One pooled bar asked half the dials to clear a number that was never theirs: on FFHQ a
    random direction moves 6.2, 4.3 and 6.7 levels through the three bands."""
    from ganlive.dials.derive import rank
    from ganlive.models import stylegan2 as S2

    net, d = _banded()
    kept = rank(net, d, "cpu", torch.float32, amount=2.0, into=net.mapping.push)

    assert kept.random_by_range is not None, "a banded basis must report a bar per band"
    assert [name for name, _level in kept.random_by_range] == [
        name for name, _lo, _hi in S2.BANDS]
    assert all(level > 0 for _name, level in kept.random_by_range)
    for name, level in kept.random_by_range:
        assert f"{name} {level:.1f}" in kept.report()
    # The push belongs in the report too: the ratio is not scale-invariant, so a bare "3x
    # random" does not say what was measured.
    assert kept.amount == 2.0 and "at a push of 2" in kept.report()


def test_a_dial_is_judged_on_both_halves_of_its_travel(net):
    """**The strip's travel is symmetric and the gate saw one side of it.** The encoder goes
    both ways from centre and an eigenvector's sign is whatever `eigh` returned, so which half
    got measured was arbitrary -- `gv-2048-ft` ships rows at 48.7 levels one way and 31.1 the
    other."""
    from ganlive.dials.derive import orient, rank, sefa, travel

    d = sefa(net, 32, count=8)
    up, down = travel(net, d, "cpu", torch.float32, amount=2.0)
    assert len(up) == len(down) == 8
    assert up != down, "an untrained generator's rows are not perfectly symmetric"

    # `rank` scores the mean of the two, which is never above the stronger half alone. The
    # level is read off the halves rather than stored beside them, so what is worth pinning is
    # that the formula is the mean: on a lopsided row, one half alone would read higher.
    kept = rank(net, d, "cpu", torch.float32, amount=2.0, relative=0.0, floor=0.0)
    for level, (strong, weak) in zip(kept.levels, kept.halves, strict=True):
        assert weak <= level <= strong or strong == weak
    lopsided = [i for i, (strong, weak) in enumerate(kept.halves) if strong > weak + 0.05]
    assert lopsided, "an untrained generator ships at least one lopsided row"
    for i in lopsided:
        assert kept.levels[i] < kept.halves[i][0], (
            "the level must be the mean of the halves, not the stronger one")

    turned, tup, tdown = orient(d, up, down)
    assert all(a >= b for a, b in zip(tup, tdown, strict=True)), (
        "after orienting, turning up must be the stronger half of every dial")
    # Orienting negates rows; it must not change what the basis spans or how long a row is.
    assert torch.allclose(turned.basis.abs(), d.basis.abs())
    assert torch.allclose(turned.basis.norm(dim=1), d.basis.norm(dim=1))


def test_the_pool_is_wider_than_the_strip_and_measurement_picks_from_it(net):
    """SeFa's eigenvalue order is a poor selector in W-space, so `rank` is handed a pool and
    caps what survives rather than being handed the answer."""
    from ganlive.dials.derive import CANDIDATES, rank, sefa, shortlist

    assert CANDIDATES > 4, "a pool the size of the strip is not a pool"
    pool = sefa(net, 32, count=CANDIDATES)
    kept = rank(net, pool, "cpu", torch.float32, amount=2.0, relative=0.0, floor=0.0,
                keep_best=4)
    assert len(kept) == 4, f"keep_best must cap the survivors, got {len(kept)}"
    assert all(a >= b for a, b in zip(kept.levels, kept.levels[1:], strict=False)), (
        "within one band the survivors must be the best ones, in order")

    # The cheap pass only narrows; it must not reorder, rescale or relabel.
    short = shortlist(net, pool, "cpu", torch.float32, amount=2.0, keep=6)
    assert len(short) == 6 and short.source == pool.source
    rows = {tuple(row.tolist()) for row in pool.basis}
    assert all(tuple(row.tolist()) in rows for row in short.basis)
    assert shortlist(net, pool, "cpu", torch.float32, amount=2.0,
                     keep=CANDIDATES + 5) is pool, "nothing to cut, nothing to measure"


def test_a_wide_pool_keeps_each_band_its_own_share():
    """A pool is per band and so is the cap; one strong band must not eat the strip."""
    from ganlive.dials.derive import CANDIDATES, rank, shortlist
    from ganlive.models import stylegan2 as S2

    nbands = len(S2.BANDS)
    net, pool = _banded(CANDIDATES * nbands)
    assert len(pool) == CANDIDATES * nbands

    short = shortlist(net, pool, "cpu", torch.float32, amount=2.0, into=net.mapping.push,
                      keep=3 * nbands)
    assert [short.ranges.count(name) for name, _lo, _hi in S2.BANDS] == [3] * nbands

    kept = rank(net, pool, "cpu", torch.float32, amount=2.0, relative=0.0, floor=0.0,
                into=net.mapping.push, keep_best=3 * nbands)
    assert [kept.ranges.count(name) for name, _lo, _hi in S2.BANDS] == [3] * nbands
