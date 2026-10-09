"""The dial gate: which of a model's dials actually move its picture, measured at load.

A dial that reaches nothing is drawn dark on the strip rather than offered as a control.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from ganlive.clock import WalkConfig
from ganlive.dials import derive, table
from ganlive.dials.table import live_dials
from ganlive.models.calibrate import TARGET_LEVELS
from ganlive.models.common import first_image, latent
from ganlive.pixels import FLOOR_LEVELS, RANDOM_FLOOR, levels

__all__ = ["live_dials"]


@torch.no_grad()
def measure_dials(net, settings, layout, nz: int, device, dtype):
    """Drive every unmeasured dial that writes the model to each end of its travel, and
    measure how far the picture moved. Returns the layout with `measured` filled in.

    The dials are driven through `Surface.apply`, the same path a hand takes, so what is
    measured is what playing it would do."""
    surface, walk = table.Surface(layout=layout), WalkConfig()

    def frame(z, **held):
        surface.values.update(layout.rests)
        surface.set_held(held)
        surface.apply(settings, walk)
        settings.commit()
        return first_image(net(z))

    z = latent(nz, 0, device, dtype)
    # A copy: a captured generator replays into one output buffer, so `base` would be every frame.
    base = frame(z).clone()
    out = []
    for knob in layout.knobs:
        if not knob.writes or knob.measured is not None:
            out.append(knob)
            continue
        moved = max((levels(frame(z, **{knob.name: x}), base)
                     for x in (0.0, 1.0) if abs(x - knob.rest) > 1e-6), default=0.0)
        out.append(replace(knob, measured=round(moved, 3)))
    settings.reset()
    return table.Layout(tuple(out))


def verified(model, device, dtype):
    """The same model with every MODEL dial measured, and its live set worked out from that.

    Every family passes through here, whether its dials were tuned by hand, swept at load or
    read out of a foreign file."""
    layout = measure_dials(model.net, model.settings, model.layout, model.cfg.nz, device, dtype)
    live = live_dials(model.settings, model.directions, layout)
    writing = [k for k in layout.knobs if k.writes]
    dark = [k for k in writing if k.name not in live]
    print(f"verified: {len(writing) - len(dark)} of {len(writing)} model dial(s) move the "
          f"picture at full travel"
          + ("; dark: " + ", ".join(f"{k.name} {k.measured:.2f}" for k in dark)
             if dark else "") + f" (floor {FLOOR_LEVELS:g} 8-bit levels)", flush=True)
    return replace(model, layout=layout, dials_live=live)


def directions_for(net, nz: int, device, dtype, read=None, into=None,
                   floor: float = RANDOM_FLOOR, path=None):
    """This model's latent directions, measured and ranked at load, or None if none qualify.

    A basis saved beside the checkpoint by `ganlive dials` wins over `read`, the family's own
    quick proposal. `into` is the tensor a StyleGAN2 takes its `w` push in; `floor` is how many
    times a random direction's effect a direction must beat to earn a dial."""
    # A pool several times what the strip can show, because the eigenvalue order is a poor
    # selector; `rank` picks by measurement. `read` sizes its own pool.
    read = read or (lambda net, nz: derive.sefa(net, nz, count=derive.CANDIDATES))
    try:
        cached = None if path is None else derive.saved(
            path, nz, None if into is None else tuple(into.shape), net)
        found = cached if cached is not None else read(net, nz)
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"no latent directions for this model: {exc}", flush=True)
        return None
    # The cheap pass first, so the dearer ones below run over a shortlist.
    found = derive.shortlist(net, found, device, dtype, table.DIRECTION_RANGE, into=into,
                        keep=2 * table.DIRECTIONS)
    found = derive.equalise(net, found, device, dtype, amount=table.DIRECTION_RANGE,
                       target=TARGET_LEVELS, into=into)
    found = derive.rank(net, found, device, dtype, amount=table.DIRECTION_RANGE, into=into,
                   relative=floor, keep_best=table.DIRECTIONS)
    if not len(found):
        print(f"no latent direction on this model beats a random one: {found.report()}",
              flush=True)
        return None
    print(f"directions: {found.report()}", flush=True)
    return found
