"""Rewriting runs of modules inside every `nn.Sequential` of a built net, in place.

A rule is a function of the next three modules. It returns `(replacements, consumed, tag)`
-- what to put in their place, how many it used up, and a name for the count -- or `None` to
leave the first one alone. `fold.fold_norms` and `onnx_rewrite.split_gated_convs` are the
two rules.
"""

from __future__ import annotations

from collections import Counter

from torch import nn


def rewrite_sequential(net: nn.Module, rule) -> Counter:
    """Apply `rule` down every `nn.Sequential` in `net`. Returns what it matched, by tag.

    Near the end of a sequence the second and third modules a rule sees are `None`."""
    counts: Counter = Counter()
    for parent in net.modules():
        if not isinstance(parent, nn.Sequential):
            continue
        items, out, i = list(parent), [], 0
        while i < len(items):
            got = rule(items[i],
                       items[i + 1] if i + 1 < len(items) else None,
                       items[i + 2] if i + 2 < len(items) else None)
            if got is None:
                out.append(items[i])
                i += 1
                continue
            replacements, consumed, tag = got
            out.extend(replacements)
            counts[tag] += 1
            i += consumed
        # Rebuilt only when the length changed, so an untouched `Sequential` keeps its numbering.
        if len(out) != len(items):
            parent._modules.clear()
            for j, module in enumerate(out):
                parent._modules[str(j)] = module
    return counts
