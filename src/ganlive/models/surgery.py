"""Rewriting a built module tree in place. The traversal, without any rule.

Two rewrites here walk a net the same way -- fold a BatchNorm into the convolution feeding
it, split a gated convolution on its weights -- and each had written the walk out: the
`nn.Sequential` test, the three-wide window, the replacement list, and the rebuild through
the private `_modules` dict. Thirteen identical lines twice, including the one piece that
reaches past the public API, which is the piece that has to be right in both.

A rule is a function of the next three modules. It returns `(replacements, consumed, tag)`
-- what to put in their place, how many it used up, and a name for the count -- or `None` to
leave the first one alone. Nothing about what a rule may match is decided here.
"""

from __future__ import annotations

from collections import Counter

from torch import nn


def rewrite_sequential(net: nn.Module, rule) -> Counter:
    """Apply `rule` down every `nn.Sequential` in `net`. Returns what it matched, by tag.

    The window is three wide because that is the longest run either caller matches, and `b`
    and `c` are `None` near the end rather than absent, so a rule never indexes off the list.
    """
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
        # Only when the shape changed: an untouched `Sequential` keeps its own `_modules`,
        # so a rule that matches nothing cannot renumber a net behind its caller's back.
        if len(out) != len(items):
            parent._modules.clear()
            for j, module in enumerate(out):
                parent._modules[str(j)] = module
    return counts
