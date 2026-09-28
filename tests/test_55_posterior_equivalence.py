"""
The breakpoint posterior is the same floats after its kernel was tabulated.

`placer/core/hypotheses.compute_breakpoint_position_posterior` now adds each
signal's kernel from a table, signals outermost, and skips the offsets where
the kernel is exactly zero. Its credible-interval width and entropy are
recorded observables, so they must be the same bits. The implementation
before the change is kept below verbatim and compared on random signal sets:
precise and imprecise classes, duplicates, spans past the 2 kb grid cap,
positions that are not usable.
"""

from __future__ import annotations

import math
import random

from placer.core import hypotheses as H
from placer.core.clustering import (
    CANDIDATE_LONG_INSERTION,
    CANDIDATE_SOFT_CLIP,
    CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
    BreakpointCandidate,
)


def _old_posterior(candidates):
    summary = H.BreakpointPosteriorSummary()
    signals: list[tuple[int, float]] = []
    for bp in candidates:
        if bp.pos < 0:
            continue
        signals.append((bp.pos, H._signal_sigma(bp.class_mask)))
    if len(signals) < 2:
        return summary

    # Taken from `signals` rather than tracked as running `lo`/`hi` in the
    # loop above. Same values, but the reason they cannot be None is now
    # structural -- the early return proves the list is non-empty -- rather
    # than a correlation between two variables that a reader (or a type
    # checker) has to notice.
    grid_lo = min(pos for pos, _ in signals) - H.POSTERIOR_GRID_PAD_BP
    grid_hi = max(pos for pos, _ in signals) + H.POSTERIOR_GRID_PAD_BP
    if (grid_hi - grid_lo) > H.POSTERIOR_GRID_MAX_SPAN_BP:
        grid_hi = grid_lo + H.POSTERIOR_GRID_MAX_SPAN_BP
    n = grid_hi - grid_lo + 1

    density = [0.0] * n
    total = 0.0
    for gi in range(n):
        x = float(grid_lo + gi)
        value = 0.0
        for pos, sigma in signals:
            z = (x - pos) / sigma
            value += math.exp(-0.5 * z * z) / sigma
        density[gi] = value
        total += value
    if total <= 0.0:
        return summary

    entropy = 0.0
    for gi in range(n):
        density[gi] /= total
        if density[gi] > 0.0:
            entropy -= density[gi] * math.log(density[gi])
    # Normalised by log(n) so the value is comparable across grid sizes: 1 is a
    # flat posterior over the whole grid, 0 is a point mass.
    summary.entropy = entropy / math.log(n)

    cumulative = 0.0
    q05 = 0
    for gi in range(n):
        cumulative += density[gi]
        if cumulative >= 0.05:
            q05 = gi
            break
    cumulative = 0.0
    q95 = n - 1
    for gi in range(n):
        cumulative += density[gi]
        if cumulative >= 0.95:
            q95 = gi
            break
    summary.ci_width = float(q95 - q05)
    return summary


def test_the_tabulated_posterior_is_the_same_floats():
    rng = random.Random(41)
    masks = (0, CANDIDATE_SOFT_CLIP, CANDIDATE_LONG_INSERTION,
             CANDIDATE_SPLIT_SA_SUPPLEMENTARY, CANDIDATE_SOFT_CLIP | CANDIDATE_LONG_INSERTION)
    compared = 0
    for _ in range(400):
        centre = rng.randint(0, 10_000_000)
        spread = rng.choice((0, 3, 20, 200, 1500, 5000))
        count = rng.choice((0, 1, 2, 3, 10, 60))
        candidates = []
        for _ in range(count):
            pos = centre + rng.randint(-spread, spread)
            if rng.random() < 0.05:
                pos = -1
            candidates.append(BreakpointCandidate(pos=pos, class_mask=rng.choice(masks)))
        if candidates and rng.random() < 0.3:
            candidates += candidates[:rng.randint(1, len(candidates))]   # duplicates
        new = H.compute_breakpoint_position_posterior(candidates)
        old = _old_posterior(candidates)
        assert new == old, (centre, spread, count)
        assert math.isfinite(new.entropy)
        compared += 1
    assert compared == 400
