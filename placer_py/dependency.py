"""
The dependency bound: what makes a product of block e-values an e-value.

Ported from `src/pipeline/mechanistic_evidence.cpp` and
`src/pipeline/pipeline_finalization_stage.inc`, pinned by
`tests/test_04_dependency.py`.

Summing the block log-likelihood-ratios multiplies their e-values, and a product
of DEPENDENT e-values is not an e-value until it is divided by an upper bound on
its null expectation (Wang 2025, Biometrika 112(2):asaf020, Sec. 5b):

    sigma >= E_null[ prod_k E_k ]   =>   (prod_k E_k) / sigma  is an e-value.

Releases up to 0.0.4 subtracted a fixed 0.65 nats, i.e. asserted
sigma <= e^0.65 = 1.9155 with no evidence. sigma is a null expectation and the
run computes its own null rows, so it is measured instead.

The estimator has NO tuning parameters of its own. Three quantities a first
implementation reaches for are derived:

  * No minimum null count. The empirical Bernstein bound holds for any n >= 2
    and its WIDTH already encodes "too few controls": few rows -> wide bound ->
    large penalty -> nothing passes. Correct behaviour, not a failure mode.
  * No confidence level. alpha is tied to the FDR target q, so a run carries one
    risk level rather than two.
  * No cap. e-BH selects at rank r once an e-value reaches m/(q*r), so the
    largest threshold it can ever face is m/q at rank 1 and no e-value needs to
    exceed it. cap = log(m/q) is therefore decision-neutral and fully determined
    by the candidate count and the FDR target.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from placer_py import mathx


@dataclass
class PenaltySide:
    sigma_mean: float = 1.0
    sigma_upper: float = 1.0
    log_penalty: float = 0.0


@dataclass
class PenaltyEstimate:
    vs_artifact: PenaltySide = field(default_factory=PenaltySide)
    vs_non_te: PenaltySide = field(default_factory=PenaltySide)
    cap_log: float = 0.0
    null_count: int = 0
    estimated: bool = False


#: Re-exported from `placer_py/mathx.py`, which defines the NaN policy
#: these copies disagreed on -- see its docstring.
_clamp = mathx.clamp


def empirical_bernstein_upper(values: Sequence[float], log_inv_alpha: float,
                              cap: float) -> float:
    """
    One-sided upper confidence bound on a mean (Maurer & Pontil 2009):

        mean + sqrt(2 v ln(1/alpha) / n) + 7 R ln(1/alpha) / (3 (n - 1))

    `v` is the sample variance with the population convention (divide by n), and
    `R` is `max(values)` -- NOT `max - min`. That is what the C++ does, and it is
    the conservative reading, since the values are non-negative e-values whose
    lower bound is 0 anyway.

    Floored at 1.0: sigma < 1 would mean the blocks are anti-correlated under
    the null and the product is already conservative. That is not turned into a
    bonus, so the penalty never goes below zero. Clamped above at `cap`.

    Why Bernstein rather than Hoeffding: the product distribution is heavy
    tailed -- most null rows contribute almost nothing and a few contribute
    everything -- so `R` is large while the variance is small. Hoeffding's width
    depends only on `R` and would be uselessly wide; Bernstein's leading term
    carries sqrt(v/n) and `R` appears only in the O(1/n) remainder.
    """
    n = len(values)
    if n < 2:
        return math.exp(cap)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    observed_range = max(values)
    slack = (math.sqrt(2.0 * var * log_inv_alpha / n)
             + (7.0 * observed_range * log_inv_alpha) / (3.0 * (n - 1.0)))
    return _clamp(mean + slack, 1.0, math.exp(cap))


def _side(null_raw_log_bf: Iterable[float], cap_log: float,
          log_inv_alpha: float) -> PenaltySide:
    products: list[float] = []
    for log_bf in null_raw_log_bf:
        if not math.isfinite(log_bf):
            continue
        e = math.exp(min(log_bf, cap_log))
        if not math.isfinite(e):
            continue
        products.append(e)

    side = PenaltySide()
    if len(products) < 2:
        # No variance exists, so the only honest bound is the cap itself:
        # maximally conservative, nothing passes. That is the right answer when
        # there is no null to calibrate against, and it is why no
        # minimum-count PARAMETER is needed.
        side.sigma_mean = math.exp(cap_log)
        side.sigma_upper = math.exp(cap_log)
        side.log_penalty = cap_log
        return side

    n = len(products)
    side.sigma_mean = sum(products) / n
    side.sigma_upper = empirical_bernstein_upper(products, log_inv_alpha, cap_log)
    side.log_penalty = math.log(side.sigma_upper)
    return side


def estimate_dependency_penalty(null_raw_log_bf_vs_artifact: Sequence[float],
                                null_raw_log_bf_vs_non_te: Sequence[float],
                                target_fdr: float,
                                candidate_count: int) -> PenaltyEstimate:
    """
    Port of `placer::estimate_dependency_penalty`.

    Each side gets its OWN bound: the two aggregates are different products over
    differently-weighted blocks and so have different null expectations.
    Estimating both is what removed the unexplained 0.60 factor that used to
    relate them.

    `null_count` is the max of the two input sizes BEFORE dropping non-finite
    values, matching the C++; `estimated` is False exactly when either side has
    fewer than two raw inputs.
    """
    out = PenaltyEstimate()
    out.null_count = max(len(null_raw_log_bf_vs_artifact),
                         len(null_raw_log_bf_vs_non_te))

    q = _clamp(target_fdr, 1e-6, 1.0)
    m = float(max(candidate_count, 1))
    out.cap_log = math.log(m / q)
    log_inv_alpha = -math.log(q)

    out.vs_artifact = _side(null_raw_log_bf_vs_artifact, out.cap_log,
                            log_inv_alpha)
    out.vs_non_te = _side(null_raw_log_bf_vs_non_te, out.cap_log, log_inv_alpha)
    out.estimated = (len(null_raw_log_bf_vs_artifact) >= 2
                     and len(null_raw_log_bf_vs_non_te) >= 2)
    return out


def apply_penalty(raw_log_bf: float, cap_log: float, log_penalty: float) -> float:
    """
    `min(raw_log_bf, cap_log) - log_penalty`.

    THE CAP MUST BE APPLIED BEFORE THE SUBTRACTION, and this is a validity
    requirement rather than a numerical nicety. sigma bounds
    `E_null[min(Y, C)]`, so only `min(Y, C) / sigma` is guaranteed to have null
    expectation <= 1; the uncapped `Y / sigma` is not, because `E_null[Y]` can
    exceed `E_null[min(Y, C)]`. Applying the penalty to the uncapped aggregate
    builds e-values that e-BH is entitled to assume and the estimator never
    certified. That was a real bug in the C++, and it is fixed.

    Cost of doing it correctly: the largest attainable e-value becomes
    `C / sigma` rather than `C`, so e-BH's rank-1 threshold is unreachable
    whenever sigma > 1. That only matters when a single candidate in the whole
    run is real; at any selection size r > 1 the threshold m/(q*r) is far below C.
    """
    return min(raw_log_bf, cap_log) - log_penalty


def calibration_rows(ledger: Sequence[dict]) -> list[int]:
    """
    Indices of the rows sigma is estimated on: EVERY row carrying a mechanistic
    certificate. Nothing is excluded.

    Deliberately NOT the conformal null-control set, and the two must not be
    merged. The conformal selector needs rows that were not selected -- that is
    what makes its p-value conformal -- and excluding promotable rows is correct
    there. This sample must not be selected by `Lambda` itself, because sigma is
    a MEAN dominated by its right tail and promotability includes `log_e > 0` on
    the very aggregate being calibrated. Selecting that way removes exactly the
    mass sigma is made of and the estimate collapses onto its floor of 1.

    Measured: removing the high-aggregate rows takes sigma from 17.7 to 1.0 and
    the penalty from 2.87 nats to zero. The C++ was returning the floor by
    construction until this was separated.

    The fix follows from the directions of the two available biases. Including
    true positives inflates the empirical mean, so the bound still bounds the
    null mean provided the aggregate is not anti-informative -- conservative,
    costs recall. Excluding rows by the aggregate truncates the right tail and
    deflates the estimate -- anti-conservative, breaks the guarantee. Only the
    first is acceptable.

    `has_conformal_feature_certificate` in the C++ is exactly "the mechanistic
    blocks field is populated", with no dependence on the aggregate's value, so
    that is the whole predicate here too.
    """
    keep: list[int] = []
    for index, row in enumerate(ledger):
        blocks = row.get("mechanistic_blocks", "")
        if blocks and blocks != "NA":
            keep.append(index)
    return keep
