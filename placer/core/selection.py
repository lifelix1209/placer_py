"""
Selection: e-BH as the primary route, the conformal dominance route second.

Why e-BH is primary. It controls FDR <= q under ARBITRARY dependence among the
e-values with no correction factor (Wang & Ramdas 2022, JRSS-B 84(3):822),
because its proof uses only linearity of expectation. The conformal p-values
have to be selected with Benjamini-Yekutieli, whose harmonic factor H_m is the
price of admitting arbitrary dependence among *p*-values -- about 7.49 at
m = 1000, i.e. a nominal q = 0.10 becomes an effective 0.0134. Switching the
currency from tail probabilities to expectations removes that entirely.

This is only sound if the inputs really are e-values, which is what the measured
dependency bound provides. The two are a pair; neither is correct alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Number of per-call e-value constructions (mechanistic, low-allele-fraction,
#: reference-unopposed bilateral partial anchor). Fixed, NOT the count of the
#: ones that happen to fire -- see `combine_e_values`.
E_VALUE_CONSTRUCTIONS = 3


def combine_e_values(candidates: Sequence[float],
                     n_constructions: int = E_VALUE_CONSTRUCTIONS) -> float:
    """
    Combine the per-call constructions by AVERAGING, not by taking the maximum.

    The maximum of e-values is NOT an e-value: `E_null[max_k E_k]` can exceed 1.
    It does obey `max_k E_k <= sum_k E_k`, so `max / K` is valid, which means a
    plain `max()` is anti-conservative by at most a factor of K -- ln 3 = 1.10
    nats here. The arithmetic mean fixes that with no assumption at all, since
    expectation is linear:

        E_null[(1/K) sum_k E_k] = (1/K) sum_k E_null[E_k] <= 1

    and it DOMINATES `max / K`, because it keeps the evidence from the other
    constructions instead of discarding it (Vovk & Wang 2021). A construction
    that does not apply contributes 0, itself a valid e-value.

    `n_constructions` is the number of CONSTRUCTIONS, not the number of non-zero
    ones. Dividing by the non-zero count would make the divisor data-dependent
    and hand back the anti-conservatism it was introduced to remove.
    """
    total = 0.0
    for value in candidates:
        if value > 0.0 and math.isfinite(value):
            total += value
    return total / float(n_constructions)


def ebh_select(e_values: Sequence[float], q: float) -> list[int]:
    """
    e-BH step-up. Sort descending, take the largest rank `r` with
    `E_(r) >= m / (q * r)`, select that prefix. Returns selected indices.

    `m` is the total number of hypotheses tested, never a subset chosen after
    looking at another test's outcome -- the earlier C++ skipped calls that
    already held a conformal certificate, which made the candidate set
    data-dependent and broke the guarantee.

    A hypothesis with no usable e-value still counts towards `m`: pass it as 0
    so it stays in the denominator and can never be selected. Dropping it from
    `m` is the same data-dependence in another guise.

    Ties break by original index ascending, matching the C++ comparator, so a
    run of equal e-values selects the same set in both languages.
    """
    m = len(e_values)
    if m == 0 or q <= 0.0:
        return []
    q = min(q, 1.0)

    order = sorted(range(m), key=lambda i: (-e_values[i], i))
    selected_prefix = 0
    for rank in range(1, m + 1):
        threshold = m / (q * rank)
        if e_values[order[rank - 1]] >= threshold:
            selected_prefix = rank
    return order[:selected_prefix]


def benjamini_yekutieli(p_values: Sequence[float], q: float) -> list[int]:
    """
    BY step-up: the threshold at rank `r` is `r * q / (m * H_m)` with
    `H_m = sum_{i=1..m} 1/i`. Returns selected indices.

    Used for the conformal route only. `H_m` is not an approximation -- it is
    the price of admitting arbitrary dependence among p-values, and the reason
    the e-value route is primary.
    """
    m = len(p_values)
    if m == 0 or q <= 0.0:
        return []
    h_m = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    selected_prefix = 0
    for rank in range(1, m + 1):
        threshold = rank * q / (m * h_m)
        if p_values[order[rank - 1]] <= threshold:
            selected_prefix = rank
    return order[:selected_prefix]


# ---------------------------------------------------------------------------
# Dominance conformal p-values
# ---------------------------------------------------------------------------
def _dominates(null_row: Sequence[float], candidate: Sequence[float]) -> bool:
    """A null dominates the candidate when it is at least as TE-like in every
    coordinate. Callers pass coordinates already sign-oriented so that LARGER is
    always more TE-like -- `ref_span_reads` is negated by the caller."""
    return all(n >= c for n, c in zip(null_row, candidate))


def dominance_conformal_p(candidates: Sequence[Sequence[float]],
                          nulls: Sequence[Sequence[float]]) -> list[float]:
    """
    Add-one-smoothed dominance conformal p-value:

        p = (1 + #{nulls dominating the candidate}) / (1 + #{relevant nulls})

    Using dominance rather than a distance avoids assuming the features' joint
    distribution: it never asks how identity and alt-read count are correlated,
    only how many nulls are no worse in every coordinate. That is the same
    evasion the explanation comparison uses to avoid exchange rates, applied to
    a different problem.

    This implementation is the straightforward O(m*n*d) one, kept because it is
    obviously correct and because it is what the invariant tests pin. It does
    NOT scale: at m = 1e5 and n = 1e6 it takes about 1159 s, where a sort plus a
    running dominance structure takes 115 ms -- ten thousand times, from the
    ALGORITHM rather than the language. Dominance counting is a
    computational-geometry problem with O(n log^(d-1) n) solutions; swap one in
    before running this on a real ledger, and keep this version as the reference
    the faster one is differentially tested against.
    """
    denominator = 1.0 + len(nulls)
    out: list[float] = []
    for candidate in candidates:
        dominating = 0
        for null_row in nulls:
            if _dominates(null_row, candidate):
                dominating += 1
        out.append((1.0 + dominating) / denominator)
    return out


def orient_conformal_features(alt_struct_reads: float, identity: float,
                              query_coverage: float, cross_family_margin: float,
                              ref_span_reads: float) -> list[float]:
    """
    Build a feature vector with every coordinate oriented so LARGER is more
    TE-like.

    Four pro-TE coordinates plus `ref_span_reads` NEGATED -- fewer
    reference-spanning reads is more TE-like, so it has to enter with the
    opposite sign or the dominance test is asking the wrong question on that
    axis.
    """
    return [alt_struct_reads, identity, query_coverage, cross_family_margin,
            -ref_span_reads]
