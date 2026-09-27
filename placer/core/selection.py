"""
Selection: e-BH.

It controls FDR <= q under ARBITRARY dependence among the e-values with no
correction factor (Wang & Ramdas 2022, JRSS-B 84(3):822), because its proof
uses only linearity of expectation. A p-value route under the same dependence
has to pay Benjamini-Yekutieli's harmonic factor H_m -- about 7.49 at m = 1000,
i.e. a nominal q = 0.10 becomes an effective 0.0134. Switching the currency from
tail probabilities to expectations removes that entirely.

This is only sound if the inputs really are e-values: the decoy check in
`core/mechanism_selection.py` is what verifies that they are.
"""

from __future__ import annotations

from collections.abc import Sequence


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
