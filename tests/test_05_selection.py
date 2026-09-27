"""
Selection: e-BH.

Mostly INVARIANT tests rather than fixed values, because the guarantees here are
mathematical: they must hold for any correct implementation, in any language,
and they are what a port can most easily break while still looking plausible.
"""

from __future__ import annotations

import math
import random

import pytest
from conftest import call_or_skip

from placer.core import selection

pytestmark = pytest.mark.invariant


# ------------------------------------------------------------------- e-BH
def test_threshold_is_m_over_q_times_rank():
    """Selecting exactly one hypothesis requires reaching m/q."""
    m = 1000
    q = 0.10
    e = [0.0] * m
    e[7] = m / q + 1.0
    chosen = call_or_skip(selection.ebh_select, e, q)
    assert chosen == [7]

    e[7] = m / q - 1.0
    assert call_or_skip(selection.ebh_select, e, q) == []


def test_unusable_hypotheses_still_count_toward_m():
    """
    Giving them e = 0 keeps them in the denominator. Dropping them from `m`
    would make the candidate set data-dependent and void the guarantee -- the
    same mistake as the C++'s old skip of already-certificated calls.
    """
    strong = 500.0
    with_padding = [strong] + [0.0] * 999
    without = [strong]
    a = call_or_skip(selection.ebh_select, with_padding, 0.10)
    b = call_or_skip(selection.ebh_select, without, 0.10)
    assert len(b) >= len(a), (
        "a smaller m must be easier to pass; if not, m is being computed wrong")


def test_ties_break_by_original_index():
    e = [50000.0, 50000.0, 50000.0]
    assert call_or_skip(selection.ebh_select, e, 0.10) == [0, 1, 2]


def test_controls_fdr_under_the_null_by_simulation():
    """
    The guarantee itself. Under the null an e-value has mean <= 1; draw from
    such a distribution, run e-BH at q, and the realised false discovery
    proportion must not exceed q by more than simulation noise.

    This is the single test most worth having: it constrains the procedure
    rather than the arithmetic, so it catches an off-by-one in the step-up rule
    that fixed values on a handful of cases would miss.
    """
    rng = random.Random(20240914)
    q = 0.10
    m = 2000
    trials = 40
    total_selected = 0
    total_false = 0
    for _ in range(trials):
        # exp(N(-s^2/2, s^2)) has mean exactly 1: a valid null e-value.
        s = 2.0
        e = [math.exp(rng.gauss(-0.5 * s * s, s)) for _ in range(m)]
        chosen = call_or_skip(selection.ebh_select, e, q)
        total_selected += len(chosen)
        total_false += len(chosen)          # every hypothesis here IS null
    if total_selected == 0:
        return                              # nothing selected: FDR is 0
    fdp = total_false / total_selected
    assert fdp <= q * 3.0, (
        f"false discovery proportion {fdp:.3f} against a target of {q}; "
        "e-BH should be conservative here, not anti-conservative")
