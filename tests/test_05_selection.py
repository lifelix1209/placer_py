"""
Selection: e-BH, the conformal route, and the combination rule.

Mostly INVARIANT tests rather than fixed values, because the guarantees here are
mathematical: they must hold for any correct implementation, in any language,
and they are what a port can most easily break while still looking plausible.
"""

from __future__ import annotations

import math
import random

import pytest
from conftest import call_or_skip, close

from placer_py.core import selection

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


# --------------------------------------------------- combining constructions
@pytest.mark.regression
def test_combination_is_the_mean_not_the_max():
    """
    The maximum of e-values is NOT an e-value: `E_null[max_k E_k]` can exceed 1.
    It obeys `max <= sum`, so `max / K` is valid, which means a plain `max()` is
    anti-conservative by at most a factor of K = 3 (ln 3 = 1.10 nats).

    The arithmetic mean is valid with no assumption, by linearity of
    expectation, and DOMINATES `max / K` because it keeps the other
    constructions' evidence instead of discarding it.

    The C++ used `max()` until this was fixed.
    """
    got = call_or_skip(selection.combine_e_values, [900.0, 0.0, 0.0])
    close(got, 300.0, "one construction firing")
    assert got != 900.0, "this is max(), which is not an e-value"

    both = call_or_skip(selection.combine_e_values, [900.0, 300.0, 0.0])
    close(both, 400.0, "two constructions firing")
    assert both > got, (
        "the mean must reward a second construction; max() would ignore it")


@pytest.mark.regression
def test_the_divisor_is_the_construction_count_not_the_nonzero_count():
    """Dividing by the number of NON-ZERO constructions would make the divisor
    data-dependent and hand back the anti-conservatism."""
    close(call_or_skip(selection.combine_e_values, [300.0, 0.0, 0.0]),
          100.0, "1 of 3 firing")
    close(call_or_skip(selection.combine_e_values, [300.0, 300.0, 300.0]),
          300.0, "3 of 3 firing")


def test_zero_is_a_valid_contribution():
    close(call_or_skip(selection.combine_e_values, [0.0, 0.0, 0.0]), 0.0,
          "nothing applies")


# ---------------------------------------------------------- conformal route
def _features(alt, identity, coverage, margin, ref_span):
    """Go through the orientation helper rather than hand-building vectors.

    My first fixtures passed raw `ref_span_reads`, forgetting that
    `dominance_conformal_p` takes PRE-ORIENTED coordinates where larger is
    always more TE-like. A candidate with 9 reference-spanning reads then looked
    BETTER than a null with 0 on that axis, and the "dominated by every null"
    test failed. Using the helper makes the sign convention impossible to get
    wrong in a test, which is most of what the helper is for.
    """
    return call_or_skip(selection.orient_conformal_features, alt, identity,
                        coverage, margin, ref_span)


def test_the_orientation_helper_negates_reference_support():
    """Fewer reference-spanning reads is more TE-like, so that coordinate must
    enter with the opposite sign or the dominance test asks the wrong question
    on that axis."""
    vector = _features(10.0, 0.95, 0.9, 0.2, 7.0)
    assert vector[:4] == [10.0, 0.95, 0.9, 0.2]
    assert vector[4] == -7.0


def test_dominance_p_is_add_one_smoothed_and_never_zero():
    cand = [_features(10.0, 0.95, 0.9, 0.2, 0.0)]
    nulls = [_features(1.0, 0.3, 0.2, 0.0, 5.0) for _ in range(99)]
    p = call_or_skip(selection.dominance_conformal_p, cand, nulls)
    assert len(p) == 1
    assert p[0] > 0.0, "add-one smoothing must keep p strictly positive"
    close(p[0], 1.0 / 100.0, "no null dominates a clearly better candidate")


def test_a_candidate_dominated_by_every_null_gets_p_near_one():
    cand = [_features(1.0, 0.3, 0.2, 0.0, 9.0)]
    nulls = [_features(10.0, 0.95, 0.9, 0.2, 0.0) for _ in range(99)]
    p = call_or_skip(selection.dominance_conformal_p, cand, nulls)
    assert p[0] > 0.9


def test_a_single_worse_coordinate_defeats_dominance():
    """Dominance is a conjunction over all five axes, which is what makes it a
    partial order: one axis where the null is worse is enough."""
    cand = [_features(10.0, 0.95, 0.9, 0.2, 0.0)]
    almost = [_features(11.0, 0.96, 0.91, 0.19, 0.0) for _ in range(99)]
    p = call_or_skip(selection.dominance_conformal_p, cand, almost)
    close(p[0], 1.0 / 100.0, "margin 0.19 < 0.20 must break dominance")


def test_benjamini_yekutieli_pays_the_harmonic_factor():
    """
    H_m is the price of admitting arbitrary dependence among p-values, and it is
    why the e-value route is primary: at m = 1000 it is about 7.49, turning a
    nominal q = 0.10 into an effective 0.0134.
    """
    m = 1000
    q = 0.10
    h_m = sum(1.0 / i for i in range(1, m + 1))
    assert 7.4 < h_m < 7.6
    # A p-value just inside the rank-1 BY threshold is selected; just outside is not.
    inside = [q / (m * h_m) * 0.99] + [0.9] * (m - 1)
    outside = [q / (m * h_m) * 1.01] + [0.9] * (m - 1)
    assert call_or_skip(selection.benjamini_yekutieli, inside, q) == [0]
    assert call_or_skip(selection.benjamini_yekutieli, outside, q) == []


def test_ebh_is_less_conservative_than_by_on_the_same_evidence():
    """
    The whole reason for the currency switch. An e-value of E corresponds to a
    p-value no larger than 1/E by Markov, so feed both routes the matched
    evidence and e-BH must select at least as much.
    """
    m = 1000
    q = 0.10
    e = [2000.0] * 20 + [1.0] * (m - 20)
    p = [min(1.0, 1.0 / v) for v in e]
    n_e = len(call_or_skip(selection.ebh_select, e, q))
    n_p = len(call_or_skip(selection.benjamini_yekutieli, p, q))
    assert n_e >= n_p
