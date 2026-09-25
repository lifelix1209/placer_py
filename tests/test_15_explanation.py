"""Pareto explanation comparison. Ported from event_explanation.cpp."""

from __future__ import annotations

import pytest
from conftest import call_or_skip

from placer.core import explanation as E


def _res(s, m, u, b, r, rc, a, la, pc):
    return E.ExplanationResidual(
        structural_conflicts=s, missing_required_components=m,
        unexplained_high_complexity_bases=u, breakpoint_disagreement_bp=b,
        read_assignment_conflicts=r, reference_counterevidence=rc,
        artifact_evidence=a, label_ambiguity=la, path_complexity=pc)


def _exp(kind, res, family="NA", subfamily="NA"):
    return E.EventExplanation(kind=kind, residual=res, family=family,
                              subfamily=subfamily)


K = E.ExplanationKind

#: Hand-written explanation sets, one per decision the comparison can reach.
SCENARIOS = {
    "te_dominates_closed": ([
        _exp(K.TE, _res(0, 0, 0, 0, 0, 0, 0, 0, 1), "L1", "L1HS"),
        _exp(K.INSERTION_NON_TE, _res(0, 0, 160, 0, 0, 0, 0, 0, 1)),
        _exp(K.REFERENCE, _res(1, 1, 0, 0, 3, 3, 0, 0, 0)),
        _exp(K.ARTIFACT, _res(0, 0, 106, 0, 1, 3, 3, 0, 1)),
    ], True),
    "te_dominates_imprecise": ([
        _exp(K.TE, _res(0, 0, 0, 25, 0, 0, 0, 0, 1), "L1", "L1HS"),
        _exp(K.INSERTION_NON_TE, _res(0, 0, 160, 0, 0, 0, 0, 0, 1)),
        _exp(K.REFERENCE, _res(1, 1, 0, 0, 3, 3, 0, 0, 0)),
    ], False),
    # MISNAMED ON PURPOSE, kept because it caught a false assumption of mine:
    # TE's primary residual sum is HIGHER here (2 vs 0), so non-TE dominates and
    # the answer is PASS_NONTE_INSERTION. I had asserted this produced
    # TE_AMBIGUOUS and the test failed, correctly.
    "te_loses_to_nonte": ([
        _exp(K.TE, _res(0, 0, 0, 0, 2, 0, 0, 0, 1), "L1", "L1HS"),
        _exp(K.INSERTION_NON_TE, _res(0, 0, 0, 0, 0, 0, 0, 0, 1)),
    ], True),
    # A REAL one: identical primary residuals, so neither dominates. TE sorts
    # first only because its kind ordinal outranks non-TE's, then fails the
    # unique-dominance test against an equal rival.
    "te_ambiguous": ([
        _exp(K.TE, _res(0, 0, 0, 0, 1, 0, 0, 0, 0), "L1", "L1HS"),
        _exp(K.INSERTION_NON_TE, _res(0, 0, 0, 0, 1, 0, 0, 0, 0)),
    ], True),
    "incomplete": ([
        _exp(K.TE, _res(0, 1, 0, 0, 0, 0, 0, 0, 1), "L1", "L1HS"),
        _exp(K.REFERENCE, _res(1, 1, 0, 0, 3, 3, 0, 0, 0)),
    ], True),
    "nonte_insertion": ([
        _exp(K.INSERTION_NON_TE, _res(0, 0, 0, 0, 0, 0, 0, 0, 1)),
        _exp(K.TE, _res(0, 0, 300, 0, 0, 0, 1, 1, 2), "UNKNOWN", "UNKNOWN"),
        _exp(K.ARTIFACT, _res(0, 0, 106, 0, 1, 3, 3, 0, 1)),
    ], True),
    "artifact_wins": ([
        _exp(K.ARTIFACT, _res(0, 0, 0, 0, 0, 0, 0, 0, 1)),
        _exp(K.TE, _res(0, 0, 400, 0, 5, 9, 2, 1, 2), "UNKNOWN", "UNKNOWN"),
    ], True),
    "te_unknown_family_closed": ([
        _exp(K.TE, _res(0, 0, 0, 0, 0, 0, 0, 1, 2), "UNKNOWN", "UNKNOWN"),
        _exp(K.REFERENCE, _res(1, 1, 0, 0, 3, 3, 0, 0, 0)),
    ], True),
}

#: (a, b, a dominates b, b dominates a). The expected answers follow from the
#: definition: no worse on every primary coordinate and strictly better on at
#: least one, with the three tie-break coordinates ignored.
PAIRS = {
    "strictly_better_everywhere": (_res(0, 0, 0, 0, 0, 0, 0, 0, 0),
                                   _res(1, 1, 1, 1, 1, 1, 1, 1, 1),
                                   True, False),
    "equal": (_res(1, 1, 1, 0, 1, 1, 1, 0, 0), _res(1, 1, 1, 0, 1, 1, 1, 0, 0),
              False, False),
    "one_better_one_worse": (_res(0, 1, 0, 0, 0, 0, 0, 0, 0),
                             _res(1, 0, 0, 0, 0, 0, 0, 0, 0), False, False),
    "one_strictly_better_rest_equal": (_res(0, 1, 1, 0, 1, 1, 1, 0, 0),
                                       _res(1, 1, 1, 0, 1, 1, 1, 0, 0),
                                       True, False),
    "tie_break_fields_ignored": (_res(1, 1, 1, 99, 1, 1, 1, 99, 99),
                                 _res(1, 1, 1, 0, 1, 1, 1, 0, 0),
                                 False, False),
}


@pytest.mark.invariant
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_the_dominance_truth_table(name):
    a, b, a_dom_b, b_dom_a = PAIRS[name]
    assert call_or_skip(E.dominates_primary_residuals, a, b) == a_dom_b, (
        f"{name}: a dominates b")
    assert call_or_skip(E.dominates_primary_residuals, b, a) == b_dom_a, (
        f"{name}: b dominates a")


@pytest.mark.invariant
def test_tie_break_coordinates_are_excluded_from_dominance():
    """
    `breakpoint_disagreement_bp`, `label_ambiguity` and `path_complexity` break
    ties only. A candidate must never be rejected for an imprecise breakpoint
    alone, which is why an otherwise-identical residual with a 99 bp
    disagreement still fails to be dominated.
    """
    imprecise = _res(1, 1, 1, 99, 1, 1, 1, 99, 99)
    precise = _res(1, 1, 1, 0, 1, 1, 1, 0, 0)
    assert not call_or_skip(E.dominates_primary_residuals, precise, imprecise)
    assert not call_or_skip(E.dominates_primary_residuals, imprecise, precise)


@pytest.mark.invariant
def test_dominance_is_irreflexive_and_asymmetric():
    r = _res(2, 1, 5, 0, 3, 1, 0, 0, 0)
    assert not call_or_skip(E.dominates_primary_residuals, r, r), (
        "nothing strictly dominates itself")
    better = _res(1, 1, 5, 0, 3, 1, 0, 0, 0)
    assert call_or_skip(E.dominates_primary_residuals, better, r)
    assert not call_or_skip(E.dominates_primary_residuals, r, better)


@pytest.mark.invariant
def test_a_higher_primary_sum_loses_even_for_te():
    """Kind does not rescue a worse residual: the kind ordinal breaks TIES, it
    does not outrank the dominance test or the primary sum."""
    explanations, closed = SCENARIOS["te_loses_to_nonte"]
    decision = call_or_skip(E.compare_event_explanations, explanations, closed)
    assert decision.best.kind == K.INSERTION_NON_TE
    assert decision.final_qc == "PASS_NONTE_INSERTION"


@pytest.mark.invariant
def test_te_ambiguous_is_abstention_not_rejection():
    """
    The state a weighted sum cannot produce. TE is the best explanation and has
    no missing components, but it does not uniquely dominate -- evidence
    sufficient, not exclusive -- so the honest action is to abstain.
    """
    explanations, closed = SCENARIOS["te_ambiguous"]
    decision = call_or_skip(E.compare_event_explanations, explanations, closed)
    assert decision.final_qc == "TE_AMBIGUOUS"
    assert decision.best.kind == K.TE, "TE really is the best explanation"
    assert not decision.emit_te_call, "but nothing is emitted"


@pytest.mark.invariant
def test_missing_components_short_circuit_before_the_kind_is_consulted():
    """A structurally incomplete best explanation is refused whatever its kind,
    so `NO_CALL_INCOMPLETE` cannot be reached past a TE branch."""
    explanations, closed = SCENARIOS["incomplete"]
    decision = call_or_skip(E.compare_event_explanations, explanations, closed)
    assert decision.final_qc == "NO_CALL_INCOMPLETE"
    assert not decision.emit_te_call


def test_an_empty_explanation_set_is_a_no_call():
    decision = call_or_skip(E.compare_event_explanations, [], True)
    assert decision.final_qc == "NO_CALL_INCOMPLETE"
    assert decision.alternatives == []
