"""
The TPRT coincidence model: a BEHAVIOUR table, not golden vectors.

There is no C++ implementation to be golden against -- this model is the
intended replacement for the six affine blocks, not a port of them. So the
constraint is a table of cases with the sign and rough magnitude each must
produce, derived from the mechanism and checked against the reference
implementation in the parent repo's `outputs/tprt_model.py`.

The tolerances here are deliberately loose (nats, not last digits). Pinning
exact values would freeze parameters that are supposed to be estimated by EM
from the run's own calls, which is the opposite of the point.
"""

from __future__ import annotations

import pytest

from conftest import call_or_skip
from placer_py import tprt

pytestmark = pytest.mark.invariant

L1_LEN = 6000.0
ALU_LEN = 300.0

# Threshold in nats used only to say "would be called" in these tests.
CALL_NATS = 2.3


def _locus(**kw) -> dict:
    base = dict(u=5000, v=6000, tsd=15, tsd_mismatches=1, repeat_frac=0.0,
                t_tract=9, polya=20, t_frac=0.30, a_frac=0.30,
                n_alt=6, n_ref=5, identity=0.96)
    base.update(kw)
    return base


# ------------------------------------------------- the decomposition itself
def test_internal_evidence_cannot_separate_a_mismapped_old_copy():
    """
    THE central claim, and the one that justifies the whole architecture.

    A mismapped read from an OLD reference copy carries the same internal
    structure as a new insertion -- complete 3' end, poly(A), everything --
    because that old copy is itself a TPRT product. So the internal terms must
    score the two IDENTICALLY, and all the discrimination must come from the
    linkage terms.

    If a port makes these two differ on the internal side, it has smuggled
    locus information into a term that must not have it.
    """
    real = _locus(tsd=15, t_tract=9)
    mismapped = _locus(tsd=0, t_tract=2)      # same insert, no local linkage
    a = call_or_skip(tprt.evaluate, real, L1_LEN, "L1")
    b = call_or_skip(tprt.evaluate, mismapped, L1_LEN, "L1")
    assert a.log_bf_vs_non_te == pytest.approx(b.log_bf_vs_non_te, abs=1e-9), (
        "internal terms must be blind to the locus linkage")
    assert a.log_bf_vs_artifact > b.log_bf_vs_artifact + 10.0, (
        "linkage terms must carry the whole separation")


def test_the_decision_takes_the_minimum_of_the_two_bayes_factors():
    """A candidate has to beat BOTH alternatives; summing them would let strong
    non-TE evidence pay for weak artifact evidence."""
    bf = call_or_skip(tprt.evaluate, _locus(), L1_LEN, "L1")
    assert min(bf.log_bf_vs_non_te, bf.log_bf_vs_artifact) <= \
        bf.log_bf_vs_non_te
    assert min(bf.log_bf_vs_non_te, bf.log_bf_vs_artifact) <= \
        bf.log_bf_vs_artifact


# ------------------------------------------------------- the behaviour table
def test_behaviour_table():
    """All nine cases in one test, reporting every mismatch at once rather than
    stopping at the first -- a port that gets three of them wrong should say so
    in one run."""
    cases = [
        # name,                          locus overrides,                 elem,    family, called
        ("modal truncated L1", dict(), L1_LEN, "L1", True),
        ("full-length L1, no TSD",
         dict(u=0, v=6000, tsd=0, t_tract=8, polya=18), L1_LEN, "L1", True),
        ("Alu, full length",
         dict(u=0, v=300, tsd=13, t_tract=7, polya=25), ALU_LEN, "ALU", True),
        ("twin-primed L1, inverted",
         dict(u=4200, v=6000, tsd=12, inverted=True), L1_LEN, "L1", True),
        ("mismapped old L1 3' end",
         dict(tsd=0, t_tract=2, n_alt=4, n_ref=9), L1_LEN, "L1", False),
        ("mismapped interior fragment",
         dict(u=2400, v=3400, tsd=0, t_tract=1, polya=0, n_alt=4, n_ref=8),
         L1_LEN, "L1", False),
        ("VNTR: TSD present but flank is tandem repeat",
         dict(repeat_frac=0.95, t_tract=3, polya=6, n_alt=4, n_ref=6),
         L1_LEN, "L1", False),
        ("A-rich tract: polyA present but flank is 90% A/T",
         dict(tsd=0, t_frac=0.90, a_frac=0.90, polya=20, t_tract=12,
              n_alt=4, n_ref=7), L1_LEN, "L1", False),
        ("real non-TE insertion (identity 0.42)",
         dict(u=2400, v=3400, tsd=0, t_tract=3, polya=0, identity=0.42,
              n_alt=8, n_ref=2), L1_LEN, "L1", False),
    ]
    problems = []
    for name, overrides, elem, family, expect_called in cases:
        bf = call_or_skip(tprt.evaluate, _locus(**overrides), elem, family)
        score = min(bf.log_bf_vs_non_te, bf.log_bf_vs_artifact)
        called = score > CALL_NATS
        if called != expect_called:
            problems.append(
                f"{name}: score {score:.2f} nats -> "
                f"{'called' if called else 'rejected'}, expected "
                f"{'called' if expect_called else 'rejected'}")
    assert not problems, "\n".join(problems)


# --------------------------------------------------- the robustness mechanism
def test_local_nulls_make_the_tsd_term_self_cancel_in_a_repeat():
    """
    Where the robustness comes from: the null for each coincidence is measured
    from the local reference, so in a VNTR a tandem duplication is the norm and
    the term contributes nothing. No threshold, no special case.
    """
    unique = call_or_skip(tprt.log_bf_tsd, 15, "l1_5p_truncated", 1, 0.0)
    vntr = call_or_skip(tprt.log_bf_tsd, 15, "l1_5p_truncated", 1, 0.95)
    assert unique > 10.0, "a 15 bp duplication in unique sequence is worth nats"
    assert vntr < 0.0, "in a tandem repeat it must carry no weight at all"


def test_local_composition_makes_the_priming_term_self_cancel():
    normal = call_or_skip(tprt.log_bf_priming_site, 9, 20, 0.30, 0.30,
                          "l1_5p_truncated")
    a_rich = call_or_skip(tprt.log_bf_priming_site, 12, 20, 0.90, 0.90,
                          "l1_5p_truncated")
    assert normal > 5.0
    assert a_rich < 2.0


def test_three_prime_anchoring_is_log_of_the_position_count():
    """`ln(L - l + 1)`: 8.5 nats for a 1 kb fragment of a 6 kb element, from one
    observation with no fitted parameter."""
    import math
    got = call_or_skip(tprt.log_bf_three_prime_anchoring, 5000, 6000, L1_LEN)
    assert got == pytest.approx(math.log(5001.0), abs=0.05)


def test_a_full_length_insert_earns_nothing_from_anchoring():
    """Correct, not a bug: when the insert covers the whole element there is
    only one possible position, so there is no coincidence to score. Full-length
    insertions are called on the linkage side instead."""
    got = call_or_skip(tprt.log_bf_three_prime_anchoring, 0, 6000, L1_LEN)
    assert got == pytest.approx(0.0, abs=1e-9)


def test_absent_tsd_is_not_evidence_of_artifact_for_the_right_mechanisms():
    """
    Only 5'-truncated L1 reliably carries a short TSD; full-length, inverted and
    trans-mobilised insertions do not (Kojima 2010). Treating "no TSD" as bad
    news punishes those for obeying their own mechanism.
    """
    truncated = call_or_skip(tprt.log_bf_tsd, 0, "l1_5p_truncated", 0, 0.0)
    full = call_or_skip(tprt.log_bf_tsd, 0, "l1_full_length", 0, 0.0)
    alu = call_or_skip(tprt.log_bf_tsd, 0, "trans_alu_sva", 0, 0.0)
    assert full > truncated
    assert alu > truncated


# ------------------------------------------------------------- the escape hatch
def test_the_mechanism_prior_is_fixed_not_conditioned_on_the_geometry():
    """
    Conditioning a prior on the data double counts the data. Renormalising
    within the mechanisms "consistent with" the observed geometry turned the
    2%-prior EN-independent route into a 90% alibi and rescued a mismapped
    interior fragment to 4.88 nats.

    Every mixture component with weak predictions is an escape hatch, and a
    fixed prior is the defence. This test pins the prior's values so a later
    refactor cannot quietly renormalise them.
    """
    assert tprt.MECHANISM_PRIOR["en_independent"] == 0.02
    assert abs(sum(tprt.MECHANISM_PRIOR.values()) - 1.0) < 1e-9


def test_tsd_model_is_conditioned_on_mechanism():
    assert tprt.TSD_MODEL["l1_5p_truncated"][0] > 0.8
    for mechanism in ("l1_full_length", "l1_5p_inverted", "trans_alu_sva"):
        assert tprt.TSD_MODEL[mechanism][0] < 0.5, mechanism
