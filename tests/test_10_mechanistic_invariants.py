"""
The MECHANISTIC layer, constrained by invariants -- not by C++ golden values.

This file targets the EXISTING implementation in `placer_py/redesign/model/mechanistic.py`, so it
runs today and its assertions are about real behaviour.

Why invariants here and golden values in the selection layer: the two codebases
are strong in different halves. The C++ has the statistically correct selection
layer (measured sigma, e-BH, the three fixes from this round) and the Python
does not have one at all -- so there, the C++ numbers bind. Conversely the
Python's mechanistic layer is ahead of the C++: it scores the endonuclease
motif, which the C++ does not model at all; it treats the TSD as a log-LR
against a background probability rather than a +0.15 bonus; and its TE-body term
really is 5'-truncation tolerant. Requiring it to reproduce C++ numbers would
be requiring it to get worse.

So what is pinned here are properties that follow from the MECHANISM, plus a
handful of measured facts about the current constants -- some of which are
recorded as problems rather than as requirements.
"""

from __future__ import annotations

import math

import pytest

from placer_py.redesign.model import mechanistic as M

pytestmark = pytest.mark.invariant


# ===================================================== endonuclease motif
def test_the_consensus_scores_highest_and_a_scrambled_site_lowest():
    """L1 ORF2p nicks at a degenerate 5'-TTTT|AA-3'. Note the consensus: it is
    TTTTAA, not TTAAAA, which is easy to misremember."""
    assert M._pwm_logodds("TTTTAA") == pytest.approx(7.343, abs=0.01)
    assert M._pwm_logodds("GGGGGG") == pytest.approx(-9.657, abs=0.01)
    assert M._pwm_logodds("TTTTAA") > M._pwm_logodds("TTTTAG") \
        > M._pwm_logodds("TTTTGG")


def test_the_motif_is_scored_on_both_strands():
    """
    An insertion into the minus strand presents the consensus reverse
    complemented, so a site that is perfect on one strand must score the same
    either way round. Missing this would halve the term's sensitivity.

    Getting the fixtures right takes care, and my first attempt did not: the
    forward window is `left[-4:] + right[:2]`, while the reverse is
    `revcomp(right[:4] + left[-2:])`. For the reverse to read TTTTAA the
    concatenation must be revcomp("TTTTAA") = "TTAAAA", i.e. `right` must start
    with "TTAA" and `left` must end with "AA".
    """
    plus, ok_plus = M.endonuclease_motif_score("CCCCTTTT", "AAGGGG")
    minus, ok_minus = M.endonuclease_motif_score("CCCCCCAA", "TTAAGGGG")
    assert ok_plus and ok_minus
    assert plus == pytest.approx(7.343, abs=0.01)
    assert minus == pytest.approx(7.343, abs=0.01), (
        "a minus-strand insertion site must score like a plus-strand one")
    # And the max() over strands must not let a poor forward window win.
    forward_window = "CCAA" + "TT"
    assert M._pwm_logodds(forward_window) < minus


def test_short_flanks_are_reported_as_not_evaluated_rather_than_as_zero():
    """A missing measurement and a measurement of zero are different claims,
    and the caller must be able to tell them apart -- `mechanistic_te_score`
    relies on the flag to avoid crediting an unscored motif."""
    score, evaluated = M.endonuclease_motif_score("TTT", "AA")
    assert not evaluated
    assert score == 0.0
    score, evaluated = M.endonuclease_motif_score("CCCCTTTT", "A")
    assert not evaluated


def test_an_unevaluated_motif_contributes_nothing_and_costs_nothing():
    with_flank = M.mechanistic_te_score(
        endonuclease=7.3, endonuclease_evaluated=True, tsd_len=15,
        tsd_significant=True, tsd_bg_p=1e-5, polya_fraction=0.08,
        insert_len=1000, te_core_fraction=0.95, identity=0.96, core_len=950)
    without = M.mechanistic_te_score(
        endonuclease=0.0, endonuclease_evaluated=False, tsd_len=15,
        tsd_significant=True, tsd_bg_p=1e-5, polya_fraction=0.08,
        insert_len=1000, te_core_fraction=0.95, identity=0.96, core_len=950)
    assert without.loglr < with_flank.loglr, "a scored motif must be worth something"
    assert without.endonuclease_score == 0.0
    assert without.hallmark_count == with_flank.hallmark_count - 1


@pytest.mark.xfail(strict=True, reason=(
    "The hallmark gate at endonuclease > 2.0 admits at most ONE mismatch "
    "(2 mismatches score 1.676). The biology is more permissive: the consensus "
    "is degenerate, priming needs as few as 4 matching nt at the primer 3' end "
    "and tolerates terminal mismatches compensated within the last 10 bases "
    "(Monot 2013, PLoS Genet), and the endonuclease is promiscuous in vitro, "
    "cutting largely on DNA STRUCTURE rather than sequence (Miller 2025, JBC). "
    "So genuine insertions at imperfect T-tracts silently lose this hallmark. "
    "The threshold should come from the priming-efficiency curve, not from a "
    "round number."))
def test_a_two_mismatch_target_site_still_counts_as_a_hallmark():
    two_mismatch = M._pwm_logodds("TTTTGG")
    score = M.mechanistic_te_score(
        endonuclease=two_mismatch, endonuclease_evaluated=True, tsd_len=0,
        tsd_significant=False, tsd_bg_p=0.5, polya_fraction=0.0,
        insert_len=1000, te_core_fraction=0.0, identity=0.0, core_len=0)
    assert score.hallmark_count >= 1


# ================================================================= TSD
def test_an_absent_tsd_is_mildly_negative_not_fatal():
    """A TSD is evidence, not a requirement: it must not veto."""
    assert M.tsd_loglr(0, False, 0.5) == pytest.approx(-0.4, abs=1e-9)
    assert M.tsd_loglr(0, False, 0.5) > -1.0


def test_an_in_range_tsd_beats_an_out_of_range_one():
    """Staggered top/bottom-strand nicking gives ~7-20 bp, and the code admits
    4-25."""
    assert M.tsd_loglr(15, False, 0.5) > M.tsd_loglr(40, False, 0.5)
    assert M.tsd_loglr(15, False, 0.5) > M.tsd_loglr(2, False, 0.5)


def test_a_rarer_duplication_is_worth_more():
    """The background probability is what ties the TSD to THIS locus: a
    duplication that is common in the local sequence says nothing."""
    rare = M.tsd_loglr(15, True, 1e-6)
    common = M.tsd_loglr(15, True, 0.5)
    assert rare > common
    assert M.tsd_loglr(15, True, 1e-9) <= 3.0, "clamped, so it cannot swamp"


@pytest.mark.xfail(strict=True, reason=(
    "TSD absence is scored at a flat -0.4 regardless of mechanism, so a "
    "full-length L1, a 5'-inverted (twin-primed) L1 and a trans-mobilised Alu "
    "or SVA are all penalised for obeying their own mechanism. Only "
    "5'-TRUNCATED L1 insertions are reliably associated with short TSDs "
    "(Kojima 2010, Mobile DNA). P(TSD | mechanism) has to be conditioned, which "
    "makes the insertion type a latent variable -- see placer_py/core/tprt.py's "
    "TSD_MODEL. The C++ has the same defect, so this is not a porting "
    "regression."))
def test_tsd_absence_is_conditioned_on_the_insertion_mechanism():
    """There is currently no way to express "no TSD expected here"."""
    import inspect
    params = inspect.signature(M.tsd_loglr).parameters
    assert any("mech" in p or "type" in p for p in params), (
        f"tsd_loglr takes {list(params)} -- no mechanism argument")


# ============================================================== poly(A)
def test_polya_is_monotone_in_tail_length():
    """TPRT primes from the poly(A) tail, and retrotransposition rate rises
    with tail length."""
    previous = -math.inf
    for tail in (0, 5, 10, 15, 18, 25, 40, 60, 100):
        value = M.polya_loglr(tail / 1000.0, 1000)
        assert value >= previous, f"non-monotone at tail={tail}"
        previous = value


def test_polya_has_a_lag_and_a_plateau():
    """Sigmoidal, centred near 18 nt: below the lag it is worth little, above
    the plateau extra length adds nothing."""
    lag = M.polya_loglr(8 / 1000.0, 1000)
    mid = M.polya_loglr(18 / 1000.0, 1000)
    plateau = M.polya_loglr(60 / 1000.0, 1000)
    far = M.polya_loglr(200 / 1000.0, 1000)
    assert lag < mid < plateau
    assert far - plateau < 0.05, "the plateau must actually saturate"


def test_polya_cannot_dominate_the_score():
    """Bounded to roughly [-0.3, 1.3], so a homopolymer alone never carries a
    call. This matters because A-rich tracts are common in the human genome --
    precisely because of past TE insertions."""
    assert M.polya_loglr(1.0, 100000) < 1.4
    assert M.polya_loglr(0.0, 0) > -0.35


# =========================================================== TE body
def test_the_te_body_term_is_truncation_tolerant():
    """
    The claim in its docstring, and it holds: past ~120 bp of core the term is
    flat, so a 1 kb 5'-truncated L1 scores the same as a 6 kb full-length one.

    This is the single place where the Python is clearly ahead of the C++, whose
    `sequence_signal` weights `best_query_coverage` at 2.45 and so penalises the
    modal true positive -- 5' truncation happens in ~95% of L1 insertions
    (Levin 2025, Genetics).
    """
    truncated = M.te_body_loglr(1.0, 0.96, 1000)
    full = M.te_body_loglr(1.0, 0.96, 6000)
    assert truncated == pytest.approx(full, abs=1e-9)
    assert truncated > 2.0


def test_a_very_short_core_is_discounted():
    """Below ~120 bp the length factor bites, which is right: a 40 bp match to a
    TE consensus is not much of an observation."""
    assert M.te_body_loglr(1.0, 0.96, 40) < M.te_body_loglr(1.0, 0.96, 200)


def test_identity_enters_relative_to_an_ambient_divergence():
    """Centred at 0.70, so a copy at ambient reference divergence earns nothing
    from identity and only a young one is rewarded."""
    assert M.te_body_loglr(1.0, 0.70, 1000) < M.te_body_loglr(1.0, 0.96, 1000)
    assert M.te_body_loglr(0.0, 0.0, 0) == pytest.approx(-1.0, abs=1e-9)


# ====================================== the conjunction, and the key claim
def test_no_single_hallmark_can_reach_the_strong_tier():
    """The precision lever is the CONJUNCTION. Any single block reaching STRONG
    on its own would defeat the design."""
    for kwargs in (
        dict(endonuclease=7.3, endonuclease_evaluated=True),
        dict(tsd_len=15, tsd_significant=True, tsd_bg_p=1e-9),
        dict(polya_fraction=0.10, insert_len=1000),
        dict(te_core_fraction=1.0, identity=0.99, core_len=6000),
    ):
        base = dict(endonuclease=0.0, endonuclease_evaluated=False, tsd_len=0,
                    tsd_significant=False, tsd_bg_p=0.5, polya_fraction=0.0,
                    insert_len=1000, te_core_fraction=0.0, identity=0.0,
                    core_len=0)
        base.update(kwargs)
        score = M.mechanistic_te_score(**base)
        assert score.qc != "MECH_TPRT_STRONG", f"{kwargs} reached STRONG alone"
        assert score.hallmark_count <= 1


def test_linkage_hallmarks_separate_a_mismapped_old_copy():
    """
    THE central discrimination, and the existing model gets the structure right.

    The dominant false positive is a read mismapped from an OLD reference copy,
    which is itself a TPRT product -- so it carries a real poly(A) and a real TE
    body. Those INTERNAL terms cannot distinguish it from a new insertion. What
    can are the LINKAGE terms, which are properties of THIS locus: the
    endonuclease motif in the local reference, and the duplication of the local
    flanks.

    Measured: a real insertion gets 4 hallmarks / +9.76 nats / STRONG, and the
    mismapped copy with identical internals gets 2 / +3.66 / SUPPORTED. The
    internal blocks are byte-identical between them; the separation is entirely
    in the motif and the TSD.

    The C++ cannot do this at all: it has no endonuclease term, and its TSD is a
    +0.15 bonus on a bounded signal.
    """
    internals = dict(polya_fraction=0.08, insert_len=1000,
                     te_core_fraction=0.95, identity=0.96, core_len=950)
    real = M.mechanistic_te_score(
        endonuclease=7.3, endonuclease_evaluated=True, tsd_len=15,
        tsd_significant=True, tsd_bg_p=1e-5, **internals)
    mismapped = M.mechanistic_te_score(
        endonuclease=-3.0, endonuclease_evaluated=True, tsd_len=0,
        tsd_significant=False, tsd_bg_p=0.5, **internals)

    assert real.polya_loglr == pytest.approx(mismapped.polya_loglr, abs=1e-12)
    assert real.te_body_loglr == pytest.approx(mismapped.te_body_loglr, abs=1e-12)
    assert real.qc == "MECH_TPRT_STRONG"
    assert mismapped.qc != "MECH_TPRT_STRONG"
    assert real.loglr > mismapped.loglr + 5.0


def test_hallmark_count_drives_the_qc_tier():
    def score(n_hallmarks: int):
        parts = [
            dict(endonuclease=7.3, endonuclease_evaluated=True),
            dict(tsd_len=15, tsd_significant=True, tsd_bg_p=1e-9),
            dict(polya_fraction=0.10, insert_len=1000),
            dict(te_core_fraction=1.0, identity=0.99, core_len=6000),
        ]
        base = dict(endonuclease=0.0, endonuclease_evaluated=False, tsd_len=0,
                    tsd_significant=False, tsd_bg_p=0.5, polya_fraction=0.0,
                    insert_len=1000, te_core_fraction=0.0, identity=0.0,
                    core_len=0)
        for part in parts[:n_hallmarks]:
            base.update(part)
        return M.mechanistic_te_score(**base)

    assert score(0).qc == "MECH_NONE"
    assert score(1).qc == "MECH_SINGLE_HALLMARK"
    assert score(2).qc == "MECH_TPRT_SUPPORTED"
    assert score(3).qc == "MECH_TPRT_STRONG"
    assert score(4).qc == "MECH_TPRT_STRONG"


# ============================== the high-precision gate, and what it excludes
def test_the_high_precision_statistic_is_length_times_quality():
    assert M.high_precision_te_score(0.96, 0.95, 1000) == pytest.approx(912.0)
    assert M.high_precision_te_score(0.0, 1.0, 6000) == 0.0


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED CONSEQUENCE of the calibrated gate, recorded because it is a "
    "recall property nobody would notice from the code. hp_score is "
    "identity * coverage * insert_len and the threshold is 3500, so at identity "
    "0.95 and coverage 0.90 the insert must exceed 4094 bp. That excludes, "
    "structurally and regardless of evidence quality: every Alu (~300 bp, the "
    "most common new insertion in humans) and every 5'-truncated L1 below 4 kb "
    "-- i.e. the MODAL insertion, since ~95% of L1 insertions are 5'-truncated "
    "(Levin 2025). The tier reaches ~0.94 precision against TLDR by selecting "
    "near-full-length L1, which is the easy case. The statistic conflates 'is "
    "it TE' with 'is it long', and the threshold then acts as a length filter. "
    "Separating the two -- a length-independent quality term, and length only "
    "where it belongs -- is the fix."))
def test_a_perfect_alu_can_reach_the_confident_tier():
    hp = M.high_precision_te_score(1.0, 1.0, 300)
    assert M.is_confident_te_insertion(hp, 1.0, True)


def test_the_confident_tier_requires_a_resolved_family():
    hp = M.high_precision_te_score(0.96, 0.95, 6000)
    assert M.is_confident_te_insertion(hp, 0.95, True)
    assert not M.is_confident_te_insertion(hp, 0.95, False), (
        "an unresolved family must not reach the precision-first tier")


def test_the_threshold_is_an_external_calibration_not_a_derivation():
    """
    Pinned so it is visible rather than buried: 3500 and 0.80 come from
    `scripts/calibrate_mechanistic_vs_tldr.py`, i.e. from fitting to another
    caller's output.

    That makes the quoted ~0.94 a precision AGAINST TLDR, not a precision, and
    it imports TLDR's biases as ground truth. It is the "where do the weights
    come from" problem: the gold standard for "is there a TE insertion here" is
    the thing being produced, so using another caller's calls as labels learns
    its blind spots too. Both defaults should eventually be derived or replaced
    by the FDR machinery in the selection layer.
    """
    assert M.DEFAULT_HP_MIN == 3500.0
    assert M.DEFAULT_HP_COV_MIN == 0.80
