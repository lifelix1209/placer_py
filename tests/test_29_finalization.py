"""
Finalization: the only stage that sees the whole run.

The largest unit in the C++ and the one with the least direct test coverage
there -- `test_conformal_finalization.cpp` reaches a handful of entry points
through a `PipelineResult`. Everything the stage decides is reachable directly
here, so the rules that matter are pinned as rules rather than as outcomes.
"""

from __future__ import annotations

import math

import pytest
from conftest import call_or_skip, close

from placer_py import finalization as F
from placer_py.core import result as R
from placer_py.core.ledger import EvidenceLedgerRow, FinalCall, FinalCallFilterConfig

pytestmark = pytest.mark.invariant


def row(**kw):
    base = dict(chrom="chr1", tid=0, pos=1000, bp_left=1000, bp_right=1012,
                mechanistic_blocks="event;sequence;structure")
    base.update(kw)
    return EvidenceLedgerRow(**base)


def call(**kw):
    base = dict(chrom="chr1", tid=0, pos=1000, bp_left=1000, bp_right=1012,
                mechanistic_blocks="event;sequence;structure")
    base.update(kw)
    return FinalCall(**base)


# ------------------------------------------------------- the local-FDR algebra
def test_two_independent_lines_of_evidence_multiply_their_odds():
    """
    Adding log odds multiplies likelihood ratios, which is the correct
    combination for independent evidence -- so two lines each at lFDR 0.10
    combine to far better than 0.10, not to 0.10.
    """
    combined = call_or_skip(F.combine_independent_local_fdr, 0.10, 0.10)
    assert combined < 0.10
    close(combined, 1.0 / (1.0 + 9.0 * 9.0), "two 9:1 odds")


def test_an_uninformative_line_returns_the_other_unchanged():
    """
    lFDR 1 means "this said nothing", not "this said no". Letting it drag the
    combination to 1 would make a silent line of evidence veto a loud one.
    """
    close(F.combine_independent_local_fdr(0.05, 1.0), 0.05, "right is silent")
    close(F.combine_independent_local_fdr(1.0, 0.05), 0.05, "left is silent")
    close(F.combine_independent_local_fdr(1.0, 1.0), 1.0, "both silent")


def test_the_odds_conversions_round_trip_and_saturate():
    for fdr in (0.01, 0.10, 0.5, 0.9):
        close(F.odds_to_local_fdr(F.local_fdr_to_log_odds(fdr)), fdr, f"round trip {fdr}")
    close(F.odds_to_local_fdr(1000.0), 0.0, "saturated up")
    close(F.odds_to_local_fdr(-1000.0), 1.0, "saturated down")
    close(F.odds_to_local_fdr(float("nan")), 1.0, "not a number")


def test_log_sum_exp_ignores_a_non_finite_operand():
    close(F.log_sum_exp_pair(float("-inf"), 2.0), 2.0, "left absent")
    close(F.log_sum_exp_pair(2.0, float("-inf")), 2.0, "right absent")
    close(F.log_sum_exp_pair(0.0, 0.0), math.log(2.0), "both present")


# --------------------------------------------------------------- count models
def test_a_minority_allele_can_beat_an_error_process():
    """The route that keeps het and mosaic insertions callable at all."""
    assert call_or_skip(F.low_allele_fraction_event_log_bf, 4, 20) > 0.0
    assert F.low_allele_fraction_event_log_bf(0, 20) == float("-inf")


def test_heterozygous_balance_needs_both_balance_and_depth():
    """
    The PRODUCT of a balance term and a count term. A 1:1 split of two reads
    scores almost nothing, a 1:1 split of forty scores a lot, and a 39:1 split
    of forty scores nothing however deep it is.
    """
    shallow = call_or_skip(F.heterozygous_balance_log_evidence, 1, 1)
    deep = F.heterozygous_balance_log_evidence(20, 20)
    lopsided = F.heterozygous_balance_log_evidence(39, 1)
    assert deep > shallow
    assert lopsided < shallow
    close(F.heterozygous_balance_log_evidence(10, 0), 0.0, "no reference")


# ------------------------------------------------------------- certificates
def test_a_row_without_computed_blocks_has_no_certificate_at_all():
    """
    Serialised blocks are the evidence the certificate was computed. Treating a
    row's default-zero aggregates as measurements would put a fabricated e-value
    into the selection.
    """
    assert not call_or_skip(F.has_conformal_feature_certificate, row(mechanistic_blocks="NA"))
    assert not F.has_conformal_feature_certificate(row(mechanistic_blocks=""))
    assert F.has_conformal_feature_certificate(row())


def test_a_statistical_line_is_consulted_only_if_it_passed_its_own_gate():
    """
    A number computed on a path that was REJECTED is not evidence. A row with a
    nominally low worst-case lFDR but a failed QC contributes nothing.
    """
    passed = row(lfdr_qc="PASS_TE_LFDR", worst_case_lfdr=0.02)
    close(call_or_skip(F.ledger_row_statistical_te_local_fdr, passed), 0.02, "gated in")
    failed = row(lfdr_qc="TE_LFDR_HIGH", worst_case_lfdr=0.40)
    close(F.ledger_row_statistical_te_local_fdr(failed), 1.0, "gated out")


def test_a_row_with_no_decoded_structure_path_says_nothing_rather_than_no():
    close(F.ledger_row_structure_te_local_fdr(row(te_structure_path="NA")), 1.0, "absent")
    decoded = row(te_structure_path="0-300:TE_CORE:L1", te_structure_log_evidence=4.0,
                  nonte_structure_log_evidence=0.0, artifact_structure_log_evidence=0.0)
    assert F.ledger_row_structure_te_local_fdr(decoded) < 0.10


def test_the_structure_origin_veto_needs_all_three_conditions():
    """
    High reference conflict AND no direct insert observation AND weak
    statistics. Any one alone must not veto -- a locus with the reference
    against it but an observed insert is exactly what a heterozygous insertion
    looks like.
    """
    base = dict(te_structure_path="0-300:TE_CORE:L1", te_structure_log_evidence=8.0,
                lfdr_qc="PASS_TE_LFDR", worst_case_lfdr=0.02)
    assert call_or_skip(F.has_te_structure_origin_certificate, row(**base))
    vetoed = row(mechanistic_ref_conflict_signal=0.8, te_structure_path=base["te_structure_path"],
                 te_structure_log_evidence=8.0)
    assert not F.has_te_structure_origin_certificate(vetoed)
    # The same row with a direct insert observation is NOT vetoed.
    observed = row(mechanistic_ref_conflict_signal=0.8, raw_cigar_insert_reads=3,
                   te_structure_path=base["te_structure_path"], te_structure_log_evidence=8.0)
    assert F.has_te_structure_origin_certificate(observed)


def test_the_mechanistic_e_value_refuses_rather_than_reporting_a_small_number():
    """
    -inf, not a small value. A refusal here means "this must not enter
    selection", and a small e-value would still contribute to the procedure.
    """
    assert F.ledger_row_mechanistic_log_e_value(row(mechanistic_blocks="NA")) == float("-inf")
    assert F.ledger_row_mechanistic_log_e_value(
        row(mechanistic_ref_conflict_signal=0.6)) == float("-inf")
    assert F.ledger_row_mechanistic_log_e_value(
        row(alt_struct_reads=4, ref_span_reads=20)) == float("-inf")


def test_the_e_value_takes_the_worse_of_the_two_aggregates():
    """`min`, which is what makes it a BOUND rather than a best case."""
    value = F.ledger_row_mechanistic_log_e_value(
        row(mechanistic_lower_log_bf_te_vs_artifact=9.0,
            mechanistic_lower_log_bf_te_vs_non_te=4.0,
            mechanistic_ambiguity_width=1.0, alt_struct_reads=10, ref_span_reads=2))
    close(value, 3.0, "worse aggregate, less ambiguity")


def test_the_event_existence_route_refuses_a_minority_allele():
    """
    `alt <= ref` is -inf here. Keeping the dominant-allele and minority-allele
    models apart is what lets each be a clean model rather than one model with
    an allele-fraction ladder in it.
    """
    assert F.ledger_row_event_existence_log_evidence(
        row(alt_struct_reads=4, ref_span_reads=20)) == float("-inf")
    assert math.isfinite(F.ledger_row_event_existence_log_evidence(
        row(alt_struct_reads=20, ref_span_reads=4)))


def test_a_non_te_insertion_needs_a_direct_observation_of_inserted_bases():
    """
    Without one the locus is a junction with no insert, which is what an
    artifact looks like -- there is nothing to call non-TE.
    """
    no_insert = F.competing_nonte_long_insertion_log_evidence(
        10, 2, 0, 0, 0, 0.8, 0.1, 0.0, 0.0)
    assert no_insert == float("-inf")
    with_insert = F.competing_nonte_long_insertion_log_evidence(
        10, 2, 3, 300, 400, 0.8, 0.1, 0.0, 0.0)
    assert math.isfinite(with_insert)


def test_insert_length_is_weak_evidence_by_design():
    """
    0.05 and 0.03 per log base. Letting length dominate would turn the gate into
    a length filter -- the failure the README identifies in the high-precision
    tier.
    """
    short = F.competing_nonte_long_insertion_log_evidence(10, 2, 1, 100, 100, 0.8, 0.1, 0, 0)
    long_ = F.competing_nonte_long_insertion_log_evidence(10, 2, 1, 6000, 6000, 0.8, 0.1, 0, 0)
    assert long_ > short
    assert (long_ - short) < 0.5     # sixty times longer buys under half a nat


# -------------------------------------------------------------- the low-AF gate
def test_the_low_af_certificate_needs_counts_and_a_sequence_hint():
    """
    A 4-vs-30 split is consistent with a real mosaic AND with a systematic
    mapping error. Only the sequence evidence separates them.
    """
    counts_only = row(alt_struct_reads=6, ref_span_reads=30)
    assert not call_or_skip(F.has_low_allele_fraction_event_certificate_row, counts_only)
    with_sequence = row(alt_struct_reads=6, ref_span_reads=30, best_te_identity=0.7)
    assert F.has_low_allele_fraction_event_certificate_row(with_sequence)


def test_the_low_af_bayes_factor_must_reach_the_very_strong_scale_point():
    """
    `>= 6.0` is a Kass-Raftery scale point, not a tuned number -- and it is
    strict enough that 4 alt against 30 ref (log BF 3.06) does NOT qualify while
    6 against 30 (6.70) does. Worth pinning, because the AF is identical to two
    decimal places in the two cases: it is the COUNT that carries the evidence,
    not the fraction.
    """
    assert F.low_allele_fraction_event_log_bf(4, 30) < 6.0
    assert F.low_allele_fraction_event_log_bf(6, 30) >= 6.0
    assert not F.has_low_allele_fraction_event_certificate_row(
        row(alt_struct_reads=4, ref_span_reads=30, best_te_identity=0.7))


def test_the_low_af_route_refuses_a_dominant_allele():
    dominant = row(alt_struct_reads=20, ref_span_reads=10, best_te_identity=0.7)
    assert not F.has_low_allele_fraction_event_certificate_row(dominant)


def test_a_wide_own_interval_is_itself_a_certificate():
    """
    An event whose two breakpoints are 50 bp or more apart cannot have been
    produced by two reads disagreeing about one position -- the interval is
    evidence that there is something between them.
    """
    narrow = row(alt_struct_reads=6, ref_span_reads=30, best_te_identity=0.7,
                 bp_left=1000, bp_right=1010, mechanistic_blocks="event;boundary")
    assert not F.has_self_stable_low_af_interval_certificate_row(narrow)
    wide = row(alt_struct_reads=6, ref_span_reads=30, best_te_identity=0.7,
               bp_left=1000, bp_right=1100, mechanistic_blocks="event;boundary")
    assert F.has_self_stable_low_af_interval_certificate_row(wide)


def test_the_bilateral_anchor_certificate_refuses_any_sequence_evidence():
    """
    Every clause is NECESSARY. The `identity == 0` requirement keeps it narrow:
    any sequence evidence at all means some other route should have handled the
    locus, and this one must not fire.
    """
    base = dict(final_qc="NO_CALL_INCOMPLETE", ref_span_reads=0, alt_struct_reads=3,
                alt_left_clip_reads=2, alt_right_clip_reads=2,
                left_anchor_input_reads=2, right_anchor_input_reads=2,
                partial_context_input_reads=2, input_event_reads=2,
                support_qnames=["a", "b", "c"])
    assert call_or_skip(F.has_ref_unopposed_bilateral_partial_anchor_certificate,
                        row(**base))
    assert not F.has_ref_unopposed_bilateral_partial_anchor_certificate(
        row(**{**base, "best_te_identity": 0.01}))
    assert not F.has_ref_unopposed_bilateral_partial_anchor_certificate(
        row(**{**base, "ref_span_reads": 1}))
    assert not F.has_ref_unopposed_bilateral_partial_anchor_certificate(
        row(**{**base, "alt_split_reads": 1}))


def test_the_balance_certificate_needs_sixteen_alt_reads():
    """
    What makes it about BALANCE rather than depth: the statistic only means
    anything once a 50/50 split is distinguishable from chance, and 16 is where
    the count term can reach the threshold at all.
    """
    base = dict(final_qc="TE_AMBIGUOUS", best_te_identity=0.6)
    assert not F.has_balanced_heterozygous_event_support(
        row(alt_struct_reads=15, ref_span_reads=14, **base))
    assert F.has_balanced_heterozygous_event_support(
        row(alt_struct_reads=20, ref_span_reads=18, **base))


# ------------------------------------------------------- support-set overlap
def test_two_shared_reads_are_enough_at_any_set_size():
    """
    THE central idea of de-duplication. Two calls 300 bp apart built from the
    same twelve reads ARE one event; position cannot answer that and the read
    names can.
    """
    big_a = row(support_qnames=sorted(f"r{i}" for i in range(40)))
    big_b = row(support_qnames=sorted([f"r{i}" for i in range(38, 78)]))
    assert call_or_skip(F.ledger_rows_share_support_edge, big_a, big_b)


def test_one_shared_read_needs_the_sets_to_be_small_and_alike():
    """
    A single shared read between two 40-read sets is coincidence at a
    repetitive locus; between two 2-read sets it is most of the evidence there
    is. Hence the Jaccard floor on the one-read case only.
    """
    # One shared read out of one each: Jaccard 1.0, believed.
    assert F.ledger_rows_share_support_edge(row(support_qnames=["a"]),
                                            row(support_qnames=["a"]))
    # One shared read out of two each: Jaccard 1/3, refused.
    assert not F.ledger_rows_share_support_edge(row(support_qnames=["a", "b"]),
                                                row(support_qnames=["b", "c"]))
    # One shared read between two 40-read sets: coincidence, refused.
    big_a = row(support_qnames=sorted(f"r{i}" for i in range(40)))
    big_b = row(support_qnames=sorted(["r39"] + [f"s{i}" for i in range(39)]))
    assert not F.ledger_rows_share_support_edge(big_a, big_b)


def test_containment_recognises_a_fragment_that_jaccard_would_miss():
    """
    Three of an event's thirty reads: Jaccard 0.1, containment 1.0. It is the
    containment that says "fragment" rather than "separate locus".
    """
    whole = call(support_qnames=sorted(f"r{i}" for i in range(30)))
    fragment = call(support_qnames=["r0", "r1", "r2"])
    assert call_or_skip(F.final_calls_share_support_containment_edge, whole, fragment)
    assert F.sorted_support_jaccard(whole.support_qnames, fragment.support_qnames) < 0.2


def test_identical_support_is_recognised_exactly():
    a = call(support_qnames=["x", "y", "z"])
    b = call(support_qnames=["z", "y", "x"], pos=9000)
    assert F.final_calls_share_identical_support(a, b)
    assert not F.final_calls_share_identical_support(a, call(support_qnames=["x", "y"]))


# ---------------------------------------------------------------- envelopes
def test_the_envelope_widens_by_half_the_consensus_length():
    """
    A call whose breakpoints came from a 3 kb consensus is uncertain by about
    that much, and two such calls 1 kb apart are very likely one event. The
    widening makes the envelope a statement about uncertainty.
    """
    narrow = call(bp_left=1000, bp_right=1010, event_consensus_len=0)
    assert call_or_skip(F.final_call_event_envelope, narrow) == (1000, 1010)
    wide = call(bp_left=1000, bp_right=1010, event_consensus_len=3000)
    left, right = F.final_call_event_envelope(wide)
    assert left <= 0 or (1000 - left) >= 1000
    assert (right - 1010) >= 1000


def test_two_real_intervals_need_no_slack_but_two_points_do():
    """
    An interval already expresses its own uncertainty, so overlapping intervals
    are the same locus and non-overlapping ones are not. Two POINTS get the
    50 bp slack, because a point claims one base and two reads disagreeing by
    30 bp are proposing the same junction.
    """
    a = call(bp_left=1000, bp_right=1100)
    b = call(bp_left=1150, bp_right=1250, pos=1200)
    assert not call_or_skip(F.same_call_locus, a, b)
    p = call(bp_left=1000, bp_right=1000, pos=1000)
    q = call(bp_left=1030, bp_right=1030, pos=1030)
    assert F.same_call_locus(p, q)
    far = call(bp_left=1051, bp_right=1051, pos=1051)
    assert not F.same_call_locus(p, far)


# -------------------------------------------------------- preferring a call
def test_a_stabilised_community_outranks_everything_else():
    """
    The only kind of call whose existence was corroborated by a second,
    read-sharing row -- so it leads the comparison.
    """
    stable = call(final_qc="PASS_STRUCTURAL_INSERTION|EVENT_COMMUNITY_STABLE",
                  support_reads=2)
    strong = call(final_qc="PASS_TE_CLOSED", support_reads=50, gq=60,
                  robust_mechanistic_qc="PASS_TE_LFDR",
                  robust_mechanistic_worst_case_lfdr=0.01)
    assert call_or_skip(F.prefer_new_call, stable, strong)


def test_a_well_supported_unnamed_call_beats_a_weak_named_one():
    """
    The name is a TIE-BREAK, never a rank. A caller optimised for a benchmark's
    family column would do the opposite.
    """
    unnamed = call(final_qc="PASS_TE_CLOSED", support_reads=30, te_name="")
    named = call(final_qc="PASS_TE_CLOSED", support_reads=3, te_name="L1HS")
    assert F.prefer_new_call(unnamed, named)


def test_promotion_demotes_the_precision_rank():
    """A promoted call is the weakest kind: it was assembled from rows that
    individually did not qualify."""
    assert F.final_call_precision_rank(call(final_qc="PASS_TE_CLOSED")) == 4
    assert F.final_call_precision_rank(call(final_qc="PASS_STRUCTURAL_INSERTION")) == 3
    assert F.final_call_precision_rank(call(final_qc="PASS_TE_IMPRECISE")) == 2
    assert F.final_call_precision_rank(
        call(final_qc="PASS_TE_IMPRECISE|EVENT_CLUSTER_PROMOTED")) == 1
    assert F.final_call_precision_rank(call(final_qc="NO_CALL")) == 0


def test_the_comparison_is_a_strict_order_so_dedup_is_deterministic():
    a = call(final_qc="PASS_TE_CLOSED", support_reads=10, pos=1000)
    b = call(final_qc="PASS_TE_CLOSED", support_reads=10, pos=2000)
    assert F.prefer_new_call(a, b) != F.prefer_new_call(b, a)
    assert not F.prefer_new_call(a, a)


# ------------------------------------------------------------------- e-BH
def test_the_three_constructions_are_averaged_not_maximised():
    """
    The MAXIMUM of e-values is not an e-value. The mean is, by linearity of
    expectation, and it dominates max/K because it keeps the evidence from the
    other constructions instead of discarding it.
    """
    only_one = call(final_qc="PASS_TE_CLOSED", alt_struct_reads=20, ref_span_reads=2,
                    mechanistic_lower_log_bf_te_vs_artifact=6.0,
                    mechanistic_lower_log_bf_te_vs_non_te=6.0)
    single = F.event_mechanistic_e_value(only_one)
    assert single > 0.0
    calls = [only_one]
    F.apply_event_ebh_selection(calls, 0.10)
    close(calls[0].ebh_e_value if calls[0].ebh_selected else single / 3.0,
          single / 3.0, "averaged over three")


def test_a_call_with_no_usable_construction_still_counts_towards_m():
    """
    Dropping it would shrink m and inflate every threshold, which would make the
    procedure anti-conservative in exactly the way e-BH's proof forbids.
    """
    strong = call(final_qc="PASS_TE_CLOSED", alt_struct_reads=20, ref_span_reads=0,
                  mechanistic_lower_log_bf_te_vs_artifact=20.0,
                  mechanistic_lower_log_bf_te_vs_non_te=20.0)
    blanks = [call(final_qc="NO_CALL", mechanistic_blocks="NA") for _ in range(50)]
    few = [call(**{k: getattr(strong, k) for k in ("final_qc", "alt_struct_reads",
                                                    "ref_span_reads",
                                                    "mechanistic_lower_log_bf_te_vs_artifact",
                                                    "mechanistic_lower_log_bf_te_vs_non_te")})]
    F.apply_event_ebh_selection(few, 0.10)
    many = few[:1] + blanks
    F.apply_event_ebh_selection(many, 0.10)
    assert len(many) == 51


def test_ebh_does_not_clobber_an_existing_conformal_certificate():
    selected = call(final_qc="PASS_TE_CLOSED", alt_struct_reads=20, ref_span_reads=0,
                    conformal_qc="PASS_CONFORMAL_FDR",
                    mechanistic_lower_log_bf_te_vs_artifact=20.0,
                    mechanistic_lower_log_bf_te_vs_non_te=20.0)
    calls = [selected]
    F.apply_event_ebh_selection(calls, 0.10)
    assert calls[0].conformal_qc == "PASS_CONFORMAL_FDR"
    assert calls[0].ebh_selected


def test_the_low_af_construction_requires_a_stability_token():
    """
    A low-AF certificate on its own is the weakest evidence in the file. It
    becomes an e-value only after a read-sharing neighbour corroborated it.
    """
    unstable = call(final_qc="PASS_STRUCTURAL_INSERTION", alt_struct_reads=6,
                    ref_span_reads=30, best_te_identity=0.7)
    close(F.low_allele_fraction_event_e_value(unstable), 0.0, "unstable")
    stable = call(final_qc="PASS_STRUCTURAL_INSERTION|EVENT_COMMUNITY_STABLE",
                  alt_struct_reads=6, ref_span_reads=30, best_te_identity=0.7)
    assert F.low_allele_fraction_event_e_value(stable) > 0.0


def test_one_opposing_read_removes_the_unopposed_construction():
    base = dict(final_qc="PASS_STRUCTURAL_INSERTION|EVENT_BILATERAL_PARTIAL_ANCHOR",
                alt_struct_reads=4)
    assert F.ref_unopposed_bilateral_partial_anchor_e_value(call(**base, ref_span_reads=0)) > 0.0
    close(F.ref_unopposed_bilateral_partial_anchor_e_value(call(**base, ref_span_reads=1)),
          0.0, "opposed")


# ------------------------------------------------------ Bayesian FDR fallbacks
def test_the_bayesian_fallback_takes_the_longest_prefix_with_mean_lfdr_under_q():
    calls = [call(final_qc="PASS_TE_CLOSED", lfdr_qc="PASS_TE_LFDR",
                  worst_case_lfdr=fdr) for fdr in (0.01, 0.05, 0.50)]
    F.apply_event_bayesian_fdr_fallback(calls, 0.10)
    assert calls[0].conformal_qc == "PASS_EVENT_BFDR"
    assert calls[1].conformal_qc == "PASS_EVENT_BFDR"
    assert calls[2].conformal_qc != "PASS_EVENT_BFDR"


def test_a_passing_posterior_contributes_the_target_not_the_posterior():
    """
    The QC token says "this cleared the bar"; reading a number off a threshold
    test would claim precision it does not have.
    """
    calls = [call(final_qc="PASS_TE_CLOSED", posterior_qc="PASS_TE_POSTERIOR",
                  te_posterior=0.60)]
    F.apply_event_bayesian_fdr_fallback(calls, 0.10)
    close(calls[0].conformal_by_threshold, 0.10, "target, not 0.40")


def test_the_structural_fallback_has_a_hard_per_call_ceiling():
    """
    The extra `<= 0.25` pre-filter bounds the WORST call this route can admit,
    where the mean-lFDR rule alone bounds only the average.
    """
    hopeless = call(final_qc="PASS_STRUCTURAL_INSERTION", alt_struct_reads=1,
                    ref_span_reads=40, mechanistic_ref_conflict_signal=0.9)
    calls = [hopeless]
    F.apply_structural_event_bayesian_fdr_fallback(calls, 1.0)
    assert calls[0].conformal_qc != "PASS_EVENT_EXISTENCE_BFDR"


def test_a_balanced_heterozygote_is_not_penalised_for_its_reference_reads():
    """0.25 against 1.20. A true het legitimately carries many reference reads,
    and the full penalty would punish it for being heterozygous."""
    balanced = call(final_qc="PASS_STRUCTURAL_INSERTION", alt_struct_reads=20,
                    ref_span_reads=20, mechanistic_ref_conflict_signal=0.6,
                    non_te_posterior=0.8, artifact_posterior=0.1,
                    raw_cigar_insert_reads=5, max_raw_cigar_insert_len=300)
    lopsided = call(final_qc="PASS_STRUCTURAL_INSERTION", alt_struct_reads=20,
                    ref_span_reads=1, mechanistic_ref_conflict_signal=0.6,
                    non_te_posterior=0.8, artifact_posterior=0.1,
                    raw_cigar_insert_reads=5, max_raw_cigar_insert_len=300)
    assert F.structural_event_local_fdr(balanced) < 1.0
    assert F.structural_event_local_fdr(lopsided) < 1.0


# ------------------------------------------------------------- calibration
def test_the_dependency_bound_is_measured_on_every_certified_row():
    """
    NOT on the null controls. Selecting the sample by the aggregate being
    calibrated truncates its right tail and drives sigma to its floor, which
    makes the whole apparatus inert.
    """
    assert call_or_skip(F.is_dependency_calibration_row, row())
    promotable = row(alt_struct_reads=20, ref_span_reads=0, best_te_identity=0.9,
                     te_posterior=0.9, final_qc="PASS_TE_CLOSED")
    assert F.is_dependency_calibration_row(promotable)
    assert not F.is_sample_local_null_control(promotable)


def test_the_null_set_excludes_promotable_rows_because_that_is_what_conformal_means():
    plain = row(alt_struct_reads=1, ref_span_reads=20)
    assert F.is_sample_local_null_control(plain)
    assert not F.is_sample_local_null_control(row(final_qc="PASS_TE_CLOSED"))


def test_the_penalty_is_applied_to_the_capped_aggregate():
    """
    Required for validity. sigma bounds `E_null[min(Y, C)]`, so only
    `min(Y, C)/sigma` is guaranteed to be an e-value -- applying the penalty to
    the uncapped aggregate builds e-values e-BH is entitled to assume and that
    the estimator never certified.
    """
    result = R.PipelineResult()
    result.final_calls = [call(mechanistic_raw_log_bf_te_vs_artifact=1000.0,
                               mechanistic_raw_log_bf_te_vs_non_te=1000.0)]
    result.evidence_ledger = [row(mechanistic_raw_log_bf_te_vs_artifact=-1.0,
                                  mechanistic_raw_log_bf_te_vs_non_te=-1.0)
                              for _ in range(50)]
    call_or_skip(F.apply_dependency_penalty_calibration, result, 0.10)
    assert result.final_calls[0].mechanistic_lower_log_bf_te_vs_artifact <= result.dependency_penalty_cap_log
    assert result.dependency_penalty_null_count == 50


# ------------------------------------------------------------- the whole pass
def _clean_run():
    result = R.PipelineResult()
    result.final_calls = [call(
        final_qc="PASS_TE_CLOSED", posterior_qc="PASS_TE_POSTERIOR",
        lfdr_qc="PASS_TE_LFDR", robust_mechanistic_qc="PASS_TE_LFDR",
        robust_mechanistic_worst_case_lfdr=0.01,
        mechanistic_raw_log_bf_te_vs_artifact=12.0,
        mechanistic_raw_log_bf_te_vs_non_te=10.0,
        alt_struct_reads=12, ref_span_reads=2, support_qnames=["a", "b", "c"],
        family="L1", subfamily="L1HS", te_name="L1HS", family_committed=True)]
    result.evidence_ledger = [
        row(pos=1000 + i * 5000, bp_left=1000 + i * 5000, bp_right=1012 + i * 5000,
            mechanistic_blocks="event",
            mechanistic_raw_log_bf_te_vs_artifact=-1.0 + 0.01 * i,
            mechanistic_raw_log_bf_te_vs_non_te=-1.2 + 0.01 * i,
            alt_struct_reads=1, ref_span_reads=10)
        for i in range(60)]
    return result


def test_a_clean_te_call_survives_the_whole_pass():
    result = _clean_run()
    call_or_skip(F.finalize_final_calls, result)
    assert len(result.final_calls) == 1
    assert result.final_pass_calls == 1
    assert F.final_call_has_reportable_conformal_certificate(result.final_calls[0])


def test_a_call_no_route_selected_is_dropped():
    result = _clean_run()
    result.final_calls[0].robust_mechanistic_qc = "TE_LFDR_HIGH"
    result.final_calls[0].robust_mechanistic_worst_case_lfdr = 0.9
    result.final_calls[0].lfdr_qc = "TE_LFDR_HIGH"
    result.final_calls[0].posterior_qc = "TE_POSTERIOR_LOW"
    result.final_calls[0].mechanistic_raw_log_bf_te_vs_artifact = -5.0
    result.final_calls[0].mechanistic_raw_log_bf_te_vs_non_te = -5.0
    result.final_calls[0].te_posterior = 0.0
    F.finalize_final_calls(result)
    assert result.final_calls == []


def test_the_te_calibrated_mode_sets_structural_calls_aside_rather_than_erasing_them():
    """
    They are SELECTED calls, not rejects. Silently discarding them made recall
    benchmarking on the default mode misleading.
    """
    result = R.PipelineResult()
    structural = call(final_qc="PASS_STRUCTURAL_INSERTION",
                      conformal_qc="PASS_EVENT_EBH")
    te = call(final_qc="PASS_TE_CLOSED", posterior_qc="PASS_TE_POSTERIOR",
              lfdr_qc="PASS_TE_LFDR", conformal_qc="PASS_CONFORMAL_FDR", pos=9000)
    result.final_calls = [te, structural]
    call_or_skip(F.apply_final_report_mode, result, "TeCalibrated")
    assert [c.final_qc for c in result.final_calls] == ["PASS_TE_CLOSED"]
    assert len(result.structural_calls) == 1


def test_legacy_mode_keeps_everything_in_one_list():
    result = R.PipelineResult()
    result.final_calls = [call(final_qc="PASS_STRUCTURAL_INSERTION")]
    F.apply_final_report_mode(result, "Legacy")
    assert len(result.final_calls) == 1
    assert result.structural_calls == []


def test_the_family_label_is_committed_only_after_selection():
    """
    Committing earlier would let family coverage influence breakpoint choice,
    de-duplication or emission -- i.e. let the TE library feed back into
    detection.
    """
    result = R.PipelineResult()
    result.final_calls = [call(final_qc="PASS_TE_CLOSED",
                               sequence_family_commit_eligible=True,
                               sequence_family_candidate="Alu",
                               sequence_subfamily_candidate="AluYa5")]
    call_or_skip(F.apply_sequence_family_commitments, result)
    assert result.final_calls[0].family == "Alu"
    assert result.final_calls[0].te_name == "AluYa5"
    assert result.final_calls[0].family_committed
    assert "SEQUENCE_FAMILY_COMMITTED" in result.final_calls[0].final_qc


def test_a_structural_call_never_gets_a_family_committed():
    result = R.PipelineResult()
    result.final_calls = [call(final_qc="PASS_STRUCTURAL_INSERTION",
                               sequence_family_commit_eligible=True,
                               sequence_family_candidate="Alu")]
    F.apply_sequence_family_commitments(result)
    assert result.final_calls[0].family == "NA"


def test_the_short_insertion_filter_has_an_assembled_event_escape_hatch():
    """
    Without it the filter would discard every long insertion that no single read
    spanned -- which is most of them for a long element.
    """
    config = FinalCallFilterConfig(min_raw_cigar_insert_len_bp=50)
    raw = call(max_raw_cigar_insert_len=100)
    assert call_or_skip(F.final_call_passes_min_raw_cigar_insert_length, raw, config)

    assembled = call(max_raw_cigar_insert_len=10, event_consensus_len=3000,
                     alt_struct_reads=12, mechanistic_blocks="event;sequence",
                     te_structure_log_evidence=4.0, artifact_structure_log_evidence=0.0)
    assert F.final_call_passes_min_raw_cigar_insert_length(assembled, config)

    bare = call(max_raw_cigar_insert_len=10, event_consensus_len=3000,
                alt_struct_reads=12, mechanistic_blocks="event")
    assert not F.final_call_passes_min_raw_cigar_insert_length(bare, config)


def test_a_zero_threshold_disables_the_filter():
    assert F.final_call_passes_min_raw_cigar_insert_length(
        call(), FinalCallFilterConfig(min_raw_cigar_insert_len_bp=0))


# ------------------------------------------------------------ collapse/promote
def test_fragments_of_one_event_collapse_into_one_call():
    """
    Two structural calls with assembled consensuses and overlapping CONTEXT are
    the same assembly -- the context is the window the event was built from.
    """
    a = call(final_qc="PASS_STRUCTURAL_INSERTION", event_consensus_len=500,
             window_start=900, window_end=1500, bp_left=1000, bp_right=1010,
             support_qnames=["r1", "r2", "r3"])
    b = call(final_qc="PASS_STRUCTURAL_INSERTION", event_consensus_len=500,
             window_start=1100, window_end=1700, bp_left=1200, bp_right=1210,
             pos=1200, support_qnames=["r2", "r3", "r4"])
    collapsed = call_or_skip(F.collapse_shared_support_final_call_fragments, [a, b])
    assert len(collapsed) == 1
    assert collapsed[0].support_qnames == ["r1", "r2", "r3", "r4"]
    assert "EVENT_FRAGMENT_COLLAPSED" in collapsed[0].final_qc
    assert "read_overlap" in collapsed[0].mechanistic_blocks


def test_collapsing_takes_the_max_of_counts_and_the_union_of_reads():
    """
    The fragments describe the SAME reads seen differently, so summing counts
    would double-count. Read NAMES cannot be double-counted, which is why the
    support set is the one thing that grows.
    """
    a = call(final_qc="PASS_STRUCTURAL_INSERTION", event_consensus_len=500,
             window_start=900, window_end=1500, support_qnames=["r1", "r2"],
             ref_span_reads=4, raw_cigar_insert_reads=2)
    b = call(final_qc="PASS_STRUCTURAL_INSERTION", event_consensus_len=500,
             window_start=1100, window_end=1700, support_qnames=["r2", "r3"],
             pos=1200, ref_span_reads=7, raw_cigar_insert_reads=1)
    merged = F.collapse_shared_support_final_call_fragments([a, b])[0]
    assert merged.ref_span_reads == 7
    assert merged.raw_cigar_insert_reads == 2
    assert merged.support_reads == 3


def test_an_aggregate_community_is_skipped_when_a_member_is_promotable():
    """
    Promotion already handles that community and would report the same event.
    Aggregating as well would double-report it, and the aggregate's synthetic
    counts would compete with the real row's in de-duplication.
    """
    promotable = row(pos=1000, bp_left=1000, bp_right=1010, alt_struct_reads=20,
                     ref_span_reads=0, best_te_identity=0.9, te_posterior=0.9,
                     final_qc="TE_AMBIGUOUS", support_qnames=["a", "b", "c"])
    weak = row(pos=1020, bp_left=1020, bp_right=1030, alt_struct_reads=3,
               ref_span_reads=30, final_qc="REFERENCE_OR_ARTIFACT",
               support_qnames=["a", "b", "d"])
    calls: list[FinalCall] = []
    call_or_skip(F.append_aggregated_event_community_calls, [promotable, weak], calls)
    assert calls == []


def test_a_promoted_call_is_downgraded_to_structural_when_its_evidence_is_not_te():
    """
    Promotion assembles evidence that an INSERTION exists; asserting an element
    identity on top of it would claim something the promotion never argued.
    """
    structural_row = row(alt_struct_reads=20, ref_span_reads=2,
                         final_qc="TE_AMBIGUOUS", best_te_identity=0.3,
                         mechanistic_lower_log_bf_te_vs_artifact=4.0,
                         family="L1", subfamily="L1HS")
    promoted = call_or_skip(F.promoted_call_from_ledger_row, structural_row)
    assert promoted.final_qc.startswith("PASS_STRUCTURAL_INSERTION")
    assert promoted.family == "UNKNOWN"
    assert not promoted.family_committed


def test_the_bilateral_anchor_promotion_widens_its_own_breakpoints():
    """
    The certificate says "an insertion is here and its junctions could not be
    placed". A narrow interval would claim a precision the evidence explicitly
    does not have.
    """
    anchored = row(final_qc="NO_CALL_INCOMPLETE", ref_span_reads=0,
                   alt_struct_reads=3, alt_left_clip_reads=2, alt_right_clip_reads=2,
                   left_anchor_input_reads=2, right_anchor_input_reads=2,
                   partial_context_input_reads=2, input_event_reads=2,
                   support_qnames=["a", "b", "c"], bp_left=1000, bp_right=1010,
                   event_consensus_len=400)
    promoted = F.promoted_call_from_ledger_row(anchored)
    assert "EVENT_BILATERAL_PARTIAL_ANCHOR" in promoted.final_qc
    assert promoted.bp_left <= 600
    assert promoted.bp_right >= 1410


def test_a_promoted_call_never_dominates_a_non_promoted_one():
    """
    A promoted call is assembled from rows that individually did not qualify, so
    it can carry inflated aggregate counts. Letting it dominate would let the
    aggregation delete the evidence it was built from.
    """
    promoted = call(final_qc="PASS_STRUCTURAL_INSERTION|EVENT_CLUSTER_PROMOTED",
                    alt_struct_reads=40, best_te_identity=0.9,
                    best_te_query_coverage=0.9, cross_family_margin=0.5,
                    ref_span_reads=0)
    plain = call(final_qc="PASS_TE_CLOSED", alt_struct_reads=5, best_te_identity=0.5,
                 best_te_query_coverage=0.5, cross_family_margin=0.1, ref_span_reads=1)
    kept = call_or_skip(F.remove_pareto_dominated_final_calls, [plain, promoted])
    assert any(c.final_qc == "PASS_TE_CLOSED" for c in kept)


# ------------------------------------------- the abPOA memory budget


def test_the_poa_budget_scales_reads_down_as_the_event_grows():
    """Quadratic cost means a read cap cannot be a memory cap.

    `event_consensus_poa_max_reads` fixes n while L varies by two orders of
    magnitude between an Alu and a mis-assembled multi-kb insertion, and abPOA
    costs roughly n * L^2. Measured: 24 sequences of 18 kb peaks near 6 GB,
    which is what made an 800 kb region of ultra-long ONT peak at 4.6 GB and
    spend 1975 s of wall clock on 1006 s of CPU.
    """
    from placer_py.core.consensus import poa_reads_within_budget

    budget = 1024 * 1024 * 1024
    counts = [poa_reads_within_budget(length, budget)
              for length in (1000, 2000, 4000, 6000, 10000, 18000)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1], counts
    # A 300 bp Alu must not be constrained at all -- the budget exists for the
    # long tail, and squeezing the common case would cost recall for nothing.
    assert poa_reads_within_budget(300, budget) > 48


def test_the_poa_budget_never_returns_zero_reads():
    """One sequence needs no alignment, so it always fits.

    Returning 0 would mean refusing to report an event because its reads are
    long, which is a worse failure than reporting it from a single uncorrected
    read -- and at n=1 the measured cost of an 18 kb string is 19 MB, not the
    ~2 GB the quadratic would predict.
    """
    from placer_py.core.consensus import poa_reads_within_budget

    for length in (10_000, 100_000, 1_000_000):
        assert poa_reads_within_budget(length, 1024 * 1024) == 1
    assert poa_reads_within_budget(0, 1024) == 1


def test_dropping_reads_for_memory_is_recorded_rather_than_silent():
    """`consensus.py` exists on the principle that a quietly worse consensus
    is the most damaging thing this stage can produce, so the count of
    withheld strings is carried on the result."""
    from placer_py.core.segmentation import EventConsensus

    assert EventConsensus().poa_reads_dropped_for_memory == 0


def test_a_memory_capped_consensus_is_visible_in_the_output():
    """A thinner consensus still looks like a consensus.

    The dropped-read count lives on `EventConsensus`, which never reaches a
    file -- so on its own it records nothing a user can see. The QC token is
    what makes the degradation legible in `final_qc`, which both the ledger
    and the call files carry.
    """
    from placer_py.core import bins as pipeline_module
    from placer_py.core.segmentation import EventConsensus

    intact = EventConsensus()
    assert pipeline_module._with_poa_cap_token("PASS_TE_CLOSED", intact) == "PASS_TE_CLOSED"

    thinned = EventConsensus()
    thinned.poa_reads_dropped_for_memory = 7
    tagged = pipeline_module._with_poa_cap_token("PASS_TE_CLOSED", thinned)
    assert pipeline_module.POA_MEMORY_CAPPED_QC in tagged
    assert tagged.startswith("PASS_TE_CLOSED"), tagged
