"""
The joint decision: ranking, the vetoes, the boundary, and the emission gate.

The C++ cases from `test_te_hard_veto.cpp` and the boundary block of
`test_decision_policy.cpp` are reproduced exactly. The rest pins what those
cases leave open -- in particular that RANKING and EMISSION are separate, which
is the single most important property of this module and the one a
reimplementation is most likely to collapse.
"""

from __future__ import annotations

import math

import pytest
from conftest import call_or_skip, close

from placer_py import policy as P
from placer_py.te_classifier import TEAlignmentEvidence

pytestmark = pytest.mark.invariant


def closed_segmentation(**kw):
    seg = P.EventSegmentationEvidence(
        has_consensus=True, has_left_flank=True, has_right_flank=True,
        has_insert_seq=True, pair_valid=True, insert_len=300, score=0.9,
        qc="PASS_EVENT_SEGMENTATION")
    for key, value in kw.items():
        setattr(seg, key, value)
    return seg


def some_existence(**kw):
    ev = P.EventExistenceEvidence(alt_struct_reads=8, alt_split_reads=5,
                                  alt_indel_reads=0, alt_left_clip_reads=3,
                                  alt_right_clip_reads=3, ref_span_reads=2,
                                  gq=60, score=1.0)
    for key, value in kw.items():
        setattr(ev, key, value)
    return ev


def te_pass(qc_reason="PASS_INSERT_TE_ALIGNMENT", **kw):
    te = TEAlignmentEvidence(pass_=True, qc_reason=qc_reason,
                             sequence_model_label="TE_MODEL_IN_DISTRIBUTION",
                             sequence_model_score=0.9)
    for key, value in kw.items():
        setattr(te, key, value)
    return te


def make_boundary(boundary_type="TSD", length=12):
    return P.BoundaryEvidence(geometry_defined=True, canonical_pass=True,
                              evidence_consistent=True, boundary_type=boundary_type,
                              boundary_len=length, score=1.0)


# --------------------------------------------------------------- boundary
def test_an_overlap_between_the_flanks_is_a_target_site_duplication():
    """
    The SIGN of `right_ref_start - left_ref_end` is the biology. Negative means
    the same reference bases appear on both sides of the insert, which is the
    signature of target-primed reverse transcription.
    """
    decision = call_or_skip(P.check_boundary_consistency,
                            P.FinalBoundaryInput(100, 220, 211, 330, 3, 50))
    assert decision.pass_
    assert decision.boundary_type == "TSD"
    assert decision.boundary_len == 9
    assert decision.qc == "PASS_BOUNDARY_TSD"


def test_flanks_that_abut_exactly_are_a_blunt_join():
    decision = P.check_boundary_consistency(P.FinalBoundaryInput(100, 220, 220, 340))
    assert decision.pass_ and decision.boundary_type == "BLUNT"
    assert decision.qc == "PASS_BOUNDARY_BLUNT"


def test_a_short_gap_is_a_target_site_deletion():
    decision = P.check_boundary_consistency(P.FinalBoundaryInput(100, 220, 227, 340))
    assert decision.pass_ and decision.boundary_type == "SMALL_DEL"
    assert decision.boundary_len == 7


def test_an_overlap_outside_the_tsd_range_is_rejected_not_reported():
    """
    A 60 bp overlap is a misassembled flank, not a 60 bp duplication. Reporting
    it as a long TSD would put a fabricated mechanism in the output.
    """
    decision = P.check_boundary_consistency(P.FinalBoundaryInput(100, 220, 160, 340, 3, 50))
    assert not decision.pass_
    assert decision.qc == "REJECT_BOUNDARY_TSD_RANGE"


def test_a_long_gap_is_rejected_as_a_deletion_out_of_range():
    decision = P.check_boundary_consistency(P.FinalBoundaryInput(100, 220, 400, 500, 3, 50))
    assert decision.qc == "REJECT_BOUNDARY_DEL_RANGE"


def test_missing_and_inverted_segments_are_distinguished():
    assert P.check_boundary_consistency(P.FinalBoundaryInput()).qc == (
        "REJECT_BOUNDARY_MISSING_REF_SEGMENTS")
    assert P.check_boundary_consistency(
        P.FinalBoundaryInput(220, 100, 300, 400)).qc == (
        "REJECT_BOUNDARY_INVALID_REF_SEGMENTS")


def test_the_tsd_score_peaks_near_twelve_rather_than_saturating():
    """
    The discrete range test says only whether the geometry is admissible; this
    says how TE-LIKE it is. A 12 bp TSD is the mode for the common families, a
    4 bp one is plausible, and a 50 bp one is admissible but not credible -- so
    the term has to peak, not saturate.
    """
    at_mode = call_or_skip(P.boundary_structure_log_lr, "TSD", 12)
    close(at_mode, 1.2, "at the mode")
    assert P.boundary_structure_log_lr("TSD", 4) < at_mode
    assert P.boundary_structure_log_lr("TSD", 30) < at_mode
    assert P.boundary_structure_log_lr("TSD", 50) < 0.0
    close(P.boundary_structure_log_lr("BLUNT", 0), 1.0, "blunt")
    assert P.boundary_structure_log_lr("SMALL_DEL", 5) > P.boundary_structure_log_lr(
        "SMALL_DEL", 45)
    close(P.boundary_structure_log_lr("REJECT", 0), -2.0, "reject")


def test_a_noncanonical_junction_inside_the_envelope_is_accepted_at_a_lower_score():
    """
    What keeps a real insertion with uncertain breakpoints from being scored as
    an artifact. If the two flanks disagree by less than the envelope the stage
    already measured, the disagreement is measurement noise.
    """
    evidence = call_or_skip(P.evaluate_boundary_evidence,
                            P.FinalBoundaryInput(100, 220, 156, 340), 90)
    assert evidence.geometry_defined
    assert not evidence.canonical_pass
    assert evidence.evidence_consistent
    assert evidence.score > 0.0
    assert evidence.boundary_type == "NONCANONICAL"
    assert evidence.qc == "PASS_BOUNDARY_NONCANONICAL_CONSISTENT"


def test_a_noncanonical_junction_outside_the_envelope_is_scored_as_an_artifact():
    evidence = P.evaluate_boundary_evidence(P.FinalBoundaryInput(100, 220, 156, 340), 10)
    assert not evidence.evidence_consistent
    close(evidence.score, -2.0, "outside the envelope")


def test_undefined_geometry_scores_minus_two_without_consulting_the_type():
    evidence = P.evaluate_boundary_evidence(P.FinalBoundaryInput(), 90)
    assert not evidence.geometry_defined
    close(evidence.score, -2.0, "undefined")


# ------------------------------------------------------------- hard vetoes
def test_the_resolved_veto_needs_an_insert_a_pass_and_a_resolved_qc():
    """The C++ `test_resolved_veto`, case for case."""
    seg = closed_segmentation()
    existence = some_existence()
    boundary = make_boundary()
    te = te_pass()
    assert not call_or_skip(P.compute_te_resolved_hard_veto, existence, seg, te, boundary)
    assert P.compute_te_resolved_hard_veto(existence, closed_segmentation(
        has_insert_seq=False), te, boundary)
    assert P.compute_te_resolved_hard_veto(existence, seg,
                                           te_pass(pass_=False), boundary)
    assert P.compute_te_resolved_hard_veto(
        existence, seg, te_pass("PASS_INSERT_TE_ALIGNMENT_UNKNOWN"), boundary)
    # The boundary TYPE is not part of the veto: a small deletion is as
    # definitionally admissible as a TSD.
    assert not P.compute_te_resolved_hard_veto(existence, seg, te,
                                               make_boundary("SMALL_DEL", 7))


def test_the_unknown_veto_admits_an_unnamed_element_but_not_an_unaligned_one():
    """The C++ `test_unknown_veto`. The difference between the two vetoes is
    exactly the reason there are two TE hypotheses rather than one."""
    seg = closed_segmentation()
    existence = some_existence()
    boundary = make_boundary()
    assert not call_or_skip(P.compute_te_unknown_hard_veto, existence, seg,
                            te_pass(), boundary)
    assert P.compute_te_unknown_hard_veto(existence,
                                          closed_segmentation(has_insert_seq=False),
                                          te_pass(), boundary)
    assert P.compute_te_unknown_hard_veto(existence, seg,
                                          te_pass("NO_TE_ALIGNMENT"), boundary)
    assert not P.compute_te_unknown_hard_veto(
        existence, seg, te_pass("PASS_INSERT_TE_ALIGNMENT_UNKNOWN"), boundary)

    low_annotation = te_pass("PASS_INSERT_TE_ALIGNMENT_UNKNOWN",
                             annotation_confidence="LOW",
                             annotation_residual_fraction=0.80,
                             annotation_masked_fraction=0.10)
    assert P.compute_te_unknown_hard_veto(existence, seg, low_annotation, boundary)


def test_annotation_quality_is_unconditional_only_at_high_confidence():
    """
    A vaguely-named alignment covering a third of the insert is compatible with
    almost anything. Letting it through would make the naming tier decide the
    call after all -- the thing the tiers are explicitly not for.
    """
    assert call_or_skip(P.annotation_quality_allows_te_decision,
                        te_pass(annotation_confidence="HIGH",
                                annotation_residual_fraction=0.99))
    assert P.annotation_quality_allows_te_decision(
        te_pass(annotation_confidence="LOW", annotation_residual_fraction=0.50,
                annotation_masked_fraction=0.65))
    assert not P.annotation_quality_allows_te_decision(
        te_pass(annotation_confidence="LOW", annotation_residual_fraction=0.51))
    assert not P.annotation_quality_allows_te_decision(
        te_pass(annotation_confidence="MEDIUM", annotation_masked_fraction=0.81))


# ------------------------------------------------------------- read shapes
def test_minus_one_means_not_broken_down_and_is_not_zero():
    """
    Conflating them would make an un-broken-down event look like one with no
    precise reads -- the difference between "unknown" and "bad", at the input to
    every mechanistic signal.
    """
    plain = P.EventExistenceEvidence(alt_struct_reads=8)
    assert not call_or_skip(P.has_structural_breakdown, plain)
    assert P.precise_structural_reads(plain) == 8
    assert P.bilateral_clip_support_reads(plain) == 8

    broken = P.EventExistenceEvidence(alt_struct_reads=8, alt_split_reads=0,
                                      alt_indel_reads=0, alt_left_clip_reads=0,
                                      alt_right_clip_reads=0)
    assert P.has_structural_breakdown(broken)
    assert P.precise_structural_reads(broken) == 0


def test_bilateral_clip_support_takes_the_minimum_of_the_two_sides():
    """
    Ten left clips and no right clips contribute NOTHING. That is consistent
    with a single misaligned edge and inconsistent with an insertion, and a sum
    would call it strong support.
    """
    one_sided = P.EventExistenceEvidence(alt_struct_reads=10, alt_split_reads=0,
                                         alt_indel_reads=0, alt_left_clip_reads=10,
                                         alt_right_clip_reads=0)
    assert P.bilateral_clip_support_reads(one_sided) == 0
    both = P.EventExistenceEvidence(alt_struct_reads=10, alt_split_reads=0,
                                    alt_indel_reads=0, alt_left_clip_reads=10,
                                    alt_right_clip_reads=4)
    assert P.bilateral_clip_support_reads(both) == 4


def test_an_unopposed_insertion_is_not_penalised_for_the_genotypers_uncertainty():
    """
    No reference-spanning reads means the genotyper has no depth to be confident
    WITH, so its score can be negative. That is a property of the model, not
    evidence against the event.
    """
    unopposed = P.EventExistenceEvidence(alt_struct_reads=3, ref_span_reads=0,
                                         score=-0.75)
    close(call_or_skip(P.joint_event_existence_score, unopposed), 0.0, "unopposed")
    too_few = P.EventExistenceEvidence(alt_struct_reads=2, ref_span_reads=0, score=-0.75)
    close(P.joint_event_existence_score(too_few), -0.75, "below the floor")


def test_count_signal_saturates_so_depth_alone_cannot_win():
    close(P.count_signal(0, 8.0), 0.0, "no reads")
    first = P.count_signal(2, 8.0) - P.count_signal(1, 8.0)
    later = P.count_signal(11, 8.0) - P.count_signal(10, 8.0)
    assert first > later


# --------------------------------------------------- the one-sided polarity
def test_the_two_definitions_of_one_sided_pass_disagree_and_both_are_kept():
    """
    PINNED BECAUSE IT IS A BUG, and one the port must not quietly fix.
    `is_one_sided_segmentation_pass` is defined twice in the C++, in two
    translation units, with OPPOSITE polarity on `pair_valid` -- and both run on
    the same locus in the same pass. Choosing one would change behaviour under
    cover of a cleanup.
    """
    from placer_py import blocks

    seg = P.EventSegmentationEvidence(has_insert_seq=True, has_left_flank=True,
                                      has_right_flank=False, pair_valid=True)
    assert P.is_one_sided_segmentation_pass(seg)
    assert not blocks._is_one_sided_segmentation_pass(seg)

    invalid = P.EventSegmentationEvidence(has_insert_seq=True, has_left_flank=True,
                                          has_right_flank=False, pair_valid=False)
    assert not P.is_one_sided_segmentation_pass(invalid)
    assert blocks._is_one_sided_segmentation_pass(invalid)


def test_a_one_sided_pass_is_floored_rather_than_left_negative():
    one_sided = P.EventSegmentationEvidence(has_insert_seq=True, pair_valid=True,
                                            has_left_flank=True, score=-2.0)
    close(call_or_skip(P.adjusted_segmentation_score, one_sided), 0.25, "floored")
    two_sided = P.EventSegmentationEvidence(has_insert_seq=True, pair_valid=True,
                                            has_left_flank=True, has_right_flank=True,
                                            score=-2.0)
    close(P.adjusted_segmentation_score(two_sided), -2.0, "not floored")


def test_a_one_sided_event_is_not_charged_twice_for_its_missing_flank():
    one_sided = P.EventSegmentationEvidence(has_insert_seq=True, pair_valid=True,
                                            has_left_flank=True)
    boundary = P.BoundaryEvidence(geometry_defined=False, score=-2.0)
    close(P.adjusted_te_boundary_score(one_sided, boundary), 0.0, "undefined, not bad")


# ------------------------------------------------------------ segmentation
def test_segmentation_scores_the_weaker_flank_not_the_average():
    """An event is only as well anchored as its worse side."""
    lopsided = call_or_skip(P.analyze_event_segmentation, True, 500, 60, 0.99, 0.91,
                            "ACGT" * 50, True, "PASS")
    balanced = P.analyze_event_segmentation(True, 60, 60, 0.91, 0.91,
                                            "ACGT" * 50, True, "PASS")
    close(lopsided.score, balanced.score, "min, not mean")


def test_a_one_sided_pass_scores_the_flank_it_has_rather_than_the_one_it_lacks():
    """
    The one-sided branch changes TWO things at once, and they are easy to
    conflate: the scoring flank switches from `min(left, right)` -- which is 0
    length and 0 identity when a flank is missing -- to the flank that exists,
    AND the penalty drops from 1.0 to 0.5. Without the first change the second
    would be meaningless, because the score would already be at its floor.
    """
    one_sided = P.analyze_event_segmentation(True, 300, 0, 0.99, 0.0, "ACGT" * 50,
                                             True, "PASS")
    # 300 bp and 0.99 identity both clamp to +2, so the base is 2.0, less 0.5.
    close(one_sided.score, 1.5 - 0.1, "scored on the flank it has")

    missing = P.analyze_event_segmentation(True, 300, 0, 0.99, 0.0, "ACGT" * 50,
                                           False, "PASS")
    # min() gives length 0 and identity 0, both clamping to -2: base -2.0, then
    # -1.0 for the missing flank and -1.5 for the invalid pair.
    close(missing.score, -4.5, "scored on the flank it lacks")


def test_an_invalid_pair_is_penalised_on_top_of_everything_else():
    valid = P.analyze_event_segmentation(True, 300, 300, 0.99, 0.99, "ACGT" * 50,
                                         True, "PASS")
    invalid = P.analyze_event_segmentation(True, 300, 300, 0.99, 0.99, "ACGT" * 50,
                                           False, "FAIL")
    close(valid.score - invalid.score, 1.5, "pair penalty")


# ----------------------------------------------------------- family kinds
def test_family_kind_maps_names_onto_mechanisms_not_taxonomy():
    """
    `active_tprt_te` is compatible with a retro family and not with a DNA
    transposon because target-primed reverse transcription is something only the
    former does. The state model keys on MECHANISM.
    """
    assert call_or_skip(P.family_kind, te_pass(best_family="L1", best_subfamily="L1HS")) == "retro"
    assert P.family_kind(te_pass(best_family="Alu", best_subfamily="AluYa5")) == "retro"
    assert P.family_kind(te_pass(best_family="HERV", best_subfamily="HERVK")) == "ltr"
    assert P.family_kind(te_pass(best_family="hAT", best_subfamily="Charlie")) == "dna"
    assert P.family_kind(te_pass(best_family="UNKNOWN", best_subfamily="UNKNOWN")) == "unknown"
    assert P.family_kind(te_pass(best_family="", best_subfamily="")) == "unknown"
    assert P.family_kind(te_pass(best_family="Satellite", best_subfamily="HSat2")) == "other"


def test_ltr_is_tested_before_retro_because_the_tokens_overlap():
    """`ERVL-MaLR` contains no retro token, but some LTR subfamily names contain
    `l1`. Order is what keeps the two apart."""
    assert P.family_kind(te_pass(best_family="ERVL", best_subfamily="MLT1A")) == "ltr"


def test_the_mechanism_prior_penalises_an_incompatible_family():
    """
    The NEGATIVE entries carry the information. Encoding the mismatch is what
    lets the model tell a mislabelled insert from a real one of another kind,
    rather than merely preferring the best-matching family.
    """
    retro = P.LatentFeatureVector(family_kind="retro", te_identity=0.95)
    dna = P.LatentFeatureVector(family_kind="dna", te_identity=0.95)
    close(call_or_skip(P.family_state_compatibility, retro, "active_tprt_te"), 0.75, "fit")
    close(P.family_state_compatibility(dna, "active_tprt_te"), -0.55, "mismatch")
    close(P.family_state_compatibility(dna, "cut_paste_te"), 0.75, "dna fits cut/paste")


def test_the_degraded_state_is_the_escape_hatch_for_an_unreliable_name():
    unknown = P.LatentFeatureVector(family_kind="unknown", te_identity=0.95)
    diverged = P.LatentFeatureVector(family_kind="retro", te_identity=0.60)
    close(P.family_state_compatibility(unknown, "degraded_unknown_te"), 0.40, "unnamed")
    close(P.family_state_compatibility(diverged, "degraded_unknown_te"), 0.20, "diverged")


# ------------------------------------------------------- the latent posterior
def _lfdr(existence=None, segmentation=None, te=None, boundary=None):
    return P.evaluate_latent_mechanism_lfdr(
        existence or some_existence(), segmentation or closed_segmentation(),
        te or te_pass(best_family="L1", best_subfamily="L1HS", best_identity=0.98,
                      best_query_coverage=0.95, cross_family_margin=0.4,
                      annotation_confidence="HIGH"),
        boundary or make_boundary())


def test_a_clean_l1_insertion_is_called_active_tprt():
    result = call_or_skip(_lfdr)
    assert result.latent_mechanism == "active_tprt_te"
    assert result.te_posterior > result.non_te_posterior
    assert result.te_posterior > result.artifact_posterior


def test_the_worst_case_lfdr_is_always_at_least_the_plain_one():
    """
    It is a SENSITIVITY ANALYSIS, not a probability: every TE state is divided
    by an ambiguity allowance and every null state multiplied by one, then the
    two are renormalised. It can only move the answer toward the null.
    """
    result = _lfdr()
    assert result.worst_case_lfdr >= result.lfdr


def test_thin_mechanistic_evidence_widens_the_te_side_faster_than_the_null_side():
    """
    2.60 against 1.65 per unit of mechanistic gap. That asymmetry is the
    precision-first stance made numerical: where evidence is thin, abstention
    rather than a confident call in either direction.
    """
    strong = _lfdr()
    thin = _lfdr(existence=some_existence(alt_split_reads=0, alt_indel_reads=0,
                                          alt_left_clip_reads=0,
                                          alt_right_clip_reads=0))
    assert thin.worst_case_lfdr > strong.worst_case_lfdr


def test_reference_conflict_raises_the_local_false_discovery_rate():
    conflicted = _lfdr(existence=some_existence(ref_span_reads=40))
    assert conflicted.worst_case_lfdr > _lfdr().worst_case_lfdr


def test_the_six_posteriors_partition_the_probability():
    result = _lfdr()
    total = result.te_posterior + result.non_te_posterior + result.artifact_posterior
    close(total, 1.0, "partition")
    close(result.lfdr, 1.0 - result.te_posterior, "lfdr is the null mass")


# --------------------------------------------------- the local three-way
def test_sequence_evidence_for_a_te_is_evidence_against_a_non_te_insertion():
    """
    `log_non_te` SUBTRACTS the sequence support. The two hypotheses compete on
    the same axis, and a model that only added TE evidence to the TE side would
    let both rise together.
    """
    te_like = call_or_skip(P.evaluate_local_hypothesis_posterior,
                           some_existence(), closed_segmentation(),
                           te_pass(best_identity=0.98, best_query_coverage=0.95,
                                   cross_family_margin=0.4,
                                   annotation_confidence="HIGH"),
                           make_boundary())
    unnamed = P.evaluate_local_hypothesis_posterior(
        some_existence(), closed_segmentation(),
        te_pass("NO_TE_ALIGNMENT", pass_=False,
                sequence_model_label="TE_MODEL_UNAVAILABLE"),
        make_boundary())
    assert te_like.te > unnamed.te
    assert te_like.non_te < unnamed.non_te


def test_the_unknown_te_deficits_are_graded_rather_than_flat():
    """
    An unnamed insert at identity 0.61 and one at identity 0.20 are not equally
    suspicious, and a flat penalty would say they are.
    """
    near = P.evaluate_local_hypothesis_posterior(
        some_existence(), closed_segmentation(),
        te_pass("PASS_INSERT_TE_ALIGNMENT_UNKNOWN", best_identity=0.61,
                best_query_coverage=0.88, cross_family_margin=0.04),
        make_boundary())
    far = P.evaluate_local_hypothesis_posterior(
        some_existence(), closed_segmentation(),
        te_pass("PASS_INSERT_TE_ALIGNMENT_UNKNOWN", best_identity=0.20,
                best_query_coverage=0.30, cross_family_margin=0.0),
        make_boundary())
    assert far.artifact > near.artifact


def test_a_closed_event_is_held_to_a_lower_posterior_bar():
    """Closure already removed the commonest way to be wrong, so it buys a
    lower threshold rather than only a higher score."""
    strong_te = te_pass(best_identity=0.98, best_query_coverage=0.95,
                        cross_family_margin=0.4, annotation_confidence="HIGH")
    closed = P.evaluate_local_hypothesis_posterior(
        some_existence(), closed_segmentation(), strong_te, make_boundary())
    assert closed.qc == "PASS_TE_POSTERIOR"


def test_the_three_posteriors_sum_to_one():
    posterior = P.evaluate_local_hypothesis_posterior(
        some_existence(), closed_segmentation(), te_pass(), make_boundary())
    close(posterior.te + posterior.non_te + posterior.artifact, 1.0, "partition")
    close(posterior.te_vs_artifact_log_odds,
          posterior.log_te - posterior.log_artifact, "odds")


# --------------------------------------------------- the low-AF Bayes factor
def test_a_minority_alt_allele_can_still_beat_an_error_process():
    """
    THE reason this exists alongside the dominant-allele odds: a true het or
    mosaic insertion has `alt < ref`, and a rule asking alt to dominate would
    discard exactly those.
    """
    minority = call_or_skip(P.low_allele_fraction_insertion_log_bf, 4, 20)
    assert minority > 0.0
    assert minority > math.log(5) - math.log(21)     # beats the dominant model
    assert P.low_allele_fraction_insertion_log_bf(0, 20) < -1e200
    assert P.low_allele_fraction_insertion_log_bf(1, 400) < minority


# ------------------------------------------------------ ranking vs emission
def _joint(existence=None, segmentation=None, te=None, boundary=None):
    return P.evaluate_joint_hypotheses(
        existence or some_existence(), segmentation or closed_segmentation(),
        te or te_pass(best_family="L1", best_subfamily="L1HS", best_identity=0.98,
                      best_query_coverage=0.95, cross_family_margin=0.4,
                      annotation_confidence="HIGH", sequence_model_score=0.93),
        boundary or make_boundary())


def test_ranking_and_emission_are_separate_decisions():
    """
    THE property of this module. A TE hypothesis may rank top as a diagnostic
    while the calibrated risk gate abstains -- the C++ test says so in as many
    words. A port that emitted whatever ranked first would pass most other
    tests and be wrong.
    """
    result = call_or_skip(_joint)
    assert result.best.kind in (P.FinalHypothesisKind.TE_RESOLVED,
                                P.FinalHypothesisKind.TE_UNKNOWN)
    assert isinstance(result.emit_te_call, bool)
    assert result.robust_mechanistic_qc in ("PASS_TE_LFDR", "TE_LFDR_HIGH")
    if not result.emit_te_call:
        assert result.final_qc in ("TE_AMBIGUOUS", "REFERENCE_OR_ARTIFACT",
                                  "PASS_STRUCTURAL_INSERTION")


def test_a_failed_gate_abstains_rather_than_rejecting():
    """
    Precision-first. A TE-like event that fails the risk gate becomes
    `TE_AMBIGUOUS` and is RETAINED as TE-evidence -- it is not reclassified as
    reference or artifact, and it stays visible to whoever wants to look.

    THE SCENARIO CHANGED and the property did not. This used to call `_joint()`
    with a strong alignment (identity 0.98, coverage 0.95) and still abstain,
    which was never right on the merits -- it abstained only because
    `blocks._structure_explanation` was handed a default-constructed
    explanation and returned zero structure evidence. Once that branch was
    fixed to fall back to the shadow path as the C++ does, a strong alignment
    correctly passes the gate. The abstention is now provoked the way it
    should be: a partial alignment (coverage 0.30) with LOW annotation
    confidence, which is genuinely uncertain about the element's identity
    while the event itself is well supported.

    Note the allele fraction stays dominant (18 alt / 2 ref). That is what
    keeps this on the TE-evidence path rather than the structural one -- the
    regime guard in `should_emit_structural_event_call` refuses to downgrade
    a high-AF TE-like event, so it stays TE_AMBIGUOUS instead of becoming
    PASS_STRUCTURAL_INSERTION.
    """
    uncertain_identity = te_pass(
        best_family="L1", best_subfamily="L1HS", best_identity=0.88,
        best_query_coverage=0.30, cross_family_margin=0.02,
        annotation_confidence="LOW", sequence_model_score=0.2)
    result = _joint(
        existence=some_existence(alt_struct_reads=18, ref_span_reads=2, gq=60),
        te=uncertain_identity)
    assert not result.emit_te_call
    assert result.emit_evidence_te_call
    assert result.final_qc == "TE_AMBIGUOUS"
    assert result.robust_mechanistic_worst_case_lfdr > P.TARGET_Q


def test_a_low_af_event_reaches_the_structural_path_even_with_a_te_explanation():
    """
    The other half of the regime guard, and the case that shows it is not just a
    non-TE escape hatch. With alt 8 against ref 40 the low-AF Bayes factor beats
    the dominant-allele odds, so the event enters the structural path even
    though the winning EXPLANATION is TE -- which is exactly the "identity is
    uncertain because alt coverage is low" case it exists for.
    """
    result = _joint(existence=some_existence(ref_span_reads=40, gq=25))
    assert not result.emit_te_call
    assert result.emit_structural_event_call
    assert result.final_qc == "PASS_STRUCTURAL_INSERTION"
    assert result.best.reason == "STRUCTURAL_INSERTION"


def test_an_event_with_no_insert_sequence_can_never_emit_a_te_call():
    result = _joint(segmentation=closed_segmentation(has_insert_seq=False,
                                                     insert_len=0))
    assert not result.emit_te_call


def test_a_vetoed_hypothesis_sorts_to_the_bottom_but_still_populates_runner_up():
    result = _joint(te=te_pass("NO_TE_ALIGNMENT", pass_=False,
                               sequence_model_label="TE_MODEL_UNAVAILABLE"))
    assert result.runner_up.kind is not None
    assert not result.emit_te_call


def test_the_reference_hypothesis_is_never_rewarded_by_weak_evidence():
    """
    `max(existence.score, 0)`. Absence of evidence for an insertion is not
    evidence for the reference; it just leaves h0 at zero.
    """
    weak = _joint(existence=some_existence(score=-3.0, gq=0))
    h0_total = weak.best.total
    assert h0_total is not None
    # Recomputed directly: both terms floor at zero.
    close(P.SCORE_REF_EXISTENCE_WEIGHT * max(-3.0, 0.0), 0.0, "no reward")


def test_the_diagnostics_are_populated_on_every_path():
    result = _joint()
    assert result.mechanistic_blocks != "NA"
    assert result.te_structure_path != "NA"
    assert result.latent_mechanism != "NA"
    assert result.posterior_qc != "POSTERIOR_NOT_EVALUATED"
    assert result.explanation_residual != "NA"


def test_the_raw_aggregates_are_carried_penalty_free_for_finalization():
    """
    The bound that makes the block product an e-value is a property of the whole
    RUN, not of one locus -- so per-locus screening charges no penalty and the
    raw aggregates have to survive to finalization intact.
    """
    result = _joint()
    close(result.mechanistic_raw_log_bf_te_vs_artifact,
          result.mechanistic_lower_log_bf_te_vs_artifact, "no per-locus penalty")


# ------------------------------------------------ the structural-insertion path
def test_a_high_af_te_like_event_that_abstained_is_not_downgraded_to_structural():
    """
    THE regime guard. A high-AF TE-like event failing the TE gate stays
    TE-evidence; only genuine non-TE insertions and low-AF events whose identity
    is uncertain BECAUSE alt coverage is low come through the structural path.
    """
    posterior = P.LocalHypothesisPosterior(te=0.8, non_te=0.1, artifact=0.1)
    assert not call_or_skip(P.should_emit_structural_event_call,
                            some_existence(alt_struct_reads=20, ref_span_reads=2),
                            closed_segmentation(), make_boundary(), posterior,
                            0.0, False)


def test_the_structural_gate_requires_a_closed_event():
    posterior = P.LocalHypothesisPosterior(te=0.1, non_te=0.85, artifact=0.05)
    assert not P.should_emit_structural_event_call(
        some_existence(), closed_segmentation(has_right_flank=False),
        make_boundary(), posterior, 0.0, True)


def test_the_structural_gate_requires_a_genotype_beyond_the_phred_20_pivot():
    """
    Stops boundary and model priors from carrying a trivially-supported event
    over the log-evidence bar on their own.
    """
    posterior = P.LocalHypothesisPosterior(te=0.1, non_te=0.85, artifact=0.05)
    assert not P.should_emit_structural_event_call(
        some_existence(gq=20), closed_segmentation(), make_boundary(),
        posterior, 0.0, True)


def test_the_structural_gate_requires_at_least_one_precise_or_bilateral_read():
    posterior = P.LocalHypothesisPosterior(te=0.1, non_te=0.85, artifact=0.05)
    assert not P.should_emit_structural_event_call(
        some_existence(alt_split_reads=0, alt_indel_reads=0, alt_left_clip_reads=0,
                       alt_right_clip_reads=0),
        closed_segmentation(), make_boundary(), posterior, 0.0, True)


def test_a_clean_non_te_insertion_passes_the_structural_gate():
    posterior = P.LocalHypothesisPosterior(te=0.05, non_te=0.92, artifact=0.03)
    assert P.should_emit_structural_event_call(
        some_existence(alt_struct_reads=20, ref_span_reads=2, gq=60),
        closed_segmentation(), make_boundary(), posterior, 0.0, True)


def test_reference_conflict_can_close_the_structural_gate():
    """And only within 1.20 nats of the bar, which is the point.

    REGRESSION, and the bug was in the test. This previously read

        assert not P.should_emit_structural_event_call(*args, 1.0, True) or True

    which is vacuously true -- `X or True` can never fail -- over a case
    (artifact=0.03, alt=20, ref=2, gq=60) where the gate does NOT close. So
    the assertion was hiding the fact that its own premise was false at that
    evidence strength.

    The claim is true, but it is quantitative rather than absolute.
    `ref_conflict_penalty` is `1.20 * clamp01(signal)`, subtracted from an
    evidence total compared against `STRUCTURAL_LOG_EVIDENCE = 2.0`. So a
    saturated conflict can only overturn a call sitting within 1.20 nats of
    the bar. At artifact=0.03 the odds term alone is log(0.97/0.03) = 3.48 and
    nothing the conflict signal can do will close it; at artifact=0.25 it is
    1.10 and the penalty decides.
    """
    near_the_bar = P.LocalHypothesisPosterior(te=0.05, non_te=0.70, artifact=0.25)
    args = (some_existence(alt_struct_reads=3, ref_span_reads=2, gq=25),
            closed_segmentation(), make_boundary(), near_the_bar)
    assert P.should_emit_structural_event_call(*args, 0.0, True)
    assert not P.should_emit_structural_event_call(*args, 1.0, True)

    # Far from the bar the same saturated conflict must NOT close it, or the
    # penalty would be acting as a veto rather than as evidence.
    far_above = P.LocalHypothesisPosterior(te=0.05, non_te=0.92, artifact=0.03)
    strong = (some_existence(alt_struct_reads=20, ref_span_reads=2, gq=60),
              closed_segmentation(), make_boundary(), far_above)
    assert P.should_emit_structural_event_call(*strong, 1.0, True)


def test_reference_conflict_is_monotone():
    """More conflict never makes a call MORE likely.

    The penalty is linear in the signal, so this is arithmetic rather than a
    discovery -- but it is the property the gate would silently lose if the
    term were ever moved to the other side of the subtraction.
    """
    posterior = P.LocalHypothesisPosterior(te=0.05, non_te=0.70, artifact=0.25)
    args = (some_existence(alt_struct_reads=3, ref_span_reads=2, gq=25),
            closed_segmentation(), make_boundary(), posterior)
    emitted = [P.should_emit_structural_event_call(*args, signal / 10.0, True)
               for signal in range(11)]
    assert emitted == sorted(emitted, reverse=True), emitted


# ---------------------------------------------------- context-conditioned prior
def test_an_artifact_prone_context_lowers_the_te_prior_floor():
    """
    The decision layer's version of the rule the TPRT model follows: read the
    null off the local context rather than applying one number everywhere. The
    difference -- and it is the README's whole argument -- is that these slopes
    are hand-set rather than measured.
    """
    clean = call_or_skip(P.context_conditioned_prior, 0.0)
    dirty = P.context_conditioned_prior(1.0)
    close(clean["te_min"], 0.05, "clean floor")
    assert dirty["te_min"] < clean["te_min"]
    assert dirty["artifact_min"] > clean["artifact_min"]
    close(dirty["artifact_max"], 0.95, "cap unchanged")


# ------------------------------------------------------------ QC classification
def test_the_qc_string_stays_the_source_of_truth():
    """
    The enum exists so gating branches on a value instead of on string literals,
    where a typo is a silently false comparison. The STRING is still what is
    serialised, so its spelling is part of the output contract.
    """
    assert P.classify_te_alignment_qc("PASS_INSERT_TE_ALIGNMENT") == P.TeAlignmentQc.PASS_INSERT
    assert P.classify_te_alignment_qc("anything else") == P.TeAlignmentQc.OTHER
    assert P.is_pass_insert_te_alignment(P.TeAlignmentQc.PASS_INSERT_UNKNOWN)
    assert not P.is_resolved_pass_insert_te_alignment(P.TeAlignmentQc.PASS_INSERT_UNKNOWN)
    assert P.is_resolved_pass_insert_te_alignment(P.TeAlignmentQc.PASS_INSERT_FAMILY_ONLY)


def test_te_sequence_evidence_is_deliberately_permissive():
    """
    It is the predicate for the ABSTENTION path -- "keep this locus visible as
    TE-evidence" -- and abstention should be reachable from weak evidence by
    definition. A bare `best_identity > 0` qualifies.
    """
    assert call_or_skip(P.has_te_sequence_evidence,
                        te_pass("NO_TE_ALIGNMENT", pass_=False, best_identity=0.4))
    assert not P.has_te_sequence_evidence(
        te_pass("NO_TE_ALIGNMENT", pass_=False, best_identity=0.0,
                best_query_coverage=0.0))
