"""
Evidence-certificate behaviour of `placer_py.core.blocks`.
"""

from __future__ import annotations

import math

import pytest
from conftest import call_or_skip, close

from placer_py.core import blocks
from placer_py.core import policy as P
from placer_py.core.te_classifier import TEAlignmentEvidence


def _ex(alt, split, indel, lclip, rclip, ref, af, gq):
    return P.EventExistenceEvidence(
        alt_struct_reads=alt, alt_split_reads=split, alt_indel_reads=indel,
        alt_left_clip_reads=lclip, alt_right_clip_reads=rclip,
        ref_span_reads=ref, af=af, gq=gq)


def _seg(pair_valid, left, right, insert_len):
    return P.EventSegmentationEvidence(
        has_insert_seq=True, has_left_flank=left, has_right_flank=right,
        pair_valid=pair_valid, insert_len=insert_len)


def _te(identity, coverage, margin, qc, model, model_score, confidence,
        residual):
    return TEAlignmentEvidence(
        pass_=True, best_family="L1", best_subfamily="L1HS",
        best_identity=identity, best_query_coverage=coverage,
        cross_family_margin=margin, qc_reason=qc,
        sequence_model_label=model, sequence_model_score=model_score,
        annotation_confidence=confidence,
        annotation_residual_fraction=residual,
        annotation_masked_fraction=0.0, second_score=0.0)


def _bd(geometry, canonical, consistent, btype, blen):
    return P.BoundaryEvidence(
        geometry_defined=geometry, canonical_pass=canonical,
        evidence_consistent=consistent, boundary_type=btype,
        boundary_len=blen)


#: Hand-written evidence bundles spanning a strong resolved TE, a reference
#: conflict, a sequence-model outlier, family-only and unknown-TE annotation, a
#: one-sided rescue, low identity, and an artifact-shaped locus.
SCENARIOS = {
    "strong_resolved_te": (
        _ex(18, 8, 3, 5, 5, 0, 0.65, 60), _seg(True, True, True, 320),
        _te(0.96, 0.90, 0.18, "PASS_INSERT_TE_ALIGNMENT",
            "TE_MODEL_IN_DISTRIBUTION", 0.35, "HIGH", 0.0),
        _bd(True, True, True, "TSD", 12)),
    "strong_with_reference_conflict": (
        _ex(18, 8, 3, 5, 5, 18, 0.65, 60), _seg(True, True, True, 320),
        _te(0.96, 0.90, 0.18, "PASS_INSERT_TE_ALIGNMENT",
            "TE_MODEL_IN_DISTRIBUTION", 0.35, "HIGH", 0.0),
        _bd(True, True, True, "TSD", 12)),
    "sequence_model_outlier": (
        _ex(18, 8, 3, 5, 5, 0, 0.65, 60), _seg(True, True, True, 320),
        _te(0.96, 0.90, 0.18, "PASS_INSERT_TE_ALIGNMENT",
            "TE_MODEL_OUTLIER", -0.80, "HIGH", 0.0),
        _bd(True, True, True, "TSD", 12)),
    "family_only": (
        _ex(12, 5, 2, 3, 3, 1, 0.45, 42), _seg(True, True, True, 280),
        _te(0.88, 0.72, 0.03, "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY",
            "TE_MODEL_EDGE", 0.05, "MEDIUM", 0.10),
        _bd(True, False, True, "BLUNT", 0)),
    "one_sided_rescue": (
        _ex(6, 2, 1, 2, 0, 2, 0.30, 28), _seg(False, True, False, 410),
        _te(0.93, 0.80, 0.09, "PASS_INSERT_TE_ALIGNMENT",
            "TE_MODEL_IN_DISTRIBUTION", 0.20, "HIGH", 0.05),
        _bd(True, False, True, "SMALL_DEL", 3)),
    "low_identity": (
        _ex(9, 4, 1, 2, 2, 3, 0.35, 33), _seg(True, True, True, 260),
        _te(0.71, 0.55, 0.01, "TE_ALIGNMENT_LOW_IDENTITY",
            "TE_MODEL_EDGE", -0.10, "LOW", 0.30),
        _bd(True, False, False, "NONE", 0)),
    "unknown_te": (
        _ex(14, 6, 2, 4, 4, 1, 0.55, 51), _seg(True, True, True, 350),
        _te(0.90, 0.84, 0.00, "PASS_INSERT_TE_ALIGNMENT_UNKNOWN",
            "TE_MODEL_IN_DISTRIBUTION", 0.28, "MEDIUM", 0.18),
        _bd(True, True, True, "TSD", 15)),
    "artifact_shaped": (
        _ex(4, 1, 0, 1, 0, 11, 0.15, 12), _seg(False, False, False, 120),
        _te(0.78, 0.40, 0.00, "TE_ALIGNMENT_LOW_IDENTITY",
            "TE_MODEL_OUTLIER", -0.90, "LOW", 0.55),
        _bd(False, False, False, "NONE", 0)),
}


@pytest.mark.regression
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_per_locus_stage_charges_no_penalty(name):
    """`lower == raw` exactly. Earlier releases subtracted a fixed 0.65 nats
    here, and that invented constant is what the current design removed."""
    ex, seg, te, bd = SCENARIOS[name]
    cert = call_or_skip(blocks.build_certificate, ex, seg, te, bd, None)
    assert cert.lower_log_bf_te_vs_artifact == cert.raw_log_bf_te_vs_artifact
    assert cert.lower_log_bf_te_vs_non_te == cert.raw_log_bf_te_vs_non_te


# The aggregate weightings, transcribed from src/pipeline/mechanistic_evidence.cpp.
ART_WEIGHTS = {
    "event": 1.00, "independent": 1.00, "sequence": 1.00,
    "structure": 0.90, "boundary": 1.00, "ref_conflict": 1.00,
}
NON_WEIGHTS = {
    "event": 0.40, "independent": 0.70, "sequence": 1.00,
    "structure": 0.90, "boundary": 0.85, "ref_conflict": 0.35,
}
PRIOR_TE_MIN = 0.05
PRIOR_NULL_ODDS_UPPER = 0.95 + 0.30      # artifact_max + non_te_max


def _cert(name):
    ex, seg, te, bd = SCENARIOS[name]
    return call_or_skip(blocks.build_certificate, ex, seg, te, bd, None)


@pytest.mark.invariant
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_aggregates_are_the_weighted_block_sums(name):
    cert = _cert(name)
    for weights, got in ((ART_WEIGHTS, cert.raw_log_bf_te_vs_artifact),
                         (NON_WEIGHTS, cert.raw_log_bf_te_vs_non_te)):
        expected = sum(w * getattr(cert, f"{block}_lower_log_lr")
                       for block, w in weights.items())
        close(got, expected, f"{name} aggregate")


@pytest.mark.invariant
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_structure_block_is_te_minus_the_best_null(name):
    """`clamp(te - max(nonte, artifact), -4, 5)`."""
    cert = _cert(name)
    raw = cert.structure_te_log_evidence - max(
        cert.structure_nonte_log_evidence, cert.structure_artifact_log_evidence)
    close(cert.structure_lower_log_lr, min(5.0, max(-4.0, raw)), name)


@pytest.mark.invariant
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_certificate_shape_and_ranges(name):
    cert = _cert(name)
    assert [b.name for b in cert.blocks] == [
        "event", "independent", "sequence", "structure", "boundary",
        "ref_conflict"]
    widths = [b.ambiguity_width for b in cert.blocks]
    close(cert.ambiguity_width, sum(widths) / len(widths),
          f"{name}: ambiguity width is the block mean")
    for key in ("mechanistic_support_signal", "ref_conflict_signal",
                "artifact_context_signal"):
        assert 0.0 <= getattr(cert, key) <= 1.0, f"{name}.{key}"


@pytest.mark.invariant
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_robust_lfdr_matches_its_closed_form(name):
    """`null_upper / (null_upper + exp(clamp(min_bf - amb, -60, 60)) * te_min)`,
    and the QC label follows it at the 0.10 threshold."""
    cert = _cert(name)
    got = call_or_skip(blocks.evaluate_robust_lfdr, cert, None, 0.10)
    min_bf = min(cert.lower_log_bf_te_vs_artifact,
                 cert.lower_log_bf_te_vs_non_te)
    te_odds = math.exp(
        max(-60.0, min(60.0, min_bf - cert.ambiguity_width))) * PRIOR_TE_MIN
    expected = PRIOR_NULL_ODDS_UPPER / max(1e-12, PRIOR_NULL_ODDS_UPPER + te_odds)
    assert math.isclose(got["worst_case_lfdr"], expected,
                        rel_tol=1e-10, abs_tol=1e-12), name
    assert got["qc"] == ("PASS_TE_LFDR" if got["worst_case_lfdr"] <= 0.10
                         else "TE_LFDR_HIGH"), name


@pytest.mark.invariant
def test_reference_conflict_lowers_the_artifact_bayes_factor():
    """The same locus, opposed by 18 reference-spanning reads, must score
    lower."""
    clean = _cert("strong_resolved_te")
    conflicted = _cert("strong_with_reference_conflict")
    assert conflicted.raw_log_bf_te_vs_artifact < clean.raw_log_bf_te_vs_artifact
    assert conflicted.ref_conflict_signal > clean.ref_conflict_signal


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN inconsistency: the per-block vs-non-TE log-LRs use weights "
    "0.45/0.65/1/0.80/0.50 while the aggregate uses 0.40/0.70/1/0.85/0.35, and "
    "the structure block differs too. So the serialized blocks do not sum to "
    "the aggregate. Harmless for the decision (only the aggregate decides) but "
    "it defeats anyone trying to re-derive a call by hand from "
    "evidence_ledger.tsv."))
def test_serialized_blocks_sum_to_the_non_te_aggregate():
    for name in SCENARIOS:
        cert = _cert(name)
        total = sum(b.te_vs_non_te for b in cert.blocks)
        assert math.isclose(cert.raw_log_bf_te_vs_non_te, total,
                            rel_tol=1e-9, abs_tol=1e-9), name
