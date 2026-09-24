"""
Golden certificates. Skips until `placer_py.core.blocks` is ported.

These assert EQUALITY to full double precision against the C++, because the C++
test suite pins only signs and orderings -- a port could compute a different
function and pass every assertion over there.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer_py.core import blocks
from placer_py.core import policy as P
from placer_py.core.te_classifier import TEAlignmentEvidence

pytestmark = pytest.mark.golden

FIELDS = (
    "event_lower_log_lr", "independent_lower_log_lr", "sequence_lower_log_lr",
    "structure_lower_log_lr", "boundary_lower_log_lr",
    "ref_conflict_lower_log_lr", "structure_te_log_evidence",
    "structure_nonte_log_evidence", "structure_artifact_log_evidence",
    "raw_log_bf_te_vs_artifact", "raw_log_bf_te_vs_non_te",
    "lower_log_bf_te_vs_artifact", "lower_log_bf_te_vs_non_te",
    "mechanistic_support_signal", "ref_conflict_signal",
    "artifact_context_signal", "ambiguity_width",
)


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


#: All eight scenarios from tools/dump_oracle.cpp, mirrored by hand.
#:
#: Kept in step with that file deliberately rather than generated, so that a
#: scenario added there and not here fails loudly instead of being skipped.
#: My first version mirrored only the first scenario and the other seven were
#: silently skipped -- which is exactly the failure mode a "skips are the
#: migration surface" convention is supposed to make visible, and did not.
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


def _inputs(name: str):
    if name not in SCENARIOS:
        pytest.fail(f"golden scenario {name!r} is not mirrored in this test; "
                    f"add it here rather than letting it be skipped")
    return SCENARIOS[name]


def test_certificate_matches_the_cpp_field_for_field(oracle):
    checked = 0
    for golden in oracle["certificates"]:
        ex, seg, te, bd = _inputs(golden["name"])
        cert = call_or_skip(blocks.build_certificate, ex, seg, te, bd, None)
        for field in FIELDS:
            close(getattr(cert, field), golden[field],
                  f"{golden['name']}.{field}")
        checked += 1
    assert checked, "no scenario was actually compared"


def test_block_list_matches_name_for_name(oracle):
    golden = oracle["certificates"][0]
    ex, seg, te, bd = _inputs(golden["name"])
    cert = call_or_skip(blocks.build_certificate, ex, seg, te, bd, None)
    assert [b.name for b in cert.blocks] == [b["name"] for b in golden["blocks"]]
    for got, want in zip(cert.blocks, golden["blocks"]):
        close(got.raw_signal, want["raw_signal"], f"{got.name}.raw_signal")
        close(got.te_vs_artifact, want["te_vs_artifact"],
              f"{got.name}.te_vs_artifact")
        close(got.te_vs_non_te, want["te_vs_non_te"], f"{got.name}.te_vs_non_te")
        close(got.ambiguity_width, want["ambiguity_width"],
              f"{got.name}.ambiguity_width")


def test_robust_lfdr_matches_the_cpp(oracle):
    golden = oracle["certificates"][0]
    ex, seg, te, bd = _inputs(golden["name"])
    cert = call_or_skip(blocks.build_certificate, ex, seg, te, bd, None)
    got = call_or_skip(blocks.evaluate_robust_lfdr, cert, None, 0.10)
    close(got["worst_case_lfdr"], golden["robust_worst_case_lfdr"],
          "robust_worst_case_lfdr")
    assert got["qc"] == golden["robust_qc"]


@pytest.mark.regression
def test_per_locus_stage_charges_no_penalty(oracle):
    """`lower == raw` exactly. Earlier releases subtracted a fixed 0.65 nats
    here, and that invented constant is what the current design removed."""
    golden = oracle["certificates"][0]
    ex, seg, te, bd = _inputs(golden["name"])
    cert = call_or_skip(blocks.build_certificate, ex, seg, te, bd, None)
    assert cert.lower_log_bf_te_vs_artifact == cert.raw_log_bf_te_vs_artifact
    assert cert.lower_log_bf_te_vs_non_te == cert.raw_log_bf_te_vs_non_te
