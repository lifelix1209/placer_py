"""
The six mechanistic evidence blocks and the two aggregates.

Ported from `src/pipeline/mechanistic_evidence.cpp`, pinned by the eight golden
certificates in `tests/test_01_blocks.py` and the cross-field algebra in
`tests/test_09_oracle_selfconsistency.py`.

WHAT THIS IS FOR, and it is not the decision path.

These six affine maps carry nineteen hand-set constants, and the measurement in
`tests/test_12_head_to_head.py` shows they cannot support genome-scale FDR
control at all: each block is clamped to a small range, the total caps near 9.9
nats, and e-BH's rank-1 threshold at m=1060, q=0.10 is 9.27 nats before any
penalty. Worse, their null expectation is an unknown number that has to be
measured, and `placer_py/decoys.py` shows that it cannot be -- both candidate
samples fail, in opposite directions.

So this module exists to reproduce the C++ EXACTLY, for the diff, and not to be
used for calling. `placer_py/tprt.py` is the decision path: real log-LRs whose
null expectation is 1 by construction, measured at 0.186 on simulated nulls, and
which recover 96% recall at FDP 0.000 where these score nothing.

Ported unchanged anyway, including two inconsistencies, because a migration that
changes behaviour cannot be validated by diffing against the thing it replaces:

  * the per-locus stage charges NO dependency penalty, so `lower == raw`;
  * the per-block `te_vs_non_te` values use weights 0.45 / 0.65 / 1.0 / 0.80 /
    0.50 while the AGGREGATE uses 0.40 / 0.70 / 1.0 / 0.85 / 0.35, and the
    structure block differs too, so the serialized blocks do NOT sum to the
    aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import structure as structure_module


@dataclass
class Block:
    name: str
    raw_signal: float
    te_vs_artifact: float
    te_vs_non_te: float
    ambiguity_width: float


@dataclass
class Certificate:
    blocks: list[Block] = field(default_factory=list)
    event_lower_log_lr: float = 0.0
    independent_lower_log_lr: float = 0.0
    sequence_lower_log_lr: float = 0.0
    structure_lower_log_lr: float = 0.0
    boundary_lower_log_lr: float = 0.0
    ref_conflict_lower_log_lr: float = 0.0
    structure_te_log_evidence: float = 0.0
    structure_nonte_log_evidence: float = 0.0
    structure_artifact_log_evidence: float = 0.0
    raw_log_bf_te_vs_artifact: float = 0.0
    raw_log_bf_te_vs_non_te: float = 0.0
    lower_log_bf_te_vs_artifact: float = 0.0
    lower_log_bf_te_vs_non_te: float = 0.0
    mechanistic_support_signal: float = 0.0
    ref_conflict_signal: float = 0.0
    artifact_context_signal: float = 0.0
    ambiguity_width: float = 0.0


DEFAULT_PRIOR = {
    "te_min": 0.05, "te_max": 0.80,
    "non_te_min": 1e-6, "non_te_max": 0.30,
    "artifact_min": 0.50, "artifact_max": 0.95,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _positive(value: int) -> int:
    return max(0, int(value))


def count_signal(count: int, scale: float) -> float:
    """`1 - exp(-count / scale)`, clamped to [0,1]; 0 for count <= 0.

    The scale is one of the hand-set constants (4.0 for mechanistic reads, 8.0
    for support reads and reference counterevidence). It is really asking "how
    many reads count as many", which has an objective answer in the read-length
    distribution and the coverage.
    """
    if count <= 0:
        return 0.0
    return _clamp01(1.0 - math.exp(-count / max(scale, 1e-6)))


def _is_one_sided_segmentation_pass(segmentation: dict) -> bool:
    return bool(segmentation.get("has_insert_seq")
                and not segmentation.get("pair_valid")
                and (bool(segmentation.get("has_left_flank"))
                     != bool(segmentation.get("has_right_flank"))))


def mechanistic_read_signal(existence: dict,
                            clip_insert_concordance: dict | None) -> float:
    alt = _positive(existence.get("alt_struct_reads", 0))
    mechanistic_reads = (
        _positive(existence.get("alt_split_reads", 0))
        + _positive(existence.get("alt_indel_reads", 0))
        + min(_positive(existence.get("alt_left_clip_reads", 0)),
              _positive(existence.get("alt_right_clip_reads", 0))))
    if clip_insert_concordance is not None and clip_insert_concordance.get("pass"):
        mechanistic_reads += _positive(
            clip_insert_concordance.get("full_insert_reads", 0))
        mechanistic_reads += min(
            _positive(clip_insert_concordance.get("left_clip_reads", 0)),
            _positive(clip_insert_concordance.get("right_clip_reads", 0)))
    if alt <= 0 or mechanistic_reads <= 0:
        return 0.0
    count_part = count_signal(mechanistic_reads, 4.0)
    fraction_part = _clamp01(mechanistic_reads / alt)
    return math.sqrt(count_part * fraction_part)


def ref_conflict_signal(existence: dict, segmentation: dict) -> float:
    alt = _positive(existence.get("alt_struct_reads", 0))
    ref = _positive(existence.get("ref_span_reads", 0))
    signal = (ref / (alt + ref)) if (alt + ref) > 0 else 0.0
    signal = max(signal, 0.60 * count_signal(ref, 8.0))
    if _is_one_sided_segmentation_pass(segmentation) and ref > 0:
        signal = max(signal, 0.35)
    return _clamp01(signal)


def event_signal(existence: dict, independent_signal: float) -> float:
    support = count_signal(_positive(existence.get("alt_struct_reads", 0)), 8.0)
    quality = _clamp01(float(existence.get("gq", 0)) / 60.0)
    return _clamp01((0.45 * support)
                    + (0.30 * independent_signal)
                    + (0.15 * quality)
                    + (0.10 * _clamp01(float(existence.get("af", 0.0)))))


def sequence_signal(te: dict) -> float:
    qc = te.get("qc_reason", "")
    if not te.get("pass_", te.get("pass", False)) and qc != "TE_ALIGNMENT_LOW_IDENTITY":
        return 0.0
    signal = ((0.48 * _clamp01(float(te.get("best_identity", 0.0))))
              + (0.32 * _clamp01(float(te.get("best_query_coverage", 0.0))))
              + (0.20 * _clamp01(float(te.get("cross_family_margin", 0.0)) * 4.0)))
    if qc == "PASS_INSERT_TE_ALIGNMENT":
        signal += 0.15
    elif qc == "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY":
        signal += 0.08
    elif qc == "PASS_INSERT_TE_ALIGNMENT_UNKNOWN":
        signal -= 0.08
    elif qc == "TE_ALIGNMENT_LOW_IDENTITY":
        signal -= 0.25

    label = te.get("sequence_model_label", "")
    if label == "TE_MODEL_IN_DISTRIBUTION":
        signal += 0.12 + (0.10 * _clamp01(float(te.get("sequence_model_score", 0.0))))
    elif label == "TE_MODEL_EDGE":
        signal -= 0.10
    elif label == "TE_MODEL_OUTLIER":
        signal -= 0.45

    confidence = te.get("annotation_confidence", "")
    if confidence == "HIGH":
        signal += 0.08
    elif confidence == "LOW":
        signal -= 0.12

    signal -= 0.20 * _clamp01(float(te.get("annotation_residual_fraction", 0.0)))
    return _clamp01(signal)


def boundary_signal(segmentation: dict, boundary: dict) -> float:
    if not segmentation.get("has_insert_seq"):
        return 0.0
    signal = 0.20
    if (segmentation.get("pair_valid") and segmentation.get("has_left_flank")
            and segmentation.get("has_right_flank")):
        signal += 0.35
    elif segmentation.get("pair_valid"):
        signal += 0.20
    elif _is_one_sided_segmentation_pass(segmentation):
        signal += 0.10
    if bool(segmentation.get("has_left_flank")) != bool(segmentation.get("has_right_flank")):
        signal += 0.05
    if boundary.get("geometry_defined") and boundary.get("canonical_pass"):
        signal += 0.35
    elif boundary.get("geometry_defined") and boundary.get("evidence_consistent"):
        signal += 0.22
    elif not boundary.get("geometry_defined"):
        signal -= 0.10
    btype = boundary.get("boundary_type", "")
    if btype == "TSD":
        signal += 0.15
    elif btype in ("BLUNT", "SMALL_DEL"):
        signal += 0.05
    return _clamp01(signal)


def artifact_context_signal(segmentation: dict, te: dict, boundary: dict,
                            ref_conflict: float) -> float:
    signal = 0.80 * ref_conflict
    if not segmentation.get("has_insert_seq"):
        signal = max(signal, 1.0)
    label = te.get("sequence_model_label", "")
    if label == "TE_MODEL_OUTLIER":
        signal = max(signal, 0.90)
    elif label == "TE_MODEL_EDGE":
        signal = max(signal, 0.45)
    if (boundary.get("geometry_defined") and not boundary.get("canonical_pass")
            and not boundary.get("evidence_consistent")):
        signal = max(signal, 0.55)
    if te.get("annotation_confidence") == "LOW":
        signal = max(signal, 0.35)
    return _clamp01(signal)


def _structure_ambiguity_width(segmentation: dict,
                               explanation) -> float:
    insert_len = int(segmentation.get("insert_len", 0))
    residual_fraction = (
        _clamp(max(0, explanation.unexplained_high_complexity_bp) / insert_len,
               0.0, 1.0) if insert_len > 0 else 1.0)
    return (0.18
            + (0.45 * (1.0 - _clamp(explanation.structure_path_confidence, 0.0, 1.0)))
            + (0.30 * residual_fraction))


def _structure_explanation(segmentation: dict, te: dict):
    """
    The C++ uses `te_alignment.te_sequence_explanation` when populated and
    otherwise falls back to `explain_te_alignment_shadow`.

    TWO DETAILS of the shadow path that a reasonable reimplementation gets
    wrong, and mine did:

      * the sequence it passes is `insert_len` copies of 'N', not the empty
        string. 'N' is neither A nor T, so `terminal_poly_at_run` returns 0 and
        the residual entropy is 0 -- but the LENGTH still drives
        `te_core_bases`, `residual_len` and the rounded residual counters, so an
        empty string is not equivalent.
      * it hard-codes `sequence_model_label = "TE_MODEL_UNAVAILABLE"` and
        `sequence_model_score = 0.0`, DISCARDING the alignment's own model
        label. So `model_support` is 0 on this path even when the alignment says
        `TE_MODEL_IN_DISTRIBUTION`.

    Passing the te's real label instead put `structure_te_log_evidence` at
    3.977 against the C++'s 3.486 -- the 0.491 being exactly the
    `0.35 + 0.25*0.35` of model support plus the poly(A) and transduction states
    that an empty sequence leaves at their open-odds logistics.
    """
    supplied = te.get("te_sequence_explanation")
    if supplied is not None:
        return supplied
    insert_len = max(0, int(segmentation.get("insert_len", 0)))
    return structure_module.explain_te_sequence_structure(
        "N" * insert_len, te.get("qc_reason", ""),
        te.get("best_family", "UNKNOWN"), te.get("best_subfamily", "UNKNOWN"),
        float(te.get("best_identity", 0.0)),
        float(te.get("best_query_coverage", 0.0)),
        float(te.get("annotation_residual_fraction", 0.0)),
        float(te.get("annotation_masked_fraction", 0.0)),
        float(te.get("cross_family_margin", 0.0)),
        float(te.get("second_score", 0.0)),
        "TE_MODEL_UNAVAILABLE", 0.0)


def build_certificate(existence: dict, segmentation: dict, te_alignment: dict,
                      boundary: dict,
                      clip_insert_concordance: dict | None = None
                      ) -> Certificate:
    """Port of `placer::build_mechanistic_evidence_certificate`."""
    cert = Certificate()
    independent = mechanistic_read_signal(existence, clip_insert_concordance)
    event = event_signal(existence, independent)
    sequence = sequence_signal(te_alignment)
    explanation = _structure_explanation(segmentation, te_alignment)
    boundary_mechanism = boundary_signal(segmentation, boundary)
    ref_conflict = ref_conflict_signal(existence, segmentation)
    artifact_context = artifact_context_signal(segmentation, te_alignment,
                                               boundary, ref_conflict)

    cert.mechanistic_support_signal = _clamp01(
        (0.35 * independent) + (0.25 * event)
        + (0.25 * boundary_mechanism) + (0.15 * sequence))
    cert.ref_conflict_signal = ref_conflict
    cert.artifact_context_signal = artifact_context

    cert.event_lower_log_lr = -0.75 + (3.00 * event) - (2.20 * ref_conflict)
    cert.independent_lower_log_lr = -1.00 + (4.00 * independent) - (1.20 * ref_conflict)
    cert.sequence_lower_log_lr = -1.40 + (4.20 * sequence) - (1.00 * artifact_context)
    cert.structure_te_log_evidence = explanation.te_structure_log_evidence
    cert.structure_nonte_log_evidence = explanation.nonte_structure_log_evidence
    cert.structure_artifact_log_evidence = explanation.artifact_structure_log_evidence
    cert.structure_lower_log_lr = _clamp(
        cert.structure_te_log_evidence
        - max(cert.structure_nonte_log_evidence,
              cert.structure_artifact_log_evidence), -4.0, 5.0)
    cert.boundary_lower_log_lr = -0.70 + (2.40 * boundary_mechanism) - (0.70 * ref_conflict)
    cert.ref_conflict_lower_log_lr = -3.00 * ref_conflict - (0.80 * artifact_context)

    cert.blocks = [
        Block("event", event, cert.event_lower_log_lr,
              0.45 * cert.event_lower_log_lr, 0.25 + (0.30 * (1.0 - event))),
        Block("independent", independent, cert.independent_lower_log_lr,
              0.65 * cert.independent_lower_log_lr,
              0.30 + (0.45 * (1.0 - independent))),
        Block("sequence", sequence, cert.sequence_lower_log_lr,
              cert.sequence_lower_log_lr, 0.25 + (0.40 * (1.0 - sequence))),
        Block("structure", explanation.structure_path_confidence,
              _clamp(cert.structure_te_log_evidence
                     - cert.structure_artifact_log_evidence, -4.0, 5.0),
              _clamp(cert.structure_te_log_evidence
                     - cert.structure_nonte_log_evidence, -4.0, 5.0),
              _structure_ambiguity_width(segmentation, explanation)),
        Block("boundary", boundary_mechanism, cert.boundary_lower_log_lr,
              0.80 * cert.boundary_lower_log_lr,
              0.20 + (0.35 * (1.0 - boundary_mechanism))),
        Block("ref_conflict", ref_conflict, cert.ref_conflict_lower_log_lr,
              0.50 * cert.ref_conflict_lower_log_lr,
              0.20 + (0.50 * ref_conflict)),
    ]
    cert.ambiguity_width = sum(b.ambiguity_width for b in cert.blocks) / max(
        1, len(cert.blocks))

    clamped_structure = _clamp(cert.structure_lower_log_lr, -4.0, 5.0)
    cert.raw_log_bf_te_vs_artifact = (
        cert.event_lower_log_lr + cert.independent_lower_log_lr
        + cert.sequence_lower_log_lr + (0.90 * clamped_structure)
        + cert.boundary_lower_log_lr + cert.ref_conflict_lower_log_lr)
    cert.raw_log_bf_te_vs_non_te = (
        (0.40 * cert.event_lower_log_lr)
        + (0.70 * cert.independent_lower_log_lr)
        + cert.sequence_lower_log_lr
        + (0.90 * clamped_structure)
        + (0.85 * cert.boundary_lower_log_lr)
        + (0.35 * cert.ref_conflict_lower_log_lr))

    # Per-locus screening charges NO dependency penalty: the certified bound is
    # a whole-run quantity that does not exist yet here, and filling the gap
    # with an invented constant is the thing this design removed.
    cert.lower_log_bf_te_vs_artifact = cert.raw_log_bf_te_vs_artifact
    cert.lower_log_bf_te_vs_non_te = cert.raw_log_bf_te_vs_non_te
    return cert


def evaluate_robust_lfdr(certificate: Certificate, prior: dict | None = None,
                         target_q: float = 0.10) -> dict:
    """
    Port of `placer::evaluate_robust_mechanistic_lfdr`.

    Demoted to a diagnostic in the current pipeline -- it no longer decides
    emission -- but still reported, so the port has to match it.
    """
    p = dict(DEFAULT_PRIOR)
    if prior:
        p.update(prior)
    lower_log_bf = min(certificate.lower_log_bf_te_vs_artifact,
                       certificate.lower_log_bf_te_vs_non_te)
    te_odds_lower = math.exp(
        _clamp(lower_log_bf - certificate.ambiguity_width, -60.0, 60.0)
    ) * max(p["te_min"], 1e-12)
    null_odds_upper = max(0.0, p["artifact_max"]) + max(0.0, p["non_te_max"])
    worst = null_odds_upper / max(1e-12, null_odds_upper + te_odds_lower)
    return {
        "lfdr": worst,
        "worst_case_lfdr": worst,
        "qc": ("PASS_TE_LFDR" if worst <= _clamp(target_q, 0.0, 1.0)
               else "TE_LFDR_HIGH"),
    }


def serialize_blocks(certificate: Certificate) -> str:
    if not certificate.blocks:
        return "NA"
    parts = [f"{b.name}:raw={b.raw_signal},te_art={b.te_vs_artifact},"
             f"te_non={b.te_vs_non_te},amb={b.ambiguity_width}"
             for b in certificate.blocks]
    return ";".join(parts)
