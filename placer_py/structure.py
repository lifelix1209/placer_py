"""
The insert's structural decode: TE core, optional 3' transduction, poly(A).

Ported from `src/component/te_sequence_explainer.cpp`, pinned by the six golden
cases in `tests/test_03_structure.py`.

This is the closest thing in the C++ to a generative model of the biology -- a
minimal duration-aware HSMM over the left-to-right chain

    TE_CORE -> [TRANSDUCTION] -> [POLYA] -> residual

where each optional state opens when its summed per-base emission log-LR against
a generic-residual null, plus a log prior-odds, is positive.

Ported faithfully, INCLUDING two known defects, which are pinned as xfail in the
tests so a later fix has to change them deliberately:

  1. The HSMM's output is DEMOTED to a feature. The state posterior is
     multiplied by 0.55 and added into an affine scorer alongside identity and
     coverage. A log-likelihood-ratio times 0.55 is not a log-likelihood ratio
     of anything.
  2. A real 3' transduction is NET PENALISED despite having a dedicated state.
     The residual counters are computed BEFORE the transduction decode and never
     reduced by it, so the Viterbi path is reporting-only. For a 200 bp
     transduction in a 1 kb insert: coverage falls 0.2 and costs 2.45*0.2, the
     high-complexity residual rises and costs 1.25*0.2, and the transduction
     posterior returns only +0.35. Net -0.39 nats.

Note also that `terminal_poly_at_run` conflates A and T, discarding the
insertion's orientation -- poly(A) on the + strand and poly(T) on the - strand
are the same tail from opposite sides, and which one appears must agree with the
element alignment strand and the TSD geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class TeAnnotationStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    EMPTY = "EMPTY"
    TOO_SHORT = "TOO_SHORT"
    NO_CANDIDATE = "NO_CANDIDATE"
    RESOLVED = "RESOLVED"
    FAMILY_ONLY = "FAMILY_ONLY"
    UNKNOWN_TE = "UNKNOWN_TE"
    NON_TE_LIKE = "NON_TE_LIKE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class StructureSegment:
    state: str
    start: int
    end: int
    label: str


@dataclass
class Residual:
    unexplained_high_complexity_bases: int = 0
    low_complexity_only_bases: int = 0
    edit_distance: int = 0
    family_conflicts: int = 0
    subfamily_conflicts: int = 0
    segment_breaks: int = 0


@dataclass
class SequenceExplanation:
    status: TeAnnotationStatus = TeAnnotationStatus.UNAVAILABLE
    family: str = "UNKNOWN"
    subfamily: str = "UNKNOWN"
    te_structure_log_evidence: float = 0.0
    nonte_structure_log_evidence: float = 0.0
    artifact_structure_log_evidence: float = 0.0
    structure_path_confidence: float = 0.0
    polyA_posterior: float = 0.0
    transduction_posterior: float = 0.0
    te_core_coverage: float = 0.0
    unexplained_high_complexity_bp: int = 0
    residual: Residual = field(default_factory=Residual)
    path: list[StructureSegment] = field(default_factory=list)


# Emission constants, transcribed from the C++.
POLY_PER_BASE_LOG_LR = 1.28        # stated as log(0.90 / 0.25)
POLY_OPEN_LOG_ODDS = -3.0
TRANS_BASELINE_ENTROPY = 0.55
TRANS_OPEN_LOG_ODDS = -1.6


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _logistic(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _rounded_bases(insert_len: int, fraction: float) -> int:
    if insert_len <= 0 or not math.isfinite(fraction):
        return 0
    return int(math.floor(insert_len * _clamp01(fraction) + 0.5))


def _normalized_label(value: str, fallback: str) -> str:
    if not value or value == "NA":
        return fallback
    return value


def terminal_poly_at_run(seq: str) -> int:
    """
    Length of the terminal A/T-dominant homopolymer run.

    Conflates A and T, which discards the insertion's ORIENTATION. Reproduced as
    is so the diff against the C++ closes; recovering the orientation is a
    modelling change and belongs with the TPRT work.
    """
    if not seq:
        return 0
    base = seq[-1]
    if base not in ("A", "T"):
        return 0
    run = 0
    for char in reversed(seq):
        if char != base:
            break
        run += 1
    return run


def interval_entropy_norm(seq: str, start: int, end: int) -> float:
    """Shannon entropy over ACGT in [start, end), normalised by log 4."""
    start = max(0, min(start, len(seq)))
    end = max(start, min(end, len(seq)))
    length = end - start
    if length <= 0:
        return 0.0
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    for index in range(start, end):
        base = seq[index]
        if base in counts:
            counts[base] += 1
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / length
        entropy -= p * math.log(p)
    return _clamp01(entropy / math.log(4.0))


def status_from_qc(qc_reason: str, family: str,
                   subfamily: str) -> TeAnnotationStatus:
    table = {
        "TE_LIBRARY_UNAVAILABLE": TeAnnotationStatus.UNAVAILABLE,
        "EMPTY_INSERT_SEQUENCE": TeAnnotationStatus.EMPTY,
        "INSERT_SEQ_TOO_SHORT": TeAnnotationStatus.TOO_SHORT,
        "NO_TE_ALIGNMENT_SHORTLIST": TeAnnotationStatus.NO_CANDIDATE,
        "NO_TE_ALIGNMENT_MATCH": TeAnnotationStatus.NO_CANDIDATE,
        "NO_TE_ALIGNMENT": TeAnnotationStatus.NO_CANDIDATE,
        "PASS_INSERT_TE_ALIGNMENT": TeAnnotationStatus.RESOLVED,
        "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY": TeAnnotationStatus.FAMILY_ONLY,
        "PASS_INSERT_TE_ALIGNMENT_UNKNOWN": TeAnnotationStatus.UNKNOWN_TE,
        "TE_ALIGNMENT_LOW_IDENTITY": TeAnnotationStatus.NON_TE_LIKE,
        "TE_ALIGNMENT_LOW_QUERY_COVERAGE": TeAnnotationStatus.NON_TE_LIKE,
    }
    if qc_reason in table:
        return table[qc_reason]
    if not family or family == "UNKNOWN" or not subfamily or subfamily == "NA":
        return TeAnnotationStatus.AMBIGUOUS
    return TeAnnotationStatus.AMBIGUOUS


def _has_te_path(status: TeAnnotationStatus) -> bool:
    return status in (TeAnnotationStatus.RESOLVED,
                      TeAnnotationStatus.FAMILY_ONLY,
                      TeAnnotationStatus.UNKNOWN_TE,
                      TeAnnotationStatus.AMBIGUOUS)


def _te_path_label(status: TeAnnotationStatus, family: str,
                   subfamily: str) -> str:
    if status == TeAnnotationStatus.UNKNOWN_TE:
        return "UNKNOWN/UNKNOWN"
    normalized_family = _normalized_label(family, "UNKNOWN")
    normalized_subfamily = ("UNKNOWN" if status == TeAnnotationStatus.FAMILY_ONLY
                            else _normalized_label(subfamily, "UNKNOWN"))
    return f"{normalized_family}/{normalized_subfamily}"


def explain_te_sequence_structure(insert_seq: str, qc_reason: str, family: str,
                                  subfamily: str, best_identity: float,
                                  effective_query_coverage: float,
                                  annotation_residual_fraction: float,
                                  annotation_masked_fraction: float,
                                  cross_family_margin: float,
                                  second_score: float,
                                  sequence_model_label: str,
                                  sequence_model_score: float
                                  ) -> SequenceExplanation:
    """Port of `placer::explain_te_sequence_structure`."""
    insert_len = len(insert_seq)
    out = SequenceExplanation()
    out.status = status_from_qc(qc_reason, family, subfamily)
    out.family = ("UNKNOWN" if out.status == TeAnnotationStatus.UNKNOWN_TE
                  else _normalized_label(family, "UNKNOWN"))
    out.subfamily = ("UNKNOWN" if out.status in (TeAnnotationStatus.FAMILY_ONLY,
                                                 TeAnnotationStatus.UNKNOWN_TE)
                     else _normalized_label(subfamily, "UNKNOWN"))

    high_complexity_residual = max(0.0, annotation_residual_fraction
                                   - annotation_masked_fraction)
    out.residual.unexplained_high_complexity_bases = _rounded_bases(
        insert_len, high_complexity_residual)
    out.residual.low_complexity_only_bases = _rounded_bases(
        insert_len, annotation_masked_fraction)
    out.residual.edit_distance = _rounded_bases(
        insert_len, max(0.0, 1.0 - best_identity)
        * max(0.0, effective_query_coverage))
    out.residual.family_conflicts = (
        1 if (second_score > 0.0 and out.family == "UNKNOWN") else 0)
    out.residual.subfamily_conflicts = (
        1 if (out.status in (TeAnnotationStatus.FAMILY_ONLY,
                             TeAnnotationStatus.UNKNOWN_TE)
              or out.subfamily == "UNKNOWN") else 0)
    out.residual.segment_breaks = (
        1 if out.residual.unexplained_high_complexity_bases > 0 else 0)
    out.unexplained_high_complexity_bp = \
        out.residual.unexplained_high_complexity_bases
    out.te_core_coverage = _clamp01(effective_query_coverage)

    te_core_bases = (max(1, min(insert_len,
                                _rounded_bases(insert_len,
                                               effective_query_coverage)))
                     if insert_len > 0 and _has_te_path(out.status) else 0)
    residual_start = te_core_bases
    residual_len = max(0, insert_len - residual_start)
    poly_bases = min(terminal_poly_at_run(insert_seq), residual_len)
    residual_before_poly = max(0, residual_len - poly_bases)
    residual_entropy = interval_entropy_norm(
        insert_seq, residual_start, residual_start + residual_before_poly)

    poly_state_score = (poly_bases * POLY_PER_BASE_LOG_LR
                        + POLY_OPEN_LOG_ODDS)
    out.polyA_posterior = _logistic(poly_state_score)

    transduction_state_score = (
        residual_before_poly * 0.06 * (residual_entropy - TRANS_BASELINE_ENTROPY)
        + TRANS_OPEN_LOG_ODDS
        + 0.6 * _clamp01(best_identity))
    out.transduction_posterior = _logistic(transduction_state_score)

    identity = _clamp01(best_identity)
    coverage = out.te_core_coverage
    margin_signal = _clamp01(cross_family_margin * 4.0)
    residual_fraction = _clamp01(annotation_residual_fraction)
    high_complexity_residual_fraction = _clamp01(high_complexity_residual)
    masked_fraction = _clamp01(annotation_masked_fraction)
    ambiguity_penalty = (0.45 if out.status == TeAnnotationStatus.UNKNOWN_TE
                         else 0.25 if out.status == TeAnnotationStatus.FAMILY_ONLY
                         else 0.0)

    model_support = 0.0
    if sequence_model_label == "TE_MODEL_IN_DISTRIBUTION":
        model_support = 0.35 + (0.25 * _clamp01(sequence_model_score))
    elif sequence_model_label == "TE_MODEL_EDGE":
        model_support = -0.15
    elif sequence_model_label == "TE_MODEL_OUTLIER":
        model_support = -0.85

    out.te_structure_log_evidence = (
        -1.10
        + (2.00 * identity)
        + (2.45 * coverage)
        + (0.55 * margin_signal)
        + (0.55 * out.polyA_posterior)
        + (0.35 * out.transduction_posterior)
        + model_support
        - (1.25 * high_complexity_residual_fraction)
        - (0.55 * masked_fraction)
        - ambiguity_penalty)
    if not _has_te_path(out.status):
        out.te_structure_log_evidence -= 2.0

    out.nonte_structure_log_evidence = (
        -0.25
        + (1.65 * high_complexity_residual_fraction)
        + (0.75 * residual_fraction)
        - (1.25 * coverage)
        - (0.85 * identity)
        - (0.25 * out.polyA_posterior))

    out.artifact_structure_log_evidence = (
        -0.10
        + (1.10 * masked_fraction)
        + (1.25 if sequence_model_label == "TE_MODEL_OUTLIER" else 0.0)
        + (0.65 if qc_reason == "TE_ALIGNMENT_LOW_IDENTITY" else 0.0)
        - (0.95 * coverage)
        - (0.75 * identity))

    best_null = max(out.nonte_structure_log_evidence,
                    out.artifact_structure_log_evidence)
    out.structure_path_confidence = _logistic(
        out.te_structure_log_evidence - best_null)

    if _has_te_path(out.status) and insert_len > 0:
        out.path.append(StructureSegment(
            "TE_CORE", 0, te_core_bases,
            _te_path_label(out.status, out.family, out.subfamily)))
        cursor = te_core_bases
        poly_start = insert_len - poly_bases
        add_transduction = residual_before_poly > 0 and transduction_state_score > 0.0
        add_poly = poly_bases > 0 and poly_state_score > 0.0
        if add_transduction and cursor < poly_start:
            out.path.append(StructureSegment("TRANSDUCTION", cursor, poly_start,
                                             "HIGH_COMPLEXITY_TAIL"))
            cursor = poly_start
        if add_poly and cursor < insert_len:
            out.path.append(StructureSegment(
                "POLYA", max(cursor, poly_start), insert_len,
                "POLY_N" if not insert_seq else f"POLY_{insert_seq[-1]}"))
            cursor = insert_len
        if masked_fraction > 0.20 and cursor < insert_len:
            out.path.append(StructureSegment("LOW_COMPLEXITY", cursor,
                                             insert_len, "MASKED_RESIDUAL"))
    return out


def explain_te_alignment_shadow(insert_len: int, qc_reason: str, family: str,
                                subfamily: str, best_identity: float,
                                effective_query_coverage: float,
                                annotation_residual_fraction: float,
                                annotation_masked_fraction: float,
                                cross_family_margin: float,
                                second_score: float) -> SequenceExplanation:
    """The decode run WITHOUT the insert sequence, from its length alone.

    Two details a reasonable reimplementation gets wrong, and both are load
    bearing:

      * the sequence passed down is `insert_len` copies of 'N', NOT the empty
        string. 'N' is neither A nor T, so `terminal_poly_at_run` returns 0 and
        the residual entropy is 0 -- but the LENGTH still drives
        `te_core_bases`, `residual_len` and the rounded residual counters, so an
        empty string is not equivalent.
      * it hard-codes `TE_MODEL_UNAVAILABLE` and a model score of 0, DISCARDING
        the alignment's own model label. So `model_support` is 0 on this path
        even when the alignment says `TE_MODEL_IN_DISTRIBUTION`, and the
        structure evidence comes out about 0.49 nats lower than it would with
        the real label.
    """
    return explain_te_sequence_structure(
        "N" * max(0, insert_len), qc_reason, family, subfamily, best_identity,
        effective_query_coverage, annotation_residual_fraction,
        annotation_masked_fraction, cross_family_margin, second_score,
        "TE_MODEL_UNAVAILABLE", 0.0)


def te_annotation_status_name(status: TeAnnotationStatus) -> str:
    return status.value


def serialize_te_path_residual(residual: Residual) -> str:
    """The residual, as the ledger spells it.

    `orientation_conflicts` is always 0 in this decode and is emitted anyway --
    the column exists for the alignment-driven decode that would set it, and
    dropping it here would change the ledger's shape depending on which decode
    ran.
    """
    return (f"unexplained={residual.unexplained_high_complexity_bases}"
            f";edit_distance={residual.edit_distance}"
            f";family_conflicts={residual.family_conflicts}"
            f";subfamily_conflicts={residual.subfamily_conflicts}"
            f";orientation_conflicts=0"
            f";segment_breaks={residual.segment_breaks}"
            f";low_complexity_only={residual.low_complexity_only_bases}")


def serialize_te_sequence_path(explanation: SequenceExplanation) -> str:
    """`start-end:kind:label` per segment, or `NA` for an empty path."""
    if not explanation.path:
        return "NA"
    return ",".join(f"{segment.start}-{segment.end}:{segment.state}:{segment.label}"
                    for segment in explanation.path)
