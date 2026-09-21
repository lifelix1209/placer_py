from __future__ import annotations

import math
from dataclasses import dataclass

from placer_py.redesign.evidence.reference_boundary import ReferenceEvidence
from placer_py.redesign.evidence.segmental_explanation import (
    SegmentalExplanation,
    segmental_explanation_from_features,
)
from placer_py.redesign.model.mechanistic import (
    DEFAULT_HP_COV_MIN,
    DEFAULT_HP_MIN,
    high_precision_te_score,
    is_confident_te_insertion,
)
from placer_py.redesign.models import BamEvidence, InsertionCandidate, SequenceFeatures, TeHit


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class EventExistenceEvidence:
    best_gt: str = "./."
    af: float = 0.0
    gq: int = 0
    alt_struct_reads: int = 0
    alt_split_reads: int = 0
    alt_indel_reads: int = 0
    alt_left_clip_reads: int = 0
    alt_right_clip_reads: int = 0
    ref_span_reads: int = 0
    depth: int = 0
    score: float = -3.0


@dataclass(frozen=True)
class EventSegmentationEvidence:
    has_consensus: bool = False
    has_left_flank: bool = False
    has_right_flank: bool = False
    has_insert_seq: bool = False
    pair_valid: bool = False
    left_align_len: int = 0
    right_align_len: int = 0
    left_identity: float = 0.0
    right_identity: float = 0.0
    insert_len: int = 0
    score: float = -2.0
    qc: str = "NO_INSERT_SEQUENCE"


@dataclass(frozen=True)
class TeAlignmentEvidence:
    best_family: str = ""
    best_subfamily: str = ""
    best_identity: float = 0.0
    best_query_coverage: float = 0.0
    cross_family_margin: float = 0.0
    pass_alignment: bool = False
    qc_reason: str = "NO_TE_ALIGNMENT"
    sequence_model_label: str = "TE_MODEL_UNAVAILABLE"
    sequence_model_score: float = 0.0
    annotation_confidence: str = "NA"
    annotation_residual_fraction: float = 0.0
    annotation_masked_fraction: float = 0.0
    orientation: str = ""


@dataclass(frozen=True)
class BoundaryEvidence:
    geometry_defined: bool = False
    canonical_pass: bool = False
    evidence_consistent: bool = False
    boundary_type: str = "REJECT"
    boundary_len: int = 0
    score: float = -2.0
    qc: str = "BOUNDARY_NOT_EVALUATED"
    reference_qc: str = "REFERENCE_NOT_EVALUATED"
    resolved_reference_chrom: str = ""
    reference_breakpoint0: int = -1
    reference_right_breakpoint0: int = -1
    left_ref_anchor_len: int = 0
    right_ref_anchor_len: int = 0
    left_ref_anchor_identity: float = 0.0
    right_ref_anchor_identity: float = 0.0
    remap_qc: str = "REFERENCE_REMAP_NOT_EVALUATED"
    insert_core_len: int = 0
    tsd_type: str = "NONE"
    tsd_len: int = 0
    tsd_sequence: str = ""
    tsd_bg_p: float = 1.0
    tsd_qc: str = "NO_TSD"
    reference_n_fraction: float = 0.0
    reference_max_n_run: int = 0


@dataclass(frozen=True)
class LocalHypothesisPosterior:
    log_te: float = -1.0
    log_non_te: float = -0.5
    log_artifact: float = 0.0
    te: float = 0.186324
    non_te: float = 0.307196
    artifact: float = 0.506480
    te_vs_artifact_log_odds: float = -1.0
    te_vs_non_te_log_odds: float = -0.5
    qc: str = "TE_POSTERIOR_LOW"


@dataclass(frozen=True)
class JointDecisionResult:
    label: str = "artifact"
    best_hypothesis: str = "REFERENCE"
    final_qc: str = "REFERENCE_OR_ARTIFACT"
    posterior_qc: str = "TE_POSTERIOR_LOW"
    te_posterior: float = 0.0
    non_te_posterior: float = 0.0
    artifact_posterior: float = 1.0
    te_vs_artifact_log_odds: float = 0.0
    te_vs_non_te_log_odds: float = 0.0
    emit_te_call: bool = False
    emit_unknown_te: bool = False
    emit_evidence_te_call: bool = False
    latent_mechanism: str = "artifact_reference"
    lfdr: float = 1.0
    robust_mechanistic_lfdr: float = 1.0
    te_alignment_qc: str = "NO_TE_ALIGNMENT"
    sequence_model_label: str = "TE_MODEL_UNAVAILABLE"
    boundary_qc: str = "BOUNDARY_NOT_EVALUATED"
    segmentation_qc: str = "NO_INSERT_SEQUENCE"
    reference_qc: str = "REFERENCE_NOT_EVALUATED"
    resolved_reference_chrom: str = ""
    reference_breakpoint0: int = -1
    reference_right_breakpoint0: int = -1
    left_ref_anchor_len: int = 0
    right_ref_anchor_len: int = 0
    left_ref_anchor_identity: float = 0.0
    right_ref_anchor_identity: float = 0.0
    remap_qc: str = "REFERENCE_REMAP_NOT_EVALUATED"
    insert_core_len: int = 0
    tsd_type: str = "NONE"
    tsd_len: int = 0
    tsd_sequence: str = ""
    tsd_bg_p: float = 1.0
    tsd_qc: str = "NO_TSD"
    reference_n_fraction: float = 0.0
    reference_max_n_run: int = 0
    seq_core_fraction: float = 0.0
    seq_core_contiguity: float = 0.0
    seq_polya_fraction: float = 0.0
    seq_low_complexity_fraction: float = 0.0
    seq_unexplained_fraction: float = 1.0
    seq_family_margin: float = 0.0
    seq_explanation_qc: str = "NO_INSERT_SEQUENCE"
    worst_case_lfdr: float = 1.0
    target_q: float = 0.10
    te_log_evidence: float = 0.0
    nonte_log_evidence: float = 0.0
    artifact_log_evidence: float = 0.0
    lfdr_qc: str = "TE_LFDR_REJECT"
    high_precision_score: float = 0.0
    emit_te_high_precision: bool = False

    def to_tsv_fields(self) -> dict[str, str]:
        fields = {
            "final_qc": self.final_qc,
            "posterior_qc": self.posterior_qc,
            "best_hypothesis": self.best_hypothesis,
            "emit_te_call": "1" if self.emit_te_call else "0",
            "emit_unknown_te": "1" if self.emit_unknown_te else "0",
            "emit_evidence_te_call": "1" if self.emit_evidence_te_call else "0",
            "te_vs_artifact_log_odds": _fmt_float(self.te_vs_artifact_log_odds),
            "te_vs_non_te_log_odds": _fmt_float(self.te_vs_non_te_log_odds),
            "latent_mechanism": self.latent_mechanism,
            "robust_mechanistic_lfdr": _fmt_float(self.robust_mechanistic_lfdr),
            "te_alignment_qc": self.te_alignment_qc,
            "sequence_model_label": self.sequence_model_label,
            "boundary_qc": self.boundary_qc,
            "segmentation_qc": self.segmentation_qc,
        }
        fields.update(
            {
                "reference_qc": self.reference_qc,
                "resolved_reference_chrom": self.resolved_reference_chrom,
                "reference_breakpoint0": str(self.reference_breakpoint0),
                "reference_right_breakpoint0": str(self.reference_right_breakpoint0),
                "left_ref_anchor_len": str(self.left_ref_anchor_len),
                "right_ref_anchor_len": str(self.right_ref_anchor_len),
                "left_ref_anchor_identity": _fmt_float(self.left_ref_anchor_identity),
                "right_ref_anchor_identity": _fmt_float(self.right_ref_anchor_identity),
                "remap_qc": self.remap_qc,
                "insert_core_len": str(self.insert_core_len),
                "tsd_type": self.tsd_type,
                "tsd_len": str(self.tsd_len),
                "tsd_sequence": self.tsd_sequence,
                "tsd_bg_p": _fmt_float(self.tsd_bg_p),
                "tsd_qc": self.tsd_qc,
                "reference_n_fraction": _fmt_float(self.reference_n_fraction),
                "reference_max_n_run": str(self.reference_max_n_run),
                "seq_core_fraction": _fmt_float(self.seq_core_fraction),
                "seq_core_contiguity": _fmt_float(self.seq_core_contiguity),
                "seq_polya_fraction": _fmt_float(self.seq_polya_fraction),
                "seq_low_complexity_fraction": _fmt_float(self.seq_low_complexity_fraction),
                "seq_unexplained_fraction": _fmt_float(self.seq_unexplained_fraction),
                "seq_family_margin": _fmt_float(self.seq_family_margin),
                "seq_explanation_qc": self.seq_explanation_qc,
                "worst_case_lfdr": _fmt_float(self.worst_case_lfdr),
                "target_q": _fmt_float(self.target_q),
                "te_log_evidence": _fmt_float(self.te_log_evidence),
                "nonte_log_evidence": _fmt_float(self.nonte_log_evidence),
                "artifact_log_evidence": _fmt_float(self.artifact_log_evidence),
                "lfdr_qc": self.lfdr_qc,
                "high_precision_score": _fmt_float(self.high_precision_score),
                "emit_te_high_precision": "1" if self.emit_te_high_precision else "0",
            }
        )
        return fields


@dataclass(frozen=True)
class DiscriminatorEvidence:
    existence: EventExistenceEvidence
    segmentation: EventSegmentationEvidence
    te_alignment: TeAlignmentEvidence
    boundary: BoundaryEvidence
    segmental: SegmentalExplanation = SegmentalExplanation()


def _gq_proxy(alt: int, ref: int) -> int:
    if alt <= 0:
        return 0
    log_odds = math.log((alt + 1.0) / (ref + 1.0))
    return int(round(_clamp(20.0 + (20.0 * log_odds), 0.0, 99.0)))


def _sequence_model_label(features: SequenceFeatures) -> str:
    if features.length <= 0:
        return "TE_MODEL_UNAVAILABLE"
    if features.low_complexity_fraction >= 0.65 or features.entropy < 0.75:
        return "TE_MODEL_OUTLIER"
    if features.low_complexity_fraction >= 0.35 or features.entropy < 1.25:
        return "TE_MODEL_EDGE"
    return "TE_MODEL_IN_DISTRIBUTION"


def _te_alignment_qc(hit: TeHit) -> tuple[bool, str]:
    if not hit.family or hit.query_coverage <= 0.0:
        return False, "NO_TE_ALIGNMENT"
    if hit.family == "UNKNOWN" or hit.subfamily == "UNKNOWN":
        return True, "PASS_INSERT_TE_ALIGNMENT_UNKNOWN"
    if hit.query_coverage >= 0.75:
        return True, "PASS_INSERT_TE_ALIGNMENT"
    if hit.query_coverage >= 0.35:
        return True, "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY"
    return False, "TE_ALIGNMENT_LOW_COVERAGE"


def build_discriminator_evidence(
    candidate: InsertionCandidate,
    bam_evidence: BamEvidence,
    sequence_features: SequenceFeatures,
    te_hit: TeHit,
    reference_evidence: ReferenceEvidence | None = None,
    segmental: SegmentalExplanation | None = None,
) -> DiscriminatorEvidence:
    clip_support = bam_evidence.left_clip_reads + bam_evidence.right_clip_reads
    alt_struct = max(0, max(candidate.support, bam_evidence.raw_cigar_insert_reads + clip_support))
    ref_span = max(0, bam_evidence.ref_span_reads)
    depth = max(bam_evidence.local_depth, alt_struct + ref_span)
    gq = _gq_proxy(alt_struct, ref_span)
    existence = EventExistenceEvidence(
        best_gt="0/1" if alt_struct > ref_span and alt_struct > 0 else "0/0",
        af=alt_struct / float(max(1, alt_struct + ref_span)),
        gq=gq,
        alt_struct_reads=alt_struct,
        alt_split_reads=max(0, bam_evidence.split_read_support),
        alt_indel_reads=max(0, bam_evidence.raw_cigar_insert_reads),
        alt_left_clip_reads=max(0, bam_evidence.left_clip_reads),
        alt_right_clip_reads=max(0, bam_evidence.right_clip_reads),
        ref_span_reads=ref_span,
        depth=depth,
        score=_clamp((gq - 20.0) / 20.0, -3.0, 3.0),
    )

    has_insert_seq = sequence_features.length > 0
    has_left_flank = bool(reference_evidence and reference_evidence.has_left_flank)
    has_right_flank = bool(reference_evidence and reference_evidence.has_right_flank)
    pair_valid = has_insert_seq and (
        reference_evidence.pair_valid if reference_evidence is not None else True
    )
    if reference_evidence is not None and has_insert_seq:
        if reference_evidence.pair_valid:
            segmentation_score = 1.0
            segmentation_qc = "PASS_EVENT_SEGMENTATION"
        elif reference_evidence.has_left_flank or reference_evidence.has_right_flank:
            segmentation_score = 0.55
            segmentation_qc = "PASS_EVENT_SEGMENTATION_ONE_SIDED"
        else:
            segmentation_score = 0.25
            segmentation_qc = "PASS_INSERT_SEQUENCE_AVAILABLE"
    else:
        segmentation_score = 0.25 if has_insert_seq else -2.0
        segmentation_qc = "PASS_INSERT_SEQUENCE_AVAILABLE" if has_insert_seq else "NO_INSERT_SEQUENCE"
    segmentation = EventSegmentationEvidence(
        has_consensus=has_insert_seq,
        has_left_flank=has_left_flank,
        has_right_flank=has_right_flank,
        has_insert_seq=has_insert_seq,
        pair_valid=pair_valid,
        left_align_len=reference_evidence.left_anchor_len if reference_evidence is not None else 0,
        right_align_len=reference_evidence.right_anchor_len if reference_evidence is not None else 0,
        left_identity=reference_evidence.left_anchor_identity if reference_evidence is not None else 0.0,
        right_identity=reference_evidence.right_anchor_identity if reference_evidence is not None else 0.0,
        insert_len=sequence_features.length,
        score=segmentation_score,
        qc=segmentation_qc,
    )

    pass_alignment, qc_reason = _te_alignment_qc(te_hit)
    if segmental is None:
        segmental = segmental_explanation_from_features(
            query_coverage=te_hit.query_coverage,
            low_complexity_fraction=sequence_features.low_complexity_fraction,
            pass_alignment=pass_alignment,
            has_partial_alignment=qc_reason == "TE_ALIGNMENT_LOW_COVERAGE",
        )
    sequence_model_label = _sequence_model_label(sequence_features)
    te_alignment = TeAlignmentEvidence(
        best_family=te_hit.family,
        best_subfamily=te_hit.subfamily,
        best_identity=_clamp(te_hit.identity, 0.0, 1.0),
        best_query_coverage=_clamp(te_hit.query_coverage, 0.0, 1.0),
        cross_family_margin=_clamp(segmental.family_margin, 0.0, 1.0),
        pass_alignment=pass_alignment,
        qc_reason=qc_reason,
        sequence_model_label=sequence_model_label,
        sequence_model_score=0.40 if sequence_model_label == "TE_MODEL_IN_DISTRIBUTION" else 0.0,
        annotation_confidence=(
            "HIGH"
            if pass_alignment and te_hit.query_coverage >= 0.75
            else "MEDIUM"
            if pass_alignment
            else "NA"
        ),
        annotation_residual_fraction=1.0 - _clamp(te_hit.query_coverage, 0.0, 1.0),
        orientation=te_hit.orientation,
    )

    if reference_evidence is not None:
        canonical_pass = reference_evidence.boundary_qc.startswith("PASS_BOUNDARY_")
        boundary = BoundaryEvidence(
            geometry_defined=canonical_pass,
            canonical_pass=canonical_pass,
            evidence_consistent=canonical_pass or reference_evidence.remap_qc != "NO_REFERENCE_EDGE_REMAP",
            boundary_type=reference_evidence.boundary_type,
            boundary_len=reference_evidence.boundary_len,
            score=reference_evidence.boundary_score,
            qc=reference_evidence.boundary_qc,
            reference_qc=reference_evidence.reference_qc,
            resolved_reference_chrom=reference_evidence.resolved_chrom,
            reference_breakpoint0=reference_evidence.breakpoint0,
            reference_right_breakpoint0=reference_evidence.right_breakpoint0,
            left_ref_anchor_len=reference_evidence.left_anchor_len,
            right_ref_anchor_len=reference_evidence.right_anchor_len,
            left_ref_anchor_identity=reference_evidence.left_anchor_identity,
            right_ref_anchor_identity=reference_evidence.right_anchor_identity,
            remap_qc=reference_evidence.remap_qc,
            insert_core_len=len(reference_evidence.insert_core_sequence or ""),
            tsd_type=reference_evidence.tsd_type,
            tsd_len=reference_evidence.tsd_len,
            tsd_sequence=reference_evidence.tsd_sequence,
            tsd_bg_p=reference_evidence.tsd_bg_p,
            tsd_qc=reference_evidence.tsd_qc,
            reference_n_fraction=reference_evidence.reference_n_fraction,
            reference_max_n_run=reference_evidence.reference_max_n_run,
        )
    else:
        boundary = BoundaryEvidence()

    return DiscriminatorEvidence(
        existence=existence,
        segmentation=segmentation,
        te_alignment=te_alignment,
        boundary=boundary,
        segmental=segmental,
    )


def _logsumexp3(a: float, b: float, c: float) -> float:
    top = max(a, b, c)
    return top + math.log(math.exp(a - top) + math.exp(b - top) + math.exp(c - top))


def _precise_structural_reads(existence: EventExistenceEvidence) -> int:
    return max(0, existence.alt_split_reads) + max(0, existence.alt_indel_reads)


def _bilateral_clip_reads(existence: EventExistenceEvidence) -> int:
    return min(max(0, existence.alt_left_clip_reads), max(0, existence.alt_right_clip_reads))


def _has_te_sequence_evidence(te_alignment: TeAlignmentEvidence) -> bool:
    return (
        te_alignment.pass_alignment
        or te_alignment.best_identity > 0.0
        or te_alignment.best_query_coverage > 0.0
        or te_alignment.qc_reason
        in {
            "PASS_INSERT_TE_ALIGNMENT",
            "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY",
            "PASS_INSERT_TE_ALIGNMENT_UNKNOWN",
            "TE_ALIGNMENT_LOW_COVERAGE",
        }
    )


def segmental_te_log_support(segmental: SegmentalExplanation) -> float:
    """Block log-likelihood ratio for the TE-insertion hypothesis from the
    segmental sequence explanation.

    The signal is graded in the contiguous, family-specific TE core rather than
    a constant identity, and it is discounted for low-complexity and poly(A)
    composition so simple-repeat inserts do not receive TE credit. A poly(A)
    tail contributes a modest retrotransposition-hallmark bonus only when a real
    TE core is also present.
    """
    core = _clamp(segmental.te_core_fraction, 0.0, 1.0)
    contiguity = _clamp(segmental.te_core_contiguity, 0.0, 1.0)
    margin = _clamp(segmental.family_margin, 0.0, 1.0)
    identity = _clamp(segmental.identity_proxy, 0.0, 1.0)
    low_complexity = _clamp(segmental.low_complexity_fraction, 0.0, 1.0)
    polya = _clamp(segmental.polya_fraction, 0.0, 1.0)
    unexplained = _clamp(segmental.unexplained_fraction, 0.0, 1.0)

    if core <= 0.0 and polya <= 0.0:
        return _clamp(-2.0 - 1.2 * low_complexity, -3.0, 3.0)

    # A short exact match is weak TE evidence: a ~40 bp k-mer coincidence is not
    # comparable to a ~300 bp TE body (cf. TLDR's ShortIns filter). Scale the core
    # credit by a soft ramp on the absolute TE-core length when it is known.
    if segmental.length > 0:
        core_len = core * segmental.length
        length_factor = _clamp(core_len / 120.0, 0.25, 1.0)
    else:
        length_factor = 1.0

    # Contiguous, family-specific coverage is the primary TE evidence.
    score = -1.2 + (4.0 * core * (0.5 + 0.5 * contiguity) * length_factor)
    score += 0.9 * margin
    score += 0.6 * (identity - 0.5)

    # Retrotransposition hallmark: a poly(A)/poly(T) tail alongside a TE core.
    if polya > 0.0 and core >= 0.15:
        score += min(0.6, 2.0 * polya)

    # Composition that is not TE-like: discourage, do not credit.
    score -= 1.8 * low_complexity
    score -= 0.6 * unexplained
    return _clamp(score, -3.0, 3.0)


# Backwards-compatible alias retained for callers that import the old name.
def te_alignment_log_support(te_alignment: TeAlignmentEvidence) -> float:
    return segmental_te_log_support(
        segmental_explanation_from_features(
            query_coverage=te_alignment.best_query_coverage,
            low_complexity_fraction=0.0,
            pass_alignment=te_alignment.pass_alignment,
            has_partial_alignment=te_alignment.qc_reason == "TE_ALIGNMENT_LOW_COVERAGE",
        )
    )


def evaluate_local_hypothesis_posterior(evidence: DiscriminatorEvidence) -> LocalHypothesisPosterior:
    existence = evidence.existence
    segmentation = evidence.segmentation
    boundary = evidence.boundary
    segmental = evidence.segmental
    alt = max(0, existence.alt_struct_reads)
    ref = max(0, existence.ref_span_reads)
    precise = _precise_structural_reads(existence)
    bilateral_clip = _bilateral_clip_reads(existence)

    count_balance = _clamp(math.log(alt + 1.0) - math.log(ref + 1.0), -2.0, 2.0)
    precise_balance = _clamp(math.log(precise + bilateral_clip + 1.0), 0.0, 2.0)
    genotype_support = _clamp((float(existence.gq) - 20.0) / 40.0, 0.0, 1.5)
    structural_support = _clamp(0.65 * count_balance + 0.45 * precise_balance + genotype_support, -2.0, 3.0)
    segmentation_support = _clamp(segmentation.score, -2.0, 2.0)
    sequence_support = segmental_te_log_support(segmental)

    low_complexity = _clamp(segmental.low_complexity_fraction, 0.0, 1.0)
    polya = _clamp(segmental.polya_fraction, 0.0, 1.0)
    te_core = _clamp(segmental.te_core_fraction, 0.0, 1.0)

    one_sided_or_open = not (segmentation.has_left_flank and segmentation.has_right_flank)
    closed = segmentation.pair_valid and segmentation.has_left_flank and segmentation.has_right_flank and boundary.geometry_defined
    boundary_support = 1.0 if closed else -0.35 if segmentation.has_insert_seq else -1.0

    artifact_context = 0.0
    if not segmentation.has_insert_seq:
        artifact_context += 1.6
    if one_sided_or_open:
        artifact_context += 0.8
    if not boundary.geometry_defined:
        artifact_context += 0.7
    if precise == 0 and bilateral_clip == 0:
        artifact_context += 0.5
    artifact_context += min(1.0, float(ref) * 0.08)
    # Low-complexity and lone poly(A) sequence is likely a mapping/repeat
    # artifact rather than a real TE insertion.
    artifact_context += 1.2 * low_complexity
    if polya >= 0.5 and te_core < 0.15:
        artifact_context += 0.6

    log_te = -0.25 + structural_support + (0.75 * segmentation_support) + sequence_support + boundary_support - (0.35 * artifact_context)
    log_non_te = 0.10 + (0.85 * structural_support) + (0.60 * segmentation_support) - (0.75 * sequence_support) + (0.25 if segmentation.has_insert_seq else -0.75)
    log_artifact = -0.10 + artifact_context - (0.25 * structural_support) - (0.15 * segmentation_support) - (0.35 * max(sequence_support, 0.0))

    denom = _logsumexp3(log_te, log_non_te, log_artifact)
    te_posterior = math.exp(log_te - denom)
    non_te_posterior = math.exp(log_non_te - denom)
    artifact_posterior = math.exp(log_artifact - denom)
    te_vs_artifact = log_te - log_artifact
    te_vs_non_te = log_te - log_non_te
    qc = (
        "PASS_TE_POSTERIOR"
        if te_posterior >= 0.55 and te_vs_artifact >= 0.5 and te_vs_non_te >= 0.0
        else "TE_POSTERIOR_LOW"
    )
    return LocalHypothesisPosterior(
        log_te=log_te,
        log_non_te=log_non_te,
        log_artifact=log_artifact,
        te=te_posterior,
        non_te=non_te_posterior,
        artifact=artifact_posterior,
        te_vs_artifact_log_odds=te_vs_artifact,
        te_vs_non_te_log_odds=te_vs_non_te,
        qc=qc,
    )


DEFAULT_TARGET_Q = 0.10


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _log_beta_binomial(k: int, n: int, alpha: float, beta: float) -> float:
    """Log-probability of ``k`` alt reads out of ``n`` under BetaBinomial priors."""
    if n <= 0:
        return 0.0
    k = max(0, min(k, n))
    log_choose = math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
    return (
        log_choose
        + _log_beta(k + alpha, n - k + beta)
        - _log_beta(alpha, beta)
    )


def _count_log_evidence(existence: EventExistenceEvidence) -> tuple[float, float]:
    """Beta-binomial log-evidence for an insertion being present vs a reference/
    artifact allele, from alt/ref read counts with overdispersion.

    Returns ``(present_log_evidence, artifact_log_evidence)`` on a common
    log-likelihood scale. Overdispersed priors are used because ONT and
    repetitive regions violate the independent-read binomial assumption.
    """
    alt = max(0, existence.alt_struct_reads)
    ref = max(0, existence.ref_span_reads)
    depth = alt + ref
    if depth <= 0:
        return -1.5, -0.5
    # Reference / mapping artifact: alt fraction near the error rate.
    artifact = _log_beta_binomial(alt, depth, alpha=1.0, beta=19.0)
    # Present insertion: mixture over heterozygous (~0.5) and homozygous (~0.9),
    # both overdispersed.
    het = _log_beta_binomial(alt, depth, alpha=8.0, beta=8.0)
    hom = _log_beta_binomial(alt, depth, alpha=18.0, beta=2.0)
    top = max(het, hom)
    present = top + math.log(0.5 * math.exp(het - top) + 0.5 * math.exp(hom - top))
    return present, artifact


@dataclass(frozen=True)
class RobustLocalFdr:
    te_log_evidence: float
    nonte_log_evidence: float
    artifact_log_evidence: float
    lfdr: float
    worst_case_lfdr: float
    target_q: float
    lfdr_qc: str


def _logsumexp2(a: float, b: float) -> float:
    top = max(a, b)
    return top + math.log(math.exp(a - top) + math.exp(b - top))


def evaluate_robust_local_fdr(
    evidence: DiscriminatorEvidence,
    target_q: float = DEFAULT_TARGET_Q,
) -> RobustLocalFdr:
    """Precision-first robust local false discovery rate.

    Follows ``.research/precision_first_lfdr_repo_implementation_notes.md``:
    build block log-evidence for ``H_TE``/``H_nonTE``/``H_artifact`` (beta-binomial
    counts + segmental sequence block + boundary/segmentation blocks), then form a
    conservative worst-case LFDR using interval priors and per-hypothesis
    likelihood uncertainty half-widths. A final TE call should require
    ``worst_case_lfdr <= target_q`` rather than a plug-in posterior threshold.
    """
    existence = evidence.existence
    segmentation = evidence.segmentation
    boundary = evidence.boundary
    segmental = evidence.segmental

    present_ev, artifact_ev = _count_log_evidence(existence)
    sequence_support = segmental_te_log_support(segmental)
    segmentation_support = _clamp(segmentation.score, -2.0, 2.0)
    closed = (
        segmentation.pair_valid
        and segmentation.has_left_flank
        and segmentation.has_right_flank
        and boundary.geometry_defined
    )
    boundary_support = 1.0 if closed else -0.35 if segmentation.has_insert_seq else -1.0

    low_complexity = _clamp(segmental.low_complexity_fraction, 0.0, 1.0)
    unexplained = _clamp(segmental.unexplained_fraction, 0.0, 1.0)
    family_margin = _clamp(segmental.family_margin, 0.0, 1.0)
    one_sided = not (segmentation.has_left_flank and segmentation.has_right_flank)

    # Block log-evidence per hypothesis (unnormalised log-likelihoods).
    ell_te = present_ev + (0.75 * segmentation_support) + sequence_support + boundary_support
    ell_non = (
        present_ev
        + (0.60 * segmentation_support)
        - (0.75 * sequence_support)
        + (0.25 if segmentation.has_insert_seq else -0.75)
    )
    artifact_context = 1.2 * low_complexity
    if segmental.polya_fraction >= 0.5 and segmental.te_core_fraction < 0.15:
        artifact_context += 0.6
    if one_sided:
        artifact_context += 0.4
    if not boundary.geometry_defined:
        artifact_context += 0.3
    ell_art = artifact_ev + artifact_context - (0.35 * max(sequence_support, 0.0))

    # Point posterior LFDR (plug-in).
    denom = _logsumexp3(ell_te, ell_non, ell_art)
    math.exp(ell_te - denom)
    post_non = math.exp(ell_non - denom)
    post_art = math.exp(ell_art - denom)
    lfdr = _clamp(post_non + post_art, 0.0, 1.0)

    # Likelihood uncertainty half-widths (model misspecification).
    delta = 0.10 + (0.35 * unexplained) + (0.30 * low_complexity)
    if one_sided:
        delta += 0.10
    if not boundary.geometry_defined:
        delta += 0.10
    delta_te = delta + 0.15 * (1.0 - family_margin)

    # Interval priors (precision-first: pessimistic on TE, but not so pessimistic
    # that a strong imprecise/one-sided TE can never clear the target risk --
    # imprecise TEs are the majority class in the reference truth set).
    log_pi_te_lo = math.log(0.25)
    log_pi_non_hi = math.log(0.45)
    log_pi_art_hi = math.log(0.45)

    log_bad_wc = _logsumexp2(
        log_pi_non_hi + ell_non + delta,
        log_pi_art_hi + ell_art + delta,
    )
    log_denom_wc = _logsumexp2(log_pi_te_lo + ell_te - delta_te, log_bad_wc)
    worst_case_lfdr = _clamp(math.exp(log_bad_wc - log_denom_wc), 0.0, 1.0)

    if worst_case_lfdr <= target_q:
        lfdr_qc = "PASS_TE_LFDR"
    elif lfdr <= target_q:
        lfdr_qc = "TE_LFDR_NOT_ROBUST"
    else:
        lfdr_qc = "TE_LFDR_REJECT"
    return RobustLocalFdr(
        te_log_evidence=ell_te,
        nonte_log_evidence=ell_non,
        artifact_log_evidence=ell_art,
        lfdr=lfdr,
        worst_case_lfdr=worst_case_lfdr,
        target_q=target_q,
        lfdr_qc=lfdr_qc,
    )


def evaluate_joint_hypotheses(
    evidence: DiscriminatorEvidence,
    target_q: float = DEFAULT_TARGET_Q,
    hp_min: float = DEFAULT_HP_MIN,
    hp_cov_min: float = DEFAULT_HP_COV_MIN,
) -> JointDecisionResult:
    posterior = evaluate_local_hypothesis_posterior(evidence)
    robust = evaluate_robust_local_fdr(evidence, target_q=target_q)
    existence = evidence.existence
    segmentation = evidence.segmentation
    te_alignment = evidence.te_alignment
    boundary = evidence.boundary

    segmental = evidence.segmental
    has_te = _has_te_sequence_evidence(te_alignment)
    has_structural = existence.alt_struct_reads > 0 and segmentation.has_insert_seq
    # A resolved TE requires a real TE core, not merely low-complexity or a lone
    # poly(A) tail that happens to share k-mers with the repeat library.
    seq_te_like = (
        segmental.te_core_fraction >= 0.15
        or segmental.qc in {"TE_CORE_DOMINANT", "TE_CORE_PARTIAL", "FALLBACK_FROM_TEHIT"}
    )
    lfdr = robust.lfdr
    robust_lfdr = robust.worst_case_lfdr

    # High-precision mechanistic tier: a family-resolved TE body with enough
    # correctly-aligned bases (biophysical alignment-likelihood statistic),
    # calibrated for >=90% concordance with the reference TE caller.
    family_resolved = bool(segmental.best_family) and segmental.best_family not in {"", "UNKNOWN", "Unknown"}
    high_precision_score = high_precision_te_score(
        segmental.identity_proxy, segmental.te_core_fraction, segmental.length
    )
    emit_te_high_precision = is_confident_te_insertion(
        high_precision_score, segmental.query_coverage, family_resolved, hp_min, hp_cov_min
    )

    # Precision-first emission: require the conservative worst-case LFDR to clear
    # the target risk, not merely a plug-in posterior threshold.
    emit_te = (
        emit_te_high_precision
        or (
            has_te
            and segmentation.has_insert_seq
            and seq_te_like
            and robust.worst_case_lfdr <= target_q
        )
    )
    emit_evidence_te = has_te and segmentation.has_insert_seq
    emit_unknown = emit_te and (
        te_alignment.best_family in {"", "UNKNOWN"}
        or te_alignment.best_subfamily in {"", "UNKNOWN"}
    )

    if emit_te:
        best_hypothesis = "TE_UNKNOWN" if emit_unknown else "TE_RESOLVED"
        if emit_te_high_precision:
            final_qc = "PASS_TE_HIGH_PRECISION"
        elif boundary.geometry_defined and boundary.canonical_pass:
            final_qc = "PASS_TE_CLOSED"
        else:
            final_qc = "PASS_TE_IMPRECISE"
        label = "TE"
        latent = "degraded_unknown_te" if emit_unknown else "cut_paste_te"
    elif posterior.artifact >= posterior.non_te and posterior.artifact >= posterior.te:
        best_hypothesis = "REFERENCE"
        final_qc = "REFERENCE_OR_ARTIFACT"
        label = "artifact"
        latent = "artifact_reference"
    elif posterior.non_te >= posterior.artifact and has_structural:
        best_hypothesis = "INSERTION_NON_TE"
        final_qc = "PASS_STRUCTURAL_INSERTION"
        label = "nonTE"
        latent = "non_te_insert"
    elif emit_evidence_te:
        best_hypothesis = "TE_UNKNOWN" if te_alignment.best_family in {"", "UNKNOWN"} else "TE_RESOLVED"
        final_qc = "TE_AMBIGUOUS"
        label = "TE"
        latent = "degraded_unknown_te"
    else:
        best_hypothesis = "REFERENCE"
        final_qc = "REFERENCE_OR_ARTIFACT"
        label = "artifact"
        latent = "artifact_reference"

    return JointDecisionResult(
        label=label,
        best_hypothesis=best_hypothesis,
        final_qc=final_qc,
        posterior_qc=posterior.qc,
        te_posterior=posterior.te,
        non_te_posterior=posterior.non_te,
        artifact_posterior=posterior.artifact,
        te_vs_artifact_log_odds=posterior.te_vs_artifact_log_odds,
        te_vs_non_te_log_odds=posterior.te_vs_non_te_log_odds,
        emit_te_call=emit_te,
        emit_unknown_te=emit_unknown,
        emit_evidence_te_call=emit_evidence_te,
        latent_mechanism=latent,
        lfdr=lfdr,
        robust_mechanistic_lfdr=robust_lfdr,
        te_alignment_qc=te_alignment.qc_reason,
        sequence_model_label=te_alignment.sequence_model_label,
        boundary_qc=boundary.qc,
        segmentation_qc=segmentation.qc,
        reference_qc=boundary.reference_qc,
        resolved_reference_chrom=boundary.resolved_reference_chrom,
        reference_breakpoint0=boundary.reference_breakpoint0,
        reference_right_breakpoint0=boundary.reference_right_breakpoint0,
        left_ref_anchor_len=boundary.left_ref_anchor_len,
        right_ref_anchor_len=boundary.right_ref_anchor_len,
        left_ref_anchor_identity=boundary.left_ref_anchor_identity,
        right_ref_anchor_identity=boundary.right_ref_anchor_identity,
        remap_qc=boundary.remap_qc,
        insert_core_len=boundary.insert_core_len,
        tsd_type=boundary.tsd_type,
        tsd_len=boundary.tsd_len,
        tsd_sequence=boundary.tsd_sequence,
        tsd_bg_p=boundary.tsd_bg_p,
        tsd_qc=boundary.tsd_qc,
        reference_n_fraction=boundary.reference_n_fraction,
        reference_max_n_run=boundary.reference_max_n_run,
        seq_core_fraction=evidence.segmental.te_core_fraction,
        seq_core_contiguity=evidence.segmental.te_core_contiguity,
        seq_polya_fraction=evidence.segmental.polya_fraction,
        seq_low_complexity_fraction=evidence.segmental.low_complexity_fraction,
        seq_unexplained_fraction=evidence.segmental.unexplained_fraction,
        seq_family_margin=evidence.segmental.family_margin,
        seq_explanation_qc=evidence.segmental.qc,
        worst_case_lfdr=robust.worst_case_lfdr,
        target_q=robust.target_q,
        te_log_evidence=robust.te_log_evidence,
        nonte_log_evidence=robust.nonte_log_evidence,
        artifact_log_evidence=robust.artifact_log_evidence,
        lfdr_qc=robust.lfdr_qc,
        high_precision_score=high_precision_score,
        emit_te_high_precision=emit_te_high_precision,
    )
