"""
The joint decision: four hypotheses, one emission gate, and the diagnostics.

Ported from the second half of `src/pipeline/decision_policy.cpp` -- the half
`placer_py/genotype.py` left behind -- and pinned by
`tests/test_24_policy.py`.

WHAT THIS MODULE DECIDES, and what it explicitly does not.

`evaluate_joint_hypotheses` RANKS four hypotheses (reference, non-TE insertion,
TE-unknown, TE-resolved) on a weighted sum of four supports. That ranking is
DIAGNOSTIC: it picks which hypothesis to describe the locus with, and it is the
key the breakpoint stage uses to choose between competing candidates. It does
not gate emission.

Emission is gated by ONE thing: the robust worst-case local FDR against a single
target risk `q`. The C++ header says so at length, and the history is visible in
`include/decision_thresholds.h`, which is mostly a list of constants that were
DELETED -- read-count floors, GQ floors, insert-length floors, segmentation-score
floors. They were hand-drawn decision surfaces fitted one site at a time, and
replacing them with a single risk level is the same move the selection layer
makes at genome scale.

Three things survive as real gates, and each is definitional rather than tuned:

  * the two HARD VETOES -- a hypothesis is ineligible when the event cannot
    definitionally be a TE call (no insert sequence, no TE alignment at all,
    annotation too poor to interpret). These are necessary conditions, not
    evidence thresholds.
  * `structural_sanity` -- there must be an insert sequence and some TE sequence
    evidence.
  * the structural-insertion path, which is a separate log-evidence gate for
    "there is a real insertion here" when the TE identity is unresolved.

WHAT ABSTENTION MEANS HERE. A TE-like event that fails the lFDR gate is NOT
reclassified as reference or artifact. It becomes `TE_AMBIGUOUS` and is retained
as TE-evidence. That is the precision-first design: the caller declines to
decide rather than deciding wrongly, and the locus stays visible to whoever
wants to look.

ONE INCONSISTENCY, PINNED. `is_one_sided_segmentation_pass` is defined twice in
the C++, in two translation units, with OPPOSITE polarity on `pair_valid`:
`mechanistic_evidence.cpp` requires `!pair_valid`, this file requires
`pair_valid`. Both are reachable for the same locus, from the same evidence, in
the same run. The port keeps both under distinct names rather than silently
picking one -- see `is_one_sided_segmentation_pass` here and
`_is_one_sided_segmentation_pass` in `placer_py/blocks.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from placer_py import blocks as blocks_module
from placer_py import mathx
from placer_py import structure as structure_module
from placer_py.explanation import (
    EventExplanation,
    ExplanationDecision,
    ExplanationKind,
    compare_event_explanations,
    explanation_kind_name,
    serialize_explanation_path,
    serialize_residual,
)
from placer_py.genotype import genotype_from_alt_vs_ref
from placer_py.structure import SequenceExplanation

# --------------------------------------------------------------------------
# DecisionThresholds. What is left after the ladders were deleted.
# --------------------------------------------------------------------------
#: The single emission knob: a target false-call risk. A strict mode lowers it.
TARGET_Q = 0.10
#: Artifact-posterior guards. Probabilities, not feature ladders.
ARTIFACT_POSTERIOR_CAP = 0.65
ARTIFACT_POSTERIOR_STRICT_CAP = 0.35
#: A Bayes-factor scale point -- log e ~ 2 is "decisive" on the Kass-Raftery
#: scale -- not a tuned read count.
STRUCTURAL_LOG_EVIDENCE = 2.0
#: Weights that RANK the four hypotheses. Diagnostic; they gate nothing.
SCORE_REF_EXISTENCE_WEIGHT = -1.0
SCORE_REF_SEGMENTATION_WEIGHT = -0.5
SCORE_NON_TE_SEGMENTATION_WEIGHT = 0.7
SCORE_NON_TE_MODEL_WEIGHT = -0.4
SCORE_TE_SEGMENTATION_WEIGHT = 0.8
SCORE_TE_BOUNDARY_WEIGHT = 0.2

#: The six latent mechanisms. Four are TE; the other two are the null.
LATENT_MECHANISM_STATES = ("active_tprt_te", "cut_paste_te", "ltr_complex_te",
                           "degraded_unknown_te", "non_te_insert",
                           "artifact_reference")

_NEG_INF = -1e9
_LOG_ZERO = -1e300


class FinalHypothesisKind(IntEnum):
    REFERENCE = 0
    INSERTION_NON_TE = 1
    TE_UNKNOWN = 2
    TE_RESOLVED = 3


# --------------------------------------------------------------------------
# TE alignment QC, as a typed value.
# --------------------------------------------------------------------------
class TeAlignmentQc(IntEnum):
    OTHER = 0
    NO_TE_ALIGNMENT = 1
    PASS_INSERT = 2
    PASS_INSERT_FAMILY_ONLY = 3
    PASS_INSERT_UNKNOWN = 4
    LOW_IDENTITY = 5


_QC_BY_REASON = {
    "NO_TE_ALIGNMENT": TeAlignmentQc.NO_TE_ALIGNMENT,
    "PASS_INSERT_TE_ALIGNMENT": TeAlignmentQc.PASS_INSERT,
    "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY": TeAlignmentQc.PASS_INSERT_FAMILY_ONLY,
    "PASS_INSERT_TE_ALIGNMENT_UNKNOWN": TeAlignmentQc.PASS_INSERT_UNKNOWN,
    "TE_ALIGNMENT_LOW_IDENTITY": TeAlignmentQc.LOW_IDENTITY,
}


def classify_te_alignment_qc(qc_reason: str) -> TeAlignmentQc:
    """Typed classification of the QC string.

    The STRING stays the source of truth -- it is serialised into the output and
    its spelling is part of the contract. This exists so the gating logic
    branches on a value instead of on string literals, where a typo is a silently
    false comparison rather than an error.
    """
    return _QC_BY_REASON.get(qc_reason, TeAlignmentQc.OTHER)


def is_pass_insert_te_alignment(qc: TeAlignmentQc) -> bool:
    return qc in (TeAlignmentQc.PASS_INSERT, TeAlignmentQc.PASS_INSERT_FAMILY_ONLY,
                  TeAlignmentQc.PASS_INSERT_UNKNOWN)


def is_resolved_pass_insert_te_alignment(qc: TeAlignmentQc) -> bool:
    return qc in (TeAlignmentQc.PASS_INSERT, TeAlignmentQc.PASS_INSERT_FAMILY_ONLY)


# --------------------------------------------------------------------------
# The evidence records the decision reads.
# --------------------------------------------------------------------------
@dataclass
class EventGenotypeInput:
    alt_struct_reads: int = 0
    alt_split_reads: int = -1
    alt_indel_reads: int = -1
    alt_left_clip_reads: int = -1
    alt_right_clip_reads: int = -1
    ref_span_reads: int = 0
    min_gq: int = 20
    error_rate: float = 0.02
    #: Intra-class correlation of the counts. rho -> 0 is the independent-read
    #: binomial limit; larger rho models the overdispersion that mapping bias
    #: and local alignment ambiguity produce in repeats.
    overdispersion: float = 0.02
    event_length: int = 0
    alt_observed_lengths: list[int] = field(default_factory=list)


@dataclass
class EventExistenceEvidence:
    best_gt: str = "./."
    af: float = 0.0
    gq: int = 0
    alt_struct_reads: int = 0
    #: -1 means "not broken down", which is NOT the same as 0. See
    #: `has_structural_breakdown`.
    alt_split_reads: int = -1
    alt_indel_reads: int = -1
    alt_left_clip_reads: int = -1
    alt_right_clip_reads: int = -1
    ref_span_reads: int = 0
    depth: int = 0
    best_nonref_minus_ref_ll: float = 0.0
    score: float = -3.0


@dataclass
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
    score: float = -3.0
    qc: str = "NO_EVENT_SEGMENTATION"


@dataclass
class BoundaryEvidence:
    geometry_defined: bool = False
    canonical_pass: bool = False
    evidence_consistent: bool = False
    boundary_type: str = "REJECT"
    boundary_len: int = 0
    score: float = -3.0
    qc: str = "REJECT_BOUNDARY_UNSET"


@dataclass
class ClipInsertConcordanceEvidence:
    pass_: bool = False
    full_insert_reads: int = 0
    left_clip_reads: int = 0
    right_clip_reads: int = 0
    max_left_identity: float = 0.0
    max_right_identity: float = 0.0
    qc: str = "NO_CLIP_INSERT_CONCORDANCE"


@dataclass
class FinalBoundaryInput:
    left_ref_start: int = -1
    left_ref_end: int = -1
    right_ref_start: int = -1
    right_ref_end: int = -1
    tsd_min_len: int = 3
    tsd_max_len: int = 50


@dataclass
class FinalBoundaryDecision:
    pass_: bool = False
    boundary_type: str = "REJECT"
    boundary_len: int = 0
    qc: str = "REJECT_BOUNDARY_UNSET"


@dataclass
class JointHypothesisScore:
    kind: FinalHypothesisKind = FinalHypothesisKind.REFERENCE
    total: float = _NEG_INF
    existence: float = 0.0
    segmentation: float = 0.0
    te: float = 0.0
    boundary: float = 0.0
    hard_veto: bool = False
    reason: str = ""


@dataclass
class LocalHypothesisPosterior:
    log_te: float = _NEG_INF
    log_non_te: float = _NEG_INF
    log_artifact: float = _NEG_INF
    te: float = 0.0
    non_te: float = 0.0
    artifact: float = 0.0
    te_vs_artifact_log_odds: float = 0.0
    te_vs_non_te_log_odds: float = 0.0
    qc: str = "TE_POSTERIOR_LOW"


@dataclass
class LatentMechanismEvidence:
    latent_mechanism: str = "artifact_reference"
    family_activity_prior: float = 0.0
    te_posterior: float = 0.0
    non_te_posterior: float = 0.0
    artifact_posterior: float = 0.0
    lfdr: float = 1.0
    worst_case_lfdr: float = 1.0
    lfdr_qc: str = "TE_LFDR_HIGH"


@dataclass
class LatentFeatureVector:
    te_identity: float = 0.0
    event_signal: float = 0.0
    independent_signal: float = 0.0
    quality_signal: float = 0.0
    ref_conflict_signal: float = 0.0
    artifact_context_signal: float = 0.0
    family_activity_prior: float = 0.0
    family_kind: str = "unknown"


@dataclass
class JointDecisionResult:
    best: JointHypothesisScore = field(default_factory=JointHypothesisScore)
    runner_up: JointHypothesisScore = field(default_factory=JointHypothesisScore)
    emit_te_call: bool = False
    emit_structural_event_call: bool = False
    emit_unknown_te: bool = False
    emit_evidence_te_call: bool = False
    final_qc: str = "REJECT_EVENT_EXISTENCE"
    te_posterior: float = 0.0
    non_te_posterior: float = 0.0
    artifact_posterior: float = 0.0
    te_vs_artifact_log_odds: float = 0.0
    te_vs_non_te_log_odds: float = 0.0
    posterior_qc: str = "POSTERIOR_NOT_EVALUATED"
    latent_mechanism: str = "NA"
    family_activity_prior: float = 0.0
    lfdr: float = 1.0
    worst_case_lfdr: float = 1.0
    lfdr_qc: str = "LFDR_NOT_EVALUATED"
    mechanistic_lower_log_bf_te_vs_artifact: float = 0.0
    #: Penalty-FREE aggregates. Finalization needs them because the bound that
    #: makes the product an e-value is a property of the whole run, not of one
    #: locus -- see placer_py/dependency.py.
    mechanistic_raw_log_bf_te_vs_artifact: float = 0.0
    mechanistic_raw_log_bf_te_vs_non_te: float = 0.0
    mechanistic_lower_log_bf_te_vs_non_te: float = 0.0
    mechanistic_ref_conflict_signal: float = 0.0
    mechanistic_ambiguity_width: float = 0.0
    mechanistic_blocks: str = "NA"
    robust_mechanistic_lfdr: float = 1.0
    robust_mechanistic_worst_case_lfdr: float = 1.0
    robust_mechanistic_qc: str = "TE_LFDR_HIGH"
    explanation_decision: ExplanationDecision = field(
        default_factory=ExplanationDecision)
    best_explanation: str = "NA"
    explanation_residual: str = "NA"
    explanation_path: str = "NA"
    te_structure_path: str = "NA"
    te_structure_log_evidence: float = 0.0
    nonte_structure_log_evidence: float = 0.0
    artifact_structure_log_evidence: float = 0.0
    te_structure_path_confidence: float = 0.0
    polyA_posterior: float = 0.0
    transduction_posterior: float = 0.0
    te_core_coverage: float = 0.0
    unexplained_high_complexity_bp: int = 0


# --------------------------------------------------------------------------
# Small numeric helpers.
# --------------------------------------------------------------------------
def clamp_score(value: float, lo: float = -3.0, hi: float = 3.0) -> float:
    """`mathx.clamp` with the policy layer's default evidence bounds.

    Kept as a wrapper rather than an alias because the +/-3.0 defaults are
    used at most of its 29 call sites and are part of what the name means
    here -- a score bound, not a general clamp.
    """
    return mathx.clamp(value, lo, hi)


_clamp01 = mathx.clamp01


def count_signal(count: int, scale: float) -> float:
    """Saturating count evidence: `1 - exp(-n/scale)`.

    Diminishing returns by construction. The tenth supporting read adds far less
    than the second, which is what stops a high-coverage locus from out-scoring a
    well-supported one purely on depth.
    """
    if count <= 0:
        return 0.0
    return _clamp01(1.0 - math.exp(-count / max(1e-6, scale)))


def beta_feature_loglike(x: float, mean: float, strength: float) -> float:
    """Beta-shaped log-likelihood of a feature in [0,1] under one state.

    Not a normalised Beta density -- the normalising constant is dropped because
    every state is evaluated at the same `x`, so it cancels in the posterior.
    """
    eps = 1e-6
    bounded_x = clamp_score(x, eps, 1.0 - eps)
    bounded_mean = clamp_score(mean, eps, 1.0 - eps)
    return strength * ((bounded_mean * math.log(bounded_x))
                       + ((1.0 - bounded_mean) * math.log(1.0 - bounded_x)))


#: PROPAGATING form. In the policy layer `-inf` is an impossible hypothesis,
#: not a line of evidence that abstained, so it must reach the normaliser --
#: the opposite of `finalization.log_sum_exp_pair`. Both semantics are wanted
#: and `mathx.log_sum_exp` keeps them one function with an explicit flag, so
#: a caller cannot get the other one by accident.
logsumexp_values = mathx.log_sum_exp


def logsumexp_pair(lhs: float, rhs: float) -> float:
    return mathx.log_sum_exp((lhs, rhs))


def logsumexp3(a: float, b: float, c: float) -> float:
    return mathx.log_sum_exp((a, b, c))


#: Re-exported: the suite names these directly and the C++ has them in this
#: translation unit. Bodies in `placer_py/mathx.py`.
log_choose_count = mathx.log_choose
beta_binomial_log_pmf = mathx.beta_binomial_log_pmf
binomial_log_pmf = mathx.binomial_log_pmf

#: Prior on the alt fraction of a real low-AF insertion: Beta(1, 9), i.e. a
#: mean of 0.1. Deliberately generous to mosaics and to a het whose alt reads
#: were lost to mapping bias.
ALT_FRACTION_PRIOR_ALPHA = 1.0
ALT_FRACTION_PRIOR_BETA = 9.0
#: The error rate the artifact alternative is scored at.
ARTIFACT_ERROR_RATE = 0.02


def low_allele_fraction_insertion_log_bf(alt: int, ref: int) -> float:
    """Real minority allele vs. a run of errors, as a log Bayes factor.

    THE POINT of having this alongside the dominant-allele odds: a true
    heterozygous or mosaic insertion has `alt < ref`, and a rule that asked for
    alt to dominate would discard exactly those. This asks a different question
    -- are these counts better explained by a real low-frequency allele than by
    a 2% error process -- and the structural gate takes the MAX of the two, so a
    locus only has to satisfy one of them.
    """
    total = max(0, alt) + max(0, ref)
    if alt <= 0 or total <= 0:
        return _LOG_ZERO
    return (beta_binomial_log_pmf(alt, total, ALT_FRACTION_PRIOR_ALPHA,
                                  ALT_FRACTION_PRIOR_BETA)
            - binomial_log_pmf(alt, total, ARTIFACT_ERROR_RATE))


def logistic_logit(x: float) -> float:
    eps = 1e-6
    bounded = clamp_score(x, eps, 1.0 - eps)
    return math.log(bounded / (1.0 - bounded))


# --------------------------------------------------------------------------
# Read-count shapes.
# --------------------------------------------------------------------------
def positive_or_zero(value: int) -> int:
    return max(0, value)


def joint_event_existence_score(existence: EventExistenceEvidence) -> float:
    """The existence score, floored at 0 when nothing opposes the insertion.

    `ref_span_reads == 0` with three or more alt reads means the reference
    allele was never observed. The genotyper cannot be confident there (no
    depth to be confident WITH), so its score can be negative -- and a negative
    existence score for an unopposed insertion is an artefact of the model, not
    evidence against the event.
    """
    unopposed_alt_min_reads = 3
    if existence.ref_span_reads == 0 and existence.alt_struct_reads >= unopposed_alt_min_reads:
        return max(0.0, existence.score)
    return existence.score


def is_one_sided_segmentation_pass(segmentation: EventSegmentationEvidence) -> bool:
    """A valid pair with exactly ONE anchored flank.

    NOTE the polarity: this requires `pair_valid`, while the identically-named
    function in `mechanistic_evidence.cpp` (ported as
    `placer_py.blocks._is_one_sided_segmentation_pass`) requires `NOT
    pair_valid`. Both run on the same locus in the same pass. The port keeps
    both rather than choosing, because choosing would change behaviour under
    cover of a cleanup.
    """
    return (segmentation.pair_valid and segmentation.has_insert_seq
            and (segmentation.has_left_flank != segmentation.has_right_flank))


def has_structural_breakdown(existence: EventExistenceEvidence) -> bool:
    """Is the alt count broken down by evidence kind at all?

    -1 means "not broken down" and 0 means "broken down, and none of this
    kind". Conflating them would make an un-broken-down event look like one with
    no precise reads, which is the difference between "unknown" and "bad".
    """
    return (existence.alt_split_reads >= 0 or existence.alt_indel_reads >= 0
            or existence.alt_left_clip_reads >= 0
            or existence.alt_right_clip_reads >= 0)


def precise_structural_reads(existence: EventExistenceEvidence) -> int:
    """Split + indel reads: the ones that place the breakpoint exactly."""
    if not has_structural_breakdown(existence):
        return existence.alt_struct_reads
    return max(0, existence.alt_split_reads) + max(0, existence.alt_indel_reads)


def bilateral_clip_support_reads(existence: EventExistenceEvidence) -> int:
    """`min(left, right)` -- the MINIMUM, and that is the whole idea.

    Clips on one side only are what a mapping artifact produces. Requiring both
    sides and counting the smaller means ten left clips and no right clips
    contribute nothing, which is correct: they are consistent with a single
    misaligned edge and inconsistent with an insertion.
    """
    if not has_structural_breakdown(existence):
        return existence.alt_struct_reads
    return min(max(0, existence.alt_left_clip_reads),
               max(0, existence.alt_right_clip_reads))


# --------------------------------------------------------------------------
# Adjusted supports.
# --------------------------------------------------------------------------
#: Annotation-quality ceilings, by confidence tier. A poorly-named alignment is
#: allowed to support a TE decision only while it also explains most of the
#: insert -- residual is what it FAILED to explain.
LOW_ANNOTATION_MAX_RESIDUAL = 0.50
LOW_ANNOTATION_MAX_MASKED = 0.65
MEDIUM_ANNOTATION_MAX_RESIDUAL = 0.65
MEDIUM_ANNOTATION_MAX_MASKED = 0.80


def annotation_quality_allows_te_decision(te_alignment) -> bool:
    """May an alignment of this confidence be used to decide TE at all?

    HIGH confidence is unconditional. LOW and MEDIUM must additionally explain
    enough of the insert -- because a vaguely-named alignment covering a third of
    the sequence is compatible with almost anything, and letting it through would
    make the naming tier decide the call after all.
    """
    if te_alignment.annotation_confidence == "LOW":
        return (te_alignment.annotation_residual_fraction <= LOW_ANNOTATION_MAX_RESIDUAL
                and te_alignment.annotation_masked_fraction <= LOW_ANNOTATION_MAX_MASKED)
    if te_alignment.annotation_confidence == "MEDIUM":
        return (te_alignment.annotation_residual_fraction <= MEDIUM_ANNOTATION_MAX_RESIDUAL
                and te_alignment.annotation_masked_fraction <= MEDIUM_ANNOTATION_MAX_MASKED)
    return True


def adjusted_segmentation_score(segmentation: EventSegmentationEvidence) -> float:
    """Floor a one-sided pass at 0.25 rather than letting it go negative.

    A one-sided event is weak, not wrong: only one flank anchored, which is
    exactly what a long insertion produces when the read runs out before the
    far side. The floor keeps it in the ranking; the artifact context signal is
    where it pays for the missing flank.
    """
    if is_one_sided_segmentation_pass(segmentation):
        return max(0.25, segmentation.score)
    return segmentation.score


def adjusted_nonref_existence_score(existence: EventExistenceEvidence,
                                    segmentation: EventSegmentationEvidence,
                                    te_model_support: float) -> float:
    """Floor the existence score at 0 for a strongly TE-like, alt-dominant event.

    Both conditions are needed. Strong sequence support alone would let a
    well-matched but poorly-supported locus float up; alt dominance alone would
    do it for any high-coverage noise.
    """
    adjusted = joint_event_existence_score(existence)
    has_strong_structural_te_support = (segmentation.pair_valid
                                        and segmentation.has_insert_seq
                                        and te_model_support >= 0.75)
    alt_not_weaker_than_ref = (existence.alt_struct_reads > 0
                               and existence.alt_struct_reads >= existence.ref_span_reads)
    if has_strong_structural_te_support and alt_not_weaker_than_ref:
        adjusted = max(0.0, adjusted)
    return adjusted


def adjusted_te_boundary_score(segmentation: EventSegmentationEvidence,
                               boundary: BoundaryEvidence) -> float:
    """A one-sided pass with no boundary geometry scores 0, not -3.

    There IS no second flank to form a junction with, so the geometry is
    undefined rather than bad, and charging the full boundary penalty would
    punish the event twice for the same missing flank.
    """
    if not boundary.geometry_defined and is_one_sided_segmentation_pass(segmentation):
        return 0.0
    return boundary.score


def te_sequence_model_score(te_alignment) -> float:
    """The composition model's contribution to the hypothesis RANKING.

    `TE_MODEL_UNAVAILABLE` and `TE_MODEL_EDGE` both score 0 -- "we could not
    look" and "it is on the boundary" are equally uninformative for ranking,
    even though they differ in the artifact context signal.
    """
    label = te_alignment.sequence_model_label
    if label == "TE_MODEL_IN_DISTRIBUTION":
        return 0.75 + clamp_score(te_alignment.sequence_model_score, 0.0, 0.50)
    if label == "TE_MODEL_OUTLIER":
        return clamp_score(te_alignment.sequence_model_score, -1.0, -0.50)
    return 0.0


# --------------------------------------------------------------------------
# Family kind: four mechanisms, from the name.
# --------------------------------------------------------------------------
_LTR_TOKENS = ("ltr", "erv", "gypsy", "copia", "bel-pao", "dirs")
_DNA_TOKENS = ("dna", "hat", "pif", "harbinger", "tc1", "mariner", "piggybac",
               "mutator", "merlin", "cmc", "dong")
_RETRO_TOKENS = ("l1", "l2", "line", "sine", "alu", "sva", "rte", "r2", "cr1",
                 "penelope", "5s-deu-l2")


def family_kind(te_alignment) -> str:
    """Map a family/subfamily name onto one of four MECHANISMS.

    The point is mechanism, not taxonomy: `active_tprt_te` is compatible with a
    retro family and incompatible with a DNA transposon because target-primed
    reverse transcription is something only the former does. Matching on tokens
    inside `family:subfamily` is crude, and the ORDER matters -- `ltr` is tested
    first because "HERVK-int#LTR/ERVK" contains no retro token but "L1" would
    match inside some LTR subfamily names.
    """
    family = (te_alignment.best_family or "").lower()
    subfamily = (te_alignment.best_subfamily or "").lower()
    name = f"{family}:{subfamily}"
    unknown_labels = ("", "unknown", "na", "none")
    if family in unknown_labels and subfamily in unknown_labels:
        return "unknown"
    if any(token in name for token in _LTR_TOKENS):
        return "ltr"
    if any(token in name for token in _DNA_TOKENS):
        return "dna"
    if any(token in name for token in _RETRO_TOKENS):
        return "retro"
    return "other"


def build_latent_feature_vector(existence: EventExistenceEvidence,
                                segmentation: EventSegmentationEvidence,
                                te_alignment,
                                boundary: BoundaryEvidence,
                                clip_insert_concordance: ClipInsertConcordanceEvidence | None
                                ) -> LatentFeatureVector:
    """Six features in [0,1], plus the family kind and the activity prior.

    `independent_signal` is the geometric mean of a COUNT term and a FRACTION
    term, and the choice of geometric mean is doing real work: it is near zero
    unless BOTH are high. Twenty supporting reads of which two are precise, and
    two reads both of which are precise, are each unconvincing for different
    reasons, and an arithmetic mean would rate both as moderate.
    """
    features = LatentFeatureVector()
    alt = max(0, existence.alt_struct_reads)
    ref = max(0, existence.ref_span_reads)
    precise = precise_structural_reads(existence)
    bilateral_clip = bilateral_clip_support_reads(existence)
    mechanistic_reads = precise + bilateral_clip
    if clip_insert_concordance is not None and clip_insert_concordance.pass_:
        mechanistic_reads += max(0, clip_insert_concordance.full_insert_reads)
        mechanistic_reads += min(max(0, clip_insert_concordance.left_clip_reads),
                                 max(0, clip_insert_concordance.right_clip_reads))

    features.te_identity = _clamp01(te_alignment.best_identity)
    independent_count_signal = count_signal(mechanistic_reads, 4.0)
    independent_fraction_signal = _clamp01(mechanistic_reads / alt) if alt > 0 else 0.0
    features.independent_signal = math.sqrt(independent_count_signal
                                            * independent_fraction_signal)
    features.quality_signal = _clamp01(existence.gq / 60.0)
    support_signal = count_signal(alt, 8.0)
    segmentation_signal = _clamp01((adjusted_segmentation_score(segmentation) + 1.0) / 3.0)
    features.event_signal = _clamp01((0.35 * support_signal)
                                     + (0.30 * features.independent_signal)
                                     + (0.20 * segmentation_signal)
                                     + (0.15 * _clamp01(existence.af)))

    ref_fraction = (ref / (alt + ref)) if (alt + ref) > 0 else 0.0
    features.ref_conflict_signal = max(ref_fraction, 0.60 * count_signal(ref, 8.0))
    if is_one_sided_segmentation_pass(segmentation) and ref > 0:
        features.ref_conflict_signal = max(features.ref_conflict_signal, 0.35)
    features.ref_conflict_signal = _clamp01(features.ref_conflict_signal)

    artifact_context = 0.0
    if not segmentation.has_insert_seq:
        artifact_context = max(artifact_context, 1.0)
    if te_alignment.sequence_model_label == "TE_MODEL_OUTLIER":
        artifact_context = max(artifact_context, 1.0)
    if te_alignment.sequence_model_label == "TE_MODEL_EDGE":
        artifact_context = max(artifact_context, 0.45)
    if not boundary.geometry_defined and not is_one_sided_segmentation_pass(segmentation):
        artifact_context = max(artifact_context, 0.35)
    if (boundary.geometry_defined and not boundary.canonical_pass
            and not boundary.evidence_consistent):
        artifact_context = max(artifact_context, 0.50)
    if te_alignment.annotation_confidence == "LOW":
        artifact_context = max(artifact_context, 0.35)
    artifact_context = max(artifact_context, 0.80 * features.ref_conflict_signal)
    features.artifact_context_signal = _clamp01(artifact_context)

    features.family_kind = family_kind(te_alignment)
    family_known = 0.0 if features.family_kind == "unknown" else 1.0
    features.family_activity_prior = clamp_score(
        (0.30 * features.te_identity) + (0.25 * features.event_signal)
        + (0.20 * features.independent_signal) + (0.15 * features.quality_signal)
        + (0.10 * family_known) - (0.30 * features.artifact_context_signal)
        - (0.10 * features.ref_conflict_signal),
        0.02, 0.98)
    return features


def family_state_compatibility(features: LatentFeatureVector, state: str) -> float:
    """How well the named family fits the mechanism, in log-prior units.

    The NEGATIVE entries carry most of the information: an `active_tprt_te`
    state for a DNA transposon is -0.55, because target-primed reverse
    transcription is not something a cut-and-paste element does. Encoding the
    mismatch is what lets the model tell a mislabelled insert from a real one of
    another kind, instead of merely preferring the best-matching family.

    `degraded_unknown_te` is the escape hatch: it rewards `unknown` outright, and
    also rewards low identity in ANY family, because an old diverged copy is
    exactly the case where the name is unreliable.
    """
    kind = features.family_kind
    if state == "active_tprt_te":
        return {"retro": 0.75, "ltr": 0.20, "dna": -0.55, "unknown": -0.35}.get(kind, 0.0)
    if state == "cut_paste_te":
        return {"dna": 0.75, "retro": -0.40, "ltr": -0.40, "unknown": -0.20}.get(kind, 0.0)
    if state == "ltr_complex_te":
        return {"ltr": 0.75, "retro": -0.10, "dna": -0.50, "unknown": -0.30}.get(kind, 0.0)
    if state == "degraded_unknown_te":
        if kind == "unknown":
            return 0.40
        if features.te_identity < 0.75:
            return 0.20
    return 0.0


def is_te_latent_state(state: str) -> bool:
    return state in ("active_tprt_te", "cut_paste_te", "ltr_complex_te",
                     "degraded_unknown_te")


def latent_state_log_prior(features: LatentFeatureVector, state: str) -> float:
    """Base rate of each mechanism, tilted by the locus's own activity prior.

    The tilt is asymmetric -- `+0.55 * logit` for a TE state, `-0.20 * logit`
    for a null state -- so evidence of activity helps the TE states about three
    times as much as it hurts the null ones. That asymmetry is a modelling
    choice with no derivation, and it is the kind of hand-set constant the
    README's argument against this whole layer is about.
    """
    base = -0.75
    if state in ("active_tprt_te", "cut_paste_te"):
        base = -1.35
    elif state == "ltr_complex_te":
        base = -1.50
    elif state == "degraded_unknown_te":
        base = -1.80
    elif state == "non_te_insert":
        base = -0.85
    activity_logit = logistic_logit(features.family_activity_prior)
    base += (0.55 * activity_logit) if is_te_latent_state(state) else (-0.20 * activity_logit)
    return base + family_state_compatibility(features, state)


#: Per-state feature means and strengths, in the order
#: (te_identity, event, independent, quality, artifact, ref_conflict).
#: The default row is `artifact_reference`.
_LATENT_STATE_PARAMS = {
    "active_tprt_te": (0.94, 12.0, 0.86, 5.5, 0.80, 6.0, 0.88, 3.0, 0.06, 4.5, 0.08, 2.5),
    "cut_paste_te": (0.92, 11.0, 0.84, 5.5, 0.76, 6.0, 0.86, 3.0, 0.07, 4.5, 0.08, 2.5),
    "ltr_complex_te": (0.90, 10.0, 0.80, 5.0, 0.70, 5.5, 0.84, 3.0, 0.10, 4.0, 0.12, 2.5),
    "degraded_unknown_te": (0.72, 3.5, 0.62, 4.0, 0.50, 5.0, 0.72, 2.5, 0.22, 3.0, 0.20, 2.0),
    "non_te_insert": (0.30, 5.5, 0.82, 5.0, 0.68, 4.0, 0.82, 3.0, 0.12, 3.5, 0.12, 2.0),
}
_ARTIFACT_PARAMS = (0.38, 3.5, 0.22, 4.0, 0.12, 4.0, 0.35, 3.0, 0.72, 5.0, 0.55, 3.0)


def latent_state_feature_log_likelihood(features: LatentFeatureVector,
                                        state: str) -> float:
    """Six independent Beta-shaped terms, summed.

    INDEPENDENCE IS ASSUMED AND IS FALSE. `event_signal` is built partly from
    `independent_signal`, and `artifact_context_signal` is built partly from
    `ref_conflict_signal`, so at least two of the six pairs are correlated by
    construction. The posterior is therefore overconfident. The port keeps it
    exactly; the honest fix is the same one the selection layer makes -- stop
    treating a hand-built score as a likelihood -- and it is not a porting
    decision.

    `non_te_insert` is the row to read carefully: it expects LOW te_identity
    (0.30) but HIGH event and independent signals. It is the model's statement
    that a real non-TE insertion looks structurally identical to a TE one and
    differs only in sequence.
    """
    params = _LATENT_STATE_PARAMS.get(state, _ARTIFACT_PARAMS)
    values = (features.te_identity, features.event_signal,
              features.independent_signal, features.quality_signal,
              features.artifact_context_signal, features.ref_conflict_signal)
    total = 0.0
    for i, value in enumerate(values):
        total += beta_feature_loglike(value, params[2 * i], params[2 * i + 1])
    return total


def evaluate_latent_mechanism_lfdr(existence: EventExistenceEvidence,
                                   segmentation: EventSegmentationEvidence,
                                   te_alignment,
                                   boundary: BoundaryEvidence,
                                   clip_insert_concordance: ClipInsertConcordanceEvidence | None = None,
                                   target_q: float = TARGET_Q) -> LatentMechanismEvidence:
    """Posterior over the six mechanisms, and a worst-case local FDR.

    THE WORST-CASE CONSTRUCTION is the interesting half. Rather than reporting
    `1 - P(TE)`, it re-weights the two sides by an AMBIGUITY allowance before
    renormalising: every TE state is divided by `exp(te_ambiguity)` and every
    null state multiplied by `exp(null_ambiguity)`. Both allowances grow with the
    mechanistic gap (`1 - independent_signal`) and with reference conflict, and
    the TE side's grows FASTER (2.60 vs 1.65 per unit of gap).

    That asymmetry is the precision-first stance made numerical: where the
    evidence is thin, the TE hypothesis is penalised more than the null is
    rewarded, so thin evidence produces abstention rather than a confident call
    in either direction. It is a sensitivity analysis, not a probability -- and
    `worst_case_lfdr` is the number the emission gate actually reads.
    """
    out = LatentMechanismEvidence()
    features = build_latent_feature_vector(existence, segmentation, te_alignment,
                                           boundary, clip_insert_concordance)
    out.family_activity_prior = features.family_activity_prior

    log_evidence = [latent_state_log_prior(features, state)
                    + latent_state_feature_log_likelihood(features, state)
                    for state in LATENT_MECHANISM_STATES]

    denom = logsumexp_values(log_evidence)
    best_log = _LOG_ZERO
    for state, log_value in zip(LATENT_MECHANISM_STATES, log_evidence):
        posterior = math.exp(log_value - denom)
        if is_te_latent_state(state):
            out.te_posterior += posterior
        elif state == "non_te_insert":
            out.non_te_posterior = posterior
        elif state == "artifact_reference":
            out.artifact_posterior = posterior
        if log_value > best_log:
            best_log = log_value
            out.latent_mechanism = state
    out.lfdr = out.non_te_posterior + out.artifact_posterior

    mechanistic_gap = 1.0 - features.independent_signal
    te_ambiguity = 0.95 + (2.60 * mechanistic_gap) + (0.75 * features.ref_conflict_signal)
    null_ambiguity = 0.95 + (1.65 * mechanistic_gap) + (0.55 * features.ref_conflict_signal)
    log_te_wc = _LOG_ZERO
    log_null_wc = _LOG_ZERO
    for state, log_value in zip(LATENT_MECHANISM_STATES, log_evidence):
        if is_te_latent_state(state):
            log_te_wc = logsumexp_pair(log_te_wc, log_value - te_ambiguity)
        else:
            log_null_wc = logsumexp_pair(log_null_wc, log_value + null_ambiguity)
    out.worst_case_lfdr = math.exp(log_null_wc - logsumexp_pair(log_te_wc, log_null_wc))
    out.lfdr_qc = ("PASS_TE_LFDR" if out.worst_case_lfdr <= _clamp01(target_q)
                   else "TE_LFDR_HIGH")
    return out


# --------------------------------------------------------------------------
# The local three-way posterior.
# --------------------------------------------------------------------------
def is_te_interpretable_without_threshold_gate(te_alignment) -> bool:
    """Can this alignment be read as naming a TE at all, ignoring strength?"""
    if not getattr(te_alignment, "pass_", False):
        return False
    if te_alignment.sequence_model_label == "TE_MODEL_OUTLIER":
        return False
    return is_pass_insert_te_alignment(classify_te_alignment_qc(te_alignment.qc_reason))


def has_closed_te_breakpoints(segmentation: EventSegmentationEvidence,
                              boundary: BoundaryEvidence) -> bool:
    """Both flanks anchored AND a usable junction. Definitional, not tuned.

    "Closed" is the difference between an insertion whose two ends are both
    accounted for and one where the far end is an assumption. Every use of it is
    a necessary condition, never a score.
    """
    return (segmentation.pair_valid and segmentation.has_left_flank
            and segmentation.has_right_flank and boundary.geometry_defined
            and (boundary.canonical_pass or boundary.evidence_consistent))


def te_alignment_log_support(te_alignment) -> float:
    """Sequence support, on a log-LR-ish scale bounded to [-3, 3].

    The three PASS tiers get different INTERCEPTS and different slopes, and the
    pivots (identity 0.78 / 0.68 / 0.55, coverage 0.72 / 0.55 / 0.70) say where
    each tier is considered neutral. These are nineteen hand-set constants in
    total across this file, which is precisely what `placer_py/blocks.py`'s
    docstring argues cannot support genome-scale FDR control -- and why this
    number feeds the RANKING and the diagnostics rather than the gate.
    """
    identity = _clamp01(te_alignment.best_identity)
    coverage = _clamp01(te_alignment.best_query_coverage)
    margin = _clamp01(te_alignment.cross_family_margin)

    qc = classify_te_alignment_qc(te_alignment.qc_reason)
    score = -1.5
    if qc == TeAlignmentQc.PASS_INSERT:
        score = 1.2 + (3.0 * (identity - 0.78)) + (1.4 * (coverage - 0.72)) + min(1.0, 2.0 * margin)
    elif qc == TeAlignmentQc.PASS_INSERT_FAMILY_ONLY:
        score = 0.65 + (2.2 * (identity - 0.68)) + (1.0 * (coverage - 0.55)) + min(0.7, 1.5 * margin)
    elif qc == TeAlignmentQc.PASS_INSERT_UNKNOWN:
        score = -0.45 + (1.9 * (identity - 0.55)) + (0.9 * (coverage - 0.70)) + min(0.45, 1.0 * margin)
    elif qc == TeAlignmentQc.LOW_IDENTITY:
        score = -0.85 + (1.4 * (identity - 0.50)) + (0.6 * (coverage - 0.40))
    elif qc in (TeAlignmentQc.NO_TE_ALIGNMENT, TeAlignmentQc.OTHER):
        if not getattr(te_alignment, "pass_", False):
            score = -2.0

    label = te_alignment.sequence_model_label
    if label == "TE_MODEL_IN_DISTRIBUTION":
        score += 0.55 + clamp_score(te_alignment.sequence_model_score, 0.0, 0.50)
    elif label == "TE_MODEL_EDGE":
        score -= 0.25
    elif label == "TE_MODEL_OUTLIER":
        score -= 1.0

    confidence = te_alignment.annotation_confidence
    if confidence == "LOW":
        score -= 0.65
    elif confidence == "MEDIUM":
        score -= 0.20
    elif confidence == "HIGH":
        score += 0.25
    score -= _clamp01(te_alignment.annotation_residual_fraction) * 0.8
    score -= _clamp01(te_alignment.annotation_masked_fraction) * 0.3
    return clamp_score(score, -3.0, 3.0)


def evaluate_local_hypothesis_posterior(existence: EventExistenceEvidence,
                                        segmentation: EventSegmentationEvidence,
                                        te_alignment,
                                        boundary: BoundaryEvidence
                                        ) -> LocalHypothesisPosterior:
    """Three unnormalised log-scores, softmaxed.

    The artifact score is built ADDITIVELY from named defects -- no insert
    sequence (+1.6), one-sided (+1.1), no usable junction (+0.9), no precise or
    bilateral read (+0.5), plus a term in the reference-read count. Each is a
    concrete way the locus could be a mapping artifact, and reading the sum tells
    you which ones fired.

    The unknown-TE deficits are the subtlest part: when the element could not be
    named, the artifact side is charged in proportion to HOW FAR SHORT the
    identity, coverage and margin fall of their pivots, rather than by a flat
    penalty. An unnamed insert at identity 0.61 and an unnamed insert at
    identity 0.20 are not equally suspicious.
    """
    posterior = LocalHypothesisPosterior()
    alt = max(0, existence.alt_struct_reads)
    ref = max(0, existence.ref_span_reads)
    precise = precise_structural_reads(existence)
    bilateral_clip = bilateral_clip_support_reads(existence)
    one_sided = is_one_sided_segmentation_pass(segmentation)
    closed = has_closed_te_breakpoints(segmentation, boundary)
    unknown_te = (classify_te_alignment_qc(te_alignment.qc_reason)
                  == TeAlignmentQc.PASS_INSERT_UNKNOWN
                  or te_alignment.best_family == "UNKNOWN"
                  or te_alignment.best_subfamily == "UNKNOWN")

    count_balance = clamp_score(math.log(alt + 1) - math.log(ref + 1), -2.0, 2.0)
    precise_balance = clamp_score(math.log(precise + bilateral_clip + 1), 0.0, 2.0)
    genotype_support = clamp_score((existence.gq - 20.0) / 40.0, 0.0, 1.5)
    structural_support = clamp_score(
        0.65 * count_balance + 0.45 * precise_balance + genotype_support, -2.0, 3.0)
    segmentation_support = clamp_score(adjusted_segmentation_score(segmentation), -2.0, 2.0)
    sequence_support = te_alignment_log_support(te_alignment)
    te_identity = _clamp01(te_alignment.best_identity)
    te_coverage = _clamp01(te_alignment.best_query_coverage)
    te_margin = _clamp01(te_alignment.cross_family_margin)

    boundary_support = -1.0
    if closed:
        boundary_support = 1.0
    elif one_sided and segmentation.has_insert_seq:
        boundary_support = -0.35
    elif boundary.geometry_defined and boundary.evidence_consistent:
        boundary_support = 0.25

    artifact_context = 0.0
    if not segmentation.has_insert_seq:
        artifact_context += 1.6
    if one_sided:
        artifact_context += 1.1
    if not boundary.geometry_defined or (not boundary.canonical_pass
                                         and not boundary.evidence_consistent):
        artifact_context += 0.9
    if unknown_te:
        identity_deficit = clamp_score((0.62 - te_identity) / 0.16, 0.0, 1.0)
        coverage_deficit = clamp_score((0.90 - te_coverage) / 0.40, 0.0, 1.0)
        margin_deficit = clamp_score((0.05 - te_margin) / 0.05, 0.0, 1.0)
        if te_alignment.annotation_confidence == "LOW":
            artifact_context += 0.50
        elif te_alignment.annotation_confidence == "MEDIUM":
            artifact_context += 0.20
        artifact_context += ((0.45 * identity_deficit) + (0.20 * coverage_deficit)
                             + (0.15 * margin_deficit))
    if precise == 0 and bilateral_clip == 0:
        artifact_context += 0.5
    artifact_context += min(1.0, ref * 0.08)

    posterior.log_te = (-0.25 + structural_support + (0.75 * segmentation_support)
                        + sequence_support + boundary_support
                        - (0.35 * artifact_context))
    # The non-TE score SUBTRACTS sequence support: evidence that the insert is a
    # TE is evidence against its being a non-TE insertion, and the two
    # hypotheses have to compete on the same axis.
    posterior.log_non_te = (0.10 + (0.85 * structural_support)
                            + (0.60 * segmentation_support)
                            - (0.75 * sequence_support)
                            + (0.25 if segmentation.has_insert_seq else -0.75))
    posterior.log_artifact = (-0.10 + artifact_context - (0.25 * structural_support)
                              - (0.15 * segmentation_support)
                              - (0.35 * max(sequence_support, 0.0)))

    denom = logsumexp3(posterior.log_te, posterior.log_non_te, posterior.log_artifact)
    posterior.te = math.exp(posterior.log_te - denom)
    posterior.non_te = math.exp(posterior.log_non_te - denom)
    posterior.artifact = math.exp(posterior.log_artifact - denom)
    posterior.te_vs_artifact_log_odds = posterior.log_te - posterior.log_artifact
    posterior.te_vs_non_te_log_odds = posterior.log_te - posterior.log_non_te

    # A closed event is held to a LOWER bar, because closure already removed the
    # commonest way to be wrong.
    min_te_posterior = 0.65 if closed else 0.72
    min_artifact_odds = 0.75 if closed else 1.25
    min_non_te_odds = 0.55 if unknown_te else 0.25
    if (posterior.te >= min_te_posterior
            and posterior.te_vs_artifact_log_odds >= min_artifact_odds
            and posterior.te_vs_non_te_log_odds >= min_non_te_odds):
        posterior.qc = "PASS_TE_POSTERIOR"
    return posterior


# --------------------------------------------------------------------------
# Boundary geometry.
# --------------------------------------------------------------------------
def check_boundary_consistency(input_: FinalBoundaryInput) -> FinalBoundaryDecision:
    """Classify the junction from the two reference segments alone.

    `right_ref_start - left_ref_end` is the whole test, and its SIGN is the
    biology:

      * negative -- the two flanks OVERLAP on the reference, i.e. the same bases
        appear on both sides of the insert. That is a target-site duplication,
        the signature of target-primed reverse transcription.
      * zero -- a blunt join.
      * positive -- reference bases are missing, a target-site deletion, which is
        credible but decays with size.

    An overlap outside [tsd_min, tsd_max] is REJECTED rather than reported as a
    long TSD: a 200 bp overlap is a misassembled flank, not a duplication.
    """
    decision = FinalBoundaryDecision()
    if (input_.left_ref_start < 0 or input_.left_ref_end < 0
            or input_.right_ref_start < 0 or input_.right_ref_end < 0):
        decision.qc = "REJECT_BOUNDARY_MISSING_REF_SEGMENTS"
        return decision
    if (input_.left_ref_start >= input_.left_ref_end
            or input_.right_ref_start >= input_.right_ref_end):
        decision.qc = "REJECT_BOUNDARY_INVALID_REF_SEGMENTS"
        return decision

    tsd_min_len = max(1, input_.tsd_min_len)
    tsd_max_len = max(tsd_min_len, input_.tsd_max_len)
    link_delta = input_.right_ref_start - input_.left_ref_end

    if link_delta < 0:
        overlap = -link_delta
        if overlap < tsd_min_len or overlap > tsd_max_len:
            decision.qc = "REJECT_BOUNDARY_TSD_RANGE"
            return decision
        return FinalBoundaryDecision(True, "TSD", overlap, "PASS_BOUNDARY_TSD")
    if link_delta == 0:
        return FinalBoundaryDecision(True, "BLUNT", 0, "PASS_BOUNDARY_BLUNT")
    if link_delta <= tsd_max_len:
        return FinalBoundaryDecision(True, "SMALL_DEL", link_delta,
                                     "PASS_BOUNDARY_SMALL_DEL")
    decision.qc = "REJECT_BOUNDARY_DEL_RANGE"
    return decision


#: Median TSD length for the common families, and the log-scale spread around
#: it. Roughly log-normal: ~12 bp is typical, 4 bp and 30 bp are both plausible,
#: 55 bp is not.
TSD_LOG_MEDIAN = 12.0
TSD_LOG_SIGMA = 0.7


def boundary_structure_log_lr(boundary_type: str, length: int) -> float:
    """Grade the junction by how TE-like its geometry is, not just pass/fail.

    The discrete range test says only whether the geometry is admissible. This
    says how much it LOOKS like a genuine insertion: a TSD near 12 bp is strong
    evidence, a 55 bp "TSD" inside the admissible range is weak, and a target-
    site deletion decays linearly with size. The quadratic in log-space is what
    makes the TSD term peak rather than saturate.
    """
    if boundary_type == "BLUNT":
        return 1.0
    if boundary_type == "TSD":
        x = max(1, length)
        z = (math.log(x) - math.log(TSD_LOG_MEDIAN)) / TSD_LOG_SIGMA
        return clamp_score(1.2 - (0.5 * z * z), -2.0, 1.2)
    if boundary_type == "SMALL_DEL":
        return clamp_score(0.8 - (max(0, length) / 80.0), 0.2, 0.8)
    if boundary_type == "NONCANONICAL":
        return 0.25
    return -2.0


def evaluate_boundary_evidence(input_: FinalBoundaryInput,
                               breakpoint_envelope_width: int) -> BoundaryEvidence:
    """Canonical geometry first; otherwise, is the junction within the envelope?

    THE NONCANONICAL PATH is what keeps a real insertion whose breakpoints are
    uncertain from being scored as an artifact. If the two flanks disagree by
    less than the breakpoint envelope the stage already measured, the
    disagreement is measurement noise rather than a contradiction, and the
    junction is accepted at a reduced score (0.25 against a TSD's 1.2).
    """
    evidence = BoundaryEvidence()
    canonical = check_boundary_consistency(input_)
    evidence.geometry_defined = (input_.left_ref_start >= 0 and input_.left_ref_end >= 0
                                 and input_.right_ref_start >= 0
                                 and input_.right_ref_end >= 0
                                 and input_.left_ref_start < input_.left_ref_end
                                 and input_.right_ref_start < input_.right_ref_end)
    evidence.canonical_pass = canonical.pass_
    evidence.boundary_type = canonical.boundary_type
    evidence.boundary_len = canonical.boundary_len
    evidence.qc = canonical.qc

    if not evidence.geometry_defined:
        evidence.score = -2.0
        return evidence
    if canonical.pass_:
        evidence.evidence_consistent = True
        evidence.score = boundary_structure_log_lr(canonical.boundary_type,
                                                   canonical.boundary_len)
        return evidence

    link_delta = input_.right_ref_start - input_.left_ref_end
    noncanonical_span = abs(link_delta)
    evidence.evidence_consistent = (breakpoint_envelope_width > 0
                                    and noncanonical_span <= breakpoint_envelope_width)
    if evidence.evidence_consistent:
        evidence.boundary_type = "NONCANONICAL"
        evidence.boundary_len = noncanonical_span
        evidence.qc = "PASS_BOUNDARY_NONCANONICAL_CONSISTENT"
        evidence.score = boundary_structure_log_lr("NONCANONICAL", noncanonical_span)
    else:
        evidence.score = -2.0
    return evidence


# --------------------------------------------------------------------------
# Existence evidence, from the genotyper.
# --------------------------------------------------------------------------
def build_event_existence_evidence(input_: EventGenotypeInput) -> EventExistenceEvidence:
    """Run the genotyper and convert GQ into an existence SCORE.

    `(gq - 20) / 20`, clamped to [-3, 3]: Phred 20 (1% error) is the pivot, so
    a locus the genotyper is exactly 99% sure about contributes nothing either
    way. Everything downstream reads the score, not the GQ, which is how the
    existence evidence stays on the same scale as the other three supports.
    """
    evidence = EventExistenceEvidence()
    decision = genotype_from_alt_vs_ref(
        input_.alt_struct_reads, input_.ref_span_reads,
        error_rate=input_.error_rate, overdispersion=input_.overdispersion,
        min_gq=input_.min_gq, event_length=input_.event_length,
        alt_observed_lengths=list(input_.alt_observed_lengths))
    evidence.best_gt = decision.best_gt
    evidence.af = decision.allele_fraction
    evidence.gq = decision.gq
    evidence.alt_struct_reads = max(0, input_.alt_struct_reads)
    evidence.alt_split_reads = input_.alt_split_reads
    evidence.alt_indel_reads = input_.alt_indel_reads
    evidence.alt_left_clip_reads = input_.alt_left_clip_reads
    evidence.alt_right_clip_reads = input_.alt_right_clip_reads
    evidence.ref_span_reads = max(0, input_.ref_span_reads)
    evidence.depth = decision.depth
    evidence.best_nonref_minus_ref_ll = decision.best_nonref_minus_ref_ll
    evidence.score = clamp_score((decision.gq - 20.0) / 20.0)
    return evidence


# --------------------------------------------------------------------------
# Segmentation evidence.
# --------------------------------------------------------------------------
def analyze_event_segmentation(has_consensus: bool, left_flank_align_len: int,
                               right_flank_align_len: int, left_flank_identity: float,
                               right_flank_identity: float, insert_seq: str,
                               pair_valid: bool, qc_reason: str
                               ) -> EventSegmentationEvidence:
    """Score how well the assembled event splits into flank / insert / flank.

    The scoring flank is the WEAKER of the two -- `min` on both length and
    identity -- because an event is only as well anchored as its worse side. The
    one-sided case is the exception: there is no second side to be weaker, so the
    one that exists is scored and a flat 0.5 is deducted for the absence.

    The two penalties do not stack the way a reading might suggest: a one-sided
    pass costs 0.5, a genuinely missing flank costs 1.0, and they are exclusive.
    """
    evidence = EventSegmentationEvidence(
        has_consensus=has_consensus,
        has_left_flank=left_flank_align_len > 0,
        has_right_flank=right_flank_align_len > 0,
        has_insert_seq=bool(insert_seq),
        pair_valid=pair_valid,
        left_align_len=left_flank_align_len,
        right_align_len=right_flank_align_len,
        left_identity=left_flank_identity,
        right_identity=right_flank_identity,
        insert_len=len(insert_seq),
        qc=qc_reason)

    scoring_flank_len = min(evidence.left_align_len, evidence.right_align_len)
    scoring_flank_identity = min(evidence.left_identity, evidence.right_identity)
    one_sided_pass = (evidence.pair_valid and evidence.has_insert_seq
                      and (evidence.has_left_flank != evidence.has_right_flank))
    if one_sided_pass:
        scoring_flank_len = (evidence.left_align_len if evidence.has_left_flank
                             else evidence.right_align_len)
        scoring_flank_identity = (evidence.left_identity if evidence.has_left_flank
                                  else evidence.right_identity)

    # Pivots: 50 bp of flank and 90% identity are neutral.
    evidence.score = (0.5 * clamp_score((scoring_flank_len - 50.0) / 25.0, -2.0, 2.0)
                      + 0.5 * clamp_score((scoring_flank_identity - 0.90) / 0.05, -2.0, 2.0))
    if one_sided_pass:
        evidence.score -= 0.5
    elif not evidence.has_left_flank or not evidence.has_right_flank:
        evidence.score -= 1.0
    if not evidence.pair_valid:
        evidence.score -= 1.5
    return evidence


# --------------------------------------------------------------------------
# The four competing explanations.
#
# Each is a RESIDUAL vector -- "what this explanation failed to account for" --
# and they are compared by dominance rather than by a weighted sum, so no
# exchange rate between "50 unexplained bases" and "2 conflicting reads" ever
# has to be declared. See placer_py/explanation.py for the comparison itself.
# --------------------------------------------------------------------------
def has_te_sequence_evidence(te_alignment) -> bool:
    """Is there ANY sequence evidence about a TE here, of any strength?

    Deliberately permissive -- a bare `best_identity > 0` qualifies. It is the
    predicate for "keep this locus visible as TE-evidence", i.e. the abstention
    path, and abstention should be reachable from weak evidence by definition.
    """
    qc = classify_te_alignment_qc(te_alignment.qc_reason)
    return (getattr(te_alignment, "pass_", False)
            or is_pass_insert_te_alignment(qc)
            or qc == TeAlignmentQc.LOW_IDENTITY
            or te_alignment.best_identity > 0.0
            or te_alignment.best_query_coverage > 0.0)


def te_structure_explanation_for_decision(segmentation: EventSegmentationEvidence,
                                          te_alignment) -> SequenceExplanation:
    """Use the alignment's own decode, or re-derive one from the length alone."""
    supplied = getattr(te_alignment, "te_sequence_explanation", None)
    if supplied is not None and supplied.status != structure_module.TeAnnotationStatus.UNAVAILABLE:
        return supplied
    return structure_module.explain_te_alignment_shadow(
        segmentation.insert_len, te_alignment.qc_reason, te_alignment.best_family,
        te_alignment.best_subfamily, te_alignment.best_identity,
        te_alignment.best_query_coverage, te_alignment.annotation_residual_fraction,
        te_alignment.annotation_masked_fraction, te_alignment.cross_family_margin,
        te_alignment.second_score)


def make_reference_explanation(existence: EventExistenceEvidence,
                               segmentation: EventSegmentationEvidence
                               ) -> EventExplanation:
    """"There is nothing here." Its residual is everything that says otherwise.

    An insert sequence is a structural conflict AND `max(1, insert_len)`
    unexplained bases -- the `max(1, ...)` matters, because a 0-length insert
    that nonetheless exists must still cost this explanation something.
    """
    out = EventExplanation(kind=ExplanationKind.REFERENCE, status="REFERENCE")
    out.residual.structural_conflicts = 1 if segmentation.has_insert_seq else 0
    out.residual.unexplained_high_complexity_bases = (
        max(1, segmentation.insert_len) if segmentation.has_insert_seq else 0)
    out.residual.read_assignment_conflicts = positive_or_zero(existence.alt_struct_reads)
    out.residual.path_complexity = 0
    return out


def make_non_te_explanation(existence: EventExistenceEvidence,
                            segmentation: EventSegmentationEvidence,
                            te_alignment) -> EventExplanation:
    """"A real insertion, but not a TE."

    It explains the insert PARTIALLY -- `insert_len / 2` if a TE alignment
    passed, `/ 4` otherwise -- which reads backwards until you see what it is
    for. A passing TE alignment is evidence AGAINST this explanation, so it must
    raise this residual; without a TE alignment there is less to contradict, so
    less is left unexplained.
    """
    out = EventExplanation(kind=ExplanationKind.INSERTION_NON_TE,
                           status="NON_TE_INSERTION")
    te_pass = getattr(te_alignment, "pass_", False)
    out.residual.missing_required_components = 0 if segmentation.has_insert_seq else 1
    out.residual.unexplained_high_complexity_bases = (
        max(0, segmentation.insert_len // (2 if te_pass else 4))
        if segmentation.has_insert_seq else 0)
    out.residual.read_assignment_conflicts = positive_or_zero(existence.ref_span_reads)
    out.residual.label_ambiguity = 1 if te_pass else 0
    out.residual.path_complexity = 1 if segmentation.has_insert_seq else 0
    return out


def make_te_explanation(existence: EventExistenceEvidence,
                        segmentation: EventSegmentationEvidence, te_alignment,
                        boundary: BoundaryEvidence) -> EventExplanation:
    """"A TE inserted here." The residual counts what that fails to explain.

    `breakpoint_disagreement_bp` is 0 or a flat 25 -- not a measured
    disagreement. Without closed breakpoints the far end is an assumption, and
    25 bp is a stand-in for "unknown by about a TSD's worth". It is one of the
    places where the residual vector is a rank rather than a measurement.
    """
    out = EventExplanation(kind=ExplanationKind.TE)
    out.family = te_alignment.best_family or "UNKNOWN"
    out.subfamily = te_alignment.best_subfamily or "NA"
    out.status = "TE_UNKNOWN" if out.family == "UNKNOWN" else "TE_RESOLVED"
    interpretable = is_te_interpretable_without_threshold_gate(te_alignment)
    out.residual.missing_required_components = ((0 if segmentation.has_insert_seq else 1)
                                                + (0 if interpretable else 1))
    out.residual.structural_conflicts = 0 if segmentation.has_insert_seq else 1
    out.residual.unexplained_high_complexity_bases = (
        0 if interpretable else max(1, segmentation.insert_len))
    out.residual.breakpoint_disagreement_bp = (
        0 if has_closed_te_breakpoints(segmentation, boundary) else 25)
    out.residual.read_assignment_conflicts = positive_or_zero(existence.ref_span_reads)
    out.residual.artifact_evidence = (
        1 if te_alignment.sequence_model_label == "TE_MODEL_EDGE" else 0)
    out.residual.label_ambiguity = 1 if out.family == "UNKNOWN" else 0
    out.residual.path_complexity = 2 if out.family == "UNKNOWN" else 1
    return out


def make_artifact_explanation(existence: EventExistenceEvidence,
                              segmentation: EventSegmentationEvidence, te_alignment,
                              boundary: BoundaryEvidence) -> EventExplanation:
    """"A mapping artifact." The residual is what that fails to explain.

    `artifact_evidence` is INVERTED here relative to every other explanation: a
    `TE_MODEL_OUTLIER` scores 0 and anything else scores 2, because an
    out-of-distribution insert is what an artifact is SUPPOSED to look like.
    A well-formed TE-like sequence is the thing this explanation cannot account
    for, and defined geometry adds one more.
    """
    out = EventExplanation(kind=ExplanationKind.ARTIFACT, status="ARTIFACT")
    out.residual.unexplained_high_complexity_bases = (
        max(0, segmentation.insert_len // 3) if segmentation.has_insert_seq else 0)
    out.residual.read_assignment_conflicts = (
        positive_or_zero(existence.alt_split_reads)
        + positive_or_zero(existence.alt_indel_reads))
    out.residual.reference_counterevidence = positive_or_zero(existence.ref_span_reads)
    out.residual.artifact_evidence = (
        (0 if te_alignment.sequence_model_label == "TE_MODEL_OUTLIER" else 2)
        + (1 if boundary.geometry_defined else 0))
    out.residual.path_complexity = 1
    return out


def evaluate_event_explanations(existence: EventExistenceEvidence,
                                segmentation: EventSegmentationEvidence,
                                te_alignment, boundary: BoundaryEvidence,
                                clip_insert_concordance=None) -> ExplanationDecision:
    explanations = [
        make_reference_explanation(existence, segmentation),
        make_non_te_explanation(existence, segmentation, te_alignment),
        make_te_explanation(existence, segmentation, te_alignment, boundary),
        make_artifact_explanation(existence, segmentation, te_alignment, boundary),
    ]
    return compare_event_explanations(explanations,
                                      has_closed_te_breakpoints(segmentation, boundary))


# --------------------------------------------------------------------------
# The two hard vetoes: necessary conditions, not evidence thresholds.
# --------------------------------------------------------------------------
def compute_te_unknown_hard_veto(existence: EventExistenceEvidence,
                                 segmentation: EventSegmentationEvidence,
                                 te_alignment, boundary: BoundaryEvidence) -> bool:
    """Ineligible when the event DEFINITIONALLY cannot be a TE call.

    `existence` and `boundary` are unused -- the read-count and one-sided
    ladders that read them were deleted -- and are kept in the signature for
    symmetry with the resolved veto. Whether an ELIGIBLE hypothesis is emitted
    is decided by the worst-case local FDR, not here.
    """
    qc = classify_te_alignment_qc(te_alignment.qc_reason)
    return (not segmentation.has_insert_seq
            or qc == TeAlignmentQc.NO_TE_ALIGNMENT
            or not annotation_quality_allows_te_decision(te_alignment))


def compute_te_resolved_hard_veto(existence: EventExistenceEvidence,
                                  segmentation: EventSegmentationEvidence,
                                  te_alignment, boundary: BoundaryEvidence) -> bool:
    """As above, plus: a RESOLVED call needs a resolved QC reason.

    `PASS_INSERT_TE_ALIGNMENT_UNKNOWN` vetoes h3 specifically. An element that
    aligned but could not be named is a perfectly good TE-unknown call and not a
    resolved one, and the distinction is the whole reason there are two TE
    hypotheses rather than one.
    """
    return (not segmentation.has_insert_seq
            or not getattr(te_alignment, "pass_", False)
            or classify_te_alignment_qc(te_alignment.qc_reason) == TeAlignmentQc.PASS_INSERT_UNKNOWN
            or not annotation_quality_allows_te_decision(te_alignment))


# --------------------------------------------------------------------------
# The structural-insertion gate.
# --------------------------------------------------------------------------
def should_emit_structural_event_call(existence: EventExistenceEvidence,
                                      segmentation: EventSegmentationEvidence,
                                      boundary: BoundaryEvidence,
                                      posterior: LocalHypothesisPosterior,
                                      mechanistic_ref_conflict_signal: float,
                                      nonte_explanation: bool) -> bool:
    """"There is a real insertion here", even when its identity is unresolved.

    A SINGLE log-evidence gate, with no read-count or length ladders. Three
    parts are worth reading closely:

    THE ODDS USE `te + non_te`, not `non_te` alone. The question is whether an
    insertion exists, not what it is -- which is what lets a TE-like event that
    abstained on the TE lFDR gate still be reported as a real insertion.

    THE COUNT EVIDENCE IS A MAX of two models. `log((alt+1)/(ref+1))` asks
    whether the alt allele dominates; the beta-binomial Bayes factor asks
    whether a MINORITY alt allele beats a 2% error process. A het or mosaic
    insertion satisfies only the second, a clean homozygote only the first, and
    taking the max means neither regime needs an allele-fraction threshold.

    THE REGIME GUARD is the subtle one. `!nonte_explanation && !low_af_regime`
    returns False, so a HIGH-AF TE-like event that failed the TE gate is NOT
    downgraded to a structural call -- it stays TE-evidence, i.e. abstention.
    Only genuine non-TE insertions and low-AF events whose identity is uncertain
    BECAUSE alt coverage is low come through here.

    And the genotype requirement at the end stops boundary and model priors from
    carrying a trivially-supported event over the bar on their own.
    """
    alt = max(0, existence.alt_struct_reads)
    ref = max(0, existence.ref_span_reads)
    precise = precise_structural_reads(existence)
    bilateral_clip = bilateral_clip_support_reads(existence)

    closed_structural_event = (segmentation.pair_valid and segmentation.has_insert_seq
                               and segmentation.has_left_flank
                               and segmentation.has_right_flank
                               and boundary.geometry_defined
                               and (boundary.canonical_pass or boundary.evidence_consistent))
    if not closed_structural_event:
        return False

    real_insertion_posterior = clamp_score(posterior.te + posterior.non_te, 1e-6, 1.0)
    structural_log_odds = (math.log(real_insertion_posterior)
                           - math.log(max(_clamp01(posterior.artifact), 1e-6)))
    dominant_allele_log_odds = math.log(alt + 1) - math.log(ref + 1)
    low_af_log_bf = low_allele_fraction_insertion_log_bf(alt, ref)
    low_af_regime = (alt < ref and math.isfinite(low_af_log_bf)
                     and low_af_log_bf > dominant_allele_log_odds)
    count_evidence = low_af_log_bf if low_af_regime else dominant_allele_log_odds
    # A low-AF het/mosaic legitimately carries many reference-spanning reads, so
    # it is held to the looser artifact cap; dominant-allele events are not.
    artifact_cap = ARTIFACT_POSTERIOR_CAP if low_af_regime else ARTIFACT_POSTERIOR_STRICT_CAP

    independent_support = math.log(precise + bilateral_clip + 1)
    boundary_support = 0.70 if boundary.canonical_pass else 0.35
    genotype_support = clamp_score((existence.gq - 20.0) / 40.0, 0.0, 1.0)
    ref_conflict_penalty = 1.20 * _clamp01(mechanistic_ref_conflict_signal)
    event_existence_log_evidence = (structural_log_odds + (0.65 * count_evidence)
                                    + (0.35 * independent_support) + boundary_support
                                    + (0.40 * genotype_support) - ref_conflict_penalty)

    if not nonte_explanation and not low_af_regime:
        return False
    return (event_existence_log_evidence > STRUCTURAL_LOG_EVIDENCE
            and posterior.artifact <= artifact_cap
            and (precise + bilateral_clip) > 0
            and genotype_support > 0.0)


# --------------------------------------------------------------------------
# The context-conditioned prior.
# --------------------------------------------------------------------------
def context_conditioned_prior(artifact_context_signal: float) -> dict:
    """Strengthen the null prior where artifacts are common.

    The prior over TE / non-TE / artifact is NOT universal -- it depends on the
    local context. In an artifact-prone one (low-complexity or tandem insert,
    one-sided or reference-conflicted junction, an edge or outlier sequence
    model) the artifact prior rises and the TE prior floor falls, so the
    worst-case lFDR gate demands more evidence there and less in a clean one.

    This is the decision layer's version of the same rule the TPRT model follows
    for its four null terms: read the null off the local data rather than
    applying one number everywhere. The difference -- and it is the whole of the
    README's argument -- is that these are hand-set slopes rather than measured
    quantities.
    """
    a = _clamp01(artifact_context_signal)
    return {
        "te_min": clamp_score(0.05 * (1.0 - (0.7 * a)), 5e-3, 0.05),
        "te_max": clamp_score(0.80 - (0.20 * a), 0.10, 0.80),
        "non_te_min": 1e-6,
        "non_te_max": 0.30,
        "artifact_min": clamp_score(0.50 + (0.25 * a), 0.0, 0.95),
        "artifact_max": 0.95,
    }


# --------------------------------------------------------------------------
# The joint decision.
# --------------------------------------------------------------------------
def evaluate_joint_hypotheses(existence: EventExistenceEvidence,
                              segmentation: EventSegmentationEvidence,
                              te_alignment, boundary: BoundaryEvidence,
                              clip_insert_concordance: ClipInsertConcordanceEvidence | None = None
                              ) -> JointDecisionResult:
    """Rank four hypotheses, then gate emission on ONE risk level.

    THE ORDER OF OPERATIONS IS THE DESIGN:

      1. score and rank h0..h3 -- diagnostic, and the key the breakpoint stage
         uses to choose between competing candidates;
      2. run the explanation comparator, which ADVISES which kind of event this
         is (`emit_te_call` and friends come from it);
      3. override that advice with the calibrated worst-case lFDR: an advised TE
         call that fails the gate becomes `TE_AMBIGUOUS`, retained as evidence
         rather than rejected;
      4. failing that, consider the structural-insertion path;
      5. finally, make `best` agree with whatever was emitted.

    Step 3 is why the explanation engine's own `emit_te_call` cannot be read as
    the answer: it proposes, and the risk gate disposes.
    """
    result = JointDecisionResult()
    te_model_support = te_sequence_model_score(te_alignment)
    existence_support = adjusted_nonref_existence_score(existence, segmentation,
                                                        te_model_support)
    segmentation_support = adjusted_segmentation_score(segmentation)
    te_boundary_support = adjusted_te_boundary_score(segmentation, boundary)

    posterior = evaluate_local_hypothesis_posterior(existence, segmentation,
                                                    te_alignment, boundary)
    latent_lfdr = evaluate_latent_mechanism_lfdr(existence, segmentation, te_alignment,
                                                 boundary, clip_insert_concordance)
    certificate = blocks_module.build_certificate(
        existence, segmentation, te_alignment, boundary,
        clip_insert_concordance)
    te_structure = te_structure_explanation_for_decision(segmentation, te_alignment)
    robust_mechanistic = blocks_module.evaluate_robust_lfdr(
        certificate, context_conditioned_prior(certificate.artifact_context_signal),
        TARGET_Q)

    h0 = JointHypothesisScore(kind=FinalHypothesisKind.REFERENCE, reason="REFERENCE")
    # NOTE the `max(..., 0)`: the reference hypothesis is penalised by POSITIVE
    # existence and segmentation evidence and is never REWARDED by negative
    # evidence. Absence of evidence for an insertion is not evidence for the
    # reference; it just leaves h0 at zero.
    h0.existence = SCORE_REF_EXISTENCE_WEIGHT * max(existence.score, 0.0)
    h0.segmentation = SCORE_REF_SEGMENTATION_WEIGHT * max(segmentation_support, 0.0)
    h0.te = 0.0
    h0.total = h0.existence + h0.segmentation + h0.te

    h1 = JointHypothesisScore(kind=FinalHypothesisKind.INSERTION_NON_TE,
                              reason="NON_TE_INSERTION")
    h1.existence = existence_support
    h1.segmentation = SCORE_NON_TE_SEGMENTATION_WEIGHT * segmentation_support
    h1.te = SCORE_NON_TE_MODEL_WEIGHT * max(te_model_support, 0.0)
    h1.total = h1.existence + h1.segmentation + h1.te

    h2 = JointHypothesisScore(kind=FinalHypothesisKind.TE_UNKNOWN, reason="TE_UNKNOWN")
    h2.existence = existence_support
    h2.segmentation = SCORE_TE_SEGMENTATION_WEIGHT * segmentation_support
    h2.te = te_model_support
    h2.boundary = SCORE_TE_BOUNDARY_WEIGHT * te_boundary_support
    h2.total = h2.existence + h2.segmentation + h2.te + h2.boundary
    h2.hard_veto = compute_te_unknown_hard_veto(existence, segmentation, te_alignment,
                                                boundary)

    h3 = JointHypothesisScore(kind=FinalHypothesisKind.TE_RESOLVED, reason="TE_RESOLVED")
    h3.existence = existence_support
    h3.segmentation = SCORE_TE_SEGMENTATION_WEIGHT * segmentation_support
    h3.te = te_model_support
    h3.boundary = SCORE_TE_BOUNDARY_WEIGHT * te_boundary_support
    h3.total = h3.existence + h3.segmentation + h3.te + h3.boundary
    h3.hard_veto = compute_te_resolved_hard_veto(existence, segmentation, te_alignment,
                                                 boundary)

    # A vetoed hypothesis sorts to the bottom by score substitution rather than
    # being removed, so `runner_up` is always populated. Ties break toward the
    # HIGHER kind, i.e. toward the more specific hypothesis: h3 (resolved) over
    # h2 (unknown) over h1 over h0 when the totals are equal.
    ranked = sorted([h0, h1, h2, h3],
                    key=lambda h: (-(_NEG_INF if h.hard_veto else h.total), -int(h.kind)))
    result.best = ranked[0]
    result.runner_up = ranked[1]

    explanation_decision = evaluate_event_explanations(existence, segmentation,
                                                       te_alignment, boundary,
                                                       clip_insert_concordance)
    result.explanation_decision = explanation_decision
    result.best_explanation = explanation_kind_name(explanation_decision.best.kind)
    result.explanation_residual = serialize_residual(explanation_decision.best.residual)
    result.explanation_path = serialize_explanation_path(explanation_decision.best)
    result.emit_te_call = explanation_decision.emit_te_call
    result.emit_unknown_te = explanation_decision.emit_unknown_te
    result.emit_evidence_te_call = explanation_decision.emit_evidence_te_call
    result.final_qc = explanation_decision.final_qc

    _populate_joint_diagnostics(result, posterior, latent_lfdr, certificate,
                                robust_mechanistic, te_structure)

    # PRIMARY EMISSION GATE.
    robust_te_lfdr_pass = (result.robust_mechanistic_qc == "PASS_TE_LFDR"
                           and result.robust_mechanistic_worst_case_lfdr <= TARGET_Q)
    structural_sanity = (segmentation.has_insert_seq
                         and has_te_sequence_evidence(te_alignment))
    if result.emit_te_call and not (robust_te_lfdr_pass and structural_sanity):
        result.emit_te_call = False
        result.emit_unknown_te = False
        result.emit_evidence_te_call = has_te_sequence_evidence(te_alignment)
        result.final_qc = ("TE_AMBIGUOUS" if result.emit_evidence_te_call
                           else "REFERENCE_OR_ARTIFACT")

    if not result.emit_te_call and should_emit_structural_event_call(
            existence, segmentation, boundary, posterior,
            certificate.ref_conflict_signal,
            explanation_decision.best.kind == ExplanationKind.INSERTION_NON_TE):
        result.emit_structural_event_call = True
        result.emit_unknown_te = True
        result.emit_evidence_te_call = True
        result.final_qc = "PASS_STRUCTURAL_INSERTION"
        result.best = h1
        result.best.hard_veto = False
        result.best.reason = "STRUCTURAL_INSERTION"

    if result.emit_te_call:
        result.best = h2 if result.emit_unknown_te else h3
        result.best.hard_veto = False
        result.best.reason = "TE_UNKNOWN" if result.emit_unknown_te else "TE_RESOLVED"
    return result


def _populate_joint_diagnostics(result: JointDecisionResult,
                                posterior: LocalHypothesisPosterior,
                                latent_lfdr: LatentMechanismEvidence,
                                certificate, robust_mechanistic: dict,
                                te_structure: SequenceExplanation) -> None:
    """Pure plumbing, split out so the decision above stays legible."""
    result.te_posterior = posterior.te
    result.non_te_posterior = posterior.non_te
    result.artifact_posterior = posterior.artifact
    result.te_vs_artifact_log_odds = posterior.te_vs_artifact_log_odds
    result.te_vs_non_te_log_odds = posterior.te_vs_non_te_log_odds
    result.posterior_qc = posterior.qc
    result.latent_mechanism = latent_lfdr.latent_mechanism
    result.family_activity_prior = latent_lfdr.family_activity_prior
    result.lfdr = latent_lfdr.lfdr
    result.worst_case_lfdr = latent_lfdr.worst_case_lfdr
    result.lfdr_qc = latent_lfdr.lfdr_qc
    result.mechanistic_lower_log_bf_te_vs_artifact = certificate.lower_log_bf_te_vs_artifact
    result.mechanistic_lower_log_bf_te_vs_non_te = certificate.lower_log_bf_te_vs_non_te
    result.mechanistic_raw_log_bf_te_vs_artifact = certificate.raw_log_bf_te_vs_artifact
    result.mechanistic_raw_log_bf_te_vs_non_te = certificate.raw_log_bf_te_vs_non_te
    result.mechanistic_ref_conflict_signal = certificate.ref_conflict_signal
    result.mechanistic_ambiguity_width = certificate.ambiguity_width
    result.mechanistic_blocks = blocks_module.serialize_blocks(certificate)
    result.robust_mechanistic_lfdr = robust_mechanistic["lfdr"]
    result.robust_mechanistic_worst_case_lfdr = robust_mechanistic["worst_case_lfdr"]
    result.robust_mechanistic_qc = robust_mechanistic["qc"]
    result.te_structure_path = structure_module.serialize_te_sequence_path(te_structure)
    result.te_structure_log_evidence = te_structure.te_structure_log_evidence
    result.nonte_structure_log_evidence = te_structure.nonte_structure_log_evidence
    result.artifact_structure_log_evidence = te_structure.artifact_structure_log_evidence
    result.te_structure_path_confidence = te_structure.structure_path_confidence
    result.polyA_posterior = te_structure.polyA_posterior
    result.transduction_posterior = te_structure.transduction_posterior
    result.te_core_coverage = te_structure.te_core_coverage
    result.unexplained_high_complexity_bp = te_structure.unexplained_high_complexity_bp
