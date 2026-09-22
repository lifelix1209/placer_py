"""
The two records finalization works on: a ledger row and a final call.

Ported from `struct EvidenceLedgerRow` and `struct FinalCall` in
`include/pipeline.h`, pinned by `tests/test_31_outputs.py`.

WHY THERE ARE TWO, and the distinction matters for reading any PLACER output.

A LEDGER ROW is every candidate the pipeline examined, whatever it concluded.
It is the sample's own null set as well as its candidate set -- the dependency
bound in `placer_py/dependency.py` and the conformal null set in
`placer_py/conformal.py` are both measured from rows that were NOT selected. So
the ledger must contain the rejections, and a ledger filtered to the calls would
silently destroy both calibrations.

A FINAL CALL is a row promoted to an answer, carrying the genotype, the TSD, the
family commitment and the selection outcome. Finalization's job is to decide
which rows become calls, and much of it is about NOT letting one event become
several calls.

THE SCHEMA IS THE SEAM. `placer_py/schema.py` pins which columns the decision
layer reads and with what dtype; this module is the in-memory form of the same
contract. A field added here that is not in the schema is a diagnostic; a field
in the schema that is missing here is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from placer_py.policy import EventGenotypeInput


@dataclass
class EvidenceLedgerRow:
    """One examined candidate, whatever the pipeline concluded about it."""

    chrom: str = ""
    tid: int = -1
    pos: int = -1
    bp_left: int = -1
    bp_right: int = -1
    coverage_left: int = -1
    coverage_right: int = -1
    #: The bin that owned this candidate. Two rows from different owners at the
    #: same position are the same event seen twice, and finalization uses these
    #: to recognise that.
    owner_context_left: int = -1
    owner_context_right: int = -1
    family: str = "NA"
    subfamily: str = "NA"
    family_alignment_resolved: bool = False
    final_qc: str = "NA"
    posterior_qc: str = "NA"
    lfdr_qc: str = "NA"
    candidate_retention_reason: str = "NA"
    alt_struct_reads: int = 0
    alt_split_reads: int = 0
    alt_indel_reads: int = 0
    alt_left_clip_reads: int = 0
    alt_right_clip_reads: int = 0
    full_context_input_reads: int = 0
    raw_cigar_insert_reads: int = 0
    max_raw_cigar_insert_len: int = 0
    partial_context_input_reads: int = 0
    left_anchor_input_reads: int = 0
    right_anchor_input_reads: int = 0
    input_event_reads: int = 0
    event_consensus_len: int = 0
    left_flank_align_len: int = 0
    right_flank_align_len: int = 0
    insert_seq: str = ""
    ref_span_reads: int = 0
    support_qnames: list[str] = field(default_factory=list)
    best_te_identity: float = 0.0
    best_te_query_coverage: float = 0.0
    cross_family_margin: float = 0.0
    te_structure_path: str = "NA"
    te_structure_log_evidence: float = 0.0
    nonte_structure_log_evidence: float = 0.0
    artifact_structure_log_evidence: float = 0.0
    te_structure_path_confidence: float = 0.0
    polyA_posterior: float = 0.0
    transduction_posterior: float = 0.0
    te_core_coverage: float = 0.0
    unexplained_high_complexity_bp: int = 0
    te_posterior: float = 0.0
    non_te_posterior: float = 0.0
    artifact_posterior: float = 0.0
    lfdr: float = 1.0
    worst_case_lfdr: float = 1.0
    mechanistic_lower_log_bf_te_vs_artifact: float = 0.0
    mechanistic_lower_log_bf_te_vs_non_te: float = 0.0
    #: PENALTY-FREE. Finalization subtracts the measured dependency bound from
    #: these to produce the certified values above -- the bound is a property of
    #: the whole run, so it cannot be applied when the row is written.
    mechanistic_raw_log_bf_te_vs_artifact: float = 0.0
    mechanistic_raw_log_bf_te_vs_non_te: float = 0.0
    mechanistic_ref_conflict_signal: float = 0.0
    mechanistic_ambiguity_width: float = 0.0
    mechanistic_blocks: str = "NA"
    robust_mechanistic_lfdr: float = 1.0
    robust_mechanistic_worst_case_lfdr: float = 1.0
    robust_mechanistic_qc: str = "TE_LFDR_HIGH"
    conformal_null_p: float = 1.0
    conformal_by_threshold: float = 0.0
    conformal_dominated_nulls: int = 0
    conformal_null_count: int = 0
    conformal_qc: str = "CONFORMAL_NOT_EVALUATED"


@dataclass
class FinalCall:
    """A row promoted to an answer."""

    chrom: str = ""
    tid: int = -1
    pos: int = -1
    bp_left: int = -1
    bp_right: int = -1

    #: Breakpoint-position posterior summary. Each supporting signal contributes
    #: a Gaussian kernel at its position, scaled by its precision; `bp_ci_width`
    #: is the 90% credible interval of the mixture. A call can be TE-positive
    #: and breakpoint-imprecise, and these two fields are what say so -- they are
    #: reported rather than used as a gate, because the two questions are
    #: genuinely separate.
    bp_ci_width: float = 0.0
    bp_posterior_entropy: float = 0.0

    window_start: int = -1
    window_end: int = -1
    te_name: str = ""

    tsd_type: str = "NONE"
    tsd_len: int = 0
    tsd_seq: str = "NA"
    tsd_bg_p: float = 1.0
    tsd_mismatches: int = 0

    support_reads: int = 0
    alt_struct_reads: int = 0
    raw_cigar_insert_reads: int = 0
    max_raw_cigar_insert_len: int = 0
    ref_span_reads: int = 0
    low_mapq_ref_span_reads: int = 0
    support_qnames: list[str] = field(default_factory=list)
    genotype: str = "./."
    af: float = 0.0
    gq: int = 0
    #: The exact likelihood inputs the genotype came from. Finalization re-runs
    #: the genotyper with the per-sample overdispersion and MUST reuse these, or
    #: the reported GQ would come from a different model than the one that
    #: decided the call.
    genotype_likelihood_input: EventGenotypeInput = field(
        default_factory=EventGenotypeInput)

    family: str = "NA"
    subfamily: str = "NA"
    #: Value and ABSTENTION are separate, because a TE library may legitimately
    #: contain a family named "Unknown".
    family_committed: bool = False
    #: A resolved sequence annotation may be retained while the event-level
    #: decision abstains on family. The commitment happens only after detection
    #: selection is complete, so family coverage cannot influence breakpoint
    #: choice, de-duplication or emission -- which is what keeps the TE library
    #: from feeding back into detection.
    sequence_family_candidate: str = "NA"
    sequence_subfamily_candidate: str = "NA"
    sequence_family_commit_eligible: bool = False
    strand: str = "NA"
    insert_len: int = 0
    best_te_identity: float = 0.0
    best_te_query_coverage: float = 0.0
    cross_family_margin: float = 0.0
    te_consensus_start: int = -1
    te_consensus_end: int = -1
    te_sequence_model_label: str = "TE_MODEL_UNAVAILABLE"
    te_sequence_model_score: float = 0.0
    te_sequence_model_gc: float = 0.0
    te_sequence_model_entropy: float = 0.0
    te_sequence_model_tandem_fraction: float = 0.0
    te_sequence_model_low_complexity_fraction: float = 0.0
    te_sequence_model_jsd_k5: float = 0.0
    te_sequence_model_jsd_k6: float = 0.0
    te_sequence_model_k9_containment: float = 0.0
    te_annotation_confidence: str = "NA"
    te_annotation_class: str = "NA"
    te_annotation_order: str = "NA"
    te_annotation_intervals: str = "NA"
    te_annotation_residual_fraction: float = 0.0
    te_annotation_masked_fraction: float = 0.0
    left_flank_align_len: int = 0
    right_flank_align_len: int = 0
    event_consensus_len: int = 0
    insert_seq: str = ""
    te_qc: str = "NA"
    final_qc: str = "NA"
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
    mechanistic_lower_log_bf_te_vs_non_te: float = 0.0
    mechanistic_raw_log_bf_te_vs_artifact: float = 0.0
    mechanistic_raw_log_bf_te_vs_non_te: float = 0.0
    mechanistic_ref_conflict_signal: float = 0.0
    mechanistic_ambiguity_width: float = 0.0
    mechanistic_blocks: str = "NA"
    robust_mechanistic_lfdr: float = 1.0
    robust_mechanistic_worst_case_lfdr: float = 1.0
    robust_mechanistic_qc: str = "TE_LFDR_HIGH"
    conformal_null_p: float = 1.0
    conformal_by_threshold: float = 0.0
    conformal_dominated_nulls: int = 0
    conformal_null_count: int = 0
    conformal_qc: str = "CONFORMAL_NOT_EVALUATED"
    #: The e-BH route, reported ALONGSIDE the conformal one rather than
    #: replacing it, so the two can be compared on the same run. That comparison
    #: is the point: e-BH needs the e-values to be e-values, the conformal route
    #: needs no distributional assumption at all, and a locus that survives both
    #: has cleared a model-based argument and an assumption-free one.
    ebh_e_value: float = 0.0
    ebh_selected: bool = False


@dataclass
class FinalCallFilterConfig:
    min_raw_cigar_insert_len_bp: int = 0
    report_mode: str = "TeCalibrated"
