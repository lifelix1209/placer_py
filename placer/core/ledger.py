"""
The two records finalization works on: a ledger row and a final call.

Ported from `struct EvidenceLedgerRow` and `struct FinalCall` in
`include/pipeline.h`, pinned by `tests/test_31_outputs.py`.

WHY THERE ARE TWO, and the distinction matters for reading any PLACER output.

A LEDGER ROW is every candidate the pipeline examined, whatever it concluded.
It is the sample's own null set as well as its candidate set -- the
overdispersion the genotyper uses is estimated over every row, and e-BH's m
counts every locus, selected or not. So the ledger must contain the
rejections, and a ledger filtered to the calls would silently bias both.

A FINAL CALL is a row promoted to an answer, carrying the genotype, the TSD, the
family commitment and the selection outcome. Finalization's job is to decide
which rows become calls, testing each locus once so that one event does not
become several calls.

THE SCHEMA IS THE SEAM. `placer/schema.py` pins which columns the decision
layer reads and with what dtype; this module is the in-memory form of the same
contract. A field added here that is not in the schema is a diagnostic; a field
in the schema that is missing here is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from placer.core.policy import EventGenotypeInput


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
    #: The best hit's class and superfamily (`core/taxonomy.py`), its strand
    #: relative to the reference, and where it lands on an element of what
    #: length -- carried on the row so a call promoted from it keeps them.
    te_annotation_class: str = "NA"
    te_annotation_order: str = "NA"
    te_strand: str = "NA"
    te_consensus_start: int = -1
    te_consensus_end: int = -1
    te_element_length: int = -1
    #: The oriented poly(A) and 3' transduction (`core/element_structure.py`).
    polya_len: int = 0
    transduction_len: int = -1
    ltr_form: str = "NA"
    #: Named from the insert's two ends, assembled from clips (no read spans it).
    te_from_clip_sides: bool = False
    #: TEBench's TE rule on the insert (`te_classifier.measure_te_coverage`).
    te_union_covered_bp: int = 0
    te_union_coverage: float = 0.0
    te_dominant_family: str = "NA"
    te_dominant_class: str = "NA"
    te_dominant_covered_bp: int = 0
    #: The per-class likelihood ratios (`core/mechanism.py`) and the
    #: shifted-breakpoint decoys (`core/locus_evidence.py`) the decision is
    #: taken on (`core/mechanism_selection.py`).
    mech_log_lr_vs_non_te: float = 0.0
    mech_log_lr_vs_artifact: float = 0.0
    mech_decoy_count: int = 0
    mech_decoy_mean_exp_linkage: float = 0.0
    mech_terms: str = "NA"
    #: The inputs the sequence term was computed from.
    mech_aligned_len: int = 0
    mech_sequence_term: float = 0.0
    #: Set at finalization by `core/mechanism_selection.py` on each locus's
    #: representative row: its e-value, whether the decision selects it as a
    #: TE call or as a structural call, and the precise breakpoint it is placed
    #: at (-1: its own).
    mech_e_value: float = 0.0
    mech_ebh_selected: bool = False
    mech_structural_selected: bool = False
    mech_call_pos: int = -1
    #: Inside an alignment-collapse region, where the e-value was set to 0
    #: (`mechanism_selection.collapse_region_items`).
    mech_collapse_region: bool = False
    family_alignment_resolved: bool = False
    final_qc: str = "NA"
    posterior_qc: str = "NA"
    lfdr_qc: str = "NA"
    candidate_retention_reason: str = "NA"
    alt_struct_reads: int = 0
    alt_split_reads: int = 0
    alt_indel_reads: int = 0
    #: Of those, counted by the same-allele carrier rule (`events.CARRIER_WINDOW_BP`).
    alt_carrier_reads: int = 0
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
    #: The alt reads' own measurements of the insertion's length, one per read
    #: (`EventReadEvidence.alt_measured_lengths`): CIGAR or SA-implied, never a
    #: clip. A diagnostic for replay; no decision reads it.
    alt_measured_lengths: list[int] = field(default_factory=list)
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
    # ---- RECORDED FOR REPLAY: no decision reads these ----------------------
    # `compare=False` on every one, so the per-bin de-duplication, which
    # compares rows field by field, and with it every call, is unchanged by them.
    #: The TSD term's inputs (`locus_evidence.LocusScore`): the locus's own TSD
    #: length (-1: not recorded) and model p, and the decoys split by whether
    #: they found a duplication, so a change of `p_present` replays exactly.
    mech_tsd_len: int = field(default=-1, compare=False)
    mech_tsd_p_present: float = field(default=-1.0, compare=False)
    mech_decoy_tsd_p_present: float = field(default=-1.0, compare=False)
    mech_decoy_tsd_hits: int = field(default=-1, compare=False)
    mech_decoy_sum_absent: float = field(default=0.0, compare=False)
    mech_decoy_sum_present_per_p: float = field(default=0.0, compare=False)
    #: The allele-level tally (`events.collect_allele_evidence`): the allele's
    #: length (-1: not computed), its carriers (offset from bp_left, length,
    #: k-mer similarity, whether it is an own alt read), and three tallies --
    #: `bylen`, `byseq` and `wide` (`events.AlleleEvidence`) -- each with its
    #: alt and ref reads, span (relative to bp_left), carriers beyond the row's
    #: own, and the counts term over the span (the row's own counts term when
    #: there are none).
    allele_length: int = field(default=-1, compare=False)
    allele_carrier_offsets: list[int] = field(default_factory=list, compare=False)
    allele_carrier_lengths: list[int] = field(default_factory=list, compare=False)
    allele_carrier_similarity: list[float] = field(default_factory=list, compare=False)
    allele_carrier_own: list[int] = field(default_factory=list, compare=False)
    allele_bylen_alt_reads: int = field(default=-1, compare=False)
    allele_bylen_ref_reads: int = field(default=-1, compare=False)
    allele_bylen_span_lo: int = field(default=0, compare=False)
    allele_bylen_span_hi: int = field(default=0, compare=False)
    allele_bylen_extra_carriers: int = field(default=0, compare=False)
    mech_counts_allele_bylen: float = field(default=0.0, compare=False)
    allele_byseq_alt_reads: int = field(default=-1, compare=False)
    allele_byseq_ref_reads: int = field(default=-1, compare=False)
    allele_byseq_span_lo: int = field(default=0, compare=False)
    allele_byseq_span_hi: int = field(default=0, compare=False)
    allele_byseq_extra_carriers: int = field(default=0, compare=False)
    mech_counts_allele_byseq: float = field(default=0.0, compare=False)
    allele_wide_alt_reads: int = field(default=-1, compare=False)
    allele_wide_ref_reads: int = field(default=-1, compare=False)
    allele_wide_span_lo: int = field(default=0, compare=False)
    allele_wide_span_hi: int = field(default=0, compare=False)
    allele_wide_extra_carriers: int = field(default=0, compare=False)
    mech_counts_allele_wide: float = field(default=0.0, compare=False)
    #: The counts term's local null, measured (`events.COUNTS_BACKGROUND_WINDOWS`):
    #: the model's error rate for the row's own counts term and for each tally
    #: (-1: not recorded), and each tally's signal where the insertion is not --
    #: read-windows spanned, and those showing the signal (-1: not measured).
    mech_counts_eps: float = field(default=-1.0, compare=False)
    counts_bg_own_reads: int = field(default=-1, compare=False)
    counts_bg_own_hits: int = field(default=-1, compare=False)
    allele_bylen_eps: float = field(default=-1.0, compare=False)
    allele_bylen_bg_reads: int = field(default=-1, compare=False)
    allele_bylen_bg_hits: int = field(default=-1, compare=False)
    allele_byseq_eps: float = field(default=-1.0, compare=False)
    allele_byseq_bg_reads: int = field(default=-1, compare=False)
    allele_byseq_bg_hits: int = field(default=-1, compare=False)
    allele_wide_eps: float = field(default=-1.0, compare=False)
    allele_wide_bg_reads: int = field(default=-1, compare=False)
    allele_wide_bg_hits: int = field(default=-1, compare=False)


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
    alt_indel_reads: int = 0
    #: `pos` before finalization places the call: the evaluated hypothesis's
    #: own position.
    hypothesis_pos: int = -1
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
    te_element_length: int = -1
    polya_len: int = 0
    transduction_len: int = -1
    ltr_form: str = "NA"
    #: Named from the insert's two ends, assembled from clips (no read spans it).
    te_from_clip_sides: bool = False
    #: TEBench's TE rule on the insert (`te_classifier.measure_te_coverage`).
    te_union_covered_bp: int = 0
    te_union_coverage: float = 0.0
    te_dominant_family: str = "NA"
    te_dominant_class: str = "NA"
    te_dominant_covered_bp: int = 0
    #: The per-class likelihood ratios (`core/mechanism.py`) and the
    #: shifted-breakpoint decoys (`core/locus_evidence.py`) the decision is
    #: taken on (`core/mechanism_selection.py`).
    mech_log_lr_vs_non_te: float = 0.0
    mech_log_lr_vs_artifact: float = 0.0
    mech_decoy_count: int = 0
    mech_decoy_mean_exp_linkage: float = 0.0
    mech_terms: str = "NA"
    #: The inputs the sequence term was computed from.
    mech_aligned_len: int = 0
    mech_sequence_term: float = 0.0
    mech_e_value: float = 0.0
    mech_ebh_selected: bool = False
    mech_structural_selected: bool = False
    mech_call_pos: int = -1
    #: Inside an alignment-collapse region, where the e-value was set to 0
    #: (`mechanism_selection.collapse_region_items`).
    mech_collapse_region: bool = False
    #: The TE alignment's own best family and subfamily, whatever the scan's
    #: joint decision (`core/policy.py`) then did to `family` (it overwrites it
    #: with UNKNOWN for a call it deems structural). Finalization names a TE
    #: call from these.
    te_best_family: str = "NA"
    te_best_subfamily: str = "NA"
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
