"""
Writing the three output files, and the column contracts they carry.

Ported from the writers in `src/main.cpp`, pinned by `tests/test_31_outputs.py`.

THREE FILES, AND WHAT EACH IS FOR:

  * `scientific.txt` -- the TE calls, with a run summary header. The answer.
  * `structural_calls.tsv` -- the structural insertions the TE-calibrated mode
    set aside. SELECTED calls, not rejects, and carrying the IDENTICAL schema so
    the two can be concatenated or diffed without reshaping.
  * `evidence_ledger.tsv` -- every candidate examined, whatever the verdict.
    This is the sample's own null set as well as its candidate set; see
    `placer_py/ledger.py`.

THE COLUMN ORDER IS THE CONTRACT. `placer_py/schema.py` pins which columns the
decision layer READS and with what dtype; this module pins the order they are
WRITTEN in. A reader that indexes by position -- and every downstream script
eventually does -- breaks silently if the order changes, so the header lists
here are the single source of truth for both files and are asserted against the
schema in the tests.

TWO DEBUG COLUMNS are optional (`insert_seq`, `support_qnames`) because they
dominate the file size on a whole-genome run: insert sequences are kilobases
each and support lists are hundreds of read names. They are off by default and
the header changes with them, which is why every reader must parse the header
rather than assume a fixed layout.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from placer_py.ledger import EvidenceLedgerRow, FinalCall

#: Columns of `scientific.txt` and `structural_calls.tsv`, in order, WITHOUT
#: the optional `insert_seq` (which is inserted after `consensus_len`).
FINAL_CALL_COLUMNS: tuple[str, ...] = (
    "chrom", "pos", "bp_left", "bp_right", "te", "family", "subfamily",
    "family_status", "strand", "insert_len",
    "support_reads", "alt_struct_reads", "raw_cigar_insert_reads",
    "max_raw_cigar_insert_len", "ref_span_reads", "low_mapq_ref_span_reads",
    "gt", "af", "gq",
    "best_te_identity", "best_te_query_coverage", "cross_family_margin",
    "te_sequence_model_label", "te_sequence_model_score",
    "te_sequence_model_gc", "te_sequence_model_entropy",
    "te_sequence_model_tandem_fraction", "te_sequence_model_low_complexity_fraction",
    "te_sequence_model_jsd_k5", "te_sequence_model_jsd_k6",
    "te_sequence_model_k9_containment",
    "te_annotation_confidence", "te_annotation_class", "te_annotation_order",
    "te_annotation_intervals",
    "te_annotation_residual_fraction", "te_annotation_masked_fraction",
    "tsd_type", "tsd_len", "tsd_seq", "tsd_bg_p", "tsd_mismatches",
    "left_flank_align_len", "right_flank_align_len", "consensus_len",
    "qc", "best_explanation", "explanation_residual", "explanation_path",
    "te_structure_path", "te_structure_log_evidence", "nonte_structure_log_evidence",
    "artifact_structure_log_evidence", "te_structure_path_confidence",
    "polyA_posterior", "transduction_posterior",
    "te_core_coverage", "unexplained_high_complexity_bp",
    "te_posterior", "non_te_posterior", "artifact_posterior",
    "te_vs_artifact_log_odds", "te_vs_non_te_log_odds", "posterior_qc",
    "latent_mechanism", "family_activity_prior", "lfdr",
    "worst_case_lfdr", "lfdr_qc",
    "mechanistic_lower_log_bf_te_vs_artifact", "mechanistic_lower_log_bf_te_vs_non_te",
    "mechanistic_ref_conflict_signal", "mechanistic_ambiguity_width",
    "mechanistic_blocks",
    "mechanistic_raw_log_bf_te_vs_artifact", "mechanistic_raw_log_bf_te_vs_non_te",
    "ebh_e_value", "ebh_selected",
    "robust_mechanistic_lfdr", "robust_mechanistic_worst_case_lfdr",
    "robust_mechanistic_qc",
    "conformal_null_p", "conformal_by_threshold", "conformal_dominated_nulls",
    "conformal_null_count", "conformal_qc",
    "bp_ci_width", "bp_posterior_entropy",
    "te_consensus_start", "te_consensus_end",
)

#: Where `insert_seq` goes when enabled.
_FINAL_CALL_INSERT_SEQ_AFTER = "consensus_len"

#: Columns of `evidence_ledger.tsv`, in order, without the two optional ones.
EVIDENCE_LEDGER_COLUMNS: tuple[str, ...] = (
    "chrom", "pos", "bp_left", "bp_right", "coverage_left", "coverage_right",
    "family", "subfamily",
    "final_qc", "posterior_qc", "lfdr_qc", "candidate_retention_reason",
    "alt_struct_reads", "alt_split_reads", "alt_indel_reads",
    "alt_left_clip_reads", "alt_right_clip_reads",
    "raw_cigar_insert_reads", "max_raw_cigar_insert_len",
    "full_context_input_reads", "partial_context_input_reads",
    "left_anchor_input_reads", "right_anchor_input_reads",
    "input_event_reads", "event_consensus_len",
    "left_flank_align_len", "right_flank_align_len",
    "ref_span_reads", "support_qname_count",
    "best_te_identity", "best_te_query_coverage", "cross_family_margin",
    "te_structure_path", "te_structure_log_evidence", "nonte_structure_log_evidence",
    "artifact_structure_log_evidence", "te_structure_path_confidence",
    "polyA_posterior", "transduction_posterior",
    "te_core_coverage", "unexplained_high_complexity_bp",
    "te_posterior", "non_te_posterior", "artifact_posterior",
    "lfdr", "worst_case_lfdr",
    "mechanistic_lower_log_bf_te_vs_artifact", "mechanistic_lower_log_bf_te_vs_non_te",
    "mechanistic_ref_conflict_signal", "mechanistic_ambiguity_width",
    "mechanistic_blocks",
    "robust_mechanistic_lfdr", "robust_mechanistic_worst_case_lfdr",
    "robust_mechanistic_qc",
    "conformal_null_p", "conformal_by_threshold", "conformal_dominated_nulls",
    "conformal_null_count", "conformal_qc",
)

_LEDGER_INSERT_SEQ_AFTER = "event_consensus_len"
_LEDGER_SUPPORT_QNAMES_AFTER = "support_qname_count"


def _insert_after(columns: Sequence[str], anchor: str, name: str) -> list[str]:
    out = list(columns)
    out.insert(out.index(anchor) + 1, name)
    return out


def final_call_header(include_insert_seq: bool = False) -> list[str]:
    columns = list(FINAL_CALL_COLUMNS)
    if include_insert_seq:
        columns = _insert_after(columns, _FINAL_CALL_INSERT_SEQ_AFTER, "insert_seq")
    return columns


def evidence_ledger_header(include_insert_seq: bool = False,
                           include_support_qnames: bool = False) -> list[str]:
    columns = list(EVIDENCE_LEDGER_COLUMNS)
    if include_insert_seq:
        columns = _insert_after(columns, _LEDGER_INSERT_SEQ_AFTER, "insert_seq")
    if include_support_qnames:
        columns = _insert_after(columns, _LEDGER_SUPPORT_QNAMES_AFTER, "support_qnames")
    return columns


def _text(value: str) -> str:
    """Empty becomes NA.

    A blank field is ambiguous in a TSV -- it reads as either "missing" or "the
    empty string" depending on the parser -- so every optional text column is
    written as NA when unset.
    """
    return value if value else "NA"


def _number(value) -> str:
    """Full double round-trip for floats, plain text for ints.

    `repr` rather than a fixed precision: the ledger's log Bayes factors are
    re-read by the dependency calibration, and rounding them in the file would
    change the measured bound.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def serialize_support_qnames(qnames: Sequence[str]) -> str:
    return ",".join(qnames) if qnames else "NA"


def final_call_row(call: FinalCall, include_insert_seq: bool = False) -> list[str]:
    """One call as a list of fields, in `final_call_header` order."""
    values = [
        call.chrom, call.pos, call.bp_left, call.bp_right,
        _text(call.te_name), call.family, call.subfamily,
        # The label and the ABSTENTION are separate columns' worth of meaning in
        # one: a library may contain a family literally named "Unknown", so the
        # status has to say whether the label was committed.
        "COMMITTED" if call.family_committed else "ABSTAINED",
        call.strand, call.insert_len,
        call.support_reads, call.alt_struct_reads, call.raw_cigar_insert_reads,
        call.max_raw_cigar_insert_len, call.ref_span_reads, call.low_mapq_ref_span_reads,
        call.genotype, call.af, call.gq,
        call.best_te_identity, call.best_te_query_coverage, call.cross_family_margin,
        call.te_sequence_model_label, call.te_sequence_model_score,
        call.te_sequence_model_gc, call.te_sequence_model_entropy,
        call.te_sequence_model_tandem_fraction,
        call.te_sequence_model_low_complexity_fraction,
        call.te_sequence_model_jsd_k5, call.te_sequence_model_jsd_k6,
        call.te_sequence_model_k9_containment,
        call.te_annotation_confidence, call.te_annotation_class,
        call.te_annotation_order, call.te_annotation_intervals,
        call.te_annotation_residual_fraction, call.te_annotation_masked_fraction,
        call.tsd_type, call.tsd_len, _text(call.tsd_seq), call.tsd_bg_p,
        call.tsd_mismatches,
        call.left_flank_align_len, call.right_flank_align_len, call.event_consensus_len,
    ]
    if include_insert_seq:
        values.append(call.insert_seq)
    values += [
        call.final_qc, call.best_explanation, call.explanation_residual,
        call.explanation_path,
        call.te_structure_path, call.te_structure_log_evidence,
        call.nonte_structure_log_evidence, call.artifact_structure_log_evidence,
        call.te_structure_path_confidence,
        call.polyA_posterior, call.transduction_posterior,
        call.te_core_coverage, call.unexplained_high_complexity_bp,
        call.te_posterior, call.non_te_posterior, call.artifact_posterior,
        call.te_vs_artifact_log_odds, call.te_vs_non_te_log_odds, call.posterior_qc,
        call.latent_mechanism, call.family_activity_prior, call.lfdr,
        call.worst_case_lfdr, call.lfdr_qc,
        call.mechanistic_lower_log_bf_te_vs_artifact,
        call.mechanistic_lower_log_bf_te_vs_non_te,
        call.mechanistic_ref_conflict_signal, call.mechanistic_ambiguity_width,
        call.mechanistic_blocks,
        call.mechanistic_raw_log_bf_te_vs_artifact,
        call.mechanistic_raw_log_bf_te_vs_non_te,
        call.ebh_e_value, call.ebh_selected,
        call.robust_mechanistic_lfdr, call.robust_mechanistic_worst_case_lfdr,
        call.robust_mechanistic_qc,
        call.conformal_null_p, call.conformal_by_threshold,
        call.conformal_dominated_nulls, call.conformal_null_count, call.conformal_qc,
        call.bp_ci_width, call.bp_posterior_entropy,
        call.te_consensus_start, call.te_consensus_end,
    ]
    return [_number(value) for value in values]


def evidence_ledger_row(row: EvidenceLedgerRow, include_insert_seq: bool = False,
                        include_support_qnames: bool = False) -> list[str]:
    values = [
        row.chrom, row.pos, row.bp_left, row.bp_right,
        row.coverage_left, row.coverage_right, row.family, row.subfamily,
        row.final_qc, row.posterior_qc, row.lfdr_qc, row.candidate_retention_reason,
        row.alt_struct_reads, row.alt_split_reads, row.alt_indel_reads,
        row.alt_left_clip_reads, row.alt_right_clip_reads,
        row.raw_cigar_insert_reads, row.max_raw_cigar_insert_len,
        row.full_context_input_reads, row.partial_context_input_reads,
        row.left_anchor_input_reads, row.right_anchor_input_reads,
        row.input_event_reads, row.event_consensus_len,
    ]
    if include_insert_seq:
        values.append(row.insert_seq)
    values += [row.left_flank_align_len, row.right_flank_align_len,
               row.ref_span_reads, len(row.support_qnames)]
    if include_support_qnames:
        values.append(serialize_support_qnames(row.support_qnames))
    values += [
        row.best_te_identity, row.best_te_query_coverage, row.cross_family_margin,
        row.te_structure_path, row.te_structure_log_evidence,
        row.nonte_structure_log_evidence, row.artifact_structure_log_evidence,
        row.te_structure_path_confidence,
        row.polyA_posterior, row.transduction_posterior,
        row.te_core_coverage, row.unexplained_high_complexity_bp,
        row.te_posterior, row.non_te_posterior, row.artifact_posterior,
        row.lfdr, row.worst_case_lfdr,
        row.mechanistic_lower_log_bf_te_vs_artifact,
        row.mechanistic_lower_log_bf_te_vs_non_te,
        row.mechanistic_ref_conflict_signal, row.mechanistic_ambiguity_width,
        row.mechanistic_blocks,
        row.robust_mechanistic_lfdr, row.robust_mechanistic_worst_case_lfdr,
        row.robust_mechanistic_qc,
        row.conformal_null_p, row.conformal_by_threshold,
        row.conformal_dominated_nulls, row.conformal_null_count, row.conformal_qc,
    ]
    return [_number(value) for value in values]


def _summary_lines(result) -> list[str]:
    """The run summary that heads `scientific.txt`.

    Includes the MEASURED calibration constants -- sigma, the penalty, the cap,
    the null count -- because a reader needs them to interpret the e-values in
    the rows below. A file reporting selections without reporting what they were
    calibrated against cannot be audited.
    """
    return [
        "#PLACER streaming pipeline summary",
        f"total_reads\t{result.total_reads}",
        f"gate1_passed\t{result.gate1_passed}",
        f"processed_bins\t{result.processed_bins}",
        f"components\t{result.built_components}",
        f"event_consensus_calls\t{result.event_consensus_calls}",
        f"genotype_calls\t{result.genotype_calls}",
        f"final_pass_calls\t{result.final_pass_calls}",
        f"estimated_overdispersion\t{_number(result.estimated_overdispersion)}",
        f"estimated_dependency_penalty\t{_number(result.estimated_dependency_penalty)}",
        f"estimated_dependency_penalty_non_te\t"
        f"{_number(result.estimated_dependency_penalty_non_te)}",
        f"estimated_dependency_sigma\t{_number(result.estimated_dependency_sigma)}",
        f"estimated_dependency_sigma_non_te\t"
        f"{_number(result.estimated_dependency_sigma_non_te)}",
        f"dependency_penalty_cap_log\t{_number(result.dependency_penalty_cap_log)}",
        f"dependency_penalty_null_count\t{result.dependency_penalty_null_count}",
        f"dependency_penalty_estimated\t"
        f"{1 if result.dependency_penalty_estimated else 0}",
    ]


def render_scientific_txt(result, include_insert_seq: bool = False) -> str:
    """The summary block, a blank line, then the call table.

    The blank line before the `#`-prefixed header is what lets a reader skip the
    summary by reading to the first empty line, which is how the C++ format is
    meant to be parsed.
    """
    lines = _summary_lines(result)
    lines.append("")
    lines.append("#" + "\t".join(final_call_header(include_insert_seq)))
    for call in result.final_calls:
        lines.append("\t".join(final_call_row(call, include_insert_seq)))
    return "\n".join(lines) + "\n"


def render_structural_calls_tsv(result, include_insert_seq: bool = False) -> str:
    """The set-aside structural calls, with the IDENTICAL call schema."""
    lines = [f"structural_calls\t{len(result.structural_calls)}", ""]
    lines.append("#" + "\t".join(final_call_header(include_insert_seq)))
    for call in result.structural_calls:
        lines.append("\t".join(final_call_row(call, include_insert_seq)))
    return "\n".join(lines) + "\n"


def render_evidence_ledger_tsv(result, include_insert_seq: bool = False,
                               include_support_qnames: bool = False) -> str:
    """Every examined candidate. No leading `#`: the ledger is data, not a
    report, and its first line is the header a TSV reader expects."""
    lines = ["\t".join(evidence_ledger_header(include_insert_seq,
                                              include_support_qnames))]
    for row in result.evidence_ledger:
        lines.append("\t".join(evidence_ledger_row(row, include_insert_seq,
                                                   include_support_qnames)))
    return "\n".join(lines) + "\n"
