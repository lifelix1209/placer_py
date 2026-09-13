from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from placer_py.redesign.models import TeExplanation


LEDGER_FIELDS = [
    "candidate_id",
    "label",
    "te_posterior",
    "non_te_posterior",
    "artifact_posterior",
    "lfdr",
    "family",
    "subfamily",
    "te_identity",
    "te_query_coverage",
    "te_orientation",
    "sequence_len",
    "sequence_gc",
    "sequence_entropy",
    "sequence_low_complexity",
    "raw_cigar_insert_reads",
    "max_raw_cigar_insert_len",
    "split_read_support",
    "left_clip_reads",
    "right_clip_reads",
    "ref_span_reads",
    "low_mapq_ref_span_reads",
    "forward_reads",
    "reverse_reads",
    "local_depth",
    "final_qc",
    "posterior_qc",
    "best_hypothesis",
    "emit_te_call",
    "emit_unknown_te",
    "emit_evidence_te_call",
    "te_vs_artifact_log_odds",
    "te_vs_non_te_log_odds",
    "latent_mechanism",
    "robust_mechanistic_lfdr",
    "te_alignment_qc",
    "sequence_model_label",
    "boundary_qc",
    "segmentation_qc",
    "reference_qc",
    "resolved_reference_chrom",
    "reference_breakpoint0",
    "reference_right_breakpoint0",
    "left_ref_anchor_len",
    "right_ref_anchor_len",
    "left_ref_anchor_identity",
    "right_ref_anchor_identity",
    "remap_qc",
    "insert_core_len",
    "tsd_type",
    "tsd_len",
    "tsd_sequence",
    "tsd_bg_p",
    "tsd_qc",
    "reference_n_fraction",
    "reference_max_n_run",
    "seq_core_fraction",
    "seq_core_contiguity",
    "seq_polya_fraction",
    "seq_low_complexity_fraction",
    "seq_unexplained_fraction",
    "seq_family_margin",
    "seq_explanation_qc",
    "worst_case_lfdr",
    "target_q",
    "te_log_evidence",
    "nonte_log_evidence",
    "artifact_log_evidence",
    "lfdr_qc",
    "high_precision_score",
    "emit_te_high_precision",
]


def write_ledger(path: str | Path, explanations: Iterable[TeExplanation]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for explanation in explanations:
            writer.writerow(explanation.to_tsv_fields())
