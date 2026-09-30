"""One bin, all the way from reads to final calls and ledger rows.

Ported from `src/pipeline/pipeline_bin_processing_stage.inc`, pinned by
`tests/test_32_pipeline.py`.

THIS IS THE PER-BIN HALF OF THE RUN, and the split from the whole-run half is
the point of having two modules. Everything here knows about one bin and
nothing about the rest of the genome: it clusters, fetches the local reads,
extracts fragments, enumerates breakpoint hypotheses, triages them, and sends
only a shortlist to the expensive stages. `placer/core/finalize.py` is the
other half, and it is the only stage that can measure the run's own null,
control FDR across it, or notice that two bins reported one event.

WHERE THE COST IS, and why the triage exists. Everything up to the shortlist is
linear in reads and cheap. Consensus, segmentation and TE alignment are
thousands of times more expensive per candidate, so the structure here is
mostly about deciding what NOT to send to them -- which is why
`placer/core/hypotheses.py` is a triage stage rather than an evaluation one.

NOTHING HERE READS ANOTHER BIN, and `placer/parallel.py` depends on that:
a bin is a function of the reads that start in it, indexed fetches and the
stateless hooks, and it only APPENDS to the result. That is what lets the scan
be cut at bin boundaries, run on several processes and rejoined in genome
order with byte-identical output. Anything added here that read state left by
an earlier bin would break `--threads` silently -- `tests/test_38_parallel.py`
is what would notice. The per-bin statistics that ARE observable (component
counts, consensus calls, genotype calls) are kept because they appear in
`scientific.txt`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Callable

from placer.alignment import AlignedRead, compute_ref_end
from placer.config import PipelineConfig
from placer.core import breakpoints as bp_module
from placer.core import consensus as consensus_module
from placer.core import events as events_module
from placer.core import fragments as fragments_module
from placer.core import hypotheses as hyp_module
from placer.core import interval_cache as cache_module
from placer.core import locus_evidence as locus_module
from placer.core import policy as policy_module
from placer.core import segmentation as seg_module
from placer.core import te_classifier as te_classifier_module
from placer.core.clustering import ComponentCall, build_component_calls
from placer.core.contracts import StageHooks
from placer.core.ledger import EvidenceLedgerRow, FinalCall
from placer.core.result import PipelineResult
from placer.core.te_classifier import TEAlignmentEvidence

#: How far around a component's seed breakpoints to fetch reads.
LOCAL_EVENT_FETCH_SLACK_BP = 1000
#: Gap below which two local fetches are merged into one.
LOCAL_INTERVAL_MERGE_GAP_BP = 128


def _bin_index_for(read: AlignedRead, bin_size: int) -> int:
    return max(0, read.pos // max(1, bin_size))


def _read_end(read: AlignedRead) -> int:
    """The last reference base a read touches -- where its last signature can be."""
    return max(read.pos, compute_ref_end(read))


def group_reads_into_bins(reads: Iterable[AlignedRead], bin_size: int,
                          bin_range: tuple[int, int | None] | None = None
                          ) -> Iterator[tuple[int, int, list[AlignedRead]]]:
    """Group a stream into `(tid, bin_index, reads)`, in order.

    A BIN GETS EVERY READ THAT OVERLAPS IT, in stream order, and is handed over
    once the stream has passed its end -- a read starting at or beyond the bin's
    end can add nothing to it, since every signature a read carries lies inside
    its own alignment. Components are then kept only by the bin that owns their
    ANCHOR (`process_bin_records`), so each is processed once, with all of its
    reads.

    It used to be "a read belongs to the bin of its START", as in the C++. Then
    a component was discovered only from reads that started in its own bin, and
    the reads that started earlier -- most of them, when reads are longer than
    bins -- formed the same component in the bin they started in, where the
    anchor filter threw it away. An insertion 14 bp into a bin whose ten
    carriers all started in the previous one was never called at all
    (`examples/make_example_data.py`, `l1_minus`); on ultra-long ONT, with reads
    of tens of kb against 10 kb bins, typically about a tenth of an insertion's
    carriers start in its own bin.

    `bin_range` = `(first, last)` restricts the bins emitted to indices in
    `[first, last)` (`last` None for unbounded): a chunk of a parallel run, or a
    `--region`, owns only its own bins even though the reads it was given
    overlap others.

    A GENERATOR, which the coordinate-sorted input is what makes possible, and
    what keeps the scan streaming. The reads held at any moment are those that
    overlap a bin not yet emitted: bounded by depth times read length over bin
    size, not by the region. Draining the stream into a list -- as the caller
    once did -- made a 10 Mb region of ultra-long ONT cost 2.2 GB; each
    AlignedRead holds its full `seq`.
    """
    size = max(1, bin_size)
    first, last = bin_range if bin_range is not None else (0, None)
    tid: int | None = None
    active: list[tuple[int, AlignedRead]] = []
    next_bin = 0

    def emit_until(limit: int | None) -> Iterator[tuple[int, int, list[AlignedRead]]]:
        nonlocal active, next_bin
        while active and (limit is None or next_bin < limit):
            index = next_bin
            lo, hi = index * size, (index + 1) * size
            members = [read for end, read in active if read.pos < hi and end >= lo]
            if members and index >= first and (last is None or index < last):
                assert tid is not None
                yield tid, index, members
            next_bin += 1
            active = [(end, read) for end, read in active if end >= next_bin * size]

    for read in reads:
        start_bin = _bin_index_for(read, size)
        if read.tid != tid:
            yield from emit_until(None)
            tid, active = read.tid, []
        if not active:
            next_bin = start_bin
        yield from emit_until(start_bin)
        active.append((_read_end(read), read))
    yield from emit_until(None)


def _summary_ledger_row(component: ComponentCall, summary: hyp_module.HypothesisSummary,
                        retention_reason: str, owner_bin_start: int,
                        owner_bin_end: int) -> EvidenceLedgerRow:
    """A ledger row for a hypothesis that never reached the expensive stages.

    THESE ROWS ARE WHY THE LEDGER IS A NULL SET. A hypothesis triaged away is
    exactly a locus the pipeline looked at and declined, which is what the
    overdispersion estimate needs. Their `final_qc` says so
    explicitly rather than leaving them indistinguishable from evaluated
    rejections.
    """
    row = EvidenceLedgerRow()
    row.chrom = component.chrom
    row.tid = component.tid
    row.bp_left = summary.bp_left
    row.bp_right = summary.bp_right
    row.pos = (summary.bp_left + ((summary.bp_right - summary.bp_left) // 2)
               if (summary.bp_left >= 0 and summary.bp_right >= 0)
               else component.anchor_pos)
    row.owner_context_left = (owner_bin_start if owner_bin_start < owner_bin_end
                              else component.bin_start)
    row.owner_context_right = (owner_bin_end - 1 if owner_bin_start < owner_bin_end
                               else component.bin_end)
    if row.bp_left >= 0 and row.bp_right >= 0:
        row.coverage_left = min(row.bp_left, row.bp_right)
        row.coverage_right = max(row.bp_left, row.bp_right)
    elif row.pos >= 0:
        row.coverage_left = row.pos
        row.coverage_right = row.pos
    row.final_qc = "NOT_EVALUATED_PRE_EXPENSIVE_STAGE"
    row.posterior_qc = "POSTERIOR_NOT_EVALUATED"
    row.lfdr_qc = "LFDR_NOT_EVALUATED"
    row.candidate_retention_reason = retention_reason
    row.alt_struct_reads = summary.alt_struct_reads
    row.alt_split_reads = summary.alt_split_reads
    row.alt_indel_reads = summary.alt_indel_reads
    row.alt_left_clip_reads = summary.alt_left_clip_reads
    row.alt_right_clip_reads = summary.alt_right_clip_reads
    row.raw_cigar_insert_reads = summary.raw_cigar_insert_reads
    row.max_raw_cigar_insert_len = summary.max_raw_cigar_insert_len
    row.ref_span_reads = summary.ref_span_reads
    row.support_qnames = list(summary.support_qnames)
    row.alt_measured_lengths = list(summary.event_evidence.alt_measured_lengths)
    return row


#: Appended to `final_qc` when the abPOA memory budget withheld event
#: strings. The consensus is then built from fewer reads than were available,
#: which is a real loss of accuracy -- and one that is invisible in every
#: other column, since a thinner consensus still looks like a consensus.
POA_MEMORY_CAPPED_QC = "EVENT_CONSENSUS_POA_MEMORY_CAPPED"


def _with_poa_cap_token(qc: str, consensus: seg_module.EventConsensus) -> str:
    if consensus.poa_reads_dropped_for_memory <= 0:
        return qc
    return bp_module.append_qc_token(qc, POA_MEMORY_CAPPED_QC)


def _evaluated_ledger_row(component: ComponentCall,
                          evidence: events_module.EventReadEvidence,
                          consensus: seg_module.EventConsensus,
                          segmentation: seg_module.EventSegmentation,
                          te_alignment: TEAlignmentEvidence,
                          joint: policy_module.JointDecisionResult,
                          owner_bin_start: int,
                          owner_bin_end: int) -> EvidenceLedgerRow:
    """A ledger row for a hypothesis that WAS evaluated, carrying its verdict.

    Everything whole-run -- the decoy check, e-BH, the overdispersion -- is
    left to finalization (`placer/core/finalize.py`).
    """
    row = EvidenceLedgerRow()
    row.chrom = component.chrom
    row.tid = component.tid
    row.bp_left = evidence.bp_left
    row.bp_right = evidence.bp_right
    row.pos = (evidence.bp_left + ((evidence.bp_right - evidence.bp_left) // 2)
               if (evidence.bp_left >= 0 and evidence.bp_right >= 0)
               else component.anchor_pos)
    row.owner_context_left = (owner_bin_start if owner_bin_start < owner_bin_end
                              else component.bin_start)
    row.owner_context_right = (owner_bin_end - 1 if owner_bin_start < owner_bin_end
                               else component.bin_end)
    row.coverage_left = min(row.bp_left, row.bp_right) if row.bp_left >= 0 else row.pos
    row.coverage_right = max(row.bp_left, row.bp_right) if row.bp_right >= 0 else row.pos
    row.family = te_alignment.best_family or "NA"
    row.subfamily = te_alignment.best_subfamily or "NA"
    row.te_annotation_class = te_alignment.annotation_class
    row.te_annotation_order = te_alignment.annotation_order
    row.te_strand = te_alignment.te_strand
    row.te_consensus_start = te_alignment.te_consensus_start
    row.te_consensus_end = te_alignment.te_consensus_end
    row.te_element_length = te_alignment.te_element_length
    row.polya_len = te_alignment.element_structure.polya_len
    row.transduction_len = te_alignment.element_structure.transduction_len
    row.ltr_form = te_alignment.ltr_form
    row.te_from_clip_sides = te_alignment.from_clip_sides
    row.te_union_covered_bp = te_alignment.te_union_covered_bp
    row.te_union_coverage = te_alignment.te_union_coverage
    row.te_dominant_family = te_alignment.te_dominant_family
    row.te_dominant_class = te_alignment.te_dominant_class
    row.te_dominant_covered_bp = te_alignment.te_dominant_covered_bp
    row.family_alignment_resolved = bool(getattr(te_alignment, "pass_", False))
    row.final_qc = _with_poa_cap_token(joint.final_qc, consensus)
    row.posterior_qc = joint.posterior_qc
    row.lfdr_qc = joint.lfdr_qc
    row.candidate_retention_reason = "EVALUATED"
    row.alt_struct_reads = evidence.alt_struct_reads
    row.alt_split_reads = evidence.alt_split_reads
    row.alt_indel_reads = evidence.alt_indel_reads
    row.alt_carrier_reads = evidence.alt_carrier_reads
    row.alt_left_clip_reads = evidence.alt_left_clip_reads
    row.alt_right_clip_reads = evidence.alt_right_clip_reads
    row.raw_cigar_insert_reads = evidence.raw_cigar_insert_reads
    row.max_raw_cigar_insert_len = evidence.max_raw_cigar_insert_len
    row.ref_span_reads = evidence.ref_span_reads
    row.support_qnames = list(evidence.support_qnames)
    row.alt_measured_lengths = list(evidence.alt_measured_lengths)
    row.full_context_input_reads = consensus.full_context_input_reads
    row.partial_context_input_reads = consensus.partial_context_input_reads
    row.left_anchor_input_reads = consensus.left_anchor_input_reads
    row.right_anchor_input_reads = consensus.right_anchor_input_reads
    row.input_event_reads = consensus.input_event_reads
    row.event_consensus_len = consensus.consensus_len
    row.left_flank_align_len = segmentation.left_flank_align_len
    row.right_flank_align_len = segmentation.right_flank_align_len
    row.insert_seq = segmentation.insert_seq
    row.best_te_identity = te_alignment.best_identity
    row.best_te_query_coverage = te_alignment.best_query_coverage
    row.cross_family_margin = te_alignment.cross_family_margin
    row.te_structure_path = joint.te_structure_path
    row.te_structure_log_evidence = joint.te_structure_log_evidence
    row.nonte_structure_log_evidence = joint.nonte_structure_log_evidence
    row.artifact_structure_log_evidence = joint.artifact_structure_log_evidence
    row.te_structure_path_confidence = joint.te_structure_path_confidence
    row.polyA_posterior = joint.polyA_posterior
    row.transduction_posterior = joint.transduction_posterior
    row.te_core_coverage = joint.te_core_coverage
    row.unexplained_high_complexity_bp = joint.unexplained_high_complexity_bp
    row.te_posterior = joint.te_posterior
    row.non_te_posterior = joint.non_te_posterior
    row.artifact_posterior = joint.artifact_posterior
    row.lfdr = joint.lfdr
    row.worst_case_lfdr = joint.worst_case_lfdr
    row.mechanistic_lower_log_bf_te_vs_artifact = joint.mechanistic_lower_log_bf_te_vs_artifact
    row.mechanistic_lower_log_bf_te_vs_non_te = joint.mechanistic_lower_log_bf_te_vs_non_te
    row.mechanistic_raw_log_bf_te_vs_artifact = joint.mechanistic_raw_log_bf_te_vs_artifact
    row.mechanistic_raw_log_bf_te_vs_non_te = joint.mechanistic_raw_log_bf_te_vs_non_te
    row.mechanistic_ref_conflict_signal = joint.mechanistic_ref_conflict_signal
    row.mechanistic_ambiguity_width = joint.mechanistic_ambiguity_width
    row.mechanistic_blocks = joint.mechanistic_blocks
    row.robust_mechanistic_lfdr = joint.robust_mechanistic_lfdr
    row.robust_mechanistic_worst_case_lfdr = joint.robust_mechanistic_worst_case_lfdr
    row.robust_mechanistic_qc = joint.robust_mechanistic_qc
    return row


def _final_call_from_evaluation(component: ComponentCall,
                                evidence: events_module.EventReadEvidence,
                                consensus: seg_module.EventConsensus,
                                segmentation: seg_module.EventSegmentation,
                                te_alignment: TEAlignmentEvidence,
                                joint: policy_module.JointDecisionResult,
                                genotype, config: PipelineConfig,
                                hooks: StageHooks) -> FinalCall:
    """Assemble a call from one evaluated hypothesis.

    THE FAMILY LOGIC IS THE SUBTLE PART, and it has three states rather than two:

      * `emit_unknown_te` (a TE-unknown call or a structural insertion) forces
        the label to UNKNOWN outright -- the event is real, the element is not
        named;
      * otherwise the label is the alignment's, and `family_committed` says
        whether it may be TRUSTED: only a passing alignment with a resolved or
        family-only QC reason commits;
      * separately, a SEQUENCE FAMILY CERTIFICATE may be recorded as a
        CANDIDATE. It requires everything at once -- a high-confidence resolved
        alignment, a family margin over the configured floor, a structure decode
        that AGREES with the alignment on both family and subfamily, and a path
        confidence of at least `1 - q`. It is committed only in finalization,
        after selection, so the library cannot feed back into detection.
    """
    call = FinalCall()
    call.chrom = component.chrom
    call.tid = component.tid
    call.bp_left = evidence.bp_left
    call.bp_right = evidence.bp_right
    call.alt_indel_reads = evidence.alt_indel_reads
    call.pos = (evidence.bp_left + ((evidence.bp_right - evidence.bp_left) // 2)
                if (evidence.bp_left >= 0 and evidence.bp_right >= 0)
                else component.anchor_pos)
    # The hypothesis's own position, before finalization places the call: the
    # one the ledger row carries, and the one the decision groups loci by.
    call.hypothesis_pos = call.pos
    call.window_start = component.bin_start
    call.window_end = component.bin_end

    posterior = hyp_module.compute_breakpoint_position_posterior(
        component.breakpoint_candidates)
    call.bp_ci_width = posterior.ci_width
    call.bp_posterior_entropy = posterior.entropy

    emit_unknown_te = joint.emit_unknown_te or joint.emit_structural_event_call
    sequence_explanation = te_alignment.te_sequence_explanation
    te_pass = bool(getattr(te_alignment, "pass_", False))
    sequence_family_certificate = (
        emit_unknown_te is False
        and joint.emit_te_call and te_pass
        and te_alignment.qc_reason == "PASS_INSERT_TE_ALIGNMENT"
        and te_alignment.annotation_confidence == "HIGH"
        and bool(te_alignment.best_family) and bool(te_alignment.best_subfamily)
        and te_alignment.cross_family_margin >= config.te_family_margin_min
        and sequence_explanation is not None
        and str(getattr(sequence_explanation.status, "value",
                        sequence_explanation.status)) == "RESOLVED"
        and sequence_explanation.family == te_alignment.best_family
        and sequence_explanation.subfamily == te_alignment.best_subfamily
        and joint.te_structure_path_confidence >= (1.0 - config.final_fdr_q))

    call.family = "UNKNOWN" if emit_unknown_te else (te_alignment.best_family or "NA")
    call.subfamily = "UNKNOWN" if emit_unknown_te else (te_alignment.best_subfamily or "NA")
    call.family_committed = (not emit_unknown_te and te_pass
                             and te_alignment.qc_reason in
                             ("PASS_INSERT_TE_ALIGNMENT",
                              "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY")
                             and bool(te_alignment.best_family))
    if sequence_family_certificate:
        call.sequence_family_candidate = te_alignment.best_family
        call.sequence_subfamily_candidate = te_alignment.best_subfamily
        call.sequence_family_commit_eligible = True
    call.te_name = call.subfamily if call.subfamily != "NA" else call.family
    call.strand = te_alignment.te_strand

    call.insert_len = len(segmentation.insert_seq)
    call.insert_seq = segmentation.insert_seq
    call.best_te_identity = te_alignment.best_identity
    call.best_te_query_coverage = te_alignment.best_query_coverage
    call.cross_family_margin = te_alignment.cross_family_margin
    call.te_consensus_start = te_alignment.te_consensus_start
    call.te_consensus_end = te_alignment.te_consensus_end
    call.te_element_length = te_alignment.te_element_length
    call.polya_len = te_alignment.element_structure.polya_len
    call.transduction_len = te_alignment.element_structure.transduction_len
    call.ltr_form = te_alignment.ltr_form
    call.te_from_clip_sides = te_alignment.from_clip_sides
    call.te_union_covered_bp = te_alignment.te_union_covered_bp
    call.te_union_coverage = te_alignment.te_union_coverage
    call.te_dominant_family = te_alignment.te_dominant_family
    call.te_dominant_class = te_alignment.te_dominant_class
    call.te_dominant_covered_bp = te_alignment.te_dominant_covered_bp
    call.te_sequence_model_label = te_alignment.sequence_model_label
    call.te_sequence_model_score = te_alignment.sequence_model_score
    call.te_sequence_model_gc = te_alignment.sequence_model_gc
    call.te_sequence_model_entropy = te_alignment.sequence_model_entropy
    call.te_sequence_model_tandem_fraction = te_alignment.sequence_model_tandem_fraction
    call.te_sequence_model_low_complexity_fraction = te_alignment.sequence_model_low_complexity_fraction
    call.te_sequence_model_jsd_k5 = te_alignment.sequence_model_jsd_k5
    call.te_sequence_model_jsd_k6 = te_alignment.sequence_model_jsd_k6
    call.te_sequence_model_k9_containment = te_alignment.sequence_model_k9_containment
    call.te_annotation_confidence = te_alignment.annotation_confidence
    call.te_annotation_class = te_alignment.annotation_class
    call.te_annotation_order = te_alignment.annotation_order
    call.te_annotation_intervals = te_alignment.annotation_intervals
    call.te_annotation_residual_fraction = te_alignment.annotation_residual_fraction
    call.te_annotation_masked_fraction = te_alignment.annotation_masked_fraction
    call.left_flank_align_len = segmentation.left_flank_align_len
    call.right_flank_align_len = segmentation.right_flank_align_len
    call.event_consensus_len = consensus.consensus_len
    call.te_qc = te_alignment.qc_reason

    call.support_reads = evidence.alt_struct_reads
    call.alt_struct_reads = evidence.alt_struct_reads
    call.raw_cigar_insert_reads = evidence.raw_cigar_insert_reads
    call.max_raw_cigar_insert_len = evidence.max_raw_cigar_insert_len
    call.ref_span_reads = evidence.ref_span_reads
    call.low_mapq_ref_span_reads = evidence.low_mapq_ref_span_reads
    call.support_qnames = list(evidence.support_qnames)
    call.genotype = genotype.best_gt
    call.af = genotype.allele_fraction
    call.gq = genotype.gq

    if hooks.detect_tsd is not None:
        detection = hooks.detect_tsd(component.chrom, evidence.bp_left,
                                     evidence.bp_right, segmentation.insert_seq)
        if detection is not None:
            # The detector's own field names, which differ from the call's --
            # `type`/`length`/`sequence` there, `tsd_*` here. Mapped explicitly
            # rather than by a loop so a renamed field is a visible break.
            call.tsd_type = detection.type
            call.tsd_len = detection.length
            call.tsd_seq = detection.sequence or "NA"
            call.tsd_bg_p = detection.bg_p
            call.tsd_mismatches = detection.mismatches

    call.final_qc = _with_poa_cap_token(joint.final_qc, consensus)
    call.best_explanation = joint.best_explanation
    call.explanation_residual = joint.explanation_residual
    call.explanation_path = joint.explanation_path
    call.te_structure_path = joint.te_structure_path
    call.te_structure_log_evidence = joint.te_structure_log_evidence
    call.nonte_structure_log_evidence = joint.nonte_structure_log_evidence
    call.artifact_structure_log_evidence = joint.artifact_structure_log_evidence
    call.te_structure_path_confidence = joint.te_structure_path_confidence
    call.polyA_posterior = joint.polyA_posterior
    call.transduction_posterior = joint.transduction_posterior
    call.te_core_coverage = joint.te_core_coverage
    call.unexplained_high_complexity_bp = joint.unexplained_high_complexity_bp
    call.te_posterior = joint.te_posterior
    call.non_te_posterior = joint.non_te_posterior
    call.artifact_posterior = joint.artifact_posterior
    call.te_vs_artifact_log_odds = joint.te_vs_artifact_log_odds
    call.te_vs_non_te_log_odds = joint.te_vs_non_te_log_odds
    call.posterior_qc = joint.posterior_qc
    call.latent_mechanism = joint.latent_mechanism
    call.family_activity_prior = joint.family_activity_prior
    call.lfdr = joint.lfdr
    call.worst_case_lfdr = joint.worst_case_lfdr
    call.lfdr_qc = joint.lfdr_qc
    call.mechanistic_lower_log_bf_te_vs_artifact = joint.mechanistic_lower_log_bf_te_vs_artifact
    call.mechanistic_lower_log_bf_te_vs_non_te = joint.mechanistic_lower_log_bf_te_vs_non_te
    call.mechanistic_raw_log_bf_te_vs_artifact = joint.mechanistic_raw_log_bf_te_vs_artifact
    call.mechanistic_raw_log_bf_te_vs_non_te = joint.mechanistic_raw_log_bf_te_vs_non_te
    call.mechanistic_ref_conflict_signal = joint.mechanistic_ref_conflict_signal
    call.mechanistic_ambiguity_width = joint.mechanistic_ambiguity_width
    call.mechanistic_blocks = joint.mechanistic_blocks
    call.robust_mechanistic_lfdr = joint.robust_mechanistic_lfdr
    call.robust_mechanistic_worst_case_lfdr = joint.robust_mechanistic_worst_case_lfdr
    call.robust_mechanistic_qc = joint.robust_mechanistic_qc
    return call


@dataclass
class _PreparedEvaluation:
    """One shortlisted hypothesis, taken as far as it can go before TE alignment.

    WHY THE EVALUATION IS CUT HERE. Everything before this point is in-process;
    the alignment is an external `blastn`, whose fixed start-up cost (~0.8 s of
    CPU for BLAST+ 2.17, measured on `blastn -version`) is larger than the
    search itself for an insert of a few hundred bases. Stopping every
    hypothesis of a bin here hands the aligner all of the bin's inserts at
    once, so their processes can run side by side -- see `process_bin_records`.
    """

    component: ComponentCall
    local_records: list[AlignedRead]
    fragments: list[fragments_module.InsertionFragment]
    shortlisted: hyp_module.ShortlistedHypothesis
    evidence: events_module.EventReadEvidence
    consensus: seg_module.EventConsensus
    segmentation: seg_module.EventSegmentation
    seg_evidence: policy_module.EventSegmentationEvidence
    existence: policy_module.EventExistenceEvidence
    genotype: object
    #: The insert's two ends assembled separately from clips, when no read
    #: spans it (`consensus.build_side_consensuses`); None otherwise.
    sides: consensus_module.SideConsensus | None = None
    #: The likelihood inputs the genotype was computed from. They travel with
    #: the call, because finalization re-genotypes with the sample's
    #: overdispersion and must use the same model.
    genotype_input: policy_module.EventGenotypeInput | None = None

    @property
    def insert_seq_to_align(self) -> str | None:
        return (self.segmentation.insert_seq if self.seg_evidence.has_insert_seq
                else None)

    @property
    def seqs_to_align(self) -> list[str | None]:
        """The insert, then its start and end sides (None where absent)."""
        sides = self.sides
        return [self.insert_seq_to_align,
                (sides.start_seq or None) if sides else None,
                (sides.end_seq or None) if sides else None]


def _prepare_shortlisted(component: ComponentCall, local_records: list[AlignedRead],
                         fragments: list[fragments_module.InsertionFragment],
                         shortlisted: hyp_module.ShortlistedHypothesis,
                         config: PipelineConfig, hooks: StageHooks,
                         result: PipelineResult) -> _PreparedEvaluation:
    """The expensive stages up to, and not including, the TE alignment.

    THE PARTIAL-CONTEXT RETRY is the one control-flow subtlety. A consensus
    built wholesale from full-context reads may fail to segment for a reason
    that is about the CONSENSUS rather than the locus -- one long read's errors
    dominating a single-read consensus, say. When that happens AND there is
    substantial partial-context support, the consensus is rebuilt from the clip
    strings alone and segmentation retried. The retry is accepted only if it
    PASSES, so it can only add calls, never change a successful one.
    """
    evidence = shortlisted.validator.event_evidence
    consensus = consensus_module.build_event_consensus(
        local_records, fragments, evidence, config,
        consensus_fn=hooks.consensus_fn)
    result.event_consensus_calls += 1

    from placer.core.genotype import genotype_from_alt_vs_ref

    # Zygosity from the allele counts alone, as finalization genotypes. The
    # length-concordance term that used to ride on it read the component's own
    # breakpoint candidates, so two components evaluating one hypothesis got
    # different GQs, and rows the per-bin de-duplication should have merged
    # both survived (`docs/departures-from-cpp.md` section 10).
    genotype = genotype_from_alt_vs_ref(
        evidence.alt_struct_reads, evidence.ref_span_reads,
        error_rate=config.genotype_error_rate,
        overdispersion=config.genotype_overdispersion)
    result.genotype_calls += 1

    genotype_input = policy_module.EventGenotypeInput(
        alt_struct_reads=evidence.alt_struct_reads,
        alt_split_reads=evidence.alt_split_reads,
        alt_indel_reads=evidence.alt_indel_reads,
        alt_left_clip_reads=evidence.alt_left_clip_reads,
        alt_right_clip_reads=evidence.alt_right_clip_reads,
        ref_span_reads=evidence.ref_span_reads,
        error_rate=config.genotype_error_rate,
        overdispersion=config.genotype_overdispersion)
    existence = policy_module.build_event_existence_evidence(genotype_input)

    def segment(consensus_to_segment):
        gate = seg_module.pre_segmentation_gate_reason(
            evidence.alt_split_reads, evidence.alt_indel_reads, consensus_to_segment)
        if gate:
            refused = seg_module.EventSegmentation()
            refused.qc_reason = gate
            return refused
        return seg_module.segment_event_consensus(
            component.chrom, evidence.bp_left, evidence.bp_right,
            evidence.alt_struct_reads, evidence.ref_span_reads,
            consensus_to_segment, config, hooks.fetch_reference)

    segmentation = segment(consensus)
    can_retry_partial = (
        not segmentation.pass_ and consensus.used_full_context
        and consensus.partial_context_input_reads >= max(
            4, config.event_consensus_poa_min_reads * 2)
        and consensus.left_anchor_input_reads > 0
        and consensus.right_anchor_input_reads > 0
        and segmentation.qc_reason in ("NO_LEFT_FLANK_MATCH", "NO_RIGHT_FLANK_MATCH",
                                       "EMPTY_EVENT_INSERT_SEGMENT"))
    if can_retry_partial:
        partial_consensus = consensus_module.build_event_consensus(
            local_records, fragments, evidence, config,
            mode=consensus_module.ConsensusContextMode.PARTIAL_ONLY,
            consensus_fn=hooks.consensus_fn)
        result.event_consensus_calls += 1
        if partial_consensus.qc_pass:
            partial_segmentation = segment(partial_consensus)
            if partial_segmentation.pass_:
                consensus = partial_consensus
                segmentation = partial_segmentation

    # NO READ SPANS IT: assemble the insert's two ends separately from the
    # clips that reach into them, so the element can still be named. Only when
    # the event consensus came from partial reads or did not segment -- a
    # spanning read's insert is better evidence than either end.
    sides = None
    if not consensus.used_full_context or not segmentation.pass_:
        sides = consensus_module.build_side_consensuses(
            local_records, fragments, evidence, consensus_fn=hooks.consensus_fn)
        if not (sides.start_seq or sides.end_seq):
            sides = None

    seg_evidence = policy_module.analyze_event_segmentation(
        consensus.qc_pass, segmentation.left_flank_align_len,
        segmentation.right_flank_align_len, segmentation.left_flank_identity,
        segmentation.right_flank_identity, segmentation.insert_seq,
        segmentation.pass_, segmentation.qc_reason)

    return _PreparedEvaluation(
        component=component, local_records=local_records, fragments=fragments,
        shortlisted=shortlisted, evidence=evidence, consensus=consensus,
        segmentation=segmentation, seg_evidence=seg_evidence, existence=existence,
        genotype=genotype, sides=sides, genotype_input=genotype_input)


def _finish_shortlisted(prepared: _PreparedEvaluation,
                        alignments: list[TEAlignmentEvidence | None],
                        config: PipelineConfig):
    """The rest of the expensive stages, once the insert has been aligned.

    `alignments` is `[insert, start side, end side]`, None where there was
    nothing to align; the sides can name an element the insert cannot.
    """
    main, start, end = alignments
    te_alignment = main if main is not None else TEAlignmentEvidence()
    if prepared.sides is not None:
        te_alignment = te_classifier_module.combine_side_alignments(
            te_alignment, start, prepared.sides.start_seq,
            end, prepared.sides.end_seq)
    evidence = prepared.evidence
    segmentation = prepared.segmentation
    fragments = prepared.fragments
    consensus = prepared.consensus
    seg_evidence = prepared.seg_evidence
    existence = prepared.existence
    genotype = prepared.genotype

    boundary = policy_module.evaluate_boundary_evidence(
        policy_module.FinalBoundaryInput(
            left_ref_start=segmentation.left_ref_start,
            left_ref_end=segmentation.left_ref_end,
            right_ref_start=segmentation.right_ref_start,
            right_ref_end=segmentation.right_ref_end,
            tsd_min_len=config.tsd_min_len, tsd_max_len=config.tsd_max_len),
        max(0, abs(evidence.bp_right - evidence.bp_left)))

    # The clip/insert concordance is computed only once segmentation produced an
    # insert to compare against, and it enters the decision as the optional
    # fifth evidence block -- it can only ADD independent read support, never
    # remove it.
    concordance = consensus_module.analyze_clip_insert_concordance(
        evidence, segmentation, fragments, config)
    clip_evidence = policy_module.ClipInsertConcordanceEvidence(
        pass_=concordance.pass_, full_insert_reads=concordance.full_insert_reads,
        left_clip_reads=concordance.left_clip_reads,
        right_clip_reads=concordance.right_clip_reads,
        max_left_identity=concordance.max_left_identity,
        max_right_identity=concordance.max_right_identity, qc=concordance.qc)

    joint = policy_module.evaluate_joint_hypotheses(existence, seg_evidence,
                                                    te_alignment, boundary,
                                                    clip_evidence)
    return evidence, consensus, segmentation, te_alignment, joint, genotype, seg_evidence


def _record_shadow(target: EvidenceLedgerRow | FinalCall,
                   shadow: locus_module.LocusScore) -> None:
    """The shadow decision's numbers, on a row or a call (same field names)."""
    target.mech_log_lr_vs_non_te = shadow.score.vs_non_te
    target.mech_log_lr_vs_artifact = shadow.score.vs_artifact
    target.mech_decoy_count = shadow.decoy_count
    target.mech_decoy_mean_exp_linkage = shadow.decoy_mean_exp_linkage
    target.mech_aligned_len = shadow.observation.aligned_len
    target.mech_sequence_term = shadow.score.terms.get("sequence", 0.0)
    target.mech_terms = ";".join(f"{name}={value:.3f}"
                                 for name, value in shadow.score.terms.items()) or "NA"


def _record_replay_observables(row: EvidenceLedgerRow, shadow: locus_module.LocusScore,
                               allele: events_module.AlleleEvidence, insert_seq: str,
                               hooks: StageHooks,
                               own_background: tuple[int, int] = (-1, -1)) -> None:
    """What the ledger records for replay and no decision reads (the fields
    after `conformal_qc` in `EvidenceLedgerRow`). Set on the row only: a
    FinalCall does not carry them."""
    row.mech_tsd_len = shadow.observation.tsd_len
    row.mech_tsd_p_present = shadow.tsd_p_present
    row.mech_decoy_tsd_p_present = shadow.decoy_tsd_p_present
    row.mech_decoy_tsd_hits = shadow.decoy_tsd_hits
    row.mech_decoy_sum_absent = shadow.decoy_sum_absent
    row.mech_decoy_sum_present_per_p = shadow.decoy_sum_present_per_p
    row.allele_length = allele.length
    row.allele_carrier_offsets = list(allele.carrier_offsets)
    row.allele_carrier_lengths = list(allele.carrier_lengths)
    row.allele_carrier_similarity = list(allele.carrier_similarity)
    row.allele_carrier_own = list(allele.carrier_own)
    own_counts = shadow.score.terms.get("counts", 0.0)
    row.mech_counts_eps = shadow.counts_eps
    row.counts_bg_own_reads, row.counts_bg_own_hits = own_background

    def counts(tally: events_module.AlleleTally) -> tuple[float, float]:
        """The tally's counts term and the error rate it was scored with."""
        if tally.extra_carriers <= 0:
            return own_counts, shadow.counts_eps
        span_left, span_right = row.bp_left + tally.span_lo, row.bp_left + tally.span_hi
        eps = locus_module.allele_error_rate(row.chrom, span_left, span_right,
                                             hooks.fetch_reference)
        return locus_module.allele_counts_term(
            row.chrom, span_left, span_right, tally.alt_reads, tally.ref_reads,
            len(insert_seq or ""), hooks.fetch_reference), eps

    for prefix, tally in (("bylen", allele.by_length), ("byseq", allele.by_sequence),
                          ("wide", allele.wide)):
        setattr(row, f"allele_{prefix}_alt_reads", tally.alt_reads)
        setattr(row, f"allele_{prefix}_ref_reads", tally.ref_reads)
        setattr(row, f"allele_{prefix}_span_lo", tally.span_lo)
        setattr(row, f"allele_{prefix}_span_hi", tally.span_hi)
        setattr(row, f"allele_{prefix}_extra_carriers", tally.extra_carriers)
        term, eps = counts(tally)
        setattr(row, f"mech_counts_allele_{prefix}", term)
        setattr(row, f"allele_{prefix}_eps", eps)
        setattr(row, f"allele_{prefix}_bg_reads", tally.background_reads)
        setattr(row, f"allele_{prefix}_bg_hits", tally.background_hits)


def _align_bin_inserts(insert_seqs: list[str | None],
                       hooks: StageHooks) -> list[TEAlignmentEvidence]:
    """One alignment per entry, in order; `None` means "nothing to align".

    Uses the batch hook when there is one -- one request for the whole bin
    -- and falls back to the per-insert hook otherwise, so a caller that
    only supplies `align_insert` (every test, and any embedding that predates
    the batch hook) sees exactly the calls it always saw.
    """
    wanted = [seq for seq in insert_seqs if seq is not None]
    if hooks.align_inserts is not None and wanted:
        aligned = iter(hooks.align_inserts(wanted))
        return [next(aligned) if seq is not None else TEAlignmentEvidence()
                for seq in insert_seqs]
    return [hooks.align_insert(seq) if seq is not None else TEAlignmentEvidence()
            for seq in insert_seqs]


def process_bin_records(bin_records: list[AlignedRead], chrom: str, tid: int,
                        owner_bin_start: int, owner_bin_end: int,
                        config: PipelineConfig, hooks: StageHooks,
                        result: PipelineResult,
                        fetch_local: Callable[[str, int, int], list[AlignedRead]]) -> None:
    """One bin, all the way to final calls and ledger rows.

    OWNERSHIP IS WHAT STOPS DOUBLE-CALLING. A component is processed by the bin
    that owns its ANCHOR, not by every bin whose reads touch it. Long reads
    starting in a previous bin still contribute -- the local fetch re-reads the
    interval -- but the component itself belongs to exactly one bin, so an event
    spanning a boundary is emitted once.
    """
    if not bin_records:
        return
    result.processed_bins += 1

    components = build_component_calls(bin_records, chrom, tid)
    if owner_bin_start < owner_bin_end:
        components = [component for component in components
                      if owner_bin_start <= component.anchor_pos < owner_bin_end]
    result.built_components += len(components)
    if not components:
        return

    # Merge the per-component fetches into as few interval reads as possible.
    requests: list[cache_module.LocalIntervalRequest] = []
    seed_bounds_by_component: list[tuple[int, int]] = []
    for index, component in enumerate(components):
        seed_bounds = bp_module.infer_component_breakpoint_bounds(component)
        seed_bounds_by_component.append(seed_bounds)
        start = max(0, min(seed_bounds) - LOCAL_EVENT_FETCH_SLACK_BP)
        requests.append(cache_module.LocalIntervalRequest(
            chrom=component.chrom, start=start,
            end=max(start + 1, max(seed_bounds) + LOCAL_EVENT_FETCH_SLACK_BP),
            request_id=index))

    intervals = cache_module.build_canonical_local_intervals(
        requests, LOCAL_INTERVAL_MERGE_GAP_BP)
    cache_entries: list[cache_module.LocalIntervalCacheEntry] = []
    for interval in intervals:
        records = fetch_local(interval.chrom, interval.start, interval.end)
        cache_entries.append(cache_module.LocalIntervalCacheEntry(
            interval=interval, records=records,
            read_spans=events_module.read_reference_spans(records)))

    # TWO PASSES OVER THE COMPONENTS, so the bin makes ONE alignment request.
    #
    # Pass 1 takes every shortlisted hypothesis of every component up to its TE
    # alignment and holds it. The bin's inserts are then aligned together, and
    # pass 2 finishes each hypothesis and appends its ledger rows and calls in
    # EXACTLY the order the single-pass loop did -- the triaged-away rows of a
    # component, then its evaluated rows and their call candidates, component
    # by component -- because finalization reads them in order and breaks ties
    # by it.
    #
    # Reordering the work is safe because nothing a component computes feeds
    # the next one: the only shared state touched before the alignment is a
    # pair of counters on `result`, and addition does not care about order.
    pending: list[tuple[ComponentCall, list[EvidenceLedgerRow],
                        list[_PreparedEvaluation]]] = []
    for index, component in enumerate(components):
        projection = cache_module.project_cached_interval_reads(requests[index],
                                                                cache_entries)
        local_records = projection.records or bin_records
        read_spans = (projection.read_spans if projection.records
                      else events_module.read_reference_spans(bin_records))
        seed_left, seed_right = seed_bounds_by_component[index]

        # Re-index the component against the LOCAL reads before extracting: a
        # component's read indices point into the BIN's list, and the local
        # fetch returns a different, wider one. Passing the original would make
        # the extractor read the wrong records, silently.
        local_component = consensus_module.build_local_fragment_component(
            component, local_records, config)
        fragments = fragments_module.extract_fragments(local_component, local_records,
                                                       config)

        all_hypotheses = bp_module.collect_breakpoint_hypotheses(
            component, local_records, fragments, seed_left, seed_right, 0)
        hypotheses = bp_module.select_diverse_breakpoint_hypotheses(
            all_hypotheses, 0, component.anchor_pos)

        summaries: list[hyp_module.HypothesisSummary] = []
        for order, hypothesis in enumerate(hypotheses):
            evidence = events_module.collect_event_read_evidence_for_bounds(
                component, local_records, read_spans, fragments,
                hypothesis.left, hypothesis.right, config.alt_signal_min_mapq,
                config.same_allele_carrier_window_bp)
            summaries.append(hyp_module.build_hypothesis_summary(
                evidence, order, hypothesis.support, hypothesis.priority))

        collapsed = hyp_module.collapse_hypothesis_summaries(summaries)
        survivors: list[hyp_module.HypothesisSummary] = []
        summary_rows: list[EvidenceLedgerRow] = []
        for order, summary in enumerate(collapsed):
            if hyp_module.should_keep_hypothesis_for_expensive_stage(summary, order == 0):
                survivors.append(summary)
            elif hyp_module.should_record_hypothesis_in_evidence_ledger(summary):
                summary_rows.append(_summary_ledger_row(
                    component, summary, "LEDGER_ONLY_PRE_EXPENSIVE_STAGE",
                    owner_bin_start, owner_bin_end))

        validator_candidates: list[hyp_module.HypothesisValidatorEvidence] = []
        for summary in survivors:
            inputs = consensus_module.collect_event_consensus_inputs(
                local_records, fragments, summary.event_evidence)
            counts = hyp_module.ConsensusInputCounts(
                full_context_input_reads=inputs.full_context_input_reads,
                partial_context_input_reads=inputs.partial_context_input_reads,
                left_anchor_input_reads=inputs.left_anchor_input_reads,
                right_anchor_input_reads=inputs.right_anchor_input_reads,
                input_event_reads=(inputs.full_context_input_reads
                                   if inputs.full_context_input_reads > 0
                                   else inputs.partial_context_input_reads))
            validator_candidates.append(hyp_module.collect_hypothesis_validator_evidence(
                summary, counts, component.anchor_pos))

        shortlist = hyp_module.build_expensive_stage_shortlist(validator_candidates)

        prepared = [_prepare_shortlisted(component, local_records, fragments,
                                         shortlisted, config, hooks, result)
                    for shortlisted in shortlist]
        pending.append((component, summary_rows, prepared))

    # Up to three sequences per hypothesis (insert, start side, end side), all
    # in the bin's one alignment request.
    flat = _align_bin_inserts(
        [seq for _, _, prepared in pending for item in prepared
         for seq in item.seqs_to_align], hooks)
    flat_seqs = [seq for _, _, prepared in pending for item in prepared
                 for seq in item.seqs_to_align]
    alignments = iter([[flat[i] if flat_seqs[i] is not None else None
                        for i in range(k, k + 3)]
                       for k in range(0, len(flat), 3)])

    # ONE ROW PER OBSERVATION. Neighbouring components of one locus often
    # triage or evaluate the SAME hypothesis from the same reads, and each used
    # to append its own copy: 21% of the rows on a 0.9 Mb HG002 slice were
    # exact duplicates. The ledger is the run's null set -- the overdispersion
    # estimate counts its rows as separate observations -- so a duplicate is
    # one locus counted twice. A row is dropped only when EVERY field equals
    # one this bin already wrote, and a dropped row adds no call candidate
    # either.
    bin_rows: list[EvidenceLedgerRow] = []

    def append_row(row: EvidenceLedgerRow) -> bool:
        if any(row == seen for seen in bin_rows):
            return False
        bin_rows.append(row)
        result.evidence_ledger.append(row)
        return True

    for component, summary_rows, prepared in pending:
        for row in summary_rows:
            append_row(row)
        for item in prepared:
            (evidence, consensus, segmentation, te_alignment, joint, genotype,
             _) = _finish_shortlisted(item, next(alignments), config)
            shadow = locus_module.score_evaluated_locus(
                component.chrom, evidence.bp_left, evidence.bp_right,
                segmentation.insert_seq, te_alignment, evidence.alt_struct_reads,
                evidence.ref_span_reads, hooks.fetch_reference, hooks.detect_tsd)
            row = _evaluated_ledger_row(
                component, evidence, consensus, segmentation, te_alignment, joint,
                owner_bin_start, owner_bin_end)
            _record_shadow(row, shadow)
            _record_replay_observables(
                row, shadow,
                events_module.collect_allele_evidence(component, item.local_records, evidence),
                segmentation.insert_seq, hooks,
                events_module.collect_own_background(component, item.local_records, evidence))
            row_is_new = append_row(row)

            call = _final_call_from_evaluation(component, evidence, consensus,
                                               segmentation, te_alignment, joint,
                                               genotype, config, hooks)
            _record_shadow(call, shadow)
            if item.genotype_input is not None:
                # A copy: finalization sets the sample's overdispersion on it.
                call.genotype_likelihood_input = replace(item.genotype_input)
            call.te_best_family = te_alignment.best_family or "NA"
            call.te_best_subfamily = te_alignment.best_subfamily or "NA"
            # One candidate per ledger OBSERVATION: a hypothesis two components
            # evaluated identically is one row, so it is one candidate too, and
            # the decision sees exactly the population a replay of the ledger
            # sees (`tools/dream`).
            if row_is_new:
                result.candidate_calls.append(call)
