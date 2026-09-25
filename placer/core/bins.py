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
from dataclasses import dataclass
from typing import Callable

from placer.alignment import AlignedRead
from placer.config import PipelineConfig
from placer.core import breakpoints as bp_module
from placer.core import call_selection as selection_module
from placer.core import consensus as consensus_module
from placer.core import events as events_module
from placer.core import fragments as fragments_module
from placer.core import hypotheses as hyp_module
from placer.core import interval_cache as cache_module
from placer.core import policy as policy_module
from placer.core import segmentation as seg_module
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


def group_reads_into_bins(reads: Iterable[AlignedRead], bin_size: int
                          ) -> Iterator[tuple[int, int, list[AlignedRead]]]:
    """Group a stream into `(tid, bin_index, reads)`, in order.

    A read belongs to the bin of its START. A long read spanning several bins is
    therefore processed once, in the bin it starts in -- which is why the
    windowing stage allows candidate windows to extend `WINDOW_BIN_SLACK_BP`
    past the bin edge, and why components are filtered by ANCHOR position
    afterwards rather than by read overlap.

    A GENERATOR, which the coordinate-sorted input is what makes possible: a
    bin is complete the moment a read with a different key arrives, so it can
    be handed over and dropped instead of accumulated. This used to return a
    list of every bin, which -- together with the caller draining the read
    stream into a list first -- meant the whole scanned region was resident as
    Python objects. On a 10 Mb region of ultra-long ONT that was 2.2 GB and
    thrashing; each AlignedRead holds its full `seq`, and 31,773 UL reads carry
    ~590 Mbp between them.
    """
    current_key: tuple[int, int] | None = None
    batch: list[AlignedRead] = []
    for read in reads:
        key = (read.tid, _bin_index_for(read, bin_size))
        if current_key is None:
            current_key = key
        elif key != current_key:
            yield current_key[0], current_key[1], batch
            current_key, batch = key, []
        batch.append(read)
    if current_key is not None:
        yield current_key[0], current_key[1], batch


def _summary_ledger_row(component: ComponentCall, summary: hyp_module.HypothesisSummary,
                        retention_reason: str, owner_bin_start: int,
                        owner_bin_end: int) -> EvidenceLedgerRow:
    """A ledger row for a hypothesis that never reached the expensive stages.

    THESE ROWS ARE WHY THE LEDGER IS A NULL SET. A hypothesis triaged away is
    exactly a locus the pipeline looked at and declined, which is what the
    dependency bound and the conformal null need. Their `final_qc` says so
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

    Note that the raw mechanistic aggregates travel here PENALTY-FREE. The
    dependency bound is a whole-run quantity, and finalization subtracts it from
    these -- see `placer/finalization.apply_dependency_penalty_calibration`.
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
    row.family_alignment_resolved = bool(getattr(te_alignment, "pass_", False))
    row.final_qc = _with_poa_cap_token(joint.final_qc, consensus)
    row.posterior_qc = joint.posterior_qc
    row.lfdr_qc = joint.lfdr_qc
    row.candidate_retention_reason = "EVALUATED"
    row.alt_struct_reads = evidence.alt_struct_reads
    row.alt_split_reads = evidence.alt_split_reads
    row.alt_indel_reads = evidence.alt_indel_reads
    row.alt_left_clip_reads = evidence.alt_left_clip_reads
    row.alt_right_clip_reads = evidence.alt_right_clip_reads
    row.raw_cigar_insert_reads = evidence.raw_cigar_insert_reads
    row.max_raw_cigar_insert_len = evidence.max_raw_cigar_insert_len
    row.ref_span_reads = evidence.ref_span_reads
    row.support_qnames = list(evidence.support_qnames)
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
    call.pos = (evidence.bp_left + ((evidence.bp_right - evidence.bp_left) // 2)
                if (evidence.bp_left >= 0 and evidence.bp_right >= 0)
                else component.anchor_pos)
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
    call.strand = "NA"

    call.insert_len = len(segmentation.insert_seq)
    call.insert_seq = segmentation.insert_seq
    call.best_te_identity = te_alignment.best_identity
    call.best_te_query_coverage = te_alignment.best_query_coverage
    call.cross_family_margin = te_alignment.cross_family_margin
    call.te_consensus_start = te_alignment.te_consensus_start
    call.te_consensus_end = te_alignment.te_consensus_end
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

    @property
    def insert_seq_to_align(self) -> str | None:
        return (self.segmentation.insert_seq if self.seg_evidence.has_insert_seq
                else None)


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

    alt_lengths = hyp_module.collect_alt_observed_lengths(component, evidence)
    event_length = hyp_module.infer_event_length_from_alt_support(alt_lengths)
    genotype = genotype_from_alt_vs_ref(
        evidence.alt_struct_reads, evidence.ref_span_reads,
        error_rate=config.genotype_error_rate,
        overdispersion=config.genotype_overdispersion,
        event_length=event_length, alt_observed_lengths=alt_lengths)
    result.genotype_calls += 1

    existence = policy_module.build_event_existence_evidence(
        policy_module.EventGenotypeInput(
            alt_struct_reads=evidence.alt_struct_reads,
            alt_split_reads=evidence.alt_split_reads,
            alt_indel_reads=evidence.alt_indel_reads,
            alt_left_clip_reads=evidence.alt_left_clip_reads,
            alt_right_clip_reads=evidence.alt_right_clip_reads,
            ref_span_reads=evidence.ref_span_reads,
            error_rate=config.genotype_error_rate,
            overdispersion=config.genotype_overdispersion,
            event_length=event_length, alt_observed_lengths=alt_lengths))

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

    seg_evidence = policy_module.analyze_event_segmentation(
        consensus.qc_pass, segmentation.left_flank_align_len,
        segmentation.right_flank_align_len, segmentation.left_flank_identity,
        segmentation.right_flank_identity, segmentation.insert_seq,
        segmentation.pass_, segmentation.qc_reason)

    return _PreparedEvaluation(
        component=component, local_records=local_records, fragments=fragments,
        shortlisted=shortlisted, evidence=evidence, consensus=consensus,
        segmentation=segmentation, seg_evidence=seg_evidence, existence=existence,
        genotype=genotype)


def _finish_shortlisted(prepared: _PreparedEvaluation, te_alignment: TEAlignmentEvidence,
                        config: PipelineConfig):
    """The rest of the expensive stages, once the insert has been aligned."""
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
    # component, then its evaluated rows, then its selected calls, component by
    # component -- because finalization reads the ledger in order and breaks
    # ties by it.
    #
    # Reordering the work is safe because nothing a component computes feeds
    # the next one: the only shared state touched before the alignment is a
    # pair of counters on `result`, and addition does not care about order.
    pending: list[tuple[ComponentCall, list[EvidenceLedgerRow], list[_PreparedEvaluation],
                        list[selection_module.ComponentFinalCallCandidate]]] = []
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
                hypothesis.left, hypothesis.right)
            summaries.append(hyp_module.build_hypothesis_summary(
                component, evidence, order, hypothesis.support, hypothesis.priority))

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

        # Every hypothesis -- not just the shortlisted ones -- is available as a
        # retether ANCHOR, which is why they are collected before the expensive
        # stages run and independently of them.
        anchors = [selection_module.ComponentFinalCallCandidate(
            pos=hypothesis.center, emit_te=False, anchor_support=hypothesis.support,
            anchor_priority=hypothesis.priority,
            anchor_hypothesis_score=hypothesis.support
            * bp_module.breakpoint_hypothesis_support_weight(hypothesis.priority),
            component_anchor_pos=component.anchor_pos)
            for hypothesis in all_hypotheses
            if hypothesis.valid and hypothesis.center >= 0]

        prepared = [_prepare_shortlisted(component, local_records, fragments,
                                         shortlisted, config, hooks, result)
                    for shortlisted in shortlist]
        pending.append((component, summary_rows, prepared, anchors))

    alignments = _align_bin_inserts(
        [item.insert_seq_to_align for _, _, prepared, _ in pending for item in prepared],
        hooks)

    # ONE ROW PER OBSERVATION. Neighbouring components of one locus often
    # triage or evaluate the SAME hypothesis from the same reads, and each used
    # to append its own copy: 21% of the rows on a 0.9 Mb HG002 slice were
    # exact duplicates. The ledger is the run's null set -- the dependency
    # bound, the overdispersion estimate and the conformal controls all count
    # its rows as separate observations -- so a duplicate is one locus
    # counted twice. A row is dropped only when EVERY field equals one this
    # bin already wrote; the call candidates are unaffected, and finalization
    # already collapses the duplicate calls.
    bin_rows: list[EvidenceLedgerRow] = []

    def append_row(row: EvidenceLedgerRow) -> None:
        if any(row == seen for seen in bin_rows):
            return
        bin_rows.append(row)
        result.evidence_ledger.append(row)

    next_alignment = iter(alignments)
    for component, summary_rows, prepared, anchors in pending:
        for row in summary_rows:
            append_row(row)
        evaluations: list[tuple[selection_module.ComponentFinalCallCandidate, FinalCall]] = []
        for item in prepared:
            shortlisted = item.shortlisted
            (evidence, consensus, segmentation, te_alignment, joint, genotype,
             seg_evidence) = _finish_shortlisted(item, next(next_alignment), config)
            append_row(_evaluated_ledger_row(
                component, evidence, consensus, segmentation, te_alignment, joint,
                owner_bin_start, owner_bin_end))

            call = _final_call_from_evaluation(component, evidence, consensus,
                                               segmentation, te_alignment, joint,
                                               genotype, config, hooks)
            summary = shortlisted.validator.summary
            candidate = selection_module.ComponentFinalCallCandidate(
                pos=call.pos, anchor_pos=summary.bp_left,
                score=joint.best.total,
                emit_te=(joint.emit_te_call or joint.emit_structural_event_call),
                evidence_te=joint.emit_evidence_te_call,
                resolved_te=(joint.emit_te_call and not joint.emit_unknown_te),
                one_sided_segmentation=policy_module.is_one_sided_segmentation_pass(
                    seg_evidence),
                anchor_support=summary.hypothesis_support,
                anchor_ref_span_reads=summary.ref_span_reads,
                anchor_priority=summary.hypothesis_priority,
                anchor_hypothesis_score=summary.hypothesis_score,
                component_anchor_pos=component.anchor_pos)
            evaluations.append((candidate, call))

        candidates = [candidate for candidate, _ in evaluations] + anchors
        selection_module.retether_evidence_supported_final_call_positions(candidates)
        for index_, (_, call) in enumerate(evaluations):
            call.pos = candidates[index_].pos
        for chosen in selection_module.select_component_final_call_indices(candidates):
            if chosen < len(evaluations):
                result.final_calls.append(evaluations[chosen][1])
