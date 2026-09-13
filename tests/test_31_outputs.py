"""
The hypothesis triage, the breakpoint posterior, and the output contracts.

Covers `placer_py/hypotheses.py`, `placer_py/outputs.py` and the region/config
half of `placer_py/main.py`. The C++ has no tests for any of it: the triage is
in an `.inc` file, the writers are in `main.cpp`, and both need a pipeline to
reach.
"""

from __future__ import annotations

import pytest

from conftest import call_or_skip, close
from placer_py import finalization as F
from placer_py import hypotheses as H
from placer_py import main as M
from placer_py import outputs as O
from placer_py.clustering import (
    CANDIDATE_LONG_INSERTION,
    CANDIDATE_SOFT_CLIP,
    CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
    BreakpointCandidate,
    ComponentCall,
)
from placer_py.config import BamRegionScope, FinalReportMode, PipelineConfig
from placer_py.events import EventReadEvidence
from placer_py.ledger import EvidenceLedgerRow, FinalCall

pytestmark = pytest.mark.invariant


def summary(**kw):
    base = dict(bp_left=1000, bp_right=1012, alt_struct_reads=4)
    base.update(kw)
    return H.HypothesisSummary(**base)


# ------------------------------------------------------------- event lengths
def test_only_supporting_reads_contribute_an_observed_length():
    """A breakpoint candidate whose read did not end up supporting this
    hypothesis is measuring something else."""
    component = ComponentCall(chrom="chr1")
    component.breakpoint_candidates = [
        BreakpointCandidate(read_id="in", ins_len=300),
        BreakpointCandidate(read_id="out", ins_len=9000)]
    evidence = EventReadEvidence(support_qnames=["in"])
    assert call_or_skip(H.collect_alt_observed_lengths, component, evidence) == [300]


def test_a_clip_length_is_used_only_when_no_insertion_length_exists():
    """It is a LOWER bound, not a measurement -- the read ran out. That biases
    the inferred length downward for events no read spans, which is honest."""
    component = ComponentCall(chrom="chr1")
    component.breakpoint_candidates = [
        BreakpointCandidate(read_id="a", ins_len=300, clip_len=80),
        BreakpointCandidate(read_id="b", clip_len=80)]
    evidence = EventReadEvidence(support_qnames=["a", "b"])
    assert H.collect_alt_observed_lengths(component, evidence) == [300, 80]


def test_the_event_length_is_a_median_so_one_chimera_cannot_set_it():
    assert call_or_skip(H.infer_event_length_from_alt_support,
                        [300, 310, 290, 12000]) == 310
    assert H.infer_event_length_from_alt_support([]) == -1


# -------------------------------------------------------------- collapsing
def test_collapsing_needs_both_position_and_support_to_agree():
    """
    Two hypotheses at the same position built from DIFFERENT reads are
    genuinely different (a tandem array); two built from the same reads at
    different positions are genuinely different too (one insertion's two
    junctions). Only agreeing on both makes them one.
    """
    shared = ["r1", "r2", "r3", "r4", "r5"]
    same = [summary(original_index=0, support_qnames=shared),
            summary(original_index=1, bp_left=1010, bp_right=1022,
                    support_qnames=shared)]
    assert len(call_or_skip(H.collapse_hypothesis_summaries, same)) == 1

    different_reads = [summary(original_index=0, support_qnames=shared),
                       summary(original_index=1, bp_left=1010, bp_right=1022,
                               support_qnames=["x1", "x2", "x3", "x4", "x5"])]
    assert len(H.collapse_hypothesis_summaries(different_reads)) == 2

    different_places = [summary(original_index=0, support_qnames=shared),
                        summary(original_index=1, bp_left=5000, bp_right=5012,
                                support_qnames=shared)]
    assert len(H.collapse_hypothesis_summaries(different_places)) == 2


def test_the_narrower_interval_wins_between_equal_hypotheses():
    """
    The one that commits to a tighter interval is the more useful description,
    and the more likely to be right: a wide interval usually means the two ends
    came from different reads.
    """
    wide = summary(original_index=0, bp_left=1000, bp_right=1100,
                   support_qnames=["a", "b"])
    narrow = summary(original_index=1, bp_left=1000, bp_right=1005,
                     support_qnames=["a", "b"])
    assert call_or_skip(H.better_summary_representative, narrow, wide)
    assert not H.better_summary_representative(wide, narrow)


# ---------------------------------------------------------------- the triage
def test_the_top_ranked_survivor_always_reaches_the_expensive_stage():
    """
    Every component gets at least one expensive evaluation. Refusing all of
    them would decide a locus with no evidence at the one stage that could have
    produced some.
    """
    hopeless = summary(alt_struct_reads=1, alt_split_reads=0, alt_indel_reads=0,
                       ref_span_reads=40)
    assert call_or_skip(H.should_keep_hypothesis_for_expensive_stage, hopeless, True)
    assert not H.should_keep_hypothesis_for_expensive_stage(hopeless, False)


def test_one_precise_read_is_enough_on_its_own():
    assert H.should_keep_hypothesis_for_expensive_stage(
        summary(alt_split_reads=1, ref_span_reads=40), False)
    assert H.should_keep_hypothesis_for_expensive_stage(
        summary(alt_indel_reads=1, ref_span_reads=40), False)


def test_bilateral_clips_qualify_where_unilateral_ones_need_much_more():
    """Clips on both sides bracket an insertion; one side alone is a mapping
    artifact's signature."""
    bilateral = summary(alt_left_clip_reads=1, alt_right_clip_reads=1,
                        alt_struct_reads=2)
    assert H.should_keep_hypothesis_for_expensive_stage(bilateral, False)

    two_unilateral = summary(alt_left_clip_reads=2, alt_struct_reads=2)
    assert not H.should_keep_hypothesis_for_expensive_stage(two_unilateral, False)

    strong_unilateral = summary(alt_left_clip_reads=3, alt_struct_reads=3,
                                ref_span_reads=1)
    assert H.should_keep_hypothesis_for_expensive_stage(strong_unilateral, False)


def test_an_unopposed_locus_clears_the_conflict_test_at_two_reads():
    """`max(1, ref)` is what makes it `alt >= 2` rather than `alt >= 0` when
    nothing opposes the insertion."""
    unopposed = summary(alt_left_clip_reads=3, alt_struct_reads=3, ref_span_reads=0)
    assert H.should_keep_hypothesis_for_expensive_stage(unopposed, False)


def test_the_ledger_records_a_hypothesis_with_only_reference_reads():
    """
    Deliberately the weakest possible test, because the ledger is also the NULL
    SET: a reference-only locus is exactly the row the dependency bound and the
    conformal null are measured on.
    """
    assert call_or_skip(H.should_record_hypothesis_in_evidence_ledger,
                        summary(alt_struct_reads=0, ref_span_reads=3))
    assert not H.should_record_hypothesis_in_evidence_ledger(
        summary(alt_struct_reads=0, ref_span_reads=0))


def test_collapsing_runs_before_the_triage():
    """
    Otherwise a component whose first three hypotheses were three views of one
    junction would spend its guaranteed slot on a duplicate.
    """
    shared = ["r1", "r2", "r3", "r4", "r5"]
    hopeless = dict(alt_struct_reads=1, ref_span_reads=40, support_qnames=shared)
    summaries = [summary(original_index=0, **hopeless),
                 summary(original_index=1, bp_left=1010, bp_right=1022, **hopeless),
                 summary(original_index=2, bp_left=9000, bp_right=9012,
                         alt_struct_reads=1, ref_span_reads=40,
                         support_qnames=["z1", "z2"])]
    kept = call_or_skip(H.select_hypothesis_summaries_for_expensive_stage, summaries)
    assert len(kept) == 1


# ------------------------------------------------------------- the validator
def _validator(**kw):
    counts = H.ConsensusInputCounts(**kw)
    return H.collect_hypothesis_validator_evidence(summary(), counts, 1000)


def test_the_validator_refuses_what_cannot_be_assembled():
    """Definitional, not a threshold: without reads anchoring both sides there
    is nothing to assemble around."""
    assert call_or_skip(_validator, left_anchor_input_reads=2,
                        right_anchor_input_reads=0).qc_reason == "VALIDATOR_NO_BILATERAL_ANCHOR"
    assert _validator(left_anchor_input_reads=2, right_anchor_input_reads=2,
                      partial_context_input_reads=1,
                      input_event_reads=1).qc_reason == "VALIDATOR_NO_PRECISE_OR_FULL_CONTEXT"
    assert _validator(left_anchor_input_reads=2, right_anchor_input_reads=2,
                      full_context_input_reads=1,
                      input_event_reads=1).feasible_for_expensive_stage


def test_the_shortlist_keeps_a_distinct_challenger_but_not_a_duplicate():
    """
    Ranking uses read counts and the expensive stages can disagree with them --
    a hypothesis with fewer reads may segment cleanly where the top one does
    not. "Distinct" is a DISJUNCTION of position and support.
    """
    def candidate(index, bp_left, qnames):
        evidence = H.collect_hypothesis_validator_evidence(
            summary(original_index=index, bp_left=bp_left, bp_right=bp_left + 12,
                    support_qnames=qnames, alt_split_reads=1),
            H.ConsensusInputCounts(full_context_input_reads=2,
                                   left_anchor_input_reads=2,
                                   right_anchor_input_reads=2, input_event_reads=2),
            1000)
        return evidence

    shared = ["a", "b", "c", "d"]
    duplicate = [candidate(0, 1000, shared), candidate(1, 1010, shared)]
    assert len(call_or_skip(H.build_expensive_stage_shortlist, duplicate)) == 1

    distinct = [candidate(0, 1000, shared), candidate(1, 5000, shared)]
    assert len(H.build_expensive_stage_shortlist(distinct)) == 2


def test_an_infeasible_candidate_never_reaches_the_shortlist():
    infeasible = H.collect_hypothesis_validator_evidence(
        summary(), H.ConsensusInputCounts(), 1000)
    assert H.build_expensive_stage_shortlist([infeasible]) == []


# ----------------------------------------------------- breakpoint posterior
def test_precise_signals_give_a_narrow_posterior_and_clips_a_wide_one():
    """
    The same precisions the priority ladder encodes, expressed as standard
    deviations. REPORTED, not gated: a call can be confidently TE-positive and
    breakpoint-imprecise.
    """
    tight = [BreakpointCandidate(pos=1000, class_mask=CANDIDATE_SPLIT_SA_SUPPLEMENTARY)
             for _ in range(4)]
    loose = [BreakpointCandidate(pos=1000 + 40 * i, class_mask=CANDIDATE_SOFT_CLIP)
             for i in range(4)]
    tight_posterior = call_or_skip(H.compute_breakpoint_position_posterior, tight)
    loose_posterior = H.compute_breakpoint_position_posterior(loose)
    assert tight_posterior.ci_width < loose_posterior.ci_width
    assert tight_posterior.entropy < loose_posterior.entropy


def test_a_single_signal_reports_zeros_rather_than_perfect_precision():
    """There is no agreement to measure with one measurement, and a width of 0
    would claim perfect precision from a single read."""
    single = [BreakpointCandidate(pos=1000, class_mask=CANDIDATE_LONG_INSERTION)]
    posterior = H.compute_breakpoint_position_posterior(single)
    assert (posterior.ci_width, posterior.entropy) == (0.0, 0.0)
    assert H.compute_breakpoint_position_posterior([]).ci_width == 0.0


def test_the_grid_is_capped_so_two_distant_signals_cannot_explode_it():
    far = [BreakpointCandidate(pos=1000, class_mask=CANDIDATE_LONG_INSERTION),
           BreakpointCandidate(pos=900000, class_mask=CANDIDATE_LONG_INSERTION)]
    posterior = H.compute_breakpoint_position_posterior(far)
    assert posterior.ci_width <= H.POSTERIOR_GRID_MAX_SPAN_BP


# ------------------------------------------------------------ output shape
def test_every_call_row_matches_its_header_exactly():
    """
    The column ORDER is the contract. Every downstream reader eventually indexes
    by position, and a mismatch here breaks them silently.
    """
    call = FinalCall(chrom="chr1", pos=1000, te_name="L1HS", family="L1")
    assert len(call_or_skip(O.final_call_row, call)) == len(O.final_call_header())
    assert len(O.final_call_row(call, True)) == len(O.final_call_header(True))


def test_every_ledger_row_matches_its_header_exactly():
    row = EvidenceLedgerRow(chrom="chr1", pos=1000, support_qnames=["a", "b"])
    for insert_seq in (False, True):
        for qnames in (False, True):
            assert len(O.evidence_ledger_row(row, insert_seq, qnames)) == len(
                O.evidence_ledger_header(insert_seq, qnames))


def test_the_optional_columns_go_where_the_header_says():
    header = O.final_call_header(True)
    assert header[header.index("consensus_len") + 1] == "insert_seq"
    ledger = O.evidence_ledger_header(True, True)
    assert ledger[ledger.index("event_consensus_len") + 1] == "insert_seq"
    assert ledger[ledger.index("support_qname_count") + 1] == "support_qnames"


def test_the_two_call_files_carry_the_identical_schema():
    """So the two can be concatenated or diffed without reshaping -- the
    structural calls are SELECTED calls, not rejects."""
    result = F.PipelineResult()
    result.final_calls = [FinalCall(chrom="chr1", pos=1000)]
    result.structural_calls = [FinalCall(chrom="chr1", pos=2000)]
    scientific = O.render_scientific_txt(result).splitlines()
    structural = O.render_structural_calls_tsv(result).splitlines()
    assert [line for line in scientific if line.startswith("#chrom")] == [
        line for line in structural if line.startswith("#chrom")]


def test_floats_round_trip_at_full_precision():
    """
    The ledger's log Bayes factors are re-read by the dependency calibration.
    Rounding them in the file would change the measured bound.
    """
    row = EvidenceLedgerRow(chrom="chr1", mechanistic_lower_log_bf_te_vs_artifact=1 / 3)
    fields = O.evidence_ledger_row(row)
    index = O.evidence_ledger_header().index("mechanistic_lower_log_bf_te_vs_artifact")
    assert float(fields[index]) == row.mechanistic_lower_log_bf_te_vs_artifact
    assert repr(1 / 3) in fields


def test_an_unset_text_field_is_written_as_na_not_blank():
    """A blank field reads as either "missing" or "the empty string" depending
    on the parser."""
    fields = O.final_call_row(FinalCall(chrom="chr1", te_name="", tsd_seq=""))
    header = O.final_call_header()
    assert fields[header.index("te")] == "NA"
    assert fields[header.index("tsd_seq")] == "NA"


def test_the_family_status_column_reports_abstention_separately_from_the_label():
    """A library may contain a family literally named "Unknown", so the status
    has to say whether the label was committed."""
    header = O.final_call_header()
    committed = O.final_call_row(FinalCall(family="L1", family_committed=True))
    abstained = O.final_call_row(FinalCall(family="L1", family_committed=False))
    assert committed[header.index("family_status")] == "COMMITTED"
    assert abstained[header.index("family_status")] == "ABSTAINED"


def test_the_summary_reports_what_the_e_values_were_calibrated_against():
    """A file reporting selections without reporting the calibration cannot be
    audited."""
    result = F.PipelineResult(estimated_dependency_sigma=2.5,
                              dependency_penalty_null_count=1060)
    text = O.render_scientific_txt(result)
    assert "estimated_dependency_sigma\t2.5" in text
    assert "dependency_penalty_null_count\t1060" in text


def test_the_scientific_summary_ends_at_a_blank_line():
    """Which is how a reader skips it without knowing its length."""
    lines = O.render_scientific_txt(F.PipelineResult()).splitlines()
    blank = lines.index("")
    assert lines[blank + 1].startswith("#chrom")


def test_the_ledger_has_no_leading_hash_because_it_is_data():
    text = O.render_evidence_ledger_tsv(F.PipelineResult())
    assert text.startswith("chrom\t")


# ------------------------------------------------------------- the CLI
def test_a_bare_contig_means_the_whole_contig():
    """`end = -1` rather than 0, because position 0 is a real coordinate."""
    scope = call_or_skip(M.parse_region_scope, "chr1")
    assert (scope.enabled, scope.chrom, scope.start, scope.end) == (True, "chr1", 0, -1)


def test_a_region_is_one_based_inclusive_on_the_way_in():
    scope = M.parse_region_scope("chr1:1000-2000")
    assert (scope.start, scope.end) == (999, 2000)


def test_a_malformed_region_raises_rather_than_widening_to_the_contig():
    """Silently widening a region the user restricted would run the whole genome
    when they asked for 100 kb."""
    for bad in ("", "chr1:1000", "chr1:2000-1000", ":1000-2000", "chr1:0-100"):
        with pytest.raises(ValueError):
            M.parse_region_scope(bad)


def test_a_colon_in_a_contig_name_is_handled_by_taking_the_last_one():
    scope = M.parse_region_scope("HLA-A*01:01:01:01:100-200")
    assert scope.chrom == "HLA-A*01:01:01:01"
    assert (scope.start, scope.end) == (99, 200)


def test_environment_overrides_apply_and_bad_values_are_ignored():
    """
    An environment variable is often set once and inherited by unrelated runs.
    Aborting a whole-genome run over a stale value would be worse than using
    the default.
    """
    config = call_or_skip(M.apply_environment_config, PipelineConfig(),
                          {"PLACER_BIN_SIZE": "5000", "PLACER_TSD_ENABLE": "0",
                           "PLACER_BAM_THREADS": "not-a-number",
                           "PLACER_TE_KMER_SIZES": "11,13"})
    assert config.bin_size == 5000
    assert not config.tsd_enable
    assert config.bam_threads == 2          # default kept
    assert config.te_kmer_sizes_csv == "11,13"


def test_a_flag_beats_an_inherited_environment_variable():
    args = M.build_arg_parser().parse_args(
        ["--final-fdr-q", "0.01", "--final-report-mode", "legacy",
         "a.bam", "ref.fa", "te.fa"])
    config = call_or_skip(M.config_from_args, args, {"PLACER_BIN_SIZE": "5000"})
    close(config.final_fdr_q, 0.01, "flag wins")
    assert config.final_report_mode == FinalReportMode.LEGACY
    assert config.bin_size == 5000
