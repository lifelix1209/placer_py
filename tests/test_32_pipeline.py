"""
The whole pipeline, end to end, and the de novo command.

THE ACCEPTANCE TEST FOR THE SCANNER HALF, in the same sense that
`test_13_end_to_end.py` is the acceptance test for the decision half. It builds
a synthetic locus -- reads carrying a known insertion against a known reference
-- and runs every stage from `AlignedRead`s to written files, with no BAM, no
reference index and no BLAST.

That is possible because every external dependency is a callable
(`placer_py.core.contracts.StageHooks`), which is the main structural
difference from
the C++ and the reason this test exists at all: the C++ equivalent would need a
real BAM, a real indexed FASTA and a real BLAST installation.
"""

from __future__ import annotations

import random

import pytest
from conftest import call_or_skip

from placer_py import denovo as D
from placer_py import main as M
from placer_py.alignment import CIGAR_I, CIGAR_M, CIGAR_S, AlignedRead
from placer_py.config import FinalReportMode, PipelineConfig
from placer_py.core.bins import group_reads_into_bins
from placer_py.core.contracts import StageHooks
from placer_py.pipeline import run_pipeline
from placer_py.report import tsv as O
from placer_py.te_classifier import FragmentTEHit, TEAlignmentEvidence

pytestmark = pytest.mark.invariant

REFERENCE = "".join(random.Random(11).choice("ACGT") for _ in range(20000))
#: Unlike anything in the reference, so segmentation cannot place it.
INSERT = "TTTTGGGGCCCCAAAA" * 20


def te_evidence(seq: str) -> TEAlignmentEvidence:
    return TEAlignmentEvidence(
        best_family="L1", best_subfamily="L1HS", best_identity=0.98,
        best_query_coverage=0.95, best_score=0.93, cross_family_margin=0.4,
        sequence_model_label="TE_MODEL_IN_DISTRIBUTION", sequence_model_score=0.93,
        annotation_confidence="HIGH", annotation_residual_fraction=0.05,
        qc_reason="PASS_INSERT_TE_ALIGNMENT", pass_=True)


def synthetic_reads(alt: int = 8, ref: int = 3) -> list[AlignedRead]:
    """Alt reads carrying the insert, ref reads spanning the locus cleanly."""
    reads = []
    for i in range(alt):
        start = 9000 + i * 3
        end = 11000 - i * 3
        reads.append(AlignedRead(
            qname=f"alt{i}", tid=0, pos=start, mapq=60,
            cigar=[(CIGAR_M, 10000 - start), (CIGAR_I, len(INSERT)),
                   (CIGAR_M, end - 10000)],
            seq=REFERENCE[start:10000] + INSERT + REFERENCE[10000:end]))
    reads.extend(AlignedRead(qname=f"ref{i}", tid=0, pos=9000, mapq=60,
                             cigar=[(CIGAR_M, 2000)], seq=REFERENCE[9000:11000])
                 for i in range(ref))
    return reads


def hooks(reads):
    return StageHooks(
        fetch_reference=lambda chrom, start, end: REFERENCE[max(0, start):max(0, end)],
        align_insert=te_evidence)


def run(reads, **config_kw):
    config = PipelineConfig(bin_size=100000, **config_kw)
    return run_pipeline(reads, lambda tid: "chr1", lambda c, s, e: reads,
                        config, hooks(reads))


# ----------------------------------------------------------------- binning
def test_a_read_belongs_to_the_bin_of_its_start():
    """
    A long read spanning several bins is processed once, in the bin it starts
    in -- which is why windows may cross the bin edge and why components are
    filtered by ANCHOR rather than by read overlap.
    """
    reads = [AlignedRead(qname="a", tid=0, pos=5, cigar=[(CIGAR_M, 100000)]),
             AlignedRead(qname="b", tid=0, pos=10005, cigar=[(CIGAR_M, 10)])]
    bins = call_or_skip(group_reads_into_bins, reads, 10000)
    assert [(tid, index, [r.qname for r in rs]) for tid, index, rs in bins] == [
        (0, 0, ["a"]), (0, 1, ["b"])]


def test_a_contig_change_always_starts_a_new_bin():
    reads = [AlignedRead(qname="a", tid=0, pos=0), AlignedRead(qname="b", tid=1, pos=0)]
    assert len(list(group_reads_into_bins(reads, 10000))) == 2


def test_binning_yields_each_bin_before_reading_the_next():
    """The streaming property, pinned so it cannot regress into a list.

    `group_reads_into_bins` must hand a bin over as soon as a read with a
    different key arrives, rather than accumulating every bin first: each
    AlignedRead holds its full sequence, so materialising the whole scan is
    what made a 10 Mb region of ultra-long ONT take 2.2 GB. Consuming from a
    generator that records how far the source has advanced is the only way to
    observe the difference from outside.
    """
    reads = [AlignedRead(qname="a", tid=0, pos=5),
             AlignedRead(qname="b", tid=0, pos=10005),
             AlignedRead(qname="c", tid=0, pos=20005)]
    consumed = []

    def source():
        for read in reads:
            consumed.append(read.qname)
            yield read

    bins = group_reads_into_bins(source(), 10000)
    first = next(bins)
    assert [r.qname for r in first[2]] == ["a"]
    # "b" had to be read to close bin 0; "c" must NOT have been.
    assert consumed == ["a", "b"], consumed


# ------------------------------------------------------------- end to end
def test_the_whole_pipeline_recovers_a_synthetic_insertion():
    """
    THE acceptance test. Eight reads carrying a 320 bp insert that is not in the
    reference, three clean reference reads. The pipeline must build the
    component, assemble the consensus, segment it back to exactly the inserted
    bases, and report the locus.
    """
    reads = synthetic_reads()
    result = call_or_skip(run, reads)

    assert result.total_reads == 11
    assert result.gate1_passed == 11
    assert result.processed_bins == 1
    assert result.built_components == 1
    assert result.event_consensus_calls >= 1
    assert result.genotype_calls >= 1
    assert len(result.evidence_ledger) >= 1

    row = result.evidence_ledger[0]
    assert row.chrom == "chr1"
    assert row.alt_struct_reads == 8
    assert row.ref_span_reads == 3

    # The insert is recovered to within a few bases rather than exactly, and
    # that is CORRECT rather than a tolerance. The flank search accepts 0.90
    # identity, so a flank may extend a base or two into the insert when those
    # bases happen to match the reference -- which is what a real aligner does
    # at a junction, and what makes the TSD detector a separate component.
    assert abs(len(row.insert_seq) - len(INSERT)) <= 8
    assert row.insert_seq in INSERT or INSERT[4:-4] in row.insert_seq


def test_the_locus_is_reported_somewhere_and_exactly_once():
    """
    Either as a TE call or -- when the calibrated risk gate abstains on the
    element identity -- as a structural insertion. What must NOT happen is the
    event being reported twice or dropped entirely.
    """
    result = run(synthetic_reads())
    reported = result.final_calls + result.structural_calls
    assert len(reported) == 1
    call = reported[0]
    assert call.chrom == "chr1"
    assert abs(call.pos - 10000) <= 50
    assert call.alt_struct_reads == 8


def test_the_genotype_comes_out_heterozygous_at_eight_versus_three():
    result = run(synthetic_reads())
    call = (result.final_calls + result.structural_calls)[0]
    assert call.genotype == "0/1"
    assert call.gq > 0
    assert 0.5 < call.af < 0.9


def test_an_abstaining_call_is_set_aside_rather_than_lost():
    """
    Precision-first, end to end: the identity gate abstains, and the event is
    still reported as a structural insertion with UNKNOWN family rather than
    being discarded.
    """
    result = run(synthetic_reads())
    if not result.final_calls:
        assert len(result.structural_calls) == 1
        call = result.structural_calls[0]
        assert call.final_qc.startswith("PASS_STRUCTURAL_INSERTION")
        assert call.family == "UNKNOWN"
        assert not call.family_committed


def test_legacy_mode_keeps_the_call_in_the_main_output():
    result = run(synthetic_reads(), final_report_mode=FinalReportMode.LEGACY)
    assert len(result.final_calls) == 1
    assert result.structural_calls == []


def test_a_locus_with_no_alt_reads_produces_no_call():
    reference_only = [AlignedRead(qname=f"ref{i}", tid=0, pos=9000, mapq=60,
                                  cigar=[(CIGAR_M, 2000)], seq=REFERENCE[9000:11000])
                      for i in range(6)]
    result = run(reference_only)
    assert result.final_calls == []
    assert result.structural_calls == []


def test_the_dependency_bound_is_measured_from_the_runs_own_ledger():
    """Not asserted. The reported sigma and null count are what make the
    e-values auditable."""
    result = run(synthetic_reads())
    assert result.dependency_penalty_null_count >= 1
    assert result.estimated_dependency_sigma >= 1.0
    assert result.dependency_penalty_cap_log > 0.0


def test_the_run_writes_three_files_even_when_two_are_empty():
    """
    A missing `structural_calls.tsv` is ambiguous between "none were set aside"
    and "the run died before writing it".
    """
    import tempfile

    result = run(synthetic_reads())
    with tempfile.TemporaryDirectory() as output_dir:
        paths = call_or_skip(M.write_outputs, result, output_dir)
        assert set(paths) == {"scientific_txt", "structural_calls_tsv",
                              "evidence_ledger_tsv"}
        for path in paths.values():
            with open(path) as handle:
                assert handle.read()


def test_the_written_ledger_parses_back_with_the_right_column_count():
    result = run(synthetic_reads())
    text = O.render_evidence_ledger_tsv(result)
    lines = text.splitlines()
    header = lines[0].split("\t")
    assert header == O.evidence_ledger_header()
    for line in lines[1:]:
        assert len(line.split("\t")) == len(header)


# ------------------------------------------------------------------ de novo
CHILD_SCIENTIFIC = (
    "#PLACER streaming pipeline summary\n"
    "total_reads\t100\n"
    "\n"
    "#chrom\tpos\tte\tsupport_reads\tbp_left\tbp_right\n"
    "chr1\t1000\tL1HS\t5\t1000\t1012\n"
    "chr1\t2000\tNA\t9\t2000\t2012\n"
    "chr1\t3000\tAluYa5\t1\t3000\t3012\n")


class _Classifier:
    """A stand-in that names every fragment the same thing."""

    def __init__(self, te_name: str = "L1HS") -> None:
        self.te_name = te_name

    def is_enabled(self) -> bool:
        return True

    def classify(self, fragments):
        return [FragmentTEHit(fragment_id=f.fragment_id, te_name=self.te_name)
                for f in fragments]


def _parent_read(qname="p1", pos=1000, clip=200):
    return AlignedRead(qname=qname, tid=0, pos=pos, mapq=60,
                       cigar=[(CIGAR_S, clip), (CIGAR_M, 500)], seq="A" * (clip + 500))


def _denovo_config(**kw):
    base = dict(parent_bam_paths=["mum.bam"], child_min_support_reads=2)
    base.update(kw)
    return D.DenovoConfig(**base)


def test_only_named_child_calls_with_enough_support_are_considered():
    """
    The name requirement is not cosmetic: the scan compares the parent's
    fragments against the child's ELEMENT, so an unnamed call has nothing to
    compare against and would be reported de novo by default.
    """
    config = _denovo_config()
    candidates, total = call_or_skip(D.load_denovo_child_candidates,
                                     CHILD_SCIENTIFIC, config)
    assert total == 3
    assert [c.te_name for c in candidates] == ["L1HS"]


def test_a_parent_read_matching_the_element_vetoes_the_call():
    config = _denovo_config()
    scanner = D.ParentPoolScanner(config, _Classifier("L1HS"),
                                  [lambda c, s, e: [_parent_read()]])
    result = D.run_denovo(config, CHILD_SCIENTIFIC, scanner)
    assert result.parent_veto_calls == 1
    assert result.calls[0].status == "PARENT_VETO"
    assert result.calls[0].de_novo == "0"
    assert result.calls[0].parent_summary.exact_te_reads == 1


def test_nothing_in_any_parent_passes_the_call():
    config = _denovo_config()
    scanner = D.ParentPoolScanner(config, _Classifier(), [lambda c, s, e: []])
    result = D.run_denovo(config, CHILD_SCIENTIFIC, scanner)
    assert result.denovo_pass_calls == 1
    assert result.calls[0].de_novo == "1"


def test_an_unclassifiable_parent_fragment_yields_review_not_a_pass():
    """
    THE point of a three-way verdict. A parent read carrying inserted sequence
    the classifier could not name may be the same element, badly assembled --
    and reporting it as de novo would turn the commonest failure of trio
    analysis into a confident positive.
    """
    class _Silent(_Classifier):
        def classify(self, fragments):
            return [FragmentTEHit(fragment_id=f.fragment_id, te_name="")
                    for f in fragments]

    config = _denovo_config()
    scanner = D.ParentPoolScanner(config, _Silent(), [lambda c, s, e: [_parent_read()]])
    result = D.run_denovo(config, CHILD_SCIENTIFIC, scanner)
    assert result.review_calls == 1
    assert result.calls[0].status == "PARENT_AMBIGUOUS_SIGNAL"
    assert result.calls[0].de_novo == "REVIEW"


def test_the_family_veto_can_be_turned_off_into_review():
    """
    A parent read matching the FAMILY but not the element is either the same
    inherited insertion seen badly, or one of a million other copies of that
    family. Which it is depends on the family, and the flag is where that
    judgement is made explicitly.
    """
    vetoing = _denovo_config(family_match_veto=True)
    scanner = D.ParentPoolScanner(vetoing, _Classifier("L1PA2"),
                                  [lambda c, s, e: [_parent_read()]])
    assert D.run_denovo(vetoing, CHILD_SCIENTIFIC, scanner).parent_veto_calls == 1

    reviewing = _denovo_config(family_match_veto=False)
    scanner = D.ParentPoolScanner(reviewing, _Classifier("L1PA2"),
                                  [lambda c, s, e: [_parent_read()]])
    result = D.run_denovo(reviewing, CHILD_SCIENTIFIC, scanner)
    assert result.parent_veto_calls == 0
    assert result.review_calls == 1


def test_suppressing_review_turns_an_uncertain_locus_into_a_pass():
    """
    Pinned because it is the one setting here that can manufacture a confident
    positive, and a reader of the output should be able to see that it was
    chosen.
    """
    class _Silent(_Classifier):
        def classify(self, fragments):
            return [FragmentTEHit(fragment_id=f.fragment_id, te_name="")
                    for f in fragments]

    config = _denovo_config(emit_review_status=False)
    scanner = D.ParentPoolScanner(config, _Silent(), [lambda c, s, e: [_parent_read()]])
    result = D.run_denovo(config, CHILD_SCIENTIFIC, scanner)
    assert result.review_calls == 0
    assert result.denovo_pass_calls == 1


def test_one_parent_read_produces_one_verdict_however_many_signals_it_has():
    """
    Otherwise a single parent read could veto twice, and the summary counts
    would overstate how much of the parent supports the locus.
    """
    busy = AlignedRead(qname="p1", tid=0, pos=1000, mapq=60,
                       cigar=[(CIGAR_S, 200), (CIGAR_M, 100), (CIGAR_I, 300),
                              (CIGAR_M, 100), (CIGAR_S, 200)], seq="A" * 900)
    config = _denovo_config(default_match_window=500)
    scanner = D.ParentPoolScanner(config, _Classifier("L1HS"),
                                  [lambda c, s, e: [busy]])
    summary = scanner.scan_candidate(
        D.DenovoChildCandidate(chrom="chr1", pos=1000, te_name="L1HS",
                               event_start=1000, event_end=1012))
    assert summary.exact_te_reads == 1
    assert summary.total_support_reads == 1


def test_an_ambiguous_read_withholds_a_call_without_supporting_a_genotype():
    """
    `ambiguous_signal_reads` is deliberately NOT added to
    `total_support_reads`: an ambiguous read cannot support a parental
    genotype, only withhold a de novo call.
    """
    class _Silent(_Classifier):
        def classify(self, fragments):
            return [FragmentTEHit(fragment_id=f.fragment_id, te_name="")
                    for f in fragments]

    config = _denovo_config()
    scanner = D.ParentPoolScanner(config, _Silent(), [lambda c, s, e: [_parent_read()]])
    summary = scanner.scan_candidate(
        D.DenovoChildCandidate(chrom="chr1", pos=1000, te_name="L1HS",
                               event_start=1000, event_end=1012))
    assert summary.ambiguous_signal_reads == 1
    assert summary.total_support_reads == 0


def test_the_match_window_grows_with_the_events_own_width():
    """
    A 6 kb insertion's breakpoints are less certain than a 300 bp one's. A fixed
    window would either miss a long event's parent evidence or admit a short
    one's neighbours.
    """
    config = D.DenovoConfig(default_match_window=30, max_match_window=100)
    narrow = call_or_skip(D.build_scan_window,
                          D.DenovoChildCandidate(event_start=1000, event_end=1010), config)
    wide = D.build_scan_window(
        D.DenovoChildCandidate(event_start=1000, event_end=7000), config)
    assert (narrow.match_end - narrow.match_start) < (wide.match_end - wide.match_start)
    # ... but clamped, so it cannot grow without limit.
    assert (wide.match_end - 7000) == config.max_match_window


def test_the_fetch_window_is_wider_than_the_match_window():
    """So a read whose breakpoint sits just outside the match window is still
    examined and can still produce an ambiguous signal."""
    window = D.build_scan_window(
        D.DenovoChildCandidate(event_start=1000, event_end=1010), D.DenovoConfig())
    assert window.fetch_start < window.match_start
    assert window.fetch_end > window.match_end


def test_family_inference_handles_a_bare_subfamily_name():
    """
    Coarser than the library-header parser on purpose: this reads the `te`
    column of an earlier run's output, which may hold a bare subfamily with no
    class path.
    """
    assert call_or_skip(D.infer_te_family, "AluYa5") == "ALU"
    assert D.infer_te_family("L1HS") == "LINE1"
    assert D.infer_te_family("SVA_E") == "SVA"
    assert D.infer_te_family("HERVK-int") == "ERV"
    assert D.infer_te_family("MIR3#SINE/MIR") == "MIR3"
    assert D.infer_te_family("NA") == "NA"
    assert D.infer_te_family("") == "NA"


def test_a_dry_run_reports_every_candidate_without_scanning():
    config = _denovo_config(dry_run=True)
    result = D.run_denovo(config, CHILD_SCIENTIFIC, None)
    assert result.implementation_status == "DRY_RUN"
    assert [c.de_novo for c in result.calls] == ["DRY_RUN"]


def test_the_three_denovo_outputs_render():
    config = _denovo_config()
    scanner = D.ParentPoolScanner(config, _Classifier("L1HS"),
                                  [lambda c, s, e: [_parent_read()]])
    result = D.run_denovo(config, CHILD_SCIENTIFIC, scanner)
    calls = call_or_skip(D.render_calls_tsv, result).splitlines()
    assert calls[0].startswith("#chrom")
    assert len(calls) == 2
    veto = D.render_veto_reads_tsv(result).splitlines()
    assert len(veto) == 2
    assert "denovo_pass_calls\t0" in D.render_summary_txt(result)
