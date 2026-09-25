"""
The whole pipeline, end to end.

THE ACCEPTANCE TEST FOR THE SCANNER HALF, in the same sense that
`test_13_end_to_end.py` is the acceptance test for the decision half. It builds
a synthetic locus -- reads carrying a known insertion against a known reference
-- and runs every stage from `AlignedRead`s to written files, with no BAM, no
reference index and no BLAST.

That is possible because every external dependency is a callable
(`placer.core.contracts.StageHooks`), which is the main structural
difference from
the C++ and the reason this test exists at all: the C++ equivalent would need a
real BAM, a real indexed FASTA and a real BLAST installation.
"""

from __future__ import annotations

import random

import pytest
from conftest import call_or_skip

from placer import main as M
from placer.alignment import CIGAR_I, CIGAR_M, AlignedRead
from placer.config import FinalReportMode, PipelineConfig
from placer.core.bins import group_reads_into_bins
from placer.core.contracts import StageHooks
from placer.core.te_classifier import TEAlignmentEvidence
from placer.pipeline import run_pipeline
from placer.report import tsv as O

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


def test_the_run_writes_five_files_even_when_three_are_empty():
    """
    A missing `structural_calls.tsv` is ambiguous between "none were set aside"
    and "the run died before writing it". The same argument is why an empty run
    still produces a VCF with a full header and a CSV with its header row.

    The exact key set is asserted so that adding an output is a deliberate,
    visible change rather than something that can be slipped in.
    """
    import tempfile

    result = run(synthetic_reads())
    with tempfile.TemporaryDirectory() as output_dir:
        paths = call_or_skip(M.write_outputs, result, output_dir)
        assert set(paths) == {"scientific_txt", "structural_calls_tsv",
                              "evidence_ledger_tsv", "calls_vcf", "calls_csv"}
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
