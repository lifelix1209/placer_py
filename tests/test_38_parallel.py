"""
The parallel scan must give the answer the single stream gives.

Where to look: `placer_py/parallel.py` argues WHY the answer is the same -- a
bin depends only on the reads that start in it, so the run can be cut at bin
boundaries and rejoined in genome order before finalization. This file holds
the three places that argument could fail:

  * the chunk planner, which must cut ONLY at bin boundaries and give every
    read to exactly one chunk -- including the long reads an indexed fetch
    hands to several, and the reads a `--region` fetch returns from before the
    region;
  * the merge, which must reproduce what one longer scan would have
    accumulated;
  * a real run, where `--threads 3` has to write the same bytes as
    `--threads 1`. That one needs pysam, pyabpoa and BLAST+ and skips without
    them; the first two run anywhere, including under the zero-dependency
    runner.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path

import pytest

from placer_py.alignment import AlignedRead
from placer_py.core.ledger import EvidenceLedgerRow, FinalCall
from placer_py.core.result import PipelineResult
from placer_py.core.scan import merge_scan_results
from placer_py.parallel import auto_chunk_bp, plan_chunks, round_up_to_bins

pytestmark = pytest.mark.invariant

BIN = 10_000
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "data"
OUTPUTS = ("scientific.txt", "evidence_ledger.tsv", "structural_calls.tsv",
           "calls.vcf", "calls.csv")


# ------------------------------------------------------------- the planner
def test_every_cut_is_a_bin_boundary():
    """A bin split between two workers would be processed twice, half each."""
    chunks = plan_chunks([("chr1", 12_345, 987_654)], 55_555, BIN, region_scoped=True)
    assert len(chunks) > 5
    for chunk in chunks[1:]:
        assert chunk.fetch_start % BIN == 0, chunk


def test_the_chunks_tile_the_span_in_order():
    chunks = plan_chunks([("chr1", 5, 250_000), ("chr2", 0, 90_000)], 50_000, BIN,
                         region_scoped=False)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chrom, start, end in (("chr1", 5, 250_000), ("chr2", 0, 90_000)):
        mine = [c for c in chunks if c.chrom == chrom]
        assert mine[0].fetch_start == start and mine[-1].fetch_end == end
        for left, right in zip(mine, mine[1:]):
            assert left.fetch_end == right.fetch_start


def _read(pos: int) -> AlignedRead:
    return AlignedRead(qname=f"r{pos}", tid=0, pos=pos)


@pytest.mark.parametrize("region_scoped", [True, False])
def test_every_read_belongs_to_exactly_one_chunk(region_scoped):
    """The property the whole design rests on.

    A read reaches EVERY chunk it overlaps (that is what an indexed fetch
    returns), and must be kept by exactly one of them: the one its START is
    in, which is the one whose bins the single stream would have put it in.
    """
    start, end = 20_000, 400_000
    chunks = plan_chunks([("chr1", start, end)], 30_000, BIN, region_scoped)
    rng = random.Random(7)
    # A region fetch also returns reads that start BEFORE the region and
    # overlap it; a whole-contig span has none, since reads start at >= 0.
    lowest = start - 50_000 if region_scoped else start
    for _ in range(3_000):
        pos = rng.randrange(lowest, end)
        span = rng.randrange(1, 60_000)
        overlapping = [c for c in chunks
                       if pos < c.fetch_end and pos + span > c.fetch_start]
        if not overlapping:
            continue
        keepers = [c.index for c in overlapping if c.keeps(_read(pos))]
        assert len(keepers) == 1, (pos, span, keepers)


def test_a_region_run_gives_the_pre_region_reads_to_the_first_chunk():
    chunks = plan_chunks([("chr1", 20_000, 100_000)], 30_000, BIN, region_scoped=True)
    assert chunks[0].keeps(_read(1_000))
    assert not any(c.keeps(_read(1_000)) for c in chunks[1:])


def test_automatic_chunks_are_bounded_and_bin_aligned():
    assert auto_chunk_bp(1_000, 8, BIN) == 50_000
    assert auto_chunk_bp(3_100_000_000, 8, BIN) == 5_000_000
    assert round_up_to_bins(55_555, BIN) == 60_000
    assert round_up_to_bins(1, BIN) == BIN


# --------------------------------------------------------------- the merge
def _scan(tag: str, n: int) -> PipelineResult:
    result = PipelineResult(total_reads=10 * n, gate1_passed=5 * n, processed_bins=n,
                            built_components=2 * n, event_consensus_calls=n,
                            genotype_calls=n)
    for i in range(n):
        call = FinalCall()
        call.chrom, call.pos = tag, i
        row = EvidenceLedgerRow()
        row.chrom, row.pos = tag, i
        result.final_calls.append(call)
        result.evidence_ledger.append(row)
    return result


def test_merging_is_what_one_longer_scan_would_have_accumulated():
    merged = PipelineResult()
    for part in (_scan("a", 2), _scan("b", 0), _scan("c", 3)):
        merge_scan_results(merged, part)
    assert (merged.total_reads, merged.gate1_passed, merged.processed_bins,
            merged.built_components, merged.event_consensus_calls,
            merged.genotype_calls) == (50, 25, 5, 10, 5, 5)
    assert [(c.chrom, c.pos) for c in merged.final_calls] == [
        ("a", 0), ("a", 1), ("c", 0), ("c", 1), ("c", 2)]
    assert [(r.chrom, r.pos) for r in merged.evidence_ledger] == [
        ("a", 0), ("a", 1), ("c", 0), ("c", 1), ("c", 2)]


def test_a_finalized_scan_cannot_be_merged():
    """Its calibration describes that part alone; there is no way to combine
    two of them into what the whole run would have measured."""
    finalized = _scan("a", 1)
    finalized.structural_calls.append(FinalCall())
    with pytest.raises(ValueError):
        merge_scan_results(PipelineResult(), finalized)


# ------------------------------------------------------------ a real run
def _real_run_available() -> str:
    if not (EXAMPLE / "mini.bam").exists():
        return "examples/data is missing (examples/make_example_data.py builds it)"
    import importlib.util

    for module in ("pysam", "pyabpoa"):
        if importlib.util.find_spec(module) is None:
            return f"{module} is not installed"
    for tool in ("blastn", "makeblastdb"):
        if shutil.which(tool) is None:
            return f"{tool} is not on PATH"
    return ""


def _run(out_dir: str, workers: int, region: str | None) -> None:
    from placer_py.config import PipelineConfig
    from placer_py.main import parse_region_scope, run_pipeline_once

    config = PipelineConfig(bam_path=str(EXAMPLE / "mini.bam"),
                            reference_fasta_path=str(EXAMPLE / "mini_ref.fa"),
                            te_fasta_path=str(EXAMPLE / "mini_te.fa"))
    config.scan_workers = workers
    # One bin per chunk, so the 60 kb contig is cut five times and every
    # 8 kb read near a cut is handed to two chunks by the fetch.
    config.scan_chunk_bp = config.bin_size
    if region:
        config.bam_region_scope = parse_region_scope(region)
    assert run_pipeline_once(config, out_dir) == 0


@pytest.mark.parametrize("region", [None, "chr1:5001-55000"])
def test_three_workers_write_the_same_bytes_as_one(region):
    reason = _real_run_available()
    if reason:
        pytest.skip(reason)
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as three:
        _run(one, 1, region)
        _run(three, 3, region)
        for name in OUTPUTS:
            left = Path(one, name).read_bytes()
            right = Path(three, name).read_bytes()
            assert left == right, f"{name} differs between 1 and 3 workers"
        # Not vacuous: the dataset plants four insertions and the run calls some.
        assert os.path.getsize(Path(one, "calls.vcf")) > 0
        assert b"\nchr1\t" in Path(one, "scientific.txt").read_bytes()
