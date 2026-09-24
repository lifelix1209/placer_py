"""The scan, on several processes, with the same answer as one.

WHY PROCESSES AND NOT THREADS. Everything the scan does between two external
calls is pure Python, so a thread pool would take turns on one core. Each
worker is a separate interpreter with its own BAM handles, reference handle and
TE aligner, opened once and reused for every chunk it is given.

WHY THE ANSWER IS THE SAME, which is the whole design constraint. A bin is a
function of three things only: the reads that START in it (`group_reads_into_
bins` keys on `read.pos`), indexed fetches around its candidates, and the
stateless hooks. Nothing one bin computes is read by another; the per-bin
stage only appends to `PipelineResult`. So the run can be cut anywhere a bin
boundary falls, each piece scanned independently, and the pieces joined in
genome order -- which is the order the single stream would have produced them
in -- before `finalize_run` sees anything. Finalization, the only stage that
looks at the run as a whole, runs once, in the parent, exactly as before.

THE ONE PLACE THIS COULD GO WRONG is which reads a chunk gets. A worker reads
its chunk with an indexed fetch, which returns every read OVERLAPPING the
chunk, including long reads that start in an earlier one. Those belong to the
earlier chunk's bins, so each chunk keeps only the reads that start inside it
-- with one exception that mirrors the single stream: a `--region` fetch also
returns reads that start BEFORE the region and overlap it, and those go to the
first chunk, as they go to the first bins of a sequential run. The filter is
applied before the gate, so the gate's tallies count every read once.

`tests/test_38_parallel.py` holds both halves to that: the chunk planner and
the merge on literals, and a real BAM run in which `--threads 3` must write
byte-identical files to `--threads 1`.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

from placer_py.alignment import AlignedRead
from placer_py.config import PipelineConfig
from placer_py.core.contracts import ReadSource, StageHooks
from placer_py.core.finalize import finalize_run
from placer_py.core.result import PipelineResult
from placer_py.core.scan import merge_scan_results, run_scan
from placer_py.io.gate import gate_reads

#: Chunk width bounds for the automatic choice. The floor keeps per-chunk
#: set-up (a fetch, a generator) negligible against the work; the ceiling
#: keeps a whole-genome run at hundreds of chunks, so one TE-dense chunk
#: cannot leave the other workers idle at the end of the run.
MIN_AUTO_CHUNK_BP = 50_000
MAX_AUTO_CHUNK_BP = 5_000_000
#: Aim for this many chunks per worker, so the pool can balance load.
CHUNKS_PER_WORKER = 8


@dataclass(frozen=True)
class ScanChunk:
    """One worker task: fetch `[fetch_start, fetch_end)`, keep reads starting
    in `[keep_from, keep_until)`. `None` on either side means unbounded."""

    index: int
    chrom: str
    fetch_start: int
    fetch_end: int
    keep_from: int | None
    keep_until: int | None

    def keeps(self, read: AlignedRead) -> bool:
        if self.keep_from is not None and read.pos < self.keep_from:
            return False
        return self.keep_until is None or read.pos < self.keep_until


def auto_chunk_bp(total_bp: int, workers: int, bin_size: int) -> int:
    target = total_bp // max(1, workers * CHUNKS_PER_WORKER)
    return max(MIN_AUTO_CHUNK_BP, min(MAX_AUTO_CHUNK_BP, target))


def round_up_to_bins(chunk_bp: int, bin_size: int) -> int:
    bin_size = max(1, bin_size)
    return max(bin_size, -(-chunk_bp // bin_size) * bin_size)


def plan_chunks(spans: list[tuple[str, int, int]], chunk_bp: int, bin_size: int,
                region_scoped: bool) -> list[ScanChunk]:
    """Cut `(chrom, start, end)` spans at multiples of `chunk_bp`.

    The cut points are ABSOLUTE multiples of a multiple of `bin_size`, so they
    are bin boundaries whatever the span's own start is -- a region starting
    mid-bin puts that whole first bin in the first chunk.

    `region_scoped` is True for a `--region` run, whose single stream also
    carries the reads that start before the region and overlap it; the first
    chunk of the span keeps those. A whole-file run has no such reads, since
    every mapped read starts inside its own contig.
    """
    step = round_up_to_bins(chunk_bp, bin_size)
    chunks: list[ScanChunk] = []
    for chrom, start, end in spans:
        if end <= start:
            continue
        cuts = [start]
        cut = (start // step + 1) * step
        while cut < end:
            cuts.append(cut)
            cut += step
        cuts.append(end)
        for i in range(len(cuts) - 1):
            first, last = i == 0, i == len(cuts) - 2
            chunks.append(ScanChunk(
                index=len(chunks), chrom=chrom, fetch_start=cuts[i],
                fetch_end=cuts[i + 1],
                keep_from=None if (first and region_scoped) else cuts[i],
                keep_until=None if (last and not region_scoped) else cuts[i + 1]))
    return chunks


def plan_run_chunks(reader, config: PipelineConfig, workers: int) -> list[ScanChunk]:
    """The chunks for this run's BAM and region scope."""
    scope = reader.region_scope
    if scope.enabled:
        length = 0
        for tid in range(reader.chromosome_count()):
            if reader.chromosome_name(tid) == scope.chrom:
                length = reader.chromosome_length(tid)
        end = scope.end if scope.end > 0 else length
        spans = [(scope.chrom, scope.start, end)]
    else:
        spans = [(reader.chromosome_name(tid), 0, reader.chromosome_length(tid))
                 for tid in range(reader.chromosome_count())]
    total = sum(end - start for _, start, end in spans)
    chunk_bp = config.scan_chunk_bp or auto_chunk_bp(total, workers, config.bin_size)
    return plan_chunks(spans, chunk_bp, config.bin_size, scope.enabled)


# ------------------------------------------------------------------ the worker
#: Per-process state, set once by `_init_worker`. Module globals because a
#: `multiprocessing` initializer has nowhere else to put it.
_WORKER: dict = {}


def _init_worker(config: PipelineConfig) -> None:
    from placer_py.io.bam import make_bam_reader
    from placer_py.io.reference import ReferenceFetcher
    from placer_py.io.te_library import load_te_library
    from placer_py.wiring import build_stage_hooks

    # One decompression thread per handle: the pool already has a process
    # per core, and htslib's own threads would only compete with them.
    reader = make_bam_reader(config.bam_path, 1, None)
    reference = ReferenceFetcher(config.reference_fasta_path)
    entries = load_te_library(config.te_fasta_path)
    _WORKER.update(config=config, reader=reader, reference=reference,
                   hooks=build_stage_hooks(config, reference, entries))


def scan_chunk(chunk: ScanChunk, reader, config: PipelineConfig,
               hooks: StageHooks) -> PipelineResult:
    """One chunk, gated and scanned, UNFINALIZED."""
    result = PipelineResult()
    kept: Iterator[AlignedRead] = (
        read for read in reader.stream_interval(chunk.chrom, chunk.fetch_start,
                                                chunk.fetch_end)
        if chunk.keeps(read))
    source = ReadSource(reads=gate_reads(kept, result),
                        chromosome_name=reader.chromosome_name,
                        fetch_local=reader.fetch)
    return run_scan(source, config, hooks, result)


def _scan_chunk_in_worker(chunk: ScanChunk) -> PipelineResult:
    return scan_chunk(chunk, _WORKER["reader"], _WORKER["config"], _WORKER["hooks"])


# ------------------------------------------------------------------ the parent
def run_pipeline_parallel(reader, config: PipelineConfig, workers: int,
                          target_fdr: float | None = None,
                          progress: bool = True) -> PipelineResult:
    """Scan on `workers` processes, merge in genome order, finalize once.

    `reader` is the parent's own open reader, used only to plan the chunks;
    the workers open their own. The caller is expected to have built the BLAST
    database already (building the stage hooks and aligning once does it), so
    that the workers find it on disk instead of racing to create it.
    """
    chunks = plan_run_chunks(reader, config, workers)
    merged = PipelineResult()
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=workers, initializer=_init_worker,
                      initargs=(config,)) as pool:
        # `imap`, not `imap_unordered`: results arrive in chunk order, which
        # is genome order, so the merge is a plain append.
        for done, part in enumerate(pool.imap(_scan_chunk_in_worker, chunks), 1):
            merge_scan_results(merged, part)
            if progress:
                print(f"[PLACER] scanned chunk {done}/{len(chunks)} "
                      f"{chunks[done - 1].chrom}:{chunks[done - 1].fetch_start}-"
                      f"{chunks[done - 1].fetch_end} "
                      f"(ledger rows {len(merged.evidence_ledger)}, "
                      f"{time.perf_counter() - started:.0f}s)",
                      file=sys.stderr, flush=True)
    return finalize_run(merged, config, target_fdr)
