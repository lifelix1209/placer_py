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

THE ONE PLACE THIS COULD GO WRONG is which reads a chunk gets. A bin is
scanned with every read that OVERLAPS it (`group_reads_into_bins`), so a chunk
needs the long reads that started in an earlier chunk as well as its own --
which is exactly what its indexed fetch returns. Each chunk then scans only the
bins it owns (`ScanChunk.bin_range`), and those are disjoint because chunks are
cut at bin boundaries. The gate's tallies count only the reads a chunk `keeps`
-- those starting inside it, plus, for the first chunk of a `--region`, the
reads that start before the region, as the single stream counts them -- so
every read is counted once.

A WORKER THAT DIES FAILS THE RUN. `multiprocessing.Pool` does not notice a
worker killed by a signal: the task it held is never answered and the parent
waits forever. That happened on the first cluster run -- a pyabpoa build
compiled for a CPU the node did not have died of SIGILL in every worker, and the
job sat idle for an hour with no output (bioconda pyabpoa 1.5.3 on AMD
EPYC Zen 3; 1.5.4-1.5.7 run, hence the `!=1.5.3` in pyproject.toml). `ProcessPoolExecutor` raises
`BrokenProcessPool` instead, and `run_pipeline_parallel` turns that into
`WorkerDiedError` with a message saying what to check.

`tests/test_38_parallel.py` holds both halves to that: the chunk planner and
the merge on literals, and a real BAM run in which `--threads 3` must write
byte-identical files to `--threads 1`.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

from placer.alignment import AlignedRead
from placer.config import PipelineConfig
from placer.core.contracts import ReadSource, StageHooks
from placer.core.finalize import finalize_run
from placer.core.result import PipelineResult
from placer.core.scan import merge_scan_results, run_scan
from placer.io.gate import gate_reads
from placer.io.perf import PerfLog, snapshot, span

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

    def bin_range(self, bin_size: int) -> tuple[int, int]:
        """The bins this chunk owns: those intersecting `[fetch_start, fetch_end)`.

        Chunks are cut at bin boundaries, so these are disjoint across the run
        and together the bins a one-process run scans.
        """
        size = max(1, bin_size)
        return self.fetch_start // size, (self.fetch_end - 1) // size + 1

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


#: A chunk estimated to hold more than this many times the median chunk's
#: alignment data is cut into pieces (`split_heavy_chunks`).
HEAVY_CHUNK_FACTOR = 2.0
MAX_PIECES_PER_CHUNK = 32


def chunk_data_estimates(reader, chunks: list[ScanChunk]) -> list[int]:
    """Compressed bytes of alignment data each chunk starts, from the index.

    A chunk's data runs from its first read to the next chunk's first read
    (offsets rise through a coordinate-sorted file, across contigs too); the
    last chunk's to the file position at its end. A chunk with no read, or a
    reader with no index, estimates 0. A millisecond or so per chunk.
    """
    if not hasattr(reader, "compressed_offset_at") or not chunks:
        return [0] * len(chunks)
    offsets = [reader.compressed_offset_at(chunk.chrom, chunk.fetch_start, chunk.fetch_end)
               for chunk in chunks]
    offsets.append(reader.compressed_offset_at(chunks[-1].chrom, chunks[-1].fetch_end))
    estimates = []
    for index in range(len(chunks)):
        here = offsets[index]
        following = next((offset for offset in offsets[index + 1:] if offset is not None),
                         None)
        estimates.append(following - here if here is not None and following is not None
                         and following > here else 0)
    return estimates


def split_heavy_chunks(chunks: list[ScanChunk], estimates: list[int],
                       bin_size: int) -> list[ScanChunk]:
    """Cut every chunk estimated above HEAVY_CHUNK_FACTOR x the median into
    (estimate / median)^2 pieces, at bin boundaries.

    WHY. On HG002 chr1 three of 128 chunks -- the two pericentromeric ones and
    1q21 -- were 52% of the CPU; one took 49 min on one worker while the run's
    ideal wall time was 14. Pieces of a heavy chunk run on several workers.
    SQUARED because the cost of a collapsed region grows much faster than its
    data: 1q21 held about 4x the median chunk's bytes and cost 63x its CPU.
    THE OUTPUT CANNOT CHANGE: a cut at a bin boundary is a chunk boundary like
    any other (module docstring), so each piece keeps the reads that START in
    it and scans the bins it owns, and the first and last pieces keep the
    chunk's own outer bounds.
    """
    positive = sorted(estimate for estimate in estimates if estimate > 0)
    if not positive:
        return chunks
    median = positive[len(positive) // 2]
    size = max(1, bin_size)
    out: list[ScanChunk] = []
    for chunk, estimate in zip(chunks, estimates):
        bins = chunk.bin_range(size)
        span_bins = bins[1] - bins[0]
        pieces = 1
        if estimate > HEAVY_CHUNK_FACTOR * median and span_bins > 1:
            pieces = min(MAX_PIECES_PER_CHUNK, span_bins,
                         -(-(estimate * estimate) // (median * median)))
        if pieces <= 1:
            out.append(ScanChunk(len(out), chunk.chrom, chunk.fetch_start, chunk.fetch_end,
                                 chunk.keep_from, chunk.keep_until))
            continue
        cuts = [chunk.fetch_start]
        for piece in range(1, pieces):
            cut = (bins[0] + (span_bins * piece) // pieces) * size
            if cuts[-1] < cut < chunk.fetch_end:
                cuts.append(cut)
        cuts.append(chunk.fetch_end)
        for i in range(len(cuts) - 1):
            out.append(ScanChunk(
                len(out), chunk.chrom, cuts[i], cuts[i + 1],
                chunk.keep_from if i == 0 else cuts[i],
                chunk.keep_until if i == len(cuts) - 2 else cuts[i + 1]))
    return out


def plan_run_chunks(reader, config: PipelineConfig, workers: int) -> list[ScanChunk]:
    """The chunks for this run's BAM and region scope.

    With the automatic chunk width, heavy chunks are then split
    (`split_heavy_chunks`); a width set by hand is taken as given.
    """
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
    chunks = plan_chunks(spans, chunk_bp, config.bin_size, scope.enabled)
    if config.scan_chunk_bp or workers <= 1:
        return chunks
    return split_heavy_chunks(chunks, chunk_data_estimates(reader, chunks), config.bin_size)


# ------------------------------------------------------------------ the worker
#: Per-process state, set once by `_init_worker`. Module globals because a
#: `multiprocessing` initializer has nowhere else to put it.
_WORKER: dict = {}


def _init_worker(config: PipelineConfig) -> None:
    from placer.io.bam import make_bam_reader
    from placer.io.reference import ReferenceFetcher
    from placer.io.te_library import load_te_library
    from placer.wiring import build_stage_hooks

    # One decompression thread per handle: the pool already has a process
    # per core, and htslib's own threads would only compete with them.
    reader = make_bam_reader(config.bam_path, 1, None)
    reference = ReferenceFetcher(config.reference_fasta_path)
    entries = load_te_library(config.te_fasta_path)
    _WORKER.update(config=config, reader=reader, reference=reference,
                   hooks=build_stage_hooks(config, reference, entries))


def scan_chunk(chunk: ScanChunk, reader, config: PipelineConfig,
               hooks: StageHooks) -> PipelineResult:
    """One chunk, gated and scanned, UNFINALIZED.

    The chunk is given EVERY read overlapping it, because its bins need the
    reads that started in an earlier chunk too (`group_reads_into_bins`), and
    it scans only its own bins. The gate's tallies count only the reads the
    chunk `keeps` -- those starting inside it -- so each read is counted once
    across the run.
    """
    result = PipelineResult()
    # The local fetches are answered from the chunk's own stream wherever it
    # provably holds every read (`placer/io/bam.StreamedReadBuffer`).
    overlapping: Iterator[AlignedRead]
    overlapping, fetch_local = reader.buffered_stream_interval(
        chunk.chrom, chunk.fetch_start, chunk.fetch_end)
    source = ReadSource(reads=gate_reads(overlapping, result, count=chunk.keeps),
                        chromosome_name=reader.chromosome_name,
                        fetch_local=fetch_local)
    return run_scan(source, config, hooks, result,
                    bin_range=chunk.bin_range(config.bin_size))


def _scan_chunk_in_worker(chunk: ScanChunk) -> tuple[PipelineResult, dict]:
    """The chunk's result, and what it cost this worker (`placer/io/perf.py`)."""
    before = snapshot()
    part = scan_chunk(chunk, _WORKER["reader"], _WORKER["config"], _WORKER["hooks"])
    return part, span(before, f"{chunk.chrom}:{chunk.fetch_start}-{chunk.fetch_end}")


# ------------------------------------------------------------------ the parent
class WorkerDiedError(RuntimeError):
    """A scan worker process died without returning its chunk."""


def map_in_worker_processes(fn, items, workers: int, initializer=None,
                            initargs: tuple = (), start_order=None) -> Iterator:
    """`map(fn, items)` on `workers` spawned processes, in input order.

    `start_order`, a permutation of the item indices, is the order the items
    are STARTED in; they are still yielded in input order, so a caller that
    merges by appending gets genome order whatever it is.

    Raises `WorkerDiedError` if a worker process dies -- killed by a signal,
    or exited -- where `multiprocessing.Pool` would wait for it forever.
    """
    items = list(items)
    order = list(range(len(items))) if start_order is None else list(start_order)
    if sorted(order) != list(range(len(items))):
        raise ValueError("start_order must be a permutation of the item indices")
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=initializer,
                                 initargs=initargs) as pool:
            futures = [None] * len(items)
            for index in order:
                futures[index] = pool.submit(fn, items[index])
            for index, future in enumerate(futures):
                result = future.result()
                futures[index] = None      # the merged result is the caller's now
                yield result
    except BrokenProcessPool as error:
        raise WorkerDiedError(str(error)) from error


def expensive_first(reader, chunks: list[ScanChunk]) -> list[int]:
    """Chunk indices, the ones holding the most alignment data first.

    WHY. A chunk costs roughly what it holds, and in a collapsed
    pericentromere far more; started in genome order, such a chunk begins
    halfway through the run and one worker finishes it long after the rest
    are idle (HG002 chr1: 2 h 55 min on 16 cores at 352% CPU, 1.0.0a1).
    Started first, it overlaps everything else. The estimate is the
    compressed-byte span between chunk starts from the BAM index, a
    millisecond per chunk; ties and chunks it cannot estimate keep genome
    order. Only the start order changes -- the merge is still by index.
    """
    if len(chunks) < 2:
        return list(range(len(chunks)))
    costs = chunk_data_estimates(reader, chunks)
    return sorted(range(len(chunks)), key=lambda index: (-costs[index], index))


def run_pipeline_parallel(reader, config: PipelineConfig, workers: int,
                          target_fdr: float | None = None,
                          progress: bool = True,
                          perf_log: PerfLog | None = None) -> PipelineResult:
    """Scan on `workers` processes, merge in genome order, finalize once.

    `reader` is the parent's own open reader, used only to plan the chunks;
    the workers open their own. The caller is expected to have built the BLAST
    database already (building the stage hooks and aligning once does it), so
    that the workers find it on disk instead of racing to create it.

    Each chunk's own cost is on its progress line, and is a row of `perf_log`
    when there is one. Lines come in genome order, so a slow chunk holds back
    the lines of the chunks after it; the elapsed time at the end of the line
    is the run's, the chunk's own is in the middle.
    """
    chunks = plan_run_chunks(reader, config, workers)
    merged = PipelineResult()
    started = time.perf_counter()
    done = 0
    try:
        for part, usage in map_in_worker_processes(_scan_chunk_in_worker, chunks,
                                                   workers, _init_worker, (config,),
                                                   start_order=expensive_first(reader,
                                                                               chunks)):
            done += 1
            merge_scan_results(merged, part)
            if perf_log is not None:
                perf_log.write(usage)
            if progress:
                cpu = (usage["user_s"] + usage["sys_s"] + usage["child_user_s"]
                       + usage["child_sys_s"])
                print(f"[PLACER] scanned chunk {done}/{len(chunks)} "
                      f"{chunks[done - 1].chrom}:{chunks[done - 1].fetch_start}-"
                      f"{chunks[done - 1].fetch_end} "
                      f"(ledger rows {len(merged.evidence_ledger)}, "
                      f"chunk {usage['wall_s']:.0f}s wall {cpu:.0f}s cpu, "
                      f"{time.perf_counter() - started:.0f}s)",
                      file=sys.stderr, flush=True)
    except WorkerDiedError as error:
        raise WorkerDiedError(
            f"a scan worker died after {done} of {len(chunks)} chunks. A worker "
            "killed by a signal leaves no Python traceback: run once with "
            "--threads 1 on the same region to see the crash, and check the "
            "compiled dependencies (pysam, pyabpoa) import and run on this "
            "machine's CPU -- the bioconda pyabpoa 1.5.3 build dies of SIGILL "
            "on AMD EPYC (Zen 3) nodes; 1.5.4 and later do not.") from error.__cause__
    return finalize_run(merged, config, target_fdr)
