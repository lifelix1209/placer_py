"""The scan: drive every bin, and stop there.

WHY THIS IS A SEPARATE FUNCTION FROM FINALIZATION. They are two different
kinds of computation and only one of them can be done while reading. The scan
is per-bin and local -- it never needs to know what happened on another
chromosome -- so it can run in a single streaming pass over a coordinate-sorted
file. Finalization needs the whole run at once. Fusing them, which is what
`run_pipeline` used to do, meant there was no way to ask for one without the
other: no way to scan and inspect the raw ledger before anything was
calibrated, and no way to re-run a calibration on a scan you already have.

THE STREAMING PROPERTY LIVES HERE and is easy to destroy by accident.
`source.reads` is an `Iterator`, `group_reads_into_bins` is a generator, and a
bin is handed over the moment a read with a different key arrives. Wrapping
either in a `list()` would produce byte-identical output and make a 10 Mb
region of ultra-long ONT cost 2.2 GB, because every `AlignedRead` holds its
full sequence. `tests/test_35_stages.py` and `tests/test_32_pipeline.py` both
watch for that from outside, by counting how far the source has advanced.

IT TAKES AN OPTIONAL `result` because the input stage's gate has already been
writing to one: `placer/io/gate.py` increments `total_reads` and
`gate1_passed` as it yields. Passing the same record in is what keeps those
tallies and the per-bin counters on one object without a copy step.
"""

from __future__ import annotations

from placer.config import PipelineConfig
from placer.core.bins import group_reads_into_bins, process_bin_records
from placer.core.contracts import ReadSource, StageHooks
from placer.core.result import PipelineResult


def run_scan(source: ReadSource, config: PipelineConfig, hooks: StageHooks,
             result: PipelineResult | None = None) -> PipelineResult:
    """Stream the gated reads through every per-bin stage.

    Returns the result UNFINALIZED: the calls are per-component selections, the
    ledger carries raw penalty-free aggregates, and nothing has been calibrated
    or de-duplicated across bins yet. `placer/core/finalize.py` is what turns
    that into an answer.
    """
    result = result if result is not None else PipelineResult()
    bin_size = max(1, config.bin_size)
    for tid, bin_index, bin_reads in group_reads_into_bins(source.reads, bin_size):
        process_bin_records(bin_reads, source.chromosome_name(tid), tid,
                            bin_index * bin_size, (bin_index + 1) * bin_size,
                            config, hooks, result, source.fetch_local)
    return result


#: The counters `run_scan` and the gate accumulate. Every other field of
#: `PipelineResult` is written by finalization, which has not run yet.
_SCAN_COUNTERS = ("total_reads", "gate1_passed", "processed_bins", "built_components",
                  "event_consensus_calls", "genotype_calls")


def merge_scan_results(into: PipelineResult, part: PipelineResult) -> PipelineResult:
    """Append one UNFINALIZED scan onto another, as if it had followed it.

    For the parallel scan (`placer/parallel.py`): `part` must be the scan of
    the reads that come after everything already in `into`, and then the
    merge is exactly what one longer scan would have accumulated -- counters
    add, lists concatenate in order. It refuses a finalized part, because a
    part's calibration fields describe that part alone and there is no way to
    combine them into what the whole run would have measured.
    """
    if part.structural_calls or part.final_pass_calls:
        raise ValueError("merge_scan_results takes unfinalized scans only")
    for name in _SCAN_COUNTERS:
        setattr(into, name, getattr(into, name) + getattr(part, name))
    into.final_calls.extend(part.final_calls)
    into.evidence_ledger.extend(part.evidence_ledger)
    return into
