"""Binding the algorithm's four hooks to real I/O.

`placer/core/contracts.py` declares what the algorithm cannot do for itself
-- fetch a reference window, align an insert against a TE library, build a
consensus, detect a TSD -- as four callables. This module is the only place
those are filled in with something that opens a file or spawns a process.

WHY IT IS NOT IN `main.py`. The CLI's job is argv, the environment table and
exit codes. Deciding that the consensus comes from abPOA, or that the TSD
detector should be handed the insert sequence as well as the breakpoints, is
not a command-line concern -- it is the assembly of a run, and a caller
embedding this package wants it without inheriting an argument parser.

WHY IT IS NOT IN `placer/io/` EITHER. It imports both `core` and `io`, which
the layering rule allows for composition modules and forbids everywhere else
(`tests/test_37_layering.py`). Putting it under `io/` would make that package
depend on `core`, which is the one direction the split exists to prevent.

THE TE ALIGNMENT IS HANDED OVER PER BIN. The bin loop takes every shortlisted
hypothesis of a bin up to its alignment, then calls `align_inserts` once for
the lot (`placer/core/bins.py`). `TeLibraryAligner` packs them into sorted
batches of `BLAST_QUERIES_PER_CALL` and runs a bin's batches concurrently.
BLAST+ 2.17 spends ~0.8 s of CPU starting up, more than the search for a
few-hundred-base insert, so the start-ups were where the time was.
Handing over per BIN rather than per run keeps the streaming property -- no
bin's reads are held past the bin.
"""

from __future__ import annotations

import os

from placer.config import PipelineConfig
from placer.core.contracts import StageHooks
from placer.core.seqtools import build_te_sequence_background
from placer.core.tsd import TsdConfig
from placer.core.tsd import detect as detect_tsd
from placer.core.tsd import detect_from_insertion as detect_tsd_from_insertion
from placer.io.poa import pyabpoa_consensus
from placer.io.te_library import TeLibraryAligner

#: See `blast_jobs_per_worker`.
MIN_BLAST_JOBS = 2


def available_cpus() -> int:
    """CPUs this process may run on: the SLURM/cgroup allocation, not the machine.

    `os.cpu_count()` reports every core on the node. On a 128-core cluster node
    with a 16-CPU allocation it said 128, each of 16 workers ran 8 blastn at
    once, and two such jobs put the node at a load of 233 -- which made every
    timing on it meaningless and every job on it slow. The affinity mask is
    what the scheduler actually granted.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):      # macOS has no sched_getaffinity
        return max(1, os.cpu_count() or 1)


def blast_jobs_per_worker(config: PipelineConfig) -> int:
    """How many `blastn` processes one scan process may run at once.

    `te_blast_jobs` when set; otherwise the process's allocated CPUs shared
    between the scan workers, so `--threads N` does not multiply into N x cores
    blastn processes -- but never fewer than `MIN_BLAST_JOBS`, so a worker has
    one batch aligning while it prepares the next. Since inserts are batched 32
    to a process (`io/te_library.py`), start-up no longer dominates, and the
    floor of 4 this used to have -- chosen on a laptop to overlap one-insert
    start-ups -- only oversubscribed a cluster allocation fourfold.
    """
    if config.te_blast_jobs > 0:
        return config.te_blast_jobs
    return max(MIN_BLAST_JOBS, available_cpus() // max(1, config.scan_workers))


def build_stage_hooks(config: PipelineConfig, reference, entries) -> StageHooks:
    """The four external dependencies, bound.

    `entries` is the loaded TE library rather than a path: the caller has
    already had to read it to know whether it was empty, and re-reading it per
    run would make the emptiness check and the alignment disagree about what
    the library is.
    """
    background = build_te_sequence_background([entry.sequence for entry in entries])
    aligner = TeLibraryAligner(config, entries, background,
                               jobs=blast_jobs_per_worker(config))

    tsd_config = TsdConfig(tsd_min_len=config.tsd_min_len,
                           tsd_max_len=config.tsd_max_len,
                           tsd_flank_window=config.tsd_flank_window,
                           tsd_bg_p_max=config.tsd_bg_p_max,
                           tsd_max_mismatch_rate=config.tsd_max_mismatch_rate,
                           tsd_max_mismatches=config.tsd_max_mismatches)

    def detect(chrom: str, bp_left: int, bp_right: int, insert_seq: str):
        """Pick the detector the evidence can actually support.

        Distinct breakpoints mean the caller resolved both edges of the event,
        and their overlap (or gap) is measurable in the reference -- that is
        `detect`. Equal breakpoints mean the aligner emitted a single CIGAR
        `I`, so the reference carries no trace of the duplication and the only
        place left to look is the inserted sequence itself.
        """
        if not config.tsd_enable:
            return None
        if bp_left != bp_right:
            return detect_tsd(reference.fetch_window, chrom, bp_left, bp_right,
                              tsd_config)
        if insert_seq:
            return detect_tsd_from_insertion(reference.fetch_window, chrom,
                                             bp_left, insert_seq, tsd_config)
        return None

    return StageHooks(fetch_reference=reference.fetch_window,
                      align_insert=aligner.align_one,
                      align_inserts=aligner.align,
                      consensus_fn=pyabpoa_consensus,
                      detect_tsd=detect)
