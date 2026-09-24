"""Binding the algorithm's four hooks to real I/O.

`placer_py/core/contracts.py` declares what the algorithm cannot do for itself
-- fetch a reference window, align an insert against a TE library, build a
consensus, detect a TSD -- as four callables. This module is the only place
those are filled in with something that opens a file or spawns a process.

WHY IT IS NOT IN `main.py`. The CLI's job is argv, the environment table and
exit codes. Deciding that the consensus comes from abPOA, or that the TSD
detector should be handed the insert sequence as well as the breakpoints, is
not a command-line concern -- it is the assembly of a run, and a caller
embedding this package wants it without inheriting an argument parser.

WHY IT IS NOT IN `placer_py/io/` EITHER. It imports both `core` and `io`, which
the layering rule allows for composition modules and forbids everywhere else
(`tests/test_37_layering.py`). Putting it under `io/` would make that package
depend on `core`, which is the one direction the split exists to prevent.

THE TE ALIGNMENT IS HANDED OVER PER BIN. The bin loop takes every shortlisted
hypothesis of a bin up to its alignment, then calls `align_inserts` once for
the lot (`placer_py/core/bins.py`). `TeLibraryAligner` still gives each insert
its own `blastn` -- packing them into one run changes blastn's answer for
repetitive inserts -- but runs a bin's processes concurrently and remembers
the hits per sequence across bins. BLAST+ 2.17 spends ~0.8 s of CPU starting
up, more than the search, so overlapping the start-ups is where the time is.
Handing over per BIN rather than per run keeps the streaming property -- no
bin's reads are held past the bin.
"""

from __future__ import annotations

import os

from placer_py.config import PipelineConfig
from placer_py.core.contracts import StageHooks
from placer_py.core.seqtools import build_te_sequence_background
from placer_py.core.tsd import TsdConfig
from placer_py.core.tsd import detect as detect_tsd
from placer_py.core.tsd import detect_from_insertion as detect_tsd_from_insertion
from placer_py.io.poa import pyabpoa_consensus
from placer_py.io.te_library import TeLibraryAligner

#: See `blast_jobs_per_worker`.
MIN_BLAST_JOBS = 4


def blast_jobs_per_worker(config: PipelineConfig) -> int:
    """How many `blastn` processes one scan process may run at once.

    `te_blast_jobs` when set; otherwise the machine's cores shared between the
    scan workers, so `--threads N` does not multiply into N x cores blastn
    processes -- but never fewer than `MIN_BLAST_JOBS`. With one job, a
    worker alternates between its own Python and a queue of `blastn`
    launches (~0.8 s of CPU and ~1.3 s of wall time each on the measurement
    laptop), and one core sits idle through each half. A few in flight keep
    it busy; the oversubscription is bounded by `workers x MIN_BLAST_JOBS`,
    and measured on 0.9 Mb of HG002 with 8 workers it was the faster choice.
    """
    if config.te_blast_jobs > 0:
        return config.te_blast_jobs
    return max(MIN_BLAST_JOBS, (os.cpu_count() or 1) // max(1, config.scan_workers))


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
