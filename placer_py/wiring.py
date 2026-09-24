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

ONE KNOWN INEFFICIENCY, recorded here because this is where it lives.
`align_insert` wraps `align_insert_sequences` -- a BATCH api with by-sequence
de-duplication -- in a single-element list, so every hypothesis that reaches
the expensive stages spawns its own `blastn`. On a whole-genome run that is
10^4 to 10^5 subprocesses where a handful would do. Fixing it means collecting
the shortlist's insert sequences before aligning any of them, which changes
the scan from one pass into two and interacts with the streaming memory
property, so it is a separate measured change rather than something to slip in
alongside a restructuring.
"""

from __future__ import annotations

from placer_py.config import PipelineConfig
from placer_py.core.contracts import StageHooks
from placer_py.core.seqtools import build_te_sequence_background
from placer_py.core.tsd import TsdConfig
from placer_py.core.tsd import detect as detect_tsd
from placer_py.core.tsd import detect_from_insertion as detect_tsd_from_insertion
from placer_py.io.poa import pyabpoa_consensus
from placer_py.io.te_library import align_insert_sequences


def build_stage_hooks(config: PipelineConfig, reference, entries) -> StageHooks:
    """The four external dependencies, bound.

    `entries` is the loaded TE library rather than a path: the caller has
    already had to read it to know whether it was empty, and re-reading it per
    run would make the emptiness check and the alignment disagree about what
    the library is.
    """
    background = build_te_sequence_background([entry.sequence for entry in entries])

    def align_insert(insert_seq: str):
        return align_insert_sequences(config, entries, [insert_seq],
                                      background)[0]

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
                      align_insert=align_insert,
                      consensus_fn=pyabpoa_consensus,
                      detect_tsd=detect)
