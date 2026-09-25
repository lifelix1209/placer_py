"""The composition root: gate, scan, finalize.

Ported from `src/pipeline/pipeline_entrypoints.inc` and the stage wiring in
`src/pipeline/pipeline.cpp`, pinned by `tests/test_32_pipeline.py`.

THE SHAPE OF A RUN:

    stream the BAM                                    placer/io/bam
      -> gate1: is this read worth carrying?           placer/io/gate
      -> group into bins                               placer/core/bins
      -> cluster signatures into components            placer/clustering
      -> fetch each component's local reads once       placer/interval_cache
      -> extract insert fragments                      placer/fragments
      -> enumerate breakpoint hypotheses               placer/breakpoints
      -> tally reads per hypothesis, collapse, triage  placer/events,
                                                       placer/hypotheses
      -> for the shortlist only:
           consensus                                   placer/consensus
           segmentation                                placer/segmentation
           TE alignment                                placer/te_classifier
           boundary + joint decision                   placer/policy
      -> select one call per component, re-anchor      placer/call_selection
    then, once, over the whole run:
      -> finalize: aggregate, dedup, calibrate, select placer/core/finalize

WHY THIS MODULE IS ALMOST EMPTY, and why that is the improvement. It used to
hold all of the above; now it holds the three-line sentence that joins them.
The per-bin work is `placer/core/bins.py`, the loop that drives it is
`placer/core/scan.py`, the whole-run stage is `placer/core/finalize.py`,
and the read filter is `placer/io/gate.py`. Each can now be called, tested
and replaced on its own.

THIS IS ALSO THE ONE MODULE IN THE PACKAGE ALLOWED TO IMPORT BOTH SIDES.
`placer/core/**` may not import `placer.io` -- that rule is what keeps
the algorithm runnable with no BAM, no BLAST and no abPOA, and
`tests/test_37_layering.py` enforces it. A composition root exists precisely so
that the rule can hold everywhere else.

`run_pipeline` KEEPS ITS SIGNATURE, including the parameter order, because
`tests/test_32_pipeline.py` calls it positionally and reads the gate counters
off its return value. A caller that wants an ungated scan, or a second
finalization over one scan, calls `run_scan` and `finalize_run` directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from placer.alignment import AlignedRead
from placer.config import PipelineConfig
from placer.core.contracts import ReadSource, StageHooks
from placer.core.finalize import finalize_run
from placer.core.result import PipelineResult
from placer.core.scan import run_scan
from placer.io.gate import gate_reads

__all__ = ["ReadSource", "StageHooks", "run_pipeline"]


def run_pipeline(reads: Iterable[AlignedRead], chromosome_name: Callable[[int], str],
                 fetch_local: Callable[[str, int, int], list[AlignedRead]],
                 config: PipelineConfig, hooks: StageHooks,
                 target_fdr: float | None = None) -> PipelineResult:
    """Gate the reads, scan every bin, then finalize once.

    THE SPLIT IS THE POINT. Everything inside the scan is per-bin and knows
    nothing about the rest of the genome; `finalize_run` runs once and is the
    only stage that can measure the run's own null, control FDR across it, or
    recognise that two bins reported one event. A pipeline that made final
    decisions inside the loop could do none of those.

    The gate writes its tallies straight into `result`, which is why it is
    constructed here and threaded through rather than returned from `run_scan`.
    """
    result = PipelineResult()
    source = ReadSource(reads=gate_reads(reads, result),
                        chromosome_name=chromosome_name,
                        fetch_local=fetch_local)
    run_scan(source, config, hooks, result)
    return finalize_run(result, config, target_fdr)
