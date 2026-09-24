"""Finalization, as the run-level stage the scan cannot be.

This is a thin adapter over `placer_py/finalization.finalize_final_calls`: it
unpacks the two policy numbers out of `PipelineConfig` and nothing else. The
2800 lines of actual finalization stay where they are; what is added here is a
NAME for the boundary, so that "scan" and "finalize" are two things a caller
can ask for separately rather than one fused call.

WHY THE BOUNDARY IS WORTH NAMING. Three of finalization's jobs are impossible
locally and that is exactly what makes it a different stage: measuring the
run's own null (the dependency bound is a whole-run expectation), controlling
FDR across the run (e-BH and the conformal route both need the full candidate
set), and recognising that a long insertion reported in several bins is one
event. A pipeline that made final decisions inside the bin loop could do none
of them.

`target_fdr` is an explicit override rather than only a config field because a
caller comparing selection routes wants to sweep it over one scan without
rebuilding the config -- which is the whole reason for splitting scan from
finalize in the first place.
"""

from __future__ import annotations

from placer_py.config import PipelineConfig
from placer_py.core.ledger import FinalCallFilterConfig
from placer_py.core.result import PipelineResult
from placer_py.finalization import finalize_final_calls


def finalize_run(result: PipelineResult, config: PipelineConfig,
                 target_fdr: float | None = None) -> PipelineResult:
    """Aggregate, de-duplicate, calibrate and select, once, over the whole run.

    Mutates `result` in place and also returns it, so it reads naturally at the
    end of a pipeline expression.
    """
    finalize_final_calls(
        result,
        target_fdr if target_fdr is not None else config.final_fdr_q,
        FinalCallFilterConfig(
            min_raw_cigar_insert_len_bp=config.min_final_raw_cigar_insert_len_bp,
            report_mode=config.final_report_mode.value))
    return result
