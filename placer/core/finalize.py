"""Finalization, as the run-level stage the scan cannot be.

This is a thin adapter over `placer/finalization.finalize_final_calls`: it
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

from placer.config import PipelineConfig
from placer.core.finalization import finalize_final_calls
from placer.core.ledger import FinalCallFilterConfig
from placer.core.result import PipelineResult


def finalize_run(result: PipelineResult, config: PipelineConfig,
                 target_fdr: float | None = None) -> PipelineResult:
    """Aggregate, de-duplicate, calibrate and select, once, over the whole run.

    Mutates `result` in place and also returns it, so it reads naturally at the
    end of a pipeline expression.
    """
    q = target_fdr if target_fdr is not None else config.final_fdr_q
    if config.decision_mode == "mechanism":
        finalize_mechanism_calls(result, q, config.mechanism_te_rule)
        return result
    finalize_final_calls(
        result,
        q,
        FinalCallFilterConfig(
            min_raw_cigar_insert_len_bp=config.min_final_raw_cigar_insert_len_bp,
            report_mode=config.final_report_mode.value))
    return result


def finalize_mechanism_calls(result: PipelineResult, q: float,
                             te_rule: str = "likelihood") -> None:
    """The mechanism decision: decoy check and e-BH over every evaluated call.

    Each locus's best call is tested once (`core/mechanism_selection.py`). The
    TE-selected ones are the run's calls, named from their own alignment -- the
    legacy decision overwrote the family of a call it judged structural -- and
    the structural-selected ones go to `structural_calls`. The ledger gets the
    same selection marked on its rows, for the summary and for audit.
    """
    from placer.core.finalization import (
        apply_sample_overdispersion_calibration,
        final_call_sort_less,
    )
    from placer.core.mechanism_selection import (
        apply_mechanism_shadow_selection,
        select_loci,
        select_loci_coverage,
    )
    coverage = te_rule == "coverage"
    if coverage:
        select_loci_coverage([row for row in result.evidence_ledger
                              if row.candidate_retention_reason == "EVALUATED"], q)
        result.mech_shadow = select_loci_coverage(result.candidate_calls, q)
    else:
        apply_mechanism_shadow_selection(result.evidence_ledger, q)
        result.mech_shadow = select_loci(result.candidate_calls, q)
    te_calls, structural = [], []
    for call in result.candidate_calls:
        if coverage and (call.mech_ebh_selected or call.mech_structural_selected):
            # Placed at the locus's precise hypothesis, not wherever the
            # legacy retether moved it; named after the family covering most
            # of the insert.
            if call.mech_call_pos >= 0:
                call.pos = call.bp_left = call.bp_right = call.mech_call_pos
            else:
                call.pos = call.hypothesis_pos
            if call.mech_ebh_selected and call.te_best_family != call.te_dominant_family:
                # The best-scoring subfamily belongs to another family.
                call.te_best_subfamily = "NA"
            if call.mech_ebh_selected:
                call.te_annotation_class = call.te_dominant_class
                call.te_best_family = call.te_dominant_family
        if call.mech_ebh_selected:
            call.family = call.te_best_family if call.te_best_family else "UNKNOWN"
            call.subfamily = call.te_best_subfamily if call.te_best_subfamily else "NA"
            call.te_name = call.subfamily if call.subfamily not in ("NA", "") else call.family
            call.family_committed = call.te_annotation_class not in ("NA", "Unknown", "NonTE")
            call.final_qc = _with_token(call.final_qc, "PASS_TE_MECHANISM")
            call.ebh_selected, call.ebh_e_value = True, call.mech_e_value
            # 1/e bounds a valid p-value (Markov), so the VCF's QUAL, the phred
            # transform of this field, reads 10*log10(e).
            call.lfdr = min(1.0, 1.0 / call.mech_e_value) if call.mech_e_value > 0 else 1.0
            te_calls.append(call)
        elif call.mech_structural_selected:
            call.family, call.subfamily, call.te_name = "UNKNOWN", "UNKNOWN", "UNKNOWN"
            call.family_committed = False
            call.final_qc = _with_token(call.final_qc, "PASS_STRUCTURAL_MECHANISM")
            # Selected by e-BH too, on the artifact question alone, so FDR-
            # controlled -- not the UNCALIB route.
            call.ebh_selected = True
            structural.append(call)
    result.final_calls = te_calls
    apply_sample_overdispersion_calibration(result)
    result.final_calls.sort(key=final_call_sort_less)
    structural.sort(key=final_call_sort_less)
    result.structural_calls = structural
    result.candidate_calls = []
    result.final_pass_calls = len(result.final_calls)


def _with_token(qc: str, token: str) -> str:
    """Append a decision token, keeping the evidence tokens (IMPRECISE...)."""
    return token if not qc or qc == "NA" else f"{qc}|{token}"
