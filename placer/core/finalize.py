"""Finalization, as the run-level stage the scan cannot be.

WHY THE BOUNDARY IS WORTH NAMING. The scan is per-bin and local. Finalization
needs the whole run at once, for three jobs no bin can do:
  * the decoy check, which measures the TSD and endonuclease-motif nulls
    over the run;
  * e-BH, which controls FDR across the full candidate set;
  * the overdispersion the genotyper uses, estimated from the whole ledger.

THE DECISION (`core/mechanism_selection.select_loci_coverage`, validated on
HG002 chr2-8 on 2026-09-26):
  1. alignment-collapse regions get e = 0;
  2. one e-BH on the decoy-adjusted artifact ratio: the insertions;
  3. a TE call when TEBench's coverage rule holds on the insert. An insertion
     whose insert is not a TE is recorded (the ledger's
     `mech_structural_selected`, the summary's count) and not reported:
     PLACER is a TE caller;
  4. precise placement.
The legacy decision (`--decision legacy`, the old `core/finalization.py`) was
removed after it: its VCF reported calls kilobytes from their own positions.

`target_fdr` is an explicit override rather than only a config field, so that
a caller can sweep q over one scan without rebuilding the config.
"""

from __future__ import annotations

from placer.config import PipelineConfig
from placer.core.ledger import FinalCall
from placer.core.result import PipelineResult


def finalize_run(result: PipelineResult, config: PipelineConfig,
                 target_fdr: float | None = None) -> PipelineResult:
    """Select, name, place and genotype, once, over the whole run.

    Mutates `result` in place and also returns it, so it reads naturally at the
    end of a pipeline expression.
    """
    q = target_fdr if target_fdr is not None else config.final_fdr_q
    finalize_mechanism_calls(result, q)
    return result


def finalize_mechanism_calls(result: PipelineResult, q: float) -> None:
    """The decision over every evaluated call, and the ledger's marks.

    Each locus is tested once (`core/mechanism_selection.py`). The TE-selected
    ones are the run's calls, named after the family covering the most of their
    insert and placed at the locus's precise hypothesis. The ledger gets the
    same selection marked on its rows, the structural ones included, for the
    summary and for replay.
    """
    from placer.core.mechanism_selection import select_loci_coverage

    select_loci_coverage([row for row in result.evidence_ledger
                          if row.candidate_retention_reason == "EVALUATED"], q)
    result.mech_shadow = select_loci_coverage(result.candidate_calls, q)
    te_calls = []
    for call in result.candidate_calls:
        if call.mech_ebh_selected:
            # Placed at the locus's precise hypothesis, when the testing one
            # was a wide interval, and otherwise where the hypothesis itself
            # sits.
            if call.mech_call_pos >= 0:
                call.pos = call.bp_left = call.bp_right = call.mech_call_pos
            else:
                call.pos = call.hypothesis_pos
            if call.te_best_family != call.te_dominant_family:
                # The best-scoring subfamily belongs to another family.
                call.te_best_subfamily = "NA"
            call.te_annotation_class = call.te_dominant_class
            call.te_best_family = call.te_dominant_family
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
    result.final_calls = te_calls
    apply_sample_overdispersion_calibration(result)
    result.final_calls.sort(key=final_call_sort_less)
    result.candidate_calls = []
    result.final_pass_calls = len(result.final_calls)


def _with_token(qc: str, token: str) -> str:
    """Append a decision token, keeping the evidence tokens (IMPRECISE...)."""
    return token if not qc or qc == "NA" else f"{qc}|{token}"


def final_call_sort_less(call: FinalCall) -> tuple:
    """A total order for output. Position first, then window, then name."""
    return (call.tid, call.pos, call.window_start, call.window_end, call.chrom,
            call.te_name)


def apply_sample_overdispersion_calibration(result: PipelineResult) -> None:
    """Re-estimate the beta-binomial overdispersion, then re-genotype.

    Over the WHOLE ledger, not just the calls: the overdispersion is a property
    of the sample's sequencing and mapping, and estimating it from the selected
    calls alone would measure it on the loci least representative of the rest.

    The configured error rate and min-GQ threshold the call was decided with
    are reused; rebuilding a default input here would silently drop them.

    The length-concordance term is NOT. It is existence evidence -- do the alt
    reads measure this event? -- and existence was settled by the scan and
    e-BH before this runs. The reported genotype is zygosity, from the allele
    counts. With the term, the model charges the mean discordance of the few
    reads that report a length (soft-clip lengths among them, which are only
    lower bounds) once per alt read, while the sample's overdispersion caps the
    count evidence. So the penalty outgrows the evidence with depth.
    HG002 chr1-8, whole genome, 1.0.0a2: 22% of PASS calls came out 0/0 at
    full depth, 0% at 5x. Every one of the 165 true positives called 0/0 had
    counts favouring an alt genotype (66 alt, 0 ref at chr2:171976066).
    Dropping it moves no call. Genotype concordance with GIAB on the chr2-8
    release runs went from 83.6% to 95.1%.
    """
    from placer.core.genotype import estimate_overdispersion, genotype_from_alt_vs_ref

    observations = [(max(0, row.alt_struct_reads),
                     max(0, row.alt_struct_reads) + max(0, row.ref_span_reads))
                    for row in result.evidence_ledger]
    rho = estimate_overdispersion(observations, result.estimated_overdispersion)
    result.estimated_overdispersion = rho

    for call in result.final_calls:
        depth = max(0, call.alt_struct_reads) + max(0, call.ref_span_reads)
        if depth <= 0:
            continue
        inputs = call.genotype_likelihood_input
        inputs.alt_struct_reads = call.alt_struct_reads
        inputs.ref_span_reads = call.ref_span_reads
        inputs.overdispersion = rho
        decision = genotype_from_alt_vs_ref(
            inputs.alt_struct_reads, inputs.ref_span_reads,
            error_rate=inputs.error_rate, overdispersion=rho,
            min_gq=inputs.min_gq)
        call.gq = decision.gq
        call.af = decision.allele_fraction
        call.genotype = decision.best_gt
        call.genotype_likelihood_input = inputs
