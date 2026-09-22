"""
Finalization: from a ledger of candidates to a set of calls, with a risk level.

Ported from `src/pipeline/pipeline_finalization_stage.inc` -- the largest single
unit in the C++ -- and pinned by `tests/test_31_finalization.py`.

WHAT THIS STAGE IS FOR. Everything before it works on ONE locus at a time.
Finalization is the only stage that sees the whole run, and that is what it is
for: three of its jobs are impossible locally.

  1. MEASURE THE RUN'S OWN NULL. The dependency bound that makes a product of
     block e-values an e-value is `E_null[prod E_k]`, a whole-run expectation.
     Per-locus screening charges no penalty at all (see
     `placer_py/blocks.py`), so the raw aggregates travel here and the certified
     ones are computed from them. Same for the beta-binomial overdispersion and
     for the conformal null set.

  2. CONTROL FDR ACROSS THE RUN. e-BH and the conformal/BY route both need the
     full set of candidates. A per-locus threshold cannot bound the proportion
     of wrong calls among the reported ones.

  3. STOP ONE EVENT BECOMING SEVERAL CALLS. A long insertion produces evidence
     at both breakpoints, in several bins, from overlapping read sets. Most of
     the code here is about recognising that and collapsing it -- and the
     recognition is by SHARED SUPPORTING READS rather than by position alone,
     because two calls 300 bp apart built from the same twelve reads are one
     event and two calls 30 bp apart built from disjoint reads may not be.

THE LOCAL-FDR ALGEBRA. Several independent lines of evidence each give a local
false discovery rate, and they are combined in LOG-ODDS space -- adding log odds
is multiplying likelihood ratios, which is the correct combination for
independent evidence and is what `combine_independent_local_fdr` does. The
independence is assumed and is imperfect, exactly as in the latent state model;
what makes it defensible here is that the result is used as ONE input to a
procedure whose guarantee does not depend on it.

THE THREE E-VALUE CONSTRUCTIONS are averaged, not maximised. The maximum of
e-values is NOT an e-value -- that is one of the pinned regressions in
`placer_py/selection.py` -- while their mean is, by linearity of expectation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from placer_py import mathx
from placer_py.ledger import EvidenceLedgerRow, FinalCall, FinalCallFilterConfig

#: The target risk. One number for the whole run.
DEFAULT_FINAL_CONFORMAL_FDR = 0.10
#: Two calls closer than this are the same locus for de-duplication.
FINAL_CALL_DEDUP_DISTANCE_BP = 50
#: The prior on a real low-AF insertion's alt fraction, and the error rate the
#: artifact alternative is scored at. Same numbers as the decision policy's.
ALT_FRACTION_PRIOR_ALPHA = 1.0
ALT_FRACTION_PRIOR_BETA = 9.0
ARTIFACT_ERROR_RATE = 0.02

_INF = float("inf")


# ---------------------------------------------------------------------------
# String predicates on the QC tokens.
# ---------------------------------------------------------------------------
def starts_with_text(value: str, prefix: str) -> bool:
    return value.startswith(prefix)


def is_final_te_qc_for_conformal(final_qc: str) -> bool:
    return starts_with_text(final_qc, "PASS_TE")


def is_final_event_qc_for_conformal(final_qc: str) -> bool:
    """TE calls AND structural insertions. The conformal null set is about
    whether an EVENT is there, not about what kind it is."""
    return (is_final_te_qc_for_conformal(final_qc)
            or starts_with_text(final_qc, "PASS_STRUCTURAL_INSERTION"))


def final_call_has_reportable_conformal_certificate(call: FinalCall) -> bool:
    """Did SOME selection route select this call?

    Five accepted tokens, one per route. They are listed rather than pattern
    matched because each represents a different guarantee -- conformal/BY, the
    lFDR fallback, e-BH, and two Bayesian-FDR fallbacks -- and a run should be
    readable as "which argument selected this".
    """
    return call.conformal_qc in ("PASS_CONFORMAL_FDR", "PASS_EVENT_LFDR_FALLBACK",
                                 "PASS_EVENT_EBH", "PASS_EVENT_BFDR",
                                 "PASS_EVENT_EXISTENCE_BFDR")


def final_call_passes_te_calibrated_report_gate(call: FinalCall) -> bool:
    """All four: a TE verdict, a posterior, an lFDR, and a selection route."""
    return (starts_with_text(call.final_qc, "PASS_TE")
            and call.posterior_qc == "PASS_TE_POSTERIOR"
            and call.lfdr_qc == "PASS_TE_LFDR"
            and final_call_has_reportable_conformal_certificate(call))


def has_conformal_feature_certificate(row: EvidenceLedgerRow) -> bool:
    """Was the mechanistic certificate actually computed for this row?

    Serialised blocks are the evidence that it was. A row without them reached
    the ledger by some path that never evaluated the blocks, and treating its
    default-zero aggregates as measurements would put a fabricated e-value into
    the selection.
    """
    return bool(row.mechanistic_blocks) and row.mechanistic_blocks != "NA"


def is_promoted_event_cluster_call(call: FinalCall) -> bool:
    return "EVENT_CLUSTER_PROMOTED" in call.final_qc


def mechanistic_blocks_include(blocks: str, token: str) -> bool:
    return bool(blocks) and blocks != "NA" and token in blocks


# ---------------------------------------------------------------------------
# The local-FDR algebra.
# ---------------------------------------------------------------------------
def log_sum_exp_pair(lhs: float, rhs: float) -> float:
    """Non-finite operands are IGNORED rather than propagating.

    `-inf` here means "this line of evidence said nothing", not "impossible", so
    it must not poison the sum.
    """
    if not math.isfinite(lhs):
        return rhs
    if not math.isfinite(rhs):
        return lhs
    top = max(lhs, rhs)
    return top + math.log(math.exp(lhs - top) + math.exp(rhs - top))


def odds_to_local_fdr(log_odds: float) -> float:
    """`1 / (1 + exp(log_odds))`, saturated at +/- 60 nats.

    The saturation avoids overflow and costs nothing: e^60 is already far beyond
    any evidence this pipeline can produce.
    """
    if not math.isfinite(log_odds):
        return 1.0
    if log_odds >= 60.0:
        return 0.0
    if log_odds <= -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(log_odds))


def local_fdr_to_log_odds(local_fdr: float) -> float:
    """The inverse. An lFDR of exactly 1 gives -inf, i.e. "no evidence at all",
    which `combine_independent_local_fdr` then drops rather than combining."""
    fdr = min(1.0, max(0.0, local_fdr))
    if fdr <= 0.0:
        return 60.0
    if fdr >= 1.0:
        return -_INF
    return math.log1p(-fdr) - math.log(fdr)


def combine_independent_local_fdr(lhs: float, rhs: float) -> float:
    """Combine two local FDRs by ADDING their log odds.

    Adding log odds multiplies likelihood ratios, which is the correct
    combination for independent evidence. The independence is assumed and
    imperfect -- the statistical and structural lines both read the same
    alignment -- so the result is deliberately used as ONE input to a procedure
    whose guarantee does not rest on it, rather than as the answer.

    An uninformative side (`lFDR = 1`, log odds `-inf`) returns the OTHER side
    unchanged rather than dragging the combination to 1.
    """
    lhs_log_odds = local_fdr_to_log_odds(lhs)
    rhs_log_odds = local_fdr_to_log_odds(rhs)
    if not math.isfinite(lhs_log_odds):
        return rhs
    if not math.isfinite(rhs_log_odds):
        return lhs
    return odds_to_local_fdr(lhs_log_odds + rhs_log_odds)


# ---------------------------------------------------------------------------
# Count models. Same formulas as the decision policy's, re-derived here because
# the C++ redefines them in this translation unit.
# ---------------------------------------------------------------------------
#: Re-exported: the suite names these directly and the C++ has them in this
#: translation unit. Bodies in `placer_py/mathx.py`.
log_choose_count = mathx.log_choose
beta_binomial_log_pmf = mathx.beta_binomial_log_pmf
binomial_log_pmf = mathx.binomial_log_pmf


def low_allele_fraction_event_log_bf(alt: int, ref: int) -> float:
    """Real minority allele vs. a 2% error process, as a log Bayes factor."""
    total = max(0, alt) + max(0, ref)
    if alt <= 0 or total <= 0:
        return -_INF
    return (beta_binomial_log_pmf(alt, total, ALT_FRACTION_PRIOR_ALPHA,
                                  ALT_FRACTION_PRIOR_BETA)
            - binomial_log_pmf(alt, total, ARTIFACT_ERROR_RATE))


def heterozygous_balance_log_evidence(alt: int, ref: int) -> float:
    """How het-like the counts look, scaled by how many reads say so.

    `balance` peaks at AF 0.5 and falls linearly to 0 at 0 or 1; the count term
    is `log1p(min(alt, ref))`. The PRODUCT is what makes this useful: a 1:1
    split of two reads scores almost nothing, while a 1:1 split of forty scores
    a lot, and a 39:1 split of forty scores nothing however deep it is.
    """
    if alt <= 0 or ref <= 0:
        return 0.0
    depth = float(alt + ref)
    af = alt / depth
    balance = 1.0 - min(1.0, abs(af - 0.5) / 0.5)
    count_signal = math.log1p(min(alt, ref))
    return balance * count_signal


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


# ---------------------------------------------------------------------------
# Per-row and per-call certificates.
# ---------------------------------------------------------------------------
def ledger_row_statistical_te_local_fdr(row: EvidenceLedgerRow) -> float:
    """The best local FDR any statistical line of evidence offers for this row.

    A `min` over two lines, each of which is CONSULTED ONLY IF IT PASSED ITS OWN
    GATE. That guard is what stops a row with a nominally low `worst_case_lfdr`
    but a failed QC from contributing it: a number computed on a path that was
    rejected is not evidence.
    """
    local_fdr = 1.0
    if row.lfdr_qc == "PASS_TE_LFDR" or row.worst_case_lfdr <= 0.10:
        local_fdr = min(local_fdr, _clamp01(row.worst_case_lfdr))
    if row.posterior_qc == "PASS_TE_POSTERIOR" or row.te_posterior >= 0.95:
        local_fdr = min(local_fdr, 1.0 - _clamp01(row.te_posterior))
    return local_fdr


def ledger_row_structure_te_local_fdr(row: EvidenceLedgerRow) -> float:
    """TE structure evidence against BOTH competitors, as a local FDR.

    `log_sum_exp` of the non-TE and artifact structure evidence is the marginal
    "not a TE" evidence, and the odds against it become the local FDR. A row
    with no decoded path is 1.0 -- no evidence, not evidence against.
    """
    if not row.te_structure_path or row.te_structure_path == "NA":
        return 1.0
    competing = log_sum_exp_pair(row.nonte_structure_log_evidence,
                                 row.artifact_structure_log_evidence)
    return odds_to_local_fdr(row.te_structure_log_evidence - competing)


def has_te_structure_origin_certificate(row: EvidenceLedgerRow) -> bool:
    """A TE origin argued from structure plus statistics, with one veto.

    THE VETO IS THE INTERESTING PART. A row with strong reference conflict
    (> 0.50) AND no direct observation of inserted sequence AND weak statistics
    is refused outright, whatever its structure evidence says. That combination
    describes a locus where the reference allele is well supported and nothing
    ever actually saw an insert -- and a structure decode run on an insert that
    may not exist is describing a hypothesis, not data.
    """
    if not has_conformal_feature_certificate(row):
        return False
    direct_insert_observation = (row.raw_cigar_insert_reads > 0
                                 or row.max_raw_cigar_insert_len > 0
                                 or row.event_consensus_len > 0)
    statistical_local_fdr = ledger_row_statistical_te_local_fdr(row)
    if (row.mechanistic_ref_conflict_signal > 0.50 and not direct_insert_observation
            and statistical_local_fdr > DEFAULT_FINAL_CONFORMAL_FDR):
        return False
    structure_local_fdr = ledger_row_structure_te_local_fdr(row)
    return combine_independent_local_fdr(statistical_local_fdr,
                                         structure_local_fdr) <= DEFAULT_FINAL_CONFORMAL_FDR


def has_pass_te_gate(row: EvidenceLedgerRow) -> bool:
    """A calibrated route PLUS any corroborating QC.

    Two-part by design: the first half asks whether a CALIBRATED argument exists
    (the robust lFDR or the structure-origin certificate), the second whether
    any local verdict agrees. Neither alone is enough -- a calibrated number
    with every local verdict against it usually means the calibration is
    reading a different locus than the verdicts are.
    """
    robust_pass = (row.robust_mechanistic_qc == "PASS_TE_LFDR"
                   and row.robust_mechanistic_worst_case_lfdr <= 0.10)
    structure_origin_pass = has_te_structure_origin_certificate(row)
    if not robust_pass and not structure_origin_pass:
        return False
    return (row.posterior_qc == "PASS_TE_POSTERIOR" or row.lfdr_qc == "PASS_TE_LFDR"
            or row.robust_mechanistic_qc == "PASS_TE_LFDR" or structure_origin_pass)


def has_calibrated_te_statistical_support(row: EvidenceLedgerRow) -> bool:
    """`has_pass_te_gate` with a wider second half.

    The extra alternatives (`te_posterior >= 0.95`, `worst_case_lfdr <= 0.10`)
    accept the NUMBERS where the gate wanted the QC tokens. Used where the
    question is "is there calibrated support" rather than "should this be
    reported", and the difference is deliberate.
    """
    robust_pass = (row.robust_mechanistic_qc == "PASS_TE_LFDR"
                   and row.robust_mechanistic_worst_case_lfdr <= 0.10)
    structure_origin_pass = has_te_structure_origin_certificate(row)
    if not robust_pass and not structure_origin_pass:
        return False
    return (row.posterior_qc == "PASS_TE_POSTERIOR" or row.lfdr_qc == "PASS_TE_LFDR"
            or robust_pass or structure_origin_pass or row.te_posterior >= 0.95
            or row.worst_case_lfdr <= 0.10)


def ledger_row_mechanistic_log_e_value(row: EvidenceLedgerRow) -> float:
    """The row's log e-value, or -inf when it must not enter selection.

    THREE REFUSALS, each returning -inf rather than a small number:

      * no computed certificate -- there is no e-value to report;
      * reference conflict above 0.50 -- the reference allele is well supported
        here, and an e-value computed as though it were not would be wrong in
        the one direction FDR control cannot absorb;
      * more than `max(2, alt/2)` reference-spanning reads -- the same judgement
        from the counts.

    The value itself is the WORSE of the two aggregates minus the ambiguity
    width. Taking the minimum is what makes it a bound rather than a best case.
    """
    if not has_conformal_feature_certificate(row):
        return -_INF
    if row.mechanistic_ref_conflict_signal > 0.50:
        return -_INF
    if row.ref_span_reads > max(2, row.alt_struct_reads // 2):
        return -_INF
    lower_log_bf = min(row.mechanistic_lower_log_bf_te_vs_artifact,
                       row.mechanistic_lower_log_bf_te_vs_non_te)
    return lower_log_bf - max(0.0, row.mechanistic_ambiguity_width)


def has_event_evalue_support(row: EvidenceLedgerRow) -> bool:
    log_e = ledger_row_mechanistic_log_e_value(row)
    return math.isfinite(log_e) and log_e > 0.0


def ledger_row_event_existence_log_evidence(row: EvidenceLedgerRow) -> float:
    """Evidence that an INSERTION exists here, TE or not.

    `alt <= ref` returns -inf: this is the dominant-allele route, and a
    minority-allele event is handled by `low_allele_fraction_event_log_bf`
    instead. Keeping the two apart is what lets each be a clean model rather
    than one model with an allele-fraction ladder in it.
    """
    if not has_conformal_feature_certificate(row):
        return -_INF
    alt = max(0, row.alt_struct_reads)
    ref = max(0, row.ref_span_reads)
    if alt <= 0 or alt <= ref:
        return -_INF
    if row.mechanistic_ref_conflict_signal > 0.65:
        return -_INF
    support_log_odds = math.log(alt + 1) - math.log(ref + 1)
    structural_log_bf = (row.mechanistic_lower_log_bf_te_vs_artifact
                         - max(0.0, row.mechanistic_ambiguity_width))
    return (support_log_odds + (0.50 * structural_log_bf)
            - (1.20 * _clamp01(row.mechanistic_ref_conflict_signal)))


def competing_nonte_long_insertion_log_evidence(alt_struct_reads: int,
                                                ref_span_reads: int,
                                                raw_cigar_insert_reads: int,
                                                max_raw_cigar_insert_len: int,
                                                event_consensus_len: int,
                                                non_te_posterior: float,
                                                artifact_posterior: float,
                                                ref_conflict_signal: float,
                                                ambiguity_width: float) -> float:
    """Evidence for a real NON-TE long insertion.

    Requires a direct observation of inserted sequence -- a raw CIGAR insert, a
    measured insert length, or an assembled consensus. Without one there is
    nothing to call non-TE; the locus is a junction with no insert, which is
    what an artifact looks like.

    The two insert terms are deliberately weak (0.05 and 0.03 per log base):
    length is WEAK evidence of reality, and letting it dominate would turn the
    gate into a length filter -- the failure the README identifies in the
    high-precision tier.
    """
    alt = max(0, alt_struct_reads)
    ref = max(0, ref_span_reads)
    if alt <= 0:
        return -_INF
    has_long_insert_observation = (raw_cigar_insert_reads > 0
                                   or max_raw_cigar_insert_len > 0
                                   or event_consensus_len > 0)
    if not has_long_insert_observation:
        return -_INF
    posterior_log_odds = (math.log(max(_clamp01(non_te_posterior), 1e-6))
                          - math.log(max(_clamp01(artifact_posterior), 1e-6)))
    support_log_odds = math.log(alt + 1) - math.log(ref + 1)
    raw_insert_log_bf = low_allele_fraction_event_log_bf(max(0, raw_cigar_insert_reads), ref)
    direct_insert_evidence = max(0.0, raw_insert_log_bf) if math.isfinite(raw_insert_log_bf) else 0.0
    assembled_insert_evidence = (0.05 * math.log1p(max(0, max_raw_cigar_insert_len))
                                 + 0.03 * math.log1p(max(0, event_consensus_len)))
    return (posterior_log_odds + support_log_odds + direct_insert_evidence
            + assembled_insert_evidence - max(0.0, ambiguity_width)
            - (0.35 * _clamp01(ref_conflict_signal)))


def has_competing_nonte_long_insertion_support(row: EvidenceLedgerRow) -> bool:
    """Is there a decisive NON-TE explanation competing with the TE one?

    Short-circuits when the TE origin already dominates: if the TE posterior
    beats both competitors AND the structure agrees, there is no competition to
    weigh and the answer is no regardless of the arithmetic.
    """
    if not has_conformal_feature_certificate(row):
        return False
    te_structure_local_fdr = ledger_row_structure_te_local_fdr(row)
    te_origin_dominates = (row.te_posterior > row.non_te_posterior
                           and row.te_posterior > row.artifact_posterior
                           and te_structure_local_fdr <= DEFAULT_FINAL_CONFORMAL_FDR)
    if te_origin_dominates:
        return False
    log_evidence = competing_nonte_long_insertion_log_evidence(
        row.alt_struct_reads, row.ref_span_reads, row.raw_cigar_insert_reads,
        row.max_raw_cigar_insert_len, row.event_consensus_len, row.non_te_posterior,
        row.artifact_posterior, row.mechanistic_ref_conflict_signal,
        row.mechanistic_ambiguity_width)
    return math.isfinite(log_evidence) and log_evidence > 2.0


# ---------------------------------------------------------------------------
# Low-allele-fraction certificates. Three variants, on purpose.
# ---------------------------------------------------------------------------
def _low_af_shape(alt: int, ref: int) -> bool:
    """At least 2 alt reads, some reference, and AF below 0.35.

    The AF ceiling is what makes this the MINORITY-allele route: above it the
    dominant-allele model applies and this one would be double-counting.
    """
    alt = max(0, alt)
    ref = max(0, ref)
    if alt < 2 or ref <= 0:
        return False
    return (alt / max(1, alt + ref)) < 0.35


def has_low_allele_fraction_event_certificate_row(row: EvidenceLedgerRow) -> bool:
    """A het or mosaic insertion, argued from the counts plus any sequence hint.

    The `>= 6.0` on the Bayes factor is a Kass-Raftery "very strong" scale
    point, not a tuned number, and the sequence-or-insert requirement stops the
    counts alone from carrying it: a 4-vs-30 split is consistent with a real
    mosaic AND with a systematic mapping error, and only the sequence evidence
    separates them.
    """
    if not has_conformal_feature_certificate(row):
        return False
    if not _low_af_shape(row.alt_struct_reads, row.ref_span_reads):
        return False
    sequence_or_insert_signal = (row.best_te_query_coverage >= 0.50
                                 or row.best_te_identity >= 0.60
                                 or row.cross_family_margin >= 0.015
                                 or row.final_qc == "TE_AMBIGUOUS"
                                 or is_final_event_qc_for_conformal(row.final_qc))
    if not sequence_or_insert_signal:
        return False
    log_bf = low_allele_fraction_event_log_bf(max(0, row.alt_struct_reads),
                                              max(0, row.ref_span_reads))
    return math.isfinite(log_bf) and log_bf >= 6.0


def has_self_stable_low_af_interval_certificate_row(row: EvidenceLedgerRow) -> bool:
    """A low-AF event whose own breakpoint INTERVAL is wider than the dedup
    distance, with a boundary block.

    The width requirement is the certificate: an event whose two breakpoints are
    50 bp or more apart cannot have been produced by two reads disagreeing about
    one position -- the interval is itself evidence that there is something
    between them.
    """
    if not has_low_allele_fraction_event_certificate_row(row):
        return False
    if row.bp_left < 0 or row.bp_right < 0:
        return False
    left = min(row.bp_left, row.bp_right)
    right = max(row.bp_left, row.bp_right)
    if left < 0 or right <= left:
        return False
    if not mechanistic_blocks_include(row.mechanistic_blocks, "boundary"):
        return False
    return (right - left) >= FINAL_CALL_DEDUP_DISTANCE_BP


def has_ref_unopposed_bilateral_partial_anchor_certificate(row: EvidenceLedgerRow) -> bool:
    """The narrowest certificate in the file, and every clause is a NECESSARY one.

    It describes exactly one situation: a locus where nothing was called
    (`NO_CALL_INCOMPLETE`), no reference read opposes it at all, no precise read
    supports it either, but two or more reads clip on BOTH sides and the
    consensus anchored from both -- and no TE sequence evidence exists at all.

    That is a real insertion whose sequence could not be classified: the clips
    agree from both directions, nothing contradicts them, and the only reason
    there is no call is that the insert never got named. The `best_te_identity
    == 0` clauses are what keep it narrow -- any sequence evidence at all means
    some other route should have handled it, and this one must not fire.
    """
    if not has_conformal_feature_certificate(row):
        return False
    if row.final_qc != "NO_CALL_INCOMPLETE":
        return False
    if row.ref_span_reads != 0 or row.mechanistic_ref_conflict_signal > 1e-9:
        return False
    if row.alt_split_reads > 0 or row.alt_indel_reads > 0:
        return False
    if row.alt_struct_reads < 2 or len(row.support_qnames) < 2:
        return False
    if not (row.alt_left_clip_reads > 0 and row.alt_right_clip_reads > 0):
        return False
    if not (row.left_anchor_input_reads > 0 and row.right_anchor_input_reads > 0):
        return False
    if row.partial_context_input_reads < 2 or row.input_event_reads < 2:
        return False
    if (row.best_te_identity > 0.0 or row.best_te_query_coverage > 0.0
            or row.cross_family_margin > 0.0):
        return False
    left = min(row.bp_left, row.bp_right)
    right = max(row.bp_left, row.bp_right)
    return left >= 0 and right >= left


def has_low_allele_fraction_event_certificate_call(call: FinalCall) -> bool:
    """The same shape, re-tested on a CALL after selection.

    Re-tested rather than carried, because finalization rewrites the counts
    (aggregation merges support sets) and a certificate computed before the
    rewrite would describe a locus that no longer exists.
    """
    if not starts_with_text(call.final_qc, "PASS_STRUCTURAL_INSERTION"):
        return False
    if not call.mechanistic_blocks or call.mechanistic_blocks == "NA":
        return False
    if not _low_af_shape(call.alt_struct_reads, call.ref_span_reads):
        return False
    sequence_or_insert_signal = (call.best_te_query_coverage >= 0.50
                                 or call.best_te_identity >= 0.60
                                 or call.cross_family_margin >= 0.015)
    if not sequence_or_insert_signal:
        return False
    log_bf = low_allele_fraction_event_log_bf(max(0, call.alt_struct_reads),
                                              max(0, call.ref_span_reads))
    return math.isfinite(log_bf) and log_bf >= 6.0


def has_aggregated_low_allele_fraction_event_certificate(call: FinalCall) -> bool:
    """As above, for a call built by AGGREGATING a community of rows.

    The extra requirements -- both community tokens in the QC, and both
    `community` and `read_overlap` in the blocks -- are how the call proves it
    really came from the aggregation path. Without them a single row could carry
    an aggregate's certificate.
    """
    if (not starts_with_text(call.final_qc, "PASS_STRUCTURAL_INSERTION")
            or "EVENT_COMMUNITY_STABLE" not in call.final_qc
            or "EVENT_COMMUNITY_AGGREGATED" not in call.final_qc):
        return False
    if not (mechanistic_blocks_include(call.mechanistic_blocks, "community")
            and mechanistic_blocks_include(call.mechanistic_blocks, "read_overlap")):
        return False
    if not _low_af_shape(call.alt_struct_reads, call.ref_span_reads):
        return False
    log_bf = low_allele_fraction_event_log_bf(max(0, call.alt_struct_reads),
                                              max(0, call.ref_span_reads))
    return math.isfinite(log_bf) and log_bf >= 6.0


def has_self_stable_low_af_interval_certificate_call(call: FinalCall) -> bool:
    if (not starts_with_text(call.final_qc, "PASS_STRUCTURAL_INSERTION")
            or "EVENT_INTERVAL_STABLE" not in call.final_qc):
        return False
    if not mechanistic_blocks_include(call.mechanistic_blocks, "boundary"):
        return False
    left = min(call.bp_left, call.bp_right)
    right = max(call.bp_left, call.bp_right)
    if left < 0 or right <= left or (right - left) < FINAL_CALL_DEDUP_DISTANCE_BP:
        return False
    if not _low_af_shape(call.alt_struct_reads, call.ref_span_reads):
        return False
    log_bf = low_allele_fraction_event_log_bf(max(0, call.alt_struct_reads),
                                              max(0, call.ref_span_reads))
    return math.isfinite(log_bf) and log_bf >= 6.0


# ---------------------------------------------------------------------------
# Support-set overlap: how one event is recognised in several calls.
# ---------------------------------------------------------------------------
def normalized_support_qnames(values: list[str]) -> list[str]:
    return sorted(set(values))


def sorted_support_intersection_size(lhs: list[str], rhs: list[str]) -> int:
    """Merge-walk over two SORTED lists. O(n+m) rather than O(n*m)."""
    i = j = intersect = 0
    while i < len(lhs) and j < len(rhs):
        if lhs[i] == rhs[j]:
            intersect += 1
            i += 1
            j += 1
        elif lhs[i] < rhs[j]:
            i += 1
        else:
            j += 1
    return intersect


def sorted_support_jaccard(lhs: list[str], rhs: list[str]) -> float:
    """Jaccard over support sets: |A n B| / |A u B|."""
    intersect = sorted_support_intersection_size(lhs, rhs)
    union = len(lhs) + len(rhs) - intersect
    return (intersect / union) if union > 0 else 0.0


def ledger_rows_share_support_edge(lhs: EvidenceLedgerRow,
                                   rhs: EvidenceLedgerRow) -> bool:
    """Do two rows share enough READS to be the same event?

    THE CENTRAL IDEA of the de-duplication half of this stage. Two calls 300 bp
    apart built from the same twelve reads ARE one event; two calls 30 bp apart
    built from disjoint reads may not be. Position cannot answer that and the
    read names can.

    TWO shared reads is enough on its own, at any set size. ONE shared read is
    enough only at Jaccard >= 0.50 -- i.e. when the sets are small and mostly
    the same, where a single shared read is most of the evidence there is. The
    asymmetry is the point: a single shared read between two 40-read sets is
    coincidence at a repetitive locus, and between two 2-read sets it is not.

    NOTE that the ledger version does NOT normalise its inputs: ledger rows
    carry `support_qnames` already sorted and deduplicated, and re-sorting on
    every pairwise comparison would be quadratic work for nothing.
    """
    if not lhs.support_qnames or not rhs.support_qnames:
        return False
    overlap = sorted_support_intersection_size(lhs.support_qnames, rhs.support_qnames)
    if overlap == 0:
        return False
    if overlap >= 2:
        return True
    return sorted_support_jaccard(lhs.support_qnames, rhs.support_qnames) >= 0.50


def final_calls_share_support_edge(lhs: FinalCall, rhs: FinalCall) -> bool:
    """The same rule for calls, which DO get normalised first -- a call's
    support set is rebuilt by aggregation and may not be sorted."""
    if not lhs.support_qnames or not rhs.support_qnames:
        return False
    left = normalized_support_qnames(lhs.support_qnames)
    right = normalized_support_qnames(rhs.support_qnames)
    overlap = sorted_support_intersection_size(left, right)
    if overlap == 0:
        return False
    if overlap >= 2:
        return True
    return sorted_support_jaccard(left, right) >= 0.50


def final_calls_share_identical_support(lhs: FinalCall, rhs: FinalCall) -> bool:
    """The same reads exactly. Two calls from an identical read set are two
    descriptions of one event, whatever their positions say."""
    if not lhs.support_qnames or not rhs.support_qnames:
        return False
    left = normalized_support_qnames(lhs.support_qnames)
    right = normalized_support_qnames(rhs.support_qnames)
    overlap = sorted_support_intersection_size(left, right)
    return overlap == len(left) and overlap == len(right)


def final_calls_share_support_containment_edge(lhs: FinalCall, rhs: FinalCall) -> bool:
    """Half or more of the SMALLER support set is shared.

    Containment rather than Jaccard because the two are asymmetric in a way that
    matters here: a fragment of an event supported by 3 of the event's 30 reads
    has a Jaccard of 0.1 and a containment of 1.0, and it is the containment
    that says it is a fragment rather than a separate locus.
    """
    if not lhs.support_qnames or not rhs.support_qnames:
        return False
    left = normalized_support_qnames(lhs.support_qnames)
    right = normalized_support_qnames(rhs.support_qnames)
    overlap = sorted_support_intersection_size(left, right)
    smaller = min(len(left), len(right))
    return smaller > 0 and (overlap * 2) >= smaller


# ---------------------------------------------------------------------------
# Spatial relations between rows and between calls.
# ---------------------------------------------------------------------------
def ledger_row_breakpoint_event_start(row: EvidenceLedgerRow) -> int:
    if row.bp_left >= 0 and row.bp_right >= 0:
        return min(row.bp_left, row.bp_right)
    return row.pos if row.pos >= 0 else -1


def ledger_row_breakpoint_event_end(row: EvidenceLedgerRow) -> int:
    if row.bp_left >= 0 and row.bp_right >= 0:
        return max(row.bp_left, row.bp_right)
    return row.pos if row.pos >= 0 else -1


def ledger_row_breakpoint_event_center(row: EvidenceLedgerRow) -> int:
    start = ledger_row_breakpoint_event_start(row)
    end = ledger_row_breakpoint_event_end(row)
    if start >= 0 and end >= 0:
        return start + ((end - start) // 2)
    return row.pos


def ledger_row_context_start(row: EvidenceLedgerRow) -> int:
    """The COVERAGE interval when known, otherwise the breakpoint interval.

    Wider than the breakpoints: the context is where the evidence came from, not
    where the event is, and two rows built from overlapping read sets have
    overlapping contexts even when their breakpoints differ.
    """
    if row.coverage_left >= 0 and row.coverage_right >= 0:
        return min(row.coverage_left, row.coverage_right)
    return ledger_row_breakpoint_event_start(row)


def ledger_row_context_end(row: EvidenceLedgerRow) -> int:
    if row.coverage_left >= 0 and row.coverage_right >= 0:
        return max(row.coverage_left, row.coverage_right)
    return ledger_row_breakpoint_event_end(row)


def _intervals_within(a_start: int, a_end: int, b_start: int, b_end: int,
                      distance: int) -> bool:
    if a_start < 0 or a_end < 0 or b_start < 0 or b_end < 0:
        return False
    if a_end < b_start:
        return (b_start - a_end) <= distance
    if b_end < a_start:
        return (a_start - b_end) <= distance
    return True


def ledger_rows_are_spatial_event_neighbors(lhs: EvidenceLedgerRow,
                                            rhs: EvidenceLedgerRow) -> bool:
    if lhs.tid != rhs.tid or lhs.chrom != rhs.chrom:
        return False
    return _intervals_within(ledger_row_breakpoint_event_start(lhs),
                             ledger_row_breakpoint_event_end(lhs),
                             ledger_row_breakpoint_event_start(rhs),
                             ledger_row_breakpoint_event_end(rhs),
                             FINAL_CALL_DEDUP_DISTANCE_BP)


def final_call_event_envelope(call: FinalCall) -> tuple[int, int]:
    """The interval a call could plausibly occupy.

    Widened by HALF THE CONSENSUS LENGTH either side of the position, because a
    call whose breakpoints came from a 3 kb consensus is uncertain by about that
    much -- and two such calls 1 kb apart are very likely the same event. The
    widening is what makes the envelope a statement about uncertainty rather
    than about position.
    """
    left = min(call.bp_left, call.bp_right)
    right = max(call.bp_left, call.bp_right)
    if left < 0 or right < 0:
        left = call.pos
        right = call.pos
    if call.event_consensus_len > 0 and call.pos >= 0:
        uncertainty = call.event_consensus_len // 2
        left = min(left, max(0, call.pos - uncertainty))
        right = max(right, call.pos + uncertainty)
    if left < 0 or right < 0:
        return -1, -1
    if left > right:
        left, right = right, left
    return left, right


def final_call_event_envelopes_overlap(lhs: FinalCall, rhs: FinalCall) -> bool:
    if lhs.tid != rhs.tid or lhs.chrom != rhs.chrom:
        return False
    a_left, a_right = final_call_event_envelope(lhs)
    b_left, b_right = final_call_event_envelope(rhs)
    if a_left < 0 or a_right < 0 or b_left < 0 or b_right < 0:
        return False
    return a_right >= b_left and b_right >= a_left


def final_calls_have_overlapping_event_context(lhs: FinalCall, rhs: FinalCall) -> bool:
    if lhs.tid != rhs.tid or lhs.chrom != rhs.chrom:
        return False
    if (lhs.window_start < 0 or lhs.window_end < 0
            or rhs.window_start < 0 or rhs.window_end < 0):
        return False
    a_left, a_right = min(lhs.window_start, lhs.window_end), max(lhs.window_start,
                                                                 lhs.window_end)
    b_left, b_right = min(rhs.window_start, rhs.window_end), max(rhs.window_start,
                                                                 rhs.window_end)
    return a_right >= b_left and b_right >= a_left


def final_calls_have_overlapping_breakpoint_intervals(lhs: FinalCall,
                                                      rhs: FinalCall) -> bool:
    if lhs.tid != rhs.tid or lhs.chrom != rhs.chrom:
        return False
    a_left, a_right = min(lhs.bp_left, lhs.bp_right), max(lhs.bp_left, lhs.bp_right)
    b_left, b_right = min(rhs.bp_left, rhs.bp_right), max(rhs.bp_left, rhs.bp_right)
    if a_left < 0 or a_right < 0 or b_left < 0 or b_right < 0:
        return False
    return a_right >= b_left and b_right >= a_left


# ---------------------------------------------------------------------------
# Row classification: promotable, null control, calibration row.
# ---------------------------------------------------------------------------
def ledger_row_is_structural_neighbor_evidence(row: EvidenceLedgerRow) -> bool:
    """Could this row corroborate a neighbour's low-AF event?

    Permissive on reference conflict (<= 0.95) because the row is being used as
    CORROBORATION, not as a call: a neighbouring row with heavy reference
    support still tells you reads were assembled into an event there.
    """
    if not has_conformal_feature_certificate(row):
        return False
    if row.alt_struct_reads <= 0:
        return False
    sequence_or_insert_signal = (row.best_te_query_coverage >= 0.50
                                 or row.best_te_identity >= 0.60
                                 or row.cross_family_margin >= 0.015
                                 or row.final_qc == "TE_AMBIGUOUS"
                                 or is_final_event_qc_for_conformal(row.final_qc))
    if not sequence_or_insert_signal:
        return False
    return row.mechanistic_ref_conflict_signal <= 0.95


def ledger_row_is_aggregatable_event_community_evidence(row: EvidenceLedgerRow) -> bool:
    """May this row be merged into an aggregated community call?

    Explicitly EXCLUDES rows that are already TE calls: aggregation exists to
    build one event out of several inconclusive rows, and folding a decided TE
    call into it would both double-count the call and blur its breakpoints.
    """
    if not has_conformal_feature_certificate(row):
        return False
    if row.pos < 0 and row.bp_left < 0 and row.bp_right < 0:
        return False
    if row.alt_struct_reads <= 0 or not row.support_qnames:
        return False
    if is_final_te_qc_for_conformal(row.final_qc):
        return False
    interpretable_event = (mechanistic_blocks_include(row.mechanistic_blocks, "event")
                           or row.final_qc == "TE_AMBIGUOUS"
                           or row.final_qc == "REFERENCE_OR_ARTIFACT")
    if not interpretable_event:
        return False
    return row.final_qc in ("REFERENCE_OR_ARTIFACT", "TE_AMBIGUOUS",
                            "PASS_NONTE_INSERTION")


def event_cluster_has_stable_low_af_support(focal_index: int,
                                            rows: list[EvidenceLedgerRow]) -> bool:
    """Does a NEIGHBOURING row, sharing reads, corroborate this low-AF event?

    Both conditions are needed and they are different: spatial neighbourhood
    says the rows are about the same place, the shared support edge says they
    are about the same READS. Either alone is satisfied routinely at a
    repetitive locus.
    """
    focal = rows[focal_index]
    if not has_low_allele_fraction_event_certificate_row(focal):
        return False
    for idx, neighbor in enumerate(rows):
        if idx == focal_index:
            continue
        if not ledger_rows_are_spatial_event_neighbors(focal, neighbor):
            continue
        if not ledger_row_is_structural_neighbor_evidence(neighbor):
            continue
        if ledger_rows_share_support_edge(focal, neighbor):
            return True
    return False


def has_event_existence_support(row: EvidenceLedgerRow) -> bool:
    log_evidence = ledger_row_event_existence_log_evidence(row)
    return math.isfinite(log_evidence) and log_evidence > 2.0


def has_balanced_heterozygous_event_support(row: EvidenceLedgerRow) -> bool:
    """A deep, well-balanced, alt-dominant locus with some sequence signal.

    `alt >= 16` is the striking requirement, and it is what makes this
    certificate about BALANCE rather than about depth: the balance statistic
    only means anything once there are enough reads for a 50/50 split to be
    distinguishable from chance, and 16 is where `log1p(min(alt, ref))` can
    reach the 1.75 threshold at all.
    """
    if not has_conformal_feature_certificate(row):
        return False
    alt = max(0, row.alt_struct_reads)
    ref = max(0, row.ref_span_reads)
    if alt < 16 or alt <= ref:
        return False
    if (heterozygous_balance_log_evidence(alt, ref) < 1.75
            or row.mechanistic_ref_conflict_signal > 0.65):
        return False
    event_interpretable = (row.final_qc in ("TE_AMBIGUOUS", "PASS_NONTE_INSERTION")
                           or is_final_event_qc_for_conformal(row.final_qc))
    if not event_interpretable:
        return False
    return (row.best_te_query_coverage >= 0.80 or row.best_te_identity >= 0.54
            or row.cross_family_margin >= 0.015
            or row.mechanistic_lower_log_bf_te_vs_artifact > -1.0)


def ledger_row_is_final_te(row: EvidenceLedgerRow) -> bool:
    return is_final_te_qc_for_conformal(row.final_qc)


def ledger_row_is_promotable_event_evidence(row: EvidenceLedgerRow) -> bool:
    """Could this row become a call by any route?

    THE STRUCTURE IS TWO GATES, and it is worth reading as such.

    The first asks whether ANY certificate applies -- low reference conflict, or
    one of the four special certificates, or calibrated TE support. The second
    then requires either (a statistical TE signal AND some sequence evidence) or
    a structural event signal on its own.

    The `and` in the first branch is the important one: a statistical signal
    with NO sequence evidence at all is a locus where the counts look like an
    insertion and nothing was ever observed to be inserted. The structural
    branch needs no sequence evidence because its certificates already require
    a direct observation of inserted bases.

    This predicate also defines what a null control is NOT -- see
    `is_sample_local_null_control`.
    """
    if not has_conformal_feature_certificate(row):
        return False
    if row.pos < 0 and row.bp_left < 0 and row.bp_right < 0:
        return False
    if row.alt_struct_reads <= 0:
        return False

    low_reference_conflict = (row.mechanistic_ref_conflict_signal <= 0.50
                              and row.ref_span_reads <= max(2, row.alt_struct_reads // 2))
    balanced_heterozygous_event = has_balanced_heterozygous_event_support(row)
    low_allele_fraction_event = has_low_allele_fraction_event_certificate_row(row)
    bilateral_partial_anchor_event = has_ref_unopposed_bilateral_partial_anchor_certificate(row)
    competing_nonte_event = has_competing_nonte_long_insertion_support(row)
    if not (low_reference_conflict or balanced_heterozygous_event
            or low_allele_fraction_event or bilateral_partial_anchor_event
            or competing_nonte_event or has_calibrated_te_statistical_support(row)):
        return False

    statistical_te_signal = (is_final_event_qc_for_conformal(row.final_qc)
                             or has_pass_te_gate(row) or row.te_posterior >= 0.50
                             or row.lfdr <= 0.50 or row.worst_case_lfdr <= 0.50
                             or row.robust_mechanistic_worst_case_lfdr <= 0.50
                             or has_event_evalue_support(row))
    structural_event_signal = (has_event_existence_support(row)
                               or balanced_heterozygous_event
                               or low_allele_fraction_event
                               or bilateral_partial_anchor_event
                               or competing_nonte_event)
    sequence_signal = (row.best_te_identity > 0.0 or row.best_te_query_coverage > 0.0
                       or row.cross_family_margin > 0.0 or row.family == "UNKNOWN"
                       or row.subfamily == "UNKNOWN")
    return (statistical_te_signal and sequence_signal) or structural_event_signal


def ledger_row_requires_stable_event_community(row: EvidenceLedgerRow) -> bool:
    """A low-AF row with NO other certificate needs a corroborating neighbour.

    Five negations: the row has the low-AF certificate and nothing else. A low
    allele fraction alone is the weakest evidence in the file -- it is what a
    real mosaic AND a systematic mapping error both look like -- so it is the
    one case required to be corroborated by another row sharing its reads.
    """
    return (has_low_allele_fraction_event_certificate_row(row)
            and not has_self_stable_low_af_interval_certificate_row(row)
            and not has_balanced_heterozygous_event_support(row)
            and not has_event_existence_support(row)
            and not has_ref_unopposed_bilateral_partial_anchor_certificate(row)
            and not has_calibrated_te_statistical_support(row))


def is_sample_local_null_control(row: EvidenceLedgerRow) -> bool:
    """A row the conformal selector may use as a null.

    Defined by EXCLUSION: it has a computed certificate, it is not promotable,
    and it is not already a TE call. Excluding promotable rows is what makes the
    resulting p-value conformal -- the null set must be rows the procedure did
    not select.
    """
    if not has_conformal_feature_certificate(row):
        return False
    if ledger_row_is_promotable_event_evidence(row):
        return False
    return not is_final_te_qc_for_conformal(row.final_qc)


def is_dependency_calibration_row(row: EvidenceLedgerRow) -> bool:
    """Rows used to measure the dependency bound. EVERY row with a certificate.

    DELIBERATELY NOT `is_sample_local_null_control`, and the two must not be
    merged back together. The conformal selector needs rows that were NOT
    selected -- that is what makes its p-value conformal. The dependency bound
    needs an estimate of `E_null[exp(Lambda)]`, and estimating that on a sample
    SELECTED BY Lambda is circular: promotability includes
    `has_event_evalue_support`, i.e. `log_e > 0` on the very aggregate being
    calibrated, and sigma is a MEAN dominated by its right tail -- so excluding
    rows for having a large aggregate removes exactly the mass sigma is made of.

    Measured on a synthetic ledger (4000 null rows, 10% repeat-region rows with
    TE-like sequence and no reference counterevidence), excluding the
    high-aggregate rows took sigma from 119.6 to its floor of 1.0: 4.8 nats of
    penalty never charged, and the whole measured-sigma apparatus inert.

    The two available biases point in opposite directions and only one is
    acceptable:

      * INCLUDING true positives inflates the empirical mean, so the bound
        bounds something at least as large as the null mean -- and therefore
        still bounds the null mean, provided the aggregate is not
        anti-informative, which is its entire purpose. Conservative: costs
        recall.
      * EXCLUDING rows by the aggregate truncates the right tail and deflates
        the estimate. Anti-conservative: breaks the guarantee.

    So nothing is excluded, and the residual contamination is made auditable
    through `dependency_penalty_null_count`.
    """
    return has_conformal_feature_certificate(row)


def ledger_rows_share_event_cluster(lhs: EvidenceLedgerRow,
                                    rhs: EvidenceLedgerRow) -> bool:
    return ledger_rows_are_spatial_event_neighbors(lhs, rhs)


def ledger_rows_have_overlapping_breakpoint_intervals(lhs: EvidenceLedgerRow,
                                                      rhs: EvidenceLedgerRow) -> bool:
    if lhs.tid != rhs.tid or lhs.chrom != rhs.chrom:
        return False
    a_start, a_end = ledger_row_breakpoint_event_start(lhs), ledger_row_breakpoint_event_end(lhs)
    b_start, b_end = ledger_row_breakpoint_event_start(rhs), ledger_row_breakpoint_event_end(rhs)
    if a_start < 0 or a_end < 0 or b_start < 0 or b_end < 0:
        return False
    return a_end >= b_start and b_end >= a_start


# ---------------------------------------------------------------------------
# The conformal feature vector.
# ---------------------------------------------------------------------------
def conformal_context_bucket(left_flank_align_len: int,
                             right_flank_align_len: int) -> int:
    """Two buckets: one-sided (1) and everything else (0).

    A one-sided event -- exactly one flank anchored -- is structurally
    artifact-prone, so it is calibrated against one-sided NULLS rather than
    against the pooled set. Flank lengths are the only context signal available
    on both the ledger rows and the final calls, which is why the bucket is
    coarse: a finer one would not be computable on both sides of the seam.
    """
    has_left = left_flank_align_len > 0
    has_right = right_flank_align_len > 0
    return 1 if (has_left != has_right) else 0


def _conformal_feature(chrom: str, pos: int, bp_left: int, bp_right: int, index: int,
                       alt_struct_reads: int, best_te_identity: float,
                       best_te_query_coverage: float, cross_family_margin: float,
                       ref_span_reads: int, left_flank_align_len: int,
                       right_flank_align_len: int):
    from placer_py.conformal import ConformalFeatureVector

    return ConformalFeatureVector(
        id=f"{chrom}:{pos}:{bp_left}:{bp_right}:{index}",
        pro_te=[float(max(0, alt_struct_reads)), max(0.0, best_te_identity),
                max(0.0, best_te_query_coverage), max(0.0, cross_family_margin)],
        ref_span_reads=float(max(0, ref_span_reads)),
        context=conformal_context_bucket(left_flank_align_len, right_flank_align_len))


def conformal_feature_from_call(call: FinalCall, index: int):
    """Four pro-TE coordinates and one against.

    The four are deliberately NOT combined into a score: the conformal route's
    whole evasion is that it never declares an exchange rate between them, and
    a candidate must be no worse than a null in EVERY coordinate to be dominated
    by it.
    """
    return _conformal_feature(call.chrom, call.pos, call.bp_left, call.bp_right, index,
                              call.alt_struct_reads, call.best_te_identity,
                              call.best_te_query_coverage, call.cross_family_margin,
                              call.ref_span_reads, call.left_flank_align_len,
                              call.right_flank_align_len)


def conformal_feature_from_row(row: EvidenceLedgerRow, index: int):
    return _conformal_feature(row.chrom, row.pos, row.bp_left, row.bp_right, index,
                              row.alt_struct_reads, row.best_te_identity,
                              row.best_te_query_coverage, row.cross_family_margin,
                              row.ref_span_reads, row.left_flank_align_len,
                              row.right_flank_align_len)


# ---------------------------------------------------------------------------
# The three e-value constructions.
# ---------------------------------------------------------------------------
#: Overflow guard only. The substantive cap is applied upstream, where the
#: dependency penalty is subtracted: `log_e <= log(m/q)` already. 40 nats is
#: reached only past m/q ~ 2.4e17.
E_VALUE_LOG_CAP = 40.0
#: The number of constructions the sum is divided by -- see
#: `apply_event_ebh_selection`.
E_VALUE_CONSTRUCTIONS = 3.0


def event_mechanistic_e_value(call: FinalCall) -> float:
    """The mechanistic aggregate, exponentiated -- or 0.

    Zero is a VALID e-value (expectation 0 <= 1), which is what lets every
    refusal here return 0 rather than having to exclude the call from the
    procedure. A call with no usable construction still counts towards m.
    """
    if not call.mechanistic_blocks or call.mechanistic_blocks == "NA":
        return 0.0
    if call.mechanistic_ref_conflict_signal > 0.50:
        return 0.0
    if call.ref_span_reads > max(2, call.alt_struct_reads // 2):
        return 0.0
    lower_log_bf = min(call.mechanistic_lower_log_bf_te_vs_artifact,
                       call.mechanistic_lower_log_bf_te_vs_non_te)
    log_e = lower_log_bf - max(0.0, call.mechanistic_ambiguity_width)
    if not math.isfinite(log_e) or log_e <= 0.0:
        return 0.0
    return math.exp(min(log_e, E_VALUE_LOG_CAP))


def low_allele_fraction_event_e_value(call: FinalCall) -> float:
    """The low-AF Bayes factor, but ONLY for a stabilised call.

    The stability token requirement is the guard: a low-AF certificate on its
    own is the weakest evidence in the file, and it becomes an e-value only
    after a neighbouring row sharing its reads has corroborated it.
    """
    has_certificate = (has_low_allele_fraction_event_certificate_call(call)
                       or has_aggregated_low_allele_fraction_event_certificate(call)
                       or has_self_stable_low_af_interval_certificate_call(call))
    stabilised = ("EVENT_COMMUNITY_STABLE" in call.final_qc
                  or "EVENT_INTERVAL_STABLE" in call.final_qc)
    if not has_certificate or not stabilised:
        return 0.0
    log_bf = low_allele_fraction_event_log_bf(max(0, call.alt_struct_reads),
                                              max(0, call.ref_span_reads))
    if not math.isfinite(log_bf) or log_bf < 6.0:
        return 0.0
    return math.exp(min(log_bf, E_VALUE_LOG_CAP))


def ref_unopposed_bilateral_partial_anchor_e_value(call: FinalCall) -> float:
    """The unopposed bilateral-clip construction.

    `ref != 0` refuses outright: this construction's whole argument is that
    NOTHING opposes the insertion, and one opposing read removes it.
    """
    if "EVENT_BILATERAL_PARTIAL_ANCHOR" not in call.final_qc:
        return 0.0
    if (not starts_with_text(call.final_qc, "PASS_STRUCTURAL_INSERTION")
            or not call.mechanistic_blocks or call.mechanistic_blocks == "NA"):
        return 0.0
    alt = max(0, call.alt_struct_reads)
    ref = max(0, call.ref_span_reads)
    if alt < 2 or ref != 0 or call.mechanistic_ref_conflict_signal > 1e-9:
        return 0.0
    log_bf = low_allele_fraction_event_log_bf(alt, ref)
    if not math.isfinite(log_bf) or log_bf <= 0.0:
        return 0.0
    return math.exp(min(log_bf, E_VALUE_LOG_CAP))


def apply_event_ebh_selection(calls: list[FinalCall], target_fdr: float) -> None:
    """e-BH as the PRIMARY selection route. Mutates the calls in place.

    WHY e-BH AND NOT BY. Conformal p-values have to be selected with
    Benjamini-Yekutieli, whose harmonic factor is what admitting arbitrary
    dependence among p-VALUES costs -- about 7.49 at m = 1000, turning a nominal
    q = 0.10 into an effective 0.0134. e-BH controls FDR <= q under arbitrary
    dependence among the e-VALUES with NO such factor (Wang & Ramdas 2022),
    because its proof uses only linearity of expectation. Switching the currency
    from tail probabilities to expectations removes the dependence penalty.

    THE THREE CONSTRUCTIONS ARE AVERAGED, NOT MAXIMISED. The maximum of e-values
    is not an e-value -- `E_null[max_k E_k]` can exceed 1. It does obey
    `max <= sum`, so `max/K` is valid, meaning the historical `max()` was
    anti-conservative by at most `ln 3 = 1.10` nats. The arithmetic mean fixes
    that with no assumption at all, since expectation is linear, and it
    DOMINATES `max/K` because it keeps the evidence from the other two
    constructions instead of discarding it (Vovk & Wang 2021).

    TWO SUBTLETIES THAT ARE EASY TO GET WRONG:

      * calls that already hold a conformal certificate are NOT skipped. Skipping
        them would make the candidate set data-dependent, which invalidates the
        e-BH guarantee -- m must be fixed before any e-value is looked at.
      * a call with no usable construction gets e = 0 and STAYS in the
        denominator. Dropping it would shrink m and inflate every threshold.

    This is sound only if the inputs really are e-values, which is what the
    measured dependency bound provides. The two changes are a pair; neither is
    correct alone.
    """
    if not calls:
        return

    evalues: list[tuple[float, int]] = []
    for i, call in enumerate(calls):
        candidates = (event_mechanistic_e_value(call),
                      low_allele_fraction_event_e_value(call),
                      ref_unopposed_bilateral_partial_anchor_e_value(call))
        total = sum(c for c in candidates if c > 0.0 and math.isfinite(c))
        evalue = total / E_VALUE_CONSTRUCTIONS
        evalues.append((evalue if (evalue > 0.0 and math.isfinite(evalue)) else 0.0, i))

    evalues.sort(key=lambda item: (-item[0], item[1]))
    q = min(1.0, max(0.0, target_fdr))
    m = float(len(evalues))
    selected_prefix = 0
    for rank in range(1, len(evalues) + 1):
        if q > 0.0 and evalues[rank - 1][0] >= (m / (q * rank)):
            selected_prefix = rank

    for rank in range(selected_prefix):
        evalue, index = evalues[rank]
        call = calls[index]
        call.ebh_e_value = evalue
        call.ebh_selected = True
        # Do not clobber a conformal certificate: either route is sufficient,
        # and keeping the conformal QC preserves its diagnostics.
        if not final_call_has_reportable_conformal_certificate(call):
            call.conformal_qc = "PASS_EVENT_EBH"
            call.conformal_by_threshold = evalue


def apply_event_bayesian_fdr_fallback(calls: list[FinalCall], target_fdr: float) -> None:
    """A Bayesian-FDR fallback for calls no other route selected.

    The rule is the classic one: sort by local FDR ascending and take the
    longest prefix whose MEAN local FDR is at or below q. The mean of local FDRs
    over a selected set is an estimate of the FDP of that set, so this is a
    direct Bayesian analogue -- and, unlike e-BH, its validity rests on the
    local FDRs being well calibrated, which is why it is a fallback and not the
    primary route.

    Calls already selected by conformal, the lFDR fallback or e-BH are skipped
    (they are already reported), and only CALIBRATED calls are eligible.
    """
    if not calls:
        return

    local_fdrs: list[tuple[float, int]] = []
    for i, call in enumerate(calls):
        if call.conformal_qc in ("PASS_CONFORMAL_FDR", "PASS_EVENT_LFDR_FALLBACK",
                                 "PASS_EVENT_EBH"):
            continue
        calibrated = (call.lfdr_qc == "PASS_TE_LFDR"
                      or call.posterior_qc == "PASS_TE_POSTERIOR"
                      or call.robust_mechanistic_qc == "PASS_TE_LFDR"
                      or call.worst_case_lfdr <= 0.10
                      or call.robust_mechanistic_worst_case_lfdr <= 0.10
                      or call.te_posterior >= 0.95)
        if not calibrated:
            continue
        local_fdr = 1.0
        if call.lfdr_qc == "PASS_TE_LFDR" or call.worst_case_lfdr <= 0.10:
            local_fdr = min(local_fdr, _clamp01(call.worst_case_lfdr))
        if (call.robust_mechanistic_qc == "PASS_TE_LFDR"
                or call.robust_mechanistic_worst_case_lfdr <= 0.10):
            local_fdr = min(local_fdr, _clamp01(call.robust_mechanistic_worst_case_lfdr))
        if call.posterior_qc == "PASS_TE_POSTERIOR":
            # A passing posterior contributes the TARGET, not the posterior
            # itself: the QC token says "this cleared the bar", and reading a
            # number off a threshold test would claim precision it does not have.
            local_fdr = min(local_fdr, 0.10)
        elif call.te_posterior >= 0.95:
            local_fdr = min(local_fdr, 1.0 - _clamp01(call.te_posterior))
        local_fdrs.append((local_fdr, i))

    if not local_fdrs:
        return

    local_fdrs.sort(key=lambda item: (item[0], item[1]))
    q = min(1.0, max(0.0, target_fdr))
    cumulative = 0.0
    selected_prefix = 0
    for rank in range(1, len(local_fdrs) + 1):
        cumulative += local_fdrs[rank - 1][0]
        if (cumulative / rank) <= q:
            selected_prefix = rank

    for rank in range(selected_prefix):
        local_fdr, index = local_fdrs[rank]
        calls[index].conformal_qc = "PASS_EVENT_BFDR"
        calls[index].conformal_by_threshold = local_fdr


def is_structural_insertion_call(call: FinalCall) -> bool:
    return starts_with_text(call.final_qc, "PASS_STRUCTURAL_INSERTION")


# ---------------------------------------------------------------------------
# Comparing and ordering calls.
# ---------------------------------------------------------------------------
def final_call_sort_less(call: FinalCall) -> tuple:
    """A total order for output. Position first, then window, then name."""
    return (call.tid, call.pos, call.window_start, call.window_end, call.chrom,
            call.te_name)


def same_call_locus(lhs: FinalCall, rhs: FinalCall) -> bool:
    """Are these two calls about the same place?

    Two REAL INTERVALS (width > 0) are the same locus when they OVERLAP -- with
    no slack at all, because an interval already expresses its own uncertainty.
    A point and an interval, or two points, get the 50 bp dedup slack instead,
    because a point is a claim about one base and two reads disagreeing by 30 bp
    are proposing the same junction.
    """
    if lhs.tid != rhs.tid:
        return False
    if lhs.chrom and rhs.chrom and lhs.chrom != rhs.chrom:
        return False

    def interval(call: FinalCall) -> tuple[int, int]:
        if call.bp_left >= 0 and call.bp_right >= 0:
            return min(call.bp_left, call.bp_right), max(call.bp_left, call.bp_right)
        if call.pos >= 0:
            return call.pos, call.pos
        return -1, -1

    a_left, a_right = interval(lhs)
    b_left, b_right = interval(rhs)
    if a_left >= 0 and a_right >= 0 and b_left >= 0 and b_right >= 0:
        if (a_right > a_left) and (b_right > b_left):
            return a_right >= b_left and b_right >= a_left
        if a_right < b_left:
            return (b_left - a_right) <= FINAL_CALL_DEDUP_DISTANCE_BP
        if b_right < a_left:
            return (a_left - b_right) <= FINAL_CALL_DEDUP_DISTANCE_BP
        return True
    return (lhs.pos >= 0 and rhs.pos >= 0
            and abs(lhs.pos - rhs.pos) <= FINAL_CALL_DEDUP_DISTANCE_BP)


def final_call_has_robust_te_lfdr(call: FinalCall) -> bool:
    return (call.robust_mechanistic_qc == "PASS_TE_LFDR"
            and call.robust_mechanistic_worst_case_lfdr <= 0.10)


def final_call_precision_rank(call: FinalCall) -> int:
    """How precisely this call is characterised. 4 is best, 0 is unranked.

    A CLOSED TE call outranks a structural insertion, which outranks an
    imprecise TE call, which outranks anything promoted from an event cluster.
    Note that promotion DEMOTES: a promoted call is the weakest kind, because it
    was assembled from rows that individually did not qualify.
    """
    if call.final_qc.startswith("PASS_TE_CLOSED"):
        return 4
    if (call.final_qc.startswith("PASS_STRUCTURAL_INSERTION")
            and "EVENT_CLUSTER_PROMOTED" not in call.final_qc):
        return 3
    if (call.final_qc.startswith("PASS_TE_IMPRECISE")
            and "EVENT_CLUSTER_PROMOTED" not in call.final_qc):
        return 2
    if "EVENT_CLUSTER_PROMOTED" in call.final_qc:
        return 1
    return 0


def prefer_new_call(cur: FinalCall, incumbent: FinalCall) -> bool:
    """Is `cur` a better representative of this locus than `incumbent`?

    THE ORDER OF KEYS IS THE POLICY, and it is calibration before evidence
    before description:

      1. a stabilised event community wins -- it is the only kind of call whose
         existence was corroborated by a second, read-sharing row;
      2. a robust TE lFDR wins, and between two of them the lower one;
      3. precision rank;
      4. then the ordinary evidence: support reads, GQ, family margin, identity;
      5. a NAMED element beats an unnamed one, but only as a tie-break -- the
         name never outranks the statistics;
      6. consensus length, coverage, and finally position for determinism.

    Step 5 is the one worth noticing: a well-supported unnamed call beats a
    weakly-supported named one, which is the opposite of what a caller optimised
    for a benchmark's family column would do.
    """
    cur_stable = "EVENT_COMMUNITY_STABLE" in cur.final_qc
    incumbent_stable = "EVENT_COMMUNITY_STABLE" in incumbent.final_qc
    if cur_stable != incumbent_stable:
        return cur_stable

    cur_robust = final_call_has_robust_te_lfdr(cur)
    incumbent_robust = final_call_has_robust_te_lfdr(incumbent)
    if cur_robust != incumbent_robust:
        return cur_robust
    if (cur_robust and incumbent_robust
            and cur.robust_mechanistic_worst_case_lfdr != incumbent.robust_mechanistic_worst_case_lfdr):
        return cur.robust_mechanistic_worst_case_lfdr < incumbent.robust_mechanistic_worst_case_lfdr

    cur_precision = final_call_precision_rank(cur)
    incumbent_precision = final_call_precision_rank(incumbent)
    if cur_precision != incumbent_precision:
        return cur_precision > incumbent_precision
    if cur.support_reads != incumbent.support_reads:
        return cur.support_reads > incumbent.support_reads
    if cur.gq != incumbent.gq:
        return cur.gq > incumbent.gq
    if cur.cross_family_margin != incumbent.cross_family_margin:
        return cur.cross_family_margin > incumbent.cross_family_margin
    if cur.best_te_identity != incumbent.best_te_identity:
        return cur.best_te_identity > incumbent.best_te_identity
    if cur.te_name != incumbent.te_name:
        if not incumbent.te_name or incumbent.te_name == "UNK":
            return True
        if not cur.te_name or cur.te_name == "UNK":
            return False
    if cur.event_consensus_len != incumbent.event_consensus_len:
        return cur.event_consensus_len > incumbent.event_consensus_len
    if cur.best_te_query_coverage != incumbent.best_te_query_coverage:
        return cur.best_te_query_coverage > incumbent.best_te_query_coverage
    return cur.pos < incumbent.pos


def final_call_has_explanation_posteriors(call: FinalCall) -> bool:
    """Did the explanation stage actually run? The three posteriors sum to 1
    when it did, and to 0 when it did not."""
    total = call.te_posterior + call.non_te_posterior + call.artifact_posterior
    return math.isfinite(total) and total > 0.50


def final_call_non_artifact_posterior(call: FinalCall) -> float:
    return _clamp01(call.te_posterior) + _clamp01(call.non_te_posterior)


def final_call_is_te_specific_event(call: FinalCall) -> bool:
    """A TE verdict WITH a concrete family.

    `family_committed` is checked alongside a legacy literal test because the
    commitment flag was added later; a library may legitimately contain a family
    literally named "Unknown", which is why the flag exists at all.
    """
    legacy_concrete_label = (call.family not in ("UNKNOWN", "NA") and bool(call.family))
    return (starts_with_text(call.final_qc, "PASS_TE")
            and (call.family_committed or legacy_concrete_label))


def final_call_statistical_te_local_fdr(call: FinalCall) -> float:
    local_fdr = 1.0
    if call.lfdr_qc == "PASS_TE_LFDR" or call.worst_case_lfdr <= DEFAULT_FINAL_CONFORMAL_FDR:
        local_fdr = min(local_fdr, _clamp01(call.worst_case_lfdr))
    if (call.posterior_qc == "PASS_TE_POSTERIOR"
            or call.te_posterior >= 1.0 - DEFAULT_FINAL_CONFORMAL_FDR):
        local_fdr = min(local_fdr, 1.0 - _clamp01(call.te_posterior))
    return local_fdr


def final_call_structure_te_local_fdr(call: FinalCall) -> float:
    """Read straight off the path CONFIDENCE, not from the evidence odds.

    Different from the ledger-row version, which compares the TE structure
    evidence against both competitors. The call carries only the confidence
    scalar by this point, and `1 - confidence` is the honest reading of it.
    """
    if not call.te_structure_path or call.te_structure_path == "NA":
        return 1.0
    return 1.0 - _clamp01(call.te_structure_path_confidence)


def final_call_has_te_origin_certificate(call: FinalCall) -> bool:
    if not final_call_is_te_specific_event(call):
        return False
    return combine_independent_local_fdr(
        final_call_statistical_te_local_fdr(call),
        final_call_structure_te_local_fdr(call)) <= DEFAULT_FINAL_CONFORMAL_FDR


def promoted_event_envelopes_should_compete(lhs: FinalCall, rhs: FinalCall) -> bool:
    """Two promoted calls compete only when they DISAGREE about being TE.

    Two promoted structural calls, or two promoted TE calls, are left alone --
    they may genuinely be two events. It is the mixed pair that is suspicious:
    the same reads described once as a TE insertion and once as a structural
    one, which is one event with two descriptions.
    """
    if not is_promoted_event_cluster_call(lhs) or not is_promoted_event_cluster_call(rhs):
        return False
    if final_call_is_te_specific_event(lhs) == final_call_is_te_specific_event(rhs):
        return False
    return final_call_event_envelopes_overlap(lhs, rhs)


def prefer_promoted_event_envelope_representative(candidate: FinalCall,
                                                  incumbent: FinalCall) -> bool:
    """Which of two competing promoted calls survives.

    A TE ORIGIN CERTIFICATE decides first -- it is the strongest statement
    available -- then the posteriors, and only then `prefer_new_call`. Note the
    posterior comparison is skipped entirely when only ONE call has posteriors:
    comparing a computed posterior against a default zero would make "we did not
    evaluate this" look like "we evaluated it and it is an artifact".
    """
    candidate_te_origin = final_call_has_te_origin_certificate(candidate)
    incumbent_te_origin = final_call_has_te_origin_certificate(incumbent)
    if candidate_te_origin != incumbent_te_origin:
        return candidate_te_origin

    candidate_has_posteriors = final_call_has_explanation_posteriors(candidate)
    incumbent_has_posteriors = final_call_has_explanation_posteriors(incumbent)
    if candidate_has_posteriors and incumbent_has_posteriors:
        candidate_non_artifact = final_call_non_artifact_posterior(candidate)
        incumbent_non_artifact = final_call_non_artifact_posterior(incumbent)
        if candidate_non_artifact != incumbent_non_artifact:
            return candidate_non_artifact > incumbent_non_artifact
        if candidate.artifact_posterior != incumbent.artifact_posterior:
            return candidate.artifact_posterior < incumbent.artifact_posterior
    elif candidate_has_posteriors != incumbent_has_posteriors:
        return candidate_has_posteriors

    candidate_te_specific = final_call_is_te_specific_event(candidate)
    incumbent_te_specific = final_call_is_te_specific_event(incumbent)
    if candidate_te_specific != incumbent_te_specific:
        return candidate_te_specific
    return prefer_new_call(candidate, incumbent)


def final_call_passes_min_raw_cigar_insert_length(call: FinalCall,
                                                  filter_config: FinalCallFilterConfig) -> bool:
    """The minimum-length filter, WITH an escape hatch for assembled events.

    A raw CIGAR insert longer than the threshold passes outright. Otherwise the
    call may still pass on an ASSEMBLED consensus longer than the threshold --
    but only with mechanistic blocks proving the sequence or structure was
    examined, and only when the structure or the posteriors positively favour an
    event over an artifact.

    Without the escape hatch this filter would discard every long insertion that
    no single read spanned, which is most of them for a long element.
    """
    threshold = filter_config.min_raw_cigar_insert_len_bp
    if threshold <= 0:
        return True
    if call.max_raw_cigar_insert_len > threshold:
        return True
    if call.alt_struct_reads <= 0 or call.event_consensus_len <= threshold:
        return False
    if not call.mechanistic_blocks or call.mechanistic_blocks == "NA":
        return False
    has_long_insertion_structure = ("sequence" in call.mechanistic_blocks
                                    or "structure" in call.mechanistic_blocks)
    if not has_long_insertion_structure:
        return False
    event_structure_log_bf = (max(call.te_structure_log_evidence,
                                  call.nonte_structure_log_evidence)
                              - call.artifact_structure_log_evidence)
    event_posterior = _clamp01(call.te_posterior) + _clamp01(call.non_te_posterior)
    return event_structure_log_bf > 0.0 or event_posterior > _clamp01(call.artifact_posterior)


def filter_short_final_insertions(calls: list[FinalCall],
                                  filter_config: FinalCallFilterConfig) -> list[FinalCall]:
    return [call for call in calls
            if final_call_passes_min_raw_cigar_insert_length(call, filter_config)]


# ---------------------------------------------------------------------------
# Collapsing one event reported as several calls.
# ---------------------------------------------------------------------------
def conformal_feature_dominates(lhs, rhs) -> bool:
    """Pareto dominance: no worse in every pro-TE coordinate AND fewer ref reads.

    The SAME dominance relation the conformal selector uses, applied here for a
    different purpose -- there it avoids assuming the features' joint
    distribution, here it avoids declaring an exchange rate between them when
    choosing a representative. Both are the same evasion.
    """
    if not lhs.pro_te or len(lhs.pro_te) != len(rhs.pro_te):
        return False
    strictly_better = lhs.ref_span_reads < rhs.ref_span_reads
    for left, right in zip(lhs.pro_te, rhs.pro_te):
        if left < right:
            return False
        if left > right:
            strictly_better = True
    if lhs.ref_span_reads > rhs.ref_span_reads:
        return False
    return strictly_better


def remove_pareto_dominated_final_calls(calls: list[FinalCall]) -> list[FinalCall]:
    """Drop a call that another call at the SAME LOCUS beats in every coordinate.

    Only within a locus -- a dominated call elsewhere is a different event, not
    a worse description of this one.

    THE PROMOTION GUARD is the subtle clause: a promoted call never dominates a
    non-promoted one. A promoted call is assembled from rows that individually
    did not qualify, so it can carry inflated aggregate counts; letting it
    dominate would let the aggregation delete the evidence it was built from.
    """
    if len(calls) < 2:
        return list(calls)
    features = [conformal_feature_from_call(call, i) for i, call in enumerate(calls)]
    dominated = [False] * len(calls)
    for i in range(len(features)):
        for j in range(len(features)):
            if i == j:
                continue
            if not same_call_locus(calls[i], calls[j]):
                continue
            if is_promoted_event_cluster_call(calls[j]) and not is_promoted_event_cluster_call(calls[i]):
                continue
            if conformal_feature_dominates(features[j], features[i]):
                dominated[i] = True
                break
    return [call for call, drop in zip(calls, dominated) if not drop]


def resolve_overlapping_promoted_event_envelopes(calls: list[FinalCall]) -> list[FinalCall]:
    """Keep one representative per group of competing promoted calls.

    The scan breaks out of the inner loop on a TID change, which is safe only
    because the list is sorted by `final_call_sort_less` first -- the C++ relies
    on that and so does this.
    """
    if len(calls) < 2:
        return list(calls)
    removed = [False] * len(calls)
    for i in range(len(calls)):
        if removed[i] or not is_promoted_event_cluster_call(calls[i]):
            continue
        best = i
        for j in range(i + 1, len(calls)):
            if removed[j]:
                continue
            if calls[best].tid != calls[j].tid:
                break
            if not promoted_event_envelopes_should_compete(calls[best], calls[j]):
                continue
            if prefer_promoted_event_envelope_representative(calls[j], calls[best]):
                removed[best] = True
                best = j
            else:
                removed[j] = True
    return [call for call, drop in zip(calls, removed) if not drop]


def final_call_is_collapsible_event_fragment(call: FinalCall) -> bool:
    return is_final_event_qc_for_conformal(call.final_qc)


def final_call_is_structural_event_context_representative(call: FinalCall) -> bool:
    return (starts_with_text(call.final_qc, "PASS_STRUCTURAL_INSERTION")
            and (is_promoted_event_cluster_call(call)
                 or "EVENT_FRAGMENT_COLLAPSED" in call.final_qc
                 or "EVENT_COMMUNITY" in call.final_qc))


def final_call_is_te_event_context_representative(call: FinalCall) -> bool:
    return (starts_with_text(call.final_qc, "PASS_TE")
            and (final_call_has_te_origin_certificate(call)
                 or is_promoted_event_cluster_call(call)))


def final_calls_are_mixed_te_structural_event_context(lhs: FinalCall,
                                                      rhs: FinalCall) -> bool:
    """One structural representative and one TE representative, same context.

    The commonest way a single long insertion becomes two calls: the structural
    route reports the event, the TE route reports the element, and they are the
    same thing. Both need an assembled consensus for this to apply, because the
    shared CONTEXT is what identifies them as the same event.
    """
    if lhs.event_consensus_len <= 0 or rhs.event_consensus_len <= 0:
        return False
    a_structural = final_call_is_structural_event_context_representative(lhs)
    b_structural = final_call_is_structural_event_context_representative(rhs)
    a_te = final_call_is_te_event_context_representative(lhs)
    b_te = final_call_is_te_event_context_representative(rhs)
    if not ((a_structural and b_te) or (b_structural and a_te)):
        return False
    return final_calls_have_overlapping_event_context(lhs, rhs)


def final_calls_are_same_supported_event_fragment(lhs: FinalCall, rhs: FinalCall) -> bool:
    """Are these two calls fragments of ONE event?

    FOUR routes to yes, in descending order of how much they assume:

      1. two structural calls with assembled consensuses and overlapping
         CONTEXT -- the context is the window the event was assembled from, so
         two structural calls sharing one are the same assembly;
      2. a mixed TE/structural pair sharing a context;
      3. NON-overlapping breakpoints, but overlapping envelopes AND contexts,
         AND either both structural or one's support contained in the other's;
      4. overlapping breakpoints and a shared support edge.

    Route 3 is where the read names do the work that position cannot: the
    breakpoints disagree, so position says these are different events, and the
    containment says one is a fragment of the other.
    """
    if not (final_call_is_collapsible_event_fragment(lhs)
            and final_call_is_collapsible_event_fragment(rhs)):
        return False

    structural_owner_context_pair = (
        starts_with_text(lhs.final_qc, "PASS_STRUCTURAL_INSERTION")
        and starts_with_text(rhs.final_qc, "PASS_STRUCTURAL_INSERTION")
        and lhs.event_consensus_len > 0 and rhs.event_consensus_len > 0
        and final_calls_have_overlapping_event_context(lhs, rhs))
    if structural_owner_context_pair:
        return True
    if final_calls_are_mixed_te_structural_event_context(lhs, rhs):
        return True

    if not final_calls_have_overlapping_breakpoint_intervals(lhs, rhs):
        if (final_call_event_envelopes_overlap(lhs, rhs)
                and final_calls_have_overlapping_event_context(lhs, rhs)
                and ((starts_with_text(lhs.final_qc, "PASS_STRUCTURAL_INSERTION")
                      and starts_with_text(rhs.final_qc, "PASS_STRUCTURAL_INSERTION"))
                     or final_calls_share_support_containment_edge(lhs, rhs))):
            return True
        return final_calls_share_identical_support(lhs, rhs)
    return final_calls_share_support_edge(lhs, rhs)


def prefer_fragment_component_representative(candidate: FinalCall,
                                             incumbent: FinalCall) -> bool:
    """Which fragment represents the collapsed event.

    Support-set SIZE comes before `prefer_new_call` here, and only here: when
    collapsing fragments of one event, the fragment that saw the most reads is
    the best description of it, whatever its QC tokens say.
    """
    if final_calls_are_mixed_te_structural_event_context(candidate, incumbent):
        return prefer_promoted_event_envelope_representative(candidate, incumbent)
    if promoted_event_envelopes_should_compete(candidate, incumbent):
        return prefer_promoted_event_envelope_representative(candidate, incumbent)
    if len(candidate.support_qnames) != len(incumbent.support_qnames):
        return len(candidate.support_qnames) > len(incumbent.support_qnames)
    return prefer_new_call(candidate, incumbent)


def collapsed_fragment_component_call(component: list[int],
                                      calls: list[FinalCall]) -> FinalCall:
    """One call from several fragments: widest envelope, union of support.

    The counts take the MAXIMUM rather than the sum -- the fragments describe
    the SAME reads seen differently, so summing would double-count. The support
    set is the union and `support_reads` is `max(existing, |union|)`, which is
    the one place a count grows: distinct read NAMES cannot be double-counted.

    The merged call keeps the representative's fields and two QC markers, so a
    reader can see that the call was assembled and on what basis.
    """
    import copy

    best = component[0]
    merged_left = None
    merged_right = None
    for idx in component:
        call_left, call_right = final_call_event_envelope(calls[idx])
        if call_left >= 0:
            merged_left = call_left if merged_left is None else min(merged_left, call_left)
        if call_right >= 0:
            merged_right = call_right if merged_right is None else max(merged_right, call_right)
        if prefer_fragment_component_representative(calls[idx], calls[best]):
            best = idx

    merged = copy.deepcopy(calls[best])
    if merged_left is not None:
        merged.bp_left = merged_left
    if merged_right is not None and merged_right >= 0:
        merged.bp_right = merged_right
    if merged.bp_left > merged.bp_right:
        merged.bp_left, merged.bp_right = merged.bp_right, merged.bp_left

    support: list[str] = []
    for idx in component:
        support.extend(calls[idx].support_qnames)
        merged.raw_cigar_insert_reads = max(merged.raw_cigar_insert_reads,
                                            calls[idx].raw_cigar_insert_reads)
        merged.max_raw_cigar_insert_len = max(merged.max_raw_cigar_insert_len,
                                              calls[idx].max_raw_cigar_insert_len)
        merged.ref_span_reads = max(merged.ref_span_reads, calls[idx].ref_span_reads)
        merged.low_mapq_ref_span_reads = max(merged.low_mapq_ref_span_reads,
                                             calls[idx].low_mapq_ref_span_reads)
    merged.support_qnames = normalized_support_qnames(support)
    unique_support = len(merged.support_qnames)
    merged.support_reads = max(merged.support_reads, unique_support)
    merged.alt_struct_reads = max(merged.alt_struct_reads, unique_support)

    if "EVENT_FRAGMENT_COLLAPSED" not in merged.final_qc:
        merged.final_qc += "|EVENT_FRAGMENT_COLLAPSED"
    if not merged.mechanistic_blocks or merged.mechanistic_blocks == "NA":
        merged.mechanistic_blocks = "event;read_overlap"
    elif "read_overlap" not in merged.mechanistic_blocks:
        merged.mechanistic_blocks += ";read_overlap"
    return merged


def collapse_shared_support_final_call_fragments(calls: list[FinalCall]) -> list[FinalCall]:
    """Union-find over "same event" edges, then one call per component.

    A union-find rather than pairwise merging because the relation is
    TRANSITIVE-by-intent but not transitive-by-test: A shares reads with B, B
    with C, and A and C may share none -- yet all three are one event. Connected
    components capture that; pairwise merging in list order would not, and would
    give a different answer depending on the order.

    The inner loop breaks on a TID change, which requires the caller to have
    sorted first.
    """
    if len(calls) < 2:
        return list(calls)

    parent = list(range(len(calls)))

    def find_root(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            parent[value], value = root, parent[value]
        return root

    def unite(lhs: int, rhs: int) -> None:
        lhs_root, rhs_root = find_root(lhs), find_root(rhs)
        if lhs_root != rhs_root:
            parent[rhs_root] = lhs_root

    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            if calls[i].tid != calls[j].tid:
                break
            if final_calls_are_same_supported_event_fragment(calls[i], calls[j]):
                unite(i, j)

    components: dict[int, list[int]] = {}
    for i in range(len(calls)):
        components.setdefault(find_root(i), []).append(i)

    collapsed: list[FinalCall] = []
    for i in range(len(calls)):
        root = find_root(i)
        found = components.pop(root, None)
        if found is None:
            continue
        if len(found) == 1:
            collapsed.append(calls[i])
        else:
            collapsed.append(collapsed_fragment_component_call(found, calls))
    return collapsed


# ---------------------------------------------------------------------------
# Promotion: turning ledger rows into calls.
# ---------------------------------------------------------------------------
def ledger_rows_share_promotable_event_cluster(lhs: EvidenceLedgerRow,
                                               rhs: EvidenceLedgerRow) -> bool:
    """Spatial neighbours, unless one of them has a self-stable INTERVAL.

    The extra condition is the interesting one. A row whose own breakpoint
    interval is wide (a self-stable low-AF interval) covers a lot of ground, so
    the plain 50 bp spatial test would swallow unrelated neighbours into its
    cluster. When one is present, the pair must also genuinely overlap or share
    reads.
    """
    if not ledger_rows_share_event_cluster(lhs, rhs):
        return False
    if not (has_self_stable_low_af_interval_certificate_row(lhs)
            or has_self_stable_low_af_interval_certificate_row(rhs)):
        return True
    return (ledger_rows_have_overlapping_breakpoint_intervals(lhs, rhs)
            or ledger_rows_share_support_edge(lhs, rhs))


def ledger_row_promotion_score(row: EvidenceLedgerRow) -> float:
    """A hand-weighted score used ONLY to pick a cluster representative.

    Eleven constants, and they gate nothing: this decides which of several rows
    describing one event is reported, not whether it is reported. That is the
    right place for a hand-weighted score -- the choice between equivalent
    descriptions has no null hypothesis attached and therefore nothing to
    calibrate.

    The two penalties dominate deliberately: reference reads cost 1.5 each and
    reference conflict up to 10, so a row with the reference against it loses to
    one without even when its TE evidence is better.
    """
    score = float(max(0, row.alt_struct_reads))
    score += _clamp01(row.te_posterior) * 8.0
    score += (1.0 - _clamp01(row.worst_case_lfdr)) * 4.0
    score += (1.0 - _clamp01(row.robust_mechanistic_worst_case_lfdr)) * 6.0
    score += max(0.0, row.best_te_identity) * 3.0
    score += max(0.0, row.best_te_query_coverage) * 3.0
    score += max(0.0, row.cross_family_margin) * 2.0
    score -= float(max(0, row.ref_span_reads)) * 1.5
    score -= _clamp01(row.mechanistic_ref_conflict_signal) * 10.0
    if row.robust_mechanistic_qc == "PASS_TE_LFDR":
        score += 6.0
    if row.lfdr_qc == "PASS_TE_LFDR":
        score += 4.0
    if row.posterior_qc == "PASS_TE_POSTERIOR":
        score += 2.0
    return score


def prefer_ledger_row_for_promotion(candidate: EvidenceLedgerRow,
                                    incumbent: EvidenceLedgerRow) -> bool:
    candidate_score = ledger_row_promotion_score(candidate)
    incumbent_score = ledger_row_promotion_score(incumbent)
    if candidate_score != incumbent_score:
        return candidate_score > incumbent_score
    if candidate.alt_struct_reads != incumbent.alt_struct_reads:
        return candidate.alt_struct_reads > incumbent.alt_struct_reads
    if candidate.robust_mechanistic_worst_case_lfdr != incumbent.robust_mechanistic_worst_case_lfdr:
        return candidate.robust_mechanistic_worst_case_lfdr < incumbent.robust_mechanistic_worst_case_lfdr
    if candidate.ref_span_reads != incumbent.ref_span_reads:
        return candidate.ref_span_reads < incumbent.ref_span_reads
    return ledger_row_breakpoint_event_center(candidate) < ledger_row_breakpoint_event_center(incumbent)


def final_call_overlaps_ledger_event(call: FinalCall, row: EvidenceLedgerRow) -> bool:
    if call.tid != row.tid or call.chrom != row.chrom:
        return False
    call_left, call_right = min(call.bp_left, call.bp_right), max(call.bp_left, call.bp_right)
    row_left = ledger_row_breakpoint_event_start(row)
    row_right = ledger_row_breakpoint_event_end(row)
    if call_left < 0 or call_right < 0 or row_left < 0 or row_right < 0:
        center = ledger_row_breakpoint_event_center(row)
        return (call.pos >= 0 and center >= 0
                and abs(call.pos - center) <= FINAL_CALL_DEDUP_DISTANCE_BP)
    return _intervals_within(call_left, call_right, row_left, row_right,
                             FINAL_CALL_DEDUP_DISTANCE_BP)


def _annotate(call: FinalCall, token: str) -> None:
    if token not in call.final_qc:
        call.final_qc += f"|{token}"


def annotate_stable_event_community_certificate(call: FinalCall) -> None:
    _annotate(call, "EVENT_COMMUNITY_STABLE")


def annotate_stable_event_interval_certificate(call: FinalCall) -> None:
    _annotate(call, "EVENT_INTERVAL_STABLE")


def annotate_aggregated_event_community_certificate(call: FinalCall) -> None:
    _annotate(call, "EVENT_COMMUNITY_AGGREGATED")


def promoted_call_from_ledger_row(row: EvidenceLedgerRow) -> FinalCall:
    """Build a call from a ledger row the pipeline did not call itself.

    THE DEFAULT VERDICT IS `PASS_TE_IMPRECISE`, downgraded to
    `PASS_STRUCTURAL_INSERTION` with the family forced to UNKNOWN whenever the
    row's evidence is for an EVENT rather than for a TE. That downgrade is the
    honest one: promotion assembles evidence that an insertion exists, and
    asserting an element identity on top of it would be claiming something the
    promotion never argued.

    THE BILATERAL-ANCHOR WIDENING is the most unusual step here. When the
    certificate applies, the breakpoints are widened by the larger of the
    consensus length and the current interval -- IN BOTH DIRECTIONS. The
    certificate says "an insertion is here and its junctions could not be
    placed"; a narrow interval would claim a precision the evidence explicitly
    does not have, and the widened one makes the uncertainty visible to the
    de-duplication that follows.
    """

    call = FinalCall()
    bilateral_partial_anchor_event = has_ref_unopposed_bilateral_partial_anchor_certificate(row)
    call.chrom = row.chrom
    call.tid = row.tid
    call.bp_left = row.bp_left if row.bp_left >= 0 else ledger_row_breakpoint_event_center(row)
    call.bp_right = row.bp_right if row.bp_right >= 0 else call.bp_left
    if call.bp_left > call.bp_right:
        call.bp_left, call.bp_right = call.bp_right, call.bp_left
    call.pos = ledger_row_breakpoint_event_center(row)
    if call.pos < 0:
        call.pos = call.bp_left

    call.window_start = ledger_row_context_start(row)
    call.window_end = ledger_row_context_end(row)
    if (not ledger_row_is_final_te(row)
            and row.owner_context_left >= 0 and row.owner_context_right >= 0):
        call.window_start = min(row.owner_context_left, row.owner_context_right)
        call.window_end = max(row.owner_context_left, row.owner_context_right)
    if not ledger_row_is_final_te(row) and row.event_consensus_len > 0 and call.pos >= 0:
        uncertainty = row.event_consensus_len // 2
        call.window_start = min(call.window_start if call.window_start >= 0 else call.pos,
                                max(0, call.pos - uncertainty))
        call.window_end = max(call.window_end if call.window_end >= 0 else call.pos,
                              call.pos + uncertainty)

    call.family = row.family if row.family else "UNKNOWN"
    if call.family == "NA":
        call.family = "UNKNOWN"
    call.subfamily = row.subfamily if row.subfamily else "UNKNOWN"
    if call.subfamily == "NA":
        call.subfamily = call.family
    call.te_name = call.subfamily if call.subfamily != "NA" else call.family
    call.family_committed = (row.family_alignment_resolved
                             and not (call.family == "UNKNOWN" and call.subfamily == "UNKNOWN"))

    call.alt_struct_reads = row.alt_struct_reads
    call.support_reads = row.alt_struct_reads
    call.raw_cigar_insert_reads = row.raw_cigar_insert_reads
    call.max_raw_cigar_insert_len = row.max_raw_cigar_insert_len
    call.ref_span_reads = row.ref_span_reads
    call.support_qnames = normalized_support_qnames(row.support_qnames)
    call.best_te_identity = row.best_te_identity
    call.best_te_query_coverage = row.best_te_query_coverage
    call.cross_family_margin = row.cross_family_margin
    call.te_structure_path = row.te_structure_path
    call.te_structure_log_evidence = row.te_structure_log_evidence
    call.nonte_structure_log_evidence = row.nonte_structure_log_evidence
    call.artifact_structure_log_evidence = row.artifact_structure_log_evidence
    call.te_structure_path_confidence = row.te_structure_path_confidence
    call.polyA_posterior = row.polyA_posterior
    call.transduction_posterior = row.transduction_posterior
    call.te_core_coverage = row.te_core_coverage
    call.unexplained_high_complexity_bp = row.unexplained_high_complexity_bp
    call.event_consensus_len = row.event_consensus_len
    call.left_flank_align_len = row.left_flank_align_len
    call.right_flank_align_len = row.right_flank_align_len
    call.insert_seq = row.insert_seq
    call.final_qc = "PASS_TE_IMPRECISE|EVENT_CLUSTER_PROMOTED"

    if (not ledger_row_is_final_te(row)
            and (has_event_existence_support(row)
                 or has_balanced_heterozygous_event_support(row)
                 or has_low_allele_fraction_event_certificate_row(row)
                 or has_competing_nonte_long_insertion_support(row)
                 or bilateral_partial_anchor_event)):
        call.family = "UNKNOWN"
        call.subfamily = "UNKNOWN"
        call.te_name = "UNKNOWN"
        call.family_committed = False
        call.final_qc = "PASS_STRUCTURAL_INSERTION|EVENT_CLUSTER_PROMOTED"

    if bilateral_partial_anchor_event:
        left = min(row.bp_left, row.bp_right)
        right = max(row.bp_left, row.bp_right)
        uncertainty = max(0, max(row.event_consensus_len, right - left))
        call.bp_left = max(0, left - uncertainty)
        call.bp_right = right + uncertainty
        call.final_qc += "|EVENT_BILATERAL_PARTIAL_ANCHOR"

    call.te_qc = "EVENT_CLUSTER_LEDGER_EVIDENCE"
    call.best_explanation = "EVENT_CLUSTER"
    call.explanation_residual = "PROMOTED_FROM_LEDGER"
    call.explanation_path = "NA"
    call.te_posterior = row.te_posterior
    call.non_te_posterior = row.non_te_posterior
    call.artifact_posterior = row.artifact_posterior
    call.lfdr = row.lfdr
    call.worst_case_lfdr = row.worst_case_lfdr
    call.posterior_qc = row.posterior_qc
    call.lfdr_qc = row.lfdr_qc
    call.mechanistic_lower_log_bf_te_vs_artifact = row.mechanistic_lower_log_bf_te_vs_artifact
    call.mechanistic_lower_log_bf_te_vs_non_te = row.mechanistic_lower_log_bf_te_vs_non_te
    call.mechanistic_raw_log_bf_te_vs_artifact = row.mechanistic_raw_log_bf_te_vs_artifact
    call.mechanistic_raw_log_bf_te_vs_non_te = row.mechanistic_raw_log_bf_te_vs_non_te
    call.mechanistic_ref_conflict_signal = row.mechanistic_ref_conflict_signal
    call.mechanistic_ambiguity_width = row.mechanistic_ambiguity_width
    call.mechanistic_blocks = row.mechanistic_blocks
    call.robust_mechanistic_lfdr = row.robust_mechanistic_lfdr
    call.robust_mechanistic_worst_case_lfdr = row.robust_mechanistic_worst_case_lfdr
    call.robust_mechanistic_qc = row.robust_mechanistic_qc
    return call


def append_promoted_event_cluster_calls(ledger: list[EvidenceLedgerRow],
                                        calls: list[FinalCall]) -> None:
    """Cluster the promotable rows and promote ONE representative per cluster.

    WHY PROMOTION EXISTS AT ALL. The per-locus decision abstains whenever the
    risk gate is not met, and abstention is right for one locus -- but several
    abstaining rows describing one event, taken together, can be conclusive when
    none of them is alone. Promotion is the only place that argument is made,
    and it is deliberately conservative: one call per cluster, verdict downgraded
    to structural unless the row itself was a TE call, and a low-AF row with no
    other certificate refused unless a read-sharing neighbour corroborates it.
    """
    promotable = [i for i, row in enumerate(ledger)
                  if ledger_row_is_promotable_event_evidence(row)]
    if not promotable:
        return
    promotable.sort(key=lambda i: (ledger[i].tid,
                                   ledger_row_breakpoint_event_start(ledger[i]), i))

    cluster_start = 0
    while cluster_start < len(promotable):
        best = promotable[cluster_start]
        cluster_end = cluster_start + 1
        # NOTE the cluster membership test is against the cluster's FIRST row,
        # not the running best or the previous row: a chain of rows each near
        # the last would otherwise merge an arbitrarily long run into one.
        while (cluster_end < len(promotable)
               and ledger_rows_share_promotable_event_cluster(
                   ledger[promotable[cluster_start]], ledger[promotable[cluster_end]])):
            idx = promotable[cluster_end]
            if prefer_ledger_row_for_promotion(ledger[idx], ledger[best]):
                best = idx
            cluster_end += 1

        requires_stable_community = ledger_row_requires_stable_event_community(ledger[best])
        has_stable_community = event_cluster_has_stable_low_af_support(best, ledger)
        if not requires_stable_community or has_stable_community:
            call = promoted_call_from_ledger_row(ledger[best])
            if has_stable_community:
                annotate_stable_event_community_certificate(call)
            if has_self_stable_low_af_interval_certificate_row(ledger[best]):
                annotate_stable_event_interval_certificate(call)
            calls.append(call)
        cluster_start = cluster_end


# ---------------------------------------------------------------------------
# Event communities: one call from several rows that individually said nothing.
# ---------------------------------------------------------------------------
def merged_support_qnames_for_component(component: list[int],
                                        rows: list[EvidenceLedgerRow]) -> list[str]:
    support: list[str] = []
    for idx in component:
        support.extend(rows[idx].support_qnames)
    return normalized_support_qnames(support)


def median_ref_span_for_component(component: list[int],
                                  rows: list[EvidenceLedgerRow]) -> int:
    """MEDIAN, not max and not sum.

    The rows describe overlapping views of one locus, so their reference counts
    are repeated measurements of the same quantity. The median is the robust
    summary; the max would inflate the denominator with whichever row happened
    to have the widest window, and the sum would be nonsense.
    """
    ref_spans = sorted(max(0, rows[idx].ref_span_reads) for idx in component)
    if not ref_spans:
        return 0
    return ref_spans[len(ref_spans) // 2]


def aggregate_event_community_row(component: list[int],
                                  rows: list[EvidenceLedgerRow]) -> EvidenceLedgerRow:
    """Fuse a read-sharing community of rows into one synthetic row.

    THE AGGREGATION RULES DIFFER PER FIELD, and each choice is a statement:

      * `alt_struct_reads` becomes the size of the UNION of support sets -- read
        names cannot be double-counted, so this is the one count that legitimately
        grows;
      * `ref_span_reads` is the MEDIAN (see above);
      * `raw_cigar_insert_reads` is SUMMED -- these are separate observations of
        inserted bases, not repeated views of one;
      * the sequence fields take the BEST of the community: the rows are views of
        one insert, and the best-resolved view is the truest one;
      * `ref_conflict` takes the MINIMUM and `ambiguity_width` the MAXIMUM --
        both conservative in the same direction, favouring the community's best
        case on conflict and its worst case on ambiguity;
      * the breakpoints are widened by the dedup distance on each side, so the
        aggregate visibly covers the whole community.

    The two mechanistic log-BFs are RESET TO ZERO. The aggregate is not a locus
    the mechanistic blocks ever ran on, and carrying a member's aggregate would
    attach a certificate to evidence that was never evaluated. Its
    `mechanistic_blocks` string is set to a synthetic list instead, which is
    what the later certificates match `community` and `read_overlap` against.
    """
    import copy

    aggregate = copy.deepcopy(rows[component[0]])
    support = merged_support_qnames_for_component(component, rows)

    left: int | None = None
    right: int | None = None
    context_left: int | None = None
    context_right: int | None = None
    best_identity = best_coverage = best_margin = best_te_posterior = 0.0
    best_lfdr = best_worst_case_lfdr = 1.0
    min_ref_conflict = 1.0
    max_ambiguity = 0.0
    raw_cigar_insert_reads = 0
    max_raw_cigar_insert_len = 0
    max_left_flank_align_len = 0
    max_right_flank_align_len = 0
    representative_insert_seq = ""

    for idx in component:
        row = rows[idx]
        row_left = ledger_row_breakpoint_event_start(row)
        row_right = ledger_row_breakpoint_event_end(row)
        if row_left >= 0:
            left = row_left if left is None else min(left, row_left)
        if row_right >= 0:
            right = row_right if right is None else max(right, row_right)
        row_context_left = ledger_row_context_start(row)
        row_context_right = ledger_row_context_end(row)
        if row_context_left >= 0:
            context_left = row_context_left if context_left is None else min(context_left,
                                                                             row_context_left)
        if row_context_right >= 0:
            context_right = row_context_right if context_right is None else max(context_right,
                                                                                row_context_right)
        best_identity = max(best_identity, row.best_te_identity)
        best_coverage = max(best_coverage, row.best_te_query_coverage)
        best_margin = max(best_margin, row.cross_family_margin)
        best_te_posterior = max(best_te_posterior, row.te_posterior)
        best_lfdr = min(best_lfdr, row.lfdr)
        best_worst_case_lfdr = min(best_worst_case_lfdr, row.worst_case_lfdr)
        min_ref_conflict = min(min_ref_conflict, row.mechanistic_ref_conflict_signal)
        raw_cigar_insert_reads += max(0, row.raw_cigar_insert_reads)
        max_raw_cigar_insert_len = max(max_raw_cigar_insert_len, row.max_raw_cigar_insert_len)
        max_left_flank_align_len = max(max_left_flank_align_len, row.left_flank_align_len)
        max_right_flank_align_len = max(max_right_flank_align_len, row.right_flank_align_len)
        if len(row.insert_seq) > len(representative_insert_seq):
            representative_insert_seq = row.insert_seq
        max_ambiguity = max(max_ambiguity, row.mechanistic_ambiguity_width)

    if left is None:
        left = ledger_row_breakpoint_event_center(rows[component[0]])
    if right is None or right < 0:
        right = left
    if context_left is None:
        context_left = left
    if context_right is None or context_right < 0:
        context_right = right

    aggregate.pos = left + ((right - left) // 2)
    aggregate.bp_left = max(0, left - FINAL_CALL_DEDUP_DISTANCE_BP)
    aggregate.bp_right = right + FINAL_CALL_DEDUP_DISTANCE_BP
    aggregate.coverage_left = min(context_left, aggregate.bp_left)
    aggregate.coverage_right = max(context_right, aggregate.bp_right)
    aggregate.family = "UNKNOWN"
    aggregate.subfamily = "UNKNOWN"
    aggregate.final_qc = ("PASS_STRUCTURAL_INSERTION|EVENT_CLUSTER_PROMOTED|"
                          "EVENT_COMMUNITY_STABLE|EVENT_COMMUNITY_AGGREGATED")
    aggregate.posterior_qc = "TE_POSTERIOR_LOW"
    aggregate.lfdr_qc = "TE_LFDR_HIGH"
    aggregate.candidate_retention_reason = "EVENT_COMMUNITY_AGGREGATE"
    aggregate.alt_struct_reads = len(support)
    aggregate.raw_cigar_insert_reads = raw_cigar_insert_reads
    aggregate.max_raw_cigar_insert_len = max_raw_cigar_insert_len
    aggregate.left_flank_align_len = max_left_flank_align_len
    aggregate.right_flank_align_len = max_right_flank_align_len
    aggregate.insert_seq = representative_insert_seq
    aggregate.ref_span_reads = median_ref_span_for_component(component, rows)
    aggregate.support_qnames = support
    aggregate.best_te_identity = best_identity
    aggregate.best_te_query_coverage = best_coverage
    aggregate.cross_family_margin = best_margin
    aggregate.te_posterior = best_te_posterior
    aggregate.lfdr = best_lfdr
    aggregate.worst_case_lfdr = best_worst_case_lfdr
    aggregate.mechanistic_lower_log_bf_te_vs_artifact = 0.0
    aggregate.mechanistic_lower_log_bf_te_vs_non_te = 0.0
    aggregate.mechanistic_ref_conflict_signal = min_ref_conflict
    aggregate.mechanistic_ambiguity_width = max_ambiguity
    aggregate.mechanistic_blocks = "event;community;read_overlap;boundary;ref_conflict"
    aggregate.robust_mechanistic_lfdr = 1.0
    aggregate.robust_mechanistic_worst_case_lfdr = 1.0
    aggregate.robust_mechanistic_qc = "TE_LFDR_HIGH"
    return aggregate


def aggregate_event_community_has_low_af_certificate(row: EvidenceLedgerRow) -> bool:
    if ("EVENT_COMMUNITY_AGGREGATED" not in row.final_qc
            or "EVENT_COMMUNITY_STABLE" not in row.final_qc):
        return False
    return has_low_allele_fraction_event_certificate_row(row)


def aggregated_event_community_call_from_row(row: EvidenceLedgerRow) -> FinalCall:
    """The aggregate as a call, always UNKNOWN family.

    An aggregate of rows that individually could not name an element cannot name
    one either -- and saying so is what keeps the aggregation from manufacturing
    family calls out of weak evidence.
    """
    call = promoted_call_from_ledger_row(row)
    call.family = "UNKNOWN"
    call.subfamily = "UNKNOWN"
    call.te_name = "UNKNOWN"
    call.final_qc = ("PASS_STRUCTURAL_INSERTION|EVENT_CLUSTER_PROMOTED|"
                     "EVENT_COMMUNITY_STABLE|EVENT_COMMUNITY_AGGREGATED")
    call.te_qc = "EVENT_COMMUNITY_AGGREGATE"
    call.best_explanation = "EVENT_COMMUNITY"
    call.explanation_residual = "AGGREGATED_WEAK_LEDGER_EVIDENCE"
    call.explanation_path = "READ_OVERLAP_COMMUNITY"
    return call


def append_aggregated_event_community_calls(ledger: list[EvidenceLedgerRow],
                                            calls: list[FinalCall]) -> None:
    """Find read-sharing communities of INCONCLUSIVE rows and aggregate them.

    THE GUARD THAT MAKES THIS SAFE: a community containing any individually
    PROMOTABLE row is skipped entirely. Promotion already handles that community
    and would report the same event; aggregating it as well would double-report
    it, and the aggregate's synthetic counts would then compete with the real
    row's in de-duplication.

    So this fires only where every member said nothing on its own -- which is
    exactly the case the per-locus decision cannot reach.
    """
    candidates = [i for i, row in enumerate(ledger)
                  if ledger_row_is_aggregatable_event_community_evidence(row)]
    if len(candidates) < 2:
        return
    candidates.sort(key=lambda i: (ledger[i].tid, ledger[i].chrom,
                                   ledger_row_breakpoint_event_start(ledger[i]), i))

    parent = list(range(len(candidates)))

    def find_root(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            parent[value], value = root, parent[value]
        return root

    def unite(lhs: int, rhs: int) -> None:
        lhs_root, rhs_root = find_root(lhs), find_root(rhs)
        if lhs_root != rhs_root:
            parent[rhs_root] = lhs_root

    for i in range(len(candidates)):
        a = ledger[candidates[i]]
        a_end = ledger_row_breakpoint_event_end(a)
        for j in range(i + 1, len(candidates)):
            b = ledger[candidates[j]]
            if a.tid != b.tid or a.chrom != b.chrom:
                break
            b_start = ledger_row_breakpoint_event_start(b)
            # The sorted order lets the scan stop early rather than comparing
            # every pair -- the rows are ordered by start, so once one is beyond
            # the dedup distance every later one is too.
            if a_end >= 0 and b_start >= 0 and (b_start - a_end) > FINAL_CALL_DEDUP_DISTANCE_BP:
                break
            if (ledger_rows_are_spatial_event_neighbors(a, b)
                    and ledger_rows_share_support_edge(a, b)):
                unite(i, j)

    components: dict[int, list[int]] = {}
    for i in range(len(candidates)):
        components.setdefault(find_root(i), []).append(candidates[i])

    for component in components.values():
        if len(component) < 2:
            continue
        if any(ledger_row_is_promotable_event_evidence(ledger[idx]) for idx in component):
            continue
        aggregate = aggregate_event_community_row(component, ledger)
        if not aggregate_event_community_has_low_af_certificate(aggregate):
            continue
        calls.append(aggregated_event_community_call_from_row(aggregate))


# ---------------------------------------------------------------------------
# The run-level result, and the calibration passes that need it.
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    total_reads: int = 0
    gate1_passed: int = 0
    processed_bins: int = 0
    built_components: int = 0
    event_consensus_calls: int = 0
    genotype_calls: int = 0
    final_pass_calls: int = 0
    #: Re-estimated at finalization from this run's own count distribution.
    estimated_overdispersion: float = 0.02
    #: The measured dependency bound. Earlier releases asserted 0.65 nats
    #: (sigma = 1.92) with no evidence; these fields report what was used.
    estimated_dependency_penalty: float = 0.0
    estimated_dependency_penalty_non_te: float = 0.0
    estimated_dependency_sigma: float = 1.0
    estimated_dependency_sigma_non_te: float = 1.0
    #: Derived, not chosen: log(m/q), the largest e-value e-BH can ever require.
    dependency_penalty_cap_log: float = 0.0
    dependency_penalty_null_count: int = 0
    dependency_penalty_estimated: bool = False

    final_calls: list[FinalCall] = field(default_factory=list)
    #: Selected structural insertions the TE-calibrated mode sets aside. They
    #: are SELECTED calls, not rejects -- moving them here rather than erasing
    #: them keeps the TE output TE-only while leaving them auditable. Silently
    #: discarding them made recall benchmarking on the default mode misleading.
    structural_calls: list[FinalCall] = field(default_factory=list)
    evidence_ledger: list[EvidenceLedgerRow] = field(default_factory=list)


def apply_dependency_penalty_calibration(result: PipelineResult,
                                         target_fdr: float) -> None:
    """Measure the dependency bound, then apply it to every call and row.

    Summing the block log-LRs multiplies their e-values, and a product of
    DEPENDENT e-values is not an e-value until divided by an upper bound on its
    null expectation (Wang 2025, Biometrika 112(2):asaf020, Sec. 5b). Earlier
    releases subtracted a fixed 0.65 nats -- i.e. asserted sigma <= 1.9155 with
    no evidence. sigma is a null expectation and the run computes its own null
    rows, so it is measured instead.

    EACH COMPARISON GETS ITS OWN BOUND. The two aggregates are different
    products over differently-weighted blocks, so their null expectations
    differ. Estimating both is what removed the unexplained 0.60 factor that
    used to relate them.

    THE CAP IS REQUIRED FOR VALIDITY, not for convenience. sigma bounds
    `E_null[min(Y, C)]` with `C = exp(cap_log)`, so only `min(Y, C)/sigma` is
    guaranteed to have null expectation <= 1 -- the uncapped `Y/sigma` is not,
    because `E_null[Y]` can exceed `E_null[min(Y, C)]`. Applying the penalty to
    the uncapped aggregate, which is what the code did before, built e-values
    that e-BH is entitled to assume and that the estimator never certified.

    Its cost: the largest attainable e-value becomes `C/sigma` rather than `C`,
    so e-BH's rank-1 threshold is out of reach whenever sigma > 1. That corner
    matters only when a single candidate in the whole run is real; at any
    selection size r > 1 the threshold m/(q*r) is far below C.

    NO MINIMUM NULL COUNT. With few controls the bound is wide, the penalty
    large, and nothing passes -- which is the right answer when there is no null
    to calibrate against.
    """
    from placer_py.dependency import estimate_dependency_penalty

    null_art: list[float] = []
    null_non: list[float] = []
    for row in result.evidence_ledger:
        # is_dependency_calibration_row, NOT is_sample_local_null_control: see
        # that predicate. Selecting this sample by the aggregate being
        # calibrated truncates its right tail and drives sigma to its floor.
        if not is_dependency_calibration_row(row):
            continue
        if math.isfinite(row.mechanistic_raw_log_bf_te_vs_artifact):
            null_art.append(row.mechanistic_raw_log_bf_te_vs_artifact)
        if math.isfinite(row.mechanistic_raw_log_bf_te_vs_non_te):
            null_non.append(row.mechanistic_raw_log_bf_te_vs_non_te)

    estimate = estimate_dependency_penalty(null_art, null_non, target_fdr,
                                           len(result.final_calls))
    result.estimated_dependency_penalty = estimate.vs_artifact.log_penalty
    result.estimated_dependency_penalty_non_te = estimate.vs_non_te.log_penalty
    result.estimated_dependency_sigma = estimate.vs_artifact.sigma_upper
    result.estimated_dependency_sigma_non_te = estimate.vs_non_te.sigma_upper
    result.dependency_penalty_cap_log = estimate.cap_log
    result.dependency_penalty_null_count = estimate.null_count
    result.dependency_penalty_estimated = estimate.estimated

    def apply(raw: float, cap_log: float, penalty: float) -> float:
        return min(raw, cap_log) - penalty

    for call in result.final_calls:
        call.mechanistic_lower_log_bf_te_vs_artifact = apply(
            call.mechanistic_raw_log_bf_te_vs_artifact, estimate.cap_log,
            estimate.vs_artifact.log_penalty)
        call.mechanistic_lower_log_bf_te_vs_non_te = apply(
            call.mechanistic_raw_log_bf_te_vs_non_te, estimate.cap_log,
            estimate.vs_non_te.log_penalty)
    for row in result.evidence_ledger:
        row.mechanistic_lower_log_bf_te_vs_artifact = apply(
            row.mechanistic_raw_log_bf_te_vs_artifact, estimate.cap_log,
            estimate.vs_artifact.log_penalty)
        row.mechanistic_lower_log_bf_te_vs_non_te = apply(
            row.mechanistic_raw_log_bf_te_vs_non_te, estimate.cap_log,
            estimate.vs_non_te.log_penalty)


def apply_sample_overdispersion_calibration(result: PipelineResult) -> None:
    """Re-estimate the beta-binomial overdispersion, then re-genotype.

    Over the WHOLE ledger, not just the calls: the overdispersion is a property
    of the sample's sequencing and mapping, and estimating it from the selected
    calls alone would measure it on the loci least representative of the rest.

    The likelihood inputs the call was DECIDED with are reused and only the
    overdispersion changes. Rebuilding a default input here would silently drop
    the configured error rate, the min-GQ threshold and the length-concordance
    term, so the reported GQ and AF would describe a different model than the
    decision came from.
    """
    from placer_py.genotype import estimate_overdispersion, genotype_from_alt_vs_ref

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
            min_gq=inputs.min_gq, event_length=inputs.event_length,
            alt_observed_lengths=list(inputs.alt_observed_lengths))
        call.gq = decision.gq
        call.af = decision.allele_fraction
        call.genotype = decision.best_gt
        call.genotype_likelihood_input = inputs


def apply_sequence_family_commitments(result: PipelineResult) -> None:
    """Commit a family label, but only AFTER selection is finished.

    THE ORDERING IS THE POINT. A resolved sequence annotation may be retained
    while the event-level decision abstains on family, and committing the label
    earlier would let family coverage influence breakpoint choice,
    de-duplication or emission -- i.e. let the TE library feed back into
    detection. Running it last makes the label a description of a call that was
    already decided.
    """
    from placer_py.breakpoints import append_qc_token

    for call in result.final_calls:
        if (not call.sequence_family_commit_eligible or call.family_committed
                or not starts_with_text(call.final_qc, "PASS_TE")):
            continue
        call.family = call.sequence_family_candidate
        call.subfamily = call.sequence_subfamily_candidate
        call.te_name = call.subfamily
        call.family_committed = True
        call.final_qc = append_qc_token(call.final_qc, "SEQUENCE_FAMILY_COMMITTED")


def apply_sample_local_conformal_selector(result: PipelineResult,
                                          target_fdr: float) -> None:
    """The conformal/BY route, plus two fallbacks, plus the final drop.

    Runs AFTER e-BH and does not downgrade a call e-BH selected: either route is
    sufficient, and keeping the conformal QC where it passed preserves its
    diagnostics.

    THE SMALL-NULL FALLBACK. Below 20 null controls a conformal p-value cannot
    be smaller than 1/21, so the BY threshold is unreachable and the route would
    silently select nothing. Rather than report that as a rejection, calls with
    a robust lFDR pass are admitted under their own certificate.

    THE LAST STEP IS A DROP: any call that no route selected is removed.
    """
    from placer_py.conformal import ConformalNullSelector

    if not result.final_calls or not result.evidence_ledger:
        return

    selector = ConformalNullSelector()
    for i, row in enumerate(result.evidence_ledger):
        if is_sample_local_null_control(row):
            selector.add_null_control(conformal_feature_from_row(row, i))

    candidates = [conformal_feature_from_call(call, i)
                  for i, call in enumerate(result.final_calls)]
    decisions = {decision.id: decision
                 for decision in selector.select(candidates, target_fdr)}

    for i, call in enumerate(result.final_calls):
        decision = decisions.get(conformal_feature_from_call(call, i).id)
        if decision is None:
            continue
        call.conformal_null_p = decision.conformal_p
        call.conformal_by_threshold = decision.by_threshold
        call.conformal_dominated_nulls = decision.dominated_null_count
        call.conformal_null_count = decision.null_count
        if not (call.ebh_selected and decision.qc != "PASS_CONFORMAL_FDR"):
            call.conformal_qc = decision.qc

    def passes_event_lfdr_fallback(call: FinalCall) -> bool:
        q = min(1.0, max(0.0, target_fdr))
        if not call.mechanistic_blocks or call.mechanistic_blocks == "NA":
            return False
        if call.mechanistic_ref_conflict_signal > 0.50:
            return False
        if call.ref_span_reads > max(2, call.alt_struct_reads // 2):
            return False
        return (call.robust_mechanistic_qc == "PASS_TE_LFDR"
                and call.robust_mechanistic_worst_case_lfdr <= max(q, 0.10))

    if selector.null_count() < 20:
        for call in result.final_calls:
            if call.conformal_qc != "PASS_CONFORMAL_FDR" and passes_event_lfdr_fallback(call):
                call.conformal_qc = "PASS_EVENT_LFDR_FALLBACK"
                call.conformal_null_count = selector.null_count()

    apply_event_bayesian_fdr_fallback(result.final_calls, target_fdr)
    apply_structural_event_bayesian_fdr_fallback(result.final_calls, target_fdr)

    result.final_calls = [call for call in result.final_calls
                          if final_call_has_reportable_conformal_certificate(call)]
    result.final_calls = remove_pareto_dominated_final_calls(result.final_calls)


def structural_event_local_fdr(call: FinalCall) -> float:
    """The local FDR that an INSERTION (not a TE) exists here.

    Mirrors `competing_nonte_long_insertion_log_evidence` with one addition: the
    reference-conflict penalty is 0.25 for a balanced heterozygous locus and
    1.20 otherwise. A true het legitimately carries many reference reads, and
    charging it the full penalty would penalise it for being heterozygous.

    The veto above it is the same shape as elsewhere: high reference conflict
    with no balance, no direct insert and no structure support returns 1.0.
    """
    if not is_structural_insertion_call(call):
        return 1.0
    alt = max(0, call.alt_struct_reads)
    ref = max(0, call.ref_span_reads)
    if alt <= 0:
        return 1.0
    if not call.mechanistic_blocks or call.mechanistic_blocks == "NA":
        return 1.0

    heterozygous_balance = heterozygous_balance_log_evidence(alt, ref)
    raw_insert_log_bf = low_allele_fraction_event_log_bf(max(0, call.raw_cigar_insert_reads), ref)
    direct_insert_evidence = max(0.0, raw_insert_log_bf) if math.isfinite(raw_insert_log_bf) else 0.0
    assembled_insert_evidence = (0.05 * math.log1p(max(0, call.max_raw_cigar_insert_len))
                                 + 0.03 * math.log1p(max(0, call.event_consensus_len)))
    has_structure_model = "structure" in call.mechanistic_blocks
    event_structure_log_bf = ((max(call.te_structure_log_evidence,
                                   call.nonte_structure_log_evidence)
                               - call.artifact_structure_log_evidence)
                              if has_structure_model else -_INF)
    event_structure_evidence = event_structure_log_bf if math.isfinite(event_structure_log_bf) else 0.0
    structure_supports_event = event_structure_evidence > 0.0
    balanced_heterozygous = (heterozygous_balance >= 1.75 and alt >= 8
                             and call.mechanistic_ref_conflict_signal <= 0.65)
    direct_raw_insertion = direct_insert_evidence > 0.0
    if (call.mechanistic_ref_conflict_signal > 0.50 and not balanced_heterozygous
            and not direct_raw_insertion and not structure_supports_event):
        return 1.0

    artifact_posterior = _clamp01(call.artifact_posterior)
    non_te_posterior = _clamp01(call.non_te_posterior)
    event_posterior = _clamp01(non_te_posterior + _clamp01(call.te_posterior))
    support_log_evidence = math.log(alt + 1) - math.log(ref + 1)
    ref_conflict_penalty = ((0.25 if balanced_heterozygous else 1.20)
                            * _clamp01(call.mechanistic_ref_conflict_signal))
    log_event_odds = (math.log(max(event_posterior, 1e-6))
                      - math.log(max(artifact_posterior, 1e-6))
                      + (0.35 * support_log_evidence) + event_structure_evidence
                      + direct_insert_evidence + assembled_insert_evidence
                      + (0.80 * heterozygous_balance) - ref_conflict_penalty)
    return 1.0 / (1.0 + math.exp(min(60.0, max(-60.0, log_event_odds))))


def apply_structural_event_bayesian_fdr_fallback(calls: list[FinalCall],
                                                 target_fdr: float) -> None:
    """The last route, for structural insertions only.

    Note the extra `local_fdr <= 0.25` PRE-FILTER, which the TE fallback does
    not have: a structural call whose local FDR is above a quarter is not
    eligible at all, however short the selected prefix would be. It bounds the
    worst call this route can admit, where the mean-lFDR rule alone bounds only
    the average.
    """
    if not calls:
        return
    local_fdrs: list[tuple[float, int]] = []
    for i, call in enumerate(calls):
        if call.conformal_qc in ("PASS_CONFORMAL_FDR", "PASS_EVENT_LFDR_FALLBACK",
                                 "PASS_EVENT_EBH", "PASS_EVENT_BFDR"):
            continue
        local_fdr = structural_event_local_fdr(call)
        if local_fdr <= 0.25:
            local_fdrs.append((local_fdr, i))
    if not local_fdrs:
        return

    local_fdrs.sort(key=lambda item: (item[0], item[1]))
    q = min(1.0, max(0.0, target_fdr))
    cumulative = 0.0
    selected_prefix = 0
    for rank in range(1, len(local_fdrs) + 1):
        cumulative += local_fdrs[rank - 1][0]
        if (cumulative / rank) <= q:
            selected_prefix = rank
    for rank in range(selected_prefix):
        local_fdr, index = local_fdrs[rank]
        calls[index].conformal_qc = "PASS_EVENT_EXISTENCE_BFDR"
        calls[index].conformal_by_threshold = local_fdr


def apply_final_report_mode(result: PipelineResult, report_mode: str) -> None:
    """Split TE calls from structural ones, keeping both.

    The set-aside calls are SELECTED structural insertions, not rejects. Moving
    them to `structural_calls` rather than erasing them keeps the TE output
    TE-only while leaving them auditable -- silently discarding them made recall
    benchmarking on the default mode misleading.
    """
    if report_mode == "Legacy":
        return
    kept = [call for call in result.final_calls
            if final_call_passes_te_calibrated_report_gate(call)]
    result.structural_calls = [call for call in result.final_calls
                               if not final_call_passes_te_calibrated_report_gate(call)]
    result.final_calls = kept


def finalize_final_calls(result: PipelineResult,
                         target_fdr: float = DEFAULT_FINAL_CONFORMAL_FDR,
                         filter_config: FinalCallFilterConfig | None = None) -> None:
    """The whole stage, in the order the C++ runs it.

    THE ORDER IS LOad-BEARING and every step depends on the one before:

      1. AGGREGATE communities, then PROMOTE clusters -- both add calls, and
         aggregation runs first so promotion can see and skip its communities;
      2. RE-GENOTYPE with the sample's own overdispersion;
      3. DE-DUPLICATE by locus, keeping the preferred representative;
      4. resolve promoted envelopes and COLLAPSE fragments of one event;
      5. filter short insertions;
      6. MEASURE the dependency bound -- everything below consumes the
         calibrated log Bayes factors, and the e-values are only e-values once
         the product has been divided by a bound on its null expectation;
      7. e-BH as the primary route, then the conformal route and its fallbacks;
      8. split the report, and only then commit family labels.

    Step 6 before step 7 is the one that cannot move: e-BH's guarantee assumes
    its inputs are e-values, and step 6 is what makes them so.
    """
    filter_config = filter_config or FinalCallFilterConfig()

    append_aggregated_event_community_calls(result.evidence_ledger, result.final_calls)
    append_promoted_event_cluster_calls(result.evidence_ledger, result.final_calls)
    apply_sample_overdispersion_calibration(result)
    result.final_calls.sort(key=final_call_sort_less)

    deduped: list[FinalCall] = []
    cluster_start = 0
    while cluster_start < len(result.final_calls):
        best = result.final_calls[cluster_start]
        cluster_end = cluster_start + 1
        while (cluster_end < len(result.final_calls)
               and same_call_locus(result.final_calls[cluster_start],
                                   result.final_calls[cluster_end])):
            if prefer_new_call(result.final_calls[cluster_end], best):
                best = result.final_calls[cluster_end]
            cluster_end += 1
        deduped.append(best)
        cluster_start = cluster_end

    result.final_calls = deduped
    result.final_calls.sort(key=final_call_sort_less)
    result.final_calls = resolve_overlapping_promoted_event_envelopes(result.final_calls)
    result.final_calls = collapse_shared_support_final_call_fragments(result.final_calls)
    result.final_calls.sort(key=final_call_sort_less)
    result.final_calls = filter_short_final_insertions(result.final_calls, filter_config)

    apply_dependency_penalty_calibration(result, target_fdr)
    apply_event_ebh_selection(result.final_calls, target_fdr)
    apply_sample_local_conformal_selector(result, target_fdr)
    apply_final_report_mode(result, filter_config.report_mode)
    apply_sequence_family_commitments(result)

    result.final_pass_calls = len(result.final_calls)


#: How far around a call the reference is checked for N-rich context.
FINAL_CALL_REFERENCE_N_CONTEXT_BP = 250


def drop_calls_in_n_rich_reference(result: PipelineResult, detector) -> None:
    """Remove calls whose reference context is mostly N.

    A reference gap produces exactly the signature of an insertion -- reads stop
    explaining the reference, clip, and carry unexplained sequence -- for a
    reason that has nothing to do with the sample. The check is done BEFORE
    finalization proper so these never enter the null set either, where they
    would make the null look more TE-like than it is.
    """
    if not detector.can_fetch_reference() or not result.final_calls:
        return

    def overlaps_n_rich(call: FinalCall) -> bool:
        if not call.chrom:
            return False
        left = min(call.bp_left, call.bp_right)
        right = max(call.bp_left, call.bp_right)
        anchor = call.pos if call.pos >= 0 else (
            left + ((right - left) // 2) if (left >= 0 and right >= 0) else -1)
        interval_start = (max(0, left - FINAL_CALL_REFERENCE_N_CONTEXT_BP) if left >= 0
                          else (max(0, anchor - FINAL_CALL_REFERENCE_N_CONTEXT_BP)
                                if anchor >= 0 else -1))
        interval_end = (right + FINAL_CALL_REFERENCE_N_CONTEXT_BP + 1 if right >= 0
                        else (anchor + FINAL_CALL_REFERENCE_N_CONTEXT_BP + 1
                              if anchor >= 0 else -1))
        if (interval_start >= 0 and interval_end > interval_start
                and detector.reference_interval_is_n_rich(call.chrom, interval_start,
                                                          interval_end)):
            return True
        return ((anchor >= 0 and detector.reference_position_is_poly_n(call.chrom, anchor))
                or (call.bp_left >= 0
                    and detector.reference_position_is_poly_n(call.chrom, call.bp_left))
                or (call.bp_right >= 0
                    and detector.reference_position_is_poly_n(call.chrom, call.bp_right)))

    result.final_calls = [call for call in result.final_calls if not overlaps_n_rich(call)]
