"""
Ranking breakpoint hypotheses, and the breakpoint-position posterior.

Ported from `src/pipeline/pipeline_hypothesis_emission_stage.inc`, pinned by
`tests/test_31_hypotheses.py`.

WHAT THIS STAGE DOES. `placer_py/breakpoints.py` enumerates candidate
breakpoints; this one evaluates each against the reads, collapses the ones that
turn out to be the same hypothesis, and decides which survive to the expensive
stages (consensus, segmentation, TE alignment). That triage is what makes the
pipeline affordable -- the expensive stages cost orders of magnitude more than
everything before them, so the question is which hypotheses deserve them.

THE POSTERIOR IS REPORTED, NOT USED. `bp_ci_width` and `bp_posterior_entropy`
describe how well the supporting signals AGREE about where the breakpoint is,
and they gate nothing. A call can be confidently TE-positive and
breakpoint-imprecise -- those are different questions, and collapsing them into
one number is what a caller does when it wants a single quality score rather
than an honest answer.

EACH SIGNAL IS A NOISY MEASUREMENT, with a precision set by its kind:

    split / SA supplementary   sigma = 2 bp
    CIGAR insertion            sigma = 3 bp
    soft clip                  sigma = 6 bp
    anything else              sigma = 8 bp

Those are the same precisions the hypothesis priority ladder encodes, expressed
as standard deviations instead of weights -- and the mixture of their kernels is
a genuine posterior over position rather than a point estimate with an error bar
bolted on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from placer_py import supports
from placer_py.alignment import median_i32
from placer_py.breakpoints import (
    RESCUE_PRECISE_ANCHOR_MIN_DISTANCE_BP,
    breakpoint_hypothesis_support_weight,
)
from placer_py.clustering import (
    CANDIDATE_LONG_INSERTION,
    CANDIDATE_SOFT_CLIP,
    CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
    BreakpointCandidate,
    ComponentCall,
)
from placer_py.events import EventReadEvidence

#: Two hypotheses whose ends both agree within this, and whose support sets
#: agree by this Jaccard, are the same hypothesis.
HYPOTHESIS_COLLAPSE_SLACK_BP = 30
HYPOTHESIS_COLLAPSE_MIN_JACCARD = 0.8
#: Padding and cap on the posterior grid.
POSTERIOR_GRID_PAD_BP = 20
POSTERIOR_GRID_MAX_SPAN_BP = 2000
#: Per-signal precision, in bases. See the module docstring.
SIGMA_SPLIT_SA = 2.0
SIGMA_LONG_INSERTION = 3.0
SIGMA_SOFT_CLIP = 6.0
SIGMA_DEFAULT = 8.0


@dataclass
class HypothesisSummary:
    original_index: int = 0
    event_evidence: EventReadEvidence = field(default_factory=EventReadEvidence)
    bp_left: int = -1
    bp_right: int = -1
    alt_split_reads: int = 0
    alt_indel_reads: int = 0
    alt_left_clip_reads: int = 0
    alt_right_clip_reads: int = 0
    alt_struct_reads: int = 0
    raw_cigar_insert_reads: int = 0
    max_raw_cigar_insert_len: int = 0
    ref_span_reads: int = 0
    inferred_event_length: int = -1
    hypothesis_support: int = 0
    hypothesis_priority: int = 1 << 30
    hypothesis_score: float = 0.0
    support_qnames: list[str] = field(default_factory=list)


@dataclass
class BreakpointPosteriorSummary:
    ci_width: float = 0.0
    entropy: float = 0.0


def collect_alt_observed_lengths(component: ComponentCall,
                                 event_evidence: EventReadEvidence) -> list[int]:
    """Each SUPPORTING read's own idea of how long the insertion is.

    Only from reads in the support set -- a breakpoint candidate whose read did
    not end up supporting this hypothesis is measuring something else.

    A clip length is used when no insertion length is available, and it is a
    LOWER BOUND rather than a measurement: the read ran out. That biases the
    inferred length downward for events no read spans, which is the honest
    direction, and the genotyper's length-concordance term is where it matters.
    """
    support_qnames = set(event_evidence.support_qnames)
    lengths: list[int] = []
    for bp in component.breakpoint_candidates:
        if not bp.read_id or bp.read_id not in support_qnames:
            continue
        if bp.ins_len > 0:
            lengths.append(bp.ins_len)
        elif bp.clip_len > 0:
            lengths.append(bp.clip_len)
    return lengths


def infer_event_length_from_alt_support(lengths: list[int]) -> int:
    """The median observed length, or -1 when nothing observed one.

    Median rather than mean or max: one read's chimeric 12 kb clip must not set
    the event length, and the length feeds the genotyper's concordance term.
    """
    return median_i32(lengths)


#: Re-exported from `placer_py/supports.py`; `finalization.py` had the same
#: merge-walk, factored into two functions instead of inlined.
support_jaccard = supports.jaccard


def better_summary_representative(lhs: HypothesisSummary,
                                  rhs: HypothesisSummary) -> bool:
    """Which of two equivalent hypotheses describes the event better.

    Support, then PRECISE support, then the NARROWER breakpoint interval. The
    narrowness tie-break is the one worth naming: between two hypotheses with
    identical evidence, the one that commits to a tighter interval is the more
    useful description, and it is also the one more likely to be right, since a
    wide interval usually means the two ends came from different reads.
    """
    if lhs.alt_struct_reads != rhs.alt_struct_reads:
        return lhs.alt_struct_reads > rhs.alt_struct_reads
    lhs_precise = lhs.alt_split_reads + lhs.alt_indel_reads
    rhs_precise = rhs.alt_split_reads + rhs.alt_indel_reads
    if lhs_precise != rhs_precise:
        return lhs_precise > rhs_precise
    lhs_width = abs(lhs.bp_right - lhs.bp_left)
    rhs_width = abs(rhs.bp_right - rhs.bp_left)
    if lhs_width != rhs_width:
        return lhs_width < rhs_width
    return lhs.original_index < rhs.original_index


def collapse_hypothesis_summaries(summaries: list[HypothesisSummary]
                                  ) -> list[HypothesisSummary]:
    """Merge hypotheses that turned out to be the same one.

    BOTH ENDS within 30 bp AND a support Jaccard of at least 0.8. The
    conjunction matters: two hypotheses at the same position built from
    different reads are genuinely different (a tandem array), and two built from
    the same reads at different positions are genuinely different too (a long
    insertion's two junctions). Only agreeing on both makes them one.

    First-match-wins rather than best-match, matching the C++: the list arrives
    ranked, so the incumbent is already the better-ranked hypothesis, and the
    representative is then chosen by evidence rather than by rank.
    """
    kept: list[HypothesisSummary] = []
    for summary in summaries:
        merged = False
        for index, incumbent in enumerate(kept):
            close_left = abs(summary.bp_left - incumbent.bp_left) <= HYPOTHESIS_COLLAPSE_SLACK_BP
            close_right = abs(summary.bp_right - incumbent.bp_right) <= HYPOTHESIS_COLLAPSE_SLACK_BP
            if not close_left or not close_right:
                continue
            if support_jaccard(summary.support_qnames,
                               incumbent.support_qnames) < HYPOTHESIS_COLLAPSE_MIN_JACCARD:
                continue
            if better_summary_representative(summary, incumbent):
                kept[index] = summary
            merged = True
            break
        if not merged:
            kept.append(summary)
    return kept


def should_keep_hypothesis_for_expensive_stage(summary: HypothesisSummary,
                                               is_top_ranked_survivor: bool) -> bool:
    """Does this hypothesis deserve consensus, segmentation and alignment?

    THE TOP-RANKED SURVIVOR ALWAYS PASSES. Every component gets at least one
    expensive evaluation, whatever its evidence -- refusing all of them would
    mean deciding a locus with no evidence at the one stage that could have
    produced some.

    After that, three routes, in decreasing order of directness:

      * ANY precise read (split or CIGAR insertion). One is enough: a read that
        names the base is worth an alignment on its own.
      * BILATERAL clips with 2+ supporting reads. Clips on both sides bracket an
        insertion; one side alone is a mapping artifact's signature.
      * UNILATERAL clips, but only with 3+ clip reads, 3+ total, AND alt at
        least twice the reference. All four together describe a locus where the
        reference allele is nearly absent and several reads clip the same way,
        which is a real insertion whose far end no read reached.

    The `max(1, ref)` in the conflict test is what makes an UNOPPOSED locus pass
    it: with no reference reads the requirement becomes `alt >= 2`, not `alt >= 0`.
    """
    if is_top_ranked_survivor:
        return True
    if summary.alt_split_reads > 0 or summary.alt_indel_reads > 0:
        return True
    bilateral_clip = summary.alt_left_clip_reads > 0 and summary.alt_right_clip_reads > 0
    if bilateral_clip and summary.alt_struct_reads >= 2:
        return True
    unilateral_clip = (summary.alt_left_clip_reads > 0) != (summary.alt_right_clip_reads > 0)
    unilateral_clip_reads = max(max(0, summary.alt_left_clip_reads),
                                max(0, summary.alt_right_clip_reads))
    ref = max(0, summary.ref_span_reads)
    low_reference_conflict = summary.alt_struct_reads >= (2 * max(1, ref))
    return (unilateral_clip and unilateral_clip_reads >= 3
            and summary.alt_struct_reads >= 3 and low_reference_conflict)


def select_hypothesis_summaries_for_expensive_stage(summaries: list[HypothesisSummary]
                                                    ) -> list[HypothesisSummary]:
    """Collapse, then keep the ones worth an expensive evaluation.

    The collapse runs FIRST, so "top ranked" means top among distinct
    hypotheses -- otherwise a component whose first three hypotheses were three
    views of one junction would spend its guaranteed slot on a duplicate.
    """
    collapsed = collapse_hypothesis_summaries(summaries)
    return [summary for i, summary in enumerate(collapsed)
            if should_keep_hypothesis_for_expensive_stage(summary, i == 0)]


def should_record_hypothesis_in_evidence_ledger(summary: HypothesisSummary) -> bool:
    """Any read at all, alt OR reference.

    Deliberately the weakest possible test, because the ledger is also the null
    set: a locus with only reference reads is exactly the kind of row the
    dependency bound and the conformal null are measured on, and filtering it
    out would bias both toward looking more TE-like than the genome is.
    """
    return (summary.alt_struct_reads > 0 or summary.alt_split_reads > 0
            or summary.alt_indel_reads > 0 or summary.alt_left_clip_reads > 0
            or summary.alt_right_clip_reads > 0 or summary.ref_span_reads > 0)


def build_hypothesis_summary(component: ComponentCall,
                             event_evidence: EventReadEvidence,
                             original_index: int, hypothesis_support: int,
                             hypothesis_priority: int) -> HypothesisSummary:
    """Package one evaluated hypothesis, with its ladder score.

    `hypothesis_score` is `support x weight(priority)` -- the same product the
    breakpoint stage ranks by, recomputed here so the two stages cannot drift
    apart on how a hypothesis is scored.
    """
    summary = HypothesisSummary(original_index=original_index,
                                event_evidence=event_evidence)
    summary.bp_left = event_evidence.bp_left
    summary.bp_right = event_evidence.bp_right
    summary.alt_split_reads = event_evidence.alt_split_reads
    summary.alt_indel_reads = event_evidence.alt_indel_reads
    summary.alt_left_clip_reads = event_evidence.alt_left_clip_reads
    summary.alt_right_clip_reads = event_evidence.alt_right_clip_reads
    summary.alt_struct_reads = event_evidence.alt_struct_reads
    summary.raw_cigar_insert_reads = event_evidence.raw_cigar_insert_reads
    summary.max_raw_cigar_insert_len = event_evidence.max_raw_cigar_insert_len
    summary.ref_span_reads = event_evidence.ref_span_reads
    summary.inferred_event_length = infer_event_length_from_alt_support(
        collect_alt_observed_lengths(component, event_evidence))
    summary.hypothesis_support = hypothesis_support
    summary.hypothesis_priority = hypothesis_priority
    summary.hypothesis_score = (hypothesis_support
                                * breakpoint_hypothesis_support_weight(hypothesis_priority))
    summary.support_qnames = event_evidence.support_qnames
    return summary


def _signal_sigma(class_mask: int) -> float:
    if class_mask & CANDIDATE_SPLIT_SA_SUPPLEMENTARY:
        return SIGMA_SPLIT_SA
    if class_mask & CANDIDATE_LONG_INSERTION:
        return SIGMA_LONG_INSERTION
    if class_mask & CANDIDATE_SOFT_CLIP:
        return SIGMA_SOFT_CLIP
    return SIGMA_DEFAULT


def compute_breakpoint_position_posterior(candidates: list[BreakpointCandidate]
                                          ) -> BreakpointPosteriorSummary:
    """A posterior over breakpoint position, as a mixture of per-signal kernels.

    Each signal contributes `exp(-z^2/2) / sigma` at every grid position -- a
    Gaussian kernel scaled by its own precision, so a precise signal is both
    narrower AND taller. Summing rather than multiplying is deliberate: the
    signals are competing MEASUREMENTS of one latent position, not independent
    likelihoods, and multiplying would let two disagreeing precise reads produce
    a posterior with no mass anywhere.

    REPORTED, NOT GATED. The 90% credible-interval width and the normalised
    entropy say how well the reads agree, which is a different question from
    whether the insertion is real.

    FEWER THAN TWO SIGNALS RETURNS ZEROS -- there is no agreement to measure with
    one measurement, and reporting a width of 0 would claim perfect precision
    from a single read.
    """
    summary = BreakpointPosteriorSummary()
    signals: list[tuple[int, float]] = []
    lo = None
    hi = None
    for bp in candidates:
        if bp.pos < 0:
            continue
        signals.append((bp.pos, _signal_sigma(bp.class_mask)))
        lo = bp.pos if lo is None else min(lo, bp.pos)
        hi = bp.pos if hi is None else max(hi, bp.pos)
    if len(signals) < 2:
        return summary

    grid_lo = lo - POSTERIOR_GRID_PAD_BP
    grid_hi = hi + POSTERIOR_GRID_PAD_BP
    if (grid_hi - grid_lo) > POSTERIOR_GRID_MAX_SPAN_BP:
        grid_hi = grid_lo + POSTERIOR_GRID_MAX_SPAN_BP
    n = grid_hi - grid_lo + 1

    density = [0.0] * n
    total = 0.0
    for gi in range(n):
        x = float(grid_lo + gi)
        value = 0.0
        for pos, sigma in signals:
            z = (x - pos) / sigma
            value += math.exp(-0.5 * z * z) / sigma
        density[gi] = value
        total += value
    if total <= 0.0:
        return summary

    entropy = 0.0
    for gi in range(n):
        density[gi] /= total
        if density[gi] > 0.0:
            entropy -= density[gi] * math.log(density[gi])
    # Normalised by log(n) so the value is comparable across grid sizes: 1 is a
    # flat posterior over the whole grid, 0 is a point mass.
    summary.entropy = entropy / math.log(n)

    cumulative = 0.0
    q05 = 0
    for gi in range(n):
        cumulative += density[gi]
        if cumulative >= 0.05:
            q05 = gi
            break
    cumulative = 0.0
    q95 = n - 1
    for gi in range(n):
        cumulative += density[gi]
        if cumulative >= 0.95:
            q95 = gi
            break
    summary.ci_width = float(q95 - q05)
    return summary


# ---------------------------------------------------------------------------
# The validator: which hypotheses can even be evaluated, and which two are.
# ---------------------------------------------------------------------------
#: How far apart two shortlisted hypotheses' ends must be, or how different
#: their support, to count as distinct.
SHORTLIST_SPATIAL_DISTINCT_BP = 60
SHORTLIST_SUPPORT_DISTINCT_JACCARD = 0.5


@dataclass
class ConsensusInputCounts:
    full_context_input_reads: int = 0
    partial_context_input_reads: int = 0
    left_anchor_input_reads: int = 0
    right_anchor_input_reads: int = 0
    input_event_reads: int = 0


@dataclass
class HypothesisValidatorEvidence:
    summary: HypothesisSummary = field(default_factory=HypothesisSummary)
    event_evidence: EventReadEvidence = field(default_factory=EventReadEvidence)
    precise_support: int = 0
    breakpoint_width: int = 0
    anchor_distance: int = 0
    full_context_input_reads: int = 0
    partial_context_input_reads: int = 0
    left_anchor_input_reads: int = 0
    right_anchor_input_reads: int = 0
    input_event_reads: int = 0
    feasible_for_expensive_stage: bool = False
    qc_reason: str = "VALIDATOR_UNSET"


@dataclass
class ShortlistedHypothesis:
    validator: HypothesisValidatorEvidence = field(
        default_factory=HypothesisValidatorEvidence)
    is_primary: bool = False


def collect_hypothesis_validator_evidence(summary: HypothesisSummary,
                                          inputs: ConsensusInputCounts,
                                          anchor_pos: int) -> HypothesisValidatorEvidence:
    """Can a consensus be built for this hypothesis at all?

    The QC reasons are DEFINITIONAL refusals, not thresholds: without reads
    anchoring both sides there is nothing to assemble around, and without either
    a precise read or an identifiable bilateral partial context there is nothing
    to assemble. Running the expensive stages anyway would spend the run's time
    proving that.

    `input_event_reads` prefers the full-context count and falls back to the
    partial one -- it is "how many reads will the consensus actually be built
    from", and the consensus uses full context wholesale when any exists.
    """
    from placer_py.segmentation import has_identifiable_bilateral_partial_context_support

    out = HypothesisValidatorEvidence(summary=summary,
                                      event_evidence=summary.event_evidence)
    out.precise_support = summary.alt_split_reads + summary.alt_indel_reads
    out.breakpoint_width = max(0, summary.bp_right - summary.bp_left)
    out.anchor_distance = min(abs(summary.bp_left - anchor_pos),
                              abs(summary.bp_right - anchor_pos))
    out.full_context_input_reads = inputs.full_context_input_reads
    out.partial_context_input_reads = inputs.partial_context_input_reads
    out.left_anchor_input_reads = inputs.left_anchor_input_reads
    out.right_anchor_input_reads = inputs.right_anchor_input_reads
    out.input_event_reads = inputs.input_event_reads

    has_bilateral_anchor = (out.left_anchor_input_reads > 0
                            and out.right_anchor_input_reads > 0)
    has_precise_or_full_context = (out.full_context_input_reads > 0
                                   or out.precise_support > 0)
    has_partial = has_identifiable_bilateral_partial_context_support(
        out.left_anchor_input_reads, out.right_anchor_input_reads,
        out.partial_context_input_reads, out.input_event_reads)

    if not has_bilateral_anchor and not has_precise_or_full_context:
        out.qc_reason = "VALIDATOR_NO_BILATERAL_ANCHOR"
        return out
    if not (has_precise_or_full_context or has_partial):
        out.qc_reason = "VALIDATOR_NO_PRECISE_OR_FULL_CONTEXT"
        return out

    out.feasible_for_expensive_stage = True
    out.qc_reason = "VALIDATOR_PASS"
    return out


def hypothesis_validator_priority_key(evidence: HypothesisValidatorEvidence) -> tuple:
    """Order for the shortlist: precise support, context, BALANCE, then width.

    `min(left_anchor, right_anchor)` is the balance term, and taking the minimum
    is the same rule as the bilateral clip count: a consensus anchored by ten
    reads on one side and none on the other cannot be segmented, so the weaker
    side is what determines whether the expensive stage will succeed.
    """
    return (-evidence.precise_support, -evidence.full_context_input_reads,
            -min(evidence.left_anchor_input_reads, evidence.right_anchor_input_reads),
            evidence.breakpoint_width, evidence.anchor_distance,
            -evidence.summary.alt_struct_reads, evidence.summary.original_index)


def build_expensive_stage_shortlist(candidates: list[HypothesisValidatorEvidence]
                                    ) -> list[ShortlistedHypothesis]:
    """The hypotheses worth assembling: a primary, a challenger, and a rescue.

    WHY MORE THAN ONE. The primary is the best-ranked feasible hypothesis, but
    ranking uses read counts and the expensive stages can disagree with them --
    a hypothesis with fewer reads may segment cleanly where the top one does
    not. So a SPATIALLY OR SUPPORT-DISTINCT challenger is kept.

    "Distinct" is a DISJUNCTION: either end more than 60 bp away, OR a support
    Jaccard under 0.5. A hypothesis at the same position built from different
    reads is a genuinely different claim, and so is one at a different position
    built from the same reads.

    THE RESCUE is the same idea as in the breakpoint stage: a SINGLE-READ
    hypothesis from split or CIGAR-insertion evidence, at least 500 bp from
    everything shortlisted, is added regardless. One such read names a base.
    """
    feasible = [candidate for candidate in candidates
                if candidate.feasible_for_expensive_stage]
    if not feasible:
        return []
    feasible.sort(key=hypothesis_validator_priority_key)

    out = [ShortlistedHypothesis(feasible[0], True)]
    for candidate in feasible[1:]:
        redundant = False
        for selected in out:
            incumbent = selected.validator
            spatially_distinct = (
                abs(candidate.summary.bp_left - incumbent.summary.bp_left) > SHORTLIST_SPATIAL_DISTINCT_BP
                or abs(candidate.summary.bp_right - incumbent.summary.bp_right) > SHORTLIST_SPATIAL_DISTINCT_BP)
            support_distinct = support_jaccard(
                candidate.summary.support_qnames,
                incumbent.summary.support_qnames) < SHORTLIST_SUPPORT_DISTINCT_JACCARD
            if not spatially_distinct and not support_distinct:
                redundant = True
                break
        if not redundant:
            out.append(ShortlistedHypothesis(candidate, False))

    rescue = None
    for candidate in feasible:
        summary = candidate.summary
        if (summary.hypothesis_support != 1 or summary.hypothesis_priority not in (0, 1)
                or summary.bp_left != summary.bp_right):
            continue
        already_selected = False
        spatially_distinct = True
        for selected in out:
            incumbent = selected.validator.summary
            if (incumbent.original_index == summary.original_index
                    or (incumbent.bp_left == summary.bp_left
                        and incumbent.bp_right == summary.bp_right)):
                already_selected = True
                break
            incumbent_center = incumbent.bp_left + ((incumbent.bp_right - incumbent.bp_left) // 2)
            if abs(summary.bp_left - incumbent_center) < RESCUE_PRECISE_ANCHOR_MIN_DISTANCE_BP:
                spatially_distinct = False
                break
        if already_selected or not spatially_distinct:
            continue
        if rescue is None:
            rescue = candidate
            continue
        key = (summary.hypothesis_priority, -summary.hypothesis_score, summary.bp_left)
        best_key = (rescue.summary.hypothesis_priority, -rescue.summary.hypothesis_score,
                    rescue.summary.bp_left)
        if key < best_key:
            rescue = candidate
    if rescue is not None:
        out.append(ShortlistedHypothesis(rescue, False))
    return out
