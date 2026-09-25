"""
Choosing one call per component, and re-anchoring the ones that drifted.

Ported from `src/pipeline/pipeline_call_selection.inc`, pinned by
`tests/test_30_call_selection.py`.

WHERE THIS SITS. Each component produces several breakpoint hypotheses, each
evaluated independently, so a component can end up with several emittable calls
describing one event at slightly different positions. This stage picks one --
and, before picking, moves calls whose position came from a WEAK source onto a
nearby PRECISE one.

RETETHERING IS THE UNUSUAL PART, and it is worth being clear about what it does:
it MOVES a call's reported position to another hypothesis's position. That is a
strong thing to do, so it is hedged in four ways:

  * only calls that are already emitting TE evidence are moved;
  * only onto anchors from a precise source (split, CIGAR insertion) unless the
    call is one-sided;
  * only over 100..3000 bp -- under 100 the move is noise, over 3000 the anchor
    belongs to a different event;
  * never for a one-sided segmentation, which is already positioned by its own
    consensus-to-reference flank alignment and has a sequence certificate for
    where it is.

The last exclusion is the one that was learned the hard way: moving a one-sided
call to an unrelated component hypothesis turns a correctly anchored call into a
distant duplicate, and there is no sequence evidence to justify it.
"""

from __future__ import annotations

from dataclasses import dataclass

from placer.core.finalization import FINAL_CALL_DEDUP_DISTANCE_BP

#: An anchor must have at least this much support to tether anything.
MIN_RETETHER_ANCHOR_SUPPORT = 1
#: Under this distance the move is within the noise of the original position.
MIN_RETETHER_DISTANCE_BP = 100
#: Over this distance the anchor belongs to a different event.
MAX_RETETHER_DISTANCE_BP = 3000
#: Radii around the component anchor, used to score how "central" an anchor is.
CORE_ANCHOR_RADIUS_BP = 100
LOCAL_ANCHOR_RADIUS_BP = 450
MAX_ANCHOR_COMPONENT_DISTANCE_BP = 1000

#: Hypothesis priorities that NAME a base: fragment split, fragment indel, raw
#: split, raw indel. Clip-derived priorities (3, 5) are excluded -- a clip
#: cannot re-anchor anything, because it does not know where the junction is.
_PRECISE_ANCHOR_PRIORITIES = (0, 1, 2, 4)


@dataclass
class ComponentFinalCallCandidate:
    pos: int = -1
    anchor_pos: int = -1
    score: float = 0.0
    emit_te: bool = False
    evidence_te: bool = False
    resolved_te: bool = False
    one_sided_segmentation: bool = False
    anchor_support: int = 0
    anchor_ref_span_reads: int = 0
    anchor_priority: int = 1 << 30
    anchor_hypothesis_score: float = 0.0
    component_anchor_pos: int = -1
    #: -1 to allow only leftward moves, +1 only rightward, 0 for either. Set
    #: from the segmentation's own orientation, so a call that knows which side
    #: its junction is on cannot be dragged the wrong way.
    retether_direction: int = 0


def is_precise_anchor(priority: int) -> bool:
    return priority in _PRECISE_ANCHOR_PRIORITIES


def select_component_final_call_indices(
        candidates: list[ComponentFinalCallCandidate]) -> list[int]:
    """One call per 50 bp cluster: the highest-scoring, ties to the lower index.

    Only emitting candidates are considered at all -- the rest are kept in the
    list because retethering reads them as ANCHORS, not because they could be
    selected.

    Clustering is against the cluster's FIRST position, held fixed, not against
    a running one. So a chain of candidates each 40 bp from the last does NOT
    collapse into a single call -- it breaks every time the fixed anchor is
    exceeded. That bounds a cluster's width at 50 bp rather than letting it
    creep across a repeat, and it is the behaviour the C++ has.
    """
    order = sorted((i for i, candidate in enumerate(candidates) if candidate.emit_te),
                   key=lambda i: (candidates[i].pos, i))
    if not order:
        return []

    selected: list[int] = []
    cluster_start = 0
    while cluster_start < len(order):
        anchor_pos = candidates[order[cluster_start]].pos
        best = order[cluster_start]
        cluster_end = cluster_start
        while (cluster_end < len(order)
               and abs(candidates[order[cluster_end]].pos - anchor_pos)
               <= FINAL_CALL_DEDUP_DISTANCE_BP):
            idx = order[cluster_end]
            if (candidates[idx].score > candidates[best].score
                    or (candidates[idx].score == candidates[best].score and idx < best)):
                best = idx
            cluster_end += 1
        selected.append(best)
        cluster_start = cluster_end

    return sorted(selected, key=lambda i: (candidates[i].pos, i))


def _retether_anchor_score(candidate: ComponentFinalCallCandidate,
                           anchor: ComponentFinalCallCandidate) -> float:
    """How good an anchor is for this candidate.

    The two bonuses are deliberately enormous (1000 for a precise source, 2000
    for sitting within 100 bp of the component anchor) so they act as a
    LEXICOGRAPHIC order rather than as weights: a central anchor beats any
    off-centre one, and among equally central ones a precise source wins. The
    two small distance penalties then break ties smoothly.
    """
    distance = abs(candidate.pos - anchor.pos)
    component_distance = (abs(anchor.pos - candidate.component_anchor_pos)
                          if candidate.component_anchor_pos >= 0 else (1 << 30))
    score = anchor.anchor_hypothesis_score
    if is_precise_anchor(anchor.anchor_priority):
        score += 1000.0
    if component_distance <= CORE_ANCHOR_RADIUS_BP:
        score += 2000.0
    if component_distance <= LOCAL_ANCHOR_RADIUS_BP:
        score += 50.0
    score -= distance * 0.01
    score -= component_distance * 0.001
    return score


def retether_evidence_supported_final_call_positions(
        candidates: list[ComponentFinalCallCandidate]) -> None:
    """Move drifted calls onto precise anchors. Mutates in place.

    FOUR PATHS, tried in order, and each is narrower than the one before:

      1. SELF-ANCHOR, two-sided. The candidate's own hypothesis anchor is
         precise and well supported -- move to it. Requires either a real
         distance (>= 100 bp) or a resolved TE, so a resolved call snaps to its
         anchor even over a few bases.
      2. SELF-ANCHOR, resolved TE. Same, unconditional on distance, and it
         `continue`s -- a resolved TE with a precise anchor is finished.
      3. SELF-ANCHOR, unresolved but strongly supported (>= 2 reads, score >=
         10) AND with reference reads at the anchor. The reference requirement
         is the guard: without it this path would move calls onto anchors in
         unmapped regions.
      4. CROSS-ANCHOR. Search every NON-emitting candidate for the best anchor.
         This is the only path that moves a call onto a different hypothesis,
         and the direction constraint applies here. NOTE that it starts from the
         candidate's OWN anchor as the incumbent tether, so a call whose self
         anchor sits 100..3000 bp away is moved onto it here even when paths 1-3
         all declined -- the earlier paths differ by CONTINUING (skipping this
         search), not by being the only way to move.

    A one-sided segmentation skips all four: it is positioned by its own
    consensus-to-reference flank and has a sequence certificate for where it is.
    """
    for candidate in candidates:
        if not candidate.emit_te or not candidate.evidence_te or candidate.pos < 0:
            continue
        if candidate.one_sided_segmentation:
            continue

        # Path 1.
        if (candidate.anchor_pos >= 0 and is_precise_anchor(candidate.anchor_priority)
                and candidate.anchor_support >= MIN_RETETHER_ANCHOR_SUPPORT
                and (candidate.anchor_hypothesis_score >= 8.0 or candidate.resolved_te)):
            self_anchor_distance = abs(candidate.pos - candidate.anchor_pos)
            if (self_anchor_distance <= MAX_RETETHER_DISTANCE_BP
                    and (self_anchor_distance >= MIN_RETETHER_DISTANCE_BP
                         or candidate.resolved_te)):
                candidate.pos = candidate.anchor_pos

        # Path 2.
        if (candidate.resolved_te and is_precise_anchor(candidate.anchor_priority)
                and candidate.anchor_pos >= 0
                and candidate.anchor_support >= MIN_RETETHER_ANCHOR_SUPPORT):
            if abs(candidate.pos - candidate.anchor_pos) <= MAX_RETETHER_DISTANCE_BP:
                candidate.pos = candidate.anchor_pos
                continue

        # Path 3.
        if (not candidate.resolved_te and candidate.anchor_pos >= 0
                and candidate.anchor_support >= 2
                and candidate.anchor_hypothesis_score >= 10.0
                and candidate.anchor_ref_span_reads > 0):
            self_anchor_distance = abs(candidate.pos - candidate.anchor_pos)
            self_component_distance = (abs(candidate.anchor_pos - candidate.component_anchor_pos)
                                       if candidate.component_anchor_pos >= 0 else (1 << 30))
            if (self_component_distance <= LOCAL_ANCHOR_RADIUS_BP
                    and self_anchor_distance <= MAX_RETETHER_DISTANCE_BP):
                if self_anchor_distance >= MIN_RETETHER_DISTANCE_BP:
                    candidate.pos = candidate.anchor_pos
                continue

        # Path 4.
        tether_pos = candidate.anchor_pos
        tether_support = candidate.anchor_support
        tether_ref_span = candidate.anchor_ref_span_reads
        tether_hypothesis_score = candidate.anchor_hypothesis_score
        best_tether_score = float("-inf")

        for anchor in candidates:
            # NON-emitting candidates only: an emitting one is a call in its own
            # right, and tethering to it would merge two calls by moving one
            # onto the other.
            if anchor.emit_te or anchor.pos < 0:
                continue
            if (anchor.anchor_support < MIN_RETETHER_ANCHOR_SUPPORT
                    or anchor.anchor_hypothesis_score <= 0.0):
                continue
            if not candidate.one_sided_segmentation and not is_precise_anchor(anchor.anchor_priority):
                continue
            distance = abs(candidate.pos - anchor.pos)
            if distance < MIN_RETETHER_DISTANCE_BP or distance > MAX_RETETHER_DISTANCE_BP:
                continue
            if (candidate.component_anchor_pos >= 0
                    and abs(anchor.pos - candidate.component_anchor_pos)
                    > MAX_ANCHOR_COMPONENT_DISTANCE_BP):
                continue
            # An imprecise anchor may not displace a candidate that already has
            # a better-scoring anchor of its own, unless it is central.
            if (not is_precise_anchor(anchor.anchor_priority)
                    and candidate.anchor_hypothesis_score > 0.0
                    and not (candidate.component_anchor_pos >= 0
                             and abs(anchor.pos - candidate.component_anchor_pos)
                             <= CORE_ANCHOR_RADIUS_BP)
                    and anchor.anchor_hypothesis_score + 1e-9 < candidate.anchor_hypothesis_score):
                continue
            if candidate.retether_direction < 0 and anchor.pos > candidate.pos:
                continue
            if candidate.retether_direction > 0 and anchor.pos < candidate.pos:
                continue

            score = _retether_anchor_score(candidate, anchor)
            key = (-score, -anchor.anchor_hypothesis_score, -anchor.anchor_support,
                   anchor.anchor_ref_span_reads, anchor.pos)
            best_key = (-best_tether_score, -tether_hypothesis_score, -tether_support,
                        tether_ref_span, tether_pos)
            if key < best_key:
                tether_pos = anchor.pos
                tether_support = anchor.anchor_support
                tether_ref_span = anchor.anchor_ref_span_reads
                tether_hypothesis_score = anchor.anchor_hypothesis_score
                best_tether_score = score

        if tether_support < MIN_RETETHER_ANCHOR_SUPPORT or tether_pos < 0:
            continue
        distance = abs(candidate.pos - tether_pos)
        if distance < MIN_RETETHER_DISTANCE_BP or distance > MAX_RETETHER_DISTANCE_BP:
            continue
        candidate.pos = tether_pos
