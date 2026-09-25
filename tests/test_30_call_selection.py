"""
One call per component, re-anchoring, and the local-interval cache.

The `select_component_final_call_indices` and retethering cases from
`tests/test_component_multi_emit_selection.cpp`, and the canonical-interval case
from `tests/test_parallel_local_interval_cache.cpp`, are reproduced verbatim --
these are among the few C++ tests that reach a pipeline helper directly.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer.alignment import AlignedRead
from placer.core import call_selection as S
from placer.core import interval_cache as I
from placer.core.events import ReadReferenceSpan

pytestmark = pytest.mark.invariant

C = S.ComponentFinalCallCandidate


# ----------------------------------------------------------- call selection
def test_two_calls_far_enough_apart_are_both_kept():
    """The C++ case: 1835 bp apart is two events, not one."""
    candidates = [C(pos=17680845, score=4.19922, emit_te=True),
                  C(pos=17682680, score=5.95314, emit_te=True)]
    assert call_or_skip(S.select_component_final_call_indices, candidates) == [0, 1]


def test_two_calls_inside_the_dedup_distance_keep_the_higher_score():
    """And a rejected candidate never competes, however high its score."""
    candidates = [C(pos=1000, score=2.0, emit_te=True),
                  C(pos=1040, score=3.0, emit_te=True),
                  C(pos=1400, score=9.0, emit_te=False)]
    assert S.select_component_final_call_indices(candidates) == [1]


def test_a_score_tie_breaks_toward_the_lower_index():
    """Determinism: the same component must select the same call on every run."""
    candidates = [C(pos=1000, score=3.0, emit_te=True),
                  C(pos=1020, score=3.0, emit_te=True)]
    assert S.select_component_final_call_indices(candidates) == [0]


def test_no_emitting_candidate_selects_nothing():
    assert S.select_component_final_call_indices([C(pos=1000, score=9.0)]) == []
    assert S.select_component_final_call_indices([]) == []


def test_clustering_is_against_a_fixed_anchor_not_a_running_one():
    """
    Pinned because it is the kind of detail a reimplementation flips. The
    cluster window is measured from the cluster's FIRST position, held fixed, so
    a chain each 40 bp from the last does NOT collapse into one call -- it
    breaks every time the fixed anchor is exceeded. That bounds a cluster at
    50 bp instead of letting it creep across a repeat.
    """
    candidates = [C(pos=1000 + 40 * i, score=float(i), emit_te=True) for i in range(5)]
    assert S.select_component_final_call_indices(candidates) == [1, 3, 4]


def test_selected_indices_come_back_in_position_order():
    candidates = [C(pos=5000, score=1.0, emit_te=True),
                  C(pos=1000, score=1.0, emit_te=True)]
    assert S.select_component_final_call_indices(candidates) == [1, 0]


# -------------------------------------------------------------- retethering
def test_a_one_sided_call_is_never_moved():
    """
    THE exclusion that was learned the hard way. A one-sided event is positioned
    by its own consensus-to-reference flank alignment -- it has a sequence
    certificate for where it is. Moving it to an unrelated component hypothesis
    turns a correctly anchored call into a distant duplicate.
    """
    candidates = [C(pos=28237107, anchor_pos=28235587, score=4.39, emit_te=True,
                    evidence_te=True, one_sided_segmentation=True,
                    anchor_support=6, anchor_ref_span_reads=8)]
    call_or_skip(S.retether_evidence_supported_final_call_positions, candidates)
    assert candidates[0].pos == 28237107


def test_a_call_carrying_no_te_evidence_is_never_moved():
    """The C++ case: `evidence_te` false, so the call keeps its position however
    well supported the anchor is."""
    candidates = [C(pos=38193873, anchor_pos=38195168, score=4.37, emit_te=True,
                    evidence_te=False, anchor_support=12)]
    S.retether_evidence_supported_final_call_positions(candidates)
    assert candidates[0].pos == 38193873


def test_a_resolved_te_snaps_to_its_own_precise_anchor():
    """
    Path 2, and it is unconditional on distance below the 3 kb ceiling: a
    resolved TE with a precise anchor is finished, and the anchor is a better
    position than the one the hypothesis scoring produced.
    """
    candidates = [C(pos=1000, anchor_pos=1020, emit_te=True, evidence_te=True,
                    resolved_te=True, anchor_priority=0, anchor_support=4)]
    S.retether_evidence_supported_final_call_positions(candidates)
    assert candidates[0].pos == 1020


def test_a_move_shorter_than_a_hundred_bases_is_noise_and_is_refused():
    candidates = [C(pos=1000, anchor_pos=1050, emit_te=True, evidence_te=True,
                    anchor_priority=0, anchor_support=4,
                    anchor_hypothesis_score=20.0)]
    S.retether_evidence_supported_final_call_positions(candidates)
    assert candidates[0].pos == 1000


def test_a_move_beyond_three_kilobases_belongs_to_a_different_event():
    candidates = [C(pos=1000, anchor_pos=9000, emit_te=True, evidence_te=True,
                    anchor_priority=0, anchor_support=4,
                    anchor_hypothesis_score=20.0)]
    S.retether_evidence_supported_final_call_positions(candidates)
    assert candidates[0].pos == 1000


def test_a_clip_derived_anchor_cannot_retether_a_two_sided_call():
    """
    A clip does not know where the junction is -- it knows only that the read
    stopped explaining the reference nearby. Only the four precise priorities
    may re-anchor.
    """
    assert call_or_skip(S.is_precise_anchor, 0)
    assert S.is_precise_anchor(1) and S.is_precise_anchor(2) and S.is_precise_anchor(4)
    assert not S.is_precise_anchor(3)
    assert not S.is_precise_anchor(5)

    candidates = [C(pos=1000, emit_te=True, evidence_te=True),
                  C(pos=1500, anchor_priority=5, anchor_support=4,
                    anchor_hypothesis_score=20.0)]
    S.retether_evidence_supported_final_call_positions(candidates)
    assert candidates[0].pos == 1000


def test_a_cross_anchor_move_uses_a_non_emitting_candidate_only():
    """
    Tethering onto an EMITTING candidate would merge two calls by moving one
    onto the other -- de-duplication's job, not this stage's.
    """
    candidates = [C(pos=1000, emit_te=True, evidence_te=True),
                  C(pos=1500, emit_te=False, anchor_priority=0, anchor_support=4,
                    anchor_hypothesis_score=20.0)]
    S.retether_evidence_supported_final_call_positions(candidates)
    assert candidates[0].pos == 1500

    both_emitting = [C(pos=1000, emit_te=True, evidence_te=True),
                     C(pos=1500, emit_te=True, anchor_priority=0, anchor_support=4,
                       anchor_hypothesis_score=20.0)]
    S.retether_evidence_supported_final_call_positions(both_emitting)
    assert both_emitting[0].pos == 1000


def test_the_direction_constraint_stops_a_call_being_dragged_the_wrong_way():
    """
    A call that knows which side its junction is on sets `retether_direction`,
    and an anchor on the other side is then ineligible however good it scores.
    """
    leftward = [C(pos=2000, emit_te=True, evidence_te=True, retether_direction=-1),
                C(pos=2500, emit_te=False, anchor_priority=0, anchor_support=4,
                  anchor_hypothesis_score=20.0)]
    S.retether_evidence_supported_final_call_positions(leftward)
    assert leftward[0].pos == 2000

    rightward = [C(pos=2000, emit_te=True, evidence_te=True, retether_direction=1),
                 C(pos=2500, emit_te=False, anchor_priority=0, anchor_support=4,
                   anchor_hypothesis_score=20.0)]
    S.retether_evidence_supported_final_call_positions(rightward)
    assert rightward[0].pos == 2500


def test_the_anchor_score_ranks_centrality_above_precision():
    """
    The 2000 and 1000 bonuses act as a LEXICOGRAPHIC order rather than as
    weights: a central anchor beats any off-centre one, and among equally
    central ones a precise source wins.
    """
    candidate = C(pos=1000, component_anchor_pos=1500)
    central_imprecise = C(pos=1500, anchor_priority=5, anchor_hypothesis_score=1.0)
    distant_precise = C(pos=900, anchor_priority=0, anchor_hypothesis_score=1.0)
    assert (call_or_skip(S._retether_anchor_score, candidate, central_imprecise)
            > S._retether_anchor_score(candidate, distant_precise))

    # Among two EQUALLY central anchors, precision decides.
    also_central_precise = C(pos=1400, anchor_priority=0, anchor_hypothesis_score=1.0)
    assert (S._retether_anchor_score(candidate, also_central_precise)
            > S._retether_anchor_score(candidate, central_imprecise))


def test_the_cross_anchor_search_falls_back_to_the_calls_own_anchor():
    """
    WORTH PINNING, because the earlier paths read as though they were the only
    ways to move. They are not: the cross-anchor search starts with the
    candidate's OWN anchor as the incumbent tether, so a self anchor 100..3000 bp
    away moves the call even when paths 1-3 all declined. What the earlier paths
    buy is the `continue` that SKIPS this search, not the move itself.
    """
    unresolved = [C(pos=1000, anchor_pos=1300, emit_te=True, evidence_te=True,
                    anchor_support=4, anchor_hypothesis_score=12.0,
                    anchor_ref_span_reads=0, component_anchor_pos=1300)]
    S.retether_evidence_supported_final_call_positions(unresolved)
    assert unresolved[0].pos == 1300


def test_an_anchor_with_no_support_moves_nothing():
    """The one thing that stops the fallback: `anchor_support` below 1."""
    unsupported = [C(pos=1000, anchor_pos=1300, emit_te=True, evidence_te=True,
                     anchor_support=0, anchor_hypothesis_score=12.0)]
    S.retether_evidence_supported_final_call_positions(unsupported)
    assert unsupported[0].pos == 1000


# ------------------------------------------------------- the interval cache
def test_nearby_requests_merge_into_one_fetchable_interval():
    """The C++ case, exactly: two overlapping requests merge, a distant one does
    not, and each canonical interval remembers which requests it covers."""
    requests = [I.LocalIntervalRequest("chr1", 1000, 2100, 0),
                I.LocalIntervalRequest("chr1", 1500, 2600, 1),
                I.LocalIntervalRequest("chr1", 8000, 9000, 2)]
    intervals = call_or_skip(I.build_canonical_local_intervals, requests, 128)
    assert len(intervals) == 2
    assert (intervals[0].chrom, intervals[0].start, intervals[0].end) == ("chr1", 1000, 2600)
    assert intervals[0].request_ids == [0, 1]
    assert (intervals[1].start, intervals[1].end) == (8000, 9000)
    assert intervals[1].request_ids == [2]


def test_the_merge_gap_admits_a_gap_it_does_not_have_to_overlap():
    near = [I.LocalIntervalRequest("chr1", 1000, 2000, 0),
            I.LocalIntervalRequest("chr1", 2100, 3000, 1)]
    assert len(I.build_canonical_local_intervals(near, 128)) == 1
    far = [I.LocalIntervalRequest("chr1", 1000, 2000, 0),
           I.LocalIntervalRequest("chr1", 2200, 3000, 1)]
    assert len(I.build_canonical_local_intervals(far, 128)) == 2


def test_requests_on_different_contigs_never_merge():
    requests = [I.LocalIntervalRequest("chr1", 1000, 2000, 0),
                I.LocalIntervalRequest("chr2", 1000, 2000, 1)]
    assert len(I.build_canonical_local_intervals(requests, 100000)) == 2


def test_a_request_sees_only_the_reads_overlapping_its_own_interval():
    """
    THE correctness-critical half. A request that shared a fetch with a
    neighbour 3 kb away must not see the neighbour's reads -- its evidence
    counts would then include another event's support.
    """
    entry = I.LocalIntervalCacheEntry(
        interval=I.CanonicalLocalInterval("chr1", 1000, 6000, [0, 1]),
        records=[AlignedRead(qname="near"), AlignedRead(qname="far")],
        read_spans=[ReadReferenceSpan(True, 0, 1000, 2000),
                    ReadReferenceSpan(True, 0, 5000, 6000)])
    projection = call_or_skip(I.project_cached_interval_reads,
                              I.LocalIntervalRequest("chr1", 1000, 2100, 0), [entry])
    assert [r.qname for r in projection.records] == ["near"]


def test_a_read_with_no_valid_span_is_dropped():
    """It cannot be shown to overlap, and including it would put an unplaceable
    record into a positional analysis."""
    entry = I.LocalIntervalCacheEntry(
        interval=I.CanonicalLocalInterval("chr1", 1000, 2000, [0]),
        records=[AlignedRead(qname="unplaced")],
        read_spans=[ReadReferenceSpan(valid=False)])
    projection = I.project_cached_interval_reads(
        I.LocalIntervalRequest("chr1", 1000, 2000, 0), [entry])
    assert projection.records == []


def test_a_request_not_covered_by_any_entry_projects_nothing():
    entry = I.LocalIntervalCacheEntry(
        interval=I.CanonicalLocalInterval("chr1", 1000, 2000, [0]),
        records=[AlignedRead(qname="r")],
        read_spans=[ReadReferenceSpan(True, 0, 1000, 2000)])
    assert I.project_cached_interval_reads(
        I.LocalIntervalRequest("chr1", 1000, 2000, 99), [entry]).records == []
    assert I.project_cached_interval_reads(
        I.LocalIntervalRequest("chr2", 1000, 2000, 0), [entry]).records == []


def test_the_reuse_ratio_reports_one_when_nothing_was_fetched():
    """"No requests" is not infinitely efficient, it is simply not measured."""
    close(call_or_skip(I.local_interval_reuse_ratio, I.LocalIntervalReuseStats()),
          1.0, "nothing fetched")
    close(I.local_interval_reuse_ratio(I.LocalIntervalReuseStats(10, 4)), 2.5, "reuse")
