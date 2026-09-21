"""
Where exactly: hypothesis enumeration, the priority ladder, diverse selection.

The two `select_diverse_breakpoint_hypotheses` cases from
`tests/test_breakpoint_hypothesis_enumeration.cpp` are reproduced verbatim --
they are the only part of this stage the C++ can reach without a BAM, and both
turn on the anchor-proximal rescue.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer_py import breakpoints as B
from placer_py import clustering as C
from placer_py import fragments as F
from placer_py.alignment import CIGAR_I, CIGAR_M, CIGAR_S, AlignedRead

pytestmark = pytest.mark.invariant

M, I, S = CIGAR_M, CIGAR_I, CIGAR_S
H = B.LocalBreakpointHypothesis


def read(qname="r0", pos=1000, cigar=None, mapq=60, sa=None, tid=0):
    cigar = cigar or [(M, 500), (I, 300), (M, 500)]
    length = sum(n for op, n in cigar if op in (M, I, S))
    return AlignedRead(qname=qname, tid=tid, pos=pos, mapq=mapq, cigar=cigar,
                       seq="A" * length, tags=({"SA": sa} if sa else {}))


def fragment(ref_junc_pos, source, ref_side=F.ReferenceSide.UNKNOWN):
    return F.InsertionFragment(ref_junc_pos=ref_junc_pos, source=source,
                               ref_side=ref_side, sequence="ACGT", length=4)


def component(anchor=1500, tid=0, chrom="chr1", candidates=()):
    call = C.ComponentCall(chrom=chrom, tid=tid, anchor_pos=anchor)
    call.breakpoint_candidates = list(candidates)
    return call


# ------------------------------------------------------------ the ladder
def test_one_precise_read_outweighs_seven_clip_reads():
    """
    THE ratio of this stage, and it is correct rather than conservative. A clip
    says "the read stops explaining the reference near here", which is
    compatible with an insertion anywhere within the clip's own length; a split
    or a CIGAR insertion names a base.
    """
    precise = call_or_skip(B.hypothesis_score, H(True, 1000, 1000, 1000, 1, 0))
    seven_clips = B.hypothesis_score(H(True, 1000, 1020, 1010, 7, 5))
    assert precise == 8
    assert seven_clips == 7
    assert precise > seven_clips


def test_fragment_evidence_outranks_raw_evidence_of_the_same_kind():
    """
    8 against 7 for splits, 8 against 6 for insertions. The fragment stage
    already checked that the junction is consistent with an EXTRACTABLE insert,
    which the raw read signal has not.
    """
    assert B.breakpoint_hypothesis_support_weight(0) > B.breakpoint_hypothesis_support_weight(2)
    assert B.breakpoint_hypothesis_support_weight(1) > B.breakpoint_hypothesis_support_weight(4)


def test_both_clip_rows_weigh_one_so_clips_break_ties_rather_than_decide():
    assert B.breakpoint_hypothesis_support_weight(3) == 1
    assert B.breakpoint_hypothesis_support_weight(5) == 1
    assert B.breakpoint_hypothesis_support_weight(99) == 1


# ---------------------------------------------------------------- clusters
def test_positions_are_single_linked_at_a_seventy_five_base_gap():
    clusters = call_or_skip(B.collect_breakpoint_position_clusters,
                            [1000, 1030, 1074, 1200, 1210])
    assert len(clusters) == 2
    assert clusters[0].support == 3
    assert clusters[1].support == 2
    assert clusters[0].center == 1030


def test_a_gap_of_exactly_seventy_five_does_not_split():
    assert len(B.collect_breakpoint_position_clusters([1000, 1075])) == 1
    assert len(B.collect_breakpoint_position_clusters([1000, 1076])) == 2


def test_the_cluster_centre_is_a_position_some_read_proposed():
    clusters = B.collect_breakpoint_position_clusters([1000, 1001, 1050])
    assert clusters[0].center in (1000, 1001, 1050)


def test_the_mad_is_median_based_so_one_stray_read_cannot_inflate_it():
    """
    A single clip 5 kb away would dominate a standard deviation and make a
    well-agreed breakpoint look uncertain.
    """
    tight = [1000, 1001, 1000, 999, 1002]
    close(call_or_skip(B.mad_from_positions, tight), 1.0, "tight")
    close(B.mad_from_positions(tight + [6000]), 1.0, "with an outlier")
    close(B.mad_from_positions([]), 0.0, "empty")


# ------------------------------------------------------------- QC tokens
def test_qc_tokens_match_whole_tokens_and_never_substrings():
    """
    `PASS_TE` must not match inside `PASS_TE_CALIBRATED`; a substring test would
    make every token a prefix trap for every later one.
    """
    qc = call_or_skip(B.append_qc_token, "", "PASS_TE_CALIBRATED")
    assert B.has_qc_token(qc, "PASS_TE_CALIBRATED")
    assert not B.has_qc_token(qc, "PASS_TE")
    assert not B.has_qc_token(qc, "")


def test_appending_a_token_twice_leaves_one():
    qc = B.append_qc_token(B.append_qc_token("", "A"), "B")
    assert B.append_qc_token(qc, "A") == "A|B"
    assert B.append_qc_token(qc, "") == "A|B"


# --------------------------------------------------------- banded identity
def test_the_band_makes_the_alignment_linear_in_the_edit_budget():
    """
    Full edit distance is O(n*m) and this runs for every flank against every
    candidate placement. Restricting the DP to `|i - j| <= max_edits` makes it
    O(n * max_edits) -- ten columns instead of a hundred at 0.90 identity on a
    100 bp flank.
    """
    budget = call_or_skip(B.max_edits_for_identity_threshold, 100, 100, 0.90)
    assert budget == 10
    close(B.edit_identity_if_at_least("A" * 100, "A" * 100, budget), 1.0, "identical")


def test_a_sequence_outside_the_budget_returns_none_rather_than_a_low_number():
    """
    The caller is asking a YES/NO question with a threshold attached, and a
    number below the threshold was never computed exactly -- so returning one
    would be inventing precision.
    """
    assert B.edit_identity_if_at_least("A" * 100, "C" * 100, 10) is None
    assert B.edit_identity_if_at_least("A" * 100, "A" * 60, 10) is None
    assert B.edit_identity_if_at_least("", "ACGT", 10) is None
    assert B.edit_identity_if_at_least("ACGT", "ACGT", -1) is None


def test_a_length_difference_costs_identity_because_the_longer_side_is_the_denominator():
    """
    Comparing a 100 bp flank against a 60 bp one can never exceed 0.6 identity,
    however well the 60 bases match. That is what stops a short flank from
    looking like a perfect anchor.
    """
    close(B.fixed_window_edit_identity("A" * 60, "A" * 100), 0.6, "shorter query")
    close(B.max_edits_for_identity_threshold(0, 0, 0.9), 0, "no length")


def test_unbanded_identity_answers_regardless_of_threshold():
    close(call_or_skip(B.fixed_window_edit_identity, "ACGTACGT", "ACGTTCGT"),
          1.0 - 1.0 / 8.0, "one mismatch")
    close(B.fixed_window_edit_identity("", "ACGT"), 0.0, "empty")


# ------------------------------------------------------- per-read signals
def test_short_insertions_are_recorded_but_cannot_locate_a_breakpoint():
    """
    `max_raw_cigar_insert_len` is set for EVERY insertion in the window,
    including sub-threshold ones; `indel_pos` only above the floor. A short
    insertion says the locus carries inserted sequence without saying where the
    junction is, and finalization reads that field to decide whether a call has
    raw CIGAR support at all.
    """
    signal = call_or_skip(B.classify_local_event_signal,
                          read(cigar=[(M, 500), (I, 30), (M, 500)]), "chr1", 0, 3000)
    assert signal.max_raw_cigar_insert_len == 30
    assert not signal.indel
    assert signal.indel_pos == -1


def test_the_chosen_insertion_is_the_one_nearest_the_window_centre():
    """
    Not the longest and not the first. The window already decided where the
    event is; the read is being asked which of its insertions is that one.
    """
    cigar = [(M, 100), (I, 400), (M, 900), (I, 60), (M, 100)]
    signal = B.classify_local_event_signal(read(pos=1000, cigar=cigar), "chr1", 1900, 2100)
    assert signal.indel_pos == 2000        # the 60 bp one, not the 400 bp one


def test_signals_outside_the_window_are_ignored():
    signal = B.classify_local_event_signal(read(pos=1000), "chr1", 5000, 6000)
    assert not signal.any()


def test_every_split_position_is_kept_here_unlike_in_the_geometry_stage():
    """
    The geometry stage keeps only the LONGEST split because it is measuring
    support; this keeps them all because it is enumerating positions to
    consider.
    """
    split = read(pos=1000, cigar=[(M, 500), (S, 500)],
                 sa="chr1,1601,+,700S300M,60,1;chr1,3001,+,700S300M,60,1;")
    positions = call_or_skip(B.robust_local_split_insertion_positions, split, "chr1")
    assert len(positions) >= 1
    assert positions == sorted(set(positions))


def test_a_read_with_no_sa_tag_proposes_no_split_positions():
    assert B.robust_local_split_insertion_positions(read(), "chr1") == []


# ---------------------------------------------------- one read, one vote
def test_a_read_keeps_only_its_breakpoint_nearest_the_anchor():
    """
    One read can propose a clip at each end plus an insertion. Letting it
    contribute all three would make a single noisy read look like three
    supporting reads.
    """
    bp_by_read: dict[int, int] = {}
    call_or_skip(B.update_best_bp_by_read, bp_by_read, 0, 1000, 1500)
    B.update_best_bp_by_read(bp_by_read, 0, 1450, 1500)
    B.update_best_bp_by_read(bp_by_read, 0, 2000, 1500)
    assert bp_by_read == {0: 1450}


def test_a_tie_breaks_toward_the_smaller_position_so_order_does_not_matter():
    forward: dict[int, int] = {}
    B.update_best_bp_by_read(forward, 0, 1400, 1500)
    B.update_best_bp_by_read(forward, 0, 1600, 1500)
    backward: dict[int, int] = {}
    B.update_best_bp_by_read(backward, 0, 1600, 1500)
    B.update_best_bp_by_read(backward, 0, 1400, 1500)
    assert forward == backward == {0: 1400}


def test_a_negative_position_is_ignored():
    bp_by_read: dict[int, int] = {}
    B.update_best_bp_by_read(bp_by_read, 0, -1, 1500)
    assert bp_by_read == {}


# ------------------------------------------------------------ enumeration
def test_cigar_insertion_reads_produce_a_single_position_hypothesis():
    """
    A CIGAR insertion has no two sides: it names one reference position, with
    the inserted bases between it and the next base. `left == right` is the
    honest encoding, and the boundary stage reads the zero link delta as BLUNT.
    """
    reads = [read(f"r{i}", 1000) for i in range(4)]
    comp = component(anchor=1500)
    hypotheses = call_or_skip(B.collect_breakpoint_hypotheses, comp, reads, [],
                              1500, 1500, 5)
    assert len(hypotheses) == 1
    assert hypotheses[0].left == hypotheses[0].right == 1500
    assert hypotheses[0].priority == 4         # raw indel
    assert hypotheses[0].support == 4


def test_fragment_evidence_wins_the_ranking_against_more_numerous_raw_clips():
    reads = [read(f"clip{i}", 1000, cigar=[(M, 500), (S, 300)]) for i in range(8)]
    frags = [fragment(1500, F.InsertionFragmentSource.CIGAR_INSERTION)]
    hypotheses = B.collect_breakpoint_hypotheses(component(anchor=1500), reads, frags,
                                                 1500, 1500, 5)
    assert hypotheses[0].priority == 1         # fragment indel
    assert hypotheses[0].center == 1500


def test_the_search_widens_toward_precise_fragment_evidence():
    """
    A component whose median anchor is off by several hundred bases can still
    find its real breakpoint. The seed alone would only reach 200 bp.
    """
    frags = [fragment(2400, F.InsertionFragmentSource.CIGAR_INSERTION)]
    hypotheses = B.collect_breakpoint_hypotheses(component(anchor=1500), [], frags,
                                                 1500, 1500, 5)
    assert any(h.center == 2400 for h in hypotheses)


def test_a_precise_fragment_beyond_the_link_distance_cannot_drag_the_search():
    """The 1 kb link keeps a fragment belonging to a DIFFERENT event from
    pulling this component's search onto it."""
    frags = [fragment(9000, F.InsertionFragmentSource.CIGAR_INSERTION)]
    hypotheses = B.collect_breakpoint_hypotheses(component(anchor=1500), [], frags,
                                                 1500, 1500, 5)
    assert not any(h.center == 9000 for h in hypotheses)


def test_a_paired_hypothesis_wider_than_the_compatibility_span_is_dropped():
    frags = [fragment(1000, F.InsertionFragmentSource.SPLIT_SA, F.ReferenceSide.REF_LEFT),
             fragment(1600, F.InsertionFragmentSource.SPLIT_SA, F.ReferenceSide.REF_RIGHT)]
    hypotheses = B.collect_breakpoint_hypotheses(component(anchor=1300), [], frags,
                                                 1000, 1600, 5)
    assert not any(h.left == 1000 and h.right == 1600 for h in hypotheses)


def test_one_sided_evidence_still_yields_a_hypothesis():
    """
    A locus where every read clips on the same side still HAS a breakpoint -- it
    just has no junction geometry, and the boundary stage is where that costs it.
    """
    frags = [fragment(1500, F.InsertionFragmentSource.SPLIT_SA, F.ReferenceSide.REF_LEFT)]
    hypotheses = B.collect_breakpoint_hypotheses(component(anchor=1500), [], frags,
                                                 1500, 1500, 5)
    assert hypotheses
    assert hypotheses[0].left == hypotheses[0].right == 1500


def test_the_hypothesis_list_is_never_empty():
    """
    A component that reached this stage exists. Refusing to name a breakpoint
    for it would silently drop it, and the evidence stages are where it should
    be judged.
    """
    hypotheses = B.collect_breakpoint_hypotheses(component(anchor=1500), [], [],
                                                 1500, 1500, 5)
    assert len(hypotheses) == 1
    assert hypotheses[0].center == 1500
    assert hypotheses[0].support == 1


def test_duplicate_pairs_keep_the_priority_of_the_strongest_source():
    """
    De-duplication is on `(left, right)` and FIRST WINS, with batches appended
    in priority order -- so a pair found by fragment evidence keeps priority 1
    even when raw clips propose the same pair.
    """
    reads = [read(f"clip{i}", 1500, cigar=[(M, 500), (S, 300)]) for i in range(4)]
    frags = [fragment(1500, F.InsertionFragmentSource.CIGAR_INSERTION)]
    hypotheses = B.collect_breakpoint_hypotheses(component(anchor=1500), reads, frags,
                                                 1500, 1500, 5)
    at_1500 = [h for h in hypotheses if h.left == 1500 and h.right == 1500]
    assert len(at_1500) == 1
    assert at_1500[0].priority == 1


def test_bounds_come_from_the_candidates_or_fall_back_to_the_anchor():
    with_candidates = component(anchor=1500, candidates=[
        C.BreakpointCandidate(pos=1400), C.BreakpointCandidate(pos=1600),
        C.BreakpointCandidate(pos=-1)])
    assert call_or_skip(B.infer_component_breakpoint_bounds, with_candidates) == (1400, 1600)
    assert B.infer_component_breakpoint_bounds(component(anchor=1500)) == (1500, 1500)


# ------------------------------------------------------- diverse selection
def test_near_duplicate_hypotheses_do_not_each_take_a_slot():
    """
    The C++ case. The ranked list at a real locus is mostly variants of one
    junction -- clips at 1000/1020, fragments at 1003/1029 -- and taking the top
    three by score would spend every slot on it. A hypothesis not selected here
    is never evaluated at all.
    """
    selected = call_or_skip(B.select_diverse_breakpoint_hypotheses, [
        H(True, 1000, 1020, 1010, 39, 5),
        H(True, 1003, 1029, 1016, 32, 3),
        H(True, 1300, 1400, 1350, 19, 3),
        H(True, 1100, 1100, 1100, 2, 1)], 3, 1100)
    assert len(selected) == 3
    pairs = {(h.left, h.right) for h in selected}
    assert (1300, 1400) in pairs
    assert (1100, 1100) in pairs
    assert (1000, 1020) in pairs or (1003, 1029) in pairs


def test_the_anchor_proximal_rescue_replaces_the_weakest_incumbent():
    """
    The second C++ case. Nothing among the top three is within 100 bp of the
    anchor at 2000, so the best hypothesis that IS displaces the weakest --
    the anchor is where the geometry stage said the event is, and a list that
    ignores it is usually chasing a nearby repeat.
    """
    selected = B.select_diverse_breakpoint_hypotheses([
        H(True, 1000, 1000, 1000, 6, 1),
        H(True, 1400, 1400, 1400, 5, 1),
        H(True, 1700, 1700, 1700, 4, 1),
        H(True, 2000, 2000, 2000, 2, 4)], 3, 2000)
    assert len(selected) == 3
    pairs = {(h.left, h.right) for h in selected}
    assert pairs == {(1000, 1000), (1400, 1400), (2000, 2000)}


def test_a_single_precise_read_far_from_everything_is_rescued_beyond_top_k():
    """
    The only place the list may EXCEED `top_k`, and deliberately: one
    split-or-insertion read names a base, and the clip cloud that outvoted it
    does not. The 500 bp distance makes the rescue add a LOCUS rather than a
    variant of one already selected.
    """
    selected = B.select_diverse_breakpoint_hypotheses([
        H(True, 1000, 1020, 1010, 39, 5),
        H(True, 1100, 1120, 1110, 30, 5),
        H(True, 1200, 1220, 1210, 20, 5),
        H(True, 5000, 5000, 5000, 1, 0)], 3, 1010)
    assert len(selected) == 4
    assert any(h.left == 5000 for h in selected)


def test_a_precise_read_close_to_a_selection_is_not_rescued():
    selected = B.select_diverse_breakpoint_hypotheses([
        H(True, 1000, 1020, 1010, 39, 5),
        H(True, 1100, 1120, 1110, 30, 5),
        H(True, 1200, 1220, 1210, 20, 5),
        H(True, 1300, 1300, 1300, 1, 0)], 3, 1010)
    assert len(selected) == 3


def test_a_list_no_longer_than_top_k_is_returned_untouched():
    hypotheses = [H(True, 1000, 1000, 1000, 6, 1), H(True, 1400, 1400, 1400, 5, 1)]
    assert B.select_diverse_breakpoint_hypotheses(hypotheses, 3, 1000) == hypotheses
    assert B.select_diverse_breakpoint_hypotheses(hypotheses, 0, 1000) == hypotheses


def test_resolving_bounds_takes_the_single_best_hypothesis():
    reads = [read(f"r{i}", 1000) for i in range(4)]
    assert call_or_skip(B.resolve_event_breakpoint_bounds, component(anchor=1500),
                        reads, [], 1500, 1500) == (1500, 1500)


# ------------------------------------------- the banded DP, against a spec


def _textbook_banded_levenshtein(lhs: str, rhs: str, max_edits: int):
    """The obvious implementation of what `edit_identity_if_at_least` computes.

    Deliberately written for clarity, not speed: a fresh full-width row per
    iteration, `min` of the three moves, no early exit. This is the
    specification the optimised version in `placer_py/breakpoints.py` must
    agree with EXACTLY.
    """
    n, m = len(lhs), len(rhs)
    if n <= 0 or m <= 0 or max_edits < 0:
        return None
    if abs(n - m) > max_edits:
        return None
    infinity = max_edits + 1
    previous = [infinity] * (m + 1)
    for j in range(min(m, max_edits) + 1):
        previous[j] = j
    for i in range(1, n + 1):
        current = [infinity] * (m + 1)
        if i <= max_edits:
            current[0] = i
        j_lo, j_hi = max(1, i - max_edits), min(m, i + max_edits)
        if j_lo > j_hi:
            return None
        char = lhs[i - 1]
        for j in range(j_lo, j_hi + 1):
            current[j] = min(previous[j - 1] + (0 if char == rhs[j - 1] else 1),
                             previous[j] + 1,
                             current[j - 1] + 1)
        previous = current
    distance = previous[m]
    if distance > max_edits:
        return None
    return min(1.0, max(0.0, 1.0 - (distance / max(n, m))))


def test_the_optimised_dp_agrees_with_the_textbook_one():
    """A differential test, because this function is 69% of a real run.

    Measured on a 200 kb slice of ultra-long ONT: 821,575 calls, 214 s of
    461 s in its own frame plus 101 s in 1.27e9 calls to `min`. That made it
    worth rewriting -- two reused buffers instead of a row allocation per i,
    comparisons instead of `min`, rolling locals instead of list indexing, and
    an early exit once a row's minimum passes the budget.

    Every one of those is supposed to be exact, and none of them is obviously
    exact, which is what this test is for. The buffer reuse in particular is
    only safe because of which cells each row reads; get the `curr[j_hi+1]`
    reset wrong and a stale value from two rows back leaks into the band,
    which would show up here as a wrong distance rather than a crash.
    """
    import random

    rng = random.Random(4242)
    checked = 0
    for _ in range(600):
        lhs = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 40)))
        if rng.random() < 0.3:
            rhs = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 40)))
        else:
            edited = list(lhs)
            for _ in range(rng.randint(0, 6)):
                at = rng.randrange(len(edited))
                roll = rng.random()
                if roll < 0.5:
                    edited[at] = rng.choice("ACGT")
                elif roll < 0.75:
                    edited.insert(at, rng.choice("ACGT"))
                elif len(edited) > 1:
                    edited.pop(at)
            rhs = "".join(edited)
        for max_edits in (0, 1, 2, 3, 5, 12, max(len(lhs), len(rhs))):
            actual = call_or_skip(B.edit_identity_if_at_least, lhs, rhs, max_edits)
            expected = _textbook_banded_levenshtein(lhs, rhs, max_edits)
            assert actual == expected, (lhs, rhs, max_edits, actual, expected)
            checked += 1
    assert checked > 4000, checked


def test_the_early_exit_cannot_reject_a_match_that_exists():
    """The row minimum is non-decreasing, so the early exit is sound.

    Pinned separately from the differential test because it is the one change
    that could silently cost RECALL rather than produce a wrong number: a
    rejection is reported as None, which the caller reads as "this placement
    does not match", and a caller cannot tell that apart from a real no.
    """
    identical = "ACGTACGTACGTACGTACGTAAGGCCTT"
    assert call_or_skip(B.edit_identity_if_at_least, identical, identical, 0) == 1.0
    # A single substitution at the very END, so every earlier row is clean and
    # the budget is only consumed on the last one.
    tail_mismatch = identical[:-1] + ("A" if identical[-1] != "A" else "C")
    assert call_or_skip(B.edit_identity_if_at_least, identical, tail_mismatch, 1) is not None
    assert call_or_skip(B.edit_identity_if_at_least, identical, tail_mismatch, 0) is None
