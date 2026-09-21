"""
The tripartite decode: flank | insert | flank, placed on the reference.

The most consequential stage in the pipeline -- `insert_seq` is what the TE
library, the poly(A) model, the TSD detector and the structure decode all read,
and `left_ref_end` / `right_ref_start` are what the boundary stage subtracts.

The C++ tests for this stage (`test_event_segmentation.cpp`,
`test_pre_segmentation_gate.cpp`) go through a `Pipeline` holding a reference
handle. Here the reference is a callback, so the whole stage runs on a string.
"""

from __future__ import annotations

import random

import pytest
from conftest import call_or_skip, close

from placer_py import segmentation as S
from placer_py.config import PipelineConfig

pytestmark = pytest.mark.invariant


def random_reference(length=4000, seed=7):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


REFERENCE = random_reference()
INSERT = "TTTTGGGGCCCCAAAA" * 20      # 320 bp, unlike anything in the reference


def fetch(chrom, start, end):
    return REFERENCE[max(0, start):max(0, end)]


def consensus(seq, **kw):
    out = S.EventConsensus(consensus_seq=seq, qc_pass=True, input_event_reads=6,
                           left_anchor_input_reads=3, right_anchor_input_reads=3,
                           partial_context_input_reads=5, full_context_input_reads=4)
    for key, value in kw.items():
        setattr(out, key, value)
    out.consensus_len = len(out.consensus_seq)
    return out


def segment(seq, bp_left=1000, bp_right=1000, alt=8, ref=4, config=None, **kw):
    return S.segment_event_consensus("chr1", bp_left, bp_right, alt, ref,
                                     consensus(seq, **kw), config or PipelineConfig(),
                                     fetch)


# ----------------------------------------------------------- the happy path
def test_a_clean_insertion_recovers_the_insert_and_both_junctions():
    """
    The whole stage in one case: 100 bp of reference, 320 bp that is not in the
    reference, 100 bp of reference. Both flanks place at identity 1.0, the
    insert comes back exactly, and the junction is blunt.
    """
    result = call_or_skip(segment, REFERENCE[900:1000] + INSERT + REFERENCE[1000:1100])
    assert result.pass_
    assert result.qc_reason == "PASS_EVENT_SEGMENTATION"
    assert result.insert_seq == INSERT
    assert (result.left_ref_start, result.left_ref_end) == (900, 1000)
    assert (result.right_ref_start, result.right_ref_end) == (1000, 1100)
    close(result.left_flank_identity, 1.0, "left identity")
    close(result.right_flank_identity, 1.0, "right identity")


def test_segmentation_does_not_discover_a_tsd_it_follows_the_given_breakpoints():
    """
    WORTH KNOWING, because the name `left_ref_end`/`right_ref_start` invites the
    opposite assumption. Given a consensus that genuinely carries a 12 bp target
    site duplication, the segmenter does NOT report overlapping flanks: the
    comparator ranks breakpoint proximity above flank length, so it places each
    flank as close as it can to the breakpoints it was handed, and the
    duplicated bases end up INSIDE the insert.

    So a TSD reaches the ledger only when the breakpoint stage already proposed
    two separated positions -- or from `placer_py/tsd.py`, which reads the
    reference around the breakpoints directly and is the component actually
    responsible for finding one.
    """
    tsd = REFERENCE[988:1000]
    duplicated = REFERENCE[900:1000] + INSERT + tsd + REFERENCE[1000:1088]

    blunt = segment(duplicated, bp_left=1000, bp_right=1000)
    assert blunt.pass_
    assert blunt.left_ref_end == blunt.right_ref_start == 1000
    assert len(blunt.insert_seq) == len(INSERT) + 12

    separated = segment(duplicated, bp_left=988, bp_right=1000)
    assert separated.pass_
    assert (separated.left_ref_end, separated.right_ref_start) == (988, 1000)


def test_the_breakpoint_order_does_not_change_the_answer():
    """`min`/`max` are applied on entry, so a caller that reports the pair the
    other way round gets the same segmentation."""
    duplicated = REFERENCE[900:1000] + INSERT + REFERENCE[988:1000] + REFERENCE[1000:1088]
    forward = segment(duplicated, bp_left=988, bp_right=1000)
    reversed_ = segment(duplicated, bp_left=1000, bp_right=988)
    assert (forward.left_ref_end, forward.right_ref_start) == (
        reversed_.left_ref_end, reversed_.right_ref_start)


def test_a_noisy_consensus_still_places_its_flanks():
    """0.90 identity is low for a flank against its own locus, and deliberately
    so: it has to admit a long-read consensus with real errors in it."""
    flank = list(REFERENCE[900:1000])
    for i in range(0, 100, 15):
        flank[i] = "ACGT"[("ACGT".index(flank[i]) + 1) % 4]
    result = segment("".join(flank) + INSERT + REFERENCE[1000:1100])
    assert result.pass_
    assert result.left_flank_identity < 1.0


def test_the_insert_is_what_is_left_over_and_not_what_failed_to_align():
    """
    Pinned because it is the difference between this and a local aligner. The
    insert is defined by the two flank PLACEMENTS, so it is exactly the
    consensus between them -- including any part of it that would happen to
    align somewhere in the window.
    """
    repeat_like = REFERENCE[2000:2100]
    result = segment(REFERENCE[900:1000] + repeat_like + INSERT + REFERENCE[1000:1100])
    assert result.pass_
    assert result.insert_seq == repeat_like + INSERT


# --------------------------------------------------------------- the refusals
def test_a_consensus_that_did_not_pass_is_not_segmented():
    result = segment(REFERENCE[900:1000] + INSERT + REFERENCE[1000:1100], qc_pass=False)
    assert not result.pass_
    assert result.qc_reason == "NO_EVENT_CONSENSUS_TO_SEGMENT"


def test_a_consensus_too_short_to_hold_a_flank_and_an_insert_is_refused():
    result = segment("ACGT" * 12)
    assert result.qc_reason == "EVENT_CONSENSUS_TOO_SHORT"


def test_invalid_breakpoints_are_refused_rather_than_clamped():
    result = segment(REFERENCE[900:1000] + INSERT + REFERENCE[1000:1100], bp_left=-1)
    assert result.qc_reason == "INVALID_EVENT_BREAKPOINTS"


def test_a_reference_window_that_cannot_be_fetched_is_reported_as_such():
    result = S.segment_event_consensus(
        "chr1", 1000, 1000, 8, 4,
        consensus(REFERENCE[900:1000] + INSERT + REFERENCE[1000:1100]),
        PipelineConfig(), lambda chrom, start, end: "")
    assert result.qc_reason == "REFERENCE_WINDOW_FETCH_FAILED"


def test_a_consensus_that_matches_nothing_reaches_the_no_segmentation_verdict():
    result = segment("TTTTGGGGCCCCAAAA" * 40, alt=1, ref=10)
    assert not result.pass_
    assert result.qc_reason == "NO_TRIPARTITE_EVENT_SEGMENTATION"


# -------------------------------------------------------------- one-sided
def test_one_flank_alone_still_yields_an_insert_but_no_junction():
    """
    THE case `pair_valid` exists for. A one-sided segmentation produces an
    insert sequence AND collapses the unanchored side onto the read-derived
    breakpoint, so `right_ref_start == right_ref_end`. The boundary stage must
    read that as INVALID rather than as a blunt join -- which is why the
    decision policy carries `pair_valid` separately from `has_insert_seq`.
    """
    result = segment(REFERENCE[900:1000] + INSERT + "TTTTGGGGCCCCAAAA" * 8)
    assert result.pass_
    assert result.qc_reason == "PASS_EVENT_SEGMENTATION_ONE_SIDED_LEFT"
    assert result.right_ref_start == result.right_ref_end
    assert result.right_flank_align_len == 0
    assert result.insert_seq.startswith(INSERT[:16])


def test_a_right_only_consensus_takes_the_mirror_path():
    result = segment("TTTTGGGGCCCCAAAA" * 8 + INSERT + REFERENCE[1000:1100])
    assert result.pass_
    assert result.qc_reason == "PASS_EVENT_SEGMENTATION_ONE_SIDED_RIGHT"
    assert result.left_ref_start == result.left_ref_end
    assert result.left_flank_align_len == 0


# ------------------------------------------------------- reverse complement
def test_a_consensus_assembled_the_wrong_way_round_is_retried_reversed():
    """
    The consensus orientation comes from whichever reads the assembler started
    from. A consensus built backwards fails every placement for a reason that
    has nothing to do with the locus, so the whole search is retried on the
    reverse complement -- once, guarded so it cannot recurse.
    """
    from placer_py.seqtools import reverse_complement

    forward = REFERENCE[900:1000] + INSERT + REFERENCE[1000:1100]
    stats = S.SegmentationSearchStats()
    result = S.segment_event_consensus("chr1", 1000, 1000, 8, 4,
                                       consensus(reverse_complement(forward)),
                                       PipelineConfig(), fetch, stats)
    assert result.pass_
    assert stats.reverse_complement_retries == 1


def test_the_reverse_retry_cannot_recurse():
    result = segment("TTTTGGGGCCCCAAAA" * 40, alt=1, ref=10,
                     qc_reason="PASS_EVENT_CONSENSUS_REVCOMP_RETRY")
    assert result.qc_reason == "NO_TRIPARTITE_EVENT_SEGMENTATION"


# ----------------------------------------------------------- unplaced insert
def test_the_unplaced_fallback_needs_an_unopposed_and_heavily_anchored_locus():
    """
    The last resort, and it asserts an insertion WITHOUT having placed either
    junction -- so it is gated on the locus being unopposed (no reference-
    spanning read), heavily supported (8+ alt reads, at least twice what the
    consensus used), and on the consensus itself having two anchors per side
    and four partial contexts. That combination is what a long insertion into a
    segmental duplication looks like.
    """
    unplaceable = "TTTTGGGGCCCCAAAA" * 40
    result = segment(unplaceable, alt=20, ref=0, input_event_reads=6,
                     partial_context_input_reads=5)
    assert result.pass_
    assert result.qc_reason == "PASS_EVENT_SEGMENTATION_UNPLACED_INSERT"
    assert result.insert_seq == unplaceable
    assert result.left_ref_start == result.left_ref_end


def test_one_reference_spanning_read_closes_the_unplaced_fallback():
    unplaceable = "TTTTGGGGCCCCAAAA" * 40
    assert segment(unplaceable, alt=20, ref=1).qc_reason == "NO_TRIPARTITE_EVENT_SEGMENTATION"


def test_a_weakly_anchored_consensus_closes_the_unplaced_fallback():
    unplaceable = "TTTTGGGGCCCCAAAA" * 40
    assert segment(unplaceable, alt=20, ref=0,
                   partial_context_input_reads=3).qc_reason == "NO_TRIPARTITE_EVENT_SEGMENTATION"


# ------------------------------------------------------------- the seeding
def test_seed_bins_rank_by_support_so_a_two_kmer_repeat_cannot_outrank_the_locus():
    """
    Binning k-mer hits by the DIAGONAL they imply is what makes the search
    linear rather than quadratic in the window. Ranking by support first means a
    repeat elsewhere in the window with two matching k-mers cannot outrank the
    real locus with forty.
    """
    window = REFERENCE[900:1200]
    bins = call_or_skip(S.collect_anchor_seed_bins, REFERENCE[900:1000], 1000, 900,
                        window, True)
    assert bins
    assert bins[0].support >= 40
    assert bins[0].ref_bin_start in range(896, 904)
    assert len(bins) <= S.SEED_TOP_BINS


def test_a_query_sharing_no_kmer_with_the_window_produces_no_bins():
    assert S.collect_anchor_seed_bins("A" * 100, 1000, 900, REFERENCE[900:1200], True) == []


# ----------------------------------------------------------- the comparators
def test_a_pair_is_ranked_by_its_worse_flank_first():
    """
    A pair with one perfect and one marginal flank is worse than a pair with two
    good ones -- the MARGINAL flank is the one that decides where the junction
    is. Ranking by the mean alone would let 1.00/0.90 beat 0.96/0.96.
    """
    P = S.EventFlankPlacement
    lopsided = (P(identity=1.00, align_len=100), P(identity=0.90, align_len=100))
    balanced = (P(identity=0.96, align_len=100), P(identity=0.96, align_len=100))
    assert S._Segmenter.pair_key(*balanced) < S._Segmenter.pair_key(*lopsided)


def test_a_single_placement_is_ranked_by_identity_before_breakpoint_proximity():
    """
    The expected breakpoint came from clustering read signals and is itself
    uncertain; identity is measured. So a better-matching placement wins even
    when it sits further from the expectation.
    """
    P = S.EventFlankPlacement
    accurate = P(identity=0.99, breakpoint_delta=180, align_len=100)
    close_by = P(identity=0.94, breakpoint_delta=2, align_len=100)
    assert S.better_event_flank_placement(accurate) < S.better_event_flank_placement(close_by)


# ------------------------------------------------------- the pre-segmentation gate
def test_the_gate_requires_reads_anchoring_both_sides():
    """Definitional, not a threshold: without both sides anchored there is no
    junction to find."""
    one_sided = S.EventConsensus(left_anchor_input_reads=4, right_anchor_input_reads=0)
    assert call_or_skip(S.pre_segmentation_gate_reason, 2, 2,
                        one_sided) == "PRESEG_NO_BILATERAL_ANCHOR"


def test_the_gate_accepts_a_bilateral_partial_context_without_a_spanning_read():
    """
    A long insertion may have NO read spanning it, and requiring one would make
    event LENGTH decide detectability -- the same bias the README criticises in
    the high-precision tier.
    """
    partial = S.EventConsensus(left_anchor_input_reads=2, right_anchor_input_reads=2,
                               partial_context_input_reads=2, input_event_reads=2,
                               full_context_input_reads=0)
    assert S.pre_segmentation_gate_reason(0, 0, partial) == ""
    thin = S.EventConsensus(left_anchor_input_reads=2, right_anchor_input_reads=2,
                            partial_context_input_reads=1, input_event_reads=2)
    assert S.pre_segmentation_gate_reason(0, 0, thin) == "PRESEG_NO_PRECISE_OR_FULL_CONTEXT"


def test_one_precise_read_is_enough_on_its_own():
    anchored = S.EventConsensus(left_anchor_input_reads=1, right_anchor_input_reads=1)
    assert S.pre_segmentation_gate_reason(1, 0, anchored) == ""


# ---------------------------------------------------------------- the stats
def test_the_search_counters_are_filled_in():
    """Diagnostic only, but they are how a pathological locus is recognised: a
    run where `edit_distance_calls` explodes is a repeat, not a bug."""
    stats = S.SegmentationSearchStats()
    S.segment_event_consensus("chr1", 1000, 1000, 8, 4,
                              consensus(REFERENCE[900:1000] + INSERT + REFERENCE[1000:1100]),
                              PipelineConfig(), fetch, stats)
    assert stats.paired_searches >= 2
    assert stats.seed_bins_total > 0
    assert stats.edit_distance_calls > 0
    assert stats.edit_distance_cache_misses == stats.edit_distance_calls
