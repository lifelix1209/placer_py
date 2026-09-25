"""
Read gating and CIGAR summarisation.

Tested against the decision table, case by case, with each case naming the
fuse it exercises.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip

from placer import reads as R

pytestmark = pytest.mark.invariant

M, I, D, S, H, EQ, X = (R.CIGAR_M, R.CIGAR_I, R.CIGAR_D, R.CIGAR_S,
                        R.CIGAR_H, R.CIGAR_EQ, R.CIGAR_X)


def _gate(cigar, flag=0, seq_len=5000, mapq=60, has_sa=False, nm=None):
    return call_or_skip(R.pass_preliminary, cigar, flag, seq_len, mapq,
                        has_sa, nm)


# ------------------------------------------------------- CIGAR summarisation
def test_max_match_block_resets_on_any_non_match():
    """
    The longest UNINTERRUPTED anchor, not the total. 200 matched bases split
    into twenty blocks of 10 is not an anchor, and the two fields have to
    distinguish that -- fuse 1 reads the block, not the total.
    """
    fragmented = [(M, 10), (D, 1)] * 20
    summary = call_or_skip(R.summarize_cigar, fragmented)
    assert summary.total_match_bases == 200
    assert summary.max_match_block == 10


def test_equal_and_mismatch_ops_count_as_anchor():
    """=, X and M all consume both reference and query, so all three anchor."""
    for op in (M, EQ, X):
        summary = call_or_skip(R.summarize_cigar, [(op, 300)])
        assert summary.max_match_block == 300, op
        assert summary.total_match_bases == 300, op


def test_clip_adjacent_anchors_stop_at_the_first_non_match():
    """
    The flank measured beside a clip is the CONTIGUOUS match run, so an
    intervening deletion truncates it. That is the point: a clip separated from
    its anchor by an indel is not anchored to that position.
    """
    summary = call_or_skip(R.summarize_cigar,
                           [(S, 300), (M, 150), (D, 5), (M, 400), (S, 200)])
    assert summary.leading_soft_clip == 300
    assert summary.right_anchor_after_leading == 150, "stops at the D"
    assert summary.trailing_soft_clip == 200
    assert summary.left_anchor_before_trailing == 400


def test_hard_clips_are_skipped_when_locating_the_ends():
    summary = call_or_skip(R.summarize_cigar,
                           [(H, 50), (S, 300), (M, 500), (S, 200), (H, 40)])
    assert summary.leading_soft_clip == 300
    assert summary.trailing_soft_clip == 200
    assert summary.right_anchor_after_leading == 500


def test_an_empty_cigar_summarises_to_zeros():
    summary = call_or_skip(R.summarize_cigar, [])
    assert summary.total_match_bases == 0
    assert summary.max_soft_clip == 0


# --------------------------------------------------------------- hard drops
def test_unmapped_and_secondary_reads_are_dropped_before_anything_else():
    strong = [(M, 5000), (I, 500), (M, 5000)]
    assert _gate(strong) is True
    assert _gate(strong, flag=R.FLAG_UNMAP) is False
    assert _gate(strong, flag=R.FLAG_SECONDARY) is False


def test_short_reads_are_dropped():
    assert _gate([(M, 5000)], seq_len=49) is False
    assert _gate([(M, 5000)], seq_len=50) is True


# ----------------------------------------------------------- the signal test
def test_a_read_with_no_signal_is_kept_only_as_background_coverage():
    """And the MAPQ comparison is STRICT, so a read exactly at the floor goes."""
    plain = [(M, 5000)]
    assert _gate(plain, mapq=21) is True
    assert _gate(plain, mapq=20) is False, "the floor is exclusive"
    assert _gate(plain, mapq=0) is False


def test_each_of_the_four_signals_is_sufficient_on_its_own():
    anchor = [(M, 5000)]
    assert _gate(anchor, flag=R.FLAG_SUPPLEMENTARY, mapq=0) is True
    assert _gate(anchor, has_sa=True, mapq=0) is True
    assert _gate([(S, 100), (M, 5000)], mapq=0) is True
    assert _gate([(M, 2500), (I, 50), (M, 2500)], mapq=0) is True


# -------------------------------------------------------------- the fuses
def test_fuse_one_requires_a_solid_anchor():
    """A signal without a 200 bp uninterrupted anchor cannot locate anything."""
    fragmented = [(S, 300)] + [(M, 100), (D, 1)] * 30
    assert _gate(fragmented) is False
    assert _gate([(S, 300), (M, 200), (M, 0)] if False else
                 [(S, 300), (M, 200)]) is True


def test_fuse_two_vetoes_a_long_clip_beside_a_short_flank():
    """A long clip next to a short anchor is a chimera, not a breakpoint."""
    assert _gate([(S, 300), (M, 250), (D, 2), (M, 3000)]) is True
    assert _gate([(S, 300), (M, 119), (D, 2), (M, 3000)]) is False
    # The trailing end is checked the same way.
    assert _gate([(M, 3000), (D, 2), (M, 119), (S, 300)]) is False


def test_fuse_two_is_skipped_when_a_long_insertion_is_also_present():
    """
    Deliberate asymmetry: direct insertion evidence is breakpoint-specific, so
    an unrelated clip on the same noisy alignment must not veto it.
    """
    clip_only = [(S, 300), (M, 119), (D, 2), (M, 3000)]
    with_insertion = [(S, 300), (M, 119), (I, 80), (M, 3000)]
    assert _gate(clip_only) is False
    assert _gate(with_insertion) is True


def test_fuse_three_rejects_noisy_alignments_unless_an_insertion_is_present():
    """Same asymmetry, for the NM rate."""
    clip_signal = [(S, 300), (M, 3000)]
    assert _gate(clip_signal, nm=100) is True          # 3.3%
    assert _gate(clip_signal, nm=700) is False         # 23%
    insertion_signal = [(M, 1500), (I, 80), (M, 1500)]
    assert _gate(insertion_signal, nm=700) is True, (
        "a long insertion survives a global mismatch burden")


def test_a_missing_nm_tag_does_not_reject():
    assert _gate([(S, 300), (M, 3000)], nm=None) is True


# ------------------------------------------------------ split-read arithmetic
def test_insertion_length_is_the_query_gap_less_the_reference_gap():
    # 500 bp of unaligned query, 0 bp of reference skipped -> 500 bp insert.
    assert call_or_skip(R.insertion_length_from_split, 1000, 5000, 1500,
                        5000) == 500
    # 500 bp of query, 200 bp of reference skipped -> 300 bp insert.
    assert call_or_skip(R.insertion_length_from_split, 1000, 5000, 1500,
                        5200) == 300


def test_overlapping_segments_do_not_inflate_the_insertion():
    """
    A negative reference gap is clamped to zero. Without the clamp an overlap
    would be treated as a negative requirement and ADD to the insertion length,
    which is the wrong direction.
    """
    assert call_or_skip(R.insertion_length_from_split, 1000, 5000, 1500,
                        4900) == 500
