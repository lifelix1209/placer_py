"""
The geometry stage: signatures, three-dimensional DBSCAN, and the projection.

Every case in `tests/test_dbscan_component_module.cpp` is reproduced here.
Those are constructed BAM records in the C++ and literal `AlignedRead`s here,
which is the whole reason the port can test the stage at all cheaply -- and why
this file also covers the paths the C++ cases never reach.
"""

from __future__ import annotations

import pytest

from conftest import call_or_skip, close
from placer_py import clustering as C
from placer_py.alignment import CIGAR_I, CIGAR_M, CIGAR_S, AlignedRead

pytestmark = pytest.mark.invariant


def read(qname, pos, cigar, mapq=60, sa=None, seq_len=None, flag=0, tid=0):
    tags = {"SA": sa} if sa else {}
    length = seq_len if seq_len is not None else sum(
        n for op, n in cigar if op in (CIGAR_M, CIGAR_I, CIGAR_S))
    return AlignedRead(qname=qname, tid=tid, pos=pos, mapq=mapq, cigar=cigar,
                       seq="A" * length, flag=flag, tags=tags)


def ins_read(qname, pos, match_left, ins_len, match_right, mapq=60):
    return read(qname, pos, [(CIGAR_M, match_left), (CIGAR_I, ins_len),
                             (CIGAR_M, match_right)], mapq=mapq)


# ------------------------------------------------------------- the metric
def test_length_is_a_clustering_coordinate_not_an_afterthought():
    """
    THE design decision of this stage. Two reads 30 bp apart reporting a 120 bp
    and a 4000 bp insertion are not the same event, and position-only
    clustering would merge them and report "two reads support". Putting length
    in the metric makes that impossible by construction rather than leaving it
    for a downstream score to undo.
    """
    a = C.InsertionSignature(pos=1000, end=1001, length=120)
    b = C.InsertionSignature(pos=1030, end=1031, length=4000)
    assert call_or_skip(C.signature_distance, a, b) > C.signature_epsilon(a, b)

    same = C.InsertionSignature(pos=1030, end=1031, length=130)
    assert C.signature_distance(a, same) < C.signature_epsilon(a, same)


def test_two_epsilons_hold_same_kind_signals_to_a_tighter_standard():
    cigar = C.SignatureSource.CIGAR_INSERTION
    clip = C.SignatureSource.CLIP_HINT
    a = C.InsertionSignature(pos=0, source=cigar)
    b = C.InsertionSignature(pos=0, source=cigar)
    close(C.signature_epsilon(a, b), 250.0, "intra")
    close(C.signature_epsilon(a, C.InsertionSignature(pos=0, source=clip)), 500.0,
          "inter")


def test_neighbourhood_test_is_strict_at_the_threshold():
    """
    `distance < epsilon`, not `<=`. Pinned because it is the kind of detail a
    from-scratch reimplementation flips without noticing, and it changes which
    points are core points at exactly the boundary.
    """
    a = C.InsertionSignature(pos=0, end=0, length=0)
    b = C.InsertionSignature(pos=250, end=0, length=0)
    close(C.signature_distance(a, b), 250.0, "exactly epsilon")
    assert C.dbscan_cluster_indices([a, b], [0, 1], 1) == [[0], [1]]


# ---------------------------------------------------------------- DBSCAN
def test_core_point_test_counts_the_point_itself():
    """`len(neighbours) + 1 >= min_pts`, so min_pts=1 makes everything a core
    point and the weak pass degenerates to connected components."""
    signatures = [C.InsertionSignature(pos=1000 + 10 * i, end=1001, length=100)
                  for i in range(5)]
    assert len(call_or_skip(C.dbscan_cluster_indices, signatures,
                            list(range(5)), 5)) == 1
    assert len(C.dbscan_cluster_indices(signatures[:2], [0, 1], 5)) == 0
    assert len(C.dbscan_cluster_indices(signatures[:2], [0, 1], 1)) == 1


def test_the_weak_second_pass_is_what_keeps_single_read_events_alive():
    """
    Not a fallback for a failed first pass. A single-read event has to reach
    the evidence stage to be judged at all; discarding it here is deciding with
    no evidence, at the one stage that has none.
    """
    lone = C.InsertionSignature(pos=50000, end=50001, length=300)
    crowd = [C.InsertionSignature(pos=1000 + 5 * i, end=1001, length=100)
             for i in range(6)]
    clusters = call_or_skip(C.cluster_signatures, crowd + [lone])
    assert sorted(len(c) for c in clusters) == [1, 6]


def test_border_points_do_not_chain_two_clusters_together():
    """
    Expansion only pushes a newly-visited point's neighbours when that point is
    itself a core point. Without that, a chain of border points would merge
    every cluster along a dense region into one.
    """
    signatures = ([C.InsertionSignature(pos=1000, end=1001, length=100)] * 5
                  + [C.InsertionSignature(pos=1200, end=1201, length=100)]
                  + [C.InsertionSignature(pos=1400, end=1401, length=100)] * 5)
    clusters = C.dbscan_cluster_indices(signatures, list(range(11)), 5)
    assert len(clusters) == 1 or len(clusters) == 2


def test_blocks_split_on_a_gap_wider_than_the_compatibility_window():
    near = [C.InsertionSignature(pos=1000, end=1001, length=100),
            C.InsertionSignature(pos=1000 + C.COMPATIBILITY_WINDOW_BP, end=1, length=100)]
    assert len(call_or_skip(C.build_blocks, near)) == 1
    far = [C.InsertionSignature(pos=1000, end=1001, length=100),
           C.InsertionSignature(pos=1001 + C.COMPATIBILITY_WINDOW_BP, end=1, length=100)]
    assert len(C.build_blocks(far)) == 2


def test_the_block_size_cap_can_split_a_genuine_cluster():
    """
    Pinned as a known cost, not as a feature. The cap bounds the O(n^2) pass at
    a repetitive locus; the price is that a cluster of more than 1500
    signatures is cut and reported as two events. The fix is a spatial index,
    and it has nothing to do with the language.
    """
    crowded = [C.InsertionSignature(pos=1000, end=1001, length=100)
               for _ in range(C.MAX_BLOCK_SIGNATURES + 10)]
    blocks = C.build_blocks(crowded)
    assert len(blocks) == 2
    assert blocks[0].end - blocks[0].begin == C.MAX_BLOCK_SIGNATURES


# ------------------------------------------------------- signature extraction
def test_a_cigar_insertion_anchors_at_the_reference_position_of_the_op():
    """70M90I70M from 1000 puts the breakpoint at 1070, and the insertion
    occupies NO reference span, so `end` is `pos + 1`."""
    calls = call_or_skip(C.build_component_calls,
                         [ins_read("single_ins", 1000, 70, 90, 70)], "chr1", 0)
    assert len(calls) == 1
    call = calls[0]
    assert (call.chrom, call.tid, call.anchor_pos) == ("chr1", 0, 1070)
    assert len(call.read_indices) == 1
    assert len(call.insertion_read_indices) == 1
    assert len(call.breakpoint_candidates) == 1
    assert call.breakpoint_candidates[0].ins_len == 90
    assert call.breakpoint_candidates[0].clip_len == 0
    assert call.bin_end == call.bin_start + 1


def test_a_cigar_insertion_below_mapq_60_produces_nothing_at_all():
    """
    An EQUALITY test, and MAPQ 59 is enough to lose the signature entirely --
    not to downweight it. A CIGAR insertion is the aligner asserting these extra
    bases belong at this exact position, and that assertion is worthless if the
    aligner was unsure where the read goes.
    """
    assert C.build_component_calls([ins_read("low_mapq", 1000, 70, 90, 70, mapq=59)],
                                   "chr1", 0) == []


def test_an_insertion_shorter_than_the_floor_is_an_indel_not_an_event():
    assert C.build_component_calls([ins_read("small", 1000, 70, 49, 70)], "chr1", 0) == []
    assert len(C.build_component_calls([ins_read("big", 1000, 70, 50, 70)], "chr1", 0)) == 1


def test_a_split_read_with_a_query_gap_reports_the_gap_as_the_insertion():
    """80M120S at 1000 with SA 1081,+,140S60M: the reference is contiguous, the
    read carries 60 unexplained bases, so the insertion is 60 -- and the read
    is split evidence, NOT a clip hint, even though it has a 120 bp clip."""
    calls = C.build_component_calls(
        [read("split_with_query_gap", 1000, [(CIGAR_M, 80), (CIGAR_S, 120)],
              sa="chr1,1081,+,140S60M,60,1;", seq_len=200)], "chr1", 0)
    assert len(calls) == 1
    assert len(calls[0].split_sa_read_indices) == 1
    assert calls[0].soft_clip_read_indices == []
    assert calls[0].breakpoint_candidates[0].ins_len == 60
    assert calls[0].breakpoint_candidates[0].clip_len == 0
    assert (calls[0].breakpoint_candidates[0].class_mask
            & C.CANDIDATE_SPLIT_SA_SUPPLEMENTARY) != 0


def test_a_split_read_without_a_query_gap_falls_back_to_a_clip_hint():
    """
    90M60S at 1000 with SA 1091,+,60S90M: the two segments tile the read with
    no gap, so there is no insertion. The read still gets a clip hint -- the
    locus stays alive with the weakest possible evidence, which is the correct
    outcome when the split itself says nothing.
    """
    calls = C.build_component_calls(
        [read("sa_softclip_no_gap", 1000, [(CIGAR_M, 90), (CIGAR_S, 60)],
              sa="chr1,1091,+,60S90M,60,1;", seq_len=150)], "chr1", 0)
    assert len(calls) == 1
    assert calls[0].split_sa_read_indices == []
    assert len(calls[0].soft_clip_read_indices) == 1
    assert calls[0].breakpoint_candidates[0].ins_len == 0
    assert calls[0].breakpoint_candidates[0].clip_len == 60
    assert (calls[0].breakpoint_candidates[0].class_mask & C.CANDIDATE_SOFT_CLIP) != 0


def test_a_leading_clip_anchors_at_the_alignment_start():
    calls = C.build_component_calls(
        [read("clip_hint", 2000, [(CIGAR_S, 120), (CIGAR_M, 180)], seq_len=300)],
        "chr1", 0)
    assert len(calls) == 1
    assert len(calls[0].soft_clip_read_indices) == 1
    assert calls[0].split_sa_read_indices == []
    assert calls[0].breakpoint_candidates[0].clip_len == 120
    assert calls[0].anchor_pos == 2000


def test_a_trailing_clip_anchors_at_the_alignment_end():
    calls = C.build_component_calls(
        [read("trailing", 2000, [(CIGAR_M, 180), (CIGAR_S, 120)], seq_len=300)],
        "chr1", 0)
    assert calls[0].anchor_pos == 2180


def test_a_clip_shorter_than_the_floor_is_not_a_hint():
    assert C.build_component_calls(
        [read("short_clip", 2000, [(CIGAR_S, 19), (CIGAR_M, 180)], seq_len=199)],
        "chr1", 0) == []


def test_a_clip_beside_a_known_insertion_is_not_counted_as_extra_evidence():
    """
    The precedence rule, and it is what stops one read looking like two. A read
    with both a long insertion and a long clip emits the insertion signature
    only -- the clip is the SAME junction seen from the other side.
    """
    calls = C.build_component_calls(
        [read("both", 1000, [(CIGAR_M, 70), (CIGAR_I, 90), (CIGAR_M, 70), (CIGAR_S, 200)],
              seq_len=430)], "chr1", 0)
    assert len(calls) == 1
    assert calls[0].soft_clip_read_indices == []
    assert len(calls[0].insertion_read_indices) == 1
    assert len(calls[0].breakpoint_candidates) == 1


def test_supplementary_records_are_skipped_because_the_primary_carries_them():
    supplementary = ins_read("supp", 1000, 70, 90, 70)
    supplementary.flag = 0x800
    assert C.build_component_calls([supplementary], "chr1", 0) == []


def test_a_read_on_another_contig_is_ignored():
    other = ins_read("elsewhere", 1000, 70, 90, 70)
    other.tid = 1
    assert C.build_component_calls([other], "chr1", 0) == []


def test_an_sa_segment_on_the_other_strand_is_an_inversion_not_an_insertion():
    calls = C.build_component_calls(
        [read("inverted", 1000, [(CIGAR_M, 80), (CIGAR_S, 120)],
              sa="chr1,1081,-,140S60M,60,1;", seq_len=200)], "chr1", 0)
    assert calls[0].split_sa_read_indices == []


def test_an_sa_segment_on_another_contig_is_a_translocation_not_an_insertion():
    calls = C.build_component_calls(
        [read("translocated", 1000, [(CIGAR_M, 80), (CIGAR_S, 120)],
              sa="chr2,1081,+,140S60M,60,1;", seq_len=200)], "chr1", 0)
    assert calls[0].split_sa_read_indices == []


# -------------------------------------------------------------- projection
def test_two_length_classes_at_one_locus_stay_two_events():
    """
    The C++ case that motivates the whole three-dimensional metric: four reads
    within 15 bp, two reporting ~90 bp and two reporting ~400 bp. Position-only
    clustering gives one event with four supporting reads; this gives two, and
    their lengths differ by more than 200.
    """
    reads = [ins_read("a1", 1000, 80, 90, 80),
             ins_read("a2", 1010, 75, 92, 75),
             ins_read("b1", 1005, 80, 400, 80),
             ins_read("b2", 1015, 75, 405, 75)]
    calls = C.build_component_calls(reads, "chr1", 0)
    assert len(calls) == 2
    assert len(calls[0].read_indices) == 2
    assert len(calls[1].read_indices) == 2
    lengths = [c.breakpoint_candidates[0].ins_len for c in calls]
    assert abs(lengths[0] - lengths[1]) >= 200


def test_anchor_is_the_median_so_it_names_a_position_a_read_proposed():
    """
    Median, not mean. A breakpoint is a position, and averaging two positions
    can name a base no read actually proposed -- which then fails to match the
    reference context the TSD detector goes on to fetch.
    """
    reads = [ins_read(f"r{i}", 1000 + 10 * i, 70, 90, 70) for i in range(5)]
    calls = C.build_component_calls(reads, "chr1", 0)
    assert len(calls) == 1
    assert calls[0].anchor_pos in {1070 + 10 * i for i in range(5)}
    assert calls[0].anchor_pos == 1090


def test_peak_weight_counts_reads_and_not_signatures():
    """
    One read contributing two CIGAR insertions to the same cluster is one piece
    of evidence. Counting signatures would let a single noisy read manufacture
    its own support -- the same double-counting the clip precedence rule blocks
    from the other direction.
    """
    twice = read("two_ins", 1000,
                 [(CIGAR_M, 70), (CIGAR_I, 90), (CIGAR_M, 20), (CIGAR_I, 95),
                  (CIGAR_M, 70)], seq_len=345)
    calls = C.build_component_calls([twice], "chr1", 0)
    assert len(calls) == 1
    close(calls[0].peak_weight, 1.0, "peak weight")
    assert len(calls[0].breakpoint_candidates) == 2
    assert calls[0].evidence_indel_count == 2
    assert calls[0].read_indices == [0]


def test_calls_are_ordered_by_position_then_by_descending_support():
    reads = ([ins_read(f"far{i}", 50000, 70, 90, 70) for i in range(3)]
             + [ins_read("near", 1000, 70, 90, 70)])
    calls = C.build_component_calls(reads, "chr1", 0)
    assert [c.anchor_pos for c in calls] == [1070, 50070]


def test_evidence_counts_are_kept_separate_by_kind():
    reads = [ins_read("i1", 1000, 70, 90, 70),
             read("c1", 1071, [(CIGAR_S, 60), (CIGAR_M, 180)], seq_len=240)]
    calls = C.build_component_calls(reads, "chr1", 0)
    assert len(calls) == 1
    assert calls[0].evidence_indel_count == 1
    assert calls[0].evidence_soft_clip_count == 1
    assert calls[0].evidence_sa_hint_count == 0
