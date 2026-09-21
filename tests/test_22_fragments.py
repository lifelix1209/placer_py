"""
Fragment extraction: which bases get cut out of a read, and from where.

The stage that can fail SILENTLY. A fragment made of 200 bp of flanking genome
and 800 bp of element still aligns to the element, still reports high identity
and still gets called, so "it produced a fragment" is not evidence that the
fragment is right. These tests pin the boundaries, not just the counts.

`tests/test_cigar_insertion_fragment_local_retention.cpp` is reproduced in
`test_two_by_length_then_two_by_locality...` and the MAPQ case below.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip

from placer_py import clustering as C
from placer_py import fragments as F
from placer_py.alignment import CIGAR_D, CIGAR_I, CIGAR_M, CIGAR_S, AlignedRead
from placer_py.config import PipelineConfig

pytestmark = pytest.mark.invariant

M, I, D, S = CIGAR_M, CIGAR_I, CIGAR_D, CIGAR_S


def make_read(qname="r1", pos=1000, cigar=None, seq=None, mapq=60, flag=0,
              sa=None, nm=None, tid=0):
    cigar = cigar or [(M, 500), (I, 300), (M, 500)]
    if seq is None:
        seq = "A" * sum(n for op, n in cigar if op in (M, I, S))
    tags = {}
    if sa is not None:
        tags["SA"] = sa
    if nm is not None:
        tags["NM"] = nm
    return AlignedRead(qname=qname, tid=tid, pos=pos, mapq=mapq, cigar=cigar,
                       seq=seq, flag=flag, tags=tags)


def component(chrom="chr1", anchor=1500, read_indices=(0,), **kw):
    call = C.ComponentCall(chrom=chrom, tid=0, anchor_pos=anchor,
                           read_indices=list(read_indices))
    for key, value in kw.items():
        setattr(call, key, value)
    return call


# ------------------------------------------------------------- clip analysis
def test_the_anchor_beside_a_clip_stops_at_the_first_non_match():
    """
    A 200 bp clip beside twenty 10 bp match blocks is a chimera, not a
    breakpoint, and `right_anchor_after_leading` has to be able to say so. The
    total matched bases cannot.
    """
    fragmented = [(S, 200)] + [(M, 10), (D, 1)] * 20
    info = call_or_skip(F.analyze_clip_info, make_read(cigar=fragmented))
    assert info.leading == 200
    assert info.right_anchor_after_leading == 10

    clean = [(S, 200), (M, 400)]
    assert F.analyze_clip_info(make_read(cigar=clean)).right_anchor_after_leading == 400


def test_clip_info_of_a_cigarless_record_is_empty_rather_than_an_error():
    info = F.analyze_clip_info(AlignedRead(pos=1234))
    assert (info.leading, info.trailing, info.ref_end) == (0, 0, 1234)


def test_both_terminal_clips_are_measured_independently():
    info = F.analyze_clip_info(make_read(cigar=[(S, 100), (M, 400), (S, 250)]))
    assert (info.leading, info.trailing) == (100, 250)
    assert info.left_anchor_before_trailing == 400


# -------------------------------------------------------- insertion discovery
def test_the_left_anchor_resets_on_any_non_match_operation():
    """
    `min(left_anchor, right_anchor)` is what the fragment reports, so an
    insertion is only as well placed as its WEAKER side. An insertion reached
    through a string of short match blocks is not anchored, however many bases
    those blocks total.
    """
    cigar = [(M, 50), (D, 5), (M, 30), (I, 200), (M, 400)]
    ops = call_or_skip(F.find_long_insertions, make_read(cigar=cigar), 50)
    assert len(ops) == 1
    assert ops[0].left_anchor == 30      # not 80
    assert ops[0].right_anchor == 400


def test_insertion_reference_position_accounts_for_deletions_before_it():
    cigar = [(M, 100), (D, 50), (M, 100), (I, 300), (M, 100)]
    ops = F.find_long_insertions(make_read(pos=1000, cigar=cigar), 50)
    assert ops[0].ref_pos == 1250
    assert ops[0].start == 200          # read coordinate skips the deletion


def test_insertions_below_the_floor_are_not_offered_at_all():
    assert F.find_long_insertions(make_read(cigar=[(M, 100), (I, 49), (M, 100)]), 50) == []
    assert len(F.find_long_insertions(make_read(cigar=[(M, 100), (I, 50), (M, 100)]), 50)) == 1


# ------------------------------------------------------------ insertion choice
def test_two_by_length_then_two_by_locality():
    """
    The C++ locality-retention case. One read carrying 861, 1689 and 387 bp
    insertions, with the component anchored on the 387 bp one: the two longest
    are kept unconditionally, and the 387 bp insertion -- the one the component
    is actually about -- is kept by the locality pass.

    The first pass is a HEDGE against a wrong anchor. If locality were the only
    filter, an anchor off by 2 kb would extract nothing at all.
    """
    pos0 = 21734874
    cigar = [(M, 4837), (I, 861), (M, 10910), (I, 1689), (M, 1721), (I, 387), (M, 120)]
    read = make_read("multi_insert_read", pos=pos0, cigar=cigar, mapq=60)
    call = component(chrom="4", anchor=21752342, insertion_read_indices=[0])
    call.breakpoint_candidates = [
        C.BreakpointCandidate(chrom="4", pos=21752342, ins_len=387,
                              read_id="multi_insert_read")]

    frags = call_or_skip(F.extract_fragments, call, [read], PipelineConfig())
    placed = {(f.ref_junc_pos, f.length) for f in frags
              if f.source == F.InsertionFragmentSource.CIGAR_INSERTION}
    assert (21750621, 1689) in placed
    assert (21739711, 861) in placed
    assert (21752342, 387) in placed


def test_a_single_mapq_below_sixty_loses_every_insertion_fragment():
    pos0 = 21734874
    cigar = [(M, 4837), (I, 861), (M, 10910), (I, 1689), (M, 1721), (I, 387), (M, 120)]
    read = make_read("low_mapq", pos=pos0, cigar=cigar, mapq=59)
    call = component(chrom="4", anchor=21752342)
    assert F.extract_fragments(call, [read], PipelineConfig()) == []


def test_at_most_four_insertions_per_read_and_the_shortest_non_local_one_goes():
    cigar = [(M, 100)]
    for length in (500, 400, 300, 200, 100):
        cigar += [(I, length), (M, 100)]
    read = make_read("five", pos=1000, cigar=cigar, mapq=60)
    call = component(anchor=1000)   # everything is local
    frags = [f for f in F.extract_fragments(call, [read], PipelineConfig())
             if f.source == F.InsertionFragmentSource.CIGAR_INSERTION]
    assert len(frags) == F.MAX_INSERTION_FRAGMENTS_PER_READ
    assert 100 not in {f.length for f in frags}


def test_locality_is_tested_against_breakpoints_and_not_only_the_anchor():
    """
    The anchor is a MEDIAN. An event whose breakpoints are bimodal can have its
    anchor sitting between the two real positions and near neither, so the
    locality test has to consult the candidates as well.
    """
    op = F.InsOp(start=0, len=300, ref_pos=50000)
    call = component(anchor=1000)
    assert not F.insertion_is_component_local(op, call)
    call.breakpoint_candidates = [C.BreakpointCandidate(pos=50100)]
    assert F.insertion_is_component_local(op, call)


# ------------------------------------------------------------ clip fragments
def test_a_leading_clip_is_taken_from_the_read_start_and_junctions_at_pos():
    read = make_read(cigar=[(S, 120), (M, 400)], seq="C" * 120 + "A" * 400, pos=2000)
    frags = F.extract_fragments(component(anchor=2000), [read], PipelineConfig())
    clip = [f for f in frags if f.source == F.InsertionFragmentSource.CLIP_REF_LEFT]
    assert len(clip) == 1
    assert (clip[0].start, clip[0].length) == (0, 120)
    assert clip[0].sequence == "C" * 120
    assert clip[0].ref_junc_pos == 2000
    assert clip[0].ref_side == F.ReferenceSide.REF_LEFT


def test_a_trailing_clip_is_taken_from_the_read_end_and_junctions_at_ref_end():
    read = make_read(cigar=[(M, 400), (S, 120)], seq="A" * 400 + "G" * 120, pos=2000)
    frags = F.extract_fragments(component(anchor=2400), [read], PipelineConfig())
    clip = [f for f in frags if f.source == F.InsertionFragmentSource.CLIP_REF_RIGHT]
    assert len(clip) == 1
    assert (clip[0].start, clip[0].length) == (400, 120)
    assert clip[0].sequence == "G" * 120
    assert clip[0].ref_junc_pos == 2400
    assert clip[0].ref_side == F.ReferenceSide.REF_RIGHT


def test_a_clip_below_the_floor_produces_no_fragment():
    read = make_read(cigar=[(S, 49), (M, 400)], pos=2000)
    assert F.extract_fragments(component(anchor=2000), [read], PipelineConfig()) == []


def test_a_fragment_with_no_bases_is_dropped_silently():
    """A hard-clipped record has coordinates but no sequence, and there is
    nothing to classify."""
    from placer_py.alignment import CIGAR_H
    read = AlignedRead(qname="hard", tid=0, pos=2000, mapq=60,
                       cigar=[(CIGAR_H, 120), (M, 400)], seq="")
    assert F.extract_fragments(component(anchor=2000), [read], PipelineConfig()) == []


# ----------------------------------------------------- short-insertion relax
def test_the_floor_relaxes_only_when_split_or_indel_reads_dominate():
    """
    The floor keeps ordinary indel noise out. Where split and CIGAR-insertion
    reads already outnumber the clip reads, the evidence that something is there
    no longer comes from any one insertion's length, so a short Alu-scale insert
    can be let through.
    """
    config = PipelineConfig()
    dominant = component(insertion_read_indices=[0, 1], soft_clip_read_indices=[2])
    assert call_or_skip(F.relaxed_min_insertion_length, dominant, config) == 35

    clip_heavy = component(insertion_read_indices=[0],
                           soft_clip_read_indices=[1, 2, 3])
    assert F.relaxed_min_insertion_length(clip_heavy, config) == 50


def test_equal_split_and_clip_support_still_relaxes():
    """`>=`, not `>`. Pinned because it is a one-character difference that
    changes which loci get a short-insert fragment at all."""
    config = PipelineConfig()
    tied = component(insertion_read_indices=[0, 1], soft_clip_read_indices=[2, 3])
    assert F.relaxed_min_insertion_length(tied, config) == 35


def test_the_relaxed_floor_never_goes_below_ten():
    config = PipelineConfig(short_ins_min_len=1)
    call = component(insertion_read_indices=[0, 1])
    assert F.relaxed_min_insertion_length(call, config) == 10


def test_relaxation_can_be_switched_off_entirely():
    config = PipelineConfig(short_ins_enable=False)
    call = component(insertion_read_indices=[0, 1])
    assert F.relaxed_min_insertion_length(call, config) == 50


# ------------------------------------------------------------ split geometry
def test_a_split_pair_on_another_strand_or_contig_has_no_insertion_length():
    a = F.NormalizedAln(chrom="chr1", ref_start=1000, ref_end=1080, qstart=0, qend=80)
    b = F.NormalizedAln(chrom="chr2", ref_start=1080, ref_end=1140, qstart=140, qend=200)
    assert call_or_skip(F.split_sa_insertion_len, a, b) == -1
    b.chrom = "chr1"
    b.is_reverse = True
    assert F.split_sa_insertion_len(a, b) == -1
    b.is_reverse = False
    assert F.split_sa_insertion_len(a, b) == 60


def test_the_query_junction_prefers_the_flank_edge_the_segments_abut_at():
    """
    Overlap at the junction is NORMAL -- microhomology at a TPRT junction is
    aligned by both segments -- so "abutting" is tested with the tolerance
    added, not against exact equality.
    """
    flank = F.NormalizedAln(qstart=0, qend=100)
    after = F.NormalizedAln(qstart=90, qend=200)
    assert call_or_skip(F._query_junction, flank, after, 20) == 100

    before = F.NormalizedAln(qstart=0, qend=110)
    assert F._query_junction(F.NormalizedAln(qstart=100, qend=200), before, 20) == 100


def test_a_nested_mate_junctions_at_the_midpoint_and_a_wide_one_is_refused():
    """
    The third branch, reachable only when the mate sits INSIDE the flank's
    query span. Beyond the tolerance the overlap is no longer microhomology but
    a real disagreement about where the read goes, and the read is refused
    rather than cut at a guessed point.
    """
    flank = F.NormalizedAln(qstart=100, qend=200)
    assert F._query_junction(flank, F.NormalizedAln(qstart=150, qend=160), 20) == 155
    assert F._query_junction(flank, F.NormalizedAln(qstart=130, qend=180), 20) is None


def test_the_opposite_side_window_excludes_the_flanks_own_aligned_core():
    """
    THE step that makes the fragment an insert rather than a mixture. A naive
    window around the junction would include the flank's own aligned bases
    whenever the junction sits inside it -- and those bases are reference
    genome, so the element call would be made partly on flanking sequence.
    """
    flank = F.NormalizedAln(qstart=0, qend=400)
    start, end = call_or_skip(F._opposite_side_window, 400, flank, 1000, 100, 100, 12)
    assert start >= flank.qend - 12
    assert end <= 500


def test_the_reference_side_is_unknown_within_the_slack():
    """Asserting a side for a junction 2 bp from the anchor would make a coin
    flip look like evidence."""
    assert F._reference_side(1000, 1000, 5) == F.ReferenceSide.UNKNOWN
    assert F._reference_side(900, 1000, 5) == F.ReferenceSide.REF_LEFT
    assert F._reference_side(1100, 1000, 5) == F.ReferenceSide.REF_RIGHT


def test_flank_choice_charges_five_nm_equivalents_for_a_large_indel():
    """
    The one hand-set constant in this module. It says "an indel beside the
    junction makes the junction's POSITION doubtful", which is a different kind
    of doubt from mismatch noise and which no count of mismatches expresses.
    """
    clean = F.NormalizedAln(anchor_len=100, nm=4)
    indel = F.NormalizedAln(anchor_len=100, nm=0, has_large_indel_near_bp=True)
    assert call_or_skip(F.compare_flank_candidate, clean, indel, 1000, 1.0) < 0

    better_anchor = F.NormalizedAln(anchor_len=106, nm=0, has_large_indel_near_bp=True)
    assert F.compare_flank_candidate(better_anchor, clean, 1000, 1.0) < 0


def test_flank_comparison_is_a_total_order_so_the_fragment_is_deterministic():
    """
    Ties are common -- two supplementary alignments of one read often have
    identical anchor and NM -- and an unstable order here would make the
    extracted SEQUENCE depend on iteration order.
    """
    a = F.NormalizedAln(anchor_len=100, nm=2, mapq=60, ref_start=1000, ref_end=1100)
    b = F.NormalizedAln(anchor_len=100, nm=2, mapq=60, ref_start=2000, ref_end=2100)
    assert F.compare_flank_candidate(a, b, 1500, 1.0) < 0
    assert F.compare_flank_candidate(b, a, 1500, 1.0) > 0
    assert F.compare_flank_candidate(a, a, 1500, 1.0) == 0


# ------------------------------------------------- anchored bases at a window
def test_deletions_count_as_anchor_but_also_raise_the_doubt_flag():
    read = make_read(pos=1000, cigar=[(M, 100), (D, 30), (M, 100)],
                     seq="A" * 200)
    anchored, large_indel = call_or_skip(F.anchor_len_from_bam, read, 1100, 150)
    assert anchored == 230
    assert large_indel


def test_a_small_deletion_near_the_breakpoint_does_not_raise_the_flag():
    read = make_read(pos=1000, cigar=[(M, 100), (D, 19), (M, 100)], seq="A" * 200)
    _, large_indel = F.anchor_len_from_bam(read, 1100, 150)
    assert not large_indel


def test_anchored_bases_outside_the_window_do_not_count():
    read = make_read(pos=1000, cigar=[(M, 1000)], seq="A" * 1000)
    anchored, _ = F.anchor_len_from_bam(read, 1100, 150)
    assert anchored == 251          # [950, 1251) clipped to the alignment


# ----------------------------------------------------------- the split path
def _split_read(sa="chr1,1601,+,700S300M,60,1;"):
    # 500M500S at 1000: reference ends at 1500. The SA segment starts at 1600
    # and covers query [700, 1000), so 200 query bases are unexplained and the
    # reference gap is 100 -> a 100 bp insertion.
    return make_read("split", pos=1000, cigar=[(M, 500), (S, 500)],
                     seq="A" * 500 + "T" * 200 + "G" * 300, sa=sa, nm=2)


def test_a_split_read_yields_a_fragment_from_the_far_side_of_the_junction():
    read = _split_read()
    call = component(anchor=1500)
    frags = F.extract_split_sa_fragments(call, [read], PipelineConfig())
    assert frags
    robust = frags[0]
    assert robust.source == F.InsertionFragmentSource.SPLIT_SA
    assert "|qj=" in robust.fragment_id
    assert robust.start >= 488          # at or past the flank core's end
    assert robust.sequence


def test_an_sa_record_with_fewer_than_six_fields_is_not_usable_here():
    """
    The fragment extractor needs mapq and NM to choose between flanks, so it
    requires all six fields -- where the clustering parser accepts four, because
    it only needs to know where the segment went.
    """
    read = _split_read(sa="chr1,1601,+,700S300M;")
    assert F.extract_split_sa_fragments(component(anchor=1500), [read],
                                        PipelineConfig()) == []


def test_split_extraction_is_disabled_by_a_non_positive_sa_cap():
    read = _split_read()
    config = PipelineConfig(max_sa_per_read=0)
    assert F.extract_split_sa_fragments(component(anchor=1500), [read], config) == []


def test_an_sa_claim_contradicted_by_an_observed_supplementary_is_unreliable():
    """
    An SA tag is a CLAIM about a segment; a supplementary record is that segment
    observed. When both exist the claim is cross-checked, and a near-miss is not
    a near-truth -- it means the two describe different query spans, so the
    junction inferred from the tag would be in the wrong place.
    """
    sa = F.NormalizedAln(qstart=700, qend=1000)
    matching = [F.NormalizedAln(qstart=700, qend=1000)]
    assert call_or_skip(F._sa_reliability, sa, matching)
    disagreeing = [F.NormalizedAln(qstart=300, qend=1000)]
    assert not F._sa_reliability(sa, disagreeing)


def test_an_sa_record_with_no_supplementary_in_the_window_is_believed():
    """
    The segment may simply align outside the bin. Refusing every such record
    would discard exactly the long-range split evidence the tag exists to
    provide.
    """
    assert F._sa_reliability(F.NormalizedAln(qstart=0, qend=100), None)
    assert F._sa_reliability(F.NormalizedAln(qstart=0, qend=100), [])


# ---------------------------------------------------------------- identifiers
def test_fragment_ids_survive_a_read_name_containing_the_delimiter():
    read = make_read("weird|name with space", cigar=[(S, 120), (M, 400)],
                     seq="C" * 120 + "A" * 400, pos=2000)
    frags = F.extract_fragments(component(anchor=2000), [read], PipelineConfig())
    assert "read=weird/name_with_space" in frags[0].fragment_id
    assert frags[0].fragment_id.count("|") == 6


def test_the_fasta_dump_wraps_at_eighty_columns():
    frag = F.InsertionFragment(fragment_id="frag 1", sequence="A" * 200)
    text = call_or_skip(F.fragments_to_fasta, [frag])
    lines = text.splitlines()
    assert lines[0] == ">frag_1"
    assert [len(line) for line in lines[1:]] == [80, 80, 40]
