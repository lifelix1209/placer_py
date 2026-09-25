"""
The read view and the CIGAR/SA parsing under it.

The C++ has no tests for any of this. `ReadView` wraps a `bam1_t`, so pinning
it needs htslib on the link line, and the helpers are duplicated privately into
four translation units where nothing can reach them. Making the read a value
object costs one adapter and buys the whole layer a specification.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip

from placer import alignment as A

pytestmark = pytest.mark.invariant

M, I, D, N, S, H = A.CIGAR_M, A.CIGAR_I, A.CIGAR_D, A.CIGAR_N, A.CIGAR_S, A.CIGAR_H


# ------------------------------------------------------------- reference span
def test_ref_end_counts_only_reference_consuming_operations():
    read = A.AlignedRead(pos=1000, cigar=[(S, 50), (M, 100), (I, 200), (D, 10),
                                          (M, 100), (S, 50)], seq="A" * 500)
    # 100 M + 10 D + 100 M = 210 reference bases. The 200 bp insertion and both
    # clips consume none.
    assert call_or_skip(A.compute_ref_end, read) == 1210


def test_ref_end_of_a_cigarless_record_is_its_start_rather_than_an_error():
    """
    A zero-length interval, which every caller's own `end > start` check then
    drops. Raising here would end a whole run over one unusable record in a
    six-million-read BAM.
    """
    assert A.compute_ref_end(A.AlignedRead(pos=1000)) == 1000


def test_skipped_reference_counts_toward_the_span():
    assert A.compute_ref_end(A.AlignedRead(pos=0, cigar=[(M, 10), (N, 90), (M, 10)])) == 110


# ------------------------------------------------------------------ read view
def test_seq_len_falls_back_to_the_cigar_when_the_record_carries_no_bases():
    """
    A hard-clipped supplementary legitimately has no sequence, and its own CIGAR
    is then the only statement of how long the query is. Returning 0 would make
    every query-interval computation on it fail.
    """
    hard = A.AlignedRead(cigar=[(H, 100), (M, 400)], seq="")
    assert hard.seq_len == 400
    assert A.AlignedRead(cigar=[(S, 100), (M, 400)], seq="").seq_len == 500


def test_flag_accessors_read_the_documented_bits():
    assert A.AlignedRead(flag=0x10).is_reverse
    assert A.AlignedRead(flag=0x800).is_supplementary
    assert A.AlignedRead(flag=0x100).is_secondary
    assert A.AlignedRead(flag=0x4).is_unmapped
    assert not A.AlignedRead(flag=0).is_reverse


def test_tag_accessors_return_none_rather_than_raising_on_a_bad_value():
    read = A.AlignedRead(tags={"NM": "not a number", "SA": "chr1,1,+,10M,60,0;"})
    assert read.get_int_tag("NM") is None
    assert read.get_int_tag("XX") is None
    assert read.has_sa_tag()
    assert not A.AlignedRead(tags={"SA": ""}).has_sa_tag()


def test_subsequence_decoding_is_clamped_rather_than_index_erroring():
    read = A.AlignedRead(seq="ACGTACGT")
    assert read.decode_subsequence(2, 3) == "GTA"
    assert read.decode_subsequence(-1, 3) == ""
    assert read.decode_subsequence(100, 3) == ""
    assert read.decode_subsequence(6, 100) == "GT"


# ----------------------------------------------------------------- SA parsing
def test_sa_records_are_parsed_independently_so_one_bad_field_costs_one_record():
    entries = call_or_skip(A.parse_sa_tag_z,
                           "chr1,1000,+,50M,60,1;,,,;chr2,2000,-,60M,60,0;")
    assert [(e.rname, e.pos, e.strand, e.cigar) for e in entries] == [
        ("chr1", 1000, "+", "50M"), ("chr2", 2000, "-", "60M")]


def test_sa_records_missing_a_required_field_are_dropped():
    assert A.parse_sa_tag_z("") == []
    assert A.parse_sa_tag_z(",1000,+,50M;") == []          # no contig
    assert A.parse_sa_tag_z("chr1,0,+,50M;") == []         # position is 1-based
    assert A.parse_sa_tag_z("chr1,1000,+,;") == []         # no cigar
    assert A.parse_sa_tag_z("chr1,1000,+;") == []          # too few fields


def test_integer_parsing_requires_the_whole_string():
    """
    `strtol` prefix parsing would read "12abc" as 12, which in an SA tag means
    silently accepting a corrupt position and placing a segment at the wrong
    breakpoint.
    """
    assert A.parse_int32("12") == 12
    assert A.parse_int32("-12") == -12
    assert A.parse_int32("12abc") is None
    assert A.parse_int32("") is None
    assert A.parse_int32("abc") is None


# -------------------------------------------------------------- CIGAR strings
def test_cigar_string_parsing_refuses_anything_malformed():
    ops = call_or_skip(A.parse_cigar_ops, "100S50M2I48M")
    assert [(o.length, o.op) for o in ops] == [(100, "S"), (50, "M"), (2, "I"), (48, "M")]
    assert A.parse_cigar_ops("") is None
    assert A.parse_cigar_ops("M") is None        # operator with no count
    assert A.parse_cigar_ops("0M") is None       # zero-length op
    assert A.parse_cigar_ops("100S50") is None   # count with no operator


def test_a_leading_soft_clip_states_the_query_offset_directly():
    ops = A.parse_cigar_ops("100S400M")
    interval, ref_len = A.cigar_to_query_interval(ops, 500)
    assert (interval.qstart, interval.qend, ref_len) == (100, 500, 400)


def test_a_trailing_soft_clip_states_it_from_the_far_end():
    ops = A.parse_cigar_ops("400M100S")
    interval, ref_len = A.cigar_to_query_interval(ops, 500)
    assert (interval.qstart, interval.qend, ref_len) == (0, 400, 400)


def test_a_fully_aligned_segment_starts_at_zero():
    interval, ref_len = A.cigar_to_query_interval(A.parse_cigar_ops("500M"), 500)
    assert (interval.qstart, interval.qend, ref_len) == (0, 500, 500)


def test_a_hard_clipped_partial_segment_is_refused_rather_than_guessed():
    """
    The third branch, and refusing is the right answer. A segment with only hard
    clips and a partial query span cannot be located in the primary's
    coordinates, and defaulting to offset 0 would put every such segment at the
    read start -- inventing an insertion at the wrong breakpoint rather than
    reporting none.
    """
    assert A.cigar_to_query_interval(A.parse_cigar_ops("100H400M"), 500) is None
    assert A.cigar_to_query_interval(A.parse_cigar_ops("400M"), 500) is None


def test_query_interval_rejects_degenerate_inputs():
    assert A.cigar_to_query_interval([], 500) is None
    assert A.cigar_to_query_interval(A.parse_cigar_ops("100S400M"), 0) is None
    assert A.cigar_to_query_interval(A.parse_cigar_ops("500D"), 500) is None


# -------------------------------------------------------- normalized segments
def test_an_sa_position_is_converted_from_one_based_to_zero_based():
    entry = A.parse_sa_tag_z("chr1,1001,+,100S400M,60,2;")[0]
    normalized = call_or_skip(A.normalized_sa_alignment, entry, 500)
    assert (normalized.chrom, normalized.ref_start, normalized.ref_end) == ("chr1", 1000, 1400)
    assert (normalized.qstart, normalized.qend) == (100, 500)
    assert not normalized.is_reverse


def test_the_primary_segment_is_normalized_onto_the_same_shape():
    read = A.AlignedRead(pos=1000, cigar=[(M, 400), (S, 100)], seq="A" * 500, flag=0x10)
    normalized = A.normalized_primary_alignment(read, "chr1")
    assert (normalized.ref_start, normalized.ref_end) == (1000, 1400)
    assert (normalized.qstart, normalized.qend) == (0, 400)
    assert normalized.is_reverse


def test_a_split_pair_yields_the_insertion_as_query_gap_minus_reference_gap():
    """
    The two coordinate systems together are what make a split interpretable:
    the reference gap is what the genome skipped, the query gap is what the read
    carried, and the difference is sequence the read has and the reference does
    not.
    """
    read = A.AlignedRead(pos=1000, cigar=[(M, 80), (S, 120)], seq="A" * 200)
    primary = A.normalized_primary_alignment(read, "chr1")
    mate = A.normalized_sa_alignment(A.parse_sa_tag_z("chr1,1081,+,140S60M;")[0], 200)
    query_gap = mate.qstart - primary.qend
    ref_gap = max(0, mate.ref_start - primary.ref_end)
    assert query_gap - ref_gap == 60


# ------------------------------------------------------------------- median
def test_median_is_the_upper_middle_and_never_an_average():
    """
    `values[n/2]`. For an even count the result is the upper of the two middle
    values -- not their mean, because a breakpoint has to be a base some read
    actually proposed.
    """
    assert call_or_skip(A.median_i32, [1, 2, 3, 4]) == 3
    assert A.median_i32([3, 1, 2]) == 2
    assert A.median_i32([]) == -1
    assert A.median_i32([7]) == 7


def test_op_character_and_code_round_trip():
    for code, char in enumerate("MIDNSHP=X"):
        assert A.op_to_char(code) == char
        assert A.char_to_op(char) == code
    assert A.char_to_op("Z") == -1
