"""
Sequence primitives and the library-independent composition model.

The C++ pins these in `tests/test_sequence_model_features.cpp` against
HAND-COMPUTABLE values rather than regression snapshots, which is the one place
in the C++ suite that is a real specification rather than a directional check.
Those cases are reproduced here verbatim, so the two implementations are pinned
to the same arithmetic and not merely to each other.
"""

from __future__ import annotations

import math

import pytest
from conftest import call_or_skip, close

from placer_py import seqtools as S

pytestmark = pytest.mark.invariant

TE1 = "ACGTGACTTGCAAGTCCATGGATCCAGTTACGGCATTAGCCATGGACTTAGCAATTGCCA"
TE2 = "GGCATTAGCCATGGACTTAGCAATTGCCACGTAAGGTTCCAAGGTTACGATCAGGTACAT"


# ------------------------------------------------------------- GC and entropy
def test_gc_excludes_non_acgt_from_the_denominator():
    """
    N-padding must not dilute GC. A run of N is missing information, not
    AT-rich sequence, and a caller that conflates the two reads a reference gap
    as a poly(A) tail.
    """
    close(call_or_skip(S.compute_te_sequence_composition, "ACGT").gc, 0.5, "gc ACGT")
    close(S.compute_te_sequence_composition("AAAA").gc, 0.0, "gc AAAA")
    close(S.compute_te_sequence_composition("GGCC").gc, 1.0, "gc GGCC")
    close(S.compute_te_sequence_composition("ACGTNNNN").gc, 0.5, "gc padded")


def test_entropy_is_in_bits_over_the_acgt_alphabet():
    comp = S.compute_te_sequence_composition
    close(comp("ACGT").entropy_bits, 2.0, "uniform")
    close(comp("ACAC").entropy_bits, 1.0, "two bases")
    close(comp("A" * 8).entropy_bits, 0.0, "homopolymer")
    close(comp("AAAC").entropy_bits,
          -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)), "3:1 mix")


# ---------------------------------------------------- tandem and low complexity
def test_tandem_fraction_covers_whole_repeats_and_unions_periods():
    """
    A poly(A) run is a period-1 repeat AND a period-2 repeat AND a period-3
    repeat. Coverage is the union over periods, so it is 1.0 and not 3.0 --
    which is why the field can be read as a fraction at all.
    """
    comp = S.compute_te_sequence_composition
    close(call_or_skip(comp, "A" * 20).tandem_fraction, 1.0, "homopolymer")
    close(comp("AC" * 10).tandem_fraction, 1.0, "period 2")
    assert comp("ACGTGACTTGCAAGTCCATG").tandem_fraction < 0.5


def test_low_complexity_flags_at_rich_and_two_base_windows():
    comp = S.compute_te_sequence_composition
    close(comp("A" * 32).low_complexity_fraction, 1.0, "poly A")
    close(comp("AT" * 16).low_complexity_fraction, 1.0, "AT microsatellite")
    # Balanced 4-base sequence: top-2 is 50% in every window, nothing flagged.
    close(comp("ACGT" * 8).low_complexity_fraction, 0.0, "balanced")


def test_every_composition_feature_stays_in_its_declared_range():
    comp = S.compute_te_sequence_composition("ACGTGACTTGCAAGTCCATGGATCCAGTTACG")
    assert 0.0 <= comp.gc <= 1.0
    assert 0.0 <= comp.entropy_bits <= 2.0
    assert 0.0 <= comp.tandem_fraction <= 1.0
    assert 0.0 <= comp.low_complexity_fraction <= 1.0


# ---------------------------------------------------------------- k-mer tables
def test_jsd_against_a_sequences_own_distribution_is_exactly_zero():
    self_only = call_or_skip(S.build_te_canonical_kmer_frequencies, [TE1], 5)
    assert self_only
    close(S.te_kmer_jsd_vs_background(TE1, 5, self_only), 0.0, "self JSD")


def test_jsd_is_maximal_for_a_sequence_sharing_no_kmer():
    library = S.build_te_canonical_kmer_frequencies([TE1, TE2], 5)
    poly = S.te_kmer_jsd_vs_background("A" * 60, 5, library)
    close(poly, 1.0, "poly-A JSD")
    member = S.te_kmer_jsd_vs_background(TE1, 5, library)
    assert 0.0 <= member <= 1.0
    assert member < poly


def test_jsd_returns_zero_rather_than_nan_on_degenerate_input():
    """
    Every degenerate path returns 0, not NaN. A NaN here would propagate into
    the ledger and then into a log, and the first place it would be noticed is
    a missing call rather than an error.
    """
    library = S.build_te_canonical_kmer_frequencies([TE1, TE2], 5)
    close(S.te_kmer_jsd_vs_background("", 5, library), 0.0, "empty")
    close(S.te_kmer_jsd_vs_background("ACG", 5, library), 0.0, "shorter than k")
    close(S.te_kmer_jsd_vs_background(TE1, 5, {}), 0.0, "no background")
    assert S.build_te_canonical_kmer_frequencies([TE1], 0) == {}
    assert S.build_te_canonical_kmer_frequencies([TE1], 20) == {}


def test_containment_is_one_for_a_library_member_and_zero_for_a_stranger():
    present = call_or_skip(S.build_te_canonical_kmer_presence, [TE1, TE2], 9)
    close(S.te_kmer_containment(TE1, 9, present), 1.0, "member")
    close(S.te_kmer_containment("A" * 60, 9, present), 0.0, "poly A")
    close(S.te_kmer_containment("ACG", 9, present), 0.0, "shorter than k")
    close(S.te_kmer_containment(TE1, 9, set()), 0.0, "no library")


def test_canonical_key_folds_a_kmer_onto_its_reverse_complement():
    """
    Strand folding is what makes the k-mer features orientation-free. An
    insertion is reported on the reference strand but the element may be in
    either orientation, so an unfolded table would score one orientation as a
    library member and the other as a stranger.
    """
    forward = dict(S.for_each_valid_kmer("ACGTACGTA", 9))
    revcomp = dict(S.for_each_valid_kmer(S.reverse_complement("ACGTACGTA"), 9))
    fkey = S.canonical_kmer_key(next(iter(forward.values())), 9)
    rkey = S.canonical_kmer_key(next(iter(revcomp.values())), 9)
    assert fkey == rkey


def test_kmers_spanning_a_non_acgt_base_are_skipped_not_recoded():
    assert list(S.for_each_valid_kmer("ACGTN", 5)) == []
    assert [start for start, _ in S.for_each_valid_kmer("ACGTNACGTA", 5)] == [5]


def test_background_is_invalid_when_any_table_is_empty():
    assert not call_or_skip(S.build_te_sequence_background, []).valid
    assert S.build_te_sequence_background([TE1, TE2]).valid


# ------------------------------------------------------------ edit identity
def test_semiglobal_identity_charges_the_query_and_not_the_target():
    """
    Free end gaps on the target only. A 1 kb 5'-truncated L1 placed inside a
    6 kb consensus must score 1.0 -- the 5 kb it does not cover is the
    ELEMENT's, not the fragment's, and charging the fragment for it is the
    truncation intolerance the README says the C++ blocks suffer from.
    """
    close(call_or_skip(S.semiglobal_edit_identity, "ACGTACGT", "TTTTACGTACGTTTTT"),
          1.0, "contained exactly")
    close(S.semiglobal_edit_identity("ACGTACGT", "ACGTACGT"), 1.0, "identical")
    # One substitution in eight bases is one edit: 1 - 1/8.
    close(S.semiglobal_edit_identity("ACGTACGT", "TTACGTTCGTTT"), 1.0 - 1.0 / 8.0,
          "one mismatch")
    close(S.semiglobal_edit_identity("", "ACGT"), 0.0, "empty query")
    close(S.semiglobal_edit_identity("ACGT", ""), 0.0, "empty target")


def test_identity_is_clamped_into_the_unit_interval():
    assert 0.0 <= S.semiglobal_edit_identity("A" * 40, TE1) <= 1.0


# ---------------------------------------------------------------- naming
def test_repeatmasker_and_dfam_headers_reach_the_same_family():
    """
    Both conventions have to collapse onto ONE family key. `family_kind` in the
    decision policy branches on it, so a library that spells L1 three ways
    would otherwise produce three families and three different priors for the
    same element.
    """
    for header in ("L1HS#LINE/L1", "L1:L1HS", "L1HS"):
        parts = call_or_skip(S.parse_te_name_parts, header)
        assert parts.family == "L1", header
        assert parts.family_key == "L1", header

    alu = S.parse_te_name_parts("AluYa5#SINE/Alu")
    assert (alu.family, alu.subfamily, alu.class_label, alu.order_label) == (
        "Alu", "AluYa5", "SINE", "Alu")
    assert S.parse_te_name_parts("SVA_E#Retroposon/SVA").family == "SVA"
    assert S.parse_te_name_parts("HERVK#LTR/HERVK").family == "HERV"


def test_family_is_read_from_the_class_path_and_not_from_the_subfamily():
    """
    Pinned because it is surprising and the C++ does it too: for
    `HERVK-int#LTR/ERVK` the family comes from the LAST path component, so the
    element is family `ERVK` even though its subfamily plainly says HERV. The
    four canonical keys are matched against the path, never against the name.
    """
    parts = S.parse_te_name_parts("HERVK-int#LTR/ERVK")
    assert parts.subfamily == "HERVK-int"
    assert parts.family == "ERVK"


def test_an_unrecognised_family_is_kept_verbatim_rather_than_dropped():
    """
    An unknown family is reported, not silently renamed to NA. The selection
    layer can weigh "named, but not one of the four" differently from
    "unnamed"; collapsing them would destroy that distinction at the source.
    """
    parts = S.parse_te_name_parts("MIR3#SINE/MIR")
    assert parts.family == "MIR"
    assert parts.family_key == "MIR"


def test_header_description_after_whitespace_is_ignored():
    assert S.parse_te_name_parts("AluY#SINE/Alu  some description").subfamily == "AluY"


# --------------------------------------------------------------- small helpers
def test_homopolymer_run_and_at_fraction():
    assert call_or_skip(S.max_homopolymer_run, "") == 0
    assert S.max_homopolymer_run("AACCCGT") == 3
    close(S.at_fraction("ACGT"), 0.5, "at")
    close(S.at_fraction("NNNN"), 0.0, "at of N only")


def test_kmer_uniqueness_separates_a_repeat_from_unique_sequence():
    close(S.kmer_uniqueness_ratio("A" * 40, 5), 1.0 / 36.0, "poly A")
    assert S.kmer_uniqueness_ratio(TE1, 5) > 0.9
    close(S.kmer_uniqueness_ratio("ACGTGACTTGCAAGTCC", 5), 1.0, "all distinct")
    close(S.kmer_uniqueness_ratio("ACG", 5), 0.0, "shorter than k")


def test_kmer_size_csv_drops_bad_entries_instead_of_failing():
    """
    A typo in a performance flag must not stop a run that would otherwise be
    correct, so out-of-range and unparsable entries are dropped.
    """
    assert call_or_skip(S.parse_kmer_sizes_csv, "11,13,17", 0) == [11, 13, 17]
    assert S.parse_kmer_sizes_csv("3,13,99", 0) == [13]
    assert S.parse_kmer_sizes_csv("", 0) == [13]
    assert S.parse_kmer_sizes_csv("13", 15) == [13, 15]
    assert S.parse_kmer_sizes_csv("13,13", 13) == [13]


def test_confidence_tracks_naming_specificity_not_evidence_strength():
    assert S.confidence_from_qc_reason("PASS_INSERT_TE_ALIGNMENT") == "HIGH"
    assert S.confidence_from_qc_reason("PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY") == "MEDIUM"
    assert S.confidence_from_qc_reason("PASS_INSERT_TE_ALIGNMENT_UNKNOWN") == "LOW"
    assert S.confidence_from_qc_reason("FAIL_WHATEVER") == "NA"


def test_fnv1a_is_stable_and_order_sensitive():
    """
    The cache key for a TE library build. It only has to be deterministic and
    to change when any input changes -- but it must be ORDER sensitive, or two
    different libraries with the same entry names would share a cache entry.
    """
    base = S.FNV1A_OFFSET_BASIS
    assert S.fnv1a_append_string(base, "abc") == S.fnv1a_append_string(base, "abc")
    assert S.fnv1a_append_string(base, "abc") != S.fnv1a_append_string(base, "cba")
    assert S.fnv1a_append_int32(base, 13) != S.fnv1a_append_int32(base, 14)
    assert 0 <= S.fnv1a_append_int32(base, -1) < (1 << 64)
