"""
The structural hallmarks by class, and the class- and strand-aware decode.

Every insert is hand-built: an element body of known class, oriented on a known
strand, with or without the hallmark its class makes. The expected numbers are
counts.
"""

import random

import pytest

from placer.core import element_structure as E
from placer.core import structure as S
from placer.core.seqtools import reverse_complement

pytestmark = pytest.mark.invariant

RNG = random.Random(11)
#: A random body, framed in G/C so no A or T run at either end can merge with a
#: tail and change the count.
BODY = ("GCG" + "".join(RNG.choice("ACGT") for _ in range(294)) + "GCG").replace(
    "AAAAAA", "AACAAA").replace("TTTTTT", "TTGTTT")


def _decode(insert, te_class="NA", strand="NA"):
    return S.explain_te_sequence_structure(
        insert, "PASS_INSERT_TE_ALIGNMENT", "L1", "L1HS", 0.95,
        len(BODY) / len(insert), 1.0 - len(BODY) / len(insert), 0.0, 0.3, 0.0,
        "TE_MODEL_IN_DISTRIBUTION", 0.9, te_class=te_class, te_strand=strand)


# ------------------------------------------------------------------ poly(A)
def test_a_minus_strand_tail_is_the_poly_t_at_the_reference_five_prime_end():
    element = BODY + "A" * 25                     # element orientation
    reference_oriented = reverse_complement(element)
    assert reference_oriented.startswith("T" * 25)
    plus = E.measure(element, "LINE", "+")
    minus = E.measure(reference_oriented, "LINE", "-")
    assert plus.polya_len == minus.polya_len == 25


def test_a_and_t_are_not_conflated_once_the_strand_is_known():
    """A T-run at the element's 3' end is not a poly(A) tail."""
    assert E.measure(BODY + "T" * 25, "LINE", "+").polya_len == 0
    # With no strand the old A-or-T reading is all there is.
    assert E.measure(BODY + "T" * 25, "LINE", "NA").polya_len == 25


def test_a_short_or_impure_run_is_not_a_tail():
    assert E.measure(BODY + "A" * 5, "SINE", "+").polya_len == 0
    impure = BODY + "AAAAGAAAAAGAAAAA"
    assert E.measure(impure, "SINE", "+").polya_len == 16
    # Two interruptions in a row end it; so does one not followed by 3 more A.
    assert E.measure(BODY + "AAAGGAAAAAAAA", "SINE", "+").polya_len == 8
    assert E.measure(BODY + "AAGAAAAAAAA", "SINE", "+").polya_len == 8


def test_the_tail_is_expected_only_where_the_mechanism_makes_one():
    assert E.measure(BODY, "LINE", "+").tail_expected
    assert E.measure(BODY, "SINE", "+").tail_expected
    assert E.measure(BODY, "Unknown", "+").tail_expected
    assert not E.measure(BODY, "LTR", "+").tail_expected
    assert not E.measure(BODY, "DNA", "+").tail_expected
    assert not E.measure(BODY, "RC", "+").tail_expected


# ------------------------------------------------------- geometry and termini
def test_end_completeness_tolerates_a_short_miss():
    full = E.measure(BODY, "DNA", "+", consensus_start=4, consensus_end=298,
                     element_length=300)
    assert full.both_ends_complete
    truncated = E.measure(BODY, "LINE", "+", consensus_start=4000,
                          consensus_end=6010, element_length=6020)
    assert truncated.three_prime_complete and not truncated.five_prime_complete


def test_ltr_termini_are_read_in_element_orientation():
    ltr = "TG" + BODY + "CA"
    assert (E.measure(ltr, "LTR", "+").ltr_start_matches,
            E.measure(ltr, "LTR", "+").ltr_end_matches) == (2, 2)
    # TG...CA reverse-complements to TG...CA, so this also checks the flip.
    minus = E.measure(reverse_complement(ltr), "LTR", "-")
    assert (minus.ltr_start_matches, minus.ltr_end_matches) == (2, 2)


def test_a_terminal_inverted_repeat_is_recognised_and_chance_is_not():
    tir = "CAGGGGTGTCCAAAACTTTT"
    with_tir = tir + BODY + reverse_complement(tir)
    assert E.measure(with_tir, "DNA", "+").tir_identity == 1.0
    assert E.measure(BODY, "DNA", "+").tir_identity < 0.6


def test_helitron_termini():
    helitron = "TC" + BODY + "CTAG"
    m = E.measure(helitron, "RC", "+")
    assert (m.helitron_start_matches, m.helitron_end_matches) == (2, 4)


def test_transduction_is_what_lies_between_the_core_and_the_tail():
    insert = BODY + "G" * 40 + "A" * 20
    m = E.measure(insert, "LINE", "+", core_end_on_insert=len(BODY))
    assert (m.polya_len, m.transduction_len) == (20, 40)


# ------------------------------------------------------------ the live decode
def test_the_decode_finds_a_minus_strand_tail_it_used_to_miss():
    element = BODY + "A" * 30
    reference_oriented = reverse_complement(element)
    blind = _decode(reference_oriented)                      # no strand, no class
    aware = _decode(reference_oriented, "LINE", "-")
    assert blind.polyA_posterior < 0.5          # poly(T) is at the wrong end
    assert aware.polyA_posterior > 0.99
    assert aware.te_structure_log_evidence > blind.te_structure_log_evidence


def test_an_ltr_or_dna_element_gets_no_tail_or_transduction_credit():
    a_rich_end = BODY + "A" * 30
    line = _decode(a_rich_end, "LINE", "+")
    for te_class in ("LTR", "DNA", "RC"):
        other = _decode(a_rich_end, te_class, "+")
        assert other.polyA_posterior == 0.0
        assert other.transduction_posterior == 0.0
        assert other.te_structure_log_evidence < line.te_structure_log_evidence
        assert not any(seg.state in ("POLYA", "TRANSDUCTION") for seg in other.path)


def test_without_class_or_strand_the_decode_is_unchanged():
    insert = BODY + "A" * 30
    legacy = S.explain_te_sequence_structure(
        insert, "PASS_INSERT_TE_ALIGNMENT", "L1", "L1HS", 0.95,
        len(BODY) / len(insert), 1.0 - len(BODY) / len(insert), 0.0, 0.3, 0.0,
        "TE_MODEL_IN_DISTRIBUTION", 0.9)
    assert legacy == _decode(insert)
