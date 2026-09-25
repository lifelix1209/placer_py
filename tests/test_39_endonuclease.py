"""
The L1 endonuclease motif, moved out of the redesign with its strand bug fixed.

Every case is hand-built: a perfect site placed on one strand or the other at a
known breakpoint, so the right answer is determined by counting, not by a
reference implementation.
"""

import math

import pytest

from placer_py.core.endonuclease import (
    EN_CONSENSUS,
    endonuclease_motif_log_odds,
    junction_windows,
    pwm_log_odds,
)
from placer_py.core.seqtools import reverse_complement

PERFECT = 6 * math.log(0.85 / 0.25)


def test_perfect_top_strand_site_scores_the_maximum_on_plus():
    # 5'-...TTTT|AA...-3' on the top strand, breakpoint at the bar.
    left, right = "GCGCGCTTTT", "AAGCGCGC"
    score, strand = endonuclease_motif_log_odds(left, right)
    assert strand == "+"
    assert score == pytest.approx(PERFECT)


def test_perfect_bottom_strand_site_scores_the_maximum_on_minus():
    # The same site on the bottom strand reads 5'-TT|AAAA-3' on the top.
    left, right = "GCGCGCGCTT", "AAAAGCGC"
    plus, minus = junction_windows(left, right)
    assert minus == EN_CONSENSUS
    score, strand = endonuclease_motif_log_odds(left, right)
    assert strand == "-"
    assert score == pytest.approx(PERFECT)


def test_minus_window_is_the_bottom_strand_reading_of_the_junction():
    """The regression: the redesign built `revcomp(right[:4] + left[-2:])`.

    For this junction that gives AATTTT -- the right bases in the wrong order --
    and the perfect bottom-strand site above scored 4 mismatches of 6, not 0.
    """
    left, right = "GCGCGCGCTT", "AAAAGCGC"
    buggy = reverse_complement(right[:4] + left[-2:])
    assert buggy != EN_CONSENSUS
    assert pwm_log_odds(buggy) < PERFECT - 5.0
    assert junction_windows(left, right)[1] == reverse_complement(left[-2:] + right[:4])


def test_the_two_strands_are_symmetric_under_reverse_complement():
    """Reverse-complementing the whole locus swaps which flank is which and
    which strand is nicked; the best score must not change."""
    left, right = "ACGTTCTTTT", "AAGCATGA"
    rc_left, rc_right = reverse_complement(right), reverse_complement(left)
    fwd_score, fwd_strand = endonuclease_motif_log_odds(left, right)
    rev_score, rev_strand = endonuclease_motif_log_odds(rc_left, rc_right)
    assert fwd_score == pytest.approx(rev_score)
    assert {fwd_strand, rev_strand} == {"+", "-"}


def test_short_flanks_are_not_evaluated_rather_than_scored_zero():
    assert endonuclease_motif_log_odds("TTT", "AAGC") is None
    assert endonuclease_motif_log_odds("GCTTTT", "AAG") is None
    assert endonuclease_motif_log_odds("", "") is None


def test_pwm_range_and_input_checks():
    assert pwm_log_odds(EN_CONSENSUS) == pytest.approx(PERFECT)
    assert pwm_log_odds("GGGGCC") == pytest.approx(6 * math.log(0.05 / 0.25))
    assert pwm_log_odds("ttttaa") == pytest.approx(PERFECT)
    with pytest.raises(ValueError):
        pwm_log_odds("TTTTA")
