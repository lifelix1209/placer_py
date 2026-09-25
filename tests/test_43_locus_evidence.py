"""
Building the mechanism observation from a locus, and the shifted decoys.

On literal references and a stub TSD detector, so every number is countable.
"""

import random

import pytest

from placer.core import locus_evidence as L
from placer.core.element_structure import ElementStructure
from placer.core.te_classifier import TEAlignmentEvidence

pytestmark = pytest.mark.invariant

RNG = random.Random(17)
REFERENCE = "".join(RNG.choice("ACGT") for _ in range(6000))


def fetch(chrom, start, end):
    return REFERENCE[max(0, start):max(0, end)]


class Dup:
    def __init__(self, seq):
        self.type, self.length, self.sequence = "DUP", len(seq), seq
        self.mismatches, self.bg_p = 0, 0.001


def test_the_tsd_copy_is_removed_from_whichever_end_carries_it():
    assert L.insert_without_tsd_copy("ELEMENTAAAAAACGTACGT", "CGTACGT", 0) == "ELEMENTAAAAAA"
    assert L.insert_without_tsd_copy("CGTACGTTTTTTTELEMENT", "CGTACGT", 0) == "TTTTTTELEMENT"
    assert L.insert_without_tsd_copy("CGTACGAAAAAA", "CGTACGT", 0) == "CGTACGAAAAAA"
    assert L.insert_without_tsd_copy("ELEMENTCGTACGA", "CGTACGT", 1) == "ELEMENT"


def test_the_tsd_null_reads_the_repeat_content_of_the_tsd_span_only():
    window = "ACGT" * 5 + "CACACACACACACACACACA" + "GATTACAGGT" * 3
    assert L.repeat_fraction_at(window, 20, 40) == pytest.approx(1.0)
    assert L.repeat_fraction_at(window, 40, 60) < 0.5
    assert L.repeat_fraction_at("", 0, 10) == 0.0


def _alignment(te_class="LINE", strand="+"):
    te = TEAlignmentEvidence(best_family="L1", best_subfamily="L1HS", best_identity=0.97,
                             annotation_class=te_class, annotation_order="L1",
                             te_strand=strand, te_consensus_start=400,
                             te_consensus_end=1200, te_element_length=1200,
                             te_query_start=0, te_query_end=800, pass_=True)
    te.element_structure = ElementStructure(te_class=te_class, strand=strand,
                                            three_prime_complete=True)
    return te


def test_a_real_tsd_scores_and_its_copy_is_trimmed_before_the_tail_is_read():
    bp = 3000
    tsd = REFERENCE[bp - 14:bp]
    insert = "G" * 800 + "A" * 30 + tsd          # plus strand: tail, then the TSD copy
    scored = L.score_evaluated_locus("c", bp, bp, insert, _alignment(), 10, 10, fetch,
                                     lambda *a: Dup(tsd), with_decoys=False)
    assert scored.observation.tsd_len == 14
    assert scored.observation.polya_len == 30
    assert scored.score.terms["tsd"] > 10.0


def test_decoys_are_scored_at_shifted_breakpoints_and_summarised():
    bp = 3000
    tsd = REFERENCE[bp - 14:bp]

    def detect(chrom, left, right, insert):
        # A duplication exists only at the real junction.
        return Dup(tsd) if left == bp else None

    scored = L.score_evaluated_locus("c", bp, bp, "G" * 800 + "A" * 30 + tsd,
                                     _alignment(), 10, 10, fetch, detect)
    assert scored.decoy_count == L.DECOY_COUNT
    assert scored.observation.tsd_empirical_null == pytest.approx(1 / (L.DECOY_COUNT + 1))
    # No duplication at any decoy: each scores the "no TSD" term, and the EN
    # motif against random flanks, so the mean sits well under 1.
    assert 0.0 < scored.decoy_mean_exp_linkage < 1.0
