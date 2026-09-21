import unittest

from placer_py.redesign.evidence.segmental_explanation import (
    build_segmental_explanation_from_alignment,
)
from placer_py.redesign.model.mechanistic import (
    endonuclease_motif_score,
    high_precision_te_score,
    is_confident_te_insertion,
    mechanistic_te_score,
    polya_loglr,
    tsd_loglr,
)
from placer_py.redesign.model.placer_discriminator import (
    build_discriminator_evidence,
    evaluate_joint_hypotheses,
)
from placer_py.redesign.models import BamEvidence, InsertionCandidate, SequenceFeatures, TeHit


class MechanisticBlocksTest(unittest.TestCase):
    def test_endonuclease_motif_prefers_consensus(self):
        strong, ok = endonuclease_motif_score("GCGCGCTTTT", "AAGCGC")
        weak, _ = endonuclease_motif_score("GCGCGCGCGC", "GCGCGC")

        self.assertTrue(ok)
        self.assertGreater(strong, weak)

    def test_tsd_in_range_and_significant_scores_high(self):
        self.assertGreater(tsd_loglr(12, True, 0.001), tsd_loglr(0, False, 1.0))

    def test_polya_sigmoid_increases_with_tail(self):
        self.assertGreater(polya_loglr(0.5, 100), polya_loglr(0.0, 100))

    def test_high_precision_score_scales_with_identity_core_and_length(self):
        self.assertGreater(
            high_precision_te_score(0.95, 0.9, 4000),
            high_precision_te_score(0.95, 0.9, 200),
        )
        self.assertGreater(
            high_precision_te_score(0.95, 0.9, 1000),
            high_precision_te_score(0.70, 0.9, 1000),
        )

    def test_confident_gate_requires_family_length_and_coverage(self):
        self.assertTrue(is_confident_te_insertion(4000.0, 0.9, True))
        self.assertFalse(is_confident_te_insertion(4000.0, 0.9, False))  # no family
        self.assertFalse(is_confident_te_insertion(1000.0, 0.9, True))  # too few bases
        self.assertFalse(is_confident_te_insertion(4000.0, 0.5, True))  # low coverage

    def test_hallmark_count_rises_with_cooccurring_signatures(self):
        strong = mechanistic_te_score(
            endonuclease=5.0, endonuclease_evaluated=True, tsd_len=14, tsd_significant=True,
            tsd_bg_p=0.001, polya_fraction=0.4, insert_len=300, te_core_fraction=0.8,
            identity=0.9, core_len=250,
        )
        weak = mechanistic_te_score(
            endonuclease=0.0, endonuclease_evaluated=False, tsd_len=0, tsd_significant=False,
            tsd_bg_p=1.0, polya_fraction=0.0, insert_len=300, te_core_fraction=0.0,
            identity=0.0, core_len=0.0,
        )
        self.assertGreater(strong.hallmark_count, weak.hallmark_count)
        self.assertEqual(strong.qc, "MECH_TPRT_STRONG")


class HighPrecisionTierTest(unittest.TestCase):
    def _decide(self, seq, identity, coverage, family="L1"):
        candidate = InsertionCandidate("chr1", 101, 101, "c", len(seq), "PASS", 20, alt_sequence=seq)
        segmental = build_segmental_explanation_from_alignment(
            seq, identity=identity, query_coverage=coverage,
            aln_query_start=0, aln_query_end=int(len(seq) * coverage), mapq=60,
            family=family, subfamily=family,
        )
        bam = BamEvidence(raw_cigar_insert_reads=18, left_clip_reads=4, right_clip_reads=4, ref_span_reads=1, local_depth=30)
        features = SequenceFeatures(length=len(seq), gc_fraction=0.5, entropy=1.9, low_complexity_fraction=0.05)
        hit = TeHit(family=family, subfamily=family, identity=identity, query_coverage=coverage)
        evidence = build_discriminator_evidence(candidate, bam, features, hit, segmental=segmental)
        return evaluate_joint_hypotheses(evidence)

    def test_long_high_identity_family_resolved_is_high_precision(self):
        result = self._decide("ACGATCTGACTAGCTAGGCA" * 250, identity=0.92, coverage=0.95)

        self.assertTrue(result.emit_te_high_precision)
        self.assertEqual(result.final_qc, "PASS_TE_HIGH_PRECISION")
        self.assertTrue(result.emit_te_call)

    def test_short_insert_is_not_high_precision(self):
        result = self._decide("ACGATCTGACTAGCTAGGCA" * 3, identity=0.92, coverage=0.95)

        self.assertFalse(result.emit_te_high_precision)


if __name__ == "__main__":
    unittest.main()
