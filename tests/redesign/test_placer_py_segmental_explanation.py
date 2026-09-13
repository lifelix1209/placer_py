import unittest

from placer_py.redesign.evidence.segmental_explanation import (
    build_segmental_explanation,
    build_segmental_explanation_from_alignment,
    segmental_explanation_from_features,
)
from placer_py.redesign.evidence.sequence import build_te_kmer_index
from placer_py.redesign.model.placer_discriminator import (
    build_discriminator_evidence,
    evaluate_joint_hypotheses,
    segmental_te_log_support,
)
from placer_py.redesign.models import BamEvidence, InsertionCandidate, SequenceFeatures, TeHit


TE_REFERENCE = (
    "ACGTTGCAATCGATCGGATCCTAGCTAGCATCGATCGTAGCTAGCTAGGCTAACGTGACAG"
    "TTCAGGCATGCATTAGCGCGATTACGATCGTTAGCCATGGTACCGATCGTACGATCGTAGC"
)


class SegmentalExplanationTest(unittest.TestCase):
    def setUp(self):
        self.references = {"L1:L1-1": TE_REFERENCE}
        self.index = build_te_kmer_index(self.references, k=17)

    def test_real_te_core_dominates(self):
        query = TE_REFERENCE[10:90]

        explanation = build_segmental_explanation(query, self.references, self.index, k=17)

        self.assertGreater(explanation.te_core_fraction, 0.5)
        self.assertEqual(explanation.qc, "TE_CORE_DOMINANT")
        self.assertEqual(explanation.best_family, "L1")

    def test_homopolymer_is_low_complexity_not_core(self):
        query = "A" * 120

        explanation = build_segmental_explanation(query, self.references, self.index, k=17)

        self.assertLess(explanation.te_core_fraction, 0.15)
        self.assertIn(explanation.qc, {"POLYA_TAIL_DOMINANT", "LOW_COMPLEXITY_DOMINANT"})

    def test_polya_tail_is_flagged(self):
        query = TE_REFERENCE[10:80] + "A" * 30

        explanation = build_segmental_explanation(query, self.references, self.index, k=17)

        self.assertGreater(explanation.polya_fraction, 0.1)
        self.assertGreater(explanation.te_core_fraction, 0.2)

    def test_empty_sequence_is_uninformative(self):
        explanation = build_segmental_explanation("", self.references, self.index, k=17)

        self.assertEqual(explanation.qc, "NO_INSERT_SEQUENCE")
        self.assertEqual(explanation.te_core_fraction, 0.0)

    def test_short_core_scores_below_long_core(self):
        short = build_segmental_explanation_from_alignment(
            "ACGTTGCAATCGATCGG" * 3, identity=0.95, query_coverage=1.0,
            aln_query_start=0, aln_query_end=51, mapq=60,
        )
        long = build_segmental_explanation_from_alignment(
            "ACGTTGCAATCGATCGG" * 24, identity=0.95, query_coverage=1.0,
            aln_query_start=0, aln_query_end=408, mapq=60,
        )

        self.assertLess(segmental_te_log_support(short), segmental_te_log_support(long))

    def test_alignment_identity_is_used_directly(self):
        explanation = build_segmental_explanation_from_alignment(
            TE_REFERENCE, identity=0.82, query_coverage=0.9,
            aln_query_start=0, aln_query_end=len(TE_REFERENCE), mapq=60,
        )

        self.assertAlmostEqual(explanation.identity_proxy, 0.82, places=5)

    def test_low_complexity_scores_below_real_core(self):
        real = segmental_explanation_from_features(
            query_coverage=0.9, low_complexity_fraction=0.05, pass_alignment=True
        )
        simple = build_segmental_explanation("AT" * 60, self.references, self.index, k=17)

        self.assertGreater(segmental_te_log_support(real), segmental_te_log_support(simple))


class SegmentalDiscriminatorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.references = {"L1:L1-1": TE_REFERENCE}
        self.index = build_te_kmer_index(self.references, k=17)
        self.bam = BamEvidence(raw_cigar_insert_reads=10, left_clip_reads=3, right_clip_reads=3, ref_span_reads=1, local_depth=20)

    def _decide(self, query: str):
        candidate = InsertionCandidate("chr1", 101, 101, "c", len(query), "PASS", 12, alt_sequence=query)
        segmental = build_segmental_explanation(query, self.references, self.index, k=17)
        features = SequenceFeatures(
            length=len(query),
            gc_fraction=0.5,
            entropy=1.9,
            low_complexity_fraction=segmental.low_complexity_fraction,
        )
        hit = TeHit(family=segmental.best_family, subfamily=segmental.best_subfamily, identity=1.0, query_coverage=segmental.te_core_contiguity)
        evidence = build_discriminator_evidence(candidate, self.bam, features, hit, segmental=segmental)
        return evaluate_joint_hypotheses(evidence)

    def test_low_complexity_insert_is_not_resolved_te(self):
        result = self._decide("A" * 200)

        self.assertFalse(result.emit_te_call)

    def test_real_te_core_insert_is_resolved_te(self):
        result = self._decide(TE_REFERENCE * 2)

        self.assertTrue(result.emit_te_call)
        self.assertEqual(result.label, "TE")


if __name__ == "__main__":
    unittest.main()
