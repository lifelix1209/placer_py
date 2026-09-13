import unittest

from placer_py.redesign.model.placer_discriminator import (
    build_discriminator_evidence,
    evaluate_robust_local_fdr,
)
from placer_py.redesign.models import BamEvidence, InsertionCandidate, SequenceFeatures, TeHit


def _evidence(ref_span: int, coverage: float, alt: int = 8, low_complexity: float = 0.05):
    candidate = InsertionCandidate("chr1", 101, 101, "c", 300, "PASS", alt, alt_sequence="X" * 300)
    bam = BamEvidence(raw_cigar_insert_reads=alt, ref_span_reads=ref_span, local_depth=alt + ref_span + 2)
    features = SequenceFeatures(length=300, gc_fraction=0.5, entropy=1.9, low_complexity_fraction=low_complexity)
    hit = TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=coverage, orientation="+")
    return build_discriminator_evidence(candidate, bam, features, hit)


class RobustLocalFdrTest(unittest.TestCase):
    def test_worst_case_lfdr_non_decreasing_in_ref_span(self):
        low = evaluate_robust_local_fdr(_evidence(ref_span=0, coverage=0.9)).worst_case_lfdr
        high = evaluate_robust_local_fdr(_evidence(ref_span=25, coverage=0.9)).worst_case_lfdr

        self.assertGreaterEqual(high, low)

    def test_worst_case_lfdr_non_increasing_in_te_coverage(self):
        weak = evaluate_robust_local_fdr(_evidence(ref_span=2, coverage=0.35)).worst_case_lfdr
        strong = evaluate_robust_local_fdr(_evidence(ref_span=2, coverage=0.95)).worst_case_lfdr

        self.assertLessEqual(strong, weak)

    def test_worst_case_lfdr_is_at_least_point_lfdr(self):
        robust = evaluate_robust_local_fdr(_evidence(ref_span=3, coverage=0.8))

        self.assertGreaterEqual(robust.worst_case_lfdr, robust.lfdr)

    def test_strong_evidence_passes_target_q(self):
        robust = evaluate_robust_local_fdr(_evidence(ref_span=0, coverage=0.95, alt=12))

        self.assertLessEqual(robust.worst_case_lfdr, robust.target_q)
        self.assertEqual(robust.lfdr_qc, "PASS_TE_LFDR")

    def test_ref_dominated_candidate_is_rejected(self):
        robust = evaluate_robust_local_fdr(_evidence(ref_span=30, coverage=0.2, alt=1))

        self.assertGreater(robust.worst_case_lfdr, robust.target_q)
        self.assertEqual(robust.lfdr_qc, "TE_LFDR_REJECT")


if __name__ == "__main__":
    unittest.main()
