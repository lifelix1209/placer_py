import unittest

from placer_py.redesign.model.placer_discriminator import (
    BoundaryEvidence,
    EventExistenceEvidence,
    EventSegmentationEvidence,
    JointDecisionResult,
    LocalHypothesisPosterior,
    TeAlignmentEvidence,
    build_discriminator_evidence,
    evaluate_joint_hypotheses,
    evaluate_local_hypothesis_posterior,
)
from placer_py.redesign.models import BamEvidence, InsertionCandidate, SequenceFeatures, TeHit


class PlacerDiscriminatorModelsTest(unittest.TestCase):
    def test_evidence_defaults_are_conservative(self):
        existence = EventExistenceEvidence()
        segmentation = EventSegmentationEvidence()
        te = TeAlignmentEvidence()
        boundary = BoundaryEvidence()

        self.assertEqual(existence.best_gt, "./.")
        self.assertEqual(existence.alt_struct_reads, 0)
        self.assertFalse(segmentation.has_insert_seq)
        self.assertEqual(segmentation.qc, "NO_INSERT_SEQUENCE")
        self.assertFalse(te.pass_alignment)
        self.assertEqual(te.qc_reason, "NO_TE_ALIGNMENT")
        self.assertFalse(boundary.geometry_defined)
        self.assertEqual(boundary.qc, "BOUNDARY_NOT_EVALUATED")

    def test_decision_result_serializes_discriminator_fields(self):
        result = JointDecisionResult(
            label="TE",
            best_hypothesis="TE_UNKNOWN",
            final_qc="PASS_TE_IMPRECISE",
            posterior_qc="PASS_TE_POSTERIOR",
            te_posterior=0.75,
            non_te_posterior=0.15,
            artifact_posterior=0.10,
            te_vs_artifact_log_odds=2.0,
            te_vs_non_te_log_odds=1.6,
            emit_te_call=True,
            emit_unknown_te=True,
            emit_evidence_te_call=True,
            latent_mechanism="degraded_unknown_te",
            lfdr=0.25,
            robust_mechanistic_lfdr=0.30,
            te_alignment_qc="PASS_INSERT_TE_ALIGNMENT_UNKNOWN",
            sequence_model_label="TE_MODEL_IN_DISTRIBUTION",
            boundary_qc="BOUNDARY_NOT_EVALUATED",
            segmentation_qc="PASS_INSERT_SEQUENCE_AVAILABLE",
        )

        row = result.to_tsv_fields()

        self.assertEqual(row["final_qc"], "PASS_TE_IMPRECISE")
        self.assertEqual(row["best_hypothesis"], "TE_UNKNOWN")
        self.assertEqual(row["emit_te_call"], "1")
        self.assertEqual(row["te_vs_artifact_log_odds"], "2.000000")
        self.assertEqual(row["robust_mechanistic_lfdr"], "0.300000")

    def test_local_posterior_defaults_to_artifact_dominant(self):
        posterior = LocalHypothesisPosterior()

        self.assertLess(posterior.te, posterior.artifact)
        self.assertEqual(posterior.qc, "TE_POSTERIOR_LOW")


class PlacerDiscriminatorAdapterTest(unittest.TestCase):
    def test_build_discriminator_evidence_maps_existing_shadow_features(self):
        candidate = InsertionCandidate(
            chrom="chr1",
            pos=101,
            end=101,
            candidate_id="c1",
            svlen=120,
            filter="PASS",
            support=6,
            alt_sequence="A" * 120,
        )
        bam = BamEvidence(
            raw_cigar_insert_reads=4,
            left_clip_reads=2,
            right_clip_reads=1,
            ref_span_reads=1,
            local_depth=14,
        )
        features = SequenceFeatures(length=120, gc_fraction=0.45, entropy=1.9, low_complexity_fraction=0.05)
        hit = TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.92, orientation="+")

        evidence = build_discriminator_evidence(candidate, bam, features, hit)

        self.assertEqual(evidence.existence.alt_struct_reads, 7)
        self.assertEqual(evidence.existence.alt_indel_reads, 4)
        self.assertEqual(evidence.existence.alt_left_clip_reads, 2)
        self.assertEqual(evidence.existence.alt_right_clip_reads, 1)
        self.assertEqual(evidence.existence.ref_span_reads, 1)
        self.assertTrue(evidence.segmentation.has_insert_seq)
        self.assertEqual(evidence.segmentation.insert_len, 120)
        self.assertEqual(evidence.segmentation.qc, "PASS_INSERT_SEQUENCE_AVAILABLE")
        self.assertEqual(evidence.te_alignment.best_family, "L1")
        self.assertTrue(evidence.te_alignment.pass_alignment)
        self.assertEqual(evidence.te_alignment.qc_reason, "PASS_INSERT_TE_ALIGNMENT")
        self.assertEqual(evidence.boundary.qc, "BOUNDARY_NOT_EVALUATED")

    def test_symbolic_insert_without_sequence_is_not_resolved_te(self):
        candidate = InsertionCandidate(
            chrom="chr1",
            pos=101,
            end=101,
            candidate_id="c2",
            svlen=120,
            filter="PASS",
            support=8,
            alt_sequence=None,
        )
        evidence = build_discriminator_evidence(candidate, BamEvidence(raw_cigar_insert_reads=3), SequenceFeatures(), TeHit())

        self.assertFalse(evidence.segmentation.has_insert_seq)
        self.assertFalse(evidence.te_alignment.pass_alignment)
        self.assertEqual(evidence.te_alignment.qc_reason, "NO_TE_ALIGNMENT")


class PlacerDiscriminatorDecisionTest(unittest.TestCase):
    def test_strong_te_evidence_emits_imprecise_te_without_boundary(self):
        candidate = InsertionCandidate("chr1", 101, 101, "te1", 500, "PASS", 12, alt_sequence="A" * 500)
        evidence = build_discriminator_evidence(
            candidate,
            BamEvidence(raw_cigar_insert_reads=10, left_clip_reads=3, right_clip_reads=3, ref_span_reads=0, local_depth=20),
            SequenceFeatures(length=500, gc_fraction=0.45, entropy=1.9, low_complexity_fraction=0.05),
            TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.95, orientation="+"),
        )

        result = evaluate_joint_hypotheses(evidence)

        self.assertEqual(result.label, "TE")
        self.assertEqual(result.best_hypothesis, "TE_RESOLVED")
        self.assertEqual(result.final_qc, "PASS_TE_IMPRECISE")
        self.assertTrue(result.emit_te_call)
        self.assertGreater(result.te_posterior, result.non_te_posterior)
        self.assertGreater(result.te_vs_artifact_log_odds, 0.0)

    def test_structural_insert_with_weak_te_evidence_is_non_te(self):
        candidate = InsertionCandidate("chr1", 101, 101, "nonte1", 160, "PASS", 14, alt_sequence="C" * 160)
        evidence = build_discriminator_evidence(
            candidate,
            BamEvidence(raw_cigar_insert_reads=12, left_clip_reads=2, right_clip_reads=2, ref_span_reads=1, local_depth=20),
            SequenceFeatures(length=160, gc_fraction=0.5, entropy=1.8, low_complexity_fraction=0.1),
            TeHit(),
        )

        result = evaluate_joint_hypotheses(evidence)

        self.assertEqual(result.label, "nonTE")
        self.assertEqual(result.final_qc, "PASS_STRUCTURAL_INSERTION")
        self.assertFalse(result.emit_te_call)
        self.assertGreater(result.non_te_posterior, result.te_posterior)

    def test_ref_conflict_dominates_weak_insert_support(self):
        candidate = InsertionCandidate("chr1", 101, 101, "artifact1", 90, "PASS", 1, alt_sequence="ACGT" * 20)
        evidence = build_discriminator_evidence(
            candidate,
            BamEvidence(raw_cigar_insert_reads=1, ref_span_reads=30, low_mapq_ref_span_reads=5, local_depth=40),
            SequenceFeatures(length=80, gc_fraction=0.5, entropy=2.0, low_complexity_fraction=0.05),
            TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.20, orientation="+"),
        )

        result = evaluate_joint_hypotheses(evidence)

        self.assertEqual(result.label, "artifact")
        self.assertEqual(result.final_qc, "REFERENCE_OR_ARTIFACT")
        self.assertGreater(result.artifact_posterior, result.te_posterior)

    def test_more_te_coverage_does_not_reduce_te_posterior(self):
        candidate = InsertionCandidate("chr1", 101, 101, "mono_te", 100, "PASS", 8, alt_sequence="A" * 100)
        bam = BamEvidence(raw_cigar_insert_reads=6, ref_span_reads=2, local_depth=16)
        features = SequenceFeatures(length=100, gc_fraction=0.45, entropy=1.8, low_complexity_fraction=0.1)
        weak = build_discriminator_evidence(candidate, bam, features, TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.40))
        strong = build_discriminator_evidence(candidate, bam, features, TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.90))

        weak_p = evaluate_local_hypothesis_posterior(weak).te
        strong_p = evaluate_local_hypothesis_posterior(strong).te

        self.assertGreaterEqual(strong_p, weak_p)

    def test_more_ref_conflict_does_not_reduce_artifact_posterior(self):
        candidate = InsertionCandidate("chr1", 101, 101, "mono_artifact", 100, "PASS", 5, alt_sequence="A" * 100)
        features = SequenceFeatures(length=100, gc_fraction=0.45, entropy=1.8, low_complexity_fraction=0.1)
        hit = TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.60)
        low_ref = build_discriminator_evidence(candidate, BamEvidence(raw_cigar_insert_reads=4, ref_span_reads=1), features, hit)
        high_ref = build_discriminator_evidence(candidate, BamEvidence(raw_cigar_insert_reads=4, ref_span_reads=20), features, hit)

        low_p = evaluate_local_hypothesis_posterior(low_ref).artifact
        high_p = evaluate_local_hypothesis_posterior(high_ref).artifact

        self.assertGreaterEqual(high_p, low_p)


if __name__ == "__main__":
    unittest.main()
