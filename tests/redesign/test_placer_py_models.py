import csv
import tempfile
import unittest
from pathlib import Path

from placer_py.redesign.models import (
    BamEvidence,
    InsertionCandidate,
    SequenceFeatures,
    TeExplanation,
    TeHit,
)
from placer_py.redesign.io import write_ledger
from placer_py.redesign.model.placer_discriminator import JointDecisionResult
from placer_py.redesign.model.te_explainer import explain_candidate


class PlacerPyModelsTest(unittest.TestCase):
    def test_insertion_candidate_preserves_sniffles_fields(self):
        cand = InsertionCandidate(
            chrom="chr1",
            pos=101,
            end=101,
            candidate_id="Sniffles2.INS.1",
            svlen=75,
            filter="PASS",
            support=8,
            stdev_pos=1.5,
            stdev_len=2.5,
            alt_sequence="ACGT",
            genotype="0/1",
            source="sniffles2",
        )

        self.assertEqual(cand.chrom, "chr1")
        self.assertEqual(cand.pos0, 100)
        self.assertEqual(cand.abs_svlen, 75)
        self.assertEqual(cand.alt_sequence, "ACGT")
        self.assertEqual(cand.source, "sniffles2")

    def test_evidence_defaults_are_zero_and_serializable(self):
        evidence = BamEvidence()
        self.assertEqual(evidence.raw_cigar_insert_reads, 0)
        self.assertEqual(evidence.max_raw_cigar_insert_len, 0)
        self.assertEqual(evidence.to_tsv_fields()["ref_span_reads"], "0")

    def test_te_explanation_row_contains_probabilities(self):
        explanation = TeExplanation(
            candidate_id="c1",
            label="TE",
            te_posterior=0.8,
            non_te_posterior=0.15,
            artifact_posterior=0.05,
            lfdr=0.2,
            te_hit=TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=1.0, orientation="+"),
            sequence_features=SequenceFeatures(length=12, gc_fraction=0.5, entropy=2.0, low_complexity_fraction=0.0),
            bam_evidence=BamEvidence(raw_cigar_insert_reads=3, ref_span_reads=4),
        )

        row = explanation.to_tsv_fields()
        self.assertEqual(row["candidate_id"], "c1")
        self.assertEqual(row["label"], "TE")
        self.assertEqual(row["family"], "L1")
        self.assertEqual(row["te_posterior"], "0.800000")
        self.assertEqual(row["raw_cigar_insert_reads"], "3")


class PlacerPyWriterAndExplainerTest(unittest.TestCase):
    def test_explain_candidate_prefers_te_when_te_coverage_and_insert_support_are_high(self):
        cand = InsertionCandidate(
            chrom="chr1",
            pos=101,
            end=101,
            candidate_id="c1",
            svlen=100,
            filter="PASS",
            support=5,
        )
        hit = TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.95, orientation="+")
        evidence = BamEvidence(raw_cigar_insert_reads=4, ref_span_reads=1)
        features = SequenceFeatures(length=100, gc_fraction=0.45, entropy=1.9, low_complexity_fraction=0.1)

        explanation = explain_candidate(cand, evidence, features, hit)

        self.assertEqual(explanation.label, "TE")
        self.assertGreater(explanation.te_posterior, explanation.non_te_posterior)
        self.assertLess(explanation.lfdr, 0.5)

    def test_write_ledger_produces_stable_header(self):
        explanation = TeExplanation(
            candidate_id="c1",
            label="nonTE",
            te_posterior=0.1,
            non_te_posterior=0.8,
            artifact_posterior=0.1,
            lfdr=0.9,
            te_hit=TeHit(),
            sequence_features=SequenceFeatures(),
            bam_evidence=BamEvidence(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ledger.tsv"
            write_ledger(output, [explanation])
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(rows[0]["candidate_id"], "c1")
        self.assertEqual(rows[0]["label"], "nonTE")
        self.assertIn("te_posterior", rows[0])
        self.assertIn("raw_cigar_insert_reads", rows[0])


class PlacerPyDiscriminatorOutputTest(unittest.TestCase):
    def test_te_explanation_includes_discriminator_fields_when_available(self):
        explanation = TeExplanation(
            candidate_id="c1",
            label="TE",
            te_posterior=0.7,
            non_te_posterior=0.2,
            artifact_posterior=0.1,
            lfdr=0.3,
            te_hit=TeHit(family="L1", subfamily="L1-1", identity=1.0, query_coverage=0.9, orientation="+"),
            sequence_features=SequenceFeatures(length=100, gc_fraction=0.5, entropy=1.8, low_complexity_fraction=0.1),
            bam_evidence=BamEvidence(raw_cigar_insert_reads=5),
            discriminator=JointDecisionResult(final_qc="PASS_TE_IMPRECISE", best_hypothesis="TE_RESOLVED", emit_te_call=True),
        )

        row = explanation.to_tsv_fields()

        self.assertEqual(row["final_qc"], "PASS_TE_IMPRECISE")
        self.assertEqual(row["best_hypothesis"], "TE_RESOLVED")
        self.assertEqual(row["emit_te_call"], "1")
        self.assertIn("robust_mechanistic_lfdr", row)

    def test_write_ledger_header_contains_discriminator_fields(self):
        explanation = TeExplanation(
            candidate_id="c1",
            label="artifact",
            te_posterior=0.1,
            non_te_posterior=0.2,
            artifact_posterior=0.7,
            lfdr=0.9,
            te_hit=TeHit(),
            sequence_features=SequenceFeatures(),
            bam_evidence=BamEvidence(),
            discriminator=JointDecisionResult(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ledger.tsv"
            write_ledger(output, [explanation])
            header = output.read_text(encoding="utf-8").splitlines()[0].split("\t")

        self.assertIn("final_qc", header)
        self.assertIn("posterior_qc", header)
        self.assertIn("best_hypothesis", header)
        self.assertIn("te_alignment_qc", header)


if __name__ == "__main__":
    unittest.main()
