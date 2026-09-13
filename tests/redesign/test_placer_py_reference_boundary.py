import tempfile
import unittest
from pathlib import Path

import pysam

from placer_py.redesign.evidence.reference_boundary import collect_reference_evidence
from placer_py.redesign.models import InsertionCandidate


class ReferenceBoundaryEvidenceTest(unittest.TestCase):
    def _write_indexed_fasta(self, root: Path, text: str) -> Path:
        fasta = root / "ref.fa"
        fasta.write_text(text, encoding="utf-8")
        pysam.faidx(str(fasta))
        return fasta

    def test_reference_tsd_detection_matches_direct_repeat_between_breakpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = self._write_indexed_fasta(
                root,
                ">chr1\nAAAAATGACCCCCTGACGGGGG\n",
            )
            candidate = InsertionCandidate(
                chrom="chr1",
                pos=9,
                end=13,
                candidate_id="tsd",
                svlen=8,
                filter="PASS",
                support=6,
                alt_sequence="TGACNNNN",
            )

            with pysam.FastaFile(str(fasta)) as reference:
                evidence = collect_reference_evidence(candidate, reference, tsd_bg_p_max=0.20)

        self.assertTrue(evidence.reference_available)
        self.assertEqual(evidence.tsd_type, "DUP")
        self.assertEqual(evidence.tsd_len, 4)
        self.assertEqual(evidence.tsd_sequence, "TGAC")
        self.assertEqual(evidence.boundary_qc, "PASS_BOUNDARY_TSD")
        self.assertEqual(evidence.boundary_type, "TSD")

    def test_reference_edge_remap_trims_insert_core_and_closes_blunt_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = self._write_indexed_fasta(
                root,
                ">chr1\nAAAACCCCGGGGTTTT\n",
            )
            candidate = InsertionCandidate(
                chrom="chr1",
                pos=8,
                end=8,
                candidate_id="blunt",
                svlen=12,
                filter="PASS",
                support=8,
                alt_sequence="CCCCTTAAGGGG",
            )

            with pysam.FastaFile(str(fasta)) as reference:
                evidence = collect_reference_evidence(candidate, reference, min_anchor_len=4)

        self.assertEqual(evidence.left_anchor_len, 4)
        self.assertEqual(evidence.right_anchor_len, 4)
        self.assertEqual(evidence.insert_core_sequence, "TTAA")
        self.assertEqual(evidence.remap_qc, "PASS_REMAP_BILATERAL")
        self.assertEqual(evidence.boundary_qc, "PASS_BOUNDARY_BLUNT")
        self.assertEqual(evidence.boundary_type, "BLUNT")

    def test_contig_alias_allows_chr_prefix_difference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = self._write_indexed_fasta(root, ">1\nAAAACCCCGGGG\n")
            candidate = InsertionCandidate(
                chrom="chr1",
                pos=8,
                end=8,
                candidate_id="alias",
                svlen=4,
                filter="PASS",
                support=3,
                alt_sequence="ACGT",
            )

            with pysam.FastaFile(str(fasta)) as reference:
                evidence = collect_reference_evidence(candidate, reference)

        self.assertTrue(evidence.reference_available)
        self.assertEqual(evidence.resolved_chrom, "1")

    def test_bilateral_remap_without_insert_core_does_not_close_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = self._write_indexed_fasta(root, ">chr1\nAAAACCCCGGGGTTTT\n")
            candidate = InsertionCandidate(
                chrom="chr1",
                pos=8,
                end=8,
                candidate_id="empty-core",
                svlen=8,
                filter="PASS",
                support=5,
                alt_sequence="CCCCGGGG",
            )

            with pysam.FastaFile(str(fasta)) as reference:
                evidence = collect_reference_evidence(candidate, reference, min_anchor_len=4)

        self.assertEqual(evidence.remap_qc, "PASS_REMAP_BILATERAL")
        self.assertFalse(evidence.pair_valid)
        self.assertEqual(evidence.insert_core_sequence, "")
        self.assertEqual(evidence.boundary_qc, "BOUNDARY_REMAP_NO_INSERT_CORE")
        self.assertEqual(evidence.boundary_type, "REJECT")


if __name__ == "__main__":
    unittest.main()
