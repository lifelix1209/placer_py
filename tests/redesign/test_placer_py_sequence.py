import tempfile
import unittest
from pathlib import Path

from placer_py.redesign.evidence.sequence import (
    best_exact_te_hit,
    best_kmer_te_hit,
    build_te_kmer_index,
    read_fasta_records,
    sequence_features,
)


class SequenceFeatureTest(unittest.TestCase):
    def test_sequence_features_reports_gc_entropy_and_low_complexity(self):
        features = sequence_features("AAAACCCCGGGGTTTT")

        self.assertEqual(features.length, 16)
        self.assertEqual(features.gc_fraction, 0.5)
        self.assertGreater(features.entropy, 1.9)
        self.assertLess(features.low_complexity_fraction, 0.6)

    def test_read_fasta_records_extracts_family_and_subfamily(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = Path(tmpdir) / "te.fa"
            fasta.write_text(">L1:L1-1\nACGTACGT\n>DNA\nTTTTAAAA\n", encoding="utf-8")

            records = read_fasta_records(fasta)

        self.assertEqual(records["L1:L1-1"], "ACGTACGT")
        self.assertEqual(records["DNA"], "TTTTAAAA")

    def test_best_exact_te_hit_finds_orientation_and_coverage(self):
        references = {"L1:L1-1": "GGGGACGTACGTCCCC", "DNA": "TTTTAAAA"}

        hit = best_exact_te_hit("ACGTACGT", references)

        self.assertEqual(hit.family, "L1")
        self.assertEqual(hit.subfamily, "L1-1")
        self.assertEqual(hit.identity, 1.0)
        self.assertEqual(hit.query_coverage, 1.0)
        self.assertEqual(hit.orientation, "+")

    def test_best_kmer_te_hit_breaks_ties_by_reference_order(self):
        references = {
            "First:Sub": "TTTACGTACGTAAA",
            "Second:Sub": "GGGACGTACGTCCC",
        }
        index = build_te_kmer_index(references, k=3)

        hit = best_kmer_te_hit("ACGTACGT", references, index, k=3)

        self.assertEqual(hit.family, "First")
        self.assertEqual(hit.subfamily, "Sub")
        self.assertEqual(hit.query_coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
