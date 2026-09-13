import tempfile
import unittest
from pathlib import Path

import pysam

from placer_py.redesign.evidence.bam_local import collect_bam_evidence, collect_bam_evidence_from_handle


def make_segment(name, start, cigar, seq, mapq=60, reverse=False):
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = seq
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = mapq
    read.cigartuples = cigar
    read.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    return read


class BamLocalEvidenceTest(unittest.TestCase):
    def test_collect_bam_evidence_counts_insert_clip_and_ref_span_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bam_path = Path(tmpdir) / "tiny.bam"
            header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
            with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
                reads = [
                    make_segment("span_low", 70, [(0, 170)], "A" * 170, mapq=5),
                    make_segment("clip_right", 80, [(0, 80), (4, 40)], "G" * 120, reverse=True),
                    make_segment("span_hi", 80, [(0, 160)], "T" * 160, mapq=60),
                    make_segment("ins1", 100, [(0, 50), (1, 60), (0, 50)], "A" * 160),
                    make_segment("clip_left", 151, [(4, 40), (0, 80)], "C" * 120),
                ]
                for read in reads:
                    bam.write(read)
            pysam.index(str(bam_path))

            evidence = collect_bam_evidence(bam_path, "chr1", 151, window=75, low_mapq=20)
            with pysam.AlignmentFile(bam_path, "rb") as bam:
                handle_evidence = collect_bam_evidence_from_handle(bam, "chr1", 151, window=75, low_mapq=20)

        self.assertEqual(evidence.raw_cigar_insert_reads, 1)
        self.assertEqual(evidence.max_raw_cigar_insert_len, 60)
        self.assertEqual(evidence.left_clip_reads, 1)
        self.assertEqual(evidence.right_clip_reads, 1)
        self.assertGreaterEqual(evidence.ref_span_reads, 2)
        self.assertEqual(evidence.low_mapq_ref_span_reads, 1)
        self.assertGreater(evidence.local_depth, 0)
        self.assertGreater(evidence.forward_reads, 0)
        self.assertGreater(evidence.reverse_reads, 0)
        self.assertEqual(handle_evidence, evidence)


if __name__ == "__main__":
    unittest.main()
