import tempfile
import unittest
from pathlib import Path

from placer_py.redesign.candidates.sniffles_vcf import iter_sniffles_insertions


VCF_TEXT = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##contig=<ID=chr2,length=1000>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Support">
##INFO=<ID=STDEV_POS,Number=1,Type=Float,Description="Position stdev">
##INFO=<ID=STDEV_LEN,Number=1,Type=Float,Description="Length stdev">
##INFO=<ID=COVERAGE,Number=.,Type=Float,Description="Coverage">
##INFO=<ID=STRAND,Number=1,Type=String,Description="Strand">
##FILTER=<ID=GT,Description="Genotype filter">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr1\t101\tSniffles2.INS.1\tN\tACGTACGT\t60\tPASS\tSVTYPE=INS;SVLEN=8;END=101;SUPPORT=5;STDEV_POS=0.5;STDEV_LEN=1.5;COVERAGE=10,11,12,13,14;STRAND=+-\tGT\t0/1
chr1\t200\tSniffles2.DEL.1\tN\t<DEL>\t60\tPASS\tSVTYPE=DEL;SVLEN=-50;END=250;SUPPORT=4\tGT\t0/1
chr2\t300\tSniffles2.INS.2\tN\t<INS>\t60\tGT\tSVTYPE=INS;SVLEN=75;END=300;SUPPORT=3\tGT\t./.
"""


class SnifflesVcfParserTest(unittest.TestCase):
    def test_iter_sniffles_insertions_keeps_only_ins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcf_path = Path(tmpdir) / "calls.vcf"
            vcf_path.write_text(VCF_TEXT, encoding="utf-8")

            candidates = list(iter_sniffles_insertions(vcf_path))

        self.assertEqual(len(candidates), 2)
        first = candidates[0]
        self.assertEqual(first.chrom, "chr1")
        self.assertEqual(first.pos, 101)
        self.assertEqual(first.end, 101)
        self.assertEqual(first.candidate_id, "Sniffles2.INS.1")
        self.assertEqual(first.svlen, 8)
        self.assertEqual(first.filter, "PASS")
        self.assertEqual(first.support, 5)
        self.assertEqual(first.stdev_pos, 0.5)
        self.assertEqual(first.stdev_len, 1.5)
        self.assertEqual(first.coverage, "10,11,12,13,14")
        self.assertEqual(first.strand, "+-")
        self.assertEqual(first.alt_sequence, "ACGTACGT")
        self.assertEqual(first.genotype, "0/1")

        second = candidates[1]
        self.assertEqual(second.filter, "GT")
        self.assertIsNone(second.alt_sequence)
        self.assertEqual(second.genotype, "./.")


if __name__ == "__main__":
    unittest.main()
