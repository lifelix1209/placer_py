import csv
import tempfile
import unittest
from pathlib import Path

from placer_py.redesign.cli import run_shadow_pipeline

VCF_TEXT = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Support">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t101\tSniffles2.INS.1\tN\tACGTACGT\t60\tPASS\tSVTYPE=INS;SVLEN=8;END=101;SUPPORT=5
"""


class PlacerPyCliTest(unittest.TestCase):
    def test_run_shadow_pipeline_writes_ledger_without_bam(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vcf = root / "calls.vcf"
            te = root / "te.fa"
            out = root / "out"
            vcf.write_text(VCF_TEXT, encoding="utf-8")
            te.write_text(">Fam:Sub\nTTTACGTACGTAAA\n", encoding="utf-8")

            run_shadow_pipeline(vcf_path=vcf, te_fasta=te, output_dir=out, bam_path=None)

            ledger = out / "evidence_ledger_py.tsv"
            self.assertTrue(ledger.exists())
            with ledger.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_id"], "Sniffles2.INS.1")
        self.assertEqual(rows[0]["family"], "Fam")
        self.assertEqual(rows[0]["subfamily"], "Sub")

    def test_run_shadow_pipeline_uses_discriminator_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vcf = root / "calls.vcf"
            te = root / "te.fa"
            out = root / "out"
            vcf.write_text(VCF_TEXT, encoding="utf-8")
            te.write_text(">Fam:Sub\nTTTACGTACGTAAA\n", encoding="utf-8")

            run_shadow_pipeline(vcf_path=vcf, te_fasta=te, output_dir=out, bam_path=None)

            with (out / "evidence_ledger_py.tsv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(len(rows), 1)
        self.assertIn("final_qc", rows[0])
        self.assertIn("best_hypothesis", rows[0])
        self.assertIn("emit_te_call", rows[0])
        self.assertIn(
            rows[0]["final_qc"],
            {"PASS_TE_IMPRECISE", "TE_AMBIGUOUS", "PASS_STRUCTURAL_INSERTION", "REFERENCE_OR_ARTIFACT"},
        )

    def test_run_shadow_pipeline_uses_te_besthits_alignment(self):
        besthits = (
            "query\tquery_len\tte_hit\tte_len\tqstart\tqend\ttstart\ttend\tstrand\t"
            "matches\talnblock\tmapq\tidentity\tquery_cov\ttarget_cov\n"
            "Sniffles2.INS.1|chr1:101\t8\tFam:Sub\t100\t0\t8\t10\t18\t+\t8\t8\t60\t0.9\t1.0\t0.08\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vcf = root / "calls.vcf"
            te = root / "te.fa"
            bh = root / "besthits.tsv"
            out = root / "out"
            vcf.write_text(VCF_TEXT, encoding="utf-8")
            te.write_text(">Fam:Sub\nTTTACGTACGTAAA\n", encoding="utf-8")
            bh.write_text(besthits, encoding="utf-8")

            run_shadow_pipeline(vcf_path=vcf, te_fasta=te, output_dir=out, bam_path=None, te_besthits_path=bh)

            with (out / "evidence_ledger_py.tsv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(len(rows), 1)
        self.assertIn("seq_explanation_qc", rows[0])
        self.assertIn("worst_case_lfdr", rows[0])

    def test_run_shadow_pipeline_uses_reference_boundary_and_remap_evidence(self):
        vcf_text = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Support">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t8\tSniffles2.INS.2\tN\tCCCCTTAAGGGG\t60\tPASS\tSVTYPE=INS;SVLEN=12;END=8;SUPPORT=8
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vcf = root / "calls.vcf"
            te = root / "te.fa"
            reference = root / "ref.fa"
            out = root / "out"
            vcf.write_text(vcf_text, encoding="utf-8")
            te.write_text(">Fam:Sub\nAATTAAGG\n", encoding="utf-8")
            reference.write_text(">chr1\nAAAACCCCGGGGTTTT\n", encoding="utf-8")

            run_shadow_pipeline(
                vcf_path=vcf,
                te_fasta=te,
                output_dir=out,
                bam_path=None,
                reference_fasta=reference,
            )

            with (out / "evidence_ledger_py.tsv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(rows[0]["reference_qc"], "PASS_REFERENCE_CONTEXT")
        self.assertEqual(rows[0]["remap_qc"], "PASS_REMAP_BILATERAL")
        self.assertEqual(rows[0]["boundary_qc"], "PASS_BOUNDARY_BLUNT")
        self.assertEqual(rows[0]["left_ref_anchor_len"], "4")
        self.assertEqual(rows[0]["right_ref_anchor_len"], "4")
        self.assertEqual(rows[0]["insert_core_len"], "4")
        self.assertEqual(rows[0]["sequence_len"], "12")


if __name__ == "__main__":
    unittest.main()
