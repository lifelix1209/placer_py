#!/usr/bin/env python3
"""Build the mini example dataset: reference, TE library, and an aligned BAM.

WHY THIS EXISTS. Nothing in the test suite touches the real I/O path --
`tests/test_32_pipeline.py` drives `run_pipeline` with stub `StageHooks` and
synthetic reads, so pysam, BLAST and abPOA are exercised by no test at all.
Every refactor of the scan therefore lands with no coverage. This produces a
few hundred kB of input that makes the whole binary runnable end to end, with
the answer known in advance.

WHAT IS SIMULATED, and what is deliberately not. The reads are written with
the CIGAR they would have had if an aligner had placed them, rather than being
aligned: the point is to exercise PLACER, not minimap2. Sequencing error is
applied to the read bases but the CIGAR stays `M`, which is what a real
aligner produces for a substitution and is why `NM` is set from the actual
mismatch count.

The truth is in `truth.tsv` next to the outputs, so a smoke test can assert
against it rather than against a golden run of the tool itself -- otherwise
the test only pins whatever the tool happened to do the day it was written.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pysam

CHROM = "chr1"
REF_LEN = 60_000
READ_LEN = 8_000
MEAN_DEPTH = 24
ERROR_RATE = 0.008  # long-read-ish substitution rate, low enough to stay callable


def _weighted_sequence(rng: random.Random, n: int) -> str:
    """Reference-like bases: 41% GC, and never a long homopolymer.

    Uniform random ACGT would be unrealistically high-complexity, which makes
    the caller's job easier than it should be. Capping the homopolymer run
    keeps the composition gates (`te_softclip_low_complexity_*`) meaningful
    rather than trivially satisfied.
    """
    bases: list[str] = []
    weights = {"A": 0.295, "C": 0.205, "G": 0.205, "T": 0.295}
    pool = list(weights)
    probs = [weights[b] for b in pool]
    run_base, run_len = "", 0
    while len(bases) < n:
        base = rng.choices(pool, probs)[0]
        if base == run_base:
            run_len += 1
            if run_len >= 6:
                continue
        else:
            run_base, run_len = base, 1
        bases.append(base)
    return "".join(bases)


def _mutate(rng: random.Random, seq: str, divergence: float) -> str:
    """Substitutions only: subfamily divergence from the library consensus."""
    out = list(seq)
    for i, base in enumerate(out):
        if rng.random() < divergence:
            out[i] = rng.choice([b for b in "ACGT" if b != base])
    return "".join(out)


def build_te_library(rng: random.Random) -> dict[str, str]:
    """Three elements, RepeatMasker-style headers (`name#class/order`).

    `parse_te_name_parts` collapses the family onto a canonical key, so the
    `#SINE/Alu` half is what the decision policy's family compatibility keys
    on -- a library that omits it produces family `NA` and a weaker call.
    """
    alu = _weighted_sequence(rng, 281) + "A" * 20
    line1 = _weighted_sequence(rng, 1_200)
    sva = _weighted_sequence(rng, 480)
    return {
        "AluY#SINE/Alu": alu,
        "L1HS#LINE/L1": line1,
        "SVA_E#Retroposon/SVA": sva,
    }


class Insertion:
    def __init__(self, pos: int, te_name: str, te_seq: str, tsd_len: int,
                 polya_len: int, af: float, label: str) -> None:
        self.pos = pos                # 0-based reference position of the TSD start
        self.te_name = te_name
        self.te_seq = te_seq
        self.tsd_len = tsd_len
        self.polya_len = polya_len
        self.af = af                  # fraction of spanning reads carrying it
        self.label = label

    def inserted_sequence(self, reference: str) -> str:
        """TE body, poly(A), then the duplicated target site.

        The TSD is the reference segment at the insertion point appearing a
        SECOND time after the element, which is what target-primed reverse
        transcription leaves behind and what `tsd.detect` looks for.
        """
        tsd = reference[self.pos:self.pos + self.tsd_len]
        return self.te_seq + ("A" * self.polya_len) + tsd


def build_insertions(rng: random.Random, library: dict[str, str]) -> list[Insertion]:
    """Four events chosen to cover the cases the caller distinguishes."""
    alu = library["AluY#SINE/Alu"]
    line1 = library["L1HS#LINE/L1"]
    sva = library["SVA_E#Retroposon/SVA"]
    return [
        # Full-length Alu, homozygous, clean 15 bp TSD: the easy positive.
        Insertion(12_000, "AluY#SINE/Alu", _mutate(rng, alu, 0.02),
                  tsd_len=15, polya_len=28, af=0.95, label="alu_hom"),
        # 5'-truncated L1, heterozygous. Truncation is the common case and the
        # TE-body term has to tolerate it.
        Insertion(24_000, "L1HS#LINE/L1", _mutate(rng, line1[400:], 0.03),
                  tsd_len=13, polya_len=35, af=0.48, label="l1_het_5ptrunc"),
        # SVA, heterozygous, short TSD.
        Insertion(36_000, "SVA_E#Retroposon/SVA", _mutate(rng, sva, 0.025),
                  tsd_len=8, polya_len=22, af=0.45, label="sva_het"),
        # Full-length Alu again, low allele fraction: the one that should be
        # hard, and whose fate on a 24x mini dataset is genuinely informative.
        Insertion(48_000, "AluY#SINE/Alu", _mutate(rng, alu, 0.02),
                  tsd_len=16, polya_len=30, af=0.22, label="alu_low_af"),
    ]


def _apply_errors(rng: random.Random, seq: str) -> tuple[str, int]:
    out = list(seq)
    mismatches = 0
    for i, base in enumerate(out):
        if rng.random() < ERROR_RATE:
            out[i] = rng.choice([b for b in "ACGT" if b != base])
            mismatches += 1
    return "".join(out), mismatches


def write_bam(path: Path, reference: str, insertions: list[Insertion],
              rng: random.Random) -> int:
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": CHROM, "LN": len(reference)}]}

    records: list[pysam.AlignedSegment] = []
    n_reads = (len(reference) * MEAN_DEPTH) // READ_LEN
    tmp = pysam.AlignmentHeader.from_dict(header)

    for i in range(n_reads):
        start = rng.randrange(0, max(1, len(reference) - READ_LEN))
        end = start + READ_LEN

        # Does this read span an insertion site with enough flank on both
        # sides to be informative? A read that merely touches the site cannot
        # show the duplication.
        carried = None
        for ins in insertions:
            if start + 300 < ins.pos and ins.pos + ins.tsd_len + 300 < end:
                if rng.random() < ins.af:
                    carried = ins
                break

        rec = pysam.AlignedSegment(tmp)
        rec.query_name = f"read{i:05d}"
        rec.reference_id = 0
        rec.reference_start = start
        rec.mapping_quality = 60
        rec.flag = 0 if rng.random() < 0.5 else 16

        if carried is None:
            seq = reference[start:end]
            cigar = [(0, len(seq))]
        else:
            split = carried.pos + carried.tsd_len      # insert lands after the TSD
            left = reference[start:split]
            insert = carried.inserted_sequence(reference)
            right = reference[split:end]
            seq = left + insert + right
            cigar = [(0, len(left)), (1, len(insert)), (0, len(right))]

        seq, mismatches = _apply_errors(rng, seq)
        rec.query_sequence = seq
        rec.cigartuples = cigar
        rec.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
        rec.set_tag("NM", mismatches, value_type="i")
        records.append(rec)

    records.sort(key=lambda r: r.reference_start)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for rec in records:
            out.write(rec)
    pysam.index(str(path))
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the placer-py mini example dataset.")
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    reference = _weighted_sequence(rng, REF_LEN)
    library = build_te_library(rng)
    insertions = build_insertions(rng, library)

    ref_path = out / "mini_ref.fa"
    with ref_path.open("w") as handle:
        handle.write(f">{CHROM}\n")
        for i in range(0, len(reference), 60):
            handle.write(reference[i:i + 60] + "\n")
    pysam.faidx(str(ref_path))

    te_path = out / "mini_te.fa"
    with te_path.open("w") as handle:
        for name, seq in library.items():
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                handle.write(seq[i:i + 60] + "\n")

    bam_path = out / "mini.bam"
    n_reads = write_bam(bam_path, reference, insertions, rng)

    truth_path = out / "truth.tsv"
    with truth_path.open("w") as handle:
        handle.write("label\tchrom\tpos\tte_name\tte_len\ttsd_len\tpolya_len\taf\n")
        for ins in insertions:
            handle.write(f"{ins.label}\t{CHROM}\t{ins.pos}\t{ins.te_name}\t"
                         f"{len(ins.te_seq)}\t{ins.tsd_len}\t{ins.polya_len}\t{ins.af}\n")

    print(f"reference   {ref_path}  ({len(reference):,} bp)")
    print(f"TE library  {te_path}  ({len(library)} elements)")
    print(f"BAM         {bam_path}  ({n_reads} reads, ~{MEAN_DEPTH}x)")
    print(f"truth       {truth_path}  ({len(insertions)} insertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
