#!/usr/bin/env python3
"""Cut a GIAB HG002 evaluation slice: ONT reads, reference, and a TE truth set.

WHY SLICE RATHER THAN DOWNLOAD. The HG002 ultra-long ONT BAM is hundreds of
GB. `samtools` reads an indexed BAM over HTTPS with range requests, so a 10 Mb
region costs seconds and a few MB. Nothing here needs a bulk download.

WHERE THE TRUTH COMES FROM, and what it is not. GIAB's Tier1 v0.6 SV callset
is not a TE callset -- it is every SV, with the inserted sequence attached.
This annotates the PASS insertions by aligning their ALT sequence against a
Dfam consensus library, and keeps the ones that are recognisably a
transposable element. So the truth is "GIAB says there is an insertion here,
and its sequence is an Alu/L1/SVA", which is the best available answer and is
NOT derived from PLACER in any way.

TWO THINGS THAT KEEP THIS HONEST.

  * Only FILTER=PASS records are used. Tier1 carries many records flagged
    `NoConsensusGT` or `LongReadHomRef`; scoring against those measures
    agreement with a filter, not with the genome.
  * A region is either DEVELOPMENT or HOLDOUT, and the script says which in
    the manifest it writes. The holdout exists so that a change can be shown
    to generalise rather than to have been fitted; if you tune against a
    holdout it stops being one, and you should cut a new region instead.

    python3 tools/make_giab_eval.py --region 21:10000000-20000000 \
        --te-library dfam_human.fa --out-dir eval/dev --role development
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

GIAB = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab"
ONT_BAM = (f"{GIAB}/data/AshkenazimTrio/HG002_NA24385_son/"
           "UCSC_Ultralong_OxfordNanopore_Promethion/"
           "HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam")
REFERENCE = f"{GIAB}/release/references/GRCh37/hs37d5.fa.gz"
TRUTH_VCF = (f"{GIAB}/release/AshkenazimTrio/HG002_NA24385_son/NIST_SV_v0.6/"
             "HG002_SVs_Tier1_v0.6.vcf.gz")
TRUTH_BED = (f"{GIAB}/release/AshkenazimTrio/HG002_NA24385_son/NIST_SV_v0.6/"
             "HG002_SVs_Tier1_v0.6.bed")

#: An insertion shorter than this is not callable by PLACER anyway
#: (`min_long_ins_for_seq_extract`), so counting it as a miss would measure
#: the threshold rather than the caller.
MIN_INSERTION_BP = 50


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""),
          file=sys.stderr)
    return subprocess.run(cmd, check=True, **kwargs)


def parse_region(region: str) -> tuple[str, int, int]:
    contig, _, span = region.partition(":")
    if not span:
        raise ValueError("region must be contig:start-end")
    start, _, end = span.partition("-")
    return contig, int(start), int(end)


def fetch_bam_slice(region: str, out: Path) -> int:
    _run(["samtools", "view", "-b", "-o", str(out), ONT_BAM, region])
    _run(["samtools", "index", str(out)])
    return int(subprocess.run(["samtools", "view", "-c", str(out)],
                              capture_output=True, text=True, check=True).stdout.strip())


def fetch_reference_contig(contig: str, out: Path) -> None:
    """The WHOLE contig, not the region.

    A sub-region FASTA would renumber the coordinates and no longer agree with
    the BAM. chr21 is 48 Mb, which is the price of keeping positions honest.
    """
    with out.open("w") as handle:
        _run(["samtools", "faidx", REFERENCE, contig], stdout=handle)
    _run(["samtools", "faidx", str(out)])


def _download(url: str, out: Path) -> None:
    if out.exists():
        print(f"  (cached) {out.name}", file=sys.stderr)
        return
    print(f"  downloading {url.rsplit('/', 1)[-1]}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=1800) as response, out.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def load_confident_intervals(bed: Path, contig: str) -> list[tuple[int, int]]:
    spans = []
    for line in bed.read_text().splitlines():
        if not line or line.startswith(("#", "track")):
            continue
        fields = line.split("\t")
        if fields[0] == contig:
            spans.append((int(fields[1]), int(fields[2])))
    return sorted(spans)


def _inside(spans: list[tuple[int, int]], pos: int) -> bool:
    lo, hi = 0, len(spans) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end = spans[mid]
        if pos < start:
            hi = mid - 1
        elif pos >= end:
            lo = mid + 1
        else:
            return True
    return False


def extract_insertions(vcf: Path, contig: str, start: int, end: int,
                       spans: list[tuple[int, int]]) -> list[dict]:
    out = []
    with gzip.open(vcf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] != contig or f[6] != "PASS":
                continue
            pos = int(f[1])
            if not (start <= pos <= end) or "SVTYPE=INS" not in f[7]:
                continue
            alt = f[4]
            if not alt or alt.startswith("<"):
                continue
            inserted = alt[len(f[3]):] if len(alt) > len(f[3]) else alt
            if len(inserted) < MIN_INSERTION_BP:
                continue
            out.append({"chrom": contig, "pos": pos, "gt": f[9].split(":")[0],
                        "insert_len": len(inserted), "insert_seq": inserted,
                        "in_confident_region": _inside(spans, pos)})
    return out


def annotate_with_te(records: list[dict], te_library: Path, work: Path) -> None:
    """Label each insertion with the Dfam family its sequence aligns to."""
    if not records:
        return
    query = work / "insertions.fa"
    with query.open("w") as handle:
        for i, rec in enumerate(records):
            handle.write(f">ins{i}\n{rec['insert_seq']}\n")

    db = work / "te_db"
    _run(["makeblastdb", "-in", str(te_library), "-dbtype", "nucl",
          "-out", str(db)], stdout=subprocess.DEVNULL)
    hits = work / "hits.tsv"
    with hits.open("w") as handle:
        _run(["blastn", "-task", "dc-megablast", "-query", str(query),
              "-db", str(db), "-outfmt", "6 qseqid sseqid pident length qlen",
              "-max_target_seqs", "5", "-evalue", "1e-5"], stdout=handle)

    best: dict[int, tuple[float, str, float]] = {}
    for line in hits.read_text().splitlines():
        q, s, pident, length, qlen = line.split("\t")
        idx = int(q[3:])
        covered = int(length) / max(1, int(qlen))
        score = covered * (float(pident) / 100.0)
        if idx not in best or score > best[idx][0]:
            best[idx] = (score, s, covered)

    for i, rec in enumerate(records):
        score, subject, covered = best.get(i, (0.0, "", 0.0))
        rec["te_hit"] = subject or "NA"
        rec["te_family"] = subject.split("#")[1].split("/")[-1] if "#" in subject else "NA"
        rec["te_coverage"] = round(covered, 4)
        # A TE insertion is one whose inserted sequence is MOSTLY the element.
        # 0.50 is a labelling threshold for the truth set, not a caller
        # parameter -- it is never tuned against PLACER's output.
        rec["is_te"] = bool(subject) and covered >= 0.50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", required=True, help="contig:start-end, GRCh37")
    parser.add_argument("--te-library", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--role", required=True, choices=("development", "holdout"),
                        help="holdout regions must not be looked at while iterating")
    parser.add_argument("--cache-dir", default="/tmp/giab")
    args = parser.parse_args()

    contig, start, end = parse_region(args.region)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] BAM slice {args.region}", file=sys.stderr)
    n_reads = fetch_bam_slice(args.region, out / "reads.bam")

    print(f"[2/5] reference contig {contig}", file=sys.stderr)
    fetch_reference_contig(contig, out / "reference.fa")

    print("[3/5] truth VCF and confident regions", file=sys.stderr)
    vcf = cache / "HG002_SVs_Tier1_v0.6.vcf.gz"
    bed = cache / "HG002_SVs_Tier1_v0.6.bed"
    _download(TRUTH_VCF, vcf)
    _download(TRUTH_BED, bed)

    print("[4/5] PASS insertions in region", file=sys.stderr)
    spans = load_confident_intervals(bed, contig)
    records = extract_insertions(vcf, contig, start, end, spans)

    print(f"[5/5] TE-annotating {len(records)} insertions", file=sys.stderr)
    annotate_with_te(records, Path(args.te_library), out)

    truth = out / "truth.tsv"
    cols = ["chrom", "pos", "gt", "insert_len", "te_hit", "te_family",
            "te_coverage", "is_te", "in_confident_region"]
    with truth.open("w") as handle:
        handle.write("\t".join(cols) + "\n")
        for rec in records:
            handle.write("\t".join(str(rec[c]) for c in cols) + "\n")

    n_te = sum(1 for r in records if r["is_te"])
    n_te_conf = sum(1 for r in records if r["is_te"] and r["in_confident_region"])
    manifest = {
        "role": args.role, "region": args.region, "reference_build": "GRCh37/hs37d5",
        "platform": "ONT-UL (HG002, UCSC 20200508, phased)",
        "truth": "GIAB Tier1 v0.6, FILTER=PASS insertions, Dfam-annotated",
        "reads": n_reads, "pass_insertions": len(records),
        "te_insertions": n_te, "te_insertions_in_confident_region": n_te_conf,
        "min_insertion_bp": MIN_INSERTION_BP,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
