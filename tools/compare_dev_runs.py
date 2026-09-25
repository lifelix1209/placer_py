#!/usr/bin/env python3
"""Compare runs of the development slices against each other and a TE truth set.

    python3 tools/compare_dev_runs.py RUN_A RUN_B [RUN_C ...] \\
        --truth .../TE_bechmark/results/truth/human_hg002/calls.tsv.gz \\
        --confident .../TE_bechmark/results/truth/human_hg002/confident.bed \\
        --region chr1:10000001-20000000

Each RUN is an `--output-dir` from `tools/run_dev_slices.sh` for ONE dataset
(e.g. `runs/6240dd0/human_hg002`). The first run is the reference: every other
run is diffed against it, locus by locus.

WHAT THIS IS FOR. A regression check between commits on a fixed slice, not a
benchmark: a 10 Mb slice holds a handful of truth insertions, so a recall of
5/6 against 6/6 is one locus, and the listing of gained and lost calls matters
more than the rates. The benchmark is TEBench.

MATCHING. A call matches a truth insertion within `--window` bp (500 by
default, truvari's reference distance), one-to-one, nearest first. Only truth
records inside the confident regions and the region are counted, and a call
outside the confident regions is neither a TP nor an FP. Structural calls (the
TE-calibrated mode's set-aside) are listed but scored separately: they are
insertions the caller did not name as a TE.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Call:
    chrom: str
    pos: int
    call_set: str          # "final" (TE calls) or "structural"
    label: str             # subfamily or family as reported
    te_class: str
    strand: str
    genotype: str
    length: int


@dataclass
class Truth:
    chrom: str
    pos: int
    te_class: str
    subfamily: str
    genotype: str
    length: int


def parse_region(text: str) -> tuple[str, int, int]:
    chrom, _, span = text.partition(":")
    if not span:
        return chrom, 0, 1 << 62
    start, _, end = span.replace(",", "").partition("-")
    return chrom, int(start) - 1, int(end)


def read_calls(run_dir: Path) -> list[Call]:
    """From calls.csv, which carries both call sets and every column."""
    import csv
    path = run_dir / "calls.csv"
    calls = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("subfamily") or row.get("te") or "NA"
            if label in ("NA", ""):
                label = row.get("family", "NA")
            calls.append(Call(
                chrom=row["chrom"], pos=int(row["pos"]), call_set=row["call_set"],
                label=label, te_class=row.get("te_annotation_class", "NA") or "NA",
                strand=row.get("strand", "NA") or "NA",
                genotype=row.get("gt", "NA") or "NA",
                length=int(float(row.get("insert_len", 0) or 0))))
    return calls


def read_truth(path: Path) -> list[Truth]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in handle:
            f = line.rstrip("\n").split("\t")
            out.append(Truth(chrom=f[col["contig"]], pos=int(f[col["pos0"]]) + 1,
                             te_class=f[col["te_class"]], subfamily=f[col["te_subfamily"]],
                             genotype=f[col["genotype"]],
                             length=int(f[col["insertion_length"]])))
    return out


def read_bed(path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    with open(path) as handle:
        for line in handle:
            f = line.split()
            if len(f) >= 3:
                out.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    for spans in out.values():
        spans.sort()
    return out


def in_bed(bed: dict[str, list[tuple[int, int]]], chrom: str, pos: int) -> bool:
    spans = bed.get(chrom, [])
    i = bisect.bisect_right(spans, (pos, 1 << 62)) - 1
    return i >= 0 and spans[i][0] <= pos - 1 < spans[i][1]


def dedupe_truth(truth: list[Truth], window: int) -> list[Truth]:
    """GIAB carries some insertions once per haplotype at the same position."""
    out: list[Truth] = []
    for t in sorted(truth, key=lambda t: (t.chrom, t.pos)):
        if out and out[-1].chrom == t.chrom and abs(out[-1].pos - t.pos) <= 10:
            continue
        out.append(t)
    return out


def match(calls: list[Call], truth: list[Truth], window: int) -> dict[int, int]:
    """One-to-one, nearest pairs first. Returns call index -> truth index."""
    pairs = sorted((abs(c.pos - t.pos), ci, ti)
                   for ci, c in enumerate(calls) for ti, t in enumerate(truth)
                   if c.chrom == t.chrom and abs(c.pos - t.pos) <= window)
    used_c, used_t, out = set(), set(), {}
    for _, ci, ti in pairs:
        if ci in used_c or ti in used_t:
            continue
        used_c.add(ci)
        used_t.add(ti)
        out[ci] = ti
    return out


def summarise(name: str, calls: list[Call], truth: list[Truth], bed, window: int) -> str:
    te = [c for c in calls if c.call_set == "final"]
    structural = [c for c in calls if c.call_set != "final"]
    te_confident = [c for c in te if bed is None or in_bed(bed, c.chrom, c.pos)]
    matched = match(te_confident, truth, window)
    any_matched = match(calls, truth, window)
    class_ok = sum(1 for ci, ti in matched.items()
                   if te_confident[ci].te_class == truth[ti].te_class)
    gt_ok = sum(1 for ci, ti in matched.items()
                if te_confident[ci].genotype == truth[ti].genotype)
    stranded = sum(1 for c in te if c.strand in ("+", "-"))
    lines = [
        f"== {name}",
        f"  TE calls {len(te)} ({len(te_confident)} in confident regions), "
        f"structural calls {len(structural)}",
        f"  truth recalled by a TE call: {len(matched)}/{len(truth)}; "
        f"by any call: {len(any_matched)}/{len(truth)}",
        f"  TE calls in confident regions with no truth within {window} bp: "
        f"{len(te_confident) - len(matched)}",
        f"  of matched: class agrees {class_ok}/{len(matched)}, "
        f"genotype agrees {gt_ok}/{len(matched)}; TE calls with a strand "
        f"{stranded}/{len(te)}",
    ]
    for ci, ti in sorted(matched.items(), key=lambda kv: te_confident[kv[0]].pos):
        c, t = te_confident[ci], truth[ti]
        lines.append(f"    TP {c.chrom}:{c.pos} {c.label} {c.te_class} {c.strand} "
                     f"GT {c.genotype}  <- truth {t.pos} {t.subfamily} {t.te_class} "
                     f"GT {t.genotype} len {t.length}")
    missed = sorted(set(range(len(truth))) - set(matched.values()))
    for ti in missed:
        t = truth[ti]
        by_structural = any(ti == v for k, v in any_matched.items()
                            if calls[k].call_set != "final")
        note = " (called as structural)" if by_structural else ""
        lines.append(f"    FN {t.chrom}:{t.pos} {t.subfamily} {t.te_class} len {t.length}{note}")
    return "\n".join(lines)


def diff(reference: list[Call], other: list[Call], window: int) -> str:
    def keyed(calls):
        return [c for c in calls if c.call_set == "final"]
    ref, oth = keyed(reference), keyed(other)
    pairs = match(oth, [Truth(c.chrom, c.pos, c.te_class, c.label, c.genotype, c.length)
                        for c in ref], window)
    kept_ref = set(pairs.values())
    lines = []
    for ci in range(len(oth)):
        if ci not in pairs:
            c = oth[ci]
            lines.append(f"    + {c.chrom}:{c.pos} {c.label} {c.te_class} len {c.length}")
    for ri in range(len(ref)):
        if ri not in kept_ref:
            c = ref[ri]
            lines.append(f"    - {c.chrom}:{c.pos} {c.label} {c.te_class} len {c.length}")
    changed = [(oth[ci], ref[ri]) for ci, ri in pairs.items()
               if oth[ci].label != ref[ri].label]
    for c, r in changed:
        lines.append(f"    ~ {c.chrom}:{c.pos} {r.label} -> {c.label}")
    return "\n".join(lines) if lines else "    (same TE call loci and labels)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--confident", type=Path)
    parser.add_argument("--region", default=None)
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()

    bed = read_bed(args.confident) if args.confident else None
    truth: list[Truth] = []
    if args.truth:
        truth = read_truth(args.truth)
        if args.region:
            chrom, start, end = parse_region(args.region)
            truth = [t for t in truth if t.chrom == chrom and start < t.pos <= end]
        if bed is not None:
            truth = [t for t in truth if in_bed(bed, t.chrom, t.pos)]
        truth = dedupe_truth(truth, args.window)

    runs = [(str(path), read_calls(path)) for path in args.runs]
    for name, calls in runs:
        print(summarise(name, calls, truth, bed, args.window))
    reference_name, reference = runs[0]
    for name, calls in runs[1:]:
        print(f"== TE-call diff, {name} against {reference_name}")
        print(diff(reference, calls, args.window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
