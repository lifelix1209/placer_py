#!/usr/bin/env python3
"""Score PLACER calls against a GIAB TE truth set built by make_giab_eval.py.

    python3 tools/score_against_truth.py --calls out/scientific.txt \\
        --truth eval/dev/truth.tsv --ledger out/evidence_ledger.tsv

WHAT IS SCORED, and the three decisions that make the number mean something.

  * ONLY truth rows with `is_te` and `in_confident_region` count. Outside
    GIAB's confident regions a "false positive" may simply be a real
    insertion GIAB did not resolve, so scoring there measures the truth set.
  * A call matches a truth insertion within `--window` bp. 500 is truvari's
    default reference distance and is used here for the same reason -- it is
    a published convention rather than a number chosen to make this caller
    look good. `--window` is reported in the output so a different choice is
    visible.
  * Non-TE PASS insertions are counted SEPARATELY, as `ambiguous`, not as
    false positives. PLACER calling a real-but-not-TE insertion is a
    different error from calling nothing at all, and folding the two together
    hides which one is happening.

The ledger is optional but worth passing: it shows how many truth insertions
were DETECTED but filtered out before the final output, which separates a
recall problem in the scan from one in the decision layer. Those are fixed in
different places.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_placer_calls(path: Path) -> list[dict]:
    """`scientific.txt` is a summary block, a blank line, then a `#`-headed TSV."""
    lines = path.read_text().splitlines()
    header_at = next((i for i, line in enumerate(lines)
                      if line.startswith("#chrom\t")), None)
    if header_at is None:
        return []
    columns = lines[header_at].lstrip("#").split("\t")
    out = []
    for line in lines[header_at + 1:]:
        if not line.strip():
            continue
        out.append(dict(zip(columns, line.split("\t"))))
    return out


def read_tsv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _match(truth_pos: int, calls: list[dict], window: int,
           used: set[int]) -> int | None:
    """Nearest unused call within the window, or None."""
    best, best_dist = None, window + 1
    for i, call in enumerate(calls):
        if i in used:
            continue
        dist = abs(int(call["pos"]) - truth_pos)
        if dist <= window and dist < best_dist:
            best, best_dist = i, dist
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()

    calls = read_placer_calls(Path(args.calls))
    truth = read_tsv(Path(args.truth))

    te_truth = [t for t in truth
                if t["is_te"] == "True" and t["in_confident_region"] == "True"]
    other_truth = [t for t in truth if t not in te_truth]

    used: set[int] = set()
    hits, misses = [], []
    for row in te_truth:
        idx = _match(int(row["pos"]), calls, args.window, used)
        if idx is None:
            misses.append(row)
        else:
            used.add(idx)
            hits.append((row, calls[idx]))

    # Unmatched calls: split into "matched a real non-TE insertion" (ambiguous)
    # and "matched nothing in the truth at all" (candidate false positive).
    ambiguous, unexplained = [], []
    for i, call in enumerate(calls):
        if i in used:
            continue
        near = any(abs(int(t["pos"]) - int(call["pos"])) <= args.window
                   for t in other_truth)
        (ambiguous if near else unexplained).append(call)

    print(f"match window: +/-{args.window} bp")
    print(f"truth: {len(te_truth)} TE insertions in confident regions "
          f"({len(other_truth)} other PASS insertions not scored as TE)")
    print(f"calls: {len(calls)}")
    print()
    recall = len(hits) / len(te_truth) if te_truth else float("nan")
    print(f"  recall          {len(hits)}/{len(te_truth)}  = {recall:.3f}")
    print(f"  matched non-TE  {len(ambiguous)}   (real insertion, not labelled TE)")
    print(f"  unexplained     {len(unexplained)}   (no truth insertion nearby)")

    if hits:
        same_family = sum(1 for t, c in hits
                          if t["te_family"].lower() in (c.get("family", "").lower(),
                                                        c.get("te", "").lower()))
        print(f"  family agrees   {same_family}/{len(hits)}")

    if args.ledger:
        ledger = read_tsv(Path(args.ledger))
        detected_but_dropped = 0
        for row in misses:
            if any(abs(int(r["pos"]) - int(row["pos"])) <= args.window
                   for r in ledger):
                detected_but_dropped += 1
        print()
        print(f"  of {len(misses)} misses, {detected_but_dropped} ARE in the ledger "
              f"(detected, then filtered) and {len(misses) - detected_but_dropped} "
              f"are not (never detected)")

    if misses:
        print("\nmissed TE insertions:")
        for row in misses:
            print(f"  {row['chrom']}:{row['pos']:>10}  {row['te_hit']:<24} "
                  f"len={row['insert_len']:>6}  gt={row['gt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
