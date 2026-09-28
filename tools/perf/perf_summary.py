#!/usr/bin/env python3
"""Summarise a `PLACER_PERF_LOG` TSV (`placer/io/perf.py`).

    python3 tools/perf/perf_summary.py RUN/perf.tsv [--top 8]

Prints the whole run's CPU (the scan workers and every blastn under them), the
chunks' own CPU split into interpreter and blastn, the most expensive chunks,
and the counters that say whether the caches are doing their job: local
fetches answered from the stream against the index, reference block loads,
abPOA cache hits.
"""

from __future__ import annotations

import argparse
import csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tsv")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)
    with open(args.tsv) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    run = [row for row in rows if row["label"] == "run"]
    chunks = [row for row in rows if row["label"] != "run"] or run

    def value(row, name):
        return float(row.get(name) or 0)

    def cpu(row):
        return sum(value(row, name) for name in
                   ("user_s", "sys_s", "child_user_s", "child_sys_s"))

    def total(name):
        return sum(value(row, name) for row in chunks)

    if run:
        whole = run[0]
        print(f"run: wall {value(whole, 'wall_s'):.0f} s, CPU {cpu(whole):.0f} s "
              f"({cpu(whole) / 3600:.2f} CPU-h), peak RSS of a process "
              f"{value(whole, 'maxrss_kb') / 1e6:.2f} GB")
    print(f"chunks: {len(chunks)}; interpreter {total('user_s'):.0f} + {total('sys_s'):.0f} s, "
          f"blastn {total('child_user_s'):.0f} + {total('child_sys_s'):.0f} s (user + sys)")
    ordered = sorted(chunks, key=cpu, reverse=True)
    costs = sorted(cpu(row) for row in chunks)
    print(f"chunk CPU: median {costs[len(costs) // 2]:.0f} s, max {costs[-1]:.0f} s; "
          f"the top {args.top} hold {sum(cpu(row) for row in ordered[:args.top]):.0f} s")
    for row in ordered[:args.top]:
        print(f"  {row['label']:28s} wall {value(row, 'wall_s'):7.0f}  CPU {cpu(row):7.0f}  "
              f"peak {value(row, 'maxrss_kb') / 1e6:5.2f} GB")
    served, fallback = total("bam_buffer_served"), total("bam_buffer_fallbacks")
    poa, hits = total("poa_calls"), total("poa_cache_hits")
    print(f"local fetches: {served:.0f} from the stream, {total('bam_fetch_calls'):.0f} "
          f"from the index ({fallback:.0f} fallbacks)")
    print(f"reference: {total('ref_fetch_calls'):.0f} windows, "
          f"{total('ref_block_loads'):.0f} block loads")
    print(f"abPOA: {poa:.0f} computed, {hits:.0f} from the cache"
          + (f" ({100 * hits / (poa + hits):.0f}%)" if poa + hits else ""))
    print(f"blastn: {total('blastn_calls'):.0f} processes, {total('blastn_queries'):.0f} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
