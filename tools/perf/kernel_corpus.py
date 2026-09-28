#!/usr/bin/env python3
"""Record a region's segmentation calls, then replay them to time and check a kernel.

A byte-identical optimisation of an inner stage needs two things a whole run
is too slow to give on every edit: the stage's real inputs, and its outputs to
compare against. `record` runs PLACER in-process on a region (one worker) and
keeps every top-level `segment_event_consensus` call -- its arguments, every
reference window it read, and its result. `bench` replays them against the
code now on disk: it reports the CPU time and fails on the first result that
differs, or on a reference window the recording never saw (which means the
stage now asks different questions).

    python3 -m tools.perf.kernel_corpus record OUT.pkl BAM REF LIB --region chr1:30000001-30500000
    python3 -m tools.perf.kernel_corpus bench OUT.pkl [--repeat 3]

Run from the repository root of the snapshot being measured.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time


def record(args) -> int:
    from placer.core import segmentation as seg_module
    from placer.main import main as placer_main

    original = seg_module.segment_event_consensus
    calls: list[dict] = []
    depth = [0]

    def recording(chrom, bp_left, bp_right, alt_struct_reads, alt_ref_span_reads,
                  consensus, config, fetch_window, stats=None, _allow_revcomp_retry=True):
        if depth[0]:
            return original(chrom, bp_left, bp_right, alt_struct_reads,
                            alt_ref_span_reads, consensus, config, fetch_window, stats,
                            _allow_revcomp_retry=_allow_revcomp_retry)
        windows: dict[tuple[str, int, int], str] = {}

        def fetch(c, s, e):
            window = fetch_window(c, s, e)
            windows[(c, s, e)] = window
            return window
        depth[0] += 1
        try:
            result = original(chrom, bp_left, bp_right, alt_struct_reads,
                              alt_ref_span_reads, consensus, config, fetch, stats,
                              _allow_revcomp_retry=_allow_revcomp_retry)
        finally:
            depth[0] -= 1
        calls.append({"args": (chrom, bp_left, bp_right, alt_struct_reads,
                               alt_ref_span_reads, consensus, config),
                      "revcomp": _allow_revcomp_retry, "windows": windows,
                      "result": result})
        return result

    seg_module.segment_event_consensus = recording
    try:
        code = placer_main([args.bam, args.ref, args.lib, "--region", args.region,
                            "--threads", "1", "--output-dir", args.out_dir])
    finally:
        seg_module.segment_event_consensus = original
    with open(args.corpus, "wb") as handle:
        pickle.dump(calls, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"recorded {len(calls)} segmentation calls -> {args.corpus}", file=sys.stderr)
    return code


def bench(args) -> int:
    from placer.core import segmentation as seg_module

    with open(args.corpus, "rb") as handle:
        calls = pickle.load(handle)
    best = None
    for _ in range(args.repeat):
        started = time.process_time()
        for index, call in enumerate(calls):
            windows = call["windows"]

            def fetch(c, s, e, windows=windows, index=index):
                try:
                    return windows[(c, s, e)]
                except KeyError:
                    raise AssertionError(
                        f"call {index}: window {c}:{s}-{e} was never fetched when "
                        "recorded -- the stage asks different questions now") from None
            result = seg_module.segment_event_consensus(
                *call["args"], fetch, _allow_revcomp_retry=call["revcomp"])
            if result != call["result"]:
                print(f"DIFFERS at call {index}: {call['args'][:3]}", file=sys.stderr)
                return 1
        spent = time.process_time() - started
        best = spent if best is None else min(best, spent)
    print(f"{len(calls)} calls identical; best of {args.repeat}: {best:.2f} CPU-s")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("corpus")
    rec.add_argument("bam")
    rec.add_argument("ref")
    rec.add_argument("lib")
    rec.add_argument("--region", required=True)
    rec.add_argument("--out-dir", default="kernel_corpus_run")
    ben = sub.add_parser("bench")
    ben.add_argument("corpus")
    ben.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)
    return record(args) if args.command == "record" else bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
