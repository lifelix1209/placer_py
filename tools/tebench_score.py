"""Score placer runs the way TEBench scores a caller, from their calls.vcf.

TEBench's own pipeline for a VCF-writing caller (workflow/Snakefile):
  1. `tebench.cli normalize` keeps the PASS records of at least 100 bp;
  2. RepeatMasker runs, from TEBench's pinned container and with the
     dataset's library, over every called insertion's sequence;
  3. `tebench.cli annotate --require-te` keeps a call only when TE hits cover
     >= 100 bp and >= 50% of it;
  4. `evaluate` matches one-to-one within 100 bp inside the confident regions.
This runs the same four steps -- TEBench's own CLI for 1 and 3, its pinned
container and command for 2 (`tools/dream/annotate.py`), its `evaluate` for 4
with the truth restricted to the run's region -- so the numbers are the ones
TEBench gives, without its Snakemake (whose container rule does not run under
Snakemake 8.30 here).

    python3 tools/tebench_score.py prepare RUN_DIR... [--submit]
    python3 tools/tebench_score.py score   RUN_DIR... [--pool NAME=RUN_DIR,RUN_DIR ...]
    python3 tools/tebench_score.py tebench [--partition development|holdout|all] [--callers A,B]

`prepare` writes `tebench/` inside each run directory and, with `--submit`,
sends its RepeatMasker job to SLURM. `score` needs those jobs finished. A run
directory's region is read from its `stderr.log` (`--region`), or given as
`RUN_DIR:REGION`.

`tebench` scores the call sets TEBench's own pipeline wrote
(`results/runs/<dataset>/<coverage>/r<rep>/<caller>/calls.tsv.gz`) on one
partition of the confident regions. TEBench can hold only one partition at a
time, because its `metrics.json` path has none, so this writes elsewhere. The
restriction and the evaluation are its `evaluate_calls` rule, line for line.
With `--partition all`, `--check` confirms every number against TEBench's own
`metrics.json`. A caller in SEALED_CALLERS is scored outside the development
contigs only with `--unseal-holdout`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# objective puts TEBench's `src` on the path, so tebench is imported after it,
# inside the functions that use it.
from tools.dream import annotate, objective  # noqa: E402

DATASETS = {
    "human_hg002": {
        "sample": "HG002",
        "library": objective.TEBENCH / "resources" / "human" / "dfam_human.freeze.fa",
        "truth": objective.TEBENCH / "results" / "truth" / "human_hg002" / "calls.tsv.gz",
        "confident": objective.TEBENCH / "results" / "truth" / "human_hg002" / "confident.bed",
        #: TEBench's config/datasets.yaml; the holdout is every other confident contig.
        "development_contigs": [f"chr{n}" for n in range(1, 9)],
    },
}
#: Callers whose holdout scores are pre-registered (docs/development-strategy.md,
#: "Release benchmark 1.0.0a3"): scoring one of them outside the development
#: contigs takes `--unseal-holdout`, so that it happens once and on purpose.
SEALED_CALLERS = frozenset({"placer"})
PARTITIONS = ("development", "holdout", "all")
#: TEBench's config/config.yaml `evaluation` block.
MIN_INSERTION_BP = 100
RM_MIN_COVERED_BP = 100
RM_MIN_FRACTION = 0.5
TOLERANCE_BP = 100


def _run_and_region(arg: str) -> tuple[Path, str]:
    path, _, region = arg.partition(":chr")
    run = Path(path)
    if region:
        return run, "chr" + region
    log = (run / "stderr.log").read_text(errors="replace")
    match = re.search(r"--region (\S+)", log)
    if not match:
        raise SystemExit(f"{run}: no --region in stderr.log; give RUN_DIR:REGION")
    return run, match.group(1)


def _tebench_cli(*args: str) -> None:
    env = {"PYTHONPATH": str(objective.TEBENCH / "src"), "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, "-m", "tebench.cli", *args], check=True,
                   cwd=objective.TEBENCH, env=env)


def prepare(runs: list[str], dataset: str, submit: bool, threads: int) -> int:
    from tebench.io import read_calls
    info = DATASETS[dataset]
    for arg in runs:
        run, _ = _run_and_region(arg)
        out = (run / "tebench").resolve()
        out.mkdir(exist_ok=True)
        _tebench_cli("normalize", "--input", str((run / "calls.vcf").resolve()),
                     "--format", "vcf", "--caller", "placer", "--sample", info["sample"],
                     "--min-insertion-length", str(MIN_INSERTION_BP),
                     "--output", str(out / "calls.pre_annotation.tsv.gz"),
                     "--fasta", str(out / "insertions.fa"))
        calls = read_calls(out / "calls.pre_annotation.tsv.gz")
        print(f"{run}: {len(calls)} PASS calls >= {MIN_INSERTION_BP} bp -> {out}")
        if submit:
            print("   ", annotate.submit(out, str(info["library"]), threads))
        else:
            print("   ", annotate.command(out, str(info["library"]), threads))
    return 0


def _score_one(run: Path, region: str, dataset: str) -> dict:
    from tebench.evaluate import evaluate
    from tebench.io import read_calls
    info = DATASETS[dataset]
    out = (run / "tebench").resolve()
    rm_out = out / "repeatmasker.out"
    if not rm_out.exists():
        raise SystemExit(f"{run}: RepeatMasker has not finished ({rm_out} missing)")
    _tebench_cli("annotate", "--calls", str(out / "calls.pre_annotation.tsv.gz"),
                 "--repeatmasker-out", str(rm_out),
                 "--min-covered-bp", str(RM_MIN_COVERED_BP),
                 "--min-fraction", str(RM_MIN_FRACTION), "--require-te",
                 "--output", str(out / "calls.tsv.gz"))
    calls = read_calls(out / "calls.pre_annotation.tsv.gz")
    annotated = read_calls(out / "calls.tsv.gz")
    truth = objective.load_truth(str(info["truth"]), str(info["confident"]), region)
    summary, matches, _, _ = evaluate(truth.calls, annotated, confident_regions=truth.confident,
                                      tolerance=TOLERANCE_BP)
    counts = summary["counts"]
    return {"run": str(run), "region": region, "pass_calls": len(calls),
            "te_calls": len(annotated), "tp": counts["tp"], "fp": counts["fp"],
            "fn": counts["fn"], "precision": summary["locus"]["precision"],
            "recall": summary["locus"]["recall"],
            "family": summary["classification"]["family"]["rate"]}


def _line(name: str, r: dict) -> str:
    p = "NA" if r["precision"] is None else f"{100 * r['precision']:.1f}%"
    rc = "NA" if r["recall"] is None else f"{100 * r['recall']:.1f}%"
    return (f"{name:28s} TP {r['tp']:5d}  FP {r['fp']:4d}  FN {r['fn']:5d}   "
            f"precision {p:>6s}  recall {rc:>6s}")


def score(runs: list[str], dataset: str, pools: list[str], output: str | None) -> int:
    results = {}
    for arg in runs:
        run, region = _run_and_region(arg)
        results[str(run)] = r = _score_one(run, region, dataset)
        print(_line(region, r))
    for spec in pools:
        name, _, members = spec.partition("=")
        rows = [results[str(Path(m))] for m in members.split(",")]
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        pooled = {"tp": tp, "fp": fp, "fn": fn,
                  "precision": tp / (tp + fp) if tp + fp else None,
                  "recall": tp / (tp + fn) if tp + fn else None}
        results[f"pool:{name}"] = pooled
        print(_line(f"POOLED {name}", pooled))
    if output:
        Path(output).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return 0


def partition_regions(confident, development: list[str], partition: str):
    """The confident regions of one partition, as TEBench's `evaluate_calls` cuts them."""
    if partition == "all":
        return confident
    development_set = set(development)
    selected = (development_set if partition == "development"
                else set(confident.intervals) - development_set)
    return confident.subset_contigs(selected)


def check_seal(callers: set[str], partition: str, unseal: bool) -> None:
    sealed = sorted(callers & SEALED_CALLERS)
    if sealed and partition != "development" and not unseal:
        raise SystemExit(
            f"{', '.join(sealed)}: partition '{partition}' scores the holdout, which is "
            "pre-registered (docs/development-strategy.md, 'Release benchmark 1.0.0a3'). "
            "Pass --unseal-holdout to score it, or leave the caller out with --callers.")


def tebench_runs(dataset: str, callers: set[str] | None) -> list[tuple[str, str, int, Path]]:
    """(caller, coverage, replicate, directory) of every call set TEBench's pipeline wrote."""
    runs = []
    for path in sorted((objective.TEBENCH / "results" / "runs" / dataset).glob("*/r*/*/calls.tsv.gz")):
        run = path.parent
        if callers is None or run.name in callers:
            runs.append((run.name, run.parent.parent.name, int(run.parent.name[1:]), run))
    return runs


def _coverage_key(coverage: str) -> int:
    return int(coverage[:-1]) if coverage.endswith("x") else 1 << 30


def tebench(dataset: str, partition: str, callers: set[str] | None, unseal: bool,
            check: bool, output: str | None) -> int:
    from tebench.evaluate import evaluate
    from tebench.io import read_calls
    from tebench.regions import RegionIndex
    if check and partition != "all":
        raise SystemExit("--check compares with TEBench's metrics.json, which are partition 'all'")
    runs = tebench_runs(dataset, callers)
    if not runs:
        raise SystemExit(f"no call sets under {objective.TEBENCH / 'results' / 'runs' / dataset}")
    check_seal({caller for caller, *_ in runs}, partition, unseal)
    info = DATASETS[dataset]
    truth = read_calls(info["truth"])
    regions = partition_regions(RegionIndex.from_bed(info["confident"]),
                                info["development_contigs"], partition)
    rows, mismatches = [], []
    for caller, coverage, replicate, run in runs:
        summary, _, _, _ = evaluate(truth, read_calls(run / "calls.tsv.gz"),
                                    confident_regions=regions, tolerance=TOLERANCE_BP)
        counts, locus = summary["counts"], summary["locus"]
        metrics = run / "run_metrics.json"
        resources = json.loads(metrics.read_text()) if metrics.exists() else {}
        rows.append({
            "dataset": dataset, "caller": caller, "coverage": coverage, "replicate": replicate,
            "partition": partition, "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            "precision": locus["precision"], "precision_ci95": locus["precision_ci95"],
            "recall": locus["recall"], "recall_ci95": locus["recall_ci95"], "f1": locus["f1"],
            "family_concordance": summary["classification"]["family"]["rate"],
            "genotype_concordance": summary["genotype"]["rate"],
            "cpu_hours": resources.get("cpu_hours"), "wall_seconds": resources.get("wall_seconds"),
            "peak_rss_kb": resources.get("peak_rss_kb"),
        })
        if check:
            reference = (objective.TEBENCH / "results" / "evaluation" / dataset / coverage
                         / f"r{replicate}" / caller / "metrics.json")
            if not reference.exists():
                mismatches.append(f"{caller} {coverage} r{replicate}: no {reference}")
                continue
            theirs = json.loads(reference.read_text())["counts"]
            mine = tuple(counts[k] for k in ("tp", "fp", "fn"))
            if mine != tuple(theirs[k] for k in ("tp", "fp", "fn")):
                mismatches.append(f"{caller} {coverage} r{replicate}: {mine} against "
                                  f"{tuple(theirs[k] for k in ('tp', 'fp', 'fn'))}")
    rows.sort(key=lambda r: (r["caller"], _coverage_key(r["coverage"]), r["replicate"]))
    print(f"partition {partition}: {len(rows)} call sets; mean over replicates")
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["caller"], r["coverage"]), []).append(r)
    for (caller, coverage), members in groups.items():
        means = {}
        for key in ("precision", "recall", "f1", "cpu_hours"):
            values = [m[key] for m in members if m[key] is not None]
            means[key] = f"{sum(values) / len(values):.3f}" if values else "NA"
        print(f"  {caller:10s} {coverage:>4s} n={len(members)}  P {means['precision']}  "
              f"R {means['recall']}  F1 {means['f1']}  CPU-h {means['cpu_hours']}")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(rows, indent=2) + "\n")
    if check:
        for line in mismatches:
            print("MISMATCH", line)
        print(f"check: {len(rows) - len(mismatches)}/{len(rows)} match TEBench's metrics.json")
        return 1 if mismatches else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools/tebench_score.py")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("runs", nargs="+")
    p.add_argument("--dataset", default="human_hg002", choices=sorted(DATASETS))
    p.add_argument("--submit", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("--dataset", default="human_hg002", choices=sorted(DATASETS))
    s.add_argument("--pool", action="append", default=[],
                   help="NAME=RUN_DIR,RUN_DIR: also report these runs pooled")
    s.add_argument("--output", help="write the numbers as JSON")
    t = sub.add_parser("tebench")
    t.add_argument("--dataset", default="human_hg002", choices=sorted(DATASETS))
    t.add_argument("--partition", default="development", choices=PARTITIONS)
    t.add_argument("--callers", help="comma-separated; default every caller with call sets")
    t.add_argument("--unseal-holdout", action="store_true",
                   help="allow a SEALED_CALLERS caller to be scored on the holdout")
    t.add_argument("--check", action="store_true",
                   help="with --partition all: every count must equal TEBench's metrics.json")
    t.add_argument("--output", help="write one row per call set as JSON")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.runs, args.dataset, args.submit, args.threads)
    if args.command == "tebench":
        callers = set(args.callers.split(",")) if args.callers else None
        return tebench(args.dataset, args.partition, callers, args.unseal_holdout,
                       args.check, args.output)
    return score(args.runs, args.dataset, args.pool, args.output)


if __name__ == "__main__":
    sys.exit(main())
