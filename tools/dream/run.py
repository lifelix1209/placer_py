"""The replay command line. Run from the repository root:

    python3 -m tools.dream.run register NAME --path RUN_DIR --dataset human_hg002 \\
        --region chr1 --scan-commit c8ea8ae [--notes ...]
    python3 -m tools.dream.run score    WORLD POLICY [--q 0.1] [--param k=v ...]
    python3 -m tools.dream.run compare  WORLD BASE CANDIDATE [--log --parent ID --note ...]
    python3 -m tools.dream.run diagnose WORLD POLICY [--limit 40]
    python3 -m tools.dream.run levels   WORLD POLICY [--limit 20]

POLICY is a module in `tools/dream/policies/` (`current`, `coverage_rule`) or
the path of a candidate `.py` file. Every `compare --log` appends a node to the
development tree (`tree.jsonl` beside `worlds.json`), accepted or not, because
a rejected candidate is also a record of what was tried.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
import time
from pathlib import Path

from tools.dream import annotate, levels, objective, world
from tools.dream import policies as policy_module

TEBENCH_TRUTH = {
    "human_hg002": (f"{objective.TEBENCH}/results/truth/human_hg002/calls.tsv.gz",
                    f"{objective.TEBENCH}/results/truth/human_hg002/confident.bed"),
}


def _params(pairs: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for pair in pairs or []:
        key, _, text = pair.partition("=")
        out[key] = world._value(key, text)
    return out


def _policy_digest(name: str) -> str:
    module = policy_module.load(name)
    source = Path(module.__file__).read_bytes() if module.__file__ else name.encode()
    return hashlib.sha1(source).hexdigest()[:10]


def _replay(w: world.World, name: str, q: float, params: dict) -> tuple[list, float]:
    policy = policy_module.load(name)
    started = time.perf_counter()
    decisions = policy.select([row.copy() for row in w.rows], q, **params)
    return decisions, time.perf_counter() - started


def _print_score(label: str, s: objective.Score, n_te: int, n_sv: int,
                 seconds: float) -> None:
    def pct(x):
        return "NA" if x is None else f"{100 * x:.1f}%"
    print(f"{label}: TE calls {n_te}, structural {n_sv} ({seconds:.1f}s)")
    print(f"    TP {s.tp}  FP {s.fp}  FN {s.fn}   precision {pct(s.precision)}  "
          f"recall {pct(s.recall)} (dedup {pct(s.recall_dedup)})   "
          f"family {pct(s.family_concordance)}   value {s.value:.4f}")


def cmd_register(args) -> int:
    path = Path(args.registry)
    entries = world.registry(path)
    truth, confident = TEBENCH_TRUTH.get(args.dataset, ("", ""))
    entries[args.name] = {"path": str(Path(args.path).resolve()), "dataset": args.dataset,
                          "region": args.region, "scan_commit": args.scan_commit,
                          "truth": args.truth or truth,
                          "confident": args.confident or confident,
                          "notes": args.notes or ""}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"registered {args.name} -> {entries[args.name]['path']}")
    return 0


def _load(args) -> tuple[world.World, objective.Truth]:
    w = world.load(args.world, Path(args.registry))
    if w.missing:
        print(f"[dream] {w.name}: columns not recorded by scan {w.scan_commit}, "
              f"defaulted: {', '.join(w.missing)}", file=sys.stderr)
    truth = objective.load_truth(w.truth, w.confident, w.region)
    w.annotation = None if getattr(args, "no_annotation", False) else annotate.load(w)
    print(f"[dream] {w.name}: TE labels "
          + ("re-annotated by RepeatMasker, as TEBench does" if w.annotation
             else "are the policy's own (world not annotated)"), file=sys.stderr)
    return w, truth


def cmd_score(args) -> int:
    w, truth = _load(args)
    params = _params(args.param)
    decisions, seconds = _replay(w, args.policy, args.q, params)
    s = objective.score(decisions, truth, w.annotation)
    _print_score(args.policy, s, sum(d.label == "TE" for d in decisions),
                 sum(d.label == "STRUCTURAL" for d in decisions), seconds)
    if args.check_invariance:
        ok = objective.check_invariance(policy_module.load(args.policy), w.rows, args.q, **params)
        print(f"    invariance under a coordinate shift: {'ok' if ok else 'FAILED'}")
    return 0


def cmd_compare(args) -> int:
    w, truth = _load(args)
    base_params, cand_params = _params(args.base_param), _params(args.param)
    base, base_s = _replay(w, args.base, args.q, base_params)
    cand, cand_s = _replay(w, args.candidate, args.q, cand_params)
    sa = objective.score(base, truth, w.annotation)
    sb = objective.score(cand, truth, w.annotation)
    _print_score(f"base {args.base}", sa, sum(d.label == "TE" for d in base),
                 sum(d.label == "STRUCTURAL" for d in base), base_s)
    _print_score(f"cand {args.candidate}", sb, sum(d.label == "TE" for d in cand),
                 sum(d.label == "STRUCTURAL" for d in cand), cand_s)
    invariant = objective.check_invariance(policy_module.load(args.candidate), w.rows,
                                           args.q, **cand_params)
    c = objective.compare(sa, sb, truth)
    accepted = c.accepted and invariant
    print(f"gain {c.gain:+.4f}  bootstrap 90% [{c.gain_low:+.4f}, {c.gain_high:+.4f}]  "
          f"invariant {'yes' if invariant else 'NO'}  -> "
          f"{'ACCEPT' if accepted else 'reject'}")
    if args.log:
        node = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "world": w.name,
                "scan_commit": w.scan_commit, "q": args.q,
                "base": {"policy": args.base, "digest": _policy_digest(args.base),
                         "params": base_params},
                "candidate": {"policy": args.candidate,
                              "digest": _policy_digest(args.candidate),
                              "params": cand_params},
                "parent": args.parent or "", "note": args.note or "",
                "invariant": invariant, "accepted": accepted, **c.as_dict()}
        node["id"] = hashlib.sha1(json.dumps(node, sort_keys=True).encode()).hexdigest()[:10]
        tree = Path(args.registry).with_name("tree.jsonl")
        with open(tree, "a") as handle:
            handle.write(json.dumps(node, sort_keys=True) + "\n")
        print(f"logged node {node['id']} to {tree}")
    return 0


def cmd_diagnose(args) -> int:
    """Why each false negative was missed, and what each false positive is."""
    w, truth = _load(args)
    decisions, _ = _replay(w, args.policy, args.q, _params(args.param))
    s = objective.score(decisions, truth, w.annotation)
    rows = sorted(w.rows, key=lambda r: int(r.pos))
    positions = [int(r.pos) + 1 for r in rows]
    by_row = {d.row._row_id: d for d in decisions}

    def near(pos0: int) -> list:
        i = bisect.bisect_left(positions, pos0 - objective.TOLERANCE_BP)
        out = []
        while i < len(positions) and positions[i] <= pos0 + objective.TOLERANCE_BP:
            out.append(rows[i])
            i += 1
        return out

    reasons: dict[str, list] = {}
    for c in s.false_negatives:
        candidates = near(c.pos0)
        labelled = [by_row[r._row_id] for r in candidates if r._row_id in by_row]
        if not candidates:
            why = "no evaluated row within 100 bp"
        elif any(d.label == "STRUCTURAL" for d in labelled):
            why = "selected, labelled structural"
        elif labelled:
            why = "selected TE call, but matched elsewhere"
        elif max(float(r.mech_log_lr_vs_artifact) for r in candidates) <= 0:
            why = "not selected: artifact ratio <= 0"
        else:
            why = "not selected: below the e-BH threshold"
        best = max(candidates, key=lambda r: float(r.mech_log_lr_vs_artifact), default=None)
        reasons.setdefault(why, []).append((c, best))
    print(f"false negatives: {s.fn}")
    for why, items in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):4d}  {why}")
        for c, best in items[:args.limit]:
            extra = ""
            if best is not None:
                extra = (f" | row fam={best.family} cls={best.te_annotation_class} "
                         f"qcov={float(best.best_te_query_coverage):.2f} "
                         f"union={float(best.te_union_coverage):.2f} "
                         f"art={float(best.mech_log_lr_vs_artifact):.1f} "
                         f"nonte={float(best.mech_log_lr_vs_non_te):.1f} "
                         f"alt={best.alt_struct_reads} ref={best.ref_span_reads} "
                         f"cons={best.event_consensus_len}")
            print(f"        {c.contig}:{c.pos0} {c.te_subfamily} {c.insertion_length}bp"
                  f" GT {c.genotype}{extra}")
    print(f"false positives: {s.fp}")
    fp_rows = {(d.row.chrom, int(d.row.pos) + 1): d for d in decisions if d.label == "TE"}
    for c in s.false_positives[:args.limit]:
        d = fp_rows.get((c.contig, c.pos0))
        if d is None:
            continue
        r = d.row
        print(f"        {c.contig}:{c.pos0} {d.family}/{d.te_class} len={r.event_consensus_len} "
              f"qcov={float(r.best_te_query_coverage):.2f} union={float(r.te_union_coverage):.2f} "
              f"id={float(r.best_te_identity):.2f} art={float(r.mech_log_lr_vs_artifact):.1f} "
              f"alt={r.alt_struct_reads} ref={r.ref_span_reads} e={d.e_value:.3g}")
    return 0


def cmd_levels(args) -> int:
    """The policy's calls level by level (`tools/dream/levels.py`)."""
    w, truth = _load(args)
    decisions, _ = _replay(w, args.policy, args.q, _params(args.param))
    all_truth = levels.load_all_insertion_truth(w.dataset, w.region)
    if all_truth is None:
        print(f"[dream] {w.dataset}: no all-insertion truth, level 2 against truth "
              "is not available", file=sys.stderr)
    ledger = Path(w.path) / "evidence_ledger.tsv"
    result = levels.measure(decisions, w.rows, truth, all_truth, w.annotation,
                            levels.triaged_positions(ledger) if ledger.exists() else None)
    print(levels.render(result, objective.score(decisions, truth, w.annotation), args.limit))
    return 0


def cmd_validate(args) -> int:
    """One fixed candidate against the base on held-out worlds, pooled: the
    online validation of section 2, step 7. Nothing here tunes anything.

    `--base-worlds` replays the base on other worlds than the candidate, one
    per candidate world and of the same region: a candidate that needs a
    changed scan is compared with the base on the scan it replaces. The truth
    is the same, so the bootstrap still pairs the two block by block."""
    base_params, cand_params = _params(args.base_param), _params(args.param)
    base_names = args.base_worlds or args.worlds
    if len(base_names) != len(args.worlds):
        raise SystemExit("--base-worlds needs one world per --worlds entry")
    scores_a, scores_b, truths = [], [], []
    for name, base_name in zip(args.worlds, base_names):
        args.world = name
        w, truth = _load(args)
        wa = w
        if base_name != name:
            args.world = base_name
            wa, _ = _load(args)
            if (wa.region, wa.truth, wa.confident) != (w.region, w.truth, w.confident):
                raise SystemExit(f"{base_name} and {name} are not the same region and truth")
        base, _ = _replay(wa, args.base, args.q, base_params)
        cand, _ = _replay(w, args.candidate, args.q, cand_params)
        sa = objective.score(base, truth, wa.annotation)
        sb = objective.score(cand, truth, w.annotation)
        _print_score(f"{name} base", sa, sum(d.label == "TE" for d in base),
                     sum(d.label == "STRUCTURAL" for d in base), 0.0)
        _print_score(f"{name} cand", sb, sum(d.label == "TE" for d in cand),
                     sum(d.label == "STRUCTURAL" for d in cand), 0.0)
        scores_a.append(sa)
        scores_b.append(sb)
        truths.append(truth)
    pa, pb = objective.pool(scores_a), objective.pool(scores_b)
    _print_score("POOLED base", pa, 0, 0, 0.0)
    _print_score("POOLED cand", pb, 0, 0, 0.0)
    c = objective.compare(pa, pb, truths)
    print(f"pooled gain {c.gain:+.4f}  bootstrap 90% [{c.gain_low:+.4f}, {c.gain_high:+.4f}]  -> "
          f"{'PASS' if c.accepted else 'fail'}")
    if args.log:
        node = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "validation",
                "worlds": args.worlds, "base_worlds": base_names, "q": args.q,
                "base": {"policy": args.base, "params": base_params},
                "candidate": {"policy": args.candidate, "params": cand_params},
                "note": args.note or "", **c.as_dict()}
        node["id"] = hashlib.sha1(json.dumps(node, sort_keys=True).encode()).hexdigest()[:10]
        with open(Path(args.registry).with_name("tree.jsonl"), "a") as handle:
            handle.write(json.dumps(node, sort_keys=True) + "\n")
        print(f"logged node {node['id']}")
    return 0


def cmd_annotate(args) -> int:
    w = world.load(args.world, Path(args.registry))
    out, distinct, mapped = annotate.export(w)
    print(f"{w.name}: {mapped} rows, {distinct} distinct inserts >= "
          f"{annotate.MIN_INSERT_BP} bp -> {out}/insertions.fa")
    if args.submit:
        print(annotate.submit(out, args.library, args.threads))
    else:
        print(annotate.command(out, args.library, args.threads))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.dream.run")
    parser.add_argument("--registry", default=str(world.DEFAULT_REGISTRY))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("register")
    p.add_argument("name")
    p.add_argument("--path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--scan-commit", required=True)
    p.add_argument("--truth")
    p.add_argument("--confident")
    p.add_argument("--notes")
    p.set_defaults(func=cmd_register)

    for name, func in (("score", cmd_score), ("diagnose", cmd_diagnose),
                       ("levels", cmd_levels)):
        p = sub.add_parser(name)
        p.add_argument("world")
        p.add_argument("policy")
        p.add_argument("--q", type=float, default=0.10)
        p.add_argument("--param", action="append", default=[])
        p.add_argument("--limit", type=int, default=40)
        p.add_argument("--check-invariance", action="store_true")
        p.add_argument("--no-annotation", action="store_true",
                       help="score the policy's own TE labels even if the world is annotated")
        p.set_defaults(func=func)

    p = sub.add_parser("validate")
    p.add_argument("base")
    p.add_argument("candidate")
    p.add_argument("--worlds", nargs="+", required=True)
    p.add_argument("--base-worlds", nargs="+",
                   help="replay the base on these instead, one per --worlds entry")
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--base-param", action="append", default=[])
    p.add_argument("--param", action="append", default=[])
    p.add_argument("--no-annotation", action="store_true")
    p.add_argument("--log", action="store_true")
    p.add_argument("--note")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("annotate")
    p.add_argument("world")
    p.add_argument("--library", required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--submit", action="store_true")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("compare")
    p.add_argument("world")
    p.add_argument("base")
    p.add_argument("candidate")
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--base-param", action="append", default=[])
    p.add_argument("--param", action="append", default=[])
    p.add_argument("--no-annotation", action="store_true")
    p.add_argument("--log", action="store_true")
    p.add_argument("--parent")
    p.add_argument("--note")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
