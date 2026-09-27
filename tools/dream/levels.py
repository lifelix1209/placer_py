"""A policy's calls, scored level by level: is there an insertion, is it a TE, which family.

PLACER is a TE caller, and TEBench's score of it is one number at the end of
three questions. This module asks them one at a time, so a loss can be put at
the level that caused it:

  LEVEL 1, DISCOVERY. Every insertion the policy selects -- TE-labelled or not,
    whatever its FILTER -- matched one-to-one within 100 bp against the TE
    truth. The target is the share of TE truth found. The same calls against
    every GIAB insertion of at least 50 bp is a diagnostic only: PLACER does not
    aim to call non-TE insertions.
  LEVEL 2, TE OR NOT. Two confusion matrices. On the level-1 matches, PLACER's
    label against the truth's (a GIAB insertion is a TE when it is in
    TEBench's TE truth). On every selected insert with a sequence, PLACER's
    label against RepeatMasker's on the same sequence -- TEBench's definition,
    applied to what PLACER assembled.
  LEVEL 3, FAMILY. Family concordance on the TE matches (TEBench's own).
  FILTER. TE-labelled matches the VCF would not write as PASS, by flag.

A DISCOVERY MISS is classified from the world's recorded rows (their
`mech_log_lr_vs_artifact`), not from what the policy scored them: for a policy
that rescores existence, "artifact ratio <= 0" is the recorded ratio's verdict.

THE WATERFALL follows each TE truth locus down ONE matching, the level-1 one:
found -> labelled TE -> PASS -> RepeatMasker calls it a TE. Every stage is a
subset of the one before, so the counts are conserved. TEBench's own score
(`objective.score`) re-matches only the PASS TE calls, so it can recover a
locus whose level-1 match was a structural call; it is reported beside the
waterfall, not in it.
"""

from __future__ import annotations

import bisect
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tools.dream import objective
from tools.dream.objective import MISSING, Call, evaluate, vcf_pass, vcf_pos0

#: The all-insertion truth for each dataset: the GIAB VCF TEBench's TE truth
#: was built from.
ALL_INSERTION_TRUTH = {
    "human_hg002": objective.TEBENCH / "resources" / "human"
    / "HG002_GRCh38_v5.0q_stvar.vcf.gz",
}
#: GIAB's structural-variant floor.
MIN_INSERTION_BP = 50
STAGES = ("te_truth", "found", "labelled_te", "pass", "repeatmasker_te")


@dataclass
class Levels:
    #: TE truth loci scored (after the haplotype merge, in confident regions),
    #: and how many survive each stage of the waterfall.
    waterfall: dict[str, int]
    #: (truth, ours) -> count, each "TE" or "SV", on the level-1 matches
    #: against every GIAB insertion. Empty without the all-insertion truth.
    confusion_truth: Counter
    #: (RepeatMasker, ours) -> count on the selected inserts with a sequence.
    #: Empty for a world without annotation.
    confusion_repeatmasker: Counter
    #: flag -> TE-labelled level-1 matches the VCF would not write as PASS.
    filter_losses: Counter
    #: reason -> TE truth loci level 1 did not find.
    discovery_misses: Counter
    #: stage -> the TE truth loci lost at that stage, with the call they lost.
    losses: dict[str, list] = field(default_factory=dict)
    #: The diagnostic: every selected insertion >= 50 bp against every GIAB
    #: insertion >= 50 bp. None without the all-insertion truth.
    all_insertions: dict | None = None


def load_all_insertion_truth(dataset: str, region: str) -> list[Call] | None:
    """GIAB's insertions of at least MIN_INSERTION_BP in `region`, or None."""
    path = ALL_INSERTION_TRUTH.get(dataset)
    if path is None or not Path(path).exists():
        return None
    from tebench.normalize import vcf_to_calls
    chrom, start, end = objective.parse_region(region)
    return [c for c in vcf_to_calls(path, caller="GIAB", sample="HG002", pass_only=False)
            if c.contig == chrom and start <= c.pos0 < end
            and (c.svtype or "").upper().startswith("INS")
            and (c.insertion_length or 0) >= MIN_INSERTION_BP]


def triaged_positions(ledger: Path) -> dict[str, list[int]]:
    """Where the scan recorded a hypothesis it did not evaluate, by contig: the
    world keeps only the evaluated rows."""
    csv.field_size_limit(sys.maxsize)
    out: dict[str, list[int]] = {}
    with open(ledger) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["candidate_retention_reason"] != "EVALUATED":
                out.setdefault(row["chrom"], []).append(int(row["pos"]))
    for positions in out.values():
        positions.sort()
    return out


def _query(decisions: list, region: tuple[str, int, int],
           min_len: int = 0) -> tuple[list[Call], dict[str, object]]:
    """Every decision as a TEBench call at the position the VCF would write."""
    chrom, start, end = region
    calls, by_id = [], {}
    for i, d in enumerate(decisions):
        pos0 = int(d.pos) if d.pos is not None else vcf_pos0(d.row)
        length = int(d.row.insert_len)
        if d.row.chrom != chrom or not start <= pos0 < end or length < min_len:
            continue
        call = Call(call_id=f"q{i}", sample="HG002", caller="placer", contig=str(d.row.chrom),
                    pos0=pos0, end0=pos0, insertion_length=length if length > 0 else None)
        calls.append(call)
        by_id[call.call_id] = d
    return calls, by_id


def repeatmasker_labels(decisions: list, annotation) -> dict[int, bool]:
    """row id -> whether RepeatMasker calls the row's insert a TE, by TEBench's
    rule, for the decisions whose insert was annotated."""
    if annotation is None:
        return {}
    out_path, sequence_of = annotation
    calls, rows = [], []
    for d in decisions:
        sid = sequence_of.get(d.row._row_id, MISSING)
        if sid == MISSING or int(d.row.insert_len) <= 0:
            continue
        calls.append(Call(call_id=str(d.row._row_id), sample="HG002", caller="placer",
                          contig=str(d.row.chrom), pos0=0, end0=0,
                          insertion_length=int(d.row.insert_len), sequence_id=sid))
        rows.append(d.row._row_id)
    annotated = objective.annotate_from_repeatmasker(calls, out_path, min_covered_bp=100,
                                                     min_fraction=0.5)
    return {row_id: c.te_family.strip().lower() not in objective._UNNAMED
            for row_id, c in zip(rows, annotated)}


def filter_flags(d) -> list[str]:
    """Why the VCF would not write this TE call as PASS (`vcf_pass`'s rules)."""
    from placer.report.vcf import _IMPRECISE_TOKENS, _qc_tokens
    flags = []
    if (d.te_class or "NA") in ("NA", "Unknown", "NonTE"):
        flags.append("FAM_ABSTAIN")
    bp_left = int(d.pos) if d.pos is not None else int(d.row.bp_left)
    imprecise = (d.imprecise if d.imprecise is not None
                 else any(t in _IMPRECISE_TOKENS for t in _qc_tokens(str(d.row.final_qc))))
    if bp_left < 0 or imprecise:
        flags.append("IMPRECISE")
    seq = getattr(d.row, "insert_seq", None)
    if isinstance(seq, str) and seq == "" and getattr(d.row, "_has_insert_seq", False):
        flags.append("ALTSEQ_MISSING")
    return flags


def _miss_reason(call: Call, rows_by_chrom, locus_of, selected_loci, triaged) -> str:
    """Why level 1 did not find a TE truth locus."""
    tol = objective.TOLERANCE_BP
    positions, rows = rows_by_chrom.get(call.contig, ([], []))
    lo = bisect.bisect_left(positions, call.pos0 - tol)
    hi = bisect.bisect_right(positions, call.pos0 + tol)
    near = rows[lo:hi]
    if not near:
        spots = triaged.get(call.contig, [])
        i = bisect.bisect_left(spots, call.pos0 - tol)
        if i < len(spots) and spots[i] <= call.pos0 + tol:
            return "triaged: no hypothesis within 100 bp reached the expensive stages"
        return "no hypothesis within 100 bp"
    if any(locus_of.get(r._row_id) in selected_loci for r in near):
        return "its locus was selected, but placed more than 100 bp away"
    if max(float(r.mech_log_lr_vs_artifact) for r in near) <= 0:
        return "not selected: artifact ratio <= 0"
    return "not selected: below the e-BH threshold"


def measure(decisions: list, rows: list, truth: objective.Truth,
            all_truth: list[Call] | None = None, annotation=None,
            triaged: dict[str, list[int]] | None = None) -> Levels:
    """The levels of one policy's `decisions` on one world's `rows`."""
    from placer.core.mechanism_selection import _loci
    triaged = triaged or {}
    query, by_id = _query(decisions, truth.region)
    summary, matches, _, misses = evaluate(truth.calls, query,
                                           confident_regions=truth.confident,
                                           tolerance=objective.TOLERANCE_BP)
    rm = repeatmasker_labels(decisions, annotation)

    waterfall = dict.fromkeys(STAGES, 0)
    waterfall["te_truth"] = summary["counts"]["tp"] + summary["counts"]["fn"]
    losses: dict[str, list] = {stage: [] for stage in STAGES[1:]}
    filter_losses: Counter = Counter()
    for m in matches:
        d = by_id[m.query.call_id]
        waterfall["found"] += 1
        if d.label != "TE":
            losses["labelled_te"].append((m.truth, d))
            continue
        waterfall["labelled_te"] += 1
        if not vcf_pass(d):
            filter_losses.update(filter_flags(d))
            losses["pass"].append((m.truth, d))
            continue
        waterfall["pass"] += 1
        if annotation is not None and not rm.get(d.row._row_id, False):
            losses["repeatmasker_te"].append((m.truth, d))
            continue
        waterfall["repeatmasker_te"] += 1

    by_chrom: dict[str, list] = {}
    for r in rows:
        by_chrom.setdefault(str(r.chrom), []).append(r)
    rows_by_chrom = {}
    for chrom, members in by_chrom.items():
        members.sort(key=lambda r: int(r.pos))
        rows_by_chrom[chrom] = ([int(r.pos) + 1 for r in members], members)
    locus_of = {r._row_id: i for i, group in enumerate(_loci(rows)) for r in group}
    selected_loci = {locus_of.get(d.row._row_id) for d in decisions}
    discovery_misses: Counter = Counter()
    for c in misses:
        why = _miss_reason(c, rows_by_chrom, locus_of, selected_loci, triaged)
        discovery_misses[why] += 1
        losses["found"].append((c, why))

    confusion_truth: Counter = Counter()
    all_insertions = None
    if all_truth is not None:
        te_positions = {(c.contig, c.pos0) for c in truth.calls}
        long_query, long_by_id = _query(decisions, truth.region, MIN_INSERTION_BP)
        s_all, m_all, _, _ = evaluate(all_truth, long_query, confident_regions=truth.confident,
                                      tolerance=objective.TOLERANCE_BP)
        for m in m_all:
            ours = "TE" if long_by_id[m.query.call_id].label == "TE" else "SV"
            theirs = "TE" if (m.truth.contig, m.truth.pos0) in te_positions else "SV"
            confusion_truth[(theirs, ours)] += 1
        all_insertions = {**s_all["counts"], "precision": s_all["locus"]["precision"],
                          "recall": s_all["locus"]["recall"]}

    confusion_repeatmasker: Counter = Counter()
    for d in decisions:
        if d.row._row_id in rm:
            confusion_repeatmasker[("TE" if rm[d.row._row_id] else "SV",
                                    "TE" if d.label == "TE" else "SV")] += 1
    return Levels(waterfall=waterfall, confusion_truth=confusion_truth,
                  confusion_repeatmasker=confusion_repeatmasker,
                  filter_losses=filter_losses, discovery_misses=discovery_misses,
                  losses=losses, all_insertions=all_insertions)


def render(levels: Levels, end_to_end: objective.Score | None = None, limit: int = 20) -> str:
    """The report `run.py levels` prints."""
    w = levels.waterfall
    lines = ["LEVEL 1  discovery (every selected insertion, no FILTER, no label)"]
    total = w["te_truth"] or 1
    lines.append(f"  TE truth found {w['found']}/{w['te_truth']} ({100 * w['found'] / total:.1f}%)")
    for why, n in levels.discovery_misses.most_common():
        lines.append(f"    missed {n:4d}  {why}")
    if levels.all_insertions is not None:
        a = levels.all_insertions
        lines.append(f"  diagnostic, every GIAB insertion >= {MIN_INSERTION_BP} bp: "
                     f"TP {a['tp']} FP {a['fp']} FN {a['fn']}  "
                     f"P {a['precision']:.3f} R {a['recall']:.3f}")
    lines.append("LEVEL 2  TE or not")
    for title, confusion, first in (("truth", levels.confusion_truth, "truth"),
                                    ("RepeatMasker", levels.confusion_repeatmasker, "RM")):
        if not confusion:
            lines.append(f"  against {title}: not available")
            continue
        agree = confusion[("TE", "TE")] + confusion[("SV", "SV")]
        n = sum(confusion.values())
        lines.append(f"  against {title} ({n}, agreement {100 * agree / n:.1f}%): "
                     f"{first} TE -> ours TE {confusion[('TE', 'TE')]}, SV {confusion[('TE', 'SV')]}; "
                     f"{first} SV -> ours TE {confusion[('SV', 'TE')]}, SV {confusion[('SV', 'SV')]}")
    lines.append("WATERFALL  (one matching, the level-1 one)")
    previous = None
    for stage in STAGES:
        lost = "" if previous is None else f"   (-{previous - w[stage]})"
        lines.append(f"  {stage:16s} {w[stage]:5d}{lost}")
        previous = w[stage]
    if levels.filter_losses:
        lines.append("  lost at PASS, by flag: " + ", ".join(
            f"{flag} {n}" for flag, n in levels.filter_losses.most_common()))
    if end_to_end is not None:
        lines.append(f"  TEBench's own score, re-matching the PASS TE calls: "
                     f"TP {end_to_end.tp} FP {end_to_end.fp}")
    for stage in STAGES[1:]:
        items = levels.losses.get(stage, [])
        if not items:
            continue
        lines.append(f"LOST AT {stage} ({len(items)})")
        for truth_call, what in items[:limit]:
            where = (f"{truth_call.contig}:{truth_call.pos0} {truth_call.te_subfamily} "
                     f"{truth_call.insertion_length}bp")
            if isinstance(what, str):
                lines.append(f"  {where}  {what}")
            else:
                r = what.row
                lines.append(f"  {where}  -> {what.label} {what.family}/{what.te_class} "
                             f"union={float(r.te_union_coverage):.2f} "
                             f"id={float(r.best_te_identity):.2f} "
                             f"art={float(r.mech_log_lr_vs_artifact):.1f} qc={r.final_qc}")
    return "\n".join(lines)
