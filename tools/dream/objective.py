"""Score a policy's calls the way TEBench scores a caller.

The matching, the confident-region filter and the counts are TEBench's own
`tebench.evaluate.evaluate`, imported from the TEBench checkout, so a replay
score and a benchmark score cannot drift apart. One-to-one within +-100 bp;
only calls and truth inside the confident regions count. A call's position
is the VCF POS placer writes, which TEBench's normaliser reads as `pos0`:
`report.vcf.vcf_pos`, the left breakpoint `bp_left`, not the midpoint `pos`.
Replays before 2026-09-26 18:30 used `pos + 1`, which put wide-interval calls
at a midpoint the output never reports.

TEBench also re-annotates every call's insert with RepeatMasker and drops
those under its TE rule. A world annotated by `tools/dream/annotate.py` gets
the same step, through TEBench's own function. An unannotated world falls back
to the policy's own label, which counts TEBench's silent drops as false
positives.

THE REPLAY OBJECTIVE is TE recall at precision >= 0.95. Below the floor, the
value is recall minus PENALTY times the shortfall, so the search is still
pointed somewhere. Recall is TEBench's: a truth set that lists one
insertion twice counts it twice. The recall with duplicate truth ids merged
is reported beside it.

ACCEPTANCE (`compare`) is a paired bootstrap over 1 Mb blocks of the region.
Each TP and FN falls in the block of its truth position, each FP in the block
of its call, and both policies are rescored on the same resampled blocks. A
candidate is accepted when the 5th percentile of its gain is above 0. On
chr1, one truth insertion is 0.4% of recall, so a gain of one or two loci
does not pass.
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TEBENCH = Path(os.environ.get("TEBENCH",
                              "/mnt/home1/miska/hl725/scratch/projects/TE_bechmark"))
if str(TEBENCH / "src") not in sys.path:
    sys.path.insert(0, str(TEBENCH / "src"))

from tebench.evaluate import evaluate  # noqa: E402
from tebench.io import read_calls  # noqa: E402
from tebench.model import MISSING, Call  # noqa: E402
from tebench.normalize import annotate_from_repeatmasker  # noqa: E402
from tebench.regions import RegionIndex  # noqa: E402

PRECISION_FLOOR = 0.95
PENALTY = 10.0
TOLERANCE_BP = 100
BLOCK_BP = 1_000_000
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260926
ACCEPT_QUANTILE = 0.05


def parse_region(text: str) -> tuple[str, int, int]:
    """`chrom` or `chrom:start-end` (1-based, inclusive) as 0-based half-open."""
    chrom, _, span = text.partition(":")
    if not span:
        return chrom, 0, 1 << 62
    start, _, end = span.replace(",", "").partition("-")
    return chrom, int(start) - 1, int(end)


@dataclass
class Truth:
    calls: list[Call]
    confident: RegionIndex
    region: tuple[str, int, int]


def load_truth(truth_path: str, confident_path: str, region: str) -> Truth:
    chrom, start, end = parse_region(region)
    calls = [c for c in read_calls(truth_path)
             if c.contig == chrom and start <= c.pos0 < end]
    window = RegionIndex.from_intervals({chrom: [(start, min(end, 1 << 40))]})
    confident = RegionIndex.from_bed(confident_path).intersect(window)
    return Truth(calls, confident, (chrom, start, end))


#: `tebench.cli annotate --require-te` keeps a call only when its family is one of these.
_UNNAMED = {"", MISSING, "na", "n/a", "unknown", "unclassified"}


def query_calls(decisions: list, truth: Truth, label: str = "TE",
                annotation: tuple[Path, dict[int, str]] | None = None) -> list[Call]:
    chrom, start, end = truth.region
    out = []
    for i, d in enumerate(decisions):
        if d.label != label or not vcf_pass(d):
            continue
        pos0 = int(d.pos) if d.pos is not None else vcf_pos0(d.row)
        if d.row.chrom != chrom or not start <= pos0 < end:
            continue
        length = int(d.row.insert_len) if int(d.row.insert_len) > 0 else None
        sequence_id = MISSING
        if annotation is not None:
            sequence_id = annotation[1].get(d.row._row_id, MISSING)
        out.append(Call(call_id=f"q{i}", sample="HG002", caller="placer",
                        contig=str(d.row.chrom), pos0=pos0, end0=pos0,
                        insertion_length=length, te_family=d.family or ".",
                        te_class=d.te_class or ".", sequence_id=sequence_id))
    if annotation is not None:
        out = [c for c in annotate_from_repeatmasker(out, annotation[0],
                                                     min_covered_bp=100, min_fraction=0.5)
               if c.te_family.strip().lower() not in _UNNAMED]
    return out


def vcf_pass(d) -> bool:
    """Would placer write this TE call with FILTER=PASS? TEBench's normaliser
    keeps nothing else. This mirrors `placer.report.vcf.vcf_filters` for a TE
    call:
      FAM_ABSTAIN     the class is not committed (NA, Unknown, NonTE);
      IMPRECISE       an imprecise QC token, or no left breakpoint;
      ALTSEQ_MISSING  no insert sequence, where the world recorded them.
    The token set is the VCF module's own."""
    from placer.report.vcf import _IMPRECISE_TOKENS, _qc_tokens
    if (d.te_class or "NA") in ("NA", "Unknown", "NonTE"):
        return False
    bp_left = int(d.pos) if d.pos is not None else int(d.row.bp_left)
    imprecise = (d.imprecise if d.imprecise is not None
                 else any(t in _IMPRECISE_TOKENS for t in _qc_tokens(str(d.row.final_qc))))
    if bp_left < 0 or imprecise:
        return False
    seq = getattr(d.row, "insert_seq", None)
    return not (isinstance(seq, str) and seq == "" and getattr(d.row, "_has_insert_seq", False))


def vcf_pos0(row) -> int:
    """TEBench's pos0 for a call placed where its row is: placer's VCF POS
    (`placer.report.vcf.vcf_pos`), the left breakpoint when there is one."""
    bp_left = int(row.bp_left)
    if bp_left >= 1:
        return bp_left
    if bp_left == 0:
        return 1
    return int(row.pos) + 1


def value_of(precision: float | None, recall: float | None) -> float:
    p = 0.0 if precision is None else precision
    r = 0.0 if recall is None else recall
    return r if p >= PRECISION_FLOOR else r - PENALTY * (PRECISION_FLOOR - p)


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    recall_dedup: float | None
    value: float
    family_concordance: float | None
    matches: list = field(default_factory=list)
    false_positives: list = field(default_factory=list)
    false_negatives: list = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": self.precision, "recall": self.recall,
                "recall_dedup": self.recall_dedup, "value": self.value,
                "family_concordance": self.family_concordance}


def score(decisions: list, truth: Truth,
          annotation: tuple[Path, dict[int, str]] | None = None) -> Score:
    query = query_calls(decisions, truth, annotation=annotation)
    summary, matches, fps, fns = evaluate(truth.calls, query,
                                          confident_regions=truth.confident,
                                          tolerance=TOLERANCE_BP)
    counts = summary["counts"]
    locus = summary["locus"]
    scored_ids = {c.call_id for c in truth.calls
                  if truth.confident.contains(c.contig, c.pos0)}
    matched_ids = {m.truth.call_id for m in matches}
    recall_dedup = len(matched_ids) / len(scored_ids) if scored_ids else None
    family = summary["classification"]["family"]["rate"]
    return Score(tp=counts["tp"], fp=counts["fp"], fn=counts["fn"],
                 precision=locus["precision"], recall=locus["recall"],
                 recall_dedup=recall_dedup,
                 value=value_of(locus["precision"], locus["recall"]),
                 family_concordance=family, matches=matches,
                 false_positives=fps, false_negatives=fns)


def _block(call: Call) -> tuple[str, int]:
    return call.contig, call.pos0 // BLOCK_BP


def _block_counts(result: Score) -> dict[tuple[str, int], list[int]]:
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for m in result.matches:
        counts[_block(m.truth)][0] += 1
    for c in result.false_positives:
        counts[_block(c)][1] += 1
    for c in result.false_negatives:
        counts[_block(c)][2] += 1
    return counts


def _value_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return value_of(precision, recall)


@dataclass
class Comparison:
    base: Score
    candidate: Score
    gain: float
    gain_low: float
    gain_high: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {"base": self.base.as_dict(), "candidate": self.candidate.as_dict(),
                "gain": self.gain, "gain_p05": self.gain_low, "gain_p95": self.gain_high,
                "accepted": self.accepted}


def compare(base: Score, candidate: Score, truth: Truth | list[Truth]) -> Comparison:
    truths = truth if isinstance(truth, list) else [truth]
    a, b = _block_counts(base), _block_counts(candidate)
    blocks = sorted(set(a) | set(b)
                    | {_block(c) for t in truths for c in t.calls
                       if t.confident.contains(c.contig, c.pos0)})
    rng = random.Random(BOOTSTRAP_SEED)
    gains = []
    zero = [0, 0, 0]
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [rng.choice(blocks) for _ in blocks]
        ta = [sum(a.get(k, zero)[i] for k in sample) for i in range(3)]
        tb = [sum(b.get(k, zero)[i] for k in sample) for i in range(3)]
        gains.append(_value_from_counts(*tb) - _value_from_counts(*ta))
    gains.sort()
    low = gains[int(ACCEPT_QUANTILE * len(gains))]
    high = gains[int((1 - ACCEPT_QUANTILE) * len(gains)) - 1]
    gain = candidate.value - base.value
    return Comparison(base, candidate, gain, low, high, accepted=low > 0.0)


def pool(scores: list[Score]) -> Score:
    """Several worlds' scores as one: counts summed, per-call lists joined, so
    `compare` bootstraps over every world's blocks at once."""
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    dedup = [s.recall_dedup for s in scores if s.recall_dedup is not None]
    return Score(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                 recall_dedup=None if not dedup else
                 sum(d * (s.tp + s.fn) for d, s in zip(dedup, scores)) / max(1, tp + fn),
                 value=value_of(precision, recall), family_concordance=None,
                 matches=[m for s in scores for m in s.matches],
                 false_positives=[c for s in scores for c in s.false_positives],
                 false_negatives=[c for s in scores for c in s.false_negatives])


def check_invariance(policy, rows: list, q: float, **params) -> bool:
    """The policy must not read coordinates as features: shifting every row
    by the same distance, on a renamed contig, must give the same calls."""
    def key(decisions):
        return sorted((d.row._row_id, d.label) for d in decisions)

    shift = 7_777_777
    shifted = []
    for row in rows:
        moved = row.copy()
        moved.pos = int(row.pos) + shift
        moved.chrom = f"{row.chrom}_shifted"
        shifted.append(moved)
    if getattr(policy, "USES_REFERENCE", False) and params.get("reference"):
        # A policy may read the reference AT the locus: shift the reference
        # with the rows, so only reading the coordinates themselves can differ.
        base = policy._fetcher(params["reference"])
        plain = dict(params, fetch=base)

        def shifted_fetch(chrom, start, end):
            return base(chrom.removesuffix("_shifted"), start - shift, end - shift)
        moved_params = dict(params, fetch=shifted_fetch)
        return key(policy.select([r.copy() for r in rows], q, **plain)) == \
            key(policy.select(shifted, q, **moved_params))
    return key(policy.select([r.copy() for r in rows], q, **params)) == \
        key(policy.select(shifted, q, **params))
