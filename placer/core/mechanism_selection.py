"""Whole-run selection on the likelihood ratios: decoy check, e-BH, TE rule.

THE STEPS, over every evaluated hypothesis the scan recorded:

  1. ALIGNMENT-COLLAPSE REGIONS. Where the sample's sequence does not fit the
     reference -- a centromere model, a satellite array -- no read spans the
     reference, and every hypothesis looks like a homozygous insertion. The
     counts model's null does not hold there, so those e-values are not valid.
     They are set to 0 and stay in m. This is the conservative choice: an
     e-value of 0 is valid whatever rule chose it, and removing the loci from m
     instead cost the same on HG002 chr1. On chr1 this is the pericentromere
     (24% of the evaluated rows, 1,323 structural calls, no confident bases);
     on the cichlid slice it is nothing. It was adopted 2026-09-26 as a
     validity fix (docs/development-strategy.md, section 2).
  2. DECOY CHECK, per class. Each row carries the mean of exp(linkage) over its
     shifted-breakpoint decoys (`core/locus_evidence.py`). Pooled over a class,
     that estimates E_null[exp(linkage)], which is at most 1 when the TSD and
     endonuclease-motif nulls hold on this genome. A one-sided upper bound
     above 1 means they do not, and the class's artifact ratio is divided by it
     (in log space, reduced by log of it): e/c is still an e-value when c
     bounds its null expectation, so FDR control survives at a cost in recall.
     The shifted null is not contaminated by true insertions -- a decoy is
     where the insertion is not -- which is what defeated calibrating on the
     candidates themselves (README, "The blocker").
  3. LOCI. Rows within LOCUS_MERGE_BP of each other are one locus, and
     consecutive loci whose alleles overlap are one locus too
     (`_merge_by_allele`). Each is tested once. Its e-value is the MEAN of its
     rows' e-values: the max is not an e-value (under the null its mean can
     reach the number of rows, so a locus evaluated as many hypotheses would
     get that many chances), and the mean is one under any dependence. Its
     best-scoring row by the row's OWN ratio represents it: names, labels and
     places the call.
  4. ONE e-BH over all loci, on the decoy-adjusted exp of the ALLELE's
     artifact ratio (`allele_log_lr_vs_artifact`): the insertions. FDR is
     controlled on "an insertion is here".
  5. TEBench's rule on each selected insert: a TE call when TE hits cover at
     least half of it and at least 100 bp, and the alt reads measure the
     insertion at 100 bp or more (`measured_insert_length`); a structural call
     otherwise.
  6. Precise placement (`_placement`).

TESTING AN ALLELE, NOT A WINDOW (round 20, 2026-10-02; docs/development-
strategy.md, section 8). In a tandem repeat the aligner puts one insertion at
different offsets in different reads, and a row's +-25 bp tally counts the
carriers it misses as reference. The scan's same-sequence tally
(`events.collect_allele_evidence`, `byseq`) gathers them; where it finds
carriers beyond the row's own, the allele's counts term replaces the row's
own in the ratio the locus is TESTED on. Which row represents the locus, and
so its label, family and position, is still chosen by the rows' own ratios:
letting the allele term choose it moved 51 true calls on the whole-genome
DREAM contigs to structural or misplaced rows.

THE MEASURED-LENGTH FLOOR (same round). The assembly inflates short
insertions: 19 of 26 FPs whose GIAB insertion is not TE truth were 62-99 bp
in GIAB and 108-227 bp assembled. An insertion its own reads measure at under
100 bp cannot hold the rule's 100 bp of TE. Reads are biased short (ONT
deletions, about 9%), so the floor errs towards fewer TE calls.

The vs_non_te ratio is reported as the transposition evidence, not a gate: on
HG002 chr1 gating on it lost the SVAs, whose identity to the library's
consensus is 0.77-0.84, and kept inserts TEBench does not call TEs.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field

from placer.core.ledger import EvidenceLedgerRow
from placer.core.selection import ebh_select

#: Rows this close are one locus. 50 bp split one Alu into three loci 88-94 bp
#: apart on the human dev slice (its hypotheses' breakpoints spread that far).
LOCUS_MERGE_BP = 300
#: One-sided normal quantile for the decoy upper bound (95%).
DECOY_UPPER_Z = 1.645
#: Alignment collapse: at least COLLAPSE_MIN_HYPOTHESES evaluated hypotheses
#: within +-COLLAPSE_HALF_WINDOW_BP with at most COLLAPSE_MAX_REF_SPAN reads
#: spanning the reference. On HG002 chr1 the 22 pericentromeric 100 kb windows
#: hold 153-1,110 such rows and the densest window elsewhere 25. The cichlid
#: slice's densest holds 53. 100 sits in that gap on both.
COLLAPSE_HALF_WINDOW_BP = 50_000
COLLAPSE_MAX_REF_SPAN = 2
COLLAPSE_MIN_HYPOTHESES = 100
#: Classes with fewer loci than this share the pooled all-class estimate. Loci,
#: not decoys: a locus's decoys share its insert and neighbourhood, so one
#: locus with 100 decoys is one sample, and its upper bound is infinite.
MIN_LOCI_PER_CLASS = 10


@dataclass
class DecoyCheck:
    te_class: str
    loci: int
    decoys: int
    mean: float
    upper: float
    #: What the class's artifact e-values are divided by: max(1, upper).
    factor: float


@dataclass
class ShadowSelection:
    checks: dict[str, DecoyCheck] = field(default_factory=dict)
    loci: int = 0
    te_selected: int = 0
    structural_selected: int = 0
    #: Items inside alignment-collapse regions, whose e-values were set to 0.
    collapse_items: int = 0


def collapse_region_items(items: list) -> set[int]:
    """Positions in `items` of the hypotheses inside an alignment-collapse
    region: at least COLLAPSE_MIN_HYPOTHESES hypotheses within
    +-COLLAPSE_HALF_WINDOW_BP with at most COLLAPSE_MAX_REF_SPAN reference-
    spanning reads. Relative distances only, so where a grid falls cannot
    matter."""
    collapsed: dict[str, list[int]] = {}
    for item in items:
        if max(0, item.ref_span_reads) <= COLLAPSE_MAX_REF_SPAN:
            collapsed.setdefault(item.chrom, []).append(hypothesis_pos(item))
    for positions in collapsed.values():
        positions.sort()
    out: set[int] = set()
    for index, item in enumerate(items):
        on_contig = collapsed.get(item.chrom)
        if not on_contig:
            continue
        pos = hypothesis_pos(item)
        n = (bisect.bisect_right(on_contig, pos + COLLAPSE_HALF_WINDOW_BP)
             - bisect.bisect_left(on_contig, pos - COLLAPSE_HALF_WINDOW_BP))
        if n >= COLLAPSE_MIN_HYPOTHESES:
            out.add(index)
    return out


def _decoy_check(te_class: str, rows: list) -> DecoyCheck:
    samples = [(row.mech_decoy_count, row.mech_decoy_mean_exp_linkage)
               for row in rows if row.mech_decoy_count > 0]
    n = sum(count for count, _ in samples)
    if n == 0:
        return DecoyCheck(te_class, 0, 0, 0.0, 0.0, 1.0)
    mean = sum(count * value for count, value in samples) / n
    # Per-locus means as the sampling unit: decoys of one locus share its
    # insert and neighbourhood, so they are not independent of each other.
    k = len(samples)
    if k > 1:
        per_locus = [value for _, value in samples]
        mu = sum(per_locus) / k
        var = sum((v - mu) ** 2 for v in per_locus) / (k - 1)
        upper = mean + DECOY_UPPER_Z * math.sqrt(var / k)
    else:
        upper = math.inf
    return DecoyCheck(te_class, k, n, mean, upper, max(1.0, upper))


#: Rows whose insert names no TE class. They can only be structural calls, and
#: their inserts are mostly local duplications, whose decoys say nothing about
#: how the TE classes' linkage terms behave -- so they do not enter the pool.
_NOT_TE_CLASSES = ("NA", "NonTE", "")


def decoy_checks(rows: list) -> dict[str, DecoyCheck]:
    """Per class, with small classes falling back to the pooled TE estimate."""
    pooled = _decoy_check("ALL", [row for row in rows
                                  if (row.te_annotation_class or "") not in _NOT_TE_CLASSES])
    by_class: dict[str, list] = {}
    for row in rows:
        by_class.setdefault(row.te_annotation_class or "NA", []).append(row)
    out = {"ALL": pooled}
    for te_class, members in sorted(by_class.items()):
        check = _decoy_check(te_class, members)
        if check.loci < MIN_LOCI_PER_CLASS:
            check = DecoyCheck(te_class, check.loci, check.decoys, check.mean,
                               pooled.upper, pooled.factor)
        out[te_class] = check
    return out


def hypothesis_pos(item) -> int:
    """Where the evaluated hypothesis sits: a ledger row's `pos`, or a call's
    `hypothesis_pos` -- its position before finalization placed it."""
    pos = getattr(item, "hypothesis_pos", -1)
    return pos if pos is not None and pos >= 0 else item.pos


def _loci(rows: list) -> list[list]:
    ordered = sorted(rows, key=lambda r: (r.tid, r.chrom, hypothesis_pos(r)))
    groups: list[list[EvidenceLedgerRow]] = []
    for row in ordered:
        if (groups and groups[-1][-1].chrom == row.chrom
                and hypothesis_pos(row) - hypothesis_pos(groups[-1][-1]) <= LOCUS_MERGE_BP):
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


# ---------------------------------------------------------------------------
# The coverage-rule decision (accepted in replay 2026-09-26; see
# docs/development-strategy.md, section 8)
# ---------------------------------------------------------------------------

#: TEBench's TE rule, on the insert: TE hits cover at least this share of it and
#: at least this many bases (`te_classifier.measure_te_coverage`).
TE_MIN_COVERAGE_FRACTION = 0.5
TE_MIN_COVERED_BP = 100
#: Precise placement: a locus tested by a wide breakpoint interval is placed at
#: its single-position hypothesis with the most indel reads, among those within
#: PLACEMENT_WINDOW_BP of the interval holding at least PLACEMENT_MIN_INDEL_SHARE
#: of the testing row's indel reads. On HG002 chr1 the midpoint of the wide
#: interval put 16 true insertions 100-500 bp off, where sniffles2 put 14 of
#: them within 100 bp. The rule is a plateau: windows of 100, 200 and unlimited
#: give the same calls.
PLACEMENT_WINDOW_BP = 100
PLACEMENT_MIN_INDEL_SHARE = 0.5
#: The TE rule's floor on the insertion as its alt reads measure it.
TE_MIN_MEASURED_LEN_BP = TE_MIN_COVERED_BP
#: Consecutive loci are one allele when their same-sequence tallies' spans
#: overlap and their allele lengths are within this share of the longer one.
ALLELE_LENGTH_TOLERANCE = 0.3


def _is_te_insert(item) -> bool:
    return (item.te_union_coverage >= TE_MIN_COVERAGE_FRACTION
            and item.te_union_covered_bp >= TE_MIN_COVERED_BP
            and (item.te_dominant_class or "") not in _NOT_TE_CLASSES)


def measured_insert_length(item) -> float | None:
    """The median of the alt reads' measured insertion lengths, or None when
    no read measured one."""
    lengths = item.alt_measured_lengths
    return statistics.median(lengths) if lengths else None


def _is_te_call(item) -> bool:
    """TEBench's rule on the insert, and the insertion long enough to hold
    it as the reads measure it. A call no read measured keeps the rule."""
    measured = measured_insert_length(item)
    return _is_te_insert(item) and (measured is None or measured >= TE_MIN_MEASURED_LEN_BP)


def allele_log_lr_vs_artifact(item) -> float:
    """The artifact ratio with the allele's counts term in place of the row's
    own, where the same-sequence tally found carriers beyond the row's own
    reads; the row's own ratio otherwise."""
    if item.allele_byseq_extra_carriers > 0:
        return (item.mech_log_lr_vs_artifact - item.mech_counts_term
                + item.mech_counts_allele_byseq)
    return item.mech_log_lr_vs_artifact


def _merge_by_allele(loci: list[list]) -> list[list]:
    """Join consecutive loci that are one allele: their rows' same-sequence
    spans overlap, and their allele lengths agree within
    ALLELE_LENGTH_TOLERANCE. A locus with no extra carriers joins nothing."""
    out: list[list] = []
    last: tuple[str, int, int, float] | None = None
    for group in loci:
        spans = [(item.bp_left + item.allele_byseq_span_lo,
                  item.bp_left + item.allele_byseq_span_hi, item.allele_length)
                 for item in group if item.allele_byseq_extra_carriers > 0]
        here = None
        if spans:
            here = (group[0].chrom, min(s[0] for s in spans), max(s[1] for s in spans),
                    float(statistics.median(s[2] for s in spans)))
        if (out and here is not None and last is not None and here[0] == last[0]
                and here[1] <= last[2] and last[1] <= here[2]
                and abs(here[3] - last[3]) <= ALLELE_LENGTH_TOLERANCE * max(here[3], last[3])):
            out[-1].extend(group)
            last = (last[0], min(last[1], here[1]), max(last[2], here[2]), last[3])
        else:
            out.append(list(group))
            last = here
    return out


def _placement(group: list, rep, artifact) -> int:
    """The precise breakpoint to report instead of the testing row's own, or -1
    to keep the row's own (its `bp_left`, which is what the VCF writes)."""
    if rep.bp_left == rep.bp_right:
        return -1
    lo = rep.bp_left - PLACEMENT_WINDOW_BP
    hi = rep.bp_right + PLACEMENT_WINDOW_BP
    need = PLACEMENT_MIN_INDEL_SHARE * max(1, rep.alt_indel_reads)
    precise = [item for item in group
               if item.bp_left == item.bp_right >= 0
               and lo <= hypothesis_pos(item) <= hi and item.alt_indel_reads >= need]
    if not precise:
        return -1
    return max(precise, key=lambda item: (item.alt_indel_reads, artifact(item))).bp_left


def select_loci_coverage(items: list, q: float) -> ShadowSelection:
    """The decision replayed as `coverage_rule` + precise placement, testing
    alleles (round 20).

      1. Alignment-collapse regions get e = 0 and stay in m.
      2. The per-class decoy check adjusts the artifact ratios, the row's own
         and the allele's alike (the decoys measure the linkage terms only).
      3. Loci, merged by allele, each tested once: ONE e-BH over all loci, on
         the mean of the locus's rows' exp(adjusted ALLELE ratio). These are
         the insertions. The row with the highest OWN ratio represents the
         locus.
      4. A selected insertion is a TE call when TEBench's rule holds on its
         insert and its reads measure it at 100 bp or more (`_is_te_call`),
         and a structural call otherwise. It is named after the family
         covering the most of it.
      5. It is placed by `_placement`, on the rows' own ratios.

    Marks the representative item: `mech_e_value`, `mech_ebh_selected` (TE) or
    `mech_structural_selected`, and `mech_call_pos`: the precise breakpoint to
    report, or -1 to keep the item's own.
    """
    collapse = {id(items[i]) for i in collapse_region_items(items)}
    out = ShadowSelection(checks=decoy_checks(items), collapse_items=len(collapse))

    def log_factor(item) -> float:
        return math.log(out.checks.get(item.te_annotation_class or "NA",
                                       out.checks["ALL"]).factor)

    def artifact(item) -> float:
        return item.mech_log_lr_vs_artifact - log_factor(item)

    def adjusted(item) -> float:
        return -math.inf if id(item) in collapse else artifact(item)

    def e_value(item) -> float:
        if id(item) in collapse:
            return 0.0
        return math.exp(min(allele_log_lr_vs_artifact(item) - log_factor(item), 700.0))

    for item in items:
        item.mech_collapse_region = id(item) in collapse
        item.mech_e_value = 0.0
        item.mech_ebh_selected = False
        item.mech_structural_selected = False
        item.mech_call_pos = -1
    loci = _merge_by_allele(_loci(items))
    out.loci = len(loci)
    best = [max(group, key=adjusted) for group in loci]
    # The mean, as a sum of e/n so that many rows at the cap cannot overflow.
    # Adopted 2026-09-27 as a validity fix (docs/development-strategy.md,
    # section 8): chr1 157 / 10 -> 154 / 9, chr2-8 858 / 42 -> 849 / 38.
    e_values = [sum(e_value(item) / len(group) for item in group) for group in loci]
    for item, e in zip(best, e_values):
        item.mech_e_value = e
    for index in ebh_select(e_values, q):
        rep = best[index]
        rep.mech_call_pos = _placement(loci[index], rep, artifact)
        if _is_te_call(rep):
            rep.mech_ebh_selected = True
            out.te_selected += 1
        else:
            rep.mech_structural_selected = True
            out.structural_selected += 1
    return out
