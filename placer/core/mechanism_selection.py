"""Whole-run selection on the per-class likelihood ratios: decoy check, e-BH.

SHADOW, for now: it writes what the new decision WOULD select onto the ledger
and the run summary, beside the current decision, and changes no call.

THE STEPS, over every evaluated hypothesis the scan recorded (not only the ones
the current decision emitted -- a new decision has to be able to find what the
old one missed):

  1. DECOY CHECK, per class. Each row carries the mean of exp(linkage) over its
     shifted-breakpoint decoys (`core/locus_evidence.py`). Pooled over a class,
     that estimates E_null[exp(linkage)], which is at most 1 when the TSD and
     endonuclease-motif nulls hold on this genome. A one-sided upper bound
     above 1 means they do not, and the class's artifact ratio is divided by it
     (in log space, reduced by log of it): e/c is still an e-value when c
     bounds its null expectation, so FDR control survives at a cost in recall.
     The shifted null is not contaminated by true insertions -- a decoy is
     where the insertion is not -- which is what defeated calibrating on the
     candidates themselves (README, "The blocker").
  2. LOCI. Rows within LOCUS_MERGE_BP of each other are one locus, represented
     by its best-scoring row, so a locus evaluated as several hypotheses is
     tested once.
  3. e-BH on exp(min(vs_non_te, vs_artifact)) over all loci: the TE calls.
  4. e-BH on exp(vs_artifact) over the loci not selected as TE: insertions
     that are real but not TEs -- the structural calls.

ALIGNMENT-COLLAPSE REGIONS come first, before any of it. Where the sample's
sequence does not fit the reference -- a centromere model, a satellite array
-- no read spans the reference, and every hypothesis looks like a homozygous
insertion. The counts model's null does not hold there, so those e-values are
not valid. They are set to 0 and stay in m. This is the conservative choice:
an e-value of 0 is valid whatever rule chose it, and removing the loci from m
instead cost the same on HG002 chr1. On chr1 this is the pericentromere
(24% of the evaluated rows, 1,323 structural calls, no confident bases); on
the cichlid slice it is nothing. It was adopted 2026-09-26 as a validity fix
(docs/development-strategy.md, section 2).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from placer.core.ledger import EvidenceLedgerRow
from placer.core.mechanism import Q_AMBIENT, Q_GRID, log_bf_te_derived
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
class IdentityPriors:
    """The TE hypothesis's identity (age) distribution, as fitted to this sample."""

    weights: tuple[float, ...] | None = None
    q_ambient: float = Q_AMBIENT
    #: Loci the fit drew on; 0 means the uniform prior was kept.
    loci: int = 0

    @property
    def mean_identity(self) -> float:
        w = self.weights or tuple(1.0 / len(Q_GRID) for _ in Q_GRID)
        return sum(q * wj for q, wj in zip(Q_GRID, w))


#: What a locus must show to enter the fit: an insertion is certainly there
#: (artifact ratio at least this many nats), a TE class, and enough aligned
#: bases for its identity to mean something.
IDENTITY_FIT_MIN_ARTIFACT = 10.0
IDENTITY_FIT_MIN_ALIGNED = 100
IDENTITY_FIT_MIN_LOCI = 20
#: Pseudo-loci of the uniform prior, spread over the grid: keeps every age
#: possible where the sample has none.
IDENTITY_PRIOR_LOCI = 10.0


def estimate_identity_priors(items: list) -> IdentityPriors:
    """Fit the age distribution to the sample: uniform prior, sample update.

    What sets a real insertion's observed identity is not only its age but the
    consensus's accuracy -- ONT error survives in an insert consensus, and a
    de novo library's consensus is itself approximate -- so the distribution
    is a property of the sample and the library (on the cichlid dev slice
    real insertions tldr calls aligned at 0.83-0.91; on the human one, truth
    Alus at 0.85-0.92 and young ones at 0.97).

    The loci that certainly carry an insertion (artifact ratio >= 10 nats, a
    TE class, >= 100 aligned bases) each contribute their posterior over
    `Q_GRID`; the weights are those summed with IDENTITY_PRIOR_LOCI spread
    uniformly. The null, chance alignment at `Q_AMBIENT`, is a property of
    alignment rather than of the sample and is not fitted: fitting a null
    from these loci, which include insertions of old-element sequence,
    pulled it to 0.86 and rejected the human truth Alus.
    """
    data = [(item.best_te_identity, item.mech_aligned_len) for item in items
            if item.mech_aligned_len >= IDENTITY_FIT_MIN_ALIGNED
            and item.mech_log_lr_vs_artifact >= IDENTITY_FIT_MIN_ARTIFACT
            and (item.te_annotation_class or "") not in _NOT_TE_CLASSES
            and 0.5 < item.best_te_identity <= 1.0]
    if len(data) < IDENTITY_FIT_MIN_LOCI:
        return IdentityPriors()
    totals = [IDENTITY_PRIOR_LOCI / len(Q_GRID)] * len(Q_GRID)
    for identity, n in data:
        k = identity * n
        ll = [k * math.log(q) + (n - k) * math.log(1.0 - q) for q in Q_GRID]
        m = max(ll)
        post = [math.exp(v - m) for v in ll]
        z = sum(post)
        for j, p in enumerate(post):
            totals[j] += p / z
    z = sum(totals)
    return IdentityPriors(tuple(t / z for t in totals), Q_AMBIENT, len(data))


def apply_identity_priors(items: list, priors: IdentityPriors) -> None:
    """Recompute each item's sequence term, and so its TE ratio, with them."""
    if priors.loci == 0:
        return
    for item in items:
        if item.mech_aligned_len <= 0:
            continue
        new = log_bf_te_derived(item.mech_aligned_len, item.best_te_identity,
                                priors.weights, priors.q_ambient)
        item.mech_log_lr_vs_non_te += new - item.mech_sequence_term
        item.mech_sequence_term = new


@dataclass
class ShadowSelection:
    identity: IdentityPriors = field(default_factory=IdentityPriors)
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
        positions = collapsed.get(item.chrom)
        if not positions:
            continue
        pos = hypothesis_pos(item)
        n = (bisect.bisect_right(positions, pos + COLLAPSE_HALF_WINDOW_BP)
             - bisect.bisect_left(positions, pos - COLLAPSE_HALF_WINDOW_BP))
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
    `hypothesis_pos` -- its position before any retether moved it."""
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


def select_loci(items: list, q: float) -> ShadowSelection:
    """The decoy check and both e-BH passes over `items` -- ledger rows or
    calls, anything with the `mech_*` fields, `te_annotation_class`, `chrom`,
    `tid` and `pos`. Marks each locus's representative item.

    First updates the identity priors from the sample and rescores the TE
    ratio with them (`estimate_identity_priors`)."""
    collapse = {id(items[i]) for i in collapse_region_items(items)}
    for item in items:
        item.mech_collapse_region = id(item) in collapse
    priors = estimate_identity_priors([item for item in items if id(item) not in collapse])
    apply_identity_priors(items, priors)
    out = ShadowSelection(identity=priors, checks=decoy_checks(items),
                          collapse_items=len(collapse))

    def adjusted_artifact(item) -> float:
        if id(item) in collapse:
            return -math.inf
        check = out.checks.get(item.te_annotation_class or "NA", out.checks["ALL"])
        return item.mech_log_lr_vs_artifact - math.log(check.factor)

    def te_score(item) -> float:
        return min(item.mech_log_lr_vs_non_te, adjusted_artifact(item))

    loci = _loci(items)
    out.loci = len(loci)
    best = [max(group, key=te_score) for group in loci]
    for item in items:
        item.mech_e_value = 0.0
        item.mech_ebh_selected = False
        item.mech_structural_selected = False
    e_values = [math.exp(min(te_score(item), 700.0)) for item in best]
    for item, e in zip(best, e_values):
        item.mech_e_value = e
    for index in ebh_select(e_values, q):
        best[index].mech_ebh_selected = True
        out.te_selected += 1

    # Structural: an insertion is here but it is not (shown to be) a TE.
    # Tested over ALL loci, with the TE-selected ones given e = 0, so m is the
    # same fixed set as above rather than a subset chosen by the first test.
    structural_e = [0.0 if item.mech_ebh_selected
                    else math.exp(min(adjusted_artifact(item), 700.0)) for item in best]
    for index in ebh_select(structural_e, q):
        best[index].mech_structural_selected = True
        out.structural_selected += 1
    return out


def apply_mechanism_shadow_selection(ledger: list[EvidenceLedgerRow],
                                     q: float) -> ShadowSelection:
    """Mark what the new decision would select over the ledger's evaluated rows."""
    return select_loci([row for row in ledger
                        if row.candidate_retention_reason == "EVALUATED"], q)


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


def _is_te_insert(item) -> bool:
    return (item.te_union_coverage >= TE_MIN_COVERAGE_FRACTION
            and item.te_union_covered_bp >= TE_MIN_COVERED_BP
            and (item.te_dominant_class or "") not in _NOT_TE_CLASSES)


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
    """The decision replayed as `coverage_rule` + precise placement.

      1. Alignment-collapse regions get e = 0 and stay in m.
      2. The per-class decoy check adjusts the artifact ratio.
      3. Loci, each tested once by its row with the highest adjusted artifact
         ratio: ONE e-BH on exp(that), over all loci. These are the insertions.
      4. A selected insertion is a TE call when TEBench's rule holds on its
         insert, and a structural call otherwise. It is named after the family
         covering the most of it.
      5. It is placed by `_placement`.

    Marks the representative item: `mech_e_value`, `mech_ebh_selected` (TE) or
    `mech_structural_selected`, and `mech_call_pos`: the precise breakpoint to
    report, or -1 to keep the item's own.
    """
    collapse = {id(items[i]) for i in collapse_region_items(items)}
    out = ShadowSelection(checks=decoy_checks(items), collapse_items=len(collapse))

    def artifact(item) -> float:
        check = out.checks.get(item.te_annotation_class or "NA", out.checks["ALL"])
        return item.mech_log_lr_vs_artifact - math.log(check.factor)

    def adjusted(item) -> float:
        return -math.inf if id(item) in collapse else artifact(item)

    for item in items:
        item.mech_collapse_region = id(item) in collapse
        item.mech_e_value = 0.0
        item.mech_ebh_selected = False
        item.mech_structural_selected = False
        item.mech_call_pos = -1
    loci = _loci(items)
    out.loci = len(loci)
    best = [max(group, key=adjusted) for group in loci]
    e_values = [math.exp(min(adjusted(item), 700.0)) for item in best]
    for item, e in zip(best, e_values):
        item.mech_e_value = e
    for index in ebh_select(e_values, q):
        rep = best[index]
        rep.mech_call_pos = _placement(loci[index], rep, artifact)
        if _is_te_insert(rep):
            rep.mech_ebh_selected = True
            out.te_selected += 1
        else:
            rep.mech_structural_selected = True
            out.structural_selected += 1
    return out
