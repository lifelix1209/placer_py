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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from placer.core.ledger import EvidenceLedgerRow
from placer.core.selection import ebh_select

#: Rows this close are one locus. 50 bp split one Alu into three loci 88-94 bp
#: apart on the human dev slice (its hypotheses' breakpoints spread that far).
LOCUS_MERGE_BP = 300
#: One-sided normal quantile for the decoy upper bound (95%).
DECOY_UPPER_Z = 1.645
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


def _loci(rows: list) -> list[list]:
    ordered = sorted(rows, key=lambda r: (r.tid, r.chrom, r.pos))
    groups: list[list[EvidenceLedgerRow]] = []
    for row in ordered:
        if (groups and groups[-1][-1].chrom == row.chrom
                and row.pos - groups[-1][-1].pos <= LOCUS_MERGE_BP):
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def select_loci(items: list, q: float) -> ShadowSelection:
    """The decoy check and both e-BH passes over `items` -- ledger rows or
    calls, anything with the `mech_*` fields, `te_annotation_class`, `chrom`,
    `tid` and `pos`. Marks each locus's representative item."""
    out = ShadowSelection(checks=decoy_checks(items))

    def adjusted_artifact(item) -> float:
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
