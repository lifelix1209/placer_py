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

from placer.core import tprt
from placer.core.ledger import EvidenceLedgerRow
from placer.core.mechanism import Q_AMBIENT, Q_YOUNG
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
class IdentityPriors:
    """The sequence term's two identities, as estimated from this sample."""

    q_young: float = Q_YOUNG
    q_ambient: float = Q_AMBIENT
    #: Loci the estimate drew on; 0 means the literature values were kept.
    loci: int = 0


#: What a locus must show to enter the identity estimate: an insertion is
#: certainly there (artifact ratio at least this many nats), a TE class, and
#: enough aligned bases for its identity to mean something.
IDENTITY_FIT_MIN_ARTIFACT = 10.0
IDENTITY_FIT_MIN_ALIGNED = 100
IDENTITY_FIT_MIN_LOCI = 20
#: Weight of each literature value, in pseudo-bases: enough to hold the fit
#: where the sample says little, small against a real sample's evidence.
IDENTITY_PRIOR_BASES = 2000.0


def estimate_identity_priors(items: list) -> IdentityPriors:
    """Fit (q_young, q_ambient) to the sample: literature prior, sample update.

    Among loci that certainly carry an insertion, identity to the consensus is
    a mixture of two populations -- new copies of an active element, and
    copies of the genome's resident old elements (a duplication carrying one,
    an old element re-inserted in trans) -- which is the distinction the TE
    question needs. What sets the observed identity of a NEW copy is not only
    the element's youth but the consensus's own accuracy: ONT error survives
    in an insert consensus, and a de novo library's consensus is itself
    approximate. On the cichlid dev slice real insertions called by tldr
    aligned at 0.83-0.91, which the literature 0.95/0.80 scored as old
    copies.

    A two-component per-base binomial mixture, fitted by EM with Beta priors
    at the literature values (IDENTITY_PRIOR_BASES each), keeping
    q_young > q_ambient. Too few loci: the literature values stand.
    """
    data = [(item.best_te_identity, item.mech_aligned_len) for item in items
            if item.mech_aligned_len >= IDENTITY_FIT_MIN_ALIGNED
            and item.mech_log_lr_vs_artifact >= IDENTITY_FIT_MIN_ARTIFACT
            and (item.te_annotation_class or "") not in _NOT_TE_CLASSES
            and 0.5 < item.best_te_identity <= 1.0]
    if len(data) < IDENTITY_FIT_MIN_LOCI:
        return IdentityPriors()
    qy, qa, w = Q_YOUNG, Q_AMBIENT, 0.5
    n0 = IDENTITY_PRIOR_BASES
    for _ in range(100):
        ky = ny = ka = na = 0.0
        wy = 0.0
        for identity, n in data:
            k = identity * n
            ly = math.log(w) + k * math.log(qy) + (n - k) * math.log(1.0 - qy)
            la = math.log(1.0 - w) + k * math.log(qa) + (n - k) * math.log(1.0 - qa)
            m = max(ly, la)
            r = math.exp(ly - m) / (math.exp(ly - m) + math.exp(la - m))
            ky += r * k
            ny += r * n
            ka += (1 - r) * k
            na += (1 - r) * n
            wy += r
        new_qy = (Q_YOUNG * n0 + ky) / (n0 + ny)
        new_qa = (Q_AMBIENT * n0 + ka) / (n0 + na)
        new_w = min(0.99, max(0.01, wy / len(data)))
        if new_qy <= new_qa + 0.01:
            new_qa = new_qy - 0.01
        if abs(new_qy - qy) < 1e-6 and abs(new_qa - qa) < 1e-6:
            qy, qa, w = new_qy, new_qa, new_w
            break
        qy, qa, w = new_qy, new_qa, new_w
    return IdentityPriors(min(qy, 0.9995), max(qa, 0.61), len(data))


def apply_identity_priors(items: list, priors: IdentityPriors) -> None:
    """Recompute each item's sequence term, and so its TE ratio, with them."""
    if priors.loci == 0:
        return
    for item in items:
        if item.mech_aligned_len <= 0:
            continue
        new = tprt.log_bf_sequence(item.mech_aligned_len, item.best_te_identity,
                                   q_young=priors.q_young, q_ambient=priors.q_ambient)
        item.mech_log_lr_vs_non_te += new - item.mech_sequence_term
        item.mech_sequence_term = new


@dataclass
class ShadowSelection:
    identity: IdentityPriors = field(default_factory=IdentityPriors)
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
    `tid` and `pos`. Marks each locus's representative item.

    First updates the identity priors from the sample and rescores the TE
    ratio with them (`estimate_identity_priors`)."""
    priors = estimate_identity_priors(items)
    apply_identity_priors(items, priors)
    out = ShadowSelection(identity=priors, checks=decoy_checks(items))

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
