"""pi_1: FDR on whether an insertion is there; the TE label by coverage.

  1. The decoy check, per class, exactly as placer runs it
     (`mechanism_selection.decoy_checks`).
  2. Loci as placer groups them, each represented by its row with the highest
     decoy-adjusted artifact ratio.
  3. ONE e-BH, on exp(adjusted vs_artifact) over all loci: the insertions.
  4. A selected insertion is a TE call when TEBench's rule holds on its insert
     -- TE hits cover at least `min_fraction` of it and at least `min_bp`
     bases -- and a structural call otherwise. It is named after the family
     that covers the most of it.

vs_non_te is not used: its hallmarks become reported evidence, not a gate.
Alignment-collapse regions get e = 0, as in placer (`collapse=False` to replay
without it).

Older worlds did not record the union coverage (`te_union_*`, then -1); there
the best family's coverage stands in for it, which undercounts inserts covered
by several families. Where it was recorded, this is placer's own rule
(`mechanism_selection.select_loci_coverage`).
"""

from __future__ import annotations

import math

from placer.core.mechanism_selection import _loci, collapse_region_items, decoy_checks
from placer.core.selection import ebh_select
from tools.dream.policies import Decision

NAME = "coverage_rule"
DESCRIPTION = ("one e-BH on the decoy-adjusted artifact ratio; TE iff TE hits "
               "cover >= 50% and >= 100 bp of the insert (TEBench's rule)")


def _coverage(row) -> tuple[float, int, str, str]:
    if row.te_union_covered_bp >= 0:
        return (row.te_union_coverage, row.te_union_covered_bp,
                str(row.te_dominant_family), str(row.te_dominant_class))
    fraction = float(row.best_te_query_coverage)
    return (fraction, int(round(fraction * row.insert_len)), str(row.family),
            str(row.te_annotation_class))


def select(rows: list, q: float, min_fraction: float = 0.5, min_bp: int = 100,
           collapse: bool = True, **params) -> list[Decision]:
    checks = decoy_checks(rows)
    # Placer's own validity fix: e = 0 inside alignment-collapse regions,
    # which stay in m (`mechanism_selection.collapse_region_items`).
    collapsed = {id(rows[i]) for i in collapse_region_items(rows)} if collapse else set()

    def adjusted_artifact(row) -> float:
        if id(row) in collapsed:
            return -math.inf
        check = checks.get(row.te_annotation_class or "NA", checks["ALL"])
        return row.mech_log_lr_vs_artifact - math.log(check.factor)

    loci = _loci(rows)
    best = [max(group, key=adjusted_artifact) for group in loci]
    e_values = [math.exp(min(adjusted_artifact(row), 700.0)) for row in best]
    out = []
    for index in ebh_select(e_values, q):
        row = best[index]
        fraction, covered, family, te_class = _coverage(row)
        is_te = (fraction >= min_fraction and covered >= min_bp
                 and te_class not in ("NA", "NonTE", ""))
        out.append(Decision(row=row, label="TE" if is_te else "STRUCTURAL",
                            e_value=e_values[index], family=family, te_class=te_class))
    return out
