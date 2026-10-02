"""PLACER's coverage-rule decision, as finalization runs it.

It calls `placer.core.mechanism_selection.select_loci_coverage` on copies of
the rows, so a replay of this policy is the online decision. It was promoted
from the candidates `coverage_rule` + `pi2_position` (place=precise,
window=100, frac=0.5), and must replay to exactly their decisions.

Since round 20 (2026-10-02) the function tests each locus on its allele's
counts and applies the TE rule's floor to the reads' measured length. It was
promoted from `round20_measured_length_te_rule` (tally=byseq merge=1
gate=median; the pre-registered copy in `placer_dev/wgsdream/prereg`), and
replays to exactly its decisions on the nine rec2 worlds.
"""

from __future__ import annotations

from placer.core.mechanism_selection import select_loci_coverage
from tools.dream.policies import Decision

NAME = "coverage_placed"
DESCRIPTION = ("collapse regions e=0; one e-BH on the decoy-adjusted allele artifact ratio; "
               "TE iff TEBench's rule holds on the insert and the reads measure >= 100 bp; "
               "precise placement")


def select(rows: list, q: float, **params) -> list[Decision]:
    items = [row.copy() for row in rows]
    select_loci_coverage(items, q)
    return [Decision(row=item, label="TE" if item.mech_ebh_selected else "STRUCTURAL",
                     e_value=item.mech_e_value,
                     family=str(item.te_dominant_family if item.mech_ebh_selected
                                else item.family),
                     te_class=str(item.te_dominant_class if item.mech_ebh_selected
                                  else item.te_annotation_class),
                     pos=int(item.mech_call_pos) if item.mech_call_pos >= 0 else None)
            for item in items if item.mech_ebh_selected or item.mech_structural_selected]
