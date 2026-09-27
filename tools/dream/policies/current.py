"""PLACER's own mechanism decision, as the scan runs it: pi_0 of every replay.

It calls `placer.core.mechanism_selection.select_loci` itself, on copies of the
rows, so a replay of this policy is the online decision. The only thing that
can differ is the world: a world recorded by an older scan holds what that scan
measured.
"""

from __future__ import annotations

from placer.core.mechanism_selection import select_loci
from tools.dream.policies import Decision

NAME = "current"
DESCRIPTION = ("placer's mechanism decision: decoy check, e-BH on "
               "exp(min(vs_non_te, vs_artifact)) for TE calls, then on "
               "exp(vs_artifact) for structural ones")


def select(rows: list, q: float, **params) -> list[Decision]:
    items = [row.copy() for row in rows]
    select_loci(items, q)
    return [Decision(row=item, label="TE" if item.mech_ebh_selected else "STRUCTURAL",
                     e_value=item.mech_e_value, family=str(item.family),
                     te_class=str(item.te_annotation_class))
            for item in items if item.mech_ebh_selected or item.mech_structural_selected]
