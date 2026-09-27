"""
Overlap between two support sets, as a merge-walk over sorted read names.

WHY A MODULE OF ITS OWN. Two stages asked the same question -- "do these two
candidates rest on the same reads?" -- and neither could import the other.
`hypotheses.py` runs inside the per-bin loop and the legacy `finalization.py`
ran once at the end, so a dependency either way would have inverted the
pipeline. They had therefore grown the same merge-walk twice. Since the legacy
finalization was deleted (2026-09-27), `hypotheses.py` is its only user.

WHY A MERGE-WALK RATHER THAN SETS. The support lists arrive sorted, and
`set(lhs) & set(rhs)` would allocate two hash sets per comparison -- inside an
O(k^2) shortlist scan in `hypotheses.py`. The walk is O(n+m) with no
allocation.

SORTEDNESS IS A PRECONDITION, not something checked. An unsorted input
silently under-counts the intersection -- the walk advances past a name it
will never see again. Callers get their lists from
`normalized_support_qnames`, which sorts.
"""

from __future__ import annotations

from collections.abc import Sequence


def intersection_size(lhs: Sequence[str], rhs: Sequence[str]) -> int:
    """`|A n B|` for two SORTED sequences, in O(n+m)."""
    i = j = intersect = 0
    while i < len(lhs) and j < len(rhs):
        if lhs[i] == rhs[j]:
            intersect += 1
            i += 1
            j += 1
        elif lhs[i] < rhs[j]:
            i += 1
        else:
            j += 1
    return intersect


def jaccard(lhs: Sequence[str], rhs: Sequence[str]) -> float:
    """`|A n B| / |A u B|`, and 0.0 when both are empty.

    Two empty support sets are not "identical" for any purpose this pipeline
    has -- a candidate with no supporting reads shares nothing with another
    one -- so the degenerate case returns 0 rather than 1.
    """
    intersect = intersection_size(lhs, rhs)
    union = len(lhs) + len(rhs) - intersect
    return (intersect / union) if union > 0 else 0.0
