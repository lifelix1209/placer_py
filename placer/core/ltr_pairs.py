"""LTR element entries: which are internal regions, and what form an insert has.

Dfam, RepeatModeler2 and EDTA all split an LTR retrotransposon into its LTR
(`MLT1J1`, `MER41A`, `Gypsy-12_LTR`) and its internal region (`MLT1J-int`,
`MER41-int`, `Gypsy-12_I`, `Gypsy-12_INT`). A new insertion of such an element
is either the full provirus -- LTR, internal region, LTR -- or, after
recombination between its two LTRs, a SOLO LTR, and which it is matters to a
user as much as the family: a solo LTR is the commonest form in most genomes,
and a full-length one is the one that can still be active.

`insert_form` reads the form off the BLAST hits of one insert.
"""

from __future__ import annotations

import re

#: Suffixes that mark an internal-region entry. Case-insensitive, and only as
#: a suffix, so an LTR entry that merely contains "int" is not one.
_INTERNAL = re.compile(r"(?i)(?:[-_](?:int|i)|_internal)$")
_LTR_PART = re.compile(r"(?i)(?:[-_]ltr)$")

#: How near an insert's end an LTR hit must start or stop to count as that
#: end's LTR.
END_SLACK_BP = 50
#: Fraction of the insert a single LTR must explain to be a solo LTR.
SOLO_COVERAGE = 0.80


def is_internal_entry(name: str) -> bool:
    return bool(_INTERNAL.search(name.strip()))


def stem(name: str) -> str:
    """The name without its internal/LTR suffix: what an LTR and its internal
    region have in common (`MER41-int` -> `MER41`, `Gypsy-12_LTR` -> `Gypsy-12`)."""
    token = name.strip()
    return _LTR_PART.sub("", _INTERNAL.sub("", token))


def insert_form(hits, insert_len: int) -> str:
    """"full", "solo", "internal" or "partial", from one family's hits.

    `hits` are objects with `name_parts.subfamily` (the entry name), `query_start`
    and `query_end` (reference orientation, 0-based half-open).

      full      an LTR hit at each end of the insert and an internal hit
                between them
      solo      only LTR hits, one of which explains most of the insert
      internal  only internal-region hits
      partial   anything else
    """
    if insert_len <= 0 or not hits:
        return "NA"
    ltr = [h for h in hits if not is_internal_entry(h.name_parts.subfamily)]
    internal = [h for h in hits if is_internal_entry(h.name_parts.subfamily)]
    left = any(h.query_start <= END_SLACK_BP for h in ltr)
    right = any(h.query_end >= insert_len - END_SLACK_BP for h in ltr)
    if left and right and internal:
        return "full"
    if ltr and not internal and any(
            (h.query_end - h.query_start) >= SOLO_COVERAGE * insert_len for h in ltr):
        return "solo"
    if internal and not ltr:
        return "internal"
    return "partial"
