"""Read real TE alignments (minimap2 besthits) for segmental explanation.

The besthits table is the per-insert best TE alignment produced upstream
(minimap2 of inserted sequences vs the TE library). Using it gives the
segmental explanation a real graded identity/coverage and a contiguous aligned
span, instead of the divergence-blind exact-k-mer proxy.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from placer_py.redesign.evidence.segmental_explanation import (
    SegmentalExplanation,
    build_segmental_explanation_from_alignment,
)

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


@dataclass(frozen=True)
class TeAlignmentRecord:
    family: str
    subfamily: str
    identity: float
    query_coverage: float
    query_start: int
    query_end: int
    mapq: int


# `str | None`, because `csv.DictReader` yields None for a column the row is
# too short to have. Both already returned the default for it, via the
# `TypeError` arm; the annotation was the only thing claiming it could not
# happen, and the None case is now a statement rather than a side effect of
# exception handling.
def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _to_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _short_id(query: str) -> str:
    return query.split("|", 1)[0]


def read_te_besthits(path: str | Path) -> dict[str, TeAlignmentRecord]:
    """Map candidate id -> best TE alignment record.

    Keys are the candidate id prefix (before the first ``|``) so they join to
    Sniffles candidate ids regardless of the coordinate suffix.
    """
    out: dict[str, TeAlignmentRecord] = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            hit = row.get("te_hit", "") or ""
            family, _, subfamily = hit.partition(":")
            out[_short_id(row.get("query", ""))] = TeAlignmentRecord(
                family=family,
                subfamily=subfamily or family,
                identity=_to_float(row.get("identity")),
                query_coverage=_to_float(row.get("query_cov")),
                query_start=_to_int(row.get("qstart")),
                query_end=_to_int(row.get("qend")),
                mapq=_to_int(row.get("mapq")),
            )
    return out


def segmental_from_record(
    seq: str | None,
    record: TeAlignmentRecord | None,
) -> SegmentalExplanation:
    """Segmental explanation from a besthit record, or empty when no alignment.

    An absent record means no TE alignment for this insert, which is genuine
    absence of TE sequence evidence -- not a reason to fall back to the
    divergence-blind exact-k-mer coverage.
    """
    if record is None or not seq:
        return SegmentalExplanation()
    return build_segmental_explanation_from_alignment(
        seq,
        identity=record.identity,
        query_coverage=record.query_coverage,
        aln_query_start=record.query_start,
        aln_query_end=record.query_end,
        mapq=record.mapq,
        family=record.family,
        subfamily=record.subfamily,
    )
