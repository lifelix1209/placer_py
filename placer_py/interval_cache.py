"""
Fetching each stretch of reads once, and projecting it back per request.

Ported from `include/../src/pipeline/local_interval_cache.cpp` and its header,
pinned by
`tests/test_30_call_selection.py`.

WHY IT EXISTS. Every component needs the reads around its breakpoints, and
neighbouring components ask for overlapping intervals. Fetching each request
separately re-reads the same BGZF blocks repeatedly; merging nearby requests into
canonical intervals and fetching those once is what makes the local-fetch stage
affordable.

THE PROJECTION IS THE CORRECTNESS-CRITICAL HALF. The cache holds a canonical
interval's reads, but each request must see ONLY the reads overlapping its own
narrower interval -- otherwise a component's evidence counts would include reads
from a neighbouring event, which is precisely the double-counting the rest of
the pipeline works to avoid. `project_cached_interval_reads` re-filters on every
request rather than trusting the merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from placer_py.alignment import AlignedRead
from placer_py.events import ReadReferenceSpan


@dataclass
class LocalIntervalRequest:
    chrom: str = ""
    start: int = -1
    end: int = -1
    request_id: int = 0


@dataclass
class CanonicalLocalInterval:
    chrom: str = ""
    start: int = -1
    end: int = -1
    request_ids: list[int] = field(default_factory=list)


@dataclass
class LocalIntervalCacheEntry:
    interval: CanonicalLocalInterval = field(default_factory=CanonicalLocalInterval)
    records: list[AlignedRead] = field(default_factory=list)
    read_spans: list[ReadReferenceSpan] = field(default_factory=list)


@dataclass
class LocalIntervalProjection:
    records: list[AlignedRead] = field(default_factory=list)
    read_spans: list[ReadReferenceSpan] = field(default_factory=list)


@dataclass
class LocalIntervalReuseStats:
    request_count: int = 0
    canonical_interval_count: int = 0


def build_canonical_local_intervals(requests: list[LocalIntervalRequest],
                                    merge_gap_bp: int) -> list[CanonicalLocalInterval]:
    """Merge requests within `merge_gap_bp` into one fetchable interval.

    Sorted by (chrom, start, end) first, so the merge is a single pass. Each
    canonical interval REMEMBERS which requests it covers, which is what lets
    the projection find the right entry without re-deciding the merge.
    """
    ordered = sorted(requests, key=lambda r: (r.chrom, r.start, r.end))
    out: list[CanonicalLocalInterval] = []
    for request in ordered:
        if (not out or out[-1].chrom != request.chrom
                or request.start > (out[-1].end + merge_gap_bp)):
            out.append(CanonicalLocalInterval(chrom=request.chrom, start=request.start,
                                              end=request.end,
                                              request_ids=[request.request_id]))
            continue
        back = out[-1]
        back.end = max(back.end, request.end)
        back.request_ids.append(request.request_id)
    return out


def project_cached_interval_reads(request: LocalIntervalRequest,
                                  cache_entries: list[LocalIntervalCacheEntry]
                                  ) -> LocalIntervalProjection:
    """The reads overlapping THIS request, from the entry that covers it.

    Re-filters by span rather than returning the whole canonical interval. A
    request that shared a fetch with a neighbour 3 kb away must not see the
    neighbour's reads -- its evidence counts would then include another event's
    support.

    A read with no valid span is dropped: it cannot be shown to overlap, and
    including it would put an unplaceable record into a positional analysis.
    """
    out = LocalIntervalProjection()
    for entry in cache_entries:
        if entry.interval.chrom != request.chrom:
            continue
        if request.request_id not in entry.interval.request_ids:
            continue
        for record, span in zip(entry.records, entry.read_spans):
            if not span.valid:
                continue
            if span.end <= request.start or span.start >= request.end:
                continue
            out.records.append(record)
            out.read_spans.append(span)
        break
    return out


def local_interval_reuse_ratio(stats: LocalIntervalReuseStats) -> float:
    """Requests per canonical fetch. 1.0 means no reuse at all.

    Returns 1.0 rather than dividing by zero when nothing was fetched -- "no
    requests" is not infinitely efficient, it is simply not measured.
    """
    if stats.canonical_interval_count == 0:
        return 1.0
    return stats.request_count / stats.canonical_interval_count
