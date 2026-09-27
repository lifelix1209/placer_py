"""
The local-interval cache.

The canonical-interval case from `tests/test_parallel_local_interval_cache.cpp`
is reproduced verbatim -- one of the few C++ tests that reach a pipeline helper
directly.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer.alignment import AlignedRead
from placer.core import interval_cache as I
from placer.core.events import ReadReferenceSpan

pytestmark = pytest.mark.invariant


# ------------------------------------------------------- the interval cache
def test_nearby_requests_merge_into_one_fetchable_interval():
    """The C++ case, exactly: two overlapping requests merge, a distant one does
    not, and each canonical interval remembers which requests it covers."""
    requests = [I.LocalIntervalRequest("chr1", 1000, 2100, 0),
                I.LocalIntervalRequest("chr1", 1500, 2600, 1),
                I.LocalIntervalRequest("chr1", 8000, 9000, 2)]
    intervals = call_or_skip(I.build_canonical_local_intervals, requests, 128)
    assert len(intervals) == 2
    assert (intervals[0].chrom, intervals[0].start, intervals[0].end) == ("chr1", 1000, 2600)
    assert intervals[0].request_ids == [0, 1]
    assert (intervals[1].start, intervals[1].end) == (8000, 9000)
    assert intervals[1].request_ids == [2]


def test_the_merge_gap_admits_a_gap_it_does_not_have_to_overlap():
    near = [I.LocalIntervalRequest("chr1", 1000, 2000, 0),
            I.LocalIntervalRequest("chr1", 2100, 3000, 1)]
    assert len(I.build_canonical_local_intervals(near, 128)) == 1
    far = [I.LocalIntervalRequest("chr1", 1000, 2000, 0),
           I.LocalIntervalRequest("chr1", 2200, 3000, 1)]
    assert len(I.build_canonical_local_intervals(far, 128)) == 2


def test_requests_on_different_contigs_never_merge():
    requests = [I.LocalIntervalRequest("chr1", 1000, 2000, 0),
                I.LocalIntervalRequest("chr2", 1000, 2000, 1)]
    assert len(I.build_canonical_local_intervals(requests, 100000)) == 2


def test_a_request_sees_only_the_reads_overlapping_its_own_interval():
    """
    THE correctness-critical half. A request that shared a fetch with a
    neighbour 3 kb away must not see the neighbour's reads -- its evidence
    counts would then include another event's support.
    """
    entry = I.LocalIntervalCacheEntry(
        interval=I.CanonicalLocalInterval("chr1", 1000, 6000, [0, 1]),
        records=[AlignedRead(qname="near"), AlignedRead(qname="far")],
        read_spans=[ReadReferenceSpan(True, 0, 1000, 2000),
                    ReadReferenceSpan(True, 0, 5000, 6000)])
    projection = call_or_skip(I.project_cached_interval_reads,
                              I.LocalIntervalRequest("chr1", 1000, 2100, 0), [entry])
    assert [r.qname for r in projection.records] == ["near"]


def test_a_read_with_no_valid_span_is_dropped():
    """It cannot be shown to overlap, and including it would put an unplaceable
    record into a positional analysis."""
    entry = I.LocalIntervalCacheEntry(
        interval=I.CanonicalLocalInterval("chr1", 1000, 2000, [0]),
        records=[AlignedRead(qname="unplaced")],
        read_spans=[ReadReferenceSpan(valid=False)])
    projection = I.project_cached_interval_reads(
        I.LocalIntervalRequest("chr1", 1000, 2000, 0), [entry])
    assert projection.records == []


def test_a_request_not_covered_by_any_entry_projects_nothing():
    entry = I.LocalIntervalCacheEntry(
        interval=I.CanonicalLocalInterval("chr1", 1000, 2000, [0]),
        records=[AlignedRead(qname="r")],
        read_spans=[ReadReferenceSpan(True, 0, 1000, 2000)])
    assert I.project_cached_interval_reads(
        I.LocalIntervalRequest("chr1", 1000, 2000, 99), [entry]).records == []
    assert I.project_cached_interval_reads(
        I.LocalIntervalRequest("chr2", 1000, 2000, 0), [entry]).records == []


def test_the_reuse_ratio_reports_one_when_nothing_was_fetched():
    """"No requests" is not infinitely efficient, it is simply not measured."""
    close(call_or_skip(I.local_interval_reuse_ratio, I.LocalIntervalReuseStats()),
          1.0, "nothing fetched")
    close(I.local_interval_reuse_ratio(I.LocalIntervalReuseStats(10, 4)), 2.5, "reuse")
