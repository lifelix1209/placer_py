"""
Segmentation answers a repeated flank search exactly as it answered the first.

`placer/core/segmentation._Segmenter.collect_candidates` caches its answer by
argument, because the one-sided searches repeat the paired ones exactly when
both flank limits are the maximum. The segmentation is what the TE library,
the TSD detector and the boundary stage all read, so this compares, on many
loci built to be hard -- mutated flanks, flanks inside tandem repeats and
duplicated reference, inserts that resemble the reference, endpoint slack --
every placement list and every final segmentation against the same searches
computed afresh every time.
"""

from __future__ import annotations

import random

import pytest

from placer.config import PipelineConfig
from placer.core import segmentation as S


def _reference(seed: int) -> str:
    rng = random.Random(seed)
    unit = "".join(rng.choice("ACGT") for _ in range(rng.choice((2, 5, 13, 37))))
    parts = []
    while sum(map(len, parts)) < 6000:
        kind = rng.random()
        if kind < 0.2:
            parts.append(unit * rng.randint(5, 40))           # tandem array
        elif kind < 0.35 and parts:
            copy = rng.choice(parts)                          # a duplicated block
            parts.append(_mutate(rng, copy, 0.03))
        else:
            parts.append("".join(rng.choice("ACGT") for _ in range(rng.randint(50, 400))))
    return "".join(parts)


def _mutate(rng: random.Random, seq: str, rate: float) -> str:
    out = []
    for base in seq:
        roll = rng.random()
        if roll < rate / 3:
            continue                                          # deletion
        if roll < 2 * rate / 3:
            out.append(rng.choice("ACGT"))                    # substitution
        elif roll < rate:
            out.append(base + rng.choice("ACGT"))             # insertion
        else:
            out.append(base)
    return "".join(out)


def _case(seed: int):
    rng = random.Random(seed)
    reference = _reference(seed)
    bp = rng.randint(1500, len(reference) - 1500)
    flank = rng.choice((60, 90, 130, 200))
    if rng.random() < 0.5:
        insert = "".join(rng.choice("ACGT") for _ in range(rng.randint(60, 500)))
    else:                                                     # insert looks like the reference
        start = rng.randint(0, len(reference) - 600)
        insert = _mutate(rng, reference[start:start + rng.randint(60, 500)], 0.05)
    rate = rng.choice((0.0, 0.01, 0.04, 0.08))
    seq = (_mutate(rng, reference[bp - flank:bp], rate) + insert
           + _mutate(rng, reference[bp:bp + flank], rate))
    jitter = rng.randint(-40, 40)
    return reference, seq, bp + jitter, bp + jitter + rng.choice((0, 0, 7, 15))


def _consensus(seq: str) -> S.EventConsensus:
    out = S.EventConsensus(consensus_seq=seq, qc_pass=True, input_event_reads=6,
                           left_anchor_input_reads=3, right_anchor_input_reads=3,
                           partial_context_input_reads=5, full_context_input_reads=4)
    out.consensus_len = len(seq)
    return out


def _run(reference: str, seq: str, bp_left: int, bp_right: int):
    def fetch(chrom, start, end):
        return reference[max(0, start):max(0, end)]
    config = PipelineConfig()
    result = S.segment_event_consensus("chr1", bp_left, bp_right, 8, 4,
                                       _consensus(seq), config, fetch)
    # And the raw placement lists, both sides, with and without endpoint slack.
    seg = S._Segmenter("chr1", bp_left, bp_right, _consensus(seq), config, fetch,
                       S.SegmentationSearchStats())
    flank = min(S.MAX_FLANK_QUERY_BP, len(seq) - S.MIN_INSERT_BP)
    lists = []
    for is_left, point in ((True, bp_left), (False, bp_right)):
        start = max(0, point - 300)
        window = fetch("chr1", start, point + 300)
        lists.extend(seg.collect_candidates(is_left, point, start, window, flank,
                                            slack, False)
                     for slack in (0, S.ENDPOINT_SLACK_BP))
    return result, lists


def test_a_cached_search_answers_exactly_as_a_fresh_one():
    cached_search = S._Segmenter.collect_candidates

    def fresh_search(self, *args):
        return self._collect_candidates(*args)

    passing = 0
    for seed in range(40):
        case = _case(seed)
        cached = _run(*case)
        S._Segmenter.collect_candidates = fresh_search
        try:
            fresh = _run(*case)
        finally:
            S._Segmenter.collect_candidates = cached_search
        assert cached[0] == fresh[0], seed
        assert cached[1] == fresh[1], seed
        passing += cached[0].pass_
    # Not vacuous: most of these segment.
    assert passing >= 20


def test_the_chain_search_places_exactly_what_the_scan_places():
    """The compiled-kernel search (one diagonal chain at a time) against the
    scan it replaces, on every hard case, every list and every result."""
    from placer.core import breakpoints as B
    if B.levenshtein_kernel() is None:
        pytest.skip("no compiled kernel: the scan is what runs")
    kernel = S.levenshtein_kernel
    placements = 0
    for seed in range(60):
        case = _case(seed)
        chains = _run(*case)
        S.levenshtein_kernel = lambda: None
        try:
            scan = _run(*case)
        finally:
            S.levenshtein_kernel = kernel
        assert chains[0] == scan[0], seed
        assert chains[1] == scan[1], seed
        placements += sum(len(found) for found in scan[1])
    assert placements > 1000


def test_a_cached_answer_is_a_new_list_each_time():
    reference, seq, bp_left, bp_right = _case(3)

    def fetch(chrom, start, end):
        return reference[max(0, start):max(0, end)]
    seg = S._Segmenter("chr1", bp_left, bp_right, _consensus(seq), PipelineConfig(),
                       fetch, S.SegmentationSearchStats())
    start = max(0, bp_left - 300)
    window = fetch("chr1", start, bp_left + 300)
    first = seg.collect_candidates(True, bp_left, start, window, 120, 0, False)
    first.clear()
    again = seg.collect_candidates(True, bp_left, start, window, 120, 0, True)
    assert again == seg._collect_candidates(True, bp_left, start, window, 120, 0, True)
    assert seg.stats.segmentation_cache_hits == 1
