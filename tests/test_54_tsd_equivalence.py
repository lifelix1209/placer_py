"""
The TSD detector finds exactly what it found before its windows were hoisted.

`placer/core/tsd.detect` now slices two reference windows instead of fetching
two per candidate length, and counts mismatches in C. The locus and each of its
hundred decoys run it, and its answer is a mechanism term of the locus test, so
it must not move by one base or one mismatch. The implementation from before
the change is kept below verbatim and compared on planted duplications, noisy
ones, bare reference, breakpoints at and past both contig ends, against both a
truncating reference (the real `ReferenceFetcher`'s semantics) and the strict
in-memory one (`fetcher_from_string`, which refuses a window that overhangs).
"""

from __future__ import annotations

import math
import random

from placer.core import tsd as T
from placer.core.tsd import TsdConfig, TsdDetection


# ----------------------------------------------- the implementation before
def _old_mismatch_count_within(lhs, rhs, budget):
    if len(lhs) != len(rhs):
        return budget + 1
    mismatches = 0
    for a, b in zip(lhs, rhs):
        if a != b:
            mismatches += 1
            if mismatches > budget:
                return budget + 1
    return mismatches


def _old_background_occurrence_fraction(region, motif, max_mismatches):
    if not region or not motif or len(region) < len(motif):
        return 1.0
    budget = max(0, max_mismatches)
    total = 0
    hit = 0
    for i in range(len(region) - len(motif) + 1):
        total += 1
        window = region[i:i + len(motif)]
        if budget == 0:
            if window == motif:
                hit += 1
        elif _old_mismatch_count_within(window, motif, budget) <= budget:
            hit += 1
    return (hit / total) if total > 0 else 1.0


def _old_detect(fetch, chrom, left_bp, right_bp, config=None):
    cfg = config or TsdConfig()
    out = TsdDetection()
    if not chrom:
        return out
    left_bp = max(0, left_bp)
    right_bp = max(0, right_bp)
    min_len = max(1, cfg.tsd_min_len)
    max_len = max(min_len, cfg.tsd_max_len)
    flank = max(10, cfg.tsd_flank_window)
    bg_p_max = min(max(cfg.tsd_bg_p_max, 0.0), 1.0)
    mismatch_rate = min(max(cfg.tsd_max_mismatch_rate, 0.0), 1.0)
    mismatch_cap = max(0, cfg.tsd_max_mismatches)
    passes = 2 if (mismatch_rate > 0.0 and mismatch_cap > 0) else 1
    for current_pass in range(passes):
        for length in range(max_len, min_len - 1, -1):
            if left_bp - length < 0:
                continue
            budget = (0 if current_pass == 0
                      else T.tsd_mismatch_budget(length, mismatch_rate, mismatch_cap))
            if current_pass == 1 and budget == 0:
                continue
            left = fetch(chrom, left_bp - length, left_bp)
            right = fetch(chrom, right_bp, right_bp + length)
            if len(left) != length or len(right) != length:
                continue
            if not T.has_only_acgt(left) or not T.has_only_acgt(right):
                continue
            mismatches = _old_mismatch_count_within(left, right, budget)
            if mismatches > budget:
                continue
            bg_region = fetch(chrom, max(0, left_bp - flank), right_bp + flank)
            p = _old_background_occurrence_fraction(bg_region, left, budget)
            out.type = "DUP" if p <= bg_p_max else "UNCERTAIN"
            out.length = length
            out.sequence = left
            out.mismatches = mismatches
            out.bg_p = p
            out.significant = p <= bg_p_max
            return out
    delta = right_bp - left_bp
    if min_len <= delta <= max_len:
        del_seq = fetch(chrom, left_bp, right_bp)
        if len(del_seq) == delta and T.has_only_acgt(del_seq):
            bg_region = fetch(chrom, max(0, left_bp - flank), right_bp + flank)
            p = _old_background_occurrence_fraction(bg_region, del_seq, 0)
            out.type = "DEL" if p <= bg_p_max else "UNCERTAIN"
            out.length = delta
            out.sequence = del_seq
            out.mismatches = 0
            out.bg_p = p
            out.significant = p <= bg_p_max
    return out


def _old_detect_from_insertion(fetch, chrom: str, pos: int,
                          insert_seq: str,
                          config=None):
    cfg = config or TsdConfig()
    out = TsdDetection()
    if not chrom or not insert_seq or pos < 0:
        return out

    min_len = max(1, cfg.tsd_min_len)
    max_len = max(min_len, cfg.tsd_max_len)
    flank = max(10, cfg.tsd_flank_window)
    bg_p_max = min(max(cfg.tsd_bg_p_max, 0.0), 1.0)
    mismatch_rate = min(max(cfg.tsd_max_mismatch_rate, 0.0), 1.0)
    mismatch_cap = max(0, cfg.tsd_max_mismatches)

    insert = insert_seq.upper()
    passes = 2 if (mismatch_rate > 0.0 and mismatch_cap > 0) else 1

    # Hoisted out of the length loop: every window below is a slice of one of
    # these two, so the loop costs no I/O at all. The previous shape of this
    # search re-fetched per length, which is up to 2*(max_len-min_len+1) calls
    # into the reference for a single detection.
    upstream = fetch(chrom, max(0, pos - max_len), pos)
    downstream = fetch(chrom, pos, pos + max_len)

    for current_pass in range(passes):
        for length in range(max_len, min_len - 1, -1):
            if length > len(insert):
                continue
            budget = (0 if current_pass == 0
                      else T.tsd_mismatch_budget(length, mismatch_rate, mismatch_cap))
            if current_pass == 1 and budget == 0:
                continue              # already covered exactly by pass 0

            for candidate, flank_seq in (
                    (insert[-length:], upstream[len(upstream) - length:]
                     if len(upstream) >= length else ""),
                    (insert[:length], downstream[:length])):
                if len(flank_seq) != length or not T.has_only_acgt(flank_seq):
                    continue
                if not T.has_only_acgt(candidate):
                    continue
                mismatches = _old_mismatch_count_within(candidate, flank_seq, budget)
                if mismatches > budget:
                    continue
                bg_region = fetch(chrom, max(0, pos - flank), pos + flank)
                p = _old_background_occurrence_fraction(bg_region, flank_seq, budget)
                out.type = "DUP" if p <= bg_p_max else "UNCERTAIN"
                out.length = length
                out.sequence = flank_seq
                out.mismatches = mismatches
                out.bg_p = p
                out.significant = p <= bg_p_max
                return out
    return out


# ------------------------------------------------------------ the cases
def _reference(rng: random.Random) -> str:
    parts = []
    while sum(map(len, parts)) < 3000:
        roll = rng.random()
        if roll < 0.15:
            parts.append(rng.choice(("A", "AT", "CAG", "TTAGGG")) * rng.randint(4, 40))
        elif roll < 0.2:
            parts.append("N" * rng.randint(1, 30))
        elif roll < 0.25:
            parts.append("".join(rng.choice("acgt") for _ in range(rng.randint(5, 60))))
        else:
            parts.append("".join(rng.choice("ACGT") for _ in range(rng.randint(20, 300))))
    return "".join(parts)


def _truncating(sequence: str):
    upper = sequence.upper()

    def fetch(chrom, start, end):
        if chrom != "chr1" or end <= start:
            return ""
        return upper[max(0, start):end]
    return fetch


def _configs():
    yield TsdConfig()
    tolerant = TsdConfig()
    tolerant.tsd_max_mismatch_rate = 0.15
    tolerant.tsd_max_mismatches = 3
    yield tolerant
    short = TsdConfig()
    short.tsd_min_len, short.tsd_max_len, short.tsd_flank_window = 2, 12, 15
    yield short
    wide = TsdConfig()
    wide.tsd_min_len, wide.tsd_max_len = 1, 64
    wide.tsd_max_mismatch_rate, wide.tsd_max_mismatches = 0.2, 5
    yield wide


def test_the_hoisted_detector_equals_the_per_length_one():
    rng = random.Random(29)
    compared = found = 0
    for _ in range(40):
        reference = _reference(rng)
        n = len(reference)
        for fetch in (_truncating(reference), T.fetcher_from_string(reference)):
            for config in _configs():
                points = [0, 1, 5, 49, 50, n - 50, n - 5, n - 1, n, n + 20]
                points += [rng.randint(0, n) for _ in range(12)]
                for left in points:
                    for right in (left, left - rng.randint(1, 40), left + rng.randint(1, 40),
                                  rng.randint(0, n + 10)):
                        for chrom in ("chr1", "", "chrZ"):
                            old = _old_detect(fetch, chrom, left, right, config)
                            new = T.detect(fetch, chrom, left, right, config)
                            assert new == old, (left, right, chrom)
                            compared += 1
                            found += old.type != "NONE"
    assert compared > 10_000 and found > 200


def test_the_outward_tally_finds_what_the_per_length_insertion_search_found():
    rng = random.Random(37)
    compared = found = 0
    for _ in range(40):
        reference = _reference(rng)
        n = len(reference)
        for fetch in (_truncating(reference), T.fetcher_from_string(reference)):
            for config in _configs():
                for _ in range(40):
                    pos = rng.choice((0, 3, 49, n - 3, n, n + 5, rng.randint(0, n)))
                    tau = rng.randint(1, 30)
                    body = "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 80)))
                    site_up = reference[max(0, pos - tau):pos]
                    site_down = reference[pos:pos + tau]
                    for insert in (body + site_up, site_down + body, body,
                                   _mutate_bases(rng, body + site_up), site_up.lower() + body,
                                   "N" + body + site_up, site_up[-3:], ""):
                        for chrom in ("chr1", "chrZ"):
                            old = _old_detect_from_insertion(fetch, chrom, pos, insert, config)
                            new = T.detect_from_insertion(fetch, chrom, pos, insert, config)
                            assert new == old, (pos, insert, chrom)
                            compared += 1
                            found += old.type != "NONE"
    assert compared > 10_000 and found > 1000


def _mutate_bases(rng: random.Random, seq: str) -> str:
    return "".join(rng.choice("ACGT") if rng.random() < 0.08 else c for c in seq)


def test_mismatch_counting_and_the_background_rate_equal_the_loops():
    rng = random.Random(31)
    for _ in range(3000):
        size = rng.randint(0, 40)
        lhs = "".join(rng.choice("ACGTN") for _ in range(size))
        rhs = "".join(rng.choice("ACGT") if rng.random() < 0.3 else c for c in lhs)
        if rng.random() < 0.1:
            rhs = rhs[:-1]
        budget = rng.randint(-2, 5)
        assert T.mismatch_count_within(lhs, rhs, budget) \
            == _old_mismatch_count_within(lhs, rhs, budget)
    for _ in range(400):
        region = "".join(rng.choice("AC" if rng.random() < 0.5 else "ACGT")
                         for _ in range(rng.randint(0, 300)))
        motif = region[rng.randint(0, max(0, len(region) - 8)):][:rng.randint(0, 12)]
        if rng.random() < 0.3:
            motif = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 9)))
        budget = rng.randint(-1, 3)
        new = T.background_occurrence_fraction(region, motif, budget)
        old = _old_background_occurrence_fraction(region, motif, budget)
        assert new == old or (math.isnan(new) and math.isnan(old)), (region, motif, budget)
