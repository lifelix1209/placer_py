"""
Target-site duplication detection.

Ported from `src/component/tsd_detector.cpp`, pinned by
`tests/test_20_tsd.py`.

NO GOLDEN VECTORS FOR THIS ONE, and the reason matters. `TSDDetector::detect`
reads the reference through a faidx handle, so it cannot be linked into the
oracle dumper without an indexed FASTA. The pure helpers it is built from live
in an anonymous namespace and are not reachable either. So this module is tested
against HAND-VERIFIABLE cases -- sequences constructed so the right answer is
determinable by counting -- rather than against frozen C++ output. That is a
weaker contract than the rest of the suite and it is stated rather than hidden.

The reference access is injected as a callback here rather than owned, which is
better design anyway: it makes the algorithm testable with an in-memory string
and lets the shifted-decoy construction reuse it.

THE ALGORITHM, and why it is two passes.

Pass 0 requires the two flanks to be byte-identical, which is exactly the
historical behaviour -- so every TSD found before is still found, at the same
length. Pass 1 runs ONLY when no exact TSD exists anywhere in the length range,
and allows a length-scaled mismatch budget. That recovers real TSDs that a
single sequencing or reference difference used to hide; before, they fell
through to the deletion branch or went uncalled entirely.

Lengths are tried LONGEST FIRST, so the longest duplication consistent with the
budget wins rather than the shortest.

THE PART THAT IS EASY TO GET WRONG: the background p-value must be computed with
the SAME mismatch budget that accepted the TSD. A mismatch-tolerant match judged
against an exact-match null understates how often the motif turns up by chance,
which would make tolerant TSDs look more significant than strict ones -- exactly
backwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

#: Reference fetcher: (chrom, start, end) -> sequence, half-open, or "" if
#: unavailable. Injected so the algorithm is testable without an indexed FASTA.
ReferenceFetcher = Callable[[str, int, int], str]


@dataclass
class TsdConfig:
    tsd_min_len: int = 4
    tsd_max_len: int = 30
    tsd_flank_window: int = 60
    tsd_bg_p_max: float = 0.05
    tsd_max_mismatch_rate: float = 0.10
    tsd_max_mismatches: int = 2


@dataclass
class TsdDetection:
    type: str = "NONE"          # DUP / DEL / NONE / UNCERTAIN
    length: int = 0
    sequence: str = ""
    mismatches: int = 0
    bg_p: float = 1.0
    significant: bool = False


def has_only_acgt(seq: str) -> bool:
    """Empty is NOT acceptable, matching the C++ `!s.empty()` conjunct."""
    return bool(seq) and all(c in "ACGT" for c in seq)


def sequence_is_n_rich_reference_context(seq: str, min_run: int = 20,
                                         max_fraction: float = 0.50) -> bool:
    """An assembly gap: either a long continuous N run or half the window."""
    if not seq:
        return False
    n_bases = 0
    current_run = 0
    max_run = 0
    for char in seq:
        if char == "N":
            n_bases += 1
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run >= min_run or (n_bases / len(seq)) > max_fraction


def mismatch_count_within(lhs: str, rhs: str, budget: int) -> int:
    """Counts mismatches, short-circuiting once the budget is exceeded.

    Returns `budget + 1` on a length mismatch or on overrun, so the caller can
    test `> budget` without distinguishing the two.
    """
    if len(lhs) != len(rhs):
        return budget + 1
    mismatches = 0
    for a, b in zip(lhs, rhs):
        if a != b:
            mismatches += 1
            if mismatches > budget:
                return budget + 1
    return mismatches


def tsd_mismatch_budget(length: int, rate: float, cap: int) -> int:
    """
    Length-scaled allowance: `min(cap, floor(length * rate))`.

    Scaled rather than fixed so a SHORT TSD stays exact -- at 5 bp a single
    tolerated mismatch would mostly buy chance matches -- while a long one
    absorbs the read/reference divergence that makes exact matching fail on
    noisy long-read data. `rate <= 0` or `cap <= 0` disables tolerance entirely.
    """
    if length <= 0 or rate <= 0.0 or cap <= 0:
        return 0
    scaled = int(math.floor(length * min(max(rate, 0.0), 1.0)))
    return max(0, min(cap, scaled))


def background_occurrence_fraction(region: str, motif: str,
                                   max_mismatches: int) -> float:
    """
    Empirical local occurrence rate of `motif` in `region`, counting a position
    as a hit when it is within `max_mismatches`.

    The budget MUST match the one that accepted the TSD. Judging a
    mismatch-tolerant match against an exact-match null understates how often
    the motif appears by chance, which would make a tolerant TSD look MORE
    significant than a strict one.
    """
    if not region or not motif or len(region) < len(motif):
        return 1.0
    budget = max(0, max_mismatches)
    total = 0
    hit = 0
    for i in range(0, len(region) - len(motif) + 1):
        total += 1
        window = region[i:i + len(motif)]
        if budget == 0:
            if window == motif:
                hit += 1
        elif mismatch_count_within(window, motif, budget) <= budget:
            hit += 1
    return (hit / total) if total > 0 else 1.0


def detect(fetch: ReferenceFetcher, chrom: str, left_bp: int, right_bp: int,
           config: TsdConfig | None = None) -> TsdDetection:
    """Port of `placer::TSDDetector::detect`."""
    cfg = config or TsdConfig()
    out = TsdDetection()
    if not chrom:
        return out

    # NO NORMALISING SWAP HERE, and that is the whole correctness of this
    # function. The sign of `right_bp - left_bp` is the ONLY thing separating
    # the two geometries below:
    #
    #   right_bp < left_bp  -- the breakpoints OVERLAP, and the overlap is the
    #                          target-site duplication (the DUP branch);
    #   right_bp > left_bp  -- the breakpoints leave a GAP, which is a small
    #                          deletion (the DEL branch).
    #
    # An unconditional `if left_bp > right_bp: swap` used to stand here. It
    # made `right_bp - left_bp` non-negative always, so every genuine TSD was
    # reported as a DELETION of the same length, and the DUP branch could fire
    # only when the REFERENCE itself carried a tandem repeat -- never for a
    # novel insertion, which is the case this detector exists for.
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
        # Longest first, so the longest duplication consistent with the budget
        # wins rather than the shortest.
        for length in range(max_len, min_len - 1, -1):
            if left_bp - length < 0:
                continue
            budget = (0 if current_pass == 0
                      else tsd_mismatch_budget(length, mismatch_rate,
                                               mismatch_cap))
            if current_pass == 1 and budget == 0:
                continue          # already covered exactly by pass 0
            left = fetch(chrom, left_bp - length, left_bp)
            right = fetch(chrom, right_bp, right_bp + length)
            if len(left) != length or len(right) != length:
                continue
            if not has_only_acgt(left) or not has_only_acgt(right):
                continue
            mismatches = mismatch_count_within(left, right, budget)
            if mismatches > budget:
                continue

            bg_region = fetch(chrom, max(0, left_bp - flank), right_bp + flank)
            p = background_occurrence_fraction(bg_region, left, budget)
            out.type = "DUP" if p <= bg_p_max else "UNCERTAIN"
            out.length = length
            out.sequence = left
            out.mismatches = mismatches
            out.bg_p = p
            out.significant = p <= bg_p_max
            return out

    # No duplication: the breakpoints may instead bracket a small DELETION,
    # which is the other geometry TPRT can leave behind.
    delta = right_bp - left_bp
    if min_len <= delta <= max_len:
        del_seq = fetch(chrom, left_bp, right_bp)
        if len(del_seq) == delta and has_only_acgt(del_seq):
            bg_region = fetch(chrom, max(0, left_bp - flank), right_bp + flank)
            p = background_occurrence_fraction(bg_region, del_seq, 0)
            out.type = "DEL" if p <= bg_p_max else "UNCERTAIN"
            out.length = delta
            out.sequence = del_seq
            out.mismatches = 0
            out.bg_p = p
            out.significant = p <= bg_p_max
    return out


def detect_from_insertion(fetch: ReferenceFetcher, chrom: str, pos: int,
                          insert_seq: str,
                          config: TsdConfig | None = None) -> TsdDetection:
    """Find the target-site duplication of an insertion placed at ONE position.

    WHY `detect` CANNOT DO THIS. `detect` compares two windows of the
    REFERENCE, which works when the caller has two breakpoints that overlap.
    A CIGAR `I` operation gives one position and a sequence -- the aligner
    collapsed both breakpoints onto the same coordinate -- so there is no
    overlap left to measure and `detect` has nothing to compare. That is the
    dominant shape of insertion evidence in long-read data, so without this
    the detector is unreachable for most real calls.

    WHERE THE EVIDENCE ACTUALLY IS. The duplication is in the READ, not the
    reference: the sample carries the target site twice and the reference once.
    Whichever way the aligner broke the tie, one copy ends up inside the
    inserted sequence and the other stays in the aligned flank, so

        S[-tau:] == ref[pos - tau : pos]     the insert was placed AFTER the
                                             first copy, and carries the second
        S[:tau]  == ref[pos : pos + tau]     the insert was placed BEFORE it,
                                             and carries the first

    are the two placements of the same event. Both are tried; the longer
    match wins, and ties go to the 3' form, which is the one TPRT produces.

    The two-pass exact-then-tolerant discipline, the length-scaled budget and
    the requirement that the background p-value use the SAME budget are all
    inherited from `detect` -- see its docstring for why the last one matters.
    """
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
                      else tsd_mismatch_budget(length, mismatch_rate, mismatch_cap))
            if current_pass == 1 and budget == 0:
                continue              # already covered exactly by pass 0

            for candidate, flank_seq in (
                    (insert[-length:], upstream[len(upstream) - length:]
                     if len(upstream) >= length else ""),
                    (insert[:length], downstream[:length])):
                if len(flank_seq) != length or not has_only_acgt(flank_seq):
                    continue
                if not has_only_acgt(candidate):
                    continue
                mismatches = mismatch_count_within(candidate, flank_seq, budget)
                if mismatches > budget:
                    continue
                bg_region = fetch(chrom, max(0, pos - flank), pos + flank)
                p = background_occurrence_fraction(bg_region, flank_seq, budget)
                out.type = "DUP" if p <= bg_p_max else "UNCERTAIN"
                out.length = length
                out.sequence = flank_seq
                out.mismatches = mismatches
                out.bg_p = p
                out.significant = p <= bg_p_max
                return out
    return out


def fetcher_from_string(sequence: str) -> ReferenceFetcher:
    """An in-memory reference, for tests and for shifted-decoy generation."""
    def fetch(chrom: str, start: int, end: int) -> str:
        del chrom
        if start < 0 or end > len(sequence) or start >= end:
            return ""
        return sequence[start:end]
    return fetch
