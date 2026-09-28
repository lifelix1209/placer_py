"""
Target-site duplication detection.

Ported from `src/component/tsd_detector.cpp`, pinned by
`tests/test_33_tsd.py`.

NO FROZEN C++ OUTPUT FOR THIS ONE, and the reason matters.
`TSDDetector::detect` reads the reference through a faidx handle, so it could
not be driven from a standalone C++ harness without an indexed FASTA. The pure
helpers it is built from live in an anonymous namespace and are not reachable
either. So this module is tested
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
import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from operator import ne
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
    """Empty is NOT acceptable, matching the C++ `!s.empty()` conjunct.

    `strip`, not a generator over the characters: anything outside ACGT
    (N, lower case, IUPAC codes) survives the strip, and it runs in C. Each
    locus's 100 decoys call this thousands of times. cProfile charged the old
    `all(...)` 19 of 89 s on a HG002 0.5 Mb, but most of that was the
    profiler's own per-call cost. Without the profiler the change was within
    run-to-run noise, with byte-identical outputs.
    """
    return bool(seq) and not seq.strip("ACGT")


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
    if budget <= 0:
        return 0 if lhs == rhs else budget + 1
    # Counted in C, then capped: the same value the early exit returned.
    mismatches = sum(map(ne, lhs, rhs))
    return mismatches if mismatches <= budget else budget + 1


@lru_cache(maxsize=4096)
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
    size = len(motif)
    total = len(region) - size + 1
    hit = 0
    if budget == 0:
        # Every position where the motif occurs, overlaps included.
        at = region.find(motif)
        while at >= 0:
            hit += 1
            at = region.find(motif, at + 1)
    else:
        for i in range(total):
            if sum(map(ne, region[i:i + size], motif)) <= budget:
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
    # Every window below is a slice of these two when both come back whole --
    # a reference returns a sub-window's bases as a slice of the window's. Near
    # a contig end, where one comes back short or empty, each length is
    # fetched on its own, exactly as before. The locus and each of its hundred
    # decoys run this, so it used to be ~180 fetches per detection.
    upstream_start = max(0, left_bp - max_len)
    upstream = fetch(chrom, upstream_start, left_bp)
    downstream = fetch(chrom, right_bp, right_bp + max_len)
    hoisted = (len(upstream) == left_bp - upstream_start
               and len(downstream) == max_len)
    # With whole windows, "only ACGT" at length L is L within the run of ACGT
    # ending at the left breakpoint and the run starting at the right one.
    acgt_left = _acgt_run(upstream, from_end=True) if hoisted else 0
    acgt_right = _acgt_run(downstream, from_end=False) if hoisted else 0

    def found(length: int, left: str, mismatches: int, budget: int) -> TsdDetection:
        bg_region = fetch(chrom, max(0, left_bp - flank), right_bp + flank)
        p = background_occurrence_fraction(bg_region, left, budget)
        out.type = "DUP" if p <= bg_p_max else "UNCERTAIN"
        out.length = length
        out.sequence = left
        out.mismatches = mismatches
        out.bg_p = p
        out.significant = p <= bg_p_max
        return out

    first_pass = 0
    if hoisted:
        first_pass = passes
        # PASS 0 BY SEARCH. An exact duplication of length L >= min_len starts
        # where the downstream window's first min_len bases occur in the
        # upstream one, L bases from its end; `find` lists those places,
        # longest L first, and only they are compared in full. The bounds are
        # the loop's skips: L <= left_bp, within both ACGT runs, <= max_len.
        size = len(upstream)
        limit = min(max_len, acgt_left, acgt_right, left_bp)
        if limit >= min_len:
            seed = downstream[:min_len]
            at = upstream.find(seed, size - limit, size)
            while at >= 0:
                length = size - at
                if upstream[at:] == downstream[:length]:
                    return found(length, upstream[at:], 0, 0)
                at = upstream.find(seed, at + 1, size)
        if passes == 2 and limit >= min_len:
            # PASS 1 BIT-PARALLEL. Both windows' ACGT runs as 2-bit integers,
            # the upstream one ending at its least significant digit: the last
            # L upstream bases are its low 2L bits and the first L downstream
            # bases its top 2L bits, and a mismatching base is a pair of bits
            # that differs. The mismatch count is that popcount -- the same
            # integer `mismatch_count_within` counts when it is within budget.
            up = int(upstream[size - limit:].translate(_BASE4), 4)
            down = int(downstream[:limit].translate(_BASE4), 4)
            for length in range(limit, min_len - 1, -1):
                budget = tsd_mismatch_budget(length, mismatch_rate, mismatch_cap)
                if budget == 0:
                    continue          # already covered exactly by pass 0
                diff = (up & ((1 << (2 * length)) - 1)) ^ (down >> (2 * (limit - length)))
                low = (_LOW_BITS[length] if length < len(_LOW_BITS)
                       else int("01" * length, 2))
                mismatches = _popcount((diff | (diff >> 1)) & low)
                if mismatches <= budget:
                    return found(length, upstream[size - length:], mismatches, budget)
    # Windows cut short by a contig end: each length fetched and compared on
    # its own, as the search always was.
    for current_pass in range(first_pass, passes):
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
            if budget == 0:
                if left != right:
                    continue
                mismatches = 0
            else:
                mismatches = mismatch_count_within(left, right, budget)
                if mismatches > budget:
                    continue
            return found(length, left, mismatches, budget)

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

    # BOTH FORMS GROW OUTWARD FROM THE JUNCTION, so one pass over each gives
    # the answer at every length. The 3' form at length L pairs the insert's
    # last L bases with the reference's last L before `pos`; the 5' form its
    # first L with the first L after. Base k of either pair is the same pair
    # of bases whatever L is, so the mismatch count and "only ACGT on both
    # sides" at length L are running totals over k <= L. The loop below then
    # asks, in the order it always did -- pass, length descending, 3' before
    # 5' -- what the per-length window comparisons used to compute.
    reach = min(max_len, len(insert))
    three_usable, three_exact, three_mismatches = _outward_tally(insert, upstream, reach,
                                                                 from_end=True)
    five_usable, five_exact, five_mismatches = _outward_tally(insert, downstream, reach,
                                                              from_end=False)

    def found(length: int, form_is_three: bool, mismatches: int, budget: int) -> TsdDetection:
        flank_seq = (upstream[len(upstream) - length:] if form_is_three
                     else downstream[:length])
        bg_region = fetch(chrom, max(0, pos - flank), pos + flank)
        p = background_occurrence_fraction(bg_region, flank_seq, budget)
        out.type = "DUP" if p <= bg_p_max else "UNCERTAIN"
        out.length = length
        out.sequence = flank_seq
        out.mismatches = mismatches
        out.bg_p = p
        out.significant = p <= bg_p_max
        return out

    # Pass 0, exact. A form matches exactly at every length up to its reach
    # (window, ACGT on both sides, no mismatch yet) and at none beyond, so the
    # longest-first search stops at the longer reach -- the 3' form's on a tie,
    # as it is tried first at each length.
    three_reach = min(max_len, three_usable, three_exact)
    five_reach = min(max_len, five_usable, five_exact)
    longest = max(three_reach, five_reach)
    if longest >= min_len:
        return found(longest, three_reach >= longest, 0, 0)
    if passes < 2:
        return out
    # Pass 1, tolerant: the running mismatch count against the length's budget.
    for length in range(max_len, min_len - 1, -1):
        budget = tsd_mismatch_budget(length, mismatch_rate, mismatch_cap)
        if budget == 0:
            continue                  # already covered exactly by pass 0
        if length <= three_usable and three_mismatches[length] <= budget:
            return found(length, True, three_mismatches[length], budget)
        if length <= five_usable and five_mismatches[length] <= budget:
            return found(length, False, five_mismatches[length], budget)
    return out


def _outward_tally(insert: str, reference: str, reach: int,
                   from_end: bool) -> tuple[int, int, list[int]]:
    """The insert paired with `reference` outward from the junction -- from
    both ends (the 3' form) or both starts (the 5' form) -- as
    `(usable, exact, mismatches)`:

      * `usable`: the longest length whose bases on both sides are all A, C,
        G or T and that the reference window is long enough for (it is
        `reach` long at most);
      * `exact`: how many pairs match before the first mismatch;
      * `mismatches[L]`: mismatches among the first L pairs.
    """
    size = min(reach, len(reference))
    if from_end:
        a = insert[len(insert) - size:][::-1]
        b = reference[len(reference) - size:][::-1]
    else:
        a = insert[:size]
        b = reference[:size]
    usable = size
    for seq in (a, b):
        bad = _NOT_ACGT.search(seq)
        if bad is not None:
            usable = min(usable, bad.start())
    mismatches = list(accumulate(map(ne, a, b), initial=0))
    return usable, bisect_right(mismatches, 0) - 1, mismatches


_ACGT = frozenset("ACGT")
_NOT_ACGT = re.compile("[^ACGT]")
#: A, C, G, T as base-4 digits; only ever applied to ACGT-only runs.
_BASE4 = str.maketrans("ACGT", "0123")
#: 0b0101...01 with L pairs, for folding each base's 2 bits onto one.
_LOW_BITS = [int("01" * length, 2) if length else 0 for length in range(257)]


def _popcount_by_bin(value: int) -> int:
    return bin(value).count("1")


#: `int.bit_count` from Python 3.10, the same count by `bin` before it.
_popcount: Callable[[int], int] = getattr(int, "bit_count", _popcount_by_bin)


def _acgt_run(seq: str, from_end: bool) -> int:
    """How many bases from the end (or the start) are all A, C, G or T."""
    run = 0
    for char in (reversed(seq) if from_end else seq):
        if char not in _ACGT:
            break
        run += 1
    return run


def fetcher_from_string(sequence: str) -> ReferenceFetcher:
    """An in-memory reference, for tests and for shifted-decoy generation."""
    def fetch(chrom: str, start: int, end: int) -> str:
        del chrom
        if start < 0 or end > len(sequence) or start >= end:
            return ""
        return sequence[start:end]
    return fetch
