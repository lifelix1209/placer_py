"""
The tripartite decode: flank | insert | flank, placed on the reference.

Ported from `src/pipeline/pipeline_segmentation_stage.inc`, pinned by
`tests/test_27_segmentation.py`.

WHAT THIS STAGE PRODUCES is the single most consequential object in the
pipeline. `insert_seq` is what gets aligned to the TE library, what the poly(A)
and TSD models read, and what the structure decode explains. `left_ref_end` and
`right_ref_start` are what the boundary stage subtracts to decide TSD vs BLUNT
vs SMALL_DEL. Everything after this point is downstream of these five fields.

THE PROBLEM IT SOLVES. The event consensus is one string containing reference
sequence, then inserted sequence, then reference sequence again -- and nothing
marks the two junctions. The stage finds them by aligning each END of the
consensus back to the reference around the expected breakpoint, and taking what
is left over in the middle as the insert.

WHY IT IS NOT A SINGLE ALIGNMENT. A full local alignment of a 5 kb consensus
against a 500 bp reference window would happily align the TE body to a similar
repeat in the flanking region and report no insertion at all. Restricting the
search to the consensus ENDS, with a breakpoint-proximity requirement, makes
that failure impossible by construction.

THE SEARCH IS SEEDED AND BANDED, in three nested loops that matter:

  * 11-mer seeds bin candidate diagonals at 8 bp, keeping the top 8. Without
    this the search would be quadratic in the window size.
  * flank lengths are tried LONGEST FIRST and every length is kept, not just
    the first that works. A longer flank is stronger evidence but a shorter one
    may sit closer to the breakpoint, and the comparator decides between them.
  * within a length, only placements whose implied breakpoint is within 200 bp
    of the expected one are considered at all.

FIVE OUTCOMES, in descending order of what they claim:

    PASS_EVENT_SEGMENTATION                  both flanks placed
    PASS_EVENT_SEGMENTATION_TETHERED_FLANK   one placed, the other found by
                                             searching from the first's edge
    PASS_EVENT_SEGMENTATION_ONE_SIDED_*      only one flank; no junction geometry
    PASS_EVENT_SEGMENTATION_UNPLACED_INSERT  neither; the whole consensus is
                                             called insert on read-count grounds
    NO_TRIPARTITE_EVENT_SEGMENTATION         nothing

The one-sided and unplaced outcomes are why the decision policy carries a
`pair_valid` flag separate from `has_insert_seq`: they produce an insert
sequence without producing a junction, and the boundary stage must not treat the
resulting `bp_left == bp_right` as a blunt join.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from placer.config import PipelineConfig
from placer.core.breakpoints import (
    edit_identity_if_at_least,
    levenshtein_kernel,
    max_edits_for_identity_threshold,
)
from placer.core.policy import FinalBoundaryInput, check_boundary_consistency
from placer.core.seqtools import for_each_valid_kmer, reverse_complement

#: A flank must align at least this far to count as anchored.
MIN_FLANK_ALIGN_BP = 50
#: ... and at least this well. 0.90 is low for a flank against its own locus;
#: it has to admit a noisy long-read consensus.
MIN_FLANK_IDENTITY = 0.90
#: How far from the expected breakpoint a placement may imply its junction.
BREAKPOINT_SLACK_BP = 200
#: How far into the consensus the flank may START. Retried only when the
#: zero-slack search found nothing, because allowing it always would let the
#: flank slide over the junction and eat into the insert.
ENDPOINT_SLACK_BP = 32
#: One base is enough to be an insert. The LENGTH filter lives downstream: a TE
#: call needs 100 TE-covered bases (`mechanism_selection.TE_MIN_COVERED_BP`).
MIN_INSERT_BP = 1
#: The longest flank tried. Beyond this the alignment cost grows without
#: improving the placement.
MAX_FLANK_QUERY_BP = 120
#: Placements within this identity of the best are ALL kept, so the pair
#: comparator can choose between them rather than the single-flank comparator
#: deciding alone.
UNIQUENESS_MARGIN = 0.02
SEED_K = 11
SEED_BIN_BP = 8
SEED_TOP_BINS = 8
#: Alternative placements must share a diagonal within this to be believed --
#: see `_try_tethered_pair`.
PLACEMENT_DIAGONAL_TOLERANCE_BP = 5


@dataclass
class EventConsensus:
    consensus_seq: str = ""
    input_event_reads: int = 0
    consensus_len: int = 0
    full_context_input_reads: int = 0
    partial_context_input_reads: int = 0
    left_anchor_input_reads: int = 0
    right_anchor_input_reads: int = 0
    used_full_context: bool = False
    #: Event strings withheld from abPOA to stay inside the memory budget.
    #: Non-zero means the consensus is built from fewer reads than were
    #: available, which is a real loss of accuracy -- recorded rather than
    #: silent, because `consensus.py` exists on the principle that a quietly
    #: worse consensus is the most damaging thing this stage can produce.
    poa_reads_dropped_for_memory: int = 0
    qc_pass: bool = False
    qc_reason: str = "NO_EVENT_CONSENSUS"


@dataclass
class EventSegmentation:
    left_flank_seq: str = ""
    insert_seq: str = ""
    right_flank_seq: str = ""
    left_ref_start: int = -1
    left_ref_end: int = -1
    right_ref_start: int = -1
    right_ref_end: int = -1
    left_flank_align_len: int = 0
    right_flank_align_len: int = 0
    left_flank_identity: float = 0.0
    right_flank_identity: float = 0.0
    pass_: bool = False
    qc_reason: str = "NO_EVENT_SEGMENTATION"


@dataclass
class EventFlankPlacement:
    query_start: int = -1
    query_end: int = -1
    ref_start: int = -1
    ref_end: int = -1
    align_len: int = 0
    breakpoint_delta: int = 1 << 30
    endpoint_offset: int = 1 << 30
    identity: float = 0.0


@dataclass
class AnchorSeedBin:
    ref_bin_start: int = 0
    support: int = 0
    best_breakpoint_delta: int = 1 << 30


@dataclass
class SegmentationSearchStats:
    """Counters. Diagnostic only, but they are how a pathological locus is
    recognised: a run where `edit_distance_calls` explodes is a repeat, not a
    bug."""

    edit_distance_calls: int = 0
    edit_distance_cache_hits: int = 0
    edit_distance_cache_misses: int = 0
    seed_bins_total: int = 0
    paired_searches: int = 0
    one_sided_searches: int = 0
    endpoint_slack_searches: int = 0
    reverse_complement_retries: int = 0
    segmentation_cache_hits: int = 0
    segmentation_cache_misses: int = 0


def has_identifiable_bilateral_partial_context_support(left_anchor_input_reads: int,
                                                       right_anchor_input_reads: int,
                                                       partial_context_input_reads: int,
                                                       input_event_reads: int) -> bool:
    """Both sides anchored by SOME read, with at least two partial contexts.

    The alternative to requiring a full-context read (one that spans the whole
    event). A long insertion may have NO read spanning it, and requiring one
    would make event length decide detectability -- which is exactly the bias
    the README criticises in the high-precision tier.
    """
    has_bilateral_anchor = left_anchor_input_reads > 0 and right_anchor_input_reads > 0
    return (has_bilateral_anchor and partial_context_input_reads >= 2
            and input_event_reads >= 2)


def pre_segmentation_gate_reason(alt_split_reads: int, alt_indel_reads: int,
                                 consensus: EventConsensus) -> str:
    """Refuse to run the expensive stage, with a reason, or return "".

    The gate is definitional, not a threshold ladder: without reads anchoring
    BOTH sides there is no junction to find, and without either a precise read
    or an identifiable bilateral partial context there is nothing to place.
    """
    if consensus.left_anchor_input_reads <= 0 or consensus.right_anchor_input_reads <= 0:
        return "PRESEG_NO_BILATERAL_ANCHOR"
    has_precise_or_full_context = (consensus.full_context_input_reads > 0
                                   or (alt_split_reads + alt_indel_reads) > 0)
    if not has_precise_or_full_context and not has_identifiable_bilateral_partial_context_support(
            consensus.left_anchor_input_reads, consensus.right_anchor_input_reads,
            consensus.partial_context_input_reads, consensus.input_event_reads):
        return "PRESEG_NO_PRECISE_OR_FULL_CONTEXT"
    return ""


def better_event_flank_placement(placement: EventFlankPlacement) -> tuple:
    """Sort key: identity, then breakpoint proximity, then endpoint, then length.

    IDENTITY LEADS, which is the choice worth noticing. A placement that matches
    the reference better is preferred even when it sits further from the
    expected breakpoint -- because the expected breakpoint came from clustering
    read signals and is itself uncertain, while identity is measured.
    """
    return (-placement.identity, placement.breakpoint_delta,
            placement.endpoint_offset, -placement.align_len,
            placement.ref_start, placement.ref_end)


def reference_seed_index(ref_window: str) -> dict[int, list[int]]:
    """Every SEED_K-mer of the window, by key, with its positions in order."""
    ref_hits: dict[int, list[int]] = {}
    for pos, key in for_each_valid_kmer(ref_window, SEED_K):
        ref_hits.setdefault(key, []).append(pos)
    return ref_hits


def collect_anchor_seed_bins(query: str, breakpoint: int, ref_window_start: int,
                             ref_window: str, is_left: bool,
                             ref_hits: dict[int, list[int]] | None = None
                             ) -> list[AnchorSeedBin]:
    """Bin shared 11-mers by the DIAGONAL they imply, keeping the best eight.

    The diagonal `ref_start = ref_window_start + ref_pos - query_pos` is where
    the flank would have to start for this k-mer match to be part of a gapless
    alignment. Binning at 8 bp tolerates small indels between the consensus and
    the reference while still collapsing a true alignment's many k-mer hits onto
    one candidate.

    Ranked by SUPPORT first and breakpoint proximity second, so a repeat
    elsewhere in the window with two matching k-mers cannot outrank the real
    locus with forty.
    """
    if ref_hits is None:
        ref_hits = reference_seed_index(ref_window)

    bins_by_start: dict[int, AnchorSeedBin] = {}
    for query_pos, key in for_each_valid_kmer(query, SEED_K):
        hits = ref_hits.get(key)
        if not hits:
            continue
        for ref_pos in hits:
            ref_start = ref_window_start + ref_pos - query_pos
            ref_bin_start = (ref_start // SEED_BIN_BP) * SEED_BIN_BP
            seed_bin = bins_by_start.get(ref_bin_start)
            if seed_bin is None:
                seed_bin = AnchorSeedBin(ref_bin_start=ref_bin_start)
                bins_by_start[ref_bin_start] = seed_bin
            seed_bin.support += 1
            candidate_breakpoint = (ref_start + len(query)) if is_left else ref_start
            seed_bin.best_breakpoint_delta = min(seed_bin.best_breakpoint_delta,
                                                 abs(candidate_breakpoint - breakpoint))

    bins = sorted(bins_by_start.values(),
                  key=lambda b: (-b.support, b.best_breakpoint_delta, b.ref_bin_start))
    return bins[:SEED_TOP_BINS]


class _Segmenter:
    """One segmentation run. A class only because the C++ closure captures a
    dozen locals and threading them through free functions would be worse."""

    def __init__(self, chrom: str, bp_left: int, bp_right: int,
                 consensus: EventConsensus, config: PipelineConfig,
                 fetch_window: Callable[[str, int, int], str],
                 stats: SegmentationSearchStats) -> None:
        self.chrom = chrom
        self.bp_left = bp_left
        self.bp_right = bp_right
        self.event_consensus = consensus
        self.consensus = consensus.consensus_seq
        self.consensus_len = len(self.consensus)
        self.config = config
        self.fetch_window = fetch_window
        self.stats = stats
        self.edit_memo: dict[tuple[int, int, int], tuple[bool, float]] = {}
        #: `collect_candidates` answers by argument. The one-sided searches
        #: repeat the paired ones exactly whenever both flank limits are
        #: MAX_FLANK_QUERY_BP, endpoint-slack retries included.
        self.search_cache: dict[tuple, list[EventFlankPlacement]] = {}
        #: `reference_seed_index` of each reference window searched: the
        #: paired, slack and one-sided searches of a side share one window.
        self.seed_index: dict[str, dict[int, list[int]]] = {}
        #: Set by the caller before `emit_unplaced_insert`. Kept off the
        #: constructor because only that one fallback reads them.
        self.alt_struct_reads = 0
        self.alt_ref_span_reads = 0

    # ------------------------------------------------------------ searching
    def collect_candidates(self, is_left: bool, breakpoint: int, ref_window_start: int,
                           ref_window: str, flank_query_len: int,
                           query_endpoint_slack: int,
                           is_one_sided_search: bool) -> list[EventFlankPlacement]:
        """Every acceptable placement of one flank, best first.

        THE LENGTH LOOP RUNS LONGEST FIRST AND KEEPS EVERY LENGTH. A longer
        flank is stronger evidence, but a shorter one may sit closer to the
        breakpoint, and only the pair comparator can weigh those against each
        other -- so the decision is deferred rather than made here.

        Within a length, everything within `UNIQUENESS_MARGIN` of the best
        identity is kept for the same reason: a 0.99 placement 180 bp from the
        breakpoint and a 0.98 one at 2 bp are both live hypotheses.
        """
        if not ref_window or flank_query_len < MIN_FLANK_ALIGN_BP:
            return []
        # The answer is a function of these arguments alone -- the memos only
        # hold values -- so a repeated search is answered from the first.
        # `is_one_sided_search` changes only the diagnostic counters.
        cache_key = (is_left, breakpoint, ref_window_start, ref_window,
                     flank_query_len, query_endpoint_slack)
        cached = self.search_cache.get(cache_key)
        if cached is not None:
            self.stats.segmentation_cache_hits += 1
            return list(cached)
        self.stats.segmentation_cache_misses += 1
        found = self._collect_candidates(is_left, breakpoint, ref_window_start,
                                         ref_window, flank_query_len,
                                         query_endpoint_slack, is_one_sided_search)
        self.search_cache[cache_key] = found
        return list(found)

    def _collect_candidates(self, is_left: bool, breakpoint: int, ref_window_start: int,
                            ref_window: str, flank_query_len: int,
                            query_endpoint_slack: int,
                            is_one_sided_search: bool) -> list[EventFlankPlacement]:
        """`collect_candidates`, computed."""
        if is_one_sided_search:
            self.stats.one_sided_searches += 1
        else:
            self.stats.paired_searches += 1
        if query_endpoint_slack > 0:
            self.stats.endpoint_slack_searches += 1

        consensus = self.consensus
        consensus_len = self.consensus_len
        query_seed_len = min(flank_query_len + query_endpoint_slack, consensus_len)
        query_seed_start = 0 if is_left else (consensus_len - query_seed_len)
        seed_query = (consensus[:query_seed_len] if is_left
                      else consensus[query_seed_start:])
        ref_hits = self.seed_index.get(ref_window)
        if ref_hits is None:
            ref_hits = self.seed_index[ref_window] = reference_seed_index(ref_window)
        seed_bins = collect_anchor_seed_bins(seed_query, breakpoint, ref_window_start,
                                             ref_window, is_left, ref_hits)
        self.stats.seed_bins_total += len(seed_bins)
        if not seed_bins:
            return []
        kernel = levenshtein_kernel()
        if kernel is not None:
            return self._chain_candidates(kernel, is_left, breakpoint, ref_window_start,
                                          ref_window, flank_query_len,
                                          query_endpoint_slack, query_seed_len, seed_bins)
        return self._scan_candidates(is_left, breakpoint, ref_window_start, ref_window,
                                     flank_query_len, query_endpoint_slack,
                                     query_seed_start, seed_bins)

    def _scan_candidates(self, is_left: bool, breakpoint: int, ref_window_start: int,
                         ref_window: str, flank_query_len: int,
                         query_endpoint_slack: int, query_seed_start: int,
                         seed_bins: list[AnchorSeedBin]) -> list[EventFlankPlacement]:
        """The search, one (length, query start, reference start) at a time.

        What runs without the compiled kernel, and the definition
        `_chain_candidates` is checked against (`tests/test_53`).
        """
        consensus = self.consensus
        consensus_len = self.consensus_len
        ref_window_end = ref_window_start + len(ref_window)
        search_lo = max(0, breakpoint - BREAKPOINT_SLACK_BP)
        search_hi = breakpoint + BREAKPOINT_SLACK_BP
        candidates: list[EventFlankPlacement] = []
        for align_len in range(flank_query_len, MIN_FLANK_ALIGN_BP - 1, -1):
            max_query_start = consensus_len - align_len
            if max_query_start < 0:
                continue
            max_edits = max_edits_for_identity_threshold(align_len, align_len,
                                                         MIN_FLANK_IDENTITY)
            endpoint_slack = max(0, min(query_endpoint_slack, max_query_start))
            query_lo = 0 if is_left else max(0, max_query_start - endpoint_slack)
            query_hi = min(max_query_start, endpoint_slack) if is_left else max_query_start

            placements: list[EventFlankPlacement] = []
            # The two bounds tests of every (seed bin, offset) are solved for
            # the offset range instead, so the loop visits exactly the offsets
            # it used to keep, in the same order. `shift` is where the
            # candidate breakpoint sits relative to ref_start: the flank's end
            # on the left side, its start on the right.
            shift = align_len if is_left else 0
            offset_lo = max(search_lo - shift, ref_window_start)
            offset_hi = min(search_hi - shift, ref_window_end - align_len)
            edit_memo = self.edit_memo
            cache_hits = 0
            cache_misses = 0
            for query_start in range(query_lo, query_hi + 1):
                query = consensus[query_start:query_start + align_len]
                endpoint_offset = (query_start if is_left
                                   else consensus_len - (query_start + align_len))
                seed_query_offset = query_start - query_seed_start

                for seed_bin in seed_bins:
                    base = seed_bin.ref_bin_start + seed_query_offset
                    first = max(0, offset_lo - base)
                    last = min(SEED_BIN_BP - 1, offset_hi - base)
                    for ref_start in range(base + first, base + last + 1):
                        ref_end = ref_start + align_len
                        memo_key = (query_start, ref_start, align_len)
                        memoized = edit_memo.get(memo_key)
                        if memoized is not None:
                            cache_hits += 1
                            pass_identity, identity = memoized
                        else:
                            cache_misses += 1
                            offset = ref_start - ref_window_start
                            result = edit_identity_if_at_least(
                                query, ref_window[offset:offset + align_len], max_edits)
                            pass_identity = result is not None
                            identity = result if result is not None else 0.0
                            edit_memo[memo_key] = (pass_identity, identity)
                        if not pass_identity:
                            continue

                        candidate_breakpoint = ref_start + shift
                        placements.append(EventFlankPlacement(
                            query_start=query_start, query_end=query_start + align_len,
                            ref_start=ref_start, ref_end=ref_end, align_len=align_len,
                            identity=identity,
                            breakpoint_delta=abs(candidate_breakpoint - breakpoint),
                            endpoint_offset=endpoint_offset))
            self.stats.edit_distance_cache_hits += cache_hits
            self.stats.edit_distance_cache_misses += cache_misses
            self.stats.edit_distance_calls += cache_misses

            if not placements:
                continue
            placements.sort(key=better_event_flank_placement)
            best_identity = placements[0].identity
            for placement in placements:
                if (placement.identity + UNIQUENESS_MARGIN) < best_identity:
                    break
                candidates.append(placement)

        candidates.sort(key=better_event_flank_placement)
        return candidates

    def _chain_candidates(self, kernel, is_left: bool, breakpoint: int,
                          ref_window_start: int, ref_window: str, flank_query_len: int,
                          query_endpoint_slack: int, query_seed_len: int,
                          seed_bins: list[AnchorSeedBin]) -> list[EventFlankPlacement]:
        """`_scan_candidates`, one diagonal CHAIN at a time.

        THE CHAINS. On the left, a placement with endpoint offset e starts its
        query at e and its reference at `bin + e + j` for every length, so the
        pairs one `(e, bin, j)` visits as the length shrinks are PREFIXES of
        one another; on the right they END at fixed positions and are suffixes.
        The bounds the scan applies per length (the search window around the
        breakpoint, the reference window, the consensus) leave each chain one
        contiguous range of lengths, solved for below. Every
        (length, query start, reference start) the scan visits is on exactly
        one chain.

        WHY FEW DISTANCES ARE ENOUGH. Along a diagonal of one edit-distance
        matrix the distance never decreases and grows by at most one per base
        (Ukkonen), so over a chain it is a non-decreasing step function of the
        length. Asked with the chain's largest budget as the cutoff, the kernel
        returns the exact distance wherever it can pass and "over" elsewhere,
        which is still non-decreasing. Equal values at two lengths fix every
        length between them; a value already over the budget at the longer end
        of a range fails the whole range. Bisection fills in the rest. A
        placement that passes gets the identity `edit_identity_if_at_least`
        would give -- the same expression of the same integer -- and the
        per-length ranking and uniqueness margin are the scan's, so the result
        is the scan's list (`tests/test_53_segmentation_search_cache.py` holds
        the two against each other).
        """
        consensus = self.consensus
        cl = self.consensus_len
        rws = ref_window_start
        rwe = rws + len(ref_window)
        search_lo = max(0, breakpoint - BREAKPOINT_SLACK_BP)
        search_hi = breakpoint + BREAKPOINT_SLACK_BP
        top = min(flank_query_len, cl)
        if top < MIN_FLANK_ALIGN_BP:
            return []
        budget = [0] * (top + 1)
        for length in range(MIN_FLANK_ALIGN_BP, top + 1):
            budget[length] = max_edits_for_identity_threshold(length, length,
                                                              MIN_FLANK_IDENTITY)
        slack = max(0, query_endpoint_slack)
        by_length: dict[int, list[tuple]] = {}
        calls = 0
        for e in range(slack + 1):
            if cl - e < MIN_FLANK_ALIGN_BP:
                break
            for seed_bin in seed_bins:
                for j in range(SEED_BIN_BP):
                    if is_left:
                        fixed = seed_bin.ref_bin_start + e + j        # ref start
                        if fixed < rws:
                            continue
                        lo = max(MIN_FLANK_ALIGN_BP, search_lo - fixed)
                        hi = min(top, cl - e, search_hi - fixed, rwe - fixed)
                    else:
                        fixed = seed_bin.ref_bin_start + query_seed_len - e + j  # ref end
                        if fixed > rwe:
                            continue
                        lo = max(MIN_FLANK_ALIGN_BP, fixed - search_hi)
                        hi = min(top, cl - e, fixed - max(search_lo, rws))
                    if lo > hi:
                        continue
                    cutoff = budget[hi]
                    offset = fixed - rws

                    def distance(length: int, e: int = e, offset: int = offset,
                                 cutoff: int = cutoff) -> int:
                        if is_left:
                            return kernel(consensus[e:e + length],
                                          ref_window[offset:offset + length],
                                          score_cutoff=cutoff)
                        return kernel(consensus[cl - e - length:cl - e],
                                      ref_window[offset - length:offset],
                                      score_cutoff=cutoff)

                    known = {lo: distance(lo)}
                    calls += 1
                    if known[lo] > cutoff:
                        continue                  # over every budget in the range
                    if hi > lo:
                        known[hi] = distance(hi)
                        calls += 1
                        pending = [(lo, hi)]
                        while pending:
                            x, y = pending.pop()
                            if y - x <= 1 or known[x] == known[y] or known[x] > budget[y]:
                                continue
                            mid = (x + y) // 2
                            known[mid] = distance(mid)
                            calls += 1
                            pending.append((x, mid))
                            pending.append((mid, y))
                    points = sorted(known)
                    for index, x in enumerate(points):
                        value = known[x]
                        span = range(x, x + 1)
                        if index + 1 < len(points) and known[points[index + 1]] == value:
                            span = range(x, points[index + 1])
                        for length in span:
                            if value > budget[length]:
                                continue
                            identity = min(1.0, max(0.0, 1.0 - (value / length)))
                            if is_left:
                                query_start, ref_start = e, fixed
                                delta = abs(fixed + length - breakpoint)
                            else:
                                query_start, ref_start = cl - e - length, fixed - length
                                delta = abs(ref_start - breakpoint)
                            by_length.setdefault(length, []).append(
                                (-identity, delta, e, ref_start, identity, query_start))
        self.stats.edit_distance_calls += calls
        self.stats.edit_distance_cache_misses += calls

        candidates: list[EventFlankPlacement] = []
        for length, rows in by_length.items():
            # The scan's per-length order: identity, then breakpoint distance,
            # endpoint offset and reference start (unique within a length).
            rows.sort()
            best_identity = rows[0][4]
            for _, delta, e, ref_start, identity, query_start in rows:
                if (identity + UNIQUENESS_MARGIN) < best_identity:
                    break
                candidates.append(EventFlankPlacement(
                    query_start=query_start, query_end=query_start + length,
                    ref_start=ref_start, ref_end=ref_start + length, align_len=length,
                    identity=identity, breakpoint_delta=delta, endpoint_offset=e))
        candidates.sort(key=better_event_flank_placement)
        return candidates

    # ----------------------------------------------------------- comparators
    @staticmethod
    def better_one_sided(placement: EventFlankPlacement) -> tuple:
        return (-placement.identity, placement.breakpoint_delta, -placement.align_len,
                placement.endpoint_offset, placement.ref_start)

    @staticmethod
    def pair_key(left: EventFlankPlacement, right: EventFlankPlacement) -> tuple:
        """Rank a flank PAIR by its WORSE side first, then by its mean.

        The min-first rule is the important one: a pair with one perfect and one
        marginal flank is worse than a pair with two good ones, because the
        marginal flank is the one that decides where the junction is. Ranking by
        the mean alone would let a 1.00/0.90 pair beat a 0.96/0.96 pair.
        """
        min_identity = min(left.identity, right.identity)
        mean_identity = 0.5 * (left.identity + right.identity)
        return (-min_identity, -mean_identity,
                left.breakpoint_delta + right.breakpoint_delta,
                -(left.align_len + right.align_len), left.ref_start, right.ref_start)

    def canonical_pair(self, left: EventFlankPlacement,
                       right: EventFlankPlacement) -> bool:
        """Do these two flanks leave room for an insert AND a legal junction?

        The boundary consistency test is reused here rather than reimplemented,
        so the pair the segmenter accepts is exactly the pair the boundary stage
        will later be able to classify.
        """
        if left.query_end + MIN_INSERT_BP > right.query_start:
            return False
        tsd_min_len = max(1, self.config.tsd_min_len)
        return check_boundary_consistency(FinalBoundaryInput(
            left_ref_start=left.ref_start, left_ref_end=left.ref_end,
            right_ref_start=right.ref_start, right_ref_end=right.ref_end,
            tsd_min_len=tsd_min_len,
            tsd_max_len=max(tsd_min_len, self.config.tsd_max_len))).pass_

    # -------------------------------------------------------------- emitters
    def emit_left_sided(self, best_left: EventFlankPlacement) -> EventSegmentation:
        out = EventSegmentation()
        insert_start = best_left.query_end
        insert_len = self.consensus_len - insert_start
        if insert_len < MIN_INSERT_BP:
            out.qc_reason = "EMPTY_EVENT_INSERT_SEGMENT"
            return out
        out.left_flank_seq = self.consensus[best_left.query_start:
                                            best_left.query_start + best_left.align_len]
        out.insert_seq = self.consensus[insert_start:insert_start + insert_len]
        out.left_ref_start = best_left.ref_start
        out.left_ref_end = best_left.ref_end
        # The unanchored side collapses onto the expected breakpoint, so
        # `right_ref_start == right_ref_end`. The boundary stage reads that as
        # INVALID rather than blunt -- which is why `pair_valid` has to travel
        # separately from `has_insert_seq`.
        out.right_ref_start = self.bp_right
        out.right_ref_end = self.bp_right
        out.left_flank_align_len = best_left.align_len
        out.left_flank_identity = best_left.identity
        out.pass_ = True
        out.qc_reason = "PASS_EVENT_SEGMENTATION_ONE_SIDED_LEFT"
        return out

    def emit_right_sided(self, best_right: EventFlankPlacement) -> EventSegmentation:
        out = EventSegmentation()
        insert_len = best_right.query_start
        if insert_len < MIN_INSERT_BP:
            out.qc_reason = "EMPTY_EVENT_INSERT_SEGMENT"
            return out
        out.insert_seq = self.consensus[:insert_len]
        out.right_flank_seq = self.consensus[best_right.query_start:
                                             best_right.query_start + best_right.align_len]
        out.left_ref_start = self.bp_left
        out.left_ref_end = self.bp_left
        out.right_ref_start = best_right.ref_start
        out.right_ref_end = best_right.ref_end
        out.right_flank_align_len = best_right.align_len
        out.right_flank_identity = best_right.identity
        out.pass_ = True
        out.qc_reason = "PASS_EVENT_SEGMENTATION_ONE_SIDED_RIGHT"
        return out

    def emit_unplaced_insert(self) -> EventSegmentation:
        """Call the WHOLE consensus an insert, on read-count grounds alone.

        The last resort, and it is gated hard because it asserts an insertion
        without having placed either junction:

          * no reference-spanning read at all, and at least 8 alt reads, of
            which the consensus used at most half -- i.e. the locus is
            unopposed and heavily supported;
          * the consensus itself had two anchoring reads on EACH side and four
            partial contexts.

        Together those say "many reads agree something is here and none disagree,
        but the flanks will not align" -- which is what a long insertion into a
        segmental duplication looks like. Not attempted on the reverse-complement
        retry, so it cannot fire twice for one event.
        """
        out = EventSegmentation()
        consensus = self.event_consensus
        if consensus.qc_reason == "PASS_EVENT_CONSENSUS_REVCOMP_RETRY":
            return out
        strong_unopposed_alt = (self.alt_ref_span_reads == 0
                                and self.alt_struct_reads >= 8
                                and self.alt_struct_reads >= consensus.input_event_reads * 2)
        anchored_consensus = (consensus.qc_pass
                              and consensus.left_anchor_input_reads >= 2
                              and consensus.right_anchor_input_reads >= 2
                              and consensus.partial_context_input_reads >= 4)
        if not strong_unopposed_alt or not anchored_consensus:
            return out
        out.insert_seq = self.consensus
        out.left_ref_start = self.bp_left
        out.left_ref_end = self.bp_left
        out.right_ref_start = self.bp_right
        out.right_ref_end = self.bp_right
        out.pass_ = True
        out.qc_reason = "PASS_EVENT_SEGMENTATION_UNPLACED_INSERT"
        return out


def _window_bounds(bp_left: int, bp_right: int,
                   one_sided_max_flank_query_len: int) -> tuple[int, int, int, int]:
    """The two reference windows, ASYMMETRIC by design.

    The left window extends BACKWARD by a flank length (the left flank ends at
    the breakpoint, so it starts a flank-length before it); the right window
    extends FORWARD by one. A symmetric window would either miss the flank or
    double the search cost for nothing.
    """
    left_start = max(0, bp_left - BREAKPOINT_SLACK_BP - one_sided_max_flank_query_len)
    left_end = max(left_start + 1, bp_left + BREAKPOINT_SLACK_BP)
    right_start = max(0, bp_right - BREAKPOINT_SLACK_BP)
    right_end = max(right_start + 1,
                    bp_right + BREAKPOINT_SLACK_BP + one_sided_max_flank_query_len)
    return left_start, left_end, right_start, right_end


def segment_event_consensus(chrom: str, bp_left_in: int, bp_right_in: int,
                            alt_struct_reads: int, alt_ref_span_reads: int,
                            consensus: EventConsensus, config: PipelineConfig,
                            fetch_window: Callable[[str, int, int], str],
                            stats: SegmentationSearchStats | None = None,
                            _allow_revcomp_retry: bool = True) -> EventSegmentation:
    """Split the event consensus into flank | insert | flank.

    The order of attempts is the order of how much each claims:

      1. a PAIR of independently-placed flanks;
      2. a TETHERED pair -- one flank placed, the other searched for from the
         first one's edge rather than from the read-derived breakpoint;
      3. the best ONE-SIDED placement;
      4. the whole thing again on the reverse complement;
      5. the unplaced-insert fallback.

    Step 4 exists because the consensus orientation comes from whichever reads
    the assembler happened to start from, and a consensus built the wrong way
    round will fail every placement for a reason that has nothing to do with the
    locus. The retry is guarded so it cannot recurse.
    """
    stats = stats if stats is not None else SegmentationSearchStats()
    out = EventSegmentation()

    if not consensus.qc_pass or not consensus.consensus_seq:
        out.qc_reason = "NO_EVENT_CONSENSUS_TO_SEGMENT"
        return out

    consensus_len = len(consensus.consensus_seq)
    if consensus_len < (MIN_FLANK_ALIGN_BP + MIN_INSERT_BP):
        out.qc_reason = "EVENT_CONSENSUS_TOO_SHORT"
        return out

    bp_left = min(bp_left_in, bp_right_in)
    bp_right = max(bp_left_in, bp_right_in)
    if not chrom or bp_left < 0 or bp_right < 0:
        out.qc_reason = "INVALID_EVENT_BREAKPOINTS"
        return out

    paired_max_flank_query_len = min(MAX_FLANK_QUERY_BP,
                                     consensus_len - MIN_FLANK_ALIGN_BP - MIN_INSERT_BP)
    one_sided_max_flank_query_len = min(MAX_FLANK_QUERY_BP, consensus_len - MIN_INSERT_BP)

    left_start, left_end, right_start, right_end = _window_bounds(
        bp_left, bp_right, one_sided_max_flank_query_len)
    left_ref_window = fetch_window(chrom, left_start, left_end)
    right_ref_window = fetch_window(chrom, right_start, right_end)
    if not left_ref_window or not right_ref_window:
        out.qc_reason = "REFERENCE_WINDOW_FETCH_FAILED"
        return out

    seg = _Segmenter(chrom, bp_left, bp_right, consensus, config, fetch_window, stats)
    seg.alt_struct_reads = alt_struct_reads
    seg.alt_ref_span_reads = alt_ref_span_reads

    def search(is_left: bool, breakpoint: int, window_start: int, window: str,
               flank_len: int, one_sided: bool) -> list[EventFlankPlacement]:
        # Endpoint slack is a RETRY, never the first attempt: allowing the flank
        # to start inside the consensus lets it slide over the junction and eat
        # into the insert, so it is only permitted when nothing else worked.
        found = seg.collect_candidates(is_left, breakpoint, window_start, window,
                                       flank_len, 0, one_sided)
        if not found and ENDPOINT_SLACK_BP > 0:
            found = seg.collect_candidates(is_left, breakpoint, window_start, window,
                                           flank_len, ENDPOINT_SLACK_BP, one_sided)
        return found

    left_search = search(True, bp_left, left_start, left_ref_window,
                         paired_max_flank_query_len, False)
    right_search = search(False, bp_right, right_start, right_ref_window,
                          paired_max_flank_query_len, False)

    best_pair: tuple[EventFlankPlacement, EventFlankPlacement] | None = None
    # A pair's key leads with -min(identity), both lists run identity-first,
    # and the best pair's min only rises: a flank whose identity is already
    # strictly below it cannot be in a better pair, nor can any after it.
    best_key: tuple | None = None
    for left_candidate in left_search:
        if best_key is not None and left_candidate.identity < -best_key[0]:
            break
        for right_candidate in right_search:
            if best_key is not None and right_candidate.identity < -best_key[0]:
                break
            if left_candidate.query_end + MIN_INSERT_BP > right_candidate.query_start:
                continue
            key = _Segmenter.pair_key(left_candidate, right_candidate)
            if best_key is None or key < best_key:
                best_pair, best_key = (left_candidate, right_candidate), key

    tethered_pair = False
    if best_pair is None:
        for anchored_search, anchor_is_left in ((left_search, True), (right_search, False)):
            rescued = _try_tethered_pair(seg, anchored_search, anchor_is_left,
                                         one_sided_max_flank_query_len, search)
            if rescued is not None and (best_pair is None
                                        or _Segmenter.pair_key(*rescued)
                                        < _Segmenter.pair_key(*best_pair)):
                best_pair = rescued
                tethered_pair = True

    if best_pair is None:
        one_sided_left = search(True, bp_left, left_start, left_ref_window,
                                one_sided_max_flank_query_len, True)
        one_sided_right = search(False, bp_right, right_start, right_ref_window,
                                 one_sided_max_flank_query_len, True)
        if one_sided_left or one_sided_right:
            if one_sided_left and not one_sided_right:
                result = seg.emit_left_sided(one_sided_left[0])
            elif one_sided_right and not one_sided_left:
                result = seg.emit_right_sided(one_sided_right[0])
            elif (_Segmenter.better_one_sided(one_sided_left[0])
                  < _Segmenter.better_one_sided(one_sided_right[0])):
                result = seg.emit_left_sided(one_sided_left[0])
            else:
                result = seg.emit_right_sided(one_sided_right[0])
            if result.pass_:
                return result

        if _allow_revcomp_retry and consensus.qc_reason != "PASS_EVENT_CONSENSUS_REVCOMP_RETRY":
            reverse = EventConsensus(**consensus.__dict__)
            reverse.consensus_seq = reverse_complement(consensus.consensus_seq)
            reverse.consensus_len = len(reverse.consensus_seq)
            if reverse.consensus_seq and reverse.consensus_seq != consensus.consensus_seq:
                reverse.qc_reason = "PASS_EVENT_CONSENSUS_REVCOMP_RETRY"
                stats.reverse_complement_retries += 1
                retry = segment_event_consensus(chrom, bp_left, bp_right,
                                                alt_struct_reads, alt_ref_span_reads,
                                                reverse, config, fetch_window, stats,
                                                _allow_revcomp_retry=False)
                if retry.pass_:
                    return retry

        fallback = seg.emit_unplaced_insert()
        if fallback.pass_:
            return fallback
        out.qc_reason = "NO_TRIPARTITE_EVENT_SEGMENTATION"
        return out

    best_left, best_right = best_pair
    insert_start = best_left.query_end
    insert_len = best_right.query_start - best_left.query_end
    if insert_len < MIN_INSERT_BP:
        out.qc_reason = "EMPTY_EVENT_INSERT_SEGMENT"
        return out

    out.left_flank_seq = consensus.consensus_seq[best_left.query_start:
                                                 best_left.query_start + best_left.align_len]
    out.insert_seq = consensus.consensus_seq[insert_start:insert_start + insert_len]
    out.right_flank_seq = consensus.consensus_seq[best_right.query_start:
                                                  best_right.query_start + best_right.align_len]
    out.left_ref_start = best_left.ref_start
    out.left_ref_end = best_left.ref_end
    out.right_ref_start = best_right.ref_start
    out.right_ref_end = best_right.ref_end
    out.left_flank_align_len = best_left.align_len
    out.right_flank_align_len = best_right.align_len
    out.left_flank_identity = best_left.identity
    out.right_flank_identity = best_right.identity
    out.pass_ = True
    out.qc_reason = ("PASS_EVENT_SEGMENTATION_TETHERED_FLANK" if tethered_pair
                     else "PASS_EVENT_SEGMENTATION")
    return out


def _try_tethered_pair(seg: _Segmenter, anchored_search: list[EventFlankPlacement],
                       anchor_is_left: bool, one_sided_max_flank_query_len: int,
                       search) -> tuple[EventFlankPlacement, EventFlankPlacement] | None:
    """Find the second flank by searching from the FIRST one's edge.

    WHY IT IS NEEDED. The two breakpoints come from clustering read signals, and
    for a TSD-bearing insertion they can be tens of bases apart from the truth in
    OPPOSITE directions. The independent searches then each look in the wrong
    window and no pair forms -- even though one flank placed cleanly. Searching
    for the second flank around the FIRST one's reference edge removes the
    read-derived estimate from the second search entirely.

    THE DIAGONAL CHECK IS THE SAFEGUARD, and it is a refusal rather than a
    ranking. If the anchor -- or the rescued opposite flank -- has near-equal
    alternative placements on DIFFERENT diagonals, then the flank sequence
    occurs more than once in the window and the tether would be attaching to an
    arbitrary copy. The whole rescue is abandoned in that case rather than
    picking the best-scoring copy, because a repeat is exactly where a wrong
    junction looks most convincing.
    """
    if not anchored_search:
        return None
    anchor = anchored_search[0]
    anchor_boundary = anchor.ref_end if anchor_is_left else anchor.ref_start
    anchor_diagonal = anchor.ref_start - anchor.query_start
    for alternative in anchored_search:
        if alternative.identity + UNIQUENESS_MARGIN < anchor.identity:
            break
        if abs((alternative.ref_start - alternative.query_start) - anchor_diagonal) \
                > PLACEMENT_DIAGONAL_TOLERANCE_BP:
            return None

    missing_is_left = not anchor_is_left
    if anchor_boundary < 0:
        return None
    if missing_is_left:
        window_start = max(0, anchor_boundary - BREAKPOINT_SLACK_BP
                           - one_sided_max_flank_query_len)
        window_end = max(window_start + 1, anchor_boundary + BREAKPOINT_SLACK_BP)
    else:
        window_start = max(0, anchor_boundary - BREAKPOINT_SLACK_BP)
        window_end = max(window_start + 1, anchor_boundary + BREAKPOINT_SLACK_BP
                         + one_sided_max_flank_query_len)
    ref_window = seg.fetch_window(seg.chrom, window_start, window_end)
    if not ref_window:
        return None

    opposite_search = search(missing_is_left, anchor_boundary, window_start, ref_window,
                             one_sided_max_flank_query_len, False)

    best: tuple[EventFlankPlacement, EventFlankPlacement] | None = None
    for opposite in opposite_search:
        left = anchor if anchor_is_left else opposite
        right = opposite if anchor_is_left else anchor
        if not seg.canonical_pair(left, right):
            continue
        if best is None or _Segmenter.pair_key(left, right) < _Segmenter.pair_key(*best):
            best = (left, right)
    if best is None:
        return None

    best_opposite = best[1] if anchor_is_left else best[0]
    best_opposite_diagonal = best_opposite.ref_start - best_opposite.query_start
    for opposite in opposite_search:
        if opposite.identity + UNIQUENESS_MARGIN < best_opposite.identity:
            break
        left = anchor if anchor_is_left else opposite
        right = opposite if anchor_is_left else anchor
        if not seg.canonical_pair(left, right):
            continue
        candidate = right if anchor_is_left else left
        if abs((candidate.ref_start - candidate.query_start) - best_opposite_diagonal) \
                > PLACEMENT_DIAGONAL_TOLERANCE_BP:
            return None
    return best
