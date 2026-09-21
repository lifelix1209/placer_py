"""
Where exactly: breakpoint hypotheses, ranked by the kind of evidence behind them.

Ported from `src/pipeline/pipeline_breakpoint_helpers.inc`,
`src/pipeline/pipeline_breakpoint_stage.inc` and the split-read helper in
`src/pipeline/pipeline_local_alignment_helpers.inc`, pinned by
`tests/test_26_breakpoints.py`.

THE PRIORITY LADDER IS THE WHOLE STAGE. Six sources of a breakpoint position,
each weighted by how specifically it locates the junction:

    priority  source                 weight
    0         fragment split          8
    1         fragment indel          8
    2         raw split               7
    3         fragment clip pair      1
    4         raw indel               6
    5         raw clip pair           1

A hypothesis's rank is `support x weight`, so ONE fragment-split read outweighs
SEVEN clip reads. That is the correct ratio and not a conservative one: a clip
says "the read stops explaining the reference near here", which is compatible
with an insertion anywhere within the clip's own length, while a split or a
CIGAR insertion names a base.

TWO THINGS TO NOTICE IN THE WEIGHTS. Fragment-derived evidence outranks the raw
read signal of the SAME kind (8 vs 7, 8 vs 6) because the fragment stage already
checked that the junction is consistent with an extractable insert. And both
clip rows are 1 rather than, say, 3 -- clips are kept to break ties and to keep a
clip-only locus alive, not to decide anything on their own.

THE CONSENSUS IS NOT PORTED. The C++ calls abPOA (a C library) to build a
partial-order-alignment consensus of the event strings. There is no pure-Python
equivalent worth writing, and a worse consensus would change every downstream
number -- so `consensus_of` implements the two cases that need no library (zero
and one sequence) and otherwise raises. `placer_py/consensus.py` documents the
options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from placer_py.alignment import (
    CIGAR_I,
    CIGAR_S,
    AlignedRead,
    consumes_ref,
    find_first_non_hard_clip,
    find_last_non_hard_clip,
    median_i32,
    normalized_primary_alignment,
    normalized_sa_alignment,
    parse_sa_tag_z,
)
from placer_py.clustering import (
    CANDIDATE_LONG_INSERTION,
    CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
    ComponentCall,
)
from placer_py.fragments import InsertionFragment, InsertionFragmentSource, ReferenceSide
from placer_py.windows import LONG_INSERTION_SIGNAL_MIN, SOFT_CLIP_SIGNAL_MIN

#: Positions further apart than this are different breakpoints.
BREAKPOINT_CLUSTER_GAP_BP = 75
#: A paired hypothesis whose two sides are further apart than this is not one
#: insertion. 250 bp is generous -- it has to admit a real TSD plus breakpoint
#: uncertainty -- and the pair is dropped rather than reported as a long one.
BREAKPOINT_PAIR_COMPATIBILITY_BP = 250
MAX_CLUSTERS_PER_SIDE = 6
MAX_SINGLE_CLUSTERS = 2
#: How far outside the seed interval to look for positions.
EVENT_BREAKPOINT_SEARCH_SLACK_BP = 200
PRECISE_FRAGMENT_SEARCH_SLACK_BP = 200
#: A precise fragment further than this from the seed belongs to another event.
PRECISE_FRAGMENT_COMPONENT_LINK_BP = 1000


@dataclass
class BreakpointPositionCluster:
    center: int = -1
    start: int = -1
    end: int = -1
    support: int = 0


@dataclass
class LocalBreakpointHypothesis:
    valid: bool = False
    left: int = -1
    right: int = -1
    center: int = -1
    support: int = 0
    priority: int = 1 << 30


@dataclass
class LocalEventSignal:
    """What one read says about the event, within one window."""

    split: bool = False
    indel: bool = False
    left_clip: bool = False
    right_clip: bool = False
    split_left_pos: int = -1
    split_right_pos: int = -1
    indel_pos: int = -1
    max_raw_cigar_insert_len: int = 0
    left_clip_pos: int = -1
    right_clip_pos: int = -1

    def any(self) -> bool:
        return self.split or self.indel or self.left_clip or self.right_clip


def collect_breakpoint_position_clusters(positions: list[int]
                                         ) -> list[BreakpointPositionCluster]:
    """Single-linkage clustering on a sorted line, at a 75 bp gap.

    The centre is a MEDIAN of the cluster's members, so it names a position some
    read proposed. `support` counts POSITIONS, not distinct reads -- one read
    contributing two positions to the same cluster counts twice -- which is why
    the callers feed in one position per read per kind.
    """
    clusters: list[BreakpointPositionCluster] = []
    if not positions:
        return clusters
    ordered = sorted(positions)

    cluster_begin = 0
    for i in range(1, len(ordered) + 1):
        split_cluster = (i == len(ordered)
                         or (ordered[i] - ordered[i - 1]) > BREAKPOINT_CLUSTER_GAP_BP)
        if not split_cluster:
            continue
        members = ordered[cluster_begin:i]
        clusters.append(BreakpointPositionCluster(
            center=median_i32(members), start=members[0], end=members[-1],
            support=len(members)))
        cluster_begin = i
    return clusters


def breakpoint_hypothesis_support_weight(priority: int) -> int:
    """The ladder. See the module docstring for why the ratios are these."""
    return {0: 8, 1: 8, 2: 7, 3: 1, 4: 6, 5: 1}.get(priority, 1)


def mad_from_positions(positions: list[int]) -> float:
    """Median absolute deviation: how tightly the reads agree on the position.

    Median-based rather than a standard deviation because a single mislocated
    clip 5 kb away would otherwise dominate the spread and make a well-agreed
    breakpoint look uncertain.
    """
    if not positions:
        return 0.0
    center = median_i32(positions)
    return float(median_i32([abs(pos - center) for pos in positions]))


# ---------------------------------------------------------------------------
# QC tokens: a `|`-delimited set, kept deduplicated and order-preserving.
# ---------------------------------------------------------------------------
def has_qc_token(qc: str, token: str) -> bool:
    """WHOLE-token match, not a substring.

    `PASS_TE` must not match inside `PASS_TE_CALIBRATED`, and a substring test
    would make every token a prefix trap for every later one.
    """
    if not token:
        return False
    return token in qc.split("|")


def append_qc_token(qc: str, token: str) -> str:
    if not token or has_qc_token(qc, token):
        return qc
    return f"{qc}|{token}" if qc else token


def has_prefix(value: str, prefix: str) -> bool:
    return value.startswith(prefix)


# ---------------------------------------------------------------------------
# Banded edit identity.
# ---------------------------------------------------------------------------
def max_edits_for_identity_threshold(lhs_len: int, rhs_len: int,
                                     min_identity: float) -> int:
    """How many edits an identity threshold allows, over the LONGER sequence.

    Using the longer length as the denominator is what makes a length difference
    cost identity: comparing a 100 bp flank against a 60 bp one can never exceed
    0.6 identity, however well the 60 bases match.
    """
    denom = max(lhs_len, rhs_len)
    if denom <= 0:
        return 0
    clamped = min(1.0, max(0.0, min_identity))
    return max(0, math.floor(((1.0 - clamped) * denom) + 1e-9))


def edit_identity_if_at_least(lhs: str, rhs: str, max_edits: int) -> float | None:
    """Banded Levenshtein: the identity, or None if it cannot reach the bound.

    The band is the point. Full edit distance is O(n*m) and this stage runs it
    for every flank against every candidate placement; restricting the DP to
    `|i - j| <= max_edits` makes it O(n * max_edits), which for a 0.90 identity
    threshold on a 100 bp flank is ten columns instead of a hundred.

    Returning None rather than a low identity is deliberate -- the caller is
    asking a YES/NO question with a threshold attached, and a number below the
    threshold was never computed exactly.
    """
    n = len(lhs)
    m = len(rhs)
    if n <= 0 or m <= 0 or max_edits < 0:
        return None
    if abs(n - m) > max_edits:
        return None

    inf = max_edits + 1
    prev = [inf] * (m + 1)
    curr = [inf] * (m + 1)
    for j in range(min(m, max_edits) + 1):
        prev[j] = j

    # THE INNER LOOP IS 69% OF A REAL RUN, so it is written for the
    # interpreter rather than for looks. Measured on a 200 kb slice of
    # ultra-long ONT: 821,575 calls to this function, 214 s of 461 s in its
    # own frame, plus 101 s in 1.27 BILLION calls to builtins.min -- three per
    # cell. Every change below is exact; the returned distance is unchanged.
    #
    #   * two buffers are reused instead of allocating a full-width row per
    #     i. Only the band is ever read, so `[inf] * (m+1)` per row was
    #     allocating ~121 slots to use ~25 of them, n times.
    #   * `min(a, b, c)` becomes two comparisons. The builtin has to build an
    #     argument tuple and dispatch; at 1.27e9 calls that dominates the
    #     arithmetic it performs.
    #   * `prev[j-1]` and `curr[j-1]` become rolling locals (`diag`, `left`),
    #     which removes two list index operations per cell.
    #
    # REUSING THE BUFFERS IS ONLY SAFE because of what each row reads. Row i
    # writes [j_lo-1, j_hi] and then forces j_hi+1 to inf; row i+1 reads
    # prev over [j_lo'-1, j_hi'] with j_lo' >= j_lo and j_hi' <= j_hi+1, so
    # every cell it reads was written by row i. Drop the `curr[j_hi+1] = inf`
    # line and a stale value from two rows back leaks into the band.
    for i in range(1, n + 1):
        j_lo = i - max_edits
        if j_lo < 1:
            j_lo = 1
        j_hi = i + max_edits
        if j_hi > m:
            j_hi = m
        if j_lo > j_hi:
            return None

        left = (i if i <= max_edits else inf) if j_lo == 1 else inf
        curr[j_lo - 1] = left
        diag = prev[j_lo - 1]
        lhs_char = lhs[i - 1]

        row_min = inf
        for j in range(j_lo, j_hi + 1):
            up = prev[j]
            best = diag if lhs_char == rhs[j - 1] else diag + 1
            candidate = up + 1
            if candidate < best:
                best = candidate
            candidate = left + 1
            if candidate < best:
                best = candidate
            curr[j] = best
            if best < row_min:
                row_min = best
            diag = up
            left = best

        # EXACT EARLY EXIT, not a heuristic. The minimum of a row is
        # non-decreasing in i: every cell is built from `prev[j-1] + cost`,
        # `prev[j] + 1` or `curr[j-1] + 1`, and each of those is at least the
        # previous row's minimum (cost >= 0, and curr[j-1] >= this row's
        # minimum by induction). So once a row's minimum exceeds the budget,
        # no later row can come back under it and the answer is already None.
        #
        # This matters because roughly half of all calls return None -- the
        # caller is asking a thresholded yes/no question and most placements
        # are a no -- and those previously paid for the whole matrix.
        if row_min > max_edits:
            return None

        if j_hi < m:
            curr[j_hi + 1] = inf
        prev, curr = curr, prev

    dist = prev[m]
    if dist > max_edits:
        return None
    return min(1.0, max(0.0, 1.0 - (dist / max(n, m))))


def fixed_window_edit_identity(query: str, target: str) -> float:
    """Unbanded identity: the band is set to the longer length, so nothing is
    refused. Used where the answer is wanted regardless of threshold."""
    if not query or not target:
        return 0.0
    identity = edit_identity_if_at_least(query, target, max(len(query), len(target)))
    return identity if identity is not None else 0.0


# ---------------------------------------------------------------------------
# Per-read event signal.
# ---------------------------------------------------------------------------
def robust_local_split_insertion_positions(read: AlignedRead, chrom: str) -> list[int]:
    """Every same-strand, same-contig SA pair implying a long insertion.

    Unlike the geometry stage's `append_best_split_signature`, which keeps only
    the LONGEST, this keeps them all -- because here the question is "which
    positions should be considered", not "how much does this read support".
    """
    positions: list[int] = []
    sa_z = read.get_string_tag("SA")
    if not sa_z:
        return positions
    primary = normalized_primary_alignment(read, chrom)
    if primary is None:
        return positions

    for entry in parse_sa_tag_z(sa_z):
        if entry.rname != chrom:
            continue
        mate = normalized_sa_alignment(entry, read.seq_len)
        if mate is None or mate.is_reverse != primary.is_reverse:
            continue
        left, right = (primary, mate) if mate.qstart >= primary.qstart else (mate, primary)
        query_gap = right.qstart - left.qend
        ref_gap = max(0, right.ref_start - left.ref_end)
        if (query_gap - ref_gap) >= LONG_INSERTION_SIGNAL_MIN:
            positions.append(left.ref_end)
    return sorted(set(positions))


def classify_local_event_signal(read: AlignedRead, chrom: str, window_start: int,
                                window_end: int) -> LocalEventSignal:
    """What one read says about the event inside this window.

    `max_raw_cigar_insert_len` is recorded for EVERY insertion in the window,
    including ones below the long-insertion floor, while `indel` and `indel_pos`
    are set only above it. The short ones cannot locate a breakpoint but they do
    say the locus carries inserted sequence, and finalization reads that field
    to decide whether a call has any raw CIGAR support at all.

    The chosen `indel_pos` is the one CLOSEST TO THE WINDOW CENTRE, not the
    longest or the first: the window already decided where the event is, and the
    read is being asked which of its insertions is that one.
    """
    signal = LocalEventSignal()
    if read is None or not read.cigar:
        return signal
    split_positions = robust_local_split_insertion_positions(read, chrom)

    first = find_first_non_hard_clip(read.cigar)
    last = find_last_non_hard_clip(read.cigar)

    ref_pos = read.pos
    window_center = window_start + ((window_end - window_start) // 2)
    best_indel_dist = None
    for i, (op, length) in enumerate(read.cigar):
        if (i == first and op == CIGAR_S and length >= SOFT_CLIP_SIGNAL_MIN
                and window_start <= ref_pos <= window_end):
            signal.left_clip = True
            signal.left_clip_pos = ref_pos
        if (i == last and op == CIGAR_S and length >= SOFT_CLIP_SIGNAL_MIN
                and window_start <= ref_pos <= window_end):
            signal.right_clip = True
            signal.right_clip_pos = ref_pos
        if op == CIGAR_I and window_start <= ref_pos <= window_end:
            signal.max_raw_cigar_insert_len = max(signal.max_raw_cigar_insert_len, length)
            if length >= LONG_INSERTION_SIGNAL_MIN:
                signal.indel = True
                dist = abs(ref_pos - window_center)
                if best_indel_dist is None or dist < best_indel_dist:
                    best_indel_dist = dist
                    signal.indel_pos = ref_pos
        if consumes_ref(op):
            ref_pos += length

    for pos in split_positions:
        if pos < window_start or pos > window_end:
            continue
        signal.split = True
        if signal.split_left_pos < 0 or pos < signal.split_left_pos:
            signal.split_left_pos = pos
        if signal.split_right_pos < 0 or pos > signal.split_right_pos:
            signal.split_right_pos = pos
    return signal


def read_has_local_event_signal(read: AlignedRead, chrom: str, window_start: int,
                                window_end: int) -> bool:
    return classify_local_event_signal(read, chrom, window_start, window_end).any()


def update_best_bp_by_read(bp_by_read: dict[int, int], read_index: int, pos: int,
                           anchor_pos: int) -> None:
    """Keep ONE breakpoint per read: the one nearest the component anchor.

    One read can propose several positions -- a clip at each end plus an
    insertion -- and letting it contribute all of them would make a single noisy
    read look like three supporting reads. Ties break toward the smaller
    position so the result does not depend on iteration order.
    """
    if pos < 0:
        return
    current = bp_by_read.get(read_index)
    if current is None:
        bp_by_read[read_index] = pos
        return
    cur_dist = abs(current - anchor_pos)
    new_dist = abs(pos - anchor_pos)
    if new_dist < cur_dist or (new_dist == cur_dist and pos < current):
        bp_by_read[read_index] = pos


def infer_component_breakpoint_bounds(component: ComponentCall) -> tuple[int, int]:
    """The span of proposed breakpoints, or the anchor twice if there are none."""
    positions = [bp.pos for bp in component.breakpoint_candidates if bp.pos >= 0]
    if positions:
        return min(positions), max(positions)
    fallback = max(0, component.anchor_pos)
    return fallback, fallback


# ---------------------------------------------------------------------------
# Hypothesis enumeration.
# ---------------------------------------------------------------------------
def _sort_clusters(clusters: list[BreakpointPositionCluster],
                   anchor_pos: int) -> list[BreakpointPositionCluster]:
    """Most supported first, then nearest the anchor, then leftmost.

    The anchor tie-break matters more than it looks: two clusters with equal
    support at a repetitive locus are common, and without it the chosen one
    would depend on sort stability.
    """
    return sorted(clusters, key=lambda c: (-c.support, abs(c.center - anchor_pos),
                                           c.center))


def _single_hypotheses(positions: list[int], priority: int,
                       anchor_pos: int) -> list[LocalBreakpointHypothesis]:
    """Hypotheses from ONE-SIDED evidence: left and right are the same base.

    A CIGAR insertion has no two sides -- it names one reference position and
    the inserted bases sit between it and the next base. Reporting `left ==
    right` is the honest encoding, and the boundary stage then reads a zero
    link delta as BLUNT.
    """
    clusters = _sort_clusters(collect_breakpoint_position_clusters(positions),
                              anchor_pos)[:MAX_SINGLE_CLUSTERS]
    return [LocalBreakpointHypothesis(valid=cluster.center >= 0, left=cluster.center,
                                      right=cluster.center, center=cluster.center,
                                      support=cluster.support, priority=priority)
            for cluster in clusters]


def _paired_hypotheses(left_positions: list[int], right_positions: list[int],
                       priority: int, anchor_pos: int
                       ) -> list[LocalBreakpointHypothesis]:
    """Hypotheses from two-sided evidence: every compatible left x right pair.

    When only ONE side has clusters the pairing degenerates to single
    hypotheses rather than producing nothing -- a locus where every read clips
    on the same side still has a breakpoint, it just has no junction geometry,
    and the boundary stage is where that costs it.

    The cross product is capped at 6 x 6 and then filtered by the 250 bp
    compatibility span, so a repetitive locus cannot emit hundreds of pairs.
    """
    left_clusters = _sort_clusters(collect_breakpoint_position_clusters(left_positions),
                                   anchor_pos)[:MAX_CLUSTERS_PER_SIDE]
    right_clusters = _sort_clusters(collect_breakpoint_position_clusters(right_positions),
                                    anchor_pos)[:MAX_CLUSTERS_PER_SIDE]

    if not left_clusters and not right_clusters:
        return []
    if not left_clusters or not right_clusters:
        only = right_clusters or left_clusters
        return [LocalBreakpointHypothesis(valid=c.center >= 0, left=c.center,
                                          right=c.center, center=c.center,
                                          support=c.support, priority=priority)
                for c in only]

    out: list[LocalBreakpointHypothesis] = []
    for left_cluster in left_clusters:
        for right_cluster in right_clusters:
            left = min(left_cluster.center, right_cluster.center)
            right = max(left_cluster.center, right_cluster.center)
            if (right - left) > BREAKPOINT_PAIR_COMPATIBILITY_BP:
                continue
            out.append(LocalBreakpointHypothesis(
                valid=True, left=left, right=right, center=(left + right) // 2,
                support=left_cluster.support + right_cluster.support,
                priority=priority))
    return out


def collect_breakpoint_hypotheses(component: ComponentCall,
                                  local_records: list[AlignedRead],
                                  fragments: list[InsertionFragment],
                                  seed_left: int, seed_right: int,
                                  top_k: int = 0) -> list[LocalBreakpointHypothesis]:
    """Enumerate and rank the breakpoint hypotheses for one component.

    THE SEARCH INTERVAL EXPANDS TOWARD PRECISE EVIDENCE. It starts as the seed
    plus 200 bp, and then every precise signal -- a long-insertion or split
    breakpoint candidate, or a fragment junction within 1 kb of the seed --
    widens it to include that position plus 200. So a component whose median
    anchor is off by several hundred bases can still find its real breakpoint,
    while a precise fragment belonging to a DIFFERENT event 5 kb away cannot
    drag the search onto it.

    THE FALLBACK IS NEVER EMPTY. If no source produced a position, the component
    anchor becomes a hypothesis at the lowest possible priority. A component
    that reached this stage exists; refusing to name a breakpoint for it would
    silently drop it, and the evidence stages are where it should be judged.
    """
    search_start = max(0, min(seed_left, seed_right) - EVENT_BREAKPOINT_SEARCH_SLACK_BP)
    search_end = max(search_start + 1,
                     max(seed_left, seed_right) + EVENT_BREAKPOINT_SEARCH_SLACK_BP)

    def expand_to(pos: int) -> None:
        nonlocal search_start, search_end
        if pos < 0:
            return
        search_start = max(0, min(search_start, pos - PRECISE_FRAGMENT_SEARCH_SLACK_BP))
        search_end = max(search_end, pos + PRECISE_FRAGMENT_SEARCH_SLACK_BP)

    def linked_to_seed(pos: int) -> bool:
        if pos < 0:
            return False
        seed_min = min(seed_left, seed_right)
        seed_max = max(seed_left, seed_right)
        distance = (seed_min - pos) if pos < seed_min else (
            (pos - seed_max) if pos > seed_max else 0)
        return distance <= PRECISE_FRAGMENT_COMPONENT_LINK_BP

    for candidate in component.breakpoint_candidates:
        if ((candidate.class_mask & CANDIDATE_LONG_INSERTION)
                or (candidate.class_mask & CANDIDATE_SPLIT_SA_SUPPLEMENTARY)
                or candidate.ins_len >= LONG_INSERTION_SIGNAL_MIN):
            expand_to(candidate.pos)
    for fragment in fragments:
        if (fragment.source in (InsertionFragmentSource.CIGAR_INSERTION,
                                InsertionFragmentSource.SPLIT_SA)
                and linked_to_seed(fragment.ref_junc_pos)):
            expand_to(fragment.ref_junc_pos)

    buckets: dict[str, list[int]] = {
        "fragment_split_left": [], "fragment_split_right": [], "fragment_indel": [],
        "fragment_clip_left": [], "fragment_clip_right": [], "raw_split_left": [],
        "raw_split_right": [], "raw_indel": [], "raw_clip_left": [], "raw_clip_right": [],
    }

    def add(name: str, pos: int) -> None:
        if search_start <= pos <= search_end:
            buckets[name].append(pos)

    for read in local_records:
        if read is None or read.tid != component.tid:
            continue
        signal = classify_local_event_signal(read, component.chrom, search_start,
                                             search_end)
        has_left = signal.split_left_pos >= 0
        has_right = signal.split_right_pos >= 0
        if has_left and has_right:
            # One read, ONE split vote: the side nearer the anchor. A read whose
            # two split positions straddle the anchor would otherwise support
            # both sides of a pair by itself.
            left_dist = abs(signal.split_left_pos - component.anchor_pos)
            right_dist = abs(signal.split_right_pos - component.anchor_pos)
            if left_dist < right_dist or (left_dist == right_dist
                                          and signal.split_left_pos <= signal.split_right_pos):
                add("raw_split_left", signal.split_left_pos)
            else:
                add("raw_split_right", signal.split_right_pos)
        else:
            add("raw_split_left", signal.split_left_pos)
            add("raw_split_right", signal.split_right_pos)
        add("raw_clip_left", signal.left_clip_pos)
        add("raw_clip_right", signal.right_clip_pos)
        add("raw_indel", signal.indel_pos)

    for fragment in fragments:
        if not (search_start <= fragment.ref_junc_pos <= search_end):
            continue
        if fragment.source == InsertionFragmentSource.SPLIT_SA:
            if fragment.ref_side == ReferenceSide.REF_LEFT:
                buckets["fragment_split_left"].append(fragment.ref_junc_pos)
            elif fragment.ref_side == ReferenceSide.REF_RIGHT:
                buckets["fragment_split_right"].append(fragment.ref_junc_pos)
        elif fragment.source == InsertionFragmentSource.CIGAR_INSERTION:
            buckets["fragment_indel"].append(fragment.ref_junc_pos)
        elif fragment.source == InsertionFragmentSource.CLIP_REF_LEFT:
            buckets["fragment_clip_left"].append(fragment.ref_junc_pos)
        elif fragment.source == InsertionFragmentSource.CLIP_REF_RIGHT:
            buckets["fragment_clip_right"].append(fragment.ref_junc_pos)

    hypotheses: list[LocalBreakpointHypothesis] = []

    def append_unique(batch: list[LocalBreakpointHypothesis]) -> None:
        # De-duplicated on (left, right), FIRST WINS -- and the batches are
        # appended in priority order, so a hypothesis found by strong evidence
        # keeps its priority even when weaker evidence proposes the same pair.
        for hypothesis in batch:
            if not hypothesis.valid:
                continue
            if any(existing.left == hypothesis.left and existing.right == hypothesis.right
                   for existing in hypotheses):
                continue
            hypotheses.append(hypothesis)

    anchor = component.anchor_pos
    append_unique(_paired_hypotheses(buckets["fragment_split_left"],
                                     buckets["fragment_split_right"], 0, anchor))
    append_unique(_single_hypotheses(buckets["fragment_indel"], 1, anchor))
    append_unique(_paired_hypotheses(buckets["raw_split_left"],
                                     buckets["raw_split_right"], 2, anchor))
    append_unique(_paired_hypotheses(buckets["fragment_clip_left"],
                                     buckets["fragment_clip_right"], 3, anchor))
    append_unique(_single_hypotheses(buckets["raw_indel"], 4, anchor))
    append_unique(_paired_hypotheses(buckets["raw_clip_left"],
                                     buckets["raw_clip_right"], 5, anchor))

    if not hypotheses:
        fallback = max(0, component.anchor_pos)
        hypotheses.append(LocalBreakpointHypothesis(
            valid=True, left=fallback, right=fallback, center=fallback, support=1,
            priority=1 << 30))

    hypotheses.sort(key=lambda h: (
        -(h.support * breakpoint_hypothesis_support_weight(h.priority)),
        abs(h.center - anchor), -h.support, h.priority, h.left, h.right))
    if top_k > 0 and len(hypotheses) > top_k:
        hypotheses = hypotheses[:top_k]
    return hypotheses


# ---------------------------------------------------------------------------
# Diverse selection: the top-k that are actually DIFFERENT hypotheses.
# ---------------------------------------------------------------------------
#: Two hypotheses whose ends both agree within this are the same locus.
BREAKPOINT_DIVERSITY_SLACK_BP = 30
#: A selected hypothesis this close to the component anchor counts as covering
#: the anchor.
ANCHOR_PROXIMAL_SLACK_BP = 100
#: Score a rescued anchor-proximal hypothesis must reach. 8 is one
#: fragment-split read, i.e. the weakest evidence that names a base.
ANCHOR_PROXIMAL_MIN_SCORE = 8
#: A rescued single-read precise hypothesis must be this far from everything
#: already selected, so the rescue adds a LOCUS rather than a variant of one.
RESCUE_PRECISE_ANCHOR_MIN_DISTANCE_BP = 500


def hypothesis_score(hypothesis: LocalBreakpointHypothesis) -> int:
    return hypothesis.support * breakpoint_hypothesis_support_weight(hypothesis.priority)


def select_diverse_breakpoint_hypotheses(hypotheses: list[LocalBreakpointHypothesis],
                                         top_k: int, anchor_pos: int
                                         ) -> list[LocalBreakpointHypothesis]:
    """Take the best `top_k`, but not `top_k` variants of the same junction.

    WHY DIVERSITY AND NOT JUST RANK. The ranked list at a real locus is mostly
    near-duplicates: the same junction reported by clips at 1000/1020 and by
    fragments at 1003/1029. Taking the top three by score would spend all three
    slots on one junction and lose a genuinely different hypothesis 300 bp away
    -- and the downstream stages evaluate each hypothesis independently, so a
    hypothesis not selected here is never considered at all.

    TWO RESCUES, both for the same failure: a strong but WRONG junction crowding
    out the right one.

      * anchor-proximal -- if nothing selected is within 100 bp of the component
        anchor, the best hypothesis that is gets a slot, REPLACING the weakest
        incumbent when the list is full. The anchor is where the geometry stage
        said the event is, and a hypothesis list that ignores it is usually
        chasing a nearby repeat.
      * precise-anchor -- a SINGLE-READ hypothesis from split or CIGAR-insertion
        evidence, at least 500 bp from everything selected, is appended even
        beyond `top_k`. One such read names a base; the clip cloud that outvoted
        it does not. This is the only place the list is allowed to exceed
        `top_k`, and it is deliberate.
    """
    if top_k == 0 or len(hypotheses) <= top_k:
        return list(hypotheses)

    def same_locus(lhs, rhs) -> bool:
        return (abs(lhs.left - rhs.left) <= BREAKPOINT_DIVERSITY_SLACK_BP
                and abs(lhs.right - rhs.right) <= BREAKPOINT_DIVERSITY_SLACK_BP)

    def selected_contains(selected, hypothesis) -> bool:
        return any(incumbent.left == hypothesis.left and incumbent.right == hypothesis.right
                   for incumbent in selected)

    selected: list[LocalBreakpointHypothesis] = []
    for hypothesis in hypotheses:
        if any(same_locus(hypothesis, incumbent) for incumbent in selected):
            continue
        selected.append(hypothesis)
        if len(selected) >= top_k:
            break

    def append_rescue_precise_anchor() -> None:
        rescue = None
        for hypothesis in hypotheses:
            if selected_contains(selected, hypothesis):
                continue
            if (hypothesis.priority not in (0, 1) or hypothesis.left != hypothesis.right
                    or hypothesis.support > 1):
                continue
            if not all(abs(hypothesis.center - incumbent.center)
                       >= RESCUE_PRECISE_ANCHOR_MIN_DISTANCE_BP for incumbent in selected):
                continue
            if rescue is None:
                rescue = hypothesis
                continue
            dist = abs(hypothesis.center - anchor_pos)
            best_dist = abs(rescue.center - anchor_pos)
            if (dist < best_dist
                    or (dist == best_dist and hypothesis.priority < rescue.priority)
                    or (dist == best_dist and hypothesis.priority == rescue.priority
                        and hypothesis.center < rescue.center)):
                rescue = hypothesis
        if rescue is not None:
            selected.append(rescue)

    has_anchor_proximal = any(abs(h.center - anchor_pos) <= ANCHOR_PROXIMAL_SLACK_BP
                              for h in selected)
    if has_anchor_proximal:
        append_rescue_precise_anchor()
        return selected

    best_anchor_candidate = None
    for hypothesis in hypotheses:
        if selected_contains(selected, hypothesis):
            continue
        if hypothesis_score(hypothesis) < ANCHOR_PROXIMAL_MIN_SCORE:
            continue
        if best_anchor_candidate is None:
            best_anchor_candidate = hypothesis
            continue
        key = (abs(hypothesis.center - anchor_pos), -hypothesis_score(hypothesis),
               -hypothesis.support, hypothesis.priority, hypothesis.left, hypothesis.right)
        best_key = (abs(best_anchor_candidate.center - anchor_pos),
                    -hypothesis_score(best_anchor_candidate),
                    -best_anchor_candidate.support, best_anchor_candidate.priority,
                    best_anchor_candidate.left, best_anchor_candidate.right)
        if key < best_key:
            best_anchor_candidate = hypothesis

    if best_anchor_candidate is None:
        return selected

    if len(selected) < top_k:
        selected.append(best_anchor_candidate)
    elif selected:
        # REPLACES the weakest incumbent rather than growing the list: the
        # anchor-proximal rescue is a correction, not an addition.
        selected[-1] = best_anchor_candidate
    append_rescue_precise_anchor()
    return selected


def enumerate_breakpoint_hypotheses(component: ComponentCall,
                                    local_records: list[AlignedRead],
                                    fragments: list[InsertionFragment],
                                    seed_left: int, seed_right: int,
                                    top_k: int) -> list[LocalBreakpointHypothesis]:
    """Collect every hypothesis, then select a diverse top-k from it.

    Note the `0`: collection is UNCAPPED and the cap is applied by the diverse
    selector. Capping during collection would throw away exactly the distant
    hypotheses the diversity rule exists to keep.
    """
    return select_diverse_breakpoint_hypotheses(
        collect_breakpoint_hypotheses(component, local_records, fragments,
                                      seed_left, seed_right, 0),
        top_k, component.anchor_pos)


def resolve_event_breakpoint_bounds(component: ComponentCall,
                                    local_records: list[AlignedRead],
                                    fragments: list[InsertionFragment],
                                    seed_left: int, seed_right: int) -> tuple[int, int]:
    """The single best hypothesis's bounds, or the anchor twice."""
    hypotheses = enumerate_breakpoint_hypotheses(component, local_records, fragments,
                                                 seed_left, seed_right, 1)
    if hypotheses:
        return hypotheses[0].left, hypotheses[0].right
    anchor = max(0, component.anchor_pos)
    return anchor, anchor
