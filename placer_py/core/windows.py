"""
Where to look: evidence points, a smoothed density, and candidate windows.

Ported from `src/pipeline/pipeline_window_helpers.inc` and the constants and
structs in `src/pipeline/pipeline_local_alignment_helpers.inc`, pinned by
`tests/test_25_windows.py`.

WHAT THIS STAGE IS FOR, and why it is not the same as clustering. A 30x human
BAM is streamed in 10 kb bins; most bins contain nothing. This stage answers
"which sub-intervals of this bin deserve the expensive stages", and it answers
it from a WEIGHTED DENSITY rather than from individual reads -- so it is
tolerant of the individual signals being imprecise, which they are. The
three-dimensional DBSCAN in `placer_py/core/clustering.py` runs afterwards, on reads,
and makes the sharp judgements.

THE WEIGHTS ARE THE MODEL. Every evidence point carries one:

    soft clip : 1.00 + min(len, 500) / 180      -> 1.11 .. 3.78
    insertion : 0.80 + min(len, 400) / 200      -> 1.00 .. 2.80
    SA hint   : 0.35, flat

A long clip outweighs a long insertion at the same length, which reads
backwards until you notice what the stage is for: a clip is how a LARGE
insertion presents when the read cannot span it, so it is the signal most likely
to mark an event the CIGAR could not. The SA hint is deliberately small -- it
says only "this read goes somewhere else too" -- and its job is to break ties
between nearby peaks, not to create one.

BOTH CAPS MATTER. Without them a single 20 kb clip would outweigh ten ordinary
reads and manufacture a peak on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from placer_py.alignment import (
    CIGAR_I,
    CIGAR_S,
    AlignedRead,
    consumes_ref,
    find_first_non_hard_clip,
    find_last_non_hard_clip,
)
from placer_py.core.clustering import (
    CANDIDATE_LONG_INSERTION,
    CANDIDATE_SOFT_CLIP,
    CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
)

#: Signal floors. Below these a clip or an indel is noise.
SOFT_CLIP_SIGNAL_MIN = 20
LONG_INSERTION_SIGNAL_MIN = 50
#: An indel worth its own evidence point -- lower than the insertion floor,
#: because a density peak may be built from several sub-threshold indels.
LARGE_INDEL_EVIDENCE_MIN = 40
#: Flat, and small on purpose: an SA tag says only "this read goes elsewhere
#: too", which breaks ties between peaks but should not create one.
SA_HINT_WEIGHT = 0.35

#: Histogram geometry.
HIST_BIN_SIZE = 20
HIST_PADDING = 120
#: A smoothed bin below this is not a peak. Roughly one good clip.
PEAK_MIN_WEIGHT = 1.2
PEAK_MERGE_DISTANCE = 100
WINDOW_MERGE_GAP = 60
#: Windows may cross bin boundaries: a long read's start can fall in an
#: adjacent bin, and clipping the window to the bin would drop its evidence.
WINDOW_BIN_SLACK_BP = 5000
#: Bins merged with the current one so split support from long reads can be
#: assembled into a single component.
CROSS_BIN_CONTEXT_BINS = 2
WINDOW_MAX_EXPAND_BINS = 15
WINDOW_DROP_RATIO = 0.50
MIN_WINDOW_SPAN = 120


class EvidenceKind(IntEnum):
    SOFT_CLIP = 0
    INDEL = 1
    SA_HINT = 2


class SaHintSide(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class EvidencePoint:
    read_index: int = 0
    pos: int = -1
    weight: float = 0.0
    kind: EvidenceKind = EvidenceKind.SOFT_CLIP
    signal_len: int = 0
    class_mask: int = 0


@dataclass
class ReadSignalSummary:
    read_id: str = ""
    is_reverse: bool = False
    has_sa_or_supp: bool = False
    max_soft_clip: int = 0
    max_ins: int = 0
    class_mask: int = 0


@dataclass
class EvidenceBundle:
    points: list[EvidencePoint] = field(default_factory=list)
    #: Indexed by position in the bin, so a read with no signal still has a
    #: slot. The later stages address reads by bin index, and a compacted list
    #: would silently shift every one of them.
    read_summaries: list[ReadSignalSummary] = field(default_factory=list)


@dataclass
class DensityPeak:
    pos: int = -1
    weight: float = 0.0


@dataclass
class CandidateWindow:
    start: int = -1
    end: int = -1
    center: int = -1
    peak_weight: float = 0.0


def choose_sa_hint_side(leading_soft: int, trailing_soft: int) -> SaHintSide:
    """Which end of the read the split most likely happened at.

    An EXACT TIE between two long clips returns NONE. Both ends are equally
    plausible, so placing the hint at either would be a coin flip recorded as
    evidence -- and the read still contributes its clip points from both ends.
    """
    left = leading_soft >= SOFT_CLIP_SIGNAL_MIN
    right = trailing_soft >= SOFT_CLIP_SIGNAL_MIN
    if left and not right:
        return SaHintSide.LEFT
    if right and not left:
        return SaHintSide.RIGHT
    if left and right:
        if leading_soft > trailing_soft:
            return SaHintSide.LEFT
        if trailing_soft > leading_soft:
            return SaHintSide.RIGHT
    return SaHintSide.NONE


def weighted_median_position(pos_weights: list[tuple[int, float]]) -> int:
    """The position where the cumulative weight first reaches half the total.

    Used only as a FALLBACK, when the histogram found no peak at all but there
    is evidence. A mean would be dragged by one distant clip; the weighted
    median names a position some evidence actually supports.
    """
    rows = sorted((row for row in pos_weights if row[1] > 0.0), key=lambda r: r[0])
    if not rows:
        return -1
    total_weight = sum(weight for _, weight in rows)
    if total_weight <= 0.0:
        return rows[len(rows) // 2][0]
    half_weight = 0.5 * total_weight
    acc = 0.0
    for pos, weight in rows:
        acc += weight
        if acc >= half_weight:
            return pos
    return rows[-1][0]


def smooth_histogram(hist: list[float]) -> list[float]:
    """A fixed five-tap smoother: weights 0.25, 0.60, 1.00, 0.60, 0.25.

    NOT normalised -- the taps sum to 2.7, so smoothing INFLATES the values and
    `PEAK_MIN_WEIGHT` is calibrated against the inflated scale. That is easy to
    "fix" and doing so would raise the effective peak threshold by 2.7x and
    silence most loci.

    Its purpose is to make nearby evidence of different kinds reinforce: a clip
    at 1000 and an insertion at 1015 fall in different 20 bp bins and would
    otherwise be two weak peaks instead of one strong one.
    """
    n = len(hist)
    smooth = [0.0] * n
    for i in range(n):
        value = hist[i]
        if i >= 1:
            value += 0.60 * hist[i - 1]
        if i + 1 < n:
            value += 0.60 * hist[i + 1]
        if i >= 2:
            value += 0.25 * hist[i - 2]
        if i + 2 < n:
            value += 0.25 * hist[i + 2]
        smooth[i] = value
    return smooth


def extract_evidence_points(bin_records: list[AlignedRead],
                            expected_tid: int) -> EvidenceBundle:
    """One pass over the bin: weighted points, plus a per-read summary.

    Supplementary records are skipped for the same reason as in the geometry
    stage -- the primary already carries the split through its SA tag, and
    admitting the supplementary would let one alignment vote twice.

    Note that a read with NO CIGAR still gets a summary (and its SA class bit)
    rather than being dropped: the summaries are indexed by bin position, and a
    missing entry would shift every later index.
    """
    bundle = EvidenceBundle(read_summaries=[ReadSignalSummary()
                                            for _ in bin_records])

    for idx, read in enumerate(bin_records):
        if read is None or read.tid != expected_tid or read.is_supplementary:
            continue

        summary = ReadSignalSummary(read_id=read.qname, is_reverse=read.is_reverse,
                                    has_sa_or_supp=read.has_sa_tag() or read.is_supplementary)
        if not read.cigar:
            if summary.has_sa_or_supp:
                summary.class_mask |= CANDIDATE_SPLIT_SA_SUPPLEMENTARY
            bundle.read_summaries[idx] = summary
            continue

        leading_soft = 0
        trailing_soft = 0
        first = find_first_non_hard_clip(read.cigar)
        last = find_last_non_hard_clip(read.cigar)
        if first >= 0 and read.cigar[first][0] == CIGAR_S:
            leading_soft = read.cigar[first][1]
        if last >= 0 and read.cigar[last][0] == CIGAR_S:
            trailing_soft = read.cigar[last][1]

        local_points: list[EvidencePoint] = []
        ref_pos = read.pos
        for op, length in read.cigar:
            if op == CIGAR_S:
                summary.max_soft_clip = max(summary.max_soft_clip, length)
            elif op == CIGAR_I:
                summary.max_ins = max(summary.max_ins, length)

            if op == CIGAR_I and length >= LARGE_INDEL_EVIDENCE_MIN:
                local_points.append(EvidencePoint(
                    read_index=idx, pos=ref_pos,
                    weight=0.80 + (min(length, 400) / 200.0),
                    kind=EvidenceKind.INDEL, signal_len=length,
                    class_mask=CANDIDATE_LONG_INSERTION))
            if consumes_ref(op):
                ref_pos += length
        ref_end = ref_pos

        if leading_soft >= SOFT_CLIP_SIGNAL_MIN:
            local_points.append(EvidencePoint(
                read_index=idx, pos=read.pos,
                weight=1.00 + (min(leading_soft, 500) / 180.0),
                kind=EvidenceKind.SOFT_CLIP, signal_len=leading_soft,
                class_mask=CANDIDATE_SOFT_CLIP))
        if trailing_soft >= SOFT_CLIP_SIGNAL_MIN:
            local_points.append(EvidencePoint(
                read_index=idx, pos=ref_end,
                weight=1.00 + (min(trailing_soft, 500) / 180.0),
                kind=EvidenceKind.SOFT_CLIP, signal_len=trailing_soft,
                class_mask=CANDIDATE_SOFT_CLIP))

        if summary.has_sa_or_supp:
            hint_side = choose_sa_hint_side(leading_soft, trailing_soft)
            if hint_side != SaHintSide.NONE:
                local_points.append(EvidencePoint(
                    read_index=idx,
                    pos=read.pos if hint_side == SaHintSide.LEFT else ref_end,
                    weight=SA_HINT_WEIGHT, kind=EvidenceKind.SA_HINT,
                    class_mask=CANDIDATE_SPLIT_SA_SUPPLEMENTARY))

        if summary.has_sa_or_supp:
            summary.class_mask |= CANDIDATE_SPLIT_SA_SUPPLEMENTARY
        if summary.max_soft_clip >= SOFT_CLIP_SIGNAL_MIN:
            summary.class_mask |= CANDIDATE_SOFT_CLIP
        if summary.max_ins >= LONG_INSERTION_SIGNAL_MIN:
            summary.class_mask |= CANDIDATE_LONG_INSERTION

        bundle.points.extend(local_points)
        bundle.read_summaries[idx] = summary

    return bundle


def _indel_only_peaks(evidence: list[EvidencePoint]) -> list[DensityPeak]:
    """Peaks built from long indels alone, regardless of the smoothed density.

    WHY THIS EXISTS. A long-indel locus sitting beside a clip-heavy repeat is
    swamped in the smoothed histogram -- the clips' weights are larger and their
    peak absorbs it. But a CIGAR insertion is breakpoint-specific in a way a clip
    is not: the aligner placed those bases at that position. So indel points get
    to seed their own windows even when the density says otherwise.
    """
    indel_points = sorted((p for p in evidence if p.kind == EvidenceKind.INDEL),
                          key=lambda p: p.pos)
    peaks: list[DensityPeak] = []
    begin = 0
    while begin < len(indel_points):
        end = begin + 1
        while (end < len(indel_points)
               and abs(indel_points[end].pos - indel_points[end - 1].pos) <= PEAK_MERGE_DISTANCE):
            end += 1
        total_weight = sum(p.weight for p in indel_points[begin:end])
        if total_weight >= PEAK_MIN_WEIGHT:
            weighted_pos = sum(p.weight * p.pos for p in indel_points[begin:end])
            peaks.append(DensityPeak(pos=round(weighted_pos / total_weight),
                                     weight=total_weight))
        begin = end
    return peaks


def build_density_windows(evidence: list[EvidencePoint], bin_start: int,
                          bin_end: int) -> list[CandidateWindow]:
    """Turn weighted evidence into merged candidate windows.

    The shape of the algorithm:

      1. bin the weights at 20 bp and smooth;
      2. take local maxima above the floor, PLUS the indel-only peaks;
      3. if nothing qualified, fall back to the weighted median so a locus with
         real but diffuse evidence is still examined;
      4. merge peaks within 100 bp, weighting the merged centre;
      5. expand each peak until the density drops below half its own centre,
         capped at 15 bins either way;
      6. clip to the bin plus 5 kb of slack, and merge windows within 60 bp.

    The expansion threshold is RELATIVE (`center * 0.50`, floored at 0.50), so a
    strong peak keeps a tight window and a weak one is allowed to be broad --
    which is the right way round: a strong peak's position is well determined,
    and a weak one's is not.
    """
    if not evidence:
        return []

    min_pos = min(p.pos for p in evidence)
    max_pos = max(p.pos for p in evidence)
    hist_start = min(min_pos, bin_start) - HIST_PADDING
    hist_end = max(max_pos, bin_end) + HIST_PADDING
    num_bins = max(1, ((hist_end - hist_start) // HIST_BIN_SIZE) + 1)

    hist = [0.0] * num_bins
    for point in evidence:
        idx = (point.pos - hist_start) // HIST_BIN_SIZE
        idx = max(0, min(num_bins - 1, idx))
        hist[idx] += point.weight
    smooth = smooth_histogram(hist)

    peaks: list[DensityPeak] = []
    for i in range(num_bins):
        center = smooth[i]
        if center < PEAK_MIN_WEIGHT:
            continue
        left = smooth[i - 1] if i > 0 else center
        right = smooth[i + 1] if (i + 1) < num_bins else center
        # `>=` on both sides, so a plateau yields a peak at EVERY bin of it.
        # They are merged two steps later, and the merged centre is the
        # weighted mean of the plateau -- which is what you want.
        if center >= left and center >= right:
            peaks.append(DensityPeak(
                pos=hist_start + (i * HIST_BIN_SIZE) + (HIST_BIN_SIZE // 2),
                weight=center))

    peaks.extend(_indel_only_peaks(evidence))

    if not peaks:
        fallback_center = weighted_median_position([(p.pos, p.weight) for p in evidence])
        if fallback_center >= 0:
            peaks.append(DensityPeak(pos=fallback_center, weight=1.0))
    if not peaks:
        return []

    peaks.sort(key=lambda p: p.pos)
    merged_peaks: list[DensityPeak] = []
    for peak in peaks:
        if not merged_peaks or (peak.pos - merged_peaks[-1].pos) > PEAK_MERGE_DISTANCE:
            merged_peaks.append(DensityPeak(peak.pos, peak.weight))
            continue
        back = merged_peaks[-1]
        total_weight = back.weight + peak.weight
        if total_weight > 0.0:
            back.pos = round(((back.weight * back.pos) + (peak.weight * peak.pos))
                             / total_weight)
        # Weight ACCUMULATES on merge here, but is taken as a MAX when windows
        # merge below. Kept as the C++ has it.
        back.weight = total_weight

    windows: list[CandidateWindow] = []
    for peak in merged_peaks:
        center_idx = max(0, min(num_bins - 1,
                                (peak.pos - hist_start) // HIST_BIN_SIZE))
        center_weight = smooth[center_idx]
        expand_threshold = max(0.50, center_weight * WINDOW_DROP_RATIO)

        left = center_idx
        right = center_idx
        while (left > 0 and (center_idx - left) < WINDOW_MAX_EXPAND_BINS
               and smooth[left - 1] >= expand_threshold):
            left -= 1
        while ((right + 1) < num_bins and (right - center_idx) < WINDOW_MAX_EXPAND_BINS
               and smooth[right + 1] >= expand_threshold):
            right += 1

        start = hist_start + (left * HIST_BIN_SIZE)
        end = hist_start + ((right + 1) * HIST_BIN_SIZE)
        if (end - start) < MIN_WINDOW_SPAN:
            start = peak.pos - (MIN_WINDOW_SPAN // 2)
            end = start + MIN_WINDOW_SPAN

        start = max(bin_start - WINDOW_BIN_SLACK_BP, start)
        end = min(bin_end + WINDOW_BIN_SLACK_BP, end)
        if end <= start:
            continue

        windows.append(CandidateWindow(
            start=start, end=end,
            center=max(start, min(end - 1, peak.pos)),
            peak_weight=max(peak.weight, center_weight)))

    if not windows:
        return windows

    windows.sort(key=lambda w: (w.start, w.end))
    merged_windows: list[CandidateWindow] = []
    for window in windows:
        if not merged_windows or window.start > (merged_windows[-1].end + WINDOW_MERGE_GAP):
            merged_windows.append(window)
            continue
        # Not `back`: that name is bound earlier in this same function to a
        # DensityPeak, and reusing it for a CandidateWindow put two record
        # types under one name in one scope.
        back_window = merged_windows[-1]
        total_weight = back_window.peak_weight + window.peak_weight
        if total_weight > 0.0:
            back_window.center = round(
                ((back_window.peak_weight * back_window.center)
                 + (window.peak_weight * window.center)) / total_weight)
        back_window.start = min(back_window.start, window.start)
        back_window.end = max(back_window.end, window.end)
        back_window.peak_weight = max(back_window.peak_weight, window.peak_weight)
    return merged_windows
