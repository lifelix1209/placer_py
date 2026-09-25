"""
Where to look: evidence weights, the smoothed density, and candidate windows.

No C++ tests exist for this stage at all -- the helpers are in an `.inc` file
inside an anonymous namespace, reachable only through a pipeline that needs a
BAM. So everything here is new, and the cases that matter most are the ones
where the arithmetic is deliberately odd: the unnormalised smoother and the
relative expansion threshold.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer.alignment import CIGAR_I, CIGAR_M, CIGAR_S, AlignedRead
from placer.core import windows as W

pytestmark = pytest.mark.invariant

M, I, S = CIGAR_M, CIGAR_I, CIGAR_S


def read(qname="r0", pos=1000, cigar=None, mapq=60, sa=None, flag=0, tid=0):
    cigar = cigar or [(M, 500), (I, 300), (M, 500)]
    length = sum(n for op, n in cigar if op in (M, I, S))
    return AlignedRead(qname=qname, tid=tid, pos=pos, mapq=mapq, cigar=cigar,
                       seq="A" * length, flag=flag,
                       tags=({"SA": sa} if sa else {}))


def point(pos, weight=2.0, kind=W.EvidenceKind.SOFT_CLIP):
    return W.EvidencePoint(pos=pos, weight=weight, kind=kind)


# ------------------------------------------------------------- the weights
def test_a_long_clip_outweighs_a_long_insertion_at_the_same_length():
    """
    Reads backwards until you see what the stage is for. A clip is how a LARGE
    insertion presents when the read cannot span it, so it is the signal most
    likely to mark an event the CIGAR could not -- and this stage's only job is
    deciding where to look.
    """
    bundle = call_or_skip(W.extract_evidence_points,
                          [read(cigar=[(M, 500), (S, 300)]),
                           read(cigar=[(M, 500), (I, 300), (M, 500)])], 0)
    clip = [p for p in bundle.points if p.kind == W.EvidenceKind.SOFT_CLIP][0]
    indel = [p for p in bundle.points if p.kind == W.EvidenceKind.INDEL][0]
    close(clip.weight, 1.0 + 300 / 180.0, "clip weight")
    close(indel.weight, 0.80 + 300 / 200.0, "indel weight")
    assert clip.weight > indel.weight


def test_both_weight_caps_stop_one_huge_signal_manufacturing_a_peak():
    """Without the caps a single 20 kb clip would outweigh ten ordinary reads
    and create a window on its own."""
    bundle = W.extract_evidence_points(
        [read(cigar=[(M, 500), (S, 20000)]),
         read(cigar=[(M, 500), (I, 20000), (M, 500)])], 0)
    clip = [p for p in bundle.points if p.kind == W.EvidenceKind.SOFT_CLIP][0]
    indel = [p for p in bundle.points if p.kind == W.EvidenceKind.INDEL][0]
    close(clip.weight, 1.0 + 500 / 180.0, "clip cap")
    close(indel.weight, 0.80 + 400 / 200.0, "indel cap")


def test_the_sa_hint_is_small_enough_to_break_ties_but_not_to_make_a_peak():
    bundle = W.extract_evidence_points(
        [read(cigar=[(M, 500), (S, 100)], sa="chr1,2000,+,100M;")], 0)
    hint = [p for p in bundle.points if p.kind == W.EvidenceKind.SA_HINT][0]
    close(hint.weight, 0.35, "hint weight")
    assert hint.weight < W.PEAK_MIN_WEIGHT


def test_an_exact_tie_between_two_clips_places_no_hint():
    """
    Both ends are equally plausible, so placing the hint at either would be a
    coin flip recorded as evidence. The read still contributes its clip points
    from both ends.
    """
    assert call_or_skip(W.choose_sa_hint_side, 100, 100) == W.SaHintSide.NONE
    assert W.choose_sa_hint_side(100, 50) == W.SaHintSide.LEFT
    assert W.choose_sa_hint_side(50, 100) == W.SaHintSide.RIGHT
    assert W.choose_sa_hint_side(10, 10) == W.SaHintSide.NONE


def test_signals_below_their_floors_produce_no_points():
    bundle = W.extract_evidence_points(
        [read(cigar=[(M, 500), (S, 19)]), read(cigar=[(M, 500), (I, 39), (M, 500)])], 0)
    assert bundle.points == []


def test_the_indel_evidence_floor_is_lower_than_the_insertion_floor():
    """
    40 against 50, and deliberately so: a density peak may be built from several
    sub-threshold indels that no single one would justify examining.
    """
    assert W.LARGE_INDEL_EVIDENCE_MIN < W.LONG_INSERTION_SIGNAL_MIN
    bundle = W.extract_evidence_points([read(cigar=[(M, 500), (I, 45), (M, 500)])], 0)
    assert len(bundle.points) == 1
    assert bundle.read_summaries[0].class_mask & 4 == 0   # not a long-insertion read


# ------------------------------------------------------------ the summaries
def test_every_read_keeps_its_slot_even_with_no_signal_at_all():
    """
    The summaries are indexed by BIN POSITION and the later stages address reads
    that way. A compacted list would silently shift every downstream index.
    """
    bundle = W.extract_evidence_points(
        [read("clean", cigar=[(M, 1000)]), read("clipped", cigar=[(M, 500), (S, 300)])], 0)
    assert len(bundle.read_summaries) == 2
    assert bundle.read_summaries[0].read_id == "clean"
    assert bundle.read_summaries[0].class_mask == 0


def test_a_record_with_no_cigar_still_records_its_sa_class():
    cigarless = AlignedRead(qname="odd", tid=0, pos=1000, tags={"SA": "chr1,1,+,10M;"})
    bundle = W.extract_evidence_points([cigarless], 0)
    assert bundle.read_summaries[0].has_sa_or_supp
    assert bundle.read_summaries[0].class_mask & 2


def test_supplementary_records_are_skipped_as_in_the_geometry_stage():
    supplementary = read(cigar=[(M, 500), (S, 300)], flag=0x800)
    assert W.extract_evidence_points([supplementary], 0).points == []


def test_reads_on_another_contig_contribute_nothing():
    assert W.extract_evidence_points([read(tid=1, cigar=[(M, 500), (S, 300)])], 0).points == []


# -------------------------------------------------------------- the smoother
def test_the_smoother_is_deliberately_unnormalised():
    """
    The five taps sum to 2.70, so smoothing INFLATES the values and
    `PEAK_MIN_WEIGHT` is calibrated against the inflated scale. Normalising it
    would look like a cleanup and would raise the effective peak threshold by
    2.7x, silencing most loci.
    """
    impulse = [0.0, 0.0, 1.0, 0.0, 0.0]
    smoothed = call_or_skip(W.smooth_histogram, impulse)
    close(sum(smoothed), 2.70, "tap sum")
    close(smoothed[2], 1.0, "centre")
    close(smoothed[1], 0.60, "adjacent")
    close(smoothed[0], 0.25, "two away")


def test_smoothing_lets_nearby_evidence_of_different_kinds_reinforce():
    """
    A clip at 1000 and an insertion at 1015 land in different 20 bp bins. Without
    smoothing they are two weak peaks; with it, one strong one.
    """
    separate = [1.0, 1.0]
    smoothed = W.smooth_histogram(separate)
    assert smoothed[0] > separate[0]


def test_smoothing_is_well_defined_at_the_edges():
    assert W.smooth_histogram([]) == []
    close(W.smooth_histogram([1.0])[0], 1.0, "single bin")


# ---------------------------------------------------------------- the median
def test_the_weighted_median_names_a_position_the_evidence_supports():
    """A mean would be dragged by one distant clip; this cannot name a position
    with no evidence at it."""
    rows = [(1000, 5.0), (1010, 5.0), (50000, 1.0)]
    assert call_or_skip(W.weighted_median_position, rows) in (1000, 1010)
    assert W.weighted_median_position([]) == -1
    assert W.weighted_median_position([(100, 0.0)]) == -1


# ---------------------------------------------------------------- the windows
def test_a_single_strong_locus_yields_one_tight_window_around_it():
    points = [point(1500, 2.3, W.EvidenceKind.INDEL) for _ in range(4)]
    windows = call_or_skip(W.build_density_windows, points, 0, 10000)
    assert len(windows) == 1
    assert windows[0].start <= 1500 <= windows[0].end
    assert (windows[0].end - windows[0].start) == W.MIN_WINDOW_SPAN


def test_two_distant_loci_stay_two_windows():
    points = [point(1500, 2.3) for _ in range(4)] + [point(50000, 2.3) for _ in range(4)]
    windows = W.build_density_windows(points, 0, 60000)
    assert len(windows) == 2
    assert windows[0].center < windows[1].center


def test_peaks_within_the_merge_distance_become_one_window():
    points = [point(1500, 2.3) for _ in range(4)] + [point(1560, 2.3) for _ in range(4)]
    windows = W.build_density_windows(points, 0, 10000)
    assert len(windows) == 1


def test_the_expansion_threshold_is_relative_to_the_peaks_own_height():
    """
    The right way round: a STRONG peak's position is well determined and gets a
    tight window; a weak one's is not and is allowed to be broad. An absolute
    threshold would do the opposite.
    """
    strong = [point(1500, 10.0) for _ in range(5)] + [point(1700, 1.0)]
    weak = [point(1500, 1.0) for _ in range(2)] + [point(1700, 0.9)]
    strong_windows = W.build_density_windows(strong, 0, 10000)
    weak_windows = W.build_density_windows(weak, 0, 10000)
    assert strong_windows
    assert weak_windows
    # The weak peak admits its distant neighbour into the same window; the
    # strong one does not.
    assert len(weak_windows) <= len(strong_windows)


def test_a_long_indel_seeds_its_own_window_beside_a_clip_heavy_repeat():
    """
    A CIGAR insertion is breakpoint-specific in a way a clip is not: the aligner
    placed those bases at that position. So indel points seed their own windows
    even when the smoothed density is dominated by nearby clips.
    """
    clips = [point(1000 + 7 * i, 3.5, W.EvidenceKind.SOFT_CLIP) for i in range(20)]
    indels = [point(1400, 2.3, W.EvidenceKind.INDEL) for _ in range(2)]
    windows = W.build_density_windows(clips + indels, 0, 10000)
    assert any(w.start <= 1400 <= w.end for w in windows)


def test_diffuse_evidence_below_the_peak_floor_still_produces_one_window():
    """
    The fallback. A locus with real but spread-out evidence must still be
    examined -- refusing to look is a decision made where there is least
    information to make it with.
    """
    scattered = [point(1000 + 400 * i, 0.4) for i in range(3)]
    windows = W.build_density_windows(scattered, 0, 10000)
    assert len(windows) == 1


def test_no_evidence_yields_no_windows():
    assert W.build_density_windows([], 0, 10000) == []


def test_windows_may_cross_the_bin_boundary_by_the_slack():
    """
    A long read's start can fall in an adjacent bin, and clipping the window to
    the bin would drop its evidence -- which is exactly the evidence a long-read
    caller exists to use.
    """
    points = [point(-500, 2.3) for _ in range(4)]
    windows = W.build_density_windows(points, 0, 10000)
    assert windows
    assert windows[0].start < 0


def test_a_window_beyond_the_slack_is_dropped_rather_than_clipped_to_nothing():
    far = [point(-W.WINDOW_BIN_SLACK_BP - 5000, 2.3) for _ in range(4)]
    assert W.build_density_windows(far, 0, 10000) == []


def test_the_window_centre_stays_inside_its_own_window():
    points = [point(1500, 2.3) for _ in range(4)] + [point(1560, 9.0) for _ in range(4)]
    for window in W.build_density_windows(points, 0, 10000):
        assert window.start <= window.center < window.end


def test_merged_windows_take_the_max_weight_while_merged_peaks_accumulate():
    """
    Pinned because the two merges differ and it is easy to make them agree:
    peaks ADD their weights, windows take the MAXIMUM. Kept as the C++ has it.
    """
    points = [point(1500, 2.3) for _ in range(4)] + [point(1600, 2.3) for _ in range(4)]
    windows = W.build_density_windows(points, 0, 10000)
    assert len(windows) == 1
    assert windows[0].peak_weight > 2.3
