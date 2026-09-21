"""
Fragment extraction: turning a cluster of reads into sequence to classify.

Ported from `src/component/insert_fragment_module.cpp`, pinned by
`tests/test_22_fragments.py`.

WHAT THIS STAGE DECIDES, and it is more consequential than it looks. Everything
downstream -- the TE alignment, the poly(A) call, the TSD, the structure decode
-- reads the fragment SEQUENCE. If the wrong bases are cut out of the read, no
amount of care in the model recovers it, and the failure is silent: a fragment
that is 200 bp of flanking genome and 800 bp of element still aligns to the
element, still reports high identity, and still gets called.

THREE SOURCES, in increasing order of how much they assume:

  * `kCigarInsertion` -- the aligner placed the extra bases itself. The
    coordinates are exact and the only judgement is which insertions to keep.
  * `kClipRefLeft` / `kClipRefRight` -- a soft clip. The bases are certain, the
    boundary is certain, but the clip is only ONE SIDE of the insertion, so the
    fragment is a lower bound on it.
  * `kSplitSa` -- reconstructed from two alignments of the same read. Nothing
    marks the junction directly; it is inferred from where the two query
    intervals stop agreeing, and that inference is the bulk of this module.

THE SPLIT-SA PATH IS WHERE THE CARE IS, because it is the only one that can cut
the wrong bases. The C++ builds two fragments per split read for two different
purposes and they are easy to confuse:

  * the ROBUST fragment -- pick the best flank, find its mate, locate the query
    junction, and take the OPPOSITE side of the junction, explicitly excluding
    the flank's own core. This is the one that is meant to be the insertion.
  * the DIAGNOSTIC fragments -- one per SA record that implies a query gap
    against the primary, taken as the SA segment's own query interval. These
    are capped at `max_sa_per_read` and exist to keep a record of what the
    aligner said, not to be trusted as inserts.

Both are emitted into the same list and both reach the classifier. That is worth
knowing when reading a hit table: a `sa_idx=` in the fragment id marks the
second kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from placer_py.alignment import (
    CIGAR_D,
    CIGAR_I,
    CIGAR_N,
    CIGAR_S,
    AlignedRead,
    CigarStringOp,
    QueryInterval,
    SAEntryWithQuality,
    compute_ref_end,
    consumes_query,
    consumes_query_char,
    consumes_ref,
    consumes_ref_char,
    find_first_non_hard_clip,
    find_last_non_hard_clip,
    is_match_like,
    parse_cigar_ops,
    parse_sa_tag_z_with_quality,
)
from placer_py.clustering import (
    CANDIDATE_LONG_INSERTION,
    CANDIDATE_SOFT_CLIP,
    CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
    INSERTION_CANDIDATE_REQUIRED_MAPQ,
    ComponentCall,
)
from placer_py.config import PipelineConfig

#: An insertion this far from the component's own breakpoints belongs to some
#: other event that happens to sit on the same read.
COMPONENT_LOCAL_INSERTION_RETENTION_BP = 1000
#: The shortest split-implied insertion worth a fragment.
SPLIT_SA_INSERTION_MIN_LEN = 50
#: Two per read by length, then up to two more that are component-local. The
#: split is deliberate -- see `select_insertions_for_fragment_extraction`.
PRIMARY_INSERTION_FRAGMENTS_PER_READ = 2
MAX_INSERTION_FRAGMENTS_PER_READ = 4
#: An indel at least this long counts as "large near the breakpoint", which
#: costs a flank candidate five NM-equivalents.
LARGE_INDEL_BP = 20
#: FASTA line width for the optional fragment dump.
FASTA_LINE_WIDTH = 80


class InsertionFragmentSource(IntEnum):
    UNKNOWN = 0
    CLIP_REF_LEFT = 1
    CLIP_REF_RIGHT = 2
    CIGAR_INSERTION = 3
    SPLIT_SA = 4


class ReferenceSide(IntEnum):
    UNKNOWN = 0
    REF_LEFT = 1
    REF_RIGHT = 2


@dataclass
class InsertionFragment:
    fragment_id: str = ""
    chrom: str = ""
    anchor_pos: int = -1
    read_id: str = ""
    read_index: int = 0
    class_mask: int = 0
    is_reverse: bool = False
    source: InsertionFragmentSource = InsertionFragmentSource.UNKNOWN
    start: int = -1   # read coordinate, 0-based
    length: int = 0
    read_len: int = 0
    anchor_len: int = 0
    ref_side: ReferenceSide = ReferenceSide.UNKNOWN
    ref_junc_pos: int = -1
    nm: int = -1
    split_sa_reliable: bool = False
    sequence: str = ""


@dataclass
class ClipInfo:
    """The two terminal clips and the anchors immediately inside them.

    `right_anchor_after_leading` stops at the first NON-match operation, so it
    measures the uninterrupted anchor beside the clip rather than the total
    matched bases in the read. A 200 bp clip beside twenty 10 bp match blocks is
    a chimera, and the two fields have to be able to say so.
    """

    leading: int = 0
    trailing: int = 0
    right_anchor_after_leading: int = 0
    left_anchor_before_trailing: int = 0
    ref_end: int = 0


@dataclass
class InsOp:
    """One CIGAR insertion, with the anchors on either side of it."""

    start: int = 0       # read coordinate
    len: int = 0
    ref_pos: int = -1
    left_anchor: int = 0
    right_anchor: int = 0


@dataclass
class NormalizedAln:
    """A candidate alignment segment, with everything the flank choice needs."""

    chrom: str = ""
    ref_start: int = -1
    ref_end: int = -1
    qstart: int = -1
    qend: int = -1
    is_reverse: bool = False
    mapq: int = 0
    nm: int = -1
    is_primary: bool = False
    reliable: bool = True
    anchor_len: int = 0
    has_large_indel_near_bp: bool = False


# ---------------------------------------------------------------- geometry
def interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    uni = max(a1, b1) - min(a0, b0)
    return (inter / uni) if uni > 0 else 0.0


def overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def nonoverlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    """How much of interval A is NOT covered by B. Asymmetric, on purpose: the
    mate is chosen by how much NEW query it brings to the flank."""
    a_len = max(0, a1 - a0)
    return max(0, a_len - overlap_len(a0, a1, b0, b1))


def distance_to_breakpoint(ref_start: int, ref_end: int, bp: int) -> int:
    """0 when the breakpoint falls inside the span, otherwise the gap to it."""
    if ref_end < ref_start:
        ref_start, ref_end = ref_end, ref_start
    if bp < ref_start:
        return ref_start - bp
    if bp > ref_end:
        return bp - ref_end
    return 0


def overlap_with_window(seg_start: int, seg_end: int, win_start: int,
                        win_end: int) -> int:
    return max(0, min(seg_end, win_end) - max(seg_start, win_start))


# -------------------------------------------------------------- clip analysis
def analyze_clip_info(read: AlignedRead) -> ClipInfo:
    info = ClipInfo()
    if not read.cigar:
        info.ref_end = read.pos
        return info

    first = find_first_non_hard_clip(read.cigar)
    last = find_last_non_hard_clip(read.cigar)
    if first < 0 or last < 0 or first > last:
        info.ref_end = compute_ref_end(read)
        return info

    if read.cigar[first][0] == CIGAR_S:
        info.leading = read.cigar[first][1]
        flank = 0
        for i in range(first + 1, last + 1):
            if not is_match_like(read.cigar[i][0]):
                break
            flank += read.cigar[i][1]
        info.right_anchor_after_leading = flank

    if read.cigar[last][0] == CIGAR_S:
        info.trailing = read.cigar[last][1]
        flank = 0
        for i in range(last - 1, first - 1, -1):
            if not is_match_like(read.cigar[i][0]):
                break
            flank += read.cigar[i][1]
        info.left_anchor_before_trailing = flank

    info.ref_end = compute_ref_end(read)
    return info


def contiguous_match_run_after(cigar: list[tuple[int, int]], idx: int) -> int:
    run = 0
    for op, length in cigar[idx + 1:]:
        if not is_match_like(op):
            break
        run += length
    return run


def find_long_insertions(read: AlignedRead, min_long_ins: int) -> list[InsOp]:
    """Every CIGAR insertion at or above the floor, with its flanking anchors.

    `prev_match_run` resets on any non-match operation, so `left_anchor` is the
    uninterrupted run immediately before the insertion -- an insertion reached
    through a string of short match blocks is not anchored, and the field says
    so. Both anchors are then used as `min(left, right)`: an insertion is only
    as well placed as its WEAKER side.
    """
    ops: list[InsOp] = []
    if not read.cigar:
        return ops

    qpos = 0
    rpos = read.pos
    prev_match_run = 0
    for i, (op, length) in enumerate(read.cigar):
        if is_match_like(op):
            prev_match_run += length
            qpos += length
            rpos += length
            continue
        if op == CIGAR_I:
            if length >= min_long_ins:
                ops.append(InsOp(start=qpos, len=length, ref_pos=rpos,
                                 left_anchor=prev_match_run,
                                 right_anchor=contiguous_match_run_after(read.cigar, i)))
            qpos += length
            prev_match_run = 0
            continue
        if consumes_query(op):
            qpos += length
        if consumes_ref(op):
            rpos += length
        prev_match_run = 0
    return ops


def same_insertion_op(a: InsOp, b: InsOp) -> bool:
    return a.start == b.start and a.len == b.len and a.ref_pos == b.ref_pos


def insertion_is_component_local(op: InsOp, component: ComponentCall) -> bool:
    """Is this insertion near the event we are extracting for?

    Tested against the component anchor AND against every breakpoint candidate,
    because the anchor is a median and an event with a bimodal breakpoint
    distribution can have its anchor sitting between the two real positions.
    """
    if op.ref_pos < 0:
        return False
    if component.anchor_pos >= 0 and abs(
            op.ref_pos - component.anchor_pos) <= COMPONENT_LOCAL_INSERTION_RETENTION_BP:
        return True
    for candidate in component.breakpoint_candidates:
        if candidate.pos >= 0 and abs(
                op.ref_pos - candidate.pos) <= COMPONENT_LOCAL_INSERTION_RETENTION_BP:
            return True
    return False


def select_insertions_for_fragment_extraction(ins_ops: list[InsOp],
                                              component: ComponentCall) -> list[InsOp]:
    """At most four insertions per read: two by length, then two by locality.

    THE TWO PASSES ARE NOT THE SAME RULE and the order matters. The first takes
    the two LONGEST insertions on the read regardless of where they are, which
    is a hedge against the component anchor being wrong -- if the locality test
    were the only filter, an anchor off by 2 kb would silently extract nothing.
    The second then fills the remaining slots with insertions that ARE local,
    which is what a read spanning several events needs.

    A read with five insertions loses one, and the one it loses is the shortest
    non-local one. That is the intended trade: the cap exists because a
    pathological read in a tandem array can otherwise emit dozens of fragments,
    each of which costs an alignment.
    """
    ordered = sorted(ins_ops, key=lambda op: (-op.len, op.ref_pos))
    selected: list[InsOp] = []

    def append_unique(op: InsOp) -> None:
        if len(selected) >= MAX_INSERTION_FRAGMENTS_PER_READ:
            return
        if not any(same_insertion_op(kept, op) for kept in selected):
            selected.append(op)

    for op in ordered:
        if len(selected) >= PRIMARY_INSERTION_FRAGMENTS_PER_READ:
            break
        append_unique(op)
    for op in ordered:
        if len(selected) >= MAX_INSERTION_FRAGMENTS_PER_READ:
            break
        if insertion_is_component_local(op, component):
            append_unique(op)
    return selected


# ------------------------------------------------------------ anchor strength
def anchor_len_from_bam(read: AlignedRead, bp: int,
                        w_anchor: int) -> tuple[int, bool]:
    """Aligned bases within `w_anchor` of the breakpoint, and whether a large
    indel sits there.

    Deletions COUNT toward the anchor -- they are reference the alignment still
    explains -- but a deletion of 20 bp or more near the breakpoint also sets
    the flag, and `compare_flank_candidate` charges five NM-equivalents for it.
    That separation is deliberate: a large indel beside the junction does not
    mean the read is unanchored, it means the junction may be misplaced by about
    the indel's length, which is a different kind of doubt.
    """
    if not read.cigar:
        return 0, False
    win_start = bp - w_anchor
    win_end = bp + w_anchor + 1

    rpos = read.pos
    anchored = 0
    has_large_indel = False
    for op, length in read.cigar:
        if is_match_like(op):
            anchored += overlap_with_window(rpos, rpos + length, win_start, win_end)
            rpos += length
            continue
        if op in (CIGAR_D, CIGAR_N):
            overlap = overlap_with_window(rpos, rpos + length, win_start, win_end)
            anchored += overlap
            if length >= LARGE_INDEL_BP and overlap > 0:
                has_large_indel = True
            rpos += length
            continue
        if op == CIGAR_I:
            if win_start <= rpos < win_end:
                anchored += length
                if length >= LARGE_INDEL_BP:
                    has_large_indel = True
    return anchored, has_large_indel


def anchor_len_from_ops(ops: list[CigarStringOp], ref_start: int, bp: int,
                        w_anchor: int) -> tuple[int, bool]:
    """The same measurement for an SA segment, whose CIGAR is a string."""
    if not ops:
        return 0, False
    win_start = bp - w_anchor
    win_end = bp + w_anchor + 1

    rpos = ref_start
    anchored = 0
    has_large_indel = False
    for op in ops:
        if op.op in ("M", "=", "X"):
            anchored += overlap_with_window(rpos, rpos + op.length, win_start, win_end)
            rpos += op.length
            continue
        if op.op in ("D", "N"):
            overlap = overlap_with_window(rpos, rpos + op.length, win_start, win_end)
            anchored += overlap
            if op.length >= LARGE_INDEL_BP and overlap > 0:
                has_large_indel = True
            rpos += op.length
            continue
        if op.op == "I":
            if win_start <= rpos < win_end:
                anchored += op.length
                if op.length >= LARGE_INDEL_BP:
                    has_large_indel = True
    return anchored, has_large_indel


def cigar_to_query_interval_with_indel(ops: list[CigarStringOp], read_len: int
                                       ) -> tuple[QueryInterval, int, bool] | None:
    """`cigar_to_query_interval`, plus whether the segment carries a large indel.

    The C++ keeps a second copy of the function for this one extra output. Here
    it is one function and the caller takes what it needs.
    """
    out = QueryInterval()
    ref_aligned_len = 0
    has_large_indel = False
    if not ops or read_len <= 0:
        return None

    i = 0
    while i < len(ops) and ops[i].op in ("H", "S"):
        if ops[i].op == "S":
            out.leading_s += ops[i].length
        i += 1
    j = len(ops) - 1
    while j >= 0 and ops[j].op in ("H", "S"):
        if ops[j].op == "S":
            out.trailing_s += ops[j].length
        j -= 1

    q_aligned = 0
    for op in ops:
        if consumes_query_char(op.op) and op.op != "S":
            q_aligned += op.length
        if consumes_ref_char(op.op):
            ref_aligned_len += op.length
        if op.op in ("I", "D") and op.length >= LARGE_INDEL_BP:
            has_large_indel = True
    if q_aligned <= 0:
        return None

    if out.leading_s > 0:
        qstart = out.leading_s
    elif out.trailing_s > 0:
        qstart = read_len - out.trailing_s - q_aligned
    elif q_aligned == read_len:
        qstart = 0
    else:
        return None
    if qstart < 0:
        return None
    qend = qstart + q_aligned
    if qend > read_len or qend <= qstart:
        return None
    out.qstart = qstart
    out.qend = qend
    return out, ref_aligned_len, has_large_indel


def bam_to_query_interval_with_clip(read: AlignedRead,
                                    clip: ClipInfo) -> QueryInterval | None:
    """Query interval of the primary, using an already-computed `ClipInfo`."""
    read_len = read.seq_len
    if read_len <= 0 or not read.cigar:
        return None
    q_aligned = 0
    for op, length in read.cigar:
        if is_match_like(op) or op == CIGAR_I:
            q_aligned += length
    if q_aligned <= 0:
        return None

    out = QueryInterval(leading_s=clip.leading, trailing_s=clip.trailing)
    out.qstart = clip.leading
    out.qend = read_len - clip.trailing
    if out.qend <= out.qstart or out.qend > read_len:
        out.qstart = clip.leading
        out.qend = out.qstart + q_aligned
    if out.qend <= out.qstart or out.qend > read_len:
        return None
    return out


# ------------------------------------------------------------ split geometry
def split_sa_insertion_len(a: NormalizedAln, b: NormalizedAln) -> int:
    """`query_gap - ref_gap`, or -1 when the pair is not an insertion at all.

    Different contig or different strand returns -1 rather than a length: those
    are a translocation and an inversion, and giving them a "length" would let
    an unrelated rearrangement enter the insertion path.
    """
    if a.chrom != b.chrom or a.is_reverse != b.is_reverse:
        return -1
    left, right = (a, b) if b.qstart >= a.qstart else (b, a)
    query_gap = right.qstart - left.qend
    ref_gap = max(0, right.ref_start - left.ref_end)
    return query_gap - ref_gap


def has_split_sa_query_gap_insertion(a: NormalizedAln, b: NormalizedAln) -> bool:
    return split_sa_insertion_len(a, b) >= SPLIT_SA_INSERTION_MIN_LEN


def compare_flank_candidate(a: NormalizedAln, b: NormalizedAln, bp: int,
                            alpha: float) -> int:
    """Order two flank candidates. Negative means `a` is better.

    The primary key is `anchor_len - alpha * (nm + 5 * large_indel_near_bp)`,
    i.e. anchored bases discounted by how much noise the alignment carries. The
    five-NM charge for a large indel near the breakpoint is the one hand-set
    constant in this module, and it encodes "an indel beside the junction makes
    the junction's position doubtful", which no count of mismatches expresses.

    Everything after it is deterministic tie-breaking. That matters more than it
    reads: ties are common (two supplementary alignments of the same read often
    have identical anchor and NM), and an unstable order here would make the
    extracted fragment depend on iteration order.
    """
    penalty_a = float(max(0, a.nm)) + 5.0 * (1.0 if a.has_large_indel_near_bp else 0.0)
    penalty_b = float(max(0, b.nm)) + 5.0 * (1.0 if b.has_large_indel_near_bp else 0.0)
    score_a = float(a.anchor_len) - alpha * penalty_a
    score_b = float(b.anchor_len) - alpha * penalty_b
    if score_a != score_b:
        return -1 if score_a > score_b else 1
    if a.anchor_len != b.anchor_len:
        return -1 if a.anchor_len > b.anchor_len else 1
    if a.mapq != b.mapq:
        return -1 if a.mapq > b.mapq else 1
    if a.nm != b.nm:
        return -1 if a.nm < b.nm else 1
    if a.is_primary != b.is_primary:
        return -1 if a.is_primary else 1
    if a.ref_start != b.ref_start:
        return -1 if a.ref_start < b.ref_start else 1
    dist_a = distance_to_breakpoint(a.ref_start, a.ref_end, bp)
    dist_b = distance_to_breakpoint(b.ref_start, b.ref_end, bp)
    if dist_a != dist_b:
        return -1 if dist_a < dist_b else 1
    return 0


# --------------------------------------------------------------- identifiers
def sanitize_fragment_token(text: str) -> str:
    """Make a token safe for the `|`-delimited fragment id."""
    out = []
    for c in text:
        if c.isspace():
            out.append("_")
        elif c == "|":
            out.append("/")
        else:
            out.append(c)
    return "".join(out)


def fasta_safe_header(text: str) -> str:
    return "".join("_" if c.isspace() else c for c in text)


def source_tag(source: InsertionFragmentSource) -> str:
    return {
        InsertionFragmentSource.CLIP_REF_LEFT: "clipRefL",
        InsertionFragmentSource.CLIP_REF_RIGHT: "clipRefR",
        InsertionFragmentSource.CIGAR_INSERTION: "ins",
        InsertionFragmentSource.SPLIT_SA: "splitSA",
    }.get(source, "unknown")


def wrapped_sequence(seq: str) -> str:
    return "".join(seq[i:i + FASTA_LINE_WIDTH] + "\n"
                   for i in range(0, len(seq), FASTA_LINE_WIDTH))


def _fragment_id_prefix(component: ComponentCall, read_token: str,
                        read_index: int, is_reverse: bool,
                        source: InsertionFragmentSource) -> str:
    return (f"{component.chrom}:{component.anchor_pos}"
            f"|read={read_token}"
            f"|idx={read_index}"
            f"|strand={'-' if is_reverse else '+'}"
            f"|src={source_tag(source)}")


# ---------------------------------------------------------------------------
# The split-SA extractor.
# ---------------------------------------------------------------------------
#: Search radius for a flank around the component anchor, and the anchor window.
SPLIT_SA_MIN_FLANK_RADIUS_BP = 200
SPLIT_SA_ANCHOR_WINDOW_BP = 150
#: Microhomology tolerated at the junction before the overlap is called
#: uninterpretable.
SPLIT_SA_MICROHOMOLOGY_MAX = 20
#: Bases trimmed off each end of the flank's core before excluding it, so a few
#: bases of junction ambiguity do not eat into the fragment.
SPLIT_SA_CORE_TRIM = 12
#: Slack when deciding which reference side the junction sits on.
SPLIT_SA_SIDE_EPS_BP = 5
#: Weight of the NM/indel penalty against anchored bases.
SPLIT_SA_FLANK_ALPHA = 1.0
#: IoU and endpoint agreement required before an SA record is believed to
#: describe the same alignment as an observed supplementary record.
SPLIT_SA_SUPP_MATCH_IOU = 0.9
SPLIT_SA_SUPP_MATCH_ENDPOINT_BP = 5


def _collect_supplementary_alignments(component: ComponentCall,
                                      bin_records: list[AlignedRead],
                                      w_anchor: int) -> dict[str, list[NormalizedAln]]:
    """Index the supplementary records actually present in this window.

    An SA tag is the aligner's CLAIM about a segment; a supplementary record is
    that segment, observed. When both exist the claim is cross-checked against
    the observation, and an SA record that does not match any observed
    supplementary is marked unreliable -- see `_sa_reliability`.
    """
    by_read: dict[str, list[NormalizedAln]] = {}
    for idx in component.read_indices:
        if idx >= len(bin_records) or bin_records[idx] is None:
            continue
        read = bin_records[idx]
        if not read.is_supplementary or read.is_secondary:
            continue
        clip = analyze_clip_info(read)
        interval = bam_to_query_interval_with_clip(read, clip)
        if interval is None:
            continue
        anchor_len, has_large_indel = anchor_len_from_bam(read, component.anchor_pos,
                                                          w_anchor)
        nm = read.get_int_tag("NM")
        by_read.setdefault(read.qname, []).append(NormalizedAln(
            chrom=component.chrom,
            ref_start=read.pos,
            ref_end=clip.ref_end,
            qstart=interval.qstart,
            qend=interval.qend,
            is_reverse=read.is_reverse,
            mapq=read.mapq,
            nm=nm if nm is not None else -1,
            is_primary=False,
            reliable=True,
            anchor_len=anchor_len,
            has_large_indel_near_bp=has_large_indel,
        ))
    return by_read


def _sa_reliability(sa: NormalizedAln,
                    observed: list[NormalizedAln] | None) -> bool:
    """Is this SA record corroborated by an observed supplementary alignment?

    When no supplementary record for the read is in the window at all, the SA
    tag is taken at face value -- the segment may simply align outside the bin,
    and refusing every such record would discard exactly the long-range split
    evidence the tag exists to provide.

    When supplementaries ARE present, the claim must match one of them closely:
    90% query IoU and both endpoints within 5 bp. A near-miss is not a
    near-truth here; it means the tag and the record describe different query
    spans, and the junction inferred from the tag would be in the wrong place.
    """
    if not observed:
        return True
    best_iou = 0.0
    best_start_diff = None
    best_end_diff = None
    for supp in observed:
        if supp.is_reverse != sa.is_reverse:
            continue
        iou = interval_iou(supp.qstart, supp.qend, sa.qstart, sa.qend)
        if iou > best_iou:
            best_iou = iou
            best_start_diff = abs(supp.qstart - sa.qstart)
            best_end_diff = abs(supp.qend - sa.qend)
    return (best_iou >= SPLIT_SA_SUPP_MATCH_IOU
            and best_start_diff is not None
            and best_start_diff <= SPLIT_SA_SUPP_MATCH_ENDPOINT_BP
            and best_end_diff <= SPLIT_SA_SUPP_MATCH_ENDPOINT_BP)


def _query_junction(flank: NormalizedAln, mate: NormalizedAln,
                    mh_max: int) -> int | None:
    """Where in the READ the two segments hand over, or None if they cannot.

    Three cases, and the third is the reason this is not just `flank.qend`:

      * the segments are separated or abut within the microhomology tolerance,
        so the junction is the flank's far edge;
      * they abut the other way round, so it is the flank's near edge;
      * they OVERLAP -- which is normal, because microhomology at a TPRT
        junction is aligned by both segments -- and the junction is taken as the
        midpoint of the overlap. Beyond `mh_max` the overlap is no longer
        microhomology but a genuine disagreement about where the read goes, and
        the read is refused rather than cut at a guessed point.
    """
    if flank.qend <= mate.qstart + mh_max:
        return flank.qend
    if mate.qend <= flank.qstart + mh_max:
        return flank.qstart
    ov_start = max(flank.qstart, mate.qstart)
    ov_end = min(flank.qend, mate.qend)
    if max(0, ov_end - ov_start) <= mh_max:
        return (ov_start + ov_end) // 2
    return None


def _opposite_side_window(q_junc: int, flank: NormalizedAln, read_len: int,
                          l_left: int, l_right: int, trim_q: int) -> tuple[int, int]:
    """The read window on the far side of the junction from the flank's core.

    This is the step that makes the fragment an INSERT rather than a mixture.
    A naive `[q_junc, q_junc + L]` would happily include the flank's own aligned
    bases whenever the junction sits inside the flank, and those bases are
    reference genome: the fragment would then align to the element AND to the
    locus, and the TE call would be made partly on flanking sequence.

    The flank's core is its query span minus `trim_q` at each end, so a few
    bases of junction ambiguity are not treated as reference. When the window
    and the core overlap, the window is cut back to whichever side of the core
    leaves more sequence -- keeping the longer run rather than splitting the
    fragment in two.
    """
    opp_start = max(0, q_junc - l_left)
    opp_end = min(read_len, q_junc + l_right)

    core_start = flank.qstart + trim_q
    core_end = flank.qend - trim_q
    if core_end <= core_start:
        core_start, core_end = flank.qstart, flank.qend

    if opp_start < core_end and opp_end > core_start:
        if q_junc <= core_start:
            opp_end = min(opp_end, core_start)
        elif q_junc >= core_end:
            opp_start = max(opp_start, core_end)
        else:
            left_len = max(0, core_start - opp_start)
            right_len = max(0, opp_end - core_end)
            if left_len >= right_len:
                opp_end = core_start
            else:
                opp_start = core_end
    return opp_start, opp_end


def _reference_side(ref_junc_pos: int, anchor_pos: int,
                    eps_bp: int) -> ReferenceSide:
    """Which side of the anchor the junction sits on, or UNKNOWN if both.

    Within `eps_bp` of the anchor both tests pass and the side is UNKNOWN --
    deliberately, because a junction that close to the anchor genuinely does not
    distinguish the two sides, and asserting one would make a coin flip look
    like evidence.
    """
    left = ref_junc_pos <= anchor_pos + eps_bp
    right = ref_junc_pos >= anchor_pos - eps_bp
    if left and right:
        return ReferenceSide.UNKNOWN
    if left:
        return ReferenceSide.REF_LEFT
    if right:
        return ReferenceSide.REF_RIGHT
    return ReferenceSide.UNKNOWN


def extract_split_sa_fragments(component: ComponentCall,
                               bin_records: list[AlignedRead],
                               config: PipelineConfig) -> list[InsertionFragment]:
    """`SplitSAFragmentModule::extract`."""
    out: list[InsertionFragment] = []
    if config.max_sa_per_read <= 0:
        return out

    min_len = config.min_sa_aln_len_for_seq_extract
    max_sa = config.max_sa_per_read
    w_ref = max(SPLIT_SA_MIN_FLANK_RADIUS_BP, config.bin_size // 2)
    w_anchor = SPLIT_SA_ANCHOR_WINDOW_BP
    mh_max = SPLIT_SA_MICROHOMOLOGY_MAX
    l_left = max(min_len, 100)
    l_right = max(min_len, 100)
    trim_q = SPLIT_SA_CORE_TRIM
    eps_bp = SPLIT_SA_SIDE_EPS_BP
    alpha = SPLIT_SA_FLANK_ALPHA

    supplementary_by_read = _collect_supplementary_alignments(component, bin_records,
                                                              w_anchor)

    for idx in component.read_indices:
        if idx >= len(bin_records) or bin_records[idx] is None:
            continue
        read = bin_records[idx]
        if read.is_supplementary or read.is_secondary or not read.has_sa_tag():
            continue
        sa_z = read.get_string_tag("SA")
        if not sa_z:
            continue
        entries = parse_sa_tag_z_with_quality(sa_z)
        if not entries:
            continue

        read_id = read.qname
        read_token = sanitize_fragment_token(read_id)
        read_len = read.seq_len
        is_reverse = read.is_reverse

        clip = analyze_clip_info(read)
        primary_q = bam_to_query_interval_with_clip(read, clip)
        if primary_q is None:
            continue
        primary_anchor, primary_large_indel = anchor_len_from_bam(
            read, component.anchor_pos, w_anchor)
        primary_nm = read.get_int_tag("NM")

        primary = NormalizedAln(
            chrom=component.chrom, ref_start=read.pos, ref_end=clip.ref_end,
            qstart=primary_q.qstart, qend=primary_q.qend, is_reverse=is_reverse,
            mapq=read.mapq, nm=primary_nm if primary_nm is not None else -1,
            is_primary=True, reliable=True, anchor_len=primary_anchor,
            has_large_indel_near_bp=primary_large_indel)
        core_records: list[NormalizedAln] = [primary]
        sa_all: list[tuple[SAEntryWithQuality, NormalizedAln]] = []
        observed = supplementary_by_read.get(read_id)

        for entry in entries:
            ops = parse_cigar_ops(entry.cigar)
            if ops is None:
                continue
            placed = cigar_to_query_interval_with_indel(ops, read_len)
            if placed is None:
                continue
            interval, ref_aligned, has_large_indel = placed
            if interval.qend - interval.qstart < min_len:
                continue

            ref_start = max(0, entry.pos - 1)
            anchor_len, ops_large_indel = anchor_len_from_ops(
                ops, ref_start, component.anchor_pos, w_anchor)
            sa = NormalizedAln(
                chrom=entry.rname, ref_start=ref_start,
                ref_end=ref_start + ref_aligned,
                qstart=interval.qstart, qend=interval.qend,
                is_reverse=(entry.strand == "-"), mapq=entry.mapq, nm=entry.nm,
                is_primary=False, anchor_len=anchor_len,
                has_large_indel_near_bp=ops_large_indel)
            sa.reliable = _sa_reliability(sa, observed)
            if sa.reliable:
                core_records.append(sa)
            sa_all.append((entry, sa))

        # ---- the robust fragment: best flank, its mate, the far side ----
        flank_idx = -1
        for i, row in enumerate(core_records):
            if row.chrom != component.chrom:
                continue
            if distance_to_breakpoint(row.ref_start, row.ref_end,
                                      component.anchor_pos) > w_ref:
                continue
            if flank_idx < 0 or compare_flank_candidate(
                    row, core_records[flank_idx], component.anchor_pos, alpha) < 0:
                flank_idx = i

        emitted_robust = False
        skip_read = False
        if flank_idx >= 0:
            flank = core_records[flank_idx]
            mate_idx = -1
            best_nonoverlap = -1
            for i, mate in enumerate(core_records):
                if i == flank_idx:
                    continue
                nonov = nonoverlap_len(mate.qstart, mate.qend, flank.qstart, flank.qend)
                if nonov > best_nonoverlap:
                    best_nonoverlap = nonov
                    mate_idx = i

            if mate_idx >= 0 and best_nonoverlap >= min_len:
                mate = core_records[mate_idx]
                if not has_split_sa_query_gap_insertion(flank, mate):
                    # The C++ `continue`s the whole read here, so the
                    # diagnostic fragments below are skipped too. Preserved: a
                    # split whose best pair implies no insertion should not
                    # reach the classifier by the back door.
                    skip_read = True
                else:
                    q_junc = _query_junction(flank, mate, mh_max)
                    if q_junc is not None and q_junc >= 0:
                        opp_start, opp_end = _opposite_side_window(
                            q_junc, flank, read_len, l_left, l_right, trim_q)
                        opp_len = opp_end - opp_start
                        if opp_len >= min_len:
                            flank_on_left = abs(q_junc - flank.qend) <= abs(
                                q_junc - flank.qstart)
                            if flank_on_left:
                                ref_junc_pos = (flank.ref_start if flank.is_reverse
                                                else flank.ref_end)
                            else:
                                ref_junc_pos = (flank.ref_end if flank.is_reverse
                                                else flank.ref_start)
                            frag = InsertionFragment(
                                chrom=component.chrom,
                                anchor_pos=component.anchor_pos,
                                read_id=read_id, read_index=idx,
                                class_mask=CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
                                is_reverse=is_reverse,
                                source=InsertionFragmentSource.SPLIT_SA,
                                start=opp_start, length=opp_len, read_len=read_len,
                                anchor_len=flank.anchor_len,
                                ref_side=_reference_side(ref_junc_pos,
                                                         component.anchor_pos, eps_bp),
                                ref_junc_pos=ref_junc_pos, nm=flank.nm,
                                split_sa_reliable=flank.reliable and mate.reliable,
                                sequence=read.decode_subsequence(opp_start, opp_len))
                            frag.fragment_id = (
                                _fragment_id_prefix(component, read_token, idx,
                                                    is_reverse, frag.source)
                                + f"|qj={q_junc}|rj={ref_junc_pos}|len={frag.length}")
                            if frag.sequence:
                                out.append(frag)
                                emitted_robust = True
        if skip_read:
            continue

        # ---- the diagnostic fragments: one per SA record with a query gap ----
        emitted_diag = 1 if emitted_robust else 0
        sa_index = 0
        for entry, sa in sa_all:
            if emitted_diag >= max_sa:
                break
            sa_index += 1
            if sa.qstart < 0 or sa.qend <= sa.qstart or sa.qend > read_len:
                continue
            if not has_split_sa_query_gap_insertion(primary, sa):
                continue
            length = sa.qend - sa.qstart
            frag = InsertionFragment(
                chrom=component.chrom, anchor_pos=component.anchor_pos,
                read_id=read_id, read_index=idx,
                class_mask=CANDIDATE_SPLIT_SA_SUPPLEMENTARY, is_reverse=is_reverse,
                source=InsertionFragmentSource.SPLIT_SA,
                start=sa.qstart, length=length, read_len=read_len,
                anchor_len=sa.anchor_len, ref_side=ReferenceSide.UNKNOWN,
                ref_junc_pos=-1, nm=sa.nm, split_sa_reliable=sa.reliable,
                sequence=read.decode_subsequence(sa.qstart, length))
            frag.fragment_id = (
                _fragment_id_prefix(component, read_token, idx, is_reverse, frag.source)
                + f"|sa_idx={sa_index}"
                + f"|sa={sanitize_fragment_token(entry.rname)}:{entry.pos}{entry.strand}"
                + f"|qlen={frag.length}")
            if frag.length < min_len or not frag.sequence:
                continue
            out.append(frag)
            emitted_diag += 1

    return out


# ---------------------------------------------------------------------------
# The CIGAR/clip extractor, which also drives the split-SA one.
# ---------------------------------------------------------------------------
def relaxed_min_insertion_length(component: ComponentCall,
                                 config: PipelineConfig) -> int:
    """Lower the insertion floor when the event is already well supported.

    The floor exists to keep ordinary indel noise out. But at a locus where
    split and CIGAR-insertion reads already OUTNUMBER the clip reads, the
    evidence that something is there does not come from the length of any one
    insertion -- so the floor can drop to `short_ins_min_len` and let a short
    Alu-scale insert through.

    Note `split_indel_reads >= softclip_reads`, not `>`: a locus with equal
    split and clip support still relaxes. And the floor never goes below 10, so
    the relaxation cannot turn every 1 bp indel into a fragment.
    """
    min_ins = config.min_long_ins_for_seq_extract
    if not config.short_ins_enable:
        return min_ins

    split_indel_support = set(component.split_sa_read_indices) | set(
        component.insertion_read_indices)
    split_indel_reads = len(split_indel_support)
    softclip_reads = len(component.soft_clip_read_indices)
    split_indel_dominant = split_indel_reads > 0 and split_indel_reads >= softclip_reads

    if split_indel_dominant and split_indel_reads >= max(1, config.short_ins_min_reads):
        relaxed_short_min = max(10, config.short_ins_min_len)
        min_ins = min(min_ins, relaxed_short_min)
    return min_ins


def extract_fragments(component: ComponentCall, bin_records: list[AlignedRead],
                      config: PipelineConfig) -> list[InsertionFragment]:
    """`CigarInsertionFragmentModule::extract`: every fragment for one component.

    Order is fixed and meaningful: leading clip, trailing clip, CIGAR
    insertions, then the split-SA fragments for the whole component. Downstream
    code takes "the first fragment" in a few places, and that is the leading
    clip of the first supporting read, not an arbitrary one.

    A fragment with an EMPTY sequence is dropped silently. That happens for a
    hard-clipped record, where the coordinates are known and the bases are not,
    and there is nothing to classify.
    """
    out: list[InsertionFragment] = []
    min_clip = config.min_soft_clip_for_seq_extract
    min_ins = relaxed_min_insertion_length(component, config)

    def emit(frag: InsertionFragment) -> None:
        if frag.sequence:
            out.append(frag)

    for idx in component.read_indices:
        if idx >= len(bin_records) or bin_records[idx] is None:
            continue
        read = bin_records[idx]
        read_id = read.qname
        read_token = sanitize_fragment_token(read_id)
        is_reverse = read.is_reverse

        class_mask = 0
        if read.has_sa_tag() or read.is_supplementary:
            class_mask |= CANDIDATE_SPLIT_SA_SUPPLEMENTARY

        clip = analyze_clip_info(read)
        if clip.leading >= min_clip or clip.trailing >= min_clip:
            class_mask |= CANDIDATE_SOFT_CLIP

        ins_ops = (find_long_insertions(read, min_ins)
                   if read.mapq == INSERTION_CANDIDATE_REQUIRED_MAPQ else [])
        if ins_ops:
            class_mask |= CANDIDATE_LONG_INSERTION

        nm = read.get_int_tag("NM")
        nm = nm if nm is not None else -1
        read_len = read.seq_len

        if clip.leading >= min_clip:
            frag = InsertionFragment(
                chrom=component.chrom, anchor_pos=component.anchor_pos,
                read_id=read_id, read_index=idx, class_mask=class_mask,
                is_reverse=is_reverse, source=InsertionFragmentSource.CLIP_REF_LEFT,
                start=0, length=clip.leading, read_len=read_len,
                anchor_len=clip.right_anchor_after_leading,
                ref_side=ReferenceSide.REF_LEFT, ref_junc_pos=read.pos, nm=nm,
                sequence=read.decode_subsequence(0, clip.leading))
            frag.fragment_id = (
                _fragment_id_prefix(component, read_token, idx, is_reverse, frag.source)
                + f"|rj={frag.ref_junc_pos}|len={frag.length}")
            emit(frag)

        if clip.trailing >= min_clip:
            length = clip.trailing
            start = max(0, read_len - length)
            frag = InsertionFragment(
                chrom=component.chrom, anchor_pos=component.anchor_pos,
                read_id=read_id, read_index=idx, class_mask=class_mask,
                is_reverse=is_reverse, source=InsertionFragmentSource.CLIP_REF_RIGHT,
                start=start, length=length, read_len=read_len,
                anchor_len=clip.left_anchor_before_trailing,
                ref_side=ReferenceSide.REF_RIGHT, ref_junc_pos=clip.ref_end, nm=nm,
                sequence=read.decode_subsequence(start, length))
            frag.fragment_id = (
                _fragment_id_prefix(component, read_token, idx, is_reverse, frag.source)
                + f"|rj={frag.ref_junc_pos}|len={frag.length}")
            emit(frag)

        for op in select_insertions_for_fragment_extraction(ins_ops, component):
            frag = InsertionFragment(
                chrom=component.chrom, anchor_pos=component.anchor_pos,
                read_id=read_id, read_index=idx, class_mask=class_mask,
                is_reverse=is_reverse, source=InsertionFragmentSource.CIGAR_INSERTION,
                start=op.start, length=op.len, read_len=read_len,
                anchor_len=min(op.left_anchor, op.right_anchor),
                ref_side=ReferenceSide.UNKNOWN, ref_junc_pos=op.ref_pos, nm=nm,
                sequence=read.decode_subsequence(op.start, op.len))
            frag.fragment_id = (
                _fragment_id_prefix(component, read_token, idx, is_reverse, frag.source)
                + f"|start={frag.start}|rj={frag.ref_junc_pos}|len={frag.length}")
            emit(frag)

    for frag in extract_split_sa_fragments(component, bin_records, config):
        emit(frag)
    return out


def fragments_to_fasta(fragments: list[InsertionFragment]) -> str:
    """The optional FASTA dump, as a string rather than a file handle.

    The C++ writes it under a mutex from every worker. Returning text instead
    keeps the decision of where it goes with the caller, and makes the format
    testable without a filesystem.
    """
    rows = []
    for frag in fragments:
        rows.append(f">{fasta_safe_header(frag.fragment_id)}\n")
        rows.append(wrapped_sequence(frag.sequence))
    return "".join(rows)
