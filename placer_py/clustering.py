"""
Three-dimensional DBSCAN over insertion signatures.

Ported from `src/component/dbscan_component_module.cpp`, pinned by
`tests/test_20_clustering.py`.

WHY THREE DIMENSIONS. Clustering by position alone fails in tandem repeats: two
reads 30 bp apart reporting a 120 bp and a 4000 bp insertion are plainly not the
same event, yet position-only clustering merges them and reports "two reads
support". Putting LENGTH into the metric makes that impossible by construction:

    d = sqrt(dpos^2 + dend^2 + dlength^2)

This is the first design decision in the pipeline and it moves a judgement --
"these do not belong together" -- forward into the geometry, where it is cheap
and unambiguous, instead of leaving it for a downstream score to undo.

TWO EPSILONS. 250 when both signatures are CIGAR insertions, 500 otherwise.
Same-kind signals should agree more closely, so they are held to a tighter
standard; a clip paired with a split read legitimately disagrees more.

TWO PASSES OF minPts. 5 first, to find strong clusters, then 1 over the leftover
noise. The second pass is what keeps single-read events alive: they proceed with
weak evidence and are judged later, rather than being discarded at the geometry
stage where there is no evidence to judge them with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from placer_py.alignment import (
    CIGAR_I,
    CIGAR_S,
    AlignedRead,
    compute_ref_end,
    consumes_ref,
    find_first_non_hard_clip,
    find_last_non_hard_clip,
    median_i32,
    normalized_primary_alignment,
    normalized_sa_alignment,
    parse_sa_tag_z,
)

#: Tighter epsilon when both signatures come from a CIGAR insertion.
INTRA_SIGNATURE_EPSILON = 250.0
#: Looser epsilon for any other pairing.
INTER_SIGNATURE_EPSILON = 500.0

STRONG_MIN_PTS = 5
WEAK_MIN_PTS = 1


#: A soft clip shorter than this is noise, not a breakpoint hint.
COMPONENT_SOFT_CLIP_SIGNAL_MIN = 20
#: The shortest insertion worth a signature. Below it the event is an indel.
COMPONENT_LONG_INSERTION_SIGNAL_MIN = 50
#: Signatures further apart than this cannot be the same event, so the O(n^2)
#: pass is run per block rather than genome-wide.
COMPATIBILITY_WINDOW_BP = 4000
#: Hard cap on block size. One repetitive locus can otherwise pile thousands of
#: signatures into a single block and make its cost dominate the whole run.
#: The cap can split a genuine cluster -- see `build_blocks`.
MAX_BLOCK_SIGNATURES = 1500
#: CIGAR insertions are believed only from a uniquely-mapped read. This is an
#: equality test in the C++, not a floor: minimap2 reports exactly 60 for
#: unique, and anything else means the aligner itself was unsure.
INSERTION_CANDIDATE_REQUIRED_MAPQ = 60

#: `CandidateClassMask`. Kept as a bitmask because one read can propose a
#: breakpoint for more than one reason and the reasons are not exclusive.
CANDIDATE_SOFT_CLIP = 1 << 0
CANDIDATE_SPLIT_SA_SUPPLEMENTARY = 1 << 1
CANDIDATE_LONG_INSERTION = 1 << 2


class SignatureSource(IntEnum):
    """Values follow the C++ `SignatureSource` so the two can be diffed."""

    CIGAR_INSERTION = 0
    SPLIT_INSERTION = 1
    CLIP_HINT = 2


@dataclass
class InsertionSignature:
    pos: int = 0
    end: int = 0
    length: int = 0
    read_index: int = -1
    source: SignatureSource = SignatureSource.CIGAR_INSERTION
    class_mask: int = 0
    is_reverse: bool = False
    anchor_len: int = 0
    read_id: str = ""


@dataclass
class BreakpointCandidate:
    """One read's proposal of a breakpoint, with its reason attached."""

    chrom: str = ""
    pos: int = -1
    is_reverse: bool = False
    anchor_len: int = 0
    clip_len: int = 0
    ins_len: int = 0
    read_id: str = ""
    read_index: int = 0
    class_mask: int = 0


@dataclass
class ComponentCall:
    """A cluster projected back onto the reference.

    The four read-index sets are kept SEPARATE rather than summed: the counts
    that feed the genotype model and the count Bayes factor distinguish a
    clip-only cluster from one with real insertion evidence, and collapsing them
    into a single "supporting reads" number would erase the distinction at the
    point where it is cheapest to keep.
    """

    chrom: str = ""
    tid: int = -1
    bin_start: int = -1
    bin_end: int = -1
    anchor_pos: int = -1
    peak_weight: float = 0.0
    evidence_soft_clip_count: int = 0
    evidence_indel_count: int = 0
    evidence_sa_hint_count: int = 0
    read_indices: list[int] = field(default_factory=list)
    soft_clip_read_indices: list[int] = field(default_factory=list)
    split_sa_read_indices: list[int] = field(default_factory=list)
    insertion_read_indices: list[int] = field(default_factory=list)
    breakpoint_candidates: list[BreakpointCandidate] = field(default_factory=list)


def signature_distance(lhs: InsertionSignature,
                       rhs: InsertionSignature) -> float:
    """Euclidean distance over (pos, end, length).

    Length enters UNSCALED, in the same units as position. That is a modelling
    choice rather than a neutral one -- it says a 100 bp length disagreement is
    as disqualifying as a 100 bp positional one -- and it is what the two
    epsilons are then calibrated against.
    """
    dx = float(lhs.pos - rhs.pos)
    dy = float(lhs.end - rhs.end)
    dz = float(lhs.length - rhs.length)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def signature_epsilon(lhs: InsertionSignature,
                      rhs: InsertionSignature) -> float:
    both_intra = (lhs.source == SignatureSource.CIGAR_INSERTION
                  and rhs.source == SignatureSource.CIGAR_INSERTION)
    return INTRA_SIGNATURE_EPSILON if both_intra else INTER_SIGNATURE_EPSILON


def dbscan_cluster_indices(signatures: list[InsertionSignature],
                           candidate_indices: list[int],
                           min_pts: int) -> list[list[int]]:
    """
    Port of `placer::dbscan_cluster_indices`.

    Two details that a from-scratch DBSCAN gets differently and which the port
    must preserve:

      * the neighbourhood test is STRICT (`distance < epsilon`), so a pair
        exactly at the threshold is not connected;
      * the core-point test counts the point itself
        (`len(neighbours) + 1 >= min_pts`), so `min_pts = 1` makes every point a
        core point and the second pass degenerates to connected components.

    The expansion also only pushes a newly-visited point's neighbours when that
    point is itself a core point, which is what keeps border points from
    chaining clusters together.

    Cost note: this is O(n^2) within each block, which is why the C++ caps block
    size -- a cap that can split a genuine cluster. The right fix is a spatial
    index, and it is a fix the language choice has nothing to do with.
    """
    n = len(candidate_indices)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lhs = signatures[candidate_indices[i]]
            rhs = signatures[candidate_indices[j]]
            if signature_distance(lhs, rhs) < signature_epsilon(lhs, rhs):
                adjacency[i].append(j)
                adjacency[j].append(i)

    labels = [-1] * n
    next_label = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        if len(adjacency[i]) + 1 < min_pts:
            labels[i] = 0                      # noise
            continue

        next_label += 1
        labels[i] = next_label
        stack = list(adjacency[i])
        while stack:
            j = stack.pop()
            if labels[j] == next_label:
                continue
            was_unvisited = labels[j] == -1
            labels[j] = next_label
            if was_unvisited and len(adjacency[j]) + 1 >= min_pts:
                stack.extend(adjacency[j])

    clusters: list[list[int]] = [[] for _ in range(next_label)]
    for i in range(n):
        if labels[i] > 0:
            clusters[labels[i] - 1].append(candidate_indices[i])
    return clusters


def collect_noise_indices(candidate_indices: list[int],
                          clusters: list[list[int]]) -> list[int]:
    clustered = {index for cluster in clusters for index in cluster}
    return [index for index in candidate_indices if index not in clustered]


def cluster_signatures(signatures: list[InsertionSignature],
                       candidate_indices: list[int] | None = None
                       ) -> list[list[int]]:
    """
    The two-pass policy: strong clusters at `minPts = 5`, then the leftover
    noise re-clustered at `minPts = 1`.

    The second pass is not a fallback for a failed first pass -- it is how
    single-read events survive the geometry stage at all. They carry weak
    evidence forward and are judged by the later gates, which have evidence to
    judge them with; discarding them here would be deciding with none.
    """
    indices = (list(range(len(signatures))) if candidate_indices is None
               else list(candidate_indices))
    clusters = dbscan_cluster_indices(signatures, indices, STRONG_MIN_PTS)
    noise = collect_noise_indices(indices, clusters)
    if noise:
        clusters.extend(dbscan_cluster_indices(signatures, noise, WEAK_MIN_PTS))
    return clusters


# ---------------------------------------------------------------------------
# Signature extraction: three ways a read can propose an insertion, and the
# precedence between them.
# ---------------------------------------------------------------------------
def append_insertion_signatures_from_cigar(read: AlignedRead, read_index: int,
                                           out: list[InsertionSignature]) -> bool:
    """Every long `I` op in a uniquely-mapped read becomes a signature.

    The MAPQ test is an EQUALITY against 60, not a floor. A CIGAR insertion is
    the aligner's own statement that the read's extra bases belong at this exact
    reference position, and that statement is only worth taking at face value
    when the aligner was certain where the read goes at all.

    `end` is `pos + 1`, not `pos + length`: an insertion occupies no reference
    span. Giving it one would make the second clustering coordinate track the
    insertion length twice, once as `end` and once as `length`.
    """
    if read.mapq != INSERTION_CANDIDATE_REQUIRED_MAPQ or not read.cigar:
        return False

    emitted = False
    ref_pos = read.pos
    for op, length in read.cigar:
        if op == CIGAR_I and length >= COMPONENT_LONG_INSERTION_SIGNAL_MIN:
            out.append(InsertionSignature(
                pos=ref_pos,
                end=ref_pos + 1,
                length=length,
                read_index=read_index,
                source=SignatureSource.CIGAR_INSERTION,
                class_mask=CANDIDATE_LONG_INSERTION,
                is_reverse=read.is_reverse,
                anchor_len=length,
                read_id=read.qname,
            ))
            emitted = True
        if consumes_ref(op):
            ref_pos += length
    return emitted


def append_best_split_signature(read: AlignedRead, chrom: str, read_index: int,
                                out: list[InsertionSignature]) -> bool:
    """The LONGEST insertion implied by any same-strand SA segment, or none.

    Three filters, each removing a different non-insertion:

      * a segment on another contig is a translocation, not an insertion;
      * a segment on the other strand is an inversion, not an insertion;
      * `query_gap - ref_gap` below the floor is an indel, not an insertion.

    Only the single best segment is emitted. A read split into four parts could
    support four insertions in principle, but in practice the extra segments are
    the same event re-reported, and emitting all of them would let one read
    supply its own "independent" support.
    """
    sa_z = read.get_string_tag("SA")
    if not sa_z:
        return False
    primary = normalized_primary_alignment(read, chrom)
    if primary is None:
        return False

    best_length = -1
    best_anchor = -1
    for entry in parse_sa_tag_z(sa_z):
        if entry.rname != chrom:
            continue
        mate = normalized_sa_alignment(entry, read.seq_len)
        if mate is None or mate.is_reverse != primary.is_reverse:
            continue
        left, right = (primary, mate) if mate.qstart >= primary.qstart else (mate, primary)
        query_gap = right.qstart - left.qend
        ref_gap = max(0, right.ref_start - left.ref_end)
        insertion_len = query_gap - ref_gap
        if insertion_len >= COMPONENT_LONG_INSERTION_SIGNAL_MIN and insertion_len > best_length:
            best_length = insertion_len
            best_anchor = left.ref_end

    if best_length < 0 or best_anchor < 0:
        return False

    out.append(InsertionSignature(
        pos=best_anchor,
        end=best_anchor + 1,
        length=best_length,
        read_index=read_index,
        source=SignatureSource.SPLIT_INSERTION,
        class_mask=CANDIDATE_SPLIT_SA_SUPPLEMENTARY,
        is_reverse=read.is_reverse,
        anchor_len=best_length,
        read_id=read.qname,
    ))
    return True


def append_clip_hint_signatures(read: AlignedRead, read_index: int,
                                out: list[InsertionSignature]) -> None:
    """Both terminal soft clips, as the weakest kind of proposal.

    A clip says only "the read stops explaining the reference here". Its
    `length` is the CLIP length, which is a lower bound on the insertion and
    usually a bad one -- so clip signatures cluster against each other at the
    loose epsilon, and the length coordinate carries little information. They
    exist to keep a locus alive until the evidence stage can look at it.
    """
    if not read.cigar:
        return
    first = find_first_non_hard_clip(read.cigar)
    last = find_last_non_hard_clip(read.cigar)

    if first >= 0 and read.cigar[first][0] == CIGAR_S:
        length = read.cigar[first][1]
        if length >= COMPONENT_SOFT_CLIP_SIGNAL_MIN:
            out.append(InsertionSignature(
                pos=read.pos, end=read.pos + 1, length=length,
                read_index=read_index, source=SignatureSource.CLIP_HINT,
                class_mask=CANDIDATE_SOFT_CLIP, is_reverse=read.is_reverse,
                anchor_len=length, read_id=read.qname))

    if last >= 0 and read.cigar[last][0] == CIGAR_S:
        length = read.cigar[last][1]
        if length >= COMPONENT_SOFT_CLIP_SIGNAL_MIN:
            ref_end = compute_ref_end(read)
            out.append(InsertionSignature(
                pos=ref_end, end=ref_end + 1, length=length,
                read_index=read_index, source=SignatureSource.CLIP_HINT,
                class_mask=CANDIDATE_SOFT_CLIP, is_reverse=read.is_reverse,
                anchor_len=length, read_id=read.qname))


def extract_signatures(records: list[AlignedRead], chrom: str,
                       tid: int) -> list[InsertionSignature]:
    """Turn a window of reads into sorted signatures.

    THE PRECEDENCE IS THE POINT. Clip hints are emitted only when the read
    produced neither a CIGAR insertion nor a split signature, because a clip
    beside a known insertion is the SAME evidence counted twice -- and counting
    it twice is exactly how a single read comes to look like two.

    Supplementary records are skipped outright: the primary already carries the
    split evidence through its SA tag, so admitting the supplementary would
    double-count the same alignment from the other end.
    """
    out: list[InsertionSignature] = []
    for read_index, read in enumerate(records):
        if read is None or read.tid != tid or read.is_supplementary:
            continue
        emitted_cigar = append_insertion_signatures_from_cigar(read, read_index, out)
        emitted_split = (read.has_sa_tag()
                         and append_best_split_signature(read, chrom, read_index, out))
        if not emitted_cigar and not emitted_split:
            append_clip_hint_signatures(read, read_index, out)

    out.sort(key=lambda s: (s.pos, s.end, s.length, s.read_index))
    return out


@dataclass
class SignatureBlock:
    begin: int = 0
    end: int = 0


def build_blocks(signatures: list[InsertionSignature]) -> list[SignatureBlock]:
    """Split sorted signatures into independently clusterable runs.

    A gap wider than the compatibility window means no pair across it can ever
    be within epsilon, so the split is free. The SIZE CAP is not free: it can
    cut a genuine cluster in half at a repetitive locus, and the two halves are
    then reported as two events. It is a cost accepted to bound the O(n^2) pass,
    and the real fix is a spatial index rather than a cap -- a fix that has
    nothing to do with which language this runs in.
    """
    blocks: list[SignatureBlock] = []
    if not signatures:
        return blocks

    begin = 0
    for i in range(1, len(signatures)):
        if (signatures[i].pos - signatures[i - 1].pos) > COMPATIBILITY_WINDOW_BP:
            blocks.append(SignatureBlock(begin, i))
            begin = i
    blocks.append(SignatureBlock(begin, len(signatures)))

    capped: list[SignatureBlock] = []
    for block in blocks:
        chunk_begin = block.begin
        while block.end - chunk_begin > MAX_BLOCK_SIGNATURES:
            capped.append(SignatureBlock(chunk_begin, chunk_begin + MAX_BLOCK_SIGNATURES))
            chunk_begin += MAX_BLOCK_SIGNATURES
        capped.append(SignatureBlock(chunk_begin, block.end))
    return capped


def project_cluster(signatures: list[InsertionSignature], cluster_members: list[int],
                    chrom: str, tid: int) -> ComponentCall:
    """Collapse a cluster into one call, keeping every read's own proposal.

    `anchor_pos` is the MEDIAN of the member positions, not the mean: it has to
    be a position some read actually proposed, and it must not be dragged by a
    single mislocated clip.

    `peak_weight` counts distinct READS, not signatures. One read contributing a
    CIGAR insertion and a split signature is one piece of evidence, and the four
    index sets are deduplicated for the same reason.
    """
    call = ComponentCall(chrom=chrom, tid=tid)
    bin_start: int | None = None
    bin_end: int | None = None

    anchors: list[int] = []
    all_reads: set[int] = set()
    clip_reads: set[int] = set()
    split_reads: set[int] = set()
    insertion_reads: set[int] = set()

    for index in cluster_members:
        sign = signatures[index]
        bin_start = sign.pos if bin_start is None else min(bin_start, sign.pos)
        bin_end = sign.end if bin_end is None else max(bin_end, sign.end)
        anchors.append(sign.pos)
        all_reads.add(sign.read_index)

        bp = BreakpointCandidate(
            chrom=chrom, pos=sign.pos, read_id=sign.read_id,
            read_index=sign.read_index, class_mask=sign.class_mask,
            is_reverse=sign.is_reverse, anchor_len=sign.anchor_len)
        if sign.source == SignatureSource.CIGAR_INSERTION:
            insertion_reads.add(sign.read_index)
            call.evidence_indel_count += 1
            bp.ins_len = sign.length
        elif sign.source == SignatureSource.SPLIT_INSERTION:
            split_reads.add(sign.read_index)
            call.evidence_sa_hint_count += 1
            bp.ins_len = sign.length
        else:
            clip_reads.add(sign.read_index)
            call.evidence_soft_clip_count += 1
            bp.clip_len = sign.length
        call.breakpoint_candidates.append(bp)

    call.anchor_pos = median_i32(anchors)
    call.peak_weight = float(len(all_reads))
    call.read_indices = sorted(all_reads)
    call.soft_clip_read_indices = sorted(clip_reads)
    call.split_sa_read_indices = sorted(split_reads)
    call.insertion_read_indices = sorted(insertion_reads)
    call.bin_start = bin_start if bin_start is not None else call.anchor_pos
    call.bin_end = bin_end if bin_end is not None else call.anchor_pos + 1
    return call


def build_component_calls(records: list[AlignedRead], chrom: str,
                          tid: int) -> list[ComponentCall]:
    """`DbscanComponentModule::build`: the whole geometry stage, end to end.

    The final ordering is by position, then by DESCENDING weight: when two
    clusters share an anchor the better-supported one is the representative,
    and everything downstream that takes the first match gets the right one.
    """
    signatures = extract_signatures(records, chrom, tid)
    out: list[ComponentCall] = []
    for block in build_blocks(signatures):
        indices = list(range(block.begin, block.end))
        clusters = dbscan_cluster_indices(signatures, indices, STRONG_MIN_PTS)
        noise = collect_noise_indices(indices, clusters)
        if noise:
            clusters.extend(dbscan_cluster_indices(signatures, noise, WEAK_MIN_PTS))
        for cluster in clusters:
            if cluster:
                out.append(project_cluster(signatures, cluster, chrom, tid))

    out.sort(key=lambda c: (c.anchor_pos, -c.peak_weight, -len(c.read_indices)))
    return out
