"""
Read gating and CIGAR summarisation -- the front of the pipeline.

Ported from `src/gate1/gate1_module.cpp`, pinned by `tests/test_18_reads.py`.

WHAT THIS STAGE IS FOR. A 30x human BAM holds ~6 million reads and only a tiny
fraction touch an insertion. This gate decides, per read and from the CIGAR
alone, whether the read is worth carrying forward -- so it is the one place in
the whole pipeline where the per-read cost actually matters. Everything
downstream works on candidates, of which there are ~10^5.

THE SHAPE OF THE DECISION is a signal test followed by three fuses:

  signal  : supplementary, an SA tag, a long soft clip, or a long insertion.
            No signal means the read is kept only as background coverage, and
            then only if its MAPQ clears a floor.
  fuse 1  : at least one solid reference anchor, or the alignment is not
            trustworthy enough to locate anything.
  fuse 2  : for CLIP-ONLY proposals, the clip-adjacent flank must itself be
            anchored -- a long clip beside a short flank is a chimera, not a
            breakpoint. Deliberately skipped when a long insertion is also
            present, because direct insertion evidence should not be vetoed by
            an unrelated clip on the same noisy alignment.
  fuse 3  : for NON-INSERTION proposals, reject alignments whose NM rate is too
            high. Same asymmetry: a long insertion is breakpoint-specific enough
            that a global mismatch burden elsewhere on the read should not
            suppress it.

Both asymmetries are deliberate and worth preserving: the gate trusts a direct
insertion signal more than it trusts a clip, because a clip is consistent with
many things and an insertion is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# CIGAR operation codes, as in htslib's bam_cigar_op.
CIGAR_M, CIGAR_I, CIGAR_D, CIGAR_N = 0, 1, 2, 3
CIGAR_S, CIGAR_H, CIGAR_P, CIGAR_EQ, CIGAR_X = 4, 5, 6, 7, 8

#: SAM flag bits.
FLAG_UNMAP = 0x4
FLAG_SECONDARY = 0x100
FLAG_SUPPLEMENTARY = 0x800


@dataclass
class Gate1SignalConfig:
    min_seq_len: int = 50
    long_soft_clip_min: int = 100
    long_insertion_min: int = 50
    background_mapq_min: int = 20
    min_anchor_match_bases: int = 200
    min_clip_flank_match_bases: int = 120
    max_nm_rate: float = 0.20


@dataclass
class CigarSummary:
    total_match_bases: int = 0
    max_match_block: int = 0
    max_soft_clip: int = 0
    max_insertion: int = 0
    leading_soft_clip: int = 0
    trailing_soft_clip: int = 0
    right_anchor_after_leading: int = 0
    left_anchor_before_trailing: int = 0


def is_match_like(op: int) -> bool:
    """M, = and X all consume both reference and query and count as anchor."""
    return op in (CIGAR_M, CIGAR_EQ, CIGAR_X)


def find_first_non_hard_clip(cigar: Sequence[tuple[int, int]]) -> int:
    for i, (op, _) in enumerate(cigar):
        if op != CIGAR_H:
            return i
    return -1


def find_last_non_hard_clip(cigar: Sequence[tuple[int, int]]) -> int:
    for i in range(len(cigar) - 1, -1, -1):
        if cigar[i][0] != CIGAR_H:
            return i
    return -1


#: The private spellings this module used before `alignment.py` grew an
#: identical pair. Kept so its own call sites read unchanged.
_first_non_hard_clip = find_first_non_hard_clip
_last_non_hard_clip = find_last_non_hard_clip


def summarize_cigar(cigar: list[tuple[int, int]]) -> CigarSummary:
    """
    One pass for the aggregate statistics, then the two ends for the
    clip-adjacent anchors.

    `max_match_block` resets on ANY non-match operation, so it measures the
    longest uninterrupted anchor rather than the total -- a read with 200
    matched bases split into twenty blocks of 10 is not anchored, and the two
    fields distinguish that case.
    """
    summary = CigarSummary()
    if not cigar:
        return summary

    current_block = 0
    for op, length in cigar:
        if is_match_like(op):
            summary.total_match_bases += length
            current_block += length
            summary.max_match_block = max(summary.max_match_block, current_block)
        else:
            current_block = 0
        if op == CIGAR_S:
            summary.max_soft_clip = max(summary.max_soft_clip, length)
        elif op == CIGAR_I:
            summary.max_insertion = max(summary.max_insertion, length)

    first = _first_non_hard_clip(cigar)
    last = _last_non_hard_clip(cigar)
    if first < 0 or last < 0 or first > last:
        return summary

    if cigar[first][0] == CIGAR_S:
        summary.leading_soft_clip = cigar[first][1]
        flank = 0
        for i in range(first + 1, last + 1):
            if not is_match_like(cigar[i][0]):
                break
            flank += cigar[i][1]
        summary.right_anchor_after_leading = flank

    if cigar[last][0] == CIGAR_S:
        summary.trailing_soft_clip = cigar[last][1]
        flank = 0
        for i in range(last - 1, first - 1, -1):
            if not is_match_like(cigar[i][0]):
                break
            flank += cigar[i][1]
        summary.left_anchor_before_trailing = flank

    return summary


def pass_preliminary(cigar: list[tuple[int, int]], flag: int, seq_len: int,
                     mapq: int, has_sa_tag: bool, nm: int | None = None,
                     config: Gate1SignalConfig | None = None) -> bool:
    """
    Port of `placer::SignalFirstGate1Module::pass_preliminary`.

    Takes plain values rather than a read object so the gate is testable without
    a BAM. In the pysam path the caller supplies `read.cigartuples`,
    `read.flag`, `read.query_length`, `read.mapping_quality`,
    `read.has_tag("SA")` and `read.get_tag("NM")`.
    """
    cfg = config or Gate1SignalConfig()

    if (flag & FLAG_UNMAP) != 0 or (flag & FLAG_SECONDARY) != 0:
        return False
    if seq_len < cfg.min_seq_len:
        return False

    summary = summarize_cigar(cigar)

    has_supplementary = (flag & FLAG_SUPPLEMENTARY) != 0
    has_long_soft_clip = summary.max_soft_clip >= cfg.long_soft_clip_min
    has_long_insertion = summary.max_insertion >= cfg.long_insertion_min
    has_signal = (has_supplementary or has_sa_tag or has_long_soft_clip
                  or has_long_insertion)

    if not has_signal:
        # Note the STRICT inequality: a read exactly at the floor is dropped.
        return mapq > cfg.background_mapq_min

    # Fuse 1: at least one solid anchor on the reference.
    if summary.max_match_block < cfg.min_anchor_match_bases:
        return False

    # Fuse 2: clip-only proposals need clip-adjacent anchors.
    if has_long_soft_clip and not has_long_insertion:
        leading_long = summary.leading_soft_clip >= cfg.long_soft_clip_min
        trailing_long = summary.trailing_soft_clip >= cfg.long_soft_clip_min
        if (leading_long
                and summary.right_anchor_after_leading
                < cfg.min_clip_flank_match_bases):
            return False
        if (trailing_long
                and summary.left_anchor_before_trailing
                < cfg.min_clip_flank_match_bases):
            return False

    # Fuse 3: non-insertion proposals reject noisy alignments.
    if not has_long_insertion:
        if nm is not None and nm >= 0 and summary.total_match_bases > 0:
            if (nm / summary.total_match_bases) > cfg.max_nm_rate:
                return False

    return True


def insertion_length_from_split(left_qend: int, left_ref_end: int,
                                right_qstart: int, right_ref_start: int) -> int:
    """
    `query_gap - ref_gap` for a split-read pair.

    `query_gap` is the read's own bases that failed to align; `ref_gap` is the
    reference span skipped between the two segments, which those bases would
    have had to cover. What is left over is sequence the read has and the
    reference does not.

    The `max(0, ...)` on the reference gap matters: overlapping segments give a
    negative raw gap, and treating that as a negative requirement would inflate
    the insertion length rather than leaving it unchanged.
    """
    query_gap = right_qstart - left_qend
    ref_gap = max(0, right_ref_start - left_ref_end)
    return query_gap - ref_gap
