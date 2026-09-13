from __future__ import annotations

from pathlib import Path

import pysam

from placer_py.redesign.models import BamEvidence


def _iter_cigar_positions(read):
    pos_ref = read.reference_start
    pos_read = 0
    for op, length in read.cigartuples or []:
        yield op, length, pos_ref, pos_read
        if op in (pysam.CMATCH, pysam.CDEL, pysam.CREF_SKIP, pysam.CEQUAL, pysam.CDIFF):
            pos_ref += length
        if op in (pysam.CMATCH, pysam.CINS, pysam.CSOFT_CLIP, pysam.CEQUAL, pysam.CDIFF):
            pos_read += length


def _has_near_insertion(read, pos0: int, window: int) -> tuple[bool, int]:
    max_len = 0
    found = False
    for op, length, ref_pos, _ in _iter_cigar_positions(read):
        if op == pysam.CINS and abs(ref_pos - pos0) <= window:
            found = True
            max_len = max(max_len, length)
    return found, max_len


def _clip_sides(read, pos0: int, window: int) -> tuple[bool, bool]:
    cigar = read.cigartuples or []
    left = bool(cigar and cigar[0][0] == pysam.CSOFT_CLIP and abs(read.reference_start - pos0) <= window)
    right = bool(
        cigar
        and cigar[-1][0] == pysam.CSOFT_CLIP
        and read.reference_end is not None
        and abs(read.reference_end - pos0) <= window
    )
    return left, right


def collect_bam_evidence_from_handle(
    bam: pysam.AlignmentFile,
    chrom: str,
    pos: int,
    window: int = 100,
    low_mapq: int = 20,
) -> BamEvidence:
    pos0 = max(0, pos - 1)
    start = max(0, pos0 - window)
    end = pos0 + window + 1
    raw_insert_reads = 0
    max_insert_len = 0
    left_clip_reads = 0
    right_clip_reads = 0
    ref_span_reads = 0
    low_mapq_ref_span_reads = 0
    fwd = 0
    rev = 0
    local_depth = 0

    for read in bam.fetch(chrom, start, end):
        if read.is_unmapped or read.is_secondary:
            continue
        local_depth += 1
        if read.is_reverse:
            rev += 1
        else:
            fwd += 1
        has_ins, ins_len = _has_near_insertion(read, pos0, window)
        if has_ins:
            raw_insert_reads += 1
            max_insert_len = max(max_insert_len, ins_len)
        left_clip, right_clip = _clip_sides(read, pos0, window)
        left_clip_reads += int(left_clip)
        right_clip_reads += int(right_clip)
        spans = read.reference_start <= pos0 and read.reference_end is not None and read.reference_end > pos0
        if spans and not has_ins and not left_clip and not right_clip:
            ref_span_reads += 1
            if read.mapping_quality < low_mapq:
                low_mapq_ref_span_reads += 1

    return BamEvidence(
        raw_cigar_insert_reads=raw_insert_reads,
        max_raw_cigar_insert_len=max_insert_len,
        left_clip_reads=left_clip_reads,
        right_clip_reads=right_clip_reads,
        ref_span_reads=ref_span_reads,
        low_mapq_ref_span_reads=low_mapq_ref_span_reads,
        forward_reads=fwd,
        reverse_reads=rev,
        local_depth=local_depth,
    )


def collect_bam_evidence(
    bam_path: str | Path,
    chrom: str,
    pos: int,
    window: int = 100,
    low_mapq: int = 20,
) -> BamEvidence:
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        return collect_bam_evidence_from_handle(bam, chrom, pos, window=window, low_mapq=low_mapq)
