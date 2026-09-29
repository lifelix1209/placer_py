"""
One CIGAR walk per read gives what the three walks gave.

`placer/alignment._build_cigar_index` now also records what the gate's
`summarize_cigar` and fragment extraction's `find_long_insertions` each walked
the whole CIGAR for, and those two read it from the index. Every stage reads
these facts, so this compares the new index, `cigar_summary` and the
long-insertion list with the implementations from before -- the index builder
and the long-insertion walk kept below verbatim, `summarize_cigar` itself --
on random CIGARs with every operation (hard and soft clips, padding, N, = and
X) and on the example BAM's real reads.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

from placer import alignment as A
from placer.alignment import (
    CIGAR_D,
    CIGAR_EQ,
    CIGAR_H,
    CIGAR_I,
    CIGAR_M,
    CIGAR_N,
    CIGAR_P,
    CIGAR_S,
    CIGAR_X,
    AlignedRead,
    cigar_summary,
    consumes_query,
    consumes_ref,
    find_first_non_hard_clip,
    find_last_non_hard_clip,
    is_match_like,
)
from placer.core import fragments as F
from placer.core.fragments import InsOp
from placer.reads import summarize_cigar

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "data"
INDEX_FIELDS = ("ref_end", "first", "first_op", "first_len", "first_ref_pos", "last",
                "last_op", "last_len", "last_ref_pos", "ins_ref_pos", "ins_len",
                "op_ref_start", "max_soft_clip", "max_insertion")


def _old_build_cigar_index(read: AlignedRead) -> A.CigarIndex:
    index = A.CigarIndex(ref_end=read.pos)
    cigar = read.cigar
    if not cigar:
        return index
    first = find_first_non_hard_clip(cigar)
    last = find_last_non_hard_clip(cigar)
    index.first = first
    index.last = last
    ins_ref_pos = index.ins_ref_pos
    ins_len = index.ins_len
    op_ref_start = index.op_ref_start
    consuming = A._CONSUMES_REF_OPS
    ref_pos = read.pos
    max_soft_clip = 0
    max_insertion = 0
    for i, (op, length) in enumerate(cigar):
        op_ref_start.append(ref_pos)
        if i == first:
            index.first_op, index.first_len, index.first_ref_pos = op, length, ref_pos
        if i == last:
            index.last_op, index.last_len, index.last_ref_pos = op, length, ref_pos
        if op == CIGAR_I:
            ins_ref_pos.append(ref_pos)
            ins_len.append(length)
            if length > max_insertion:
                max_insertion = length
        elif op in consuming:
            ref_pos += length
        elif op == CIGAR_S and length > max_soft_clip:
            max_soft_clip = length
    index.ref_end = ref_pos
    index.max_soft_clip = max_soft_clip
    index.max_insertion = max_insertion
    return index


def _old_contiguous_match_run_after(cigar: list[tuple[int, int]], idx: int) -> int:
    run = 0
    for op, length in cigar[idx + 1:]:
        if not is_match_like(op):
            break
        run += length
    return run


def _old_find_long_insertions(read: AlignedRead, min_long_ins: int) -> list[InsOp]:
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
                                 right_anchor=_old_contiguous_match_run_after(read.cigar, i)))
            qpos += length
            prev_match_run = 0
            continue
        if consumes_query(op):
            qpos += length
        if consumes_ref(op):
            rpos += length
        prev_match_run = 0
    return ops


def _random_cigar(rng: random.Random) -> list[tuple[int, int]]:
    body = []
    for _ in range(rng.randint(0, 40)):
        op = rng.choice((CIGAR_M, CIGAR_M, CIGAR_M, CIGAR_I, CIGAR_D, CIGAR_N,
                         CIGAR_EQ, CIGAR_X, CIGAR_P, CIGAR_S))
        body.append((op, rng.choice((1, 2, 5, 19, 20, 49, 50, 51, 300))))
    for side in (0, 1):
        if rng.random() < 0.5:
            clip = (CIGAR_S, rng.choice((1, 30, 500)))
            body = [clip] + body if side == 0 else body + [clip]
        if rng.random() < 0.3:
            hard = (CIGAR_H, rng.choice((5, 1000)))
            body = [hard] + body if side == 0 else body + [hard]
    return body


def _reads():
    rng = random.Random(67)
    for _ in range(3000):
        yield AlignedRead(qname="r", pos=rng.randint(0, 10_000), cigar=_random_cigar(rng),
                          seq="A" * 50)
    if importlib.util.find_spec("pysam") is not None and (EXAMPLE / "mini.bam").is_file():
        from placer.io.bam import make_bam_reader
        yield from make_bam_reader(str(EXAMPLE / "mini.bam"), 1).stream()


def test_one_walk_gives_the_index_the_summary_and_the_long_insertions():
    compared = 0
    for read in _reads():
        new = A._build_cigar_index(read)
        old = _old_build_cigar_index(read)
        for name in INDEX_FIELDS:
            assert getattr(new, name) == getattr(old, name), (name, read.cigar)
        fresh = AlignedRead(qname=read.qname, pos=read.pos, cigar=list(read.cigar),
                            seq=read.seq)
        assert cigar_summary(fresh) == summarize_cigar(read.cigar), read.cigar
        for floor in (1, 20, 50, 51):
            assert F._find_long_insertions(fresh, floor) == \
                _old_find_long_insertions(read, floor), (floor, read.cigar)
        compared += 1
    assert compared >= 3000
