"""
The read view, and the CIGAR/SA parsing every stage downstream of it needs.

Ported from `include/bam_io.h` (the `ReadView` accessors) and from the
CIGAR/SA helpers that `src/component/dbscan_component_module.cpp`,
`src/component/insert_fragment_module.cpp` and
`src/pipeline/pipeline_local_alignment_helpers.inc` each define privately.

WHY `AlignedRead` IS A PLAIN DATACLASS AND NOT A pysam WRAPPER. `ReadView` in
the C++ is a non-owning view over a `bam1_t`, which is why the C++ suite cannot
test any of this without linking htslib -- and why almost none of it IS tested
there. Making the read an ordinary value object with the same eight accessors
means every stage below can be exercised from a literal, so the port's tests can
pin behaviour the C++ tests never reach. `placer/io/pysam_adapter.py` adapts
a real BAM record onto the same shape, so nothing downstream knows which it is
holding -- and it lives in the input stage rather than here, so this module
stays a type the algorithm can own without the package that produces it.

THE SA TAG IS THE SECOND SOURCE OF TRUTH about an insertion, and the parsing
here is deliberately forgiving: a malformed record is dropped, never raised on.
A single bad SA field in one read of a 6-million-read BAM must not end the run,
and the read still carries its CIGAR evidence.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from placer.reads import (  # noqa: F401  (re-exported: one definition of each)
    CIGAR_D,
    CIGAR_EQ,
    CIGAR_H,
    CIGAR_I,
    CIGAR_M,
    CIGAR_N,
    CIGAR_P,
    CIGAR_S,
    CIGAR_X,
    FLAG_SECONDARY,
    FLAG_SUPPLEMENTARY,
    FLAG_UNMAP,
    find_first_non_hard_clip,
    find_last_non_hard_clip,
    is_match_like,
)

#: BAM_FREVERSE. The strand bit, needed because a split pair whose two segments
#: disagree on strand is an inversion signature and not an insertion one.
FLAG_REVERSE = 0x10

#: Per-op consumption, in the two encodings the C++ uses. `bam_cigar_type`
#: gives the integer form; the SA tag is a string, hence the char form.
_CONSUMES_QUERY_OPS = {CIGAR_M, CIGAR_EQ, CIGAR_X, CIGAR_I, CIGAR_S}
_CONSUMES_REF_OPS = {CIGAR_M, CIGAR_EQ, CIGAR_X, CIGAR_D, CIGAR_N}
_OP_CHARS = "MIDNSHP=X"
_CHAR_TO_OP = {c: i for i, c in enumerate(_OP_CHARS)}


def consumes_query(op: int) -> bool:
    return op in _CONSUMES_QUERY_OPS


def consumes_ref(op: int) -> bool:
    return op in _CONSUMES_REF_OPS


def consumes_query_char(op: str) -> bool:
    return op in ("M", "=", "X", "I", "S")


def consumes_ref_char(op: str) -> bool:
    return op in ("M", "=", "X", "D", "N")


def op_to_char(op: int) -> str:
    return _OP_CHARS[op] if 0 <= op < len(_OP_CHARS) else "?"


def char_to_op(op: str) -> int:
    return _CHAR_TO_OP.get(op, -1)


@dataclass
class AlignedRead:
    """The `ReadView` surface, as a value.

    `cigar` is a list of `(op, length)` in htslib's integer encoding, matching
    what `pysam`'s `cigartuples` already returns, so the adapter is a copy and
    not a translation.
    """

    qname: str = ""
    flag: int = 0
    tid: int = -1
    pos: int = 0
    mapq: int = 0
    cigar: list[tuple[int, int]] = field(default_factory=list)
    seq: str = ""
    #: `Any`, not `object`: pysam hands back whatever the tag's type code
    #: says -- str, int, float or an array -- and the three accessors below
    #: coerce defensively rather than trusting it. Typed as `object` the
    #: coercions do not type-check at all (`int()` does not accept
    #: `object`), which is a checker complaining about the annotation
    #: rather than about the code.
    tags: dict[str, Any] = field(default_factory=dict)
    #: Facts derived from the read, computed once on first use: the
    #: `cigar_index`, and the per-argument answers `read_memo` holds for the
    #: stages that ask the same read the same question many times. Not part
    #: of the value: excluded from `__init__`, `repr` and equality, so two
    #: reads with the same fields compare equal whether or not either has
    #: been queried. Safe only because nothing mutates a read after it is
    #: built -- every stage treats `AlignedRead` as a value, and a caller that
    #: edited `cigar` in place would have to reset both to None by hand.
    _cigar_index: CigarIndex | None = field(default=None, init=False, repr=False,
                                            compare=False)
    _memo: dict[tuple[Any, ...], Any] | None = field(default=None, init=False,
                                                    repr=False, compare=False)

    @property
    def seq_len(self) -> int:
        """Query length. Falls back to the CIGAR when the sequence is absent --
        a hard-clipped supplementary record legitimately carries no bases, and
        its own CIGAR is then the only statement of how long it is."""
        if self.seq:
            return len(self.seq)
        return sum(length for op, length in self.cigar if consumes_query(op))

    @property
    def n_cigar(self) -> int:
        return len(self.cigar)

    @property
    def is_reverse(self) -> bool:
        return (self.flag & FLAG_REVERSE) != 0

    @property
    def is_supplementary(self) -> bool:
        return (self.flag & FLAG_SUPPLEMENTARY) != 0

    @property
    def is_secondary(self) -> bool:
        return (self.flag & FLAG_SECONDARY) != 0

    @property
    def is_unmapped(self) -> bool:
        return (self.flag & FLAG_UNMAP) != 0

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def has_sa_tag(self) -> bool:
        return bool(self.tags.get("SA"))

    def has_md_tag(self) -> bool:
        return "MD" in self.tags

    def get_int_tag(self, tag: str) -> int | None:
        value = self.tags.get(tag)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def get_string_tag(self, tag: str) -> str | None:
        value = self.tags.get(tag)
        return None if value is None else str(value)

    def decode_subsequence(self, start: int, length: int) -> str:
        if start < 0 or length <= 0 or start >= len(self.seq):
            return ""
        return self.seq[start:start + length]

    def decode_sequence(self) -> str:
        return self.seq


@dataclass
class CigarIndex:
    """The facts about one CIGAR that every window asks for, computed once.

    WHY IT EXISTS. An ultra-long ONT read carries tens of thousands of CIGAR
    operations, and the event stages ask the same read about many windows --
    one per hypothesis, per component, per bin it is fetched into. Walking the
    whole CIGAR for each question made `classify_local_event_signal` and
    `compute_ref_end` a quarter of a real run. One walk builds this, and each
    window is then a pair of binary searches over `ins_ref_pos`.

    `first`/`last` are the indices `find_first_non_hard_clip` and
    `find_last_non_hard_clip` return (-1 when there is none), with the op, its
    length and the reference position at which it starts. `ins_ref_pos` and
    `ins_len` list every `I` operation in CIGAR order; the positions are
    therefore non-decreasing, which is what makes them searchable.
    """

    ref_end: int = 0
    first: int = -1
    first_op: int = -1
    first_len: int = 0
    first_ref_pos: int = 0
    last: int = -1
    last_op: int = -1
    last_len: int = 0
    last_ref_pos: int = 0
    #: `array('q')`, not `list`: a p90 ultra-long ONT read has ~5,000 CIGAR
    #: operations, about a sixth of them `I`, and every read of a bin keeps
    #: its index while the bin is processed. Eight bytes an entry instead of
    #: a pointer plus a boxed int.
    ins_ref_pos: array = field(default_factory=lambda: array("q"))
    ins_len: array = field(default_factory=lambda: array("q"))
    #: The reference position at which each CIGAR operation starts, one entry
    #: per operation. Non-decreasing, so a window's operations are a
    #: contiguous run found by binary search (`fragments.anchor_len_from_bam`).
    op_ref_start: array = field(default_factory=lambda: array("q"))
    max_soft_clip: int = 0
    max_insertion: int = 0


def _build_cigar_index(read: AlignedRead) -> CigarIndex:
    index = CigarIndex(ref_end=read.pos)
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
    consuming = _CONSUMES_REF_OPS
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


def cigar_index(read: AlignedRead) -> CigarIndex:
    """The read's `CigarIndex`, built on first use and kept on the read.

    Tolerates a read-like object without the cache slot (a test double, say):
    it then gets a fresh index every time, which is correct and merely slow.
    """
    index = getattr(read, "_cigar_index", None)
    if index is None:
        index = _build_cigar_index(read)
        with suppress(AttributeError):
            read._cigar_index = index
    return index


def read_memo(read: AlignedRead) -> dict[tuple[Any, ...], Any] | None:
    """The read's memo table, created on first use; None for a read-like
    object with no slot for one, whose callers then simply recompute.

    Keys are tuples whose first element names the question, so two stages can
    never collide. Values must be immutable (tuples, ints): the table is
    shared by every caller that asks.
    """
    memo = getattr(read, "_memo", None)
    if memo is None:
        memo = {}
        try:
            read._memo = memo
        except AttributeError:
            return None
    return memo


def compute_ref_end(read: AlignedRead) -> int:
    """One past the last reference base the alignment covers.

    Returns `pos` for a CIGAR-less record rather than raising, so an unusable
    record contributes a zero-length interval and is dropped by the callers'
    own `end > start` checks instead of by an exception.
    """
    return cigar_index(read).ref_end


def median_i32(values: Iterable[int]) -> int:
    """Upper median, and -1 for an empty input.

    NOT the average of the two middle values for an even count: a breakpoint is
    a position, and averaging two positions can name a base that no read
    actually proposed. The C++ takes `values[n/2]` and so does this.
    """
    ordered = sorted(values)
    if not ordered:
        return -1
    return ordered[len(ordered) // 2]


@dataclass
class CigarStringOp:
    length: int = 0
    op: str = "\0"


@dataclass
class QueryInterval:
    """Where an alignment sits in READ coordinates."""

    qstart: int = 0
    qend: int = 0
    leading_s: int = 0
    trailing_s: int = 0


@dataclass
class SAEntry:
    rname: str = ""
    pos: int = 0  # 1-based, as the tag stores it
    strand: str = "+"
    cigar: str = ""


@dataclass
class NormalizedAlignment:
    """One alignment segment in both coordinate systems at once.

    Reference and query intervals together are what make a split pair
    interpretable: the reference gap says what the genome skipped, the query gap
    says what the read carried, and the insertion is the difference.
    """

    chrom: str = ""
    ref_start: int = -1
    ref_end: int = -1
    qstart: int = -1
    qend: int = -1
    is_reverse: bool = False


def parse_int32(text: str) -> int | None:
    """Strict decimal parse: the WHOLE string, or nothing.

    `strtol`-style prefix parsing would read "12abc" as 12, which in an SA tag
    means silently accepting a corrupt position.
    """
    if not text:
        return None
    body = text[1:] if text[0] in "+-" else text
    if not body or not body.isdigit():
        return None
    return int(text, 10)


def parse_sa_tag_z(sa_z: str) -> list[SAEntry]:
    """Parse an `SA:Z:` tag into its records.

    Malformed records are DROPPED rather than raised on, and each record is
    independent: a corrupt third segment does not cost the first two.
    """
    out: list[SAEntry] = []
    if not sa_z:
        return out
    start = 0
    while start < len(sa_z):
        rec_end = sa_z.find(";", start)
        rec = sa_z[start:] if rec_end == -1 else sa_z[start:rec_end]
        if not rec:
            break
        fields = rec.split(",")
        if len(fields) >= 4:
            pos = parse_int32(fields[1])
            entry = SAEntry(
                rname=fields[0],
                pos=pos if pos is not None else 0,
                strand=fields[2][0] if fields[2] else "+",
                cigar=fields[3],
            )
            if entry.rname and entry.pos > 0 and entry.cigar:
                out.append(entry)
        if rec_end == -1:
            break
        start = rec_end + 1
    return out


def parse_cigar_ops(cigar: str) -> list[CigarStringOp] | None:
    """Parse a CIGAR STRING (the SA tag's form) into ops.

    Returns None on anything malformed -- a zero-length op, an operator with no
    count, or a trailing count with no operator. The caller treats None as "this
    SA record is unusable", which is the right outcome: a half-parsed CIGAR
    would place the segment at the wrong query offset.
    """
    if not cigar:
        return None
    ops: list[CigarStringOp] = []
    num = 0
    have_num = False
    for c in cigar:
        if c.isdigit():
            have_num = True
            num = (num * 10) + (ord(c) - 48)
            continue
        if not have_num or num <= 0:
            return None
        ops.append(CigarStringOp(num, c))
        num = 0
        have_num = False
    if have_num or not ops:
        return None
    return ops


def cigar_to_query_interval(ops: Sequence[CigarStringOp],
                            read_len: int) -> tuple[QueryInterval, int] | None:
    """Place an SA segment in the PRIMARY read's coordinates.

    The hard case, and the reason this is not a one-liner: an SA record's CIGAR
    is written in its own orientation and may hard-clip what the primary soft-
    clips, so the query offset has to be RECONSTRUCTED rather than read off. The
    three branches are the three things that can be known:

      * a leading soft clip states the offset directly;
      * otherwise a trailing soft clip states it from the far end;
      * otherwise the segment must consume the whole read, or its position is
        genuinely unknown and the record is refused.

    Refusing is the right third branch. An SA segment with only hard clips and a
    partial query span cannot be located, and guessing 0 would silently place
    every such segment at the read start.
    """
    out = QueryInterval()
    ref_aligned_len = 0
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

    out.qstart = qstart
    out.qend = qstart + q_aligned
    if not (out.qend > out.qstart and out.qend <= read_len):
        return None
    return out, ref_aligned_len


def bam_to_query_interval(read: AlignedRead) -> QueryInterval | None:
    """The same placement for the PRIMARY record, where the soft clips are real.

    The clips bracket the aligned span directly, so the interval is read off
    them. The `q_aligned` fallback covers the one inconsistency that occurs in
    practice: a record whose clip lengths and sequence length disagree, where
    trusting the clips would give an empty or over-long interval.
    """
    read_len = read.seq_len
    if read_len <= 0 or not read.cigar:
        return None
    out = QueryInterval()

    first = find_first_non_hard_clip(read.cigar)
    last = find_last_non_hard_clip(read.cigar)
    if first >= 0 and read.cigar[first][0] == CIGAR_S:
        out.leading_s = read.cigar[first][1]
    if last >= 0 and read.cigar[last][0] == CIGAR_S:
        out.trailing_s = read.cigar[last][1]

    q_aligned = 0
    for op, length in read.cigar:
        if op in (CIGAR_M, CIGAR_EQ, CIGAR_X, CIGAR_I):
            q_aligned += length
    if q_aligned <= 0:
        return None

    out.qstart = out.leading_s
    out.qend = read_len - out.trailing_s
    if out.qend <= out.qstart or out.qend > read_len:
        out.qend = out.qstart + q_aligned
    if not (out.qend > out.qstart and out.qend <= read_len):
        return None
    return out


def normalized_primary_alignment(read: AlignedRead,
                                 chrom: str = "") -> NormalizedAlignment | None:
    interval = bam_to_query_interval(read)
    if interval is None:
        return None
    out = NormalizedAlignment(
        chrom=chrom,
        ref_start=read.pos,
        ref_end=compute_ref_end(read),
        qstart=interval.qstart,
        qend=interval.qend,
        is_reverse=read.is_reverse,
    )
    if not (out.ref_end > out.ref_start and out.qend > out.qstart):
        return None
    return out


def normalized_sa_alignment(entry: SAEntry,
                            read_len: int) -> NormalizedAlignment | None:
    ops = parse_cigar_ops(entry.cigar)
    if ops is None:
        return None
    placed = cigar_to_query_interval(ops, read_len)
    if placed is None:
        return None
    interval, ref_aligned_len = placed
    out = NormalizedAlignment(
        chrom=entry.rname,
        ref_start=entry.pos - 1,  # the tag is 1-based
        ref_end=entry.pos - 1 + ref_aligned_len,
        qstart=interval.qstart,
        qend=interval.qend,
        is_reverse=(entry.strand == "-"),
    )
    if not (out.ref_end > out.ref_start and out.qend > out.qstart):
        return None
    return out


@dataclass
class SAEntryWithQuality(SAEntry):
    """An SA record carrying the aligner's own quality fields.

    The fragment extractor needs `mapq` and `nm` to choose between competing
    flank candidates, so it requires all SIX fields and refuses a record with
    fewer -- where the clustering parser accepts four, because it only needs to
    know where the segment went.
    """

    mapq: int = 0
    nm: int = 0


def parse_sa_tag_z_with_quality(sa_z: str) -> list[SAEntryWithQuality]:
    """Parse `SA:Z:` keeping mapq and NM.

    NOTE the two differences from :func:`parse_sa_tag_z`, both faithful to the
    C++: six fields are REQUIRED, and the only validity test is a non-empty
    CIGAR -- a record with `pos = 0` survives here and is clamped later by
    `max(0, pos - 1)`, where the other parser would have dropped it.
    """
    out: list[SAEntryWithQuality] = []
    if not sa_z:
        return out
    start = 0
    while start < len(sa_z):
        rec_end = sa_z.find(";", start)
        rec = sa_z[start:] if rec_end == -1 else sa_z[start:rec_end]
        if not rec:
            break
        fields = rec.split(",")
        if len(fields) >= 6:
            pos = parse_int32(fields[1])
            mapq = parse_int32(fields[4])
            nm = parse_int32(fields[5])
            entry = SAEntryWithQuality(
                rname=fields[0],
                pos=pos if pos is not None else 0,
                strand=fields[2][0] if fields[2] else "+",
                cigar=fields[3],
                mapq=mapq if mapq is not None else 0,
                nm=nm if nm is not None else 0,
            )
            if entry.cigar:
                out.append(entry)
        if rec_end == -1:
            break
        start = rec_end + 1
    return out
