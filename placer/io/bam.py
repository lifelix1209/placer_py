"""
Reading the BAM: a streaming reader, an indexed fetch, and a region scope.

Ported from `include/bam_io.h`, `src/stream/bam_io.cpp` and
`src/denovo/indexed_bam_reader.cpp`, pinned by
`tests/test_31_outputs.py`. The reference fetcher that used to share this
module is now `placer/io/reference.py`.

TWO ACCESS PATTERNS, and the pipeline needs both. The scan STREAMS the whole
file once in order, which is the only affordable way to touch six million reads.
The local stages then FETCH small intervals around candidates, which needs an
index. The C++ opens the file twice for exactly this reason -- one handle
positioned by the stream, one by the fetch -- and the Python reader does the
same through pysam.

THE FILTER IS APPLIED AT THE SOURCE. Secondary and unmapped records never reach
any handler. That is not an optimisation: a secondary alignment is the same read
placed somewhere else, and letting one through would let a single read support
two loci. Supplementary records DO pass here and are filtered per stage, because
the fragment extractor genuinely needs them.

`pysam` IS IMPORTED LAZILY. The decision layer needs no BAM at all, and making a
compiled dependency a hard import would put it in front of the half of the
package that does not use it.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - for the checker, never at runtime
    # Named so the handle annotations below can be real types. Importing
    # it for real would defeat the lazy imports in the two constructors,
    # which exist so the decision layer needs no compiled dependency.
    import pysam

from placer.alignment import AlignedRead
from placer.config import BamRegionScope
from placer.io.perf import count
from placer.io.pysam_adapter import read_from_pysam

#: One definition, in `reads.py`, which `alignment.py` also re-exports.
from placer.reads import FLAG_SECONDARY, FLAG_UNMAP  # noqa: F401


def normalize_region_scope(scope: BamRegionScope) -> BamRegionScope:
    """Make a region scope self-consistent, or disable it.

    An enabled scope with no contig is DISABLED rather than raising -- the
    caller asked for "everything on nothing", which is most likely a default
    left set, and refusing the run for it would be unhelpful. A negative start
    clamps to 0, and an end at or before the start becomes `start + 1` so the
    interval is never empty.
    """
    if not scope.enabled:
        return scope
    if not scope.chrom:
        scope.enabled = False
        return scope
    scope.start = max(0, scope.start)
    if scope.end > 0 and scope.end <= scope.start:
        scope.end = scope.start + 1
    return scope


class RecordCache:
    """The `AlignedRead` already built for a record, if it is still recent.

    WHY. The bin loop re-fetches the reads around every candidate, and on
    ultra-long ONT the same read turns up in the fetches of many neighbouring
    components and bins -- about five conversions per read on a real HG002
    slice. Each conversion decodes the whole sequence and CIGAR, and each new
    object starts with an empty `cigar_index` and memo, so every derived fact
    was being recomputed as often as the read was fetched. Handing back the
    SAME object for the same record keeps those.

    SAFE because `AlignedRead` is treated as a value everywhere (nothing
    mutates one, nothing compares reads by identity), so two lists holding one
    object behave exactly as two lists holding equal copies.

    THE KEY is every cheap field that distinguishes two records of one read:
    name, flag, contig, both alignment ends, mapping quality and length. A
    primary and its supplementaries differ in flag or position; secondary
    records never reach here. BOUNDED BY BASES, not entries, because a read's
    cost is its sequence -- one ultra-long read outweighs a hundred short ones.
    """

    def __init__(self, max_bases: int = 32_000_000) -> None:
        self.max_bases = max_bases
        self._reads: OrderedDict[tuple, AlignedRead] = OrderedDict()
        self._bases = 0

    def convert(self, record) -> AlignedRead:
        if self.max_bases <= 0:
            count("bam_conversions")
            return read_from_pysam(record)
        key = (record.query_name, record.flag, record.reference_id,
               record.reference_start, record.reference_end,
               record.mapping_quality, record.query_length)
        read = self._reads.get(key)
        if read is not None:
            self._reads.move_to_end(key)
            return read
        count("bam_conversions")
        read = read_from_pysam(record)
        self._reads[key] = read
        self._bases += len(read.seq)
        while self._bases > self.max_bases and len(self._reads) > 1:
            _, dropped = self._reads.popitem(last=False)
            self._bases -= len(dropped.seq)
        return read


#: How far ahead of the read being handed to the scan the stream is read, so
#: that a bin's local fetches -- its components' seeds +- 1 kb, which reach
#: past the bin's end -- find every read they need already streamed.
STREAM_LOOKAHEAD_BP = 50_000
#: How far below a local fetch's start the buffer keeps reads for the next one.
STREAM_RETAIN_BP = 50_000
#: How far fetches move on before the buffer is pruned again.
STREAM_PRUNE_STEP_BP = 10_000
#: Ceiling on the bases held, whatever the positions say. A collapsed region
#: can be thousands of reads deep, and each held read keeps its CIGAR and tags
#: as Python objects, several times its bases; at 400 Mbases 1q21 peaked 1.6 GB
#: above 1.0.0a1. Past the ceiling the earliest-ending reads are dropped and a
#: fetch that needs them goes to the index, so the answer is the same.
STREAM_BUFFER_MAX_BASES = 64_000_000
#: `PLACER_VERIFY_LOCAL_FETCH=1`: answer every local fetch both ways and fail
#: the run on any difference. A check for the buffer, never for production.
VERIFY_LOCAL_FETCH_ENV = "PLACER_VERIFY_LOCAL_FETCH"

_UNBOUNDED = sys.maxsize


@dataclass
class _ContigReads:
    """The streamed reads of one contig still held, in file order."""

    name: str
    #: (pos, htslib end, read), in file order: nondecreasing pos.
    reads: list[tuple[int, int, AlignedRead]] = field(default_factory=list)
    #: Every read with pos < this has been streamed; unbounded once the
    #: stream has left the contig.
    frontier: int = -1
    #: No read ending after this has been dropped; a fetch starting below it
    #: may be missing some.
    floor: int = 0
    #: The last limit the reads were pruned to; pruning again is only worth
    #: a pass once fetches have moved well past it.
    pruned_to: int = 0
    bases: int = 0


class StreamedReadBuffer:
    """Local fetches answered from the reads the scan's own stream has read.

    WHY. Every bin re-fetches the reads around its components from the index.
    On ultra-long ONT an indexed fetch has to inflate every record from the
    start of the longest read overlapping the interval -- often hundreds of kb
    upstream -- and neighbouring bins inflate the same records again and
    again; on BeeGFS, which keeps no page cache, each of those reads goes back
    over the network as system time. The stream has already read every one of
    those records once, in the same order.

    WHEN IT ANSWERS, which is what makes the answer the same list the index
    would give. A fetch of `[s, e)` on the fetch handle returns every kept
    (not secondary, not unmapped) record with `pos < e` and `end > s`, where
    `end` is htslib's `bam_endpos`, in file order. The buffer holds the same
    records, filtered the same way, in the same order, and answers only when
    it provably holds all of them:

      * the stream covers the interval: same contig, and `[s, e)` inside the
        streamed interval, so every read overlapping it was in the stream;
      * the stream has passed `e` (`frontier`), so every read with `pos < e`
        has been read -- which is why the stream is read `STREAM_LOOKAHEAD_BP`
        ahead of the scan;
      * nothing that ends after `s` has been dropped (`floor`).

    Anything else -- another contig, an interval reaching outside a chunk,
    `e <= s`, a reader with no index -- goes to the index, exactly as before.
    `PLACER_VERIFY_LOCAL_FETCH=1` asks both and compares every answer
    (`tests/test_51_streamed_reads.py`).
    """

    def __init__(self, reader: BamStreamReader, cover_chrom: str | None = None,
                 cover_start: int = 0, cover_end: int | None = None,
                 verify: bool | None = None) -> None:
        self._reader = reader
        #: None: the whole file, every contig in full.
        self._cover_chrom = cover_chrom
        self._cover_start = max(0, cover_start)
        self._cover_end = _UNBOUNDED if cover_end is None else cover_end
        self._contigs: OrderedDict[str, _ContigReads] = OrderedDict()
        self._current: _ContigReads | None = None
        self._names: dict[int, str] = {}
        self._bases = 0
        self.verify = (os.environ.get(VERIFY_LOCAL_FETCH_ENV, "") == "1"
                       if verify is None else verify)

    # ------------------------------------------------------------- the stream
    def tap(self, records: Iterable[tuple[AlignedRead, int]]) -> Iterator[AlignedRead]:
        """The reads, in order, while holding them for `fetch`.

        `records` pairs each kept read with its htslib end. The same reads come
        out in the same order; only how far ahead of the consumer the file has
        been read changes.
        """
        source = iter(records)
        pending: deque[AlignedRead] = deque()
        exhausted = False
        while True:
            if not pending:
                if exhausted or not self._pull(source, pending):
                    return
                continue
            head = pending[0]
            limit = head.pos + STREAM_LOOKAHEAD_BP
            while not exhausted:
                current = self._current
                if (current is None or current.name != self._name(head.tid)
                        or current.frontier >= limit):
                    break
                exhausted = not self._pull(source, pending)
            yield pending.popleft()

    def _name(self, tid: int) -> str:
        name = self._names.get(tid)
        if name is None:
            name = self._names[tid] = self._reader.chromosome_name(tid)
        return name

    def _pull(self, source: Iterator[tuple[AlignedRead, int]],
              pending: deque[AlignedRead]) -> bool:
        try:
            read, end = next(source)
        except StopIteration:
            if self._current is not None:
                self._current.frontier = _UNBOUNDED
            return False
        name = self._name(read.tid)
        current = self._current
        if current is None or current.name != name:
            if current is not None:
                current.frontier = _UNBOUNDED
            current = self._current = _ContigReads(name)
            self._contigs[name] = current
            # The previous contig's last bins are still to be scanned (the
            # stream runs ahead); anything older is done.
            while len(self._contigs) > 2:
                _, dropped = self._contigs.popitem(last=False)
                self._bases -= dropped.bases
        current.reads.append((read.pos, end, read))
        current.frontier = read.pos
        size = len(read.seq)
        current.bases += size
        self._bases += size
        if self._bases > STREAM_BUFFER_MAX_BASES:
            self._shrink()
        pending.append(read)
        return True

    def _drop_ending_by(self, state: _ContigReads, limit: int) -> None:
        """Drop the reads of `state` that end at or before `limit`."""
        kept = []
        for item in state.reads:
            if item[1] <= limit:
                state.floor = max(state.floor, item[1])
                size = len(item[2].seq)
                state.bases -= size
                self._bases -= size
            else:
                kept.append(item)
        state.reads = kept

    def _shrink(self) -> None:
        """Over the ceiling: drop other contigs, then the earliest-ending reads
        of this one, down to three quarters of it."""
        for name in [name for name in self._contigs if self._contigs[name] is not self._current]:
            self._bases -= self._contigs.pop(name).bases
        state = self._current
        if state is None or self._bases <= STREAM_BUFFER_MAX_BASES:
            return
        target = self._bases - STREAM_BUFFER_MAX_BASES * 3 // 4
        freed = 0
        limit = state.floor
        for end, size in sorted((item[1], len(item[2].seq)) for item in state.reads):
            if freed >= target:
                break
            freed += size
            limit = end
        self._drop_ending_by(state, limit)

    # ------------------------------------------------------------- the fetch
    def fetch(self, chrom: str, start: int, end: int) -> list[AlignedRead]:
        """`reader.fetch(chrom, start, end)`, from the buffer when it can be."""
        begin = max(0, start)
        state = self._contigs.get(chrom)
        if (state is None or end <= begin or begin < state.floor
                or end > state.frontier or not self._covers(chrom, begin, end)):
            count("bam_buffer_fallbacks")
            return self._reader.fetch(chrom, start, end)
        if begin - STREAM_RETAIN_BP > state.pruned_to + STREAM_PRUNE_STEP_BP:
            state.pruned_to = begin - STREAM_RETAIN_BP
            self._drop_ending_by(state, state.pruned_to)
        reads = []
        for pos, read_end, read in state.reads:
            if pos >= end:
                break
            if read_end > begin:
                reads.append(read)
        count("bam_buffer_served")
        count("bam_buffer_records", len(reads))
        if self.verify:
            expected = self._reader.fetch(chrom, start, end)
            if expected != reads:
                raise AssertionError(
                    f"streamed-read buffer differs from the index at {chrom}:{start}-{end}: "
                    f"{[r.qname for r in reads]} != {[r.qname for r in expected]}")
        return reads

    def _covers(self, chrom: str, begin: int, end: int) -> bool:
        if self._cover_chrom is None:
            return True
        return (chrom == self._cover_chrom and begin >= self._cover_start
                and end <= self._cover_end)


@dataclass
class BamReadStats:
    total: int = 0
    skipped_secondary: int = 0
    skipped_unmapped: int = 0


class BamStreamReader:
    """A pysam-backed reader with the `BamStreamReader` surface.

    Holds TWO handles when the file is indexed -- see the module docstring --
    and falls back to a stream-only reader when it is not, which is the correct
    behaviour for a name-sorted or streamed input: the scan still works, and the
    local stages that need `fetch` are told they cannot have it rather than
    getting silently wrong answers.
    """

    def __init__(self, bam_path: str, decompression_threads: int = 2,
                 region_scope: BamRegionScope | None = None) -> None:
        import pysam  # noqa: F401  (lazy: the decision layer needs no BAM)

        self.bam_path = bam_path
        self.region_scope = normalize_region_scope(region_scope or BamRegionScope())
        self.stats = BamReadStats()
        self.records = RecordCache()
        self._pysam = pysam
        # Optional because `close()` sets both to None, which is what
        # `is_valid()` and `can_fetch()` are reading. Saying so is the
        # point: without it the annotation claims a handle that is always
        # open, and every method below reads it as if that were true.
        self._stream: pysam.AlignmentFile | None = pysam.AlignmentFile(
            bam_path, "rb", threads=max(1, decompression_threads))
        self._fetch: pysam.AlignmentFile | None = None
        try:
            if self._stream.has_index():
                self._fetch = pysam.AlignmentFile(bam_path, "rb",
                                                  threads=max(1, decompression_threads))
        except (ValueError, OSError):
            self._fetch = None

    def _open_stream(self) -> pysam.AlignmentFile:
        """The stream handle, or a named error instead of an anonymous one.

        Reaching any of the four methods below after `close()` used to raise
        `AttributeError: 'NoneType' object has no attribute 'nreferences'`,
        which says nothing about what the caller did wrong.
        """
        if self._stream is None:
            raise ValueError("BamStreamReader used after close()")
        return self._stream

    # ------------------------------------------------------------- metadata
    def is_valid(self) -> bool:
        return self._stream is not None

    def chromosome_count(self) -> int:
        return self._open_stream().nreferences

    def chromosome_name(self, tid: int) -> str:
        stream = self._open_stream()
        if tid < 0 or tid >= stream.nreferences:
            return ""
        return stream.get_reference_name(tid)

    def chromosome_length(self, tid: int) -> int:
        stream = self._open_stream()
        if tid < 0 or tid >= stream.nreferences:
            return 0
        return stream.lengths[tid]

    def header_dict(self) -> dict:
        """The BAM header as plain nested dicts and lists.

        Only `@RG SM` is read from it today, for the VCF sample column. Typed
        as a plain dict rather than a pysam type so nothing downstream of the
        input stage has to know pysam exists.
        """
        return dict(self._open_stream().header.to_dict())

    def can_fetch(self) -> bool:
        return self._fetch is not None

    # ------------------------------------------------------------- reading
    def _keep(self, record) -> bool:
        if record.flag & FLAG_SECONDARY:
            self.stats.skipped_secondary += 1
            return False
        if record.flag & FLAG_UNMAP:
            self.stats.skipped_unmapped += 1
            return False
        return True

    def stream(self, progress: Callable[[int, int], bool] | None = None,
               progress_interval: int = 100000) -> Iterator[AlignedRead]:
        """Every usable record, in file order, as `AlignedRead`s.

        A generator rather than a callback: the C++ has to pass a handler
        because it owns the record memory, and Python does not. The progress
        callback is kept because it can ABORT the scan by returning False, which
        a generator's consumer cannot do from outside.
        """
        stream = self._open_stream()
        source = (stream.fetch(self.region_scope.chrom,
                               self.region_scope.start,
                               self.region_scope.end if self.region_scope.end > 0 else None)
                  if (self.region_scope.enabled and self.can_fetch())
                  else stream.fetch(until_eof=True))
        processed = 0
        last_progress = 0
        for record in source:
            if not self._keep(record):
                continue
            self.stats.total += 1
            processed += 1
            count("bam_stream_records")
            yield self.records.convert(record)
            if progress is not None and progress_interval > 0 and (
                    processed - last_progress) >= progress_interval:
                last_progress = processed
                if not progress(processed, int(record.reference_id)):
                    return

    def stream_interval(self, chrom: str, start: int,
                        end: int | None) -> Iterator[AlignedRead]:
        """`stream()` restricted to one interval, on the STREAM handle.

        What a parallel worker reads its chunk with: the same filter and the
        same tally as `stream()`, and a generator for the same reason -- a
        chunk of ultra-long reads is far too large to hold as a list. It uses
        the stream handle, not the fetch handle, so the local fetches the bin
        loop makes while this is being consumed cannot move its position.
        """
        stream = self._open_stream()
        for record in stream.fetch(chrom, max(0, start), end):
            if not self._keep(record):
                continue
            self.stats.total += 1
            count("bam_stream_records")
            yield self.records.convert(record)

    def _kept_with_ends(self, source) -> Iterator[tuple[AlignedRead, int]]:
        """Kept records as reads, each with htslib's `bam_endpos` -- the end an
        indexed fetch tests overlap against (pysam's `reference_end`, or
        pos + 1 for a record with no CIGAR)."""
        for record in source:
            if not self._keep(record):
                continue
            self.stats.total += 1
            count("bam_stream_records")
            end = record.reference_end
            yield (self.records.convert(record),
                   end if end is not None else record.reference_start + 1)

    def buffered_stream(self) -> tuple[Iterator[AlignedRead],
                                       Callable[[str, int, int], list[AlignedRead]]]:
        """`stream()` and a `fetch` that answers from it (`StreamedReadBuffer`).

        With no index there is nothing to answer for -- `fetch` returns [] --
        so the plain stream and fetch are returned unchanged.
        """
        if not self.can_fetch():
            return self.stream(), self.fetch
        stream = self._open_stream()
        scope = self.region_scope
        if scope.enabled:
            end = scope.end if scope.end > 0 else None
            buffer = StreamedReadBuffer(self, scope.chrom, scope.start, end)
            source = stream.fetch(scope.chrom, scope.start, end)
        else:
            buffer = StreamedReadBuffer(self)
            source = stream.fetch(until_eof=True)
        return buffer.tap(self._kept_with_ends(source)), buffer.fetch

    def buffered_stream_interval(self, chrom: str, start: int, end: int | None
                                 ) -> tuple[Iterator[AlignedRead],
                                            Callable[[str, int, int], list[AlignedRead]]]:
        """`stream_interval()` and a `fetch` that answers from it."""
        if not self.can_fetch():
            return self.stream_interval(chrom, start, end), self.fetch
        buffer = StreamedReadBuffer(self, chrom, start, end)
        source = self._open_stream().fetch(chrom, max(0, start), end)
        return buffer.tap(self._kept_with_ends(source)), buffer.fetch

    def fetch(self, chrom: str, start: int, end: int) -> list[AlignedRead]:
        """Records overlapping one interval, or [] when there is no index.

        Returning an empty list rather than raising matches the C++'s `can_fetch`
        contract: the caller is expected to have checked, and a run on an
        unindexed BAM should degrade to the scan-only stages rather than abort.
        """
        if self._fetch is None:
            return []
        records = [self.records.convert(record)
                   for record in self._fetch.fetch(chrom, max(0, start), end)
                   if self._keep(record)]
        count("bam_fetch_calls")
        count("bam_fetch_records", len(records))
        return records

    def compressed_offset_at(self, chrom: str, start: int,
                             end: int | None = None) -> int | None:
        """Where in the compressed file the reads of `chrom:[start, end)`
        begin, as a byte offset: just after the first record overlapping it
        (`end` defaults to `start + 1`), or None without an index or a record.
        Differences between two intervals' offsets estimate how much alignment
        data lies between them, for scheduling only."""
        if self._fetch is None:
            return None
        start = max(0, start)
        stop = start + 1 if end is None else max(start + 1, end)
        try:
            record = next(iter(self._fetch.fetch(chrom, start, stop)), None)
        except (ValueError, OSError):
            return None
        if record is None:
            return None
        return self._fetch.tell() >> 16

    def close(self) -> None:
        for handle in (self._stream, self._fetch):
            if handle is not None:
                handle.close()
        self._stream = None
        self._fetch = None

    def __enter__(self) -> BamStreamReader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def make_bam_reader(bam_path: str, decompression_threads: int = 2,
                    region_scope: BamRegionScope | None = None) -> BamStreamReader:
    return BamStreamReader(bam_path, decompression_threads, region_scope)
