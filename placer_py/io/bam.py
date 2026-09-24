"""
Reading the BAM: a streaming reader, an indexed fetch, and a region scope.

Ported from `include/bam_io.h`, `src/stream/bam_io.cpp` and
`src/denovo/indexed_bam_reader.cpp`, pinned by
`tests/test_31_outputs.py`. The reference fetcher that used to share this
module is now `placer_py/io/reference.py`.

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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - for the checker, never at runtime
    # Named so the handle annotations below can be real types. Importing
    # it for real would defeat the lazy imports in the two constructors,
    # which exist so the decision layer needs no compiled dependency.
    import pysam

from placer_py.alignment import AlignedRead
from placer_py.config import BamRegionScope
from placer_py.io.pysam_adapter import read_from_pysam

#: One definition, in `reads.py`, which `alignment.py` also re-exports.
from placer_py.reads import FLAG_SECONDARY, FLAG_UNMAP  # noqa: F401


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
            yield read_from_pysam(record)
            if progress is not None and progress_interval > 0 and (
                    processed - last_progress) >= progress_interval:
                last_progress = processed
                if not progress(processed, int(record.reference_id)):
                    return

    def fetch(self, chrom: str, start: int, end: int) -> list[AlignedRead]:
        """Records overlapping one interval, or [] when there is no index.

        Returning an empty list rather than raising matches the C++'s `can_fetch`
        contract: the caller is expected to have checked, and a run on an
        unindexed BAM should degrade to the scan-only stages rather than abort.
        """
        if self._fetch is None:
            return []
        return [read_from_pysam(record)
                for record in self._fetch.fetch(chrom, max(0, start), end)
                if self._keep(record)]

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
