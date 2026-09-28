"""Reference sequence by interval, for segmentation and the TSD detector.

Split out of the BAM reader it used to share a module with: a FASTA fetcher is
not BAM reading, and the algorithm's entire dependency on the reference is this
one callable. `placer/core/segmentation.py` and `placer/core/tsd.py` both
take `fetch_window` as a parameter for exactly that reason, so keeping the
implementation in its own file is what makes the seam visible from the outside.

Ported from `include/bam_io.h` and `src/stream/bam_io.cpp`.

`pysam` IS IMPORTED LAZILY, as everywhere in this package: the decision layer
needs no reference at all, and making a compiled dependency a hard import would
put it in front of the half of the package that does not use it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from placer.io import perf

if TYPE_CHECKING:  # pragma: no cover - for the checker, never at runtime
    import pysam

#: Reference blocks held per process: 64 kb each, 64 of them (4 Mb). A bin's
#: windows -- segmentation's, the TSD detector's, and the hundred decoys'
#: around each locus -- fall inside one or two blocks. Small enough that the
#: few thousand scattered single-base anchor fetches at the end of a run cost
#: 64 kb each, not a megabase.
REFERENCE_BLOCK_BITS = 16
REFERENCE_CACHE_BLOCKS = 64


class ReferenceFetcher:
    """Reference sequence by interval, for the TSD detector and segmentation.

    Uppercased on the way out, because every consumer compares against
    uppercased read bases and a reference with soft-masked (lower-case) repeats
    would otherwise mismatch at exactly the loci this caller is about.

    SERVED FROM CACHED BLOCKS. An evaluated locus asks for on the order of 10^4
    small windows (the TSD detector's two per candidate length, for the locus
    and each of its decoys), all within two kilobases of it. Each used to be a
    faidx read plus an `upper()`; on BeeGFS without a page cache the reads are
    system time. A window inside the contig and across at most two blocks is
    now a slice of uppercased blocks, which is the same string: faidx
    truncates at the contig end, and so does a slice. Everything else -- an
    unknown contig, a start at or past the end, a window wider than two blocks,
    a block that is not ASCII (where `upper()` could change a length) -- takes
    the direct fetch, exactly as before (`tests/test_50_reference_cache.py`).
    """

    def __init__(self, fasta_path: str) -> None:
        import pysam  # noqa: F401

        # Optional for the same reason as the BAM handles: `close()` sets
        # it to None and `fetch_window` already checks for that.
        self._fasta: pysam.FastaFile | None = pysam.FastaFile(fasta_path)
        self._lengths = dict(zip(self._fasta.references, self._fasta.lengths))
        #: (chrom, block index) -> uppercased block, or None for a block the
        #: cache will not serve (not ASCII).
        self._blocks: OrderedDict[tuple[str, int], str | None] = OrderedDict()
        #: `placer/io/perf.py` tallies, kept here: a `count()` call per window
        #: would cost a noticeable share of the window itself.
        self._calls = 0
        self._block_loads = 0
        perf.register(self._perf_counts)

    def _perf_counts(self) -> dict[str, int]:
        return {"ref_fetch_calls": self._calls, "ref_block_loads": self._block_loads}

    def can_fetch_reference(self) -> bool:
        return self._fasta is not None

    def fetch_window(self, chrom: str, start: int, end: int) -> str:
        if self._fasta is None or not chrom or end <= start:
            return ""
        start = max(0, start)
        self._calls += 1
        length = self._lengths.get(chrom)
        if length is not None and start < length:
            end = min(end, length)
            first = start >> REFERENCE_BLOCK_BITS
            last = (end - 1) >> REFERENCE_BLOCK_BITS
            if last - first <= 1:
                head = self._block(chrom, first)
                tail = head if last == first else self._block(chrom, last)
                if head is not None and tail is not None:
                    offset = first << REFERENCE_BLOCK_BITS
                    joined = head if last == first else head + tail
                    return joined[start - offset:end - offset]
        return self._fetch_direct(chrom, start, end)

    def _block(self, chrom: str, index: int) -> str | None:
        key = (chrom, index)
        if key in self._blocks:
            self._blocks.move_to_end(key)
            return self._blocks[key]
        begin = index << REFERENCE_BLOCK_BITS
        self._block_loads += 1
        block: str | None = self._fetch_direct(
            chrom, begin, min(self._lengths[chrom], begin + (1 << REFERENCE_BLOCK_BITS)))
        if not block or not block.isascii():
            block = None
        self._blocks[key] = block
        if len(self._blocks) > REFERENCE_CACHE_BLOCKS:
            self._blocks.popitem(last=False)
        return block

    def _fetch_direct(self, chrom: str, start: int, end: int) -> str:
        assert self._fasta is not None
        try:
            return self._fasta.fetch(chrom, start, end).upper()
        except (KeyError, ValueError):
            # An unknown contig or an out-of-range interval is a MISSING window,
            # not an error: the caller's own `if not window` branch reports it
            # as REFERENCE_WINDOW_FETCH_FAILED, which is the right verdict.
            return ""

    def close(self) -> None:
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
        self._blocks.clear()
