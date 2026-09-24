"""Reference sequence by interval, for segmentation and the TSD detector.

Split out of the BAM reader it used to share a module with: a FASTA fetcher is
not BAM reading, and the algorithm's entire dependency on the reference is this
one callable. `placer_py/core/segmentation.py` and `placer_py/core/tsd.py` both
take `fetch_window` as a parameter for exactly that reason, so keeping the
implementation in its own file is what makes the seam visible from the outside.

Ported from `include/bam_io.h` and `src/stream/bam_io.cpp`.

`pysam` IS IMPORTED LAZILY, as everywhere in this package: the decision layer
needs no reference at all, and making a compiled dependency a hard import would
put it in front of the half of the package that does not use it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for the checker, never at runtime
    import pysam


class ReferenceFetcher:
    """Reference sequence by interval, for the TSD detector and segmentation.

    Uppercased on the way out, because every consumer compares against
    uppercased read bases and a reference with soft-masked (lower-case) repeats
    would otherwise mismatch at exactly the loci this caller is about.
    """

    def __init__(self, fasta_path: str) -> None:
        import pysam  # noqa: F401

        # Optional for the same reason as the BAM handles: `close()` sets
        # it to None and `fetch_window` already checks for that.
        self._fasta: pysam.FastaFile | None = pysam.FastaFile(fasta_path)

    def can_fetch_reference(self) -> bool:
        return self._fasta is not None

    def fetch_window(self, chrom: str, start: int, end: int) -> str:
        if self._fasta is None or not chrom or end <= start:
            return ""
        try:
            return self._fasta.fetch(chrom, max(0, start), end).upper()
        except (KeyError, ValueError):
            # An unknown contig or an out-of-range interval is a MISSING window,
            # not an error: the caller's own `if not window` branch reports it
            # as REFERENCE_WINDOW_FETCH_FAILED, which is the right verdict.
            return ""

    def close(self) -> None:
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
