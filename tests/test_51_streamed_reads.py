"""
Local fetches answered from the stream return exactly what the index returns.

`placer/io/bam.StreamedReadBuffer` answers the bin loop's local fetches from
the reads its own stream has already read, and only when it provably holds
every read the indexed fetch would return. Every downstream stage reads those
lists, so any difference -- a missing long read, a read past the frontier, an
order change -- would change calls. This drives the buffer the way the scan
does (consume the stream, fetch around where it is) with the lookahead,
retention and memory ceiling shrunk so that every path runs: served, fallback
below the floor, fallback past the frontier, outside a chunk, and pruning.
Each answer is compared with the index's.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "data"


def _needs_example():
    if importlib.util.find_spec("pysam") is None:
        pytest.skip("needs pysam")
    if not (EXAMPLE / "mini.bam").is_file():
        pytest.skip("examples/data is missing (examples/make_example_data.py builds it)")


def _small_buffer(bam_module):
    """Shrink the buffer's constants; returns a restore callback."""
    names = ("STREAM_LOOKAHEAD_BP", "STREAM_RETAIN_BP", "STREAM_PRUNE_STEP_BP",
             "STREAM_BUFFER_MAX_BASES")
    saved = {name: getattr(bam_module, name) for name in names}
    bam_module.STREAM_LOOKAHEAD_BP = 3_000
    bam_module.STREAM_RETAIN_BP = 2_000
    bam_module.STREAM_PRUNE_STEP_BP = 500
    bam_module.STREAM_BUFFER_MAX_BASES = 150_000

    def restore():
        for name, value in saved.items():
            setattr(bam_module, name, value)
    return restore


def _outcome(fetch, chrom: str, start: int, end: int):
    """The reads, or the exception type: an interval the index refuses
    (an end below 0) must be refused the same way."""
    try:
        return fetch(chrom, start, end)
    except ValueError as error:
        return type(error)


def _drive(stream, fetch, direct_fetch, chrom: str, seed: int) -> int:
    """Consume `stream`; around each read, fetch like a bin would, and compare.
    Returns the number of reads consumed."""
    rng = random.Random(seed)
    consumed = 0
    for read in stream:
        consumed += 1
        for _ in range(3):
            start = read.pos + rng.randint(-6_000, 1_500)
            end = start + rng.choice((1, 50, 800, 2_500, 4_000))
            assert (_outcome(fetch, chrom, start, end)
                    == _outcome(direct_fetch, chrom, start, end)), (chrom, start, end)
        # And the odd interval the buffer must refuse: far behind, far ahead,
        # empty or reversed, another contig.
        for start, end in ((read.pos - 30_000, read.pos - 29_000),
                           (read.pos + 20_000, read.pos + 21_000),
                           (read.pos, read.pos), (read.pos + 5, read.pos)):
            assert (_outcome(fetch, chrom, start, end)
                    == _outcome(direct_fetch, chrom, start, end)), (chrom, start, end)
        assert _outcome(fetch, "chrZ", 0, 100) == _outcome(direct_fetch, "chrZ", 0, 100)
    return consumed


def _counts():
    from placer.io import perf
    return {name: perf.COUNTS[name] for name in
            ("bam_buffer_served", "bam_buffer_fallbacks")}


def test_the_tapped_stream_is_the_plain_stream():
    _needs_example()
    from placer.io import bam as bam_module
    from placer.io.bam import make_bam_reader

    restore = _small_buffer(bam_module)
    try:
        plain = make_bam_reader(str(EXAMPLE / "mini.bam"), 1)
        tapped = make_bam_reader(str(EXAMPLE / "mini.bam"), 1)
        stream, _ = tapped.buffered_stream()
        assert list(stream) == list(plain.stream())
        assert tapped.stats.total == plain.stats.total
    finally:
        restore()


def test_whole_file_fetches_from_the_buffer_equal_the_index():
    _needs_example()
    from placer.io import bam as bam_module
    from placer.io.bam import make_bam_reader

    restore = _small_buffer(bam_module)
    try:
        reader = make_bam_reader(str(EXAMPLE / "mini.bam"), 1)
        direct = make_bam_reader(str(EXAMPLE / "mini.bam"), 1)
        before = _counts()
        stream, fetch = reader.buffered_stream()
        assert _drive(stream, fetch, direct.fetch, "chr1", 3) > 100
        after = _counts()
        # Both paths ran, so the comparison above covered both.
        assert after["bam_buffer_served"] > before["bam_buffer_served"]
        assert after["bam_buffer_fallbacks"] > before["bam_buffer_fallbacks"]
    finally:
        restore()


def test_region_and_chunk_fetches_from_the_buffer_equal_the_index():
    _needs_example()
    from placer.config import BamRegionScope
    from placer.io import bam as bam_module
    from placer.io.bam import make_bam_reader

    restore = _small_buffer(bam_module)
    try:
        direct = make_bam_reader(str(EXAMPLE / "mini.bam"), 1)
        scope = BamRegionScope(enabled=True, chrom="chr1", start=20_000, end=70_000)
        region = make_bam_reader(str(EXAMPLE / "mini.bam"), 1, scope)
        stream, fetch = region.buffered_stream()
        assert _drive(stream, fetch, direct.fetch, "chr1", 5) > 0

        chunked = make_bam_reader(str(EXAMPLE / "mini.bam"), 1)
        stream, fetch = chunked.buffered_stream_interval("chr1", 40_000, 90_000)
        assert _drive(stream, fetch, direct.fetch, "chr1", 7) > 0
    finally:
        restore()
