"""
The reference block cache returns exactly what the direct fetch returned.

`ReferenceFetcher.fetch_window` serves windows from cached, uppercased 64 kb
blocks (`placer/io/reference.py`). Every consumer compares these strings base
for base, so a cache that differed anywhere -- at a block boundary, at a
contig end, for a start before 0 or past the end, for an unknown contig --
would change calls. This compares the cache against the pre-cache
implementation, written out below, on the edges and on random windows, with
the cache small enough to evict.
"""

from __future__ import annotations

import importlib.util
import os
import random
import tempfile

import pytest

BLOCK = 1 << 16
CONTIGS = {"chrA": 3 * BLOCK + 17, "chrB": BLOCK, "chrC": 10}


def _direct(fasta, chrom: str, start: int, end: int) -> str:
    """`fetch_window` as it was before the cache."""
    if not chrom or end <= start:
        return ""
    try:
        return fasta.fetch(chrom, max(0, start), end).upper()
    except (KeyError, ValueError):
        return ""


def _write_fasta(directory: str) -> str:
    rng = random.Random(11)
    path = os.path.join(directory, "ref.fa")
    with open(path, "w") as out:
        for name, length in CONTIGS.items():
            # Soft-masked stretches and Ns, as a real reference has.
            seq = "".join(rng.choice("ACGTacgtN") for _ in range(length))
            out.write(f">{name}\n")
            for offset in range(0, length, 60):
                out.write(seq[offset:offset + 60] + "\n")
    import pysam
    pysam.faidx(path)
    return path


def _edges() -> list[tuple[str, int, int]]:
    cases = [("", 0, 10), ("chrZ", 0, 10), ("chrA", 10, 10), ("chrA", 10, 5),
             ("chrA", -50, 20), ("chrA", -50, -10)]
    for name, length in CONTIGS.items():
        for point in (0, 1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK, length - 1,
                      length, length + 1):
            for width in (1, 2, 7, 100, BLOCK, BLOCK + 1, 2 * BLOCK + 3):
                cases.append((name, point - width // 2, point - width // 2 + width))
                cases.append((name, point, point + width))
                cases.append((name, point - width, point))
    return cases


def _random(n: int) -> list[tuple[str, int, int]]:
    rng = random.Random(5)
    cases = []
    for _ in range(n):
        name = rng.choice(list(CONTIGS))
        length = CONTIGS[name]
        start = rng.randint(-100, length + 100)
        cases.append((name, start, start + rng.choice((1, 3, 50, 400, 5000, BLOCK))))
    return cases


def test_cached_windows_equal_direct_fetches_on_every_edge_and_at_random():
    if importlib.util.find_spec("pysam") is None:
        pytest.skip("needs pysam")
    import pysam

    from placer.io import reference as reference_module
    from placer.io.reference import ReferenceFetcher

    saved = reference_module.REFERENCE_CACHE_BLOCKS
    reference_module.REFERENCE_CACHE_BLOCKS = 2  # evict constantly
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_fasta(directory)
            fetcher = ReferenceFetcher(path)
            fasta = pysam.FastaFile(path)
            cases = _edges() + _random(3000)
            for chrom, start, end in cases:
                expected = _direct(fasta, chrom, start, end)
                got = fetcher.fetch_window(chrom, start, end)
                assert got == expected, (chrom, start, end)
            assert len(fetcher._blocks) <= 2
            fetcher.close()
            fasta.close()
            assert fetcher.fetch_window("chrA", 0, 10) == ""
    finally:
        reference_module.REFERENCE_CACHE_BLOCKS = saved
