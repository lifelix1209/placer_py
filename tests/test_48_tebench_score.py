"""tools/tebench_score.py's partitions and its holdout seal.

The scorer imports TEBench's own evaluator, so these tests are skipped where
the TEBench checkout is absent or the interpreter is older than 3.10.
"""

from __future__ import annotations

import pytest


def _scorer():
    """The scorer module, or a skip where TEBench is not importable.

    `pytest.skip`, not `importorskip`: the zero-dependency runner has only the
    former."""
    try:
        from tools import tebench_score
    except ImportError as error:
        pytest.skip(f"TEBench not importable: {error}")
    return tebench_score


def test_partitions_split_the_confident_contigs_as_tebench_does():
    scorer = _scorer()
    from tebench.regions import RegionIndex

    confident = RegionIndex.from_intervals(
        {"chr1": [(0, 10)], "chr8": [(0, 10)], "chr9": [(0, 10)], "chrX": [(0, 10)]})
    development = ["chr1", "chr2", "chr8"]      # chr2 has no confident region
    parts = {p: set(scorer.partition_regions(confident, development, p).intervals)
             for p in scorer.PARTITIONS}
    assert parts["development"] == {"chr1", "chr8"}
    assert parts["holdout"] == {"chr9", "chrX"}
    assert parts["all"] == {"chr1", "chr8", "chr9", "chrX"}


def test_a_sealed_caller_reaches_the_holdout_only_when_unsealed():
    scorer = _scorer()
    sealed = next(iter(scorer.SEALED_CALLERS))
    scorer.check_seal({sealed, "sniffles2"}, "development", unseal=False)
    scorer.check_seal({"sniffles2"}, "holdout", unseal=False)
    for partition in ("holdout", "all"):
        with pytest.raises(SystemExit):
            scorer.check_seal({sealed, "sniffles2"}, partition, unseal=False)
        scorer.check_seal({sealed}, partition, unseal=True)
