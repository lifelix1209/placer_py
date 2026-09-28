"""
Opt-in run accounting (`placer/io/perf.py`): the TSV, and that it is inert.

Nothing in the scan reads the counters, so the outputs cannot depend on them;
`tests/test_38_parallel.py` already runs the real pipeline with them always
bumped. What this file pins is the shape `tools/perf` reads: a span is a delta
over what happened inside it, the file has one header and then a row per span,
and an unset `PLACER_PERF_LOG` writes nothing.
"""

from __future__ import annotations

import os
import tempfile

from placer.io import perf


def test_a_span_is_a_delta_over_what_happened_inside_it():
    perf.count("ref_fetch_calls", 5)
    before = perf.snapshot()
    perf.count("ref_fetch_calls", 3)
    perf.count_locked("blastn_calls")
    row = perf.span(before, "chr1:0-10")
    assert row["ref_fetch_calls"] == 3
    assert row["blastn_calls"] == 1
    assert row["bam_fetch_calls"] == 0
    assert row["label"] == "chr1:0-10"
    assert row["pid"] == os.getpid()
    assert row["wall_s"] >= 0 and row["user_s"] >= 0 and row["sys_s"] >= 0
    assert set(perf.COLUMNS) <= set(row)


def test_an_unset_log_writes_nothing():
    log = perf.PerfLog("")
    assert not log.enabled
    log.write(perf.span(perf.snapshot(), "run"))


def test_the_log_has_one_header_and_then_a_row_per_span():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "perf.tsv")
        log = perf.PerfLog(path)
        assert log.enabled
        before = perf.snapshot()
        log.write(perf.span(before, "a"))
        log.write(perf.span(before, "b"))
        with open(path) as handle:
            lines = handle.read().splitlines()
    assert lines[0].split("\t") == list(perf.COLUMNS)
    assert [line.split("\t")[0] for line in lines[1:]] == ["a", "b"]
    assert all(len(line.split("\t")) == len(perf.COLUMNS) for line in lines)


def test_a_repeated_abpoa_input_is_answered_from_the_cache_with_the_same_consensus():
    import importlib.util

    import pytest
    if importlib.util.find_spec("pyabpoa") is None:
        pytest.skip("needs pyabpoa")
    from placer.io import poa

    reads = ["ACGTTGCAAGGCTTACCGATGACGTTAGCATGCATCGA" * 3,
             "ACGTTGCAAGGCTTACCGTTGACGTTAGCATGCATCGA" * 3,
             "ACGTTGCAAGGCTTACCGATGACGTAAGCATGCATCGA" * 3]
    hits = perf.COUNTS["poa_cache_hits"]
    first = poa.pyabpoa_consensus(list(reads))
    again = poa.pyabpoa_consensus(list(reads))
    assert again == first and first
    assert perf.COUNTS["poa_cache_hits"] == hits + 1
    # A different list is computed, not looked up.
    assert poa.pyabpoa_consensus(reads[:2]) is not None
    assert perf.COUNTS["poa_cache_hits"] == hits + 1
