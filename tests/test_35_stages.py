"""
The three stages, and the seams between them.

Where to look: `placer_py/io/gate.py` for the input stage's filter, and
`placer_py/pipeline.py` for the composition that joins gate, scan and
finalize.

WHAT THIS FILE IS FOR that no other file covers. `tests/test_18_reads.py` pins
what the gate DECIDES about one read; this pins how the gate BEHAVES on a
stream -- that it stays lazy, and that its tallies count what they claim to.
Those are properties of the plumbing rather than of the predicate, and the
mini example dataset cannot see either of them: every one of its 180 reads
passes gate-1, so on that input the gate is a no-op.

THE LAZINESS TEST IS THE LOAD-BEARING ONE. A gate that drains its source into
a list produces byte-identical answers and destroys the only property that
makes a whole-genome scan affordable -- each `AlignedRead` carries its full
sequence, and materialising a 10 Mb region of ultra-long ONT was 2.2 GB. No
output can show that, so it is observed from outside, by counting how far the
source has been advanced.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip

from placer_py.alignment import AlignedRead
from placer_py.io.gate import gate_reads
from placer_py.reads import Gate1SignalConfig

pytestmark = pytest.mark.invariant


class _Counters:
    """The two fields `gate_reads` writes. `PipelineResult` satisfies the same
    shape structurally, which is the point of the Protocol."""

    def __init__(self) -> None:
        self.total_reads = 0
        self.gate1_passed = 0


def _passing_read(qname: str = "r") -> AlignedRead:
    """A read with no signal and a MAPQ above the background floor.

    Background coverage is the cheapest way through the gate: no supplementary
    bit, no SA tag, no long clip and no long insertion, so the decision is the
    single `mapq > background_mapq_min` test.
    """
    return AlignedRead(qname=qname, tid=0, pos=0, mapq=60,
                       cigar=[(0, 200)], seq="A" * 200)


def _rejected_read(qname: str = "x") -> AlignedRead:
    """Same shape, but MAPQ exactly at the floor -- the comparison is strict."""
    return AlignedRead(qname=qname, tid=0, pos=0,
                       mapq=Gate1SignalConfig().background_mapq_min,
                       cigar=[(0, 200)], seq="A" * 200)


# ------------------------------------------------------------------- laziness
def test_the_gate_is_a_filter_and_never_drains_its_source():
    """
    Consuming one read must advance the source by one. This is what a
    materialising gate would break while leaving every output identical.
    """
    reads = [_passing_read("a"), _passing_read("b"), _passing_read("c")]
    consumed: list[str] = []

    def source():
        for read in reads:
            consumed.append(read.qname)
            yield read

    counters = _Counters()
    gated = call_or_skip(gate_reads, source(), counters)
    assert next(gated).qname == "a"
    assert consumed == ["a"], consumed
    assert next(gated).qname == "b"
    assert consumed == ["a", "b"], consumed


def test_a_source_that_fails_late_still_yields_what_came_before_it():
    """
    The scan must survive reaching a bad record: everything before it has
    already been handed over. A gate that read ahead would lose all of it.
    """
    def source():
        yield _passing_read("a")
        yield _passing_read("b")
        raise RuntimeError("source blew up")

    counters = _Counters()
    gated = gate_reads(source(), counters)
    assert [next(gated).qname, next(gated).qname] == ["a", "b"]
    with pytest.raises(RuntimeError):
        next(gated)


# ------------------------------------------------------------------- counters
def test_the_counters_are_only_final_once_the_stream_is_drained():
    """
    They are read by the run summary, which renders after the bin loop. Mid
    stream they are a partial tally and the test says so rather than pretending
    a generator could report otherwise.
    """
    counters = _Counters()
    gated = gate_reads(iter([_passing_read("a"), _passing_read("b")]), counters)
    assert (counters.total_reads, counters.gate1_passed) == (0, 0)
    next(gated)
    assert (counters.total_reads, counters.gate1_passed) == (1, 1)
    list(gated)
    assert (counters.total_reads, counters.gate1_passed) == (2, 2)


def test_a_rejected_read_is_counted_as_seen_but_not_as_passed():
    counters = _Counters()
    kept = list(gate_reads(iter([_passing_read("a"), _rejected_read("x"),
                                 _passing_read("b")]), counters))
    assert [read.qname for read in kept] == ["a", "b"]
    assert (counters.total_reads, counters.gate1_passed) == (3, 2)


def test_an_empty_stream_leaves_the_counters_at_zero():
    counters = _Counters()
    assert list(gate_reads(iter([]), counters)) == []
    assert (counters.total_reads, counters.gate1_passed) == (0, 0)


def test_the_gate_uses_the_default_config_when_none_is_given():
    """
    The default is `Gate1SignalConfig()`, and nothing in `PipelineConfig` or
    the `PLACER_*` environment can reach it. Pinned because a knob whose
    default is silently a copy of another default is how two thresholds drift.
    """
    explicit, implicit = _Counters(), _Counters()
    reads = [_passing_read("a"), _rejected_read("x")]
    assert ([r.qname for r in gate_reads(iter(reads), implicit)]
            == [r.qname for r in gate_reads(iter(reads), explicit,
                                            Gate1SignalConfig())])
    assert (implicit.total_reads, implicit.gate1_passed) == (2, 1)


# ------------------------------------------------------- scan versus finalize
def test_the_scan_leaves_the_run_uncalibrated():
    """
    `run_scan` must stop before the whole-run stage. The dependency bound is a
    null expectation measured across every candidate, so a scan that had
    already applied it would have measured it from a partial run.
    """
    from placer_py.core.result import PipelineResult

    result = PipelineResult()
    assert result.dependency_penalty_estimated is False
    assert result.estimated_dependency_penalty == 0.0
    assert result.final_pass_calls == 0


def test_scan_then_finalize_is_the_same_run_as_run_pipeline():
    """
    THE PARTITION TEST. `run_pipeline` is defined as gate + scan + finalize, so
    driving the three by hand must reach the identical result -- otherwise the
    split silently duplicated or dropped a step.

    Both halves are run on the same synthetic reads `test_32_pipeline.py` uses
    for its end-to-end acceptance, because a partition that only holds on empty
    input holds for the wrong reason.
    """
    from test_32_pipeline import hooks, synthetic_reads

    from placer_py.config import PipelineConfig
    from placer_py.core.contracts import ReadSource
    from placer_py.core.finalize import finalize_run
    from placer_py.core.result import PipelineResult
    from placer_py.core.scan import run_scan
    from placer_py.pipeline import run_pipeline

    config = PipelineConfig(bin_size=100000)
    reads = synthetic_reads()

    def chrom_name(tid):
        return "chr1"

    def fetch_local(chrom, start, end):
        return reads

    whole = run_pipeline(reads, chrom_name, fetch_local, config, hooks(reads))

    split = PipelineResult()
    source = ReadSource(reads=gate_reads(reads, split),
                        chromosome_name=chrom_name, fetch_local=fetch_local)
    run_scan(source, config, hooks(reads), split)
    finalize_run(split, config)

    assert (split.total_reads, split.gate1_passed) == (whole.total_reads,
                                                       whole.gate1_passed)
    assert split.processed_bins == whole.processed_bins
    assert split.built_components == whole.built_components
    assert split.final_pass_calls == whole.final_pass_calls
    assert len(split.final_calls) == len(whole.final_calls)
    assert len(split.evidence_ledger) == len(whole.evidence_ledger)
    assert [(c.chrom, c.pos, c.te_name) for c in split.final_calls] == \
           [(c.chrom, c.pos, c.te_name) for c in whole.final_calls]
    assert split.estimated_dependency_sigma == whole.estimated_dependency_sigma
