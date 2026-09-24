"""Gate-1 applied to a read stream: which reads enter the rest of the run.

WHY THIS IS THE INPUT STAGE'S JOB. A 30x human BAM holds ~6 million reads and
only a tiny fraction touch an insertion. Deciding which ones to carry forward
is the last thing that happens on the way in and the one place where per-read
cost matters -- everything downstream works on ~10^5 candidates. Lifting it out
of `run_pipeline`'s body is what makes it possible to run the gate alone, to
count what it rejects, or to substitute a different one, none of which was
reachable while it was a closure inside the orchestrator.

THE PREDICATE ITSELF IS NOT HERE AND DOES NOT MOVE. `placer_py/reads.py` holds
`pass_preliminary` and `Gate1SignalConfig`, both pure functions of scalars, and
`placer_py/alignment.py` depends on that module for the CIGAR and flag
constants. It is shared vocabulary, not input-stage code, and
`tests/test_18_reads.py` pins it at that path. What is here is only the
APPLICATION of the predicate to a stream.

TWO PROPERTIES THIS MUST NOT LOSE, both of which the original closure had:

  * It is a FILTER, not a materialising pass. The scan is the only affordable
    way to touch six million reads, and each `AlignedRead` carries its full
    sequence -- draining this into a list is what made a 10 Mb region of
    ultra-long ONT take 2.2 GB. See `group_reads_into_bins`.
  * The counters are only final once the stream is DRAINED. They are read by
    the run summary in `placer_py/outputs.py`, which renders after the bin loop
    has finished, so this is a property of where they are read rather than of
    when they are written.

WHY IT TAKES A COUNTER OBJECT rather than returning counts. `PipelineResult`
already has `total_reads` and `gate1_passed`, and `tests/test_32_pipeline.py`
reads them off `run_pipeline`'s return value. Incrementing that object directly
means there is no copy step -- and therefore no "when do we copy" bug to argue
about. A separate counter that gets copied back after the loop would also be
correct, but only because of an ordering argument this version does not need.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from placer_py.alignment import AlignedRead
from placer_py.reads import Gate1SignalConfig, pass_preliminary


class GateCounters(Protocol):
    """What the gate needs to write its tallies to.

    A Protocol rather than a dataclass because `PipelineResult` already
    satisfies it structurally, and introducing a second record that had to be
    reconciled with it would be inventing the very copy step this design
    avoids.
    """

    total_reads: int
    gate1_passed: int


def gate_reads(stream: Iterable[AlignedRead], counters: GateCounters,
               config: Gate1SignalConfig | None = None) -> Iterator[AlignedRead]:
    """Yield the reads worth carrying forward, tallying as it goes.

    `config` defaults to `Gate1SignalConfig()` and there is deliberately no way
    to reach it from `PipelineConfig` or the `PLACER_*` environment: the gate
    is not tunable today, and a knob whose default is silently a copy of
    another default is how two thresholds drift apart.
    """
    cfg = config if config is not None else Gate1SignalConfig()
    for read in stream:
        counters.total_reads += 1
        nm = read.get_int_tag("NM")
        if not pass_preliminary(read.cigar, read.flag, read.seq_len,
                                read.mapq, read.has_sa_tag(), nm, cfg):
            continue
        counters.gate1_passed += 1
        yield read
