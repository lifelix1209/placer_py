"""Where a run's CPU and I/O went: opt-in accounting, per chunk.

`PLACER_PERF_LOG=PATH` makes a run write one TSV row per scanned chunk, plus
one for the whole run, to PATH. NOTHING IN THE SCAN READS ANY OF THIS, so the
four outputs are the same bytes with it on or off. The counters are integer
increments at the input-stage adapters (BAM fetch and stream, reference fetch,
blastn, abPOA), cheap enough to be always on; only the file is opt-in.

WHY SELF AND CHILDREN SEPARATELY. blastn runs as a child process, so its CPU
is in RUSAGE_CHILDREN and the interpreter's own is in RUSAGE_SELF. TEBench's
`cpu_hours` is the sum of both, but they are fixed in different places.

WHY THE I/O COLUMNS. On BeeGFS in `buffered` cache mode nothing a process reads
stays in the page cache, so every re-read of a BAM block or a reference window
goes back over the network and is paid for as system time. `rchar` and
`syscr` (from `/proc/self/io`) count what the process asked `read()` for, and
`inblock` what the kernel fetched for it. Together with the fetch counters they
tell a re-read from new data.
"""

from __future__ import annotations

import os
import resource
import threading
import time
import weakref
from collections import Counter
from typing import Callable

#: Per-process tallies, keyed by counter name. Module-global because the
#: adapters that bump them are spread across `placer/io/` and share nothing
#: else. Workers each have their own copy (spawned processes).
COUNTS: Counter[str] = Counter()
_LOCK = threading.Lock()
#: Objects that keep their own tallies because a `count()` call per event
#: would cost more than the event (the reference fetcher, ~10^4 windows per
#: evaluated locus). Weak, so a closed fetcher does not stay alive for this.
_PROVIDERS: list[weakref.WeakMethod] = []

#: The environment variable naming the TSV to write. Unset: no file.
PERF_LOG_ENV = "PLACER_PERF_LOG"

#: Counters every row carries, in this order, whether or not they moved.
COUNTER_NAMES = (
    "bam_stream_records", "bam_fetch_calls", "bam_fetch_records",
    "bam_buffer_served", "bam_buffer_records", "bam_buffer_fallbacks",
    "bam_conversions", "ref_fetch_calls", "ref_block_loads",
    "blastn_calls", "blastn_queries", "poa_calls", "poa_cache_hits",
    "poa_input_bases",
)
#: Resource columns, as deltas over the row's span.
USAGE_NAMES = (
    "wall_s", "user_s", "sys_s", "child_user_s", "child_sys_s",
    "inblock", "majflt", "minflt", "nvcsw", "nivcsw", "rchar", "syscr",
)
COLUMNS = ("label", "pid", "t_start", *USAGE_NAMES, "maxrss_kb", *COUNTER_NAMES)


def count(name: str, n: int = 1) -> None:
    """Bump a counter from the interpreter's own thread."""
    COUNTS[name] += n


def count_locked(name: str, n: int = 1) -> None:
    """Bump a counter from a helper thread (the blastn pool)."""
    with _LOCK:
        COUNTS[name] += n


def register(provider: Callable[[], dict[str, int]]) -> None:
    """Add a bound method returning an object's own cumulative tallies."""
    _PROVIDERS.append(weakref.WeakMethod(provider))


def _proc_io() -> dict[str, int]:
    try:
        with open("/proc/self/io") as handle:
            fields = dict(line.split(":", 1) for line in handle if ":" in line)
    except OSError:
        return {"rchar": 0, "syscr": 0}
    return {"rchar": int(fields.get("rchar", 0)), "syscr": int(fields.get("syscr", 0))}


def snapshot() -> dict[str, float]:
    """Cumulative usage and counters of this process, now."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    row: dict[str, float] = {
        "t_start": time.time(), "wall_s": time.perf_counter(),
        "user_s": own.ru_utime, "sys_s": own.ru_stime,
        "child_user_s": children.ru_utime, "child_sys_s": children.ru_stime,
        "inblock": own.ru_inblock, "majflt": own.ru_majflt, "minflt": own.ru_minflt,
        "nvcsw": own.ru_nvcsw, "nivcsw": own.ru_nivcsw, "maxrss_kb": own.ru_maxrss,
    }
    row.update(_proc_io())
    for name in COUNTER_NAMES:
        row[name] = COUNTS[name]
    for reference in _PROVIDERS:
        provider = reference()
        if provider is not None:
            for name, value in provider().items():
                row[name] += value
    return row


def span(before: dict[str, float], label: str) -> dict[str, float | str]:
    """The row for the span from `before` to now: deltas, except the start
    time and the peak RSS, which are absolute."""
    after = snapshot()
    row: dict[str, float | str] = {"label": label, "pid": os.getpid(),
                                   "t_start": before["t_start"],
                                   "maxrss_kb": after["maxrss_kb"]}
    for name in (*USAGE_NAMES, *COUNTER_NAMES):
        row[name] = after[name] - before[name]
    return row


class PerfLog:
    """The TSV, or nothing when `PLACER_PERF_LOG` is unset. Written only by
    the parent process, one row at a time, so rows never interleave."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path if path is not None else os.environ.get(PERF_LOG_ENV, "")
        self._wrote_header = False

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def write(self, row: dict[str, float | str]) -> None:
        if not self.path:
            return
        with open(self.path, "w" if not self._wrote_header else "a") as handle:
            if not self._wrote_header:
                handle.write("\t".join(COLUMNS) + "\n")
                self._wrote_header = True
            handle.write("\t".join(_format(row.get(name, "")) for name in COLUMNS)
                         + "\n")


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
