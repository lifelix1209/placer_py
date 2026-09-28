"""abPOA, through its Python binding -- the one consensus backend.

WHY ONLY THIS MOVED OUT OF `core/consensus.py`. `pyabpoa` is the single genuine
third-party leak in the algorithm: every other external dependency of the port
is either stdlib (`subprocess`, `tempfile`) or already confined to
`placer/io/`. Putting these eleven lines here is what turns "pyabpoa is
confined to one lazily-imported function" from an honour-system comment into a
property `tests/test_36_layering.py` can assert.

WHAT DELIBERATELY STAYED BEHIND. `single_sequence_consensus` and
`ConsensusUnavailable` are the reason the whole pipeline can be driven with no
abPOA installed -- they are `StageHooks.consensus_fn`'s default and the refusal
that `CONTRIBUTING.md` names by name. And `poa_reads_within_budget` is not a
backend detail at all: the memory cap changes WHICH READS ENTER the consensus,
which is an algorithm decision with a measured entry in
`docs/departures-from-cpp.md`. Only the call into the library is input-stage
work, and only the call moved.
"""

from __future__ import annotations

from collections import OrderedDict

from placer.core.seqtools import upper_acgt
from placer.io.perf import count

#: abPOA's answers, by their exact input, per process. The same reads are
#: assembled again and again: neighbouring components of one locus, and the
#: hypotheses of one component whose 25 bp locality selects the same fragments,
#: hand abPOA the same list -- in a collapsed pericentromere, side consensuses
#: of hundreds of 3 kb clips each. abPOA is deterministic, so an answer is a
#: function of the list alone and reusing it is exact at any scope: across
#: bins, and whatever the chunking (`tests/test_38_parallel.py`).
POA_CACHE_MAX_BASES = 64_000_000
_CACHE: OrderedDict[tuple[str, ...], str] = OrderedDict()
_cache_bases = 0


def pyabpoa_consensus(sequences: list[str]) -> str:
    """abPOA through its Python binding -- the same library the C++ links.

    Imported lazily so the package has no hard dependency on it: the decision
    layer needs no consensus at all, and only a run that starts from a BAM does.
    """
    import pyabpoa  # noqa: F401  (optional dependency)

    global _cache_bases
    if not sequences:
        return ""
    key = tuple(sequences)
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        count("poa_cache_hits")
        return cached
    size = sum(len(seq) for seq in key)
    count("poa_calls")
    count("poa_input_bases", size)
    aligner = pyabpoa.msa_aligner()
    result = aligner.msa([upper_acgt(seq) for seq in sequences], out_cons=True,
                         out_msa=False, max_n_cons=1)
    consensus = upper_acgt(result.cons_seq[0]) if result.cons_seq else ""
    if size <= POA_CACHE_MAX_BASES // 4:
        _CACHE[key] = consensus
        _cache_bases += size
        while _cache_bases > POA_CACHE_MAX_BASES and _CACHE:
            dropped, _ = _CACHE.popitem(last=False)
            _cache_bases -= sum(len(seq) for seq in dropped)
    return consensus
