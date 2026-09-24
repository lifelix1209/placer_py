"""abPOA, through its Python binding -- the one consensus backend.

WHY ONLY THIS MOVED OUT OF `core/consensus.py`. `pyabpoa` is the single genuine
third-party leak in the algorithm: every other external dependency of the port
is either stdlib (`subprocess`, `tempfile`) or already confined to
`placer_py/io/`. Putting these eleven lines here is what turns "pyabpoa is
confined to one lazily-imported function" from an honour-system comment into a
property `tests/test_36_layering.py` can assert.

WHAT DELIBERATELY STAYED BEHIND. `single_sequence_consensus` and
`ConsensusUnavailable` are the reason the whole pipeline can be driven with no
abPOA installed -- they are `StageHooks.consensus_fn`'s default and the refusal
that `CONTRIBUTING.md` names by name. And `poa_reads_within_budget` is not a
backend detail at all: the memory cap changes WHICH READS ENTER the consensus,
which is an algorithm decision with a measured entry in
`tests/EXPECTED_DIVERGENCE.md`. Only the call into the library is input-stage
work, and only the call moved.
"""

from __future__ import annotations

from placer_py.core.seqtools import upper_acgt


def pyabpoa_consensus(sequences: list[str]) -> str:
    """abPOA through its Python binding -- the same library the C++ links.

    Imported lazily so the package has no hard dependency on it: the decision
    layer needs no consensus at all, and only a run that starts from a BAM does.
    """
    import pyabpoa  # noqa: F401  (optional dependency)

    if not sequences:
        return ""
    aligner = pyabpoa.msa_aligner()
    result = aligner.msa([upper_acgt(seq) for seq in sequences], out_cons=True,
                         out_msa=False, max_n_cons=1)
    return upper_acgt(result.cons_seq[0]) if result.cons_seq else ""
