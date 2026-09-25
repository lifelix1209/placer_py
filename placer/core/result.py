"""What one run accumulates: the counters, the calibration, the three lists.

WHY THIS IS ITS OWN MODULE. `PipelineResult` is the contract between the
algorithm and the report stage -- `placer/report/tsv.py` reads six of its
fields by name -- and it used to live at the bottom of `finalization.py`, 2800
lines into the largest module in the package. A record that three stages touch
should not be reachable only by importing the stage that happens to fill in
most of it.

IT IS MUTATED IN PLACE, ON PURPOSE, and by more than one stage. The input
stage's gate increments `total_reads` and `gate1_passed` as it yields (see
`placer/io/gate.py`); the scan appends to the three lists per bin;
finalization rewrites the calibration fields once at the end. Returning fresh
copies at each step would mean the streaming scan could not report a partial
tally, and the counters are read by the run summary precisely because they are
cumulative.

THE CALIBRATION FIELDS ARE REPORTED, not just used. `scientific.txt` writes
sigma, the penalty, the cap and the null count into its header because a
reader needs them to interpret the e-values in the rows below -- a file
reporting selections without reporting what they were calibrated against
cannot be audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from placer.core.ledger import EvidenceLedgerRow, FinalCall


@dataclass
class PipelineResult:
    total_reads: int = 0
    gate1_passed: int = 0
    processed_bins: int = 0
    built_components: int = 0
    event_consensus_calls: int = 0
    genotype_calls: int = 0
    final_pass_calls: int = 0
    #: Re-estimated at finalization from this run's own count distribution.
    estimated_overdispersion: float = 0.02
    #: The measured dependency bound. Earlier releases asserted 0.65 nats
    #: (sigma = 1.92) with no evidence; these fields report what was used.
    estimated_dependency_penalty: float = 0.0
    estimated_dependency_penalty_non_te: float = 0.0
    estimated_dependency_sigma: float = 1.0
    estimated_dependency_sigma_non_te: float = 1.0
    #: Derived, not chosen: log(m/q), the largest e-value e-BH can ever require.
    dependency_penalty_cap_log: float = 0.0
    dependency_penalty_null_count: int = 0
    dependency_penalty_estimated: bool = False

    final_calls: list[FinalCall] = field(default_factory=list)
    #: Selected structural insertions the TE-calibrated mode sets aside. They
    #: are SELECTED calls, not rejects -- moving them here rather than erasing
    #: them keeps the TE output TE-only while leaving them auditable. Silently
    #: discarding them made recall benchmarking on the default mode misleading.
    structural_calls: list[FinalCall] = field(default_factory=list)
    evidence_ledger: list[EvidenceLedgerRow] = field(default_factory=list)
