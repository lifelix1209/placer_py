"""
The seam between the compiled scanner and the Python decision layer.

`evidence_ledger.tsv` already is that seam in the C++ pipeline, which is why the
migration can start in the middle instead of at the edges: the compiled side
keeps producing the ledger, the Python side consumes it, and both can run on the
same input so their outputs can be diffed.

This module is implemented rather than stubbed, because a contract that is
itself unwritten cannot constrain anything.

Column meanings live in the C++ `EvidenceLedgerRow` (include/pipeline.h).
(This used to cite "the README's output section" as well; the README has never
had one.) What is pinned HERE is narrower and more useful for a port: which
columns the decision layer actually READS, and with what dtype. Anything not
listed is diagnostic output that the port may ignore.

THIS IS A READER CONTRACT, NOT A WRITER ONE, and the distinction is worth
stating because `check_header` invites the confusion. It answers "can the
decision layer consume this ledger?" for a ledger produced ELSEWHERE -- the
C++, or an older version of this tool. It does NOT describe the ledger
`placer/report/tsv.py` writes, and running it against
`report.tsv.EVIDENCE_LEDGER_COLUMNS` reports thirteen missing columns, which is
correct rather than a bug: those are fields of the C++ row that the Python
`EvidenceLedgerRow` spells differently or does not carry, and the decision
layer reads them from the call rather than the ledger.

The writer contract lives in `tests/test_00_contract.py`, which pins every
emitted column to a field of the dataclass it is rendered from, or to an
explicitly declared derivation or rename.
"""

from __future__ import annotations

from dataclasses import dataclass

# Columns the decision layer consumes. Splitting them by role documents WHY each
# one is needed, which is the part that rots first.
#
# Ordering within each group follows the C++ struct, so a reviewer can diff the
# two side by side.

#: Identity and position. Needed for clustering, de-duplication and output.
LOCUS_COLUMNS: dict[str, str] = {
    "chrom": "str",
    "tid": "int32",
    "bp_left": "int32",
    "bp_right": "int32",
    "coverage_left": "int32",
    "coverage_right": "int32",
}

#: Read counts. Feed the genotype model and the count Bayes factor.
COUNT_COLUMNS: dict[str, str] = {
    "alt_struct_reads": "int32",
    "alt_split_reads": "int32",
    "alt_indel_reads": "int32",
    "alt_left_clip_reads": "int32",
    "alt_right_clip_reads": "int32",
    "ref_span_reads": "int32",
    "low_mapq_ref_span_reads": "int32",
    "support_qname_count": "int32",
}

#: Genotype outputs, recomputed at finalization with the sample-level rho.
GENOTYPE_COLUMNS: dict[str, str] = {
    "genotype": "str",
    "gq": "int32",
    "af": "float64",
}

#: TE sequence evidence. The current six-block scorer reads these; the TPRT
#: model needs MORE than these (see MISSING_FOR_TPRT below), which is the point.
SEQUENCE_COLUMNS: dict[str, str] = {
    "family": "str",
    "subfamily": "str",
    "best_te_identity": "float64",
    "best_te_query_coverage": "float64",
    "cross_family_margin": "float64",
    "annotation_residual_fraction": "float64",
    "annotation_masked_fraction": "float64",
}

#: Breakpoint geometry and TSD.
BOUNDARY_COLUMNS: dict[str, str] = {
    "boundary_type": "str",
    "boundary_len": "int32",
    "tsd_mismatches": "int32",
    "insert_len": "int32",
}

#: The mechanistic certificate. These are the quantities the dependency bound
#: and e-BH consume, so they are the load-bearing ones for selection.
CERTIFICATE_COLUMNS: dict[str, str] = {
    "mechanistic_blocks": "str",
    "mechanistic_raw_log_bf_te_vs_artifact": "float64",
    "mechanistic_raw_log_bf_te_vs_non_te": "float64",
    "mechanistic_lower_log_bf_te_vs_artifact": "float64",
    "mechanistic_lower_log_bf_te_vs_non_te": "float64",
    "mechanistic_ambiguity_width": "float64",
    "mechanistic_ref_conflict_signal": "float64",
}

#: QC strings that gate promotability and null-control membership.
QC_COLUMNS: dict[str, str] = {
    "final_qc": "str",
    "robust_mechanistic_qc": "str",
    "robust_mechanistic_worst_case_lfdr": "float64",
    "posterior_qc": "str",
    "lfdr_qc": "str",
    "te_posterior": "float64",
    "lfdr": "float64",
    "worst_case_lfdr": "float64",
}

LEDGER_COLUMNS: dict[str, str] = {
    **LOCUS_COLUMNS,
    **COUNT_COLUMNS,
    **GENOTYPE_COLUMNS,
    **SEQUENCE_COLUMNS,
    **BOUNDARY_COLUMNS,
    **CERTIFICATE_COLUMNS,
    **QC_COLUMNS,
}

#: Observables the TPRT model needs that the current ledger does NOT carry.
#:
#: This list is the concrete cost of the current schema's information loss, and
#: it is pinned here so the port cannot quietly proceed without deciding what to
#: do about it. The scanner has to emit these, or the TPRT terms cannot be
#: computed at all:
#:
#:   te_elem_start / te_elem_end
#:       Where the insert aligns in ELEMENT coordinates. The current ledger
#:       keeps only `best_te_query_coverage`, a ratio -- which throws away the
#:       ln(L - l + 1) nats of the 3'-anchoring term. Two integers instead of
#:       one ratio is the entire cost of recovering it.
#:   te_elem_length, te_strand
#:       Needed to know whether the 3' end is the element's 3' end, and on which
#:       strand the poly(A) must sit.
#:   polya_len
#:       Length of the terminal A/T run at the 3' junction, in the orientation
#:       implied by te_strand. A and T must NOT be conflated: which one appears
#:       tells you the insertion's orientation, and that has to agree with the
#:       TSD geometry.
#:   flank_t_tract_len
#:       Matching T's immediately 5' of the nick, in the reference. This is the
#:       priming-site observable and it is free -- it is reference sequence.
#:   local_repeat_frac, local_at_frac
#:       The measured LOCAL nulls. Without them every coincidence term has to
#:       assume a background, which is exactly what makes a model fragile in
#:       VNTRs and A-rich tracts.
MISSING_FOR_TPRT: dict[str, str] = {
    "te_elem_start": "int32",
    "te_elem_end": "int32",
    "te_elem_length": "int32",
    "te_strand": "str",
    "polya_len": "int32",
    "flank_t_tract_len": "int32",
    "local_repeat_frac": "float64",
    "local_at_frac": "float64",
}


@dataclass(frozen=True)
class SchemaCheck:
    """Result of validating a ledger header against the contract."""

    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Extra columns are fine; missing ones are not."""
        return not self.missing


def check_header(header: list[str] | tuple[str, ...]) -> SchemaCheck:
    """
    Compare a ledger header against the columns the decision layer reads.

    Extra columns are allowed on purpose: the C++ ledger carries a lot of
    diagnostics, and a port that broke whenever a diagnostic was added would be
    a liability rather than a contract.
    """
    present = set(header)
    required = set(LEDGER_COLUMNS)
    return SchemaCheck(
        missing=tuple(sorted(required - present)),
        extra=tuple(sorted(present - required)),
    )
