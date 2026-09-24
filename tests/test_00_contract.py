"""
The seam. Runs today, because `placer_py.schema` is implemented.

`evidence_ledger.tsv` is the interface between the compiled scanner and the
Python decision layer, which is why the migration can start in the middle: both
sides can run on the same input and their outputs diffed. A contract that is
itself unwritten would constrain nothing, so this module is real code and these
are real assertions.
"""

from __future__ import annotations

import pytest

from placer_py import schema
from placer_py.report import tsv as outputs

pytestmark = pytest.mark.contract


def test_every_column_declares_a_dtype():
    for name, dtype in schema.LEDGER_COLUMNS.items():
        assert dtype in ("str", "int32", "int64", "float64"), (name, dtype)


def test_the_groups_partition_the_contract():
    """No column may appear in two role groups: a column with two roles is a
    column whose meaning nobody can state."""
    groups = [schema.LOCUS_COLUMNS, schema.COUNT_COLUMNS,
              schema.GENOTYPE_COLUMNS, schema.SEQUENCE_COLUMNS,
              schema.BOUNDARY_COLUMNS, schema.CERTIFICATE_COLUMNS,
              schema.QC_COLUMNS]
    total = sum(len(g) for g in groups)
    assert total == len(schema.LEDGER_COLUMNS), (
        "a column is declared in more than one group")


def test_the_load_bearing_certificate_columns_are_present():
    """These are what the dependency bound and e-BH consume. Losing one of them
    silently disables selection rather than breaking it loudly."""
    for column in ("mechanistic_raw_log_bf_te_vs_artifact",
                   "mechanistic_raw_log_bf_te_vs_non_te",
                   "mechanistic_lower_log_bf_te_vs_artifact",
                   "mechanistic_lower_log_bf_te_vs_non_te",
                   "mechanistic_ambiguity_width",
                   "mechanistic_blocks"):
        assert column in schema.CERTIFICATE_COLUMNS, column


def test_header_check_accepts_extra_columns_but_not_missing_ones():
    full = list(schema.LEDGER_COLUMNS)
    assert schema.check_header(full).ok
    assert schema.check_header(full + ["some_new_diagnostic"]).ok, (
        "the C++ ledger carries many diagnostics; a contract that broke on "
        "every new one would be a liability")
    result = schema.check_header(full[:-1])
    assert not result.ok
    assert result.missing == (full[-1],)


def test_the_tprt_observables_are_absent_from_the_current_ledger():
    """
    The concrete cost of the current schema's information loss, pinned so the
    port cannot quietly proceed without deciding what to do about it.

    The one that matters most is the element-coordinate pair. The ledger keeps
    only `best_te_query_coverage`, a ratio, which discards the ln(L - l + 1)
    nats of the 3'-anchoring term -- about 8.5 nats for a 1 kb fragment of a
    6 kb L1. Two integers instead of one ratio is the entire cost of recovering
    it, and until the scanner emits them the TPRT model cannot be computed at
    all.
    """
    for column in schema.MISSING_FOR_TPRT:
        assert column not in schema.LEDGER_COLUMNS, (
            f"{column} is now in the ledger -- move it out of MISSING_FOR_TPRT")
    for column in ("te_elem_start", "te_elem_end", "te_elem_length",
                   "polya_len", "flank_t_tract_len",
                   "local_repeat_frac", "local_at_frac"):
        assert column in schema.MISSING_FOR_TPRT, column


def test_a_and_t_are_not_conflated_in_the_polya_contract():
    """
    `polya_len` is specified as oriented by `te_strand`, and both must be in the
    contract together.

    poly(A) on the + strand and poly(T) on the - strand are the same physical
    tail seen from opposite sides, and which one appears tells you the
    insertion's ORIENTATION -- which then has to agree with the element
    alignment strand and the TSD geometry. That agreement is a hard constraint
    a real insertion satisfies and a mismapped reference copy satisfies only by
    luck, so throwing it away is throwing away nearly free discrimination.
    """
    assert "polya_len" in schema.MISSING_FOR_TPRT
    assert "te_strand" in schema.MISSING_FOR_TPRT


# ------------------------------------------- the WRITER contract


#: Ledger columns that are not fields of `EvidenceLedgerRow`, with the reason.
#: Anything not listed here must be a field, or the header and the row have
#: drifted apart with nothing to notice.
DERIVED_LEDGER_COLUMNS = {
    # A count of `support_qnames`, which is itself only written on request.
    "support_qname_count",
}

#: The same for the call file, where several columns are deliberately spelled
#: differently from the field they come from -- the C++ header names, kept so
#: a downstream script written against the C++ output still parses.
RENAMED_CALL_COLUMNS = {
    "te",              # <- te_name
    "family_status",   # <- family_alignment_resolved, as a token
    "gt",              # <- genotype
    "qc",              # <- final_qc
    "consensus_len",   # <- event_consensus_len
}


def test_every_ledger_column_is_a_field_or_a_declared_derivation():
    """The header is hand-written; the row is generated from a dataclass.

    Nothing tied the two together. `EVIDENCE_LEDGER_COLUMNS` is a literal
    tuple of 59 names and `EvidenceLedgerRow` has 66 fields, so renaming a
    field, or adding a column without a field, drifts silently -- the file
    still has the right number of columns and the wrong contents under one of
    them.

    This is the WRITER contract, and it is a different thing from
    `LEDGER_COLUMNS` above: that one pins which columns the decision layer
    READS out of a ledger the C++ produced.
    """
    import dataclasses

    from placer_py.ledger import EvidenceLedgerRow

    fields = {f.name for f in dataclasses.fields(EvidenceLedgerRow)}
    unexplained = [c for c in outputs.EVIDENCE_LEDGER_COLUMNS
                   if c not in fields and c not in DERIVED_LEDGER_COLUMNS]
    assert not unexplained, (
        f"ledger columns with no field and no declared derivation: {unexplained}")


def test_every_call_column_is_a_field_or_a_declared_rename():
    import dataclasses

    from placer_py.ledger import FinalCall

    fields = {f.name for f in dataclasses.fields(FinalCall)}
    unexplained = [c for c in outputs.FINAL_CALL_COLUMNS
                   if c not in fields and c not in RENAMED_CALL_COLUMNS]
    assert not unexplained, (
        f"call columns with no field and no declared rename: {unexplained}")


def test_the_header_and_the_row_stay_the_same_length():
    """Including under every combination of the two optional columns.

    A header built by one function and a row by another is exactly where an
    off-by-one lands, and a TSV with a shifted column is far worse than one
    that fails to parse: every value is present and every value is under the
    wrong name.
    """
    from placer_py.ledger import EvidenceLedgerRow, FinalCall

    row = EvidenceLedgerRow()
    for insert_seq in (False, True):
        for qnames in (False, True):
            assert (len(outputs.evidence_ledger_row(row, insert_seq, qnames))
                    == len(outputs.evidence_ledger_header(insert_seq, qnames))), (
                f"ledger, insert_seq={insert_seq} qnames={qnames}")

    call = FinalCall()
    for insert_seq in (False, True):
        assert (len(outputs.final_call_row(call, insert_seq))
                == len(outputs.final_call_header(insert_seq))), (
            f"call, insert_seq={insert_seq}")
