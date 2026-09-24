"""The full flat table: every selected call, both sets, one CSV.

WHY A CSV WHEN THREE TSVs ALREADY EXIST. The VCF is for interoperation and is
deliberately narrow -- about thirty INFO keys out of a `FinalCall`'s ninety
fields. This file is the other half of that trade: everything, in one table, in
the format a spreadsheet and a dataframe both open without being told anything.

IT DOES NOT DECLARE ITS OWN COLUMNS. The body is `report/tsv.FINAL_CALL_COLUMNS`
reused verbatim, because a second list would be a second thing to keep in step
with `FinalCall` and `tests/test_00_contract.py` only checks the first one.
What is added is bracketing: two leading columns and seven trailing ones.

  call_set    `final` or `structural`. This is what lets one CSV replace two
              TSVs without losing the distinction that `FILTER=STRUCTURAL`
              carries in the VCF.
  vcf_pos     the join key to the VCF, and it is not redundant: the `pos`
              column is 0-based AND is the midpoint of the two breakpoints,
              so it is neither the VCF coordinate nor a breakpoint.
  te_qc, sequence_family_candidate, sequence_subfamily_candidate,
  sequence_family_commit_eligible
              these appear in no output file today and are the only record of
              WHY a family abstained. The CSV is their natural home.
  window_start, window_end, tid
              cheap, and they make a row self-locating.

QUOTING IS THE STDLIB'S, and three rules sit on top of it:

  * Every value is pre-rendered through `report/tsv._number` before
    `csv.writer` sees it. `csv` calls `str()` on non-strings, and `str(True)`
    is "True" where every other PLACER file writes "1".
  * `support_qnames` keeps the existing comma-joined spelling and is simply
    quoted, so the CSV and the TSV carry byte-identical strings. Switching the
    inner separator to ";" for this file would be a second spelling of one
    list.
  * Any `\\r` or `\\n` inside a value becomes a space. No field can contain one
    today, but `csv`'s answer to an embedded newline is a quoted multi-line
    record, and a CSV whose line count disagrees with its row count is how a
    downstream pipeline corrupts in silence.

`mechanistic_blocks` and `te_annotation_intervals` live here and NOT in the
VCF: the first contains `;`, `,` and `=`, all three of which are INFO
separators, and percent-encoding one field in a file where nothing else is
encoded is a trap.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from placer_py.ledger import FinalCall
from placer_py.report import tsv
from placer_py.report.vcf import vcf_pos

CSV_LEADING_COLUMNS: tuple[str, ...] = ("call_set", "vcf_pos")

CSV_TRAILING_COLUMNS: tuple[str, ...] = (
    "te_qc", "sequence_family_candidate", "sequence_subfamily_candidate",
    "sequence_family_commit_eligible", "window_start", "window_end", "tid")


def call_csv_header(include_insert_seq: bool = False,
                    include_support_qnames: bool = False) -> list[str]:
    columns = list(CSV_LEADING_COLUMNS)
    columns += tsv.final_call_header(include_insert_seq)
    columns += list(CSV_TRAILING_COLUMNS)
    if include_support_qnames:
        columns.append("support_qnames")
    return columns


def _scrub(value: str) -> str:
    """No value may contain a record separator. See the module docstring."""
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def call_csv_row(call: FinalCall, call_set: str,
                 include_insert_seq: bool = False,
                 include_support_qnames: bool = False) -> list[str]:
    """One call, in `call_csv_header` order, every field already a string."""
    values = [call_set, str(vcf_pos(call))]
    values += tsv.final_call_row(call, include_insert_seq)
    values += [tsv._number(v) for v in (
        call.te_qc, call.sequence_family_candidate,
        call.sequence_subfamily_candidate, call.sequence_family_commit_eligible,
        call.window_start, call.window_end, call.tid)]
    if include_support_qnames:
        values.append(tsv.serialize_support_qnames(call.support_qnames))
    return [_scrub(value) for value in values]


def _write(rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=",", quotechar='"', doublequote=True,
                        quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def render_csv(result, include_insert_seq: bool = False,
               include_support_qnames: bool = False) -> str:
    """Both call sets, final first, distinguished by `call_set`."""
    rows: list[Sequence[str]] = [call_csv_header(include_insert_seq,
                                                 include_support_qnames)]
    for call_set, calls in (("final", result.final_calls),
                            ("structural", result.structural_calls)):
        rows.extend(call_csv_row(call, call_set, include_insert_seq,
                                 include_support_qnames)
                    for call in calls)
    return _write(rows)
