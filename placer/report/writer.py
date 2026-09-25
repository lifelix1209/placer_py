"""The only place in the output stage that opens a file.

Every renderer beside this one returns a string, so this module is where that
choice is paid for: it holds the filenames, the directory creation and the
handles, and nothing else. Keeping it to that means a test can exercise the
whole output surface without a filesystem, and a caller embedding the library
can take the strings and put them wherever it likes.

ALL THE FILES, ALWAYS, EVEN WHEN EMPTY. A missing `structural_calls.tsv` is
ambiguous between "none were set aside" and "the run died before writing it",
and a downstream script cannot tell the difference. The same argument is why
an empty run still produces a VCF with a full header.

THE RETURNED DICT IS PART OF THE CONTRACT: `tests/test_32_pipeline.py` asserts
its exact key set, so adding an output is a deliberate, visible change rather
than something that can be slipped in.
"""

from __future__ import annotations

import os

from placer.report import csv_table, tsv, vcf
from placer.report.context import ReportContext


def write_outputs(result, output_dir: str, include_insert_seq: bool = False,
                  include_support_qnames: bool = False,
                  context: ReportContext | None = None) -> dict[str, str]:
    """Write the five output files and return their paths.

    `context` carries the contig list, the sample name and the anchor bases the
    VCF needs. Omitting it still produces a valid VCF -- contig-less and
    `N`-anchored -- which is what lets this be exercised without a BAM.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "scientific_txt": os.path.join(output_dir, "scientific.txt"),
        "structural_calls_tsv": os.path.join(output_dir, "structural_calls.tsv"),
        "evidence_ledger_tsv": os.path.join(output_dir, "evidence_ledger.tsv"),
        "calls_vcf": os.path.join(output_dir, "calls.vcf"),
        "calls_csv": os.path.join(output_dir, "calls.csv"),
    }
    with open(paths["scientific_txt"], "w") as handle:
        handle.write(tsv.render_scientific_txt(result, include_insert_seq))
    with open(paths["structural_calls_tsv"], "w") as handle:
        handle.write(tsv.render_structural_calls_tsv(result, include_insert_seq))
    with open(paths["evidence_ledger_tsv"], "w") as handle:
        handle.write(tsv.render_evidence_ledger_tsv(result, include_insert_seq,
                                                    include_support_qnames))
    with open(paths["calls_vcf"], "w") as handle:
        handle.write(vcf.render_vcf(result, context))
    with open(paths["calls_csv"], "w") as handle:
        handle.write(csv_table.render_csv(result, include_insert_seq,
                                          include_support_qnames))
    return paths
