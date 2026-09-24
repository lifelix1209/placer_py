"""The output stage: turning a finished run into files.

WHAT THIS PACKAGE IS FOR. Five files come out of a run -- three that predate
this package and two that were added with it -- and every one of them is
rendered by a PURE FUNCTION returning a string. Opening files happens in
`placer_py/report/writer.py` and nowhere else.

THAT PURITY IS NOT A STYLE PREFERENCE. The suite runs under
`tools/run_tests_without_pytest.py`, which implements no `tmp_path`, no
`monkeypatch` and no `capsys`, so a renderer that wrote to disk could not be
tested there at all. `render_*(result, ...) -> str` is what lets every output
column, every VCF field and every quoting rule be pinned on a string.

THIS PACKAGE DOES NOT IMPORT `placer_py.io`. It reads `core`'s output types --
`PipelineResult`, `FinalCall`, `EvidenceLedgerRow` -- and the few facts that
only the input stage knows (the contig list, the sample name, one reference
base per record) arrive as a `ReportContext` built by the caller. Handing a
renderer a live pysam handle instead would put a compiled dependency inside the
one layer that has to be testable on literals.

THIS `__init__` IMPORTS NOTHING, for the same reason the other two do not.
"""

from __future__ import annotations
