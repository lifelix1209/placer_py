"""The algorithm: reads in, calls and a ledger out.

WHAT THIS PACKAGE IS FOR. Everything that decides something lives here --
clustering, breakpoints, consensus, segmentation, TE naming, the policy layer,
the selection layer. Nothing here talks to anything outside the process.

THE DEPENDENCY RULE, which `tests/test_37_layering.py` enforces:

    placer/core/** may not import placer.io, placer.report,
    placer.pipeline, placer.wiring, placer.parallel or placer.main,
    and may not import pysam, pyabpoa, subprocess or tempfile -- at module
    scope OR inside a function body.

The last clause is not decoration. This codebase uses function-local imports
heavily, so a rule checked only on the top import block would be checking the
wrong half.

What core needs from outside it takes as a PARAMETER: the reference is a
callable, the TE alignment is a callable, the consensus is a callable. See
`placer/core/contracts.py`. That is what lets the whole algorithm run on
literals in a test, which is in turn what lets the suite run with no
third-party package installed at all.

THIS `__init__` IMPORTS NOTHING, so that `import placer.core.finalization`
costs exactly the decision layer and not the scan.
"""

from __future__ import annotations
