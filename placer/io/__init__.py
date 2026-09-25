"""The input stage: open the BAM, open the reference, gate the reads.

WHAT THIS PACKAGE IS FOR. Everything that talks to something outside the
process lives here -- pysam, the BLAST executables, abPOA -- and nothing else
does. `placer/core/` is the algorithm and may not import this package; the
composition root in `placer/pipeline.py` is what joins the two.

THIS `__init__` IMPORTS NOTHING, for the same reason `placer/__init__.py`
imports nothing: `placer.io.bam` needs pysam, and a package-level
`from . import bam` would make `import placer.io.gate` -- which is pure
Python and is exercised by the zero-dependency test runner -- fail in exactly
the locked-down environment the design is for.

So import what you need:

    from placer.io.gate import gate_reads        # needs nothing
    from placer.io.bam import make_bam_reader    # needs pysam
"""

from __future__ import annotations
