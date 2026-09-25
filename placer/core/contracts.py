"""What the algorithm needs from outside it, as three declarations.

THE WHOLE PACKAGE BELOW THIS ONE IS PURE, and these types are why it can be.
Each is a statement of something `placer/core/` cannot do for itself: read
the next read, fetch a reference window, align an insert against a library,
build a consensus, detect a TSD. Every one of them arrives as a PARAMETER, so
the algorithm never learns that pysam, BLAST or abPOA exist -- which is what
lets `tests/test_32_pipeline.py` drive the entire pipeline on literals with no
BAM, no reference index and no BLAST installed.

`placer/wiring.py` is the only place these are bound to real I/O.

ONE KNOWN COUPLING, recorded rather than worked around: `StageHooks`'
`consensus_fn` default is `consensus.single_sequence_consensus`, so importing
this module pulls in `consensus -> segmentation -> policy`. That is a real
cost for a module named "contracts" and it buys something specific -- a
`StageHooks()` with no arguments is a working pipeline rather than one that
fails at the first event. Replacing the default with a `default_factory` would
break the coupling and would also be a change to the shape of an existing
behaviour; it is not worth spending that on tidiness here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, Protocol

from placer.alignment import AlignedRead
from placer.core import consensus as consensus_module
from placer.core.te_classifier import TEAlignmentEvidence


@dataclass
class ReadSource:
    """Everything the scan needs from the input stage, as one value.

    These three were three loose positional parameters on `run_pipeline` with
    nothing naming them as a set, which is exactly the kind of implicit
    grouping that makes a seam hard to see. Boxing them turns "what core needs
    from the input side" into a single declaration you can grep for.

    `reads` IS ALREADY GATED and it is an `Iterator`, not a list. Both matter:
    the gate belongs to the input stage (`placer/io/gate.py`), and holding a
    list here would undo the streaming property that makes a whole-genome scan
    affordable -- each `AlignedRead` carries its full sequence.

    `fetch_local` is separate from `reads` because the two access patterns are
    genuinely different: the scan streams the file once in order, and the local
    stages then re-fetch small intervals around candidates, which needs an
    index. The C++ opens the file twice for the same reason.
    """

    reads: Iterator[AlignedRead]
    chromosome_name: Callable[[int], str]
    fetch_local: Callable[[str, int, int], list[AlignedRead]]


class TsdDetection(Protocol):
    """What the pipeline reads off a TSD detection.

    `placer.core.tsd.TsdDetection` satisfies it; so does any record with these
    five fields, which is the point -- a caller can supply its own detector
    without this module importing one.
    """

    type: str
    length: int
    sequence: str
    bg_p: float
    mismatches: int


@dataclass
class StageHooks:
    """The four external dependencies, injected rather than imported.

    Making them parameters is what lets the whole pipeline run on literals in a
    test: a reference is a callable returning a string, the TE alignment is a
    callable returning evidence, and the consensus is the callable
    `placer/core/consensus.py` argues must be chosen deliberately.

    It is also honest about what the pipeline actually needs from each -- the TE
    library is a function from an insert sequence to evidence, nothing more.
    """

    #: (chrom, start, end) -> uppercased reference bases, or "".
    fetch_reference: Callable[[str, int, int], str] = lambda chrom, start, end: ""
    #: insert sequence -> TE alignment evidence.
    align_insert: Callable[[str], TEAlignmentEvidence] = lambda seq: TEAlignmentEvidence()
    #: The same, for a whole bin at once: one evidence per sequence, in order.
    #:
    #: Optional, and `align_insert` stays the definition of what an alignment
    #: is. This exists because the real aligner is an external process whose
    #: start-up costs more than a few-hundred-base search, so seeing a bin's
    #: inserts together lets it overlap those start-ups. It must return
    #: exactly what `[align_insert(s) for s in seqs]` would; `None` means the
    #: bin loop calls `align_insert` per insert.
    align_inserts: Callable[[list[str]], list[TEAlignmentEvidence]] | None = None
    #: event strings -> one consensus sequence.
    consensus_fn: Callable[[list[str]], str] = consensus_module.single_sequence_consensus
    #: (chrom, bp_left, bp_right, insert_seq) -> a TSD detection, or None.
    #:
    #: Typed by what this module READS off the result (see `TsdDetection`)
    #: rather than as `object`, which is what it was: five attribute reads
    #: on a value the checker knew nothing about. A Protocol keeps the
    #: reason `object` was there -- importing `placer.core.tsd` for its
    #: dataclass would put the reference-fetching half of the tool in front
    #: of a pipeline that takes its reference as a callable.
    #:
    #: `insert_seq` is not redundant with the breakpoints. When the evidence
    #: is a CIGAR `I` the aligner collapses both breakpoints onto one
    #: coordinate, so bp_left == bp_right and the reference alone carries no
    #: trace of the duplication -- it exists only in the read. The detector
    #: needs the inserted bases to find it.
    detect_tsd: Callable[[str, int, int, str], TsdDetection | None] | None = None
