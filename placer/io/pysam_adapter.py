"""Adapting a `pysam.AlignedSegment` onto this package's own read view.

WHY THIS IS ITS OWN MODULE rather than a function on `AlignedRead`. The read
view is the contract between the input stage and the algorithm, so it lives at
the top level where both can see it (`placer/alignment.py`); the adapter
that builds one FROM A FOREIGN RECORD is input-stage work and belongs here.
Keeping them apart is what lets `placer/core/` hold the type without the
package that produces it.

Note this module does not import pysam either. `read_from_pysam` is duck-typed
on the record -- it reads eight attributes and nothing else -- which is why the
whole algorithm can be driven from literals in a test.
"""

from __future__ import annotations

from placer.alignment import AlignedRead


def read_from_pysam(record) -> AlignedRead:
    """Adapt a `pysam.AlignedSegment` onto :class:`AlignedRead`.

    Imported lazily by the caller: the decision layer needs no BAM at all, and
    making pysam a hard import would put a compiled dependency in front of the
    part of the pipeline that has none.
    """
    tags = dict(record.get_tags() or [])
    return AlignedRead(
        qname=record.query_name or "",
        flag=int(record.flag),
        tid=int(record.reference_id),
        pos=int(record.reference_start if record.reference_start is not None else 0),
        mapq=int(record.mapping_quality),
        cigar=[(int(op), int(length)) for op, length in (record.cigartuples or [])],
        seq=record.query_sequence or "",
        tags=tags,
    )
