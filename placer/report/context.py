"""The facts the VCF needs that no stage computes.

A VCF carries three things a PLACER run never has to know: which contigs exist
and how long they are, what the sample is called, and the single reference base
each record is anchored on. All three are properties of the INPUT, and the
output stage must stay pure, so they are gathered once after the run and handed
over as data. `placer/io/report_context.py` is what builds one.

WHY THE ANCHOR BASES ARE A RESOLVED MAPPING rather than a fetch callable. The
alternative -- give `render_vcf` a `fetch_reference` function -- reads better
until you notice it puts a live pysam handle inside the one layer that has to
be testable on literals, and makes the renderer impure in a suite whose runner
has no `monkeypatch`. Resolving first costs a few thousand one-base fetches at
the end of a run instead of one per candidate inside the loop.

THE KEY IS `(chrom, 0-based index)`, NOT A CALL INDEX, and that is deliberate:
two calls at one locus legitimately share a reference base, so a key collision
is correct rather than a bug. A missing key renders as `N` -- the spec's own
unknown base -- and never as the empty string, which would make the VCF
unparseable.

NONE OF THIS IS ON `FinalCall`. Adding `ref_anchor_base` there would give the
Python record a field the C++ struct does not have, for a reason that has
nothing to do with the algorithm, and every reviewer diffing the two would
have to be told why. A `FinalCall()` built in a unit test would also carry a
base that looks like data and is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contig:
    """One `##contig=<ID=...,length=...>` line."""

    name: str
    length: int


@dataclass(frozen=True)
class ReportContext:
    """Everything the writers need beyond the run itself.

    Every field has a default, so `ReportContext()` is valid and
    `render_vcf(result)` produces a contig-less, `N`-anchored but otherwise
    correct VCF. That is what lets the whole output surface be tested with no
    BAM and no reference.
    """

    sample_name: str = "SAMPLE"
    reference_path: str = ""
    contigs: tuple[Contig, ...] = ()
    #: (chrom, 0-based anchor index) -> one uppercase base.
    anchor_bases: Mapping[tuple[str, int], str] = field(default_factory=dict)
    #: "" omits `##fileDate`, so a rendered VCF can be byte-stable in a test.
    file_date: str = ""


def contig_ranks(context: ReportContext) -> dict[str, int]:
    """Sort keys for the declared contigs, in header order.

    A VCF must be coordinate-sorted, and "coordinate order" means the header's
    contig order rather than anything lexical -- `chr10` precedes `chr2` in a
    reference dictionary and follows it in a string sort. Contigs absent from
    the header are ranked after every declared one by the caller, so a VCF
    rendered without a context is still internally consistent.
    """
    return {contig.name: index for index, contig in enumerate(context.contigs)}
