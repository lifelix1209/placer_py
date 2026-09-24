"""Gathering the input-stage facts the VCF needs, once, after the run.

This is the only module that knows both that contigs live in a BAM header and
that a VCF needs one reference base per record. It runs AFTER the pipeline, not
inside it: a few thousand single-base fetches at the end costs nothing, while
resolving an anchor per candidate inside the bin loop would put reference I/O
back in the middle of the algorithm -- which is what the three-stage split just
took out.

WHY THE SAMPLE NAME FALLS BACK TO THE FILENAME. `@RG SM` is conventional, not
guaranteed: the example dataset's own BAM has no read groups at all, so the
fallback is exercised on every test run rather than being dead code waiting for
an unusual input. A BAM whose read groups DISAGREE about SM is a real thing
(two samples merged into one file), and it warns rather than picking silently,
in the same register as the environment-variable warnings in `main.py`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import Callable

from placer_py.ledger import FinalCall
from placer_py.report.context import Contig, ReportContext
from placer_py.report.vcf import anchor_index


def contigs_from_reader(reader) -> tuple[Contig, ...]:
    """Every reference sequence in the BAM header, in header order.

    Header order IS the sort order a VCF must be in, which is why this keeps
    the sequence rather than returning a dict.
    """
    return tuple(Contig(name=reader.chromosome_name(tid),
                        length=reader.chromosome_length(tid))
                 for tid in range(reader.chromosome_count()))


def sample_name_from_reader(reader, fallback_path: str) -> str:
    """`@RG SM`, or the BAM's basename with its extension stripped."""
    names: list[str] = []
    try:
        header = reader.header_dict()
    except (AttributeError, ValueError, OSError):
        header = {}
    for group in header.get("RG", []) or []:
        name = str(group.get("SM", "")).strip()
        if name and name not in names:
            names.append(name)
    if len(names) > 1:
        print(f"[PLACER] BAM read groups disagree about SM {names}; "
              f"using {names[0]!r} for the VCF sample column", file=sys.stderr)
    if names:
        return names[0]
    return os.path.splitext(os.path.basename(fallback_path))[0] or "SAMPLE"


def collect_anchor_bases(calls: Iterable[FinalCall],
                         fetch_reference: Callable[[str, int, int], str]
                         ) -> dict[tuple[str, int], str]:
    """One reference base per distinct anchor position.

    Keyed by `(chrom, 0-based index)` rather than by call, so two calls at one
    locus share a fetch -- a key collision here is correct, not a bug.
    """
    bases: dict[tuple[str, int], str] = {}
    for call in calls:
        if not call.chrom:
            continue
        key = (call.chrom, anchor_index(call))
        if key in bases:
            continue
        window = fetch_reference(call.chrom, key[1], key[1] + 1)
        if window:
            bases[key] = window[0].upper()
    return bases


def build_report_context(reader, fetch_reference, config, result,
                         file_date: str | None = None) -> ReportContext:
    """Everything `render_vcf` needs, assembled from the open inputs."""
    calls = list(result.final_calls) + list(result.structural_calls)
    return ReportContext(
        sample_name=sample_name_from_reader(reader, config.bam_path),
        reference_path=os.path.abspath(config.reference_fasta_path),
        contigs=contigs_from_reader(reader),
        anchor_bases=collect_anchor_bases(calls, fetch_reference),
        file_date=file_date or "")
