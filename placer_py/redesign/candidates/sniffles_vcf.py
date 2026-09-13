from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pysam

from placer_py.redesign.models import InsertionCandidate


def _first(value, default=None):
    if value is None:
        return default
    if isinstance(value, tuple):
        return value[0] if value else default
    return value


def _alt_sequence(alts: tuple[str, ...] | None) -> str | None:
    if not alts:
        return None
    alt = alts[0]
    if alt.startswith("<") and alt.endswith(">"):
        return None
    if "[" in alt or "]" in alt:
        return None
    return alt


def _sample_gt(record) -> str:
    if not record.samples:
        return ""
    sample = next(iter(record.samples.values()))
    gt = sample.get("GT")
    if gt is None:
        return ""
    return "/".join("." if allele is None else str(allele) for allele in gt)


def _filter_string(record) -> str:
    keys = list(record.filter.keys())
    if not keys:
        return "PASS"
    return ";".join(keys)


def _coverage_string(record) -> str:
    coverage_raw = _info_get(record, "COVERAGE")
    if not coverage_raw:
        return ""
    parts = []
    for value in coverage_raw:
        value_float = float(value)
        parts.append(str(int(value_float)) if value_float.is_integer() else str(value))
    return ",".join(parts)


def _info_get(record, key: str, default=None):
    try:
        return record.info.get(key, default)
    except (KeyError, ValueError):
        return default


def iter_sniffles_insertions(vcf_path: str | Path) -> Iterator[InsertionCandidate]:
    with pysam.VariantFile(str(vcf_path)) as vcf:
        for record in vcf.fetch():
            if _info_get(record, "SVTYPE") != "INS":
                continue
            svlen = int(_first(_info_get(record, "SVLEN"), 0))
            support = int(_first(_info_get(record, "SUPPORT"), 0))
            stdev_pos = _first(_info_get(record, "STDEV_POS"))
            stdev_len = _first(_info_get(record, "STDEV_LEN"))
            yield InsertionCandidate(
                chrom=record.chrom,
                pos=int(record.pos),
                end=int(_first(_info_get(record, "END"), record.pos)),
                candidate_id=record.id or f"{record.chrom}:{record.pos}:INS",
                svlen=svlen,
                filter=_filter_string(record),
                support=support,
                stdev_pos=float(stdev_pos) if stdev_pos is not None else None,
                stdev_len=float(stdev_len) if stdev_len is not None else None,
                coverage=_coverage_string(record),
                strand=str(_first(_info_get(record, "STRAND"), "")),
                alt_sequence=_alt_sequence(record.alts),
                genotype=_sample_gt(record),
            )
