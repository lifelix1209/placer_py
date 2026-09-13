from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


@dataclass(frozen=True)
class InsertionCandidate:
    chrom: str
    pos: int
    end: int
    candidate_id: str
    svlen: int
    filter: str
    support: int
    stdev_pos: Optional[float] = None
    stdev_len: Optional[float] = None
    coverage: str = ""
    strand: str = ""
    alt_sequence: Optional[str] = None
    genotype: str = ""
    source: str = "sniffles2"

    @property
    def pos0(self) -> int:
        return max(0, self.pos - 1)

    @property
    def abs_svlen(self) -> int:
        return abs(int(self.svlen))


@dataclass(frozen=True)
class BamEvidence:
    raw_cigar_insert_reads: int = 0
    max_raw_cigar_insert_len: int = 0
    split_read_support: int = 0
    left_clip_reads: int = 0
    right_clip_reads: int = 0
    ref_span_reads: int = 0
    low_mapq_ref_span_reads: int = 0
    forward_reads: int = 0
    reverse_reads: int = 0
    local_depth: int = 0

    def to_tsv_fields(self) -> dict[str, str]:
        return {
            "raw_cigar_insert_reads": str(self.raw_cigar_insert_reads),
            "max_raw_cigar_insert_len": str(self.max_raw_cigar_insert_len),
            "split_read_support": str(self.split_read_support),
            "left_clip_reads": str(self.left_clip_reads),
            "right_clip_reads": str(self.right_clip_reads),
            "ref_span_reads": str(self.ref_span_reads),
            "low_mapq_ref_span_reads": str(self.low_mapq_ref_span_reads),
            "forward_reads": str(self.forward_reads),
            "reverse_reads": str(self.reverse_reads),
            "local_depth": str(self.local_depth),
        }


@dataclass(frozen=True)
class SequenceFeatures:
    length: int = 0
    gc_fraction: float = 0.0
    entropy: float = 0.0
    low_complexity_fraction: float = 0.0

    def to_tsv_fields(self) -> dict[str, str]:
        return {
            "sequence_len": str(self.length),
            "sequence_gc": _fmt_float(self.gc_fraction),
            "sequence_entropy": _fmt_float(self.entropy),
            "sequence_low_complexity": _fmt_float(self.low_complexity_fraction),
        }


@dataclass(frozen=True)
class TeHit:
    family: str = ""
    subfamily: str = ""
    identity: float = 0.0
    query_coverage: float = 0.0
    orientation: str = ""

    def to_tsv_fields(self) -> dict[str, str]:
        return {
            "family": self.family,
            "subfamily": self.subfamily,
            "te_identity": _fmt_float(self.identity),
            "te_query_coverage": _fmt_float(self.query_coverage),
            "te_orientation": self.orientation,
        }


@dataclass(frozen=True)
class TeExplanation:
    candidate_id: str
    label: str
    te_posterior: float
    non_te_posterior: float
    artifact_posterior: float
    lfdr: float
    te_hit: TeHit
    sequence_features: SequenceFeatures
    bam_evidence: BamEvidence
    discriminator: Optional[Any] = None

    def to_tsv_fields(self) -> dict[str, str]:
        row = {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "te_posterior": _fmt_float(self.te_posterior),
            "non_te_posterior": _fmt_float(self.non_te_posterior),
            "artifact_posterior": _fmt_float(self.artifact_posterior),
            "lfdr": _fmt_float(self.lfdr),
        }
        row.update(self.te_hit.to_tsv_fields())
        row.update(self.sequence_features.to_tsv_fields())
        row.update(self.bam_evidence.to_tsv_fields())
        if self.discriminator is not None:
            row.update(self.discriminator.to_tsv_fields())
        return row
