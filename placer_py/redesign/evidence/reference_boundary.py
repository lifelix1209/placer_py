from __future__ import annotations

from dataclasses import dataclass

import pysam

from placer_py.redesign.models import InsertionCandidate


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _upper(seq: str | None) -> str:
    return (seq or "").upper()


def _has_only_acgt(seq: str) -> bool:
    return bool(seq) and all(base in "ACGT" for base in seq)


def _contig_aliases(chrom: str) -> tuple[str, ...]:
    if not chrom:
        return ()
    aliases = [chrom]
    if chrom.startswith("chr") and len(chrom) > 3:
        aliases.append(chrom[3:])
    else:
        aliases.append(f"chr{chrom}")
    return tuple(dict.fromkeys(aliases))


def resolve_reference_chrom(reference: pysam.FastaFile, chrom: str) -> str:
    names = set(reference.references)
    for alias in _contig_aliases(chrom):
        if alias in names:
            return alias
    return ""


def _fetch(reference: pysam.FastaFile, chrom: str, start: int, end: int) -> str:
    if not chrom or end <= start:
        return ""
    length = reference.get_reference_length(chrom)
    qstart = max(0, min(start, length))
    qend = max(qstart, min(end, length))
    if qend <= qstart:
        return ""
    return _upper(reference.fetch(chrom, qstart, qend))


def _background_occurrence_fraction(region: str, motif: str) -> float:
    if not region or not motif or len(region) < len(motif):
        return 1.0
    total = 0
    hits = 0
    for start in range(0, len(region) - len(motif) + 1):
        total += 1
        if region[start : start + len(motif)] == motif:
            hits += 1
    return hits / float(max(1, total))


def _n_rich(seq: str) -> tuple[bool, float, int]:
    if not seq:
        return False, 0.0, 0
    n_count = 0
    curr_run = 0
    max_run = 0
    for base in seq:
        if base == "N":
            n_count += 1
            curr_run += 1
            max_run = max(max_run, curr_run)
        else:
            curr_run = 0
    fraction = n_count / float(len(seq))
    return max_run >= 20 or fraction >= 0.50, fraction, max_run


def _longest_left_anchor(query: str, left_context: str, max_anchor_len: int) -> int:
    limit = min(len(query), len(left_context), max_anchor_len)
    for size in range(limit, 0, -1):
        if query.startswith(left_context[-size:]):
            return size
    return 0


def _longest_right_anchor(query: str, right_context: str, max_anchor_len: int) -> int:
    limit = min(len(query), len(right_context), max_anchor_len)
    for size in range(limit, 0, -1):
        if query.endswith(right_context[:size]):
            return size
    return 0


def _trim_core(query: str, left_anchor_len: int, right_anchor_len: int) -> str:
    left = max(0, min(left_anchor_len, len(query)))
    right = max(0, min(right_anchor_len, len(query) - left))
    end = len(query) - right
    return query[left:end]


@dataclass(frozen=True)
class ReferenceEvidence:
    reference_available: bool = False
    reference_qc: str = "REFERENCE_NOT_EVALUATED"
    resolved_chrom: str = ""
    breakpoint0: int = -1
    right_breakpoint0: int = -1
    left_anchor_len: int = 0
    right_anchor_len: int = 0
    left_anchor_identity: float = 0.0
    right_anchor_identity: float = 0.0
    remap_qc: str = "REFERENCE_REMAP_NOT_EVALUATED"
    insert_core_sequence: str | None = None
    tsd_type: str = "NONE"
    tsd_len: int = 0
    tsd_sequence: str = ""
    tsd_bg_p: float = 1.0
    tsd_significant: bool = False
    tsd_qc: str = "NO_TSD"
    boundary_type: str = "REJECT"
    boundary_len: int = 0
    boundary_qc: str = "BOUNDARY_NOT_EVALUATED"
    boundary_score: float = -2.0
    reference_n_fraction: float = 0.0
    reference_max_n_run: int = 0

    @property
    def has_left_flank(self) -> bool:
        return self.left_anchor_len > 0

    @property
    def has_right_flank(self) -> bool:
        return self.right_anchor_len > 0

    @property
    def pair_valid(self) -> bool:
        return self.has_left_flank and self.has_right_flank and bool(self.insert_core_sequence)


def collect_reference_evidence(
    candidate: InsertionCandidate,
    reference: pysam.FastaFile,
    inserted_sequence: str | None = None,
    flank_window: int = 150,
    tsd_min_len: int = 3,
    tsd_max_len: int = 50,
    tsd_bg_p_max: float = 0.05,
    min_anchor_len: int = 4,
) -> ReferenceEvidence:
    query = _upper(inserted_sequence if inserted_sequence is not None else candidate.alt_sequence)
    resolved = resolve_reference_chrom(reference, candidate.chrom)
    if not resolved:
        return ReferenceEvidence(reference_qc="REFERENCE_CONTIG_MISSING", insert_core_sequence=query or None)

    left_bp = max(0, int(candidate.pos))
    right_bp = max(left_bp, int(candidate.end) if int(candidate.end) > int(candidate.pos) else left_bp)
    flank = max(10, int(flank_window))
    left_context = _fetch(reference, resolved, left_bp - flank, left_bp)
    right_context = _fetch(reference, resolved, right_bp, right_bp + flank)
    combined_context = left_context + right_context
    is_n_rich, n_fraction, max_n_run = _n_rich(combined_context)
    reference_qc = "REFERENCE_N_RICH" if is_n_rich else "PASS_REFERENCE_CONTEXT"

    max_anchor_len = min(max(0, len(query)), flank)
    left_anchor = _longest_left_anchor(query, left_context, max_anchor_len) if query else 0
    right_anchor = _longest_right_anchor(query, right_context, max_anchor_len) if query else 0
    if left_anchor < min_anchor_len:
        left_anchor = 0
    if right_anchor < min_anchor_len:
        right_anchor = 0
    insert_core = _trim_core(query, left_anchor, right_anchor) if query else None

    if left_anchor and right_anchor:
        remap_qc = "PASS_REMAP_BILATERAL"
    elif left_anchor:
        remap_qc = "PASS_REMAP_LEFT"
    elif right_anchor:
        remap_qc = "PASS_REMAP_RIGHT"
    else:
        remap_qc = "NO_REFERENCE_EDGE_REMAP"

    tsd_type = "NONE"
    tsd_len = 0
    tsd_sequence = ""
    tsd_bg_p = 1.0
    tsd_significant = False
    tsd_qc = "NO_TSD"
    min_len = max(1, int(tsd_min_len))
    max_len = max(min_len, int(tsd_max_len))
    for size in range(max_len, min_len - 1, -1):
        if left_bp - size < 0:
            continue
        left = _fetch(reference, resolved, left_bp - size, left_bp)
        right = _fetch(reference, resolved, right_bp, right_bp + size)
        if len(left) != size or len(right) != size:
            continue
        if not _has_only_acgt(left) or left != right:
            continue
        bg_region = _fetch(reference, resolved, left_bp - flank, right_bp + flank)
        bg_p = _background_occurrence_fraction(bg_region, left)
        edge_supported = not query or query.startswith(left) or query.endswith(left)
        tsd_type = "DUP" if bg_p <= tsd_bg_p_max else "UNCERTAIN"
        tsd_len = size
        tsd_sequence = left
        tsd_bg_p = bg_p
        tsd_significant = bg_p <= tsd_bg_p_max and edge_supported
        tsd_qc = "PASS_TSD_DUP" if tsd_significant else "TSD_BACKGROUND_AMBIGUOUS"
        break

    boundary_type = "REJECT"
    boundary_len = 0
    boundary_qc = "BOUNDARY_REFERENCE_CONTEXT_ONLY"
    boundary_score = -0.25 if reference_qc == "PASS_REFERENCE_CONTEXT" else -2.0
    if reference_qc == "PASS_REFERENCE_CONTEXT":
        if tsd_significant:
            boundary_type = "TSD"
            boundary_len = tsd_len
            boundary_qc = "PASS_BOUNDARY_TSD"
            boundary_score = 1.0
        elif left_anchor and right_anchor:
            if not insert_core:
                boundary_qc = "BOUNDARY_REMAP_NO_INSERT_CORE"
                boundary_type = "REJECT"
                boundary_len = 0
                boundary_score = -0.25
            else:
                link_delta = right_bp - left_bp
                if link_delta == 0:
                    boundary_type = "BLUNT"
                    boundary_len = 0
                    boundary_qc = "PASS_BOUNDARY_BLUNT"
                    boundary_score = 1.0
                elif link_delta <= max_len:
                    boundary_type = "SMALL_DEL"
                    boundary_len = link_delta
                    boundary_qc = "PASS_BOUNDARY_SMALL_DEL"
                    boundary_score = 1.0
                else:
                    boundary_type = "NONCANONICAL"
                    boundary_len = link_delta
                    boundary_qc = "PASS_BOUNDARY_NONCANONICAL_CONSISTENT"
                    boundary_score = 0.25
        elif left_anchor or right_anchor:
            boundary_qc = "BOUNDARY_ONE_SIDED_REMAP"

    return ReferenceEvidence(
        reference_available=True,
        reference_qc=reference_qc,
        resolved_chrom=resolved,
        breakpoint0=left_bp,
        right_breakpoint0=right_bp,
        left_anchor_len=left_anchor,
        right_anchor_len=right_anchor,
        left_anchor_identity=1.0 if left_anchor else 0.0,
        right_anchor_identity=1.0 if right_anchor else 0.0,
        remap_qc=remap_qc,
        insert_core_sequence=insert_core,
        tsd_type=tsd_type,
        tsd_len=tsd_len,
        tsd_sequence=tsd_sequence,
        tsd_bg_p=tsd_bg_p,
        tsd_significant=tsd_significant,
        tsd_qc=tsd_qc,
        boundary_type=boundary_type,
        boundary_len=boundary_len,
        boundary_qc=boundary_qc,
        boundary_score=boundary_score,
        reference_n_fraction=n_fraction,
        reference_max_n_run=max_n_run,
    )


def reference_evidence_tsv_fields(evidence: ReferenceEvidence) -> dict[str, str]:
    return {
        "reference_qc": evidence.reference_qc,
        "resolved_reference_chrom": evidence.resolved_chrom,
        "reference_breakpoint0": str(evidence.breakpoint0),
        "reference_right_breakpoint0": str(evidence.right_breakpoint0),
        "left_ref_anchor_len": str(evidence.left_anchor_len),
        "right_ref_anchor_len": str(evidence.right_anchor_len),
        "left_ref_anchor_identity": _fmt_float(evidence.left_anchor_identity),
        "right_ref_anchor_identity": _fmt_float(evidence.right_anchor_identity),
        "remap_qc": evidence.remap_qc,
        "insert_core_len": str(len(evidence.insert_core_sequence or "")),
        "tsd_type": evidence.tsd_type,
        "tsd_len": str(evidence.tsd_len),
        "tsd_sequence": evidence.tsd_sequence,
        "tsd_bg_p": _fmt_float(evidence.tsd_bg_p),
        "tsd_qc": evidence.tsd_qc,
        "reference_n_fraction": _fmt_float(evidence.reference_n_fraction),
        "reference_max_n_run": str(evidence.reference_max_n_run),
    }
