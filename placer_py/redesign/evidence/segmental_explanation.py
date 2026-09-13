"""Segmental explanation of an inserted sequence.

Instead of a single flat TE k-mer coverage number, partition the inserted
sequence into interpretable segments -- TE core, poly(A)/poly(T) tail,
low-complexity/simple-repeat, and unexplained -- and expose graded,
composition-robust statistics for the discriminator.

This addresses two failure modes of the previous flat k-mer hit:

* identity was reported as a constant ``1.0`` for every hit, carrying no
  graded information;
* coverage was the fraction of query k-mers found *anywhere* in one reference,
  so low-complexity and poly(A) sequence inflated the TE signal.

The design follows the redesign notes in
``sources/research_te_insertion_discriminator_20260701.md``: replace flat TE
k-mer coverage with a segmental sequence explanation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from placer_py.redesign.evidence.sequence import _family_parts, best_exact_te_hit, reverse_complement


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class SegmentalExplanation:
    length: int = 0
    te_core_fraction: float = 0.0
    te_core_contiguity: float = 0.0
    polya_fraction: float = 0.0
    low_complexity_fraction: float = 0.0
    unexplained_fraction: float = 1.0
    family_margin: float = 0.0
    identity_proxy: float = 0.0
    query_coverage: float = 0.0
    best_family: str = ""
    best_subfamily: str = ""
    orientation: str = ""
    qc: str = "NO_INSERT_SEQUENCE"


def _homopolymer_and_repeat_mask(seq: str, window: int = 16, dominance: float = 0.9) -> list[bool]:
    """Mark positions inside a low-complexity window.

    A window is low-complexity when its two most frequent bases cover at least
    ``dominance`` of the window. This captures homopolymers and dinucleotide
    tandem repeats without a separate detector for each.
    """
    length = len(seq)
    mask = [False] * length
    if length == 0:
        return mask
    win = min(window, length)
    counts: Counter[str] = Counter(seq[:win])
    threshold = dominance * win

    def _dominant(counter: Counter[str]) -> bool:
        top_two = sum(count for _, count in counter.most_common(2))
        return top_two >= threshold

    for start in range(0, length - win + 1):
        if start > 0:
            counts[seq[start - 1]] -= 1
            counts[seq[start + win - 1]] += 1
        if _dominant(counts):
            for pos in range(start, start + win):
                mask[pos] = True
    return mask


def _terminal_run_fraction(seq: str, base: str, purity: float = 0.8) -> int:
    """Length of the maximal terminal run (from the right) that stays ``base``-rich."""
    length = len(seq)
    matched = 0
    best_len = 0
    for offset in range(1, length + 1):
        if seq[length - offset] == base:
            matched += 1
        if matched >= purity * offset:
            best_len = offset
    return best_len


def _polya_mask(seq: str) -> list[bool]:
    """Mark a 3' poly(A) or 5' poly(T) tail (retrotransposition hallmark)."""
    length = len(seq)
    mask = [False] * length
    if length == 0:
        return mask
    right_a = _terminal_run_fraction(seq, "A")
    right_t = _terminal_run_fraction(seq, "T")
    tail = max(right_a, right_t)
    if tail >= 6:
        for pos in range(length - tail, length):
            mask[pos] = True
    # A 5' poly(T) tail appears when a poly(A) insertion is seen on the minus
    # strand; mirror the terminal scan on the reversed sequence.
    rev = seq[::-1]
    left_a = _terminal_run_fraction(rev, "A")
    left_t = _terminal_run_fraction(rev, "T")
    head = max(left_a, left_t)
    if head >= 6:
        for pos in range(0, head):
            mask[pos] = True
    return mask


def _coverage_for_best_family(
    query: str,
    index: dict[str, set[str]],
    k: int,
) -> tuple[list[bool], str, float]:
    """Return per-position k-mer coverage for the best family plus family margin.

    Coverage is oriented so the returned mask is in forward-query coordinates.
    """
    length = len(query)
    if length < k:
        return [False] * length, "", 0.0

    best_orientation = ""
    best_name = ""
    best_count = 0
    best_total = 0
    second_count = 0
    for orientation, oriented in (("+", query), ("-", reverse_complement(query))):
        kmers = {oriented[start : start + k] for start in range(0, length - k + 1)}
        counts: Counter[str] = Counter()
        for kmer in kmers:
            for name in index.get(kmer, ()):
                counts[name] += 1
        if not counts:
            continue
        ranked = counts.most_common(2)
        name, count = ranked[0]
        if count > best_count:
            best_orientation = orientation
            best_name = name
            best_count = count
            best_total = len(kmers)
            second_count = ranked[1][1] if len(ranked) > 1 else 0

    if not best_name:
        return [False] * length, "", 0.0

    oriented = query if best_orientation == "+" else reverse_complement(query)
    covered = [False] * length
    for start in range(0, length - k + 1):
        if best_name in index.get(oriented[start : start + k], ()):
            for pos in range(start, start + k):
                covered[pos] = True
    if best_orientation == "-":
        covered = covered[::-1]

    margin = (best_count - second_count) / float(max(1, best_total))
    return covered, best_name, _clamp(margin, 0.0, 1.0)


def _longest_run(mask: list[bool]) -> int:
    best = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _classify(
    te_core: float,
    polya: float,
    low_complexity: float,
    unexplained: float,
) -> str:
    if te_core >= 0.5:
        return "TE_CORE_DOMINANT"
    if te_core >= 0.2:
        return "TE_CORE_PARTIAL"
    if polya >= 0.5:
        return "POLYA_TAIL_DOMINANT"
    if low_complexity >= 0.5:
        return "LOW_COMPLEXITY_DOMINANT"
    if unexplained >= 0.5:
        return "UNEXPLAINED_DOMINANT"
    return "MIXED_SEGMENTS"


def build_segmental_explanation(
    seq: str | None,
    references: dict[str, str],
    index: dict[str, set[str]],
    k: int = 17,
) -> SegmentalExplanation:
    if not seq:
        return SegmentalExplanation()
    seq = seq.upper()
    length = len(seq)

    low_complexity_mask = _homopolymer_and_repeat_mask(seq)
    polya_mask = _polya_mask(seq)

    if length < k:
        hit = best_exact_te_hit(seq, references)
        core = _clamp(hit.query_coverage, 0.0, 1.0)
        polya = sum(polya_mask) / float(length)
        low_complexity = sum(1 for m in low_complexity_mask if m) / float(length)
        unexplained = _clamp(1.0 - core - polya - low_complexity, 0.0, 1.0)
        return SegmentalExplanation(
            length=length,
            te_core_fraction=core,
            te_core_contiguity=1.0 if core > 0 else 0.0,
            polya_fraction=polya,
            low_complexity_fraction=low_complexity,
            unexplained_fraction=unexplained,
            family_margin=0.0,
            identity_proxy=core,
            best_family=hit.family,
            best_subfamily=hit.subfamily,
            orientation=hit.orientation,
            qc=_classify(core, polya, low_complexity, unexplained),
        )

    covered, best_name, family_margin = _coverage_for_best_family(seq, index, k)
    family, subfamily = _family_parts(best_name) if best_name else ("", "")
    return _assemble_explanation(
        seq,
        covered,
        low_complexity_mask,
        polya_mask,
        family=family,
        subfamily=subfamily,
        family_margin=family_margin,
    )


def _assemble_explanation(
    seq: str,
    covered: list[bool],
    low_complexity_mask: list[bool],
    polya_mask: list[bool],
    *,
    family: str,
    subfamily: str,
    family_margin: float,
    identity_proxy_override: float | None = None,
    query_coverage_override: float | None = None,
) -> SegmentalExplanation:
    """Partition the sequence given a k-mer/alignment coverage mask."""
    length = len(seq)
    te_core_mask = [
        covered[pos] and not low_complexity_mask[pos] and not polya_mask[pos]
        for pos in range(length)
    ]
    te_core_positions = sum(1 for flag in te_core_mask if flag)
    polya_positions = sum(1 for flag in polya_mask if flag)
    low_complexity_positions = sum(
        1 for pos in range(length) if low_complexity_mask[pos] and not polya_mask[pos]
    )

    te_core_fraction = te_core_positions / float(length)
    polya_fraction = polya_positions / float(length)
    low_complexity_fraction = low_complexity_positions / float(length)
    unexplained_fraction = _clamp(
        1.0 - te_core_fraction - polya_fraction - low_complexity_fraction, 0.0, 1.0
    )

    longest_core_run = _longest_run(te_core_mask)
    contiguity = longest_core_run / float(te_core_positions) if te_core_positions else 0.0

    covered_positions = sum(1 for flag in covered if flag)
    if query_coverage_override is not None:
        query_coverage = query_coverage_override
    else:
        query_coverage = covered_positions / float(length) if length else 0.0
    if identity_proxy_override is not None:
        identity_proxy = identity_proxy_override
    else:
        first = next((pos for pos in range(length) if covered[pos]), 0)
        last = next((pos for pos in range(length - 1, -1, -1) if covered[pos]), 0)
        span = max(1, last - first + 1) if covered_positions else 1
        identity_proxy = covered_positions / float(span) if covered_positions else 0.0

    return SegmentalExplanation(
        length=length,
        te_core_fraction=te_core_fraction,
        te_core_contiguity=_clamp(contiguity, 0.0, 1.0),
        polya_fraction=polya_fraction,
        low_complexity_fraction=low_complexity_fraction,
        unexplained_fraction=unexplained_fraction,
        family_margin=_clamp(family_margin, 0.0, 1.0),
        identity_proxy=_clamp(identity_proxy, 0.0, 1.0),
        query_coverage=_clamp(query_coverage, 0.0, 1.0),
        best_family=family,
        best_subfamily=subfamily,
        orientation="",
        qc=_classify(te_core_fraction, polya_fraction, low_complexity_fraction, unexplained_fraction),
    )


def build_segmental_explanation_from_alignment(
    seq: str | None,
    *,
    identity: float,
    query_coverage: float,
    aln_query_start: int,
    aln_query_end: int,
    mapq: int = 60,
    family: str = "",
    subfamily: str = "",
) -> SegmentalExplanation:
    """Segmental explanation driven by a real TE alignment (e.g. minimap2 PAF).

    The aligned query span ``[aln_query_start, aln_query_end)`` defines a
    contiguous TE-covered block; composition masks then subtract poly(A) and
    low-complexity positions so those are not credited as TE core. Real
    alignment ``identity`` is used directly as the identity proxy, and mapping
    quality is a proxy for family specificity (cross-family margin).
    """
    if not seq:
        return SegmentalExplanation()
    seq = seq.upper()
    length = len(seq)
    low_complexity_mask = _homopolymer_and_repeat_mask(seq)
    polya_mask = _polya_mask(seq)

    covered = [False] * length
    if identity > 0.0 and query_coverage > 0.0:
        start = max(0, min(aln_query_start, length))
        end = max(start, min(aln_query_end, length))
        if end <= start:
            # Fall back to a coverage-proportional contiguous block from the 5' end.
            end = min(length, start + int(round(_clamp(query_coverage, 0.0, 1.0) * length)))
        for pos in range(start, end):
            covered[pos] = True

    family_margin = _clamp(mapq / 60.0, 0.0, 1.0)
    return _assemble_explanation(
        seq,
        covered,
        low_complexity_mask,
        polya_mask,
        family=family,
        subfamily=subfamily,
        family_margin=family_margin,
        identity_proxy_override=_clamp(identity, 0.0, 1.0),
        query_coverage_override=_clamp(query_coverage, 0.0, 1.0),
    )


def segmental_explanation_from_features(
    query_coverage: float,
    low_complexity_fraction: float,
    pass_alignment: bool,
    has_partial_alignment: bool = False,
) -> SegmentalExplanation:
    """Fallback explanation when the raw inserted sequence is unavailable.

    Derives a monotone, low-complexity-discounted core estimate from a
    precomputed ``TeHit`` coverage. Used by unit paths that construct evidence
    directly from a ``TeHit`` rather than from sequence.
    """
    coverage = _clamp(query_coverage, 0.0, 1.0)
    low_complexity = _clamp(low_complexity_fraction, 0.0, 1.0)
    if pass_alignment:
        core = coverage
    elif has_partial_alignment:
        core = 0.5 * coverage
    else:
        core = 0.0
    core = _clamp(core * (1.0 - 0.5 * low_complexity), 0.0, 1.0)
    unexplained = _clamp(1.0 - core - low_complexity, 0.0, 1.0)
    return SegmentalExplanation(
        length=0,
        te_core_fraction=core,
        te_core_contiguity=coverage,
        polya_fraction=0.0,
        low_complexity_fraction=low_complexity,
        unexplained_fraction=unexplained,
        family_margin=0.0,
        identity_proxy=coverage,
        query_coverage=coverage,
        qc="FALLBACK_FROM_TEHIT",
    )
