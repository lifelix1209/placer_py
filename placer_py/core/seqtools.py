"""
Sequence primitives shared by every stage that looks at bases.

Ported from the anonymous-namespace helpers of
`src/component/te_quick_classifier.cpp` and the composition model declared in
`include/pipeline.h`, pinned by `tests/test_19_seqtools.py`.

WHY THIS IS ONE MODULE AND NOT FIVE. The C++ duplicates these helpers into four
translation units -- `te_quick_classifier.cpp`, `dbscan_component_module.cpp`,
`insert_fragment_module.cpp` and `pipeline_breakpoint_helpers.inc` each define
their own `char_to_2bit`, `upper_acgt`, `max_homopolymer_run` and friends,
because C++ internal linkage makes that cheaper than a shared header. The
duplicates are byte-identical, so collapsing them here is the one place this
port deliberately does NOT mirror the C++ layout: four copies of a function is
four chances for them to drift, and nothing in the behaviour depends on their
being separate.

THE COMPOSITION MODEL is the part worth reading. `identity x coverage` says how
well an insert matched one library entry; these four numbers describe the insert
on its own terms, which is what separates a real but unnamed TE insertion from a
low-complexity or tandem artifact that happens to hit something in the library.
They are library-INDEPENDENT by construction, so they cannot inherit the
library's blind spots -- which is exactly the failure mode the README flags for
the hp-tier thresholds fitted against another caller's calls.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

#: k for the two Jensen-Shannon composition features.
SEQ_MODEL_JSD_K5 = 5
SEQ_MODEL_JSD_K6 = 6
#: k for the containment feature. 9 is long enough to be specific and short
#: enough that a diverged copy still shares some.
SEQ_MODEL_CONTAINMENT_K = 9
#: dust-style low-complexity window, and the top-2 mass that flags it.
LOW_COMPLEXITY_WINDOW = 16
LOW_COMPLEXITY_TOP2_FRACTION = 0.90
#: longest tandem period considered by `sequence_tandem_fraction`.
MAX_TANDEM_PERIOD = 12

_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def char_to_2bit(base: str) -> int:
    """0..3 for ACGT, 4 for anything else (including lower case)."""
    return _CODE.get(base, 4)


def is_acgt_base(base: str) -> bool:
    return base in _CODE


def upper_ascii(seq: str) -> str:
    return seq.upper()


#: The C++ name for the same operation. It does NOT mask non-ACGT bases despite
#: the name -- kept verbatim so a reader diffing the two sides is not misled.
upper_acgt = upper_ascii


def complement_base(base: str) -> str:
    return _COMPLEMENT.get(base, "N")


def reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT.get(c, "N") for c in reversed(seq))


def take_header_token(header: str) -> str:
    """First whitespace-delimited token, or the whole string when there is none."""
    stripped = header.strip()
    if not stripped:
        return header
    return stripped.split()[0]


def for_each_valid_kmer(seq: str, k: int) -> Iterator[tuple[int, int]]:
    """Yield `(start, packed_key)` for every k-mer free of non-ACGT bases.

    A non-ACGT base resets the rolling key, so k-mers spanning an N are skipped
    rather than silently recoded -- the same contract as the C++ template.
    """
    if k <= 0 or len(seq) < k:
        return
    mask = (1 << (2 * k)) - 1 if k < 32 else (1 << 64) - 1
    key = 0
    valid = 0
    for i, base in enumerate(seq):
        code = _CODE.get(base, 4)
        if code > 3:
            key = 0
            valid = 0
            continue
        key = ((key << 2) | code) & mask
        if valid < k:
            valid += 1
        if valid >= k:
            yield i - k + 1, key


def reverse_complement_kmer_key(key: int, k: int) -> int:
    rc = 0
    for _ in range(k):
        rc = (rc << 2) | (3 - (key & 3))
        key >>= 2
    return rc


def canonical_kmer_key(key: int, k: int) -> int:
    """Strand-folded key: the smaller of the k-mer and its reverse complement."""
    return min(key, reverse_complement_kmer_key(key, k))


def max_homopolymer_run(seq: str) -> int:
    if not seq:
        return 0
    best = 1
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best


def at_fraction(seq: str) -> float:
    at = 0
    total = 0
    for c in seq:
        if c not in _CODE:
            continue
        total += 1
        if c in ("A", "T"):
            at += 1
    return (at / total) if total > 0 else 0.0


def shannon_entropy_acgt(seq: str) -> float:
    """Shannon entropy of the ACGT composition, in bits. Same as
    :func:`sequence_entropy_bits`; both C++ names are kept because the two
    translation units that define them are pinned by different tests."""
    counts = [0, 0, 0, 0]
    total = 0
    for c in seq:
        code = _CODE.get(c, 4)
        if code > 3:
            continue
        counts[code] += 1
        total += 1
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


sequence_entropy_bits = shannon_entropy_acgt


def kmer_uniqueness_ratio(seq: str, k: int) -> float:
    """Distinct k-mers over total k-mers. Low means the sequence repeats itself."""
    if k <= 0 or len(seq) < k:
        return 0.0
    uniq = set()
    total = 0
    for _, key in for_each_valid_kmer(seq, k):
        uniq.add(key)
        total += 1
    if total <= 0:
        return 0.0
    return len(uniq) / total


def sequence_gc_fraction(seq: str) -> float:
    gc = 0
    acgt = 0
    for c in seq:
        code = _CODE.get(c, 4)
        if code > 3:
            continue
        acgt += 1
        if code in (1, 2):
            gc += 1
    return (gc / acgt) if acgt > 0 else 0.0


def sequence_tandem_fraction(seq: str) -> float:
    """Fraction of bases inside an exact tandem repeat of period <= 12.

    A base is covered when it lies in a maximal run of `seq[i] == seq[i - p]`
    spanning at least one further full copy of the period. Coverage is the UNION
    over periods, so a poly(A) run (period 1, 2, 3, ...) is not counted twice.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    covered = [False] * n
    for period in range(1, MAX_TANDEM_PERIOD + 1):
        p = period
        if p >= n:
            break
        run_start = p
        run = 0
        for i in range(p, n + 1):
            match = i < n and seq[i] in _CODE and seq[i] == seq[i - p]
            if match:
                if run == 0:
                    run_start = i
                run += 1
                continue
            if run >= p:
                for j in range(run_start - p, run_start + run):
                    covered[j] = True
            run = 0
    return sum(covered) / n


def sequence_low_complexity_fraction(seq: str) -> float:
    """Fraction of bases in a 16 bp window whose top two bases hold >= 90%.

    The classic dust-style test, and the one that flags poly(A) tails and
    two-base microsatellites -- i.e. the two things a TE library will happily
    match for the wrong reason.
    """
    n = len(seq)
    if n == 0:
        return 0.0
    window = min(LOW_COMPLEXITY_WINDOW, n)
    covered = [False] * n
    for start in range(n - window + 1):
        counts = [0, 0, 0, 0]
        total = 0
        for i in range(start, start + window):
            code = _CODE.get(seq[i], 4)
            if code > 3:
                continue
            counts[code] += 1
            total += 1
        if total <= 0:
            continue
        counts.sort(reverse=True)
        if (counts[0] + counts[1]) / total >= LOW_COMPLEXITY_TOP2_FRACTION:
            for i in range(start, start + window):
                covered[i] = True
    return sum(covered) / n


@dataclass
class TeSequenceComposition:
    """Library-independent description of one insert."""

    gc: float = 0.0
    entropy_bits: float = 0.0
    tandem_fraction: float = 0.0
    low_complexity_fraction: float = 0.0


def compute_te_sequence_composition(seq: str) -> TeSequenceComposition:
    return TeSequenceComposition(
        gc=sequence_gc_fraction(seq),
        entropy_bits=sequence_entropy_bits(seq),
        tandem_fraction=sequence_tandem_fraction(seq),
        low_complexity_fraction=sequence_low_complexity_fraction(seq),
    )


def build_te_canonical_kmer_frequencies(sequences: Sequence[str],
                                        k: int) -> dict[int, float]:
    """Canonical k-mer probability table over the library.

    The C++ allocates a dense `4^k` vector; a dict is used here because the
    table is sparse for any realistic library and the lookups are by key either
    way. Empty (falsy) when k is out of range or no valid k-mer exists, which is
    the signal the callers already test for.
    """
    if k <= 0 or k > 15:
        return {}
    counts: dict[int, float] = {}
    total = 0.0
    for key, count in _canonical_kmer_counts(sequences, k).items():
        counts[key] = float(count)
        total += count
    if total <= 0.0:
        return {}
    return {key: count / total for key, count in counts.items()}


def build_te_canonical_kmer_presence(sequences: Sequence[str], k: int) -> set[int]:
    if k <= 0 or k > 15:
        return set()
    return set(_canonical_kmer_counts(sequences, k))


def _canonical_kmer_counts(sequences: Sequence[str], k: int) -> dict[int, int]:
    """Occurrences per CANONICAL k-mer, keyed in order of first occurrence.

    Counts the raw keys first and folds each DISTINCT one onto its canonical
    form once, instead of reverse-complementing every occurrence: a 1,400-family
    library is ~1.8 Mbp, so that was 5 million reverse complements for at most
    4^k distinct answers, and most of the start-up time of every scan process.

    The result is the one the per-occurrence loop built, including its key
    ORDER -- which the JSD sums iterate, so it matters. A canonical key first
    appears where the earliest of its raw keys first appears, and walking the
    raw keys in first-occurrence order meets them in exactly that order. The
    counts are integers, so summing them in a different order is exact.
    """
    raw: dict[int, int] = {}
    for sequence in sequences:
        for _, key in for_each_valid_kmer(sequence, k):
            raw[key] = raw.get(key, 0) + 1
    folded: dict[int, int] = {}
    for key, count in raw.items():
        canonical = canonical_kmer_key(key, k)
        folded[canonical] = folded.get(canonical, 0) + count
    return folded


@dataclass
class TeSequenceBackground:
    """The library-level tables the composition features are measured against."""

    freq_k5: dict[int, float] = field(default_factory=dict)
    freq_k6: dict[int, float] = field(default_factory=dict)
    present_k9: set[int] = field(default_factory=set)
    valid: bool = False


def build_te_sequence_background(sequences: Sequence[str]) -> TeSequenceBackground:
    background = TeSequenceBackground(
        freq_k5=build_te_canonical_kmer_frequencies(sequences, SEQ_MODEL_JSD_K5),
        freq_k6=build_te_canonical_kmer_frequencies(sequences, SEQ_MODEL_JSD_K6),
        present_k9=build_te_canonical_kmer_presence(sequences, SEQ_MODEL_CONTAINMENT_K),
    )
    background.valid = bool(background.freq_k5) and bool(background.freq_k6) and bool(
        background.present_k9)
    return background


def _plogp(p: float) -> float:
    return p * math.log2(p) if p > 0.0 else 0.0


def te_kmer_jsd_vs_background(seq: str, k: int,
                              background: dict[int, float]) -> float:
    """Jensen-Shannon divergence in bits between the insert and the library.

    Base-2 JSD of two distributions is bounded by 1, so the result needs no
    normalisation: 0 is "composed like the library", 1 is "shares nothing with
    it". Only k-mers present in the insert are enumerated; the background-only
    mass contributes exactly `0.5 * (1 - covered)`, because at `p = 0` the
    summand collapses to `q/2`.
    """
    if not background or len(seq) < k:
        return 0.0
    query: dict[int, float] = {}
    total = 0.0
    for _, key in for_each_valid_kmer(seq, k):
        canonical = canonical_kmer_key(key, k)
        query[canonical] = query.get(canonical, 0.0) + 1.0
        total += 1.0
    if total <= 0.0:
        return 0.0
    jsd = 0.0
    covered_background = 0.0
    for key, count in query.items():
        p = count / total
        q = background.get(key, 0.0)
        m = 0.5 * (p + q)
        jsd += (0.5 * _plogp(p)) + (0.5 * _plogp(q)) - _plogp(m)
        covered_background += q
    jsd += 0.5 * max(0.0, 1.0 - covered_background)
    return min(1.0, max(0.0, jsd))


def te_kmer_containment(seq: str, k: int, present: set[int]) -> float:
    """Fraction of the insert's DISTINCT canonical k-mers found in the library.

    Needs no alignment, so it still reports something when the insert is a
    chimera or too diverged to align end to end -- the two cases where
    identity/coverage silently go to zero and take the evidence with them.
    """
    if not present or len(seq) < k:
        return 0.0
    distinct = {canonical_kmer_key(key, k) for _, key in for_each_valid_kmer(seq, k)}
    if not distinct:
        return 0.0
    hit = sum(1 for key in distinct if key in present)
    return hit / len(distinct)


def semiglobal_edit_identity(query: str, target: str) -> float:
    """Identity of the best placement of `query` inside `target`.

    Free end gaps on the TARGET only: the query must be consumed end to end, the
    target may be entered and left anywhere. That asymmetry is the point -- an
    insert fragment is matched against a whole consensus element, so the element
    contributes no penalty for the part it does not cover, while the fragment
    does for the part it cannot explain.
    """
    n = len(query)
    m = len(target)
    if n <= 0 or m <= 0:
        return 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        qi = query[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0 if qi == target[j - 1] else 1)
            delete = prev[j] + 1
            insert = curr[j - 1] + 1
            curr[j] = sub if sub <= delete else delete
            if insert < curr[j]:
                curr[j] = insert
        prev = curr
    best = min(prev)
    identity = 1.0 - (best / max(1, n))
    return min(1.0, max(0.0, identity))


@dataclass
class TeNameParts:
    """The pieces of a library header, after the four naming conventions."""

    exact_name: str = "NA"
    class_label: str = "NA"
    order_label: str = "NA"
    family: str = "NA"
    family_key: str = "NA"
    subfamily: str = "NA"


def parse_te_name_parts(te_name: str) -> TeNameParts:
    """Split a TE library header into class/order/family/subfamily.

    Handles the two header conventions in the wild -- RepeatMasker's
    `AluYa5#SINE/Alu` and Dfam's `L1:L1HS` -- and then COLLAPSES the family onto
    one of four canonical keys. The collapse matters downstream: family is what
    `family_state_compatibility` in the decision policy keys on, so a library
    that spells the same family three ways must not produce three families.
    """
    parts = TeNameParts()
    token = take_header_token(te_name)
    if not token:
        return parts
    parts.exact_name = token

    family = ""
    hash_at = token.find("#")
    if hash_at != -1:
        left = token[:hash_at]
        right = token[hash_at + 1:]
        parts.subfamily = left if left else token
        family = right if right else parts.subfamily
        slash = right.find("/")
        if slash != -1:
            parts.class_label = right[:slash] if slash > 0 else "NA"
            parts.order_label = right[slash + 1:] if (slash + 1) < len(right) else "NA"
            last_slash = family.rfind("/")
            if last_slash != -1 and (last_slash + 1) < len(family):
                family = family[last_slash + 1:]
        elif right:
            parts.class_label = right
    else:
        colon = token.find(":")
        if colon != -1:
            family = token[:colon]
            parts.subfamily = token[colon + 1:] if (colon + 1) < len(token) else token
        else:
            parts.subfamily = token
            family = token

    if not parts.subfamily:
        parts.subfamily = token
    parts.exact_name = parts.subfamily
    if not family:
        family = parts.subfamily

    family_key = upper_ascii(family)
    for prefix, canonical in (("ALU", "Alu"), ("L1", "L1"), ("SVA", "SVA"),
                              ("HERV", "HERV")):
        if family_key.startswith(prefix):
            parts.exact_name = parts.subfamily
            parts.family = canonical
            parts.family_key = prefix
            return parts

    parts.family = family
    parts.family_key = family_key if family_key else "NA"
    return parts


def confidence_from_qc_reason(qc_reason: str) -> str:
    """The three-level confidence the ledger reports, from the QC token.

    A pure lookup, and deliberately so: confidence here means "how specifically
    was this named", not "how strong is the evidence". The strength question is
    the selection layer's, and conflating the two is what lets a tier threshold
    act as a length filter.
    """
    if qc_reason == "PASS_INSERT_TE_ALIGNMENT":
        return "HIGH"
    if qc_reason == "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY":
        return "MEDIUM"
    if qc_reason == "PASS_INSERT_TE_ALIGNMENT_UNKNOWN":
        return "LOW"
    return "NA"


def parse_kmer_sizes_csv(csv: str, fallback_k: int) -> list[int]:
    """Parse `--te-kmer-sizes`, keeping only 7..31, plus the fallback.

    Out-of-range and unparsable entries are DROPPED rather than raising: the
    flag is a performance knob, and a typo in it should not stop a run that
    would otherwise be correct. An empty result falls back to 13.
    """
    out: list[int] = []
    for token in csv.split(","):
        trimmed = token.strip()
        if not trimmed:
            continue
        try:
            parsed = int(trimmed, 10)
        except ValueError:
            continue
        if parsed < 7 or parsed > 31:
            continue
        out.append(parsed)
    if 7 <= fallback_k <= 31:
        out.append(fallback_k)
    if not out:
        out.append(13)
    return sorted(set(out))


def fnv1a_append_byte(hash_value: int, byte: int) -> int:
    hash_value ^= byte & 0xFF
    return (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF


def fnv1a_append_string(hash_value: int, value: str) -> int:
    for byte in value.encode("utf-8"):
        hash_value = fnv1a_append_byte(hash_value, byte)
    return hash_value


def fnv1a_append_int32(hash_value: int, value: int) -> int:
    unsigned = value & 0xFFFFFFFF
    for shift in (0, 8, 16, 24):
        hash_value = fnv1a_append_byte(hash_value, (unsigned >> shift) & 0xFF)
    return hash_value


FNV1A_OFFSET_BASIS = 0xCBF29CE484222325
