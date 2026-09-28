"""
The composition masks mark the same bases after they were moved into C.

`placer/core/seqtools` now slides the low-complexity window instead of
recounting it, finds tandem and microsatellite runs by comparing the sequence
with itself shifted, and remembers canonical k-mer keys. These fractions and
masks are recorded features of every insert and decide which bases count as
informative, so they must not move by one base. The implementations before
the change are kept below verbatim and compared on homopolymers, pure and
interrupted microsatellites, Ns, soft-masked and IUPAC bases, non-ASCII input
and every short length.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

from placer.core import seqtools as S
from placer.core.seqtools import (
    LOW_COMPLEXITY_TOP2_FRACTION,
    LOW_COMPLEXITY_WINDOW,
    MAX_TANDEM_PERIOD,
    MICROSATELLITE_MAX_PERIOD,
    MICROSATELLITE_MIN_BP,
    MICROSATELLITE_MIN_COPIES,
)

_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def _old_rc(key: int, k: int) -> int:
    rc = 0
    for _ in range(k):
        rc = (rc << 2) | (3 - (key & 3))
        key >>= 2
    return rc


def _old_sequence_tandem_fraction(seq: str) -> float:
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


def _old_microsatellite_mask(seq: str) -> list[bool]:
    n = len(seq)
    covered = [False] * n
    for period in range(1, MICROSATELLITE_MAX_PERIOD + 1):
        if period >= n:
            break
        min_span = max(MICROSATELLITE_MIN_BP, MICROSATELLITE_MIN_COPIES * period)
        run = 0
        run_start = period
        for i in range(period, n + 1):
            if i < n and seq[i] in _CODE and seq[i] == seq[i - period]:
                if run == 0:
                    run_start = i
                run += 1
                continue
            if run > 0 and run + period >= min_span:
                for j in range(run_start - period, run_start + run):
                    covered[j] = True
            run = 0
    return covered


def _old_low_complexity_mask(seq: str) -> list[bool]:
    n = len(seq)
    if n == 0:
        return []
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
    return covered


def _old_kmers(seq: str, k: int) -> Iterator[tuple[int, int]]:
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


def _old_homopolymer(seq: str) -> int:
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


def _old_at_fraction(seq: str) -> float:
    at = 0
    total = 0
    for c in seq:
        if c not in _CODE:
            continue
        total += 1
        if c in ("A", "T"):
            at += 1
    return (at / total) if total > 0 else 0.0


def _old_entropy(seq: str) -> float:
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


def _old_uniqueness(seq: str, k: int) -> float:
    if k <= 0 or len(seq) < k:
        return 0.0
    uniq = set()
    total = 0
    for _, key in _old_kmers(seq, k):
        uniq.add(key)
        total += 1
    if total <= 0:
        return 0.0
    return len(uniq) / total


def _sequences():
    rng = random.Random(43)
    yield from ("", "A", "AC", "AAA", "N" * 20, "acgt" * 10, "ACGTé" * 5)
    for _ in range(600):
        parts = []
        for _ in range(rng.randint(1, 8)):
            roll = rng.random()
            if roll < 0.25:
                unit = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 13)))
                array = unit * rng.randint(1, 12)
                if rng.random() < 0.4:              # an interrupted array
                    array = "".join(rng.choice("ACGT") if rng.random() < 0.05 else b
                                    for b in array)
                parts.append(array)
            elif roll < 0.35:
                parts.append(rng.choice("ACGT") * rng.randint(1, 30))
            elif roll < 0.42:
                parts.append("".join(rng.choice("NRYacgtn-") for _ in range(rng.randint(1, 6))))
            else:
                parts.append("".join(rng.choice("ACGT") for _ in range(rng.randint(1, 120))))
        yield "".join(parts)


def test_the_masks_and_fractions_equal_the_loops():
    for seq in _sequences():
        assert S.low_complexity_mask(seq) == _old_low_complexity_mask(seq), seq
        assert S.microsatellite_mask(seq) == _old_microsatellite_mask(seq), seq
        assert S.sequence_tandem_fraction(seq) == _old_sequence_tandem_fraction(seq), seq
        assert all(type(value) is bool for value in S.microsatellite_mask(seq))


def test_canonical_keys_equal_the_computed_ones():
    rng = random.Random(47)
    for k in (1, 5, 6, 9, 10, 11, 21):
        for _ in range(3000):
            key = rng.randrange(4 ** k)
            assert S.canonical_kmer_key(key, k) == min(key, _old_rc(key, k))


def _old_low_complexity_softclip(seq, at_min, homopolymer_min, entropy_min, uniqueness_min):
    return (_old_at_fraction(seq) >= at_min
            or _old_homopolymer(seq) >= homopolymer_min
            or _old_entropy(seq) < max(0.0, entropy_min)
            or _old_uniqueness(seq, 5) < min(1.0, max(0.0, uniqueness_min)))


def _clips():
    rng = random.Random(53)
    yield from _sequences()
    for _ in range(150):
        size = rng.choice((20, 300, 2000, 2900, 3100, 8000))
        kind = rng.random()
        if kind < 0.3:
            unit = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 6)))
            seq = (unit * (size // len(unit) + 1))[:size]
        elif kind < 0.4:
            seq = "A" * rng.randint(1, 120) + "".join(rng.choice("ACGT") for _ in range(size))
        else:
            seq = "".join(rng.choice("ACGT") for _ in range(size))
        if rng.random() < 0.3:
            seq = "".join(rng.choice("NacRY") if rng.random() < 0.01 else b for b in seq)
        yield seq


def test_the_clip_complexity_test_and_its_parts_equal_the_loops():
    from placer.core.fragments import InsertionFragment, InsertionFragmentSource
    from placer.core.te_classifier import is_low_complexity_softclip
    clip = InsertionFragment(source=InsertionFragmentSource.CLIP_REF_RIGHT)
    rng = random.Random(59)
    decided = 0
    for seq in _clips():
        assert S.at_fraction(seq) == _old_at_fraction(seq), seq
        assert S.shannon_entropy_acgt(seq) == _old_entropy(seq), seq
        for k in (1, 3, 5, 9, 11, 31, 32, 33):
            new, old = S.kmer_uniqueness_ratio(seq, k), _old_uniqueness(seq, k)
            assert new == old or (math.isnan(new) and math.isnan(old)), (seq, k)
        for h in (-1, 0, 1, 2, 5, 20, 80, 200):
            assert S.has_homopolymer_run(seq, h) == (_old_homopolymer(seq) >= h), (seq, h)
        for at_min, h, e_min, u_min in ((0.90, 80, 1.25, 0.35), (0.5, 5, 1.9, 0.9),
                                        (1.0, 1000, 0.0, 0.0), (0.2, 3, -1.0, 1.5),
                                        (rng.random(), rng.randint(0, 50), 2 * rng.random(),
                                         rng.random())):
            new = is_low_complexity_softclip(clip, seq, at_min, h, e_min, u_min)
            old = bool(seq) and _old_low_complexity_softclip(seq, at_min, h, e_min, u_min)
            assert new == old, (seq[:40], len(seq), at_min, h, e_min, u_min)
            decided += new
    assert decided > 100


def _old_informative_aligned_bases(hit, mask):
    intervals = hit.query_intervals or [(hit.query_start, hit.query_end)]
    covered = [False] * len(mask)
    for start, end in intervals:
        for i in range(max(0, start), min(len(mask), end)):
            covered[i] = True
    return sum(1 for i, c in enumerate(covered) if c and not mask[i])


def test_informative_bases_from_prefix_sums_equal_the_per_base_count():
    from placer.core.te_classifier import (
        BlastSubjectHit,
        informative_aligned_bases,
        informative_prefix,
    )
    rng = random.Random(61)
    for _ in range(3000):
        size = rng.randint(0, 400)
        mask = [rng.random() < rng.choice((0.0, 0.2, 0.8)) for _ in range(size)]
        intervals = [(rng.randint(-20, size + 20), rng.randint(-20, size + 20))
                     for _ in range(rng.randint(0, 5))]
        hit = BlastSubjectHit(query_start=rng.randint(-5, size), query_end=rng.randint(-5, size + 5),
                              query_intervals=intervals)
        old = _old_informative_aligned_bases(hit, mask)
        assert informative_aligned_bases(hit, mask) == old
        assert informative_aligned_bases(hit, mask, informative_prefix(mask)) == old
