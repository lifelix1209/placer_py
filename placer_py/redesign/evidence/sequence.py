from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from placer_py.redesign.models import SequenceFeatures, TeHit


def read_fasta_records(path: str | Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    name: str | None = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                records[name] = []
            elif name is not None:
                records[name].append(line.upper())
    return {key: "".join(parts) for key, parts in records.items()}


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def sequence_features(seq: str | None) -> SequenceFeatures:
    if not seq:
        return SequenceFeatures()
    seq = seq.upper()
    length = len(seq)
    gc = sum(base in "GC" for base in seq) / float(length)
    counts = {base: seq.count(base) for base in "ACGT"}
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / float(length)
            entropy -= p * math.log2(p)
    max_run = 1
    curr_run = 1
    for i in range(1, length):
        if seq[i] == seq[i - 1]:
            curr_run += 1
            max_run = max(max_run, curr_run)
        else:
            curr_run = 1
    return SequenceFeatures(
        length=length,
        gc_fraction=gc,
        entropy=entropy,
        low_complexity_fraction=max_run / float(length),
    )


def _family_parts(name: str) -> tuple[str, str]:
    if ":" in name:
        family, subfamily = name.split(":", 1)
        return family, subfamily
    return name, name


def _best_containment(query: str, reference: str) -> int:
    query = query.upper()
    reference = reference.upper()
    for size in range(len(query), 0, -1):
        for start in range(len(query) - size + 1):
            if query[start : start + size] in reference:
                return size
    return 0


def best_exact_te_hit(query: str | None, references: dict[str, str]) -> TeHit:
    if not query:
        return TeHit()
    query = query.upper()
    best_name = ""
    best_orientation = ""
    best_bases = 0
    for name, reference in references.items():
        fwd = _best_containment(query, reference)
        rev = _best_containment(reverse_complement(query), reference)
        if fwd > best_bases:
            best_name = name
            best_orientation = "+"
            best_bases = fwd
        if rev > best_bases:
            best_name = name
            best_orientation = "-"
            best_bases = rev
    if best_bases == 0:
        return TeHit()
    family, subfamily = _family_parts(best_name)
    coverage = best_bases / float(len(query))
    return TeHit(
        family=family,
        subfamily=subfamily,
        identity=1.0,
        query_coverage=coverage,
        orientation=best_orientation,
    )


def build_te_kmer_index(references: dict[str, str], k: int = 17) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for name, sequence in references.items():
        sequence = sequence.upper()
        if len(sequence) < k:
            continue
        for start in range(len(sequence) - k + 1):
            index[sequence[start : start + k]].add(name)
    return dict(index)


def best_kmer_te_hit(
    query: str | None,
    references: dict[str, str],
    index: dict[str, set[str]],
    k: int = 17,
) -> TeHit:
    if not query:
        return TeHit()
    query = query.upper()
    if len(query) < k:
        return best_exact_te_hit(query, references)

    best_name = ""
    best_orientation = ""
    best_count = 0
    best_total = 0
    reference_rank = {name: rank for rank, name in enumerate(references)}
    best_rank = len(reference_rank)
    for orientation, oriented_query in (("+", query), ("-", reverse_complement(query))):
        kmers = {oriented_query[start : start + k] for start in range(len(oriented_query) - k + 1)}
        counts: Counter[str] = Counter()
        for kmer in sorted(kmers):
            for name in sorted(index.get(kmer, ()), key=lambda item: (reference_rank.get(item, len(reference_rank)), item)):
                counts[name] += 1
        if counts:
            name, count = min(
                counts.items(),
                key=lambda item: (-item[1], reference_rank.get(item[0], len(reference_rank)), item[0]),
            )
            rank = reference_rank.get(name, len(reference_rank))
            if count > best_count or (count == best_count and rank < best_rank):
                best_name = name
                best_orientation = orientation
                best_count = count
                best_total = len(kmers)
                best_rank = rank

    if best_count == 0 or best_total == 0:
        return TeHit()
    family, subfamily = _family_parts(best_name)
    return TeHit(
        family=family,
        subfamily=subfamily,
        identity=1.0,
        query_coverage=best_count / float(best_total),
        orientation=best_orientation,
    )
