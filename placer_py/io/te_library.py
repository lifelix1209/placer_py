"""Loading the TE library, and classifying a batch of inserts against it.

THIS IS THE SEAM BETWEEN THE TWO HALVES OF TE NAMING. `align_insert_sequences`
drives the external aligner (`placer_py/io/blast.py`) and then hands the result
to the pure scorer (`placer_py/core/te_classifier.py`). It sits on the input
side because what it fundamentally does is run a program; the algorithm sees it
only as `StageHooks.align_insert`, a function from an insert sequence to
evidence.

IT IS A BATCH API AND IT MATTERS THAT IT STAYS ONE. blastn's e-values depend on
the database, not on the query set, so batching changes no reported number --
which is what makes it safe -- and the by-sequence de-duplication below is what
stops a component with twenty supporting reads from aligning twenty
near-identical consensus strings. The C++ achieves the same with a
mutex-guarded in-flight cache; here it is a dict, because there is nothing to
serialise against.

(The CLI currently defeats that batching by calling this once per insert. That
is a separate, measured change and not this module's doing -- see the note in
`placer_py/wiring.py`.)
"""

from __future__ import annotations

from placer_py.config import PipelineConfig
from placer_py.io.blast import ensure_te_blast_db, run_blastn_batch_against_te_library
from placer_py.seqtools import TeSequenceBackground
from placer_py.te_classifier import (
    TEAlignmentEvidence,
    TeEntry,
    build_insert_alignment_evidence_from_blast_hits,
    build_te_library_cache_key,
    load_te_entries_from_fasta,
    parse_kmer_sizes_csv,
)


def load_te_library(te_fasta_path: str) -> list[TeEntry]:
    """Read the TE FASTA into entries.

    Returns an EMPTY LIST rather than raising for an unreadable or empty file:
    the caller has to decide what that means, and for the main CLI it is fatal
    (an empty library makes every insert `TE_LIBRARY_UNAVAILABLE`, which is a
    silent whole-run negative) while for a de novo run against a parent pool it
    need not be.
    """
    with open(te_fasta_path) as handle:
        return load_te_entries_from_fasta(handle.read())


def align_insert_sequences(config: PipelineConfig, entries: list[TeEntry],
                           insert_seqs: list[str],
                           background: TeSequenceBackground | None = None
                           ) -> list[TEAlignmentEvidence]:
    """Classify a batch of assembled inserts, in input order.

    Deduplicated by SEQUENCE before the aligner is called, because a component
    with twenty supporting reads produces twenty near-identical consensus
    strings and the identical ones need aligning once. The C++ achieves the same
    thing with a mutex-guarded in-flight cache; here it is a dict, because there
    is nothing to serialise against.
    """
    if not entries or not config.te_fasta_path:
        return [build_insert_alignment_evidence_from_blast_hits(
            seq, False, [], config.te_subfamily_margin_min, background)
            for seq in insert_seqs]

    ks = parse_kmer_sizes_csv(config.te_kmer_sizes_csv, config.te_kmer_size)
    cache_key = build_te_library_cache_key(entries, ks, config.te_kmer_size)
    db_prefix = ensure_te_blast_db(config.te_fasta_path, config.te_makeblastdb_path,
                                   cache_key)

    unique: dict[str, str] = {}
    for seq in insert_seqs:
        if seq and seq not in unique:
            unique[seq] = f"q{len(unique)}"
    hits_by_id = run_blastn_batch_against_te_library(
        config.te_blastn_path, db_prefix,
        [(query_id, seq) for seq, query_id in unique.items()])

    return [build_insert_alignment_evidence_from_blast_hits(
        seq, True, hits_by_id.get(unique.get(seq, ""), []),
        config.te_subfamily_margin_min, background) for seq in insert_seqs]
