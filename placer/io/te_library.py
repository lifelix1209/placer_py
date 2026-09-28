"""Loading the TE library, and classifying a batch of inserts against it.

THIS IS THE SEAM BETWEEN THE TWO HALVES OF TE NAMING. `align_insert_sequences`
drives the external aligner (`placer/io/blast.py`) and then hands the result
to the pure scorer (`placer/core/te_classifier.py`). It sits on the input
side because what it fundamentally does is run a program; the algorithm sees it
only as `StageHooks.align_insert`, a function from an insert sequence to
evidence.

`align_insert_sequences` is a batch API, with by-sequence de-duplication so a
component with twenty supporting reads does not align twenty identical
consensus strings. The CLI uses `TeLibraryAligner`, which aligns a bin's
inserts in sorted, fixed-size `blastn` batches -- 21x faster than one process
per insert, with the batches chosen so that `--threads` cannot change them.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from placer.config import PipelineConfig
from placer.core.seqtools import TeSequenceBackground, parse_te_name_parts
from placer.core.taxonomy import LibrarySummary, TeTaxon, summarise_library
from placer.core.te_classifier import (
    TEAlignmentEvidence,
    TeEntry,
    build_insert_alignment_evidence_from_blast_hits,
    build_te_library_cache_key,
    load_te_entries_from_fasta,
    parse_kmer_sizes_csv,
)
from placer.io.blast import (
    BlastSubjectHit,
    ensure_te_blast_db,
    run_blastn_batch_against_te_library,
    staged_blastn,
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


def library_is_complete(config: PipelineConfig) -> bool:
    """Whether a miss against the library is evidence (see the config field)."""
    return config.te_library_completeness != "denovo"


def summarise_te_library(entries: list[TeEntry]) -> LibrarySummary:
    """How many library entries have a class, and where it came from.

    Printed once per run: a library whose headers carry no class the taxonomy
    can recognise gives every call class Unknown, which is a silent loss of the
    mechanism evidence rather than an error, so it has to be visible.
    """
    def taxon(entry: TeEntry) -> tuple[str, TeTaxon]:
        parts = parse_te_name_parts(entry.name)
        return entry.name, TeTaxon(parts.te_class, parts.superfamily,
                                   parts.taxon_source)
    return summarise_library(taxon(entry) for entry in entries)


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
            seq, False, [], config.te_subfamily_margin_min, background,
            library_is_complete(config))
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
        config.te_subfamily_margin_min, background, library_is_complete(config))
        for seq in insert_seqs]


#: Inserts per `blastn` process. Fixed, and applied to a bin's inserts in
#: sorted order, so which inserts share a process depends only on the bin --
#: never on `--threads`, the CPU count or what an earlier bin aligned.
BLAST_QUERIES_PER_CALL = 32


class TeLibraryAligner:
    """`align_insert_sequences(config, entries, seqs)` for a whole run, faster.

    INSERTS ARE BATCHED, `BLAST_QUERIES_PER_CALL` to a `blastn` process. BLAST+
    2.17 spends ~0.8 s of CPU starting up, whatever the query, and one process
    per insert made that start-up about half of a run. Measured on the 201
    distinct inserts of the human development slice (HG002, GRCh38
    chr1:10-20 Mb, Dfam 3.8 human): one process each took 116.6 s, groups of 32
    took 5.6 s, groups of 128 took 2.8 s.

    BATCHING IS NOT HIT-NEUTRAL, and the design follows from that. Which HSPs
    blastn keeps for a repetitive query depends on the other queries in its
    run: on those 201 inserts 8 raw hit lists differed between alone and
    batched (7 at groups of 8), though the evidence built from them -- family,
    subfamily, identity, coverage, cross-family margin, strand, consensus
    interval, QC -- was identical for all 201 at groups of 8, 32 and 128. So
    the batches must be a function of the input alone, or `--threads N` could
    stop writing the same bytes for every N:

      * the batches are a bin's DISTINCT inserts, SORTED, cut every
        `BLAST_QUERIES_PER_CALL` -- the bin is the same whatever the chunking;
      * nothing is remembered across bins. A memo would decide which inserts
        still need aligning, and so what shares a batch, from what this worker
        happened to align before.

    The batches of one bin run CONCURRENTLY, `jobs` at a time. Threads suffice:
    each spends its time blocked in `subprocess.run`, which releases the GIL.
    The database key is computed once per run -- `build_te_library_cache_key`
    hashes every library sequence, 0.3 s for Dfam human.
    """

    def __init__(self, config: PipelineConfig, entries: list[TeEntry],
                 background: TeSequenceBackground | None = None,
                 jobs: int = 1) -> None:
        self.config = config
        self.entries = entries
        self.background = background
        self.jobs = max(1, jobs)
        self._db_prefix: str | None = None
        self._blastn: str | None = None
        self.blastn_calls = 0

    def prepare(self) -> str:
        """Build the BLAST database now, and return its prefix.

        A parallel run calls this in the parent before starting any worker,
        so the workers find the database on disk instead of all racing to
        run `makeblastdb` into the same path.
        """
        return self._db()

    def _db(self) -> str:
        if self._db_prefix is None:
            ks = parse_kmer_sizes_csv(self.config.te_kmer_sizes_csv,
                                      self.config.te_kmer_size)
            cache_key = build_te_library_cache_key(self.entries, ks,
                                                   self.config.te_kmer_size)
            self._db_prefix = ensure_te_blast_db(self.config.te_fasta_path,
                                                 self.config.te_makeblastdb_path,
                                                 cache_key)
            # The same blastn, from node-local disk when it lives on a network
            # filesystem (`placer/io/blast.staged_blastn`).
            canary = next((entry.sequence[:300] for entry in self.entries
                           if len(entry.sequence) >= 100), "")
            self._blastn = staged_blastn(self.config.te_blastn_path,
                                         self._db_prefix, canary)
        return self._db_prefix

    def align(self, insert_seqs: list[str]) -> list[TEAlignmentEvidence]:
        """Evidence for each insert, from sorted fixed-size `blastn` batches."""
        config = self.config
        if not self.entries or not config.te_fasta_path:
            return align_insert_sequences(config, self.entries, insert_seqs,
                                          self.background)

        distinct = sorted({seq for seq in insert_seqs if seq})
        groups = [distinct[i:i + BLAST_QUERIES_PER_CALL]
                  for i in range(0, len(distinct), BLAST_QUERIES_PER_CALL)]
        hits: dict[str, list[BlastSubjectHit]] = {}
        if groups:
            db_prefix = self._db()
            blastn = self._blastn or config.te_blastn_path

            def align_group(group: list[str]) -> dict[str, list[BlastSubjectHit]]:
                found = run_blastn_batch_against_te_library(
                    blastn, db_prefix,
                    [(f"q{i}", seq) for i, seq in enumerate(group)])
                return {seq: found[f"q{i}"] for i, seq in enumerate(group)}

            self.blastn_calls += len(groups)
            if self.jobs > 1 and len(groups) > 1:
                with ThreadPoolExecutor(max_workers=min(self.jobs, len(groups))) as pool:
                    for part in pool.map(align_group, groups):
                        hits.update(part)
            else:
                for group in groups:
                    hits.update(align_group(group))

        return [build_insert_alignment_evidence_from_blast_hits(
            seq, True, list(hits.get(seq, [])) if seq else [],
            config.te_subfamily_margin_min, self.background,
            library_is_complete(config))
            for seq in insert_seqs]

    def align_one(self, insert_seq: str) -> TEAlignmentEvidence:
        return self.align([insert_seq])[0]
