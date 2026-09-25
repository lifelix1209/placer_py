"""Loading the TE library, and classifying a batch of inserts against it.

THIS IS THE SEAM BETWEEN THE TWO HALVES OF TE NAMING. `align_insert_sequences`
drives the external aligner (`placer/io/blast.py`) and then hands the result
to the pure scorer (`placer/core/te_classifier.py`). It sits on the input
side because what it fundamentally does is run a program; the algorithm sees it
only as `StageHooks.align_insert`, a function from an insert sequence to
evidence.

`align_insert_sequences` is a batch API, with by-sequence de-duplication so a
component with twenty supporting reads does not align twenty identical
consensus strings. But a multi-query `blastn` run is NOT result-neutral -- see
`run_blastn_batch_against_te_library` for the measured case -- so the CLI does
not batch queries into one process. It uses `TeLibraryAligner`, which gives
every insert its own `blastn`, exactly as a one-insert batch always did, and
recovers the speed another way: the calls for a bin run concurrently, the
database key is computed once per run, and the hits are remembered per
sequence across bins.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from placer.config import PipelineConfig
from placer.core.seqtools import TeSequenceBackground
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


class TeLibraryAligner:
    """`align_insert_sequences(config, entries, [seq])` for a whole run, faster.

    THE RESULT FOR EVERY INSERT IS WHAT A ONE-INSERT CALL RETURNS, byte for
    byte: each insert still gets a `blastn` of its own, because packing
    several into one run changes the HSPs blastn reports for repetitive ones
    (measured; see `run_blastn_batch_against_te_library`). What changes is
    what that costs.

      * `blastn`'s start-up. BLAST+ 2.17 spends ~0.8 s of CPU before it reads
        a query (`blastn -version` alone costs that), more than the search for
        a few-hundred-base insert. That CPU cannot be avoided without sharing
        a process between inserts, which is the thing that changes answers --
        but it need not be spent one launch after another. The bin loop hands
        over every insert of a bin at once and their processes run
        CONCURRENTLY, `jobs` at a time, so the scan process is not idle while
        they start.
      * The database key. `build_te_library_cache_key` hashes every library
        sequence byte by byte in Python -- 0.3 s for a 1,400-family Dfam
        library -- and the library does not change during a run.
      * Repeated inserts. Neighbouring components of one locus assemble the
        same insert, and the hits for a sequence alone depend only on that
        sequence and the database. They are remembered, bounded, per sequence.

    WHAT IS REMEMBERED is the parsed hit list, not the evidence built from it:
    the evidence object is rebuilt on every call, exactly as the per-call path
    builds it, so no two callers ever share a mutable result.
    """

    #: Enough to cover every insert a locus produces while it is being
    #: scanned, which is where repeats come from; small enough that a
    #: whole-genome run cannot grow it without bound.
    MEMO_CAPACITY = 4096

    def __init__(self, config: PipelineConfig, entries: list[TeEntry],
                 background: TeSequenceBackground | None = None,
                 jobs: int = 1) -> None:
        self.config = config
        self.entries = entries
        self.background = background
        #: Concurrent `blastn` processes. Threads suffice to drive them: each
        #: thread spends its time blocked in `subprocess.run`, which releases
        #: the GIL.
        self.jobs = max(1, jobs)
        self._db_prefix: str | None = None
        self._hits: OrderedDict[str, list[BlastSubjectHit]] = OrderedDict()
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
        return self._db_prefix

    def align(self, insert_seqs: list[str]) -> list[TEAlignmentEvidence]:
        """What `align_insert_sequences` returns for the same batch."""
        config = self.config
        if not self.entries or not config.te_fasta_path:
            return align_insert_sequences(config, self.entries, insert_seqs,
                                          self.background)

        # This batch's hits, gathered BEFORE anything is written to the memo:
        # the memo is bounded, and reading results back out of it would lose
        # any that a large batch evicted before they were used.
        batch_hits: dict[str, list[BlastSubjectHit]] = {}
        missing: dict[str, None] = {}
        for seq in insert_seqs:
            if not seq or seq in batch_hits or seq in missing:
                continue
            remembered = self._hits.get(seq)
            if remembered is not None:
                self._hits.move_to_end(seq)
                batch_hits[seq] = remembered
            else:
                missing[seq] = None
        if missing:
            db_prefix = self._db()

            def align_alone(seq: str) -> list[BlastSubjectHit]:
                # "q0", the id a one-insert `align_insert_sequences` call
                # gives it, so the process sees exactly the same input.
                return run_blastn_batch_against_te_library(
                    config.te_blastn_path, db_prefix, [("q0", seq)])["q0"]

            pending = list(missing)
            self.blastn_calls += len(pending)
            if self.jobs > 1 and len(pending) > 1:
                with ThreadPoolExecutor(max_workers=min(self.jobs, len(pending))) as pool:
                    found = list(pool.map(align_alone, pending))
            else:
                found = [align_alone(seq) for seq in pending]
            for seq, hits in zip(pending, found):
                batch_hits[seq] = hits
                self._remember(seq, hits)

        return [build_insert_alignment_evidence_from_blast_hits(
            seq, True, list(batch_hits.get(seq, [])) if seq else [],
            config.te_subfamily_margin_min, self.background)
            for seq in insert_seqs]

    def align_one(self, insert_seq: str) -> TEAlignmentEvidence:
        return self.align([insert_seq])[0]

    def _remember(self, seq: str, hits: list[BlastSubjectHit]) -> None:
        if self.MEMO_CAPACITY <= 0:
            return
        self._hits[seq] = hits
        self._hits.move_to_end(seq)
        while len(self._hits) > self.MEMO_CAPACITY:
            self._hits.popitem(last=False)
