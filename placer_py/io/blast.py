"""Driving BLAST+: the database on disk, the query file, the subprocess.

WHY THIS IS SPLIT FROM THE CLASSIFIER. `placer_py/core/te_classifier.py` used
to hold both halves of TE naming in one 1002-line module: the part that shells
out to `makeblastdb` and `blastn`, and the part that turns their tabular output
into evidence. `tests/test_23_te_classifier.py` already says out loud that the
port "splits the parsing from the process" -- this is that split reflected in
the file layout, so the sentence is now enforced rather than asserted.

WHAT STAYED BEHIND, deliberately. The HSP line parser, the collapse, the
evidence builder and `build_te_library_cache_key` are all pure functions of
text, so they are algorithm and they stay with the algorithm. So does
`BLAST_OUTFMT`: it is the column order the parser parses, and the two must
never be edited apart.

THE CACHE KEY IS OBSERVABLE, not an optimisation. It names the database files
on disk, so it has to change when the library content, the requested k, or the
k list changes -- otherwise a run silently reuses an index built for a
different configuration. That is why it is computed in the algorithm and merely
USED here.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile

from placer_py.core.te_classifier import (
    BLAST_MAX_TARGET_SEQS,
    BLAST_OUTFMT,
    BlastSubjectHit,
    collapse_blast_hsps,
    parse_blast_output,
)


def blast_work_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "placer_te_blast")
    os.makedirs(path, exist_ok=True)
    return path


def blast_db_files_exist(db_prefix: str) -> bool:
    """All three of `.nhr`, `.nin`, `.nsq`, each non-empty.

    A partially-written database from an interrupted `makeblastdb` would
    otherwise be reused, and blastn's failure mode on one is a confusing error
    rather than a rebuild.
    """
    for suffix in (".nhr", ".nin", ".nsq"):
        path = db_prefix + suffix
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            return False
    return True


def ensure_te_blast_db(te_fasta_path: str, makeblastdb_path: str,
                       cache_key: str) -> str:
    """Build the BLAST database if it is not already on disk.

    Returns an empty prefix for an empty FASTA path -- "no library configured"
    is a normal run, not an error -- but raises when a library IS configured and
    the database cannot be built, because silently proceeding would classify
    every insert as `TE_LIBRARY_UNAVAILABLE` and look like a clean negative run.
    """
    if not te_fasta_path:
        return ""
    if not makeblastdb_path:
        raise RuntimeError("BLAST+ makeblastdb path is empty")

    db_prefix = os.path.join(blast_work_dir(), f"te_library_{cache_key}")
    if blast_db_files_exist(db_prefix):
        return db_prefix

    result = subprocess.run(
        [makeblastdb_path, "-in", te_fasta_path, "-dbtype", "nucl", "-out", db_prefix],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if result.returncode != 0 or not blast_db_files_exist(db_prefix):
        raise RuntimeError(
            f"failed to build BLAST database for TE FASTA {te_fasta_path!r} "
            f"with makeblastdb {makeblastdb_path!r}")
    return db_prefix


def write_blast_batch_query_fasta(queries: list[tuple[str, str]],
                                  directory: str | None = None) -> str:
    directory = directory if directory is not None else blast_work_dir()
    handle, path = tempfile.mkstemp(prefix="insert_batch_query_", suffix=".fa",
                                    dir=directory)
    with os.fdopen(handle, "w") as out:
        for query_id, sequence in queries:
            out.write(f">{query_id}\n")
            for offset in range(0, len(sequence), 80):
                out.write(sequence[offset:offset + 80] + "\n")
    return path


def run_blastn_batch_against_te_library(blastn_path: str, blast_db_prefix: str,
                                        queries: list[tuple[str, str]]
                                        ) -> dict[str, list[BlastSubjectHit]]:
    """One blastn invocation for a whole batch of inserts.

    Batching is not only a speed choice: blastn's e-values depend on the
    database, not the query set, so batching does not change any reported
    number -- which is what makes it safe to do.

    EVERY query gets a key in the result, including ones with no hits. A missing
    key and an empty list would otherwise be indistinguishable, and "blastn was
    never asked" is a different condition from "blastn found nothing".
    """
    if not blastn_path:
        raise RuntimeError("BLAST+ blastn path is empty")
    if not blast_db_prefix:
        raise RuntimeError("BLAST database prefix is empty")

    out: dict[str, list[BlastSubjectHit]] = {}
    if not queries:
        return out

    query_lengths = {query_id: len(sequence) for query_id, sequence in queries}
    query_path = write_blast_batch_query_fasta(queries)
    output_path = os.path.splitext(query_path)[0] + ".blast.tsv"
    try:
        result = subprocess.run(
            [blastn_path, "-query", query_path, "-db", blast_db_prefix,
             "-task", "blastn", "-dust", "no", "-soft_masking", "false",
             "-max_target_seqs", str(BLAST_MAX_TARGET_SEQS),
             "-outfmt", BLAST_OUTFMT, "-out", output_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "blastn failed for batched insert consensus TE classification "
                f"with executable {blastn_path!r}")
        if not os.path.exists(output_path):
            raise RuntimeError(f"blastn did not create output file: {output_path}")
        with open(output_path) as handle:
            hsps_by_query = parse_blast_output(handle.read(), query_lengths)
    finally:
        for path in (query_path, output_path):
            with contextlib.suppress(OSError):
                os.remove(path)

    for query_id, _ in queries:
        out[query_id] = collapse_blast_hsps(hsps_by_query.get(query_id, []),
                                            query_lengths[query_id])
    return out
