"""Driving BLAST+: the database on disk, the query file, the subprocess.

WHY THIS IS SPLIT FROM THE CLASSIFIER. `placer/core/te_classifier.py` used
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
import hashlib
import os
import shutil
import subprocess
import tempfile

from placer.core.te_classifier import (
    BLAST_MAX_TARGET_SEQS,
    BLAST_OUTFMT,
    BlastSubjectHit,
    collapse_blast_hsps,
    parse_blast_output,
)
from placer.io.perf import count_locked


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

    # ONE BUILDER AT A TIME. The cache is per node and shared by every run on
    # it, so two jobs that start together both found no database and ran
    # makeblastdb into the same files; one of them failed (seen on the
    # cluster: one of eight jobs submitted at once). The lock serialises the
    # build, and whoever waited finds the finished database and uses it.
    with _exclusive(db_prefix + ".lock"):
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


#: Filesystems whose file pages do not stay cached between processes here
#: (BeeGFS in `buffered` mode), or that are remote in general.
NETWORK_FILESYSTEMS = frozenset({
    "beegfs", "nfs", "nfs4", "lustre", "gpfs", "cifs", "smb3", "ceph",
    "fuse.sshfs", "fuse.glusterfs", "panfs", "wekafs"})
#: `PLACER_STAGE_BLASTN`: "0" never stages blastn, "1" always does; unset
#: stages it when it lives on a network filesystem.
STAGE_BLASTN_ENV = "PLACER_STAGE_BLASTN"
_STAGED_OK = "STAGED_OK"


def filesystem_type(path: str, mounts_text: str | None = None) -> str:
    """The type of the filesystem `path` is on, from `/proc/mounts`; "" if
    that cannot be read."""
    if mounts_text is None:
        try:
            with open("/proc/mounts") as handle:
                mounts_text = handle.read()
        except OSError:
            return ""
    best, kind = "", ""
    for line in mounts_text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        point = fields[1]
        inside = path == point or path.startswith(point.rstrip("/") + "/")
        if inside and len(point) > len(best):
            best, kind = point, fields[2]
    return kind


def ldd_dependencies(ldd_text: str) -> dict[str, str]:
    """`{soname: resolved path}` from `ldd` output; unresolved ones map to ""."""
    found: dict[str, str] = {}
    for line in ldd_text.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        name, _, rest = line.partition("=>")
        target = rest.strip().split(" (")[0].strip()
        found[name.strip()] = "" if target == "not found" else target
    return found


def _ldd(path: str) -> dict[str, str] | None:
    try:
        result = subprocess.run(["ldd", path], capture_output=True, text=True,
                                check=False)
    except OSError:
        return None
    return ldd_dependencies(result.stdout) if result.returncode == 0 else None


def staged_blastn(blastn_path: str, canary_db: str, canary_seq: str,
                  work_dir: str | None = None, stage: bool | None = None) -> str:
    """A node-local copy of `blastn` and the shared libraries it loads from a
    network filesystem, or `blastn_path` itself.

    WHY. A conda BLAST+ is a small `blastn` that loads 86 libraries (535 MB)
    from the environment. Measured on this cluster, from BeeGFS: every exec
    took ~1,020 major page faults -- the library pages are not kept between
    processes -- and 0.10 s of system time against 0.03 s from local disk.
    With 48 scan workers on a node running blastn at once, that contention
    reached 0.6-0.9 s of system time per call, three quarters of the run's
    whole system time (`placer/io/perf.py`, perf round 1). Batching cannot remove
    the execs without changing hits (`run_blastn_batch_against_te_library`).

    WHY THE ANSWER IS UNCHANGED. The copies are the same bytes, and they are
    used only after two checks: `ldd` of the copy resolves every library either
    to its staged copy or to exactly the path the original resolves it to, and
    a canary search (`canary_seq`, a library element) against `canary_db`
    writes byte-identical output from both.
    Any failure, or no `ldd`, keeps the original binary. The copy lives in
    `blast_work_dir()` beside the database, one per binary version, built under
    the same kind of lock and reused by every run on the node.
    """
    if stage is None:
        setting = os.environ.get(STAGE_BLASTN_ENV, "")
        if setting == "0":
            return blastn_path
        stage = setting == "1" or None
    if stage is False:
        return blastn_path
    resolved = shutil.which(blastn_path)
    if not resolved or not canary_db or not canary_seq:
        return blastn_path
    real = os.path.realpath(resolved)
    if stage is None and filesystem_type(real) not in NETWORK_FILESYSTEMS:
        return blastn_path
    try:
        info = os.stat(real)
        key = hashlib.sha1(f"{real}\0{info.st_size}\0{info.st_mtime_ns}".encode()
                           ).hexdigest()[:16]
        target = os.path.join(work_dir or blast_work_dir(), f"blastn_{key}")
        staged = os.path.join(target, "bin", "blastn")
        if os.path.isfile(os.path.join(target, _STAGED_OK)):
            return staged
        with _exclusive(target + ".lock"):
            if os.path.isfile(os.path.join(target, _STAGED_OK)):
                return staged
            if not _stage_blastn_copy(real, target):
                return blastn_path
            if not _same_canary_output(real, staged, canary_db, canary_seq, target):
                return blastn_path
            with open(os.path.join(target, _STAGED_OK), "w") as handle:
                handle.write(real + "\n")
        return staged
    except (OSError, subprocess.SubprocessError):
        return blastn_path


def _stage_blastn_copy(real: str, target: str) -> bool:
    """Copy the binary to `target/bin`, and every library it resolves on a
    network filesystem to `target/lib` (its RPATH `$ORIGIN/../lib` finds them
    there). True when `ldd` then resolves the copy as the original."""
    original = _ldd(real)
    if original is None or any(not path for path in original.values()):
        return False
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(os.path.join(target, "bin"))
    os.makedirs(os.path.join(target, "lib"))
    staged = os.path.join(target, "bin", "blastn")
    shutil.copy2(real, staged)
    copied: dict[str, str] = {}
    for name, path in original.items():
        if filesystem_type(os.path.realpath(path)) in NETWORK_FILESYSTEMS:
            destination = os.path.join(target, "lib", os.path.basename(path))
            shutil.copyfile(path, destination)
            copied[name] = destination
    after = _ldd(staged)
    if after is None or set(after) != set(original):
        return False
    # ldd prints what `$ORIGIN/../lib` found unnormalised (`bin/../lib/...`).
    return all(os.path.realpath(after[name])
               == os.path.realpath(copied.get(name, original[name]))
               for name in original)


def _same_canary_output(real: str, staged: str, canary_db: str, canary_seq: str,
                        target: str) -> bool:
    """Both binaries search one query against the run's database; True when
    they write the same bytes."""
    query = os.path.join(target, "canary.fa")
    with open(query, "w") as out:
        out.write(f">canary\n{canary_seq}\n")
    outputs = []
    for binary in (real, staged):
        result = subprocess.run(
            [binary, "-query", query, "-db", canary_db, "-task", "blastn",
             "-dust", "no", "-soft_masking", "false",
             "-max_target_seqs", str(BLAST_MAX_TARGET_SEQS), "-outfmt", BLAST_OUTFMT],
            capture_output=True, check=False)
        if result.returncode != 0:
            return False
        outputs.append(result.stdout)
    return outputs[0] == outputs[1]


@contextlib.contextmanager
def _exclusive(lock_path: str):
    """An exclusive advisory lock on `lock_path` for the `with` block. Where
    `fcntl` does not exist (not POSIX) the block runs unlocked, as before."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - PLACER runs on POSIX (pysam does)
        yield
        return
    with open(lock_path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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

    BATCHING IS NOT HIT-NEUTRAL, whatever the e-value argument suggests.
    Measured with BLAST+ 2.17 on HG002 (chr21:19,546,363): a 112 bp (AT)n
    insert reported different HSPs against LTR66 and MLT2B5 when three other
    inserts shared its run than when it ran alone, which moved its
    `cross_family_margin` from 0.0610 to 0.0662. In a tandem repeat many
    alignments tie, and which one blastn keeps depends on how the batch's
    queries were packed. Such an insert is now rejected as simple repeat
    before the family ranking, and on the 201 inserts of the human
    development slice batching changed 8 raw hit lists and no evidence; the
    pipeline batches, with batches that depend only on the bin (see
    `placer/io/te_library.TeLibraryAligner`).

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
    # Batches run on the aligner's thread pool, hence the locked counter.
    count_locked("blastn_calls")
    count_locked("blastn_queries", len(queries))
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
