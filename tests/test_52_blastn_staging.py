"""
blastn staged on node-local disk is the same program, or is not used.

`placer/io/blast.staged_blastn` copies a conda `blastn` and the libraries it
loads from a network filesystem into the node's BLAST work directory, because
paging them in from BeeGFS on every exec cost most of a busy node's system
time. It may only ever change where the bytes are read from: this pins the
parsers it trusts, that a staged copy searches exactly like the original, and
that it declines rather than guesses.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from placer.io.blast import filesystem_type, ldd_dependencies, staged_blastn

MOUNTS = """\
/dev/sdb2 / ext4 rw,relatime 0 0
tmpfs /dev/shm tmpfs rw 0 0
beegfs_nodev /mnt/beegfs beegfs rw,relatime 0 0
beegfs_nodev /mnt/beegfs6 beegfs rw,relatime 0 0
server:/x /mnt/nfs nfs4 rw 0 0
"""


def test_the_filesystem_is_the_longest_mount_point_that_contains_the_path():
    assert filesystem_type("/mnt/beegfs6/env/bin/blastn", MOUNTS) == "beegfs"
    assert filesystem_type("/mnt/beegfs/x", MOUNTS) == "beegfs"
    assert filesystem_type("/mnt/beegfsX/x", MOUNTS) == "ext4"
    assert filesystem_type("/mnt/nfs", MOUNTS) == "nfs4"
    assert filesystem_type("/usr/bin/blastn", MOUNTS) == "ext4"
    assert filesystem_type("/usr/bin/blastn", "") == ""


def test_ldd_output_parses_to_resolved_paths_and_marks_the_missing():
    text = """\
\tlinux-vdso.so.1 (0x00007ffd)
\tlibblast.so => /env/bin/../lib/ncbi-blast+/libblast.so (0x00007f)
\tlibz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f)
\tlibgone.so => not found
\t/lib64/ld-linux-x86-64.so.2 (0x00007f)
"""
    assert ldd_dependencies(text) == {
        "libblast.so": "/env/bin/../lib/ncbi-blast+/libblast.so",
        "libz.so.1": "/lib/x86_64-linux-gnu/libz.so.1",
        "libgone.so": ""}


def _tools_missing() -> str:
    for tool in ("blastn", "makeblastdb", "ldd"):
        if shutil.which(tool) is None:
            return f"needs {tool}"
    return ""


LIBRARY = ("GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTC"
           "AGGAGATCGAGACCATCCCGGCTAAAACGGTGAAACCCCGTCTCTACTAAAAATACAAAAAATTAGCCGG"
           "GCGTGGTGGCGGGCGCCTGTAGTCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATGGCGTGAACCCGGGA")


def test_a_staged_blastn_searches_exactly_like_the_original():
    reason = _tools_missing()
    if reason:
        pytest.skip(reason)
    with tempfile.TemporaryDirectory() as directory:
        fasta = os.path.join(directory, "lib.fa")
        with open(fasta, "w") as out:
            out.write(f">AluY#SINE/Alu\n{LIBRARY}\n")
        db = os.path.join(directory, "db")
        subprocess.run(["makeblastdb", "-in", fasta, "-dbtype", "nucl", "-out", db],
                       check=True, capture_output=True)
        # Forced, so the test also runs where blastn is on local disk; the
        # default work directory, so the node's staged copy is reused.
        staged = staged_blastn("blastn", db, LIBRARY[:150], stage=True)
        assert staged != "blastn" and os.path.isfile(staged)
        assert staged_blastn("blastn", db, LIBRARY[:150], stage=True) == staged
        query = os.path.join(directory, "q.fa")
        with open(query, "w") as out:
            out.write(f">q\n{LIBRARY[20:190]}\n")
        outputs = [subprocess.run([binary, "-query", query, "-db", db, "-outfmt", "6"],
                                  check=True, capture_output=True).stdout
                   for binary in ("blastn", staged)]
        assert outputs[0] and outputs[0] == outputs[1]


def test_staging_declines_when_told_not_to_or_when_there_is_no_canary():
    assert staged_blastn("blastn", "/no/db", "ACGT" * 50, stage=False) == "blastn"
    assert staged_blastn("blastn", "", "ACGT" * 50, stage=True) == "blastn"
    assert staged_blastn("no-such-blastn", "/no/db", "ACGT" * 50, stage=True) \
        == "no-such-blastn"
