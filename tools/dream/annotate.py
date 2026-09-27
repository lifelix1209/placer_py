"""Re-annotate a world's inserts the way TEBench re-annotates a caller's.

TEBench does not trust a caller's TE label. It runs RepeatMasker, with the
dataset's library, over every called insertion's sequence, and keeps a call
only when TE hits cover at least 100 bp and 50% of it and name a family
(`tebench.normalize.annotate_from_repeatmasker`, then `cli annotate
--require-te`). A replay that scored the policy's own label instead would
count as false positives the calls TEBench silently drops. On HG002 chr1 that
was 26 of pi_1's 51.

So a world recorded with `--record-world` is annotated once, here:
  1. every evaluated row's insert (at least MIN_INSERT_BP) is written to FASTA,
     identical sequences once;
  2. RepeatMasker runs from TEBench's own pinned container, with TEBench's
     command line and the dataset's library;
  3. `objective.score` applies TEBench's function to the policy's calls.

    python3 -m tools.dream.run annotate WORLD --library LIB.fa [--threads 16] [--submit]
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from tools.dream import objective, world

MIN_INSERT_BP = 100
#: TEBench's `containers.repeatmasker` (config/config.yaml), which Snakemake
#: caches under the md5 of its URI.
REPEATMASKER_URI = ("docker://quay.io/biocontainers/repeatmasker@sha256:"
                    "cc2015ef40d2b837245b461997acdc792a02cd1a6f6790bf0a07d9f99ee32d67")
IMAGE = (objective.TEBENCH / ".snakemake" / "singularity"
         / f"{hashlib.md5(REPEATMASKER_URI.encode()).hexdigest()}.simg")


def annotation_dir(w: world.World) -> Path:
    return Path(w.registry).parent / "annotations" / w.name


def export(w: world.World) -> tuple[Path, int, int]:
    """Write the inserts to FASTA and the row-to-sequence map. Returns the
    directory, the number of distinct sequences and of rows mapped."""
    if "insert_seq" in w.missing:
        raise ValueError(f"world {w.name} was not recorded with --record-world: "
                         "it has no insert sequences")
    out = annotation_dir(w)
    out.mkdir(parents=True, exist_ok=True)
    ids: dict[str, str] = {}
    mapped = 0
    with open(out / "insertions.fa", "w") as fasta, open(out / "rows.tsv", "w") as rows:
        rows.write("row_id\tsequence_id\n")
        for row in w.rows:
            seq = str(row.insert_seq or "")
            if len(seq) < MIN_INSERT_BP:
                continue
            sid = ids.get(seq)
            if sid is None:
                sid = ids[seq] = f"s{len(ids)}"
                fasta.write(f">{sid}\n{seq}\n")
            rows.write(f"{row._row_id}\t{sid}\n")
            mapped += 1
    return out, len(ids), mapped


def command(out: Path, library: str, threads: int) -> str:
    """TEBench's `repeatmasker_calls` shell command, with its own HOME for the
    RepeatMasker cache (concurrent jobs race on a shared one)."""
    lib_dir = str(Path(library).resolve().parent)
    return (f"export HOME={out} && singularity exec -B {out} -B {lib_dir} {IMAGE} "
            f"RepeatMasker -pa {threads} -lib {Path(library).resolve()} -dir {out} "
            f"{out}/insertions.fa > {out}/repeatmasker.log 2>&1 && "
            f"cp {out}/insertions.fa.out {out}/repeatmasker.out")


def submit(out: Path, library: str, threads: int) -> str:
    script = out / "repeatmasker.sbatch"
    script.write_text("#!/usr/bin/env bash\n"
                      "#SBATCH --job-name=dream-repeatmasker\n"
                      "#SBATCH --partition=2204\n"
                      f"#SBATCH --cpus-per-task={threads}\n"
                      "#SBATCH --mem=32G\n"
                      "#SBATCH --time=24:00:00\n"
                      "#SBATCH --exclude=node5\n"
                      f"#SBATCH --output={out}/slurm_%j.log\n"
                      "set -euo pipefail\n"
                      f"{command(out, library, threads)}\n")
    done = subprocess.run(["sbatch", str(script)], check=True, capture_output=True, text=True)
    return done.stdout.strip()


def load(w: world.World) -> tuple[Path, dict[int, str]] | None:
    """The RepeatMasker output and the row-to-sequence map, once annotated."""
    out = annotation_dir(w)
    rm_out = out / "repeatmasker.out"
    if not rm_out.exists():
        return None
    mapping: dict[int, str] = {}
    with open(out / "rows.tsv") as handle:
        next(handle)
        for line in handle:
            row_id, sid = line.split()
            mapping[int(row_id)] = sid
    return rm_out, mapping
