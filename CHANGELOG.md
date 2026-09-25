# Changelog

Notable changes, in the terms a user of the tool would notice. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

Numerical behaviour that departs from the C++ implementation on purpose is not
summarised here: each departure is recorded in
[`docs/departures-from-cpp.md`](docs/departures-from-cpp.md) with what moved,
by how much, and how it was measured. This file says *that* something changed;
that file says what it cost.

## [Unreleased]

**0.1.0 has never been published.** The version in `pyproject.toml` is the one
this package was extracted at; there is no release on PyPI or bioconda yet, so
everything below is the state of `main` rather than an upgrade path. (The
`0.0.5` mentioned in `placer_py/config.py` is a *PLACER* release — the C++
project's numbering, not this package's.)

### Added

- **`--threads N`** (`-t`): the scan runs on N processes. The genome is cut at
  bin boundaries, the pieces are scanned independently and rejoined in genome
  order, and finalization runs once over the whole run, so the output files
  are byte-identical for any N (`tests/test_38_parallel.py` checks this on a
  real BAM). 8 processes: 0.9 Mb of HG002 ONT-UL in 21 s.
- **`calls.vcf`** — VCF 4.2, so the output can be read by `bcftools`, `truvari`,
  `SURVIVOR` and anything else that speaks the format. The inserted sequence is
  written out as the ALT allele rather than a symbolic `<INS:ME:*>`, because the
  sequence is the evidence; a call whose sequence could not be assembled keeps
  its record as `<INS>` with `FILTER=ALTSEQ_MISSING` rather than disappearing.
  `MEINFO` is declared and emitted on no record — its polarity field is not
  optional and this build does not resolve insertion orientation, so `MEI`,
  `MEISTART` and `MEIEND` carry the three components that are known.
- **`calls.csv`** — the full flat table, both call sets in one file
  distinguished by a `call_set` column, plus `te_qc` and the three
  `sequence_family_*` fields, which are the only record of *why* a family
  abstained and appear in no other output.
- Extracted into a standalone repository with an MIT `LICENSE`, making the code
  legally usable and publishable at all.
- `placer-py --version`, and `python -m placer_py` as an equivalent entry point.
- A runnable example dataset (`examples/make_example_data.py`) with a known
  truth set, and the first end-to-end run of the pipeline on real files rather
  than on literals in a test.
- A GIAB HG002 ONT-UL evaluation harness (`tools/make_giab_eval.py`) that cuts
  a development slice and a holdout slice, with the labels in the manifest: a
  change that needs the holdout to justify it is a fit, not a fix.
- `py.typed`, so the annotations are visible to a downstream mypy or pyright
  instead of being erased to `Any`.
- `docs/off-pipeline-modules.md`, stating which of the modules nothing imports
  is deliberate and why.

### Changed

- **Faster on one process, with byte-identical output**: on 0.9 Mb of HG002
  ONT-UL, CPU 977 s to 73 s and wall 1450-2425 s to 80 s. Peak memory rose
  from 469 MB to 711 MB (a per-read CIGAR index and a bounded cache of
  recently fetched reads). See "Speed" in the README for where the time went.
  `rapidfuzz` joins the `scan` extra; without it the pure-Python edit
  distance is used and the answer is the same.
- **The package is now three named stages**: `placer_py/io/` (everything that
  talks to something outside the process — pysam, BLAST, abPOA), `placer_py/
  core/` (everything that decides something) and `placer_py/report/`
  (everything that renders). `core` may import neither of the other two, at
  module scope or inside a function body, and `tests/test_37_layering.py`
  enforces that rather than leaving it to a comment. **Import paths changed**:
  `placer_py.finalization` is now `placer_py.core.finalization`,
  `placer_py.outputs` is `placer_py.report.tsv`, `placer_py.bam_io` is
  `placer_py.io.bam`, and so on. No shims were left behind, so a stale import
  fails loudly instead of resolving to something that no longer means what it
  did. Every existing output file is byte-identical across the whole move.
- Gate-1 is no longer a closure inside the orchestrator: `placer_py/io/gate.py`
  applies it and can be run, counted and replaced on its own. The predicate and
  its thresholds in `placer_py/reads.py` are unchanged to the byte.
- `run_pipeline` is now a composition of `core.scan.run_scan` and
  `core.finalize.finalize_run`, so a caller can scan without calibrating, or
  re-calibrate a scan it already has. Its own signature and behaviour are
  unchanged.
- Peak memory during a scan is now O(one bin) rather than O(genome): reads are
  streamed through the bin loop instead of being materialised first, which is
  what makes a genome-scale run possible at all.
- The banded edit-distance DP was rewritten: 2.3x less CPU, byte-identical
  output.
- The event consensus takes an explicit memory budget, and reports when the
  budget capped it, rather than being bounded indirectly by a read cap that
  cannot bound it.
- `clamp`, `log_sum_exp` and the count models have one definition each. Nine
  copies of `clamp` gave three different answers for NaN, and the surviving
  policy propagates it; `log_sum_exp` keeps both of the semantics that were in
  use, now spelled as `ignore_nonfinite=`.
- Tooling: ruff and mypy are CI gates, the test job covers Python 3.9–3.12, and
  the packaging job installs the built wheel into a clean environment and runs
  the console script.

### Removed

- The C++ golden-vector oracle (`tests/oracle/cpp_reference.json`,
  `tools/dump_oracle.cpp`, `tools/regenerate_oracle.sh`) and the tests that
  asserted equality against it are gone; placer is now the reference
  implementation. The record of how it departed from the C++ up to that point
  is kept as [`docs/departures-from-cpp.md`](docs/departures-from-cpp.md).
- **`placer-py denovo`** (trio de novo calling) and `placer_py/redesign/` (the
  pre-port implementation that took candidates from a Sniffles VCF). The one
  piece of the redesign nothing else had -- the L1 endonuclease motif -- moved
  to `placer_py/core/endonuclease.py` first; see Fixed.

### Fixed

- The L1 endonuclease motif's minus-strand window was assembled in the wrong
  order (`revcomp(right[:4] + left[-2:])`), so a perfect bottom-strand
  5'-TTTT|AA-3' site scored as four mismatches out of six. It is now
  `revcomp(left[-2:] + right[:4])`, pinned in `tests/test_39_endonuclease.py`.
  The motif is not on the calling path yet, so no call changes.

Caller fixes from the first real-data runs, each recorded with its measured
effect in `docs/departures-from-cpp.md` (5-9). On HG002 chr21:10-20 Mb: TE
calls 12 -> 9, GIAB TE truth recalled 2/3 -> 3/3, calls labelled from a simple
repeat or a <= 30 bp match 7 -> 0.

- **Microsatellites were called as transposable elements.** An (AT)n or
  (AAAG)n expansion aligns to the same repeat inside an L1 or LTR consensus,
  and was reported as that element; so were 16-30 bp matches inside short
  inserts. A hit now needs 50 aligned bases outside simple repeat to name an
  element (`TE_ALIGNMENT_UNINFORMATIVE` otherwise).
- **Heterozygous insertions could be refused for being heterozygous.** Every
  reference-spanning read counted as a conflict for "an insertion is here",
  so a het insertion with more reference than split+indel reads could not win
  the explanation comparison. Only reference reads beyond the het balance
  count now. This recovered a 3.4 kb het L1 on HG002 and the het SVA in
  `examples/data`.
- **An L1 split across library entries was under-covered.** Dfam models L1 as
  `_5end`/`_orf2`/`_3end`; coverage is now the chosen family's, not one
  entry's.
- **21.8% of evidence-ledger rows were exact duplicates**, counted as separate
  observations by every whole-run estimate. Each observation is now one row.
- Cluster-promoted calls reported `insert_len` 0 while carrying a sequence.
- **TSD detection could not report a duplication at all.** Every detection
  returned `NONE`, so no call carried a target-site duplication — the single
  most mechanistically informative field for a TPRT insertion.
- `log_sum_exp` returned NaN when every input was the impossible sentinel; it
  now returns `-inf`, which is what "nothing is possible" means.
- The structure explanation no longer zeroes the shadow path when the TE
  alignment explains nothing, matching the C++ and the golden vectors.
- `finalization` no longer carries its own inline copy of the e-BH selection it
  shares with `selection.py`, where the argument for averaging rather than
  maximising e-values is written down.
- Six module docstrings cited test files that do not exist, five of them
  pointing at a real file about a different subject.
