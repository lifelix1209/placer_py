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
`0.0.5` mentioned in `placer/config.py` is a *PLACER* release — the C++
project's numbering, not this package's.)

### Added

- **The coverage-rule decision** (introduced as `--te-rule coverage`, now the
  only one; see Removed): the decision promoted from replay.
  - **Collapse regions:** e = 0.
  - **Selection:** ONE e-BH on the decoy-adjusted artifact ratio, over all loci.
  - **TE call** when TEBench's rule holds on the insert (TE hits cover >= 50% of
    it and >= 100 bp), named after the family covering the most of it. A
    structural call otherwise.
  - **Placement:** a locus tested by a wide breakpoint interval is reported at
    its precise single-position hypothesis with the most indel reads, where one
    lies within 100 bp and has >= half the indel reads. The midpoint of the wide
    interval had put truth insertions 100-500 bp off.
  - **Online and replay agree:** calls keep `hypothesis_pos` and are placed from
    it, not from the legacy retether, and the decision sees one candidate per
    ledger observation.
  - **HG002 chr1, scored as TEBench scores:** TP 157 / FP 10 (P 94.0%, R 68.6%),
    against 128 / 13 (90.8%, 55.9%) for `--te-rule likelihood`.

- **TEBench's TE rule, measured on every insert**: the union of all TE-class
  hits in bases and as a fraction of the insert (`te_union_covered_bp`,
  `te_union_coverage`), and the family covering the most of it
  (`te_dominant_family` / `_class` / `_covered_bp`), in the ledger and the
  TSVs. It is the benchmark's own rule (at least 100 bp and 50% TE, non-TE
  classes excluded), read off this run's BLAST hits. Nothing decides on it
  yet.
- **`--record-world`** (development): write each evaluated row's insert
  sequence into the ledger, so the run can be replayed.
- **`tools/dream/`, replay-first development** (after Dream-RSI): a finished
  scan's ledger is a frozen world that decision policies are replayed against
  in under a second.
  - Scoring uses TEBench's own evaluator and RepeatMasker re-annotation.
  - A candidate is accepted by a paired block bootstrap, and a
    coordinate-shift invariance check guards against peeking.
  - Every candidate is logged.
  - The method is in [`docs/development-strategy.md`](docs/development-strategy.md),
    and `CONTRIBUTING.md` and `CLAUDE.md` make it the route for any change to
    what PLACER decides.

- **The form of an LTR insertion** -- `full` (LTR-internal-LTR), `solo`,
  `internal` or `partial` -- read across the library's separate LTR and
  internal-region entries (`MER41A` + `MER41-int`, EDTA's `_LTR`/`_INT`), in
  the ledger, the TSVs and the VCF's `LTRFORM`.
- **The per-class decision, in shadow** (`core/mechanism.py`,
  `core/locus_evidence.py`, `core/mechanism_selection.py`): every evaluated
  locus is scored by two likelihood ratios -- is the insert a TE, is there an
  insertion here -- checked against 100 shifted-breakpoint decoys per locus,
  and selected by e-BH. The ledger and `scientific.txt` record what it would
  select beside the current decision, which it does not yet change.

- **TE classes for any species' library** (`placer/core/taxonomy.py`). Every
  library entry gets a RepeatMasker class (LINE, SINE, Retroposon, PLE, LTR,
  DNA, RC, Unknown, or NonTE for satellites and RNA genes) and a superfamily:
  as stated in a `name#Class/Superfamily` header (Dfam, RepeatMasker, EDTA,
  RepeatModeler2), or inferred from the superfamily in a tldr-style
  `Superfamily:Family` header or from a bare name. EDTA's Wicker codes
  (`DNA/DTA`, `LTR/RLG`, `DNA/DHH`...) are translated. On the benchmark
  libraries: Dfam human 1,349 of 1,432 entries classified from the header, the
  cichlid MWCichlidTE-3.2 library 565 of 599 by superfamily (the other 34 are
  `Unknown` in the library itself). The run log states the counts, and warns
  when over half the library is unclassified.
- **`--library-completeness {curated,denovo}`**. With a curated library (the
  default: Dfam, RepBase) an insert that matches nothing counts against it, as
  before. With `denovo` (EDTA, RepeatModeler2 on a non-model genome) the same
  miss is left uninformative, since it may be an element the library never saw.

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
  `MEINFO` (name, 1-based start and end on the consensus, polarity) is
  emitted on every committed call whose orientation is known; `MEI`,
  `MEISTART` and `MEIEND` go on every record. `TE_CLASS` and `TE_SUPERFAMILY`
  name the element's class and superfamily.
- **`calls.csv`** — the full flat table, both call sets in one file
  distinguished by a `call_set` column, plus `te_qc` and the three
  `sequence_family_*` fields, which are the only record of *why* a family
  abstained and appear in no other output.
- Extracted into a standalone repository with an MIT `LICENSE`, making the code
  legally usable and publishable at all.
- `placer --version`, and `python -m placer` as an equivalent entry point.
- The example dataset covers every class the caller models -- a minus-strand
  L1, an LTR, a hAT DNA transposon and a Helitron join the four SINE, LINE and
  SVA events -- and `truth.tsv` records each insertion's strand.
- `tools/compare_dev_runs.py`: score runs of the development slices against a
  TE truth set and list the TE calls one run gains or loses against another.
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

- **IMPRECISE now says only that the breakpoint is uncertain, and a narrow
  interval is reported at its middle.** Recall on HG002 chr2-8, scored as
  TEBench scores, pooled: TP 849 -> 923, FP 38 -> 42 (P 95.7% -> 95.6%,
  R 68.5% -> 74.5%; bootstrap 90% gain [+0.014, +0.091]). Recall rose on
  every one of the seven chromosomes.
  - **Before.** FILTER=IMPRECISE came from the scan's joint decision, whose
    explanation comparison flagged calls it could not close on both ends. On
    chr1 that dropped 15 TE calls that matched the truth, 14 of them at an
    exact breakpoint.
  - **Now.** A call is IMPRECISE when its breakpoint is known only to an
    interval wider than 200 bp, or not at all. The scan's token stays in QC,
    as evidence.
  - **An interval up to 200 bp wide** is written at its middle, with CIPOS
    bracketing both ends. Before, it was written at its left end. The truth
    sits at either end about equally (quartiles 0.02 / 0.28 / 0.78 across the
    interval), and from the middle every point is within TEBench's 100 bp.
    200 is twice that tolerance: it comes from the evaluation, not a fit.

- **A locus's e-value is now the mean over its hypotheses, not the maximum.**
  The maximum of e-values is not an e-value: under the null its mean can
  reach the number of hypotheses, so a locus evaluated many times got that
  many chances to be called. The mean is an e-value under any dependence.
  The best hypothesis still names, labels and places the call. It is a
  validity fix, adopted at a measured cost on HG002, scored as TEBench scores:
  - chr1: TP 157 / FP 10 -> 154 / 9;
  - chr2-8 pooled: 858 / 42 -> 849 / 38 (P 95.3% -> 95.7%, R 69.2% -> 68.5%).

- **The default decision is now `--decision mechanism --te-rule coverage`.**
  Validated on HG002 chr2-8, which was never used for tuning, scored as TEBench
  scores:
  - pooled TP 858 / FP 42 (P 95.3%, R 69.2%) against 718 / 68 (91.3%, 57.9%)
    for the likelihood rule;
  - bootstrap 90% gain [+0.30, +0.64];
  - better on every one of the seven chromosomes.

  The old default, `--decision legacy`, has since been removed (see Removed).
  Its `calls.vcf` wrote
  each call at the left end of its aggregated breakpoint interval, and those
  intervals reached kilobytes: on HG002 chr1, 217 of 344 TE calls were written
  more than 100 bp from the call's own position. TEBench scores that VCF at
  P 27.7% / R 17.0% on chr1.

- **Alignment-collapse regions no longer produce calls** under `--decision
  mechanism`. A collapse region is one where the sample does not fit the
  reference (a centromere model, a satellite array): no read spans the
  reference, so every hypothesis looks like a homozygous insertion, and the
  counts model's null does not hold. The rule: at least 100 evaluated
  hypotheses within +-50 kb with <=2 reference-spanning reads. Those loci get
  e = 0 and stay in the e-BH family. The ledger marks them
  (`mech_collapse_region`) and `scientific.txt` counts them.
  - HG002 chr1: 24% of the evaluated rows and 1,323 structural calls (no
    confident bases). Replayed TE recall fell by 3 loci, because the invalid
    discoveries had been loosening the shared e-BH threshold.
  - Cichlid slice: nothing.
  - Adopted as a validity fix (`docs/development-strategy.md`, section 2).

- **The structure decode reads the insert in the element's orientation and
  by class.** A minus-strand insert is reverse-complemented first, so a poly(T)
  at the reference 5' end is recognised as the element's poly(A); the tail and
  3' transduction states exist only for TPRT classes (LINE, SINE, Retroposon,
  PLE) and elements of unknown class, so an LTR's or DNA transposon's A-rich
  end is no longer credited as a tail. `core/element_structure.py` measures
  each class's hallmarks -- the oriented poly(A) and transduction, 5'/3' end
  completeness, LTR TG...CA termini, DNA terminal inverted repeats, Helitron
  TC...CTRR -- for the decision layer, and the ledger and TSVs gain
  `polya_len` and `transduction_len`.

- **`blastn` is batched**: a bin's distinct inserts, sorted, 32 to a process,
  the batches of a bin run concurrently. On the 201 inserts of the human
  development slice this took the TE alignment from 116.6 s to 5.6 s. Batching
  changed 8 of their raw hit lists and none of the evidence built from them.
  The batches depend only on the bin, and the cross-bin memo is gone, so
  `--threads N` still writes the same bytes for every N.

- `te_annotation_class` and `te_annotation_order` in the ledger and TSVs are the
  normalised class and superfamily, where they used to be the raw text of the
  header (and `NA` for any library not in `#Class/Superfamily` form).
- **Every call has a strand.** The orientation of the best TE alignment used
  to be discarded while parsing, so `strand` was `NA` on every call; it is now
  the strand of the strongest HSP, relative to the reference, and it reaches
  `strand`, the ledger's new `te_strand`, and the VCF's `MEINFO` polarity. The
  ledger also gains `te_annotation_class`, `te_annotation_order`,
  `te_consensus_start`, `te_consensus_end` and `te_element_length` (from
  BLAST's `slen`), so a call promoted from a ledger row keeps them.
- The latent-mechanism model's family kind (`policy.family_kind`) is read from
  the class instead of from substrings of the element's name. Elements the old
  token match missed now get their mechanism prior -- SINE/MIR and tRNA-derived
  SINEs as retro, every cichlid Gypsy and Pao as LTR -- and an element whose
  class is Unknown is `unknown` rather than `other`.

- **Faster on one process, with byte-identical output**: on 0.9 Mb of HG002
  ONT-UL, CPU 977 s to 73 s and wall 1450-2425 s to 80 s. Peak memory rose
  from 469 MB to 711 MB (a per-read CIGAR index and a bounded cache of
  recently fetched reads). See "Speed" in the README for where the time went.
  `rapidfuzz` joins the `scan` extra; without it the pure-Python edit
  distance is used and the answer is the same.
- **The package is now three named stages**: `placer/io/` (everything that
  talks to something outside the process — pysam, BLAST, abPOA), `placer/
  core/` (everything that decides something) and `placer/report/`
  (everything that renders). `core` may import neither of the other two, at
  module scope or inside a function body, and `tests/test_37_layering.py`
  enforces that rather than leaving it to a comment. **Import paths changed**:
  `placer.finalization` is now `placer.core.finalization`,
  `placer.outputs` is `placer.report.tsv`, `placer.bam_io` is
  `placer.io.bam`, and so on. No shims were left behind, so a stale import
  fails loudly instead of resolving to something that no longer means what it
  did. Every existing output file is byte-identical across the whole move.
- Gate-1 is no longer a closure inside the orchestrator: `placer/io/gate.py`
  applies it and can be run, counted and replaced on its own. The predicate and
  its thresholds in `placer/reads.py` are unchanged to the byte.
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

### Changed

- **Renamed to PLACER.** The package is `placer` (was `placer_py`), the
  command is `placer` (was `placer-py`), and the distribution is `placer-te`,
  because `placer` on PyPI belongs to an unrelated project. Version
  `1.0.0.dev0`: this implementation replaces the C++ rather than porting it.

### Removed

- **Non-TE insertions are no longer reported.** PLACER is a TE caller. An
  insertion the decision selects whose insert fails TEBench's TE rule used to
  be written to three places, all of which change:
  - `structural_calls.tsv` is gone;
  - `calls.vcf` no longer writes it as `FILTER=STRUCTURAL`;
  - `calls.csv` no longer writes it as `call_set=structural`. The `call_set`
    column stays, now always `final`, so that older tables read the same.

  The selection itself is unchanged: the ledger marks those insertions
  (`mech_structural_selected`) and `scientific.txt` counts them. Checked on
  the example data (with and without `--record-world`) and on two HG002 chr1
  regions:
  - `evidence_ledger.tsv` and `scientific.txt` are byte-identical;
  - `calls.vcf` and `calls.csv` differ only by the structural records and the
    `STRUCTURAL` FILTER header line.

  The TEBench score cannot move, because TEBench keeps only PASS calls.
- **The legacy decision and its options.** `--decision`, `--te-rule`,
  `--final-report-mode` and `--min-final-raw-cigar-insert-len-bp` are gone,
  with `core/finalization.py`, `core/call_selection.py` (the per-component
  selection and retether), `core/dependency.py`, `core/conformal.py`,
  `core/integrate.py`, `core/decoys.py`, the likelihood-gated TE rule and its
  identity-prior fit, and their tests: 4,300 lines of the package and 1,700
  of tests, net. The default outputs are byte-identical: `calls.vcf`, `calls.csv`,
  `structural_calls.tsv` and `evidence_ledger.tsv` on the example data (with
  and without `--record-world`) and on two HG002 chr1 regions
  (30.0-30.5 Mb, and 200 kb around 24.6 Mb). `scientific.txt` loses one
  line, `mech_identity_priors`, which reported the unfitted uniform prior. The
  scan's joint decision (`core/policy.py`, `core/blocks.py`) stays: it still
  sets the FILTER tokens (IMPRECISE, FAM_ABSTAIN). The dependency-bound
  fields are no longer set, and keep their defaults in the VCF header and the
  summary until a change that may alter the output removes them.
- The C++ golden-vector oracle (`tests/oracle/cpp_reference.json`,
  `tools/dump_oracle.cpp`, `tools/regenerate_oracle.sh`) and the tests that
  asserted equality against it are gone; placer is now the reference
  implementation. The record of how it departed from the C++ up to that point
  is kept as [`docs/departures-from-cpp.md`](docs/departures-from-cpp.md).
- **`placer-py denovo`** (trio de novo calling) and `placer_py/redesign/` (the
  pre-port implementation that took candidates from a Sniffles VCF). The one
  piece of the redesign nothing else had -- the L1 endonuclease motif -- moved
  to `placer/core/endonuclease.py` first; see Fixed.

### Fixed

- **Opt-in (`PLACER_SAME_ALLELE_CARRIER_WINDOW_BP=500`, off by default): carriers
  of an insertion that the aligner placed elsewhere were counted as reference
  support.** In a tandem repeat or low-complexity flank, ONT reads
  carry one insertion at offsets hundreds of bp apart. A hypothesis counted an
  insertion as alt only within +-25 bp of its bounds. Any other carrier
  spanning that window counted as REFERENCE.
  - A homozygous 312 bp AluSz (HG002 chr1:24,634,837) carried by all 52
    spanning reads scored 5 alt against 35 reference, and so as an artifact.
  - A read with a uniquely mapped insertion of the allele's length (+-30% of the
    tight window's median) within 500 bp now counts for that allele, as alt and
    not as reference. The ledger reports how many (`alt_carrier_reads`).
  - At four such truth loci, alt went from 2-5 to 20-52 and the artifact ratio
    from -14 to +10 and +47.
  - Off by default: under the current decision the recovered loci are placed
    at off-mode offsets. On HG002 chr1 the replay was +3 TP and +3 FP, a gain of
    -0.140 (90% [-0.39, +0.12]), rejected until placement handles dispersed
    carriers.

- **Re-genotyping ignored the scan's genotype inputs.** Finalization
  re-genotypes each call with the sample's overdispersion and is meant to
  reuse everything else the scan used, but the scan never passed those inputs
  on. Every call was therefore re-genotyped with error rate 0.02, whatever
  `PLACER_GENOTYPE_ERROR_RATE` said, and without the length-concordance term.
  On the synthetic test locus, GQ was 16 from the defaults and 7 from the
  configured error rate of 0.07. Both decision modes were affected.

- **`blastn` concurrency is sized from the CPUs the job was given**, not from
  every core on the machine. On a 128-core cluster node with a 16-CPU SLURM
  allocation, `--threads 16` ran 128 `blastn` processes at once and two such
  jobs drove the node to a load of 233.

- **Insertions whose supporting reads started in the previous scan bin were
  never called.** A bin got only the reads that START in it, and a candidate
  is kept only by the bin that owns its anchor, so a candidate formed from
  reads that started earlier was discovered in their bin and discarded there.
  With reads longer than the 10 kb bins -- all ultra-long ONT -- most of an
  insertion's carriers start earlier. A bin now receives every read that
  overlaps it. On the example data a minus-strand L1 14 bp into a bin, with
  ten carriers, went from not called to called.

- **`--threads N` no longer hangs forever when a worker dies.** A scan worker
  killed by a signal left `multiprocessing.Pool` waiting on a result that would
  never come; on the first cluster run every worker died of SIGILL inside
  pyabpoa and the job sat idle for an hour. The pool is now a
  `ProcessPoolExecutor`, and a dead worker ends the run with exit code 1 and a
  message saying what to check.
- `pyabpoa` 1.5.3 is excluded (`pyabpoa>=1.4,!=1.5.3`): that bioconda build
  dies of SIGILL as soon as an aligner is created on AMD EPYC Zen 3 nodes.
  1.5.4 to 1.5.7 run on the same machines.

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
