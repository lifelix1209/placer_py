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

**The scan genotypes from the counts too.** 1.0.0a3 took the
length-concordance term out of the reported genotype. This takes it out of
the scan's existence evidence as well, and out of the genotyper
(`docs/departures-from-cpp.md` section 10).

**Measured.** Every release run was rerun from frozen code
(`placer_dev/frozen/a4-rc1`) with its own arguments: HG002 chr1-8 and the
cichlid slice. Each was compared with its 1.0.0a3 rerun.
- **The ledger.**
  - The two new columns are appended.
  - Every non-EVALUATED row is identical.
  - EVALUATED rows fall by 2-5%: chr1 44,614 → 43,845, cichlid 35,140 →
    33,634. Every dropped row has a kept twin that differs only in
    GQ-dependent columns.
- **TEBench.** chr1 (development) 171/10/58 and chr2-8 pooled (validation)
  923/42/316, both unchanged.
  - Four HG002 calls moved, all four true positives. chr4 lost ERV1
    135407205 and gained Alu 90372928 and Alu 142337150. chr8 lost SVA
    128825914.
- **Cichlid.** Two calls went: chr1:10526957 (PASS, hAT-Ac) and 17935507
  (IMPRECISE). Neither has a Sniffles2 or tldr call within 1 kb.
- **The replay emulation** (dream nodes 06302ca628 and 97f11604c2) predicted
  chr1's zero change and cichlid's two losses, and bounded every run's row
  count.
- **A replay of each new ledger** gives that run's PASS calls exactly.
- **The sample overdispersion rises** (chr1 0.1729 → 0.1796). Through it,
  genotype concordance on chr2-8 goes from 878/923 to 875/923; chr1 is
  unchanged at 149/171.

### Changed

- **The scan-time GQ** is zygosity from the allele counts. Two things read it:
  - the joint decision's diagnostics: the INFO fields TEPOST, LFDRMAX,
    MECHART, MECHNONTE, MECH and QC, and the ledger's posterior, lfdr and
    mechanistic_* columns;
  - the per-bin de-duplication of ledger rows.
- **Fewer duplicate rows.** Rows two components wrote for one hypothesis are
  now one row. Before, the term read each component's own breakpoint
  candidates, so the copies differed in GQ and both survived. Merging them
  can move:
  - the sample overdispersion, and so the reported GT and GQ;
  - the locus-mean e-value, and so a call.

### Added

- **Ledger columns `alt_measured_length_reads` and `alt_measured_lengths`**:
  what each alt read measured the insertion to be. That is a CIGAR insertion
  or an SA-implied one, at least 50 bp, one per read, never a clip.
  - They are recorded for a later replay candidate; no decision reads them.
  - Worlds recorded earlier load them as -1 and NA.

### Removed

- **Removed functions:**
  - `genotype.length_concordance_factor`;
  - `hypotheses.collect_alt_observed_lengths`;
  - `hypotheses.infer_event_length_from_alt_support`.
- **Removed fields:**
  - the `event_length` and `alt_observed_lengths` fields of `GenotypeInput`
    and `EventGenotypeInput`;
  - the parameters of the same names of `genotype_from_alt_vs_ref`;
  - `HypothesisSummary.inferred_event_length`, which nothing read.
- **Signature change:** `build_hypothesis_summary` no longer takes the
  component.

## [1.0.0a3] - 2026-09-29

**The same calls as 1.0.0a2, with correct genotypes.** No call, position,
FILTER, QUAL or family moved; only GT and GQ did. PASS calls were reported
0/0 when their read counts said 0/1 or 1/1. The deeper the data, the more
often this happened.

It was found in TEBench's release benchmark of 1.0.0a2
(`docs/development-strategy.md`, "Release benchmark 1.0.0a3"). It was checked
on the development contigs, HG002 chr1-8, and nothing else: the holdout had
not been scored. In the whole-genome runs, this share of PASS calls on chr1-8
was genotyped 0/0:

| coverage | 5x | 10x | 20x | 30x | full |
|---|---|---|---|---|---|
| PASS calls genotyped 0/0 | 0% | 3% | 10% | 14% | 22% |

**Measured.** Every release run was rerun from the frozen fix with its own
arguments: HG002 chr1-8 and the cichlid slice, 16 threads, `--record-world`.
- `evidence_ledger.tsv` is byte-identical in all nine.
- `calls.vcf`, `calls.csv` and `scientific.txt` are identical except for GT
  and GQ. DP, AD and AF are unchanged.
- In the PASS calls, all 299 0/0 genotypes on HG002 chr1-8 became 0/1 (203)
  or 1/1 (96). So did all 18 on the cichlid slice (10 and 8). No 0/1 or 1/1
  call changed.
- TEBench scores are unchanged: chr1 171/10/58 (P 94.5%, R 74.7%); chr2-8
  923/42/316 (P 95.6%, R 74.5%).
- Genotype concordance with GIAB v5.0q on the true positives, Wilson 95%:

| | 1.0.0a2 | 1.0.0a3 |
|---|---|---|
| chr1 (development) | 121/171 = 70.8% [63.5, 77.1] | 149/171 = 87.1% [81.3, 91.3] |
| chr2-8 pooled (validation) | 772/923 = 83.6% [81.1, 85.9] | 878/923 = 95.1% [93.5, 96.3] |

- The cost is unchanged. chr1 took 2.06 CPU-h.

### Fixed

- **The reported genotype comes from the allele counts**
  (`core/finalize.apply_sample_overdispersion_calibration`).
  - **The cause.** Finalization re-genotypes each call with the sample's
    overdispersion. Since 1.0.0a1 it also applied the scan's
    length-concordance term, which has two faults:
    - It averages over the few reads that report a length, then charges
      that mean once per alt read. At chr2:171976066, an 8.3 kb insertion
      with 66 alt reads and no reference read, two soft clips of 25 and
      28 bp cost -39 nats. They are lower bounds that say nothing about an
      8.3 kb insert.
    - The lengths it scores include clip lengths, which are only lower
      bounds.
  - **Why it grew with depth.** The whole-sample overdispersion (0.14 on
    HG002) caps the count evidence near 20 nats, while the penalty grows
    with every alt read.
  - **The fix.** Existence is settled before finalization, so the reported
    genotype no longer takes a second vote on it. The configured error rate
    and min-GQ are still used.
  - **Not changed.** The scan still uses the same term, at the default
    overdispersion of 0.02, in the GQ its own diagnostics read. Those are
    the INFO fields TEPOST, LFDRMAX, MECHART, MECHNONTE, MECH and QC.
    - None of them decides emission, PASS, position or family.
    - They do enter the equality test that de-duplicates ledger rows. So a
      fix there is checked the way this release was: frozen A/B runs, calls
      compared record for record.

## [1.0.0a2] - 2026-09-28

**The same calls as 1.0.0a1, about five times faster.** Every release run of
1.0.0a1 was rerun from the frozen 1.0.0a2 code with that run's own arguments:
HG002 chr1-8 and the cichlid slice, 16 threads, `--record-world`. All four
output files are byte-identical to 1.0.0a1's, `calls.vcf` apart from the
`##source` line that names the version. So every accuracy number of 1.0.0a1
stands unchanged.

| run, 16 CPUs | 1.0.0a1 CPU-h | 1.0.0a2 CPU-h | | 1.0.0a1 wall | 1.0.0a2 wall | peak memory of a worker (GB) |
|---|---|---|---|---|---|---|
| HG002 chr1 | 10.28 | 2.05 | 5.0x | 2 h 55 min | 12.8 min | 12.5 -> 12.6 |
| HG002 chr2 | 7.26 | 1.17 | 6.2x | 2 h 13 min | 6.4 min | 2.2 -> 2.2 |
| HG002 chr3 | 5.58 | 0.89 | 6.3x | 2 h 5 min | 5.1 min | 2.3 -> 2.4 |
| HG002 chr4 | 4.08 | 0.99 | 4.1x | 43 min | 7.3 min | 3.6 -> 4.3 |
| HG002 chr5 | 3.16 | 0.84 | 3.8x | 32 min | 4.6 min | 1.9 -> 1.9 |
| HG002 chr6 | 3.36 | 0.86 | 3.9x | 35 min | 5.0 min | 2.9 -> 3.5 |
| HG002 chr7 | 4.64 | 0.86 | 5.4x | 81 min | 4.5 min | 2.3 -> 2.1 |
| HG002 chr8 | 4.06 | 0.73 | 5.6x | 75 min | 4.0 min | 2.3 -> 2.3 |
| **HG002 chr1-8** | **42.4** | **8.38** | **5.1x** | | | |
| cichlid chr1:10-20 Mb | 6.19 | 0.60 | 10.3x | 60 min | 2.4 min | 2.0 -> 2.0 |

CPU is user + sys, as TEBench's `cpu_hours` counts it. The target for HG002
chr1 was 2.5 CPU-h. Scaled by length, chr1-8 point to about 17 CPU-h for the
whole genome, against a target of 30.

**Known: a little more memory in the worst region.** The 1q21 segmental
duplications, chr1:143.13-143.32 Mb, run by one worker with `--threads 1`:

| | CPU | peak memory |
|---|---|---|
| 1.0.0a1 | 4,434 s | 12.4 GB |
| 1.0.0a2 | 880 s | 13.3 GB |

In the 16-thread chr1 run the heaviest worker peaks at 12.6 GB, against 12.5.
chr4's heaviest worker went from 3.6 to 4.3 GB, and chr6's from 2.9 to 3.5.

### Changed

- **Nothing about the decision.** No threshold, model or output field moved.
- **Measured the same way throughout.** Every change was run from frozen
  snapshots on four workloads, with all four outputs compared byte for byte:
  typical, pericentromeric and centromeric HG002 regions; a cichlid slice;
  three concurrent 16-worker jobs; and whole HG002 chr1. Each rewritten
  kernel also has a test that keeps the old implementation verbatim beside the
  new one (`tests/test_49`-`test_57`).
- **System time: blastn is staged on node-local disk** (`io/blast.staged_blastn`).
  - **The cause.** A conda BLAST+ loads 86 libraries (535 MB) at every exec,
    and BeeGFS kept none of their pages cached. With 48 workers on a node that
    cost 0.6-0.9 s of system time per call. It was 90% of the run's sys time,
    and why sys reached 50-200% of user time on some nodes.
  - **The safeguards.** The copy is used only when `ldd` resolves it exactly
    as the original, and a canary search gives the same bytes.
    `PLACER_STAGE_BLASTN=0/1` overrides the choice.
  - **The effect.** HG002 chr1's sys time went from 13,755 s to 1,010 s.
- **Reads and reference.**
  - **Local reads come from the chunk's own stream** (`io/bam.StreamedReadBuffer`)
    wherever it provably holds every read an indexed fetch would return, and
    from the index otherwise. Filesystem input on chr1 went from 1.43e9 to
    4.4e7 blocks. `PLACER_VERIFY_LOCAL_FETCH=1` answers every fetch both ways
    and fails on any difference.
  - **The reference is read in cached 64 kb blocks.**
- **The soft-clip low-complexity test was 27% of chr1's CPU.** It ran four
  per-base passes over clips of up to tens of kb, for every fragment of every
  hypothesis.
  - The four tests now run cheapest first, on base counts.
  - A clip with more k-mers than 4^k / threshold is decided without building
    its k-mer set.
- **The flank search follows diagonal chains** (`segmentation._chain_candidates`).
  - Along a diagonal the edit distance never decreases and grows by at most
    one per base, so a few kernel calls per chain answer every length.
  - The pair search stops as soon as no remaining flank can beat the best
    pair's weaker side.
  - Replayed on 2,243 real segmentation calls, the stage is 10-20x faster
    with identical results.
- **The TSD detector**, run for every locus and each of its 100 decoys.
  - Its windows are sliced, not fetched per candidate length.
  - The exact pass is a substring search, and the tolerant pass counts
    mismatches bit-parallel.
  - The insertion form is one pass outward from the junction.
- **Answers reused for the same input:**
  - abPOA consensuses, 62% of calls on chr1;
  - insert composition features and simple-repeat masks;
  - canonical k-mer keys.
- **Other pure-Python kernels.**
  - The breakpoint posterior uses tabulated kernels, in the same summation
    order.
  - The tandem and microsatellite masks compare the sequence with itself
    shifted.
  - The low-complexity window slides.
  - Each read's CIGAR is walked once, for its index, the gate's summary and
    its long insertions.
- **Load balance.** The chunks holding the most alignment data, from the BAM
  index, start first. A chunk estimated above twice the median is cut into
  bin-aligned pieces. On chr1, three chunks (the pericentromere and 1q21) had
  been 52% of the CPU and set the wall time.

### Added

- `PLACER_PERF_LOG=PATH` writes one TSV row per scanned chunk: wall time; the
  CPU of the worker and of its blastn children; I/O; fetch and cache
  counters. `tools/perf/perf_summary.py` summarises it.
- `tools/perf/ab_step.sh` A/Bs two frozen snapshots, including three
  concurrent 16-worker jobs.
- `tools/perf/kernel_corpus.py` records a region's segmentation calls and
  replays them against the code on disk.
- `PLACER_SCAN_CHUNK_BP` and `PLACER_TE_BLAST_JOBS` set the chunk width and the
  blastn concurrency. Neither changes an output.

## [1.0.0a1] - 2026-09-27

The first tagged version: an alpha of 1.0. It is on GitHub only, not on PyPI
or bioconda, and there is no earlier version to upgrade from, so everything
below is what 1.0.0a1 is rather than what changed since a release. (The `0.0.5`
mentioned in `placer/config.py` is a *PLACER* release -- the C++ project's
numbering, not this package's.)

**Where it stands**, from this version's own runs, scored by TEBench's
pipeline (`tools/tebench_score.py`): HG002, GIAB v5.0q TE truth, confident
regions, +-100 bp.

| | TP / FP | precision | recall |
|---|---|---|---|
| chr1 (development) | 171 / 10 | 94.5% | 74.7% |
| chr2-8 pooled (validation) | 923 / 42 | 95.6% | 74.5% |
| Sniffles2 on chr2-8, for reference | 1025 / 46 | 95.7% | 82.7% |

- **Cichlid.** On the cichlid slice, 80.0% of the 180 PASS TE calls lie
  within 100 bp of a Sniffles2 insertion, and 58 of tldr's 78 PASS calls are
  matched.
- **Cost.** HG002 chr1-8 took 42.4 CPU-hours.

The README's "1.0.0a1: where it stands" describes how it decides and what it
does not do yet.

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

- **Renamed to PLACER.** The package is `placer` (was `placer_py`), the
  command is `placer` (was `placer-py`), and the distribution is `placer-te`,
  because `placer` on PyPI belongs to an unrelated project. The version
  line starts at 1.0: this implementation replaces the C++ rather than porting it.

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

- **Runs that started together on one node could fail building the BLAST
  database.** The database is cached per node and shared, and two runs that
  both found it missing ran `makeblastdb` into the same files; one of them
  died with "failed to build BLAST database". Seen on the cluster in one of
  eight jobs submitted at once. The build now holds an exclusive lock, and a
  run that waited uses the finished database.
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
