# PLACER

A Python implementation of [PLACER](https://github.com/lifelix1209/PLACER),
the long-read transposable-element insertion caller. It runs end to end — BAM
in, calls and an evidence ledger out — without the compiled binary.

This repository began as a port of the C++ and is now the reference
implementation itself: the C++ golden-vector oracle has been removed, and how
this implementation departed from the C++ up to that point is recorded in
[`docs/departures-from-cpp.md`](docs/departures-from-cpp.md). Every module's
docstring still names the C++ file it came from.

## 1.0.0a3: where it stands

An alpha of PLACER 1.0. It calls **TE insertions** from long reads (ONT): BAM
in, VCF out. An insertion that is not a TE is recorded in the evidence ledger
but not reported.

1.0.0a3 makes exactly the calls 1.0.0a1 made:
- 1.0.0a2 made them about five times faster. Every 1.0.0a1 release run was
  rerun with its own arguments, its output files were the same bytes, and
  HG002 chr1-8 takes 8.4 CPU-hours against 42.4 (see [Speed](#speed)).
- 1.0.0a3 reports their genotypes correctly. Genotype concordance on HG002
  chr2-8 went from 83.6% to 95.1% (`CHANGELOG.md`); no call moved.

**How it decides.**

1. **The scan**, per bin, on `--threads N` processes with the same bytes out
   for any N:
   - it gates the reads and clusters their insertion signals;
   - it enumerates breakpoint hypotheses and counts the alt and reference
     reads of each;
   - for the shortlisted hypotheses it assembles the insert (abPOA), segments
     it from its flanks, and aligns it to the TE library (BLAST).
   - It then measures TEBench's TE rule on the insert: how much of it TE hits
     cover, and the family covering the most.
   - It scores two per-class likelihood ratios (`core/mechanism.py`):
     - *is an insertion here* (`vs_artifact`): the read counts, a TSD by the
       superfamily, the L1 endonuclease motif;
     - *is it a TE* (`vs_non_te`): identity to the element, 3' anchoring,
       tail, termini.
   - It checks the linkage terms against 100 shifted-breakpoint decoys.
   - Every evaluated hypothesis is one ledger row.
2. **Finalization**, once, over the whole run (`core/mechanism_selection.py`):
   - hypotheses in alignment-collapse regions get e = 0;
   - a per-class decoy check bounds each class's e-values;
   - hypotheses within 300 bp are one locus, tested once, on the mean of its
     hypotheses' e-values;
   - one e-BH at q = 0.1 controls the FDR of "an insertion is here";
   - a selected insertion is a TE call when TE hits cover >= 50% and >= 100 bp
     of it, and it is named after the family covering the most;
   - a call tested by a breakpoint interval moves to the locus's exact
     hypothesis with the most indel reads, where one lies within 100 bp;
   - calls are genotyped from their allele counts, with the sample's own
     overdispersion.
3. **The VCF.**
   - FILTER=IMPRECISE only when the breakpoint is known to no better than an
     interval wider than 200 bp; a narrower interval is written at its middle,
     with CIPOS.
   - FAM_ABSTAIN marks a TE call whose class is not committed.

**How well**:
- the calls are 1.0.0a1's: 1.0.0a2 reproduces them byte for byte, and
  1.0.0a3 changes only their GT and GQ;
- scored by TEBench's pipeline (`tools/tebench_score.py`);
- HG002, GIAB v5.0q TE truth, confident regions, +-100 bp.

| caller | chr1 TP / FP | chr1 P / R | chr2-8 TP / FP | chr2-8 P / R |
|---|---|---|---|---|
| **PLACER 1.0.0a3** (= 1.0.0a1) | 171 / 10 | 94.5% / 74.7% | 923 / 42 | 95.6% / 74.5% |
| Sniffles2 | 180 / 11 | 94.2% / 78.6% | 1025 / 46 | 95.7% / 82.7% |
| GraffiTE | 166 / 10 | 94.3% / 72.5% | 909 / 49 | 94.9% / 73.4% |
| cuteSV | 171 / 15 | 91.9% / 74.7% | 957 / 63 | 93.8% / 77.2% |
| tldr | 119 / 2 | 98.3% / 52.0% | 674 / 15 | 97.8% / 54.4% |

- **Where the numbers come from.** The other callers are TEBench's own runs,
  scored the same way. chr1 is where the decision layer was developed. chr2-8
  were used only to validate candidates, never to develop them. TEBench's
  holdout contigs have not been looked at.
- **The same numbers two ways.** Each run's PASS calls are exactly the
  replay of its own evidence ledger, and the replay scores the same numbers.
- **Cichlid**, `chr1:10-20 Mb`, no truth set. 180 PASS TE calls:
  - 80.0% lie within 100 bp of a Sniffles2 insertion;
  - 58 of tldr's 78 PASS calls are matched.
- **Cost**, 16 threads, 1.0.0a2 (1.0.0a3 costs the same):
  - HG002 chr1-8 took 8.4 CPU-hours in all (1.0.0a1: 42.4);
  - chr1 took 2.05 CPU-hours and 13 min (1.0.0a1: 10.3 and 2 h 55 min);
  - one chr1 worker peaks at 12.6 GB in 1q21, the others at 2-4 GB;
  - cichlid took 0.6 CPU-hours per 10 Mb (1.0.0a1: 6.2);
  - the whole HG002 genome (TEBench full/r0, 32 threads) took 18.7
    CPU-hours and 50 min, with a peak of 19 GB. The target was 30.

**What it does not do yet.**
- **Whole-genome accuracy.** The whole genome has been run (see Cost), but it
  is scored outside chr1-8 only once TEBench's release benchmark finishes.
- **Tandem repeats.** An allele that the aligner scatters across a tandem
  repeat loses its reads to the reference count; that is most of what is
  left of the recall gap to Sniffles2.
- **Other species.** Only human is scored against a truth set. The cichlid
  slice is checked for consistency with Sniffles2 and tldr; mouse is not
  run yet.
- **Inputs and outputs.** No `--ploidy` and no CRAM.
- **Distribution.** It is not on PyPI or bioconda.

How the decision layer is developed -- replayed against frozen scans, and
accepted only on a TEBench-exact bootstrap -- is in
[`docs/development-strategy.md`](docs/development-strategy.md).

## How it got here

**The migration is complete, and it now covers the whole pipeline rather than
the decision layer alone.** The first pass ported `decision_policy.cpp`,
`mechanistic_evidence.cpp`, `conformal_selector.cpp`, `event_explanation.cpp`
and `null_control.cpp` against golden vectors. The second ported everything
upstream of them -- the BAM scan, clustering, fragment extraction, TE
classification, consensus, segmentation, the joint decision and the whole
finalization stage -- so `placer` now runs end to end from reads to
`scientific.txt` without the compiled binary.

Two things are deliberately NOT ported, and both are documented where they
would be used rather than silently stubbed (the second now has a replacement
of its own; see [Speed](#speed)):

  * **abPOA.** `placer/core/consensus.py` takes the consensus function as an
    argument. `single_sequence_consensus` handles the cases needing no
    alignment and RAISES otherwise; `pyabpoa_consensus` uses the same library
    the C++ links. A worse consensus would change the insert sequence, the TE
    identity, the poly(A) call and the structure decode without changing any QC
    field -- the run would look clean and every call would be subtly wrong.
  * **The C++ parallel executor.** `placer` has its own instead:
    `--threads N` cuts the scan at bin boundaries, runs the pieces on N
    processes and rejoins them in genome order before finalization, with
    byte-identical output (`placer/parallel.py`, `tests/test_38_parallel.py`).

Every other C++ translation unit has a Python counterpart, and each module's
docstring names the file it was ported from — so the map can be regenerated
from the source rather than maintained by hand:

| C++ | Python |
|---|---|
| `gate1_module.cpp` | `reads.py` |
| `bam_io.cpp`, `indexed_bam_reader.cpp` | `io/bam.py`, `io/reference.py`, `alignment.py` |
| `pipeline_window_helpers.inc` | `core/windows.py` |
| `dbscan_component_module.cpp` | `core/clustering.py` |
| `local_interval_cache.cpp` | `core/interval_cache.py` |
| `insert_fragment_module.cpp` | `core/fragments.py` |
| `te_quick_classifier.cpp` | `core/te_classifier.py`, `core/seqtools.py`, `io/blast.py` |
| `te_sequence_explainer.cpp` | `core/structure.py` |
| `tsd_detector.cpp` | `core/tsd.py` |
| `pipeline_breakpoint_{helpers,stage}.inc` | `core/breakpoints.py` |
| `pipeline_event_evidence_stage.inc` | `core/events.py` |
| `pipeline_consensus_stage.inc`, `pipeline_event_helpers.inc` | `core/consensus.py`, `io/poa.py` |
| `pipeline_segmentation_stage.inc` | `core/segmentation.py` |
| `pipeline_hypothesis_emission_stage.inc` | `core/hypotheses.py` |
| `decision_policy.cpp` | `core/policy.py`, `core/genotype.py` |
| `mechanistic_evidence.cpp` | `core/blocks.py` |
| `event_explanation.cpp` | `core/explanation.py` |
| `conformal_selector.cpp` | (ported, then deleted with the legacy decision) |
| `null_control.cpp` | `core/null_control.py` |
| `pipeline_call_selection.inc` | (ported, then deleted with the legacy decision) |
| `pipeline_finalization_stage.inc` | `core/finalize.py` (rewritten: `core/mechanism_selection.py`) |
| `pipeline_{entrypoints,bin_processing_stage}.inc` | `pipeline.py`, `core/scan.py`, `core/bins.py` |
| `main.cpp` | `main.py`, `wiring.py`, `report/tsv.py` |

## Why the tests had to come first

The C++ suite is 16,268 lines across 39 files, and it is almost entirely
**directional**. It asserts things like

```cpp
assert(conflicted_cert.lower_log_bf_te_vs_artifact <
       clean_cert.lower_log_bf_te_vs_artifact);
assert(cert.structure_lower_log_lr > 0.0);
```

Signs and orderings. Very little of it pins a value.

That is a reasonable regression net for C++ refactors and a **weak
specification for a port**: a Python implementation could compute a completely
different function and still satisfy every one of those assertions. So the first
step was not to write Python, it was to freeze the actual numbers from the
real C++ entry points and assert equality against them to full double
precision. That oracle has since been removed; see
[`docs/departures-from-cpp.md`](docs/departures-from-cpp.md).

## Layout

```
placer/
  THREE STAGES, and the rule between them: core/ may import neither io/ nor
  report/, at module scope or inside a function body. That is what keeps the
  decision layer runnable with no pysam, no BLAST and no abPOA installed, and
  tests/test_37_layering.py enforces it.

  --- tier 0: the vocabulary all three stages speak ----------------------
  alignment.py      AlignedRead (the ReadView surface), CIGAR/SA parsing
  reads.py          gate1's predicate: is this read worth carrying?
  config.py         PipelineConfig, every default from the C++
  schema.py         the ledger contract (a READER contract; see the file)

  --- io/: everything that talks to something outside the process --------
  io/bam.py         the pysam streaming reader and the indexed fetch
  io/reference.py   the FASTA fetcher
  io/pysam_adapter.py  a pysam record onto AlignedRead
  io/gate.py        gate1 applied to a stream, with its tallies
  io/blast.py       the makeblastdb / blastn driver
  io/te_library.py  loading the library, and classifying a batch of inserts
  io/poa.py         abPOA through pyabpoa
  io/report_context.py  the contig list, sample name and VCF anchor bases

  --- core/: everything that decides something ---------------------------
  core/contracts.py ReadSource, StageHooks -- what core needs from outside
  core/result.py    PipelineResult
  core/ledger.py    EvidenceLedgerRow and FinalCall
  core/scan.py      the bin loop
  core/bins.py      one bin, reads to calls and ledger rows
  core/finalize.py  the whole-run stage
  (the scan)      seqtools windows clustering interval_cache fragments
  (naming)        te_classifier breakpoints events consensus segmentation
                  hypotheses
  (deciding)      policy blocks structure explanation genotype tsd
  (selecting)     tprt mechanism locus_evidence null_control selection
                  mechanism_selection

  --- report/: everything that renders; every function returns a string --
  report/tsv.py     scientific.txt, evidence_ledger.tsv
  report/vcf.py     calls.vcf -- VCF 4.2, explicit inserted sequence as ALT
  report/csv_table.py  calls.csv -- the full flat table, both call sets
  report/context.py the facts a VCF needs that no stage computes
  report/writer.py  the only place in the output stage that opens a file

  --- composition: the only modules allowed to import more than one stage -
  pipeline.py       gate -> scan -> finalize
  wiring.py         binds the four hooks to real I/O
  parallel.py       the scan on N processes (`--threads`), same bytes out
  main.py           the CLI

tests/
  test_00..18           the decision layer
  test_19_seqtools.py   composition model, hand-computable pins
  test_20_clustering.py the geometry stage
  test_21_alignment.py  the read view and CIGAR/SA parsing
  test_22_fragments.py  which bases get cut out of a read
  test_23_te_classifier.py  the two classifiers
  test_24_policy.py     ranking vs emission
  test_25_windows.py    the evidence density
  test_26_breakpoints.py  the priority ladder
  test_27_segmentation.py the tripartite decode
  test_28_events.py     counts, event strings, clip concordance
  test_30_interval_cache.py  fetching each stretch of reads once
  test_31_outputs.py    triage, posterior, output contracts, the CLI
  test_32_pipeline.py   THE end-to-end acceptance test, and de novo
  test_38_parallel.py   --threads N writes the same bytes as --threads 1
  test_44_mechanism_selection.py  the decision, and finalization's use of it
tools/
  run_tests_without_pytest.py    zero-dependency runner
```

## Installing

```bash
pip install -e .            # the decision layer: no dependencies at all
pip install -e '.[scan]'    # + pysam, pyabpoa and rapidfuzz, to run from a BAM
pip install -e '.[dev]'     # + pytest, ruff, mypy, pre-commit
```

BLAST+ (`blastn`, `makeblastdb`) is also needed for the TE alignment and is not
a Python package — put it on `PATH`, or set `te_blastn_path` in the config.

Nothing is published yet: `0.1.0` is the version this package was extracted at,
not a release. [`CHANGELOG.md`](CHANGELOG.md) is what has changed since, and
[`docs/departures-from-cpp.md`](docs/departures-from-cpp.md) is what those
changes cost numerically.

## Running the whole pipeline

```bash
placer sample.bam reference.fa te_library.fa --output-dir out/ --threads 8
```

`--threads N` (`-t`) scans on N processes. It changes how long the run takes
and nothing else: the four files are the same bytes for any N.

or, without installing, `python3 -m placer.main ...` from the repository
root.

The decision layer needs none of the scan dependencies:
`placer.core.finalize` and everything it imports run on a ledger alone,
which is why they are optional rather than required.

## Speed

The run:
- HG002 ONT (TEBench, full coverage), GRCh38 chr1, `--threads 16`;
- 16 CPUs of an AMD EPYC 75F3 node, with the inputs on BeeGFS;
- from a frozen snapshot.

Both rows write the same four files, byte for byte.

| | CPU (user + sys) | of which sys | wall | peak memory, one worker |
|---|---|---|---|---|
| 1.0.0a1 | 10.28 CPU-h | 3.8 h | 2 h 55 min | 12.5 GB |
| 1.0.0a2 | 2.05 CPU-h | 0.28 h | 13 min | 12.6 GB |

What changed, each change exact (`CHANGELOG.md`, 1.0.0a2):

| was | now |
|---|---|
| every `blastn` exec paging in 86 conda libraries from the network filesystem, 0.6-0.9 s of system time per call on a busy node | blastn staged once per node on local disk, used only when it resolves and searches exactly as the original |
| every bin re-reading its ultra-long reads from the BAM index (~50x the chunk) | local reads answered from the chunk's own stream when it provably holds them all |
| ~10^4 reference fetches per evaluated locus (TSD detector, 100 decoys) | 64 kb cached blocks, and two windows per detection |
| the soft-clip complexity test walking clips of tens of kb four times per fragment (27% of chr1) | base counts, cheapest test first, and a 4^k bound that decides long clips without a k-mer set |
| the flank search computing an edit distance per (length, offset) | one diagonal chain at a time, a few distances per chain |
| abPOA re-assembling the same reads for neighbouring hypotheses (62% of calls) | answers reused for the same input |
| three chunks (the pericentromere, 1q21) running for most of the wall time on one worker | the heaviest chunks start first and are cut into pieces |

**`blastn` batches are fixed, on purpose.** Packing inserts differently changes
the HSPs blastn reports for repetitive ones: a 112 bp (AT)n insert's
`cross_family_margin` moved from 0.0610 to 0.0662. So the batches are a
function of the bin alone, 32 sorted inserts each.

**What is left**, on HG002 chr1:
- **abPOA**, about a quarter of the interpreter's time, and **blastn**, about
  a quarter of the CPU. Both are compiled code on inputs the decision fixes.
- **Collapsed pericentromeric and segmental-duplication regions.** Their loci
  are removed from the calls by the collapse rule. Skipping their expensive
  stages leaves every chr1 decision unchanged in replay, and would save about
  a quarter of the CPU. It changes the ledger, so it is a decision change,
  judged as one.

To see where a run's time goes, set `PLACER_PERF_LOG=perf.tsv` and run
`tools/perf/perf_summary.py perf.tsv`.

## The output files

A run writes four files into `--output-dir`, always, even when some are empty.
A missing file is ambiguous between "nothing qualified" and "the run died",
and a downstream script cannot tell the difference.

| file | what it is |
|---|---|
| `calls.vcf` | VCF 4.2. The TE calls, coordinate-sorted |
| `calls.csv` | the full flat table: every column of `scientific.txt`, plus four fields no other file carries |
| `scientific.txt` | the TE calls, with the run's calibration constants in a header block |
| `evidence_ledger.tsv` | every candidate examined, whatever the verdict. This is the sample's own null set as well as its candidate set |

**PLACER is a TE caller.** An insertion the decision selects whose insert is
not a TE -- under TEBench's rule, TE hits covering at least half of it and
100 bp -- is not reported. The ledger marks it (`mech_structural_selected`)
and `scientific.txt` counts it (`mech_shadow_structural_selected`). Before
2026-09-27 these went to `structural_calls.tsv` and to the VCF as
`FILTER=STRUCTURAL`.

**The VCF writes the inserted sequence as the ALT allele**, not a symbolic
`<INS:ME:ALU>`. The sequence is the evidence, and a symbolic allele sends every
consumer back to a second file to see it. The cost is that the file is roughly
the size of every insert sequence combined — tens of megabytes on a
whole-genome run, where the TSVs default `insert_seq` off for that reason.
`bgzip` handles it. A call whose sequence could not be assembled keeps its
record with `ALT=<INS>` and `FILTER=ALTSEQ_MISSING`, rather than being dropped,
so the VCF and `scientific.txt` never disagree about how many calls there were.

**`MEINFO` goes only where all four of its fields are known.** Its fourth
field is a polarity, it is not optional, and the spec has no value for
unknown, so a call with no oriented TE alignment -- or whose family abstained --
gets no `MEINFO` rather than an invented `+`. The polarity is the strand of the
best TE alignment relative to the reference, and START/END are 1-based on the
element consensus. `MEI`, `MEISTART` and `MEIEND` go out on every record, and
`TE_CLASS` / `TE_SUPERFAMILY` on every committed one.

`QUAL` is the phred transform of the local FDR, and `.` rather than `0.00`
when none was computed — a zero QUAL asserts "certainly wrong", which is a
different claim from "not assessed".

## Running the tests

```bash
pytest -q
```

No package index? There is a dependency-free runner, and it is the one CI
should use -- the suite deliberately has no third-party dependency at all, so a
green run proves that:

```bash
python3 tools/run_tests_without_pytest.py
python3 tools/run_tests_without_pytest.py test_04    # one module
```

```
total: 770 passed, 0 failed, 0 skipped, 2 xfail (known issues)
```

| module | status |
|---|---|
| `schema.py` | the ledger contract |
| `pipeline.py` | end to end, reads to calls (`test_32`) |
| `genotype.py` | GQ as posterior Phred, count invariants |
| `structure.py` | path confidence, poly(A) state |
| `blocks.py` | 8 certificates, aggregate algebra, robust lfdr |
| `selection.py` | e-BH, FDR simulation |
| `tprt.py` | the coincidence model, 10 behaviour cases |
| `mechanism_selection.py` | the decision: decoy check, loci, e-BH, TE rule, placement |

## How to read the suite

- **passed** — the seam contract, an invariant, a regression, or a behaviour
  case.
- **xfail (2)** — known problems, pinned so that fixing one is a deliberate
  modelling change: the 3' transduction net penalty, and the serialized blocks
  not summing to the aggregate.
- **no skips** — the migration surface is empty.

## The kinds of test

1. **Invariant** (`@pytest.mark.invariant`) — mathematical properties any
   correct implementation must satisfy, in any language. The most valuable is
   `test_controls_fdr_under_the_null_by_simulation`, which constrains the e-BH
   *procedure* rather than its arithmetic and so catches an off-by-one in the
   step-up rule that a handful of fixed cases would miss.
2. **Regression** (`@pytest.mark.regression`) — bugs found and fixed during this
   work, pinned so a port cannot reintroduce them:
   - GQ implemented as a likelihood difference instead of the posterior error in
     Phred;
   - a minimum-depth gate that duplicated what GQ already does.

   (Three more -- the dependency cap applied after the penalty, the
   calibration sample selected by the aggregate being calibrated, and `max()`
   instead of the mean when combining e-values -- went with the legacy
   decision they pinned, on 2026-09-27.)
3. **Contract** (`@pytest.mark.contract`) — the ledger schema, i.e. the seam.

## What joining the layers found

> History. This section and the two after it record how the decision layer
> was redesigned. `integrate.py`, `decoys.py`, `dependency.py`, `conformal.py`
> and `tests/test_13_end_to_end.py`, which they cite, were deleted with the
> legacy decision on 2026-09-27. The decision they argued for is
> `core/mechanism_selection.py`.

`integrate.py` puts the existing mechanistic score under e-BH, which gives the
Python side FDR control for the first time. Running it produced a result worth
more than the plumbing.

**The architectural claim holds.** An e-value only has to be a fixed
non-negative function divided by its own measured null expectation, so a
hand-tuned score can feed a procedure with a theorem attached. Tested by
inverting the linkage hallmarks — rewarding the ABSENCE of an endonuclease motif
and of a TSD — and FDR control survives. The hand-set constants can only cost
recall, and that is now a test rather than an argument.

**But recall is currently zero, and not because of a mistuned threshold.** The
blocks are each clamped to a small range — endonuclease `clamp(en,0,6)*0.45`
≤ 2.7, `tsd_loglr` clamped to [−1,3], `polya_loglr` bounded to about [−0.3,1.3],
TE body ~2.9 — so the total caps near **9.9 nats**. e-BH at m=1060 and q=0.10
needs `log(m/q)` = **9.27 nats just to clear the rank-1 threshold**, before any
penalty. The score's dynamic range is smaller than genome-scale multiple testing
requires.

**And the penalty is not the bound's fault.** On a pure-null sample σ = 22.24
against an empirical mean of 21.41: the Bernstein slack costs 0.04 nats, and the
other 3.06 are `log(mean)` — the honest statement that this score assigns e^2.97
≈ 20 to a typical null locus. Tightening the concentration inequality buys 0.04
nats; making the blocks real log-LRs against measured local nulls buys 3.

The single largest loss is `tsd_loglr`'s clamp at 3.0. A 15 bp exact duplication
in unique sequence is worth ~13–14 nats against a locally measured background,
so that clamp alone discards about ten. The TPRT model reaches 33 nats on the
same case for exactly that reason — which turns "it is more elegant" into "it
has the range".

**One cost I chose without checking its size.** Calibrating σ on every row is
the safe direction — contamination inflates the mean, so the bound still bounds
the null mean — but measured at m=1060 the penalty goes 3.13 → 4.83 → 5.51 →
6.22 → 6.67 nats as the true fraction goes 0 → 0.2% → 1% → 3% → 5.7%. The true
positives are setting the bar they then have to clear. On real WGS the true rate
among ledger rows is far below that, so it is milder in practice, but it is why
a robust mean estimator (median-of-means, or a Catoni M-estimator) is worth more
here than it looks: same validity, far less leverage for any single row.

## End to end

```
recall 0.93-0.98 per run, mean 0.963      pooled FDP 0.0000 at q = 0.10
E_null[e^score] = 0.0016-0.0023           validity check PASS
sigma = 1, penalty = 0                     by construction, verified
```

`tests/test_13_end_to_end.py` is the acceptance test. It also pins the failure
modes: a misspecified score is refused rather than reported, an inconclusive
null set does not block, and a run with no null set is told it is unverified
rather than given false assurance.

## The blocker, and how it was resolved

Wiring the TPRT terms in and running them head to head against the clamped score
found something that invalidates the comparison and matters more than it:

**calibrating sigma on every row is structurally incompatible with e-BH
selecting anything, for any score.**

Any score worth having puts a true insertion above `log(m/q)`, so every true
positive SATURATES the cap `C = m/q`. Calibrating sigma on all rows then gives
`sigma ~ pi * C`, where `pi` is the true fraction, because the saturating rows
dominate the mean. The largest attainable e-value is therefore `C/sigma = 1/pi`,
while e-BH at rank `r` demands `C/r`:

```
1/pi >= C/r   =>   r >= pi * C = pi * m / q
```

but `r` cannot exceed the number of true rows, about `pi * m`. So selection
needs `pi*m >= pi*m/q`, i.e. **q >= 1**. Impossible for any usable q.

Measured at m=1060, pi=0.0566, q=0.10: sigma 751, penalty 6.62 nats, headroom
2.87, and e-BH needs 5.17 at r=60. Recall zero, for the clamped score and the
TPRT score alike. And the better the score, the worse it gets, because a better
score saturates the cap more thoroughly.

Removing the cap makes it worse, not better: uncapped, each true row contributes
e^21, the mean becomes 7.4e7 and the penalty 18.1 nats against a raw score of
21. **The cap is not the cause. The contamination is, and the contamination is
precisely the signal.**

So both of the obvious sample choices are wrong, and I have now made both
mistakes:

- **Exclude rows the score likes** (what the C++ did) — the right tail goes,
  sigma collapses onto its floor of 1, the correction is inert. Anti-conservative.
- **Include everything** (what this port does) — the true positives set the bar
  they then have to clear. Valid, and useless.

The resolution has to be an estimator that is simultaneously valid, not selected
by the score, and robust to a small fraction of large values. Trimming shows the
shape: dropping the top 5% takes the penalty from 6.30 to 2.72 nats and the
headroom from 2.97 to 6.55, clearing the 5.17 e-BH needs — while dropping 2%,
below `pi`, does not. The trim fraction has to be near `pi`, and `pi` is what
e-BH is estimating, so there is a natural fixed point: sigma → selection → pi →
sigma. A median-of-means or Catoni M-estimator is the principled version.

### The resolution was a deletion

Trimming turned out to be the wrong fix too, and chasing it would have added a
parameter. The right answer is that **a genuine likelihood ratio needs no
calibration at all**:

```
E_null[ p1(X)/p0(X) ]  =  integral (p1/p0) p0  =  1
```

by construction. The entire sigma apparatus exists only because the affine
blocks are *not* likelihood ratios — they are bounded signals through invented
affine maps, whose null expectation is an unknown number that has to be
measured, and which cannot be measured. That is a complete argument against
using them in a decision path at genome scale, independent of whether their
constants are well chosen.

Measured on the TPRT terms over 20,000 simulated nulls: `E_null[e^score] =
0.186`, with only 2.7% of nulls scoring above zero at all. So `sigma = 1`, no
penalty, and e-BH works.

What remains for a null set is **verification, not estimation** — a likelihood
approach's real risk is misspecification, so the decision checks
`E_null[e^score] <= 1` on shifted-breakpoint decoys, per class, and divides a
class's e-values by the bound where it fails
(`core/mechanism_selection.decoy_checks`). That is a far weaker
requirement than calibration: an approximate null set can still falsify the
inequality, whereas estimating sigma from one would need a faithful draw.

Two null constructions, and the difference matters:

- **Permuted** — each insert paired with a different locus's flanking context.
  Inherits the host's linkage wholesale, so it is contaminated at the same rate
  as the candidate set (measured: means of 101–173, blocking every run in my
  first version). A PASS is conclusive; a FAILURE is not, because contamination
  can only push the mean up.
- **Shifted** — the same locus with the breakpoint moved, so local composition
  is preserved and the coincidences are whatever chance gives. The standard
  shifted control from peak calling, and the one that works. It cannot be built
  from the scalar ledger, because `tsd` is already the *result* of a coincidence
  test rather than the sequence it was computed from, so the evidence layer has
  to produce it.

## What the TPRT terms do establish

The dynamic-range claim holds on the scores themselves, independent of the
blocker above:

- The clamped ceiling measures at **under 11 nats**, against a rank-1 threshold
  of `log(m/q)` = 9.27 — essentially no headroom.
- A modal true insertion scores **more than 14 nats above** that threshold under
  the TPRT terms.
- Attributable to one line: `tsd_loglr` clamps at 3.0, while the same 15 bp
  duplication against the same locally measured background is worth **over 10
  nats** unclamped. That single clamp discards more than the whole score's
  headroom.

And one new limitation surfaced while checking it. `log_bf_sequence`'s crossover
is at identity **0.9433** — where the two Bernoulli models are equally likely,
not at the arithmetic midpoint 0.93 I expected. Per 1 kb: 0.88 → −120 nats,
0.92 → −44, 0.96 → +32, 0.98 → +70. But `q_ambient` is a single GLOBAL constant,
so that crossover is identical for every family, and the ambient divergence of
L1HS and of an old L1PA lineage are very different numbers. It should be measured
per subfamily from the reference's own copies of that family — the same "read the
null off the genome" rule the other three nulls already follow.

## The seam

`evidence_ledger.tsv` was the interface, which is why the migration could start
in the middle instead of at an edge: the compiled scanner kept producing the
ledger, the Python layer consumed it, and **both could run on the same input so
their call sets could be diffed**. That made this a rewrite with a ground truth —
the safest kind.

The Python side now produces the ledger too, so the diff runs in both
directions and `placer/report/tsv.py` pins the column order both halves have to
agree on. The synthetic end-to-end test in `test_32_pipeline.py` shows the
stages compose, not that they agree with the C++ on real data.

`placer.schema.MISSING_FOR_TPRT` lists eight observables the current ledger
does *not* carry and the TPRT model needs. The important one is the pair of
**element coordinates**: the ledger keeps only `best_te_query_coverage`, a
ratio, which discards about 8.5 nats of 3'-anchoring evidence for a 1 kb
fragment of a 6 kb L1. Two integers instead of one ratio is the whole cost of
recovering it, and the scanner has to emit them before the model can be
computed at all.

## The two pinned problems in the C++ decode

**A real 3' transduction is net penalised**, despite having a dedicated HSMM
state. The residual counters are computed *before* the transduction decode and
never reduced by it, so the Viterbi path is reporting-only. For a 200 bp
transduction in a 1 kb insert: coverage falls 0.2 and costs `2.45 × 0.2`, the
high-complexity residual rises and costs `1.25 × 0.2`, and the transduction
posterior returns only `+0.35`. Net **−0.39 nats** against a hallmark of genuine
L1 activity.

**The serialized blocks do not sum to the aggregate.** Per-block vs-non-TE
log-LRs use weights `0.45 / 0.65 / 1 / 0.80 / 0.50`; the aggregate uses
`0.40 / 0.70 / 1 / 0.85 / 0.35`, and the structure block differs too. Harmless
for the decision — only the aggregate decides — but it defeats anyone trying to
re-derive a call by hand from the ledger.

## Where the work is now

The porting order this section used to give — genotype, dependency,
selection, then blocks and structure, then tprt — is done. The migration is
complete, and leaving a to-do list that reads as if it were not was the most
misleading paragraph in this file.

What replaces it is not a list of modules to port but a set of open
questions, each of which has evidence attached rather than an opinion:

- **`docs/departures-from-cpp.md`** — a historical record of how this
  implementation departed from the C++ while the C++ was the reference. Every
  deliberate departure is recorded there with what moved, by how much, and how it was measured. One entry is
  marked **Open**: fixing `blocks._structure_explanation` to fall back to the
  shadow path (as the C++ does) raises the
  TE evidence of loci with no TE alignment, and the precision effect of that
  has not yet been measured on real data.
- **`tools/make_giab_eval.py`** — cuts an HG002 ONT-UL evaluation slice and a
  TE truth set from GIAB Tier1, with development and holdout regions labelled
  in the manifest. A change that needs the holdout to justify it is a fit,
  not a fix.
- **The four once-unimported modules** — `tprt.py` and `null_control.py` are
  on the calling path now, and `integrate.py` and `decoys.py` were deleted
  once the decision they prototyped replaced the legacy one:
  `docs/off-pipeline-modules.md` records which went where.
