# placer-py

A Python implementation of [PLACER](https://github.com/lifelix1209/PLACER),
the long-read transposable-element insertion caller. It runs end to end — BAM
in, calls and an evidence ledger out — without the compiled binary.

This repository began as a port of the C++ and is now the reference
implementation itself: the C++ golden-vector oracle has been removed, and how
this implementation departed from the C++ up to that point is recorded in
[`docs/departures-from-cpp.md`](docs/departures-from-cpp.md). Every module's
docstring still names the C++ file it came from.

**This is mid-refactor toward PLACER 1.0**, a multi-species caller: the
TE taxonomy, the structure grammar and the decision layer are being made
class-aware (LINE/SINE/LTR/DNA/Helitron) rather than human-specific. The
sections below describe the implementation as it stands, and this README will be
rewritten for users before the release.

**The migration is complete, and it now covers the whole pipeline rather than
the decision layer alone.** The first pass ported `decision_policy.cpp`,
`mechanistic_evidence.cpp`, `conformal_selector.cpp`, `event_explanation.cpp`
and `null_control.cpp` against golden vectors. The second ported everything
upstream of them -- the BAM scan, clustering, fragment extraction, TE
classification, consensus, segmentation, the joint decision and the whole
finalization stage -- so `placer_py` now runs end to end from reads to
`scientific.txt` without the compiled binary.

Two things are deliberately NOT ported, and both are documented where they
would be used rather than silently stubbed (the second now has a replacement
of its own; see [Speed](#speed)):

  * **abPOA.** `placer_py/core/consensus.py` takes the consensus function as an
    argument. `single_sequence_consensus` handles the cases needing no
    alignment and RAISES otherwise; `pyabpoa_consensus` uses the same library
    the C++ links. A worse consensus would change the insert sequence, the TE
    identity, the poly(A) call and the structure decode without changing any QC
    field -- the run would look clean and every call would be subtly wrong.
  * **The C++ parallel executor.** `placer_py` has its own instead:
    `--threads N` cuts the scan at bin boundaries, runs the pieces on N
    processes and rejoins them in genome order before finalization, with
    byte-identical output (`placer_py/parallel.py`, `tests/test_38_parallel.py`).

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
| `mechanistic_evidence.cpp` | `core/blocks.py`, `core/dependency.py` |
| `event_explanation.cpp` | `core/explanation.py` |
| `conformal_selector.cpp` | `core/conformal.py` |
| `null_control.cpp` | `core/null_control.py` |
| `pipeline_call_selection.inc` | `core/call_selection.py` |
| `pipeline_finalization_stage.inc` | `core/finalization.py` |
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
placer_py/
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
  (selecting)     dependency selection conformal decoys null_control
                  call_selection finalization integrate tprt

  --- report/: everything that renders; every function returns a string --
  report/tsv.py     scientific.txt, structural_calls.tsv, evidence_ledger.tsv
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
  test_29_finalization.py the whole-run stage
  test_30_call_selection.py  one call per component, the interval cache
  test_31_outputs.py    triage, posterior, output contracts, the CLI
  test_32_pipeline.py   THE end-to-end acceptance test, and de novo
  test_38_parallel.py   --threads N writes the same bytes as --threads 1
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
placer-py sample.bam reference.fa te_library.fa --output-dir out/ --threads 8
```

`--threads N` (`-t`) scans on N processes. It changes how long the run takes
and nothing else: the five files are the same bytes for any N.

or, without installing, `python3 -m placer_py.main ...` from the repository
root.

The decision layer needs none of the scan dependencies:
`placer_py.core.finalization` and everything it imports run on a ledger alone,
which is why they are optional rather than required.

## Speed

Measured on HG002 ONT-UL (GIAB, GRCh37), `21:18,704,270-19,599,876` -- 0.9 Mb,
2,065 reads, 203 evaluated candidates -- on an 11-core Apple M3 Pro laptop.
Every row writes the same five files, byte for byte, as the first.

| | wall | CPU | peak memory |
|---|---|---|---|
| before any of this | 1450-2425 s | 977-986 s | 469 MB |
| one process | 80 s | 73 s | 711 MB |
| `--threads 8` | 21 s | 109 s | 503 MB per process |

The "before" wall time is a range because it was measured twice and most of
it was spent launching `blastn` one process after another, which on this
machine varies from run to run; CPU is the stable comparison (~13x on one
process). Memory went UP on one process: each read now carries its CIGAR
index while it is in use, and up to 32 Mbp of recently fetched reads are kept
for reuse (`io/bam.RecordCache`). Both are bounded -- by the bin and by the
cache -- so neither grows with the genome.

Where the single-process time went, and what was done about it -- every change
exact, each checked by comparing the five output files:

| was | fix |
|---|---|
| 63% in the flank-placement edit distance | `rapidfuzz` when installed (optional; the pure-Python DP is the fallback and gives the same answer) |
| every window re-walking an ultra-long read's whole CIGAR (p90 ~5,000 operations) | one walk per read into a cached index; windows are binary searches |
| one `blastn` per candidate, run one after another | the same one-insert-per-process calls, run concurrently per bin, and remembered per sequence so a repeated insert is not re-aligned |
| each read decoded from pysam ~5 times, once per fetch that returned it | a bounded cache hands back the same read object |
| the TE library hashed and its k-mer tables rebuilt per call / per process | once per run; k-mers folded per distinct key |

**`blastn` is not batched, on purpose.** Packing several inserts into one
`blastn` run changes the HSPs it reports for repetitive ones (measured: a
112 bp (AT)n insert's `cross_family_margin` moved from 0.0610 to 0.0662), so
each insert still gets its own process and the processes run side by side.

**What is left is mostly `blastn` starting up.** Each launch costs ~0.8 s of
CPU and ~1.3 s of wall time on the laptop above, whatever the query -- that is
BLAST+ 2.17 initialising, and `blastn -version` alone pays it. It is about half
of the single-process time. It cannot be cut without sharing a process between
inserts, which is the change measured above to alter results, so the next
lever there is a decision rather than an optimisation. The largest pure-Python
cost left is the flank search in `core/segmentation.py` (~25%).

## The output files

A run writes five files into `--output-dir`, always, even when some are empty.
A missing file is ambiguous between "nothing qualified" and "the run died",
and a downstream script cannot tell the difference.

| file | what it is |
|---|---|
| `calls.vcf` | VCF 4.2. Both call sets, coordinate-sorted, structural ones marked `FILTER=STRUCTURAL` |
| `calls.csv` | the full flat table: every column of `scientific.txt`, plus the call set and four fields no other file carries |
| `scientific.txt` | the TE calls, with the run's calibration constants in a header block |
| `structural_calls.tsv` | the structural insertions the TE-calibrated mode set aside — selected calls, not rejects |
| `evidence_ledger.tsv` | every candidate examined, whatever the verdict. This is the sample's own null set as well as its candidate set |

**The VCF writes the inserted sequence as the ALT allele**, not a symbolic
`<INS:ME:ALU>`. The sequence is the evidence, and a symbolic allele sends every
consumer back to a second file to see it. The cost is that the file is roughly
the size of every insert sequence combined — tens of megabytes on a
whole-genome run, where the TSVs default `insert_seq` off for that reason.
`bgzip` handles it. A call whose sequence could not be assembled keeps its
record with `ALT=<INS>` and `FILTER=ALTSEQ_MISSING`, rather than being dropped,
so the VCF and `scientific.txt` never disagree about how many calls there were.

**`MEINFO` is declared in the header and emitted on no record.** Its fourth
field is a polarity, it is not optional, and the spec has no value for
unknown — but this build never resolves insertion orientation, so `+` would
invent a measurement and `.` would be read as a real one. The three components
that *are* known go out as `MEI`, `MEISTART` and `MEIEND`. A test ties that
decision to `schema.MISSING_FOR_TPRT`, so whoever wires orientation in gets a
failing test telling them to turn `MEINFO` back on in the same commit.

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
total: 736 passed, 0 failed, 2 skipped, 2 xfail (known issues)
```

| module | status |
|---|---|
| `schema.py` | the ledger contract |
| `pipeline.py` | end to end, reads to calls (`test_32`) |
| `genotype.py` | GQ as posterior Phred, count invariants |
| `structure.py` | path confidence, poly(A) state |
| `blocks.py` | 8 certificates, aggregate algebra, robust lfdr |
| `dependency.py` | bound invariants + 3 regressions |
| `selection.py` | e-BH, mean-not-max, dominance, BY, FDR simulation |
| `decoys.py` | the validity check that replaced calibration |
| `integrate.py` | both selection paths, joined to the mechanistic layer |
| `tprt.py` | the coincidence model, 10 behaviour cases |

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
   - the dependency cap applied *after* the penalty instead of before (a
     validity bug: `sigma` bounds `E[min(Y,C)]`, so only `min(Y,C)/sigma` is an
     e-value);
   - the calibration sample selected by the very aggregate being calibrated,
     which truncated `sigma`'s right tail and collapsed it onto its floor of 1;
   - `max()` instead of the mean when combining e-value constructions (the
     maximum of e-values is not an e-value);
   - GQ implemented as a likelihood difference instead of the posterior error in
     Phred;
   - a minimum-depth gate that duplicated what GQ already does.
3. **Contract** (`@pytest.mark.contract`) — the ledger schema, i.e. the seam.

## What joining the layers found

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
approach's real risk is misspecification, so `placer_py/core/decoys.py` checks
`E_null[e^score] <= 1` and refuses to proceed if it fails. That is a far weaker
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
directions and `placer_py/report/tsv.py` pins the column order both halves have to
agree on. The synthetic end-to-end test in `test_32_pipeline.py` shows the
stages compose, not that they agree with the C++ on real data.

`placer_py.schema.MISSING_FOR_TPRT` lists eight observables the current ledger
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
- **Four unimported modules** — `tprt.py`, `integrate.py`, `decoys.py` and
  `null_control.py` are reachable only from tests, and every one of them is
  deliberate: see `docs/off-pipeline-modules.md` for which is which.
  `selection.py` was the fifth and was the only genuine duplicate; it is now
  imported by `finalization.py` and the inline copy is gone.
