# placer-py

A Python implementation of [PLACER](https://github.com/lifelix1209/PLACER),
the long-read transposable-element insertion caller. It runs end to end — BAM
in, calls and an evidence ledger out — without the compiled binary.

**This repository is a port with a ground truth, and that is the whole point.**
The C++ is the oracle: the selection layer is pinned to golden vectors frozen
out of it (`tests/oracle/cpp_reference.json`), every module's docstring names
the C++ file it came from, and `tools/regenerate_oracle.sh` regenerates the
contract from a C++ checkout. Keep the C++ authoritative until the two agree
locus by locus on a real BAM.

```
placer-py/            this repository
PLACER/               the C++, expected as a sibling checkout by
                      tools/regenerate_oracle.sh (or set PLACER_SRC)
```

Tests first, **and the two layers are constrained differently on purpose**.

`placer_py/redesign/` holds a 2,602-line Python implementation that predates
the port and had no tests. That code is **not a port of the C++** — it is a
parallel redesign: it
takes candidates from a Sniffles VCF rather than scanning the BAM itself, it
scores the L1 endonuclease target motif (which the C++ does not model at all),
it treats the TSD as a log-LR against a background probability rather than a
`+0.15` bonus, and its TE-body term really is 5'-truncation tolerant.

So the two codebases are strong in different halves, and the suite reflects that:

| layer | binding constraint | why |
|---|---|---|
| **Selection** — `sigma`, the cap, e-BH, the combination rule, conformal/BY | **C++ golden values** | The redesign has no selection layer at all, and the C++ one is statistically correct and was just fixed. Reproduce it exactly. |
| **Mechanistic** — endonuclease motif, TSD, poly(A), TE body, counts | **invariants + behaviour** | The redesign is ahead of the C++ here. Requiring it to reproduce C++ numbers would require it to get worse. |

**The migration is complete, and it now covers the whole pipeline rather than
the decision layer alone.** The first pass ported `decision_policy.cpp`,
`mechanistic_evidence.cpp`, `conformal_selector.cpp`, `event_explanation.cpp`
and `null_control.cpp` against golden vectors. The second ported everything
upstream of them -- the BAM scan, clustering, fragment extraction, TE
classification, consensus, segmentation, the joint decision and the whole
finalization stage -- so `placer_py` now runs end to end from reads to
`scientific.txt` without the compiled binary.

Two things are deliberately NOT ported, and both are documented where they
would be used rather than silently stubbed:

  * **abPOA.** `placer_py/core/consensus.py` takes the consensus function as an
    argument. `single_sequence_consensus` handles the cases needing no
    alignment and RAISES otherwise; `pyabpoa_consensus` uses the same library
    the C++ links. A worse consensus would change the insert sequence, the TE
    identity, the poly(A) call and the structure decode without changing any QC
    field -- the run would look clean and every call would be subtly wrong.
  * **The parallel executor.** It exists to use more cores and asserts its own
    equivalence to the streaming path in the C++ suite. `placer_py` implements
    the streaming path and says so when `PLACER_PARALLEL` is set.

Every other C++ translation unit has a Python counterpart, and each module's
docstring names the file it was ported from — so the map can be regenerated
from the source rather than maintained by hand:

| C++ | Python |
|---|---|
| `gate1_module.cpp` | `reads.py` |
| `bam_io.cpp`, `indexed_bam_reader.cpp` | `bam_io.py`, `alignment.py` |
| `pipeline_window_helpers.inc` | `windows.py` |
| `dbscan_component_module.cpp` | `clustering.py` |
| `local_interval_cache.cpp` | `interval_cache.py` |
| `insert_fragment_module.cpp` | `fragments.py` |
| `te_quick_classifier.cpp` | `te_classifier.py`, `seqtools.py` |
| `te_sequence_explainer.cpp` | `structure.py` |
| `tsd_detector.cpp` | `tsd.py` |
| `pipeline_breakpoint_{helpers,stage}.inc` | `breakpoints.py` |
| `pipeline_event_evidence_stage.inc` | `events.py` |
| `pipeline_consensus_stage.inc`, `pipeline_event_helpers.inc` | `consensus.py` |
| `pipeline_segmentation_stage.inc` | `segmentation.py` |
| `pipeline_hypothesis_emission_stage.inc` | `hypotheses.py` |
| `decision_policy.cpp` | `policy.py`, `genotype.py` |
| `mechanistic_evidence.cpp` | `blocks.py`, `dependency.py` |
| `event_explanation.cpp` | `explanation.py` |
| `conformal_selector.cpp` | `conformal.py` |
| `null_control.cpp` | `null_control.py` |
| `pipeline_call_selection.inc` | `call_selection.py` |
| `pipeline_finalization_stage.inc` | `finalization.py` |
| `pipeline_{entrypoints,bin_processing_stage}.inc` | `pipeline.py` |
| `main.cpp` | `main.py`, `outputs.py` |
| `denovo_cli.cpp`, `parent_pool_scanner.cpp` | `denovo.py` |

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
step was not to write Python, it was to freeze the actual numbers —
`tools/dump_oracle.cpp` walks a grid of inputs through the real C++ entry points
and writes `tests/oracle/cpp_reference.json`, and the Python tests assert
equality against that to full double precision.

## Layout

```
placer_py/
  --- the scan: BAM to candidates ---------------------------------------
  bam_io.py         the pysam reader, the reference fetcher, region scope
  alignment.py      AlignedRead (the ReadView surface), CIGAR/SA parsing
  reads.py          gate1: is this read worth carrying?
  seqtools.py       sequence primitives, the composition model
  windows.py        weighted evidence density, candidate windows
  clustering.py     3-D DBSCAN over insertion signatures, components
  interval_cache.py fetch each stretch of reads once, project per request
  fragments.py      insert fragment extraction (clip / CIGAR / split-SA)
  --- naming and assembling ----------------------------------------------
  te_classifier.py  the k-mer shortlist and the BLAST classification
  breakpoints.py    hypothesis enumeration, the priority ladder
  events.py         reads into alt/ref counts, one read one vote
  consensus.py      event strings, the consensus seam, clip concordance
  segmentation.py   the flank | insert | flank decode
  hypotheses.py     triage, the validator, the breakpoint posterior
  --- deciding -----------------------------------------------------------
  policy.py         the four hypotheses, the latent model, the emission gate
  blocks.py         the six affine evidence blocks and the two aggregates
  structure.py      the TE_CORE -> [TRANSDUCTION] -> [POLYA] decode
  explanation.py    Pareto comparison of competing explanations
  genotype.py       beta-binomial genotyping, overdispersion
  tsd.py            target-site duplication detection
  --- selecting ----------------------------------------------------------
  dependency.py     sigma estimation, the cap, the calibration sample
  selection.py      e-BH, the conformal route, the combination rule
  conformal.py      the dominance route and its BY correction
  decoys.py         verifying that a score is an e-value, not calibrating it
  null_control.py   breakpoint-shift controls, the empirical null tail
  call_selection.py one call per component, re-anchoring
  finalization.py   aggregate, dedup, calibrate, select -- the whole-run stage
  --- the parallel redesign (not a port; see placer_py/redesign/) ---------
  redesign/         cli.py, io.py, models.py, candidates/, evidence/, model/
  --- plumbing -----------------------------------------------------------
  ledger.py         EvidenceLedgerRow and FinalCall
  schema.py         the ledger contract
  outputs.py        the three output files and their column contracts
  config.py         PipelineConfig, every default from the C++
  pipeline.py       the orchestration: BAM in, calls and a ledger out
  main.py           the CLI
  denovo.py         trio de novo calling
  integrate.py      the two selection paths
  tprt.py           the TPRT coincidence model
tests/
  test_00..06, 09..18   the decision layer, against golden vectors
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
  redesign/             the redesign's own unittest suite; needs pysam, and is
                        skipped without it (tests/redesign/conftest.py)
  oracle/cpp_reference.json      frozen from the C++
tools/
  dump_oracle.cpp                generates the golden vectors
  regenerate_oracle.sh           build + run + validate + install
  run_tests_without_pytest.py    zero-dependency runner
```

## Installing

```bash
pip install -e .            # the decision layer: no dependencies at all
pip install -e '.[scan]'    # + pysam and pyabpoa, to run from a BAM
pip install -e '.[dev]'     # + pytest, ruff, mypy, pre-commit
```

BLAST+ (`blastn`, `makeblastdb`) is also needed for the TE alignment and is not
a Python package — put it on `PATH`, or set `te_blastn_path` in the config.

Nothing is published yet: `0.1.0` is the version this package was extracted at,
not a release. [`CHANGELOG.md`](CHANGELOG.md) is what has changed since, and
[`tests/EXPECTED_DIVERGENCE.md`](tests/EXPECTED_DIVERGENCE.md) is what those
changes cost numerically.

## Running the whole pipeline

```bash
placer-py sample.bam reference.fa te_library.fa --output-dir out/
placer-py denovo --child-scientific out/scientific.txt \
    --parent-bam-list parents.txt --ref reference.fa --te te_library.fa
```

or, without installing, `python3 -m placer_py.main ...` from the repository
root.

The decision layer needs none of the scan dependencies:
`placer_py.finalization` and everything it imports run on a ledger alone, which
is why they are optional rather than required.

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
total: 694 passed, 0 failed, 0 skipped, 7 xfail (known issues)
```

| module | status |
|---|---|
| `schema.py` | the ledger contract |
| `pipeline.py` | end to end, reads to calls (`test_32`) |
| `genotype.py` | 14 golden cases, exact |
| `structure.py` | 6 golden cases, exact |
| `blocks.py` | 8 golden certificates, exact |
| `dependency.py` | 6 golden cases + 3 regressions |
| `selection.py` | e-BH, mean-not-max, dominance, BY, FDR simulation |
| `decoys.py` | the validity check that replaced calibration |
| `integrate.py` | both selection paths, joined to the mechanistic layer |
| `tprt.py` | the coincidence model, 10 behaviour cases |

## How to read the suite

- **passed** — either the seam contract, or a relationship the golden data must
  satisfy internally. These constrain the C++ as well as the port.
- **xfail (7)** — known problems, pinned so the port reproduces them faithfully.
  Fixing one is a modelling change and has to flip the xfail deliberately:
  the 3' transduction net penalty; the serialized blocks not summing to the
  aggregate; the endonuclease gate admitting only one mismatch; the flat TSD
  absence penalty; the high-precision tier excluding Alu and truncated L1; the
  clamped score's dynamic range; and the head-to-head comparison that the sigma
  blocker made unrunnable.
- **no skips** — the migration surface is empty.

## The four kinds of test

1. **Golden** (`@pytest.mark.golden`) — equality against the C++ to `rtol=1e-12`.
   That tolerance admits a different order of floating-point operations and
   nothing else; it is far too tight to absorb a different formula.
2. **Invariant** (`@pytest.mark.invariant`) — mathematical properties any
   correct implementation must satisfy, in any language. The most valuable is
   `test_controls_fdr_under_the_null_by_simulation`, which constrains the e-BH
   *procedure* rather than its arithmetic and so catches an off-by-one in the
   step-up rule that a handful of golden cases would miss.
3. **Regression** (`@pytest.mark.regression`) — bugs found and fixed during this
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
4. **Contract** (`@pytest.mark.contract`) — the ledger schema, i.e. the seam.

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
approach's real risk is misspecification, so `placer_py/decoys.py` checks
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
directions and `placer_py/outputs.py` pins the column order both halves have to
agree on. Keep the C++ as the oracle until the Python agrees locus by locus on
a real BAM; the synthetic end-to-end test in `test_32_pipeline.py` shows the
stages compose, not that they agree with the C++ on real data.

`placer_py.schema.MISSING_FOR_TPRT` lists eight observables the current ledger
does *not* carry and the TPRT model needs. The important one is the pair of
**element coordinates**: the ledger keeps only `best_te_query_coverage`, a
ratio, which discards about 8.5 nats of 3'-anchoring evidence for a 1 kb
fragment of a 6 kb L1. Two integers instead of one ratio is the whole cost of
recovering it, and the scanner has to emit them before the model can be
computed at all.

## What the mechanistic tests found

`test_10_mechanistic_invariants.py` runs against the existing `model/` and
confirms one real strength, plus three problems worth knowing about.

**The strength, and it is a genuine one.** The linkage hallmarks separate a
mismapped old reference copy from a new insertion, which is the dominant false
positive and the thing the C++ cannot address at all. An old copy is itself a
TPRT product, so it carries a real poly(A) and a real TE body — those internal
terms are byte-identical between the two cases. The endonuclease motif and the
TSD are properties of *this* locus, and they carry the whole separation:
4 hallmarks / +9.76 nats / `MECH_TPRT_STRONG` against 2 / +3.66 /
`MECH_TPRT_SUPPORTED`. The truncation tolerance is real too: past ~120 bp of
core the TE-body term is flat, so a 1 kb 5'-truncated L1 scores exactly like a
6 kb full-length one.

**The high-precision tier structurally excludes the modal insertion.**
`hp_score = identity × coverage × insert_len` against a threshold of 3500 means
that at identity 0.95 and coverage 0.90 the insert must exceed **4094 bp**. So
regardless of evidence quality it can never fire for an Alu (~300 bp, the most
common new insertion in humans) or for a 5'-truncated L1 below 4 kb — and ~95%
of L1 insertions are 5'-truncated. The tier reaches its quoted ~0.94 precision
by selecting near-full-length L1, which is the easy case. The statistic
conflates "is it TE" with "is it long", and the threshold then acts as a length
filter.

**The endonuclease gate admits at most one mismatch.** `> 2.0` on the PWM means
a two-mismatch target site (1.676) loses the hallmark, while the biology is more
permissive: priming needs as few as 4 matching nt at the primer 3' end and
tolerates terminal mismatches compensated within the last 10 bases, and the
endonuclease is promiscuous in vitro, cutting largely on DNA structure rather
than sequence.

**TSD absence is not conditioned on mechanism.** A flat −0.4 penalises
full-length L1, twin-primed L1 and trans-mobilised Alu/SVA for obeying their own
mechanism — only 5'-truncated L1 reliably carries a short TSD. The C++ shares
this defect, so it is not a porting regression.

**And one to weigh.** `DEFAULT_HP_MIN` and `DEFAULT_HP_COV_MIN` come from
`scripts/calibrate_mechanistic_vs_tldr.py`, i.e. from fitting to another
caller's output. That makes the quoted figure a precision *against TLDR* and
imports TLDR's blind spots as ground truth — the "where do the weights come
from" problem, whose answer is supposed to be the FDR machinery in the selection
layer rather than another caller's calls.

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

## Regenerating the golden vectors

```bash
HTSLIB_INCLUDE_DIR=/path/to/htslib/include tools/regenerate_oracle.sh
git diff -- PLACER_py/tests/oracle/cpp_reference.json
```

Headers only; nothing links against htslib. A non-empty diff means the contract
moved — decide whether that was intended before committing.

## Where the work is now

The porting order this section used to give — genotype, dependency,
selection, then blocks and structure, then tprt — is done. The migration is
complete, and leaving a to-do list that reads as if it were not was the most
misleading paragraph in this file.

What replaces it is not a list of modules to port but a set of open
questions, each of which has evidence attached rather than an opinion:

- **`tests/EXPECTED_DIVERGENCE.md`** — the frozen C++ vectors are now a
  characterisation, not a definition. Every deliberate departure is recorded
  there with what moved, by how much, and how it was measured. One entry is
  marked **Open**: fixing `blocks._structure_explanation` to fall back to the
  shadow path (as the C++ does, and as the golden vectors confirm) raises the
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
- **`placer_py/redesign/`** — a second implementation kept alongside the
  port. It holds the only TE-hit orientation in the repository and the only
  flank-evidence collector, both of which `tprt.py` would need.
