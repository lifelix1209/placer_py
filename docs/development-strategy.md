# Development strategy: replay first

How PLACER's algorithm is changed. This is the method every change to what
PLACER decides follows. It was adapted from Dream-RSI (Zheng et al.,
*Recursive Self-Improvement through Evolving Worlds*, arXiv 2609.14858, 2026)
and then shaped by the first rounds of applying it here.

## Why

**Our evaluator is expensive, and most questions do not need it.** One HG002
chr1 scan takes 2 h 17 min of wall time on a 16-core allocation: 8.5 CPU-hours
used and 37 core-hours held, since one worker spends most of the run on the
pericentromeric chunks. Replaying a decision policy against that scan's
recorded evidence takes under a second.

Before this method, a decision-layer idea was tested by rerunning the scan. The
scan got rerun on a 10 Mb slice with six truth insertions because that was all
anyone could afford, so ideas were tuned on six loci. It showed: 4cbf656 was
tuned offline to 5/6 on that slice and recalled 4/6 when rerun.

Dream-RSI's observation is that a finished search already records the outcomes
of the decisions it made. That record is a simulator for any policy that makes
those decisions differently. The expensive evaluator is needed only for what
the record does not contain.

For PLACER, the scan is the expensive evaluator and the decision layer is the
policy. The evidence ledger of a finished scan is the record.

## Vocabulary

| term | here |
|---|---|
| **world** | One scan commit's output on one dataset and region, frozen: its evidence ledger (every evaluated hypothesis, with everything measured about it), its insert sequences (`--record-world`), and their RepeatMasker annotation. Registered in `placer_dev/dream/worlds.json`. |
| **policy** | Code that maps a world's rows to calls: which loci to select, what to call them, where to place them. `tools/dream/policies/current.py` is PLACER's own decision, calling the production function. (Until 2026-09-27 it was 4cbf656's likelihood-gated decision, and the tables of rounds 1-3 below mean that by `current`; it is now `coverage_placed`.) |
| **objective** | How a policy's calls are scored. It uses TEBench's own evaluator and nothing else (`tools/dream/objective.py`). |
| **candidate** | A proposed policy. It lives in `placer_dev/dream/candidates/` until it is accepted. |
| **round** | Diagnose, propose, replay, accept or reject, log. |
| **node** | One logged candidate, accepted or not, in `placer_dev/dream/tree.jsonl`. |
| **pool** | The worlds a candidate is replayed on. |
| **online** | Running the scan for real, from frozen code. |

## 1. Decide which side of the boundary the change is on

**Replay domain.** Everything that reads what the scan recorded and decides
from it:
- selection (e-BH, the decoy correction, q);
- the TE / structural label and the family name;
- locus grouping, and which row tests a locus and which row places it;
- thresholds and model parameters over recorded terms.

These changes are developed and accepted in replay. Running the scan to test
one of them is a mistake.

**Scan domain.** Everything that changes what gets recorded:
- discovery;
- consensus and segmentation;
- alignment;
- new observables;
- admission to the expensive stage.

For admission, a world in which every component was evaluated makes most of
the question replayable.

These changes need online runs. Before making one, ask what the policy would
need recorded so that the next iteration can be replayed. **Make the world
richer before making the policy smarter.** The first round needed the insert
sequences and a RepeatMasker annotation. Without them, 26 of 51 "false
positives" were an artefact of the replay.

## 2. The round

1. **Diagnose on replay.**
   ```
   python3 -m tools.dream.run diagnose WORLD POLICY
   ```
   This lists the false negatives by cause and the false positives with their
   evidence. Work on the largest class, and confirm the cause before writing a
   candidate. In round 1, "no truth nearby" turned out to be real GIAB
   insertions that TEBench does not count as TE, and the fix belonged in the
   replay, not the policy.
2. **Propose a candidate.** It is one hypothesis, written down in the
   candidate's docstring: what it changes and which diagnosed class it targets.
   A family of variants of one hypothesis is one candidate with parameters.
3. **Replay on the pool.**
   ```
   python3 -m tools.dream.run compare WORLD current CANDIDATE --log --note "..."
   ```
   Compare against the currently accepted policy, on every world in the pool.
4. **Accept only if all of these hold:**
   - The paired block bootstrap's 5th percentile of the gain is above 0 on the
     primary world. On chr1, one truth insertion is 0.4% of recall, so a gain
     of one or two loci is noise and does not pass.
   - No world in the pool regresses beyond its own noise.
   - The coordinate-shift invariance check passes (section 4).
   - The candidate reads only recorded observables.
   **Validity fixes are the one exception.** Some changes remove loci whose
   null model is known to be misspecified, so that their e-values are
   invalid. Removing them can cost objective points, because invalid
   "discoveries" inflate k in e-BH and loosen the threshold for everyone
   else. Such a change is accepted when all of these hold:
   - the misspecification is shown from recorded data;
   - the removed loci carry no truth, or none within the confident regions;
   - no guard world regresses;
   - the objective loss is reported;
   - the maintainer signs off.

   The first was alignment-collapse regions, adopted on 2026-09-26:
   - **What.** Where at least 100 hypotheses within ±50 kb have ≤2
     reference-spanning reads, the reference and the sample disagree, and
     every position looks like a homozygous insertion.
   - **Scale on chr1.** 24% of evaluated rows and 1,323 junk structural calls,
     with no confident bases.
   - **Cost.** −3 TP through the e-BH coupling.
5. **Log every candidate**, the rejected ones too: `--log` writes a node with
   the hypothesis, the numbers and the verdict. A rejected hypothesis is a
   result, and the tree is how the next round avoids repeating it.
6. **Promote what is accepted into `placer/core/`**, and have the replay call
   the production function (`policies/current.py` does). What ships is what
   was replayed. A replay-only reimplementation of production logic drifts.
7. **Go online**:
   - every few accepted changes;
   - always before a default changes;
   - whenever the scan changed.

   Run from frozen code, on the validation contigs. Record the new scan as a
   world and add it to the pool: this is what keeps the pool from being only
   the history of old policies.
8. **Stop** after two online rounds without gain, or when the targets are met.

**Candidates are proposed one at a time.** After two rounds with no replay
gain, fan out: M = 4 subagents, each with a different hypothesis, all scored on
the same pool. This is more expensive, so confirm with the maintainer first.

## 3. The objective

**Score exactly as the benchmark scores, or the replay optimises the wrong
thing.** `objective.py` imports TEBench's `evaluate` (one-to-one matching
within ±100 bp, confident regions only).

An annotated world is re-annotated the way TEBench re-annotates a caller:
- RepeatMasker, run from TEBench's pinned container, on the call's insert;
- `annotate_from_repeatmasker`: at least 100 bp and 50% TE coverage, and a
  family name.

Only PASS calls are scored, as TEBench's normaliser keeps only those.

**Primary objective: TE recall at precision ≥ 0.95** on the human chr1 world.
- **The precision floor.** Below it the value is recall minus 10 × the
  shortfall. That puts 0.95 at the precision of sniffles2 (0.948) and graffite
  (0.943) on HG002, so the claim is higher recall at the same precision.
- **Two recalls.** Recall is reported the way TEBench computes it. The truth
  lists 567 of its 3,394 call ids twice, so recall is also reported with
  duplicates merged.

**Guards.** They are not optimised, but must not regress:
- **Per-class decoy check.** Ê ≤ 1 per class, from shifted breakpoints; above
  1 it is a hard failure.
- **Cichlid consistency**, until the validated panel exists:
  - tldr PASS calls matched;
  - the fraction of TE calls matching a Sniffles2 insertion on the same reads;
  - agreement of PLACER's TE label with RepeatMasker's.
- **Family concordance.**

**Report numbers, not adjectives.** A commit that changes what PLACER decides
cites:
- the world;
- the replay TP / FP / FN, precision and recall;
- the bootstrap interval;
- the online numbers once they exist.

## 4. No peeking

Dream-RSI requires a policy to decide only from what the replay has revealed.
Here that means:

- **A policy receives world rows and nothing else.** The truth set is held by
  the objective and is never passed in.
- **Coordinates and ids group rows; they are never features.** A policy whose
  calls change when every row is moved 7.8 Mb onto a renamed contig has
  memorised where the answers are. `objective.check_invariance` enforces this
  on every `compare`.
- **Diagnosis output is for whoever writes the candidate, not for the
  candidate.** It shows truth positions. A threshold chosen to flip a listed
  locus is not a mechanism. Every threshold needs a reason that would hold on
  another genome, and is checked on the other worlds in the pool.
- **The replay's boundary is hard.** A policy cannot use what the scan did not
  record. When it needs to, that is a scan-domain change (section 1).

## 5. Data discipline

| set | used for |
|---|---|
| **human chr1** (TEBench development contig) | The dreaming pool's primary world. |
| **cichlid D2 chr1:10–20 Mb** | Pool guard. Its truth panel is pending. |
| **human chr2–8** (development contigs) | Online validation only, never dreamt on. A change accepted in replay is confirmed here before any default switches. |
| **TEBench holdout contigs** | Not touched until the release benchmark. |
| **the 10 Mb dev slices** | Smoke tests only. With six truth insertions they can neither accept nor reject anything. |

### Release benchmark 1.0.0a3 (pre-registered 2026-09-28, amended 2026-09-29)

The human holdout is spent here, once, on the released code. This was written
before any of the runs below started.

It was first drafted for 1.0.0a1, and moved to 1.0.0a2 before any run started.
1.0.0a2 is the speed release. It writes byte-identical output on every 1.0.0a1
release run (HG002 chr1–8 and the cichlid slice; `calls.vcf` apart from its
`##source` line), so every 1.0.0a1 number below still stands.

**Amendment, 2026-09-29: 1.0.0a3.** The rule below was followed: a bug found
before the holdout was scored.
- **How it was found.** 1.0.0a2 ran all 13 conditions and was checked on
  chr1–8 only. The check found PASS calls genotyped 0/0, and their share rose
  with depth: 0% at 5x, 22% at full.
- **The diagnosis, on chr1–8 only.** The genotyper's length-concordance term
  was wrongly applied when finalization re-genotyped the calls (`CHANGELOG.md`,
  1.0.0a3).
- **The fix changes only GT and GQ.** Rerun from frozen code, the nine release
  runs keep their ledgers byte for byte. Their calls are identical apart from
  GT and GQ, and so are their TP/FP/FN.
- **What was not looked at.** The holdout was not scored, and no holdout
  record was read, for 1.0.0a2 or for the fix.
- **What happens to the 1.0.0a2 runs.** They are kept in
  `placer_dev/tebench_1.0.0a2/runs` and never scored on the holdout. The 13
  runs are repeated with 1.0.0a3, and everything below applies to 1.0.0a3.

**Deferred on purpose, 2026-09-29: the scan's use of the same term.**
- **What was found.** Also found before the holdout was scored: the scan
  still applies the length-concordance term to its own GQ.
- **What that GQ decides.** Only the joint decision's diagnostics read it:
  the INFO fields TEPOST, LFDRMAX, MECHART, MECHNONTE, MECH and QC. Its one
  route to calls is the per-bin de-duplication of ledger rows.
- **A first emulation.** The production selection, run on the 1.0.0a1
  release ledgers with that de-duplication fixed, predicts:
  - HG002 chr1's calls are unchanged;
  - two cichlid TE calls go, one of them PASS;
  - the sample overdispersion moves (chr1 0.1729 → 0.1799).
  1.0.0a4's replay round and online A/B are to confirm or correct this.
- **The decision.** The maintainer decided to fix it in 1.0.0a4, outside this
  benchmark. The benchmark stays on 1.0.0a3, and 1.0.0a4 is never scored on
  the holdout for it.

- **Code.** The code is v1.0.0a3, taken by `git archive` into
  `placer_dev/frozen/v1.0.0a3`.
  - Its `placer/` differs from `a3-rc1` in one docstring only. `a3-rc1`
    reran the release runs: identical ledgers, and calls identical apart from
    GT and GQ.
  - It is pip-installed, non-editable, into TEBench's
    `workflow/envs/placer.yaml`.
  - It is run as a user would run it: over the whole BAM, with no `--region`.
    It goes through TEBench's `run_caller` with 32 threads and
    `--record-world`.
- **Conditions.** human_hg002, under the same 13 conditions as every
  registered caller: full/r0, and 5/10/20/30x × r1–3. The input is
  `*.primary.md.bam`.
- **Metric.** TEBench's `evaluate`, unchanged.
  - It reports TP/FP/FN, precision, recall and F1, with Wilson intervals.
  - It counts PASS calls within ±100 bp inside the confident regions, after
    RepeatMasker re-annotation with `--require-te`.
  - It is reported for three partitions, for PLACER and the eight other
    registered callers alike:
    - all, which is TEBench's summary;
    - development: chr1–8;
    - holdout: chr9–22, X and Y.
  - Nothing else is scored.
- **Reference.** The per-chromosome release runs, pooled over chr2–8, gave
  923/42/316.
  - Precision was 95.6% [94.2, 96.8] and recall 74.5% [72.0, 76.8] (Wilson,
    95%).
  - The prediction under test: full/r0 on the holdout falls inside both
    intervals.
- **Order.**
  1. full/r0 runs first. It is checked only on chr1–8, and on quantities
     that need no truth:
     - its chr1–8 records against the per-chromosome release runs. They may
       differ, because three things are estimated over the whole run: e-BH's
       m, the per-class decoy factor and the genotyper's overdispersion.
     - its CPU-h and peak RSS;
     - its development score against 1094/52/374. 1.0.0a2 scored
       1097/53/371 here. 1.0.0a3 changes no call, so it must score the same.
  2. The twelve titration runs follow, checked the same way.
  3. Only when all 13 are done is every partition scored, once, for every
     caller.
- **Rules.**
  - Before the holdout is scored, a bug may be fixed. The fix is judged only
    on chr1–8 or on the 10 Mb slices. The fixed code becomes 1.0.0a3: this
    entry is amended and the runs are repeated.
  - After the holdout is scored, no code changes for this benchmark, and the
    numbers are reported whatever they are.
  - From then on, human chr9–22, X and Y cannot accept or reject any
    candidate. The next untouched set is mouse, once its truth is built.

## 6. Worlds

- **Record from frozen code.** Use `--record-world` from a snapshot or
  detached worktree that nobody edits while the job runs: spawned workers and
  lazy imports read code from disk mid-run.
  ```
  rsync -a --exclude .git <worktree>/ placer_dev/frozen/<name>/
  ```
  Then submit with `REPO=` pointing at the snapshot.
- **Register the world with its scan commit.**
  ```
  python3 -m tools.dream.run register NAME --path RUN --dataset DS --region R --scan-commit C
  ```
- **Annotate it.** This is about 2 min for the 10 Mb cichlid slice.
  ```
  python3 -m tools.dream.run annotate NAME --library LIB --submit
  ```
- **Worlds go stale.** When the scan changes what a column means, a world
  recorded by the old scan replays the old meaning. Prefer the latest scan's
  world for each dataset. Keep older worlds only to show that a change is not
  specific to one world. Columns an old scan did not record are listed at load
  and filled with defaults: do not read a default as a measurement.
- **Check a new world before using it.** Replaying `current` on it must give
  exactly the online run's selection. It did for both worlds recorded in
  round 1: 125 TE calls on chr1, and 286 TE / 183 structural on cichlid.
- **Know the world's quirks.**
  - `event_consensus_len` includes about 80 bp of reference flank on each
    side, so it is not the insert length. Use `insert_len`: the recorded insert,
    or the consensus less both flanks. Treating the consensus as the insert
    made 66–82 bp insertions pass a 100 bp floor.
  - The truth duplicates multi-allelic sites.
  - Decoy rows are not candidates.

## 7. Performance changes

Performance work is scan-domain, and each change must leave the four outputs
**byte-identical**. Run the same region through two frozen snapshots and
`cmp` every file: `tools/perf/ab_region.sh`. A change that alters the
output is not a performance change. It is a decision change and goes through
replay.

Two measurement traps, both hit on 2026-09-26:
- **CPU is user + sys, not allocation.** A chr1 scan held 16 cores for
  2 h 17 min (37 core-hours) but used 8.5 CPU-hours: one worker ran the
  pericentromeric chunks while the rest sat idle. TEBench's `cpu_hours` is
  user + sys, so report that, and treat wall time as a separate, load-balance
  question.
- **cProfile inflates tiny functions called millions of times.** It charged a
  per-character `all(...)` with 19 of 89 s. Without the profiler, the whole
  region took about 30 CPU-s, and replacing it changed nothing measurable.
  Profile by sampling (`py-spy`, `tools/perf/spy_region.sh`). Claim a
  gain only from repeated A/B user + sys times on a node that is not shared
  with another heavy job.

Perf round 1 (2026-09-28) added two more:
- **Attribute system time before guessing its cause.** On the release node,
  chr1's sys time was 37% of its CPU, and on other nodes 50-200% of user time.
  A ~50x re-read of the BAM from BeeGFS (no page cache) looked like the cause.
  It was not: a run with the BAM and the reference in `/dev/shm` paid the same
  sys time. `PLACER_PERF_LOG` splits each chunk's CPU into the interpreter's
  and its children's, and 90% of the sys time was in blastn. Each exec paged
  in 86 conda libraries from BeeGFS (~1,000 major faults), which cost 0.6-0.9 s
  with 48 processes on a node. Staging blastn on local disk
  (`io/blast.staged_blastn`) took chr1's sys time from 13,755 s to about
  1,000 s.
- **Test a single-worker win under concurrency too.** The blastn cost was
  0.09 s of sys time per call with one worker, and ten times that on a busy
  node. `tools/perf/ab_step.sh` runs three 16-worker jobs side by side.

The tools this round added:
- `PLACER_PERF_LOG=PATH` writes a TSV row per chunk: CPU split into self and
  children, I/O, and fetch/cache counters. `tools/perf/perf_summary.py`
  summarises it.
- `tools/perf/kernel_corpus.py` records a region's real segmentation calls and
  replays them. That checks an optimisation against the recorded results in
  seconds, and times it.
- An equivalence test keeps the old implementation verbatim beside the new one
  (`tests/test_53`-`test_56`).

## 8. What the rounds found (worked examples)

HG002 chr1, scored as TEBench scores. The truth has 277 rows in the confident
regions.

| policy | TP | FP | precision | recall (dedup) |
|---|---|---|---|---|
| `current` (4cbf656 decision) | 94 | 22 | 81.0% | 33.9% (41.0%) |
| `coverage_rule`: one e-BH on the artifact ratio; TE iff TEBench's rule holds on the insert | 165 | 51 | 76.4% | 59.6% (72.1%) |

`coverage_rule` was rejected: it is further below the precision floor. Its 51
false positives, diagnosed:
- **26 real GIAB insertions that TEBench does not count as TE.** The fix was
  to the replay: re-annotate with RepeatMasker as TEBench does.
- **18 breakpoints 51–227 bp to the right of the truth**, each also a false
  negative. Placement by read counts was tried and rejected. Left-normalisation
  against the reference is the next candidate.
- **4 GIAB insertions under 100 bp.**
- **3 with no GIAB insertion nearby.**

On cichlid, `coverage_rule` matched 52 of 78 tldr PASS calls against 49 for
`current`, and 58% of its TE calls matched a Sniffles2 insertion against 50%.
Its TE label agreed with RepeatMasker's on 96.2% of selected insertions.

**Round 2**, scored exactly as TEBench scores. The world is `h_chr1_dream1`,
annotated by RepeatMasker. The truth has 229 loci, after TEBench's
haplotype-record merge (below). Calls are scored at the position the VCF
writes, and only FILTER=PASS calls count.

| policy | TP | FP | precision | recall |
|---|---|---|---|---|
| `current` (+ collapse fix) | 128 | 13 | 90.8% | 55.9% |
| `coverage_rule` | 145 | 21 | 87.3% | 63.3% |
| `coverage_placed` (the rule + precise placement; production since 2026-09-26) | 157 | 10 | 94.0% | 68.6% |
| sniffles2, for reference | 180 | 11 | 94.2% | 78.6% |

Accepted: `coverage_placed` against `current`, gain +0.450, 90% [+0.050,
+0.825]. It replays to exactly the decisions of the candidate it was promoted
from, on chr1, chr8 and cichlid. On the example data its replayed calls equal
its online `calls.vcf`.

What led to precise placement:
- The locus was tested by a wide breakpoint interval, which gathers the most
  reads.
- The precise rows (bp_left == bp_right) from reads' CIGAR insertions sat at
  the truth.
- Testing by one row and placing by another fixed most of the offset calls.
- The result is a plateau: the same for windows of 100, 200 and unlimited.
  Nothing was tuned on cichlid, yet its matched tldr PASS calls rose from 52
  to 65 of 78.

**Score at the position the output reports, with the output's filter.** Until
18:30 on 2026-09-26 the objective placed every call at `pos + 1`, but the VCF
writes the left breakpoint `bp_left`. For a wide interval, `pos` is the
midpoint. The objective also counted TE calls the VCF writes as non-PASS
(IMPRECISE, FAM_ABSTAIN), which TEBench drops. Both inflated the early round-2
numbers (e.g. 180 TP for this policy). The check that would have caught it:
replay a run's own ledger and compare with that run's `calls.vcf`. It must be
identical.

**Round 3: a scan fix is judged end to end, like any other change.**
- **The defect.** Dispersed carriers of one insertion were counted as reference
  support: a homozygous Alu's 52 carriers counted as 5 alt against 35
  reference.
- **The fix.** The same-allele carrier rule. At the truth loci it rescued, alt
  went from 2–5 to 20–52.
- **The measurement.** A new chr1 world recorded with the fix, replayed with
  the accepted policy, gave +3 TP and +3 FP: gain −0.140, 90% [−0.39, +0.12].
  Rejected, and made opt-in.
- **The reason.** Every precise hypothesis of a rescued locus now gathered the
  whole allele, so the call went to an off-mode offset, 127–502 bp from the
  truth. sniffles2 reports near the carriers' median.
- **The lesson.** The evidence fix needs a placement that uses where the reads
  put the insertion. Until then it is not a gain, however right it is locally.

**Before blaming the benchmark, check the other callers.** These offsets sit
inside tandem-repeat arrays, and were first put down to how the benchmark
represents insertions there. But sniffles2 placed 14 of the 16 within 100 bp,
mostly at 0 bp, so the defect was PLACER's. A loss is a benchmark artefact only
if callers that are otherwise good lose it too.

**Round 4: the fan-out, and a scan change judged against the scan it replaces.**
After two rounds with no accepted gain, four subagents searched four directions
at once (2026-09-27), each on the chr1 carrier-rule world against
`coverage_placed` (160 TP / 13 FP there):

| direction | best TP / FP | verdict |
|---|---|---|
| place at the own-read median of the allele's rows | 163 / 10 | accepted, p05 +0.004 |
| merge loci that share carrier reads, place by allele | 162 / 7 | accepted, p05 +0.004 |
| a separate e-BH for the TE-rule loci | +3 TP, 0 FP | rejected, p05 0 |
| relabel old, diverged inserts as TE | +3 TP, 0 FP | rejected, p05 0 |

Both accepted candidates need the carrier rule in the scan, so the question
that decides is "carrier scan + candidate" against "current scan + current
policy". `validate --base-worlds` asks it: each policy replays on its own
scan's worlds, and the bootstrap pairs them by the shared truth. The
pre-registered comparison on chr2-8 (never dreamt on), the merge candidate at
its best chr1 settings, failed: pooled TP 858 against 858, FP 52 against 42,
gain -0.071, 90% [-0.159, +0.009]. The carrier rule alone is significantly
worse (-0.148, [-0.253, -0.008]); the candidate does beat the base on the
carrier scan (+0.077, [+0.005, +0.130]), but not by what the rule costs. The
rule stays opt-in and the line is closed.

**Lesson: judge a candidate against the whole change it needs.** Accepted
against the base on the candidate's own scan, both fan-out winners looked
like gains. Against production they are a loss.

**Rounds 5 and 6.**
- **Round 5.** Combining the two accepted placements gave the merge
  candidate's 162 / 7 exactly. The loci the median placement fixes are ones
  the merge already fixes.
- **Round 6.** A decoy check without the collapse rows gave identical calls on
  chr1: every class factor is 1.000 either way. It is not a gain, but it is
  the precondition for skipping the expensive stages in collapse regions.

**Validity, measured and decided by the maintainer (2026-09-27).** The
fan-out found that a locus's e-value is the MAX over its hypotheses, and the
max of e-values is not an e-value (its null mean can reach the row count); the
mean is one under any dependence. On the production scan the mean costs
chr1 157 / 10 -> 154 / 9 and chr2-8 858 / 42 -> 849 / 38. The maintainer took
it under the validity-fix exception. A separate e-BH for the TE calls (their
own FDR <= q; chr2-8 875 / 45) was declined: the decision stays two-step,
existence over all loci and then the TE label.

**Read a loss at the level that caused it (`run.py levels`).** PLACER is a
TE caller (maintainer, 2026-09-27), and TEBench's score of it is the end of
three questions: is there an insertion, is it a TE, is it PASS. `levels`
follows each TE truth locus down one matching. HG002 chr1, production after
the mean fix:

| stage | loci | lost | where they went |
|---|---|---|---|
| TE truth | 229 | | |
| found by any selected insertion | 182 | 47 | 17 artifact <= 0, 16 locus selected but placed > 100 bp away, 9 below the e-BH threshold, 5 no hypothesis |
| labelled TE | 172 | 10 | 5 no TE alignment at all (short SVA, L1MC4), 5 old elements at 34-47% coverage |
| PASS | 156 | 16 | 15 IMPRECISE, from the scan's joint decision; 1 FAM_ABSTAIN |
| RepeatMasker calls it a TE | 154 | 2 | |

- **Label agreement.** PLACER's TE/SV label agrees with the truth's on 92.6%
  of the matched insertions, and with RepeatMasker's on the same sequences
  on 92.4%. About 35 TE labels are not TEs by the truth's definition.
  TEBench's re-annotation hides this; PLACER's user sees it.
- **Every GIAB insertion (a diagnostic only).** P 0.922, R 0.534. sniffles2
  is at 0.909 / 0.678.

**Round 7 (measured for the maintainer): does level 1 need the library?**
vs_artifact adds, to the counts, a TSD term chosen by the superfamily and
the L1 endonuclease motif for LINE, SINE and Retroposon. Both depend on the
class the library alignment gives. Testing existence on the counts term
alone (`round7_counts_only`, HG002 chr1) changes the result as follows:

| | TP / FP | P | R | TE truth found at level 1 |
|---|---|---|---|---|
| production | 154 / 9 | 94.5% | 67.2% | 182 |
| counts only | 159 / 11 | 93.5% | 69.4% | 187 |

Gain -0.073, 90% [-0.201, +0.041]: rejected by the objective. The
class-specific linkage terms buy precision at a small cost in recall.

**What levels 1 and 2 lose, looked at directly (chr1).**
- **Artifact ratio <= 0.** Most of these misses have few alt reads against
  many reference reads (2 / 40, 4 / 44): the allele's reads were counted
  elsewhere. That is the carrier rule's target, and it failed validation.
- **Placed more than 100 bp away.** Most of these are wide testing intervals
  with no precise hypothesis. They are reported at `bp_left`, with the truth
  near `bp_right`.
- **Level 2 against RepeatMasker on the same inserts: 36 over-calls, 10
  under-calls.**
  - Over-calls: for 26 of the 36, RepeatMasker has no TE hit at all. They are
    simple or low-complexity inserts that BLAST matches into ERV1 and L1
    consensuses at identity 0.76-0.86. Another 6 are class Unknown, which the
    VCF writes as FAM_ABSTAIN anyway.
  - Under-calls: 4 short SVA and L1 fragments (107-147 bp) that BLAST does not
    find, and 5 old elements that BLAST covers only 42-48%.

**Round 8 (the maintainer's decision, measured before building it): level 2
by RepeatMasker on the selected inserts.** The labels come from the world's
RepeatMasker annotation of PLACER's own inserts, under TEBench's rule, and
level 1 is unchanged (`round8_repeatmasker_labels`). A policy that takes an
`annotation` is handed it by `run.py`: RepeatMasker's reading of an insert is
an observable of the insert, not the truth.

| | chr1 TP / FP | chr2-8 TP / FP | chr2-8 P / R | chr1 label agreement with the truth |
|---|---|---|---|---|
| BLAST coverage rule | 154 / 9 | 849 / 38 | 95.7% / 68.5% | 92.6% (35 non-TE labelled TE) |
| RepeatMasker | 160 / 14 | 882 / 59 | 93.7% / 71.2% | 96.8% (12 non-TE labelled TE) |

The labels are much closer to the truth's definition. The score still falls,
because TEBench's re-annotation used to discard the 35 wrong BLAST labels,
while the 12 that remain are ones it keeps. All five new chr1 FPs are real
GIAB insertions within 100 bp, where RepeatMasker calls PLACER's short
assembled insert (107-160 bp of SVA or L1) a TE and the truth's RepeatMasker
does not call GIAB's sequence one. What limits level 2 is then how closely
the assembled insert matches the real one.

**Level 1 without the library, measured end to end and not adopted
(2026-09-27).** At the maintainer's request the scan recorded a library-free
level-1 ratio: the counts term plus a TSD term under the class-free length
model, with its own decoy check. It is on branch `phase3f-level1`, and the
worlds `*_exists1` were recorded from snapshot `frozen/exists1`. On those
worlds, with levels 1 and 2 crossed:

| level 1 | level 2 | chr1 TP / FP | cichlid PASS TE / with Sniffles2 / tldr matched |
|---|---|---|---|
| class-specific (production) | BLAST | 154 / 9 | 175 / 134 / 56 |
| class-specific | RepeatMasker | 160 / 14 | 203 / 156 / 65 |
| counts only | BLAST | 159 / 11 | 195 / 134 / 55 |
| counts + class-free TSD | BLAST | 159 / 10 | 186 / 134 / 55 |
| counts + class-free TSD | RepeatMasker | 165 / 15 | 213 / 156 / 64 |

The maintainer pre-registered "counts + class-free TSD, BLAST" for chr2-8.
Against production: pooled 849 / 38 -> 865 / 44, P 95.2%, gain +0.013, 90%
[-0.071, +0.021] (node 2bd9d258a2). That is neither a proven gain nor a
proven loss, with 6 more FPs and, on cichlid, 11 more PASS calls no other
caller supports. Not adopted: level 1 keeps the class-specific linkage terms.

Two lessons carry over:
- A design principle ("level 1 should not depend on the library") is still
  a change to what PLACER decides, and is judged like any other.
- RepeatMasker at level 2 goes opposite ways on the two genomes. It gains
  most on the de novo cichlid library (tldr 56 -> 65 of 78). On human it
  loses precision at TEBench's 100 bp line, because the assembled inserts
  are 10-30 bp longer than GIAB's.

**Round 9: recall first (maintainer, 2026-09-27), and the precision floor's
noise.** `levels` put the largest single recall loss at FILTER: 15 TE calls
on chr1 that matched the truth were IMPRECISE, 14 of them at an exact
breakpoint. The flag came from the scan's joint decision. The next largest
was wide intervals reported at their left end, with the truth at either end
about equally. The candidate (`round9_filter_midpoint`) makes IMPRECISE mean
"an interval wider than 200 bp, or no breakpoint", and reports narrower
intervals at their middle.

- **chr1: 154 / 9 -> 171 / 10** (recall +7.5 points), but gain +0.074, 90%
  [-0.121, +0.283]. The rule rejects it on chr1.
- **chr2-8, pre-registered: 849 / 38 -> 923 / 42**, gain +0.060, 90%
  [+0.014, +0.091]. Adopted. This was the second use of chr2-8 for a FILTER
  rule (F1 failed there first, node 228085220d); the holdout is untouched.
- **Online, HG002 chr8, from the frozen commit (`frozen/filter1`).** The
  run's 141 PASS records equal the replay of its own ledger. On its own
  world it scores TP 101 / FP 4, as the replay of the production scan's
  chr8 world does.

**Lesson: near the precision floor, chr1 cannot see a recall gain.** chr1's
base precision was 94.5%, just under 0.95. Every bootstrap replicate lands
on either side of the floor, and the 10x penalty for the shortfall swamps a
recall difference of several points. chr2-8, at 95.7%, is far enough above
the floor to measure recall. When the development world sits at the floor, a
chr1 rejection whose point gain is positive is not evidence against the
candidate. That is when the maintainer may send it to validation anyway.

**TEBench's truth listed 567 insertions twice.** GIAB writes an insertion on
both haplotypes as two heterozygous records at one position. The maintainer's
decision (2026-09-26) was to merge them in TEBench's `evaluate()` into one 1/1
locus, applied to every caller. The replay imports `evaluate()` and so scores
the same way.

## 9. Checklist for a change to what PLACER decides

- [ ] Which side of the boundary? If scan-domain: what does the next round need
      recorded?
- [ ] Diagnosed on replay; the targeted class named in the candidate.
- [ ] Replayed on every world in the pool against the accepted policy; node
      logged.
- [ ] Bootstrap 5th percentile > 0 on the primary world; no world regresses;
      invariance passes.
- [ ] Promoted into `placer/core/`, with the replay calling the production code.
- [ ] A test pinning the new behaviour, and `CHANGELOG.md` with the numbers.
- [ ] Before a default changes: online validation on chr2–8 from frozen code,
      and the new scan recorded as a world.

## 10. Commands

```bash
python3 -m tools.dream.run register NAME --path RUN_DIR --dataset human_hg002 \
    --region chr1 --scan-commit COMMIT
python3 -m tools.dream.run annotate NAME --library LIB.fa --submit
python3 -m tools.dream.run score    NAME POLICY [--param k=v] [--check-invariance]
python3 -m tools.dream.run diagnose NAME POLICY [--limit 40]
python3 -m tools.dream.run levels   NAME POLICY [--limit 20]   # discovery, TE-or-not, FILTER, waterfall
python3 -m tools.dream.run compare  NAME current CANDIDATE --log --note "hypothesis"
python3 -m tools.dream.run validate current CANDIDATE --worlds W2 ... W8 --log \
    [--base-worlds B2 ... B8]    # base on its own scan's worlds, when the candidate needs a new scan
```

Run them from the repository root, with an environment that has pysam, for
example `placer-dev`. `TEBENCH` overrides the TEBench checkout the objective
imports.
