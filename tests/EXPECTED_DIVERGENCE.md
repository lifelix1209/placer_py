# Deliberate divergences from the C++

`tests/oracle/cpp_reference.json` is frozen out of
[lifelix1209/PLACER](https://github.com/lifelix1209/PLACER) and the suite still
asserts against it. What changed is what a failure MEANS.

The C++ is no longer the definition of correct. It is a characterisation: the
values it produced, recorded so that an unintended change to the numbers is
still caught. A golden test that fails is therefore one of two things, and the
difference is the whole point of this file:

- **an accident** — fix the code; or
- **a deliberate improvement** — add a row here, in the same commit, saying
  which value moved, from what to what, and why.

A divergence that is not written down here is an accident by definition. If you
find one that is not listed, treat it as a bug until someone proves otherwise.

---

## 1. Impossible-outcome sentinel: `-1e300` → `-inf`

**Where** `log_choose_count`, `binomial_log_pmf`, `beta_binomial_log_pmf` —
formerly three copies in `genotype.py`, `policy.py` and `finalization.py`, now
one in `placer_py/core/mathx.py`.

**Was** `-1e300` in `genotype.py` and `policy.py`; `-inf` in `finalization.py`.
The three copies never agreed.

**Now** `-inf` everywhere, as `mathx.LOG_ZERO`.

**Why** `math.isfinite` is the test the finalization stage uses to separate a
line of evidence that *abstained* from one that is merely *unlikely*
(`finalization.py:129`, `dependency.py:96`, `decoys.py:195`). `-1e300` passes
that guard while meaning the opposite. Keeping two spellings of "impossible" in
a codebase that branches on `isfinite` is a trap regardless of whether it has
been sprung yet.

**Impact measured** None. The example dataset's three output files are
byte-identical across the change, and 20,829 differential checks against all
three original implementations agree on every possible-case input. The
divergence is confined to inputs the functions reject.

**Tests updated** `tests/test_02_genotype.py::test_invalid_parameters_return_the_sentinel`,
`tests/test_14_unit_coverage.py::test_log_choose_count_matches_the_binomial_coefficient`.

---

## 2. Target-site duplication geometry

**Where** `placer_py/core/tsd.py::detect`.

**Was** An unconditional `if left_bp > right_bp: swap` at the top, which made
`right_bp - left_bp` non-negative always.

**Now** No swap; the sign of `right_bp - left_bp` distinguishes a duplication
(overlapping breakpoints) from a deletion (a gap), and `detect_from_insertion`
handles the CIGAR-`I` case the reference-only comparison cannot see.

**Why** The swap erased the only signal separating the two geometries, so every
genuine TSD was reported as a DELETION of the same length — a confident wrong
answer, not a missing one. Demonstrated on a hand-built case: correct
breakpoints for a novel insertion returned `DEL/15`; without the swap,
`DUP/15` with the correct sequence.

**Impact measured** On the example dataset, `tsd_type` went from `NONE` on
every call to `DUP` with the planted length on both calls that reach the main
output. No golden vector covers `detect` — the C++ reads the reference through
a faidx handle and cannot be linked into the oracle dumper, which is why this
was never caught.

**Tests** `tests/test_33_tsd.py` (new — the detector previously had none).

---

## 3. `log_sum_exp` of an all-impossible set: NaN → -inf

**Where** `genotype._logsumexp3`, `policy.logsumexp_values` / `logsumexp_pair`
/ `logsumexp3`, now `placer_py/mathx.log_sum_exp`.

**Was** `max(a,b,c)` of three `-inf` is `-inf`, and `exp(-inf - -inf)` is
`exp(nan)`. Both modules returned **NaN** for a locus where every hypothesis
was impossible.

**Now** `-inf`, which is what `log(0 + 0 + 0)` is.

**Why** A NaN normaliser poisons every posterior derived from it, silently.
This also interacts with divergence 1: it was unreachable in these two
modules while their sentinel was `-1e300` (finite, and it normalises fine),
and became reachable the moment the sentinel unified on `-inf`. The fix and
the change that exposed it belong together.

**Impact measured** None on real input — a scan of `genotype_from_alt_vs_ref`
over alt, ref in 0..5 produces no NaN either way, because the counts are
self-consistent by construction and the impossible branch is never taken.
The example dataset's three output files are byte-identical.

**Note** The two semantics are deliberately kept apart.
`finalization.log_sum_exp_pair` still DROPS non-finite operands, because
there `-inf` means "this line of evidence said nothing" rather than "this
hypothesis is impossible". `mathx.log_sum_exp` takes an explicit
`ignore_nonfinite` flag so neither can be reached by accident.

**Tests** `tests/test_14_unit_coverage.py::test_log_sum_exp_of_all_impossible_is_impossible_not_nan`,
`::test_the_two_log_sum_exp_semantics_stay_distinct`.

---

## 4. Structure evidence when the TE alignment explains nothing

**Where** `blocks._structure_explanation`.

**Was** The branch read `if supplied is not None: return supplied`, and
`policy._as_dict` copied `record.__dict__`, so the key was ALWAYS present and
always carried a `SequenceExplanation` — default-constructed when nothing had
been explained. Production therefore took the supplied branch every time and
got structure evidence of exactly **0** whenever the TE alignment produced no
explanation.

**Now** The branch tests `status is not UNAVAILABLE`, which is what
"populated" means for this type, and falls back to the shadow path
(`explain_te_sequence_structure("N" * insert_len, ...)`) otherwise.

**Why** Both the module's own docstring ("the C++ uses
`te_sequence_explanation` when populated and otherwise falls back to
`explain_te_alignment_shadow`") and the golden certificates say the shadow
path is right. The golden test reproduced the C++ values precisely *because*
its hand-built dicts omitted the key and hit the shadow path — so production
and its own golden test had been exercising different branches, and the test
was the one that matched the C++.

**Reachable in production** Yes, and this is the part that needs real data.
`build_insert_alignment_evidence_from_blast_hits` leaves the status
UNAVAILABLE when the TE library is missing or BLAST returns no hit, while the
insert sequence and its length are present. Those loci previously scored zero
structure evidence and now score the shadow path's.

**Impact measured — and the direction I predicted was WRONG.** The commit
that made this change said it "raises their TE evidence and makes the gate
more permissive". Measured on 21:18,800,000-19,600,000 of HG002 ONT-UL, it
does the opposite: across all 340 ledger rows with no TE family,
`mechanistic_lower_log_bf_te_vs_artifact` moves by **-2.62 nats**, i.e. those
loci are rejected as artifacts more strongly, not less.

The mechanism is consistent once seen. The shadow path computes real
structure evidence from `"N" * insert_len` with no annotation, and for a
structureless insert that evidence is NEGATIVE, where the old zeroed version
contributed exactly 0. The same path is positive for genuine TE-like
structure — which is why a strong alignment that used to abstain now passes
the gate. One mechanism, both signs, and only the sign on structureless
inserts was mispredicted.

So this is a precision improvement on the loci it touches, not a risk.

Recall and precision on the development slice are unchanged: the same five
calls at the same positions, 2/3 recall against the GIAB TE truth, family
agreement 2/2, one unexplained call. Every final call there has a populated
alignment (`family_status=COMMITTED`), so the fix does not reach them — the
+0.0007 nats they do move is the dependency calibration responding to a
separate change, not this one.

**Caveat** Three TE truths in confident regions is a small sample. This
closes the question of whether the fix DEGRADES anything on the development
slice; it does not establish the size of the precision gain. That needs the
holdout, and the holdout has not been cut.

**Tests** `tests/test_24_policy.py::test_a_failed_gate_abstains_rather_than_rejecting`
(scenario changed, property unchanged); the eight golden certificates in
`tests/test_01_blocks.py` now pass with records rather than dicts.

---

## 5-9. The first caller fixes from real data

Five changes made together, each for a failure seen on the development slice
(HG002 ONT-UL, GRCh37, `21:10,000,001-20,000,000`, 23,104 reads) and none
tuned against it: the two numbers introduced are the 50 bp structural-variant
floor the tool already uses and, in (9), the balance a diploid genotype
predicts. The holdout was not looked at.

**Measured together, on that slice** (before = `4c23fa4`):

| | before | after |
|---|---|---|
| TE calls | 12 | 9 |
| GIAB TE truth recalled (confident regions) | 2/3 | 3/3 |
| calls at a Tier1 insertion (<= 58 bp away, length within 30%) | 6 | 8 |
| ... of which the TE label came from a simple repeat or a <= 30 bp match | 3 | 0 |
| calls whose TE label came from a simple repeat or a <= 30 bp match | 7 | 0 |
| ledger rows | 5,053 | 4,052 |
| `dependency_penalty_null_count` | 2,722 | 1,836 |

The one call with no Tier1 record near it (`21:11,004,386`, a 64 bp L1 match
in the centromere) is in both columns. A fourth L1 appears that the truth file
labels non-TE -- `21:18,607,133`, 3.2 kb, where the truth's own annotation has
the defect `tools/make_giab_eval.py` is known to have (single-hit coverage of
an L1 split across Dfam entries). On `examples/data`, the heterozygous SVA
(`sva_het`, AF 0.45) is now called; before, only the Alu and the L1 were.

### 5. One ledger row per observation

**Where** `core/bins.py::process_bin_records`. **Was** each component
appended its own rows, and neighbouring components of one locus often
triaged or evaluated the same hypothesis from the same reads: 1,104 of 5,053
rows (21.8%) were exact copies. **Now** a row identical in every field to one
the bin already wrote is dropped. **Why** the ledger is the run's null set --
the dependency bound, the overdispersion estimate and the conformal controls
count its rows as observations -- so a copy is one locus counted twice. 103
rows that are identical in the TSV but belong to different owner bins remain.

### 6. `insert_len` on cluster-promoted calls

**Where** `core/finalization.py::promoted_call_from_ledger_row`. **Was**
`insert_seq` copied, `insert_len` left at 0, so a promoted call reported
length 0 in `scientific.txt` and `calls.csv` while the VCF carried its
sequence. **Now** set from the sequence.

### 7. A TE hit needs 50 informative aligned bases

**Where** `core/te_classifier.py` (`MIN_ELEMENT_ALIGNED_BP`,
`informative_aligned_bases`), `core/seqtools.py::simple_repeat_mask`.
**Was** any BLAST hit named an element. Many consensus sequences contain an
(AT)n or (AAAG)n stretch, and microsatellite expansions were called as L1,
LTR66, MER52-int and HERV9N; 16-30 bp matches inside 36-82 bp inserts were
called as L1, LTR40b and Arthur1. **Now** a hit counts only if at least 50 of
the insert bases it aligns lie outside simple repeat (exact 1-6 bp arrays of
>= 12 bp, or dust-style windows); an insert with hits but none that qualify
gets `TE_ALIGNMENT_UNINFORMATIVE` and can still be a structural call. **Why a
floor and not a fraction** 18-29% of the Dfam SVA consensus is already
masked, and a real SVA's VNTR expands well beyond it; a floor leaves a 2 kb
SVA with ~1,000 informative bases untouched.

**Tried and withdrawn** Discarding HSPs that overlap in the consensus (a
tandem array aligns there once per unit). It catches the same inserts, but it
would also discard the expanded VNTR copies of a genuine SVA, the same
under-coverage that (8) fixes for L1.

### 8. Coverage is the chosen family's

**Where** `core/te_classifier.py::build_insert_alignment_evidence_from_blast_hits`.
**Was** `best_query_coverage` was the single best subject's. Dfam models L1 as
`_5end` / `_orf2` / `_3end`, so an inserted L1 aligns to two entries, and
`21:19,517,300` (3,375 bp) was 78% covered by L1P1_orf2 while L1HS_3end
explained the other 750 bp at 98.9% identity; the rest read as a
transduction. **Now** the union over the chosen family's qualifying hits;
family ranking and identity are unchanged. `te_annotation_intervals` gains
`family_cov=` when this differs.

### 9. Reference reads a heterozygous insertion predicts are not conflicts

**Where** `core/policy.py::reference_reads_unexplained_by_an_insertion`, used
by the TE and non-TE explanations. **Was** every reference-spanning read was a
`read_assignment_conflict` for "an insertion is here". ARTIFACT keeps the same
reads in a different coordinate and counts split+indel reads as its
conflicts, so a het insertion with fewer split+indel than reference reads
could never dominate it on every coordinate, and the Pareto comparator
abstained. That L1 was `TE_AMBIGUOUS` on 24 conflicts against ARTIFACT's 21,
with 0 unexplained bases against 1,125. **Now** `max(0, ref - alt)`: only
reference reads beyond the balance a het insertion predicts. The comparator
itself, and its golden vectors, are unchanged.

**Tests** `tests/test_23_te_classifier.py` (simple repeats, short hits, the
split L1, and five fixtures whose inserts were `"ACGT" * n` -- itself a
tandem repeat -- given non-repetitive sequence; one whose 30 bp Alu hits were
lengthened to 60), `tests/test_24_policy.py::test_a_heterozygous_insertion_is_not_charged_for_its_reference_reads`.

---

## Not divergences

For the record, these look like behaviour changes and are not:

- **The banded edit-distance rewrite** (`breakpoints.py`). 480,000 differential
  cases against a textbook implementation, zero mismatches; all three output
  files byte-identical before and after.
- **Streaming the read path** (`pipeline.py`). Reordering when reads are
  consumed, not what is computed.
- **The abPOA memory budget** (`consensus.py`). It changes how many event
  strings reach the aligner on inputs that previously exhausted memory, so on
  those it is a change from "no result" to "a result from fewer reads". When
  it binds, `final_qc` carries `EVENT_CONSENSUS_POA_MEMORY_CAPPED` and
  `EventConsensus.poa_reads_dropped_for_memory` holds the count.

  Measured on 21:18,800,000-19,600,000 of HG002 ONT-UL: peak memory 4.58 GB
  to 420 MB with CPU unchanged (1006 s to 980 s), the same five calls at the
  same positions with the same genotypes and TE families, and recall against
  the GIAB TE truth unchanged at 2/3. The only numeric movement is in the
  dependency calibration — sigma 110.164 to 110.245, a 0.08% shift — which is
  the consensus being built from fewer reads at some loci, exactly the trade
  the budget makes.
