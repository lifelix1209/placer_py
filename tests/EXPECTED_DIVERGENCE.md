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
one in `placer_py/mathx.py`.

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

**Where** `placer_py/tsd.py::detect`.

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

## Not divergences

For the record, these look like behaviour changes and are not:

- **The banded edit-distance rewrite** (`breakpoints.py`). 480,000 differential
  cases against a textbook implementation, zero mismatches; all three output
  files byte-identical before and after.
- **Streaming the read path** (`pipeline.py`). Reordering when reads are
  consumed, not what is computed.
- **The abPOA memory budget** (`consensus.py`). It changes how many event
  strings reach the aligner on inputs that previously exhausted memory, so on
  those it is a change from "no result" to "a result from fewer reads" — and
  the count is recorded on `EventConsensus.poa_reads_dropped_for_memory`. On
  every input that completed before, the budget is not binding.
