# The four modules nothing imported

`tprt.py`, `integrate.py`, `decoys.py` and `null_control.py` were once
reachable only from the test suite. A code audit called them a "five-module
dead cluster, ~1187 lines" and proposed a unify-or-delete sweep. That reading
was wrong: each was deliberate, for a different reason, and this file said
which. The fifth, `selection.py`, was a genuine duplicate of logic inside the
legacy finalization, and was unified.

All four have since resolved, each the way this file said it would.

| module | then | now |
|---|---|---|
| `tprt.py` | the intended replacement for `blocks.py`'s clamped scores, missing six producers | on the calling path: `core/mechanism.py` builds the per-class likelihood ratios from its terms |
| `null_control.py` | the breakpoint-shift callback `decoys.py` was written to take | on the calling path: `core/locus_evidence.py` builds every evaluated locus's decoys with it |
| `integrate.py` | kept "until the decision layer is rewritten", with `tests/test_13_end_to_end.py` as the acceptance test of its likelihood path | deleted 2026-09-27: that path is `core/mechanism_selection.py`, tested in `tests/test_44_mechanism_selection.py` |
| `decoys.py` | a validity check, `E_null[score] <= 1`, kept off the calling path | deleted 2026-09-27: the check runs per class on the shifted-breakpoint decoys (`mechanism_selection.decoy_checks`), and a class that fails it has its e-values divided by the bound |

## What "unused" means in this repository

`tests/test_14_unit_coverage.py` opens by saying its functions are "named,
even though most were exercised through `build_certificate`". Many public
functions have no caller outside their own module. The suite is the
specification, so *reachable only from tests* is the normal state here, not a
smell, and a dead-code sweep driven by import graphs would delete working,
tested code.

The signal that does mean something is a module that is unimported **and**
duplicates live logic, or one whose purpose has been met. The first was true
of `selection.py`. The second is why `integrate.py` and `decoys.py` went: the
decision they prototyped replaced the legacy one, and was validated on HG002
chr2-8 (`docs/development-strategy.md`, section 8).
