# The four modules nothing imports

`tprt.py`, `integrate.py`, `decoys.py` and `null_control.py` are reachable
only from the test suite. A linter reports them as dead, and a code audit of
this repository did exactly that — it called them a "five-module dead cluster,
~1187 lines" and proposed a unify-or-delete sweep.

That reading is wrong, and the reason it is wrong is not written anywhere in
the code, which is the actual defect. Each module is deliberate, each has a
different reason, and deleting any of them would lose something. This file is
the missing statement of which is which.

The fifth module in that cluster, `selection.py`, *was* a genuine duplicate:
`finalization.apply_event_ebh_selection` reimplemented both of its functions
inline. It is now imported and the duplicate is gone. That is what a real
finding looks like, and it is why the other four were each checked rather
than swept.

---

## `tprt.py` — the intended replacement, not yet connectable

The TPRT coincidence model: real log-likelihood ratios against nulls measured
from the local reference, where `blocks.py` has nineteen hand-set constants
clamped to a range that `blocks.py`'s own docstring shows cannot support
genome-scale FDR control.

**Why it is not wired up.** `tprt.evaluate` needs thirteen observables and six
of them have no producer: `repeat_frac`, `t_frac`, `a_frac`, `t_tract`,
`elem_len`, and a `polya` that distinguishes A from T. A seventh, `inverted`,
is *discarded on purpose* — `te_classifier.parse_blast_hsp_line` takes
`min`/`max` of the subject coordinates and its comment says so. An eighth, the
TSD, is computed after the decision rather than before it.

This is pinned rather than implied: `schema.MISSING_FOR_TPRT` lists the
columns, and `tests/test_00_contract.py` asserts they are absent from the
ledger. The model is the destination; the evidence layer is what has to be
built to reach it.

**Decision: keep.** Connecting it is a separate, staged piece of work — build
the producers as ledger columns first, scoring nothing, then move TSD ahead of
the decision, then run both scorers side by side.

## `integrate.py` — retained for the comparison, and it says so

`select()` calibrates sigma on the candidate set, the way the C++ does and the
way this port did first; it cannot select anything at genome scale.
`select_likelihood()` is the path that works: an uncalibrated log-LR under
e-BH, checked by decoys, and `tests/test_13_end_to_end.py` is its acceptance
test.

**Decision: keep, until the decision layer is rewritten.** The multi-species
refactor puts `select_likelihood`'s construction on the calling path; at that
point `test_13` moves onto the live path and this module goes.

## `decoys.py` — a validity check, not a calibration

Given a set of rows believed to be null, it asks whether a score behaves like
an e-value at all — `E_null[score] <= 1`. It does not tune anything, and it is
deliberately not on the calling path: a validity check that runs inside the
thing it validates is not a check.

`permuted_decoys` and `shifted_decoys` build the null sets, and
`check_e_value_validity` is the test. The module's own conclusion is negative
and worth keeping visible: both candidate samples fail, in opposite
directions, which is the evidence behind `blocks.py` being demoted.

**Decision: keep.** This is the tool that would answer "is the new scorer's
null expectation really 1?" for `tprt.py`, which is the question phase 5
turns on.

## `null_control.py` — the callback `decoys.py` was written to take

`make_breakpoint_shift_controls` produces the same locus with the breakpoint
moved, so local composition is preserved and any apparent signal is whatever
chance gives. `decoys.py` argued for exactly this construction and could not
build it from the scalar ledger, which is why it takes a callback — this is
the callback.

**Decision: keep.** It is half of a validation loop whose other half is
`decoys.py`, and the loop closes as soon as the evidence layer can recompute
observables at a shifted position — the same capability `tprt.py` needs.

---

## What "unused" means in this repository

`tests/test_14_unit_coverage.py` opens by saying its functions are "named,
even though most were exercised through `build_certificate`". 247 of 578
public functions have no caller outside their own module. The suite is the
specification, so *reachable only from tests* is the normal state here, not a
smell — and a dead-code sweep driven by import graphs would delete a large
part of a working, tested codebase.

The signal that does mean something is a module that is unimported **and**
duplicates live logic. That was true of exactly one of the five.
