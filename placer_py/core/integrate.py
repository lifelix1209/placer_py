"""
Join the two layers: the existing mechanistic score, and FDR control.

This is the first time the Python side has any FDR control at all. The existing
`placer_py/redesign/model/mechanistic.py` produces a per-candidate `loglr` and a hallmark tier, and
then the decision is a threshold on those. That gives no guarantee over a
whole-genome run: several tens of thousands of candidates, each judged locally,
and nothing bounding how many of the reported ones are wrong.

THE ARGUMENT THAT MAKES THIS WORK, and it is worth stating carefully because it
is what licenses plugging a hand-tuned score into a procedure with a theorem
attached.

An e-value is any non-negative random variable with `E_null[E] <= 1`. That is
the whole definition. In particular an e-value does NOT have to be a correct
likelihood ratio: for any fixed non-negative measurable `f`,

    E = f(X) / E_null[f(X)]      =>      E_null[E] = 1

So `f` can be a hand-weighted sum with nineteen invented constants. It only has
to be FIXED IN ADVANCE, and the same `f` has to be evaluated on the null rows.
If the constants are badly chosen, the null rows' `f` values rise too, sigma
rises with them, and the penalty absorbs it. Self-calibrating.

What the constants DO control is `f`'s ability to separate -- i.e. recall. A
poor `f` gives real insertions and null rows similar values, sigma climbs, and
the real insertions stop passing too. So:

    in this architecture the hand-set constants can only cost RECALL.
    They cannot inflate the FDR.

`tests/test_11_integration.py::test_bad_mechanistic_constants_cannot_break_fdr`
demonstrates exactly that, by deliberately scrambling the weights and checking
the guarantee survives.

THE ONE THING THAT CAN BREAK IT is the sigma estimate itself, which is why
`dependency.calibration_rows` excludes nothing. sigma is a MEAN dominated by its
right tail; select its sample using the score being calibrated and the tail
goes, along with the whole correction.

NOT ON THE CALLING PATH, and that is deliberate rather than neglect --
see `docs/off-pipeline-modules.md` for which of the four unimported
modules this is and why deleting it would lose something.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

from . import dependency, selection


@dataclass
class Candidate:
    """One locus, reduced to what selection needs.

    `log_score` is whatever the mechanistic layer produced -- currently
    `placer_py.redesign.model.mechanistic.mechanistic_te_score(...).loglr`. Nothing here assumes it
    is calibrated, or a likelihood ratio, or even monotone in anything. It only
    has to be a fixed function of the data.
    """

    locus_id: str
    log_score: float
    is_null_control: bool = False
    ambiguity_width: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class SelectionResult:
    selected: list[str]
    e_values: dict[str, float]
    log_penalty: float
    sigma_upper: float
    cap_log: float
    n_calibration_rows: int
    q: float


def build_e_values(candidates: Sequence[Candidate], q: float,
                   log_penalty: float, cap_log: float) -> dict[str, float]:
    """
    `E = exp(min(log_score, cap_log) - log_penalty - ambiguity_width)`.

    The cap goes on BEFORE the penalty is subtracted. sigma bounds
    `E_null[min(Y, C)]`, so only `min(Y, C)/sigma` has a certified null
    expectation; applying the penalty to an uncapped score builds something
    e-BH is entitled to assume and the estimator never certified. This was a
    real bug in the C++.

    Subtracting a non-negative ambiguity width multiplies by `e^-w <= 1`, which
    preserves validity -- it can only make an e-value smaller.

    Scores that produce an e-value at or below 0 are floored to 0.0 rather than
    dropped, so they stay in `m`. Removing them would make the hypothesis count
    data-dependent, which is precisely what voids the e-BH guarantee.
    """
    out: dict[str, float] = {}
    for candidate in candidates:
        log_e = dependency.apply_penalty(candidate.log_score, cap_log,
                                         log_penalty)
        log_e -= max(0.0, candidate.ambiguity_width)
        if not math.isfinite(log_e) or log_e <= 0.0:
            out[candidate.locus_id] = 0.0
            continue
        # Overflow guard only: log_e <= cap_log = log(m/q) already, and that
        # reaches 40 only past m/q ~ 2.4e17.
        out[candidate.locus_id] = math.exp(min(log_e, 40.0))
    return out


@dataclass
class CalibratedSelectionResult(SelectionResult):
    """`SelectionResult` plus the validity check that replaced calibration."""

    validity_message: str = ""
    validity_mean: float = float("nan")
    validity_ok: bool = False


def select_likelihood(rows: Sequence[dict], score_fn: Callable[[dict], float],
                      q: float = 0.10,
                      null_rows: Sequence[dict] | None = None,
                      null_kind: str = "exact",
                      require_valid: bool = True) -> CalibratedSelectionResult:
    """
    The selection pass that works, and it works by NOT calibrating.

    `select()` estimates sigma on the candidates. That is valid and unusable:
    the true positives saturate the cap and set sigma themselves, the headroom
    collapses to `1/pi`, and e-BH's requirement cannot be met for any q < 1 --
    for any score, and worse the better the score is. See `placer_py/core/decoys.py`
    for the algebra and `tests/test_12_head_to_head.py` for the measurement.

    A genuine likelihood ratio does not need the estimate at all, because
    `E_null[p1/p0] = 1` holds by construction. Measured on the TPRT terms over
    20,000 simulated null loci: `E_null[e^score] = 0.186`. With no penalty,
    e-BH then recovers 90-97% of true insertions at FDP 0.000.

    So sigma is 1 here, and `null_rows` is used to VERIFY that rather than to
    set it. Verification is a far weaker requirement than calibration: an
    approximate null set can still falsify `E_null[e^score] <= 1`, whereas
    estimating sigma from it would need a faithful draw.

    With `require_valid` and a null set that fails the check, nothing is
    selected and the message says why. That is the correct response to a
    misspecified model -- the guarantee simply does not hold, and a penalty
    would be papering over it.
    """
    from . import decoys as decoy_module

    if not rows:
        return CalibratedSelectionResult([], {}, 0.0, 1.0, 0.0, 0, q)

    candidates = candidates_from_mechanistic(rows, score_fn)
    m = len(candidates)
    cap_log = math.log(m / max(q, 1e-12))

    check = None
    if null_rows:
        check = decoy_module.check_e_value_validity(null_rows, score_fn,
                                                    cap_log=cap_log,
                                                    null_kind=null_kind)

    # sigma = 1 by construction: a likelihood ratio is already an e-value.
    e_values = build_e_values(candidates, q, 0.0, cap_log)
    ordered_ids = [c.locus_id for c in candidates]

    # Only a CONCLUSIVE failure blocks. A permuted null set is contaminated by
    # construction, so its failure says nothing -- blocking on it stopped every
    # run in my first version.
    blocked = bool(check is not None and require_valid
                   and not check.valid and check.conclusive)
    chosen: list[int] = []
    if not blocked:
        chosen = selection.ebh_select([e_values[i] for i in ordered_ids], q)

    return CalibratedSelectionResult(
        selected=[ordered_ids[i] for i in chosen],
        e_values=e_values,
        log_penalty=0.0,
        sigma_upper=1.0,
        cap_log=cap_log,
        n_calibration_rows=check.n_rows if check else 0,
        q=q,
        validity_message=(check.message if check else
                          "no null set supplied: sigma = 1 is asserted from the "
                          "construction and NOT verified on this data"),
        validity_mean=check.mean_e_value if check else float("nan"),
        validity_ok=bool(check.valid) if check else False,
    )


def select(candidates: Sequence[Candidate], q: float = 0.10) -> SelectionResult:
    """
    Calibrated selection: sigma estimated on the candidate set.

    RETAINED FOR THE RECORD, not for use. This is what the C++ does and what
    this port did first, and it cannot select anything at genome scale -- see
    `select_likelihood` and `placer_py/core/decoys.py`. Kept because the golden
    vectors pin it and because the comparison in
    `tests/test_12_head_to_head.py` needs both paths.

    1. Calibrate sigma on EVERY candidate. Not on a subset chosen by the score:
       see `dependency.calibration_rows`. Contamination by true positives
       inflates the mean and stays conservative; truncation by the score
       deflates it and breaks the guarantee.
    2. Cap, then subtract the penalty, to build e-values.
    3. e-BH at `q`, with `m` the total candidate count.

    Only one side is estimated here, because the mechanistic layer produces a
    single score rather than the C++'s two directional aggregates. When the
    TPRT model lands it produces two again -- `log_bf_vs_non_te` and
    `log_bf_vs_artifact` -- and then each needs its own sigma and the decision
    takes the minimum, as the C++ does.
    """
    if not candidates:
        return SelectionResult([], {}, 0.0, 1.0, 0.0, 0, q)

    scores = [c.log_score for c in candidates]
    estimate = dependency.estimate_dependency_penalty(
        scores, scores, q, len(candidates))
    penalty = estimate.vs_artifact.log_penalty

    e_values = build_e_values(candidates, q, penalty, estimate.cap_log)
    ordered_ids = [c.locus_id for c in candidates]
    chosen = selection.ebh_select([e_values[i] for i in ordered_ids], q)

    return SelectionResult(
        selected=[ordered_ids[i] for i in chosen],
        e_values=e_values,
        log_penalty=penalty,
        sigma_upper=estimate.vs_artifact.sigma_upper,
        cap_log=estimate.cap_log,
        n_calibration_rows=len(candidates),
        q=q,
    )


def candidates_from_mechanistic(rows: Sequence[dict],
                                score_fn: Callable[[dict], float]
                                ) -> list[Candidate]:
    """
    Adapter from evidence rows to `Candidate`, so the selection layer never
    imports the mechanistic layer and the two stay independently testable.

    `score_fn` takes a row and returns a log score -- in practice a closure over
    `placer_py.redesign.model.mechanistic.mechanistic_te_score`. Keeping it injected is what lets
    the integration tests scramble the weights deliberately and check that the
    FDR guarantee survives it.
    """
    out: list[Candidate] = []
    for index, row in enumerate(rows):
        out.append(Candidate(
            locus_id=str(row.get("locus_id", index)),
            log_score=float(score_fn(row)),
            ambiguity_width=float(row.get("ambiguity_width", 0.0) or 0.0),
            extra=row,
        ))
    return out
