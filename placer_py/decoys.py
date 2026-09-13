"""
Verifying that a score is already an e-value, instead of calibrating it.

THE RESOLUTION OF THE SIGMA DILEMMA, and it is a deletion rather than an
addition.

The dependency bound needs `sigma >= E_null[min(Y, C)]`. Two samples to estimate
that on, and both are wrong:

  1. Rows the score does not like (what the C++ did). Promotability includes
     `log_e > 0` on the very aggregate being calibrated, so the sample is
     truncated exactly where sigma lives -- sigma is a MEAN, dominated by its
     right tail. Measured: sigma collapses from 17.7 to its floor of 1.0.
     Anti-conservative, and the correction is inert.

  2. Every row. Then the true positives, which saturate the cap `C = m/q` under
     any score worth having, set sigma themselves: `sigma ~ pi*C`, the largest
     attainable e-value is `C/sigma = 1/pi`, and e-BH at rank r demanding `C/r`
     needs `r >= pi*m/q` while `r <= pi*m` -- so it needs `q >= 1`. Impossible
     for any usable q, for ANY score, and worse the better the score is. Valid,
     and useless.

Neither is a tuning problem, and removing the cap makes it worse.

WHY THE DILEMMA EXISTS AT ALL
    Because the affine blocks are NOT likelihood ratios. They are bounded
    signals run through invented affine maps, so their null expectation is an
    unknown number that has to be measured -- and it cannot be measured, per the
    above. That is a complete argument against using them in a decision path at
    genome scale, independent of whether their constants are well chosen.

    A genuine likelihood ratio needs no calibration:

        E_null[ p1(X)/p0(X) ] = integral (p1/p0) p0 = 1

    MEASURED on the TPRT terms over 20,000 simulated null loci:
    `E_null[e^score] = 0.186`, comfortably below 1, with only 2.7% of nulls
    scoring above zero at all. With `sigma = 1` and no penalty, e-BH then
    recovers 90-97% of true insertions at FDP 0.000.

    So the whole sigma apparatus was solving a problem the affine blocks create
    and a likelihood ratio does not have.

WHAT REMAINS FOR A NULL SET
    Verification, not estimation. A likelihood approach's real risk is model
    MISSPECIFICATION -- if the local nulls are wrong, `E_null[e^score]` can
    exceed 1 and the guarantee goes with it. So the null set's job is to check
    that inequality on real data and refuse to proceed if it fails.

    That is a much weaker requirement than calibration: an approximate null set
    can falsify `E_null[e^score] <= 1`, whereas estimating sigma from it needs
    the sample to be a faithful draw.

THE CONSTRUCTION, AND ITS LIMITATION -- stated because my first version was wrong
    Pairing an insert with a DIFFERENT locus's flanking context looks like the
    natural decoy, and it is what `permuted_decoys` below does. But it inherits
    the host's linkage observables wholesale, so a decoy built from a true
    insert on a true host looks exactly like a true insertion -- the decoy set
    is contaminated at the same rate pi as the candidate set. Measured: decoy
    sigma came back at 238, penalty 5.47 nats, recall still zero. Useless as an
    estimator, for the same reason as sample 2.

    It remains useful as a FALSIFIER, because contamination can only push
    `E_null[e^score]` up: if the permuted set satisfies the inequality, the true
    null does too.

    The correct null for estimation is a BREAKPOINT SHIFT -- the same locus with
    the breakpoint moved a few hundred bp, so the local composition is preserved
    and the coincidences are whatever chance gives. That cannot be built from
    the scalar ledger summaries, because `tsd` is already the RESULT of the
    coincidence test rather than the sequence it was computed from. It has to be
    generated in the evidence-extraction layer, which is why `shifted_decoys`
    below takes a callback rather than doing it here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

#: Observables that belong to the INSERT and travel with it under permutation.
INTERNAL_FIELDS = ("u", "v", "insert_len", "identity", "polya", "inverted")

#: Observables that belong to the LOCUS.
LINKAGE_FIELDS = ("tsd", "tsd_mismatches", "repeat_frac", "t_tract",
                  "t_frac", "a_frac", "n_alt", "n_ref")


@dataclass
class ValidityCheck:
    """The result of checking `E_null[e^score] <= 1` on a null set.

    `conclusive` records whether a FAILURE can be believed. On an exact null set
    it can. On a contaminated one -- permuted decoys, which inherit their host's
    linkage -- contamination can only push the mean UP, so a PASS is conclusive
    and a FAILURE is not. Treating an inconclusive failure as misspecification
    would block every run, which is what my first version did: permuted decoys
    returned means of 101-173 and nothing was ever selected.
    """

    mean_e_value: float
    n_rows: int
    fraction_above_zero: float
    valid: bool
    message: str
    conclusive: bool = True


def permuted_decoys(rows: Sequence[dict], n_decoys: int | None = None,
                    seed: int = 0) -> list[dict]:
    """
    Pair each insert with another locus's flanking context.

    Null by construction in the sense that the insert did not come from the
    locus -- but see the module docstring: it inherits the host's linkage
    observables, so it is contaminated at the same rate as the candidate set and
    must NOT be used to estimate sigma. Use it only to falsify
    `E_null[e^score] <= 1`, which contamination can only make harder to satisfy.
    """
    n = len(rows)
    if n < 2:
        return []
    target = n if n_decoys is None else max(0, n_decoys)
    rng = random.Random(seed)

    out: list[dict] = []
    for index in range(target):
        donor = rows[index % n]
        offset = rng.randrange(1, n)          # never pair with its own locus
        host = rows[(index + offset) % n]
        decoy = dict(donor)
        for field in LINKAGE_FIELDS:
            if field in host:
                decoy[field] = host[field]
        decoy["locus_id"] = f"decoy_{index}"
        decoy["is_decoy"] = True
        out.append(decoy)
    return out


def shifted_decoys(rows: Sequence[dict],
                   shift_fn: Callable[[dict], dict | None],
                   seed: int = 0) -> list[dict]:
    """
    The correct null: the same locus with the breakpoint moved.

    `shift_fn` takes a row and returns the observables recomputed at a shifted
    position, or None if that is impossible there. It has to live in the
    evidence layer because `tsd` and `t_tract` are RESULTS of coincidence tests
    against the reference sequence, not inputs that can be permuted -- the
    scalar ledger has already thrown away what would be needed.

    Local composition is preserved (same neighbourhood) and the coincidences are
    whatever chance gives, which is exactly the null the model's local
    backgrounds claim to describe. That makes this set usable for estimation as
    well as falsification, and it is the standard shifted-control construction
    from peak calling.
    """
    out: list[dict] = []
    for index, row in enumerate(rows):
        shifted = shift_fn(row)
        if shifted is None:
            continue
        shifted = dict(shifted)
        shifted["locus_id"] = f"shift_{index}"
        shifted["is_decoy"] = True
        out.append(shifted)
    return out


def check_e_value_validity(null_rows: Sequence[dict],
                           score_fn: Callable[[dict], float],
                           cap_log: float | None = None,
                           tolerance: float = 1.0,
                           null_kind: str = "exact") -> ValidityCheck:
    """
    Check `E_null[e^score] <= tolerance` on a set believed to be null.

    This replaces sigma estimation for a genuine likelihood ratio: instead of
    measuring a scale factor that cannot be measured, verify the inequality the
    construction is supposed to guarantee. If it fails, the model is
    misspecified -- most likely one of the local nulls -- and the right response
    is to fix the model, not to paper over it with a penalty.

    The cap matters less here than it did for estimation, but it is applied when
    given so the quantity checked is the same one the e-values are built from.
    """
    values: list[float] = []
    above_zero = 0
    for row in null_rows:
        score = float(score_fn(row))
        if not math.isfinite(score):
            continue
        if score > 0.0:
            above_zero += 1
        if cap_log is not None:
            score = min(score, cap_log)
        values.append(math.exp(score))

    n = len(values)
    if n == 0:
        return ValidityCheck(float("nan"), 0, 0.0, False,
                             "no usable null rows: cannot verify validity, so "
                             "the guarantee is unsupported", conclusive=True)
    mean = sum(values) / n
    valid = mean <= tolerance
    # On a conservative (contaminated) null set, contamination can only raise
    # the mean, so a PASS is conclusive and a FAILURE is not.
    conclusive = valid or null_kind == "exact"

    if valid:
        message = (f"E_null[e^score] = {mean:.4g} <= {tolerance:g} on {n} "
                   f"{null_kind} null rows: the score is already an e-value "
                   f"and needs no calibration")
    elif conclusive:
        message = (f"E_null[e^score] = {mean:.4g} > {tolerance:g} on {n} exact "
                   f"null rows: MISSPECIFIED. The score is not an e-value, so "
                   f"e-BH's guarantee does not hold. Check the local nulls "
                   f"(tandem-duplication background, A/T composition, T-tract "
                   f"background, local error rate) before trusting any output.")
    else:
        message = (f"E_null[e^score] = {mean:.4g} > {tolerance:g} on {n} "
                   f"{null_kind} null rows, which is INCONCLUSIVE: this set is "
                   f"contaminated by construction and contamination can only "
                   f"raise the mean. Build an exact null (a breakpoint shift) "
                   f"to settle it.")
    return ValidityCheck(mean, n, above_zero / n, valid, message,
                         conclusive=conclusive)
