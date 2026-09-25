"""
The numeric primitives, defined once.

WHY THIS MODULE EXISTS. Nine modules had each grown their own clamp, five had
their own log-sum-exp, and three had their own binomial and beta-binomial. That
is what a port does -- the C++ redefines these per translation unit, and
`finalization.py` says so out loud -- but the copies had drifted, and the drift
was in the answers rather than in the style:

  * clamp(NaN) returned 1.0 in `blocks.py`, 0.0 in `finalization.py` and NaN in
    `te_classifier.py`. In code whose job is probability arithmetic, one
    spelling turns a broken number into certainty and another into impossibility.
  * `log_choose_count(1, 5)` -- an impossible count -- returned `-inf` in
    `finalization.py` and `-1e300` in `genotype.py` and `policy.py`.
    `math.isfinite` separates those, and `log_sum_exp_pair` below uses exactly
    that test to decide whether a line of evidence said anything at all.

THE TWO LOG-SUM-EXP SEMANTICS ARE BOTH KEPT, because both are wanted and the
difference is deliberate. Dropping non-finite operands means "-inf is a line of
evidence that abstained"; propagating them means "-inf is a hypothesis that is
impossible". Making them one function with a flag is what stops a caller
getting the other one by accident.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from math import lgamma

#: Returned for the log-probability of an impossible outcome.
#:
#: `-inf` rather than a large negative number, because the two are not
#: interchangeable: `math.isfinite` is the test `log_sum_exp` uses to tell an
#: abstaining line of evidence from a merely unlikely one, and `-1e300` reads
#: as the second while meaning the first. Arithmetic on `-1e300` also reaches
#: NaN in ways `-inf` does not (`-1e300 * 0` is `-0.0`; `-inf * 0` is NaN --
#: the NaN is the honest answer, and hiding it was never the intent).
LOG_ZERO = -math.inf


def clamp(value: float, lo: float, hi: float) -> float:
    """Confine `value` to `[lo, hi]`, PROPAGATING NaN.

    The NaN policy is the whole reason this is a shared function. `max(lo,
    min(hi, v))` silently yields `hi` for NaN and `min(hi, max(lo, v))` yields
    `lo`, so the same broken input became certainty in one module and
    impossibility in another -- and neither was visible at the call site.

    Propagating is the only one of the three that does not invent an answer.
    A NaN that reaches a clamp is a defect upstream, and a probability of
    "nan" surfaces it at the first comparison rather than at the conclusion.
    """
    if value != value:          # NaN, without importing anything to say so
        return value
    if value < lo:
        return lo
    return hi if value > hi else value


def clamp01(value: float) -> float:
    """`clamp(value, 0.0, 1.0)` -- the probability case, which is most of them."""
    return clamp(value, 0.0, 1.0)


def log_sum_exp(values: Iterable[float], *, ignore_nonfinite: bool = False) -> float:
    """`log(sum(exp(v)))`, computed without overflowing.

    `ignore_nonfinite=True` drops `-inf` and NaN operands instead of letting
    them decide the result. That is not a numerical convenience -- it encodes
    "this line of evidence said nothing", which is a different statement from
    "this hypothesis is impossible", and the finalization stage needs the
    first while the policy stage needs the second. Pass it explicitly so
    which one is meant is visible where it is used.
    """
    items = [v for v in values if not ignore_nonfinite or math.isfinite(v)]
    if not items:
        return LOG_ZERO
    if len(items) == 1:
        return items[0]
    peak = max(items)
    if peak == -math.inf:
        return LOG_ZERO
    if not math.isfinite(peak):
        return peak             # +inf, or NaN, either of which must survive
    total = sum(math.exp(v - peak) for v in items)
    return peak + math.log(total) if total > 0.0 else LOG_ZERO


def log_choose(n: int, k: int) -> float:
    """`log(C(n, k))`, or `LOG_ZERO` when the count is impossible.

    Via lgamma rather than `log(math.comb(n, k))`. The two differ in the last
    bits, and this is the spelling the three copies used, so keeping it means
    the unification changes no output.
    """
    if k < 0 or n < 0 or k > n:
        return LOG_ZERO
    return lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)


def binomial_log_pmf(alt: int, total: int, p: float) -> float:
    """Log-pmf of `alt` successes in `total` trials at rate `p`.

    `p` is confined to `[1e-6, 1 - 1e-6]`, which is the bound the copies used
    -- it is a guard against `log(0)`, not a prior, and tightening it would
    change every count term.
    """
    if alt < 0 or total < 0 or alt > total:
        return LOG_ZERO
    rate = clamp(p, 1e-6, 1.0 - 1e-6)
    return (log_choose(total, alt)
            + alt * math.log(rate)
            + (total - alt) * math.log1p(-rate))


def beta_binomial_log_pmf(alt: int, total: int, alpha: float, beta: float) -> float:
    """Log-pmf of a beta-binomial, parameterised by alpha and beta.

    THE SIGNATURE IS `(alt, total, alpha, beta)`, NOT `(k, n, mu, rho)`, and
    `genotype.py` explains why: the C++ converts mu/rho to alpha/beta inside
    `genotype_log_likelihood`, so keeping the split keeps the same two
    functions it has. The mu/rho form reads
    more naturally and is the wrong port boundary.
    """
    if alt < 0 or total < 0 or alt > total or alpha <= 0.0 or beta <= 0.0:
        return LOG_ZERO
    return (log_choose(total, alt)
            + lgamma(alt + alpha) + lgamma(total - alt + beta)
            - lgamma(total + alpha + beta)
            + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta))
