"""
The dependency bound: the invariants, and the three regressions.

This is the subsystem where porting from a reading of the code is most likely to
reintroduce a bug, so it gets the most tests.
"""

from __future__ import annotations

import math

import pytest
from conftest import call_or_skip, close

from placer.core import dependency


@pytest.mark.invariant
def test_cap_is_derived_from_m_and_q_not_chosen():
    for m in (10, 1000, 100000):
        for q in (0.01, 0.10, 0.50):
            est = call_or_skip(dependency.estimate_dependency_penalty,
                               [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], q, m)
            close(est.cap_log, math.log(m / q), f"cap for m={m} q={q}")


@pytest.mark.invariant
def test_there_is_no_minimum_null_count():
    """
    The bound's WIDTH encodes "too few controls", so no threshold is needed:
    few rows give a wide bound, a large penalty, and nothing passes.

    THE POOL HAS TO BE HOMOGENEOUS, and this is the third time that detail has
    caught me out -- twice on the C++ side and once here. My first version used
    a monotone ramp `[-1.0 + 0.001*i for i in range(4000)]`, whose 4-element
    prefix holds only the four SMALLEST values. The prefix is then not a sample
    from the same distribution as the whole, its mean is genuinely lower, and
    the comparison measures the ramp rather than the estimator. It failed, as it
    should have.

    A repeating pattern fixes it: every even-length prefix has exactly the same
    empirical distribution, so `n` is the only thing that varies and the slack
    is isolated.
    """
    import math
    pattern = [math.log(1.2), math.log(1.6)]
    pool = pattern * 2000                      # mean e-value 1.4 at any prefix
    starved = call_or_skip(dependency.estimate_dependency_penalty,
                           pool[:4], pool[:4], 0.10, 1000)
    rich = call_or_skip(dependency.estimate_dependency_penalty,
                        pool, pool, 0.10, 1000)
    assert math.isclose(starved.vs_artifact.sigma_mean,
                        rich.vs_artifact.sigma_mean, rel_tol=1e-9), (
        "the fixture is broken if the two prefixes have different means")
    assert starved.vs_artifact.log_penalty > rich.vs_artifact.log_penalty, (
        "fewer controls must give a wider bound and a larger penalty")


@pytest.mark.invariant
def test_the_slack_is_what_shrinks_with_n_not_the_penalty():
    """
    The penalty is NOT monotone in n, and asserting that it were would be
    testing a coincidence. `sigma_mean` converges toward the truth from wherever
    the small-sample mean happened to land, so the total can tick UP slightly
    even as the bound tightens. What is monotone is the SLACK,
    `sigma_upper - sigma_mean`.

    Measured on the C++ with nested prefixes of one lognormal pool: sigma_mean
    rises 1.2404 -> 1.4032 toward the true 1.4 while the slack falls
    3.18 -> 0.008, and the penalty goes 1.487 -> 0.378 -> 0.341 -> 0.345, i.e.
    it RISES at the last step.
    """
    import math
    pattern = [math.log(0.9), math.log(1.3), math.log(2.1)]
    previous_slack = math.inf
    for n in (6, 30, 150, 900, 4500):
        pool = (pattern * (n // 3))[:n]
        est = call_or_skip(dependency.estimate_dependency_penalty,
                           pool, pool, 0.10, 1000)
        slack = est.vs_artifact.sigma_upper - est.vs_artifact.sigma_mean
        assert slack >= 0.0
        assert slack < previous_slack, f"slack did not shrink at n={n}"
        previous_slack = slack


@pytest.mark.invariant
def test_never_takes_credit_for_sigma_below_one():
    """Anti-correlated blocks make the product already conservative; that is not
    converted into a bonus."""
    tiny = [-8.0] * 500
    est = call_or_skip(dependency.estimate_dependency_penalty,
                       tiny, tiny, 0.10, 1000)
    assert est.vs_artifact.sigma_mean < 1.0
    assert est.vs_artifact.sigma_upper >= 1.0
    assert est.vs_artifact.log_penalty == 0.0


@pytest.mark.regression
def test_capping_precedes_the_penalty():
    """
    A VALIDITY bug that was in the C++ and is fixed.

    sigma bounds `E_null[min(Y, C)]`, so only `min(Y, C)/sigma` is guaranteed to
    have null expectation <= 1. `Y/sigma` is not, because `E_null[Y]` can exceed
    `E_null[min(Y, C)]`. Applying the penalty to the UNCAPPED aggregate builds
    e-values that e-BH is entitled to assume and that the estimator never
    certified.

    So for a raw value above the cap, the result must be `cap - penalty`, and
    NOT `raw - penalty`.
    """
    cap_log = math.log(1000 / 0.10)
    penalty = 1.5
    huge = cap_log + 25.0
    got = call_or_skip(dependency.apply_penalty, huge, cap_log, penalty)
    close(got, cap_log - penalty, "capped-then-penalised")
    assert got < huge - penalty, "the cap was applied after the subtraction"
    # Below the cap, capping is inert.
    small = 2.0
    close(call_or_skip(dependency.apply_penalty, small, cap_log, penalty),
          small - penalty, "uncapped path")


@pytest.mark.regression
def test_the_calibration_sample_must_not_be_selected_by_the_aggregate():
    """
    The bug that made this whole subsystem inert, and the reason
    `calibration_rows` is separate from the conformal null-control predicate.

    sigma is a MEAN, dominated by its right tail. The C++ once estimated it on
    rows that were NOT "promotable", and promotability includes `log_e > 0` on
    the very aggregate being calibrated -- so the sample was truncated exactly
    where sigma lives and the estimate collapsed onto its floor of 1.

    Measured here: dropping the top 10% of rows takes the penalty from nats to
    zero. The direction matters -- contamination by true positives inflates the
    mean and stays conservative, while truncation deflates it and breaks the
    guarantee, so nothing is excluded.
    """
    bulk = [-5.5 + 0.0005 * i for i in range(3600)]
    tail = [5.0 + 0.001 * i for i in range(400)]
    full = bulk + tail
    truncated = [v for v in full if v <= 0.0]
    assert len(truncated) < len(full)

    est_full = call_or_skip(dependency.estimate_dependency_penalty,
                            full, full, 0.10, 1000)
    est_trunc = call_or_skip(dependency.estimate_dependency_penalty,
                             truncated, truncated, 0.10, 1000)
    assert est_trunc.vs_artifact.log_penalty == 0.0, (
        "a tail-free sample should bottom out on the floor")
    assert est_full.vs_artifact.log_penalty > 2.0, (
        "with the tail present the penalty must be worth nats")


@pytest.mark.regression
def test_each_side_gets_its_own_bound():
    """What removed the unexplained 0.60 factor that used to relate them."""
    strong = [2.0 + 0.001 * i for i in range(2000)]
    weak = [-3.0 + 0.001 * i for i in range(2000)]
    est = call_or_skip(dependency.estimate_dependency_penalty,
                       strong, weak, 0.10, 1000)
    assert est.vs_artifact.log_penalty > est.vs_non_te.log_penalty
    ratio = est.vs_non_te.log_penalty / max(1e-9, est.vs_artifact.log_penalty)
    assert not math.isclose(ratio, 0.60, abs_tol=0.01)


def test_non_finite_inputs_are_dropped_not_propagated():
    clean = [0.1, 0.2, 0.3, 0.4]
    dirty = clean + [float("inf"), float("-inf"), float("nan")]
    a = call_or_skip(dependency.estimate_dependency_penalty,
                     clean, clean, 0.10, 1000)
    b = call_or_skip(dependency.estimate_dependency_penalty,
                     dirty, dirty, 0.10, 1000)
    assert math.isfinite(b.vs_artifact.sigma_mean)
    assert math.isfinite(b.vs_artifact.log_penalty)
    # -inf maps to a finite e-value of 0, which legitimately lowers the mean.
    assert b.vs_artifact.sigma_mean <= a.vs_artifact.sigma_mean + 1e-9


def _grid() -> dict[str, tuple[list[float], list[float], float, int]]:
    """Deterministic inputs, no RNG: a ramp, the ramp plus a right tail, two
    tiny samples, an empty one, and two sides drawn from different pools."""
    ramp = [-6.0 + 0.01 * i for i in range(500)]
    tailed = ramp + [4.0 + 0.02 * i for i in range(50)]
    return {
        "ramp_500": (ramp, ramp, 0.10, 1000),
        "ramp_with_right_tail": (tailed, tailed, 0.10, 1000),
        "tiny_two": ([0.1, 0.3], [0.1, 0.3], 0.10, 1000),
        "single_row": ([0.5], [0.5], 0.10, 1000),
        "empty": ([], [], 0.10, 1000),
        "asymmetric_sides": (tailed, ramp, 0.05, 250),
    }


@pytest.mark.invariant
@pytest.mark.parametrize("name", sorted(_grid()))
def test_the_bound_is_well_formed(name):
    """Across the grid: the penalty is the log of the bound, the bound is
    floored at 1, it sits above the empirical mean unless it is on a clamp, and
    with fewer than two usable rows the only honest bound is the cap."""
    art, non, q, m = _grid()[name]
    est = call_or_skip(dependency.estimate_dependency_penalty, art, non, q, m)
    close(est.cap_log, math.log(m / q), f"{name}.cap_log")
    assert est.estimated == (len(art) >= 2 and len(non) >= 2), name
    for side, rows in (("vs_artifact", art), ("vs_non_te", non)):
        s = getattr(est, side)
        label = f"{name}.{side}"
        close(s.log_penalty, math.log(s.sigma_upper), label)
        assert s.sigma_upper >= 1.0, label
        assert s.log_penalty >= 0.0, label
        on_clamp = math.isclose(s.sigma_upper, 1.0, rel_tol=1e-12) or \
            math.isclose(s.sigma_upper, math.exp(est.cap_log), rel_tol=1e-9)
        assert s.sigma_upper >= s.sigma_mean or on_clamp, label
        if len(rows) < 2:
            close(s.log_penalty, est.cap_log, f"{label}: falls back to the cap")


@pytest.mark.invariant
def test_the_right_tail_is_what_the_bound_is_made_of():
    """`ramp_500` and `ramp_with_right_tail` share 500 identical rows; the
    latter adds 50 high ones. sigma is a MEAN, so those 50 rows are almost the
    whole quantity."""
    grid = _grid()
    plain = call_or_skip(dependency.estimate_dependency_penalty,
                         *grid["ramp_500"])
    tailed = call_or_skip(dependency.estimate_dependency_penalty,
                          *grid["ramp_with_right_tail"])
    close(plain.vs_artifact.sigma_upper, 1.0,
          "the tail-free sample should bottom out on the floor")
    assert tailed.vs_artifact.log_penalty > plain.vs_artifact.log_penalty + 2.0
