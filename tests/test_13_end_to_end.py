"""
The migration's acceptance test: a working caller, end to end.

Everything else in this suite constrains a piece. This file checks the thing the
pieces were for -- that a whole-genome-shaped run selects most of the true
insertions, controls FDR, and says so honestly when it cannot.

It also pins the resolution of the sigma dilemma, which was the blocker and
turned out to be a DELETION rather than an addition:

  * Estimating sigma on rows the score dislikes truncates its right tail and
    collapses it onto 1. Anti-conservative.
  * Estimating it on every row lets the true positives, which saturate the cap,
    set it themselves -- headroom `1/pi`, and e-BH needs `q >= 1`. Valid and
    useless.
  * A genuine likelihood ratio needs neither, because `E_null[p1/p0] = 1` holds
    by construction. Measured here at 0.0016-0.0023 on shifted nulls.

So sigma is 1, and the null set VERIFIES that rather than setting it.
"""

from __future__ import annotations

import math
import random

import pytest

from conftest import call_or_skip
from placer_py import decoys, integrate, tprt

pytestmark = pytest.mark.invariant

Q = 0.10
L1_LEN = 6000.0


def _locus(rng: random.Random, is_true: bool) -> dict:
    length = rng.randint(600, 3000)
    if is_true:
        return dict(u=int(L1_LEN - length), v=int(L1_LEN), insert_len=length,
                    tsd=rng.randint(9, 20), tsd_mismatches=1,
                    repeat_frac=rng.uniform(0.0, 0.10),
                    t_tract=rng.randint(6, 12), polya=rng.randint(14, 30),
                    t_frac=rng.uniform(0.25, 0.38),
                    a_frac=rng.uniform(0.25, 0.38),
                    identity=rng.uniform(0.94, 0.99),
                    n_alt=rng.randint(5, 14), n_ref=rng.randint(2, 8))
    return dict(u=int(L1_LEN - length), v=int(L1_LEN), insert_len=length,
                tsd=0, tsd_mismatches=0, repeat_frac=rng.uniform(0.0, 0.30),
                t_tract=rng.randint(0, 3), polya=rng.randint(10, 25),
                t_frac=rng.uniform(0.25, 0.38), a_frac=rng.uniform(0.25, 0.38),
                identity=rng.uniform(0.87, 0.96),
                n_alt=rng.randint(2, 7), n_ref=rng.randint(5, 14))


def _score(locus: dict) -> float:
    bf = tprt.evaluate(locus, L1_LEN, "L1")
    return min(bf.log_bf_vs_non_te, bf.log_bf_vs_artifact)


def _shift(row: dict) -> dict:
    """
    An EXACT null: the same neighbourhood with the breakpoint moved, so the
    local composition is preserved and the coincidences are whatever chance
    gives. The standard shifted control from peak calling.

    In a real pipeline this has to be produced by the evidence layer, because
    `tsd` and `t_tract` are RESULTS of coincidence tests against the reference
    rather than inputs that can be permuted -- the scalar ledger has already
    discarded what would be needed. Here it is simulated directly.
    """
    rng = random.Random(hash(row.get("locus_id", "")) & 0xFFFF)
    out = dict(row)
    out.update(tsd=0, tsd_mismatches=0, t_tract=rng.randint(0, 3),
               n_alt=rng.randint(0, 3), n_ref=rng.randint(6, 16))
    return out


def _population(n_null: int, n_true: int, seed: int):
    rng = random.Random(seed)
    rows, truth = [], {}
    for i in range(n_null):
        row = _locus(rng, False)
        row["locus_id"] = f"null_{i}"
        rows.append(row)
        truth[row["locus_id"]] = False
    for i in range(n_true):
        row = _locus(rng, True)
        row["locus_id"] = f"true_{i}"
        rows.append(row)
        truth[row["locus_id"]] = True
    rng.shuffle(rows)
    return rows, truth


# ===========================================================================
def test_the_likelihood_ratio_is_already_an_e_value():
    """
    `E_null[e^score] <= 1` by construction, and measured.

    This is what makes the whole sigma apparatus unnecessary, so it is the
    single load-bearing fact of the design. Over 20,000 simulated nulls the
    measured value is about 0.19, with only ~3% of nulls scoring above zero.
    """
    rng = random.Random(9)
    nulls = [_locus(rng, False) for _ in range(20000)]
    for row in nulls:
        row["locus_id"] = "n"
    check = call_or_skip(decoys.check_e_value_validity, nulls, _score,
                         cap_log=math.log(20000 / Q))
    assert check.valid, check.message
    assert check.mean_e_value < 1.0
    assert check.fraction_above_zero < 0.10, (
        "most nulls should score at or below zero outright")


def test_a_whole_run_recovers_most_true_insertions_at_the_target_fdr():
    """THE acceptance test."""
    recalls, fdps = [], []
    total_selected = total_false = 0
    for seed in range(8):
        rows, truth = _population(n_null=1000, n_true=60, seed=900 + seed)
        nulls = call_or_skip(decoys.shifted_decoys, rows, _shift)
        result = call_or_skip(integrate.select_likelihood, rows, _score, Q,
                              null_rows=nulls, null_kind="exact")
        assert result.validity_ok, result.validity_message
        true_positives = sum(1 for locus in result.selected if truth[locus])
        false_positives = len(result.selected) - true_positives
        total_selected += len(result.selected)
        total_false += false_positives
        recalls.append(true_positives / 60.0)
        fdps.append(false_positives / len(result.selected)
                    if result.selected else 0.0)

    mean_recall = sum(recalls) / len(recalls)
    pooled_fdp = total_false / max(1, total_selected)
    assert mean_recall > 0.85, f"mean recall {mean_recall:.3f}"
    assert pooled_fdp <= Q, (
        f"pooled false discovery proportion {pooled_fdp:.4f} against q={Q}")


def test_sigma_is_one_and_no_penalty_is_charged():
    rows, _ = _population(n_null=500, n_true=30, seed=1)
    nulls = call_or_skip(decoys.shifted_decoys, rows, _shift)
    result = call_or_skip(integrate.select_likelihood, rows, _score, Q,
                          null_rows=nulls, null_kind="exact")
    assert result.sigma_upper == 1.0
    assert result.log_penalty == 0.0


def test_calibrating_on_the_candidates_instead_selects_nothing():
    """
    The contrast that justifies the design, run side by side.

    Same rows, same score, same q -- the only difference is whether sigma is
    estimated on the candidate set or taken as 1 from the construction.
    """
    rows, truth = _population(n_null=1000, n_true=60, seed=900)
    nulls = call_or_skip(decoys.shifted_decoys, rows, _shift)

    working = call_or_skip(integrate.select_likelihood, rows, _score, Q,
                           null_rows=nulls, null_kind="exact")
    calibrated = call_or_skip(
        integrate.select,
        call_or_skip(integrate.candidates_from_mechanistic, rows, _score), Q)

    assert len(working.selected) > 40
    assert len(calibrated.selected) == 0
    assert calibrated.log_penalty > 4.0, (
        "the candidate-set penalty should be several nats, driven by the true "
        "positives saturating the cap")


def test_a_misspecified_model_is_refused_rather_than_reported():
    """
    The failure mode a likelihood approach really has is misspecification, so
    the null set's job is to catch it. A score whose null expectation exceeds 1
    must block selection and say why, rather than quietly reporting calls whose
    guarantee does not hold.
    """
    rows, _ = _population(n_null=400, n_true=20, seed=2)
    nulls = call_or_skip(decoys.shifted_decoys, rows, _shift)

    def inflated(locus: dict) -> float:
        return _score(locus) + 30.0      # an e-value 1e13 times too large

    result = call_or_skip(integrate.select_likelihood, rows, inflated, Q,
                          null_rows=nulls, null_kind="exact")
    assert not result.validity_ok
    assert result.selected == []
    assert "MISSPECIFIED" in result.validity_message


def test_an_inconclusive_null_set_does_not_block():
    """
    Permuted decoys inherit their host's linkage, so they are contaminated at
    the same rate as the candidate set and their FAILURE says nothing --
    contamination can only push the mean up. Blocking on it stopped every run in
    my first version (means of 101-173, nothing ever selected).
    """
    rows, truth = _population(n_null=600, n_true=40, seed=3)
    permuted = call_or_skip(decoys.permuted_decoys, rows, seed=0)
    result = call_or_skip(integrate.select_likelihood, rows, _score, Q,
                          null_rows=permuted, null_kind="permuted")
    assert not result.validity_ok, "permuted decoys are expected to fail"
    assert "INCONCLUSIVE" in result.validity_message
    assert len(result.selected) > 0, (
        "an inconclusive failure must not block selection")


def test_no_null_set_is_reported_as_unverified():
    rows, _ = _population(n_null=200, n_true=10, seed=4)
    result = call_or_skip(integrate.select_likelihood, rows, _score, Q,
                          null_rows=None)
    assert not result.validity_ok
    assert "NOT verified" in result.validity_message
    assert len(result.selected) > 0, (
        "absent verification the run proceeds, but the caller is told it is "
        "unverified rather than being given false assurance")


def test_a_pure_null_population_selects_essentially_nothing():
    rng = random.Random(7)
    rows = []
    for i in range(2000):
        row = _locus(rng, False)
        row["locus_id"] = f"null_{i}"
        rows.append(row)
    nulls = call_or_skip(decoys.shifted_decoys, rows, _shift)
    result = call_or_skip(integrate.select_likelihood, rows, _score, Q,
                          null_rows=nulls, null_kind="exact")
    assert len(result.selected) <= Q * len(rows)
