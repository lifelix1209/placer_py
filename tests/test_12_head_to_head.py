"""
The clamped mechanistic score against the TPRT terms, on identical loci.

The claim being tested is a DYNAMIC RANGE claim, so it has to be measured rather
than asserted: the two scores go through the same sigma calibration, the same
cap, the same e-BH at the same q, on the same simulated population, and the only
thing that differs is the statistic.

FAIRNESS OF THE MAPPING matters here, so it is worth being explicit. Each
simulated locus is generated once, as a set of physical facts -- where the
insert lands in element coordinates, how long the TSD is, how repetitive the
flank is, how many bases of T-tract and poly(A) there are, the read counts --
and then RENDERED into each model's input vocabulary. Nothing is given to one
model and withheld from the other. In particular `tsd_bg_p`, which the clamped
model takes as an argument, is computed from the same local-null function the
TPRT term uses, so the comparison is not "measured background versus assumed
background" but purely "clamped versus unclamped".
"""

from __future__ import annotations

import math
import random

import pytest

from conftest import call_or_skip
from placer_py import integrate, tprt
from placer_py.redesign.model import mechanistic as M

pytestmark = pytest.mark.invariant

Q = 0.10
L1_LEN = 6000.0


# ===========================================================================
# One locus, two renderings
# ===========================================================================
def _make_locus(rng: random.Random, is_true: bool) -> dict:
    """Physical facts about a locus, before either model sees it."""
    if is_true:
        # A real 5'-truncated L1: 3' end complete, TSD present, T-tract and
        # poly(A) annealed, few reference-spanning reads.
        length = rng.randint(600, 3000)
        return dict(
            is_true=True,
            u=int(L1_LEN - length), v=int(L1_LEN),
            insert_len=length,
            tsd=rng.randint(9, 20),
            tsd_mismatches=1,
            repeat_frac=rng.uniform(0.0, 0.10),
            t_tract=rng.randint(6, 12),
            polya=rng.randint(14, 30),
            t_frac=rng.uniform(0.25, 0.38),
            a_frac=rng.uniform(0.25, 0.38),
            identity=rng.uniform(0.94, 0.99),
            n_alt=rng.randint(5, 14),
            n_ref=rng.randint(2, 8),
        )
    # A mismapped OLD reference copy: identical internal structure -- complete
    # 3' end, real poly(A), real TE sequence -- but no linkage to this locus.
    length = rng.randint(600, 3000)
    return dict(
        is_true=False,
        u=int(L1_LEN - length), v=int(L1_LEN),
        insert_len=length,
        tsd=0,
        tsd_mismatches=0,
        repeat_frac=rng.uniform(0.0, 0.30),
        t_tract=rng.randint(0, 3),
        polya=rng.randint(10, 25),
        t_frac=rng.uniform(0.25, 0.38),
        a_frac=rng.uniform(0.25, 0.38),
        identity=rng.uniform(0.87, 0.96),
        n_alt=rng.randint(2, 7),
        n_ref=rng.randint(5, 14),
    )


def _as_mechanistic_kwargs(locus: dict) -> dict:
    """
    Render the locus into `placer_py.redesign.model.mechanistic`'s vocabulary.

    `endonuclease` is the PWM log-odds the T-tract length implies: the clamped
    model takes a motif score where the TPRT term takes matching-base counts,
    so map through the same per-base log-odds the PWM uses (log(0.85/0.25) per
    match, log(0.05/0.25) per mismatch over six positions). A 6-base T-tract is
    a perfect site; fewer bases means proportionally more mismatches.

    `tsd_bg_p` comes from the SAME local-null function the TPRT term uses, so
    neither model gets a better background than the other.
    """
    per_match = math.log(0.85 / 0.25)
    per_mismatch = math.log(0.05 / 0.25)
    matches = min(6, max(0, locus["t_tract"]))
    endonuclease = matches * per_match + (6 - matches) * per_mismatch

    bg_p = tprt.p_null_tandem_duplication(locus["tsd"],
                                          locus["tsd_mismatches"],
                                          locus["repeat_frac"])
    return dict(
        endonuclease=endonuclease,
        endonuclease_evaluated=True,
        tsd_len=locus["tsd"],
        tsd_significant=locus["tsd"] > 0 and bg_p < 1e-3,
        tsd_bg_p=max(bg_p, 1e-12),
        polya_fraction=locus["polya"] / max(1, locus["insert_len"]),
        insert_len=locus["insert_len"],
        te_core_fraction=0.97,
        identity=locus["identity"],
        core_len=float(locus["insert_len"]),
    )


def _mechanistic_score(locus: dict) -> float:
    return M.mechanistic_te_score(**_as_mechanistic_kwargs(locus)).loglr


def _tprt_score(locus: dict) -> float:
    """
    The decision statistic is the MINIMUM of the two Bayes factors -- a
    candidate must beat both alternatives -- which is what feeds selection.
    """
    bf = tprt.evaluate(locus, L1_LEN, "L1")
    return min(bf.log_bf_vs_non_te, bf.log_bf_vs_artifact)


def _population(n_null: int, n_true: int, seed: int):
    rng = random.Random(seed)
    rows, truth = [], {}
    for i in range(n_null):
        locus = _make_locus(rng, is_true=False)
        locus["locus_id"] = f"null_{i}"
        rows.append(locus)
        truth[locus["locus_id"]] = False
    for i in range(n_true):
        locus = _make_locus(rng, is_true=True)
        locus["locus_id"] = f"true_{i}"
        rows.append(locus)
        truth[locus["locus_id"]] = True
    rng.shuffle(rows)
    return rows, truth


def _run(rows, truth, score_fn):
    candidates = call_or_skip(integrate.candidates_from_mechanistic,
                              rows, score_fn)
    result = call_or_skip(integrate.select, candidates, Q)
    true_positives = sum(1 for locus in result.selected if truth[locus])
    false_positives = len(result.selected) - true_positives
    n_true = sum(1 for v in truth.values() if v)
    return dict(
        selected=len(result.selected),
        recall=true_positives / max(1, n_true),
        fdp=(false_positives / len(result.selected)) if result.selected else 0.0,
        penalty=result.log_penalty,
        sigma=result.sigma_upper,
        cap=result.cap_log,
    )


# ===========================================================================
def test_the_mapping_gives_neither_model_an_advantage():
    """Guard the fairness of the comparison before trusting its result."""
    rng = random.Random(0)
    locus = _make_locus(rng, is_true=True)
    kwargs = _as_mechanistic_kwargs(locus)
    # Same background probability from the same function.
    assert kwargs["tsd_bg_p"] == pytest.approx(
        tprt.p_null_tandem_duplication(locus["tsd"], locus["tsd_mismatches"],
                                       locus["repeat_frac"]), rel=1e-12)
    # Same TSD length, same identity, same insert length.
    assert kwargs["tsd_len"] == locus["tsd"]
    assert kwargs["identity"] == locus["identity"]
    assert kwargs["insert_len"] == locus["insert_len"]
    # A 6-base T-tract maps to the perfect-motif score.
    perfect = _as_mechanistic_kwargs({**locus, "t_tract": 6})
    assert perfect["endonuclease"] == pytest.approx(7.343, abs=0.01)


def test_the_clamped_score_is_bounded_below_the_rank_one_threshold():
    """
    The mechanism of the failure, stated as an inequality rather than an
    outcome.

    Each block is clamped -- endonuclease `clamp(en,0,6)*0.45 <= 2.7`,
    `tsd_loglr` to [-1,3], poly(A) to about [-0.3,1.3], TE body ~2.9 -- so the
    total cannot exceed about 9.9 nats. e-BH's rank-1 threshold is
    `log(m/q)`, which at m=1060, q=0.10 is 9.27 nats BEFORE any dependency
    penalty. There is almost no headroom, and the penalty removes what is left.
    """
    best_possible = M.mechanistic_te_score(
        endonuclease=7.343, endonuclease_evaluated=True, tsd_len=15,
        tsd_significant=True, tsd_bg_p=1e-12, polya_fraction=0.10,
        insert_len=3000, te_core_fraction=1.0, identity=0.999,
        core_len=3000.0).loglr
    assert best_possible < 11.0, (
        f"clamped ceiling measured at {best_possible:.2f} nats")
    rank_one = math.log(1060 / Q)
    assert best_possible - rank_one < 1.5, (
        f"ceiling {best_possible:.2f} vs rank-1 threshold {rank_one:.2f}: "
        "essentially no headroom")


def test_the_tprt_score_clears_the_threshold_with_room():
    """A single modal true insertion, scored by the TPRT terms."""
    rng = random.Random(1)
    locus = _make_locus(rng, is_true=True)
    score = _tprt_score(locus)
    rank_one = math.log(1060 / Q)
    assert score > rank_one + 5.0, (
        f"TPRT score {score:.2f} nats against a rank-1 threshold of "
        f"{rank_one:.2f}")


@pytest.mark.xfail(strict=True, reason=(
    "NEITHER score selects anything, and the reason is STRUCTURAL rather than "
    "a property of either statistic -- see "
    "test_calibrating_sigma_on_every_row_makes_selection_impossible for the "
    "algebra. Calibrating sigma on every row lets the true positives, which "
    "saturate the cap, set sigma themselves; the headroom collapses to 1/pi "
    "and e-BH's requirement cannot be met for any q < 1. So this comparison "
    "cannot be run until the sigma estimator is fixed, and the dynamic-range "
    "question it was written to answer is currently unanswerable. The other "
    "tests in this file still establish the range claim directly, on the "
    "scores themselves."))
def test_tprt_recovers_recall_that_the_clamped_score_cannot():
    """
    THE measurement. Same loci, same sigma calibration, same cap, same e-BH.

    If this fails in the direction of the clamped score, the dynamic-range
    argument is wrong and the TPRT work is not justified on these grounds.
    """
    clamped_recalls, tprt_recalls = [], []
    clamped_fdps, tprt_fdps = [], []
    for seed in range(8):
        rows, truth = _population(n_null=1000, n_true=60, seed=500 + seed)
        clamped = _run(rows, truth, _mechanistic_score)
        via_tprt = _run(rows, truth, _tprt_score)
        clamped_recalls.append(clamped["recall"])
        tprt_recalls.append(via_tprt["recall"])
        clamped_fdps.append(clamped["fdp"])
        tprt_fdps.append(via_tprt["fdp"])

    mean_clamped = sum(clamped_recalls) / len(clamped_recalls)
    mean_tprt = sum(tprt_recalls) / len(tprt_recalls)
    assert mean_tprt > mean_clamped, (
        f"TPRT recall {mean_tprt:.3f} vs clamped {mean_clamped:.3f}")
    # And it must not buy recall by breaking the guarantee.
    worst_fdp = max(tprt_fdps)
    assert worst_fdp <= Q * 2.5, (
        f"TPRT bought recall at the cost of FDR: worst fdp {worst_fdp:.3f}")


def test_both_scores_control_fdr_whatever_their_recall():
    """
    Validity is not the thing that differs between them -- that is the point of
    the architecture. Only power is.
    """
    for score_fn, label in ((_mechanistic_score, "clamped"),
                            (_tprt_score, "tprt")):
        total_selected = total_false = 0
        for seed in range(8):
            rows, truth = _population(n_null=1000, n_true=60, seed=700 + seed)
            candidates = call_or_skip(integrate.candidates_from_mechanistic,
                                      rows, score_fn)
            result = call_or_skip(integrate.select, candidates, Q)
            total_selected += len(result.selected)
            total_false += sum(1 for locus in result.selected
                               if not truth[locus])
        if total_selected == 0:
            continue
        fdp = total_false / total_selected
        assert fdp <= Q * 2.5, f"{label}: pooled fdp {fdp:.3f} against q={Q}"


def test_the_tsd_clamp_is_where_the_range_goes():
    """
    Attribute the difference to a specific line rather than to the redesign in
    general.

    `tsd_loglr` clamps to [-1, 3]. A 15 bp duplication with a one-mismatch
    budget, in unique sequence, is worth `log(P(tau|mech) / p_null)` against the
    locally measured background -- about 13 nats. So that single clamp discards
    roughly ten, which is more than the entire headroom the score has.
    """
    unique_bg = tprt.p_null_tandem_duplication(15, 1, 0.0)
    unclamped = call_or_skip(tprt.log_bf_tsd, 15, "l1_5p_truncated", 1, 0.0)
    clamped = M.tsd_loglr(15, True, unique_bg)
    assert clamped <= 3.0, "tsd_loglr is clamped at 3"
    assert unclamped > 10.0, f"unclamped TSD term is {unclamped:.2f} nats"
    assert unclamped - clamped > 7.0, (
        f"the clamp discards {unclamped - clamped:.1f} nats on this case")


# ===========================================================================
# What the head-to-head actually uncovered
# ===========================================================================
def test_calibrating_sigma_on_every_row_makes_selection_impossible():
    """
    A STRUCTURAL incompatibility, and the most important thing this suite has
    found. It is not a property of either score, and the better the score the
    worse it gets.

    Any score worth having puts a true insertion above `log(m/q)`, so every true
    positive SATURATES the cap `C = m/q`. Calibrating sigma on every row then
    gives

        sigma  ~  pi * C          (pi = the true fraction; the saturating rows
                                   dominate the mean)

    so the largest attainable e-value is

        C / sigma  =  1 / pi

    while e-BH at rank r demands `C / r`. Putting those together:

        1/pi >= C/r   =>   r >= pi * C = pi * m / q

    but `r` cannot exceed the number of true rows, which is about `pi * m`.
    So selection requires

        pi * m  >=  pi * m / q   =>   q >= 1

    IMPOSSIBLE for any usable q. Measured at m=1060, pi=0.0566, q=0.10: sigma
    751, penalty 6.62 nats, headroom 2.87 nats, and e-BH needs 5.17 at r=60.

    Removing the cap does not help -- it makes it worse. Uncapped, the true rows
    contribute e^21 each, the mean becomes 7.4e7 and the penalty 18.1 nats
    against a raw score of 21. The cap is not the cause; the contamination is,
    and the contamination is precisely the signal.

    So BOTH of the obvious sample choices are wrong:

      * exclude rows the score likes (what the C++ did) -> the right tail goes,
        sigma collapses onto its floor of 1, and the correction is inert. That
        is anti-conservative.
      * include everything (what this port does) -> the true positives set the
        bar they then have to clear, and nothing is ever selected. That is
        valid but useless.

    The resolution has to be an estimator that is simultaneously valid, not
    selected by the score, and robust to a small fraction of large values.
    Trimming shows the shape of it: dropping the top 5% takes the penalty from
    6.30 to 2.72 nats and the headroom from 2.97 to 6.55, which clears the 5.17
    that e-BH needs -- while dropping only 2% (below pi) does not. The trim
    fraction has to be near pi, and pi is what e-BH is estimating, so there is a
    natural fixed point: sigma -> selection -> pi -> sigma. A median-of-means or
    Catoni M-estimator is the principled version of the same idea.
    """
    rows, truth = _population(n_null=1000, n_true=60, seed=500)
    m = len(rows)
    pi = 60.0 / m
    cap_value = m / Q

    scores = [_tprt_score(locus) for locus in rows]
    true_scores = [s for s, locus in zip(scores, rows) if truth[locus["locus_id"]]]

    # Premise: a good score saturates the cap on MOST true rows.
    #
    # Not on all of them, and my first version asserted the minimum and failed.
    # The sequence term is `length * per_base` with `per_base` a log-LR between
    # q_young = 0.98 and q_ambient = 0.88, so an insert at identity 0.94 sits
    # halfway between the two models and earns almost nothing per base -- a
    # 600 bp insert at 0.94 scores about -3.8 nats. That is a real weakness
    # worth noting separately (q_ambient is a global constant where it should be
    # per-subfamily, read off the reference's own copies of THAT family), but it
    # does not affect the argument here, which only needs the bulk of the true
    # rows to saturate.
    true_scores.sort()
    median_true = true_scores[len(true_scores) // 2]
    assert median_true > math.log(cap_value), (
        f"premise fails: median true score {median_true:.2f} does not reach "
        f"the cap {math.log(cap_value):.2f}, so this analysis does not apply")
    saturating = sum(1 for s in true_scores if s > math.log(cap_value))
    assert saturating >= 0.5 * len(true_scores)

    result = _run(rows, truth, _tprt_score)
    # sigma lands near pi * cap_value.
    assert result["sigma"] > 0.5 * pi * cap_value
    assert result["sigma"] < 3.0 * pi * cap_value

    headroom = result["cap"] - result["penalty"]
    needed = math.log(m / (Q * 60))
    assert headroom < needed, (
        f"headroom {headroom:.2f} nats vs {needed:.2f} needed at r=60")
    assert result["recall"] == 0.0


def test_trimming_the_contaminated_tail_restores_the_headroom():
    """
    The fix direction, measured. Not implemented in `dependency.py` yet, because
    a trimmed mean needs its own validity argument -- the empirical Bernstein
    bound is for the mean of the sample it is given, and trimming changes the
    estimand. What this test establishes is only that the arithmetic works out,
    so the remaining problem is the guarantee rather than the numbers.
    """
    rows, truth = _population(n_null=1000, n_true=60, seed=500)
    m = len(rows)
    cap_log = math.log(m / Q)
    e_values = sorted(math.exp(min(_tprt_score(locus), cap_log))
                      for locus in rows)
    needed = math.log(m / (Q * 60))

    def headroom_after_trim(fraction: float) -> float:
        keep = e_values[:int(len(e_values) * (1.0 - fraction))] or e_values[:1]
        mean = sum(keep) / len(keep)
        return cap_log - math.log(max(mean, 1.0))

    assert headroom_after_trim(0.0) < needed, "no trim: cannot select"
    assert headroom_after_trim(0.02) < needed, (
        "trimming below the true fraction is not enough")
    assert headroom_after_trim(0.05) > needed, (
        "trimming at about the true fraction clears the requirement")


def test_the_sequence_terms_crossover_is_a_likelihood_ratio_not_a_midpoint():
    """
    Noticed while fixing the premise of the test above, and more interesting
    than the thing I first wrote down.

    `log_bf_sequence` is `length * per_base` with `per_base` the log-LR between
    a young-subfamily divergence (q_young = 0.98) and the ambient divergence of
    the reference's own copies (q_ambient = 0.88). I expected the term to vanish
    at the arithmetic midpoint 0.93; it vanishes at

        q* = -ln(0.02/0.12) / (ln(0.98/0.88) - ln(0.02/0.12)) = 0.9433

    which is where the two Bernoulli models are equally likely. That is the
    correct crossover for a likelihood ratio and it is NOT the midpoint -- my
    first assertion failed because 0.93 still scores -25 nats per kb, 36% of the
    magnitude at 0.98.

    Measured per 1 kb: 0.88 -> -120.3, 0.92 -> -44.3, 0.9433 -> 0, 0.96 -> +31.7,
    0.98 -> +69.6.

    The consequence worth acting on: `q_ambient` is a single GLOBAL constant, so
    the crossover is the same for every family. The ambient divergence of L1HS
    and of an old L1PA lineage are very different numbers, and using one for
    both discards exactly the discrimination this term exists to provide. It
    should be measured per subfamily from the reference's own copies of THAT
    family -- which is the same "read the null off the genome" rule the other
    three nulls already follow.
    """
    crossing = 0.9433
    assert call_or_skip(tprt.log_bf_sequence, 1000, crossing) == \
        pytest.approx(0.0, abs=0.5)
    assert call_or_skip(tprt.log_bf_sequence, 1000, 0.88) < -100.0, (
        "an insert at ambient divergence is positively evidence AGAINST youth")
    assert call_or_skip(tprt.log_bf_sequence, 1000, 0.98) > 65.0
    # Strictly increasing in identity above the floor, and length only scales it.
    previous = -math.inf
    for q in (0.70, 0.80, 0.88, 0.92, 0.96, 0.99):
        value = call_or_skip(tprt.log_bf_sequence, 1000, q)
        assert value > previous, f"non-monotone at identity {q}"
        previous = value
    assert call_or_skip(tprt.log_bf_sequence, 2000, 0.96) == \
        pytest.approx(2.0 * call_or_skip(tprt.log_bf_sequence, 1000, 0.96),
                      rel=1e-12), "the term must be linear in length"


def test_a_global_ambient_divergence_is_a_named_limitation():
    """
    Pinned so it cannot quietly become permanent: the crossover moves with
    `q_ambient`, so a per-subfamily value changes which inserts the term
    supports. The parameter is exposed, which is what makes the fix cheap.
    """
    young_family = call_or_skip(tprt.log_bf_sequence, 1000, 0.95,
                                q_young=0.99, q_ambient=0.80)
    old_family = call_or_skip(tprt.log_bf_sequence, 1000, 0.95,
                              q_young=0.99, q_ambient=0.94)
    assert young_family > old_family + 40.0, (
        "the same 0.95 identity means very different things against a family "
        "whose reference copies sit at 0.80 versus 0.94")
