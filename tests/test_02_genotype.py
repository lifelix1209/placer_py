"""Golden genotypes. Skips until `placer_py.genotype` is ported."""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer_py import genotype

pytestmark = pytest.mark.golden


def test_every_genotype_case_matches_the_cpp(oracle):
    for g in oracle["genotypes"]:
        d = call_or_skip(genotype.genotype_from_alt_vs_ref,
                         g["alt"], g["ref"],
                         error_rate=g["error_rate"],
                         overdispersion=g["overdispersion"],
                         event_length=320)
        label = f"alt={g['alt']} ref={g['ref']} err={g['error_rate']} rho={g['overdispersion']}"
        assert d.best_gt == g["best_gt"], f"{label}: genotype"
        assert d.gq == g["gq"], f"{label}: GQ (integer, must match exactly)"
        assert d.depth == g["depth"], f"{label}: depth"
        assert d.pass_ == g["pass"], f"{label}: pass"
        close(d.allele_fraction, g["allele_fraction"], f"{label}: AF")
        close(d.best_nonref_minus_ref_ll, g["best_nonref_minus_ref_ll"],
              f"{label}: best_nonref_minus_ref_ll")


@pytest.mark.regression
def test_gq_is_the_posterior_error_in_phred_not_a_likelihood_difference(oracle):
    """
    A specific trap, and one I fell into while documenting the algorithm: GQ is

        -10 log10(1 - P(best_gt | data))

    NOT the best-minus-second-best log-likelihood difference. The golden values
    distinguish the two, so a port that implements the wrong one fails here
    rather than shipping a subtly different quality score.
    """
    g = next(x for x in oracle["genotypes"]
             if x["alt"] == 3 and x["ref"] == 3
             and x["error_rate"] == 0.02 and x["overdispersion"] == 0.02)
    d = call_or_skip(genotype.genotype_from_alt_vs_ref,
                     3, 3, error_rate=0.02, overdispersion=0.02,
                     event_length=320)
    assert d.gq == g["gq"] == 26
    # The likelihood margin is a DIFFERENT number and is reported separately.
    close(d.best_nonref_minus_ref_ll, g["best_nonref_minus_ref_ll"],
          "likelihood margin")
    assert abs(d.gq - d.best_nonref_minus_ref_ll) > 10.0, (
        "if these two are close, the port is probably conflating them")


@pytest.mark.regression
def test_no_minimum_depth_gate(oracle):
    """
    `alt=2, ref=0` must genotype 1/1 with low GQ, not be refused.

    A depth gate was added to the C++ once and then reverted: a shallow locus
    already has a diffuse posterior and therefore a low GQ, so the gate was a
    second expression of the same thing. This pins the behaviour so it cannot
    come back.
    """
    d = call_or_skip(genotype.genotype_from_alt_vs_ref, 2, 0,
                     error_rate=0.02, overdispersion=0.02, event_length=320)
    golden = next(g for g in oracle["genotypes"]
                  if g["alt"] == 2 and g["ref"] == 0)
    assert d.best_gt == golden["best_gt"] == "1/1"
    assert d.gq == golden["gq"]


def test_the_rho_zero_limit_is_approximate_not_exact():
    """
    My stub for this got two things wrong, and the port caught both.

    First the signature: the C++ `beta_binomial_log_pmf` takes
    `(alt, total, alpha, beta)`, not `(k, n, mu, rho)`. The mu/rho form reads
    more naturally but it is the wrong port boundary -- the conversion lives in
    `genotype_log_likelihood`, and keeping that split means the golden
    comparison exercises the same two functions the C++ has. Calling it with
    `beta=0.0` returns the -1e300 sentinel, which is correct.

    Second the claim: at `rho <= 1e-9` the C++ does NOT switch to a closed
    binomial form. It sets `kappa = 1e9` and stays in the Beta-binomial, which
    approximates the binomial to about 6 decimal places and no further
    (-2.143985748 against -2.143980063, a difference of 5.7e-06). So a port that
    "helpfully" special-cases the exact binomial limit would DISAGREE with the
    C++ -- the opposite of what I first wrote down.
    """
    import math
    kappa = 1e9
    beta_form = call_or_skip(genotype.beta_binomial_log_pmf, 3, 10,
                             0.5 * kappa, 0.5 * kappa)
    binomial = call_or_skip(genotype.binomial_log_pmf, 3, 10, 0.5)
    assert math.isclose(beta_form, binomial, rel_tol=1e-5), (
        "kappa=1e9 should be close to the binomial")
    assert beta_form != binomial, (
        "but NOT equal -- a port that special-cases the exact limit diverges "
        "from the C++")


def test_invalid_parameters_return_the_sentinel():
    for args in ((-1, 10, 1.0, 1.0), (3, 2, 1.0, 1.0), (3, 10, 0.0, 1.0),
                 (3, 10, 1.0, 0.0)):
        assert call_or_skip(genotype.beta_binomial_log_pmf, *args) == -1e300


def test_overdispersion_estimates_match_the_cpp(oracle):
    sets = {
        0: [],
        1: [(5, 10)],
        4: [(5, 10), (6, 12), (4, 9), (7, 15)],
        200: [((i % 7) + 3, (i % 11) + 10) for i in range(200)],
    }
    for g in oracle["overdispersion"]:
        observations = sets[g["n_sites"]]
        rho = call_or_skip(genotype.estimate_overdispersion, observations, 0.02)
        close(rho, g["rho"], f"rho for n_sites={g['n_sites']}")
