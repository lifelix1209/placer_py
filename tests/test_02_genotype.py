"""Genotype likelihood and GQ behaviour of `placer_py.core.genotype`."""

from __future__ import annotations

import pytest
from conftest import call_or_skip

from placer_py.core import genotype


@pytest.mark.regression
def test_gq_is_the_posterior_error_in_phred_not_a_likelihood_difference():
    """
    A specific trap, and one I fell into while documenting the algorithm: GQ is

        -10 log10(1 - P(best_gt | data))

    NOT the best-minus-second-best log-likelihood difference. This recomputes
    the posterior independently from the per-genotype likelihoods and priors,
    so a port that implements the wrong one fails here rather than shipping a
    subtly different quality score.
    """
    import math

    for alt, ref in ((3, 3), (6, 2), (10, 1), (4, 8)):
        d = call_or_skip(genotype.genotype_from_alt_vs_ref,
                         alt, ref, error_rate=0.02, overdispersion=0.02,
                         event_length=320)
        assert d.best_gt != "0/0", (alt, ref)
        inp = genotype.GenotypeInput(alt_struct_reads=alt, ref_span_reads=ref,
                                     error_rate=0.02, overdispersion=0.02,
                                     event_length=320)
        lls = [genotype.genotype_log_likelihood(inp, f) + math.log(prior)
               for f, prior in ((0.0, genotype.PRIOR_HOM_REF),
                                (0.5, genotype.PRIOR_HET),
                                (1.0, genotype.PRIOR_HOM_ALT))]
        top = max(lls)
        norm = top + math.log(sum(math.exp(x - top) for x in lls))
        p_err = max(1e-12, 1.0 - math.exp(top - norm))
        expected = max(0, min(99, int(math.floor(-10 * math.log10(p_err) + 0.5))))
        assert d.gq == expected, (alt, ref)

    d = call_or_skip(genotype.genotype_from_alt_vs_ref,
                     3, 3, error_rate=0.02, overdispersion=0.02,
                     event_length=320)
    # The likelihood margin is a DIFFERENT number and is reported separately.
    assert abs(d.gq - d.best_nonref_minus_ref_ll) > 10.0, (
        "if these two are close, the port is probably conflating them")


@pytest.mark.regression
def test_no_minimum_depth_gate():
    """
    `alt=2, ref=0` must genotype 1/1 with low GQ, not be refused.

    A depth gate was added to the C++ once and then reverted: a shallow locus
    already has a diffuse posterior and therefore a low GQ, so the gate was a
    second expression of the same thing. This pins the behaviour so it cannot
    come back.
    """
    d = call_or_skip(genotype.genotype_from_alt_vs_ref, 2, 0,
                     error_rate=0.02, overdispersion=0.02, event_length=320)
    deep = call_or_skip(genotype.genotype_from_alt_vs_ref, 20, 0,
                        error_rate=0.02, overdispersion=0.02, event_length=320)
    assert d.best_gt == "1/1"
    assert d.depth == 2
    assert 0 < d.gq < deep.gq, "shallow support is reported as low GQ"


def test_the_rho_zero_limit_is_approximate_not_exact():
    """
    My stub for this got two things wrong, and the port caught both.

    First the signature: the C++ `beta_binomial_log_pmf` takes
    `(alt, total, alpha, beta)`, not `(k, n, mu, rho)`. The mu/rho form reads
    more naturally but it is the wrong port boundary -- the conversion lives in
    `genotype_log_likelihood`, and keeping that split keeps the same two
    functions the C++ has. Calling it with
    `beta=0.0` returns the impossible-outcome sentinel, which is correct.

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
    """DELIBERATE DIVERGENCE from the C++: the sentinel is -inf, not -1e300.

    See docs/departures-from-cpp.md. The three copies of this function did
    not agree -- `finalization.py` already returned -inf while this module and
    `policy.py` returned -1e300 -- and `math.isfinite` is the test the
    finalization stage uses to tell an abstaining line of evidence from a
    merely unlikely one. -1e300 reads as the second while meaning the first.

    Unifying on -inf changed no output: the example dataset's three files are
    byte-identical across the change.
    """
    import math

    for args in ((-1, 10, 1.0, 1.0), (3, 2, 1.0, 1.0), (3, 10, 0.0, 1.0),
                 (3, 10, 1.0, 0.0)):
        value = call_or_skip(genotype.beta_binomial_log_pmf, *args)
        assert value == -math.inf, args
        assert not math.isfinite(value), (
            "the point of the change: an impossible outcome must not pass "
            "an isfinite() guard")


GRID = [(alt, ref, err, rho)
        for alt, ref in ((0, 0), (0, 10), (1, 9), (2, 0), (3, 3), (5, 5),
                         (8, 2), (10, 0), (20, 20))
        for err in (0.02, 0.05)
        for rho in (0.0, 0.02, 0.20)]


@pytest.mark.invariant
def test_decision_fields_are_consistent_with_the_counts():
    """
    Depth is alt + ref, AF is the raw observed ratio (not a shrunken posterior
    mean), GQ is a bounded Phred, and 0/0 and 1/1 are not reachable from the
    same alt fraction.

    An earlier version asserted `(alt == 0) == (best_gt == "0/0")` and failed
    on `alt=1, ref=9`, which is correctly 0/0 -- one alt read in ten is entirely
    consistent with the error rate. Pinning the ratio is the assertion that
    actually has content.
    """
    for alt, ref, err, rho in GRID:
        d = call_or_skip(genotype.genotype_from_alt_vs_ref, alt, ref,
                         error_rate=err, overdispersion=rho, event_length=320)
        label = f"alt={alt} ref={ref} err={err} rho={rho}"
        assert d.depth == alt + ref, label
        assert 0.0 <= d.allele_fraction <= 1.0, label
        if d.depth:
            assert d.allele_fraction == alt / d.depth, label
        assert 0 <= d.gq <= 99, label
        if d.best_gt == "0/0":
            assert d.allele_fraction < 0.35, label
        if d.best_gt == "1/1":
            assert d.allele_fraction > 0.65, label


@pytest.mark.invariant
def test_overdispersion_widens_the_posterior():
    """Larger rho means more count dispersion, so the same 3v3 observation is
    less decisive."""
    tight = call_or_skip(genotype.genotype_from_alt_vs_ref, 3, 3,
                         error_rate=0.02, overdispersion=0.02, event_length=320)
    loose = call_or_skip(genotype.genotype_from_alt_vs_ref, 3, 3,
                         error_rate=0.02, overdispersion=0.20, event_length=320)
    assert loose.gq <= tight.gq


@pytest.mark.invariant
def test_overdispersion_estimator_falls_back_when_starved():
    for observations in ([], [(5, 10)], [(5, 10), (6, 12), (4, 9), (7, 15)]):
        rho = call_or_skip(genotype.estimate_overdispersion, observations, 0.02)
        assert rho == 0.02, observations
    rich = [((i % 7) + 3, (i % 11) + 10) for i in range(200)]
    rho = call_or_skip(genotype.estimate_overdispersion, rich, 0.02)
    assert 0.0 <= rho < 1.0, "an intra-class correlation"
