"""
Beta-binomial genotyping and the sample-level overdispersion estimate.

Ported from `src/pipeline/decision_policy.cpp` and pinned by the 14 golden cases
in `tests/oracle/cpp_reference.json`. Everything here is a faithful port; the
commentary records the places where a reasonable-looking reimplementation
diverges, because those are the places the golden vectors earn their keep.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

#: Genotype priors, applied as log terms to the three likelihoods.
PRIOR_HOM_REF = 0.25
PRIOR_HET = 0.50
PRIOR_HOM_ALT = 0.25


@dataclass
class GenotypeDecision:
    best_gt: str = "./."
    allele_fraction: float = 0.0
    gq: int = 0
    depth: int = 0
    best_nonref_minus_ref_ll: float = 0.0
    pass_: bool = False


@dataclass
class GenotypeInput:
    """Mirrors the C++ `EventGenotypeInput`. Only the fields the likelihood
    actually reads are kept."""

    alt_struct_reads: int = 0
    ref_span_reads: int = 0
    min_gq: int = 20
    error_rate: float = 0.02
    overdispersion: float = 0.02
    event_length: int = 0
    alt_observed_lengths: list[int] = field(default_factory=list)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_log(p: float) -> float:
    """`log(max(1e-12, p))` -- the C++ `safe_log`."""
    return math.log(max(1e-12, p))


def _phred_from_error_probability(p_error: float) -> float:
    return -10.0 * math.log10(_clamp(p_error, 1e-12, 1.0))


def _logsumexp3(a: float, b: float, c: float) -> float:
    m = max(a, b, c)
    return m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))


def log_choose_count(n: int, k: int) -> float:
    if k < 0 or n < 0 or k > n:
        return -1e300
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def beta_binomial_log_pmf(alt: int, total: int, alpha: float,
                          beta: float) -> float:
    """
    Signature matches the C++: `(alt, total, alpha, beta)`, NOT `(k, n, mu, rho)`.

    My stub declared the mu/rho form, which reads more naturally but is the wrong
    port boundary -- the C++ converts mu/rho to alpha/beta inside
    `genotype_log_likelihood`, and keeping that split means the golden
    comparison exercises the same two functions the C++ has.
    """
    if alt < 0 or total < 0 or alt > total or alpha <= 0.0 or beta <= 0.0:
        return -1e300
    return (log_choose_count(total, alt)
            + math.lgamma(alt + alpha)
            + math.lgamma(total - alt + beta)
            - math.lgamma(total + alpha + beta)
            + math.lgamma(alpha + beta)
            - math.lgamma(alpha)
            - math.lgamma(beta))


def binomial_log_pmf(alt: int, total: int, p: float) -> float:
    if alt < 0 or total < 0 or alt > total:
        return -1e300
    clamped = _clamp(p, 1e-6, 1.0 - 1e-6)
    return (log_choose_count(total, alt)
            + alt * math.log(clamped)
            + (total - alt) * math.log1p(-clamped))


def _median_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _infer_event_length(inp: GenotypeInput) -> int:
    if inp.event_length > 0:
        return inp.event_length
    return _median_int(inp.alt_observed_lengths)


def _alt_length_support_probability(observed_length: int, event_length: int,
                                    error_rate: float) -> float:
    if event_length <= 0 or observed_length <= 0:
        return _clamp(1.0 - (error_rate * 2.0), 0.55, 0.95)
    norm_factor = 20.0 if error_rate <= 0.03 else 30.0
    sigma = max(1.0, event_length / norm_factor)
    z = (observed_length - event_length) / sigma
    return max(1e-6, math.exp(-0.5 * z * z))


def length_concordance_factor(inp: GenotypeInput) -> float:
    """
    Mean length-support probability of the alt reads, or 1.0 when no lengths are
    observed.

    Low values mean the alt reads do not match the event length -- they look
    like a different event, or an artifact, rather than genuine support. It
    down-weights only the NON-REFERENCE genotypes, which is the asymmetry that
    makes it a discount on a claim rather than a generic penalty.
    """
    if not inp.alt_observed_lengths:
        return 1.0
    event_length = _infer_event_length(inp)
    acc = sum(_alt_length_support_probability(observed, event_length,
                                             inp.error_rate)
              for observed in inp.alt_observed_lengths)
    return _clamp(acc / len(inp.alt_observed_lengths), 1e-3, 1.0)


def genotype_log_likelihood(inp: GenotypeInput, alt_copy_fraction: float) -> float:
    """
    `alt | depth, GT ~ BetaBinomial(depth, alpha_GT, beta_GT)`.

    Per-genotype expected alt fraction: `mu_00 = err` (alt only from error or
    artifact), `mu_01 = alt_copy_fraction`, `mu_11 = 1 - err`. The overdispersion
    `rho` sets the concentration `kappa = (1 - rho) / rho`.

    NOTE, and this is the detail my stub got wrong: at `rho <= 1e-9` the C++ does
    NOT switch to a closed binomial form -- it sets `kappa = 1e9` and stays in
    the Beta-binomial. The two agree to about 8 decimal places, not exactly, so
    a port that "helpfully" special-cases the binomial limit fails the golden
    comparison at its 1e-12 tolerance.
    """
    alt = max(0, inp.alt_struct_reads)
    ref = max(0, inp.ref_span_reads)
    depth = alt + ref
    if depth <= 0:
        return 0.0

    err = _clamp(inp.error_rate, 1e-4, 0.2)
    if alt_copy_fraction <= 0.0:
        mu = err
    elif alt_copy_fraction >= 1.0:
        mu = 1.0 - err
    else:
        mu = _clamp(alt_copy_fraction, err, 1.0 - err)

    rho = _clamp(inp.overdispersion, 0.0, 0.99)
    kappa = 1e9 if rho <= 1e-9 else (1.0 - rho) / rho
    alpha = max(1e-6, mu * kappa)
    beta = max(1e-6, (1.0 - mu) * kappa)

    ll = beta_binomial_log_pmf(alt, depth, alpha, beta)
    if alt_copy_fraction > 0.0 and alt > 0:
        ll += alt * math.log(length_concordance_factor(inp))
    return ll


def genotype_from_alt_vs_ref(alt_struct_reads: int, ref_span_reads: int,
                             error_rate: float = 0.02,
                             overdispersion: float = 0.02,
                             min_gq: int = 20,
                             event_length: int = 0,
                             alt_observed_lengths: Sequence[int] | None = None
                             ) -> GenotypeDecision:
    """
    Port of `placer::genotype_event_from_alt_vs_ref`.

    Two details a reimplementation gets wrong, both pinned by golden cases:

    1. GQ is the POSTERIOR error in Phred, `-10 log10(1 - P(best | data))`,
       computed over all three genotypes -- not the best-minus-second-best
       likelihood difference. The likelihood margin is reported separately as
       `best_nonref_minus_ref_ll`, and at 3v3 the two are 26 and 6.69, so a port
       that conflates them is caught immediately.

    2. On a 0/0 call the function returns EARLY, so `best_nonref_minus_ref_ll`
       stays at its default 0.0 rather than holding the (negative) margin. The
       golden row for `alt=1, ref=9` has exactly 0 there, which is the tell.

    There is deliberately no minimum-depth gate: a shallow locus has a diffuse
    posterior and therefore a low GQ on its own, so a depth threshold would be a
    second expression of the same thing. `alt=2, ref=0` genotypes 1/1 with GQ 9.
    """
    inp = GenotypeInput(
        alt_struct_reads=alt_struct_reads,
        ref_span_reads=ref_span_reads,
        min_gq=min_gq,
        error_rate=error_rate,
        overdispersion=overdispersion,
        event_length=event_length,
        alt_observed_lengths=list(alt_observed_lengths or []),
    )
    decision = GenotypeDecision()

    alt = max(0, alt_struct_reads)
    ref = max(0, ref_span_reads)
    depth = alt + ref
    decision.depth = depth
    if depth <= 0:
        return decision

    decision.allele_fraction = _clamp(alt / depth, 0.0, 1.0)

    ll_00 = genotype_log_likelihood(inp, 0.0) + _safe_log(PRIOR_HOM_REF)
    ll_01 = genotype_log_likelihood(inp, 0.5) + _safe_log(PRIOR_HET)
    ll_11 = genotype_log_likelihood(inp, 1.0) + _safe_log(PRIOR_HOM_ALT)

    het_is_best_nonref = ll_01 >= ll_11
    best_nonref_ll = ll_01 if het_is_best_nonref else ll_11
    decision.best_gt = "0/1" if het_is_best_nonref else "1/1"

    if best_nonref_ll <= ll_00:
        decision.best_gt = "0/0"
        decision.gq = 0
        decision.pass_ = False
        return decision

    decision.best_nonref_minus_ref_ll = best_nonref_ll - ll_00
    best_total_ll = max(ll_00, ll_01, ll_11)
    posterior_best = math.exp(best_total_ll - _logsumexp3(ll_00, ll_01, ll_11))
    gq = _phred_from_error_probability(1.0 - posterior_best)
    # C++ uses lround, i.e. round-half-away-from-zero. Python's round() is
    # round-half-to-even, so it would disagree on exact .5 boundaries.
    decision.gq = max(0, min(99, int(math.floor(gq + 0.5))))
    decision.pass_ = decision.gq >= max(0, min_gq)
    return decision


def estimate_overdispersion(observations: Sequence[tuple[int, int]],
                            fallback: float = 0.02) -> float:
    """
    One-way random-effects (Fleiss / ANOVA) estimator of the intra-class
    correlation for proportions:

        rho = (MSB - MSW) / (MSB + (n0 - 1) MSW)

    `observations` are `(alt, depth)` pairs. Only het-like informative sites are
    kept -- `depth >= 4` and `0.15 < alt/depth < 0.85` -- because those are the
    ones carrying the allele-balance overdispersion the genotyper cares about.
    Fewer than 20 such sites returns `fallback`.

    This is the pattern the whole design leans on: a quantity that would
    otherwise be hand-set turns out to be a functional of a distribution, and
    that distribution is in the run's own data. Note the 0.5 clamp at the end --
    it is a guard, not a modelling choice, and rho near 0.5 means the counts are
    so dispersed that the genotyper has almost no information left.
    """
    kept = []
    for alt, depth in observations:
        if depth < 4 or alt < 0 or alt > depth:
            continue
        p = alt / depth
        if 0.15 < p < 0.85:
            kept.append((alt, depth))

    m = len(kept)
    if m < 20:
        return fallback

    sum_x = float(sum(alt for alt, _ in kept))
    sum_n = float(sum(depth for _, depth in kept))
    sum_n2 = float(sum(depth * depth for _, depth in kept))

    p_hat = sum_x / sum_n
    if p_hat <= 1e-6 or p_hat >= 1.0 - 1e-6:
        return fallback

    s_between = 0.0
    sum_within = 0.0
    for alt, depth in kept:
        p = alt / depth
        s_between += depth * (p - p_hat) * (p - p_hat)
        sum_within += alt * (depth - alt) / depth

    total_n = sum_n
    n0 = (total_n - (sum_n2 / total_n)) / (m - 1.0)
    msb = s_between / (m - 1.0)
    msw = sum_within / (total_n - m)
    denom = msb + ((n0 - 1.0) * msw)
    if denom <= 1e-9:
        return fallback
    return _clamp((msb - msw) / denom, 0.0, 0.5)
