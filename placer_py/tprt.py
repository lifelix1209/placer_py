"""
The TPRT coincidence model -- the intended REPLACEMENT for the clamped blocks.

THE IDEA
    A TPRT insertion is a set of COINCIDENCES between the inserted sequence and
    THIS genomic locus. The model is the log-odds of those coincidences against
    a null measured from the LOCAL reference sequence.

THE DECOMPOSITION THAT DOES THE WORK
    The dominant false positive is a read mismapped from an OLD reference copy,
    and that copy is itself a TPRT product -- complete 3' end, poly(A), the
    works. So internal structure CANNOT separate H_TE from H_artifact, and that,
    not any tuning intuition, is why a high-identity TE alignment proves nothing
    on its own.

      (a) INTERNAL terms -- where the insert's ends land in element coordinates,
          and identity against the young-subfamily consensus. Discriminate
          H_TE from H_nonTE; blind to H_artifact.
      (b) LINKAGE terms -- the flanks of THIS locus are duplicated, the nick
          site of THIS locus is a T-tract, no reads span THIS locus cleanly.
          Discriminate H_TE from H_artifact.

    Two hypotheses, two disjoint evidence sets, and the mechanism assigns each
    term. Nothing is weighted and nothing is a "gate".

WHY IT MATTERS BEYOND ELEGANCE -- the measured reason to prefer it.
    The existing clamped blocks total at most ~9.9 nats (endonuclease
    clamp(en,0,6)*0.45 <= 2.7, tsd_loglr clamped to [-1,3], poly(A) bounded to
    ~[-0.3,1.3], TE body ~2.9). e-BH at m=1060, q=0.10 needs log(m/q) = 9.27
    nats just to clear the RANK-1 threshold, before any dependency penalty. So
    the clamped score cannot support genome-scale multiple testing at all --
    recall is zero, and not because a threshold is mistuned.

    These terms are real log-LRs against measured local nulls, so they carry the
    range: a 15 bp exact duplication in unique sequence is worth ~13 nats, where
    `tsd_loglr`'s clamp at 3.0 discards about ten of them.

ROBUSTNESS COMES FROM ONE CHOICE
    Every null is measured from the local reference. In a VNTR the chance of a
    tandem duplication is ~1, so the TSD term self-cancels; in an A-rich tract
    the priming term does the same. Conservative exactly where the data is
    treacherous, with no thresholds and no special cases. The same rule applies
    four times: duplication background, A/T composition, T-tract background, and
    the local read-error rate.

LITERATURE, including one fact that changed the model's shape
  * The EN consensus is 5'-TTTTT/AA-3' (degenerate TTTT/A, often preceded by an
    imperfect T-tract) -- NOT TTAAAA. Miller et al. 2021 NAR; Monot et al. 2013
    PLoS Genet.
  * EN is promiscuous in vitro and cuts largely on DNA STRUCTURE rather than
    sequence (Miller et al. 2025 JBC), so the motif is a soft bias, never a
    requirement.
  * Priming needs as few as 4 matching nt at the primer 3' end and tolerates
    terminal mismatches compensated within the last 10 bases (Monot 2013), so
    the T-tract observable is graded and saturates at 10.
  * SHAPE-CHANGING: only 5'-TRUNCATED L1 insertions carry short TSDs. Full-length
    L1, 5'-inverted (twin-primed) L1, and trans-mobilised Alu / SVA / processed
    pseudogenes are NOT (Kojima 2010, Mobile DNA). So "TSD present" is not
    uniformly good news, and the mechanism must be a latent variable.
  * EN-independent retrotransposition integrates at atypical targets, truncates
    at the 3' END and lacks TSDs (Morrish 2002 Nat Genet) -- rare, which is why
    the 3'-anchoring term needs a mixture escape hatch rather than a hard
    constraint.

The poly(A) and the T-tract are two views of ONE physical event (the tail
annealing to the nicked strand), so they are combined into a single term rather
than multiplied as if independent. Multiplying them would double count, and the
mechanism is what tells you so.

NOT ON THE CALLING PATH, and that is deliberate rather than neglect --
see `docs/off-pipeline-modules.md` for which of the four unimported
modules this is and why deleting it would lose something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb, exp, lgamma, log

# Fixed prior over mechanism. NOT to be conditioned on the observed geometry.
#
# Conditioning a prior on the data double counts the data: renormalising within
# the mechanisms "consistent with" what was seen turned the 2%-prior
# EN-independent route into a 90% alibi and rescued a mismapped interior
# fragment to 4.88 nats. Every mixture component with weak predictions is an
# escape hatch, and the defence is a fixed prior.
MECHANISM_PRIOR: dict[str, float] = {
    "l1_5p_truncated": 0.50,
    "trans_alu_sva": 0.33,
    "l1_5p_inverted": 0.10,
    "l1_full_length": 0.05,
    "en_independent": 0.02,
}

#: P(TSD present), mean TSD length, BY MECHANISM (Kojima 2010).
TSD_MODEL: dict[str, tuple[float, float]] = {
    "l1_5p_truncated": (0.90, 15.0),
    "l1_full_length": (0.25, 12.0),
    "l1_5p_inverted": (0.30, 12.0),
    "trans_alu_sva": (0.30, 12.0),
    "en_independent": (0.02, 10.0),
}


@dataclass
class TprtBayesFactors:
    """Kept SEPARATE: they are evidence about different alternatives, and the
    decision takes the MINIMUM. A candidate must beat both."""

    log_bf_vs_non_te: float
    log_bf_vs_artifact: float
    log_bf_counts: float


# ===========================================================================
# Local nulls -- measured from the reference, no free parameters
# ===========================================================================
def p_null_tandem_duplication(tau: int, mismatches_allowed: int,
                              local_repeat_frac: float) -> float:
    """
    Null probability that the flanks of THIS locus happen to look duplicated
    over `tau` bp, allowing a mismatch budget:

        sum_{j<=b} C(tau, j) 3^j / 4^tau

    blended toward 1.0 by the measured local tandem-repeat fraction, because in
    a VNTR a duplication IS the local norm. That blend is the whole robustness
    mechanism for this term, and `local_repeat_frac` is data (the fraction of
    the flanking window under a tandem-repeat annotation, or self-alignment),
    not a tuning knob.
    """
    if tau <= 0:
        return 1.0
    total = sum(comb(tau, j) * (3.0 ** j)
                for j in range(max(0, mismatches_allowed) + 1))
    p_unique = total / (4.0 ** tau)
    frac = min(max(local_repeat_frac, 0.0), 1.0)
    return (1.0 - frac) * p_unique + frac * 1.0


def p_null_terminal_poly(a: int, local_a_frac: float) -> float:
    """Null probability of an `a`-base A-run at the junction, at the LOCAL base
    composition -- ~0.30 genome-wide but ~0.9 inside an A-rich tract, and the
    genome is full of those precisely BECAUSE of past TE insertions."""
    return min(max(local_a_frac, 1e-6), 1.0) ** max(0, a)


def p_null_t_tract(k_match: int, local_t_frac: float) -> float:
    """Null probability of `k` matching T's immediately 5' of the nick."""
    return min(max(local_t_frac, 1e-6), 1.0) ** max(0, k_match)


# ===========================================================================
# (a) INTERNAL: is this a TPRT product at all?   H_TE vs H_nonTE
# ===========================================================================
def log_bf_three_prime_anchoring(u: int, v: int, elem_len: float,
                                 p_en_independent: float = 0.02) -> float:
    """
    The insert aligns to element coordinates [u, v] of an element of length L.

    H_TPRT : v = L. RT initiates at the poly(A), so the 3' end is complete.
             Escape hatch: EN-independent events truncate at the 3' end instead
             (Morrish 2002), so allow that with a small fixed probability, under
             which the position is unconstrained.
    H_nonTE: the insert is not an element copy, so if it nonetheless aligns to
             [u, v] the position is uninformative -- maximum entropy given the
             observed length gives u ~ Unif{0 .. L-l}.

    Worth `ln(L - l + 1)` = 8.5 nats for a 1 kb fragment of a 6 kb L1, from one
    observation with no fitted parameter. The current pipeline discards it
    entirely by collapsing the alignment to a coverage ratio; keeping two
    integers instead of one ratio is the whole cost of recovering it.

    ZERO for a full-length insert, and that is correct rather than a bug: when
    the insert covers the whole element there is only one possible position, so
    there is no coincidence to score. Full-length insertions are called on the
    linkage side instead.
    """
    length = v - u
    n_positions = max(1.0, elem_len - length + 1.0)
    p_null = 1.0 / n_positions
    three_prime_complete = v >= elem_len - 5
    p_alt = ((1.0 - p_en_independent) * (1.0 if three_prime_complete else 0.0)
             + p_en_independent * p_null)
    if p_alt <= 0.0:
        p_alt = p_en_independent * p_null
    return log(p_alt) - log(p_null)


def log_bf_sequence(length: int, identity: float, q_young: float = 0.98,
                    q_ambient: float = 0.88,
                    genome_te_frac: float = 0.50) -> float:
    """
    Is this a YOUNG element copy, or some other piece of genome?

    A per-base log-LR between two EXPLICIT Bernoulli models of the match: the
    youngest active subfamily's divergence (read off the library) against the
    divergence of the reference's own copies (read off the genome). Both rates
    are measurable, there is no cap and no offset, and the scale comes from the
    length -- telling a young copy from an old one over 6 kb really is decisive,
    while over 300 bp it is not.

    Three wrong versions preceded this, all caught by running numbers: dropping
    the term (a real full-length L1 scored -2.59 nats, because "it cancels" is a
    claim about ONE ratio and does not hold against H_nonTE); log-odds against
    i.i.d. uniform background (1152 nats, swamping everything, because H_nonTE
    is "some other piece of genome" and ~50% of the human genome is TE-derived);
    and a capped z-score (0.78 nats, under-calling full-length insertions).
    """
    if identity < 0.60:
        # Not element sequence at all: this is the evidence that says non-TE.
        return log(1.0 - genome_te_frac) - log(genome_te_frac) - 2.0
    q = min(max(identity, 0.61), 0.9995)
    per_base = (q * log(q_young / q_ambient)
                + (1.0 - q) * log((1.0 - q_young) / (1.0 - q_ambient)))
    return length * per_base


# ===========================================================================
# (b) LINKAGE: did it happen HERE?   H_TE vs H_artifact
# ===========================================================================
def log_p_tsd_given_mechanism(tau: int, mechanism: str) -> float:
    """Geometric-ish duration for the second-nick offset, conditional on the
    mechanism that produced the insertion."""
    p_present, mean_tau = TSD_MODEL[mechanism]
    if tau <= 0:
        return log(max(1e-12, 1.0 - p_present))
    lam = 1.0 / mean_tau
    return log(p_present) + log(lam) - lam * tau


def log_bf_tsd(tau: int, mechanism: str, mismatches_allowed: int,
               local_repeat_frac: float) -> float:
    """
    The flanks of THIS locus are duplicated over `tau` bp.

    This is the term that ties the insert to the locus, and the reason a
    mismapped old copy cannot pass: it has a perfectly good 3' end and poly(A)
    of its own, just somewhere else.

    `tau <= 0` is NOT evidence of artifact -- for a full-length or
    trans-mobilised insertion, no TSD is the expected outcome, so the absence is
    scored against that mechanism's own P(TSD present) rather than at a flat
    penalty. A flat penalty is what the clamped model does, and it punishes
    full-length L1 and Alu for obeying their own mechanism.
    """
    if tau <= 0:
        return log_p_tsd_given_mechanism(0, mechanism)
    p_null = p_null_tandem_duplication(tau, mismatches_allowed,
                                       local_repeat_frac)
    return log_p_tsd_given_mechanism(tau, mechanism) - log(max(p_null, 1e-300))


def log_bf_priming_site(k_match: int, polya_len: int, local_t_frac: float,
                        local_a_frac: float, mechanism: str,
                        min_match: int = 4) -> float:
    """
    The nick site of THIS locus is a T-tract, and the insert carries a poly(A)
    that must anneal to it.

    ONE event seen from two sides, so it is a single term. The observable is the
    number of complementary bases at the junction, which needs >= ~4 and
    saturates within the last ~10 (Monot 2013), so the effective evidence is
    capped at 10 matching bases however long the run is.

    Below `min_match` the EN-dependent route could not have primed here. Soft,
    not fatal, because the endonuclease is promiscuous and cuts on structure
    (Miller 2025).
    """
    if mechanism == "en_independent":
        return 0.0                      # no EN, so no T-tract expectation
    effective = min(min(k_match, polya_len), 10)
    if effective < min_match:
        return log(0.05)
    p_null = (p_null_t_tract(effective, local_t_frac)
              * p_null_terminal_poly(effective, local_a_frac))
    # ONE event, so do not square-count the two views of it.
    p_null = min(1.0, math.sqrt(p_null))
    return log(0.95) - log(max(p_null, 1e-300))


# ===========================================================================
# Counts: is there an insertion at all?  (Lander-Waterman sensitivity)
# ===========================================================================
def _log_beta_binom(k: int, n: int, mu: float, rho: float) -> float:
    mu = min(max(mu, 1e-12), 1 - 1e-12)
    lc = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    if rho <= 1e-9:
        return lc + k * log(mu) + (n - k) * math.log1p(-mu)
    kappa = (1 - rho) / rho
    a, b = mu * kappa, (1 - mu) * kappa
    return (lc + lgamma(k + a) + lgamma(n - k + b) - lgamma(n + a + b)
            + lgamma(a + b) - lgamma(a) - lgamma(b))


def spanning_sensitivity(ins_len: float, read_median: float = 15000.0,
                         anchor: float = 50.0, sigma_log: float = 0.8,
                         n_bins: int = 200) -> float:
    """
    `s(L) = E_R[(R - L - 2a)+ / R]`.

    A carrier read signals only if it spans the insertion with at least `anchor`
    bp of reference on each side. Conditioned on covering the breakpoint, a read
    of length R has start uniform on [-R, 0] and covers [-a, L+a] iff the start
    lies in [L+a-R, -a], of measure (R-L-2a)+. No free parameter: `anchor` is
    the pipeline's own requirement and the read-length distribution is measured
    from the BAM.
    """
    mu = math.log(read_median)
    lo, hi = 200.0, 200000.0
    acc = 0.0
    total = 0.0
    for i in range(n_bins):
        r = lo * (hi / lo) ** (i / (n_bins - 1.0))
        w = ((1.0 / (r * sigma_log * math.sqrt(2 * math.pi)))
             * exp(-((math.log(r) - mu) ** 2) / (2 * sigma_log ** 2)))
        acc += w * max(0.0, r - ins_len - 2 * anchor) / r
        total += w
    return acc / total if total > 0 else 0.0


def local_error_rate(a_frac: float, t_frac: float, repeat_frac: float,
                     base_eps: float = 0.02) -> float:
    """
    The read-error rate is a LOCAL quantity too -- the fourth place the same rule
    applies. Basecallers mis-call homopolymer and tandem-repeat lengths far more
    often than they err in unique sequence, so an "insertion" supported by a few
    reads means much less inside an A-tract than outside one.
    """
    homopolymer = max(a_frac, t_frac)
    difficulty = max(0.0, (homopolymer - 0.30) / 0.70)
    difficulty = max(difficulty, repeat_frac)
    return min(0.45, base_eps + difficulty * 0.30)


def log_bf_counts(n_alt: int, n_ref: int, ins_len: float, eps: float = 0.02,
                  rho: float = 0.02, p_het: float = 2.0 / 3.0) -> float:
    """
    H_TE / H_nonTE: P(signal) = theta*s(L) + (1-theta)*eps, marginalised over
    theta in {1/2, 1}. H_artifact: P(signal) = eps (pure mismapping).

    Marginalising the latent carrier indicator is exactly what the
    beta-binomial does, which is why no separate "event" and "independent"
    blocks are needed -- those were two affine readings of the same latent.
    """
    n = n_alt + n_ref
    if n <= 0:
        return 0.0
    s = spanning_sensitivity(ins_len)
    terms = []
    for theta, weight in ((0.5, p_het), (1.0, 1.0 - p_het)):
        mu = theta * s + (1 - theta) * eps
        terms.append(log(weight) + _log_beta_binom(n_alt, n, mu, rho))
    m = max(terms)
    ll_ins = m + log(sum(exp(t - m) for t in terms))
    ll_art = _log_beta_binom(n_alt, n, eps, rho)
    return ll_ins - ll_art


# ===========================================================================
# Mechanism mixture and the two Bayes factors
# ===========================================================================
def log_p_geometry_given_mechanism(u: int, v: int, elem_len: float,
                                   mechanism: str, inverted: bool) -> float:
    """
    `P(observed end geometry | mechanism)`, near zero where the mechanism cannot
    produce what is seen. That is what keeps the FIXED prior honest: an
    inconsistent geometry scores 0.98*0 + 0.02*(uniform), which is small,
    instead of being renormalised into plausibility.
    """
    three_prime_complete = v >= elem_len - 5
    five_prime_complete = u <= 5
    if mechanism == "en_independent":
        return log(1.0 / max(1.0, elem_len))     # no positional prediction
    if not three_prime_complete:
        return log(1e-4)                          # EN-dependent TPRT cannot
    if mechanism == "l1_5p_inverted":
        return log(1.0 if inverted else 0.02)
    if inverted:
        return log(0.02)
    if mechanism == "l1_full_length":
        return log(1.0 if five_prime_complete else 0.02)
    if mechanism == "l1_5p_truncated":
        return log(0.02 if five_prime_complete else 1.0)
    if mechanism == "trans_alu_sva":
        return log(1.0)
    return log(0.02)


def mechanism_allows_family(mechanism: str, family: str) -> bool:
    """The aligned family is OBSERVED, so it constrains the mechanism -- as a
    likelihood on the family label, not as a reweighted prior."""
    if family.upper() in ("ALU", "SVA", "PSEUDOGENE"):
        return mechanism in ("trans_alu_sva", "en_independent")
    return mechanism != "trans_alu_sva"


def evaluate(locus: dict, elem_len: float, family: str = "L1"
             ) -> TprtBayesFactors:
    """
    Two Bayes factors, kept SEPARATE because they are evidence about different
    alternatives. The decision takes the MINIMUM: a candidate must beat both,
    and summing them would let strong non-TE evidence pay for weak artifact
    evidence.
    """
    u, v = int(locus["u"]), int(locus["v"])
    length = v - u
    inverted = bool(locus.get("inverted", False))

    # (a) INTERNAL -> vs non-TE. The sequence term lives here and only here,
    # because it CANCELS against a mismapped copy (which is also real element
    # sequence) but not against a non-TE insertion (which is not).
    internal = (log_bf_sequence(length, float(locus.get("identity", 0.96)))
                + log_bf_three_prime_anchoring(u, v, elem_len))

    # (b) LINKAGE -> vs artifact. Mixture over mechanism with the FIXED prior.
    terms = []
    for mechanism, prior in MECHANISM_PRIOR.items():
        if not mechanism_allows_family(mechanism, family):
            continue
        terms.append(
            log(prior)
            + log_p_geometry_given_mechanism(u, v, elem_len, mechanism, inverted)
            + log_bf_tsd(int(locus["tsd"]), mechanism,
                         int(locus.get("tsd_mismatches", 0)),
                         float(locus.get("repeat_frac", 0.0)))
            + log_bf_priming_site(int(locus.get("t_tract", 0)),
                                  int(locus.get("polya", 0)),
                                  float(locus.get("t_frac", 0.30)),
                                  float(locus.get("a_frac", 0.30)),
                                  mechanism))
    m = max(terms)
    linkage = m + log(sum(exp(t - m) for t in terms))
    # The mixture's own normaliser, so loci with different family labels are
    # comparable rather than carrying different constant offsets.
    linkage -= log(sum(p for mech, p in MECHANISM_PRIOR.items()
                       if mechanism_allows_family(mech, family)))

    eps = local_error_rate(float(locus.get("a_frac", 0.30)),
                           float(locus.get("t_frac", 0.30)),
                           float(locus.get("repeat_frac", 0.0)))
    counts = log_bf_counts(int(locus.get("n_alt", 0)),
                           int(locus.get("n_ref", 0)), length, eps=eps)

    # Counts bear on "an insertion exists HERE", i.e. on the artifact
    # comparison; a non-TE insertion produces the same counts, so they do not
    # enter the non-TE comparison.
    return TprtBayesFactors(log_bf_vs_non_te=internal,
                            log_bf_vs_artifact=linkage + counts,
                            log_bf_counts=counts)
