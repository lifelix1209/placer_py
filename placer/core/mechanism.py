"""The per-class likelihood ratios a call is decided on.

TWO QUESTIONS, TWO LOG-LIKELIHOOD RATIOS:

  * `vs_artifact` -- is there a new insertion HERE at all?
                     (H_TE against H_artifact: mismapping, a reference copy
                     misplaced, a sequencing artefact)
  * `vs_non_te`   -- given an insertion, is the inserted sequence a TE?
                     (H_TE against H_nonTE: a real insertion of other sequence)

Only `vs_artifact` decides (`core/mechanism_selection.py`): e-BH on it says
which insertions are there, and whether one is a TE is then TEBench's coverage
rule on the insert. `vs_non_te` is reported as the transposition evidence.
(Until 2026-09-26 a TE call had to win both, on exp(min) -- valid against
both nulls, but it lost the SVAs and kept inserts TEBench does not call TEs.)

WHAT EACH CLASS CONTRIBUTES. Only the terms a class's insertion mechanism
actually produces (`core/taxonomy.py`, `core/element_structure.py`):

  vs_non_te    sequence identity                                     all
               3' anchoring (5' truncation is expected)              TPRT
               both ends complete                                    LTR, DNA
               poly(A) tail                                          TPRT
               TG...CA termini                                       LTR
               terminal inverted repeats                             DNA
               5' TC ... 3' CTRR                                     RC
  vs_artifact  read counts (alt against ref)                         all
               TSD, by the superfamily's own length and sequence     all
               L1 endonuclease target motif                          LINE, SINE,
                                                                     Retroposon

An element of Unknown class is scored as a mixture over the mechanisms, with a
fixed prior.

EVERY HALLMARK TERM IS A ROBUST MIXTURE:

    log( w * P(x | hallmark present) / P(x | null)  +  (1 - w) )

-- a true likelihood ratio in which, with probability 1 - w, the insertion
simply does not show the hallmark (a degraded tail, a TSD lost to a read error,
an element that ignored the canonical site). It is bounded below by
log(1 - w), so no single missing hallmark can sink a real insertion, and above
only by how improbable the observation is under the null.

THE PARAMETERS are literature values (`SUPERFAMILY_TSD`, `TAIL_MEAN_BP`, ...),
and `MechanismParameters` holds them so the sample can update them
(`update_parameters`). Whether the resulting scores really are e-values on a
given genome is not assumed: the shifted-breakpoint decoys measure it.

Builds on `core/tprt.py`, whose sequence, 3'-anchoring, TSD-null and counts
terms are reused unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from placer.core import tprt
from placer.core.seqtools import reverse_complement
from placer.core.taxonomy import TeClass

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TsdModel:
    """P(TSD length | mechanism). `modal_len` 0 means "no TSD expected"."""

    p_present: float
    #: A fixed-length TSD (transposases, integrases): this length, with
    #: `spread` of the mass on +-1 bp.
    modal_len: int = 0
    spread: float = 0.2
    #: A variable-length TSD (TPRT): geometric with this mean instead.
    mean_len: float = 0.0
    #: The TSD's own sequence, when the superfamily fixes it (TA, TTAA, TAA).
    motif: str = ""

    def log_p_length(self, tau: int) -> float:
        if tau <= 0:
            return math.log(max(1e-6, 1.0 - self.p_present))
        if self.mean_len > 0.0:
            lam = 1.0 / self.mean_len
            return math.log(self.p_present) + math.log(lam) - lam * tau
        if self.modal_len <= 0:
            return math.log(max(1e-6, self.p_present)) - math.log(50.0)
        if tau == self.modal_len:
            p = 1.0 - self.spread
        elif abs(tau - self.modal_len) == 1:
            p = self.spread / 2.0 * 0.9
        else:
            # A small floor spread over 50 other lengths: a mis-measured TSD.
            p = self.spread * 0.1 / 50.0
        return math.log(self.p_present) + math.log(p)


#: TSD by superfamily (prefix of the lower-cased superfamily name), from the
#: element-biology literature (Wicker et al. 2007 Nat Rev Genet; Kojima 2010).
#: Longest matching prefix wins; `CLASS_TSD` is the fallback.
SUPERFAMILY_TSD: dict[str, TsdModel] = {
    # DNA transposons: the transposase fixes the stagger.
    "tcmar": TsdModel(0.95, modal_len=2, motif="TA"),
    "tc1": TsdModel(0.95, modal_len=2, motif="TA"),
    "mariner": TsdModel(0.95, modal_len=2, motif="TA"),
    "hat": TsdModel(0.95, modal_len=8),
    "piggybac": TsdModel(0.95, modal_len=4, motif="TTAA"),
    "mule": TsdModel(0.90, modal_len=9, spread=0.4),
    "mudr": TsdModel(0.90, modal_len=9, spread=0.4),
    "cmc": TsdModel(0.90, modal_len=3),
    "enspm": TsdModel(0.90, modal_len=3),
    "cacta": TsdModel(0.90, modal_len=3),
    "pif": TsdModel(0.90, modal_len=3, motif="TAA"),
    "harbinger": TsdModel(0.90, modal_len=3, motif="TAA"),
    "merlin": TsdModel(0.90, modal_len=9, spread=0.4),
    "p": TsdModel(0.90, modal_len=8),
    "transib": TsdModel(0.90, modal_len=5),
    "kolobok": TsdModel(0.90, modal_len=4),
    "sola": TsdModel(0.90, modal_len=4, spread=0.5),
    "maverick": TsdModel(0.90, modal_len=6),
    "polinton": TsdModel(0.90, modal_len=6),
    "crypton": TsdModel(0.05),          # tyrosine recombinase: no TSD
    # LTR retrotransposons: the integrase makes a 4-6 bp duplication.
    "erv": TsdModel(0.95, modal_len=5, spread=0.4),
    "gypsy": TsdModel(0.95, modal_len=5, spread=0.3),
    "copia": TsdModel(0.95, modal_len=5, spread=0.3),
    "pao": TsdModel(0.95, modal_len=5, spread=0.4),
    "bel": TsdModel(0.95, modal_len=5, spread=0.4),
    "dirs": TsdModel(0.05),             # tyrosine recombinase
    "ngaro": TsdModel(0.05),
    "helitron": TsdModel(0.05),
}

CLASS_TSD: dict[TeClass, TsdModel] = {
    TeClass.LINE: TsdModel(0.90, mean_len=15.0),
    TeClass.SINE: TsdModel(0.85, mean_len=12.0),
    TeClass.RETROPOSON: TsdModel(0.85, mean_len=12.0),
    TeClass.PLE: TsdModel(0.70, mean_len=10.0),
    TeClass.LTR: TsdModel(0.95, modal_len=5, spread=0.4),
    TeClass.DNA: TsdModel(0.90, modal_len=8, spread=0.6),
    TeClass.RC: TsdModel(0.05),
    TeClass.UNKNOWN: TsdModel(0.60, mean_len=10.0),
    TeClass.NON_TE: TsdModel(0.05),
}

#: The tail a TPRT insertion ends in: present in most, geometric in length
#: above the MIN_TAIL_BP floor (Alu tails run ~10-50, L1 longer).
TAIL_P_PRESENT = 0.85
TAIL_MEAN_BP = 25.0

#: P(no library element aligns | the insert is TE-derived): what an insert
#: that aligns to nothing says against being a TE. Small for a curated
#: library; the whole term is the evidence, since there is no alignment to
#: weigh.
P_NO_HIT_GIVEN_TE = 0.05

#: See `MechanismParameters.q_young`.
Q_YOUNG = 0.95
#: The identity at which sequence that is NOT TE-derived aligns to a library
#: consensus by chance, over the >= 50 informative bases a hit must have
#: (`te_classifier.MIN_ELEMENT_ALIGNED_BP`): local alignment of unrelated
#: sequence settles around 0.65-0.75.
Q_AMBIENT = 0.70

#: TE-derived sequence spans every age, so under H_TE the identity to the
#: consensus is not one number but a distribution: a grid over
#: [Q_TE_MIN, Q_TE_MAX], weighted uniformly. (A fit of the weights to the
#: sample served only the likelihood-gated TE rule, and went with it.)
Q_TE_MIN = 0.75
Q_TE_MAX = 0.995
Q_GRID = tuple(round(Q_TE_MIN + 0.005 * i, 3) for i in range(int((Q_TE_MAX - Q_TE_MIN) / 0.005) + 1))


def log_bf_te_derived(length: int, identity: float,
                      weights: tuple[float, ...] | None = None,
                      q_null: float = Q_AMBIENT) -> float:
    """Is this TE-derived sequence, or non-TE sequence aligning by chance?

    H_TE: identity q drawn from the age distribution (`weights` over
    `Q_GRID`), matches ~ Binomial(length, q). H_nonTE: matches ~
    Binomial(length, q_null), the chance level. The binomial coefficient
    cancels. A single-q H_TE, as first written here, scored an 800 bp insert
    at 0.82 identity -57 nats against q_young = 0.95, though it is plainly
    TE-derived.
    """
    n = max(0, length)
    if n <= 0:
        return 0.0
    k = min(max(identity, 0.0), 1.0) * n
    w = weights if weights is not None else tuple(1.0 / len(Q_GRID) for _ in Q_GRID)
    terms = [math.log(wj) + k * math.log(q) + (n - k) * math.log(1.0 - q)
             for q, wj in zip(Q_GRID, w) if wj > 0.0]
    m = max(terms)
    ll_te = m + math.log(sum(math.exp(t - m) for t in terms))
    ll_null = k * math.log(q_null) + (n - k) * math.log(1.0 - q_null)
    return ll_te - ll_null


#: Weights for the robust hallmark mixtures (see the module docstring).
W_TAIL = 0.85
W_EN_MOTIF = 0.60
W_TERMINI = 0.80
W_TIR = 0.70
W_ENDS = 0.80
#: Per-base agreement with a canonical terminus in a real element.
P_TERMINUS_MATCH = 0.95
P_TIR_IDENTITY = 0.90


@dataclass
class MechanismParameters:
    """The parameter set a run scores with: literature values, sample-updated."""

    superfamily_tsd: dict[str, TsdModel] = field(
        default_factory=lambda: dict(SUPERFAMILY_TSD))
    class_tsd: dict[TeClass, TsdModel] = field(default_factory=lambda: dict(CLASS_TSD))
    tail_mean_bp: float = TAIL_MEAN_BP
    #: The sequence term asks whether the insert is TE-DERIVED: identity to
    #: its consensus under a TE insertion (`q_young`) against the identity at
    #: which non-TE sequence aligns by chance (`q_ambient`). An insertion of an
    #: old element's sequence therefore counts as a TE insertion, as GIAB's and
    #: TEBench's TE truth sets count it (RepeatMasker names it). The first
    #: version asked instead whether the insert was a YOUNG copy, against the
    #: genome's resident old copies (tprt's 0.98 / 0.88, then 0.95 / 0.80), and
    #: fitting that from the sample moved q_ambient to 0.86 on the human dev
    #: slice and rejected its truth Alus at 0.85-0.92. `q_ambient` is a
    #: property of alignment, not of the sample, and stays fixed.
    q_young: float = Q_YOUNG
    q_ambient: float = Q_AMBIENT
    #: The age distribution over `Q_GRID`; None is uniform.
    identity_weights: tuple[float, ...] | None = None
    #: How many high-confidence calls each update drew on, for the run log.
    updated_from: dict[str, int] = field(default_factory=dict)

    def tsd_model(self, te_class: TeClass, superfamily: str) -> TsdModel:
        name = (superfamily or "").strip().lower()
        best = ""
        for key in self.superfamily_tsd:
            if len(key) > len(best) and name.startswith(key) and (
                    len(key) >= 2 or name == key or not name[len(key):len(key) + 1].isalpha()):
                best = key
        if best:
            return self.superfamily_tsd[best]
        return self.class_tsd.get(te_class, CLASS_TSD[TeClass.UNKNOWN])


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass
class LocusObservation:
    """Everything the ratios read about one candidate locus."""

    te_class: TeClass = TeClass.UNKNOWN
    superfamily: str = "NA"
    #: TE alignment: identity, aligned bases, interval on the element and its
    #: length (-1 when unknown).
    identity: float = 0.0
    aligned_len: int = 0
    element_start: int = -1
    element_end: int = -1
    element_length: int = -1
    #: `core/element_structure.py` measurements.
    polya_len: int = 0
    five_prime_complete: bool = False
    three_prime_complete: bool = False
    ltr_start_matches: int = -1
    ltr_end_matches: int = -1
    tir_identity: float = -1.0
    helitron_start_matches: int = -1
    helitron_end_matches: int = -1
    #: TSD detection: length (0 = none), mismatches, sequence.
    tsd_len: int = 0
    tsd_mismatches: int = 0
    tsd_seq: str = ""
    #: L1 endonuclease motif log-odds at the junction, None = not evaluated.
    en_log_odds: float | None = None
    #: Reads, and the locus's own composition (the local nulls).
    n_alt: int = 0
    n_ref: int = 0
    insert_len: int = 0
    local_a_frac: float = 0.30
    local_t_frac: float = 0.30
    local_repeat_frac: float = 0.0
    #: The fraction of this locus's shifted breakpoints at which the same
    #: insert also finds a duplication at least `tsd_len` long -- the LOCAL,
    #: measured chance of a TSD this long here (`core/locus_evidence.py`).
    #: -1 when not measured.
    tsd_empirical_null: float = -1.0


@dataclass
class MechanismScore:
    """The two ratios, their minimum, and the terms that made them."""

    vs_non_te: float = 0.0
    vs_artifact: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return min(self.vs_non_te, self.vs_artifact)

    @property
    def e_value(self) -> float:
        return math.exp(min(self.score, 700.0))


# ---------------------------------------------------------------------------
# The terms
# ---------------------------------------------------------------------------


def robust(log_lr_present: float, weight: float) -> float:
    """`log(w * exp(log_lr_present) + (1 - w))`, computed without overflow.

    `log_lr_present` may be -inf -- an observation the hallmark cannot produce
    -- and the term is then exactly the floor log(1 - w).
    """
    w = min(max(weight, 0.0), 1.0)
    if w <= 0.0:
        return 0.0
    if log_lr_present == -math.inf:
        return math.log(1.0 - w) if w < 1.0 else -math.inf
    a = math.log(w) + log_lr_present
    b = math.log(1.0 - w) if w < 1.0 else -math.inf
    m = max(a, b)
    return m + math.log(math.exp(a - m) + (math.exp(b - m) if b > -math.inf else 0.0))


def _binomial_log_lr(matches: int, n: int, p_alt: float, p_null: float) -> float:
    """Per-base Bernoulli log-LR of `matches` agreements out of `n`."""
    if n <= 0 or matches < 0:
        return 0.0
    k = min(matches, n)
    return (k * math.log(p_alt / p_null)
            + (n - k) * math.log((1.0 - p_alt) / (1.0 - p_null)))


def tail_term(obs: LocusObservation, params: MechanismParameters) -> float:
    """poly(A) at the element's 3' end, against the locus's own A content.

    Under the null the insert's last bases are local sequence, and a run of
    `a` A's has probability f^a (1 - f). Under a TPRT insertion there is a
    tail with probability TAIL_P_PRESENT, geometric in length.
    """
    f = min(max(obs.local_a_frac, 0.05), 0.95)
    a = max(0, obs.polya_len)
    p_null = (f ** a) * (1.0 - f)
    if a == 0:
        p_tail = 1.0 - TAIL_P_PRESENT
    else:
        lam = 1.0 / max(1.0, params.tail_mean_bp)
        p_tail = TAIL_P_PRESENT * lam * math.exp(-lam * a)
    return robust(math.log(max(p_tail, 1e-300)) - math.log(max(p_null, 1e-300)), W_TAIL)


def ends_term(obs: LocusObservation) -> float:
    """Both element ends present, for classes that insert whole (LTR, DNA).

    Under the null the insert's position on the element is uniform over the
    n = L - l + 1 placements, so each end is reached with probability 1/n;
    a complete insertion reaches both. A full-length insert (n = 1) reaches
    both ends under either hypothesis, and scores 0.
    """
    if obs.element_length <= 0 or obs.element_end <= obs.element_start:
        return 0.0
    length = obs.element_end - obs.element_start
    n = max(1.0, obs.element_length - length + 1.0)
    if n <= 1.0:
        return 0.0
    # A complete insertion reaches the end with probability 1, so a missing
    # end is impossible under the hallmark and falls to the mixture's floor.
    total = 0.0
    for complete in (obs.five_prime_complete, obs.three_prime_complete):
        total += robust(math.log(n) if complete else -math.inf, W_ENDS)
    return total


def termini_term(obs: LocusObservation) -> float:
    """The class's own terminal signature, against chance at 1/4 per base."""
    if obs.te_class is TeClass.LTR and obs.ltr_start_matches >= 0:
        k = obs.ltr_start_matches + obs.ltr_end_matches
        return robust(_binomial_log_lr(k, 4, P_TERMINUS_MATCH, 0.25), W_TERMINI)
    if obs.te_class is TeClass.RC and obs.helitron_start_matches >= 0:
        k = obs.helitron_start_matches + obs.helitron_end_matches
        return robust(_binomial_log_lr(k, 6, P_TERMINUS_MATCH, 0.30), W_TERMINI)
    if obs.te_class is TeClass.DNA and obs.tir_identity >= 0.0:
        n = 20
        k = int(round(obs.tir_identity * n))
        return robust(_binomial_log_lr(k, n, P_TIR_IDENTITY, 0.25), W_TIR)
    return 0.0


def tsd_term(obs: LocusObservation, model: TsdModel) -> float:
    """The target site duplicated over `tsd_len` bp, as this mechanism does it.

    `tprt.p_null_tandem_duplication` is the chance the locus's own flanks look
    duplicated over that length, blended toward 1 by the local tandem-repeat
    fraction. A superfamily with a fixed TSD sequence (TA, TTAA, TAA) earns the
    sequence too -- against 1/4 per base under a chance duplication.
    """
    tau = max(0, obs.tsd_len)
    if tau == 0:
        return model.log_p_length(0)
    p_null = tprt.p_null_tandem_duplication(tau, max(0, obs.tsd_mismatches),
                                            obs.local_repeat_frac)
    # The analytic null assumes the insert's end and the flank are independent
    # sequence. They are not when the insert was copied from this neighbourhood
    # (a tandem duplication, a VNTR expansion) or lands beside a copy of the
    # same element, and then chance duplications are everywhere nearby: on the
    # human dev slice the shifted breakpoints scored a mean exp(linkage) of
    # 3.7e5 under 4^-tau. The locus's own shifted breakpoints measure the
    # rate, with the permutation floor 1/(n+1) -- so a TSD is worth at most
    # what n shifts can certify, a few nats, and the evidence that an
    # insertion is HERE rests mainly on the read counts.
    if obs.tsd_empirical_null > 0.0:
        p_null = max(p_null, obs.tsd_empirical_null)
    out = model.log_p_length(tau) - math.log(max(p_null, 1e-300))
    if model.motif and len(model.motif) == tau and obs.tsd_seq:
        seq = obs.tsd_seq.upper()
        hit = model.motif in (seq, reverse_complement(seq))
        out += robust(len(model.motif) * math.log(4.0) if hit else -math.inf, 0.9)
    return out


def en_term(obs: LocusObservation) -> float:
    if obs.en_log_odds is None:
        return 0.0
    return robust(obs.en_log_odds, W_EN_MOTIF)


def counts_term(obs: LocusObservation) -> float:
    eps = tprt.local_error_rate(obs.local_a_frac, obs.local_t_frac,
                                obs.local_repeat_frac)
    return tprt.log_bf_counts(obs.n_alt, obs.n_ref, float(max(1, obs.insert_len)),
                              eps=eps)


# ---------------------------------------------------------------------------
# The two ratios
# ---------------------------------------------------------------------------

#: Unknown class: a fixed prior over the mechanisms it could be.
UNKNOWN_CLASS_PRIOR: dict[TeClass, float] = {
    TeClass.LINE: 0.30, TeClass.SINE: 0.20, TeClass.LTR: 0.30,
    TeClass.DNA: 0.15, TeClass.RC: 0.05,
}

_TPRT = (TeClass.LINE, TeClass.SINE, TeClass.RETROPOSON, TeClass.PLE)


def _class_terms(obs: LocusObservation, cls: TeClass,
                 params: MechanismParameters) -> tuple[dict[str, float], dict[str, float]]:
    """(vs_non_te terms, vs_artifact terms) for one class hypothesis."""
    internal: dict[str, float] = {}
    linkage: dict[str, float] = {}
    if cls in _TPRT:
        if obs.element_length > 0 and obs.element_end > obs.element_start:
            # tprt's own test calls the 3' end complete within 5 bp. A library
            # consensus often ends in its own poly(A), which does not align
            # (Dfam's Alu does), so a complete Alu read as 3'-truncated and
            # scored -3.9. element_structure's tolerance (30 bp or 3%) decides.
            end = obs.element_length if obs.three_prime_complete else obs.element_end
            start = min(obs.element_start, end - 1)
            internal["anchoring"] = tprt.log_bf_three_prime_anchoring(
                start, end, float(obs.element_length))
        internal["tail"] = tail_term(obs, params)
        if cls in (TeClass.LINE, TeClass.SINE, TeClass.RETROPOSON):
            linkage["en_motif"] = en_term(obs)
    elif cls in (TeClass.LTR, TeClass.DNA):
        internal["ends"] = ends_term(obs)
    local = replace(obs, te_class=cls)
    internal["termini"] = termini_term(local)
    linkage["tsd"] = tsd_term(obs, params.tsd_model(cls, obs.superfamily
                                                    if cls is obs.te_class else ""))
    return internal, linkage


def score_locus(obs: LocusObservation,
                params: MechanismParameters | None = None) -> MechanismScore:
    """Both ratios for one locus."""
    params = params or MechanismParameters()
    out = MechanismScore()
    sequence = log_bf_te_derived(max(0, obs.aligned_len), obs.identity,
                                 params.identity_weights, params.q_ambient)
    counts = counts_term(obs)

    if obs.te_class is TeClass.NON_TE:
        # The best match is a satellite, an RNA gene or a simple repeat: the
        # sequence itself says "not a TE". A TE that happens to align best to
        # one is allowed for at 1 in 100, and no mechanism term applies.
        non_te = math.log(0.01)
        out.terms = {"sequence": sequence, "counts": counts, "non_te_class": non_te}
        out.vs_non_te = min(sequence, 0.0) + non_te
        out.vs_artifact = counts
        return out

    if obs.aligned_len <= 0 or obs.identity <= 0.0:
        # No element aligned. The hallmarks (a tail, TG...CA, TIRs) are
        # evidence about WHICH element was inserted, and there is none: on the
        # cichlid dev slice the Unknown-class mixture's tail and termini terms
        # alone gave unaligned inserts +10 to +23 nats and made TE calls of
        # them. The TE question rests on the sequence term; the insertion can
        # still be a structural call on the artifact question.
        linkage = tsd_term(obs, params.tsd_model(TeClass.UNKNOWN, ""))
        no_hit = math.log(P_NO_HIT_GIVEN_TE)
        out.terms = {"no_te_alignment": no_hit, "counts": counts, "tsd": linkage}
        out.vs_non_te = no_hit
        out.vs_artifact = counts + linkage
        return out

    if obs.te_class is TeClass.UNKNOWN:
        internal_mix, linkage_mix = [], []
        for cls, prior in UNKNOWN_CLASS_PRIOR.items():
            internal, linkage = _class_terms(obs, cls, params)
            internal_mix.append(math.log(prior) + sum(internal.values()))
            linkage_mix.append(math.log(prior) + sum(linkage.values()))
        internal_total = _logsumexp(internal_mix)
        linkage_total = _logsumexp(linkage_mix)
        out.terms = {"sequence": sequence, "counts": counts,
                     "internal_mixture": internal_total, "linkage_mixture": linkage_total}
    else:
        internal, linkage = _class_terms(obs, obs.te_class, params)
        internal_total = sum(internal.values())
        linkage_total = sum(linkage.values())
        out.terms = {"sequence": sequence, "counts": counts, **internal, **linkage}

    out.vs_non_te = sequence + internal_total
    out.vs_artifact = counts + linkage_total
    return out


def _logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))
