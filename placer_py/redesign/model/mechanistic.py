"""Mechanistic (biophysical) model of the TE insertion process.

Rather than scoring a candidate only on read support and generic sequence
coverage, this module scores it against the *physics of retrotransposition*
(target-primed reverse transcription, TPRT) and, more generally, the co-occurrence
of independent, mechanism-specific hallmarks. Each hallmark is individually
noisy, but their conjunction is highly specific to a genuine TE insertion, which
is what lets a strict gate reach high precision.

Model blocks (each a log-likelihood ratio of "TE insertion" vs a null model):

* Endonuclease target motif. The L1 ORF2p endonuclease nicks the bottom strand
  at a degenerate 5'-TTTT|AA-3' consensus, creating the 3'-OH that primes reverse
  transcription. We score the reference sequence spanning the junction against a
  position-weight model; the log-odds vs a uniform background is proportional to
  the binding free energy (Boltzmann: P(cut) ~ exp(-E/kT)).
  Refs: NAR 2021 10.1093/nar/gkab826 (EN sequence recognition); Cell 2021 L1 TPRT structure.

* Target site duplication. Staggered top/bottom-strand nicking (+7/+11/+19 nt)
  produces TSDs of ~7-20 bp (up to ~60). TSD length in range plus low background
  probability is strong evidence.
  Refs: Cell 2021 structural mechanism of L1 TPRT.

* Poly(A) tail. TPRT primes from the L1/Alu poly(A) tail; retrotransposition rate
  rises sigmoidally with tail length (lag < 15 nt, plateau > 50 nt).
  Refs: Genomics 2005 10.1016/j.ygeno.2005.05.009 (Alu poly-A length).

* TE-body specificity. A family-resolved TE alignment over an adequate core
  length. 5'-truncation is expected (RT processivity survival), so partial
  3'-anchored coverage is not penalised.
  Refs: Genome Res 2005 15:780 (5' junctions / truncation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


# L1 endonuclease consensus spanning the nick: 5'-TTTT|AA-3' (bottom-strand nick).
_EN_CONSENSUS = "TTTTAA"
_EN_MATCH_P = 0.85
_EN_MISMATCH_P = 0.05
_BG_P = 0.25


def _pwm_logodds(window: str) -> float:
    """Log-odds of a 6 bp window under a degenerate TTTTAA model vs uniform."""
    score = 0.0
    for base, consensus in zip(window, _EN_CONSENSUS):
        p = _EN_MATCH_P if base == consensus else _EN_MISMATCH_P
        score += math.log(p / _BG_P)
    return score


def endonuclease_motif_score(left_flank: str, right_flank: str) -> tuple[float, bool]:
    """Best endonuclease log-odds over both strands at the insertion junction.

    ``left_flank`` ends at the breakpoint; ``right_flank`` starts at it. Returns
    ``(score, evaluated)``; ``evaluated`` is False when flanks are too short.
    """
    left = (left_flank or "").upper()
    right = (right_flank or "").upper()
    if len(left) < 4 or len(right) < 2:
        return 0.0, False
    forward = left[-4:] + right[:2]
    # Minus-strand insertion: the consensus appears reverse-complemented.
    reverse = _revcomp(right[:4] + left[-2:])
    scores = []
    if len(forward) == 6:
        scores.append(_pwm_logodds(forward))
    if len(reverse) == 6:
        scores.append(_pwm_logodds(reverse))
    if not scores:
        return 0.0, False
    return max(scores), True


def tsd_loglr(tsd_len: int, tsd_significant: bool, tsd_bg_p: float) -> float:
    """Log-LR for a target site duplication under staggered-nick TPRT geometry."""
    if tsd_len <= 0:
        return -0.4
    in_range = 4 <= tsd_len <= 25
    score = 0.8 if in_range else -0.4
    if tsd_significant:
        score += 1.0
    score += _clamp(-math.log10(max(tsd_bg_p, 1e-6)) * 0.3, 0.0, 1.5)
    return _clamp(score, -1.0, 3.0)


def polya_loglr(polya_fraction: float, insert_len: int) -> float:
    """Log-LR for a poly(A) tail; sigmoid in estimated tail length (nt)."""
    tail = _clamp(polya_fraction, 0.0, 1.0) * max(0, insert_len)
    sigmoid = 1.0 / (1.0 + math.exp(-(tail - 18.0) / 8.0))
    return 1.6 * sigmoid - 0.3


def te_body_loglr(te_core_fraction: float, identity: float, core_len: float) -> float:
    """Log-LR for a family-resolved TE body of adequate length and identity."""
    core = _clamp(te_core_fraction, 0.0, 1.0)
    ident = _clamp(identity, 0.0, 1.0)
    if core <= 0.0 or ident <= 0.0:
        return -1.0
    length_factor = _clamp(core_len / 120.0, 0.0, 1.0)
    return (2.5 * core * length_factor) + (1.5 * (ident - 0.7))


# Calibrated against the D2/TLDR reference (see scripts/calibrate_mechanistic_vs_tldr.py):
# hp_score >= 3500 with coverage >= 0.80 yields ~0.94 precision vs TLDR-detected
# TE loci (~0.88 vs the strict TLDR PASS subset).
DEFAULT_HP_MIN = 3500.0
DEFAULT_HP_COV_MIN = 0.80


def high_precision_te_score(identity: float, te_core_fraction: float, insert_len: int) -> float:
    """Expected number of correctly-aligned, non-low-complexity TE-derived bases.

    ``identity * te_core_fraction * insert_len`` is proportional to the alignment
    log-likelihood under a substitution model: more high-identity aligned bases
    are exponentially less likely to occur by chance, so this is a well-separated
    statistic for a genuine, family-resolved TE body. It favours long, full-length
    elements and, being 5'-truncation-tolerant, does not require full coverage.
    """
    return _clamp(identity, 0.0, 1.0) * _clamp(te_core_fraction, 0.0, 1.0) * max(0, insert_len)


def is_confident_te_insertion(
    hp_score: float,
    coverage: float,
    family_resolved: bool,
    hp_min: float = DEFAULT_HP_MIN,
    cov_min: float = DEFAULT_HP_COV_MIN,
) -> bool:
    """High-precision TE emission gate (precision-first tier)."""
    return family_resolved and hp_score >= hp_min and coverage >= cov_min


@dataclass(frozen=True)
class MechanisticScore:
    loglr: float
    hallmark_count: int
    endonuclease_score: float
    tsd_loglr: float
    polya_loglr: float
    te_body_loglr: float
    qc: str


def mechanistic_te_score(
    *,
    endonuclease: float,
    endonuclease_evaluated: bool,
    tsd_len: int,
    tsd_significant: bool,
    tsd_bg_p: float,
    polya_fraction: float,
    insert_len: int,
    te_core_fraction: float,
    identity: float,
    core_len: float,
) -> MechanisticScore:
    """Combine mechanistic blocks into a log-LR and a hallmark count.

    ``hallmark_count`` counts independent positive signatures (endonuclease
    motif, TSD, poly(A) tail, TE body). Requiring several to co-occur is the
    precision lever: no single block is decisive, their conjunction is.
    """
    en_block = 0.0
    hallmarks = 0
    if endonuclease_evaluated and endonuclease > 2.0:
        en_block = _clamp(endonuclease, 0.0, 6.0) * 0.45
        hallmarks += 1

    tsd_block = tsd_loglr(tsd_len, tsd_significant, tsd_bg_p)
    if tsd_block >= 0.8:
        hallmarks += 1

    polya_block = polya_loglr(polya_fraction, insert_len)
    if polya_block >= 0.5:
        hallmarks += 1

    te_block = te_body_loglr(te_core_fraction, identity, core_len)
    if te_block >= 1.0:
        hallmarks += 1

    loglr = en_block + tsd_block + polya_block + te_block

    if hallmarks >= 3:
        qc = "MECH_TPRT_STRONG"
    elif hallmarks == 2:
        qc = "MECH_TPRT_SUPPORTED"
    elif hallmarks == 1:
        qc = "MECH_SINGLE_HALLMARK"
    else:
        qc = "MECH_NONE"
    return MechanisticScore(
        loglr=loglr,
        hallmark_count=hallmarks,
        endonuclease_score=endonuclease if endonuclease_evaluated else 0.0,
        tsd_loglr=tsd_block,
        polya_loglr=polya_block,
        te_body_loglr=te_block,
        qc=qc,
    )
