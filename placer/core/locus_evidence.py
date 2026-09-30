"""From one evaluated hypothesis to the observation `core/mechanism.py` scores.

It gathers what the per-class likelihood ratios read -- the TE alignment and
its class, the element structure, the TSD, the endonuclease motif, the read
counts, and the locus's own composition -- and runs the SHIFTED-BREAKPOINT
DECOYS that check the ratios on this genome.

WHAT A DECOY MEASURES. The same insert and the same reads, with the breakpoint
moved to where there is no insertion (`core/null_control.py`). Only the terms
tied to the LOCUS change -- the TSD and the endonuclease motif, both read off the
reference flanks -- so a decoy measures how often the flanks alone produce the
coincidences those terms reward. That is exactly what the terms' nulls assume
(a chance duplication at 4^-tau, bases at the local composition), and the
assumption most likely to be wrong on a new genome: in a VNTR or an A-rich
tract coincidences are the norm. The terms that depend only on the insert
(sequence, tail, termini) do not move with the breakpoint and are not tested
here; nor are the counts, which at a decoy would simply be zero and make the
check pass for the wrong reason.

The decoys are summarised per locus (count and mean of exp(linkage)), not
written as rows: finalization still treats every ledger row as part of the
run's null set, and decoy rows would enter it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from placer.core import element_structure as structure_module
from placer.core import mechanism as mech
from placer.core import tprt
from placer.core.endonuclease import endonuclease_motif_log_odds
from placer.core.null_control import make_breakpoint_shift_controls
from placer.core.seqtools import microsatellite_mask
from placer.core.taxonomy import TeClass

#: Reference bases either side of the breakpoint that define the LOCAL null:
#: composition and tandem-repeat content. Wide enough to hold a TSD search
#: (`tsd_flank_window` is 60) and a few repeat units.
LOCAL_WINDOW_BP = 100
#: Shifted breakpoints per locus, their spacing, and how far they may range:
#: near enough to be the same neighbourhood, far enough apart (> the longest
#: TSD searched) that two shifts do not re-find one duplication. They serve
#: twice -- as the locus's own empirical TSD null, and as its decoys.
DECOY_COUNT = 100
DECOY_STEP_BP = 30
DECOY_RANGE_BP = 1500

_EN_CLASSES = (TeClass.LINE, TeClass.SINE, TeClass.RETROPOSON)


@dataclass
class LocusScore:
    observation: mech.LocusObservation
    score: mech.MechanismScore
    decoy_count: int = 0
    #: Mean over the decoys of exp(linkage terms): <= 1 when the TSD and motif
    #: nulls hold at this locus.
    decoy_mean_exp_linkage: float = 0.0
    #: RECORDED FOR REPLAY, read by no decision: what a change of the TSD
    #: models' `p_present` would need to be replayed exactly.
    #:   * `tsd_p_present`: the p of the model behind the locus's own "tsd"
    #:     term; -1 when it has none, or a mixture of them (Unknown class).
    #:   * the decoys, split by whether they found a duplication. Every decoy
    #:     shares one model (`decoy_tsd_p_present`), and its exp(linkage) is
    #:     max(1e-6, 1 - p) * exp(en) without a TSD and p * exp(tsd - log p + en)
    #:     with one, so the mean at any p is
    #:     (max(1e-6, 1 - p) * sum_absent + p * sum_present_per_p) / count.
    tsd_p_present: float = -1.0
    decoy_tsd_p_present: float = -1.0
    decoy_tsd_hits: int = -1
    decoy_sum_absent: float = 0.0
    decoy_sum_present_per_p: float = 0.0


def _te_class(value: str) -> TeClass:
    try:
        return TeClass(value)
    except ValueError:
        return TeClass.UNKNOWN


def local_composition(flank: str) -> tuple[float, float]:
    """(A fraction, T fraction) of a reference window."""
    seq = flank.upper()
    # str.count per base rather than a generator over the characters: this
    # runs at every one of a locus's 101 breakpoints (itself and its decoys).
    a, t = seq.count("A"), seq.count("T")
    acgt = a + seq.count("C") + seq.count("G") + t
    if acgt == 0:
        return 0.30, 0.30
    return a / acgt, t / acgt


def repeat_fraction_at(window: str, start: int, end: int) -> float:
    """Fraction of `window[start:end]` inside a simple repeat.

    The TSD null asks whether THESE bases could look duplicated by chance, and
    they can if they themselves sit in a tandem array. Measuring the whole
    neighbourhood instead was wrong: 8% of dust-masked bases somewhere in a
    200 bp window blended a clean 15 bp duplication's null from 4^-15 up to
    0.08 and erased it.

    Exact tandem arrays only (`microsatellite_mask`), not the dust-style
    windows `simple_repeat_mask` adds: those mark whole low-complexity
    windows, far coarser than a TSD span.
    """
    if not window or end <= start:
        return 0.0
    mask = microsatellite_mask(window.upper())
    lo, hi = max(0, start), min(len(mask), end)
    if hi <= lo:
        return 0.0
    return sum(mask[lo:hi]) / (hi - lo)


def insert_without_tsd_copy(insert_seq: str, tsd_seq: str, mismatches: int) -> str:
    """The insert with the second TSD copy removed, when it carries one.

    A single-position (CIGAR I) insertion carries one copy of the target site
    at one of its ends (`tsd.detect_from_insertion`), and on the plus strand
    it is AFTER the poly(A), so the tail no longer reaches the insert's end.
    """
    n = len(tsd_seq)
    if n == 0 or len(insert_seq) <= n:
        return insert_seq
    seq, tsd = insert_seq.upper(), tsd_seq.upper()

    def close(a: str) -> bool:
        return sum(1 for x, y in zip(a, tsd) if x != y) <= max(0, mismatches)

    if close(seq[-n:]):
        return insert_seq[:-n]
    if close(seq[:n]):
        return insert_seq[n:]
    return insert_seq


def _linkage_at(chrom: str, bp_left: int, bp_right: int, insert_seq: str,
                obs: mech.LocusObservation, params: mech.MechanismParameters,
                fetch_reference: Callable[[str, int, int], str],
                detect_tsd) -> tuple[mech.LocusObservation, float]:
    """Fill the locus-tied observables at (bp_left, bp_right); return the
    observation and its linkage log-LR (TSD + endonuclease motif)."""
    lo = max(0, min(bp_left, bp_right) - LOCAL_WINDOW_BP)
    hi = max(bp_left, bp_right) + LOCAL_WINDOW_BP
    window = fetch_reference(chrom, lo, hi) or ""
    obs.local_a_frac, obs.local_t_frac = local_composition(window)

    obs.tsd_len, obs.tsd_mismatches, obs.tsd_seq = 0, 0, ""
    obs.local_repeat_frac = 0.0
    if detect_tsd is not None and bp_left >= 0:
        detection = detect_tsd(chrom, bp_left, bp_right, insert_seq)
        # UNCERTAIN is a duplication whose sequence is common nearby: the
        # likelihood's own null (the repeat content of the TSD span) weighs
        # that, so it is scored rather than discarded.
        if (detection is not None and detection.type in ("DUP", "UNCERTAIN")
                and detection.length > 0):
            obs.tsd_len = detection.length
            obs.tsd_mismatches = detection.mismatches
            obs.tsd_seq = detection.sequence or ""
            cut = min(bp_left, bp_right) - lo
            obs.local_repeat_frac = repeat_fraction_at(
                window, cut - obs.tsd_len, max(cut, max(bp_left, bp_right) - lo)
                + obs.tsd_len)

    obs.en_log_odds = None
    if obs.te_class in _EN_CLASSES and window:
        cut = min(bp_left, bp_right) - lo
        found = endonuclease_motif_log_odds(window[:cut], window[cut:])
        obs.en_log_odds = found[0] if found is not None else None

    linkage = mech.tsd_term(obs, params.tsd_model(obs.te_class, obs.superfamily))
    if obs.te_class in _EN_CLASSES:
        linkage += mech.en_term(obs)
    return obs, linkage


def _own_tsd_p_present(score: mech.MechanismScore, obs: mech.LocusObservation,
                       params: mech.MechanismParameters) -> float:
    """The p of the TSD model behind `score_locus`'s own "tsd" term."""
    if "tsd" not in score.terms:
        return -1.0
    if "no_te_alignment" in score.terms:
        return params.tsd_model(TeClass.UNKNOWN, "").p_present
    return params.tsd_model(obs.te_class, obs.superfamily).p_present


def allele_counts_term(chrom: str, span_left: int, span_right: int, n_alt: int,
                       n_ref: int, insert_len: int,
                       fetch_reference: Callable[[str, int, int], str]) -> float:
    """The counts term (`mechanism.counts_term`) over an allele's whole span.

    The error rate is the span's own: its composition and its tandem-repeat
    content over [span_left, span_right], so an allele gathered across a VNTR
    is weighed against the higher artifact rate a VNTR has. RECORDED FOR
    REPLAY (`events.collect_allele_evidence`); no decision reads it.
    """
    lo = max(0, span_left - LOCAL_WINDOW_BP)
    hi = span_right + LOCAL_WINDOW_BP
    window = fetch_reference(chrom, lo, hi) or ""
    a_frac, t_frac = local_composition(window)
    repeat = repeat_fraction_at(window, span_left - lo, span_right - lo)
    eps = tprt.local_error_rate(a_frac, t_frac, repeat)
    return tprt.log_bf_counts(max(0, n_alt), max(0, n_ref), float(max(1, insert_len)),
                              eps=eps)


def score_evaluated_locus(chrom: str, bp_left: int, bp_right: int,
                          insert_seq: str, te_alignment, n_alt: int, n_ref: int,
                          fetch_reference: Callable[[str, int, int], str],
                          detect_tsd=None,
                          params: mech.MechanismParameters | None = None,
                          with_decoys: bool = True) -> LocusScore:
    """The mechanism score of one evaluated hypothesis, and its decoys."""
    params = params or mech.MechanismParameters()
    structure = te_alignment.element_structure
    aligned = 0
    if te_alignment.te_query_end > te_alignment.te_query_start >= 0:
        aligned = te_alignment.te_query_end - te_alignment.te_query_start
    obs = mech.LocusObservation(
        te_class=_te_class(te_alignment.annotation_class),
        superfamily=te_alignment.annotation_order,
        identity=te_alignment.best_identity if te_alignment.best_identity else 0.0,
        aligned_len=aligned,
        element_start=te_alignment.te_consensus_start,
        element_end=te_alignment.te_consensus_end,
        element_length=te_alignment.te_element_length,
        polya_len=structure.polya_len,
        five_prime_complete=structure.five_prime_complete,
        three_prime_complete=structure.three_prime_complete,
        ltr_start_matches=structure.ltr_start_matches,
        ltr_end_matches=structure.ltr_end_matches,
        tir_identity=structure.tir_identity,
        helitron_start_matches=structure.helitron_start_matches,
        helitron_end_matches=structure.helitron_end_matches,
        n_alt=max(0, n_alt), n_ref=max(0, n_ref), insert_len=len(insert_seq or ""))
    if not getattr(te_alignment, "pass_", False) or not te_alignment.best_family \
            or te_alignment.best_family in ("NA", "UNKNOWN"):
        obs.identity, obs.aligned_len = 0.0, 0
    obs, _ = _linkage_at(chrom, bp_left, bp_right, insert_seq, obs, params,
                         fetch_reference, detect_tsd)
    if obs.tsd_len > 0:
        # Re-measure the tail on the insert without its TSD copy.
        trimmed = insert_without_tsd_copy(insert_seq, obs.tsd_seq, obs.tsd_mismatches)
        if trimmed != insert_seq:
            obs.polya_len = max(obs.polya_len, structure_module.measure(
                trimmed, te_alignment.annotation_class, te_alignment.te_strand).polya_len)
    if not with_decoys or bp_left < 0:
        score = mech.score_locus(obs, params)
        return LocusScore(observation=obs, score=score,
                          tsd_p_present=_own_tsd_p_present(score, obs, params))

    # The shifted breakpoints: the same insert and reads, the breakpoint moved.
    width = max(0, bp_right - bp_left)
    controls = make_breakpoint_shift_controls(
        bp_left, bp_right, max(0, bp_left - DECOY_RANGE_BP),
        bp_right + DECOY_RANGE_BP, DECOY_STEP_BP, DECOY_COUNT)
    shifted: list[mech.LocusObservation] = []
    for control in controls:
        decoy = mech.LocusObservation(**obs.__dict__)
        _linkage_at(chrom, control.bp_left, control.bp_left + width, insert_seq,
                    decoy, params, fetch_reference, detect_tsd)
        shifted.append(decoy)
    lengths = [decoy.tsd_len for decoy in shifted]

    def rate_at_least(tau: int, exclude: int = -1) -> float:
        """(1 + hits) / (n + 1): the permutation estimate, never below 1/(n+1).

        With n shifts the locus can only certify a duplication as rarer than
        1/(n+1). Letting a zero count fall back to the analytic 4^-tau -- as
        the first version did -- claimed 1e-12 at loci where 1 shift in 40
        found a duplication, and the decoys scored a mean exp(linkage) of
        2.9e10 for SVA and 125 for LTR elements.
        """
        pool = [n for i, n in enumerate(lengths) if i != exclude]
        if tau <= 0 or not pool:
            return -1.0
        hits = sum(1 for n in pool if n >= tau)
        return (1 + hits) / (1 + len(pool))

    obs.tsd_empirical_null = rate_at_least(obs.tsd_len)
    out = LocusScore(observation=obs, score=mech.score_locus(obs, params))
    out.tsd_p_present = _own_tsd_p_present(out.score, obs, params)

    # Each shift as a decoy, scored exactly as the locus is -- against the
    # empirical null of the OTHER shifts (leave-one-out), so the check tests
    # the statistic actually used rather than the analytic one.
    values = []
    hits, absent, present = 0, 0.0, 0.0
    for index, decoy in enumerate(shifted):
        decoy.tsd_empirical_null = rate_at_least(decoy.tsd_len, exclude=index)
        model = params.tsd_model(decoy.te_class, decoy.superfamily)
        tsd = mech.tsd_term(decoy, model)
        linkage = tsd
        en = 0.0
        if decoy.te_class in _EN_CLASSES:
            en = mech.en_term(decoy)
            linkage += en
        values.append(math.exp(min(linkage, 700.0)))
        # The decomposition `LocusScore` documents; it changes nothing above.
        out.decoy_tsd_p_present = model.p_present
        if decoy.tsd_len > 0:
            hits += 1
            present += math.exp(min(tsd - math.log(max(1e-6, model.p_present)) + en, 700.0))
        else:
            absent += math.exp(min(en, 700.0))
    if values:
        out.decoy_count = len(values)
        out.decoy_mean_exp_linkage = sum(values) / len(values)
        out.decoy_tsd_hits = hits
        out.decoy_sum_absent = absent
        out.decoy_sum_present_per_p = present
    return out
