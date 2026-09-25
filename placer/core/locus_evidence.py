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
from placer.core.endonuclease import endonuclease_motif_log_odds
from placer.core.null_control import make_breakpoint_shift_controls
from placer.core.seqtools import microsatellite_mask
from placer.core.taxonomy import TeClass

#: Reference bases either side of the breakpoint that define the LOCAL null:
#: composition and tandem-repeat content. Wide enough to hold a TSD search
#: (`tsd_flank_window` is 60) and a few repeat units.
LOCAL_WINDOW_BP = 100
#: Decoys per locus, their spacing, and how far they may range. Far enough that
#: a decoy's flanks share nothing with the real junction; near enough that the
#: composition is the same neighbourhood's.
DECOY_COUNT = 4
DECOY_STEP_BP = 150
DECOY_RANGE_BP = 1200

_EN_CLASSES = (TeClass.LINE, TeClass.SINE, TeClass.RETROPOSON)


@dataclass
class LocusScore:
    observation: mech.LocusObservation
    score: mech.MechanismScore
    decoy_count: int = 0
    #: Mean over the decoys of exp(linkage terms): <= 1 when the TSD and motif
    #: nulls hold at this locus.
    decoy_mean_exp_linkage: float = 0.0


def _te_class(value: str) -> TeClass:
    try:
        return TeClass(value)
    except ValueError:
        return TeClass.UNKNOWN


def local_composition(flank: str) -> tuple[float, float]:
    """(A fraction, T fraction) of a reference window."""
    seq = flank.upper()
    acgt = sum(1 for base in seq if base in "ACGT")
    if acgt == 0:
        return 0.30, 0.30
    return seq.count("A") / acgt, seq.count("T") / acgt


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
    out = LocusScore(observation=obs, score=mech.score_locus(obs, params))
    if not with_decoys or bp_left < 0:
        return out

    width = max(0, bp_right - bp_left)
    controls = make_breakpoint_shift_controls(
        bp_left, bp_right, max(0, bp_left - DECOY_RANGE_BP),
        bp_right + DECOY_RANGE_BP, DECOY_STEP_BP, DECOY_COUNT)
    values = []
    for control in controls:
        decoy = mech.LocusObservation(**obs.__dict__)
        _, linkage = _linkage_at(chrom, control.bp_left, control.bp_left + width,
                                 insert_seq, decoy, params, fetch_reference, detect_tsd)
        values.append(math.exp(min(linkage, 700.0)))
    if values:
        out.decoy_count = len(values)
        out.decoy_mean_exp_linkage = sum(values) / len(values)
    return out
