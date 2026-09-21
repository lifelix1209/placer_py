"""
Target-site duplication detection, across the geometries the caller actually
produces.

THIS FILE EXISTS BECAUSE THE DETECTOR HAD NO TEST. `placer_py/tsd.py`'s
docstring cited `tests/test_20_tsd.py`, which has never existed -- the
numbering was reshuffled once and the reference was never updated. The gap was
not cosmetic: `detect` reported every genuine duplication as a DELETION, and
the whole-run consequence (tsd_type=NONE on every call) was invisible because
nothing asserted otherwise.

WHAT THE GEOMETRIES ARE. A novel insertion leaves the target site duplicated
in the SAMPLE; the reference still carries it once. How that reaches the
detector depends on what the aligner did:

  * split/clipped reads resolve both edges, so the two breakpoints OVERLAP by
    tau and `detect` can measure the overlap against the reference;
  * a single CIGAR `I` collapses both edges onto one coordinate, so the
    reference retains no trace and the duplication is only in the inserted
    bases -- `detect_from_insertion`.

A small deletion is the same shape with the opposite sign (a GAP rather than
an overlap), which is why the sign must survive: normalising it away is what
made every TSD look like a deletion.

No golden vectors here, for the reason `tsd.py` gives: the C++ detector reads
the reference through a faidx handle and cannot be linked into the oracle
dumper. These are hand-constructed cases whose answer is determinable by
counting.
"""

from __future__ import annotations

import random

import pytest
from conftest import call_or_skip

from placer_py.tsd import TsdConfig, detect, detect_from_insertion, fetcher_from_string

#: Long enough that the background window is fully inside the reference and
#: the occurrence rate of a 15 bp motif is genuinely low. A short reference
#: makes every detection UNCERTAIN for reasons that have nothing to do with
#: the geometry under test.
FLANK_BP = 3000
TSD_SEQ = "ACGTTGCAAGGCTTA"


def _config() -> TsdConfig:
    return TsdConfig(tsd_min_len=3, tsd_max_len=50, tsd_flank_window=150,
                     tsd_bg_p_max=0.05, tsd_max_mismatch_rate=0.10,
                     tsd_max_mismatches=2)


def _scene(seed: int = 7):
    """Reference carrying the target site ONCE, plus an unrelated element."""
    rng = random.Random(seed)
    def seq(n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))
    left, right = seq(FLANK_BP), seq(FLANK_BP)
    reference = left + TSD_SEQ + right
    element = seq(300)
    return reference, len(left), len(TSD_SEQ), element


@pytest.mark.invariant
def test_overlapping_breakpoints_are_a_duplication():
    """The split-read geometry: right_bp < left_bp, overlapping by tau.

    REGRESSION. An unconditional `if left_bp > right_bp: swap` used to stand
    at the top of `detect`, which made `right_bp - left_bp` non-negative
    always and sent every case here into the DELETION branch instead.
    """
    reference, pos, tau, _ = _scene()
    fetch = fetcher_from_string(reference)
    found = call_or_skip(detect, fetch, "chr1", pos + tau, pos, _config())
    assert found.type == "DUP", f"expected DUP, got {found.type}"
    assert found.length == tau
    assert found.sequence == TSD_SEQ
    assert found.significant


@pytest.mark.invariant
def test_gapped_breakpoints_are_still_a_deletion():
    """The opposite sign must keep its old meaning.

    Removing the swap is only correct if it does not turn deletions into
    duplications, so this pins the branch the fix could plausibly have broken.
    """
    reference, pos, tau, _ = _scene()
    fetch = fetcher_from_string(reference)
    found = call_or_skip(detect, fetch, "chr1", pos, pos + tau, _config())
    assert found.type == "DEL", f"expected DEL, got {found.type}"
    assert found.length == tau


@pytest.mark.invariant
def test_cigar_insertion_carrying_the_three_prime_copy():
    """`S = element + TSD`, the placement TPRT actually produces."""
    reference, pos, tau, element = _scene()
    fetch = fetcher_from_string(reference)
    found = call_or_skip(detect_from_insertion, fetch, "chr1", pos + tau,
                         element + TSD_SEQ, _config())
    assert found.type == "DUP"
    assert found.length == tau
    assert found.sequence == TSD_SEQ


@pytest.mark.invariant
def test_cigar_insertion_left_shifted_by_the_aligner():
    """`S = TSD + element`: the same event, the other tie-break.

    An aligner is free to place the insertion on either side of the duplicated
    bases. Both spellings must give the same answer, or the call depends on
    the aligner rather than the genome.
    """
    reference, pos, tau, element = _scene()
    fetch = fetcher_from_string(reference)
    found = call_or_skip(detect_from_insertion, fetch, "chr1", pos,
                         TSD_SEQ + element, _config())
    assert found.type == "DUP"
    assert found.length == tau
    assert found.sequence == TSD_SEQ


@pytest.mark.invariant
def test_insertion_without_a_duplication_reports_none():
    """The false-positive guard: an element whose ends match neither flank."""
    reference, pos, tau, element = _scene()
    fetch = fetcher_from_string(reference)
    found = call_or_skip(detect_from_insertion, fetch, "chr1", pos + tau,
                         element, _config())
    assert found.type == "NONE"
    assert found.length == 0


@pytest.mark.invariant
def test_detect_from_insertion_needs_the_inserted_sequence():
    """No sequence, no claim -- rather than falling back to a reference-only
    comparison that cannot see a novel duplication anyway."""
    reference, pos, tau, _ = _scene()
    fetch = fetcher_from_string(reference)
    found = call_or_skip(detect_from_insertion, fetch, "chr1", pos + tau, "",
                         _config())
    assert found.type == "NONE"
