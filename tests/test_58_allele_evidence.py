"""
The allele-level tally (`events.collect_allele_evidence`): recorded for replay,
read by no decision.

On synthetic reads, so every count is countable. The shape is the one that
loses dispersed insertions on HG002: one insertion that the aligner placed at
several offsets in different reads, where the +-25 bp tally counts the carriers
it cannot see as reference.
"""

from __future__ import annotations

import random

import pytest

from placer.alignment import CIGAR_I, CIGAR_M, CIGAR_S, AlignedRead
from placer.core import clustering as C
from placer.core import events as E
from placer.core.ledger import EvidenceLedgerRow

pytestmark = pytest.mark.invariant

REFERENCE = "".join(random.Random(23).choice("ACGT") for _ in range(20000))
INSERT = "".join(random.Random(5).choice("ACGT") for _ in range(320))
OTHER = "".join(random.Random(6).choice("ACGT") for _ in range(320))
SITE = 10000


def carrier(qname, offset, insert=INSERT, pieces=1, start=9000, end=11000):
    """A read carrying `insert` at SITE + offset, in `pieces` CIGAR insertions
    5 bp apart (an ONT alignment's broken insertion)."""
    site = SITE + offset
    size = len(insert) // pieces
    cigar, seq = [(CIGAR_M, site - start)], REFERENCE[start:site]
    ref = site
    for k in range(pieces):
        part = insert[k * size:(k + 1) * size if k < pieces - 1 else len(insert)]
        cigar.append((CIGAR_I, len(part)))
        seq += part
        if k < pieces - 1:
            cigar.append((CIGAR_M, 5))
            seq += REFERENCE[ref:ref + 5]
            ref += 5
    cigar.append((CIGAR_M, end - ref))
    seq += REFERENCE[ref:end]
    return AlignedRead(qname=qname, tid=0, pos=start, mapq=60, cigar=cigar, seq=seq)


def reference(qname, start=9000, end=11000):
    return AlignedRead(qname=qname, tid=0, pos=start, mapq=60,
                       cigar=[(CIGAR_M, end - start)], seq=REFERENCE[start:end])


def group(offset, n=8, **kwargs):
    return [carrier(f"c{offset}_{i}", offset, start=9000 + i * 3, end=11000 - i * 3, **kwargs)
            for i in range(n)]


def refs(n=3):
    return [reference(f"ref{i}") for i in range(n)]


def allele(reads, bp=SITE):
    component = C.ComponentCall(chrom="chr1", tid=0, anchor_pos=bp)
    evidence = E.collect_event_read_evidence_for_bounds(
        component, reads, E.read_reference_spans(reads), [], bp, bp)
    return evidence, E.collect_allele_evidence(component, reads, evidence)


def test_carriers_within_the_linkage_are_one_allele_and_never_reference():
    evidence, found = allele(group(-80) + group(0) + group(80) + refs())
    # The +-25 bp tally: the offset-0 reads are alt, and the 16 carriers 80 bp
    # away span its window with no signal inside +-75 bp, so they are "reference".
    assert (evidence.alt_struct_reads, evidence.ref_span_reads) == (8, 19)
    assert found.length == 320
    tally = found.by_length
    assert (tally.alt_reads, tally.ref_reads, tally.extra_carriers) == (24, 3, 16)
    assert (tally.span_lo, tally.span_hi) == (-80, 80)
    assert (found.by_sequence.alt_reads, found.wide.alt_reads) == (24, 24)


def test_the_linkage_stops_at_a_wide_gap_and_the_wide_tally_does_not():
    evidence, found = allele(group(-200) + group(0) + group(200) + refs())
    own = (evidence.alt_struct_reads, evidence.ref_span_reads)
    assert (found.by_length.alt_reads, found.by_length.ref_reads) == own
    assert found.by_length.extra_carriers == 0
    assert (found.wide.alt_reads, found.wide.ref_reads, found.wide.extra_carriers) == (24, 3, 16)
    assert (found.wide.span_lo, found.wide.span_hi) == (-200, 200)


def test_no_carrier_beyond_the_own_reads_leaves_every_tally_as_the_row_s_own():
    evidence, found = allele(group(0) + refs())
    for tally in (found.by_length, found.by_sequence, found.wide):
        assert tally.extra_carriers == 0
        assert (tally.alt_reads, tally.ref_reads) == (evidence.alt_struct_reads,
                                                      evidence.ref_span_reads)
        assert (tally.span_lo, tally.span_hi) == (0, 0)
    assert found.carrier_own == [1] * 8


def test_the_pieces_of_one_broken_insertion_are_one_carrier():
    """160 + 160 bp, 5 bp apart: neither piece is of the allele's length alone."""
    _, found = allele(group(0) + group(80, pieces=2) + refs())
    assert found.carrier_lengths.count(320) == 16
    assert found.by_length.extra_carriers == 8


def test_another_sequence_of_the_same_length_is_not_the_same_allele_by_sequence():
    evidence, found = allele(group(0) + group(80, insert=OTHER) + refs())
    assert found.by_length.extra_carriers == 8
    assert found.by_sequence.extra_carriers == 0
    assert found.by_sequence.alt_reads == evidence.alt_struct_reads
    others = [s for s, own in zip(found.carrier_similarity, found.carrier_own) if not own]
    assert others and max(others) < E.ALLELE_MIN_SIMILARITY
    assert min(s for s, own in zip(found.carrier_similarity, found.carrier_own) if own) == 1.0


def test_allele_reference_must_span_the_whole_allele():
    short = reference("short", start=9950)          # spans the +-25 bp window only
    evidence, found = allele(group(-80) + group(0) + group(80) + refs() + [short])
    assert evidence.ref_span_reads == 20
    assert found.by_length.ref_reads == 3


def test_a_carrier_outside_the_allele_is_not_reference_either():
    """The linkage leaves the -200 group out of the allele, but those reads
    carry an insertion of its length: they are not evidence of the reference."""
    _, found = allele(group(-200, n=4) + group(-80) + group(0) + refs())
    assert found.by_length.extra_carriers == 8
    assert found.by_length.ref_reads == 3


def test_without_a_measured_length_nothing_is_computed():
    clip = AlignedRead(qname="clip", tid=0, pos=9000, mapq=60,
                       cigar=[(CIGAR_M, SITE - 9000), (CIGAR_S, 300)],
                       seq=REFERENCE[9000:SITE] + INSERT[:300])
    evidence, found = allele([clip] + refs())
    assert not evidence.alt_measured_lengths
    assert found.length == -1
    assert found.by_length.alt_reads == -1 and found.carrier_offsets == []


def test_recorded_fields_do_not_take_part_in_row_equality():
    """The per-bin de-duplication compares rows field by field. The recorded
    observables must not change which rows it keeps, or the calls would move."""
    a = EvidenceLedgerRow(chrom="chr1", pos=1000)
    b = EvidenceLedgerRow(chrom="chr1", pos=1000, allele_length=320,
                          allele_carrier_offsets=[0, 80], allele_wide_alt_reads=24,
                          mech_tsd_len=14, mech_decoy_sum_absent=3.5)
    assert a == b


# ------------------------------------------- the counts term's local null, measured
def test_a_clean_neighbourhood_shows_no_background_signal():
    _, found = allele(group(-80) + group(0) + group(80) + refs())
    tally = found.by_length
    assert tally.background_reads > 0 and tally.background_hits == 0


def test_the_allele_s_insertion_beyond_the_carrier_window_is_its_background():
    """Reads with the allele's insertion 700 bp away are outside the carrier
    window, so not carriers: they are what the allele's signal looks like where
    the insertion is not, as in a VNTR where every read carries something."""
    _, found = allele(group(-80) + group(0) + group(80) + refs() + group(700, n=6))
    assert found.by_length.extra_carriers == 16
    assert found.by_length.background_hits == 6
    # Of another sequence, they are background by length only.
    _, other = allele(group(-80) + group(0) + group(80) + refs()
                      + group(700, n=6, insert=OTHER))
    assert other.by_length.background_hits == 6
    assert other.by_sequence.background_hits == 0


def test_a_tally_that_is_the_row_s_own_has_no_background_of_its_own():
    _, found = allele(group(0) + refs())
    assert (found.by_length.background_reads, found.by_length.background_hits) == (-1, -1)


def test_the_own_background_counts_a_clip_as_the_alt_tally_would():
    """A read clipped inside a background window does not span it, and is
    counted all the same: the +-25 bp tally counts clipped reads as alt."""
    component = C.ComponentCall(chrom="chr1", tid=0, anchor_pos=SITE)
    clipped = [AlignedRead(qname=f"clip{i}", tid=0, pos=9000, mapq=60,
                           cigar=[(CIGAR_M, 430), (CIGAR_S, 300)],
                           seq=REFERENCE[9000:9430] + INSERT[:300]) for i in range(4)]
    for reads, hits in ((group(0) + refs(), 0), (group(0) + refs() + clipped, 4)):
        evidence = E.collect_event_read_evidence_for_bounds(
            component, reads, E.read_reference_spans(reads), [], SITE, SITE)
        counted, found = E.collect_own_background(component, reads, evidence)
        assert found == hits and counted >= 11 * 8
