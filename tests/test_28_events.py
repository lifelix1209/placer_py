"""
Reads into counts, and reads into one event sequence.

Covers `placer/core/events.py` (the alt/ref tally) and `placer/core/consensus.py`
(the event strings and their consensus). The C++ has partial coverage in
`test_event_read_evidence.cpp`; the rules that matter most here -- one read one
vote, the two slack bands, the clip partner rule -- are pinned directly.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip

from placer.alignment import CIGAR_I, CIGAR_M, CIGAR_S, AlignedRead
from placer.config import PipelineConfig
from placer.core import clustering as C
from placer.core import consensus as K
from placer.core import events as E
from placer.core import fragments as F

pytestmark = pytest.mark.invariant

M, I, S = CIGAR_M, CIGAR_I, CIGAR_S


def read(qname, pos=1000, cigar=None, mapq=60, tid=0, seq=None, flag=0, sa=None):
    cigar = cigar or [(M, 500), (I, 300), (M, 500)]
    if seq is None:
        seq = "A" * sum(n for op, n in cigar if op in (M, I, S))
    return AlignedRead(qname=qname, tid=tid, pos=pos, mapq=mapq, cigar=cigar,
                       seq=seq, flag=flag, tags=({"SA": sa} if sa else {}))


def alt_read(qname, pos=1000):
    return read(qname, pos)


def ref_read(qname, pos=1000, mapq=60):
    return read(qname, pos, cigar=[(M, 1000)], mapq=mapq)


def component(anchor=1500, chrom="chr1", tid=0, candidates=()):
    call = C.ComponentCall(chrom=chrom, tid=tid, anchor_pos=anchor)
    call.breakpoint_candidates = list(candidates)
    return call


def tally(records, fragments=(), bp_left=1500, bp_right=1500, comp=None):
    comp = comp or component()
    return E.collect_event_read_evidence_for_bounds(
        comp, list(records), E.read_reference_spans(list(records)),
        list(fragments), bp_left, bp_right)


def fragment(read_id, ref_junc_pos, source, start=0, length=300, sequence=None):
    return F.InsertionFragment(read_id=read_id, ref_junc_pos=ref_junc_pos,
                               source=source, start=start, length=length,
                               sequence=sequence if sequence is not None else "C" * length)


# ------------------------------------------------------------- counting reads
def test_one_read_carrying_four_signals_is_one_alt_read():
    """
    THE rule the whole stage is arranged around. A count of SIGNALS would let a
    chimeric read manufacture its own support, and that is the dominant false
    positive at a repetitive locus.
    """
    # Left clip at 1475, insertion at 1500, right clip at 1525 -- three signals
    # inside the alt window, from one read.
    busy = read("busy", 1475, cigar=[(S, 200), (M, 25), (I, 300), (M, 25), (S, 200)],
                seq="A" * 750)
    evidence = call_or_skip(tally, [busy])
    assert evidence.alt_left_clip_reads == 1
    assert evidence.alt_right_clip_reads == 1
    assert evidence.alt_indel_reads == 1
    assert evidence.alt_struct_reads == 1
    assert evidence.support_qnames == ["busy"]


def test_alt_and_ref_reads_are_tallied_separately():
    evidence = tally([alt_read(f"alt{i}") for i in range(4)]
                     + [ref_read(f"ref{i}") for i in range(3)])
    assert evidence.alt_struct_reads == 4
    assert evidence.alt_indel_reads == 4
    assert evidence.ref_span_reads == 3
    assert evidence.support_qnames == ["alt0", "alt1", "alt2", "alt3"]


def test_support_qnames_are_sorted_so_two_runs_agree():
    """Finalization compares support sets between calls; an unordered list would
    make those comparisons depend on hash order."""
    evidence = tally([alt_read(name) for name in ("zeta", "alpha", "mu")])
    assert evidence.support_qnames == ["alpha", "mu", "zeta"]


def test_an_insertion_from_a_non_unique_read_is_not_precise_support():
    """
    The same MAPQ-60 equality the geometry stage applies, for the same reason:
    a CIGAR insertion is the aligner asserting a position, and the assertion is
    worthless if the aligner was unsure where the read goes.
    """
    evidence = tally([read("ambiguous", mapq=59)])
    assert evidence.alt_indel_reads == 0
    assert evidence.raw_cigar_insert_reads == 1     # still recorded as raw support


def test_raw_insert_length_is_recorded_even_below_the_precise_floor():
    evidence = tally([read("short", cigar=[(M, 500), (I, 30), (M, 500)])])
    assert evidence.max_raw_cigar_insert_len == 30
    assert evidence.alt_indel_reads == 0
    assert evidence.alt_struct_reads == 0


# ------------------------------------------------------------- the two bands
def test_alt_support_is_hypothesis_specific_at_twenty_five_bases():
    inside = tally([read("near", pos=1000, cigar=[(M, 480), (I, 300), (M, 500)])])
    assert inside.alt_indel_reads == 1
    outside = tally([read("far", pos=1000, cigar=[(M, 400), (I, 300), (M, 500)])])
    assert outside.alt_indel_reads == 0


def test_the_reference_exclusion_band_is_wider_than_the_alt_band():
    """
    Clipped noise 50 bp from the breakpoint is not clean reference. Counting it
    as such would inflate the denominator and suppress a real call -- so the
    disqualifying band is 75 bp where the qualifying one is 25.
    """
    assert E.REF_SIGNAL_SLACK_BP > E.ALT_SIGNAL_SLACK_BP
    noisy = read("noisy", pos=1000, cigar=[(M, 440), (S, 200)], seq="A" * 640)
    evidence = tally([noisy, ref_read("clean")])
    assert evidence.ref_span_reads == 1
    assert "clean" in evidence.ref_span_qnames


def test_a_read_must_strictly_span_the_window_to_be_reference_support():
    """
    Merely overlapping the breakpoint is not evidence that the reference allele
    is present there.
    """
    spanning = ref_read("spanning", pos=1000)          # 1000..2000
    # 1490..1590 overlaps the breakpoint but starts inside the alt window.
    overlapping = read("overlapping", pos=1490, cigar=[(M, 100)], seq="A" * 100)
    evidence = tally([spanning, overlapping])
    assert evidence.ref_span_qnames == ["spanning"]


def test_low_mapq_reference_reads_are_counted_apart_rather_than_discarded():
    """They are evidence, but not evidence the genotyper should divide by."""
    evidence = tally([ref_read("good", mapq=60), ref_read("poor", mapq=19)])
    assert evidence.ref_span_reads == 1
    assert evidence.low_mapq_ref_span_reads == 1


def test_an_alt_read_is_never_also_a_reference_read():
    evidence = tally([alt_read("both")])
    assert evidence.alt_struct_reads == 1
    assert evidence.ref_span_reads == 0


def test_supplementary_records_are_not_reference_support():
    supplementary = ref_read("supp")
    supplementary.flag = 0x800
    assert tally([supplementary]).ref_span_reads == 0


# --------------------------------------------------------- the clip partner rule
def test_one_sided_clips_are_reported_but_do_not_enter_the_genotype_denominator():
    """
    Clips on one side alone are what a mapping artifact produces. They stay in
    `alt_left_clip_reads`, where the mechanistic blocks can see and discount
    them, and out of `alt_struct_reads`, which the genotyper divides by.
    """
    left_only = [read(f"left{i}", pos=1500, cigar=[(S, 200), (M, 500)],
                      seq="A" * 700) for i in range(3)]
    evidence = tally(left_only)
    assert evidence.alt_left_clip_reads == 3
    assert evidence.alt_right_clip_reads == 0
    assert evidence.alt_struct_reads == 0


def test_clips_on_both_sides_do_enter_the_denominator():
    reads = [read(f"left{i}", pos=1500, cigar=[(S, 200), (M, 500)], seq="A" * 700)
             for i in range(2)]
    reads += [read(f"right{i}", pos=1000, cigar=[(M, 500), (S, 200)], seq="A" * 700)
              for i in range(2)]
    evidence = tally(reads)
    assert evidence.alt_left_clip_reads == 2
    assert evidence.alt_right_clip_reads == 2
    assert evidence.alt_struct_reads == 4


def test_precise_support_lets_one_sided_clips_in():
    reads = [alt_read("precise")]
    reads += [read(f"left{i}", pos=1500, cigar=[(S, 200), (M, 500)], seq="A" * 700)
              for i in range(2)]
    evidence = tally(reads)
    assert evidence.alt_struct_reads == 3


# ----------------------------------------------------- the clip supplement rule
def test_all_three_conditions_are_needed_to_admit_nearby_clips():
    """
    Clips slightly outside the tight window are admitted only when the event
    already has PRECISE support AND the clips appear on BOTH sides. Any one of
    those alone would admit ordinary clip noise.
    """
    candidates = [C.BreakpointCandidate(pos=1560)]
    comp = component(anchor=1500, candidates=candidates)
    precise = alt_read("precise")
    left = read("nearby_left", pos=1560, cigar=[(S, 200), (M, 500)], seq="A" * 700)
    right = read("nearby_right", pos=1060, cigar=[(M, 500), (S, 200)], seq="A" * 700)

    both = tally([precise, left, right], comp=comp)
    assert both.alt_left_clip_reads == 1 and both.alt_right_clip_reads == 1

    one_side = tally([precise, left], comp=comp)
    assert one_side.alt_left_clip_reads == 0


def test_fragments_contribute_support_by_read_name_too():
    frags = [fragment("fragread", 1500, F.InsertionFragmentSource.SPLIT_SA)]
    evidence = tally([], frags)
    assert evidence.alt_split_reads == 1
    assert evidence.support_qnames == ["fragread"]


def test_a_fragment_outside_the_alt_window_contributes_nothing():
    frags = [fragment("distant", 9000, F.InsertionFragmentSource.SPLIT_SA)]
    assert tally([], frags).alt_struct_reads == 0


# --------------------------------------------------------------- event strings
def _clip_read(qname="clip", insert_len=300, flank=200):
    seq = "C" * insert_len + "A" * flank
    return AlignedRead(qname=qname, tid=0, pos=1000, mapq=60,
                       cigar=[(S, insert_len), (M, flank)], seq=seq)


def _ins_read(qname="ins", left=200, insert_len=300, right=200):
    seq = "G" * left + "C" * insert_len + "T" * right
    return AlignedRead(qname=qname, tid=0, pos=1000, mapq=60,
                       cigar=[(M, left), (I, insert_len), (M, right)], seq=seq)


def test_an_event_string_carries_flanks_so_the_consensus_can_be_segmented():
    """
    Not the fragment and not the read: `flank + insert + flank`. A consensus of
    bare inserts has no ends to align back to the reference, so segmentation
    could never find the junctions.
    """
    record = _ins_read()
    frag = fragment("ins", 1200, F.InsertionFragmentSource.CIGAR_INSERTION,
                    start=200, length=300, sequence="C" * 300)
    event_string = call_or_skip(K.build_event_string_from_fragment, record, frag)
    assert len(event_string) == 80 + 300 + 80
    assert event_string.startswith("G" * 80)
    assert event_string.endswith("T" * 80)


def test_a_clip_fragment_can_only_ever_give_one_flank():
    """
    Mechanical, not policy: the insert runs off the end of the read, so the
    other side is not in the record at all.
    """
    record = _clip_read()
    frag = fragment("clip", 1000, F.InsertionFragmentSource.CLIP_REF_LEFT,
                    start=0, length=300, sequence="C" * 300)
    event_string = K.build_event_string_from_fragment(record, frag)
    assert event_string == "C" * 300 + "A" * 80
    assert not K.fragment_has_full_event_context(frag)


def test_a_read_without_enough_anchor_is_unusable():
    short_anchor = _clip_read(flank=40)
    frag = fragment("clip", 1000, F.InsertionFragmentSource.CLIP_REF_LEFT,
                    start=0, length=300, sequence="C" * 300)
    assert not call_or_skip(K.can_build_event_string_from_fragment, short_anchor, frag)
    assert K.build_event_string_from_fragment(short_anchor, frag) == ""


def test_a_flank_shorter_than_the_target_but_over_the_minimum_is_accepted():
    """Checked against the READ's extent, not the configured length: 60 bp of
    available flank clears the 50 bp minimum even though the target is 80."""
    record = _clip_read(flank=60)
    frag = fragment("clip", 1000, F.InsertionFragmentSource.CLIP_REF_LEFT,
                    start=0, length=300, sequence="C" * 300)
    assert K.can_build_event_string_from_fragment(record, frag)
    assert len(K.build_event_string_from_fragment(record, frag)) == 360


def test_the_anchor_counts_are_cross_wired_and_that_is_correct():
    """
    A `CLIP_REF_LEFT` fragment increments the RIGHT anchor count. The name says
    which side of the REFERENCE the read's aligned part sits on; the anchor it
    provides is on the opposite side of the INSERT. Inverting this would break
    the pre-segmentation gate's bilateral test.
    """
    record = _clip_read("clipL")
    frag = fragment("clipL", 1500, F.InsertionFragmentSource.CLIP_REF_LEFT,
                    start=0, length=300, sequence="C" * 300)
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500,
                                   support_qnames=["clipL"])
    summary = call_or_skip(K.collect_event_consensus_inputs, [record], [frag], evidence)
    assert summary.right_anchor_input_reads == 1
    assert summary.left_anchor_input_reads == 0
    assert summary.partial_context_input_reads == 1


def test_only_reads_in_the_support_set_contribute_event_strings():
    record = _ins_read("stranger")
    frag = fragment("stranger", 1500, F.InsertionFragmentSource.CIGAR_INSERTION,
                    start=200, length=300, sequence="C" * 300)
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500, support_qnames=[])
    assert K.collect_event_consensus_inputs([record], [frag], evidence).full_event_by_qname == {}


def test_fragment_locality_here_is_tighter_than_at_extraction():
    """
    25 bp against the extractor's 1 kb. Extraction can afford to be generous --
    a wrong fragment costs one alignment -- while a wrong event string corrupts
    the consensus that everything downstream reads.
    """
    assert K.fragment_is_local_to_event(
        fragment("r", 1520, F.InsertionFragmentSource.CIGAR_INSERTION), 1500, 1500)
    assert not K.fragment_is_local_to_event(
        fragment("r", 1600, F.InsertionFragmentSource.CIGAR_INSERTION), 1500, 1500)


def test_the_longest_record_per_read_name_is_used():
    """The event string is cut from the record's own bases, so a hard-clipped
    supplementary carries fewer of them."""
    short = AlignedRead(qname="r", seq="A" * 100, cigar=[(M, 100)])
    long_ = AlignedRead(qname="r", seq="A" * 900, cigar=[(M, 900)])
    assert call_or_skip(K.best_records_by_qname, [short, long_])["r"] is long_
    assert K.best_records_by_qname([long_, short])["r"] is long_


# ------------------------------------------------------------- the consensus
def _consensus(records, fragments, **kw):
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500,
                                   support_qnames=sorted({f.read_id for f in fragments}))
    return K.build_event_consensus(records, fragments, evidence, PipelineConfig(), **kw)


def test_a_single_full_context_read_is_its_own_consensus():
    record = _ins_read("ins")
    frag = fragment("ins", 1500, F.InsertionFragmentSource.CIGAR_INSERTION,
                    start=200, length=300, sequence="C" * 300)
    result = call_or_skip(_consensus, [record], [frag])
    assert result.qc_pass
    assert result.qc_reason == "PASS_EVENT_CONSENSUS"
    assert result.used_full_context
    assert result.input_event_reads == 1
    assert len(result.consensus_seq) == 460


def test_full_context_is_preferred_wholesale_and_not_mixed():
    """
    One full-context read beats twenty clip-derived ones. Mixing strings that
    begin at DIFFERENT points in the event produces a smear rather than a
    consensus, so the clip strings are discarded entirely.
    """
    records = [_ins_read("ins")] + [_clip_read(f"clip{i}") for i in range(4)]
    frags = [fragment("ins", 1500, F.InsertionFragmentSource.CIGAR_INSERTION,
                      start=200, length=300, sequence="C" * 300)]
    frags += [fragment(f"clip{i}", 1500, F.InsertionFragmentSource.CLIP_REF_LEFT,
                       start=0, length=300, sequence="C" * 300) for i in range(4)]
    result = _consensus(records, frags)
    assert result.used_full_context
    assert result.input_event_reads == 1
    assert result.full_context_input_reads == 1
    assert result.partial_context_input_reads == 4


def test_partial_only_mode_exists_so_the_caller_can_retry():
    records = [_ins_read("ins")] + [_clip_read(f"clip{i}") for i in range(2)]
    frags = [fragment("ins", 1500, F.InsertionFragmentSource.CIGAR_INSERTION,
                      start=200, length=300, sequence="C" * 300)]
    frags += [fragment(f"clip{i}", 1500, F.InsertionFragmentSource.CLIP_REF_LEFT,
                       start=0, length=300, sequence="C" * 300) for i in range(2)]
    result = _consensus(records, frags, mode=K.ConsensusContextMode.PARTIAL_ONLY)
    assert not result.used_full_context
    assert result.qc_reason == "PASS_EVENT_CONSENSUS_PARTIAL_RETRY"
    assert result.input_event_reads == 2


def test_one_full_context_read_suffices_but_partial_needs_the_configured_minimum():
    """
    A full-context string is already the whole event; several partial views have
    to agree before their overlap means anything.
    """
    records = [_clip_read("clip0")]
    frags = [fragment("clip0", 1500, F.InsertionFragmentSource.CLIP_REF_LEFT,
                      start=0, length=300, sequence="C" * 300)]
    result = _consensus(records, frags, mode=K.ConsensusContextMode.PARTIAL_ONLY)
    assert not result.qc_pass
    assert result.qc_reason == "INSUFFICIENT_EVENT_READS"


def test_no_usable_read_is_reported_as_such():
    result = _consensus([], [])
    assert result.qc_reason == "NO_EVENT_STRING_READS"


def test_differing_strings_refuse_rather_than_approximating_a_consensus():
    """
    THE most important refusal in this port. A longest-input or majority-vote
    stand-in would change the insert sequence without changing any QC field, and
    the run would look clean while every call was subtly wrong. So the default
    consensus raises, and a real alignment has to be supplied deliberately.
    """
    assert call_or_skip(K.single_sequence_consensus, ["ACGT"]) == "ACGT"
    assert K.single_sequence_consensus(["ACGT", "ACGT"]) == "ACGT"
    assert K.single_sequence_consensus([]) == ""
    with pytest.raises(K.ConsensusUnavailable):
        K.single_sequence_consensus(["ACGT", "ACGA"])


def test_a_supplied_consensus_function_is_used():
    records = [_clip_read(f"clip{i}") for i in range(2)]
    frags = [fragment(f"clip{i}", 1500, F.InsertionFragmentSource.CLIP_REF_LEFT,
                      start=0, length=300, sequence="C" * 300) for i in range(2)]
    result = _consensus(records, frags, mode=K.ConsensusContextMode.PARTIAL_ONLY,
                        consensus_fn=lambda seqs: "ACGT" * 25)
    assert result.qc_pass
    assert result.consensus_seq == "ACGT" * 25


def test_inputs_are_sorted_longest_first_before_the_cap():
    """Truncating to `event_consensus_poa_max_reads` then drops the LEAST
    informative strings rather than an arbitrary subset."""
    seen: list[list[str]] = []

    records = [_clip_read(f"clip{i}", flank=100 + 10 * i) for i in range(3)]
    frags = [fragment(f"clip{i}", 1500, F.InsertionFragmentSource.CLIP_REF_LEFT,
                      start=0, length=300, sequence="C" * 300) for i in range(3)]
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500,
                                   support_qnames=[f"clip{i}" for i in range(3)])
    config = PipelineConfig(event_consensus_poa_max_reads=2)

    def capture(seqs):
        seen.append(list(seqs))
        return seqs[0]

    K.build_event_consensus(records, frags, evidence, config,
                            mode=K.ConsensusContextMode.PARTIAL_ONLY,
                            consensus_fn=capture)
    assert len(seen[0]) == 2
    assert len(seen[0][0]) >= len(seen[0][1])


# ------------------------------------------------- clip / insert concordance
def _clip_fragment(read_id, source, sequence, anchor_len=200, nm=2):
    return F.InsertionFragment(read_id=read_id, source=source, sequence=sequence,
                               length=len(sequence), ref_junc_pos=1500,
                               anchor_len=anchor_len, nm=nm)


def _segmentation(insert_seq):
    from placer.core.segmentation import EventSegmentation

    return EventSegmentation(insert_seq=insert_seq, pass_=True,
                             qc_reason="PASS_EVENT_SEGMENTATION")


def test_a_clip_whose_bases_do_not_match_the_insert_is_not_concordant():
    """
    A clip counts as alt support on POSITIONAL grounds -- the read stops
    explaining the reference near the breakpoint -- which says nothing about
    whether the clipped bases are the insertion. This compares them directly.
    """
    insert = "ACGTGACTTGCAAGTCCATGGATCCAGTTACG" * 4
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500,
                                   support_qnames=["full", "clip"])
    fragments = [
        _clip_fragment("full", F.InsertionFragmentSource.CIGAR_INSERTION, insert),
        _clip_fragment("clip", F.InsertionFragmentSource.CLIP_REF_LEFT,
                       "TTTTTTTTGGGGGGGG" * 8),
    ]
    result = call_or_skip(K.analyze_clip_insert_concordance, evidence,
                          _segmentation(insert), fragments, PipelineConfig())
    assert result.left_clip_reads == 0
    assert not result.pass_


def test_a_matching_clip_is_concordant_at_the_right_end_of_the_insert():
    """
    Not symmetric: a `CLIP_REF_LEFT` fragment's bases are the START of the
    insert (the read ran off the front), so they are compared against the
    insert's start.
    """
    insert = "ACGTGACTTGCAAGTCCATGGATCCAGTTACG" * 4
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500,
                                   support_qnames=["full", "left", "right"])
    fragments = [
        _clip_fragment("full", F.InsertionFragmentSource.CIGAR_INSERTION, insert),
        _clip_fragment("left", F.InsertionFragmentSource.CLIP_REF_LEFT, insert[:60]),
        _clip_fragment("right", F.InsertionFragmentSource.CLIP_REF_RIGHT, insert[-60:]),
    ]
    result = K.analyze_clip_insert_concordance(evidence, _segmentation(insert),
                                               fragments, PipelineConfig())
    assert result.left_clip_reads == 1
    assert result.right_clip_reads == 1
    assert result.pass_
    assert result.qc == "PASS_CLIP_INSERT_CONCORDANCE"


def test_concordance_needs_a_read_that_actually_spans_the_event():
    """
    Without one there is no assembled insert to compare the clips against, only
    the clips' own consensus -- which would make the test circular.
    """
    insert = "ACGTGACTTGCAAGTCCATGGATCCAGTTACG" * 4
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500,
                                   support_qnames=["left", "right"])
    fragments = [
        _clip_fragment("left", F.InsertionFragmentSource.CLIP_REF_LEFT, insert[:60]),
        _clip_fragment("right", F.InsertionFragmentSource.CLIP_REF_RIGHT, insert[-60:]),
    ]
    result = K.analyze_clip_insert_concordance(evidence, _segmentation(insert),
                                               fragments, PipelineConfig())
    assert result.full_insert_reads == 0
    assert not result.pass_


def test_a_short_insert_cannot_be_compared_at_all():
    evidence = E.EventReadEvidence(bp_left=1500, bp_right=1500, support_qnames=["a"])
    result = K.analyze_clip_insert_concordance(evidence, _segmentation("ACGT" * 5),
                                               [], PipelineConfig())
    assert result.qc == "CLIP_INSERT_CONCORDANCE_INSERT_TOO_SHORT"


def test_a_poorly_anchored_clip_is_excluded_before_it_is_compared():
    """
    A short anchor cannot locate the junction, and a high NM rate means the
    alignment that produced the clip is itself unreliable -- the clip may be an
    artefact of a bad alignment rather than a real breakpoint.
    """
    config = PipelineConfig()
    short_anchor = _clip_fragment("a", F.InsertionFragmentSource.CLIP_REF_LEFT,
                                  "ACGT" * 20, anchor_len=5)
    assert call_or_skip(K.is_low_quality_softclip_anchor, short_anchor, config)

    noisy = _clip_fragment("a", F.InsertionFragmentSource.CLIP_REF_LEFT,
                           "ACGT" * 20, anchor_len=100, nm=50)
    assert K.is_low_quality_softclip_anchor(noisy, config)

    clean = _clip_fragment("a", F.InsertionFragmentSource.CLIP_REF_LEFT,
                           "ACGT" * 20, anchor_len=100, nm=2)
    assert not K.is_low_quality_softclip_anchor(clean, config)


def test_an_unmeasured_anchor_is_not_treated_as_a_missing_one():
    """`anchor_len == 0` means "not measured", not "no anchor"; refusing on it
    would discard every fragment from a source that does not set it."""
    unmeasured = _clip_fragment("a", F.InsertionFragmentSource.CLIP_REF_LEFT,
                                "ACGT" * 20, anchor_len=0, nm=500)
    assert not K.is_low_quality_softclip_anchor(unmeasured, PipelineConfig())


def test_a_local_component_is_reindexed_against_the_fetched_reads():
    """
    NECESSARY, because a component's read indices point into the BIN's record
    list and the local fetch returns a different, wider one. Passing the
    original to the fragment extractor would make it read the wrong records --
    silently, since both lists are just integers.
    """
    component = C.ComponentCall(chrom="chr1", tid=0, anchor_pos=1500,
                                read_indices=[0, 1])
    local_records = [read("x", cigar=[(M, 1000)]),
                     read("clipped", cigar=[(M, 500), (S, 300)]),
                     read("inserted", cigar=[(M, 500), (I, 300), (M, 500)])]
    local = call_or_skip(K.build_local_fragment_component, component, local_records,
                         PipelineConfig())
    assert local.read_indices == [0, 1, 2]
    assert local.soft_clip_read_indices == [1]
    assert local.insertion_read_indices == [2]
    assert local.chrom == "chr1" and local.anchor_pos == 1500
