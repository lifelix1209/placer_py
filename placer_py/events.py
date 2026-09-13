"""
Reads into counts: the alt/ref tally the genotyper and the count model read.

Ported from `src/pipeline/pipeline_event_evidence_stage.inc`, pinned by
`tests/test_28_events.py`.

EVERY COUNT IS A COUNT OF READ NAMES, not of signals. That single decision is
what the whole stage is arranged around: one read carrying a split, an insertion
and clips at both ends contributes ONE alt read, not four. A count of signals
would let a chimeric read manufacture its own support, and that is the dominant
false positive at a repetitive locus.

TWO SLACK BANDS, and the asymmetry is deliberate:

    alt signal   +/- 25 bp    hypothesis-specific
    ref signal   +/- 75 bp    a wider EXCLUSION band

A read is alt support only if its signal sits within 25 bp of the hypothesis --
tight, because the hypothesis is being tested. But a read is disqualified from
being reference support if it carries ANY event signal within 75 bp -- wide,
because clipped noise near the breakpoint is not clean reference and counting it
as such would inflate the denominator and suppress a real call.

THE CLIP PARTNER RULE. Clip reads join the alt set only when there is either
precise support (a split or an insertion) or clips on BOTH sides. Clips on one
side alone are what a mapping artifact produces, so they are counted in
`alt_left_clip_reads` -- where the mechanistic blocks can see them -- but kept
out of `alt_struct_reads`, which is what the genotyper divides by.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from placer_py.alignment import AlignedRead, compute_ref_end
from placer_py.breakpoints import classify_local_event_signal, read_has_local_event_signal
from placer_py.clustering import INSERTION_CANDIDATE_REQUIRED_MAPQ, ComponentCall
from placer_py.fragments import InsertionFragment, InsertionFragmentSource

#: How close a signal must be to the hypothesis to be alt support.
ALT_SIGNAL_SLACK_BP = 25
#: How close a signal must be to disqualify a read as reference support.
REF_SIGNAL_SLACK_BP = 75
#: Tolerance when matching a clip to a component breakpoint candidate.
CANDIDATE_CLIP_SLACK_BP = 25
#: How far a component candidate may sit from the hypothesis and still lend it
#: clip reads.
CANDIDATE_CLIP_SUPPLEMENT_MAX_OFFSET_BP = 75
#: A reference-spanning read below this MAPQ is counted separately rather than
#: discarded -- it is evidence, but not evidence the genotyper should trust.
REF_SPAN_MIN_MAPQ = 20


@dataclass
class ReadReferenceSpan:
    """Where a read covers the reference, precomputed once per window."""

    valid: bool = False
    tid: int = -1
    start: int = -1
    end: int = -1


@dataclass
class EventReadEvidence:
    bp_left: int = -1
    bp_right: int = -1
    alt_split_reads: int = 0
    alt_indel_reads: int = 0
    alt_left_clip_reads: int = 0
    alt_right_clip_reads: int = 0
    alt_struct_reads: int = 0
    raw_cigar_insert_reads: int = 0
    max_raw_cigar_insert_len: int = 0
    ref_span_reads: int = 0
    low_mapq_ref_span_reads: int = 0
    #: Sorted, so two runs of the same locus produce identical output and the
    #: support-sharing tests in finalization are deterministic.
    support_qnames: list[str] = field(default_factory=list)
    ref_span_qnames: list[str] = field(default_factory=list)


def read_reference_spans(records: list[AlignedRead]) -> list[ReadReferenceSpan]:
    """Precompute each record's reference span, in bin index order."""
    spans: list[ReadReferenceSpan] = []
    for read in records:
        if read is None or not read.cigar:
            spans.append(ReadReferenceSpan())
            continue
        spans.append(ReadReferenceSpan(valid=True, tid=read.tid, start=read.pos,
                                       end=compute_ref_end(read)))
    return spans


def collect_event_read_evidence_for_bounds(component: ComponentCall,
                                           local_records: list[AlignedRead],
                                           read_spans: list[ReadReferenceSpan],
                                           fragments: list[InsertionFragment],
                                           bp_left: int, bp_right: int
                                           ) -> EventReadEvidence:
    """Tally alt and reference support for ONE breakpoint hypothesis.

    Called once per hypothesis, which is why it takes the bounds rather than
    reading them off the component: the hypotheses compete on the evidence each
    of them gathers, and a hypothesis in the wrong place collects less.
    """
    evidence = EventReadEvidence()
    left = min(bp_left, bp_right)
    right = max(bp_left, bp_right)
    evidence.bp_left = left
    evidence.bp_right = right

    split_qnames: set[str] = set()
    indel_qnames: set[str] = set()
    left_clip_qnames: set[str] = set()
    right_clip_qnames: set[str] = set()
    raw_cigar_insert_qnames: set[str] = set()
    nearby_left_clip_qnames: set[str] = set()
    nearby_right_clip_qnames: set[str] = set()

    alt_signal_start = max(0, left - ALT_SIGNAL_SLACK_BP)
    alt_signal_end = max(alt_signal_start, right + ALT_SIGNAL_SLACK_BP)
    ref_signal_start = max(0, left - REF_SIGNAL_SLACK_BP)
    ref_signal_end = max(ref_signal_start, right + REF_SIGNAL_SLACK_BP)

    # Component breakpoint candidates within 75 bp of THIS hypothesis. They
    # define a second, wider clip window -- see the supplement rule below.
    nearby_candidates = sorted({
        candidate.pos for candidate in component.breakpoint_candidates
        if candidate.pos >= 0
        and ((left - candidate.pos) if candidate.pos < left
             else ((candidate.pos - right) if candidate.pos > right else 0))
        <= CANDIDATE_CLIP_SUPPLEMENT_MAX_OFFSET_BP})
    have_nearby_candidates = bool(nearby_candidates)
    candidate_clip_start = 0
    candidate_clip_end = 0
    if have_nearby_candidates:
        candidate_clip_start = max(0, nearby_candidates[0] - CANDIDATE_CLIP_SLACK_BP)
        candidate_clip_end = max(candidate_clip_start,
                                 nearby_candidates[-1] + CANDIDATE_CLIP_SLACK_BP)

    def matches_nearby_candidate(pos: int) -> bool:
        return pos >= 0 and any(abs(pos - candidate) <= CANDIDATE_CLIP_SLACK_BP
                                for candidate in nearby_candidates)

    for read in local_records:
        if read is None or read.tid != component.tid or not read.qname:
            continue
        qname = read.qname
        signal = classify_local_event_signal(read, component.chrom, alt_signal_start,
                                             alt_signal_end)
        if signal.split:
            split_qnames.add(qname)
        # An insertion is believed as PRECISE evidence only from a uniquely
        # mapped read -- the same equality test the geometry stage applies, and
        # for the same reason.
        if signal.indel and read.mapq == INSERTION_CANDIDATE_REQUIRED_MAPQ:
            indel_qnames.add(qname)
        if signal.max_raw_cigar_insert_len > 0:
            raw_cigar_insert_qnames.add(qname)
            evidence.max_raw_cigar_insert_len = max(evidence.max_raw_cigar_insert_len,
                                                    signal.max_raw_cigar_insert_len)
        if signal.left_clip:
            left_clip_qnames.add(qname)
        if signal.right_clip:
            right_clip_qnames.add(qname)

        if have_nearby_candidates:
            nearby_signal = classify_local_event_signal(read, component.chrom,
                                                        candidate_clip_start,
                                                        candidate_clip_end)
            if nearby_signal.left_clip and matches_nearby_candidate(nearby_signal.left_clip_pos):
                nearby_left_clip_qnames.add(qname)
            if nearby_signal.right_clip and matches_nearby_candidate(nearby_signal.right_clip_pos):
                nearby_right_clip_qnames.add(qname)

    for fragment in fragments:
        if (not fragment.read_id or fragment.ref_junc_pos < alt_signal_start
                or fragment.ref_junc_pos > alt_signal_end):
            continue
        if fragment.source == InsertionFragmentSource.SPLIT_SA:
            split_qnames.add(fragment.read_id)
        elif fragment.source == InsertionFragmentSource.CIGAR_INSERTION:
            indel_qnames.add(fragment.read_id)
            raw_cigar_insert_qnames.add(fragment.read_id)
            evidence.max_raw_cigar_insert_len = max(evidence.max_raw_cigar_insert_len,
                                                    fragment.length)
        elif fragment.source == InsertionFragmentSource.CLIP_REF_LEFT:
            left_clip_qnames.add(fragment.read_id)
        elif fragment.source == InsertionFragmentSource.CLIP_REF_RIGHT:
            right_clip_qnames.add(fragment.read_id)

        if have_nearby_candidates and matches_nearby_candidate(fragment.ref_junc_pos):
            if fragment.source == InsertionFragmentSource.CLIP_REF_LEFT:
                nearby_left_clip_qnames.add(fragment.read_id)
            elif fragment.source == InsertionFragmentSource.CLIP_REF_RIGHT:
                nearby_right_clip_qnames.add(fragment.read_id)

    # THE SUPPLEMENT RULE. Clips slightly outside the tight alt window are
    # admitted only when the event ALREADY has precise support and the clips
    # appear on BOTH sides. All three conditions together mean "we know the
    # event is real and these clips are its slightly-misplaced edges"; any one
    # of them alone would admit ordinary clip noise.
    has_precise_nonclip_support = bool(split_qnames) or bool(indel_qnames)
    if (has_precise_nonclip_support and nearby_left_clip_qnames
            and nearby_right_clip_qnames):
        left_clip_qnames |= nearby_left_clip_qnames
        right_clip_qnames |= nearby_right_clip_qnames

    evidence.alt_split_reads = len(split_qnames)
    evidence.alt_indel_reads = len(indel_qnames)
    evidence.alt_left_clip_reads = len(left_clip_qnames)
    evidence.alt_right_clip_reads = len(right_clip_qnames)
    evidence.raw_cigar_insert_reads = len(raw_cigar_insert_qnames)

    # THE CLIP PARTNER RULE. Clips enter `alt_struct_reads` -- the number the
    # genotyper divides by -- only with precise support or a clip on the other
    # side. One-sided clips are still reported in their own field, where the
    # mechanistic blocks can see them and discount them.
    alt_qnames = set(split_qnames) | set(indel_qnames)
    clip_support_has_partner = bool(alt_qnames) or (left_clip_qnames and right_clip_qnames)
    if clip_support_has_partner:
        alt_qnames |= left_clip_qnames
        alt_qnames |= right_clip_qnames

    evidence.alt_struct_reads = len(alt_qnames)
    evidence.support_qnames = sorted(alt_qnames)

    ref_qnames: set[str] = set()
    low_mapq_ref_qnames: set[str] = set()
    for read_idx, span in enumerate(read_spans):
        if not span.valid or span.tid != component.tid:
            continue
        # STRICTLY spanning: the read must start before the alt window and end
        # after it. A read that merely overlaps the breakpoint is not evidence
        # that the reference allele is present there.
        if span.start > alt_signal_start or span.end < alt_signal_end:
            continue
        if read_idx >= len(local_records) or local_records[read_idx] is None:
            continue
        read = local_records[read_idx]
        if read.is_supplementary or read.is_secondary:
            continue
        qname = read.qname
        if not qname or qname in alt_qnames:
            continue
        # The wide exclusion band: ANY event signal within 75 bp disqualifies
        # the read as clean reference.
        if read_has_local_event_signal(read, component.chrom, ref_signal_start,
                                       ref_signal_end):
            continue
        if read.mapq >= REF_SPAN_MIN_MAPQ:
            ref_qnames.add(qname)
        else:
            low_mapq_ref_qnames.add(qname)

    evidence.ref_span_reads = len(ref_qnames)
    evidence.low_mapq_ref_span_reads = len(low_mapq_ref_qnames)
    evidence.ref_span_qnames = sorted(ref_qnames)
    return evidence


def collect_event_read_evidence(component: ComponentCall,
                                local_records: list[AlignedRead],
                                read_spans: list[ReadReferenceSpan],
                                fragments: list[InsertionFragment],
                                seed_left: int, seed_right: int) -> EventReadEvidence:
    """Tally support for the component's own resolved breakpoint bounds."""
    from placer_py.breakpoints import resolve_event_breakpoint_bounds

    bp_left, bp_right = resolve_event_breakpoint_bounds(component, local_records,
                                                        fragments, seed_left, seed_right)
    return collect_event_read_evidence_for_bounds(component, local_records, read_spans,
                                                  fragments, bp_left, bp_right)
