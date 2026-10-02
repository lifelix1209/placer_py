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

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from statistics import median

from placer.alignment import AlignedRead, cigar_index, compute_ref_end
from placer.core.breakpoints import (
    classify_local_event_signal,
    read_has_local_event_signal,
    split_insertion_calls,
)
from placer.core.clustering import INSERTION_CANDIDATE_REQUIRED_MAPQ, ComponentCall
from placer.core.fragments import InsertionFragment, InsertionFragmentSource
from placer.core.windows import LONG_INSERTION_SIGNAL_MIN

#: How close a signal must be to the hypothesis to be alt support.
ALT_SIGNAL_SLACK_BP = 25
#: How close a signal must be to disqualify a read as reference support.
REF_SIGNAL_SLACK_BP = 75
#: Tolerance when matching a clip to a component breakpoint candidate.
CANDIDATE_CLIP_SLACK_BP = 25
#: How far a component candidate may sit from the hypothesis and still lend it
#: clip reads.
CANDIDATE_CLIP_SUPPLEMENT_MAX_OFFSET_BP = 75
#: THE SAME-ALLELE CARRIER RULE. In a tandem repeat or a low-complexity flank an
#: aligner places one insertion at different offsets in different reads.
#: Measured on HG002 chr1:
#:   * a homozygous 312 bp AluSz: all 52 spanning reads carry it, over 300 bp,
#:     only 12 of them within 25 bp of the mode;
#:   * a het 164 bp SVA_E: 47 carriers against 3 counted.
#: Every carrier outside the +-25 bp alt window was counted as REFERENCE whenever
#: it spanned that window, so the insertion looked like a few noisy reads over
#: a strong reference and scored as an artifact.
#: A uniquely mapped read with an insertion of this allele's length (the median
#: of the tight window's insertions, within +-CARRIER_LENGTH_TOLERANCE) within
#: CARRIER_WINDOW_BP of the bounds carries the same allele: it is alt support
#: and not reference. The length condition keeps a different nearby allele out.
#: Opt-in (`PipelineConfig.same_allele_carrier_window_bp`, default 0): see there
#: for why it is not on by default yet.
CARRIER_WINDOW_BP = 500
CARRIER_LENGTH_TOLERANCE = 0.3
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
    #: Of the indel reads, those counted by the same-allele carrier rule: an
    #: insertion of this allele's length outside the tight window.
    alt_carrier_reads: int = 0
    ref_span_reads: int = 0
    low_mapq_ref_span_reads: int = 0
    #: Sorted, so two runs of the same locus produce identical output and the
    #: support-sharing tests in finalization are deterministic.
    support_qnames: list[str] = field(default_factory=list)
    ref_span_qnames: list[str] = field(default_factory=list)
    #: What the alt reads MEASURED the insertion to be, one length per read,
    #: sorted. Only a read that spans the insertion measures it: a CIGAR
    #: insertion (from a uniquely mapped read, as for `alt_indel_reads`) or an
    #: SA pair's implied insertion (as for `alt_split_reads`), at least
    #: LONG_INSERTION_SIGNAL_MIN, inside this hypothesis's alt window. A clip is
    #: only a lower bound and is not a measurement. A read with several takes
    #: its CIGAR one over its split one, then the one nearest the window
    #: centre, then the longer. Taken from the reads alone, so it does not
    #: depend on which component evaluated the hypothesis. Recorded, not used:
    #: whether length concordance should inform a decision is for replay.
    alt_measured_lengths: list[int] = field(default_factory=list)


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
                                           bp_left: int, bp_right: int,
                                           min_signal_mapq: int = 0,
                                           carrier_window_bp: int = 0
                                           ) -> EventReadEvidence:
    """Tally alt and reference support for ONE breakpoint hypothesis.

    Called once per hypothesis, which is why it takes the bounds rather than
    reading them off the component: the hypotheses compete on the evidence each
    of them gathers, and a hypothesis in the wrong place collects less.

    `min_signal_mapq`: a clip or split read below this MAPQ is not counted as
    alt support. An insertion already has to come from a uniquely mapped read
    (`INSERTION_CANDIDATE_REQUIRED_MAPQ`), but clips and splits counted at any
    MAPQ, so in a repeat-rich genome multi-mapping reads -- MAPQ 0 -- added to
    a candidate's support. 0 keeps that behaviour.
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
    tight_lengths: list[int] = []
    # qname -> (kind, distance from the centre, -length, length); min() wins.
    measured: dict[str, tuple[int, int, int, int]] = {}
    nearby_left_clip_qnames: set[str] = set()
    nearby_right_clip_qnames: set[str] = set()

    alt_signal_start = max(0, left - ALT_SIGNAL_SLACK_BP)
    alt_signal_end = max(alt_signal_start, right + ALT_SIGNAL_SLACK_BP)
    alt_signal_centre = alt_signal_start + ((alt_signal_end - alt_signal_start) // 2)
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
        if read.mapq < min_signal_mapq:
            signal.split = signal.left_clip = signal.right_clip = False
        if signal.split:
            split_qnames.add(qname)
            for pos, length in split_insertion_calls(read, component.chrom):
                if alt_signal_start <= pos <= alt_signal_end:
                    _keep_measurement(measured, qname, (1, abs(pos - alt_signal_centre),
                                                        -length, length))
        # An insertion is believed as PRECISE evidence only from a uniquely
        # mapped read -- the same equality test the geometry stage applies, and
        # for the same reason.
        if signal.indel and read.mapq == INSERTION_CANDIDATE_REQUIRED_MAPQ:
            indel_qnames.add(qname)
            length = _insertion_length_at(read, signal.indel_pos)
            if length >= LONG_INSERTION_SIGNAL_MIN:
                tight_lengths.append(length)
                _keep_measurement(measured, qname, (0, abs(signal.indel_pos - alt_signal_centre),
                                                    -length, length))
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

    if tight_lengths and carrier_window_bp > 0:
        carriers = _same_allele_carriers(component, local_records, left, right,
                                         median(tight_lengths), indel_qnames,
                                         carrier_window_bp)
        evidence.alt_carrier_reads = len(carriers)
        indel_qnames |= carriers
        raw_cigar_insert_qnames |= carriers

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
    evidence.alt_measured_lengths = sorted(entry[3] for entry in measured.values())

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


def _keep_measurement(measured: dict[str, tuple[int, int, int, int]], qname: str,
                      candidate: tuple[int, int, int, int]) -> None:
    """Keep one measured length per read: the smallest key wins, so the
    choice does not depend on the order of the read's records."""
    current = measured.get(qname)
    if current is None or candidate < current:
        measured[qname] = candidate


def _insertion_length_at(read: AlignedRead, ref_pos: int) -> int:
    """The length of the read's insertion at `ref_pos` (the longest, if several)."""
    index = cigar_index(read)
    lo = bisect_left(index.ins_ref_pos, ref_pos)
    hi = bisect_right(index.ins_ref_pos, ref_pos)
    return max(index.ins_len[lo:hi], default=0)


def _same_allele_carriers(component: ComponentCall, local_records: list[AlignedRead],
                          left: int, right: int, allele_len: float,
                          already: set[str], window_bp: int) -> set[str]:
    """Reads carrying an insertion of `allele_len` (+-CARRIER_LENGTH_TOLERANCE)
    within CARRIER_WINDOW_BP of [left, right], outside the tight window: the
    same allele, placed elsewhere by the aligner. Uniquely mapped primary
    alignments only, as for any insertion evidence."""
    lo_len = max(LONG_INSERTION_SIGNAL_MIN, allele_len * (1.0 - CARRIER_LENGTH_TOLERANCE))
    hi_len = allele_len * (1.0 + CARRIER_LENGTH_TOLERANCE)
    start = max(0, left - window_bp)
    end = right + window_bp
    out: set[str] = set()
    for read in local_records:
        if (read is None or read.tid != component.tid or not read.qname
                or read.qname in already or read.is_secondary or read.is_supplementary
                or read.mapq != INSERTION_CANDIDATE_REQUIRED_MAPQ or not read.cigar):
            continue
        index = cigar_index(read)
        a = bisect_left(index.ins_ref_pos, start)
        b = bisect_right(index.ins_ref_pos, end)
        if any(lo_len <= length <= hi_len for length in index.ins_len[a:b]):
            out.add(read.qname)
    return out


def collect_event_read_evidence(component: ComponentCall,
                                local_records: list[AlignedRead],
                                read_spans: list[ReadReferenceSpan],
                                fragments: list[InsertionFragment],
                                seed_left: int, seed_right: int) -> EventReadEvidence:
    """Tally support for the component's own resolved breakpoint bounds."""
    from placer.core.breakpoints import resolve_event_breakpoint_bounds

    bp_left, bp_right = resolve_event_breakpoint_bounds(component, local_records,
                                                        fragments, seed_left, seed_right)
    return collect_event_read_evidence_for_bounds(component, local_records, read_spans,
                                                  fragments, bp_left, bp_right)


# ---------------------------------------------------------------------------
# The allele-level tally: RECORDED, read by no decision
# ---------------------------------------------------------------------------

#: THE ALLELE-LEVEL TALLY. In a tandem repeat the aligner places one insertion
#: at different offsets in different reads. On HG002 chr1, 73 of the 229 truth
#: TEs have carriers spread over more than 50 bp, and 1.0.0a4 recalled 41% of
#: them against 94% of the rest: the +-25 bp tally above sees a few carriers and
#: counts the others as reference. This tally gathers the allele's carriers
#: around the hypothesis, and counts as reference only reads that span the
#: whole allele without an insertion. It is written to the ledger for replay
#: (`docs/development-strategy.md`, section 1: record before deciding) and no
#: decision reads it.
ALLELE_WINDOW_BP = 500
ALLELE_LENGTH_TOLERANCE = 0.3
ALLELE_LENGTH_SLACK_BP = 20
#: A read's insertion pieces at least this long and this close on the reference
#: are one event: an ONT alignment often breaks one insertion into two or three
#: CIGAR insertions a few bases apart (47 of 51 carriers at chr1:45497762).
ALLELE_PIECE_MIN_BP = 20
ALLELE_PIECE_MERGE_BP = 100
#: Single linkage over the carriers' positions, grown from the hypothesis's own
#: window, with the allele's span capped.
ALLELE_CLUSTER_GAP_BP = 100
ALLELE_SPAN_CAP_BP = 1000
#: "The same sequence": the share of a carrier's k-mers found in the inserts of
#: the hypothesis's own reads. At k = 11 and ONT error rates, same-length
#: carriers of chr1's dispersed truth TEs sit at a median of 0.47, and 90% of
#: them at 0.2 or more.
ALLELE_KMER = 11
ALLELE_MIN_SIMILARITY = 0.2
#: THE COUNTS TERM'S LOCAL NULL, measured. The counts term's null is a model:
#: under H_artifact a read shows the signal with probability eps, from the
#: locus's composition (`tprt.local_error_rate`). Nothing checks it, as the
#: shifted-breakpoint decoys check the TSD and motif terms, and a tally
#: widened to an allele's span makes it more fragile: in a VNTR, reads carry
#: insertions of every length. So the rate is measured where the insertion is
#: not: in COUNTS_BACKGROUND_WINDOWS windows on each side, each as wide as the
#: tally's own window, beyond the carrier window (ALLELE_WINDOW_BP). A window's
#: reads are counted as the tally counts them: those showing its signal inside
#: the window (alt-like), and those strictly spanning it without one
#: (reference-like). Recorded for replay; no decision reads it.
COUNTS_BACKGROUND_WINDOWS = 4



@dataclass
class AlleleTally:
    """The allele's counts under one definition of "the same allele"."""

    alt_reads: int = -1
    ref_reads: int = -1
    #: The allele's span, relative to the hypothesis's bp_left.
    span_lo: int = 0
    span_hi: int = 0
    #: Carriers beyond the hypothesis's own alt reads. 0 means the tally is the
    #: hypothesis's own tally, unchanged.
    extra_carriers: int = 0
    #: The tally's signal where the insertion is not (COUNTS_BACKGROUND_WINDOWS):
    #: read-windows counted, and those showing a carrier of this allele's length
    #: (and sequence, for `by_sequence` and `wide`). -1 when extra_carriers is 0.
    background_reads: int = -1
    background_hits: int = -1


@dataclass
class AlleleEvidence:
    """The allele-level view of one hypothesis (`collect_allele_evidence`)."""

    #: The median of the hypothesis's measured lengths; -1 when it measured
    #: none, and then nothing else here was computed.
    length: int = -1
    #: Every read carrying an insertion of that length within ALLELE_WINDOW_BP,
    #: the hypothesis's own included, sorted: its offset from bp_left, its
    #: length, its k-mer similarity to the own inserts (-1 when it cannot be
    #: measured: a split, or no own insert sequence), and whether it is one of
    #: the hypothesis's own alt reads (1) or not (0). With the own alt count,
    #: that is enough to recount alt under any other grouping of the carriers.
    carrier_offsets: list[int] = field(default_factory=list)
    carrier_lengths: list[int] = field(default_factory=list)
    carrier_similarity: list[float] = field(default_factory=list)
    carrier_own: list[int] = field(default_factory=list)
    #: Three definitions of the allele:
    #:   * `by_length`: carriers of the length, grown by single linkage
    #:     (ALLELE_CLUSTER_GAP_BP) from the hypothesis's own window;
    #:   * `by_sequence`: the same, with only carriers of the same sequence;
    #:   * `wide`: every carrier of the same length and sequence in the window,
    #:     for alleles whose carriers the aligner spread with gaps wider than
    #:     the linkage (chr1:2212064: 20 carriers over 520 bp, in groups
    #:     136-232 bp apart).
    by_length: AlleleTally = field(default_factory=AlleleTally)
    by_sequence: AlleleTally = field(default_factory=AlleleTally)
    wide: AlleleTally = field(default_factory=AlleleTally)


def _distance_to(pos: int, left: int, right: int) -> int:
    if left <= pos <= right:
        return 0
    return left - pos if pos < left else pos - right


def _kmers(seq: str) -> set[str]:
    k = ALLELE_KMER
    text = seq.upper()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _allele_event(read: AlignedRead, chrom: str, left: int, right: int,
                  lo_len: float, hi_len: float) -> tuple[int, int, str] | None:
    """The read's insertion of the allele's length nearest [left, right], as
    (position, length, inserted sequence -- "" for a split), or None.

    Pieces are merged first (ALLELE_PIECE_MERGE_BP). A CIGAR event wins over a
    split one, as for `alt_measured_lengths`."""
    index = cigar_index(read)
    start = max(0, left - ALLELE_WINDOW_BP)
    end = right + ALLELE_WINDOW_BP
    a = bisect_left(index.ins_ref_pos, start - ALLELE_PIECE_MERGE_BP)
    b = bisect_right(index.ins_ref_pos, end + ALLELE_PIECE_MERGE_BP)
    groups: list[list[int]] = []
    for k in range(a, b):
        if index.ins_len[k] < ALLELE_PIECE_MIN_BP:
            continue
        if (groups and index.ins_ref_pos[k] - index.ins_ref_pos[groups[-1][-1]]
                <= ALLELE_PIECE_MERGE_BP):
            groups[-1].append(k)
        else:
            groups.append([k])
    best: tuple[tuple[int, int, int], tuple[int, int, str]] | None = None
    for group in groups:
        pos = index.ins_ref_pos[group[0]]
        length = sum(index.ins_len[k] for k in group)
        if not (start <= pos <= end and lo_len <= length <= hi_len):
            continue
        seq = ("".join(read.seq[index.ins_query_pos[k]:index.ins_query_pos[k] + index.ins_len[k]]
                       for k in group) if read.seq else "")
        key = (_distance_to(pos, left, right), -length, pos)
        if best is None or key < best[0]:
            best = (key, (pos, length, seq))
    if best is not None:
        return best[1]
    for pos, length in split_insertion_calls(read, chrom):
        if start <= pos <= end and lo_len <= length <= hi_len:
            key = (_distance_to(pos, left, right), -length, pos)
            if best is None or key < best[0]:
                best = (key, (pos, length, ""))
    return best[1] if best is not None else None


def _allele_tally(component: ComponentCall, local_records: list[AlignedRead],
                  evidence: EventReadEvidence, own: set[str],
                  carriers: list[tuple[int, str]], left: int, right: int,
                  not_reference: set[str], linked: bool = True) -> AlleleTally:
    """The allele: grown by single linkage from the carriers inside the
    hypothesis's alt window (`linked`), or all of `carriers` when at least one
    is inside it. Then its reads, and the reads that span all of it without an
    event. A read in `not_reference` -- any carrier of the length in the
    window -- is never reference, whichever carriers the allele took."""
    own_tally = AlleleTally(alt_reads=evidence.alt_struct_reads,
                            ref_reads=evidence.ref_span_reads,
                            span_lo=left - evidence.bp_left,
                            span_hi=right - evidence.bp_left, extra_carriers=0)
    ordered = sorted(carriers)
    positions = [pos for pos, _ in ordered]
    tight_lo, tight_hi = left - ALT_SIGNAL_SLACK_BP, right + ALT_SIGNAL_SLACK_BP
    seeds = [i for i, pos in enumerate(positions) if tight_lo <= pos <= tight_hi]
    if not seeds:
        return own_tally
    lo, hi = (seeds[0], seeds[-1]) if linked else (0, len(positions) - 1)
    while linked:
        options = []
        if lo > 0:
            gap = positions[lo] - positions[lo - 1]
            if (gap <= ALLELE_CLUSTER_GAP_BP
                    and positions[hi] - positions[lo - 1] <= ALLELE_SPAN_CAP_BP):
                options.append((gap, 0))
        if hi + 1 < len(positions):
            gap = positions[hi + 1] - positions[hi]
            if (gap <= ALLELE_CLUSTER_GAP_BP
                    and positions[hi + 1] - positions[lo] <= ALLELE_SPAN_CAP_BP):
                options.append((gap, 1))
        if not options:
            break
        if min(options)[1] == 0:
            lo -= 1
        else:
            hi += 1
    members = {qname for _, qname in ordered[lo:hi + 1]}
    extra = members - own
    if not extra:
        return own_tally
    span_left = min(positions[lo], left)
    span_right = max(positions[hi], right)
    names = own | members
    excluded = names | not_reference
    reference: set[str] = set()
    for read in local_records:
        if (read is None or read.tid != component.tid or not read.qname or not read.cigar
                or read.qname in excluded or read.is_secondary or read.is_supplementary
                or read.mapq < REF_SPAN_MIN_MAPQ):
            continue
        # Strictly spanning the whole allele, and clean inside its exclusion band.
        if (read.pos > span_left - ALT_SIGNAL_SLACK_BP
                or compute_ref_end(read) < span_right + ALT_SIGNAL_SLACK_BP):
            continue
        if read_has_local_event_signal(read, component.chrom,
                                       max(0, span_left - REF_SIGNAL_SLACK_BP),
                                       span_right + REF_SIGNAL_SLACK_BP):
            continue
        reference.add(read.qname)
    return AlleleTally(alt_reads=len(names), ref_reads=len(reference),
                       span_lo=span_left - evidence.bp_left,
                       span_hi=span_right - evidence.bp_left,
                       extra_carriers=len(extra))


def collect_allele_evidence(component: ComponentCall, local_records: list[AlignedRead],
                            evidence: EventReadEvidence) -> AlleleEvidence:
    """The hypothesis's allele: its carriers wherever the aligner put them.

    A carrier is a uniquely mapped primary read with an insertion (pieces
    merged) or an SA split of the allele's length, +-ALLELE_LENGTH_TOLERANCE,
    within ALLELE_WINDOW_BP. It is "of the same sequence" when its inserted
    sequence shares at least ALLELE_MIN_SIMILARITY of its k-mers with the
    hypothesis's own reads' inserts; one whose similarity cannot be measured
    is kept. The three tallies are described on `AlleleEvidence`. Where no
    carrier lies outside the hypothesis's own reads, a tally is the
    hypothesis's own, exactly.
    """
    out = AlleleEvidence()
    if evidence.bp_left < 0 or evidence.bp_right < 0 or not evidence.alt_measured_lengths:
        return out
    left = min(evidence.bp_left, evidence.bp_right)
    right = max(evidence.bp_left, evidence.bp_right)
    length = int(round(median(evidence.alt_measured_lengths)))
    out.length = length
    lo_len = max(LONG_INSERTION_SIGNAL_MIN, length * (1.0 - ALLELE_LENGTH_TOLERANCE))
    hi_len = length * (1.0 + ALLELE_LENGTH_TOLERANCE) + ALLELE_LENGTH_SLACK_BP
    own = set(evidence.support_qnames)

    def nearness(event: tuple[int, int, str]) -> tuple[int, int, int]:
        return (_distance_to(event[0], left, right), -event[1], event[0])

    events: dict[str, tuple[int, int, str]] = {}
    for read in local_records:
        if (read is None or read.tid != component.tid or not read.qname or not read.cigar
                or read.is_secondary or read.is_supplementary
                or read.mapq != INSERTION_CANDIDATE_REQUIRED_MAPQ):
            continue
        found = _allele_event(read, component.chrom, left, right, lo_len, hi_len)
        if found is None:
            continue
        current = events.get(read.qname)
        if current is None or nearness(found) < nearness(current):
            events[read.qname] = found

    tight_lo, tight_hi = left - ALT_SIGNAL_SLACK_BP, right + ALT_SIGNAL_SLACK_BP
    reference_kmers: set[str] = set()
    for qname in sorted(own):
        event = events.get(qname)
        if event is not None and event[2] and tight_lo <= event[0] <= tight_hi:
            reference_kmers |= _kmers(event[2])
    similarity: dict[str, float] = {}
    for qname, (_, _, seq) in events.items():
        kmers = _kmers(seq) if seq else set()
        similarity[qname] = (len(kmers & reference_kmers) / len(kmers)
                             if kmers and reference_kmers else -1.0)

    listed = sorted((pos - evidence.bp_left, size, round(similarity[qname], 3),
                     int(qname in own))
                    for qname, (pos, size, _) in events.items())
    out.carrier_offsets = [entry[0] for entry in listed]
    out.carrier_lengths = [entry[1] for entry in listed]
    out.carrier_similarity = [entry[2] for entry in listed]
    out.carrier_own = [entry[3] for entry in listed]

    carriers = [(pos, qname) for qname, (pos, _, _) in events.items()]
    every_carrier = set(events)
    alike = [(pos, qname) for pos, qname in carriers
             if qname in own or similarity[qname] < 0.0
             or similarity[qname] >= ALLELE_MIN_SIMILARITY]
    out.by_length = _allele_tally(component, local_records, evidence, own, carriers,
                                  left, right, every_carrier)
    out.by_sequence = _allele_tally(component, local_records, evidence, own, alike,
                                    left, right, every_carrier)
    out.wide = _allele_tally(component, local_records, evidence, own, alike,
                             left, right, every_carrier, linked=False)

    def matches(seq: str, by_sequence: bool) -> bool:
        if not by_sequence or not seq or not reference_kmers:
            return True
        kmers = _kmers(seq)
        return bool(kmers) and len(kmers & reference_kmers) / len(kmers) >= ALLELE_MIN_SIMILARITY

    for tally, by_sequence in ((out.by_length, False), (out.by_sequence, True),
                               (out.wide, True)):
        if tally.extra_carriers <= 0:
            continue
        windows = background_windows(left, right,
                                     tally.span_hi - tally.span_lo + 2 * ALT_SIGNAL_SLACK_BP)
        tally.background_reads, tally.background_hits = 0, 0
        for read in local_records:
            if (read is None or read.tid != component.tid or not read.qname or not read.cigar
                    or read.is_secondary or read.is_supplementary
                    or read.mapq != INSERTION_CANDIDATE_REQUIRED_MAPQ):
                continue
            end = compute_ref_end(read)
            touched = [(lo, hi) for lo, hi in windows if read.pos <= hi and end >= lo]
            if not touched:
                continue
            in_windows = _length_events(read, component.chrom, touched[0][0], touched[-1][1],
                                        lo_len, hi_len)
            for lo, hi in touched:
                if any(lo <= pos <= hi and matches(seq, by_sequence) for pos, seq in in_windows):
                    tally.background_reads += 1
                    tally.background_hits += 1
                elif read.pos <= lo and end >= hi:
                    tally.background_reads += 1
    return out


def background_windows(left: int, right: int, width: int) -> list[tuple[int, int]]:
    """COUNTS_BACKGROUND_WINDOWS windows of `width` on each side of the carrier
    window [left - ALLELE_WINDOW_BP, right + ALLELE_WINDOW_BP], in order."""
    width = max(1, width)
    lo_edge, hi_edge = left - ALLELE_WINDOW_BP, right + ALLELE_WINDOW_BP
    out = [(lo_edge - (k + 1) * width, lo_edge - k * width)
           for k in range(COUNTS_BACKGROUND_WINDOWS - 1, -1, -1)]
    out += [(hi_edge + k * width, hi_edge + (k + 1) * width)
            for k in range(COUNTS_BACKGROUND_WINDOWS)]
    return [(lo, hi) for lo, hi in out if lo >= 0]


def _length_events(read: AlignedRead, chrom: str, start: int, end: int,
                   lo_len: float, hi_len: float) -> list[tuple[int, str]]:
    """Every insertion event of the length in [start, end]: merged CIGAR pieces
    (position, inserted sequence) and SA splits (position, "")."""
    index = cigar_index(read)
    a = bisect_left(index.ins_ref_pos, start - ALLELE_PIECE_MERGE_BP)
    b = bisect_right(index.ins_ref_pos, end + ALLELE_PIECE_MERGE_BP)
    groups: list[list[int]] = []
    for k in range(a, b):
        if index.ins_len[k] < ALLELE_PIECE_MIN_BP:
            continue
        if (groups and index.ins_ref_pos[k] - index.ins_ref_pos[groups[-1][-1]]
                <= ALLELE_PIECE_MERGE_BP):
            groups[-1].append(k)
        else:
            groups.append([k])
    out = []
    for group in groups:
        pos = index.ins_ref_pos[group[0]]
        length = sum(index.ins_len[k] for k in group)
        if start <= pos <= end and lo_len <= length <= hi_len:
            seq = ("".join(read.seq[index.ins_query_pos[k]:index.ins_query_pos[k] + index.ins_len[k]]
                           for k in group) if read.seq else "")
            out.append((pos, seq))
    out.extend((pos, "") for pos, length in split_insertion_calls(read, chrom)
               if start <= pos <= end and lo_len <= length <= hi_len)
    return out


def collect_own_background(component: ComponentCall, local_records: list[AlignedRead],
                           evidence: EventReadEvidence) -> tuple[int, int]:
    """The +-25 bp tally's signal where the insertion is not: (read-windows
    counted, read-windows showing an alt-like signal -- a split, a clip, or a
    uniquely mapped insertion of at least LONG_INSERTION_SIGNAL_MIN), over
    COUNTS_BACKGROUND_WINDOWS windows a side as wide as the alt window. A
    window counts the reads showing the signal inside it and the reads
    strictly spanning it without one; primary reads, and MAPQ >=
    REF_SPAN_MIN_MAPQ for the spanning ones, as the reference tally counts
    them. (-1, -1) without a breakpoint. Recorded for replay; no decision
    reads it."""
    if evidence.bp_left < 0 or evidence.bp_right < 0:
        return -1, -1
    left = min(evidence.bp_left, evidence.bp_right)
    right = max(evidence.bp_left, evidence.bp_right)
    windows = background_windows(left, right, right - left + 2 * ALT_SIGNAL_SLACK_BP)
    reads = hits = 0
    for read in local_records:
        if (read is None or read.tid != component.tid or not read.qname or not read.cigar
                or read.is_secondary or read.is_supplementary):
            continue
        end = compute_ref_end(read)
        for lo, hi in windows:
            if read.pos > hi or end < lo:
                continue
            signal = classify_local_event_signal(read, component.chrom, lo, hi)
            if (signal.split or signal.left_clip or signal.right_clip
                    or (signal.indel and read.mapq == INSERTION_CANDIDATE_REQUIRED_MAPQ)):
                reads += 1
                hits += 1
            elif read.pos <= lo and end >= hi and read.mapq >= REF_SPAN_MIN_MAPQ:
                reads += 1
    return reads, hits
