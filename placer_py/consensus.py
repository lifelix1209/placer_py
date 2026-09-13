"""
One sequence for the event: event strings per read, and their consensus.

Ported from `src/pipeline/pipeline_consensus_stage.inc` and the event-string
helpers of `src/pipeline/pipeline_event_helpers.inc`, pinned by
`tests/test_29_consensus.py`.

WHAT AN "EVENT STRING" IS. Not the fragment, and not the read: it is
`left_flank + insert + right_flank`, cut from ONE read, with up to 80 bp of
anchored sequence on each available side. The flanks are what makes the
consensus segmentable -- `placer_py/segmentation.py` finds the junctions by
aligning the ends back to the reference, and a consensus of bare inserts has no
ends to align.

WHICH SIDES ARE AVAILABLE depends on the fragment's source, and the asymmetry is
mechanical rather than a policy:

    kClipRefLeft   the insert runs off the read start -> insert + right flank
    kClipRefRight  the insert runs off the read end   -> left flank + insert
    kCigarInsertion / kSplitSa                        -> both flanks

So only the last two are FULL CONTEXT. A clip fragment cannot produce both
flanks no matter how long the read is, because the read genuinely does not
contain the other side.

FULL CONTEXT IS PREFERRED WHOLESALE, not mixed. When any read gives both flanks,
the consensus is built from those reads ONLY and every clip-derived string is
discarded -- even when that means building from one read instead of twenty.
Mixing them would align strings that start at different points in the event, and
the consensus would be a smear. The `kPartialOnly` mode exists so the caller can
retry with the clip strings after a full-context consensus fails to segment.

THE CONSENSUS ITSELF IS NOT PORTED. The C++ calls abPOA, a C partial-order
alignment library. There is no pure-Python equivalent that would produce the
same sequence, and a different consensus changes every number downstream -- the
insert sequence, the TE identity, the poly(A) call, the structure decode. So
`build_event_consensus` takes the consensus function as an ARGUMENT:

  * `single_sequence_consensus` handles the cases needing no alignment (one
    input, or several identical ones) and raises otherwise;
  * a caller with `pyabpoa` installed passes `pyabpoa_consensus`;
  * anything else is a deliberate choice by the caller, not a silent fallback.

Returning a worse consensus silently would be the single most damaging thing
this port could do, because nothing downstream would report an error -- the
calls would just be subtly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from placer_py.alignment import AlignedRead
from placer_py.config import PipelineConfig
from placer_py.events import EventReadEvidence
from placer_py.fragments import InsertionFragment, InsertionFragmentSource
from placer_py.segmentation import EventConsensus
from placer_py.seqtools import upper_acgt

#: Flank taken from each available side of the insert.
EVENT_CONSENSUS_FLANK_BP = 80
#: ... of which this much must actually exist, or the read is unusable.
EVENT_CONSENSUS_MIN_ANCHOR_BP = 50
#: How far a fragment's junction may sit from the event's breakpoints.
EVENT_FRAGMENT_JUNCTION_SLACK_BP = 25


class ConsensusContextMode(Enum):
    PREFER_FULL = "PreferFull"
    PARTIAL_ONLY = "PartialOnly"


@dataclass
class ConsensusInputSummary:
    full_event_by_qname: dict[str, str] = field(default_factory=dict)
    partial_event_by_qname: dict[str, str] = field(default_factory=dict)
    full_context_input_reads: int = 0
    partial_context_input_reads: int = 0
    left_anchor_input_reads: int = 0
    right_anchor_input_reads: int = 0


class ConsensusUnavailable(RuntimeError):
    """Raised when a real multiple alignment is needed and none was supplied."""


def single_sequence_consensus(sequences: list[str]) -> str:
    """The consensus cases that need no alignment at all.

    One input is its own consensus; N identical inputs are too. Anything else
    needs a real partial-order alignment, and this RAISES rather than returning
    an approximation -- see the module docstring. A longest-input or
    majority-vote stand-in would change the insert sequence without changing any
    QC field, and the run would look clean.
    """
    if not sequences:
        return ""
    if len(set(sequences)) == 1:
        return sequences[0]
    raise ConsensusUnavailable(
        f"{len(sequences)} differing event strings need a partial-order alignment; "
        "pass `pyabpoa_consensus` (pip install pyabpoa) or another consensus "
        "callable to build_event_consensus")


def pyabpoa_consensus(sequences: list[str]) -> str:
    """abPOA through its Python binding -- the same library the C++ links.

    Imported lazily so the package has no hard dependency on it: the decision
    layer needs no consensus at all, and only a run that starts from a BAM does.
    """
    import pyabpoa  # noqa: F401  (optional dependency)

    if not sequences:
        return ""
    aligner = pyabpoa.msa_aligner()
    result = aligner.msa([upper_acgt(seq) for seq in sequences], out_cons=True,
                         out_msa=False, max_n_cons=1)
    return upper_acgt(result.cons_seq[0]) if result.cons_seq else ""


# ---------------------------------------------------------------------------
# Event strings.
# ---------------------------------------------------------------------------
def fragment_supports_event_consensus(fragment: InsertionFragment) -> bool:
    return fragment.source != InsertionFragmentSource.UNKNOWN


def fragment_has_full_event_context(fragment: InsertionFragment) -> bool:
    """Only a CIGAR insertion or a split can have both flanks on one read."""
    return fragment.source in (InsertionFragmentSource.CIGAR_INSERTION,
                               InsertionFragmentSource.SPLIT_SA)


def fragment_is_local_to_event(fragment: InsertionFragment, bp_left: int,
                               bp_right: int) -> bool:
    """Is this fragment's junction within 25 bp of the event's breakpoints?

    Tight, and deliberately tighter than the fragment extractor's own 1 kb
    locality: extraction is allowed to be generous because a wrong fragment
    costs one alignment, while a wrong event string corrupts the consensus that
    everything downstream reads.
    """
    if fragment.ref_junc_pos < 0 or bp_left < 0 or bp_right < 0:
        return False
    span_start = max(0, min(bp_left, bp_right) - EVENT_FRAGMENT_JUNCTION_SLACK_BP)
    span_end = max(span_start, max(bp_left, bp_right) + EVENT_FRAGMENT_JUNCTION_SLACK_BP)
    return span_start <= fragment.ref_junc_pos <= span_end


def collect_aligned_query_bases_before(read: AlignedRead, query_limit: int,
                                       max_bases: int) -> str:
    if query_limit <= 0 or max_bases <= 0:
        return ""
    take_start = max(0, query_limit - max_bases)
    return upper_acgt(read.decode_subsequence(take_start, query_limit - take_start))


def collect_aligned_query_bases_after(read: AlignedRead, query_start: int,
                                      max_bases: int) -> str:
    if query_start < 0 or max_bases <= 0:
        return ""
    return upper_acgt(read.decode_subsequence(query_start, max_bases))


def best_records_by_qname(local_records: list[AlignedRead]) -> dict[str, AlignedRead]:
    """One record per read name: the LONGEST, then the primary.

    Length first because the event string is cut from the record's own bases,
    and a hard-clipped supplementary carries fewer of them -- so the longest
    record is the one most likely to hold both flanks.
    """
    records_by_qname: dict[str, AlignedRead] = {}
    for record in local_records:
        if record is None or not record.qname:
            continue
        incumbent = records_by_qname.get(record.qname)
        if incumbent is None:
            records_by_qname[record.qname] = record
            continue
        if (record.seq_len > incumbent.seq_len
                or (record.seq_len == incumbent.seq_len
                    and not record.is_supplementary and incumbent.is_supplementary)):
            records_by_qname[record.qname] = record
    return records_by_qname


def can_build_event_string_from_fragment(record: AlignedRead,
                                         fragment: InsertionFragment) -> bool:
    """Does this read carry enough anchored sequence beside the insert?

    The flank requirement is what a "usable" read means here, and it is checked
    against the READ's own extent, not against the configured flank length: a
    read ending 60 bp past the insert gives a 60 bp flank, which clears the 50 bp
    minimum even though it is short of the 80 bp target.
    """
    if (record is None or not fragment.sequence or fragment.start < 0
            or fragment.length <= 0):
        return False
    insert_start = fragment.start
    insert_end = fragment.start + fragment.length
    if insert_end > record.seq_len:
        return False

    left_flank_len = insert_start - max(0, insert_start - EVENT_CONSENSUS_FLANK_BP)
    right_flank_len = max(0, min(record.seq_len, insert_end + EVENT_CONSENSUS_FLANK_BP)
                          - insert_end)

    if fragment.source == InsertionFragmentSource.CLIP_REF_LEFT:
        return right_flank_len >= EVENT_CONSENSUS_MIN_ANCHOR_BP
    if fragment.source == InsertionFragmentSource.CLIP_REF_RIGHT:
        return left_flank_len >= EVENT_CONSENSUS_MIN_ANCHOR_BP
    if fragment.source in (InsertionFragmentSource.CIGAR_INSERTION,
                           InsertionFragmentSource.SPLIT_SA):
        return (left_flank_len >= EVENT_CONSENSUS_MIN_ANCHOR_BP
                and right_flank_len >= EVENT_CONSENSUS_MIN_ANCHOR_BP)
    return False


def build_event_string_from_fragment(record: AlignedRead,
                                     fragment: InsertionFragment) -> str:
    """`[left flank] + insert + [right flank]`, from one read.

    The flanks are re-cut from the RECORD rather than taken from the fragment,
    because the fragment carries only the inserted bases -- and it is the flanks
    that let the consensus be segmented afterwards.
    """
    if not can_build_event_string_from_fragment(record, fragment):
        return ""
    insert_start = fragment.start
    insert_end = fragment.start + fragment.length
    insert_seq = upper_acgt(fragment.sequence)
    if not insert_seq:
        return ""

    left_flank = collect_aligned_query_bases_before(record, insert_start,
                                                    EVENT_CONSENSUS_FLANK_BP)
    right_flank = collect_aligned_query_bases_after(record, insert_end,
                                                    EVENT_CONSENSUS_FLANK_BP)

    if fragment.source == InsertionFragmentSource.CLIP_REF_LEFT:
        if len(right_flank) < EVENT_CONSENSUS_MIN_ANCHOR_BP:
            return ""
        return insert_seq + right_flank
    if fragment.source == InsertionFragmentSource.CLIP_REF_RIGHT:
        if len(left_flank) < EVENT_CONSENSUS_MIN_ANCHOR_BP:
            return ""
        return left_flank + insert_seq
    if fragment.source in (InsertionFragmentSource.CIGAR_INSERTION,
                           InsertionFragmentSource.SPLIT_SA):
        if (len(left_flank) < EVENT_CONSENSUS_MIN_ANCHOR_BP
                or len(right_flank) < EVENT_CONSENSUS_MIN_ANCHOR_BP):
            return ""
        return left_flank + insert_seq + right_flank
    return ""


def collect_event_consensus_inputs(local_records: list[AlignedRead],
                                   fragments: list[InsertionFragment],
                                   event_evidence: EventReadEvidence
                                   ) -> ConsensusInputSummary:
    """Build one event string per read, and count what each side anchors.

    THE ANCHOR COUNTS ARE CROSS-WIRED, and it is not a typo: a `CLIP_REF_LEFT`
    fragment increments the RIGHT anchor count. The name says which side of the
    reference the read's aligned part sits on; the anchor it provides is on the
    opposite side of the INSERT. Getting this backwards would invert the
    pre-segmentation gate's bilateral test.

    A read appearing twice keeps its LONGER event string -- more flank is
    strictly better for the alignment that follows.
    """
    summary = ConsensusInputSummary()
    support_qnames = set(event_evidence.support_qnames)
    records_by_qname = best_records_by_qname(local_records)

    full_context_qnames: set[str] = set()
    partial_context_qnames: set[str] = set()
    left_anchor_qnames: set[str] = set()
    right_anchor_qnames: set[str] = set()

    for fragment in fragments:
        if (not fragment.read_id or fragment.read_id not in support_qnames
                or not fragment_supports_event_consensus(fragment)
                or not fragment_is_local_to_event(fragment, event_evidence.bp_left,
                                                  event_evidence.bp_right)):
            continue
        record = records_by_qname.get(fragment.read_id)
        if record is None or not can_build_event_string_from_fragment(record, fragment):
            continue
        event_string = build_event_string_from_fragment(record, fragment)
        if not event_string:
            continue

        if fragment.source in (InsertionFragmentSource.CIGAR_INSERTION,
                               InsertionFragmentSource.SPLIT_SA):
            full_context_qnames.add(fragment.read_id)
            left_anchor_qnames.add(fragment.read_id)
            right_anchor_qnames.add(fragment.read_id)
        elif fragment.source == InsertionFragmentSource.CLIP_REF_LEFT:
            partial_context_qnames.add(fragment.read_id)
            right_anchor_qnames.add(fragment.read_id)
        elif fragment.source == InsertionFragmentSource.CLIP_REF_RIGHT:
            partial_context_qnames.add(fragment.read_id)
            left_anchor_qnames.add(fragment.read_id)
        else:
            continue

        target = (summary.full_event_by_qname if fragment_has_full_event_context(fragment)
                  else summary.partial_event_by_qname)
        existing = target.get(fragment.read_id)
        if existing is None or len(event_string) > len(existing):
            target[fragment.read_id] = event_string

    summary.full_context_input_reads = len(full_context_qnames)
    summary.partial_context_input_reads = len(partial_context_qnames)
    summary.left_anchor_input_reads = len(left_anchor_qnames)
    summary.right_anchor_input_reads = len(right_anchor_qnames)
    return summary


def build_event_consensus(local_records: list[AlignedRead],
                          fragments: list[InsertionFragment],
                          event_evidence: EventReadEvidence,
                          config: PipelineConfig,
                          mode: ConsensusContextMode = ConsensusContextMode.PREFER_FULL,
                          consensus_fn: Callable[[list[str]], str] = single_sequence_consensus
                          ) -> EventConsensus:
    """One consensus sequence for the event, from its reads' event strings.

    FULL CONTEXT IS ALL-OR-NOTHING. If any read gives both flanks, the consensus
    is built from those reads alone -- one full-context read beats twenty
    clip-derived ones, because mixing strings that begin at different points in
    the event produces a smear rather than a consensus. `PARTIAL_ONLY` exists so
    the caller can retry with the clip strings when segmentation of the
    full-context consensus fails.

    The MINIMUM READ COUNT differs between the two for the same reason: 1 for
    full context (the string is already the whole event) versus
    `event_consensus_poa_min_reads` for partial (several partial views have to
    agree before their overlap means anything).

    Inputs are sorted LONGEST FIRST before the cap, so truncating to
    `event_consensus_poa_max_reads` drops the least informative strings rather
    than an arbitrary subset.
    """
    consensus = EventConsensus()
    inputs = collect_event_consensus_inputs(local_records, fragments, event_evidence)

    use_full_context = (mode != ConsensusContextMode.PARTIAL_ONLY
                        and bool(inputs.full_event_by_qname))
    consensus.full_context_input_reads = inputs.full_context_input_reads
    consensus.partial_context_input_reads = inputs.partial_context_input_reads
    consensus.left_anchor_input_reads = inputs.left_anchor_input_reads
    consensus.right_anchor_input_reads = inputs.right_anchor_input_reads
    consensus.used_full_context = use_full_context

    event_by_qname = (inputs.full_event_by_qname if use_full_context
                      else inputs.partial_event_by_qname)
    event_strings = list(event_by_qname.values())
    consensus.input_event_reads = len(event_strings)
    if consensus.input_event_reads <= 0:
        consensus.qc_reason = "NO_EVENT_STRING_READS"
        return consensus

    min_event_reads = 1 if use_full_context else max(1, config.event_consensus_poa_min_reads)
    if consensus.input_event_reads < min_event_reads:
        consensus.qc_reason = "INSUFFICIENT_EVENT_READS"
        return consensus

    event_strings.sort(key=lambda s: (-len(s), s))
    if len(event_strings) > config.event_consensus_poa_max_reads:
        event_strings = event_strings[:config.event_consensus_poa_max_reads]

    consensus.consensus_seq = upper_acgt(consensus_fn(event_strings))
    consensus.consensus_len = len(consensus.consensus_seq)
    if not consensus.consensus_seq:
        consensus.qc_reason = "EMPTY_EVENT_CONSENSUS"
        return consensus

    consensus.qc_pass = True
    consensus.qc_reason = ("PASS_EVENT_CONSENSUS_PARTIAL_RETRY"
                           if mode == ConsensusContextMode.PARTIAL_ONLY
                           else "PASS_EVENT_CONSENSUS")
    return consensus


# ---------------------------------------------------------------------------
# Clip/insert concordance: do the clipped bases match the assembled insert?
# ---------------------------------------------------------------------------
#: The comparison window, and the identity it must reach.
MIN_CLIP_MATCH_LEN = 50
MAX_CLIP_MATCH_LEN = 300
MIN_CLIP_IDENTITY = 0.85


def is_low_quality_softclip_anchor(fragment: InsertionFragment,
                                   config: PipelineConfig) -> bool:
    """Is the reference side of this clip too poor to trust the clip?

    Two tests, and both are about the ANCHOR rather than the clipped sequence: a
    short anchor cannot locate the junction, and a high NM rate means the
    alignment that produced the clip is itself unreliable -- the clip may be an
    artefact of a bad alignment rather than a real breakpoint.

    `anchor_len > 0` guards both: an anchor length of 0 means "not measured",
    not "no anchor", and refusing on it would discard every fragment from a
    source that does not set it.
    """
    if fragment.source not in (InsertionFragmentSource.CLIP_REF_LEFT,
                               InsertionFragmentSource.CLIP_REF_RIGHT):
        return False
    if 0 < fragment.anchor_len < max(1, config.te_softclip_min_anchor_len):
        return True
    if fragment.anchor_len > 0 and fragment.nm >= 0:
        nm_per_bp = fragment.nm / max(1, fragment.anchor_len)
        if nm_per_bp > max(0.0, config.te_softclip_max_nm_per_bp):
            return True
    return False


@dataclass
class ClipInsertConcordanceEvidence:
    pass_: bool = False
    full_insert_reads: int = 0
    left_clip_reads: int = 0
    right_clip_reads: int = 0
    max_left_identity: float = 0.0
    max_right_identity: float = 0.0
    qc: str = "NO_CLIP_INSERT_CONCORDANCE"


def analyze_clip_insert_concordance(event_evidence: EventReadEvidence,
                                    segmentation, fragments: list[InsertionFragment],
                                    config: PipelineConfig
                                    ) -> ClipInsertConcordanceEvidence:
    """Do the clipped reads' bases actually match the assembled insert?

    WHY THIS IS WORTH DOING. A clip is counted as alt support on POSITIONAL
    grounds -- the read stops explaining the reference near the breakpoint. That
    says nothing about whether the clipped bases are the insertion. This
    compares them directly, and a clip whose sequence does NOT match the insert
    is a clip for some other reason at the same place.

    WHICH END IS COMPARED matters and is not symmetric: a `CLIP_REF_LEFT`
    fragment's bases are the START of the insert (the read ran off the front),
    so they are compared against the insert's start; a `CLIP_REF_RIGHT`
    fragment's against its end.

    THE PASS RULE needs a full-insert read AND either bilateral clips or two on
    one side. The full-insert requirement is what makes the comparison
    meaningful -- without a read that spans the event there is no assembled
    insert to compare the clips against, only the clips' own consensus.
    """
    from placer_py.breakpoints import fixed_window_edit_identity
    from placer_py.te_classifier import is_low_complexity_softclip, is_softclip_source

    evidence = ClipInsertConcordanceEvidence()
    if not segmentation.pass_ or not segmentation.insert_seq:
        evidence.qc = "NO_EVENT_SEGMENTATION_FOR_CLIP_INSERT_CONCORDANCE"
        return evidence
    insert_seq = upper_acgt(segmentation.insert_seq)
    if len(insert_seq) < MIN_CLIP_MATCH_LEN:
        evidence.qc = "CLIP_INSERT_CONCORDANCE_INSERT_TOO_SHORT"
        return evidence

    support_qnames = set(event_evidence.support_qnames)
    full_insert_qnames: set[str] = set()
    left_clip_qnames: set[str] = set()
    right_clip_qnames: set[str] = set()

    for fragment in fragments:
        if (not fragment.read_id or fragment.read_id not in support_qnames
                or not fragment_is_local_to_event(fragment, event_evidence.bp_left,
                                                  event_evidence.bp_right)):
            continue

        if fragment.source in (InsertionFragmentSource.CIGAR_INSERTION,
                               InsertionFragmentSource.SPLIT_SA):
            if len(fragment.sequence) >= MIN_CLIP_MATCH_LEN:
                full_insert_qnames.add(fragment.read_id)
            continue

        if (not is_softclip_source(fragment.source)
                or is_low_complexity_softclip(
                    fragment, upper_acgt(fragment.sequence),
                    config.te_softclip_low_complexity_at_frac_min,
                    config.te_softclip_low_complexity_homopolymer_min,
                    config.te_softclip_entropy_min,
                    config.te_softclip_kmer_uniqueness_min)
                or is_low_quality_softclip_anchor(fragment, config)):
            continue

        clip_seq = upper_acgt(fragment.sequence)
        match_len = min(len(clip_seq), len(insert_seq), MAX_CLIP_MATCH_LEN)
        if match_len < MIN_CLIP_MATCH_LEN:
            continue

        clipped_window = clip_seq[:match_len]
        insert_window = (insert_seq[:match_len]
                         if fragment.source == InsertionFragmentSource.CLIP_REF_LEFT
                         else insert_seq[len(insert_seq) - match_len:])
        identity = fixed_window_edit_identity(clipped_window, insert_window)
        if identity + 1e-9 < MIN_CLIP_IDENTITY:
            continue

        if fragment.source == InsertionFragmentSource.CLIP_REF_LEFT:
            left_clip_qnames.add(fragment.read_id)
            evidence.max_left_identity = max(evidence.max_left_identity, identity)
        else:
            right_clip_qnames.add(fragment.read_id)
            evidence.max_right_identity = max(evidence.max_right_identity, identity)

    evidence.full_insert_reads = len(full_insert_qnames)
    evidence.left_clip_reads = len(left_clip_qnames)
    evidence.right_clip_reads = len(right_clip_qnames)
    bilateral_clip = evidence.left_clip_reads >= 1 and evidence.right_clip_reads >= 1
    strong_one_sided_clip = (evidence.left_clip_reads >= 2
                             or evidence.right_clip_reads >= 2)
    evidence.pass_ = (evidence.full_insert_reads >= 1
                      and (bilateral_clip or strong_one_sided_clip))
    evidence.qc = ("PASS_CLIP_INSERT_CONCORDANCE" if evidence.pass_
                   else "NO_CLIP_INSERT_CONCORDANCE")
    return evidence


def build_local_fragment_component(component, local_records: list[AlignedRead],
                                   config: PipelineConfig):
    """Re-index a component against the LOCALLY FETCHED reads.

    NECESSARY because a component's read indices point into the BIN's record
    list, and the local fetch returns a different, wider set. Passing the
    original component to the fragment extractor would make it index the wrong
    reads -- silently, since both lists are just integers.

    The re-classification is deliberately looser than the geometry stage's: any
    read with a split position or a supplementary flag counts as split support
    here, because the question is "which local reads might carry a fragment",
    not "which reads support this event".
    """
    import copy

    from placer_py.breakpoints import robust_local_split_insertion_positions
    from placer_py.alignment import CIGAR_I, CIGAR_S
    from placer_py.clustering import INSERTION_CANDIDATE_REQUIRED_MAPQ

    local = copy.copy(component)
    local.read_indices = []
    local.soft_clip_read_indices = []
    local.split_sa_read_indices = []
    local.insertion_read_indices = []

    for index, read in enumerate(local_records):
        if read is None or read.tid != component.tid:
            continue
        local.read_indices.append(index)

        if (robust_local_split_insertion_positions(read, component.chrom)
                or read.is_supplementary):
            local.split_sa_read_indices.append(index)
        if not read.cigar:
            continue

        max_soft_clip = 0
        max_insertion = 0
        for op, length in read.cigar:
            if op == CIGAR_S:
                max_soft_clip = max(max_soft_clip, length)
            elif op == CIGAR_I:
                max_insertion = max(max_insertion, length)

        if max_soft_clip >= config.min_soft_clip_for_seq_extract:
            local.soft_clip_read_indices.append(index)
        if (read.mapq == INSERTION_CANDIDATE_REQUIRED_MAPQ
                and max_insertion >= config.min_long_ins_for_seq_extract):
            local.insertion_read_indices.append(index)
    return local
