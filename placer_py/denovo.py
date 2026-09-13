"""
De novo calling: is the child's insertion absent from the parents?

Ported from `include/denovo.h`, `src/denovo/denovo_cli.cpp` and
`src/denovo/parent_pool_scanner.cpp`, pinned by `tests/test_32_pipeline.py`.

THE QUESTION IS ASYMMETRIC, and the whole design follows from that. Calling an
insertion in the child is a detection problem; showing it is ABSENT from both
parents is not -- absence of evidence in a parent BAM has many causes that have
nothing to do with genotype (low coverage, a mapping hole, a read that clipped
for another reason). So this stage does not try to genotype the parents. It
looks for ANY read in either parent that could be the same insertion, and it is
deliberately eager to find one.

THREE VERDICTS, and the middle one is the point:

    PARENT_VETO              a parent read matches the child's element
    PARENT_AMBIGUOUS_SIGNAL  a parent read shows SOMETHING at the locus
    DENOVO_PASS              nothing at all in any parent

`REVIEW` is a first-class answer rather than a rounding of one of the others.
An unclassified insertion fragment in a parent, or a split alignment near the
locus, is exactly the evidence that cannot distinguish "inherited, poorly
assembled" from "unrelated event nearby" -- and reporting it as de novo would
turn the commonest failure of trio analysis into a confident positive.

THE FAMILY VETO IS CONFIGURABLE for the same reason. `--no-family-match-veto`
moves family-level matches from VETO to REVIEW: a parent read matching the
child's FAMILY but not its element is either the same inherited insertion seen
badly, or one of the ~1,000,000 other copies of that family in the genome. Which
of those it is depends on the family, and the flag is where that judgement is
made explicitly rather than assumed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Sequence

from placer_py.alignment import AlignedRead
from placer_py.config import PipelineConfig
from placer_py.fragments import (
    InsertionFragment,
    InsertionFragmentSource,
    analyze_clip_info,
    find_long_insertions,
)


@dataclass
class DenovoConfig:
    child_scientific_path: str = ""
    parent_bam_list_path: str = ""
    parent_bam_paths: list[str] = field(default_factory=list)
    reference_fasta_path: str = ""
    te_fasta_path: str = ""
    out_prefix: str = "trio_denovo"

    bam_threads: int = 1
    child_min_support_reads: int = 2

    #: How far around the event to FETCH reads, versus how close a signal must
    #: be to MATCH. The fetch window is much wider so a read whose breakpoint
    #: sits just outside the match window is still examined and can still
    #: produce an ambiguous signal.
    fetch_window: int = 500
    default_match_window: int = 30
    max_match_window: int = 100
    min_softclip_len: int = 50
    min_insertion_len: int = 50
    parent_mapq_min: int = 0
    family_match_veto: bool = True
    emit_review_status: bool = True
    dry_run: bool = False


@dataclass
class DenovoChildCandidate:
    chrom: str = ""
    pos: int = -1
    te_name: str = ""
    support_reads: int = 0
    event_start: int = -1
    event_end: int = -1


@dataclass
class ParentReadEvidence:
    parent_bam_path: str = ""
    read_name: str = ""
    evidence_type: str = ""
    breakpoint_pos: int = -1
    fragment_len: int = 0
    matched_te: str = "NA"
    matched_family: str = "NA"
    veto_reason: str = "NA"


@dataclass
class ParentEvidenceSummary:
    total_support_reads: int = 0
    exact_te_reads: int = 0
    family_te_reads: int = 0
    ambiguous_signal_reads: int = 0
    support_bams: list[str] = field(default_factory=list)
    veto_reads: list[ParentReadEvidence] = field(default_factory=list)
    scanner_status: str = "NOT_RUN"


@dataclass
class DenovoCall:
    child: DenovoChildCandidate = field(default_factory=DenovoChildCandidate)
    parent_summary: ParentEvidenceSummary = field(default_factory=ParentEvidenceSummary)
    status: str = "PENDING_PARENT_SCAN"
    de_novo: str = "NA"


@dataclass
class DenovoResult:
    implementation_status: str = "SKELETON_PARENT_SCAN_NOT_IMPLEMENTED"
    child_rows_total: int = 0
    child_candidates_considered: int = 0
    parent_bams: int = 0
    calls_written: int = 0
    inherited_prefilter_hits: int = 0
    parent_veto_calls: int = 0
    review_calls: int = 0
    denovo_pass_calls: int = 0
    calls: list[DenovoCall] = field(default_factory=list)


class EvidenceMatchClass(IntEnum):
    """Ordered: a stronger match always wins in `choose_better_decision`."""

    NONE = 0
    AMBIGUOUS = 1
    FAMILY = 2
    EXACT = 3


@dataclass
class ReadEvidenceDecision:
    match_class: EvidenceMatchClass = EvidenceMatchClass.NONE
    evidence: ParentReadEvidence = field(default_factory=ParentReadEvidence)


@dataclass
class CandidateScanWindow:
    match_start: int = 0
    match_end: int = 0
    fetch_start: int = 0
    fetch_end: int = 0


def sanitize_token(text: str) -> str:
    return "".join("_" if c.isspace() else c for c in text)


def infer_te_family(te_name: str) -> str:
    """Collapse an element name onto a FAMILY, for the family-level veto.

    A separate function from `placer_py.seqtools.parse_te_name_parts` and
    deliberately coarser: that one parses library headers, this one reads the
    `te` column of a `scientific.txt` written by an earlier run, which may hold
    a bare subfamily name with no class path at all. The fallback -- everything
    before the first delimiter -- is what makes it work on both.
    """
    upper = te_name.upper()
    if not upper or upper == "NA":
        return "NA"
    if upper.startswith("ALU"):
        return "ALU"
    if upper.startswith("SVA"):
        return "SVA"
    if upper.startswith("L1") or upper.startswith("LINE1") or upper.startswith("LINE_1"):
        return "LINE1"
    if upper.startswith("ERV") or "HERV" in upper:
        return "ERV"
    for index, char in enumerate(upper):
        if char in "|#:/ _-":
            return upper[:index] if index > 0 else upper
    return upper


def evidence_type_for_source(source: InsertionFragmentSource) -> str:
    if source in (InsertionFragmentSource.CLIP_REF_LEFT,
                  InsertionFragmentSource.CLIP_REF_RIGHT):
        return "soft_clip"
    if source == InsertionFragmentSource.CIGAR_INSERTION:
        return "long_insertion"
    if source == InsertionFragmentSource.SPLIT_SA:
        return "split_sa"
    return "unknown"


def build_scan_window(candidate: DenovoChildCandidate,
                      config: DenovoConfig) -> CandidateScanWindow:
    """Two nested windows, and the padding ADAPTS to the event's own width.

    A 6 kb insertion's breakpoints are less certain than a 300 bp one's, so the
    match window grows with the event span (half of it), clamped between the
    default and the maximum. A fixed window would either miss a long event's
    parent evidence or admit a short one's neighbours.
    """
    event_start = max(0, candidate.event_start)
    event_end = max(event_start, candidate.event_end)
    span = max(1, event_end - event_start + 1)
    adaptive_padding = min(max(max(config.default_match_window, span // 2),
                               config.default_match_window),
                           config.max_match_window)
    return CandidateScanWindow(
        match_start=max(0, event_start - adaptive_padding),
        match_end=event_end + adaptive_padding,
        fetch_start=max(0, event_start - config.fetch_window),
        fetch_end=event_end + config.fetch_window)


def position_in_window(pos: int, window: CandidateScanWindow) -> bool:
    return window.match_start <= pos <= window.match_end


def nearest_breakpoint_in_window(a: int, b: int,
                                 window: CandidateScanWindow) -> int | None:
    """Whichever of two positions is in the window, or the more central one."""
    a_ok = position_in_window(a, window)
    b_ok = position_in_window(b, window)
    if not a_ok and not b_ok:
        return None
    if a_ok and not b_ok:
        return a
    if b_ok and not a_ok:
        return b
    mid = (window.match_start + window.match_end) // 2
    return a if abs(a - mid) <= abs(b - mid) else b


def choose_better_decision(lhs: ReadEvidenceDecision,
                           rhs: ReadEvidenceDecision) -> ReadEvidenceDecision:
    """The stronger match class, then the longer fragment.

    ONE VERDICT PER READ, and the strongest one it produced. A read with both a
    clip and an insertion contributes once -- otherwise a single parent read
    could veto twice, and the counts in the summary would overstate how much of
    the parent supports the locus.
    """
    if rhs.match_class > lhs.match_class:
        return rhs
    if rhs.match_class < lhs.match_class:
        return lhs
    return rhs if rhs.evidence.fragment_len > lhs.evidence.fragment_len else lhs


def make_fragment(child: DenovoChildCandidate, read_name: str, tag: str,
                  breakpoint_pos: int, source: InsertionFragmentSource,
                  start: int, sequence: str) -> InsertionFragment:
    return InsertionFragment(
        fragment_id=f"{child.chrom}:{child.pos}|{sanitize_token(read_name)}|{tag}"
                    f"|{breakpoint_pos}",
        chrom=child.chrom, anchor_pos=breakpoint_pos, read_id=read_name,
        start=start, length=len(sequence), read_len=len(sequence),
        source=source, sequence=sequence)


class ParentPoolScanner:
    """Looks for any read in any parent that could be the child's insertion.

    Takes a `fetch` CALLABLE per parent rather than opening BAMs itself, so the
    scan logic can be exercised on literals. `run_denovo` supplies real indexed
    readers.
    """

    def __init__(self, config: DenovoConfig, classifier=None,
                 fetchers: Sequence[Callable[[str, int, int], list[AlignedRead]]] | None = None
                 ) -> None:
        self.config = config
        self.classifier = classifier
        self.fetchers = list(fetchers or [])
        self.implementation_status = (
            "TARGETED_PARENT_SCAN_WITH_SPLIT_FRAGMENT_CLASSIFICATION"
            if (classifier is not None and classifier.is_enabled())
            else "TARGETED_PARENT_SCAN_WITH_SPLIT_NO_TEKMER_CLASSIFIER")

    def classify_fragments(self, child: DenovoChildCandidate, bam_path: str,
                           fragments: list[InsertionFragment]) -> list[ReadEvidenceDecision]:
        """Name each parent fragment, and rank it against the child's element.

        AN UNCLASSIFIED FRAGMENT IS AMBIGUOUS, NOT ABSENT. A parent read
        carrying inserted sequence the classifier could not name is precisely
        the case that cannot be resolved -- it may be the same element, badly
        assembled. Treating it as no evidence would report the commonest
        failure mode as a confident de novo call.
        """
        decisions: list[ReadEvidenceDecision] = []
        if not fragments:
            return decisions
        hits = (self.classifier.classify(fragments)
                if self.classifier is not None and self.classifier.is_enabled() else [])

        child_te_upper = child.te_name.upper()
        child_family = infer_te_family(child.te_name)
        for index, fragment in enumerate(fragments):
            decision = ReadEvidenceDecision()
            decision.evidence.parent_bam_path = bam_path
            decision.evidence.read_name = (fragment.read_id if fragment.read_id
                                           else sanitize_token(fragment.fragment_id))
            decision.evidence.evidence_type = evidence_type_for_source(fragment.source)
            decision.evidence.breakpoint_pos = (fragment.ref_junc_pos
                                                if fragment.ref_junc_pos >= 0
                                                else fragment.anchor_pos)
            decision.evidence.fragment_len = len(fragment.sequence)

            hit_name = hits[index].te_name if index < len(hits) else ""
            if hit_name:
                hit_family = infer_te_family(hit_name)
                decision.evidence.matched_te = hit_name
                decision.evidence.matched_family = hit_family
                if hit_name.upper() == child_te_upper:
                    decision.match_class = EvidenceMatchClass.EXACT
                    decision.evidence.veto_reason = "EXACT_TE_MATCH"
                elif (child_family != "NA" and hit_family != "NA"
                      and child_family == hit_family):
                    decision.match_class = EvidenceMatchClass.FAMILY
                    decision.evidence.veto_reason = "FAMILY_MATCH"
                else:
                    # A parent insertion of a DIFFERENT element at this locus is
                    # still ambiguous rather than clean: two insertions at one
                    # position is unlikely enough that the naming is the more
                    # probable error.
                    decision.match_class = EvidenceMatchClass.AMBIGUOUS
                    decision.evidence.veto_reason = "NON_MATCHING_TE_FRAGMENT"
            else:
                decision.match_class = EvidenceMatchClass.AMBIGUOUS
                decision.evidence.veto_reason = "UNCLASSIFIED_INSERTION_FRAGMENT"
            decisions.append(decision)
        return decisions

    def classify_read(self, child: DenovoChildCandidate, bam_path: str,
                      read: AlignedRead,
                      window: CandidateScanWindow) -> ReadEvidenceDecision:
        """One parent read's strongest statement about the child's locus.

        THE SPLIT-ALIGNMENT FALLBACK at the end is the eager half of this
        stage: a read that produced no usable fragment at all, but is
        supplementary or carries an SA tag with a breakpoint in the window, is
        still recorded as AMBIGUOUS. It is a read that goes somewhere else from
        near this locus, which is what an inherited insertion looks like when
        the parent's coverage is too low to assemble one.
        """
        best = ReadEvidenceDecision()
        if read is None or read.is_unmapped or read.is_secondary:
            return best
        if read.mapq < max(0, self.config.parent_mapq_min):
            return best

        read_name = sanitize_token(read.qname)
        clip = analyze_clip_info(read)
        fragments: list[InsertionFragment] = []
        min_softclip = max(1, self.config.min_softclip_len)
        min_insertion = max(1, self.config.min_insertion_len)

        if clip.leading >= min_softclip and position_in_window(read.pos, window):
            sequence = read.decode_subsequence(0, clip.leading)
            if sequence:
                fragments.append(make_fragment(child, read_name, "softclip_left",
                                               read.pos,
                                               InsertionFragmentSource.CLIP_REF_LEFT,
                                               0, sequence))
        if clip.trailing >= min_softclip and position_in_window(clip.ref_end, window):
            start = max(0, read.seq_len - clip.trailing)
            sequence = read.decode_subsequence(start, clip.trailing)
            if sequence:
                fragments.append(make_fragment(child, read_name, "softclip_right",
                                               clip.ref_end,
                                               InsertionFragmentSource.CLIP_REF_RIGHT,
                                               start, sequence))
        for ins in find_long_insertions(read, min_insertion):
            if not position_in_window(ins.ref_pos, window):
                continue
            sequence = read.decode_subsequence(ins.start, ins.len)
            if not sequence:
                continue
            fragments.append(make_fragment(child, read_name, "ins", ins.ref_pos,
                                           InsertionFragmentSource.CIGAR_INSERTION,
                                           ins.start, sequence))

        for decision in self.classify_fragments(child, bam_path, fragments):
            best = choose_better_decision(best, decision)

        if best.match_class == EvidenceMatchClass.NONE:
            split_bp = nearest_breakpoint_in_window(read.pos, clip.ref_end, window)
            has_split = read.is_supplementary or self._has_sa_breakpoint_in_window(
                read, child.chrom, window)
            if has_split and split_bp is not None:
                ambiguous = ReadEvidenceDecision(match_class=EvidenceMatchClass.AMBIGUOUS)
                ambiguous.evidence.parent_bam_path = bam_path
                ambiguous.evidence.read_name = read_name
                ambiguous.evidence.evidence_type = "split_sa"
                ambiguous.evidence.breakpoint_pos = split_bp
                ambiguous.evidence.fragment_len = 0
                ambiguous.evidence.veto_reason = "SPLIT_ALIGNMENT_NEAR_LOCUS"
                best = choose_better_decision(best, ambiguous)
        return best

    @staticmethod
    def _has_sa_breakpoint_in_window(read: AlignedRead, chrom: str,
                                     window: CandidateScanWindow) -> bool:
        from placer_py.alignment import parse_sa_tag_z

        sa_z = read.get_string_tag("SA")
        if not sa_z:
            return False
        for entry in parse_sa_tag_z(sa_z):
            if entry.rname == chrom and position_in_window(entry.pos - 1, window):
                return True
        return False

    def scan_candidate(self, candidate: DenovoChildCandidate) -> ParentEvidenceSummary:
        """Every parent, every read near the locus, one verdict per read.

        Keyed by `(bam, read name)` so the SAME read seen twice in one parent
        counts once, while the same read name in two parents counts twice --
        which is right, because they are different molecules.
        """
        summary = ParentEvidenceSummary(scanner_status=self.implementation_status)
        if not self.fetchers:
            return summary

        window = build_scan_window(candidate, self.config)
        best_by_read_key: dict[tuple[str, str], ReadEvidenceDecision] = {}

        for index, fetch in enumerate(self.fetchers):
            bam_path = (self.config.parent_bam_paths[index]
                        if index < len(self.config.parent_bam_paths) else f"parent{index}")
            for read in fetch(candidate.chrom, window.fetch_start, window.fetch_end):
                decision = self.classify_read(candidate, bam_path, read, window)
                if decision.match_class == EvidenceMatchClass.NONE:
                    continue
                key = (bam_path, decision.evidence.read_name)
                incumbent = best_by_read_key.get(key)
                best_by_read_key[key] = (decision if incumbent is None
                                         else choose_better_decision(incumbent, decision))

        support_bams: set[str] = set()
        for decision in best_by_read_key.values():
            if decision.match_class == EvidenceMatchClass.EXACT:
                summary.total_support_reads += 1
                summary.exact_te_reads += 1
                support_bams.add(decision.evidence.parent_bam_path)
                summary.veto_reads.append(decision.evidence)
            elif decision.match_class == EvidenceMatchClass.FAMILY:
                summary.total_support_reads += 1
                summary.family_te_reads += 1
                support_bams.add(decision.evidence.parent_bam_path)
                summary.veto_reads.append(decision.evidence)
            elif decision.match_class == EvidenceMatchClass.AMBIGUOUS:
                # NOT counted as support: an ambiguous read cannot support a
                # parental genotype, only withhold a de novo call.
                summary.ambiguous_signal_reads += 1
                summary.veto_reads.append(decision.evidence)

        summary.support_bams = sorted(support_bams)
        return summary


# ---------------------------------------------------------------------------
# Reading the child's calls, and writing the three outputs.
# ---------------------------------------------------------------------------
def parse_child_candidate_row(fields: list[str], columns: dict[str, int],
                              config: DenovoConfig) -> DenovoChildCandidate:
    """One `scientific.txt` row as a candidate.

    The event interval falls back to `pos +/- default_match_window` when the
    breakpoints are missing or inverted. That fallback is what lets this read a
    `scientific.txt` from an older release, or from a caller that reports a
    position and no interval.
    """
    def field(name: str, default: str = "") -> str:
        index = columns.get(name)
        return fields[index] if index is not None and index < len(fields) else default

    def int_field(name: str, default: int = 0) -> int:
        try:
            return int(field(name, str(default)))
        except ValueError:
            return default

    candidate = DenovoChildCandidate()
    candidate.chrom = field("chrom")
    candidate.pos = int_field("pos", -1)
    candidate.te_name = field("te", "NA")
    candidate.support_reads = int_field("support_reads", 0)

    bp_left = int_field("bp_left", -1)
    bp_right = int_field("bp_right", -1)
    if bp_left >= 0 and bp_right >= bp_left:
        candidate.event_start = bp_left
        candidate.event_end = bp_right
    else:
        candidate.event_start = max(0, candidate.pos - config.default_match_window)
        candidate.event_end = candidate.pos + config.default_match_window
    return candidate


def candidate_passes_child_filters(candidate: DenovoChildCandidate,
                                   config: DenovoConfig) -> bool:
    """A NAMED element with enough support.

    The name requirement is not cosmetic: the parent scan compares the parent's
    fragments against the child's ELEMENT, so an unnamed child call has nothing
    to compare against and would be reported de novo by default.
    """
    te_name = candidate.te_name.upper()
    if not te_name or te_name == "NA":
        return False
    return candidate.support_reads >= config.child_min_support_reads


def load_denovo_child_candidates(text: str, config: DenovoConfig
                                 ) -> tuple[list[DenovoChildCandidate], int]:
    """Parse a `scientific.txt`, returning the candidates and the row total.

    The header is found by its `#chrom` prefix, so the run-summary block above
    it is skipped without having to know how many lines it has. Columns are
    resolved BY NAME -- `placer_py/outputs.py` documents why nothing may index
    this file by position.
    """
    columns: dict[str, int] = {}
    candidates: list[DenovoChildCandidate] = []
    total_rows = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            header = line[1:].split("\t")
            if header and header[0] == "chrom":
                columns = {name: index for index, name in enumerate(header)}
            continue
        if not columns:
            continue
        total_rows += 1
        candidate = parse_child_candidate_row(line.split("\t"), columns, config)
        if candidate_passes_child_filters(candidate, config):
            candidates.append(candidate)
    return candidates, total_rows


def run_denovo(config: DenovoConfig, child_text: str,
               scanner: ParentPoolScanner | None = None) -> DenovoResult:
    """Classify every child candidate against the parent pool.

    THE THREE-WAY VERDICT, and the order it is tested in:

      1. `parent_veto` -- an exact element match always vetoes; a family match
         vetoes only when `family_match_veto` is on;
      2. `review_signal` -- any ambiguous read, OR a family match when the
         family veto is OFF (it has to land somewhere, and REVIEW is where);
      3. otherwise `DENOVO_PASS`.

    Note that `emit_review_status` only controls whether step 2 is REPORTED --
    with it off, a reviewable call becomes a pass. That is a deliberate choice
    the caller makes, and it is the one setting here that can turn an uncertain
    locus into a confident positive.
    """
    result = DenovoResult()
    result.parent_bams = len(config.parent_bam_paths)

    candidates, total_rows = load_denovo_child_candidates(child_text, config)
    result.child_rows_total = total_rows
    result.child_candidates_considered = len(candidates)

    if config.dry_run:
        result.implementation_status = "DRY_RUN"
    elif scanner is not None:
        result.implementation_status = scanner.implementation_status

    for candidate in candidates:
        call = DenovoCall(child=candidate)
        if config.dry_run or scanner is None:
            call.parent_summary.scanner_status = "DRY_RUN"
            call.status = "DRY_RUN"
            call.de_novo = "DRY_RUN"
        else:
            call.parent_summary = scanner.scan_candidate(candidate)
            parent_veto = (call.parent_summary.exact_te_reads > 0
                           or (config.family_match_veto
                               and call.parent_summary.family_te_reads > 0))
            review_signal = (call.parent_summary.ambiguous_signal_reads > 0
                             or (not config.family_match_veto
                                 and call.parent_summary.family_te_reads > 0))
            if parent_veto:
                call.status = "PARENT_VETO"
                call.de_novo = "0"
                result.parent_veto_calls += 1
            elif review_signal and config.emit_review_status:
                call.status = "PARENT_AMBIGUOUS_SIGNAL"
                call.de_novo = "REVIEW"
                result.review_calls += 1
            else:
                call.status = "DENOVO_PASS"
                call.de_novo = "1"
                result.denovo_pass_calls += 1
        result.calls.append(call)

    result.calls_written = len(result.calls)
    return result


def render_calls_tsv(result: DenovoResult) -> str:
    lines = ["#" + "\t".join((
        "chrom", "pos", "te", "child_support_reads", "child_event_start",
        "child_event_end", "parent_total_support_reads", "parent_exact_te_reads",
        "parent_family_te_reads", "parent_ambiguous_signal_reads",
        "parent_support_bams", "scanner_status", "status", "de_novo"))]
    for call in result.calls:
        summary = call.parent_summary
        lines.append("\t".join((
            call.child.chrom, str(call.child.pos), call.child.te_name,
            str(call.child.support_reads), str(call.child.event_start),
            str(call.child.event_end), str(summary.total_support_reads),
            str(summary.exact_te_reads), str(summary.family_te_reads),
            str(summary.ambiguous_signal_reads),
            ",".join(summary.support_bams) if summary.support_bams else "NA",
            summary.scanner_status, call.status, call.de_novo)))
    return "\n".join(lines) + "\n"


def render_veto_reads_tsv(result: DenovoResult) -> str:
    """Every read that withheld or vetoed a call, with its reason.

    Written even for calls that PASSED, because an ambiguous read that did not
    reach the review threshold is still the evidence a reviewer would want.
    """
    lines = ["#" + "\t".join((
        "chrom", "child_pos", "child_te", "parent_bam", "read_name",
        "evidence_type", "breakpoint_pos", "fragment_len", "matched_te",
        "matched_family", "veto_reason"))]
    for call in result.calls:
        for evidence in call.parent_summary.veto_reads:
            lines.append("\t".join((
                call.child.chrom, str(call.child.pos), call.child.te_name,
                evidence.parent_bam_path, evidence.read_name, evidence.evidence_type,
                str(evidence.breakpoint_pos), str(evidence.fragment_len),
                evidence.matched_te, evidence.matched_family, evidence.veto_reason)))
    return "\n".join(lines) + "\n"


def render_summary_txt(result: DenovoResult) -> str:
    return "\n".join((
        f"implementation_status\t{result.implementation_status}",
        f"child_rows_total\t{result.child_rows_total}",
        f"child_candidates_considered\t{result.child_candidates_considered}",
        f"parent_bams\t{result.parent_bams}",
        f"calls_written\t{result.calls_written}",
        f"inherited_prefilter_hits\t{result.inherited_prefilter_hits}",
        f"parent_veto_calls\t{result.parent_veto_calls}",
        f"review_calls\t{result.review_calls}",
        f"denovo_pass_calls\t{result.denovo_pass_calls}")) + "\n"


def write_denovo_outputs(config: DenovoConfig, result: DenovoResult) -> dict[str, str]:
    paths = {
        "calls": f"{config.out_prefix}.calls.tsv",
        "veto_reads": f"{config.out_prefix}.parent_veto_reads.tsv",
        "summary": f"{config.out_prefix}.summary.txt",
    }
    for key, renderer in (("calls", render_calls_tsv),
                          ("veto_reads", render_veto_reads_tsv),
                          ("summary", render_summary_txt)):
        with open(paths[key], "w") as handle:
            handle.write(renderer(result))
    return paths


def build_denovo_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(prog="placer denovo")
    parser.add_argument("--child-scientific", required=True)
    parser.add_argument("--parent-bam-list", required=True)
    parser.add_argument("--ref", dest="reference_fasta_path", required=True)
    parser.add_argument("--te", dest="te_fasta_path", required=True)
    parser.add_argument("--out-prefix", default="trio_denovo")
    parser.add_argument("--child-min-support-reads", type=int, default=2)
    parser.add_argument("--fetch-window", type=int, default=500)
    parser.add_argument("--default-match-window", type=int, default=30)
    parser.add_argument("--max-match-window", type=int, default=100)
    parser.add_argument("--min-softclip-len", type=int, default=50)
    parser.add_argument("--min-insertion-len", type=int, default=50)
    parser.add_argument("--parent-mapq-min", type=int, default=0)
    parser.add_argument("--no-family-match-veto", action="store_true")
    parser.add_argument("--no-emit-review-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def config_from_denovo_args(args) -> DenovoConfig:
    config = DenovoConfig(
        child_scientific_path=args.child_scientific,
        parent_bam_list_path=args.parent_bam_list,
        reference_fasta_path=args.reference_fasta_path,
        te_fasta_path=args.te_fasta_path,
        out_prefix=args.out_prefix,
        child_min_support_reads=max(1, args.child_min_support_reads),
        fetch_window=max(1, args.fetch_window),
        default_match_window=max(1, args.default_match_window),
        min_softclip_len=max(1, args.min_softclip_len),
        min_insertion_len=max(1, args.min_insertion_len),
        parent_mapq_min=max(0, args.parent_mapq_min),
        family_match_veto=not args.no_family_match_veto,
        emit_review_status=not args.no_emit_review_status,
        dry_run=args.dry_run)
    # The maximum is floored at the default rather than accepted as given: a
    # max below the default would make the clamp in `build_scan_window` invert.
    config.max_match_window = max(config.default_match_window, args.max_match_window)
    return config


def load_path_list_file(path: str) -> list[str]:
    """One path per line, blanks and `#` comments skipped."""
    with open(path) as handle:
        return [line.strip() for line in handle
                if line.strip() and not line.strip().startswith("#")]


def run_denovo_cli(argv: list[str]) -> int:
    """`placer denovo ...`: four required inputs, three outputs.

    Every missing input is fatal. A de novo run that silently proceeded without
    a parent BAM would report every child call as de novo, which is the most
    damaging possible failure for this command.
    """
    args = build_denovo_arg_parser().parse_args(argv)
    config = config_from_denovo_args(args)

    for label, path in (("child scientific.txt", config.child_scientific_path),
                        ("parent BAM list", config.parent_bam_list_path),
                        ("reference FASTA", config.reference_fasta_path),
                        ("TE FASTA", config.te_fasta_path)):
        if not os.path.exists(path):
            print(f"[PLACER][denovo] {label} not found: {path}")
            return 1

    config.parent_bam_paths = load_path_list_file(config.parent_bam_list_path)
    if not config.parent_bam_paths:
        print(f"[PLACER][denovo] no parent BAMs in {config.parent_bam_list_path}")
        return 1

    with open(config.child_scientific_path) as handle:
        child_text = handle.read()

    scanner = None
    if not config.dry_run:
        from placer_py import te_classifier
        from placer_py.bam_io import make_bam_reader

        with open(config.te_fasta_path) as handle:
            entries = te_classifier.load_te_entries_from_fasta(handle.read())
        classifier_config = PipelineConfig(te_fasta_path=config.te_fasta_path,
                                           te_kmer_size=13,
                                           te_kmer_sizes_csv="9,11,13",
                                           te_low_kmer_support_trigger=0.20)
        classifier = te_classifier.TeKmerQuickClassifier(classifier_config, entries)

        readers = [make_bam_reader(path, config.bam_threads)
                   for path in config.parent_bam_paths]
        for path, reader in zip(config.parent_bam_paths, readers):
            if not reader.can_fetch():
                print(f"[PLACER][denovo] missing BAM index for parent BAM: {path}")
                return 1
        scanner = ParentPoolScanner(config, classifier,
                                    [reader.fetch for reader in readers])

    result = run_denovo(config, child_text, scanner)
    write_denovo_outputs(config, result)
    print(render_summary_txt(result), end="")
    return 0
