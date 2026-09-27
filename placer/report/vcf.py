"""VCF 4.2, with the inserted sequence written out as the ALT allele.

WHY EXPLICIT BASES RATHER THAN `<INS:ME:ALU>`. The sequence is the evidence. A
symbolic allele sends every consumer back to a second file to see what was
actually inserted, and the whole reason this caller assembles a consensus is to
have those bases. The cost is real and is stated rather than discovered: with
explicit ALTs the file is roughly the size of every insert sequence combined,
so a whole-genome run produces tens of megabytes where the TSVs default
`insert_seq` off for exactly that reason. `bgzip` handles it.

THE ANCHOR CONVENTION, which is a one-off that nothing upstream pins. `placer/core/tsd.py` fetches the left flank as `ref[bp_left - length :
bp_left]`, and `placer/core/breakpoints.py` advances `ref_pos` only on
ref-consuming ops, so at a CIGAR `I` it holds the coordinate of the first
reference base AFTER the flank. Both say the insertion sits immediately before
`ref[bp_left]`. Measured independently on the example dataset: its AluY locus
has a 15 bp TSD occupying 0-based [12000, 12015) and `bp_left == 12015`, the
first base past it. So:

    VCF POS (1-based)     = bp_left
    anchor index (0-based) = bp_left - 1
    REF                   = reference[bp_left - 1]
    ALT                   = REF + insert_seq

Getting this backwards would shift every record by one base and break no test,
which is why it is written down here and pinned in `tests/test_36_vcf_csv.py`.

MEINFO IS EMITTED ONLY WHERE ITS FOUR FIELDS ARE ALL KNOWN. Its fourth field
is a polarity, it is not optional, and the spec has no value for "unknown":
writing `+` for an unresolved call would invent a measurement, and `.` makes a
`Number=4` field that a MEI-aware parser reads as a real polarity. The polarity
is the strand of the best TE alignment (`FinalCall.strand`, "+" or "-"; the
insert is in reference orientation), so a call with no oriented hit, or no
consensus interval, gets no MEINFO. `MEI`, `MEISTART` and `MEIEND` go out on
every record regardless. MEINFO's START/END are 1-based inclusive on the
element consensus, the convention MELT writes; MEISTART/MEIEND keep their
declared 0-based half-open form.

SVLEN COMES FROM THE ALT, NOT FROM `insert_len`. The cluster-promoted path in
`placer/core/finalization.py` once set `insert_seq` without `insert_len`, so
a promoted call could carry sequence and report a length of zero. That is now
fixed at the source; the ALT is still the definition here because it is the
sequence the record actually carries.
"""

from __future__ import annotations

import math

from placer.core.ledger import FinalCall
from placer.report.context import ReportContext, contig_ranks

#: What an unfetched anchor renders as. Never "": an empty REF is unparseable.
MISSING_ANCHOR_BASE = "N"

#: `lfdr == 0.0` phred-transforms to +inf, which is not a VCF number.
QUAL_CAP = 1000.0

#: `TEAlignmentEvidence` carries the alignment strand, so MEINFO is emitted
#: where it is known. Pinned against `schema.MISSING_FOR_TPRT` in
#: `tests/test_36_vcf_csv.py` so the two cannot diverge again.
POLARITY_RESOLVED = True

#: Emitted in this order so a FILTER field is stable across runs.
FILTER_ORDER = ("FAM_ABSTAIN", "IMPRECISE", "UNCALIB", "ALTSEQ_MISSING")

#: A breakpoint the decision could narrow only to an interval is reported at
#: the interval's MIDDLE, and passes, when it is at most this wide: from the
#: middle every point of it is within 100 bp, the tolerance TEBench matches
#: calls at. A wider one is reported at its left end and flagged IMPRECISE.
#: Neither end is the right one to report: on HG002 chr1 the truth sat at
#: 0.02 / 0.28 / 0.78 of the way across (quartiles). Adopted 2026-09-27 on
#: HG002 chr2-8: TP 849 -> 923, FP 38 -> 42 (docs/development-strategy.md,
#: section 8). Until then IMPRECISE came from the scan's joint decision, whose
#: explanation comparison flagged calls that had an exact breakpoint.
MAX_REPORTED_INTERVAL_BP = 200


def breakpoint_is_imprecise(bp_left: int, bp_right: int) -> bool:
    """IMPRECISE: no breakpoint, or only an interval wider than
    MAX_REPORTED_INTERVAL_BP."""
    return bp_left < 0 or bp_right - bp_left > MAX_REPORTED_INTERVAL_BP


def reported_breakpoint(bp_left: int, bp_right: int, pos: int = -1) -> int:
    """The 1-based POS a call with these breakpoints is written at, or -1."""
    if 0 <= bp_left < bp_right and bp_right - bp_left <= MAX_REPORTED_INTERVAL_BP:
        return max(1, (bp_left + bp_right) // 2)
    if bp_left >= 1:
        return bp_left
    if bp_left == 0:
        # No preceding base exists. Anchoring on index 0 shifts the insertion
        # right by one, which is the VCF convention for a contig-start
        # insertion and not a real TE locus.
        return 1
    if pos >= 0:
        return pos + 1
    return -1

_INFO_KEYS = (
    ("SVTYPE", "1", "String", "Always INS"),
    ("SVLEN", "1", "Integer", "Inserted length in bp, from the ALT allele"),
    ("CIPOS", "2", "Integer", "Offsets from POS bracketing the insertion point"),
    ("MEINFO", "4", "String",
     "Mobile element name,start,end,polarity: start and end 1-based inclusive "
     "on the element consensus, polarity the strand of the element relative "
     "to the reference. Only on records where all four are known"),
    ("MEI", "1", "String", "Mobile element name"),
    ("TE_CLASS", "1", "String",
     "RepeatMasker class of the element: LINE, SINE, Retroposon, PLE, LTR, "
     "DNA, RC, Unknown or NonTE"),
    ("TE_SUPERFAMILY", "1", "String", "Superfamily of the element"),
    ("LTRFORM", "1", "String",
     "LTR elements only: full (LTR-internal-LTR), solo, internal or partial"),
    ("FAM", "1", "String", "TE family"),
    ("SUBFAM", "1", "String", "TE subfamily"),
    ("FAMSTATUS", "1", "String",
     "COMMITTED or ABSTAINED. Separate from the label because a library may "
     "contain a family literally named Unknown"),
    ("MEISTART", "1", "Integer", "Insert start on the element consensus, 0-based"),
    ("MEIEND", "1", "Integer", "Insert end on the element consensus, exclusive"),
    ("IDENT", "1", "Float", "Best TE alignment identity"),
    ("QCOV", "1", "Float", "Best TE alignment query coverage"),
    ("XFAM", "1", "Float", "Cross-family score margin"),
    ("TSD", "1", "String", "Target site duplication geometry"),
    ("TSDLEN", "1", "Integer", "Target site duplication length"),
    ("TSDSEQ", "1", "String", "Target site duplication sequence"),
    ("TSDMM", "1", "Integer", "Mismatches within the target site duplication"),
    ("LOWMQREF", "1", "Integer",
     "Reference-spanning reads below the mapping quality bar: the part of "
     "AD[0] that is untrustworthy"),
    ("CIGINS", "1", "Integer", "Reads with a raw CIGAR insertion at this locus"),
    ("BPCIW", "1", "Float", "90% credible width of the breakpoint posterior"),
    ("TEPOST", "1", "Float", "Posterior probability of the TE hypothesis"),
    ("LFDR", "1", "Float", "Local false discovery rate; QUAL is its phred transform"),
    ("LFDRMAX", "1", "Float", "Certified upper bound on the local FDR"),
    ("MECHART", "1", "Float",
     "Dependency-penalised log Bayes factor, TE versus artifact"),
    ("MECHNONTE", "1", "Float",
     "Dependency-penalised log Bayes factor, TE versus non-TE"),
    ("EBHE", "1", "Float", "e-BH e-value"),
    ("EBHSEL", "0", "Flag", "Selected by e-BH"),
    ("CONFQC", "1", "String", "Conformal route verdict"),
    ("TESTRUCT", "1", "String", "Decoded TE structure path"),
    ("MECH", "1", "String", "Latent insertion mechanism"),
    ("QC", "1", "String",
     "final_qc verbatim, pipe-joined. FILTER summarises this; this is the "
     "original"),
)

_FILTER_KEYS = (
    ("FAM_ABSTAIN",
     "Insertion called; TE family label not committed"),
    ("IMPRECISE",
     f"Breakpoint known only to an interval wider than {MAX_REPORTED_INTERVAL_BP} bp "
     "(see CIPOS), or not at all"),
    ("UNCALIB",
     "Reached the output by a route with no FDR control: neither e-BH "
     "selected it nor a conformal route passed it"),
    ("ALTSEQ_MISSING",
     "No assembled insert sequence, so ALT is the symbolic <INS>"),
)

_FORMAT_KEYS = (
    ("GT", "1", "String", "Genotype"),
    ("GQ", "1", "Integer", "Genotype quality"),
    ("DP", "1", "Integer", "Read depth at the locus: AD[0]+AD[1]"),
    ("AD", "R", "Integer",
     "Reference-spanning and insertion-supporting read counts"),
    ("AF", "A", "Float", "Estimated allele fraction"),
)


def vcf_pos(call: FinalCall) -> int:
    """1-based POS, or -1 when the call carries no usable coordinate.

    The breakpoint itself when it is one base; the middle of an interval up to
    MAX_REPORTED_INTERVAL_BP wide; the left end of a wider one. `pos` is the
    fallback only for a call with no breakpoint at all, which is IMPRECISE.
    """
    return reported_breakpoint(call.bp_left, call.bp_right, call.pos)


def anchor_index(call: FinalCall) -> int:
    """0-based index of the base the ALT is anchored on."""
    return max(0, vcf_pos(call) - 1)


def vcf_ref_and_alt(call: FinalCall,
                    context: ReportContext) -> tuple[str, str, bool]:
    """(REF, ALT, alt_is_symbolic)."""
    base = context.anchor_bases.get((call.chrom, anchor_index(call)),
                                    MISSING_ANCHOR_BASE)
    base = (base or MISSING_ANCHOR_BASE)[:1].upper() or MISSING_ANCHOR_BASE
    if not call.insert_seq:
        return base, "<INS>", True
    return base, base + call.insert_seq.upper(), False


def vcf_svlen(call: FinalCall, alt_is_symbolic: bool) -> int:
    """Length of the insertion.

    From the ALT when there is one. When there is not, the best surviving
    statement of the length is whichever of the three recorded measures is
    largest.
    """
    if not alt_is_symbolic:
        return len(call.insert_seq)
    return max(call.insert_len, call.max_raw_cigar_insert_len,
               call.event_consensus_len, 0)


def vcf_qual(call: FinalCall) -> str:
    """Phred of the local FDR, or "." when none was computed.

    "." rather than "0.00": a zero QUAL asserts "certainly wrong", which is a
    different claim from "not assessed".
    """
    if call.lfdr_qc == "LFDR_NOT_EVALUATED" or not math.isfinite(call.lfdr):
        return "."
    value = -10.0 * math.log10(max(call.lfdr, 1e-100))
    return f"{min(value, QUAL_CAP):.2f}"


def vcf_filters(call: FinalCall, *, alt_is_symbolic: bool) -> list[str]:
    """The FILTER column, as a list, in `FILTER_ORDER`.

    Every value is re-derivable from the record -- CIPOS, `EBHSEL`, `CONFQC`,
    the family INFO keys -- so the summary loses nothing; it just saves a
    consumer from working out "is this a clean call?".
    """
    flags = set()
    if not call.family_committed:
        flags.add("FAM_ABSTAIN")
    if breakpoint_is_imprecise(call.bp_left, call.bp_right):
        flags.add("IMPRECISE")
    if not call.ebh_selected and not call.conformal_qc.startswith("PASS_"):
        flags.add("UNCALIB")
    if alt_is_symbolic:
        flags.add("ALTSEQ_MISSING")
    return [name for name in FILTER_ORDER if name in flags]


def _meinfo(call: FinalCall) -> str:
    """`NAME,START,END,POLARITY`, or "" when any of the four is unknown.

    A call whose family ABSTAINED has no name it stands behind, so it gets no
    MEINFO either, whatever the alignment strand was.
    """
    if not POLARITY_RESOLVED or call.strand not in ("+", "-"):
        return ""
    if not call.family_committed:
        return ""
    if not (0 <= call.te_consensus_start < call.te_consensus_end):
        return ""
    name = call.te_name or "UNKNOWN"
    return (f"{name},{call.te_consensus_start + 1},{call.te_consensus_end},"
            f"{call.strand}")


def _info_pairs(call: FinalCall, *, svlen: int) -> list[str]:
    out = ["SVTYPE=INS", f"SVLEN={svlen}"]
    if call.bp_right > call.bp_left >= 0:
        pos = vcf_pos(call)
        out.append(f"CIPOS={call.bp_left - pos},{call.bp_right - pos}")
    out.append(f"MEI={call.te_name or 'UNKNOWN'}")
    meinfo = _meinfo(call)
    if meinfo:
        out.append(f"MEINFO={meinfo}")
    # Class and superfamily are claims about the element, so they follow the
    # family's commitment: an abstaining call does not name what it inserted.
    if call.family_committed:
        if call.te_annotation_class and call.te_annotation_class != "NA":
            out.append(f"TE_CLASS={call.te_annotation_class}")
        if call.te_annotation_order and call.te_annotation_order != "NA":
            out.append(f"TE_SUPERFAMILY={call.te_annotation_order}")
        if call.ltr_form and call.ltr_form != "NA":
            out.append(f"LTRFORM={call.ltr_form}")
    out.append(f"FAM={call.family}")
    out.append(f"SUBFAM={call.subfamily}")
    out.append("FAMSTATUS=" + ("COMMITTED" if call.family_committed
                               else "ABSTAINED"))
    if call.te_consensus_start >= 0:
        out.append(f"MEISTART={call.te_consensus_start}")
    if call.te_consensus_end >= 0:
        out.append(f"MEIEND={call.te_consensus_end}")
    for key, value in (("IDENT", call.best_te_identity),
                       ("QCOV", call.best_te_query_coverage),
                       ("XFAM", call.cross_family_margin)):
        if value:
            out.append(f"{key}={value:.4g}")
    out.append(f"TSD={call.tsd_type}")
    if call.tsd_len:
        out.append(f"TSDLEN={call.tsd_len}")
    if call.tsd_seq and call.tsd_seq != "NA":
        out.append(f"TSDSEQ={call.tsd_seq}")
    if call.tsd_mismatches:
        out.append(f"TSDMM={call.tsd_mismatches}")
    if call.low_mapq_ref_span_reads:
        out.append(f"LOWMQREF={call.low_mapq_ref_span_reads}")
    if call.raw_cigar_insert_reads:
        out.append(f"CIGINS={call.raw_cigar_insert_reads}")
    if call.bp_ci_width:
        out.append(f"BPCIW={call.bp_ci_width:.4g}")
    for key, value in (("TEPOST", call.te_posterior),
                       ("LFDR", call.lfdr),
                       ("LFDRMAX", call.worst_case_lfdr),
                       ("MECHART", call.mechanistic_lower_log_bf_te_vs_artifact),
                       ("MECHNONTE", call.mechanistic_lower_log_bf_te_vs_non_te),
                       ("EBHE", call.ebh_e_value)):
        if value:
            out.append(f"{key}={value:.6g}")
    if call.ebh_selected:
        out.append("EBHSEL")
    if call.conformal_qc and call.conformal_qc != "NA":
        out.append(f"CONFQC={call.conformal_qc}")
    if call.te_structure_path and call.te_structure_path != "NA":
        out.append(f"TESTRUCT={call.te_structure_path}")
    if call.latent_mechanism and call.latent_mechanism != "NA":
        out.append(f"MECH={call.latent_mechanism}")
    out.append(f"QC={call.final_qc or 'NA'}")
    return out


def vcf_sample(call: FinalCall) -> tuple[str, str]:
    """(FORMAT, the one sample column)."""
    ref_reads = max(0, call.ref_span_reads)
    alt_reads = max(0, call.alt_struct_reads)
    return ("GT:GQ:DP:AD:AF",
            f"{call.genotype}:{call.gq}:{ref_reads + alt_reads}:"
            f"{ref_reads},{alt_reads}:{call.af:.4g}")


def vcf_record(call: FinalCall, context: ReportContext) -> str:
    """One data line, tab separated."""
    ref, alt, symbolic = vcf_ref_and_alt(call, context)
    svlen = vcf_svlen(call, symbolic)
    filters = vcf_filters(call, alt_is_symbolic=symbolic)
    fmt, sample = vcf_sample(call)
    return "\t".join([
        call.chrom, str(vcf_pos(call)), ".", ref, alt, vcf_qual(call),
        ";".join(filters) if filters else "PASS",
        ";".join(_info_pairs(call, svlen=svlen)), fmt, sample])


def vcf_header_lines(result, context: ReportContext,
                     skipped_without_coordinate: int = 0) -> list[str]:
    """Everything above `#CHROM`, plus the `#CHROM` line itself."""
    from placer import __version__

    lines = ["##fileformat=VCFv4.2"]
    if context.file_date:
        lines.append(f"##fileDate={context.file_date}")
    lines.append(f"##source=placer {__version__}")
    if context.reference_path:
        lines.append(f"##reference=file://{context.reference_path}")
    lines.extend(f"##contig=<ID={c.name},length={c.length}>"
                 for c in context.contigs)
    # The calibration constants, for the same reason scientific.txt carries
    # them: a file reporting selections without reporting what they were
    # calibrated against cannot be audited.
    lines.append(f"##placerDependencySigma={result.estimated_dependency_sigma!r}")
    lines.append(f"##placerDependencyPenalty={result.estimated_dependency_penalty!r}")
    lines.append(f"##placerDependencyNullCount={result.dependency_penalty_null_count}")
    if skipped_without_coordinate:
        lines.append(f"##placerRecordsWithoutCoordinate={skipped_without_coordinate}")
    if not POLARITY_RESOLVED:
        lines.append(
            '##placerNote="MEINFO is declared but emitted on no record: this '
            "build does not resolve insertion polarity, and MEINFO's fourth "
            'field has no spec-defined value for unknown. See MEI, MEISTART '
            'and MEIEND."')
    lines.append('##ALT=<ID=INS,Description="Insertion of novel sequence whose '
                 'bases could not be assembled; length is in SVLEN">')
    lines.extend(f'##FILTER=<ID={name},Description="{desc}">'
                 for name, desc in _FILTER_KEYS)
    lines.extend(f'##INFO=<ID={key},Number={num},Type={typ},Description="{desc}">'
                 for key, num, typ, desc in _INFO_KEYS)
    lines.extend(f'##FORMAT=<ID={key},Number={num},Type={typ},Description="{desc}">'
                 for key, num, typ, desc in _FORMAT_KEYS)
    lines.append("#" + "\t".join(
        ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT",
         context.sample_name]))
    return lines


def render_vcf(result, context: ReportContext | None = None) -> str:
    """The TE calls, coordinate-sorted.

    TE CALLS ONLY. PLACER is a TE caller: an insertion the decision selects
    whose insert is not a TE is recorded in the evidence ledger
    (`mech_structural_selected`) and counted in `scientific.txt`, not
    reported. Until 2026-09-27 they were written here as `FILTER=STRUCTURAL`.

    SORTING IS REQUIRED AND IS NOT FREE HERE: `result.final_calls` is in
    finalization order, which is not coordinate order. The sort is stable, so
    two calls at one position keep that order.
    """
    context = context if context is not None else ReportContext()
    ranks = contig_ranks(context)
    unranked: dict[str, int] = {}

    def rank(chrom: str) -> int:
        if chrom in ranks:
            return ranks[chrom]
        # Undeclared contigs sort after every declared one, by first appearance.
        return len(ranks) + unranked.setdefault(chrom, len(unranked))

    usable = [call for call in result.final_calls if vcf_pos(call) >= 1]
    skipped = len(result.final_calls) - len(usable)
    usable.sort(key=lambda call: (rank(call.chrom), vcf_pos(call)))

    lines = vcf_header_lines(result, context, skipped)
    lines.extend(vcf_record(call, context) for call in usable)
    return "\n".join(lines) + "\n"
