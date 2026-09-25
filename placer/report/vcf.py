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

MEINFO IS DECLARED AND NEVER EMITTED, and this is the one place the writer
refuses to say something the tool does not know. `MEINFO`'s fourth field is a
polarity, it is not optional, and the spec has no value for "unknown".
`placer/core/bins.py` sets `FinalCall.strand` to "NA" unconditionally --
this port never resolves insertion orientation, and `placer/schema.py` lists
`te_strand` among the observables the ledger does not carry. Writing `+` would
invent a measurement; writing `.` produces a `Number=4` field that a MEI-aware
parser will read as a real polarity. So the three components that ARE known go
out under their own keys -- `MEI`, `MEISTART`, `MEIEND` -- and `MEINFO` returns
unchanged the moment `POLARITY_RESOLVED` flips. `tests/test_36_vcf_csv.py` ties
those two facts together so the next person cannot turn one on without the
other.

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

#: Flip when `TEAlignmentEvidence` starts carrying an orientation. Pinned
#: against `schema.MISSING_FOR_TPRT["te_strand"]` so the two cannot diverge.
POLARITY_RESOLVED = False

#: Emitted in this order so a FILTER field is stable across runs.
FILTER_ORDER = ("STRUCTURAL", "FAM_ABSTAIN", "IMPRECISE", "UNCALIB",
                "ALTSEQ_MISSING")

#: `final_qc` tokens that mean the breakpoint was not closed on both sides.
_IMPRECISE_TOKENS = ("PASS_TE_IMPRECISE", "EVENT_BILATERAL_PARTIAL_ANCHOR")

_INFO_KEYS = (
    ("SVTYPE", "1", "String", "Always INS"),
    ("SVLEN", "1", "Integer", "Inserted length in bp, from the ALT allele"),
    ("CIPOS", "2", "Integer", "Offsets from POS bracketing the insertion point"),
    ("MEINFO", "4", "String",
     "Mobile element name,start,end,polarity. NEVER EMITTED by this build: "
     "polarity is unresolved. Use MEI, MEISTART and MEIEND."),
    ("MEI", "1", "String", "Mobile element name"),
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
    ("STRUCTURAL",
     "A selected structural insertion that the TE-calibrated mode set aside. "
     "A call, not a reject"),
    ("FAM_ABSTAIN",
     "Insertion called; TE family label not committed"),
    ("IMPRECISE",
     "Breakpoint not closed on both sides"),
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


def _qc_tokens(qc: str) -> list[str]:
    return [token for token in (qc or "").split("|") if token]


def vcf_pos(call: FinalCall) -> int:
    """1-based POS, or -1 when the call carries no usable coordinate.

    `bp_left` is preferred over `pos` because `pos` is the MIDPOINT of the two
    breakpoints, not a breakpoint -- falling back to it is strictly worse, and
    a call that has to is marked IMPRECISE.
    """
    if call.bp_left >= 1:
        return call.bp_left
    if call.bp_left == 0:
        # No preceding base exists. Anchoring on index 0 shifts the insertion
        # right by one, which is the VCF convention for a contig-start
        # insertion and not a real TE locus.
        return 1
    if call.pos >= 0:
        return call.pos + 1
    return -1


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


def vcf_filters(call: FinalCall, *, structural: bool,
                alt_is_symbolic: bool) -> list[str]:
    """The FILTER column, as a list, in `FILTER_ORDER`.

    Every value is re-derivable from the emitted `QC=`, `EBHSEL` and `CONFQC`,
    so the summary loses nothing -- it just saves a consumer from parsing QC
    tokens to ask "is this a clean call?".
    """
    flags = set()
    if structural:
        flags.add("STRUCTURAL")
    elif not call.family_committed:
        # For a structural insertion, abstaining on family is the definition
        # rather than extra information, and emitting both would make
        # `bcftools view -f FAM_ABSTAIN` return a set the user did not mean.
        flags.add("FAM_ABSTAIN")
    tokens = _qc_tokens(call.final_qc)
    if any(token in _IMPRECISE_TOKENS for token in tokens) or call.bp_left < 0:
        flags.add("IMPRECISE")
    if not call.ebh_selected and not call.conformal_qc.startswith("PASS_"):
        flags.add("UNCALIB")
    if alt_is_symbolic:
        flags.add("ALTSEQ_MISSING")
    return [name for name in FILTER_ORDER if name in flags]


def _info_pairs(call: FinalCall, *, svlen: int) -> list[str]:
    out = ["SVTYPE=INS", f"SVLEN={svlen}"]
    if call.bp_right > call.bp_left >= 0:
        out.append(f"CIPOS=0,{call.bp_right - call.bp_left}")
    out.append(f"MEI={call.te_name or 'UNKNOWN'}")
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


def vcf_record(call: FinalCall, context: ReportContext, *,
               structural: bool) -> str:
    """One data line, tab separated."""
    ref, alt, symbolic = vcf_ref_and_alt(call, context)
    svlen = vcf_svlen(call, symbolic)
    filters = vcf_filters(call, structural=structural, alt_is_symbolic=symbolic)
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
    """Both call sets, coordinate-sorted, as one VCF.

    ONE FILE RATHER THAN TWO. Structural insertions are SELECTED calls that the
    TE-calibrated mode set aside, not rejects, and splitting them into their own
    VCF would force every consumer to learn PLACER's internal routing before it
    could read either file. `FILTER=STRUCTURAL` says the same thing in the
    vocabulary a VCF reader already has.

    SORTING IS REQUIRED AND IS NOT FREE HERE: `result.final_calls` is in
    finalization order, which is not coordinate order. The sort is stable and
    final calls are listed first, so two calls at one position keep that
    precedence.
    """
    context = context if context is not None else ReportContext()
    ranks = contig_ranks(context)
    unranked: dict[str, int] = {}

    def rank(chrom: str) -> int:
        if chrom in ranks:
            return ranks[chrom]
        # Undeclared contigs sort after every declared one, by first appearance.
        return len(ranks) + unranked.setdefault(chrom, len(unranked))

    rows = ([(call, False) for call in result.final_calls]
            + [(call, True) for call in result.structural_calls])
    usable = [(call, structural) for call, structural in rows
              if vcf_pos(call) >= 1]
    skipped = len(rows) - len(usable)
    usable.sort(key=lambda item: (rank(item[0].chrom), vcf_pos(item[0])))

    lines = vcf_header_lines(result, context, skipped)
    lines.extend(vcf_record(call, context, structural=structural)
                 for call, structural in usable)
    return "\n".join(lines) + "\n"
