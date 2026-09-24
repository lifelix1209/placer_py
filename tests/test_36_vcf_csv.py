"""
The two new outputs: `calls.vcf` and `calls.csv`.

Where to look: `placer_py/report/vcf.py`, `placer_py/report/csv_table.py` and
the `ReportContext` in `placer_py/report/context.py`.

EVERY TEST HERE RUNS ON A STRING, with no BAM and no reference, because
`ReportContext` is a frozen record of plain data rather than a callable. That
is the reason it was designed that way: the zero-dependency runner has no
`monkeypatch`, so a renderer that took a `fetch_reference` function could not
be pinned at this level at all.

THE ANCHOR TEST IS THE LOAD-BEARING ONE. Nothing in the golden vectors relates
`bp_left` to the reference base a VCF record is anchored on, so getting it
backwards would shift every record by one base and break no existing test. It
is fixed here from two independent directions -- what `tsd.py` and
`breakpoints.py` imply, and what the example dataset measures.
"""

from __future__ import annotations

import csv as csv_module
import io

import pytest
from conftest import call_or_skip

from placer_py import schema
from placer_py.core.result import PipelineResult
from placer_py.ledger import FinalCall
from placer_py.report import csv_table as C
from placer_py.report import tsv as O
from placer_py.report import vcf as V
from placer_py.report.context import Contig, ReportContext

pytestmark = pytest.mark.invariant


def a_call(**kw) -> FinalCall:
    """A resolved, committed TE call at a closed breakpoint."""
    call = FinalCall()
    call.chrom = "chr1"
    call.tid = 0
    call.bp_left = call.bp_right = call.pos = 1000
    call.te_name = "AluY"
    call.family, call.subfamily = "Alu", "AluY"
    call.family_committed = True
    call.insert_seq = "G" * 310
    call.insert_len = 310
    call.te_consensus_start, call.te_consensus_end = 1, 311
    call.genotype, call.gq, call.af = "0/1", 41, 0.458
    call.ref_span_reads, call.alt_struct_reads = 13, 11
    call.lfdr, call.lfdr_qc = 0.001, "PASS_TE_LFDR"
    call.conformal_qc = "PASS_CONFORMAL_FDR"
    call.final_qc = "PASS_TE_CLOSED"
    for key, value in kw.items():
        setattr(call, key, value)
    return call


def a_result(final=(), structural=()) -> PipelineResult:
    result = PipelineResult()
    result.final_calls = list(final)
    result.structural_calls = list(structural)
    return result


def ctx(**kw) -> ReportContext:
    base = {"anchor_bases": {("chr1", 999): "T"}, "sample_name": "NA12878",
            "contigs": (Contig("chr1", 248956422),)}
    base.update(kw)
    return ReportContext(**base)


def records(text: str) -> list[list[str]]:
    return [line.split("\t") for line in text.splitlines()
            if line and not line.startswith("#")]


def info_of(row: list[str]) -> dict[str, str]:
    out = {}
    for part in row[7].split(";"):
        key, _, value = part.partition("=")
        out[key] = value
    return out


# ------------------------------------------------- coordinates and alleles
def test_the_anchor_base_is_the_base_before_the_breakpoint_not_at_it():
    """
    THE ONE-OFF THAT DECIDES THE WHOLE FILE. `tsd.py` fetches the left flank as
    `ref[bp_left - length : bp_left]` and `breakpoints.py` advances `ref_pos`
    only on ref-consuming ops, so the insertion sits immediately BEFORE
    `ref[bp_left]`. Measured independently on the example dataset: its AluY
    locus has a 15 bp TSD at 0-based [12000, 12015) and `bp_left == 12015`.
    """
    row = call_or_skip(records, V.render_vcf(a_result([a_call()]), ctx()))[0]
    assert row[1] == "1000"          # POS is 1-based and equals bp_left
    assert row[3] == "T"             # REF is reference[bp_left - 1]
    assert V.anchor_index(a_call()) == 999


def test_an_unfetched_anchor_becomes_n_rather_than_an_empty_ref():
    """An empty REF is an unparseable VCF; N is the spec's own unknown base."""
    row = records(V.render_vcf(a_result([a_call()]), ReportContext()))[0]
    assert row[3] == V.MISSING_ANCHOR_BASE == "N"


def test_the_alt_is_the_anchor_base_followed_by_the_inserted_sequence():
    row = records(V.render_vcf(a_result([a_call()]), ctx()))[0]
    assert row[4] == "T" + "G" * 310
    assert row[4].startswith(row[3])


def test_svlen_is_the_length_of_the_alt_and_not_the_insert_len_field():
    """
    `finalization.py` sets `insert_seq` on the cluster-promoted path and never
    sets `insert_len`, so a promoted call carries a kilobase of sequence and
    reports a length of zero. Observed on the example dataset: its call at
    48016 has `insert_len == 0` and 347 inserted bases. Reading `insert_len`
    would write SVLEN=0 on a record whose ALT is 348 characters long.
    """
    row = records(V.render_vcf(a_result([a_call(insert_len=0)]), ctx()))[0]
    assert info_of(row)["SVLEN"] == "310"
    assert len(row[4]) == 311


def test_a_call_with_no_assembled_sequence_keeps_its_record():
    """
    Dropping it would make the VCF disagree with `scientific.txt` on how many
    calls there were -- the same ambiguity `write_outputs` refuses elsewhere.
    """
    text = V.render_vcf(a_result([a_call(insert_seq="", insert_len=884,
                                         max_raw_cigar_insert_len=884)]), ctx())
    row = records(text)[0]
    assert row[4] == "<INS>"
    assert "ALTSEQ_MISSING" in row[6]
    assert info_of(row)["SVLEN"] == "884"


def test_a_call_with_no_coordinate_is_counted_in_the_header_not_dropped():
    text = V.render_vcf(a_result([a_call(bp_left=-1, bp_right=-1, pos=-1)]), ctx())
    assert records(text) == []
    assert "##placerRecordsWithoutCoordinate=1" in text


def test_a_breakpoint_at_the_contig_start_never_produces_position_zero():
    row = records(V.render_vcf(a_result([a_call(bp_left=0, bp_right=0)]), ctx()))[0]
    assert row[1] == "1"


def test_a_missing_breakpoint_falls_back_to_the_midpoint_and_says_so():
    """`pos` is the MIDPOINT of two breakpoints, not a breakpoint, so falling
    back to it is strictly worse and the record is marked IMPRECISE."""
    row = records(V.render_vcf(a_result([a_call(bp_left=-1, bp_right=-1,
                                                pos=4000)]), ctx()))[0]
    assert row[1] == "4001"
    assert "IMPRECISE" in row[6]


# --------------------------------------------------------------- ordering
def test_records_are_sorted_by_contig_header_order_then_position():
    """
    `final_calls` is in finalization order, which is not coordinate order, and
    a VCF must be sorted. Header order rather than lexical: chr10 precedes chr2
    in a reference dictionary and follows it in a string sort.
    """
    calls = [a_call(chrom="chr10", bp_left=5, bp_right=5),
             a_call(chrom="chr2", bp_left=900, bp_right=900),
             a_call(chrom="chr2", bp_left=100, bp_right=100)]
    context = ctx(contigs=(Contig("chr2", 1000), Contig("chr10", 1000)))
    rows = records(V.render_vcf(a_result(calls), context))
    assert [(r[0], r[1]) for r in rows] == [("chr2", "100"), ("chr2", "900"),
                                            ("chr10", "5")]


def test_an_undeclared_contig_sorts_after_every_declared_one():
    calls = [a_call(chrom="chrUn", bp_left=5, bp_right=5),
             a_call(chrom="chr1", bp_left=900, bp_right=900)]
    rows = records(V.render_vcf(a_result(calls), ctx()))
    assert [r[0] for r in rows] == ["chr1", "chrUn"]


def test_a_final_call_precedes_a_structural_one_at_the_same_position():
    """The sort is stable and final calls are listed first."""
    rows = records(V.render_vcf(
        a_result([a_call()], [a_call(te_name="SVA")]), ctx()))
    assert [info_of(r)["MEI"] for r in rows] == ["AluY", "SVA"]


# ------------------------------------------------------ header completeness
def test_the_header_declares_every_info_key_any_record_emits():
    """Stops the writer and the header drifting apart silently."""
    calls = [a_call(), a_call(bp_right=1012, insert_seq="", ebh_selected=True,
                              ebh_e_value=812.4, tsd_type="DUP", tsd_len=15,
                              tsd_seq="ACGTACGTACGTACG", tsd_mismatches=1,
                              low_mapq_ref_span_reads=2, raw_cigar_insert_reads=7,
                              bp_ci_width=31.5, te_posterior=0.9,
                              worst_case_lfdr=0.01, latent_mechanism="TPRT",
                              te_structure_path="TE_CORE",
                              best_te_identity=0.97, best_te_query_coverage=0.9,
                              cross_family_margin=0.4,
                              mechanistic_lower_log_bf_te_vs_artifact=6.31,
                              mechanistic_lower_log_bf_te_vs_non_te=4.02)]
    text = V.render_vcf(a_result(calls), ctx())
    declared = {line.split("<ID=")[1].split(",")[0]
                for line in text.splitlines() if line.startswith("##INFO=")}
    for row in records(text):
        for key in info_of(row):
            assert key in declared, f"{key} emitted but not declared"


def test_the_header_declares_every_filter_value_any_record_emits():
    declared = {line.split("<ID=")[1].split(",")[0]
                for line in V.render_vcf(a_result(), ctx()).splitlines()
                if line.startswith("##FILTER=")}
    assert set(V.FILTER_ORDER) <= declared


def test_the_vcf_is_never_empty_even_with_no_calls():
    text = V.render_vcf(a_result(), ctx())
    assert text.startswith("##fileformat=VCFv4.2")
    assert "#CHROM\tPOS" in text and records(text) == []


# ------------------------------------------------------------ FILTER and QUAL
def test_an_abstained_family_is_filtered_rather_than_relabelled():
    row = records(V.render_vcf(a_result([a_call(family_committed=False)]),
                               ctx()))[0]
    assert "FAM_ABSTAIN" in row[6]
    assert info_of(row)["FAMSTATUS"] == "ABSTAINED"
    assert info_of(row)["FAM"] == "Alu"      # the label survives the abstention


def test_a_structural_call_shares_the_vcf_and_suppresses_the_family_flag():
    """
    For a structural insertion, abstaining on family is the definition rather
    than extra information; emitting both would make `bcftools view -f
    FAM_ABSTAIN` return a set the user did not mean.
    """
    row = records(V.render_vcf(
        a_result(structural=[a_call(family_committed=False)]), ctx()))[0]
    assert row[6] == "STRUCTURAL"


def test_an_uncalibrated_call_says_so_rather_than_passing_silently():
    row = records(V.render_vcf(
        a_result([a_call(conformal_qc="CONFORMAL_NULL_INSUFFICIENT",
                         ebh_selected=False)]), ctx()))[0]
    assert "UNCALIB" in row[6]


def test_every_filter_value_is_rederivable_from_the_emitted_columns():
    """FILTER is a summary; QC, EBHSEL and CONFQC are the original. A summary
    that lost information would make the VCF weaker than the TSV."""
    row = records(V.render_vcf(
        a_result([a_call(family_committed=False,
                         final_qc="PASS_TE_IMPRECISE",
                         conformal_qc="CONFORMAL_NULL_INSUFFICIENT")]), ctx()))[0]
    info = info_of(row)
    assert set(row[6].split(";")) == {"FAM_ABSTAIN", "IMPRECISE", "UNCALIB"}
    assert info["FAMSTATUS"] == "ABSTAINED"
    assert "PASS_TE_IMPRECISE" in info["QC"]
    assert info["CONFQC"] == "CONFORMAL_NULL_INSUFFICIENT" and "EBHSEL" not in info


def test_qual_is_missing_rather_than_zero_when_no_lfdr_was_computed():
    """A zero QUAL asserts "certainly wrong", which is not "not assessed"."""
    row = records(V.render_vcf(
        a_result([a_call(lfdr_qc="LFDR_NOT_EVALUATED")]), ctx()))[0]
    assert row[5] == "."


def test_qual_is_the_phred_of_the_lfdr_and_is_capped():
    assert V.vcf_qual(a_call(lfdr=0.001)) == "30.00"
    assert V.vcf_qual(a_call(lfdr=0.0)) == f"{V.QUAL_CAP:.2f}"


# ------------------------------------------------------------ parseability
def test_no_info_value_can_break_an_info_parser():
    """
    Splitting INFO on ';' then each part on '=' must give one or two pieces.
    This is what keeps `mechanistic_blocks` and `te_annotation_intervals` --
    which contain ';', ',' and '=' -- out of the VCF and in the CSV.
    """
    text = V.render_vcf(a_result([a_call(), a_call(bp_right=1012)]), ctx())
    for row in records(text):
        for part in row[7].split(";"):
            assert 1 <= len(part.split("=")) <= 2, part


# -------------------------------------------------------------- MEINFO tripwire
def test_meinfo_stays_unemitted_while_te_strand_is_missing_from_the_contract():
    """
    THE TRIPWIRE. `MEINFO`'s fourth field is a polarity, it is not optional,
    and this build never resolves insertion orientation -- `strand` is set to
    "NA" unconditionally and `schema.MISSING_FOR_TPRT` lists `te_strand`.
    Whoever wires orientation in gets a failing test telling them to turn
    MEINFO on in the same commit.
    """
    assert ("te_strand" in schema.MISSING_FOR_TPRT) is not V.POLARITY_RESOLVED
    text = V.render_vcf(a_result([a_call()]), ctx())
    assert "##INFO=<ID=MEINFO" in text, "it must stay declared"
    assert not any("MEINFO=" in row[7] for row in records(text))


def test_the_known_element_coordinates_are_emitted_under_their_own_keys():
    info = info_of(records(V.render_vcf(a_result([a_call()]), ctx()))[0])
    assert (info["MEI"], info["MEISTART"], info["MEIEND"]) == ("AluY", "1", "311")


# --------------------------------------------------------------------- CSV
def parse_csv(text: str) -> list[list[str]]:
    return list(csv_module.reader(io.StringIO(text)))


def test_the_csv_body_is_the_tsv_header_verbatim():
    """One column list, not two. A second would be a second thing to keep in
    step with `FinalCall`, and only the first is contract-tested."""
    header = C.call_csv_header()
    n = len(O.FINAL_CALL_COLUMNS)
    assert header[:2] == list(C.CSV_LEADING_COLUMNS)
    assert header[2:2 + n] == list(O.FINAL_CALL_COLUMNS)
    assert header[2 + n:] == list(C.CSV_TRAILING_COLUMNS)


def test_every_csv_row_has_exactly_the_header_width():
    for insert_seq in (False, True):
        for qnames in (False, True):
            text = C.render_csv(a_result([a_call(support_qnames=["r1", "r2"])],
                                         [a_call()]), insert_seq, qnames)
            rows = parse_csv(text)
            assert all(len(r) == len(rows[0]) for r in rows), (insert_seq, qnames)


def test_a_support_qname_list_is_quoted_rather_than_splitting_the_row():
    """It keeps the TSV's comma-joined spelling, so the two files carry
    byte-identical strings rather than two spellings of one list."""
    call = a_call(support_qnames=["r1", "r2", "r3"])
    rows = parse_csv(C.render_csv(a_result([call]), False, True))
    column = rows[0].index("support_qnames")
    assert rows[1][column] == O.serialize_support_qnames(["r1", "r2", "r3"])
    assert rows[1][column] == "r1,r2,r3"


def test_a_comma_bearing_annotation_field_survives_a_round_trip():
    packed = "q=12-330,t=1-311,id=0.977,cov=0.994"
    rows = parse_csv(C.render_csv(a_result([a_call(te_annotation_intervals=packed)])))
    assert rows[1][rows[0].index("te_annotation_intervals")] == packed


def test_a_semicolon_bearing_block_certificate_survives_a_round_trip():
    packed = "polya:raw=1.2,te_art=0.3;tsd:raw=2.1,te_art=0.9"
    rows = parse_csv(C.render_csv(a_result([a_call(mechanistic_blocks=packed)])))
    assert rows[1][rows[0].index("mechanistic_blocks")] == packed


def test_a_float_round_trips_at_full_precision_through_the_csv():
    rows = parse_csv(C.render_csv(a_result([a_call(lfdr=1 / 3)])))
    assert rows[1][rows[0].index("lfdr")] == repr(1 / 3)


def test_a_boolean_is_written_as_one_and_zero_like_every_other_file():
    """`csv.writer` would call str() and write True."""
    rows = parse_csv(C.render_csv(a_result([a_call(ebh_selected=True)])))
    assert rows[1][rows[0].index("ebh_selected")] == "1"


def test_every_csv_row_is_exactly_one_physical_line():
    """A CSV whose line count disagrees with its row count corrupts in
    silence. `csv`'s answer to an embedded newline is a multi-line record."""
    text = C.render_csv(a_result([a_call(best_explanation="two\nlines"),
                                  a_call(explanation_path="a\r\nb")]))
    assert len(text.splitlines()) == 3
    assert len(parse_csv(text)) == 3


def test_the_call_set_column_reconstructs_the_two_tsvs():
    rows = parse_csv(C.render_csv(a_result([a_call()], [a_call(), a_call()])))
    column = rows[0].index("call_set")
    assert [r[column] for r in rows[1:]] == ["final", "structural", "structural"]


def test_the_csv_and_the_vcf_agree_on_position():
    """`vcf_pos` is the join key, and it is not redundant: the `pos` column is
    0-based AND is the midpoint, so it is neither coordinate."""
    result = a_result([a_call(), a_call(bp_left=5000, bp_right=5012, pos=5006)])
    rows = parse_csv(C.render_csv(result))
    column = rows[0].index("vcf_pos")
    vcf_positions = [r[1] for r in records(V.render_vcf(result, ctx()))]
    assert [r[column] for r in rows[1:]] == vcf_positions
    assert rows[2][column] != rows[2][rows[0].index("pos")]
