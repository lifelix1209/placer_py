"""
The decision: decoy check per class, loci, one e-BH, TEBench's TE rule and
precise placement (`core/mechanism_selection.select_loci_coverage`), and what
finalization makes of it (`core/finalize.finalize_mechanism_calls`).
"""

import math

import pytest

from placer.core import mechanism_selection as S
from placer.core.finalize import finalize_mechanism_calls
from placer.core.ledger import EvidenceLedgerRow, FinalCall
from placer.core.result import PipelineResult

pytestmark = pytest.mark.invariant


def _fill(item, artifact, te_class, decoys, decoy_mean, coverage, covered):
    item.mech_log_lr_vs_artifact = artifact
    item.mech_decoy_count = decoys
    item.mech_decoy_mean_exp_linkage = decoy_mean
    item.te_annotation_class = te_class
    item.te_union_coverage, item.te_union_covered_bp = coverage, covered
    item.te_dominant_class = te_class
    item.te_dominant_family = "L1" if te_class == "LINE" else te_class
    return item


def row(pos, artifact, te_class="LINE", decoys=4, decoy_mean=0.1, coverage=0.9,
        covered=300):
    r = EvidenceLedgerRow(chrom="c", tid=0, pos=pos, bp_left=pos, bp_right=pos)
    r.candidate_retention_reason = "EVALUATED"
    return _fill(r, artifact, te_class, decoys, decoy_mean, coverage, covered)


def call(pos, artifact, te_class="LINE", coverage=0.9, covered=300):
    c = FinalCall(chrom="c", tid=0, pos=pos, bp_left=pos, bp_right=pos,
                  hypothesis_pos=pos)
    return _fill(c, artifact, te_class, 4, 0.1, coverage, covered)


def background(start, n=20, make=row):
    """Loci with no insertion: they fill m, and nothing selects them."""
    return [make(start + i * 1000, -5) for i in range(n)]


def wide(item, left, right, indel_reads):
    """Tested by a breakpoint interval rather than a position."""
    item.bp_left, item.bp_right, item.alt_indel_reads = left, right, indel_reads
    return item


def precise(item, indel_reads):
    item.alt_indel_reads = indel_reads
    return item


# ------------------------------------------------------------ loci and e-BH
def test_one_locus_evaluated_twice_is_tested_once_by_its_best_row():
    rows = [row(1000, 30), row(1020, 10)] + background(10_000)
    shadow = S.select_loci_coverage(rows, 0.1)
    assert shadow.loci == 21
    assert rows[0].mech_ebh_selected and not rows[1].mech_ebh_selected
    assert rows[1].mech_e_value == 0.0
    assert shadow.te_selected == 1


def test_the_insertion_test_is_on_the_artifact_ratio_alone():
    """vs_non_te is reported evidence, not a gate: a clear insertion whose
    sequence the TE model dislikes is still called, and named by the rule."""
    r = row(1000, 30)
    r.mech_log_lr_vs_non_te = -50.0
    S.select_loci_coverage([r] + background(10_000), 0.1)
    assert r.mech_ebh_selected
    assert r.mech_e_value == pytest.approx(math.exp(30))


# ------------------------------------------------------------ the TE rule
@pytest.mark.parametrize("coverage, covered, te_class, is_te", [
    (0.5, 100, "LINE", True),        # both thresholds, exactly
    (0.49, 300, "LINE", False),      # under half the insert
    (0.9, 99, "LINE", False),        # under 100 bases
    (0.9, 300, "NonTE", False),      # the covering hits are not a TE class
])
def test_the_te_rule_is_tebenchs_coverage_rule(coverage, covered, te_class, is_te):
    r = row(1000, 30, te_class=te_class, coverage=coverage, covered=covered)
    shadow = S.select_loci_coverage([r] + background(10_000), 0.1)
    assert (r.mech_ebh_selected, r.mech_structural_selected) == (is_te, not is_te)
    assert (shadow.te_selected, shadow.structural_selected) == (int(is_te), int(not is_te))


# ------------------------------------------------------------ the decoy check
def test_decoys_above_one_lower_that_class_s_artifact_evidence():
    """A class whose linkage terms score e^1 on average where there is no
    insertion is over-crediting coincidences: its artifact ratio is divided
    by the bound, the other classes are untouched."""
    lines = [row(i * 1000, 12.0, "LINE", decoys=4, decoy_mean=0.1) for i in range(20)]
    ltrs = [row(100_000 + i * 1000, 12.0, "LTR", decoys=4, decoy_mean=3.0 + 0.1 * (i % 3))
            for i in range(20)]
    shadow = S.select_loci_coverage(lines + ltrs, 0.1)
    assert shadow.checks["LINE"].factor == 1.0
    ltr = shadow.checks["LTR"]
    assert ltr.upper > ltr.mean > 1.0 and ltr.factor == ltr.upper
    assert lines[0].mech_e_value == pytest.approx(math.exp(12.0))
    assert ltrs[0].mech_e_value == pytest.approx(math.exp(12.0) / ltr.factor)


def test_a_class_with_few_decoys_uses_the_pooled_bound():
    rows = [row(i * 1000, 12.0, "LINE", decoys=4, decoy_mean=0.2) for i in range(20)]
    rows.append(row(900_000, 12.0, "RC", decoys=4, decoy_mean=5.0))
    shadow = S.select_loci_coverage(rows, 0.1)
    assert shadow.checks["RC"].loci < S.MIN_LOCI_PER_CLASS
    assert shadow.checks["RC"].factor == shadow.checks["ALL"].factor


# ------------------------------------------------------------ placement
def test_a_wide_interval_is_placed_at_its_best_supported_precise_hypothesis():
    """The testing row's interval is 400 bp wide and the VCF would write its
    left end. The call goes to the precise row, within 100 bp of the interval,
    with the most indel reads -- if it holds at least half the testing row's."""
    tested = wide(row(1200, 30), 1000, 1400, 10)
    candidates = [precise(row(1050, 5), 8),
                  precise(row(1100, 5), 6),
                  precise(row(1300, 5), 4),       # under half of 10
                  precise(row(1550, 5), 20)]      # 150 bp past the interval
    S.select_loci_coverage([tested] + candidates + background(10_000), 0.1)
    assert tested.mech_ebh_selected
    assert tested.mech_call_pos == 1050


def test_a_precise_testing_row_keeps_its_own_breakpoint():
    tested = precise(row(1000, 30), 3)
    neighbour = precise(row(1040, 5), 30)
    S.select_loci_coverage([tested, neighbour] + background(10_000), 0.1)
    assert tested.mech_ebh_selected and tested.mech_call_pos == -1


# ------------------------------------------------------------ collapse regions
def _collapsed(pos, artifact=40.0):
    r = row(pos, artifact, "NA", coverage=0.0, covered=0)
    r.ref_span_reads = 0
    return r


def test_an_alignment_collapse_region_gets_no_calls_but_stays_in_m():
    """A dense run of hypotheses that no read spans -- a centromere model the
    sample does not fit -- looks like homozygous insertions everywhere. Their
    e-values are set to 0, but they stay in the family: 0 is a valid e-value
    whatever rule chose it."""
    collapse = [_collapsed(1_000_000 + i * 400) for i in range(120)]
    elsewhere = background(5_000_000)
    lone = _collapsed(9_000_000, artifact=30.0)          # one homozygous insertion
    shadow = S.select_loci_coverage(collapse + elsewhere + [lone], 0.1)
    assert shadow.collapse_items == 120
    assert all(r.mech_collapse_region and r.mech_e_value == 0.0 for r in collapse)
    assert not any(r.mech_ebh_selected or r.mech_structural_selected for r in collapse)
    assert not lone.mech_collapse_region
    assert lone.mech_structural_selected
    assert shadow.loci > 21                               # the collapse loci still count


def test_collapse_detection_uses_relative_distances_only():
    rows = [_collapsed(1_000_000 + i * 400) for i in range(120)]
    moved = [_collapsed(r.pos + 7_777_777) for r in rows]
    assert S.collapse_region_items(rows) == S.collapse_region_items(moved)
    sparse = [_collapsed(1_000_000 + i * 1200) for i in range(120)]   # 144 kb: no +-50 kb holds 100
    assert len(S.collapse_region_items(sparse)) < 120


# ------------------------------------------------------------ finalization
def test_rows_that_were_not_evaluated_are_not_tested():
    triaged = row(5000, 99)
    triaged.candidate_retention_reason = "LEDGER_ONLY_PRE_EXPENSIVE_STAGE"
    evaluated = row(1000, 30)
    result = PipelineResult(evidence_ledger=[triaged, evaluated] + background(10_000))
    finalize_mechanism_calls(result, 0.1)
    assert evaluated.mech_ebh_selected and not triaged.mech_ebh_selected


def test_finalization_names_places_and_splits_the_calls():
    te = wide(call(1200, 30), 1000, 1400, 10)
    te.te_best_family, te.te_best_subfamily = "Alu", "AluY"   # another family's best hit
    structural = call(50_000, 30, coverage=0.2)
    result = PipelineResult(candidate_calls=[te, precise(call(1050, 5), 8), structural]
                            + background(100_000, make=call))
    finalize_mechanism_calls(result, 0.1)

    assert result.final_calls == [te] and result.structural_calls == [structural]
    assert result.candidate_calls == [] and result.final_pass_calls == 1
    # Named after the family covering the most of the insert; the best hit's
    # subfamily belongs to another family, so it is dropped rather than kept.
    assert (te.family, te.subfamily, te.te_name) == ("L1", "NA", "L1")
    assert te.te_annotation_class == "LINE" and te.family_committed
    # Placed at the precise hypothesis, as one position.
    assert te.pos == te.bp_left == te.bp_right == 1050
    assert te.final_qc == "PASS_TE_MECHANISM" and te.ebh_selected
    assert te.lfdr == pytest.approx(math.exp(-30))
    assert structural.family == "UNKNOWN" and not structural.family_committed
    assert structural.final_qc == "PASS_STRUCTURAL_MECHANISM" and structural.ebh_selected
    assert structural.pos == 50_000


def test_finalization_keeps_the_scans_evidence_tokens():
    te = call(1000, 30)
    te.final_qc = "PASS_TE_IMPRECISE"
    result = PipelineResult(candidate_calls=[te] + background(100_000, make=call))
    finalize_mechanism_calls(result, 0.1)
    assert te.final_qc.split("|") == ["PASS_TE_IMPRECISE", "PASS_TE_MECHANISM"]
