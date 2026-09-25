"""
The shadow selection: decoy check per class, loci, and the two e-BH passes.
"""

import math

import pytest

from placer.core import mechanism_selection as S
from placer.core.ledger import EvidenceLedgerRow

pytestmark = pytest.mark.invariant


def row(pos, non_te, artifact, te_class="LINE", decoys=4, decoy_mean=0.1):
    r = EvidenceLedgerRow(chrom="c", tid=0, pos=pos)
    r.candidate_retention_reason = "EVALUATED"
    r.te_annotation_class = te_class
    r.mech_log_lr_vs_non_te = non_te
    r.mech_log_lr_vs_artifact = artifact
    r.mech_decoy_count = decoys
    r.mech_decoy_mean_exp_linkage = decoy_mean
    return r


def test_one_locus_evaluated_twice_is_tested_once_by_its_best_row():
    rows = [row(1000, 30, 30), row(1020, 10, 10)] + [row(10_000 + i * 1000, -5, -5)
                                                     for i in range(20)]
    shadow = S.apply_mechanism_shadow_selection(rows, 0.1)
    assert shadow.loci == 21
    assert rows[0].mech_ebh_selected and not rows[1].mech_ebh_selected
    assert shadow.te_selected == 1


def test_an_insertion_that_is_not_a_te_is_a_structural_call():
    rows = [row(1000, -20, 30)] + [row(10_000 + i * 1000, -5, -5) for i in range(20)]
    shadow = S.apply_mechanism_shadow_selection(rows, 0.1)
    assert (shadow.te_selected, shadow.structural_selected) == (0, 1)
    assert rows[0].mech_structural_selected


def test_decoys_above_one_lower_that_class_s_artifact_evidence():
    """A class whose linkage terms score e^1 on average where there is no
    insertion is over-crediting coincidences: its artifact ratio is divided
    by the bound, the other classes are untouched."""
    lines = [row(i * 1000, 50, 12.0, "LINE", decoys=4, decoy_mean=0.1) for i in range(20)]
    ltrs = [row(100_000 + i * 1000, 50, 12.0, "LTR", decoys=4, decoy_mean=3.0 + 0.1 * (i % 3))
            for i in range(20)]
    shadow = S.apply_mechanism_shadow_selection(lines + ltrs, 0.1)
    assert shadow.checks["LINE"].factor == 1.0
    ltr = shadow.checks["LTR"]
    assert ltr.upper > ltr.mean > 1.0 and ltr.factor == ltr.upper
    assert lines[0].mech_e_value == pytest.approx(math.exp(12.0))
    assert ltrs[0].mech_e_value == pytest.approx(math.exp(12.0) / ltr.factor)


def test_a_class_with_few_decoys_uses_the_pooled_bound():
    rows = [row(i * 1000, 50, 12.0, "LINE", decoys=4, decoy_mean=0.2) for i in range(20)]
    rows.append(row(900_000, 50, 12.0, "RC", decoys=4, decoy_mean=5.0))
    shadow = S.apply_mechanism_shadow_selection(rows, 0.1)
    assert shadow.checks["RC"].loci < S.MIN_LOCI_PER_CLASS
    assert shadow.checks["RC"].factor == shadow.checks["ALL"].factor


def test_rows_that_were_not_evaluated_are_not_tested():
    triaged = row(5000, 99, 99)
    triaged.candidate_retention_reason = "LEDGER_ONLY_PRE_EXPENSIVE_STAGE"
    shadow = S.apply_mechanism_shadow_selection([triaged, row(1000, 30, 30)], 0.1)
    assert shadow.loci == 1 and not triaged.mech_ebh_selected
