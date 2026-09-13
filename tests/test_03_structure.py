"""Golden structural decode. Skips until `placer_py.structure` is ported."""

from __future__ import annotations

import pytest

from conftest import call_or_skip, close
from placer_py import structure

pytestmark = pytest.mark.golden

SEQS = {
    "core_plus_polya": ("G" * 220 + "A" * 30, 0.96, 0.88, 0.12, 0.0, 0.18,
                        "TE_MODEL_IN_DISTRIBUTION", 0.35,
                        "PASS_INSERT_TE_ALIGNMENT"),
    "core_only": ("G" * 250, 0.96, 0.99, 0.01, 0.0, 0.18,
                  "TE_MODEL_IN_DISTRIBUTION", 0.35,
                  "PASS_INSERT_TE_ALIGNMENT"),
    "core_transduction_polya": (
        "G" * 200 + "CTAGGCATTCGAATCGGATCCTAGGCATTCGAAT" + "A" * 25,
        0.95, 0.72, 0.28, 0.02, 0.15, "TE_MODEL_IN_DISTRIBUTION", 0.30,
        "PASS_INSERT_TE_ALIGNMENT"),
    "low_coverage_high_residual": ("G" * 140 + "C" * 110, 0.92, 0.55, 0.45,
                                   0.05, 0.06, "TE_MODEL_EDGE", 0.0,
                                   "PASS_INSERT_TE_ALIGNMENT"),
    "masked_residual": ("G" * 150 + "T" * 100, 0.90, 0.60, 0.40, 0.35, 0.04,
                        "TE_MODEL_EDGE", -0.05, "PASS_INSERT_TE_ALIGNMENT"),
    "unknown_family": ("G" * 200 + "A" * 20, 0.90, 0.84, 0.18, 0.0, 0.0,
                       "TE_MODEL_IN_DISTRIBUTION", 0.28,
                       "PASS_INSERT_TE_ALIGNMENT_UNKNOWN"),
}


def test_structural_decode_matches_the_cpp(oracle):
    for g in oracle["sequence_structure"]:
        seq, ident, cov, resid, masked, margin, model, score, qc = SEQS[g["name"]]
        assert len(seq) == g["insert_len"], (
            f"{g['name']}: test fixture drifted from the oracle generator")
        ex = call_or_skip(structure.explain_te_sequence_structure,
                          seq, qc, "L1", "L1HS", ident, cov, resid, masked,
                          margin, 0.0, model, score)
        for field in ("te_structure_log_evidence",
                      "nonte_structure_log_evidence",
                      "artifact_structure_log_evidence",
                      "structure_path_confidence", "polyA_posterior",
                      "transduction_posterior", "te_core_coverage"):
            close(getattr(ex, field), g[field], f"{g['name']}.{field}")
        assert ex.unexplained_high_complexity_bp == \
            g["unexplained_high_complexity_bp"], g["name"]


def test_polya_emission_constant_is_reproduced():
    """1.28 nats/base against an open-odds of -3.0, so a run of 3 is marginal
    and a run of 30 is decisive."""
    short = call_or_skip(structure.terminal_poly_at_run, "G" * 20 + "AAA")
    long = call_or_skip(structure.terminal_poly_at_run, "G" * 20 + "A" * 30)
    assert short == 3
    assert long == 30


@pytest.mark.regression
def test_a_and_t_runs_are_currently_conflated():
    """
    Pinned as CURRENT behaviour, not as desirable behaviour.

    `terminal_poly_at_run` counts an A/T-dominant run without recording which
    base it was, discarding the insertion's orientation. poly(A) on the + strand
    and poly(T) on the - strand are the same tail from opposite sides, and which
    appears must agree with the element alignment strand and the TSD geometry --
    a hard constraint a real insertion satisfies and a mismapped copy satisfies
    only by luck.

    The port must match this so the diff closes. Recovering the orientation is a
    modelling change and belongs with the TPRT work, where it has to flip this
    test deliberately.
    """
    a_run = call_or_skip(structure.terminal_poly_at_run, "G" * 20 + "A" * 25)
    t_run = call_or_skip(structure.terminal_poly_at_run, "G" * 20 + "T" * 25)
    assert a_run == t_run == 25
