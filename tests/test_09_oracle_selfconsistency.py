"""
Relationships the golden data must satisfy internally.

This file runs FULLY today, with nothing ported, and it is the most useful thing
in the suite for two reasons.

First, it pins the cross-field algebra. A port can reproduce every individual
number and still wire them together wrongly; these tests say how the fields
relate, so the port inherits the structure and not just the values.

Second, it constrains the C++ too. If a future C++ change breaks one of these,
the golden data stops being self-consistent and this file says so immediately --
which is a stronger net than the C++ suite's own directional assertions provide.

One test here asserts that a known INCONSISTENCY still exists. That is
deliberate: the per-block and aggregate non-TE weights disagree, so the
serialized blocks do not sum to the aggregate. Pinning it means the port
reproduces the current behaviour faithfully, and a later fix has to change this
test on purpose rather than discover the discrepancy by accident.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.invariant


# The aggregate weightings, transcribed from src/pipeline/mechanistic_evidence.cpp.
ART_WEIGHTS = {
    "event": 1.00, "independent": 1.00, "sequence": 1.00,
    "structure": 0.90, "boundary": 1.00, "ref_conflict": 1.00,
}
NON_WEIGHTS = {
    "event": 0.40, "independent": 0.70, "sequence": 1.00,
    "structure": 0.90, "boundary": 0.85, "ref_conflict": 0.35,
}
# The per-BLOCK non-TE weights, which are different. See the final test.
BLOCK_NON_WEIGHTS = {
    "event": 0.45, "independent": 0.65, "sequence": 1.00,
    "boundary": 0.80, "ref_conflict": 0.50,
}

PRIOR_TE_MIN = 0.05
PRIOR_NULL_ODDS_UPPER = 0.95 + 0.30      # artifact_max + non_te_max


def test_oracle_has_every_section(oracle):
    for section in ("certificates", "genotypes", "overdispersion",
                    "dependency_penalty", "sequence_structure"):
        assert section in oracle, f"golden data is missing {section!r}"
        assert oracle[section], f"golden section {section!r} is empty"


# --------------------------------------------------------------- certificates
def test_per_locus_stage_charges_no_dependency_penalty(oracle):
    """`lower_log_bf` must equal `raw_log_bf` EXACTLY at the per-locus stage.

    The certified penalty is a whole-run quantity that does not exist yet here.
    Earlier releases filled the gap with a fixed 0.65 nats; charging zero
    instead is what makes this stage a deliberately permissive screen, and it
    also tightens the eventual bound, since more rows reach the ledger and the
    calibration sample is drawn from those rows.
    """
    for cert in oracle["certificates"]:
        assert cert["lower_log_bf_te_vs_artifact"] == \
            cert["raw_log_bf_te_vs_artifact"], cert["name"]
        assert cert["lower_log_bf_te_vs_non_te"] == \
            cert["raw_log_bf_te_vs_non_te"], cert["name"]


def test_artifact_aggregate_is_the_weighted_block_sum(oracle):
    for cert in oracle["certificates"]:
        expected = (
            ART_WEIGHTS["event"] * cert["event_lower_log_lr"]
            + ART_WEIGHTS["independent"] * cert["independent_lower_log_lr"]
            + ART_WEIGHTS["sequence"] * cert["sequence_lower_log_lr"]
            + ART_WEIGHTS["structure"] * cert["structure_lower_log_lr"]
            + ART_WEIGHTS["boundary"] * cert["boundary_lower_log_lr"]
            + ART_WEIGHTS["ref_conflict"] * cert["ref_conflict_lower_log_lr"]
        )
        assert math.isclose(cert["raw_log_bf_te_vs_artifact"], expected,
                            rel_tol=1e-12, abs_tol=1e-12), cert["name"]


def test_non_te_aggregate_is_the_weighted_block_sum(oracle):
    for cert in oracle["certificates"]:
        expected = (
            NON_WEIGHTS["event"] * cert["event_lower_log_lr"]
            + NON_WEIGHTS["independent"] * cert["independent_lower_log_lr"]
            + NON_WEIGHTS["sequence"] * cert["sequence_lower_log_lr"]
            + NON_WEIGHTS["structure"] * cert["structure_lower_log_lr"]
            + NON_WEIGHTS["boundary"] * cert["boundary_lower_log_lr"]
            + NON_WEIGHTS["ref_conflict"] * cert["ref_conflict_lower_log_lr"]
        )
        assert math.isclose(cert["raw_log_bf_te_vs_non_te"], expected,
                            rel_tol=1e-12, abs_tol=1e-12), cert["name"]


def test_structure_block_is_te_minus_the_best_null(oracle):
    """`clamp(te - max(nonte, artifact), -4, 5)`."""
    for cert in oracle["certificates"]:
        raw = cert["structure_te_log_evidence"] - max(
            cert["structure_nonte_log_evidence"],
            cert["structure_artifact_log_evidence"])
        assert math.isclose(cert["structure_lower_log_lr"],
                            min(5.0, max(-4.0, raw)),
                            rel_tol=1e-12, abs_tol=1e-12), cert["name"]


def test_ambiguity_width_is_the_mean_over_blocks(oracle):
    for cert in oracle["certificates"]:
        widths = [b["ambiguity_width"] for b in cert["blocks"]]
        assert widths, cert["name"]
        assert math.isclose(cert["ambiguity_width"], sum(widths) / len(widths),
                            rel_tol=1e-12, abs_tol=1e-12), cert["name"]


def test_every_certificate_has_the_six_named_blocks(oracle):
    expected = ["event", "independent", "sequence", "structure", "boundary",
                "ref_conflict"]
    for cert in oracle["certificates"]:
        assert [b["name"] for b in cert["blocks"]] == expected, cert["name"]
        assert cert["n_blocks"] == 6


def test_signals_are_normalised_to_the_unit_interval(oracle):
    for cert in oracle["certificates"]:
        for key in ("mechanistic_support_signal", "ref_conflict_signal",
                    "artifact_context_signal"):
            assert 0.0 <= cert[key] <= 1.0, f"{cert['name']}.{key}={cert[key]}"


def test_robust_lfdr_matches_its_closed_form(oracle):
    """`null_upper / (null_upper + exp(clamp(min_bf - amb, -60, 60)) * te_min)`."""
    for cert in oracle["certificates"]:
        min_bf = min(cert["lower_log_bf_te_vs_artifact"],
                     cert["lower_log_bf_te_vs_non_te"])
        te_odds = math.exp(
            max(-60.0, min(60.0, min_bf - cert["ambiguity_width"]))
        ) * PRIOR_TE_MIN
        expected = PRIOR_NULL_ODDS_UPPER / max(
            1e-12, PRIOR_NULL_ODDS_UPPER + te_odds)
        assert math.isclose(cert["robust_worst_case_lfdr"], expected,
                            rel_tol=1e-10, abs_tol=1e-12), cert["name"]


def test_robust_qc_follows_the_lfdr(oracle):
    for cert in oracle["certificates"]:
        expected = "PASS_TE_LFDR" if cert["robust_worst_case_lfdr"] <= 0.10 \
            else "TE_LFDR_HIGH"
        assert cert["robust_qc"] == expected, cert["name"]


def test_reference_conflict_lowers_the_artifact_bayes_factor(oracle):
    """The one directional property worth keeping from the C++ suite: the same
    locus, opposed by 18 reference-spanning reads, must score lower."""
    clean = next(c for c in oracle["certificates"]
                 if c["name"] == "strong_resolved_te")
    conflicted = next(c for c in oracle["certificates"]
                      if c["name"] == "strong_with_reference_conflict")
    assert conflicted["raw_log_bf_te_vs_artifact"] < \
        clean["raw_log_bf_te_vs_artifact"]
    assert conflicted["ref_conflict_signal"] > clean["ref_conflict_signal"]


# -------------------------------------------------------- dependency penalty
def test_cap_is_log_m_over_q(oracle):
    """Derived, not chosen: e-BH's largest possible threshold is m/q at rank 1,
    so no e-value ever needs to exceed it and capping there is
    decision-neutral."""
    for entry in oracle["dependency_penalty"]:
        assert math.isclose(entry["cap_log"],
                            math.log(entry["m"] / entry["q"]),
                            rel_tol=1e-12, abs_tol=1e-12), entry["name"]


def test_penalty_is_the_log_of_the_bound(oracle):
    for entry in oracle["dependency_penalty"]:
        for side in ("art", "non"):
            assert math.isclose(entry[f"log_penalty_{side}"],
                                math.log(entry[f"sigma_upper_{side}"]),
                                rel_tol=1e-12, abs_tol=1e-12), \
                f"{entry['name']} {side}"


def test_bound_never_takes_credit_for_sigma_below_one(oracle):
    """Anti-correlated blocks would make the product already conservative. That
    is not converted into a bonus, so the penalty is floored at zero."""
    for entry in oracle["dependency_penalty"]:
        for side in ("art", "non"):
            assert entry[f"sigma_upper_{side}"] >= 1.0, f"{entry['name']} {side}"
            assert entry[f"log_penalty_{side}"] >= 0.0, f"{entry['name']} {side}"


def test_bound_lies_above_the_empirical_mean_or_on_a_clamp(oracle):
    for entry in oracle["dependency_penalty"]:
        for side in ("art", "non"):
            upper = entry[f"sigma_upper_{side}"]
            mean = entry[f"sigma_mean_{side}"]
            on_clamp = math.isclose(upper, 1.0, rel_tol=1e-12) or \
                math.isclose(upper, math.exp(entry["cap_log"]), rel_tol=1e-9)
            assert upper >= mean or on_clamp, \
                f"{entry['name']} {side}: upper {upper} < mean {mean}"


def test_estimated_flag_means_both_sides_had_two_usable_rows(oracle):
    for entry in oracle["dependency_penalty"]:
        assert entry["estimated"] == (entry["n_art"] >= 2 and entry["n_non"] >= 2), \
            entry["name"]


def test_too_few_rows_falls_back_to_the_cap(oracle):
    """No minimum-count parameter exists, and none is needed: with no variance
    the only honest bound is the cap, so nothing passes."""
    for entry in oracle["dependency_penalty"]:
        if entry["n_art"] < 2:
            assert math.isclose(entry["log_penalty_art"], entry["cap_log"],
                                rel_tol=1e-12), entry["name"]


def test_the_right_tail_is_what_the_bound_is_made_of(oracle):
    """
    The finding that justifies estimating sigma on EVERY certificated row rather
    than on the conformal null set.

    `ramp_500` and `ramp_with_right_tail` share 500 identical rows; the latter
    adds 50 high ones. sigma is a MEAN, so those 50 rows are almost the whole
    quantity -- and promotability, which the C++ once used to pick this sample,
    excludes rows for having a high aggregate. Selecting that way removes
    exactly the mass sigma is made of.
    """
    plain = next(e for e in oracle["dependency_penalty"]
                 if e["name"] == "ramp_500")
    tailed = next(e for e in oracle["dependency_penalty"]
                  if e["name"] == "ramp_with_right_tail")
    assert math.isclose(plain["sigma_upper_art"], 1.0, rel_tol=1e-12), (
        "the tail-free sample should bottom out on the floor")
    assert tailed["log_penalty_art"] > plain["log_penalty_art"] + 2.0, (
        "adding 10% high rows must move the penalty by nats, not decimals")


def test_each_side_is_bounded_separately(oracle):
    """What removed the unexplained 0.60 factor that used to relate them."""
    entry = next(e for e in oracle["dependency_penalty"]
                 if e["name"] == "asymmetric_sides")
    assert entry["log_penalty_art"] != entry["log_penalty_non"]
    ratio = entry["log_penalty_non"] / max(1e-9, entry["log_penalty_art"])
    assert not math.isclose(ratio, 0.60, abs_tol=0.01), (
        "the two sides must not be a fixed multiple of each other")


# ------------------------------------------------------------- genotypes
def test_depth_is_alt_plus_ref(oracle):
    for gt in oracle["genotypes"]:
        assert gt["depth"] == gt["alt"] + gt["ref"], gt


def test_allele_fraction_is_the_observed_ratio(oracle):
    """AF is the raw observed ratio, not a shrunken posterior mean.

    My first version of this test asserted `(alt == 0) == (best_gt == "0/0")`
    and failed on `alt=1, ref=9`, which the C++ correctly calls 0/0 -- one alt
    read in ten is entirely consistent with the error rate. The test was wrong,
    not the implementation. Pinning the ratio is the assertion that actually has
    content.
    """
    for gt in oracle["genotypes"]:
        assert 0.0 <= gt["allele_fraction"] <= 1.0, gt
        if gt["depth"] > 0:
            assert math.isclose(gt["allele_fraction"],
                                gt["alt"] / gt["depth"],
                                rel_tol=1e-12, abs_tol=1e-12), gt


def test_a_homozygous_reference_call_needs_a_low_alt_fraction(oracle):
    """0/0 and 1/1 must not both be reachable from the same alt fraction."""
    for gt in oracle["genotypes"]:
        if gt["best_gt"] == "0/0":
            assert gt["allele_fraction"] < 0.35, gt
        if gt["best_gt"] == "1/1":
            assert gt["allele_fraction"] > 0.65, gt


def test_gq_is_a_non_negative_phred(oracle):
    for gt in oracle["genotypes"]:
        assert gt["gq"] >= 0, gt
        assert gt["gq"] < 1000, gt


def test_low_depth_is_disciplined_by_gq_not_by_a_depth_gate(oracle):
    """
    There is no minimum-depth parameter, and none is needed: a shallow locus has
    a diffuse posterior and therefore a low GQ on its own. `alt=2, ref=0` must
    still genotype as 1/1 -- with low confidence -- rather than be refused.

    The vestigial depth knob was removed from the C++ for exactly this reason,
    and this test is what stops it coming back.
    """
    shallow = next(g for g in oracle["genotypes"]
                   if g["alt"] == 2 and g["ref"] == 0)
    deep = next(g for g in oracle["genotypes"]
                if g["alt"] == 10 and g["ref"] == 0)
    assert shallow["best_gt"] == "1/1"
    assert deep["best_gt"] == "1/1"
    assert shallow["gq"] < deep["gq"], (
        "shallow support must yield lower confidence, not a different call")


def test_overdispersion_widens_the_posterior(oracle):
    """Larger rho means more count dispersion, so the same 3v3 observation is
    less decisive."""
    tight = next(g for g in oracle["genotypes"]
                 if g["alt"] == 3 and g["ref"] == 3
                 and g["overdispersion"] == 0.02 and g["error_rate"] == 0.02)
    loose = next(g for g in oracle["genotypes"]
                 if g["alt"] == 3 and g["ref"] == 3
                 and g["overdispersion"] == 0.20 and g["error_rate"] == 0.02)
    assert loose["gq"] <= tight["gq"]


def test_overdispersion_estimator_returns_the_fallback_when_starved(oracle):
    starved = [e for e in oracle["overdispersion"] if e["n_sites"] <= 1]
    assert starved, "golden data should include starved cases"
    for entry in starved:
        assert math.isclose(entry["rho"], 0.02, rel_tol=1e-12), entry


def test_estimated_overdispersion_is_a_correlation(oracle):
    for entry in oracle["overdispersion"]:
        assert 0.0 <= entry["rho"] < 1.0, entry


# -------------------------------------------------- sequence structure decode
def test_structure_path_confidence_is_the_logistic_of_the_margin(oracle):
    for entry in oracle["sequence_structure"]:
        best_null = max(entry["nonte_structure_log_evidence"],
                        entry["artifact_structure_log_evidence"])
        margin = entry["te_structure_log_evidence"] - best_null
        expected = 1.0 / (1.0 + math.exp(-margin))
        assert math.isclose(entry["structure_path_confidence"], expected,
                            rel_tol=1e-10, abs_tol=1e-12), entry["name"]


def test_state_posteriors_are_probabilities(oracle):
    for entry in oracle["sequence_structure"]:
        for key in ("polyA_posterior", "transduction_posterior",
                    "structure_path_confidence", "te_core_coverage"):
            assert 0.0 <= entry[key] <= 1.0, f"{entry['name']}.{key}"


def test_a_terminal_poly_a_run_opens_the_polya_state(oracle):
    """30 A's at 1.28 nats/base against an open-odds of -3.0 must win easily."""
    with_tail = next(e for e in oracle["sequence_structure"]
                     if e["name"] == "core_plus_polya")
    without = next(e for e in oracle["sequence_structure"]
                   if e["name"] == "core_only")
    assert with_tail["polyA_posterior"] > 0.99
    assert without["polyA_posterior"] < with_tail["polyA_posterior"]


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN, and pinned rather than tolerated: a real 3' transduction is NET "
    "PENALISED despite having a dedicated HSMM state. The residual counters are "
    "computed before the transduction decode and never reduced by it, so the "
    "Viterbi path is reporting-only. Coverage falls and costs 2.45/unit, the "
    "high-complexity residual rises and costs 1.25/unit, and the transduction "
    "posterior returns only 0.35. The port must reproduce this; fixing it is a "
    "modelling change and has to flip this xfail deliberately."))
def test_transduction_is_rewarded_not_penalised(oracle):
    plain = next(e for e in oracle["sequence_structure"]
                 if e["name"] == "core_plus_polya")
    transduced = next(e for e in oracle["sequence_structure"]
                      if e["name"] == "core_transduction_polya")
    assert transduced["transduction_posterior"] > 0.5
    assert transduced["te_structure_log_evidence"] >= \
        plain["te_structure_log_evidence"]


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN inconsistency, pinned so the port reproduces it: the per-block "
    "vs-non-TE log-LRs use weights 0.45/0.65/1/0.80/0.50 while the aggregate "
    "uses 0.40/0.70/1/0.85/0.35, and the structure block differs too. So the "
    "serialized blocks do not sum to the aggregate. Harmless for the decision "
    "(only the aggregate decides) but it defeats anyone trying to re-derive a "
    "call by hand from evidence_ledger.tsv."))
def test_serialized_blocks_sum_to_the_non_te_aggregate(oracle):
    for cert in oracle["certificates"]:
        total = sum(b["te_vs_non_te"] for b in cert["blocks"])
        assert math.isclose(cert["raw_log_bf_te_vs_non_te"], total,
                            rel_tol=1e-9, abs_tol=1e-9), cert["name"]
