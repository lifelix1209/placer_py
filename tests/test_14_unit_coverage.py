"""
Direct coverage for the helpers the higher-level tests only reach indirectly.

An audit of the ported modules found nineteen public functions that no test
named, even though most were exercised through `build_certificate`, `evaluate`
and friends. Indirect exercise is not the same as a pinned property: a helper
can be wrong in a way the caller happens to cancel out, and it will stay wrong
until someone reuses it somewhere the cancellation does not apply.

So each one gets an assertion about what it is FOR, not just that it runs.
"""

from __future__ import annotations

import math

import pytest
from conftest import call_or_skip, close

from placer_py.core import blocks, dependency, genotype, integrate, structure, tprt
from placer_py.core import policy as P

pytestmark = pytest.mark.invariant


# ------------------------------------------------------------------ blocks
def test_count_signal_saturates_at_the_scale():
    """`1 - exp(-n/s)`: at n = s it is 1 - 1/e, and it never reaches 1."""
    close(call_or_skip(blocks.count_signal, 8, 8.0), 1.0 - math.exp(-1.0),
          "count_signal at n = scale")
    assert call_or_skip(blocks.count_signal, 0, 8.0) == 0.0
    assert call_or_skip(blocks.count_signal, -3, 8.0) == 0.0
    assert call_or_skip(blocks.count_signal, 10_000, 8.0) < 1.0000001
    # Smaller scale saturates faster: that is what the constant MEANS.
    assert call_or_skip(blocks.count_signal, 4, 4.0) > \
        call_or_skip(blocks.count_signal, 4, 8.0)


def test_mechanistic_read_signal_needs_both_count_and_fraction():
    """It is `sqrt(count_part * fraction_part)`, so either being zero zeroes it.

    The fraction guards against a locus with many alt reads of which only one
    carries a mechanistic signature; the count guards against a locus with one
    read that happens to be entirely mechanistic.
    """
    def existence(**kw):
        return P.EventExistenceEvidence(**kw)

    full = call_or_skip(blocks.mechanistic_read_signal,
                        existence(alt_struct_reads=10, alt_split_reads=5,
                                  alt_indel_reads=3, alt_left_clip_reads=2,
                                  alt_right_clip_reads=2), None)
    assert 0.0 < full <= 1.0

    no_alt = call_or_skip(blocks.mechanistic_read_signal,
                          existence(alt_struct_reads=0, alt_split_reads=5,
                                    alt_indel_reads=3, alt_left_clip_reads=2,
                                    alt_right_clip_reads=2), None)
    assert no_alt == 0.0
    no_mech = call_or_skip(blocks.mechanistic_read_signal,
                           existence(alt_struct_reads=10, alt_split_reads=0,
                                     alt_indel_reads=0, alt_left_clip_reads=0,
                                     alt_right_clip_reads=0), None)
    assert no_mech == 0.0
    # Clip reads count only as a MATCHED pair -- one-sided clipping is not a
    # mechanistic signature on its own.
    one_sided = call_or_skip(blocks.mechanistic_read_signal,
                             existence(alt_struct_reads=10, alt_split_reads=0,
                                       alt_indel_reads=0, alt_left_clip_reads=8,
                                       alt_right_clip_reads=0), None)
    assert one_sided == 0.0


def test_clip_insert_concordance_only_counts_when_it_passes():
    existence = P.EventExistenceEvidence(
        alt_struct_reads=10, alt_split_reads=2, alt_indel_reads=0,
        alt_left_clip_reads=0, alt_right_clip_reads=0)
    failing = P.ClipInsertConcordanceEvidence(
        pass_=False, full_insert_reads=6, left_clip_reads=3, right_clip_reads=3)
    passing = P.ClipInsertConcordanceEvidence(
        pass_=True, full_insert_reads=6, left_clip_reads=3, right_clip_reads=3)
    assert call_or_skip(blocks.mechanistic_read_signal, existence, passing) > \
        call_or_skip(blocks.mechanistic_read_signal, existence, failing)


def test_event_signal_weights_sum_to_one():
    """0.45 support + 0.30 independence + 0.15 quality + 0.10 AF, so a locus
    that maxes every input reaches exactly 1 and the weights are a convex
    combination rather than an arbitrary scale."""
    maxed = call_or_skip(blocks.event_signal,
                         P.EventExistenceEvidence(alt_struct_reads=10_000,
                                                  gq=60, af=1.0), 1.0)
    close(maxed, 1.0, "event_signal with every input maxed")
    floor = call_or_skip(blocks.event_signal,
                         P.EventExistenceEvidence(alt_struct_reads=0, gq=0,
                                                  af=0.0), 0.0)
    assert floor == 0.0


def test_boundary_signal_rewards_a_tsd_and_closed_geometry():
    def boundary(btype, defined=True, canonical=True, consistent=True):
        return P.BoundaryEvidence(geometry_defined=defined,
                                  canonical_pass=canonical,
                                  evidence_consistent=consistent,
                                  boundary_type=btype)

    seg = P.EventSegmentationEvidence(
        has_insert_seq=True, pair_valid=True, has_left_flank=True,
        has_right_flank=True)
    tsd = call_or_skip(blocks.boundary_signal, seg, boundary("TSD"))
    blunt = call_or_skip(blocks.boundary_signal, seg, boundary("BLUNT"))
    undefined = call_or_skip(blocks.boundary_signal, seg,
                             boundary("NONE", False, False, False))
    assert tsd > blunt > undefined
    # No insert sequence means no boundary evidence at all.
    assert call_or_skip(blocks.boundary_signal,
                        P.EventSegmentationEvidence(has_insert_seq=False),
                        boundary("TSD")) == 0.0


def test_serialize_blocks_round_trips_the_names():
    cert = blocks.Certificate(blocks=[
        blocks.Block("event", 0.5, 1.0, 0.45, 0.3),
        blocks.Block("sequence", 0.9, 2.0, 2.0, 0.2),
    ])
    text = call_or_skip(blocks.serialize_blocks, cert)
    assert text.count(";") == 1
    for name in ("event", "sequence", "raw=", "te_art=", "te_non=", "amb="):
        assert name in text
    assert call_or_skip(blocks.serialize_blocks, blocks.Certificate()) == "NA"


# -------------------------------------------------------------- dependency
def test_empirical_bernstein_upper_is_above_the_mean_and_shrinks_with_n():
    values = [1.0, 2.0, 3.0, 4.0] * 500
    mean = sum(values) / len(values)
    wide = call_or_skip(dependency.empirical_bernstein_upper, values[:8],
                        -math.log(0.10), 20.0)
    tight = call_or_skip(dependency.empirical_bernstein_upper, values,
                         -math.log(0.10), 20.0)
    assert wide > tight >= mean
    # Fewer than two values has no variance, so the only honest bound is the cap.
    close(call_or_skip(dependency.empirical_bernstein_upper, [3.0],
                       -math.log(0.10), 2.0),
          math.exp(2.0), "single value falls back to the cap")


def test_empirical_bernstein_upper_is_floored_at_one_and_capped():
    tiny = [0.001] * 100
    close(call_or_skip(dependency.empirical_bernstein_upper, tiny,
                       -math.log(0.10), 20.0), 1.0, "floored at 1")
    huge = [1e6] * 100
    close(call_or_skip(dependency.empirical_bernstein_upper, huge,
                       -math.log(0.10), 2.0), math.exp(2.0), "capped")


def test_calibration_rows_keeps_every_certificated_row():
    ledger = [
        {"mechanistic_blocks": "event:raw=1"},
        {"mechanistic_blocks": "NA"},
        {"mechanistic_blocks": ""},
        {"mechanistic_blocks": "sequence:raw=2"},
    ]
    assert call_or_skip(dependency.calibration_rows, ledger) == [0, 3]


# ---------------------------------------------------------------- genotype
def test_log_choose_count_matches_the_binomial_coefficient():
    close(call_or_skip(genotype.log_choose_count, 10, 3),
          math.log(120.0), "C(10,3) = 120")
    # DELIBERATE DIVERGENCE: -inf, not the C++'s -1e300. See
    # tests/EXPECTED_DIVERGENCE.md -- the three copies of this function
    # disagreed, and -1e300 survives an isfinite() guard that an impossible
    # count should not.
    assert call_or_skip(genotype.log_choose_count, 3, 10) == -math.inf
    assert call_or_skip(genotype.log_choose_count, -1, 0) == -math.inf


def test_length_concordance_factor_discounts_length_discordant_alt_reads():
    """
    1.0 when no lengths are observed, and below 1 when the alt reads do not
    match the event length -- they look like a different event rather than
    support for this one.
    """
    base = genotype.GenotypeInput(alt_struct_reads=4, ref_span_reads=4,
                                  event_length=320)
    close(call_or_skip(genotype.length_concordance_factor, base), 1.0,
          "no observed lengths")

    concordant = genotype.GenotypeInput(alt_struct_reads=4, ref_span_reads=4,
                                        event_length=320,
                                        alt_observed_lengths=[318, 320, 322, 319])
    discordant = genotype.GenotypeInput(alt_struct_reads=4, ref_span_reads=4,
                                        event_length=320,
                                        alt_observed_lengths=[60, 900, 55, 1200])
    assert call_or_skip(genotype.length_concordance_factor, concordant) > 0.9
    assert call_or_skip(genotype.length_concordance_factor, discordant) < 0.1


# --------------------------------------------------------------- integrate
def test_build_e_values_floors_at_zero_and_keeps_every_hypothesis():
    candidates = [
        integrate.Candidate("a", log_score=5.0),
        integrate.Candidate("b", log_score=-3.0),
        integrate.Candidate("c", log_score=5.0, ambiguity_width=4.0),
    ]
    values = call_or_skip(integrate.build_e_values, candidates, 0.10, 1.0, 9.0)
    assert set(values) == {"a", "b", "c"}, "every hypothesis stays in m"
    close(values["a"], math.exp(5.0 - 1.0), "capped path is inert below the cap")
    assert values["b"] == 0.0, "a negative log e-value floors at zero"
    assert values["c"] == 0.0, "the ambiguity width can only shrink an e-value"


def test_build_e_values_applies_the_cap_before_the_penalty():
    candidates = [integrate.Candidate("x", log_score=50.0)]
    values = call_or_skip(integrate.build_e_values, candidates, 0.10, 2.0, 9.0)
    close(values["x"], math.exp(9.0 - 2.0), "min(score, cap) - penalty")


# --------------------------------------------------------------- structure
def test_interval_entropy_norm_is_zero_for_a_homopolymer_and_one_for_uniform():
    close(call_or_skip(structure.interval_entropy_norm, "AAAAAAAA", 0, 8), 0.0,
          "homopolymer")
    close(call_or_skip(structure.interval_entropy_norm, "ACGTACGT", 0, 8), 1.0,
          "uniform over ACGT")
    assert call_or_skip(structure.interval_entropy_norm, "ACGT", 2, 2) == 0.0
    # Non-ACGT bases are ignored, so a run of N has no entropy.
    close(call_or_skip(structure.interval_entropy_norm, "NNNN", 0, 4), 0.0,
          "N contributes nothing")


def test_status_from_qc_maps_every_documented_reason():
    cases = {
        "PASS_INSERT_TE_ALIGNMENT": structure.TeAnnotationStatus.RESOLVED,
        "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY":
            structure.TeAnnotationStatus.FAMILY_ONLY,
        "PASS_INSERT_TE_ALIGNMENT_UNKNOWN":
            structure.TeAnnotationStatus.UNKNOWN_TE,
        "TE_ALIGNMENT_LOW_IDENTITY": structure.TeAnnotationStatus.NON_TE_LIKE,
        "EMPTY_INSERT_SEQUENCE": structure.TeAnnotationStatus.EMPTY,
        "TE_LIBRARY_UNAVAILABLE": structure.TeAnnotationStatus.UNAVAILABLE,
        "NO_TE_ALIGNMENT": structure.TeAnnotationStatus.NO_CANDIDATE,
    }
    for reason, expected in cases.items():
        assert call_or_skip(structure.status_from_qc, reason, "L1",
                            "L1HS") == expected, reason
    assert call_or_skip(structure.status_from_qc, "SOMETHING_ELSE", "L1",
                        "L1HS") == structure.TeAnnotationStatus.AMBIGUOUS


# -------------------------------------------------------------------- tprt
def test_the_local_composition_nulls_are_powers_of_the_local_frequency():
    close(call_or_skip(tprt.p_null_terminal_poly, 10, 0.30), 0.30 ** 10,
          "poly(A) null")
    close(call_or_skip(tprt.p_null_t_tract, 6, 0.25), 0.25 ** 6, "T-tract null")
    # In an A-rich tract the same run is nearly certain by chance, which is the
    # whole self-cancelling mechanism.
    assert call_or_skip(tprt.p_null_terminal_poly, 10, 0.95) > 0.5


def test_log_p_tsd_given_mechanism_is_a_proper_duration_model():
    """Absence is scored against that mechanism's own P(TSD present), which is
    what stops a full-length or Alu insertion being punished for obeying its own
    mechanism."""
    truncated_absent = call_or_skip(tprt.log_p_tsd_given_mechanism, 0,
                                    "l1_5p_truncated")
    full_absent = call_or_skip(tprt.log_p_tsd_given_mechanism, 0,
                               "l1_full_length")
    close(truncated_absent, math.log(0.10), "1 - 0.90 for 5'-truncated L1")
    close(full_absent, math.log(0.75), "1 - 0.25 for full-length L1")
    assert full_absent > truncated_absent
    # Present: exponential in tau about the mechanism's mean.
    near = call_or_skip(tprt.log_p_tsd_given_mechanism, 15, "l1_5p_truncated")
    far = call_or_skip(tprt.log_p_tsd_given_mechanism, 60, "l1_5p_truncated")
    assert near > far


def test_spanning_sensitivity_falls_with_insertion_length():
    """`s(L) = E_R[(R - L - 2a)+ / R]`: a long insertion is harder to span, and
    an insertion longer than every read cannot be spanned at all."""
    short = call_or_skip(tprt.spanning_sensitivity, 200.0)
    long = call_or_skip(tprt.spanning_sensitivity, 12000.0)
    assert 0.0 < long < short < 1.0
    assert call_or_skip(tprt.spanning_sensitivity, 500000.0) == \
        pytest.approx(0.0, abs=1e-9)


def test_local_error_rate_rises_in_homopolymers_and_repeats():
    """The fourth application of the same rule: the read-error rate is local
    too, because basecallers mis-call homopolymer and repeat lengths far more
    often than they err in unique sequence."""
    unique = call_or_skip(tprt.local_error_rate, 0.30, 0.30, 0.0)
    a_rich = call_or_skip(tprt.local_error_rate, 0.90, 0.30, 0.0)
    repeat = call_or_skip(tprt.local_error_rate, 0.30, 0.30, 0.9)
    close(unique, 0.02, "baseline in unique sequence")
    assert a_rich > unique
    assert repeat > unique
    assert call_or_skip(tprt.local_error_rate, 1.0, 1.0, 1.0) <= 0.45


def test_log_bf_counts_rewards_alt_support_and_punishes_reference_spanning():
    supported = call_or_skip(tprt.log_bf_counts, 10, 1, 1000.0)
    opposed = call_or_skip(tprt.log_bf_counts, 2, 20, 1000.0)
    assert supported > 0.0 > opposed
    assert call_or_skip(tprt.log_bf_counts, 0, 0, 1000.0) == 0.0


def test_log_p_geometry_given_mechanism_rejects_an_incomplete_three_prime_end():
    """EN-dependent TPRT cannot leave the 3' end truncated, so every
    EN-dependent mechanism scores that geometry near zero -- while the
    EN-independent route, which CAN (Morrish 2002), stays uniform."""
    incomplete = call_or_skip(tprt.log_p_geometry_given_mechanism,
                              2400, 3400, 6000.0, "l1_5p_truncated", False)
    complete = call_or_skip(tprt.log_p_geometry_given_mechanism,
                            5000, 6000, 6000.0, "l1_5p_truncated", False)
    assert complete > incomplete + 5.0
    en_free = call_or_skip(tprt.log_p_geometry_given_mechanism,
                           2400, 3400, 6000.0, "en_independent", False)
    assert en_free > incomplete, (
        "the EN-independent route is the escape hatch for this geometry")


def test_mechanism_allows_family_separates_cis_from_trans_mobilised():
    assert call_or_skip(tprt.mechanism_allows_family, "trans_alu_sva", "ALU")
    assert not call_or_skip(tprt.mechanism_allows_family, "l1_5p_truncated",
                            "ALU")
    assert call_or_skip(tprt.mechanism_allows_family, "l1_5p_truncated", "L1")
    assert not call_or_skip(tprt.mechanism_allows_family, "trans_alu_sva", "L1")
    # The EN-independent route is available to both.
    for family in ("L1", "ALU", "SVA"):
        assert call_or_skip(tprt.mechanism_allows_family, "en_independent",
                            family)


def test_log_sum_exp_of_all_impossible_is_impossible_not_nan():
    """`log(0 + 0 + 0)` is -inf. It used to be NaN.

    `max(a,b,c)` of three -inf is -inf, and `exp(-inf - -inf)` is `exp(nan)`.
    The plain three-way log-sum-exp in `genotype.py` and `policy.py` therefore
    returned NaN for a locus where every hypothesis was impossible -- a
    silently poisoned normaliser rather than a refusal.

    This became reachable when the count models unified on -inf (previously
    -1e300 in these two modules, which is finite and normalises fine), so the
    fix and the change that exposed it belong to the same piece of work.
    """
    import math

    from placer_py.core import genotype as genotype_module
    from placer_py.core import mathx
    from placer_py.core import policy as policy_module

    impossible = -math.inf
    assert mathx.log_sum_exp((impossible,) * 3) == impossible
    assert genotype_module._logsumexp3(impossible, impossible, impossible) == impossible
    assert policy_module.logsumexp3(impossible, impossible, impossible) == impossible
    assert policy_module.logsumexp_values([impossible] * 4) == impossible


def test_the_two_log_sum_exp_semantics_stay_distinct():
    """Dropping non-finite operands and propagating them are both wanted.

    `finalization.log_sum_exp_pair` reads -inf as "this line of evidence said
    nothing" and must ignore it; the policy layer reads -inf as "this
    hypothesis is impossible" and must not. Collapsing them onto one default
    would silently change one of the two.
    """
    import math

    from placer_py.core import finalization as finalization_module
    from placer_py.core import mathx
    from placer_py.core import policy as policy_module

    abstained, real = -math.inf, 3.0
    assert finalization_module.log_sum_exp_pair(abstained, real) == real
    assert mathx.log_sum_exp((abstained, real), ignore_nonfinite=True) == real
    # Propagating: -inf contributes exp(-inf) = 0, so the answer is still the
    # finite operand -- the difference shows when EVERY operand is -inf.
    assert policy_module.logsumexp_pair(abstained, real) == real
    assert mathx.log_sum_exp((abstained, abstained), ignore_nonfinite=True) == -math.inf
