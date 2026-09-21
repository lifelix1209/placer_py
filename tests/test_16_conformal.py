"""Golden conformal selection. Ported from conformal_selector.cpp."""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer_py import conformal as C

pytestmark = pytest.mark.golden


def _build(n_nulls: int, n_candidates: int, contexts: int):
    """Mirrors tools/dump_oracle.cpp's deterministic ramp exactly."""
    selector = C.ConformalNullSelector()
    for i in range(n_nulls):
        t = i / max(1, n_nulls - 1)
        selector.add_null_control(C.ConformalFeatureVector(
            id=f"null_{i}",
            pro_te=[2.0 + 6.0 * t, 0.60 + 0.30 * t, 0.50 + 0.40 * t,
                    0.01 + 0.10 * t],
            ref_span_reads=10.0 - 8.0 * t,
            context=i % max(1, contexts)))
    candidates = []
    for i in range(n_candidates):
        t = i / max(1, n_candidates - 1)
        candidates.append(C.ConformalFeatureVector(
            id=f"cand_{i}",
            pro_te=[4.0 + 8.0 * t, 0.70 + 0.28 * t, 0.60 + 0.38 * t,
                    0.02 + 0.16 * t],
            ref_span_reads=6.0 - 5.0 * t,
            context=i % max(1, contexts)))
    return selector, candidates


def test_every_conformal_scenario_matches_the_cpp(oracle):
    for golden in oracle["conformal"]:
        selector, candidates = _build(golden["n_nulls"],
                                      golden["n_candidates"],
                                      golden["contexts"])
        results = call_or_skip(selector.select, candidates, golden["q"])
        assert len(results) == len(golden["results"]), golden["name"]
        for got, want in zip(results, golden["results"]):
            label = f"{golden['name']}::{want['id']}"
            assert got.id == want["id"], label
            close(got.conformal_p, want["conformal_p"], f"{label}.p")
            close(got.by_threshold, want["by_threshold"], f"{label}.threshold")
            assert got.dominated_null_count == want["dominated"], label
            assert got.null_count == want["null_count"], label
            assert got.pass_ == want["pass"], label
            assert got.qc == want["qc"], label


@pytest.mark.invariant
def test_the_p_value_is_add_one_smoothed_and_never_zero():
    selector, candidates = _build(40, 6, 1)
    for result in call_or_skip(selector.select, candidates, 0.10):
        assert result.conformal_p > 0.0
        assert result.conformal_p <= 1.0


@pytest.mark.invariant
def test_no_nulls_means_no_selection_and_a_named_reason():
    selector = C.ConformalNullSelector()
    candidates = [C.ConformalFeatureVector(id="a", pro_te=[9.0, 0.9, 0.9, 0.2],
                                           ref_span_reads=0.0)]
    results = call_or_skip(selector.select, candidates, 0.10)
    assert results[0].qc == "CONFORMAL_NULL_INSUFFICIENT"
    assert not results[0].pass_
    assert results[0].conformal_p == 1.0, (
        "with no null to compare against, the honest p-value is 1")


@pytest.mark.invariant
def test_a_sparse_context_falls_back_to_the_pooled_null():
    """
    Below `MIN_CONTEXT_NULLS` the selector pools, so power is not lost in a
    sparsely-populated context. At 8 nulls every candidate sees all 8; at 40
    split over 2 contexts each sees only its own 20.
    """
    sparse, candidates = _build(8, 4, 2)
    for result in call_or_skip(sparse.select, candidates, 0.10):
        assert result.null_count == 8, "pooled fallback"

    rich, candidates = _build(40, 6, 2)
    for result in call_or_skip(rich.select, candidates, 0.10):
        assert result.null_count == 20, "context-conditioned"


@pytest.mark.invariant
def test_malformed_null_controls_are_dropped_not_counted():
    """A non-finite or empty feature vector must not enlarge the denominator,
    which would silently make every p-value look smaller."""
    selector = C.ConformalNullSelector()
    selector.add_null_control(C.ConformalFeatureVector(id="ok",
                                                       pro_te=[1.0, 0.5],
                                                       ref_span_reads=1.0))
    selector.add_null_control(C.ConformalFeatureVector(id="empty", pro_te=[]))
    selector.add_null_control(C.ConformalFeatureVector(
        id="nan", pro_te=[float("nan"), 0.5], ref_span_reads=1.0))
    selector.add_null_control(C.ConformalFeatureVector(
        id="inf_ref", pro_te=[1.0, 0.5], ref_span_reads=float("inf")))
    assert call_or_skip(selector.null_count) == 1


@pytest.mark.invariant
def test_dominance_requires_every_coordinate_including_the_reversed_one():
    candidate = C.ConformalFeatureVector(id="c", pro_te=[5.0, 0.8],
                                         ref_span_reads=2.0)
    better = C.ConformalFeatureVector(id="n", pro_te=[6.0, 0.9],
                                      ref_span_reads=1.0)
    one_axis_worse = C.ConformalFeatureVector(id="n", pro_te=[6.0, 0.7],
                                              ref_span_reads=1.0)
    more_reference = C.ConformalFeatureVector(id="n", pro_te=[6.0, 0.9],
                                              ref_span_reads=3.0)
    assert call_or_skip(C.null_dominates_candidate, better, candidate)
    assert not call_or_skip(C.null_dominates_candidate, one_axis_worse,
                            candidate)
    assert not call_or_skip(C.null_dominates_candidate, more_reference,
                            candidate), (
        "ref_span_reads is compared with the OPPOSITE sense -- more reference "
        "support makes a null LESS TE-like, so it cannot dominate on that axis")


@pytest.mark.invariant
def test_the_structural_support_p_value_is_taken_as_a_maximum():
    """
    A candidate whose only advantage is raw read count must not pass on that
    alone. Dominance can be generous when the other coordinates are
    uninformative, so the count is also tested on its own terms and the WORSE
    (larger) of the two p-values is used.
    """
    selector = C.ConformalNullSelector()
    for i in range(40):
        # Nulls with very high alt counts but poor sequence coordinates.
        selector.add_null_control(C.ConformalFeatureVector(
            id=f"n{i}", pro_te=[100.0, 0.1, 0.1, 0.0], ref_span_reads=0.0))
    candidate = C.ConformalFeatureVector(id="c", pro_te=[50.0, 0.9, 0.9, 0.2],
                                         ref_span_reads=0.0)
    dominance = call_or_skip(selector.dominance_p_value, candidate)
    results = call_or_skip(selector.select, [candidate], 0.10)
    assert results[0].conformal_p >= dominance, (
        "the maximum must not be smaller than the dominance p-value")
    assert results[0].conformal_p > 0.9, (
        "every null beats it on alt count, so the structural p-value is near 1")


@pytest.mark.invariant
def test_benjamini_yekutieli_harmonic_factor_is_applied():
    """The threshold at rank r is `r*q / (m * H_m)` -- the H_m is the price of
    arbitrary dependence among p-values, and it is what e-BH avoids."""
    selector, candidates = _build(40, 6, 1)
    results = call_or_skip(selector.select, candidates, 0.10)
    m = len(results)
    h_m = call_or_skip(C.harmonic_number, m)
    assert h_m > 1.0
    # Every threshold is its rank's BY threshold, and the largest possible one
    # (at rank m) is q/H_m -- the harmonic factor in plain sight.
    for rank, index in enumerate(
            sorted(range(m), key=lambda i: (results[i].conformal_p,
                                            results[i].id)), start=1):
        if results[index].pass_:
            continue                      # selected rows carry the cutoff
        close(results[index].by_threshold, (rank * 0.10) / (m * h_m),
              f"rank {rank} threshold")
    assert (m * 0.10) / (m * h_m) < 0.10, (
        "H_m strictly shrinks the effective q; at m=1000 it is 7.49x")
    close(call_or_skip(C.harmonic_number, 1000),
          sum(1.0 / i for i in range(1, 1001)), "H_1000")
