"""Breakpoint-shift controls and the empirical null tail: the shifted null Phase 3 builds on."""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer.core import null_control as N


@pytest.mark.invariant
def test_the_controls_alternate_outward_around_the_locus():
    """
    Left then right at each multiple of the step, rather than scanning one side
    first. That keeps the controls balanced around the locus, so a coverage or
    composition gradient across the window does not bias them.
    """
    controls = call_or_skip(N.make_breakpoint_shift_controls,
                            1000, 1020, 0, 5000, 100, 6)
    offsets = [c.bp_left - 1000 for c in controls]
    assert offsets == [-100, 100, -200, 200, -300, 300]


@pytest.mark.invariant
def test_a_control_never_coincides_with_the_locus_itself():
    for controls in (call_or_skip(N.make_breakpoint_shift_controls,
                                  1000, 1020, 0, 5000, 100, 8),
                     call_or_skip(N.make_breakpoint_shift_controls,
                                  1000, 1000, 0, 5000, 37, 5)):
        for control in controls:
            assert not (control.bp_left == 1000 and control.bp_right in (1000, 1020))


@pytest.mark.invariant
def test_controls_stay_inside_the_window():
    controls = call_or_skip(N.make_breakpoint_shift_controls,
                            1000, 1020, 900, 1150, 40, 20)
    for control in controls:
        assert control.bp_left >= 900
        assert control.bp_right <= 1150


@pytest.mark.invariant
def test_invalid_arguments_yield_no_controls_rather_than_raising():
    for args in ((-1, 20, 0, 100, 10, 4), (10, -1, 0, 100, 10, 4),
                 (10, 20, 100, 0, 10, 4), (10, 20, 0, 100, 0, 4),
                 (10, 20, 0, 100, 10, 0)):
        assert call_or_skip(N.make_breakpoint_shift_controls, *args) == []


@pytest.mark.invariant
def test_a_narrow_window_terminates_instead_of_looping():
    """The loop has no fixed bound, so it must detect that BOTH directions have
    left the window. A window narrower than one step yields nothing and returns."""
    assert call_or_skip(N.make_breakpoint_shift_controls,
                        1000, 1020, 995, 1045, 100, 6) == []


@pytest.mark.invariant
def test_the_tail_denominator_counts_the_observation_on_both_sides():
    """
    `(1 + ge) / (2 + n)`, not `(1 + ge) / (1 + n)`.

    The conservative convention: the observation is counted on both sides, which
    keeps the p-value strictly inside (0, 1) even when it exceeds every stored
    value.
    """
    tail = N.EmpiricalNullTail()
    for value in (1.0, 2.0, 3.0):
        call_or_skip(tail.add, value)
    close(call_or_skip(tail.upper_tail_p, 100.0), 1.0 / 5.0,
          "beyond every stored value")
    close(call_or_skip(tail.upper_tail_p, 0.0), 4.0 / 5.0, "below all of them")
    assert 0.0 < call_or_skip(tail.upper_tail_p, 100.0) < 1.0


@pytest.mark.invariant
def test_an_empty_tail_is_maximally_uninformative():
    tail = N.EmpiricalNullTail()
    close(call_or_skip(tail.upper_tail_p, 5.0), 0.5,
          "no data: (1+0)/(2+0)")
