"""
Breakpoint-shift controls and the empirical null tail.

Ported from `src/pipeline/null_control.cpp`, pinned by
`tests/test_17_null_control.py`.

`make_breakpoint_shift_controls` is the construction `placer/core/decoys.py`
argued for and could not build from the scalar ledger: the SAME locus with the
breakpoint moved, so local composition is preserved and the coincidences are
whatever chance gives. It already existed in the C++, which is why the decoy
module takes a callback -- this is the callback, once the evidence layer can
recompute observables at the shifted position.

It alternates outward, left then right, at multiples of `shift_step`, and stops
when neither direction fits inside the window. Alternating rather than
scanning one side first keeps the controls balanced around the locus, so a
coverage or composition gradient across the window does not bias them.

NOT ON THE CALLING PATH, and that is deliberate rather than neglect --
see `docs/off-pipeline-modules.md` for which of the four unimported
modules this is and why deleting it would lose something.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BreakpointShiftControl:
    bp_left: int = -1
    bp_right: int = -1
    kind: str = "breakpoint_shift"


def make_breakpoint_shift_controls(bp_left: int, bp_right: int,
                                   window_start: int, window_end: int,
                                   shift_step: int, max_controls: int
                                   ) -> list[BreakpointShiftControl]:
    """Port of `placer::make_breakpoint_shift_controls`."""
    controls: list[BreakpointShiftControl] = []
    if (bp_left < 0 or bp_right < 0 or window_start > window_end
            or shift_step <= 0 or max_controls <= 0):
        return controls

    width = max(0, bp_right - bp_left)
    multiplier = 1
    while len(controls) < max_controls:
        added = False
        for direction in (-1, 1):
            if len(controls) >= max_controls:
                break
            shifted_left = bp_left + (direction * multiplier * shift_step)
            shifted_right = shifted_left + width
            if shifted_left < window_start or shifted_right > window_end:
                continue
            if shifted_left == bp_left and shifted_right == bp_right:
                continue
            controls.append(BreakpointShiftControl(bp_left=shifted_left,
                                                   bp_right=shifted_right))
            added = True
        if (not added
                and bp_left - (multiplier * shift_step) < window_start
                and bp_left + (multiplier * shift_step) + width > window_end):
            break
        multiplier += 1
    return controls


class EmpiricalNullTail:
    """
    Add-one-smoothed upper-tail p-value against a stored empirical null.

    Note the denominator is `2 + n`, not `1 + n`: the observation itself is
    counted on BOTH sides, which is the conservative convention and keeps the
    p-value strictly inside (0, 1) even when the observation exceeds every
    stored value.
    """

    def __init__(self) -> None:
        self._values: list[float] = []

    def add(self, value: float) -> None:
        self._values.append(value)

    def upper_tail_p(self, observed: float) -> float:
        ge = sum(1 for value in self._values if value >= observed)
        return (1 + ge) / (2 + len(self._values))

    def size(self) -> int:
        return len(self._values)
