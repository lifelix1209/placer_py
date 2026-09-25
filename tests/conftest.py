"""
Shared fixtures, and the skip mechanism left over from the migration.

HOW THIS SUITE IS MEANT TO BE USED

THE PORT IS COMPLETE and this file used to open by saying the opposite --
"Nothing in `placer_py` is implemented yet" -- which was true when it was
written and has not been for a long time. `run_tests_without_pytest.py`
reports `0 skipped (unported)` on every run, so the migration surface it was
built to measure is empty.

`call_or_skip` stays anyway, and not out of sentiment: it turns a
`NotImplementedError` into a SKIP carrying its own name, so a function
removed or stubbed during a refactor shows up as a named gap rather than as
a wall of red that hides whatever else broke in the same commit. The
invariant to watch is now the opposite of the original one --

    pytest -r s            # any skip is a REGRESSION, not remaining work

-- because there is no remaining work for it to mean.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

#: The only differences an exact closed form may show against the code under
#: test come from the order of floating point operations. This tolerance
#: allows that and nothing more -- it is far too tight to absorb a genuinely
#: different formula.
RTOL = 1e-12
ATOL = 1e-12


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root. Also the conftest-defined fixture that
    `test_34_runner_parity.py` uses to check both runners resolve one."""
    return Path(__file__).resolve().parent.parent


def call_or_skip(fn, *args, **kwargs):
    """
    Invoke a port function, or skip this test if it is not written yet.

    Only `NotImplementedError` is caught. Any other exception is a real bug in
    an implementation that claims to exist, and must surface as a failure.
    """
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"not ported yet: {getattr(fn, '__qualname__', fn)}")


def close(actual: float, expected: float, what: str = "") -> None:
    """Assert equality to full double precision, with a message that says which
    field disagreed and by how much."""
    if math.isnan(expected):
        assert math.isnan(actual), f"{what}: expected NaN, got {actual!r}"
        return
    assert math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), (
        f"{what}: got {actual!r}, expected {expected!r} "
        f"(delta {actual - expected!r})"
    )
