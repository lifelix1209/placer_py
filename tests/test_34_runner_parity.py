"""
The pytest surface the suite is allowed to use, pinned in both runners.

`tools/run_tests_without_pytest.py` reimplements part of pytest, and the CI
gates on BOTH of them, so every feature it implements is a place the two can
drift apart. This file is the only test module whose subject is the runners
rather than `placer_py`: each test below passes only if pytest and the
zero-dependency runner agree about what a decorator means.

It exists because the runner's limits were previously enforced by a sentence
in its docstring -- "if the suite starts using parametrize or real fixtures,
use pytest" -- which is a rule nothing checks. That sentence was also the
thing quietly vetoing every test improvement: `parametrize` and a cached
fixture were unavailable not because they are hard but because using them
would have silently reduced the zero-dependency run to a subset. The features
are implemented now, and this is what says they still work.

Three of these tests read the state left behind by the ones above them, which
pins the third guarantee as a side effect: tests run in SOURCE order, in both
runners. The runner sorted them alphabetically until the same commit added
this file.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

_single_values_seen: set[int] = set()
_product_seen: set[tuple[str, int]] = set()
_fixture_builds: list[str] = []
_teardown_log: list[str] = []
_session_teardown_log: list[str] = []


# ===========================================================================
# parametrize
# ===========================================================================
@pytest.mark.parametrize("value", [1, 2, 3])
def test_parametrize_runs_the_body_once_per_value(value: int) -> None:
    _single_values_seen.add(value)
    assert value in (1, 2, 3)


def test_every_parametrized_value_actually_ran() -> None:
    assert _single_values_seen == {1, 2, 3}, (
        "a runner that collects parametrize but runs one case would still "
        "report the test as passing")


@pytest.mark.parametrize("left,right,expected", [(1, 1, 2), (2, 3, 5), (-1, 1, 0)])
def test_a_parametrize_row_unpacks_into_several_arguments(
        left: int, right: int, expected: int) -> None:
    assert left + right == expected


@pytest.mark.parametrize("flag", [True, False], ids=["on", "off"])
def test_explicit_ids_are_accepted(flag: bool) -> None:
    assert flag in (True, False)


@pytest.mark.parametrize("outer", ["a", "b"])
@pytest.mark.parametrize("inner", [0, 1])
def test_stacked_parametrize_marks_multiply(outer: str, inner: int) -> None:
    _product_seen.add((outer, inner))


def test_stacking_produced_the_full_cartesian_product() -> None:
    # Four cases, not two and not six. Getting this wrong in the direction of
    # too few is the dangerous one: the suite would look green while a third
    # of the combinations never ran.
    assert _product_seen == {("a", 0), ("a", 1), ("b", 0), ("b", 1)}


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture(scope="session")
def built_once() -> dict:
    _fixture_builds.append("built")
    return {"rows": 3}


@pytest.fixture(scope="session")
def session_scratch():
    _session_teardown_log.append("setup")
    yield "session"
    # Runs after the LAST test module, so nothing in the suite can observe it;
    # what is observable, and asserted below, is that it has NOT run while
    # tests are still consuming the fixture.
    _session_teardown_log.append("teardown")


@pytest.fixture
def scratch():
    _teardown_log.append("setup")
    yield "scratch"
    _teardown_log.append("teardown")


def test_a_fixture_is_resolved_by_parameter_name(built_once: dict) -> None:
    assert built_once["rows"] == 3


def test_a_session_fixture_is_built_once_not_once_per_test(built_once: dict) -> None:
    # This is the guarantee the suite's expensive fixtures depend on. A runner
    # that rebuilt per test would still be green everywhere else, just slower
    # -- so nothing but this would catch it.
    assert _fixture_builds == ["built"]


def test_a_yield_fixture_yields_its_value(scratch: str) -> None:
    assert scratch == "scratch"
    assert _teardown_log == ["setup"], "teardown must not run before the test"


def test_the_teardown_half_of_a_yield_fixture_ran() -> None:
    assert _teardown_log == ["setup", "teardown"]


def test_a_session_yield_fixture_is_set_up_once(session_scratch: str) -> None:
    assert session_scratch == "session"
    assert _session_teardown_log == ["setup"]


def test_a_session_yield_fixture_is_not_torn_down_between_tests(
        session_scratch: str) -> None:
    # The distinction this makes is scope: a function-scoped fixture would
    # have been finalised after the test above, leaving "teardown" here.
    assert _session_teardown_log == ["setup"]


def test_fixtures_defined_in_conftest_are_visible_here(repo_root) -> None:
    # `repo_root` lives in tests/conftest.py. The zero-dependency runner used
    # to special-case one conftest fixture name by inspecting the signature;
    # it now resolves conftest fixtures like any other, which is what this
    # assertion checks.
    assert (repo_root / "tests" / "conftest.py").is_file(), (
        "conftest's repo_root fixture")
