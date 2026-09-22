#!/usr/bin/env python3
"""
Run the suite with no third-party dependencies at all.

`pytest` is the intended runner and `pyproject.toml` configures it. This script
exists for two situations where that is not available:

  * a locked-down or offline environment with no package index, and
  * a fast smoke check that the test MODULES themselves are sound -- that they
    import, that the golden data loads, that the assertions execute -- without
    installing anything.

It implements the pytest surface the suite uses: `fixture` (including
`scope="session"` and yield-fixtures), `mark.parametrize`, `mark.<name>`,
`mark.xfail(strict=...)`, `skip`, `fail`, `raises`, `approx` and module-level
`pytestmark`.

WHAT IT DELIBERATELY DOES NOT IMPLEMENT: pytest's builtin fixtures
(`tmp_path`, `monkeypatch`, `capsys`, ...), test classes, plugins, and
collection from anywhere but `tests/test_*.py`. A test that needs one of those
is a test that needs pytest, and the runner says so by name rather than by
resolving the fixture to `None` and failing somewhere confusing.

The two runners have to agree: the `no-dependencies` and `pytest` CI jobs both
gate, so any behaviour here that pytest does not share is a divergence to fix,
not a feature. Where a choice was available -- xfail swallowing arbitrary
exceptions, a skip not failing the run -- parity with pytest won, and the
comment at that point says so.

    python3 tools/run_tests_without_pytest.py           # everything
    python3 tools/run_tests_without_pytest.py test_09   # one module
"""

from __future__ import annotations

import importlib.util
import inspect
import itertools
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"


class Skipped(Exception):
    pass


class Failed(Exception):
    pass


class FixtureError(Exception):
    """A fixture could not be resolved: unknown name, or a cycle."""


# ===========================================================================
# The pytest stub
# ===========================================================================
def _parameter_names(fn) -> list[str]:
    """The positional parameter names of `fn`, in order.

    `inspect.signature` would also do this, but it is ~40x slower per call and
    this runs once per test and once per fixture resolution.
    """
    code = fn.__code__
    return list(code.co_varnames[:code.co_argcount])


def _parameters_with_defaults(fn) -> set[str]:
    names = _parameter_names(fn)
    defaults = fn.__defaults__ or ()
    return set(names[len(names) - len(defaults):]) if defaults else set()


def _idify(value: object) -> str:
    """One parameter value as an id fragment, the way pytest labels cases."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return str(value)
    return type(value).__name__


def _build_pytest_stub() -> types.ModuleType:
    mod = types.ModuleType("pytest")

    class _XfailMark:
        def __init__(self, strict=False, reason=""):
            self.strict = strict
            self.reason = reason

        def __call__(self, fn):
            fn._xfail = self
            return fn

    class _NamedMark:
        def __init__(self, name):
            self.name = name

        def __call__(self, fn):
            marks = getattr(fn, "_marks", [])
            marks.append(self.name)
            fn._marks = marks
            return fn

    class _ParametrizeMark:
        """`pytest.mark.parametrize(argnames, argvalues, ids=...)`.

        Stacking is the reason this is a class rather than a dict: two
        decorators on one function are a cartesian product in pytest, and the
        product has to be taken at collection time, not at decoration time.
        """

        def __init__(self, argnames, argvalues, ids=None):
            if isinstance(argnames, str):
                self.names = [n.strip() for n in argnames.split(",") if n.strip()]
            else:
                self.names = list(argnames)
            self.values = list(argvalues)
            self.ids = None if ids is None else list(ids)

        def cases(self) -> list[tuple[str, dict]]:
            """`(id_fragment, kwargs)` for each row of this one mark."""
            out: list[tuple[str, dict]] = []
            seen: dict[str, int] = {}
            for index, row in enumerate(self.values):
                args = (row,) if len(self.names) == 1 else tuple(row)
                if len(args) != len(self.names):
                    raise Failed(
                        f"parametrize({self.names!r}) row {index} has "
                        f"{len(args)} value(s), not {len(self.names)}")
                if self.ids is not None:
                    label = str(self.ids[index])
                else:
                    label = "-".join(_idify(value) for value in args)
                    # Two non-scalar values of the same type would otherwise
                    # share an id, and an id that does not identify the case
                    # is worse than an ugly one.
                    if label in seen:
                        seen[label] += 1
                        label = f"{label}{seen[label]}"
                    else:
                        seen[label] = 0
                out.append((label, dict(zip(self.names, args))))
            return out

        def __call__(self, fn):
            marks = getattr(fn, "_parametrize", [])
            marks.append(self)
            fn._parametrize = marks
            return fn

    class _MarkFactory:
        def xfail(self, strict=False, reason=""):
            return _XfailMark(strict, reason)

        def parametrize(self, argnames, argvalues, ids=None):
            return _ParametrizeMark(argnames, argvalues, ids)

        def __getattr__(self, name):
            return _NamedMark(name)

    def fixture(*args, **kwargs):
        def wrap(fn):
            fn._is_fixture = True
            # Only `session` is distinguished from `function` below; see
            # `_Fixtures`. The declared scope is recorded verbatim anyway so
            # that the distinction lives in one place.
            fn._fixture_scope = kwargs.get("scope", "function")
            return fn
        if args and callable(args[0]):
            return wrap(args[0])
        return wrap

    def skip(reason=""):
        raise Skipped(reason)

    def fail(reason=""):
        raise Failed(reason)

    class _Raises:
        """Minimal pytest.raises: enough for `with pytest.raises(Error):`.

        Kept deliberately small -- no `match=`, no `.value` -- so it cannot
        quietly diverge from pytest on anything the suite relies on.
        """

        def __init__(self, expected):
            self.expected = expected

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise Failed(f"expected {self.expected.__name__}, nothing raised")
            return issubclass(exc_type, self.expected)

    def raises(expected):
        return _Raises(expected)

    class _Approx:
        """Minimal pytest.approx: enough for `== approx(x, abs=t)`."""

        def __init__(self, expected, abs=None, rel=None):
            self.expected = expected
            self.abs = abs
            self.rel = rel

        def __eq__(self, other):
            import math
            tol_abs = self.abs if self.abs is not None else 0.0
            tol_rel = self.rel if self.rel is not None else (
                0.0 if self.abs is not None else 1e-6)
            return math.isclose(other, self.expected,
                                rel_tol=tol_rel, abs_tol=tol_abs)

        def __repr__(self):
            return f"approx({self.expected!r}, abs={self.abs!r}, rel={self.rel!r})"

    def approx(expected, abs=None, rel=None):
        return _Approx(expected, abs=abs, rel=rel)

    mod.approx = approx
    mod.mark = _MarkFactory()
    mod.fixture = fixture
    mod.skip = skip
    mod.fail = fail
    mod.raises = raises
    return mod


# ===========================================================================
# Fixtures
# ===========================================================================
class _Session:
    """What outlives one test module: the session cache and its teardowns.

    Both have to be shared across modules, and the teardowns are the reason
    this is an object rather than a dict passed around. A session
    yield-fixture registers its generator here; if each module's `_Fixtures`
    kept its own list, the finalizers would be discarded with the module that
    happened to build the value, and the teardown half of every session
    fixture would silently never run.
    """

    def __init__(self) -> None:
        self.cache: dict[tuple[str, str], object] = {}
        self.finalizers: list[tuple[str, object]] = []


class _Fixtures:
    """Resolve test parameters by name, with the caching the scopes imply.

    Two scopes are distinguished, not four: `session` values are computed once
    per run, everything else once per test. `module` and `class` therefore
    behave as `function` here -- conservative in the only direction that can
    matter, because recomputing costs time while sharing a value pytest would
    have rebuilt could hide state leaking between tests.

    Session values are keyed by `(origin, name)`, where the origin is the
    conftest or the test module that defined the fixture. Without that, a
    session fixture defined in one test module would be served to a
    same-named fixture in the next one.
    """

    def __init__(self, defs: dict[str, tuple], session: _Session):
        self._defs = defs
        self._session = session.cache
        self._function: dict[tuple[str, str], object] = {}
        self._finalize_function: list[tuple[str, object]] = []
        self._finalize_session = session.finalizers

    def get(self, name: str, _stack: tuple[str, ...] = ()) -> object:
        if name not in self._defs:
            raise FixtureError(
                f"no fixture named {name!r}. Fixtures are resolved from "
                f"tests/conftest.py and from the test module itself; pytest's "
                f"builtins (tmp_path, monkeypatch, capsys, ...) are not "
                f"implemented, so a test that needs one needs pytest.")
        if name in _stack:
            raise FixtureError("fixture cycle: " + " -> ".join((*_stack, name)))

        fn, scope, origin = self._defs[name]
        key = (origin, name)
        cache = self._session if scope == "session" else self._function
        if key in cache:
            return cache[key]

        kwargs = {arg: self.get(arg, (*_stack, name)) for arg in _parameter_names(fn)}
        produced = fn(**kwargs)
        if inspect.isgenerator(produced):
            value = next(produced)
            finalizers = (self._finalize_session if scope == "session"
                          else self._finalize_function)
            finalizers.append((name, produced))
        else:
            value = produced
        cache[key] = value
        return value

    def end_test(self) -> list[str]:
        problems = self._drain(self._finalize_function)
        self._function.clear()
        return problems

    @staticmethod
    def _drain(finalizers: list[tuple[str, object]]) -> list[str]:
        problems: list[str] = []
        while finalizers:
            name, generator = finalizers.pop()
            try:
                next(generator)
            except StopIteration:
                continue
            except Exception as exc:                        # noqa: BLE001
                problems.append(f"fixture {name!r} teardown raised "
                                f"{type(exc).__name__}: {exc}")
                continue
            problems.append(f"fixture {name!r} yielded twice; a yield-fixture "
                            f"must yield exactly one value")
        return problems


def _collect_fixtures(namespace: dict, origin: str) -> dict[str, tuple]:
    return {
        name: (obj, getattr(obj, "_fixture_scope", "function"), origin)
        for name, obj in namespace.items()
        if callable(obj) and getattr(obj, "_is_fixture", False)
    }


# ===========================================================================
# Collection and execution
# ===========================================================================
def _import_path(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that a module importing itself by name
    # (and `from conftest import ...` in every test file) sees one instance.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cases(fn) -> list[tuple[str, dict]]:
    """`(case_id, kwargs)` for one test function; one empty case if unmarked.

    Decorators apply bottom-up, so `_parametrize` is in bottom-to-top order --
    which is also the order pytest joins the id in and the order of its nested
    loops, the bottom-most mark varying slowest. `itertools.product` varies
    its last argument fastest, so passing the list as collected reproduces
    pytest's case order and ids exactly.
    """
    marks = getattr(fn, "_parametrize", None)
    if not marks:
        return [("", {})]

    claimed: set[str] = set()
    for mark in marks:
        overlap = claimed.intersection(mark.names)
        if overlap:
            raise Failed(f"parametrize declares {sorted(overlap)} twice")
        claimed.update(mark.names)

    cases: list[tuple[str, dict]] = []
    for combination in itertools.product(*(mark.cases() for mark in marks)):
        kwargs: dict = {}
        for _, values in combination:
            kwargs.update(values)
        cases.append(("-".join(label for label, _ in combination), kwargs))
    return cases


def _tests_in_definition_order(module: types.ModuleType) -> list:
    """Test functions in source order.

    Source order, not alphabetical: these files are written to be read top to
    bottom, each section building on the one above, and a failure is easier to
    place when the run follows the file. It also makes the runner's order
    match pytest's.
    """
    found = []
    for name, obj in vars(module).items():
        if not name.startswith("test_") or not callable(obj):
            continue
        code = getattr(obj, "__code__", None)
        if code is None:                     # a callable object, not a function
            continue
        found.append((code.co_firstlineno, name, obj))
    return [(name, obj) for _, name, obj in sorted(found)]


class ModuleReport:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.xfailed = 0
        self.problems: list[str] = []
        self.xfails: list[str] = []


def run_module(path: Path, conftest_fixtures: dict, session: _Session) -> ModuleReport:
    report = ModuleReport()
    try:
        module = _import_path(path, path.stem)
    except Exception as exc:                                  # noqa: BLE001
        report.failed = 1
        report.problems.append(f"{path.name}: import failed: {exc!r}")
        return report

    defs = dict(conftest_fixtures)
    defs.update(_collect_fixtures(vars(module), path.stem))   # module overrides
    fixtures = _Fixtures(defs, session)

    for name, fn in _tests_in_definition_order(module):
        xfail = getattr(fn, "_xfail", None)
        try:
            cases = _cases(fn)
        except Failed as exc:
            report.failed += 1
            report.problems.append(f"{path.name}::{name}: bad parametrize: {exc}")
            continue

        defaulted = _parameters_with_defaults(fn)
        for case_id, case_kwargs in cases:
            label = f"{name}[{case_id}]" if case_id else name
            try:
                kwargs = dict(case_kwargs)
                for argument in _parameter_names(fn):
                    if argument in kwargs:
                        continue
                    if argument in defaulted and argument not in defs:
                        continue                 # let the default stand
                    kwargs[argument] = fixtures.get(argument)
                fn(**kwargs)
            except Skipped:
                report.skipped += 1
            except FixtureError as exc:
                report.failed += 1
                report.problems.append(f"{path.name}::{label}: {exc}")
            # Any Exception, deliberately, and not the `(AssertionError,
            # Failed, Exception)` this used to read as -- a tuple ending in
            # Exception is a catch-all written to look like an enumeration.
            # pytest's xfail also swallows arbitrary exceptions, so narrowing
            # it here would fail tests that pytest passes; what was actually
            # wrong is that the swallowing was SILENT. The type is now
            # reported, so an xfail firing for an unrelated reason -- a typo,
            # a renamed attribute -- is visible instead of looking expected.
            except Exception as exc:                          # noqa: BLE001
                if xfail is not None:
                    report.xfailed += 1
                    report.xfails.append(
                        f"{path.name}::{label}: {type(exc).__name__}: "
                        f"{str(exc)[:100]}")
                else:
                    report.failed += 1
                    report.problems.append(
                        f"{path.name}::{label}: {type(exc).__name__}: {exc}")
            else:
                if xfail is not None and xfail.strict:
                    report.failed += 1
                    report.problems.append(
                        f"{path.name}::{label}: XPASS but marked strict xfail "
                        f"-- the behaviour it documents has changed. Reason "
                        f"given was: {xfail.reason[:120]}")
                else:
                    report.passed += 1
            finally:
                report.problems.extend(
                    f"{path.name}::{label}: {problem}"
                    for problem in fixtures.end_test())
    return report


def check_oracle_present() -> None:
    """Fail fast, and say how to fix it.

    The `oracle` fixture in `tests/conftest.py` is what actually loads the
    file -- this only front-runs it, because 673 tests reporting one missing
    input file is a worse message than one line before anything runs.
    """
    path = TESTS / "oracle" / "cpp_reference.json"
    if not path.exists():
        print(f"FATAL: golden vectors missing at {path}")
        print("Regenerate with tools/regenerate_oracle.sh")
        sys.exit(2)
    with path.open() as handle:
        json.load(handle)


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(p for p in TESTS.glob("test_*.py") if pattern in p.name)
    if not files:
        print(f"no test modules matching {pattern!r}")
        return 2

    # Installed before conftest is imported, because conftest imports pytest.
    sys.modules["pytest"] = _build_pytest_stub()
    sys.path.insert(0, str(TESTS))
    sys.path.insert(0, str(ROOT))
    check_oracle_present()

    try:
        conftest = _import_path(TESTS / "conftest.py", "conftest")
    except Exception as exc:                                  # noqa: BLE001
        print(f"FATAL: tests/conftest.py failed to import: {exc!r}")
        return 2
    conftest_fixtures = _collect_fixtures(vars(conftest), "conftest")
    session = _Session()

    totals = ModuleReport()
    for path in files:
        report = run_module(path, conftest_fixtures, session)
        totals.passed += report.passed
        totals.failed += report.failed
        totals.skipped += report.skipped
        totals.xfailed += report.xfailed
        totals.problems.extend(report.problems)
        totals.xfails.extend(report.xfails)
        flag = "FAIL" if report.failed else "ok"
        print(f"  {path.name:<44} {report.passed:3d} passed  "
              f"{report.skipped:3d} skipped  {report.xfailed:3d} xfail  {flag}")

    # Session teardowns, after the last module rather than inside it: a
    # session fixture built in test_11 and torn down when test_11 finished
    # would not be session-scoped at all.
    late = _Fixtures._drain(session.finalizers)
    totals.problems.extend(late)
    totals.failed += len(late)

    print()
    if totals.problems:
        print("PROBLEMS")
        for line in totals.problems:
            print(f"  - {line}")
        print()
    if totals.xfails:
        print("XFAIL (expected failures, and what each one actually raised)")
        for line in totals.xfails:
            print(f"  - {line}")
        print()
    print(f"total: {totals.passed} passed, {totals.failed} failed, "
          f"{totals.skipped} skipped, {totals.xfailed} xfail (known issues)")
    if totals.skipped:
        # The port is complete, so this counts the opposite of what it used to:
        # see tests/conftest.py. It is reported rather than failed, because
        # pytest does not fail on a skip either and the two runners gate
        # together -- a runner stricter than pytest is a runner that blocks
        # commits pytest passes.
        print("\nA SKIP IS NOW A REGRESSION, not remaining work: the "
              "migration surface is empty, so a skip means a function that "
              "existed has been removed or stubbed. See tests/conftest.py.")
    return 1 if totals.failed else 0


if __name__ == "__main__":
    sys.exit(main())
