#!/usr/bin/env python3
"""
Run the suite with no third-party dependencies at all.

`pytest` is the intended runner and `pyproject.toml` configures it. This script
exists for two situations where that is not available:

  * a locked-down or offline environment with no package index, and
  * a fast smoke check that the test MODULES themselves are sound -- that they
    import, that the golden data loads, that the assertions execute -- without
    installing anything.

It implements just enough of the pytest surface the suite uses: `fixture`,
`mark.<name>`, `mark.xfail(strict=...)`, `skip`, `fail` and module-level
`pytestmark`. It is not a pytest replacement and makes no attempt to be one;
if the suite starts using parametrize or real fixtures, use pytest.

    python3 tools/run_tests_without_pytest.py           # everything
    python3 tools/run_tests_without_pytest.py test_09   # one module
"""

from __future__ import annotations

import importlib.util
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

    class _MarkFactory:
        def xfail(self, strict=False, reason=""):
            return _XfailMark(strict, reason)

        def __getattr__(self, name):
            return _NamedMark(name)

    def fixture(*args, **kwargs):
        def wrap(fn):
            fn._is_fixture = True
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


def load_oracle() -> dict:
    path = TESTS / "oracle" / "cpp_reference.json"
    if not path.exists():
        print(f"FATAL: golden vectors missing at {path}")
        print("Regenerate with tools/regenerate_oracle.sh")
        sys.exit(2)
    with path.open() as handle:
        return json.load(handle)


def run_module(path: Path, oracle: dict) -> tuple[int, int, int, int, list[str]]:
    sys.modules["pytest"] = _build_pytest_stub()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TESTS))

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                              # noqa: BLE001
        return 0, 1, 0, 0, [f"{path.name}: import failed: {exc!r}"]

    passed = failed = skipped = xfailed = 0
    problems: list[str] = []
    for name in sorted(vars(module)):
        if not name.startswith("test_"):
            continue
        fn = getattr(module, name)
        if not callable(fn):
            continue
        xfail = getattr(fn, "_xfail", None)
        wants_oracle = "oracle" in fn.__code__.co_varnames[:fn.__code__.co_argcount]
        try:
            fn(oracle) if wants_oracle else fn()
        except Skipped:
            skipped += 1
            continue
        except (AssertionError, Failed, Exception) as exc:  # noqa: BLE001
            if xfail is not None:
                xfailed += 1
                continue
            failed += 1
            problems.append(f"{path.name}::{name}: {type(exc).__name__}: {exc}")
            continue
        if xfail is not None and xfail.strict:
            failed += 1
            problems.append(
                f"{path.name}::{name}: XPASS but marked strict xfail -- the "
                f"behaviour it documents has changed. Reason given was: "
                f"{xfail.reason[:120]}")
            continue
        passed += 1
    return passed, failed, skipped, xfailed, problems


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    oracle = load_oracle()
    files = sorted(p for p in TESTS.glob("test_*.py") if pattern in p.name)
    if not files:
        print(f"no test modules matching {pattern!r}")
        return 2

    totals = [0, 0, 0, 0]
    all_problems: list[str] = []
    for path in files:
        p, f, s, x = 0, 0, 0, 0
        p, f, s, x, problems = run_module(path, oracle)
        totals[0] += p
        totals[1] += f
        totals[2] += s
        totals[3] += x
        all_problems.extend(problems)
        flag = "FAIL" if f else "ok"
        print(f"  {path.name:<44} {p:3d} passed  {s:3d} skipped  "
              f"{x:3d} xfail  {flag}")

    print()
    if all_problems:
        print("PROBLEMS")
        for line in all_problems:
            print(f"  - {line}")
        print()
    print(f"total: {totals[0]} passed, {totals[1]} failed, "
          f"{totals[2]} skipped (unported), {totals[3]} xfail (known issues)")
    if totals[2]:
        print("\nSkips are the remaining migration surface: each one is a "
              "function in placer_py/ that still raises NotImplementedError.")
    return 1 if totals[1] else 0


if __name__ == "__main__":
    sys.exit(main())
