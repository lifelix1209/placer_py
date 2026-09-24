"""
The dependency rule, as an assertion rather than a comment.

Where to look: `placer_py/core/__init__.py` states the rule; this file is what
makes it true. The three stages are `placer_py/io/` (everything that talks to
something outside the process), `placer_py/core/` (everything that decides
something) and `placer_py/report/` (everything that renders).

WHY THIS IS WORTH A TEST FILE. The rule is what the whole refactor buys, and
it is the kind of rule that decays silently: one convenient import inside one
function body, and `placer_py.core.finalization` stops being importable in an
environment with no pysam. Nothing else in the suite would notice, because
every existing test either has pysam available or never reaches that module.

IT WALKS THE AST, NOT THE TEXT, and at every depth rather than only the top
import block. Both choices are load-bearing. A grep matches the docstring in
`core/__init__.py` that STATES the rule and reports a violation that is not
one. And this codebase uses function-local imports heavily -- `consensus.py`,
`finalization.py`, `main.py` and `denovo.py` all do it, for good reasons -- so
a check that read only module-level imports would be checking the wrong half.

Numbered last on purpose: `tests/test_34_runner_parity.py` pins the source
order of the files above it, so new modules go after it rather than among them.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.contract

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "placer_py"

#: `core` may not reach the stages on either side of it, nor the composition
#: root that joins them.
FORBIDDEN_FOR_CORE = ("placer_py.io", "placer_py.report", "placer_py.pipeline",
                      "placer_py.wiring", "placer_py.main", "placer_py.denovo")

#: Nor anything that would put a compiled or external dependency in front of
#: the decision layer. `subprocess` and `tempfile` are stdlib and so do not
#: threaten `dependencies = []`, but they mean a module shells out, which is
#: input-stage work by definition.
FORBIDDEN_TOP_LEVEL = {"pysam", "pyabpoa", "subprocess", "tempfile",
                       "numpy", "scipy"}


def imported_modules(tree: ast.AST):
    """(line, dotted name) for every import at ANY depth.

    Relative imports are skipped: they cannot leave their own package, so they
    can never violate a rule expressed in absolute terms.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module


def modules_under(*parts: str) -> list[pathlib.Path]:
    root = PACKAGE.joinpath(*parts)
    return sorted(root.rglob("*.py")) if root.is_dir() else []


def violations(paths, forbidden_prefixes, forbidden_top):
    found = []
    for path in paths:
        for lineno, name in imported_modules(ast.parse(path.read_text())):
            if name.split(".")[0] in forbidden_top or name.startswith(forbidden_prefixes):
                found.append(f"{path.relative_to(PACKAGE.parent)}:{lineno} imports {name}")
    return found


# ------------------------------------------------------------ the three stages
def test_the_core_package_exists_and_is_not_empty():
    """If this fails the rest of the file is vacuously true, which is the one
    way a layering test can pass while meaning nothing."""
    assert len(modules_under("core")) > 20


def test_core_never_imports_the_stages_on_either_side_of_it():
    found = violations(modules_under("core"), FORBIDDEN_FOR_CORE, set())
    assert not found, "\n".join(found)


def test_core_never_imports_a_compiled_or_external_dependency():
    """
    This is what `dependencies = []` means in practice, and what the CI job
    that imports the decision layer in an empty venv is checking from the
    outside. Asserting it here names the offending line instead of failing an
    install.
    """
    found = violations(modules_under("core"), (), FORBIDDEN_TOP_LEVEL)
    assert not found, "\n".join(found)


def test_report_never_imports_the_input_stage():
    """
    The renderers take a `ReportContext` of plain data precisely so that they
    never need a live handle. A `fetch_reference` callable would read better
    and would make the whole output surface untestable under the
    zero-dependency runner.
    """
    found = violations(modules_under("report"),
                       ("placer_py.io",), {"pysam", "pyabpoa"})
    assert not found, "\n".join(found)


def test_the_external_tools_live_only_in_the_input_stage():
    """The positive half of the rule: pysam, pyabpoa, subprocess and tempfile
    appear SOMEWHERE, and that somewhere is `placer_py/io/`."""
    reached = set()
    for path in modules_under("io"):
        for _, name in imported_modules(ast.parse(path.read_text())):
            top = name.split(".")[0]
            if top in FORBIDDEN_TOP_LEVEL:
                reached.add(top)
    assert {"pysam", "pyabpoa", "subprocess", "tempfile"} <= reached, reached


# ------------------------------------------------------------- the __init__ files
@pytest.mark.parametrize("package", ["core", "io", "report"])
def test_a_stage_package_re_exports_nothing(package):
    """
    `import placer_py.io.gate` is pure Python and is exercised by the
    zero-dependency runner. A `from . import bam` in `io/__init__.py` would
    make it drag pysam in and fail in exactly the locked-down environment the
    design is for. The top-level `placer_py/__init__.py` has said so since
    before this refactor; the three stage packages inherit the rule.
    """
    init = PACKAGE / package / "__init__.py"
    assert init.is_file(), init
    imported = [name for _, name in imported_modules(ast.parse(init.read_text()))
                if name != "__future__"]
    assert imported == [], imported


# ----------------------------------------------------------------- the top level
def test_the_top_level_holds_only_the_shared_vocabulary_and_the_composition():
    """
    Everything else belongs to a stage. The two tiers that remain are the
    vocabulary all three stages speak (`alignment`, `reads`, `config`,
    `schema`) and the composition that joins them (`pipeline`, `main`,
    `denovo`).

    `reads.py` is here rather than in `io/` even though gating is input-stage
    work: `alignment.py` depends on it for the CIGAR and flag constants, so it
    is vocabulary. `placer_py/io/gate.py` holds the APPLICATION of the
    predicate, which is the part that is actually a stage.
    """
    top = {p.stem for p in PACKAGE.glob("*.py")}
    assert top == {"__init__", "__main__", "alignment", "reads", "config",
                   "schema", "pipeline", "main", "denovo"}, sorted(top)
