"""
The redesign's own suite, kept apart from the port's for three reasons.

FIRST, it needs `pysam`. Not obviously: only four of these files import a
pysam-dependent module directly, but `model/placer_discriminator.py` imports
`evidence/reference_boundary.py`, which imports pysam at module level -- so the
dependency reaches every file transitively. Guarding per-file would therefore
be wrong in a way that only shows up once someone adds an import, which is why
the whole directory is ignored when pysam is absent.

SECOND, it is unittest-style, and `tools/run_tests_without_pytest.py`
implements only the pytest surface the PORT's suite uses -- it does not collect
`unittest.TestCase`. That runner globs `tests/test_*.py` at the top level only,
so these files are invisible to it by construction rather than by an exclusion
list that could rot.

THIRD, and most importantly, the two halves are constrained differently. The
port is pinned to C++ golden vectors; the redesign is pinned to invariants,
because requiring it to reproduce the C++ numbers would require it to get worse
(see the README). Keeping the directories apart puts that distinction in the
layout instead of only in the prose.

These files are byte-identical to the ones that used to live in the C++
repository's `tests/`, apart from the import prefix, so a diff against that
history stays readable.
"""

collect_ignore_glob: list[str] = []
try:  # pragma: no cover - depends on the environment, not on the code
    import pysam  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
