"""placer-py: long-read transposable-element insertion calling, BAM to calls.

Importing this package pulls in NOTHING. The submodules are deliberately not
re-exported here, because the decision layer's whole selling point is that it
runs with no third-party package installed -- and `placer_py.io.bam` needs
pysam. A package-level `from . import io` would make `import placer_py`
fail in exactly the locked-down environment the design is for.

So import what you need:

    from placer_py.pipeline import run_pipeline        # needs pysam upstream
    from placer_py.finalization import finalize_final_calls   # needs nothing
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Resolved from the installed distribution metadata so there is ONE source of
#: truth (pyproject.toml). The fallback matters: the zero-dependency CI job and
#: `tools/run_tests_without_pytest.py` both run from a source checkout with
#: nothing installed, where the distribution does not exist.
try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("placer-py")
    except PackageNotFoundError:  # running from a source tree, not installed
        __version__ = "0.0.0+source"
except ImportError:  # pragma: no cover - importlib.metadata is stdlib >= 3.8
    __version__ = "0.0.0+source"
