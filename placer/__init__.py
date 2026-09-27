"""PLACER: long-read transposable-element insertion calling, BAM to calls.

Importing this package pulls in NOTHING. The submodules are deliberately not
re-exported here, because the decision layer's whole selling point is that it
runs with no third-party package installed -- and `placer.io.bam` needs
pysam. A package-level `from . import io` would make `import placer`
fail in exactly the locked-down environment the design is for.

So import what you need:

    from placer.pipeline import run_pipeline        # needs pysam upstream
    from placer.core.finalize import finalize_run                # needs nothing
"""

from __future__ import annotations

__all__ = ["__version__"]


def _source_tree_version() -> str:
    """The version pyproject.toml states, marked `+source`, for a run from a
    source checkout; "0.0.0+source" when there is none to read.

    Read with a pattern rather than `tomllib`, which is 3.11+ and the floor is
    3.9. Every run on the cluster is from a frozen source tree, and a VCF that
    says only "0.0.0+source" cannot say which version wrote it.
    """
    import re
    from pathlib import Path

    try:
        text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    except OSError:
        return "0.0.0+source"
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return f"{match.group(1)}+source" if match else "0.0.0+source"


#: Resolved from the installed distribution metadata so there is ONE source of
#: truth (pyproject.toml). The fallback matters: the zero-dependency CI job,
#: `tools/run_tests_without_pytest.py` and every cluster run work from a source
#: checkout with nothing installed, where the distribution does not exist.
try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("placer-te")
    except PackageNotFoundError:  # running from a source tree, not installed
        __version__ = _source_tree_version()
except ImportError:  # pragma: no cover - importlib.metadata is stdlib >= 3.8
    __version__ = _source_tree_version()
