# Changelog

Notable changes, in the terms a user of the tool would notice. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

Numerical behaviour that departs from the C++ implementation on purpose is not
summarised here: each departure is recorded in
[`tests/EXPECTED_DIVERGENCE.md`](tests/EXPECTED_DIVERGENCE.md) with what moved,
by how much, and how it was measured. This file says *that* something changed;
that file says what it cost.

## [Unreleased]

**0.1.0 has never been published.** The version in `pyproject.toml` is the one
this package was extracted at; there is no release on PyPI or bioconda yet, so
everything below is the state of `main` rather than an upgrade path. (The
`0.0.5` mentioned in `placer_py/config.py` is a *PLACER* release — the C++
project's numbering, not this package's.)

### Added

- Extracted into a standalone repository with an MIT `LICENSE`, making the code
  legally usable and publishable at all.
- `placer-py --version`, and `python -m placer_py` as an equivalent entry point.
- A runnable example dataset (`examples/make_example_data.py`) with a known
  truth set, and the first end-to-end run of the pipeline on real files rather
  than on literals in a test.
- A GIAB HG002 ONT-UL evaluation harness (`tools/make_giab_eval.py`) that cuts
  a development slice and a holdout slice, with the labels in the manifest: a
  change that needs the holdout to justify it is a fit, not a fix.
- `py.typed`, so the annotations are visible to a downstream mypy or pyright
  instead of being erased to `Any`.
- `docs/off-pipeline-modules.md`, stating which of the modules nothing imports
  is deliberate and why.

### Changed

- Peak memory during a scan is now O(one bin) rather than O(genome): reads are
  streamed through the bin loop instead of being materialised first, which is
  what makes a genome-scale run possible at all.
- The banded edit-distance DP was rewritten: 2.3x less CPU, byte-identical
  output.
- The event consensus takes an explicit memory budget, and reports when the
  budget capped it, rather than being bounded indirectly by a read cap that
  cannot bound it.
- `clamp`, `log_sum_exp` and the count models have one definition each. Nine
  copies of `clamp` gave three different answers for NaN, and the surviving
  policy propagates it; `log_sum_exp` keeps both of the semantics that were in
  use, now spelled as `ignore_nonfinite=`.
- Tooling: ruff and mypy are CI gates, the test job covers Python 3.9–3.12, and
  the packaging job installs the built wheel into a clean environment and runs
  the console script.

### Fixed

- **TSD detection could not report a duplication at all.** Every detection
  returned `NONE`, so no call carried a target-site duplication — the single
  most mechanistically informative field for a TPRT insertion.
- `log_sum_exp` returned NaN when every input was the impossible sentinel; it
  now returns `-inf`, which is what "nothing is possible" means.
- The structure explanation no longer zeroes the shadow path when the TE
  alignment explains nothing, matching the C++ and the golden vectors.
- `finalization` no longer carries its own inline copy of the e-BH selection it
  shares with `selection.py`, where the argument for averaging rather than
  maximising e-values is written down.
- Six module docstrings cited test files that do not exist, five of them
  pointing at a real file about a different subject.
