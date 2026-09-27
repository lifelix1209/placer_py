"""PLACER's own decision, as finalization runs it: pi_0 of every replay.

It is `coverage_placed`, which calls the production function
(`placer.core.mechanism_selection.select_loci_coverage`) on copies of the
rows, so a replay of this policy is the online decision. The only thing that
can differ is the world: a world recorded by an older scan holds what that scan
measured.

Until 2026-09-27 `current` was 4cbf656's likelihood-gated decision
(`select_loci`, deleted with the legacy decision). Tree nodes logged before
then against `current` compare against that.
"""

from __future__ import annotations

from tools.dream.policies.coverage_placed import select

NAME = "current"
DESCRIPTION = "placer's own decision (coverage_placed)"

__all__ = ["NAME", "DESCRIPTION", "select"]
