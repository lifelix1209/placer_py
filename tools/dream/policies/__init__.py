"""Decision policies for replay.

A policy is a module that defines `NAME`, a one-line `DESCRIPTION`, and

    select(rows: list, q: float, **params) -> list[Decision]

`rows` are a world's EVALUATED ledger rows (`world.Row`). The policy may read
any of their recorded fields except the coordinates (`chrom`, `pos`, `tid`),
which it may use only to group rows into loci. It must not mutate the rows:
`objective.check_invariance` replays it on shuffled, shifted copies and
requires the same calls.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class Decision:
    row: object
    #: "TE" or "STRUCTURAL".
    label: str
    e_value: float
    family: str
    te_class: str
    #: The breakpoint reported for the call (TEBench's pos0, the VCF POS), when
    #: the policy places it other than at its own row's VCF position: a policy
    #: may test a locus by one row and place it at another's breakpoint.
    pos: int | None = None


def load(name_or_path: str) -> ModuleType:
    """A policy by module name (`current`) or by the path of a candidate file."""
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(f"dream_candidate_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(f"tools.dream.policies.{name_or_path}")
