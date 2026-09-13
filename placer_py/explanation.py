"""
Pareto comparison of competing event explanations.

Ported from `src/pipeline/event_explanation.cpp`, pinned by
`tests/test_15_explanation.py`.

THE IDEA, and it is the answer to the second of the two unanswerable questions.

Fitting weights needs a gold standard, and "is there a TE insertion here" IS the
thing being produced. But even with weights there is a second problem: to trade
50 bp of unexplained sequence against 2 reference-spanning reads you must first
declare an exchange rate between them, and no such rate has any objective basis.

So do not declare one. Four candidate explanations -- Reference, non-TE
insertion, TE, Artifact -- each produce a residual vector whose coordinates are
all "what this explanation failed to account for", and they are compared by
DOMINANCE rather than by a sum:

    A dominates B  <=>  A is no worse in every primary coordinate
                        AND strictly better in at least one.

No exchange rate is needed, because the comparison never adds coordinates
together.

The cost is that a partial order is incomplete: sometimes nothing dominates. That
is the point rather than a defect -- when TE is best but does not uniquely
dominate, the evidence is sufficient but not exclusive, and the honest action is
to abstain. `TE_AMBIGUOUS` is that abstention. A weighted sum can never produce
it, because scalars are always comparable.

Note which coordinates are PRIMARY. `breakpoint_disagreement_bp`,
`label_ambiguity` and `path_complexity` are excluded from the dominance test and
used only to break ties, so a candidate cannot be rejected for an imprecise
breakpoint alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ExplanationKind(IntEnum):
    REFERENCE = 0
    INSERTION_NON_TE = 1
    TE = 2
    ARTIFACT = 3


KIND_NAMES = {
    ExplanationKind.REFERENCE: "REFERENCE",
    ExplanationKind.INSERTION_NON_TE: "INSERTION_NON_TE",
    ExplanationKind.TE: "TE",
    ExplanationKind.ARTIFACT: "ARTIFACT",
}


@dataclass
class ExplanationResidual:
    structural_conflicts: int = 0
    missing_required_components: int = 0
    unexplained_high_complexity_bases: int = 0
    breakpoint_disagreement_bp: int = 0
    read_assignment_conflicts: int = 0
    reference_counterevidence: int = 0
    artifact_evidence: int = 0
    label_ambiguity: int = 0
    path_complexity: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class EventExplanation:
    kind: ExplanationKind = ExplanationKind.REFERENCE
    residual: ExplanationResidual = field(default_factory=ExplanationResidual)
    path: list = field(default_factory=list)
    family: str = "NA"
    subfamily: str = "NA"
    status: str = "UNSET"


@dataclass
class ExplanationDecision:
    best: EventExplanation = field(default_factory=EventExplanation)
    alternatives: list[EventExplanation] = field(default_factory=list)
    final_qc: str = "NO_CALL_INCOMPLETE"
    emit_te_call: bool = False
    emit_unknown_te: bool = False
    emit_evidence_te_call: bool = False


#: The coordinates the dominance test uses. The three left out --
#: breakpoint_disagreement_bp, label_ambiguity, path_complexity -- break ties
#: only, so an imprecise breakpoint never rejects a candidate on its own.
PRIMARY_FIELDS = ("structural_conflicts", "missing_required_components",
                  "unexplained_high_complexity_bases",
                  "read_assignment_conflicts", "reference_counterevidence",
                  "artifact_evidence")

TIE_BREAK_FIELDS = ("breakpoint_disagreement_bp", "label_ambiguity",
                    "path_complexity")


def explanation_kind_name(kind: ExplanationKind) -> str:
    return KIND_NAMES.get(kind, "UNKNOWN")


def _primary_no_worse(lhs: ExplanationResidual,
                      rhs: ExplanationResidual) -> bool:
    return all(getattr(lhs, f) <= getattr(rhs, f) for f in PRIMARY_FIELDS)


def _primary_strictly_better(lhs: ExplanationResidual,
                             rhs: ExplanationResidual) -> bool:
    return any(getattr(lhs, f) < getattr(rhs, f) for f in PRIMARY_FIELDS)


def dominates_primary_residuals(lhs: ExplanationResidual,
                                rhs: ExplanationResidual) -> bool:
    """No worse in every primary coordinate, strictly better in at least one."""
    return _primary_no_worse(lhs, rhs) and _primary_strictly_better(lhs, rhs)


def _primary_sum(residual: ExplanationResidual) -> int:
    return sum(getattr(residual, f) for f in PRIMARY_FIELDS)


def _tie_break_sum(residual: ExplanationResidual) -> int:
    return sum(getattr(residual, f) for f in TIE_BREAK_FIELDS)


def _compare(lhs: EventExplanation, rhs: EventExplanation) -> int:
    """
    The C++ comparator, reproduced exactly -- including that it is NOT a strict
    weak ordering.

    It consults `dominates_primary_residuals` first, which is a PARTIAL order
    and can be non-transitive across a set of three or more. Reproducing it with
    a sort key rather than a comparator would change the result on any set where
    dominance cycles, so this port keeps the comparator form and uses
    `functools.cmp_to_key`. That is a faithful port of something slightly
    unsound rather than a quiet correction: tidying it changes behaviour and
    belongs with a deliberate review, not with a migration.
    """
    lhs_dominates = dominates_primary_residuals(lhs.residual, rhs.residual)
    rhs_dominates = dominates_primary_residuals(rhs.residual, lhs.residual)
    if lhs_dominates != rhs_dominates:
        return -1 if lhs_dominates else 1

    lhs_primary = _primary_sum(lhs.residual)
    rhs_primary = _primary_sum(rhs.residual)
    if lhs_primary != rhs_primary:
        return -1 if lhs_primary < rhs_primary else 1

    lhs_tie = _tie_break_sum(lhs.residual)
    rhs_tie = _tie_break_sum(rhs.residual)
    if lhs_tie != rhs_tie:
        return -1 if lhs_tie < rhs_tie else 1

    # Higher kind first: ARTIFACT > TE > INSERTION_NON_TE > REFERENCE.
    if int(lhs.kind) != int(rhs.kind):
        return -1 if int(lhs.kind) > int(rhs.kind) else 1
    return 0


def _has_missing_required_components(explanation: EventExplanation) -> bool:
    return (explanation.residual.missing_required_components > 0
            or explanation.residual.structural_conflicts > 0)


def compare_event_explanations(explanations: list[EventExplanation],
                               closed_breakpoints: bool) -> ExplanationDecision:
    """Port of `placer::compare_event_explanations`."""
    import functools

    decision = ExplanationDecision()
    if not explanations:
        return decision

    ranked = sorted(explanations, key=functools.cmp_to_key(_compare))
    decision.best = ranked[0]
    decision.alternatives = ranked

    uniquely_dominates_all = True
    for other in ranked[1:]:
        if not dominates_primary_residuals(decision.best.residual,
                                           other.residual):
            uniquely_dominates_all = False
            break

    if _has_missing_required_components(decision.best):
        decision.final_qc = "NO_CALL_INCOMPLETE"
        return decision

    if decision.best.kind == ExplanationKind.TE:
        if not uniquely_dominates_all:
            # Sufficient evidence, but not exclusive. Abstaining is the honest
            # action, and a scalar score could never express it.
            decision.final_qc = "TE_AMBIGUOUS"
            return decision
        decision.emit_te_call = True
        decision.emit_evidence_te_call = not closed_breakpoints
        decision.emit_unknown_te = (not closed_breakpoints
                                    or not decision.best.subfamily
                                    or decision.best.subfamily == "UNKNOWN"
                                    or decision.best.family == "UNKNOWN")
        decision.final_qc = ("PASS_TE_CLOSED" if closed_breakpoints
                             else "PASS_TE_IMPRECISE")
        return decision

    if (decision.best.kind == ExplanationKind.INSERTION_NON_TE
            and uniquely_dominates_all):
        decision.final_qc = "PASS_NONTE_INSERTION"
        return decision

    decision.final_qc = "REFERENCE_OR_ARTIFACT"
    return decision


def serialize_residual(residual: ExplanationResidual) -> str:
    return (f"structural={residual.structural_conflicts}"
            f";missing={residual.missing_required_components}"
            f";unexplained={residual.unexplained_high_complexity_bases}"
            f";breakpoint={residual.breakpoint_disagreement_bp}"
            f";read_conflicts={residual.read_assignment_conflicts}"
            f";ref_counter={residual.reference_counterevidence}"
            f";artifact={residual.artifact_evidence}"
            f";label_ambiguity={residual.label_ambiguity}"
            f";path_complexity={residual.path_complexity}")


def serialize_explanation_path(explanation: EventExplanation) -> str:
    """`start-end:label` per segment, or `NA`.

    The path is empty for every explanation this pipeline builds -- the four
    candidates are residual vectors, not segmentations -- so this reports `NA`
    in practice. Ported anyway because the column is part of the output
    contract, and a reader must not have to guess whether `NA` means "no path"
    or "column missing".
    """
    if not explanation.path:
        return "NA"
    return ",".join(f"{segment.query_start}-{segment.query_end}:{segment.label}"
                    for segment in explanation.path)
