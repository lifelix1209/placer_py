"""
The conformal null selector: the second selection route.

Ported from `src/pipeline/conformal_selector.cpp`, pinned by
`tests/test_16_conformal.py`.

WHY IT EXISTS ALONGSIDE e-BH. The two routes have different failure modes.
e-BH depends on the e-values really being e-values -- that is, on the score's
null expectation being at most 1, which `placer/core/decoys.py` verifies. The
conformal route depends on no distributional assumption at all: it only counts
how many rows that were NOT selected look at least as TE-like as the candidate.
A locus that fools both has to survive a model-based argument and an
assumption-free one.

THE SAME TOOL, A DIFFERENT PURPOSE. `placer/core/explanation.py` uses dominance to
avoid declaring an exchange rate between residual coordinates. Here dominance
avoids assuming the features' JOINT DISTRIBUTION -- it never asks how identity
and alt-read count covary, only how many nulls are no worse in every coordinate.
Both are the same evasion: when a quantity has no objective basis, change the
question to one that does not need it.

THE PRICE. Conformal p-values are selected with Benjamini-Yekutieli, whose
harmonic factor `H_m` is what admitting arbitrary dependence among p-VALUES
costs -- about 7.49 at m = 1000, turning a nominal q = 0.10 into an effective
0.0134. That factor is exactly what switching the currency to e-values removes,
which is why e-BH is primary and this is the second opinion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Below this many same-context nulls the selector falls back to the pooled set,
#: so power is not lost in a sparsely-populated context. Deterministic.
MIN_CONTEXT_NULLS = 12


@dataclass
class ConformalFeatureVector:
    """
    `pro_te` coordinates are oriented so LARGER is more TE-like;
    `ref_span_reads` is kept separate and compared with the opposite sense,
    because fewer reference-spanning reads is more TE-like.

    `context` is a discrete local-context bucket (clean / low-complexity tandem /
    one-sided). A candidate is compared against nulls in the SAME context, which
    makes the empirical null context-conditioned -- the same idea as measuring
    the TPRT model's backgrounds locally, applied to a different layer.
    """

    id: str = "NA"
    pro_te: list[float] = field(default_factory=list)
    ref_span_reads: float = 0.0
    context: int = 0


@dataclass
class ConformalSelectionResult:
    id: str = "NA"
    conformal_p: float = 1.0
    by_threshold: float = 0.0
    dominated_null_count: int = 0
    null_count: int = 0
    pass_: bool = False
    qc: str = "CONFORMAL_NULL_INSUFFICIENT"


def _finite(feature: ConformalFeatureVector) -> bool:
    if not math.isfinite(feature.ref_span_reads):
        return False
    return all(math.isfinite(v) for v in feature.pro_te)


def null_dominates_candidate(null_feature: ConformalFeatureVector,
                             candidate: ConformalFeatureVector) -> bool:
    """A null dominates when it is at least as TE-like in EVERY coordinate."""
    if (not null_feature.pro_te
            or len(null_feature.pro_te) != len(candidate.pro_te)
            or not _finite(null_feature) or not _finite(candidate)):
        return False
    for null_value, candidate_value in zip(null_feature.pro_te,
                                           candidate.pro_te):
        if null_value < candidate_value:
            return False
    return null_feature.ref_span_reads <= candidate.ref_span_reads


def harmonic_number(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1))


class ConformalNullSelector:
    def __init__(self) -> None:
        self._nulls: list[ConformalFeatureVector] = []

    def add_null_control(self, feature: ConformalFeatureVector) -> None:
        """Non-finite or empty feature vectors are dropped rather than stored,
        so a malformed row cannot silently enlarge the denominator."""
        if feature.pro_te and _finite(feature):
            self._nulls.append(feature)

    def null_count(self) -> int:
        return len(self._nulls)

    def _relevant_nulls(self, candidate: ConformalFeatureVector
                        ) -> list[ConformalFeatureVector]:
        same_context = [n for n in self._nulls if n.context == candidate.context]
        if len(same_context) >= MIN_CONTEXT_NULLS:
            return same_context
        return list(self._nulls)

    def dominated_null_count(self, candidate: ConformalFeatureVector) -> int:
        return sum(1 for n in self._relevant_nulls(candidate)
                   if null_dominates_candidate(n, candidate))

    def dominance_p_value(self, candidate: ConformalFeatureVector) -> float:
        """`(1 + #dominating) / (1 + #relevant)`.

        Add-one smoothing keeps the p-value strictly positive, which matters
        because a zero p-value would claim more certainty than a finite null set
        can support.
        """
        if not self._nulls or not candidate.pro_te or not _finite(candidate):
            return 1.0
        relevant = self._relevant_nulls(candidate)
        return (1 + self.dominated_null_count(candidate)) / (1 + len(relevant))

    def _structural_support_p(self, relevant: list[ConformalFeatureVector],
                              candidate: ConformalFeatureVector) -> float:
        """
        An upper-tail p-value on the FIRST pro-TE coordinate alone
        (`alt_struct_reads`).

        Taken as a MAXIMUM with the dominance p-value, so a candidate whose only
        advantage is raw read count cannot pass on that alone: dominance can be
        generous when the other coordinates are uninformative, and this keeps
        the count honest on its own terms.
        """
        if not relevant or not candidate.pro_te or not _finite(candidate):
            return 1.0
        ge = sum(1 for n in relevant
                 if n.pro_te and _finite(n)
                 and n.pro_te[0] >= candidate.pro_te[0])
        return (1 + ge) / (1 + len(relevant))

    def select(self, candidates: list[ConformalFeatureVector],
               target_fdr: float) -> list[ConformalSelectionResult]:
        """BY step-up over the conformal p-values."""
        results: list[ConformalSelectionResult] = []
        for candidate in candidates:
            relevant = self._relevant_nulls(candidate)
            result = ConformalSelectionResult(
                id=candidate.id,
                null_count=len(relevant),
                dominated_null_count=self.dominated_null_count(candidate),
                conformal_p=max(self.dominance_p_value(candidate),
                                self._structural_support_p(relevant, candidate)),
                qc=("CONFORMAL_NULL_INSUFFICIENT" if not self._nulls
                    else "CONFORMAL_FDR_REJECT"))
            results.append(result)

        if not results or not self._nulls:
            return results

        # Ties break by id, matching the C++ comparator, so the same set is
        # selected in both languages when p-values coincide.
        order = sorted(range(len(results)),
                       key=lambda i: (results[i].conformal_p, results[i].id))
        q = min(max(target_fdr, 0.0), 1.0)
        m = len(order)
        harmonic = max(1.0, harmonic_number(m))

        selected_prefix = 0
        selected_threshold = 0.0
        for rank in range(1, m + 1):
            threshold = (rank * q) / (m * harmonic)
            if results[order[rank - 1]].conformal_p <= threshold:
                selected_prefix = rank
                selected_threshold = threshold

        for rank in range(1, m + 1):
            index = order[rank - 1]
            results[index].by_threshold = (rank * q) / (m * harmonic)
            if selected_prefix > 0 and rank <= selected_prefix:
                results[index].pass_ = True
                results[index].by_threshold = selected_threshold
                results[index].qc = "PASS_CONFORMAL_FDR"
        return results
