from __future__ import annotations

import math

from placer_py.redesign.models import (
    BamEvidence,
    InsertionCandidate,
    SequenceFeatures,
    TeExplanation,
    TeHit,
)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values())
    weights = {key: math.exp(value - max_score) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def explain_candidate(
    candidate: InsertionCandidate,
    bam_evidence: BamEvidence,
    sequence_features: SequenceFeatures,
    te_hit: TeHit,
) -> TeExplanation:
    insert_support = math.log1p(candidate.support + bam_evidence.raw_cigar_insert_reads)
    ref_conflict = math.log1p(bam_evidence.ref_span_reads + bam_evidence.low_mapq_ref_span_reads)
    te_signal = te_hit.query_coverage * te_hit.identity
    sequence_signal = min(1.0, sequence_features.length / float(max(1, candidate.abs_svlen)))
    low_complexity_penalty = sequence_features.low_complexity_fraction
    scores = {
        "TE": 1.5 * te_signal + 0.6 * insert_support + 0.4 * sequence_signal - 0.8 * ref_conflict - 0.3 * low_complexity_penalty,
        "nonTE": 0.7 * insert_support + 0.5 * sequence_signal - 0.5 * te_signal - 0.5 * ref_conflict,
        "artifact": 0.9 * ref_conflict + 0.6 * low_complexity_penalty - 0.4 * insert_support,
    }
    probs = _softmax(scores)
    # `probs[k]`, not `probs.get`: `dict.get` is typed as returning
    # `float | None`, so the key function claims it can hand `max` a None.
    label = max(probs, key=lambda name: probs[name])
    return TeExplanation(
        candidate_id=candidate.candidate_id,
        label=label,
        te_posterior=probs["TE"],
        non_te_posterior=probs["nonTE"],
        artifact_posterior=probs["artifact"],
        lfdr=1.0 - probs["TE"],
        te_hit=te_hit,
        sequence_features=sequence_features,
        bam_evidence=bam_evidence,
    )
