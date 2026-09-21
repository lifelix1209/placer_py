from __future__ import annotations

import argparse
from pathlib import Path

import pysam

from placer_py.redesign.candidates.sniffles_vcf import iter_sniffles_insertions
from placer_py.redesign.evidence.bam_local import collect_bam_evidence_from_handle
from placer_py.redesign.evidence.reference_boundary import collect_reference_evidence
from placer_py.redesign.evidence.segmental_explanation import build_segmental_explanation
from placer_py.redesign.evidence.sequence import (
    best_kmer_te_hit,
    build_te_kmer_index,
    read_fasta_records,
    sequence_features,
)
from placer_py.redesign.evidence.te_alignment import read_te_besthits, segmental_from_record
from placer_py.redesign.io import write_ledger
from placer_py.redesign.model.placer_discriminator import (
    build_discriminator_evidence,
    evaluate_joint_hypotheses,
)
from placer_py.redesign.model.te_explainer import explain_candidate
from placer_py.redesign.models import BamEvidence, TeExplanation

MIN_REFERENCE_TRIMMED_MODEL_SEQUENCE_LEN = 17


def _sequence_for_model(raw_sequence: str | None, reference_core: str | None) -> str | None:
    if reference_core is not None and len(reference_core) >= MIN_REFERENCE_TRIMMED_MODEL_SEQUENCE_LEN:
        return reference_core
    return raw_sequence


def run_shadow_pipeline(
    vcf_path: str | Path,
    te_fasta: str | Path,
    output_dir: str | Path,
    bam_path: str | Path | None = None,
    reference_fasta: str | Path | None = None,
    te_besthits_path: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    te_refs = read_fasta_records(te_fasta)
    te_index = build_te_kmer_index(te_refs)
    # When real minimap2 TE alignments are supplied, drive the segmental
    # explanation from them; otherwise fall back to the exact-k-mer proxy.
    te_besthits = read_te_besthits(te_besthits_path) if te_besthits_path is not None else None
    explanations = []
    bam = pysam.AlignmentFile(str(bam_path), "rb") if bam_path is not None else None
    reference = pysam.FastaFile(str(reference_fasta)) if reference_fasta is not None else None
    try:
        for candidate in iter_sniffles_insertions(vcf_path):
            bam_evidence = (
                collect_bam_evidence_from_handle(bam, candidate.chrom, candidate.pos)
                if bam is not None
                else BamEvidence()
            )
            reference_evidence = (
                collect_reference_evidence(candidate, reference, candidate.alt_sequence)
                if reference is not None
                else None
            )
            seq = _sequence_for_model(
                candidate.alt_sequence,
                reference_evidence.insert_core_sequence if reference_evidence is not None else None,
            )
            seq_features = sequence_features(seq)
            te_hit = best_kmer_te_hit(seq, te_refs, te_index)
            if te_besthits is not None:
                segmental = segmental_from_record(seq, te_besthits.get(candidate.candidate_id))
            else:
                segmental = build_segmental_explanation(seq, te_refs, te_index)
            discriminator_evidence = build_discriminator_evidence(
                candidate,
                bam_evidence,
                seq_features,
                te_hit,
                reference_evidence=reference_evidence,
                segmental=segmental,
            )
            discriminator = evaluate_joint_hypotheses(discriminator_evidence)
            legacy_explanation = explain_candidate(candidate, bam_evidence, seq_features, te_hit)
            explanations.append(
                TeExplanation(
                    candidate_id=legacy_explanation.candidate_id,
                    label=discriminator.label,
                    te_posterior=discriminator.te_posterior,
                    non_te_posterior=discriminator.non_te_posterior,
                    artifact_posterior=discriminator.artifact_posterior,
                    lfdr=discriminator.lfdr,
                    te_hit=te_hit,
                    sequence_features=seq_features,
                    bam_evidence=bam_evidence,
                    discriminator=discriminator,
                )
            )
    finally:
        if bam is not None:
            bam.close()
        if reference is not None:
            reference.close()
    write_ledger(output_dir / "evidence_ledger_py.tsv", explanations)
    write_ledger(
        output_dir / "scientific_py.tsv",
        [e for e in explanations if e.discriminator is not None and e.discriminator.emit_te_call],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python PLACER shadow pipeline for Sniffles2 INS candidates")
    parser.add_argument("--sniffles-vcf", required=True)
    parser.add_argument("--te-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bam", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument(
        "--te-besthits",
        default=None,
        help="Optional minimap2 TE besthits TSV; drives segmental explanation from real alignments",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_shadow_pipeline(
        vcf_path=args.sniffles_vcf,
        te_fasta=args.te_fasta,
        output_dir=args.output_dir,
        bam_path=args.bam,
        reference_fasta=args.reference,
        te_besthits_path=args.te_besthits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
