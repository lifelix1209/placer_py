"""
The command line: `placer [options] <input.bam> <ref.fa> <te.fa>`.

Ported from `src/main.cpp`, pinned by `tests/test_32_pipeline.py`.

WHAT THE CLI IS RESPONSIBLE FOR, and it is deliberately little: parse the
arguments, apply the environment overrides, build the three external
dependencies (BAM, reference, TE library), run the pipeline, write three files.
Every decision lives in the stages; nothing here chooses anything.

THE FOUR FLAGS are the ones that survived. `--final-fdr-q` is the single policy
knob -- a target false-call RISK, not an evidence weight -- and the other three
are output shape. The read-count, GQ, insert-length and segmentation-score
ladders that used to be flags were deleted along with the thresholds behind
them; see `placer_py/policy.py`.

REGION PARSING IS 1-BASED INCLUSIVE ON THE WAY IN, half-open 0-based
internally, because that is what every genomics tool's `chrom:start-end` means
and what htslib's fetch wants. Getting it wrong by one is silent, so the
conversion happens in exactly one place.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from placer_py import __version__
from placer_py.config import BamRegionScope, FinalReportMode, PipelineConfig

#: `PLACER_*` environment overrides, and the config field each sets. The C++
#: reads these instead of adding flags for every tuning parameter; they are
#: pinned here so a run can be reproduced from its environment.
_ENV_INT_FIELDS = {
    "PLACER_PROGRESS_INTERVAL": "progress_interval",
    "PLACER_BIN_SIZE": "bin_size",
    "PLACER_BAM_THREADS": "bam_threads",
    "PLACER_TE_KMER_SIZE": "te_kmer_size",
    "PLACER_MIN_SOFT_CLIP_FOR_SEQ_EXTRACT": "min_soft_clip_for_seq_extract",
    "PLACER_MIN_LONG_INS_FOR_SEQ_EXTRACT": "min_long_ins_for_seq_extract",
    "PLACER_MIN_SA_ALN_LEN_FOR_SEQ_EXTRACT": "min_sa_aln_len_for_seq_extract",
    "PLACER_MAX_SA_PER_READ": "max_sa_per_read",
    "PLACER_TSD_MIN_LEN": "tsd_min_len",
    "PLACER_TSD_MAX_LEN": "tsd_max_len",
    "PLACER_TSD_FLANK_WINDOW": "tsd_flank_window",
    "PLACER_EVENT_CONSENSUS_POA_MIN_READS": "event_consensus_poa_min_reads",
    "PLACER_EVENT_CONSENSUS_POA_MAX_READS": "event_consensus_poa_max_reads",
}
_ENV_FLOAT_FIELDS = {
    "PLACER_GENOTYPE_ERROR_RATE": "genotype_error_rate",
    "PLACER_GENOTYPE_OVERDISPERSION": "genotype_overdispersion",
    "PLACER_TSD_BG_P_MAX": "tsd_bg_p_max",
    "PLACER_TE_FAMILY_MARGIN_MIN": "te_family_margin_min",
    "PLACER_TE_SUBFAMILY_MARGIN_MIN": "te_subfamily_margin_min",
}
_ENV_BOOL_FIELDS = {
    "PLACER_LOG_STAGE_BINS": "log_stage_bins",
    "PLACER_LOG_STAGE_COMPONENTS": "log_stage_components",
    "PLACER_TSD_ENABLE": "tsd_enable",
    "PLACER_SHORT_INS_ENABLE": "short_ins_enable",
    "PLACER_TE_LOW_KMER_RESCUE_ENABLE": "te_low_kmer_rescue_enable",
}
_ENV_STRING_FIELDS = {
    "PLACER_TE_KMER_SIZES": "te_kmer_sizes_csv",
    "PLACER_INS_FRAGMENTS_FASTA": "ins_fragments_fasta_path",
    "PLACER_INS_FRAGMENT_HITS_TSV": "ins_fragment_hits_tsv_path",
    "PLACER_BLASTN": "te_blastn_path",
    "PLACER_MAKEBLASTDB": "te_makeblastdb_path",
}

USAGE = ("placer [--region <chrom:start-end>] [--final-fdr-q <q>] "
         "[--final-report-mode <legacy|te-calibrated>] "
         "[--min-final-raw-cigar-insert-len-bp <bp>] <input.bam> <ref.fa> <te.fa>")


def parse_region_scope(region: str) -> BamRegionScope:
    """`chrom`, or `chrom:start-end` with 1-based inclusive bounds.

    A bare contig name means the WHOLE contig (`end = -1`), which is why the
    end sentinel is negative rather than 0 -- position 0 is a real coordinate.

    Raises on a malformed range rather than falling back to the whole contig:
    silently widening a region the user restricted would run the whole genome
    when they asked for 100 kb.
    """
    if not region:
        raise ValueError("empty region")
    scope = BamRegionScope(enabled=True)
    colon = region.rfind(":")
    if colon == -1:
        scope.chrom = region
        scope.start = 0
        scope.end = -1
        return scope

    scope.chrom = region[:colon]
    range_text = region[colon + 1:]
    dash = range_text.find("-")
    if not scope.chrom or dash == -1:
        raise ValueError(f"invalid region: {region}")
    try:
        start_1based = int(range_text[:dash])
        end_1based = int(range_text[dash + 1:])
    except ValueError as error:
        raise ValueError(f"invalid region: {region}") from error
    if start_1based <= 0 or end_1based < start_1based or end_1based > 2**31 - 1:
        raise ValueError(f"invalid region: {region}")
    scope.start = start_1based - 1
    scope.end = end_1based
    return scope


def _env_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def apply_environment_config(config: PipelineConfig,
                             environ: dict[str, str] | None = None) -> PipelineConfig:
    """Apply `PLACER_*` overrides. Unparsable values are IGNORED.

    Deliberately forgiving: an environment variable is often set once and
    inherited by unrelated runs, and aborting a whole-genome run over a stale
    `PLACER_BIN_SIZE=abc` would be worse than using the default. A value that
    parses is always applied, so a deliberate override is never silently lost.
    """
    environ = environ if environ is not None else os.environ
    for name, field_name in _ENV_INT_FIELDS.items():
        if name in environ:
            try:
                setattr(config, field_name, int(environ[name]))
            except ValueError:
                pass
    for name, field_name in _ENV_FLOAT_FIELDS.items():
        if name in environ:
            try:
                setattr(config, field_name, float(environ[name]))
            except ValueError:
                pass
    for name, field_name in _ENV_BOOL_FIELDS.items():
        if name in environ:
            setattr(config, field_name, _env_bool(environ[name]))
    for name, field_name in _ENV_STRING_FIELDS.items():
        if name in environ:
            setattr(config, field_name, environ[name])
    if "PLACER_PARALLEL" in environ and _env_bool(environ["PLACER_PARALLEL"]):
        # The parallel executor is not ported. Saying so is better than
        # accepting the flag and running single-threaded, because a user who
        # set it is expecting a speed-up and would otherwise not learn there
        # isn't one.
        print("[PLACER] PLACER_PARALLEL is not supported by placer_py; "
              "running the streaming path", file=sys.stderr)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placer", usage=USAGE, add_help=True)
    parser.add_argument("--version", action="version",
                        version=f"placer-py {__version__}",
                        help="print the version and exit")
    parser.add_argument("--region", default=None,
                        help="restrict the scan to chrom or chrom:start-end (1-based)")
    parser.add_argument("--final-fdr-q", type=float, default=None,
                        help="target false-call risk for the final selection")
    parser.add_argument("--final-report-mode", default=None,
                        choices=("legacy", "te-calibrated"),
                        help="legacy keeps structural insertions in the main output")
    parser.add_argument("--min-final-raw-cigar-insert-len-bp", type=int, default=None)
    parser.add_argument("--output-dir", default=".",
                        help="where the three output files are written")
    parser.add_argument("bam")
    parser.add_argument("reference")
    parser.add_argument("te_fasta")
    return parser


def config_from_args(args, environ: dict[str, str] | None = None) -> PipelineConfig:
    """CLI arguments and environment into one config.

    ORDER MATTERS: the environment is applied FIRST and the explicit flags
    second, so a flag always wins over an inherited variable. The C++ does the
    same, and the reverse would make a command line silently ineffective.
    """
    config = PipelineConfig(bam_path=args.bam, reference_fasta_path=args.reference,
                            te_fasta_path=args.te_fasta)
    if args.region:
        config.bam_region_scope = parse_region_scope(args.region)
    apply_environment_config(config, environ)
    if args.final_fdr_q is not None:
        config.final_fdr_q = args.final_fdr_q
    if args.final_report_mode is not None:
        config.final_report_mode = (FinalReportMode.LEGACY
                                    if args.final_report_mode == "legacy"
                                    else FinalReportMode.TE_CALIBRATED)
    if args.min_final_raw_cigar_insert_len_bp is not None:
        config.min_final_raw_cigar_insert_len_bp = args.min_final_raw_cigar_insert_len_bp
    return config


def write_outputs(result, output_dir: str, include_insert_seq: bool = False,
                  include_support_qnames: bool = False) -> dict[str, str]:
    """Write the three files and return their paths.

    All three, always, even when empty. A missing `structural_calls.tsv` is
    ambiguous between "none were set aside" and "the run died before writing
    it", and a downstream script cannot tell the difference.
    """
    from placer_py import outputs

    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "scientific_txt": os.path.join(output_dir, "scientific.txt"),
        "structural_calls_tsv": os.path.join(output_dir, "structural_calls.tsv"),
        "evidence_ledger_tsv": os.path.join(output_dir, "evidence_ledger.tsv"),
    }
    with open(paths["scientific_txt"], "w") as handle:
        handle.write(outputs.render_scientific_txt(result, include_insert_seq))
    with open(paths["structural_calls_tsv"], "w") as handle:
        handle.write(outputs.render_structural_calls_tsv(result, include_insert_seq))
    with open(paths["evidence_ledger_tsv"], "w") as handle:
        handle.write(outputs.render_evidence_ledger_tsv(result, include_insert_seq,
                                                        include_support_qnames))
    return paths


def run_pipeline_once(config: PipelineConfig, output_dir: str = ".") -> int:
    """Open the inputs, run the pipeline, write the outputs.

    THE FOUR PRECONDITIONS are checked before any work, and each is fatal
    because the run would otherwise produce a clean-looking empty result:

      * a readable BAM;
      * an INDEX on it -- the local stages re-fetch each candidate's reads, and
        without an index every event would be evaluated on bin reads alone;
      * a readable reference -- segmentation cannot place a flank without one;
      * a readable TE library -- without it every insert is
        `TE_LIBRARY_UNAVAILABLE`, which is a silent whole-run negative.
    """
    from placer_py import consensus as consensus_module
    from placer_py import te_classifier
    from placer_py.bam_io import ReferenceFetcher, make_bam_reader
    from placer_py.pipeline import StageHooks, run_pipeline
    from placer_py.seqtools import build_te_sequence_background
    from placer_py.tsd import (TsdConfig, detect as detect_tsd,
                               detect_from_insertion as detect_tsd_from_insertion)

    reader = make_bam_reader(config.bam_path, config.bam_threads,
                             config.bam_region_scope)
    if not reader.is_valid():
        print(f"[PLACER] cannot open BAM: {config.bam_path}", file=sys.stderr)
        return 1
    if not reader.can_fetch():
        print("[PLACER] BAM index required for event-level recollection",
              file=sys.stderr)
        return 1

    reference = ReferenceFetcher(config.reference_fasta_path)
    if not reference.can_fetch_reference():
        print(f"[PLACER] cannot read reference: {config.reference_fasta_path}",
              file=sys.stderr)
        return 1

    with open(config.te_fasta_path) as handle:
        entries = te_classifier.load_te_entries_from_fasta(handle.read())
    if not entries:
        print(f"[PLACER] empty or unreadable TE library: {config.te_fasta_path}",
              file=sys.stderr)
        return 1
    background = build_te_sequence_background([entry.sequence for entry in entries])

    def align_insert(insert_seq: str):
        return te_classifier.align_insert_sequences(config, entries, [insert_seq],
                                                    background)[0]

    tsd_config = TsdConfig(tsd_min_len=config.tsd_min_len,
                           tsd_max_len=config.tsd_max_len,
                           tsd_flank_window=config.tsd_flank_window,
                           tsd_bg_p_max=config.tsd_bg_p_max,
                           tsd_max_mismatch_rate=config.tsd_max_mismatch_rate,
                           tsd_max_mismatches=config.tsd_max_mismatches)

    def detect(chrom: str, bp_left: int, bp_right: int, insert_seq: str):
        """Pick the detector the evidence can actually support.

        Distinct breakpoints mean the caller resolved both edges of the event,
        and their overlap (or gap) is measurable in the reference -- that is
        `detect`. Equal breakpoints mean the aligner emitted a single CIGAR
        `I`, so the reference carries no trace of the duplication and the only
        place left to look is the inserted sequence itself.
        """
        if not config.tsd_enable:
            return None
        if bp_left != bp_right:
            return detect_tsd(reference.fetch_window, chrom, bp_left, bp_right,
                              tsd_config)
        if insert_seq:
            return detect_tsd_from_insertion(reference.fetch_window, chrom,
                                             bp_left, insert_seq, tsd_config)
        return None

    hooks = StageHooks(fetch_reference=reference.fetch_window,
                       align_insert=align_insert,
                       consensus_fn=consensus_module.pyabpoa_consensus,
                       detect_tsd=detect)

    try:
        result = run_pipeline(reader.stream(), reader.chromosome_name,
                              reader.fetch, config, hooks)
    finally:
        reader.close()
        reference.close()

    paths = write_outputs(result, output_dir)
    print(f"[PLACER] wrote scientific.txt path={paths['scientific_txt']}",
          file=sys.stderr)
    print(f"[PLACER] wrote evidence_ledger.tsv path={paths['evidence_ledger_tsv']}",
          file=sys.stderr)
    print(f"[PLACER] wrote structural_calls.tsv path={paths['structural_calls_tsv']}",
          file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "denovo":
        from placer_py.denovo import run_denovo_cli

        return run_denovo_cli(argv[1:])

    args = build_arg_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as error:
        print(f"[PLACER] {error}", file=sys.stderr)
        return 1
    return run_pipeline_once(config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
