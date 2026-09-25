"""
The command line: `placer [options] <input.bam> <ref.fa> <te.fa>`.

Ported from `src/main.cpp`, pinned by `tests/test_32_pipeline.py`.

WHAT THE CLI IS RESPONSIBLE FOR, and it is deliberately little: parse the
arguments, apply the environment overrides, build the three external
dependencies (BAM, reference, TE library), run the pipeline, write five files.
Every decision lives in the stages; nothing here chooses anything.

THE FOUR FLAGS are the ones that survived. `--final-fdr-q` is the single policy
knob -- a target false-call RISK, not an evidence weight -- and the other three
are output shape. `--threads` is a fifth that decides nothing: it sets how
many processes scan (`placer/parallel.py`), and the files are the same
bytes for any value. The read-count, GQ, insert-length and segmentation-score
ladders that used to be flags were deleted along with the thresholds behind
them; see `placer/core/policy.py`.

REGION PARSING IS 1-BASED INCLUSIVE ON THE WAY IN, half-open 0-based
internally, because that is what every genomics tool's `chrom:start-end` means
and what htslib's fetch wants. Getting it wrong by one is silent, so the
conversion happens in exactly one place.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

from placer import __version__
from placer.config import BamRegionScope, FinalReportMode, PipelineConfig
from placer.report.writer import write_outputs

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

USAGE = ("placer [--region <chrom:start-end>] [--threads <n>] [--final-fdr-q <q>] "
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
                             environ: Mapping[str, str] | None = None) -> PipelineConfig:
    """Apply `PLACER_*` overrides. Unparsable values are WARNED ABOUT and ignored.

    Deliberately forgiving: an environment variable is often set once and
    inherited by unrelated runs, and aborting a whole-genome run over a stale
    `PLACER_BIN_SIZE=abc` would be worse than using the default. A value that
    parses is always applied, so a deliberate override is never silently lost.

    But forgiving is not the same as silent, and it used to be both. A typo'd
    `PLACER_TSD_BG_P_MAX=0,05` produced no warning and no effect, so a user
    who thought they had loosened the TSD significance bar got the default
    and no way to find out. The warning costs one line on stderr and keeps
    the tolerance.

    Note this only covers variables the table below RECOGNISES. An unknown
    `PLACER_*` name is still ignored in silence, which is a separate gap:
    `PLACER_FINAL_FDR_Q` looks like it should work and does not, because the
    final risk level is a CLI flag only.
    """
    environ = environ if environ is not None else os.environ

    def _apply(name: str, field_name: str, parse) -> None:
        raw = environ[name]
        try:
            setattr(config, field_name, parse(raw))
        except ValueError:
            print(f"[PLACER] ignoring {name}={raw!r}: not a valid "
                  f"{parse.__name__} value; using the default", file=sys.stderr)

    for name, field_name in _ENV_INT_FIELDS.items():
        if name in environ:
            _apply(name, field_name, int)
    for name, field_name in _ENV_FLOAT_FIELDS.items():
        if name in environ:
            _apply(name, field_name, float)
    for name, field_name in _ENV_BOOL_FIELDS.items():
        if name in environ:
            setattr(config, field_name, _env_bool(environ[name]))
    for name, field_name in _ENV_STRING_FIELDS.items():
        if name in environ:
            setattr(config, field_name, environ[name])
    if "PLACER_PARALLEL" in environ and _env_bool(environ["PLACER_PARALLEL"]):
        # The C++ switch, which named a different executor. Saying what to use
        # instead is better than accepting it and running on one core: a user
        # who set it is expecting a speed-up.
        print("[PLACER] PLACER_PARALLEL is not read by placer; "
              "use --threads N", file=sys.stderr)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placer", usage=USAGE, add_help=True)
    parser.add_argument("--version", action="version",
                        version=f"placer {__version__}",
                        help="print the version and exit")
    parser.add_argument("--region", default=None,
                        help="restrict the scan to chrom or chrom:start-end (1-based)")
    parser.add_argument("-t", "--threads", type=int, default=None,
                        help="scan worker processes (default 1); the output is "
                             "identical for any value")
    parser.add_argument("--final-fdr-q", type=float, default=None,
                        help="target false-call risk for the final selection")
    parser.add_argument("--final-report-mode", default=None,
                        choices=("legacy", "te-calibrated"),
                        help="legacy keeps structural insertions in the main output")
    parser.add_argument("--min-final-raw-cigar-insert-len-bp", type=int, default=None)
    parser.add_argument("--output-dir", default=".",
                        help="where the five output files are written")
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
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError(f"--threads must be at least 1, got {args.threads}")
        config.scan_workers = args.threads
    if args.final_fdr_q is not None:
        config.final_fdr_q = args.final_fdr_q
    if args.final_report_mode is not None:
        config.final_report_mode = (FinalReportMode.LEGACY
                                    if args.final_report_mode == "legacy"
                                    else FinalReportMode.TE_CALIBRATED)
    if args.min_final_raw_cigar_insert_len_bp is not None:
        config.min_final_raw_cigar_insert_len_bp = args.min_final_raw_cigar_insert_len_bp
    return config


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
    from placer.io.bam import make_bam_reader
    from placer.io.reference import ReferenceFetcher
    from placer.io.report_context import build_report_context
    from placer.io.te_library import load_te_library
    from placer.pipeline import run_pipeline
    from placer.wiring import build_stage_hooks

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

    entries = load_te_library(config.te_fasta_path)
    if not entries:
        print(f"[PLACER] empty or unreadable TE library: {config.te_fasta_path}",
              file=sys.stderr)
        return 1
    try:
        if config.scan_workers > 1:
            from placer.io.te_library import TeLibraryAligner
            from placer.parallel import run_pipeline_parallel

            # The parent scans nothing, so it builds no hooks -- each worker
            # builds its own. It does build the BLAST database, once, so the
            # workers find it on disk rather than racing to create it.
            TeLibraryAligner(config, entries).prepare()
            result = run_pipeline_parallel(reader, config, config.scan_workers)
        else:
            hooks = build_stage_hooks(config, reference, entries)
            result = run_pipeline(reader.stream(), reader.chromosome_name,
                                  reader.fetch, config, hooks)
        # BUILT BEFORE THE HANDLES CLOSE, and that is the only reason it is
        # inside the `try`: the contig list comes from the BAM header and the
        # VCF anchor bases come from the reference, so both must still be open.
        # It is still AFTER the run -- a few thousand single-base fetches at
        # the end rather than reference I/O inside the bin loop.
        context = build_report_context(reader, reference.fetch_window,
                                       config, result)
    finally:
        reader.close()
        reference.close()

    paths = write_outputs(result, output_dir, context=context)
    for key, name in (("scientific_txt", "scientific.txt"),
                      ("evidence_ledger_tsv", "evidence_ledger.tsv"),
                      ("structural_calls_tsv", "structural_calls.tsv"),
                      ("calls_vcf", "calls.vcf"),
                      ("calls_csv", "calls.csv")):
        print(f"[PLACER] wrote {name} path={paths[key]}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as error:
        print(f"[PLACER] {error}", file=sys.stderr)
        return 1
    return run_pipeline_once(config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
