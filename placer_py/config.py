"""
The run configuration, as one value.

Ported from `struct PipelineConfig` in `include/pipeline.h`. Every default here
is the C++ default, verbatim, including the ones the README argues against --
`min_final_raw_cigar_insert_len_bp = 50` and the TE-calibrated report mode among
them. A migration that "improves" a default while porting cannot be validated by
diffing against the thing it replaces, because every difference then has two
possible causes.

WHY THE WHOLE CONFIG IS ONE FLAT RECORD rather than per-module structs: the C++
threads a copy of it into every module, and several thresholds are read by more
than one stage (`min_long_ins_for_seq_extract` by fragment extraction AND by the
short-insertion relaxation, `bin_size` by windowing AND by the split-SA flank
search radius). Splitting it up would mean deciding which module owns each of
those, and the honest answer is that they are shared.

The fields are grouped and commented by the stage that reads them, which is the
part of a config that rots first and the part a reader actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FinalReportMode(str, Enum):
    """What the main output file is allowed to contain.

    `LEGACY` keeps structural insertions in `final_calls`; `TE_CALIBRATED` (the
    default) moves them to `structural_calls` so the TE output is TE-only. The
    string VALUES are the C++ enum's spelling, because finalization compares
    against them.
    """

    LEGACY = "Legacy"
    TE_CALIBRATED = "TeCalibrated"


@dataclass
class BamRegionScope:
    """A 0-based half-open restriction. `end <= 0` means to the contig end."""

    enabled: bool = False
    chrom: str = ""
    start: int = 0
    end: int = -1


@dataclass
class PipelineConfig:
    # ---------------------------------------------------------------- inputs
    bam_path: str = ""
    reference_fasta_path: str = ""
    te_fasta_path: str = ""
    te_blastn_path: str = "blastn"
    te_makeblastdb_path: str = "makeblastdb"
    bam_region_scope: BamRegionScope = field(default_factory=BamRegionScope)

    bam_threads: int = 2
    progress_interval: int = 100000
    log_stage_bins: bool = False
    log_stage_components: bool = False

    #: Scan bin width. Also sets the split-SA flank search radius (bin_size/2),
    #: which is why it is not purely a performance knob.
    bin_size: int = 10000

    # ------------------------------------- module 2.1: fragment extraction
    #: Empty disables the FASTA dump. Recommended empty for a whole-genome run.
    ins_fragments_fasta_path: str = ""
    min_soft_clip_for_seq_extract: int = 50
    min_long_ins_for_seq_extract: int = 50
    min_sa_aln_len_for_seq_extract: int = 50
    max_sa_per_read: int = 3

    # -------------------------------------- module 2.2: TE classification
    ins_fragment_hits_tsv_path: str = ""
    te_kmer_size: int = 13
    te_kmer_sizes_csv: str = "9,11,13"
    te_low_kmer_support_trigger: float = 0.30
    te_low_kmer_rescue_enable: bool = True
    te_low_kmer_rescue_topn: int = 3
    te_low_kmer_rescue_min_frag_len: int = 40
    te_low_kmer_rescue_identity_min: float = 0.55
    te_low_kmer_rescue_margin_max: float = 0.08
    short_ins_enable: bool = True
    short_ins_min_len: int = 35
    short_ins_min_reads: int = 2
    te_family_margin_min: float = 0.05
    te_subfamily_margin_min: float = 0.04
    te_softclip_low_complexity_at_frac_min: float = 0.90
    te_softclip_low_complexity_homopolymer_min: int = 80
    te_softclip_entropy_min: float = 1.25
    te_softclip_kmer_uniqueness_min: float = 0.35
    te_softclip_min_anchor_len: int = 20
    te_softclip_max_nm_per_bp: float = 0.12

    # ---------------------------------------------- module 2.3: TSD detector
    tsd_enable: bool = True
    tsd_min_len: int = 3
    tsd_max_len: int = 50
    tsd_flank_window: int = 150
    tsd_bg_p_max: float = 0.05
    #: Exact flank equality is tried over the whole length range FIRST; the
    #: mismatch budget is a retry, not a relaxation of the primary search. Set
    #: either to 0 to require exact TSDs, as releases before 0.0.5 did.
    tsd_max_mismatch_rate: float = 0.10
    tsd_max_mismatches: int = 2

    # ------------------------------------------------------------ genotyping
    genotype_error_rate: float = 0.02
    #: The binomial-ish baseline. Finalization re-estimates it per sample from
    #: the run's own count distribution and re-runs the genotyper, so this is a
    #: starting value rather than an assertion about the data.
    genotype_overdispersion: float = 0.02

    # ------------------------------------------------------------- selection
    #: A statistical risk level, not an evidence weight. The distinction is the
    #: whole argument of the selection layer.
    final_fdr_q: float = 0.10
    min_final_raw_cigar_insert_len_bp: int = 50
    final_report_mode: FinalReportMode = FinalReportMode.TE_CALIBRATED

    # ------------------------------------------------------------- consensus
    event_consensus_poa_min_reads: int = 2
    event_consensus_poa_max_reads: int = 48

    # ------------------------------------------------------------ execution
    enable_parallel: bool = False
    parallel_workers: int = 0
    parallel_queue_max_tasks: int = 0  # <= 0 auto-sizes the bounded queue
    parallel_result_buffer_max: int = 0
    log_parallel_progress: bool = False
