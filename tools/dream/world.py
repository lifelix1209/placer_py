"""A finished scan's output directory, loaded as a replay world.

A world is what one scan commit recorded on one dataset and region. Its rows
are the ledger's EVALUATED rows, with every column the ledger carries. Numeric
columns become numbers, so the rows can be handed to placer's own selection
code (`placer.core.mechanism_selection.select_loci_coverage`) exactly as
finalization hands it ledger rows.

A world goes stale when the scan changes what a column means. Record the scan
commit with each world (`worlds.json`) and replay only against worlds whose
columns mean what the policy assumes. A column an older scan did not write is
filled with its default and listed in `World.missing`, so a policy can tell
"absent" from "zero".
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Columns a policy may need that older scans did not write, with the value an
#: absent column stands for.
DEFAULTS: dict[str, object] = {
    "mech_aligned_len": 0,
    "mech_sequence_term": 0.0,
    "te_from_clip_sides": 0,
    # -1: not measured by this scan (a policy may fall back); 0 is a measurement.
    "te_union_covered_bp": -1,
    "te_union_coverage": -1.0,
    "te_dominant_family": "NA",
    "te_dominant_class": "NA",
    "te_dominant_covered_bp": 0,
    "insert_seq": "",
    # Scans before 1.0.0a4 did not record what the alt reads measured.
    "alt_measured_length_reads": -1,
    "alt_measured_lengths": "NA",
    # Scans before the replay observables of the TSD term and the allele-level
    # tally (EvidenceLedgerRow, after `conformal_qc`). -1 / NA: not recorded;
    # 0 extra carriers leaves a row's own tally, so a policy that reads them
    # replays an older world as the scan decided it.
    "mech_tsd_len": -1,
    "mech_tsd_p_present": -1.0,
    "mech_decoy_tsd_p_present": -1.0,
    "mech_decoy_tsd_hits": -1,
    "mech_decoy_sum_absent": 0.0,
    "mech_decoy_sum_present_per_p": 0.0,
    "allele_length": -1,
    "allele_carrier_offsets": "NA",
    "allele_carrier_lengths": "NA",
    "allele_carrier_similarity": "NA",
    "allele_carrier_own": "NA",
    "allele_bylen_alt_reads": -1,
    "allele_bylen_ref_reads": -1,
    "allele_bylen_span_lo": 0,
    "allele_bylen_span_hi": 0,
    "allele_bylen_extra_carriers": 0,
    "mech_counts_allele_bylen": 0.0,
    "allele_byseq_alt_reads": -1,
    "allele_byseq_ref_reads": -1,
    "allele_byseq_span_lo": 0,
    "allele_byseq_span_hi": 0,
    "allele_byseq_extra_carriers": 0,
    "mech_counts_allele_byseq": 0.0,
    "allele_wide_alt_reads": -1,
    "allele_wide_ref_reads": -1,
    "allele_wide_span_lo": 0,
    "allele_wide_span_hi": 0,
    "allele_wide_extra_carriers": 0,
    "mech_counts_allele_wide": 0.0,
    # Scans before frozen/rec2 did not measure the counts term's local null.
    "mech_counts_eps": -1.0,
    "counts_bg_own_reads": -1,
    "counts_bg_own_hits": -1,
    "allele_bylen_eps": -1.0,
    "allele_bylen_bg_reads": -1,
    "allele_bylen_bg_hits": -1,
    "allele_byseq_eps": -1.0,
    "allele_byseq_bg_reads": -1,
    "allele_byseq_bg_hits": -1,
    "allele_wide_eps": -1.0,
    "allele_wide_bg_reads": -1,
    "allele_wide_bg_hits": -1,
}

#: Columns kept as text even when they look numeric.
TEXT_COLUMNS = frozenset({
    "chrom", "family", "subfamily", "te_annotation_class", "te_annotation_order",
    "te_strand", "ltr_form", "te_dominant_family", "te_dominant_class",
    "candidate_retention_reason", "final_qc", "posterior_qc", "lfdr_qc",
    "mech_terms", "insert_seq", "te_structure_path", "alt_measured_lengths",
    "allele_carrier_offsets", "allele_carrier_lengths", "allele_carrier_similarity",
    "allele_carrier_own",
})

#: The registry of worlds, beside the scan outputs.
DEFAULT_REGISTRY = Path("/mnt/beegfs/scratch/miska/hl725/placer_dev/dream/worlds.json")


class Row:
    """One ledger row. Attributes are the ledger's columns."""

    def __init__(self, values: dict[str, object]) -> None:
        self.__dict__.update(values)

    def copy(self) -> Row:
        return Row(dict(self.__dict__))


def _value(column: str, text: str) -> object:
    if column in TEXT_COLUMNS:
        return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


@dataclass
class World:
    name: str
    path: Path
    dataset: str
    region: str
    scan_commit: str
    truth: str = ""
    confident: str = ""
    rows: list[Row] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: str = ""
    #: (repeatmasker.out, row id -> sequence id) once `annotate.py` has run.
    annotation: object = None
    registry: Path = DEFAULT_REGISTRY


def insert_length(values: dict[str, object]) -> int:
    """The insert's length: the recorded insert when the world has it, else
    the event consensus less its two flank alignments.

    `event_consensus_len` alone is NOT the insert: the consensus carries
    roughly 80 bp of reference flank on each side. On HG002 chr1 it put a
    truth 283 bp insertion at 430, and made 66-82 bp insertions look 230-265 bp
    long, which passes a 100 bp floor that the insertion itself fails.
    """
    seq = values.get("insert_seq") or ""
    if isinstance(seq, str) and seq:
        return len(seq)
    consensus = int(values.get("event_consensus_len") or 0)
    flanks = int(values.get("left_flank_align_len") or 0) + int(values.get("right_flank_align_len") or 0)
    return max(0, consensus - flanks)


def load_rows(ledger: Path) -> tuple[list[Row], list[str]]:
    csv.field_size_limit(sys.maxsize)   # insert_seq can be tens of kb
    rows: list[Row] = []
    tids: dict[str, int] = {}
    with open(ledger, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = sorted(set(DEFAULTS) - columns)
        for raw in reader:
            if raw.get("candidate_retention_reason") != "EVALUATED":
                continue
            values: dict[str, object] = {k: _value(k, v) for k, v in raw.items()}
            for column in missing:
                values[column] = DEFAULTS[column]
            values["tid"] = tids.setdefault(str(values["chrom"]), len(tids))
            values["insert_len"] = insert_length(values)
            values["_row_id"] = len(rows)
            values["_has_insert_seq"] = "insert_seq" not in missing
            rows.append(Row(values))
    return rows, missing


def registry(path: Path = DEFAULT_REGISTRY) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def load(name: str, registry_path: Path = DEFAULT_REGISTRY) -> World:
    """A registered world by name, with its rows."""
    entries = registry(registry_path)
    if name not in entries:
        raise KeyError(f"no world {name!r} in {registry_path}; known: {sorted(entries)}")
    entry = entries[name]
    path = Path(entry["path"])
    rows, missing = load_rows(path / "evidence_ledger.tsv")
    return World(name=name, path=path, dataset=entry["dataset"], region=entry["region"],
                 scan_commit=entry["scan_commit"], truth=entry.get("truth", ""),
                 confident=entry.get("confident", ""), rows=rows, missing=missing,
                 notes=entry.get("notes", ""), registry=registry_path)
