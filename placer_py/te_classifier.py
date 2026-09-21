"""
Naming the insert: the k-mer shortlist, and the BLAST alignment that decides.

Ported from `src/component/te_quick_classifier.cpp`, pinned by
`tests/test_23_te_classifier.py`.

TWO CLASSIFIERS, AND THEY ANSWER DIFFERENT QUESTIONS. Confusing them is the
easiest way to misread a PLACER run:

  * `classify` is the QUICK pass, per fragment. Exact k-mer matching against a
    library index, no alignment, no gaps. Its `kmer_support` and `coverage` are
    documented in the C++ struct as "quick estimate [0,1], NOT sequence
    identity", and they are not comparable to the identity the decision policy
    reads. It exists to say "this fragment looks like it is worth aligning" and
    to put a name in the diagnostic hit table.
  * `build_insert_alignment_evidence_from_blast_hits` is the REAL classification,
    per event, from a gapped alignment of the assembled insert. Everything the
    decision policy reads -- identity, coverage, family, subfamily, the
    consensus interval -- comes from here.

WHY THE FAMILY IS NOT SIMPLY THE TOP HIT. A TE library is enormously redundant:
a thousand near-identical AluY copies, and one of them will win by float noise.
So hits are aggregated per FAMILY, each family is represented by its best copy,
and families are ranked by that. One noisy high-scoring copy of the wrong family
then cannot outrank a family whose best copy is genuinely better.

WHAT THE MARGINS BUY. `subfamily_margin_min` makes a near-tie ABSTAIN to
`FAMILY_ONLY` rather than naming an arbitrary subfamily. That is the same
evasion the conformal selector and the explanation comparator use -- when a
quantity has no objective basis, refuse to report it instead of inventing one --
and it is why `annotation_confidence` reads HIGH/MEDIUM/LOW rather than a
number: the tiers are naming SPECIFICITY, not evidence strength.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

from placer_py.config import PipelineConfig
from placer_py.fragments import InsertionFragment, InsertionFragmentSource
from placer_py.seqtools import (
    FNV1A_OFFSET_BASIS,
    TeNameParts,
    TeSequenceBackground,
    at_fraction,
    compute_te_sequence_composition,
    confidence_from_qc_reason,
    fnv1a_append_int32,
    fnv1a_append_string,
    for_each_valid_kmer,
    kmer_uniqueness_ratio,
    max_homopolymer_run,
    parse_kmer_sizes_csv,
    parse_te_name_parts,
    reverse_complement,
    shannon_entropy_acgt,
    take_header_token,
    te_kmer_containment,
    te_kmer_jsd_vs_background,
    upper_acgt,
)
from placer_py.structure import SequenceExplanation, explain_te_sequence_structure

#: Ambiguity marker in the k-mer index: this k-mer occurs in two or more
#: library entries, so it names no element.
KMER_AMBIGUOUS = -1
#: Not in the index at all.
KMER_ABSENT = -2
#: `k` used by the low-complexity uniqueness test. Fixed in the C++, not a knob.
LOW_COMPLEXITY_UNIQUENESS_K = 5


@dataclass
class TeEntry:
    """One library element, with its reverse complement precomputed.

    Both strands are indexed rather than canonicalising the key, because the
    quick classifier reports WHICH element matched and a canonical key loses the
    orientation that the `aligned_len_est` run measurement needs.
    """

    name: str = ""
    sequence: str = ""
    reverse_complement_sequence: str = ""


def load_te_entries_from_fasta(text: str) -> list[TeEntry]:
    """Parse a TE library FASTA from text.

    Takes text rather than a path so the library can come from anywhere and the
    parser can be tested without a filesystem. Entries with an empty sequence
    are dropped -- a header with no bases indexes nothing and would only
    occupy an id.
    """
    entries: list[TeEntry] = []
    name: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if name is None:
            return
        sequence = upper_acgt("".join(chunks))
        if sequence:
            entries.append(TeEntry(name=name, sequence=sequence,
                                   reverse_complement_sequence=reverse_complement(sequence)))

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            name = take_header_token(line[1:])
            chunks = []
            continue
        if name is not None:
            chunks.append(line.strip())
    flush()
    return entries


class KmerIndex:
    """Exact k-mer to element id, with ambiguity collapsed to a single marker.

    A k-mer shared by two elements maps to `KMER_AMBIGUOUS` and thereafter
    supports NEITHER. That is the important design choice: with a redundant
    library, a shared k-mer is evidence that the fragment is a TE but no
    evidence at all about which one, and crediting it to whichever element was
    inserted first would be an artefact of library order.
    """

    def __init__(self, kmer_size: int) -> None:
        self.k = kmer_size
        self.te_names: list[str] = []
        self.kmer_to_id: dict[int, int] = {}

    def build_from_entries(self, entries: list[TeEntry]) -> bool:
        self.te_names = []
        self.kmer_to_id = {}
        for te_id, entry in enumerate(entries):
            self.te_names.append(entry.name)
            self.add_sequence(te_id, entry.sequence)
            self.add_sequence(te_id, entry.reverse_complement_sequence)
        return bool(self.te_names) and bool(self.kmer_to_id)

    def add_sequence(self, te_id: int, seq: str) -> None:
        if len(seq) < self.k:
            return
        for _, key in for_each_valid_kmer(seq, self.k):
            existing = self.kmer_to_id.get(key)
            if existing is None:
                self.kmer_to_id[key] = te_id
            elif existing != te_id:
                self.kmer_to_id[key] = KMER_AMBIGUOUS

    def te_name(self, te_id: int) -> str:
        if 0 <= te_id < len(self.te_names):
            return self.te_names[te_id]
        return ""

    def lookup(self, key: int) -> int:
        return self.kmer_to_id.get(key, KMER_ABSENT)


def is_softclip_source(source: InsertionFragmentSource) -> bool:
    return source in (InsertionFragmentSource.CLIP_REF_LEFT,
                      InsertionFragmentSource.CLIP_REF_RIGHT)


def is_low_complexity_softclip(fragment: InsertionFragment, seq: str,
                              at_fraction_min: float, homopolymer_run_min: int,
                              entropy_min: float,
                              kmer_uniqueness_min: float) -> bool:
    """Four independent tests, ANY of which disqualifies -- for CLIPS only.

    The asymmetry is the point. A soft clip can be an adapter, a poly(A) tail
    read through, or a low-complexity tract that the aligner gave up on, and all
    three will match something in a TE library for the wrong reason. A CIGAR
    insertion is the aligner's own placement of extra bases at a specific
    position, so it is not subject to this veto even when its composition looks
    identical -- the evidence that it is real comes from the placement, not
    from the sequence.
    """
    if not is_softclip_source(fragment.source) or not seq:
        return False
    return (at_fraction(seq) >= at_fraction_min
            or max_homopolymer_run(seq) >= homopolymer_run_min
            or shannon_entropy_acgt(seq) < max(0.0, entropy_min)
            or kmer_uniqueness_ratio(seq, LOW_COMPLEXITY_UNIQUENESS_K)
            < min(1.0, max(0.0, kmer_uniqueness_min)))


@dataclass
class FragmentTEHit:
    """The quick pass's answer. NOT sequence identity -- see the module docstring."""

    fragment_id: str = ""
    te_name: str = ""
    fragment_len: int = 0
    aligned_len_est: int = 0
    kmer_support: float = 0.0   # quick estimate in [0,1]
    coverage: float = 0.0       # quick estimate in [0,1]
    multik_support: float = 0.0
    rescue_used: bool = False
    hit_kmers: int = 0
    total_kmers: int = 0


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


class TeKmerQuickClassifier:
    """`TEKmerQuickClassifierModule`, minus the thread plumbing.

    The C++ carries a mutex-guarded alignment cache and a process-wide shared
    library cache keyed by an FNV-1a hash of the build inputs. Both exist to
    stop N workers rebuilding the same index; neither changes what is computed,
    so the port keeps the cache KEY (which is observable -- it names files on
    disk) and drops the locking.
    """

    def __init__(self, config: PipelineConfig, entries: list[TeEntry]) -> None:
        self.config = config
        self.entries = entries
        self.te_names = [entry.name for entry in entries]
        self.te_sequences = [entry.sequence for entry in entries]
        self.indices: list[KmerIndex] = []
        self.primary_index: KmerIndex | None = None

        for k in parse_kmer_sizes_csv(config.te_kmer_sizes_csv, config.te_kmer_size):
            index = KmerIndex(k)
            if index.build_from_entries(entries):
                self.indices.append(index)
                if k == config.te_kmer_size:
                    self.primary_index = index
        if self.primary_index is None and self.indices:
            self.primary_index = self.indices[0]

    def is_enabled(self) -> bool:
        return bool(self.indices) and bool(self.te_names)

    def classify(self, fragments: list[InsertionFragment]) -> list[FragmentTEHit]:
        """One hit per fragment, in input order, including the vetoed ones.

        A fragment vetoed as a low-complexity clip still gets a row, with every
        field at its default. That is deliberate: the hit table is a record of
        what was examined, and dropping the vetoed rows would make "no hit" and
        "not looked at" indistinguishable downstream.
        """
        hits: list[FragmentTEHit] = []
        if not self.is_enabled():
            return hits

        cfg = self.config
        at_min = _clamp(cfg.te_softclip_low_complexity_at_frac_min, 0.0, 1.0)
        homopolymer_min = max(1, cfg.te_softclip_low_complexity_homopolymer_min)
        entropy_min = max(0.0, cfg.te_softclip_entropy_min)
        uniqueness_min = _clamp(cfg.te_softclip_kmer_uniqueness_min, 0.0, 1.0)
        rescue_topn = max(1, cfg.te_low_kmer_rescue_topn)
        rescue_min_frag_len = max(1, cfg.te_low_kmer_rescue_min_frag_len)
        rescue_identity_min = _clamp(cfg.te_low_kmer_rescue_identity_min, 0.0, 1.0)
        rescue_margin_max = _clamp(cfg.te_low_kmer_rescue_margin_max, 0.0, 1.0)
        support_gate = _clamp(cfg.te_low_kmer_support_trigger, 0.0, 1.0)

        for frag in fragments:
            hit = FragmentTEHit(fragment_id=frag.fragment_id,
                                fragment_len=len(frag.sequence))
            seq = upper_acgt(frag.sequence)
            if is_low_complexity_softclip(frag, seq, at_min, homopolymer_min,
                                          entropy_min, uniqueness_min):
                hits.append(hit)
                continue
            self._classify_one(hit, seq, rescue_topn, rescue_min_frag_len,
                               rescue_identity_min, rescue_margin_max, support_gate)
            hits.append(hit)
        return hits

    def _classify_one(self, hit: FragmentTEHit, seq: str, rescue_topn: int,
                      rescue_min_frag_len: int, rescue_identity_min: float,
                      rescue_margin_max: float, support_gate: float) -> None:
        weighted_support_sum: dict[int, float] = {}
        total_weight = 0.0
        primary_total = 0
        primary_counts: dict[int, int] = {}
        fallback_total = 0
        fallback_counts: dict[int, int] = {}

        for index in self.indices:
            k = index.k
            if len(seq) < k:
                continue
            counts: dict[int, int] = {}
            total = 0
            for _, key in for_each_valid_kmer(seq, k):
                total += 1
                te_id = index.lookup(key)
                if te_id >= 0:
                    counts[te_id] = counts.get(te_id, 0) + 1
            if total <= 0:
                continue

            # Longer k weighs more: a shared 13-mer is far stronger evidence of
            # common ancestry than a shared 9-mer, and weighting by k is the
            # cheapest monotone expression of that.
            weight = float(k)
            total_weight += weight
            for te_id, count in counts.items():
                weighted_support_sum[te_id] = (weighted_support_sum.get(te_id, 0.0)
                                               + weight * (count / total))
            if index is self.primary_index:
                primary_total = total
                primary_counts = counts
            if fallback_total <= 0:
                fallback_total = total
                fallback_counts = counts

        ranked: list[tuple[int, float]] = []
        if total_weight > 0.0:
            ranked = sorted(((te_id, value / total_weight)
                             for te_id, value in weighted_support_sum.items()),
                            key=lambda item: (-item[1], item[0]))

        best_id = -1
        best_score = 0.0
        second_score = 0.0
        if ranked:
            best_id, best_score = ranked[0]
            if len(ranked) >= 2:
                second_score = ranked[1][1]

        hit.multik_support = best_score
        hit.coverage = best_score
        hit.kmer_support = best_score
        hit.te_name = (parse_te_name_parts(self.te_names[best_id]).exact_name
                       if 0 <= best_id < len(self.te_names) else "")

        count_source = primary_counts if primary_total > 0 else fallback_counts
        hit.total_kmers = primary_total if primary_total > 0 else fallback_total
        if best_id >= 0 and best_id in count_source:
            hit.hit_kmers = count_source[best_id]

        hit.aligned_len_est = self._longest_run_length(seq, best_id)

        if (self.config.te_low_kmer_rescue_enable and len(seq) >= rescue_min_frag_len
                and ranked):
            self._maybe_rescue(hit, seq, ranked, best_score, second_score,
                               count_source, rescue_topn, rescue_identity_min,
                               rescue_margin_max, support_gate)

    def _longest_run_length(self, seq: str, best_id: int) -> int:
        """Bases spanned by the longest CONSECUTIVE run of winning k-mers.

        `k + max_run - 1`, on the longest index whose k the fragment can carry.
        The run has to be consecutive in START POSITION -- a gap in the k-mer
        stream (an N, or a k-mer that went ambiguous) breaks it. That is what
        separates "300 scattered matching k-mers" from "a contiguous 300 bp
        match", and only the second is an alignment length.
        """
        run_index: KmerIndex | None = None
        for index in self.indices:
            if len(seq) < index.k:
                continue
            if run_index is None or index.k > run_index.k:
                run_index = index
        if run_index is None or best_id < 0:
            return 0

        run = 0
        max_run = 0
        prev_start = -2
        for start, key in for_each_valid_kmer(seq, run_index.k):
            if start != prev_start + 1:
                run = 0
            if run_index.lookup(key) == best_id:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
            prev_start = start
        return (run_index.k + max_run - 1) if max_run > 0 else 0

    def _maybe_rescue(self, hit: FragmentTEHit, seq: str,
                      ranked: list[tuple[int, float]], best_score: float,
                      second_score: float, count_source: dict[int, int],
                      rescue_topn: int, rescue_identity_min: float,
                      rescue_margin_max: float, support_gate: float) -> None:
        """Fall back to alignment when the k-mer evidence is weak OR ambiguous.

        TWO triggers, for two different failures:

          * low support -- a diverged copy shares few exact k-mers with any
            library entry, which is precisely the old-insertion case exact
            matching is worst at;
          * a narrow margin -- two families are within `rescue_margin_max`, so
            the k-mer winner is a coin flip between them.

        The alignment is run against the top few candidates only, because it is
        O(n*m) and the k-mer pass has already ruled out everything else. The
        rescue can only RAISE `kmer_support` (`max`), never lower it: it is a
        second opinion admitted when the first is weak, not a replacement.
        """
        support_trigger = best_score < support_gate
        margin_trigger = len(ranked) >= 2 and (best_score - second_score) < rescue_margin_max
        if not (support_trigger or margin_trigger):
            return

        from placer_py.seqtools import semiglobal_edit_identity

        rescue_best_id = -1
        rescue_best_identity = 0.0
        for te_id, _ in ranked[:min(rescue_topn, len(ranked))]:
            if te_id < 0 or te_id >= len(self.te_sequences):
                continue
            identity = semiglobal_edit_identity(seq, self.te_sequences[te_id])
            if identity > rescue_best_identity or (identity == rescue_best_identity
                                                   and te_id < rescue_best_id):
                rescue_best_identity = identity
                rescue_best_id = te_id

        if rescue_best_id >= 0 and rescue_best_identity >= rescue_identity_min:
            hit.rescue_used = True
            hit.te_name = parse_te_name_parts(self.te_names[rescue_best_id]).exact_name
            hit.kmer_support = max(hit.kmer_support, rescue_best_identity)
            hit.coverage = hit.kmer_support
            if hit.total_kmers > 0 and rescue_best_id in count_source:
                hit.hit_kmers = count_source[rescue_best_id]


def fragment_hits_tsv(hits: list[FragmentTEHit]) -> str:
    """The diagnostic hit table, in the C++ column order."""
    rows = ["\t".join((
        hit.fragment_id, hit.te_name, str(hit.fragment_len),
        str(hit.hit_kmers), str(hit.total_kmers), repr(hit.coverage),
        str(hit.aligned_len_est), repr(hit.kmer_support),
        repr(hit.multik_support), "1" if hit.rescue_used else "0")) + "\n"
        for hit in hits]
    return "".join(rows)


# ---------------------------------------------------------------------------
# The BLAST half: the classification the decision policy actually reads.
# ---------------------------------------------------------------------------
#: Columns requested from blastn, in order. Changing this changes the parser.
BLAST_OUTFMT = "6 qseqid sseqid pident length qlen qstart qend sstart send bitscore evalue"
#: Reported subjects per query. A redundant library means the top 25 copies are
#: often the same family; the per-family aggregation below is what makes that
#: harmless.
BLAST_MAX_TARGET_SEQS = 25


@dataclass
class BlastHsp:
    """One high-scoring pair, in 0-based half-open coordinates."""

    query_id: str = ""
    subject_id: str = ""
    identity: float = 0.0
    alignment_length: int = 0
    query_length: int = 0
    query_start: int = -1
    query_end: int = -1
    target_start: int = -1
    target_end: int = -1
    bitscore: float = 0.0
    evalue: float = 1.0


@dataclass
class BlastSubjectHit:
    """Every HSP against one library entry, collapsed into one statement."""

    subject_id: str = ""
    name_parts: TeNameParts = field(default_factory=TeNameParts)
    identity: float = 0.0
    query_coverage: float = 0.0
    score: float = 0.0
    bitscore: float = 0.0
    best_evalue: float = 1.0
    query_start: int = -1
    query_end: int = -1
    target_start: int = -1
    target_end: int = -1


def parse_blast_hsp_line(line: str) -> BlastHsp | None:
    """Parse one `-outfmt 6` row, or None.

    THREE conversions worth naming, because each silently breaks a
    reimplementation that skips it:

      * pident is a PERCENTAGE; identity is a fraction.
      * BLAST coordinates are 1-based INCLUSIVE; the port's are 0-based half
        open, hence `min(a,b) - 1` and `max(a,b)`.
      * a minus-strand hit reports `send < sstart`. Taking min/max normalises
        the interval and DISCARDS the orientation -- which is correct here,
        because the consensus interval is used for 5'-truncation geometry and
        the strand is already carried by the fragment.
    """
    fields = line.split()
    if len(fields) < 11:
        return None
    out = BlastHsp(query_id=fields[0], subject_id=fields[1])
    try:
        out.identity = _clamp(float(fields[2]) / 100.0, 0.0, 1.0)
        out.alignment_length = max(0, int(fields[3]))
        out.query_length = max(0, int(fields[4]))
        qa, qb = int(fields[5]), int(fields[6])
        ta, tb = int(fields[7]), int(fields[8])
        out.query_start = min(qa, qb) - 1
        out.query_end = max(qa, qb)
        out.target_start = min(ta, tb) - 1
        out.target_end = max(ta, tb)
        out.bitscore = float(fields[9])
        out.evalue = float(fields[10])
    except ValueError:
        return None
    if not (out.query_length > 0 and out.alignment_length > 0
            and out.query_end > out.query_start):
        return None
    return out


def covered_fraction_from_intervals(length: int,
                                    intervals: list[tuple[int, int]]) -> float:
    """Union coverage, not the sum of the parts.

    Two HSPs against the same subject routinely OVERLAP -- a tandem repeat
    inside the element aligns twice. Summing their lengths would report coverage
    above 1 for an insert that is half explained, so the intervals are merged
    first. This is the single place where the difference between "aligned bases"
    and "explained bases" is enforced.
    """
    if length <= 0 or not intervals:
        return 0.0
    covered = 0
    current_start = -1
    current_end = -1
    for raw_start, raw_end in sorted(intervals):
        start = _clamp(raw_start, 0, length)
        end = _clamp(raw_end, 0, length)
        if end <= start:
            continue
        if current_start < 0:
            current_start, current_end = start, end
            continue
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered += current_end - current_start
        current_start, current_end = start, end
    if current_start >= 0:
        covered += current_end - current_start
    return _clamp(covered / length, 0.0, 1.0)


def collapse_blast_hsps(hsps: list[BlastHsp], query_len: int) -> list[BlastSubjectHit]:
    """Per-subject aggregation, and the ordering the family ranking depends on.

    Identity is weighted by alignment length -- a 500 bp HSP at 0.95 and a 20 bp
    HSP at 0.60 are not equally informative about the element -- while coverage
    is a union and bitscore is a plain sum, because bitscore is already additive
    over independent HSPs.

    The sort is by BEST E-VALUE first and bitscore second. e-value leads because
    it is the only field that answers "is this a hit at all"; everything after
    it is ranking among things that already are.
    """
    by_subject: dict[str, dict] = {}
    for hsp in hsps:
        if hsp.query_length > 0:
            query_len = max(query_len, hsp.query_length)
        acc = by_subject.get(hsp.subject_id)
        if acc is None:
            acc = {"subject_id": hsp.subject_id, "query_intervals": [],
                   "identity_weighted": 0.0, "aligned_bases": 0, "bitscore": 0.0,
                   "best_evalue": hsp.evalue, "query_start": -1, "query_end": -1,
                   "target_start": -1, "target_end": -1}
            by_subject[hsp.subject_id] = acc
        acc["query_intervals"].append((hsp.query_start, hsp.query_end))
        acc["identity_weighted"] += hsp.identity * hsp.alignment_length
        acc["aligned_bases"] += hsp.alignment_length
        acc["bitscore"] += hsp.bitscore
        acc["best_evalue"] = min(acc["best_evalue"], hsp.evalue)
        if acc["query_start"] < 0 or hsp.query_start < acc["query_start"]:
            acc["query_start"] = hsp.query_start
        acc["query_end"] = max(acc["query_end"], hsp.query_end)
        if acc["target_start"] < 0 or hsp.target_start < acc["target_start"]:
            acc["target_start"] = hsp.target_start
        acc["target_end"] = max(acc["target_end"], hsp.target_end)

    hits: list[BlastSubjectHit] = []
    for acc in by_subject.values():
        if acc["aligned_bases"] <= 0:
            continue
        hit = BlastSubjectHit(
            subject_id=acc["subject_id"],
            name_parts=parse_te_name_parts(acc["subject_id"]),
            identity=_clamp(acc["identity_weighted"] / acc["aligned_bases"], 0.0, 1.0),
            query_coverage=covered_fraction_from_intervals(query_len,
                                                           acc["query_intervals"]),
            bitscore=acc["bitscore"], best_evalue=acc["best_evalue"],
            query_start=acc["query_start"], query_end=acc["query_end"],
            target_start=acc["target_start"], target_end=acc["target_end"])
        #: The DISCRIMINATIVE score: a high-identity match over 5% of the insert
        #: and a mediocre one over all of it are both bad in different ways, and
        #: the product refuses to call either good.
        hit.score = hit.identity * hit.query_coverage
        hits.append(hit)

    hits.sort(key=lambda h: (h.best_evalue, -h.bitscore, -h.score, h.subject_id))
    return hits


@dataclass
class TEAlignmentEvidence:
    """Everything the decision policy knows about the insert's sequence."""

    best_family: str = ""
    best_subfamily: str = ""
    best_identity: float = 0.0
    best_query_coverage: float = 0.0
    best_score: float = 0.0
    #: Where the insert lands on the CONSENSUS. The start is the profile-depth
    #: signature of 5' truncation: a full-length insertion starts near 0, a
    #: 5'-truncated L1 starts thousands of bp in. -1 when unknown.
    te_consensus_start: int = -1
    te_consensus_end: int = -1
    coarse_prefilter_score: float = 0.0
    coarse_chain_coverage: float = 0.0
    second_family: str = "NA"
    second_score: float = 0.0
    cross_family_margin: float = 0.0
    sequence_model_label: str = "TE_MODEL_UNAVAILABLE"
    sequence_model_score: float = 0.0
    sequence_model_gc: float = 0.0
    sequence_model_entropy: float = 0.0
    sequence_model_tandem_fraction: float = 0.0
    sequence_model_low_complexity_fraction: float = 0.0
    sequence_model_jsd_k5: float = 0.0
    sequence_model_jsd_k6: float = 0.0
    sequence_model_k9_containment: float = 0.0
    annotation_confidence: str = "NA"
    annotation_class: str = "NA"
    annotation_order: str = "NA"
    annotation_intervals: str = "NA"
    annotation_residual_fraction: float = 0.0
    annotation_masked_fraction: float = 0.0
    te_sequence_explanation: SequenceExplanation = field(
        default_factory=SequenceExplanation)
    pass_: bool = False
    qc_reason: str = "NO_TE_ALIGNMENT"


def _finalize_evidence(evidence: TEAlignmentEvidence, insert_seq: str,
                       effective_query_coverage: float,
                       background: TeSequenceBackground | None) -> TEAlignmentEvidence:
    """Fill in the composition features and the structure decode, on EVERY path.

    Including the no-hit and library-unavailable paths, where they are the only
    sequence evidence there is. That is the point of a library-independent
    feature: an insert that matches nothing still has a GC content, an entropy
    and a tandem fraction, and those are exactly what distinguish "a TE we have
    no name for" from "a low-complexity artifact".
    """
    if insert_seq:
        composition = compute_te_sequence_composition(insert_seq)
        evidence.sequence_model_gc = composition.gc
        evidence.sequence_model_entropy = composition.entropy_bits
        evidence.sequence_model_tandem_fraction = composition.tandem_fraction
        evidence.sequence_model_low_complexity_fraction = composition.low_complexity_fraction
        if background is not None and background.valid:
            evidence.sequence_model_jsd_k5 = te_kmer_jsd_vs_background(
                insert_seq, 5, background.freq_k5)
            evidence.sequence_model_jsd_k6 = te_kmer_jsd_vs_background(
                insert_seq, 6, background.freq_k6)
            evidence.sequence_model_k9_containment = te_kmer_containment(
                insert_seq, 9, background.present_k9)
    evidence.te_sequence_explanation = explain_te_sequence_structure(
        insert_seq, evidence.qc_reason, evidence.best_family,
        evidence.best_subfamily, evidence.best_identity, effective_query_coverage,
        evidence.annotation_residual_fraction, evidence.annotation_masked_fraction,
        evidence.cross_family_margin, evidence.second_score,
        evidence.sequence_model_label, evidence.sequence_model_score)
    return evidence


def build_insert_alignment_evidence_from_blast_hits(
        insert_seq: str, has_blast_db: bool, hits: list[BlastSubjectHit],
        subfamily_margin_min: float,
        sequence_background: TeSequenceBackground | None = None
) -> TEAlignmentEvidence:
    """Turn collapsed BLAST hits into the evidence the decision policy reads.

    FOUR outcomes, and the `qc_reason` names which:

      * `TE_LIBRARY_UNAVAILABLE` / `EMPTY_INSERT_SEQUENCE` -- nothing was asked.
      * `NO_TE_ALIGNMENT_MATCH` -- asked, nothing found. This is the only path
        that sets a NEGATIVE sequence-model score (-0.50) and labels the insert
        a `TE_MODEL_OUTLIER`: failing to match a library of essentially every
        known human repeat is itself evidence, and the alternative -- scoring it
        0 -- would make "no TE" and "no information" the same number.
      * `PASS_..._UNKNOWN` / `..._FAMILY_ONLY` / `PASS_INSERT_TE_ALIGNMENT` --
        found, named to three different depths.

    Note that all three PASS variants set `pass = True`. The tier is about how
    SPECIFICALLY the element could be named, never about whether to believe the
    insertion is there -- that question belongs to the selection layer.
    """
    evidence = TEAlignmentEvidence()

    if not has_blast_db:
        evidence.qc_reason = "TE_LIBRARY_UNAVAILABLE"
        return _finalize_evidence(evidence, insert_seq,
                                  evidence.best_query_coverage, sequence_background)
    if not insert_seq:
        evidence.qc_reason = "EMPTY_INSERT_SEQUENCE"
        return _finalize_evidence(evidence, insert_seq,
                                  evidence.best_query_coverage, sequence_background)
    if not hits:
        evidence.qc_reason = "NO_TE_ALIGNMENT_MATCH"
        evidence.sequence_model_label = "TE_MODEL_OUTLIER"
        evidence.sequence_model_score = -0.50
        return _finalize_evidence(evidence, insert_seq,
                                  evidence.best_query_coverage, sequence_background)

    # Rank FAMILIES by their best copy, not copies by their score. A library
    # holds a thousand near-identical AluY entries and one of them wins by float
    # noise; taking the single top hit would let that noise choose the family.
    family_best: list[tuple[str, BlastSubjectHit]] = []
    family_slot: dict[str, int] = {}
    for hit in hits:
        slot = family_slot.get(hit.name_parts.family)
        if slot is None:
            family_slot[hit.name_parts.family] = len(family_best)
            family_best.append((hit.name_parts.family, hit))
        elif hit.score > family_best[slot][1].score:
            family_best[slot] = (family_best[slot][0], hit)
    family_best.sort(key=lambda item: (-item[1].score, item[1].best_evalue))

    best_family, best_hit = family_best[0]
    evidence.best_family = best_family
    if len(family_best) > 1:
        evidence.second_family = family_best[1][0]
        evidence.second_score = family_best[1][1].score
    else:
        evidence.second_family = "NA"
        evidence.second_score = 0.0

    evidence.best_subfamily = best_hit.name_parts.subfamily
    evidence.best_identity = best_hit.identity
    evidence.best_query_coverage = best_hit.query_coverage
    evidence.best_score = best_hit.score
    evidence.te_consensus_start = best_hit.target_start
    evidence.te_consensus_end = best_hit.target_end
    effective_query_coverage = evidence.best_query_coverage
    evidence.annotation_class = best_hit.name_parts.class_label
    evidence.annotation_order = best_hit.name_parts.order_label
    evidence.annotation_masked_fraction = 0.0
    evidence.annotation_residual_fraction = _clamp(1.0 - effective_query_coverage,
                                                   0.0, 1.0)
    if best_hit.query_start >= 0 and best_hit.query_end > best_hit.query_start:
        evidence.annotation_intervals = (
            f"q={best_hit.query_start}-{best_hit.query_end},"
            f"t={best_hit.target_start}-{best_hit.target_end},"
            f"id={best_hit.identity},cov={best_hit.query_coverage}")
    evidence.coarse_prefilter_score = best_hit.score
    evidence.coarse_chain_coverage = evidence.best_query_coverage
    evidence.cross_family_margin = max(0.0, evidence.best_score - evidence.second_score)
    evidence.sequence_model_label = "TE_MODEL_IN_DISTRIBUTION"
    evidence.sequence_model_score = evidence.best_score

    if not evidence.best_family or evidence.best_family == "NA":
        evidence.best_family = "UNKNOWN"
        evidence.best_subfamily = "UNKNOWN"
        evidence.pass_ = True
        evidence.qc_reason = "PASS_INSERT_TE_ALIGNMENT_UNKNOWN"
        evidence.annotation_confidence = confidence_from_qc_reason(evidence.qc_reason)
        return _finalize_evidence(evidence, insert_seq, effective_query_coverage,
                                  sequence_background)

    # Commit to a subfamily only when the winner beats every competing subfamily
    # of the SAME family by a real margin. A bare `>` would commit on float
    # noise; the margin makes a near-tie abstain to FAMILY_ONLY instead of
    # naming an arbitrary subfamily -- the same refusal-to-invent that the
    # conformal route and the explanation comparator are built on.
    subfamily_margin = max(0.0, subfamily_margin_min)
    subfamily_ambiguous = any(
        hit is not best_hit
        and hit.name_parts.family == evidence.best_family
        and hit.name_parts.subfamily != evidence.best_subfamily
        and (evidence.best_score - hit.score) <= subfamily_margin
        for hit in hits)
    if subfamily_ambiguous or not evidence.best_subfamily or evidence.best_subfamily == "NA":
        evidence.best_subfamily = ""
        evidence.pass_ = True
        evidence.qc_reason = "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY"
        evidence.annotation_confidence = confidence_from_qc_reason(evidence.qc_reason)
        return _finalize_evidence(evidence, insert_seq, effective_query_coverage,
                                  sequence_background)

    evidence.pass_ = True
    evidence.qc_reason = "PASS_INSERT_TE_ALIGNMENT"
    evidence.annotation_confidence = confidence_from_qc_reason(evidence.qc_reason)
    return _finalize_evidence(evidence, insert_seq, effective_query_coverage,
                              sequence_background)


# ---------------------------------------------------------------------------
# Driving the external aligner.
# ---------------------------------------------------------------------------
def build_te_library_cache_key(entries: list[TeEntry], ks: list[int],
                               requested_kmer_size: int) -> str:
    """FNV-1a over the library and the k choice, as lower-case hex.

    Observable, because it names the BLAST database files on disk. It must
    change when the library content changes, when the requested k changes, or
    when the k LIST changes -- otherwise a run would silently reuse an index
    built for a different configuration.
    """
    hash_value = FNV1A_OFFSET_BASIS
    hash_value = fnv1a_append_int32(hash_value, len(entries))
    for entry in entries:
        hash_value = fnv1a_append_string(hash_value, entry.name)
        hash_value = fnv1a_append_string(hash_value, entry.sequence)
    hash_value = fnv1a_append_int32(hash_value, requested_kmer_size)
    hash_value = fnv1a_append_int32(hash_value, len(ks))
    for k in ks:
        hash_value = fnv1a_append_int32(hash_value, k)
    return format(hash_value, "x")


def blast_work_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "placer_te_blast")
    os.makedirs(path, exist_ok=True)
    return path


def blast_db_files_exist(db_prefix: str) -> bool:
    """All three of `.nhr`, `.nin`, `.nsq`, each non-empty.

    A partially-written database from an interrupted `makeblastdb` would
    otherwise be reused, and blastn's failure mode on one is a confusing error
    rather than a rebuild.
    """
    for suffix in (".nhr", ".nin", ".nsq"):
        path = db_prefix + suffix
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            return False
    return True


def ensure_te_blast_db(te_fasta_path: str, makeblastdb_path: str,
                       cache_key: str) -> str:
    """Build the BLAST database if it is not already on disk.

    Returns an empty prefix for an empty FASTA path -- "no library configured"
    is a normal run, not an error -- but raises when a library IS configured and
    the database cannot be built, because silently proceeding would classify
    every insert as `TE_LIBRARY_UNAVAILABLE` and look like a clean negative run.
    """
    if not te_fasta_path:
        return ""
    if not makeblastdb_path:
        raise RuntimeError("BLAST+ makeblastdb path is empty")

    db_prefix = os.path.join(blast_work_dir(), f"te_library_{cache_key}")
    if blast_db_files_exist(db_prefix):
        return db_prefix

    result = subprocess.run(
        [makeblastdb_path, "-in", te_fasta_path, "-dbtype", "nucl", "-out", db_prefix],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if result.returncode != 0 or not blast_db_files_exist(db_prefix):
        raise RuntimeError(
            f"failed to build BLAST database for TE FASTA {te_fasta_path!r} "
            f"with makeblastdb {makeblastdb_path!r}")
    return db_prefix


def write_blast_batch_query_fasta(queries: list[tuple[str, str]],
                                  directory: str | None = None) -> str:
    directory = directory if directory is not None else blast_work_dir()
    handle, path = tempfile.mkstemp(prefix="insert_batch_query_", suffix=".fa",
                                    dir=directory)
    with os.fdopen(handle, "w") as out:
        for query_id, sequence in queries:
            out.write(f">{query_id}\n")
            for offset in range(0, len(sequence), 80):
                out.write(sequence[offset:offset + 80] + "\n")
    return path


def parse_blast_output(text: str, query_lengths: dict[str, int]
                       ) -> dict[str, list[BlastHsp]]:
    """Group parsed HSPs by query, dropping rows for queries never asked about.

    Unparsable rows are skipped rather than raised on: blastn writes warnings to
    stdout in some builds, and one warning line must not lose a whole batch.
    """
    hsps_by_query: dict[str, list[BlastHsp]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        hsp = parse_blast_hsp_line(line)
        if hsp is None or hsp.query_id not in query_lengths:
            continue
        hsps_by_query.setdefault(hsp.query_id, []).append(hsp)
    return hsps_by_query


def run_blastn_batch_against_te_library(blastn_path: str, blast_db_prefix: str,
                                        queries: list[tuple[str, str]]
                                        ) -> dict[str, list[BlastSubjectHit]]:
    """One blastn invocation for a whole batch of inserts.

    Batching is not only a speed choice: blastn's e-values depend on the
    database, not the query set, so batching does not change any reported
    number -- which is what makes it safe to do.

    EVERY query gets a key in the result, including ones with no hits. A missing
    key and an empty list would otherwise be indistinguishable, and "blastn was
    never asked" is a different condition from "blastn found nothing".
    """
    if not blastn_path:
        raise RuntimeError("BLAST+ blastn path is empty")
    if not blast_db_prefix:
        raise RuntimeError("BLAST database prefix is empty")

    out: dict[str, list[BlastSubjectHit]] = {}
    if not queries:
        return out

    query_lengths = {query_id: len(sequence) for query_id, sequence in queries}
    query_path = write_blast_batch_query_fasta(queries)
    output_path = os.path.splitext(query_path)[0] + ".blast.tsv"
    try:
        result = subprocess.run(
            [blastn_path, "-query", query_path, "-db", blast_db_prefix,
             "-task", "blastn", "-dust", "no", "-soft_masking", "false",
             "-max_target_seqs", str(BLAST_MAX_TARGET_SEQS),
             "-outfmt", BLAST_OUTFMT, "-out", output_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "blastn failed for batched insert consensus TE classification "
                f"with executable {blastn_path!r}")
        if not os.path.exists(output_path):
            raise RuntimeError(f"blastn did not create output file: {output_path}")
        with open(output_path) as handle:
            hsps_by_query = parse_blast_output(handle.read(), query_lengths)
    finally:
        for path in (query_path, output_path):
            with contextlib.suppress(OSError):
                os.remove(path)

    for query_id, _ in queries:
        out[query_id] = collapse_blast_hsps(hsps_by_query.get(query_id, []),
                                            query_lengths[query_id])
    return out


def align_insert_sequences(config: PipelineConfig, entries: list[TeEntry],
                           insert_seqs: list[str],
                           background: TeSequenceBackground | None = None
                           ) -> list[TEAlignmentEvidence]:
    """Classify a batch of assembled inserts, in input order.

    Deduplicated by SEQUENCE before the aligner is called, because a component
    with twenty supporting reads produces twenty near-identical consensus
    strings and the identical ones need aligning once. The C++ achieves the same
    thing with a mutex-guarded in-flight cache; here it is a dict, because there
    is nothing to serialise against.
    """
    if not entries or not config.te_fasta_path:
        return [build_insert_alignment_evidence_from_blast_hits(
            seq, False, [], config.te_subfamily_margin_min, background)
            for seq in insert_seqs]

    ks = parse_kmer_sizes_csv(config.te_kmer_sizes_csv, config.te_kmer_size)
    cache_key = build_te_library_cache_key(entries, ks, config.te_kmer_size)
    db_prefix = ensure_te_blast_db(config.te_fasta_path, config.te_makeblastdb_path,
                                   cache_key)

    unique: dict[str, str] = {}
    for seq in insert_seqs:
        if seq and seq not in unique:
            unique[seq] = f"q{len(unique)}"
    hits_by_id = run_blastn_batch_against_te_library(
        config.te_blastn_path, db_prefix,
        [(query_id, seq) for seq, query_id in unique.items()])

    return [build_insert_alignment_evidence_from_blast_hits(
        seq, True, hits_by_id.get(unique.get(seq, ""), []),
        config.te_subfamily_margin_min, background) for seq in insert_seqs]
