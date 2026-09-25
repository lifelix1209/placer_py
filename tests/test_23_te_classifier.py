"""
Naming the insert: the quick k-mer pass and the BLAST classification.

The two C++ cases are reproduced -- `test_te_quick_classifier_multik.cpp` in
the multi-k and rescue tests, `test_blast_te_alignment.cpp` in the family and
margin tests -- but WITHOUT the external BLAST process. The C++ has to install
a fake `blastn` shell script to test any of this; the port splits the parsing
and aggregation away from the subprocess, so the decisions can be pinned
directly and only the process plumbing needs an executable. That split is
now the file layout: the parsing is `placer/core/te_classifier.py` and the
process is `placer/io/blast.py`.
"""

from __future__ import annotations

import pytest
from conftest import call_or_skip, close

from placer.config import PipelineConfig
from placer.core import te_classifier as T
from placer.core.fragments import InsertionFragment, InsertionFragmentSource
from placer.core.seqtools import build_te_sequence_background
from placer.io import blast as B
from placer.io import te_library as L

pytestmark = pytest.mark.invariant

TEA = "ACGTTGCAACGTTGCAACGTTGCAACGTTGCAACGTTGCA"
TEB = "TTGGAACCTTGGAACCTTGGAACCTTGGAACCTTGGAACC"
LIBRARY = f">TEA\n{TEA}\n>TEB\n{TEB}\n"


def frag(sequence, fragment_id="f0",
         source=InsertionFragmentSource.CIGAR_INSERTION):
    return InsertionFragment(fragment_id=fragment_id, sequence=sequence,
                             source=source, length=len(sequence))


def insert(n: int) -> str:
    """An insert of `n` bases with no tandem structure.

    NOT `"ACGT" * k`: that is a period-4 tandem repeat, and the classifier now
    refuses to name an element from alignments to simple-repeat bases (see
    `SIMPLE_REPEAT_HIT_FRACTION`). A test about family ranking needs an insert
    that could actually be an element.
    """
    import random

    rng = random.Random(n)
    return "".join(rng.choice("ACGT") for _ in range(n))


def hsp(subject, pident=98.5, length=300, qlen=310, qstart=1, qend=300,
        sstart=1, send=300, bitscore=500.0, evalue=1e-90):
    return T.parse_blast_hsp_line(
        f"q0\t{subject}\t{pident}\t{length}\t{qlen}\t{qstart}\t{qend}\t"
        f"{sstart}\t{send}\t{bitscore}\t{evalue}")


# ---------------------------------------------------------------- the library
def test_library_entries_index_both_strands():
    """
    An insertion is reported on the reference strand but the element may be in
    either orientation. Both strands are indexed rather than canonicalising the
    key, because the quick pass reports WHICH element matched and the
    `aligned_len_est` run measurement needs the orientation preserved.
    """
    entries = call_or_skip(T.load_te_entries_from_fasta, LIBRARY)
    assert [e.name for e in entries] == ["TEA", "TEB"]
    assert entries[0].reverse_complement_sequence == "TGCAACGTTGCAACGTTGCAACGTTGCAACGTTGCAACGT"


def test_a_header_with_no_bases_is_dropped():
    assert T.load_te_entries_from_fasta(">EMPTY\n>TEA\n" + TEA + "\n")[0].name == "TEA"
    assert len(T.load_te_entries_from_fasta(">EMPTY\n>TEA\n" + TEA + "\n")) == 1


def test_a_kmer_shared_by_two_elements_supports_neither():
    """
    THE design choice of the index. With a redundant library a shared k-mer is
    evidence that the fragment is a TE, and no evidence at all about which one.
    Crediting it to whichever element was loaded first would make the family
    call depend on library ORDER.
    """
    shared = "ACGTACGTACGTACGT"
    entries = T.load_te_entries_from_fasta(f">X\n{shared}\n>Y\n{shared}\n")
    index = T.KmerIndex(9)
    assert call_or_skip(index.build_from_entries, entries)
    assert all(value == T.KMER_AMBIGUOUS for value in index.kmer_to_id.values())


def test_a_missing_kmer_is_distinguishable_from_an_ambiguous_one():
    index = T.KmerIndex(9)
    index.build_from_entries(T.load_te_entries_from_fasta(LIBRARY))
    assert index.lookup(0xDEADBEEF) == T.KMER_ABSENT
    assert T.KMER_ABSENT != T.KMER_AMBIGUOUS


# ------------------------------------------------------------- the quick pass
def _classifier(**overrides):
    config = PipelineConfig(te_fasta_path="dummy", **overrides)
    return T.TeKmerQuickClassifier(config, T.load_te_entries_from_fasta(LIBRARY))


def test_a_fragment_too_short_for_the_primary_k_still_matches_a_shorter_one():
    """
    The C++ multi-k case: a 10 bp fragment cannot carry a 13-mer at all, and
    without the 9-mer index it would report nothing. Multi-k exists so that
    fragment length does not silently decide whether a fragment is classifiable.
    """
    classifier = _classifier(te_kmer_size=13, te_kmer_sizes_csv="9,13",
                             te_low_kmer_rescue_enable=False)
    assert call_or_skip(classifier.is_enabled)
    hits = classifier.classify([frag("ACGTTGCAAC", "frag_multik")])
    assert len(hits) == 1
    assert hits[0].te_name == "TEA"
    assert hits[0].multik_support > 0.0
    assert hits[0].kmer_support > 0.0


def test_longer_k_carries_more_weight_than_shorter_k():
    """
    A shared 13-mer is far stronger evidence of common ancestry than a shared
    9-mer, and weighting each index by its own k is the cheapest monotone
    expression of that. Pinned by construction rather than by a number.
    """
    classifier = _classifier(te_kmer_size=13, te_kmer_sizes_csv="9,13")
    assert [index.k for index in classifier.indices] == [9, 13]
    assert classifier.primary_index.k == 13


def test_the_rescue_fires_on_low_support_and_can_only_raise_it():
    """
    The C++ rescue case: a 30 bp fragment one mismatch from TEA shares few
    exact 13-mers with it -- precisely the diverged-copy failure that exact
    k-mer matching is worst at. The alignment is a SECOND OPINION admitted when
    the first is weak, so it takes `max` and can never lower the support.
    """
    classifier = _classifier(te_kmer_size=13, te_kmer_sizes_csv="13",
                             te_low_kmer_support_trigger=0.95,
                             te_low_kmer_rescue_enable=True,
                             te_low_kmer_rescue_topn=2,
                             te_low_kmer_rescue_min_frag_len=20,
                             te_low_kmer_rescue_identity_min=0.80,
                             te_low_kmer_rescue_margin_max=0.20)
    hits = classifier.classify([frag("ACGTTGCAACGTTACAACGTTGCAACGTTG", "frag_rescue")])
    assert len(hits) == 1
    assert hits[0].te_name == "TEA"
    assert hits[0].rescue_used
    assert hits[0].kmer_support >= hits[0].multik_support


def test_the_rescue_also_fires_on_a_narrow_margin_between_two_families():
    """
    Two triggers for two different failures. Low support means "nothing matches
    well"; a narrow margin means "two things match equally", and the k-mer
    winner is then a coin flip. Only the second is visible when support is high.
    """
    classifier = _classifier(te_kmer_size=9, te_kmer_sizes_csv="9",
                             te_low_kmer_support_trigger=0.0,
                             te_low_kmer_rescue_margin_max=1.0,
                             te_low_kmer_rescue_min_frag_len=10,
                             te_low_kmer_rescue_identity_min=0.0)
    hits = classifier.classify([frag(TEA[:20] + TEB[:20], "chimera")])
    assert hits[0].rescue_used


def test_the_rescue_can_be_disabled_and_then_never_fires():
    classifier = _classifier(te_kmer_size=13, te_kmer_sizes_csv="13",
                             te_low_kmer_support_trigger=0.95,
                             te_low_kmer_rescue_enable=False)
    hits = classifier.classify([frag("ACGTTGCAACGTTACAACGTTGCAACGTTG")])
    assert not hits[0].rescue_used


def test_a_fragment_shorter_than_the_rescue_floor_is_not_rescued():
    classifier = _classifier(te_kmer_size=9, te_kmer_sizes_csv="9",
                             te_low_kmer_support_trigger=1.0,
                             te_low_kmer_rescue_min_frag_len=1000)
    assert not classifier.classify([frag("ACGTTGCAACGTT")])[0].rescue_used


def test_aligned_length_needs_a_consecutive_run_not_scattered_matches():
    """
    `k + max_run - 1`, where the run must be consecutive in START POSITION. A
    gap in the k-mer stream breaks it. That is what separates "300 scattered
    matching k-mers" from "a contiguous 300 bp match" -- and only the second is
    an alignment length.
    """
    classifier = _classifier(te_kmer_size=9, te_kmer_sizes_csv="9",
                             te_low_kmer_rescue_enable=False)
    contiguous = classifier.classify([frag(TEA[:30])])[0]
    assert contiguous.aligned_len_est == 30

    broken = classifier.classify([frag(TEA[:15] + "N" + TEA[15:30])])[0]
    assert broken.aligned_len_est < 30


def test_a_low_complexity_clip_is_vetoed_but_an_insertion_with_the_same_bases_is_not():
    """
    THE asymmetry of this stage. A soft clip can be an adapter, a read-through
    poly(A), or a tract the aligner gave up on, and all three match a TE library
    for the wrong reason. A CIGAR insertion is the aligner's own placement of
    extra bases at a specific position: its evidence comes from the PLACEMENT,
    not from the composition, so it is not subject to this veto.
    """
    classifier = _classifier(te_kmer_size=9, te_kmer_sizes_csv="9")
    poly_a = "A" * 100
    clip = classifier.classify([frag(poly_a, "clip",
                                     InsertionFragmentSource.CLIP_REF_LEFT)])[0]
    assert clip.te_name == "" and clip.kmer_support == 0.0

    insertion = frag(poly_a, "ins", InsertionFragmentSource.CIGAR_INSERTION)
    assert not call_or_skip(T.is_low_complexity_softclip, insertion, poly_a,
                            0.90, 80, 1.25, 0.35)


def test_a_vetoed_fragment_still_gets_a_row():
    """
    Dropping it would make "no hit" and "not looked at" indistinguishable
    downstream. The hit table is a record of what was EXAMINED.
    """
    classifier = _classifier(te_kmer_size=9, te_kmer_sizes_csv="9")
    hits = classifier.classify([
        frag("A" * 100, "vetoed", InsertionFragmentSource.CLIP_REF_LEFT),
        frag(TEA, "kept")])
    assert [h.fragment_id for h in hits] == ["vetoed", "kept"]


def test_any_one_of_the_four_low_complexity_tests_is_enough():
    """
    Four ORs, and each is shown firing with the other three disabled -- the
    thresholds are set past the sequence's value so only one can be responsible.
    They overlap heavily in practice, which is why the veto is a disjunction
    rather than a score: any one of them is sufficient reason not to trust a
    clip's library match.
    """
    clip = frag("", "c", InsertionFragmentSource.CLIP_REF_LEFT)
    assert T.is_low_complexity_softclip(clip, "ATATATATATATATATATAT", 0.90, 999, 0.0, 0.0)
    assert T.is_low_complexity_softclip(clip, "GGGGGGGGGG" + "ACGTACGTAC", 1.1, 10, 0.0, 0.0)
    assert T.is_low_complexity_softclip(clip, "GC" * 20, 1.1, 999, 1.25, 0.0)
    assert T.is_low_complexity_softclip(clip, "ACGTG" * 20, 1.1, 999, 0.0, 0.9)


def test_a_clean_complex_clip_passes_all_four():
    clip = frag("", "c", InsertionFragmentSource.CLIP_REF_LEFT)
    unique = "ACGTGACTTGCAAGTCCATGGATCCAGTTACGGCATTAGCCATGGACTTAGCAATTGCCA"
    assert not T.is_low_complexity_softclip(clip, unique, 0.90, 80, 1.25, 0.35)


def test_the_library_entry_itself_would_be_vetoed_as_a_clip():
    """
    Worth pinning because it is counter-intuitive: TEA is a tandem repeat of
    ACGTTGCA, so its 5-mer uniqueness is 0.22 and a CLIP carrying it exactly
    would be refused. That is the intended trade -- a clip matching a
    low-complexity library entry is the commonest false TE call there is -- and
    it is another reason the veto does not apply to CIGAR insertions.
    """
    clip = frag("", "c", InsertionFragmentSource.CLIP_REF_LEFT)
    assert T.is_low_complexity_softclip(clip, TEA, 0.90, 80, 1.25, 0.35)


def test_an_empty_library_disables_the_classifier_rather_than_erroring():
    classifier = T.TeKmerQuickClassifier(PipelineConfig(), [])
    assert not classifier.is_enabled()
    assert classifier.classify([frag(TEA)]) == []


# ------------------------------------------------------------- BLAST parsing
def test_blast_coordinates_are_converted_to_zero_based_half_open():
    parsed = call_or_skip(T.parse_blast_hsp_line,
                          "q0\tAluY#SINE/Alu\t98.5\t300\t310\t1\t300\t1\t300\t500.0\t1e-90")
    assert (parsed.query_start, parsed.query_end) == (0, 300)
    close(parsed.identity, 0.985, "pident is a percentage")


def test_a_minus_strand_hit_is_normalised_and_its_orientation_discarded():
    """
    blastn reports `send < sstart` on the minus strand. Taking min/max
    normalises the interval and drops the orientation, which is correct here:
    the consensus interval is used for 5'-truncation geometry, and the strand is
    already carried by the fragment.
    """
    parsed = T.parse_blast_hsp_line(
        "q0\tL1HS#LINE/L1\t95.0\t500\t600\t100\t599\t6000\t5501\t800.0\t0.0")
    assert (parsed.target_start, parsed.target_end) == (5500, 6000)
    assert (parsed.query_start, parsed.query_end) == (99, 599)


def test_a_malformed_or_degenerate_blast_row_is_dropped():
    assert T.parse_blast_hsp_line("") is None
    assert T.parse_blast_hsp_line("too\tfew\tfields") is None
    assert T.parse_blast_hsp_line("q0\ts\tNaNpct\t1\t1\t1\t1\t1\t1\t1\t1") is None
    # Zero query length: nothing can be a coverage fraction of it.
    assert T.parse_blast_hsp_line("q0\ts\t98\t300\t0\t1\t300\t1\t300\t500\t1e-9") is None


def test_warning_lines_in_blast_output_do_not_lose_the_batch():
    text = ("Warning: something happened\n"
            "q0\tAluY#SINE/Alu\t98.5\t300\t310\t1\t300\t1\t300\t500.0\t1e-90\n")
    grouped = call_or_skip(T.parse_blast_output, text, {"q0": 310})
    assert len(grouped["q0"]) == 1


def test_rows_for_a_query_that_was_never_asked_about_are_ignored():
    text = "other\tAluY#SINE/Alu\t98.5\t300\t310\t1\t300\t1\t300\t500.0\t1e-90\n"
    assert T.parse_blast_output(text, {"q0": 310}) == {}


# ------------------------------------------------------------ HSP collapsing
def test_coverage_is_a_union_and_never_exceeds_one():
    """
    Two HSPs against the same subject routinely OVERLAP -- a tandem repeat
    inside the element aligns twice. Summing their lengths would report an
    insert as more than fully explained. This is the one place where "aligned
    bases" and "explained bases" are kept apart.
    """
    close(call_or_skip(T.covered_fraction_from_intervals, 100, [(0, 60), (40, 100)]),
          1.0, "overlapping")
    close(T.covered_fraction_from_intervals(100, [(0, 40), (60, 100)]), 0.8, "disjoint")
    close(T.covered_fraction_from_intervals(100, []), 0.0, "no intervals")
    close(T.covered_fraction_from_intervals(0, [(0, 60)]), 0.0, "no length")
    close(T.covered_fraction_from_intervals(100, [(-50, 200)]), 1.0, "clamped")


def test_identity_is_weighted_by_alignment_length():
    """A 500 bp HSP at 0.95 and a 20 bp HSP at 0.60 are not equally informative
    about the element, and a plain mean would treat them as though they were."""
    hsps = [hsp("X#Fam/Sub", pident=95.0, length=500, qstart=1, qend=500,
                sstart=1, send=500),
            hsp("X#Fam/Sub", pident=60.0, length=20, qstart=501, qend=520,
                sstart=501, send=520)]
    collapsed = call_or_skip(T.collapse_blast_hsps, hsps, 600)
    close(collapsed[0].identity, (0.95 * 500 + 0.60 * 20) / 520, "weighted identity")


def test_the_score_is_discriminative_in_both_directions():
    """
    `identity x coverage`. A perfect match over 5% of the insert and a mediocre
    one over all of it are both bad, in different ways, and the product refuses
    to call either good.
    """
    narrow = T.collapse_blast_hsps([hsp("X#F/S", pident=100.0, length=15,
                                        qstart=1, qend=15, qlen=300)], 300)[0]
    assert narrow.score < 0.1


def test_hits_rank_by_evalue_first_because_that_is_the_is_it_a_hit_field():
    hits = T.collapse_blast_hsps(
        [hsp("A#F/S1", evalue=1e-90, bitscore=100.0),
         hsp("B#F/S2", evalue=1e-5, bitscore=900.0)], 310)
    assert [h.subject_id for h in hits] == ["A#F/S1", "B#F/S2"]


def test_subject_id_breaks_a_total_tie_so_the_order_is_deterministic():
    hits = T.collapse_blast_hsps([hsp("Z#F/S"), hsp("A#F/S")], 310)
    assert [h.subject_id for h in hits] == ["A#F/S", "Z#F/S"]


# ----------------------------------------------------------- family assignment
def test_the_family_is_chosen_by_its_best_copy_not_by_the_single_top_hit():
    """
    A library holds a thousand near-identical AluY entries and one of them wins
    by float noise. Ranking FAMILIES by their best copy means a single noisy
    high-scoring copy of the wrong family cannot outrank a family whose best
    copy is genuinely better.
    """
    hits = T.collapse_blast_hsps([
        hsp("L1HS#LINE/L1", pident=99.0, length=300, qstart=1, qend=300),
        # 60 bp, above the 50 informative bases a hit needs to name anything.
        hsp("AluY#SINE/Alu", pident=99.5, length=60, qstart=1, qend=60),
        hsp("AluSx#SINE/Alu", pident=99.4, length=60, qstart=1, qend=60),
    ], 310)
    evidence = call_or_skip(T.build_insert_alignment_evidence_from_blast_hits,
                            insert(320), True, hits, 0.04)
    assert evidence.best_family == "L1"
    assert evidence.second_family == "Alu"
    assert evidence.cross_family_margin > 0.0


def test_a_near_tie_between_subfamilies_abstains_to_family_only():
    """
    The same refusal-to-invent the conformal route and the explanation
    comparator are built on: when a quantity has no objective basis, do not
    report it. A bare `>` would commit on float noise.
    """
    hits = T.collapse_blast_hsps([
        hsp("AluYa5#SINE/Alu", pident=98.0, length=300, qstart=1, qend=300),
        hsp("AluYb8#SINE/Alu", pident=97.9, length=300, qstart=1, qend=300),
    ], 310)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(
        insert(320), True, hits, 0.04)
    assert evidence.qc_reason == "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY"
    assert evidence.best_family == "Alu"
    assert evidence.best_subfamily == ""
    assert evidence.annotation_confidence == "MEDIUM"


def test_a_clear_subfamily_winner_is_named():
    hits = T.collapse_blast_hsps([
        hsp("AluYa5#SINE/Alu", pident=99.0, length=300, qstart=1, qend=300),
        hsp("AluSx#SINE/Alu", pident=70.0, length=100, qstart=1, qend=100),
    ], 310)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(
        insert(320), True, hits, 0.04)
    assert evidence.qc_reason == "PASS_INSERT_TE_ALIGNMENT"
    assert evidence.best_subfamily == "AluYa5"
    assert evidence.annotation_confidence == "HIGH"
    assert evidence.pass_


def test_every_pass_tier_sets_pass_because_the_tier_is_about_naming():
    """
    HIGH/MEDIUM/LOW say how SPECIFICALLY the element could be named, never
    whether to believe the insertion is there. That second question belongs to
    the selection layer, and conflating them is how a tier threshold comes to
    act as a length filter.
    """
    for pident, expect in ((99.0, "HIGH"), (70.0, "HIGH")):
        hits = T.collapse_blast_hsps([hsp("AluYa5#SINE/Alu", pident=pident,
                                          length=300, qstart=1, qend=300)], 310)
        evidence = T.build_insert_alignment_evidence_from_blast_hits(
            insert(320), True, hits, 0.04)
        assert evidence.pass_
        assert evidence.annotation_confidence == expect


def test_no_match_is_scored_negatively_rather_than_zero():
    """
    Failing to match a library of essentially every known human repeat is
    ITSELF evidence. Scoring it 0 would make "no TE" and "no information" the
    same number, and the artifact side of the comparison would lose its
    strongest input.
    """
    evidence = T.build_insert_alignment_evidence_from_blast_hits(
        "ACGT" * 80, True, [], 0.04)
    assert evidence.qc_reason == "NO_TE_ALIGNMENT_MATCH"
    assert evidence.sequence_model_label == "TE_MODEL_OUTLIER"
    close(evidence.sequence_model_score, -0.50, "outlier score")
    assert not evidence.pass_


def test_composition_features_are_filled_in_even_with_no_library():
    """
    On EVERY exit path, because they are library-INDEPENDENT and are the only
    sequence evidence on the no-hit path. An insert that matches nothing still
    has a GC content and a tandem fraction, and those are what distinguish "a
    TE we have no name for" from "a low-complexity artifact".
    """
    evidence = T.build_insert_alignment_evidence_from_blast_hits(
        "ACGT" * 80, False, [], 0.04)
    assert evidence.qc_reason == "TE_LIBRARY_UNAVAILABLE"
    close(evidence.sequence_model_gc, 0.5, "gc")
    close(evidence.sequence_model_entropy, 2.0, "entropy")
    assert evidence.te_sequence_explanation is not None


def test_the_background_features_are_filled_in_only_when_a_background_exists():
    background = build_te_sequence_background([TEA * 10, TEB * 10])
    evidence = T.build_insert_alignment_evidence_from_blast_hits(
        TEA * 4, True, [], 0.04, background)
    assert evidence.sequence_model_k9_containment > 0.0
    without = T.build_insert_alignment_evidence_from_blast_hits(TEA * 4, True, [], 0.04)
    close(without.sequence_model_k9_containment, 0.0, "no background")


def test_an_empty_insert_is_reported_as_such_rather_than_as_no_match():
    evidence = T.build_insert_alignment_evidence_from_blast_hits("", True, [], 0.04)
    assert evidence.qc_reason == "EMPTY_INSERT_SEQUENCE"


def test_the_consensus_interval_is_carried_through_for_truncation_geometry():
    """
    `te_consensus_start` is the profile-depth signature of 5' truncation: a
    full-length insertion starts near 0, a 5'-truncated L1 starts thousands of
    bp in. It is one of the two integers the README says the ledger needs and
    does not carry.
    """
    hits = T.collapse_blast_hsps([hsp("L1HS#LINE/L1", pident=99.0, length=1000,
                                      qlen=1000, qstart=1, qend=1000,
                                      sstart=5001, send=6000)], 1000)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(
        insert(1000), True, hits, 0.04)
    assert (evidence.te_consensus_start, evidence.te_consensus_end) == (5000, 6000)
    assert "q=0-1000" in evidence.annotation_intervals


# ------------------------------------------------------------------ cache key
def test_the_cache_key_changes_with_the_library_and_with_the_k_choice():
    """
    Observable: it names the BLAST database files on disk. A key that did not
    move when k moved would let a run silently reuse an index built for a
    different configuration.
    """
    entries = T.load_te_entries_from_fasta(LIBRARY)
    base = call_or_skip(T.build_te_library_cache_key, entries, [9, 13], 13)
    assert base == T.build_te_library_cache_key(entries, [9, 13], 13)
    assert base != T.build_te_library_cache_key(entries, [9, 11, 13], 13)
    assert base != T.build_te_library_cache_key(entries, [9, 13], 11)
    assert base != T.build_te_library_cache_key(entries[:1], [9, 13], 13)


def test_an_incomplete_blast_database_is_not_reused():
    """
    All three files, each non-empty. A partially-written database from an
    interrupted makeblastdb would otherwise be reused, and blastn's failure mode
    on one is a confusing error rather than a rebuild.
    """
    assert not B.blast_db_files_exist("/nonexistent/prefix")


def test_no_configured_library_is_a_normal_run_and_not_an_error():
    assert call_or_skip(B.ensure_te_blast_db, "", "makeblastdb", "key") == ""


def test_alignment_without_a_library_reports_unavailable_for_every_insert():
    evidences = call_or_skip(L.align_insert_sequences, PipelineConfig(), [],
                             ["ACGT" * 80, ""])
    assert [e.qc_reason for e in evidences] == ["TE_LIBRARY_UNAVAILABLE",
                                                "TE_LIBRARY_UNAVAILABLE"]


# ----------------------------------------------------------- simple repeats
def test_a_hit_on_a_simple_repeat_names_no_element():
    """
    Many consensus sequences contain an (AT)n or (AAAG)n stretch, so a
    microsatellite insert aligns to them -- and the alignment says nothing
    about which element, if any, was inserted. Measured on HG002 chr21: (AT)n
    and (AAAG)n expansions were being called as L1, LTR66 and MER52-int.
    """
    at_repeat = "AT" * 56
    hits = T.collapse_blast_hsps([hsp("L1ME4b_3end#LINE/L1", pident=93.0, length=112,
                                      qlen=112, qstart=1, qend=112,
                                      sstart=689, send=800)], 112)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(at_repeat, True, hits, 0.04)
    assert evidence.qc_reason == "TE_ALIGNMENT_UNINFORMATIVE"
    assert not evidence.pass_
    assert evidence.best_family == ""


def test_a_real_element_hit_survives_a_simple_repeat_hit_beside_it():
    """The veto is per hit: an element that aligns through its own sequence
    still wins when the same insert also matches a repeat elsewhere."""
    seq = insert(300) + "AT" * 30
    hits = T.collapse_blast_hsps([
        hsp("AluY#SINE/Alu", pident=98.0, length=300, qlen=360, qstart=1, qend=300),
        hsp("L1ME4b_3end#LINE/L1", pident=99.0, length=60, qlen=360, qstart=301,
            qend=360, sstart=689, send=748, bitscore=900.0),
    ], 360)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(seq, True, hits, 0.04)
    assert evidence.pass_
    assert evidence.best_family == "Alu"


def test_a_short_hit_names_no_element():
    """Below 50 informative aligned bases a match cannot tell an element from
    chance. Measured on HG002 chr21: a 16 bp hit inside a 36 bp insert, and an
    18 bp one inside 72 bp, were calls."""
    seq = insert(72)
    hits = T.collapse_blast_hsps([hsp("Arthur1#DNA/hAT", pident=94.4, length=18,
                                      qlen=72, qstart=49, qend=66,
                                      sstart=3136, send=3153)], 72)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(seq, True, hits, 0.04)
    assert evidence.qc_reason == "TE_ALIGNMENT_UNINFORMATIVE"
    assert not evidence.pass_


def test_an_impure_repeat_array_is_still_a_simple_repeat():
    """(AAAG)n with a GAAAG or GGAAAG every few units has no exact run long
    enough for the microsatellite test; the low-complexity window catches it."""
    from placer.core.seqtools import simple_repeat_mask

    impure = "AGAAAGGAAAGAATGGAAAGAAAGGAAGAGAAAGGAAAGAAAGGAAGAAAAAGAAAGAAAGAAAGAAAGAAAG"
    assert all(simple_repeat_mask(impure))


def test_collinear_pieces_of_one_copy_all_count():
    """The same rule must not penalise one copy that BLAST split in two."""
    pieces = [hsp("L1HS#LINE/L1", pident=97.0, length=500, qlen=1000, qstart=1,
                  qend=500, sstart=5001, send=5500),
              hsp("L1HS#LINE/L1", pident=95.0, length=495, qlen=1000, qstart=506,
                  qend=1000, sstart=5496, send=6000)]
    hit = T.collapse_blast_hsps(pieces, 1000)[0]
    close(hit.query_coverage, 995 / 1000, "both pieces")


def test_the_microsatellite_mask_marks_arrays_not_short_runs():
    from placer.core.seqtools import microsatellite_mask

    assert all(microsatellite_mask("AT" * 10))
    assert all(microsatellite_mask("A" * 12))
    assert not any(microsatellite_mask("A" * 11))       # under 12 bp
    assert not any(microsatellite_mask("ACGTTGCA" * 3))  # period 8 > 6
    mixed = microsatellite_mask("GGCATCTGA" + "AAAG" * 5 + "CTGACCTGA")
    assert not any(mixed[:8]) and all(mixed[9:29]) and not any(mixed[-8:])


def test_one_element_split_across_library_entries_is_covered_as_one():
    """Dfam models L1 as `_5end`, `_orf2` and `_3end`, so an inserted L1 aligns
    to two entries over two parts of the insert. The family explains both; a
    single entry's coverage left the rest looking like a transduction."""
    seq = insert(3375)
    hits = T.collapse_blast_hsps([
        hsp("L1P1_orf2#LINE/L1", pident=97.9, length=2634, qlen=3375, qstart=751,
            qend=3375, sstart=3294, send=661, bitscore=4483.0),
        hsp("L1HS_3end#LINE/L1", pident=98.9, length=895, qlen=3375, qstart=10,
            qend=899, sstart=895, send=1, bitscore=1556.0),
        hsp("AluY#SINE/Alu", pident=90.0, length=100, qlen=3375, qstart=2000,
            qend=2099, bitscore=100.0),
    ], 3375)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(seq, True, hits, 0.04)
    assert evidence.best_family == "L1"
    close(evidence.best_query_coverage, (3375 - 9) / 3375, "both L1 pieces")
    assert "family_cov=" in evidence.annotation_intervals


def test_another_familys_hit_does_not_add_to_the_coverage():
    seq = insert(1000)
    hits = T.collapse_blast_hsps([
        hsp("L1HS_3end#LINE/L1", pident=98.0, length=600, qlen=1000, qstart=1,
            qend=600, sstart=1, send=600, bitscore=1000.0),
        hsp("AluY#SINE/Alu", pident=97.0, length=300, qlen=1000, qstart=651,
            qend=950, bitscore=500.0),
    ], 1000)
    evidence = T.build_insert_alignment_evidence_from_blast_hits(seq, True, hits, 0.04)
    assert evidence.best_family == "L1"
    close(evidence.best_query_coverage, 0.6, "L1 alone")
