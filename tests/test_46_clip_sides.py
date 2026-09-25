"""
Naming an insertion no read spans, from its two ends assembled separately.
"""

import pytest

from placer.core import consensus as C
from placer.core import te_classifier as T
from placer.core.events import EventReadEvidence
from placer.core.fragments import InsertionFragment, InsertionFragmentSource

pytestmark = pytest.mark.invariant


def frag(read_id, source, seq, pos=1000):
    return InsertionFragment(read_id=read_id, source=source, sequence=seq,
                             length=len(seq), ref_junc_pos=pos)


def test_each_side_is_assembled_only_from_the_clips_that_reach_into_it():
    evidence = EventReadEvidence(bp_left=1000, bp_right=1000,
                                 support_qnames=["a", "b", "c"])
    start = "ACGTTGCA" * 20
    end = "GGCCTTAA" * 20
    fragments = [frag("a", InsertionFragmentSource.CLIP_REF_RIGHT, start + "T" * 50),
                 frag("b", InsertionFragmentSource.CLIP_REF_LEFT, "C" * 50 + end),
                 frag("c", InsertionFragmentSource.CLIP_REF_LEFT, end)]
    sides = C.build_side_consensuses([], fragments, evidence,
                                     consensus_fn=lambda seqs: seqs[-1])
    assert (sides.start_reads, sides.end_reads) == (1, 2)
    assert sides.start_seq.startswith(start)          # starts at the junction
    assert sides.end_seq.endswith(end)                # ends at the junction


def te(family, subfamily, identity, q_start, q_end, t_start, t_end, strand="+"):
    return T.TEAlignmentEvidence(pass_=True, best_family=family, best_subfamily=subfamily,
                                 best_identity=identity, annotation_class="LINE",
                                 annotation_order=family, te_strand=strand,
                                 te_query_start=q_start, te_query_end=q_end,
                                 te_consensus_start=t_start, te_consensus_end=t_end,
                                 te_element_length=3000, qc_reason="PASS_INSERT_TE_ALIGNMENT")


def test_two_ends_of_one_element_combine_into_one_named_insertion():
    unnamed = T.TEAlignmentEvidence()
    start = te("Rex-Babar", "Rex-Babar-16", 0.86, 0, 600, 0, 600)
    end = te("Rex-Babar", "Rex-Babar-16", 0.83, 0, 400, 2600, 3000)
    out = T.combine_side_alignments(unnamed, start, "A" * 650, end, "C" * 420 + "A" * 30)
    assert out.from_clip_sides and out.qc_reason == "PASS_INSERT_TE_ALIGNMENT_CLIP_SIDES"
    assert out.best_subfamily == "Rex-Babar-16"
    assert out.te_query_end - out.te_query_start == 1000
    assert out.best_identity == pytest.approx((0.86 * 600 + 0.83 * 400) / 1000)
    assert (out.te_consensus_start, out.te_consensus_end) == (0, 3000)
    assert out.element_structure.polya_len == 30        # the 3' end is the end side


def test_ends_that_disagree_name_the_family_of_the_longer_one_only():
    out = T.combine_side_alignments(T.TEAlignmentEvidence(),
                                    te("L2", "L2-3", 0.9, 0, 700, 0, 700), "A" * 700,
                                    te("Gypsy", "Gypsy-9", 0.9, 0, 200, 0, 200), "C" * 200)
    assert out.best_family == "L2" and out.best_subfamily == "UNKNOWN"
    assert out.qc_reason.endswith("FAMILY_ONLY")


def test_an_insert_that_already_names_an_element_is_left_alone():
    named = te("Alu", "AluY", 0.97, 0, 300, 0, 300)
    side = te("L1", "L1HS", 0.9, 0, 500, 0, 500)
    assert T.combine_side_alignments(named, side, "A" * 500, None, "") is named
    assert T.combine_side_alignments(T.TEAlignmentEvidence(), None, "", None, "").best_family == ""
