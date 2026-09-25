"""
The per-class likelihood ratios, and the property that makes them e-values.

Hand-computed terms first; then, for each class, a simulation of the null the
ratio is against, checking E_null[exp(term)] <= 1 -- the one property e-BH
needs. A likelihood ratio has it by construction; these checks catch a ratio
whose null was written down wrong.
"""

import math
import random

import pytest

from placer.core import mechanism as M
from placer.core.taxonomy import TeClass

pytestmark = pytest.mark.invariant


def obs(**kw):
    base = dict(te_class=TeClass.LINE, superfamily="L1", identity=0.97,
                aligned_len=800, element_start=400, element_end=1200,
                element_length=1200, polya_len=30, three_prime_complete=True,
                tsd_len=13, en_log_odds=5.0, n_alt=10, n_ref=10, insert_len=850)
    base.update(kw)
    return M.LocusObservation(**base)


# -------------------------------------------------------------- the pieces
def test_robust_is_a_mixture_with_a_floor():
    assert M.robust(0.0, 0.8) == pytest.approx(0.0, abs=1e-12)
    assert M.robust(-math.inf, 0.8) == pytest.approx(math.log(0.2))
    assert M.robust(10.0, 0.8) == pytest.approx(math.log(0.8 * math.exp(10) + 0.2))
    assert M.robust(-50.0, 0.6) > math.log(0.4) - 1e-9


def test_the_tsd_model_is_the_superfamily_s_own():
    params = M.MechanismParameters()
    assert params.tsd_model(TeClass.DNA, "TcMar-Tc1").motif == "TA"
    assert params.tsd_model(TeClass.DNA, "hAT-Charlie").modal_len == 8
    assert params.tsd_model(TeClass.DNA, "PiggyBac").motif == "TTAA"
    assert params.tsd_model(TeClass.LTR, "ERVL-MaLR").modal_len == 5
    assert params.tsd_model(TeClass.LINE, "L1").mean_len == 15.0     # class default
    # The one-letter `p` must not swallow PIF-Harbinger or PiggyBac.
    assert params.tsd_model(TeClass.DNA, "PIF-Harbinger").motif == "TAA"


def test_a_fixed_tsd_length_puts_its_mass_on_the_superfamily_s_length():
    """hAT duplicates 8 bp. The length model has to say so; the full TSD term
    also weighs how improbable the duplication is by chance, which is why a
    long exact duplication beside a hAT is still evidence of an insertion."""
    hat = M.SUPERFAMILY_TSD["hat"]
    assert hat.log_p_length(8) - hat.log_p_length(13) > 5.0
    assert hat.log_p_length(8) > hat.log_p_length(7) > hat.log_p_length(13)


def test_a_ta_tsd_earns_the_motif_for_tcmar_and_nothing_for_other_sequence():
    base = dict(te_class=TeClass.DNA, superfamily="TcMar", tsd_len=2)
    ta = M.tsd_term(obs(**base, tsd_seq="TA"), M.SUPERFAMILY_TSD["tcmar"])
    gc = M.tsd_term(obs(**base, tsd_seq="GC"), M.SUPERFAMILY_TSD["tcmar"])
    assert ta - gc == pytest.approx(math.log(0.9 * 16 + 0.1) - math.log(0.1))


def test_a_tail_is_evidence_only_where_the_locus_is_not_already_a_rich():
    params = M.MechanismParameters()
    plain = M.tail_term(obs(polya_len=25, local_a_frac=0.30), params)
    a_rich = M.tail_term(obs(polya_len=25, local_a_frac=0.85), params)
    assert plain > a_rich > 0.0
    assert M.tail_term(obs(polya_len=0), params) < 0.0


def test_ends_of_a_complete_insertion_count_and_a_full_length_one_is_neutral():
    partial = obs(te_class=TeClass.DNA, element_start=0, element_end=300,
                  element_length=600, five_prime_complete=True,
                  three_prime_complete=False)
    # One end reached out of n = 301 placements, the other missing: log(301)
    # through the mixture, plus the floor.
    assert M.ends_term(partial) == pytest.approx(
        M.robust(math.log(301), M.W_ENDS) + math.log(1 - M.W_ENDS))
    full = obs(te_class=TeClass.DNA, element_start=0, element_end=600,
               element_length=600, five_prime_complete=True, three_prime_complete=True)
    assert M.ends_term(full) == 0.0


# ------------------------------------------------------------- the decision
def test_each_class_is_scored_only_on_its_own_hallmarks():
    line = M.score_locus(obs())
    assert {"anchoring", "tail", "en_motif", "tsd"} <= set(line.terms)
    dna = M.score_locus(obs(te_class=TeClass.DNA, superfamily="hAT", tir_identity=1.0,
                            polya_len=40))
    assert "tail" not in dna.terms and "en_motif" not in dna.terms
    assert dna.terms["termini"] > 10.0


def test_the_score_is_the_minimum_of_the_two_questions():
    s = M.score_locus(obs())
    assert s.score == min(s.vs_non_te, s.vs_artifact)
    # One read in 31, and nothing at the locus but the read: no insertion here.
    unsupported = M.score_locus(obs(n_alt=1, n_ref=30, tsd_len=0, en_log_odds=None))
    assert unsupported.vs_non_te > 0 > unsupported.vs_artifact
    assert unsupported.score < 0
    assert M.score_locus(obs(n_alt=0, n_ref=30, tsd_len=0, en_log_odds=None)).vs_artifact < 0


def test_a_diverged_old_copy_is_not_called_a_new_insertion():
    young = M.score_locus(obs(te_class=TeClass.LTR, superfamily="Gypsy", identity=0.93))
    old = M.score_locus(obs(te_class=TeClass.LTR, superfamily="Gypsy", identity=0.82))
    assert young.terms["sequence"] > 0 > old.terms["sequence"]


def test_a_non_te_repeat_is_never_a_te_call():
    s = M.score_locus(obs(te_class=TeClass.NON_TE, identity=0.99))
    assert s.vs_non_te < 0


# ------------------------------------------------ e-values under their nulls
def _mean_exp(values):
    return sum(math.exp(v) for v in values) / len(values)


@pytest.mark.parametrize("model_key", ["hat", "tcmar", "erv", "piggybac"])
def test_the_tsd_term_has_null_expectation_at_most_one(model_key):
    """
    Under the artifact null the flanks look duplicated over tau bp by chance,
    with the probability `p_null_tandem_duplication` assigns, and not at all
    otherwise. The expectation of exp(term) is then a sum we can enumerate.
    """
    from placer.core import tprt
    model = M.SUPERFAMILY_TSD[model_key]
    total = math.exp(M.tsd_term(obs(tsd_len=0), model))     # "no TSD": P ~ 1
    for tau in range(1, 51):
        p = tprt.p_null_tandem_duplication(tau, 0, 0.0) * 0.75   # exactly tau
        total += p * math.exp(M.tsd_term(obs(tsd_len=tau, tsd_seq=""), model))
    assert total <= 1.0 + 1e-6, total


def test_the_tail_term_has_null_expectation_at_most_one():
    """A-run lengths at the insert end under local composition f."""
    params = M.MechanismParameters()
    for f in (0.2, 0.3, 0.6, 0.9):
        total = 0.0
        for a in range(400):
            p = (f ** a) * (1 - f)
            total += p * math.exp(M.tail_term(obs(polya_len=a, local_a_frac=f), params))
        # Runs of 1-5 are measured as 0 by element_structure (MIN_TAIL_BP);
        # scoring them as their own length can only raise this sum.
        assert total <= 1.0 + 1e-6, (f, total)


def test_the_termini_terms_have_null_expectation_at_most_one():
    """Each probed base matches its canonical terminus with p = 1/4 by chance."""
    rng = random.Random(5)
    for te_class, fields in ((TeClass.LTR, ("ltr_start_matches", "ltr_end_matches")),):
        values = []
        for _ in range(20000):
            kw = {name: sum(rng.random() < 0.25 for _ in range(2)) for name in fields}
            values.append(M.termini_term(obs(te_class=te_class, **kw)))
        assert _mean_exp(values) <= 1.02
    values = []
    for _ in range(20000):
        k = sum(rng.random() < 0.25 for _ in range(20))
        values.append(M.termini_term(obs(te_class=TeClass.DNA, tir_identity=k / 20)))
    assert _mean_exp(values) <= 1.02


def test_three_prime_anchoring_tolerates_an_unaligned_consensus_tail():
    """Dfam's Alu ends in its own poly(A), which does not align: a complete
    Alu reaching element position 282 of 311 is 3'-complete."""
    complete = M.score_locus(obs(te_class=TeClass.SINE, superfamily="Alu",
                                 element_start=0, element_end=282, element_length=311,
                                 three_prime_complete=True))
    fragment = M.score_locus(obs(te_class=TeClass.SINE, superfamily="Alu",
                                 element_start=46, element_end=187, element_length=311,
                                 three_prime_complete=False))
    assert complete.terms["anchoring"] >= 0.0
    assert fragment.terms["anchoring"] < 0.0


def test_an_insert_no_element_aligned_to_is_never_a_te_call_whatever_its_ends():
    """An A-rich end is not a TE without a TE: the hallmark terms need an
    aligned element to be evidence about."""
    s = M.score_locus(obs(te_class=TeClass.UNKNOWN, identity=0.0, aligned_len=0,
                          polya_len=40, ltr_start_matches=2, ltr_end_matches=2))
    assert s.vs_non_te < 0
    assert "tail" not in s.terms
    assert s.vs_artifact > 0            # it can still be a structural call
