"""
Class and superfamily of a library entry, across the header conventions.

The cases are the three conventions on the libraries the benchmark actually
uses -- Dfam's `name#Class/Superfamily` (human, mouse), tldr's
`Superfamily:Family` (the cichlid MWCichlidTE-3.2 library, which states no
class at all), EDTA's Wicker codes -- and the bare name. The expected classes are
RepeatMasker's own filing of each superfamily.
"""

import pytest

from placer.core.seqtools import parse_te_name_parts
from placer.core.taxonomy import (
    Mechanism,
    TeClass,
    class_from_superfamily,
    classify,
    summarise_library,
)

pytestmark = pytest.mark.invariant


@pytest.mark.parametrize("header, te_class, superfamily", [
    ("AluYa5#SINE/Alu @Primates [S:35,85]", TeClass.SINE, "Alu"),
    ("LTR19-int#LTR/ERV1", TeClass.LTR, "ERV1"),
    ("Chompy-6_Croc#DNA/PIF-Harbinger", TeClass.DNA, "PIF-Harbinger"),
    ("SVA_A#Retroposon/SVA", TeClass.RETROPOSON, "SVA"),
    ("Helitron1_Mm#RC/Helitron", TeClass.RC, "Helitron"),
    ("HSATII#Satellite", TeClass.NON_TE, "NA"),
    ("(CA)n#Simple_repeat", TeClass.NON_TE, "NA"),
    ("Eulor4#Unknown", TeClass.UNKNOWN, "NA"),
    ("rnd-1_family-12#LINE/L2", TeClass.LINE, "L2"),       # RepeatModeler2
    ("Tigger1#DNA?/TcMar-Tigger", TeClass.DNA, "TcMar-Tigger"),  # uncertain
])
def test_a_stated_class_is_taken_as_stated(header, te_class, superfamily):
    parts = parse_te_name_parts(header)
    assert parts.te_class is te_class
    assert parts.superfamily == superfamily
    assert parts.taxon_source == ("header" if te_class is not TeClass.UNKNOWN
                                  else "none")


@pytest.mark.parametrize("header, te_class", [
    # Every superfamily spelling in MWCichlidTE-3.2 except `Unknown`.
    ("Gypsy:Gypsy-100", TeClass.LTR), ("Pao:Pao-14__dup2", TeClass.LTR),
    ("Copia:Copia-3", TeClass.LTR), ("ERV1:ERV1-2", TeClass.LTR),
    ("ERV:ERV-1", TeClass.LTR), ("DIRS:DIRS-1", TeClass.LTR),
    ("LTR:LTR-5", TeClass.LTR),
    ("L1:L1-2", TeClass.LINE), ("L2:L2-7", TeClass.LINE),
    ("L1-Tx1:L1-Tx1-3", TeClass.LINE), ("RTE-BovB:RTE-BovB-1", TeClass.LINE),
    ("Rex-Babar:Rex-Babar-9", TeClass.LINE), ("Dong-R4:Dong-R4-1", TeClass.LINE),
    ("Proto2:Proto2-1", TeClass.LINE),
    ("5S-Deu-L2:5S-Deu-L2-1", TeClass.SINE),
    ("tRNA-Core-RTE:tRNA-Core-RTE-1", TeClass.SINE),
    ("Penelope:Penelope-2", TeClass.PLE),
    ("DNA:DNA-4", TeClass.DNA), ("hAT:hAT-1", TeClass.DNA),
    ("hAT-Ac:hAT-Ac-3", TeClass.DNA), ("hAT-Charlie:hAT-Charlie-8", TeClass.DNA),
    ("hAT-Tip100:hAT-Tip100-1", TeClass.DNA),
    ("hAT-Blackjack:hAT-Blackjack-1", TeClass.DNA),
    ("CMC-EnSpm:CMC-EnSpm-10", TeClass.DNA), ("TcMar-Tc1:TcMar-Tc1-5", TeClass.DNA),
    ("TcMar-Tc2:TcMar-Tc2-1", TeClass.DNA), ("TcMar-Tigger:TcMar-Tigger-2", TeClass.DNA),
    ("TcMar-Fot1:TcMar-Fot1-1", TeClass.DNA),
    ("TcMar-ISRm11:TcMar-ISRm11-4", TeClass.DNA),
    ("PIF-Harbinger:PIF-Harbinger-2", TeClass.DNA),
    ("PIF-ISL2EU:PIF-ISL2EU-1", TeClass.DNA), ("PiggyBac:PiggyBac-3", TeClass.DNA),
    ("MULE-MuDR:MULE-MuDR-1", TeClass.DNA), ("Maverick:Maverick-2", TeClass.DNA),
    ("Helitron:Helitron-1", TeClass.RC),
])
def test_a_tldr_style_header_is_classified_by_its_superfamily(header, te_class):
    parts = parse_te_name_parts(header)
    assert parts.te_class is te_class, header
    assert parts.taxon_source == "superfamily"
    assert parts.superfamily == header.split(":")[0]


def test_an_unknown_superfamily_stays_unknown_rather_than_guessed():
    parts = parse_te_name_parts("Unknown:Unknown-12")
    assert parts.te_class is TeClass.UNKNOWN
    assert parts.taxon_source == "none"
    assert parse_te_name_parts("Zzyzx:Zzyzx-1").te_class is TeClass.UNKNOWN


@pytest.mark.parametrize("header, te_class, superfamily, family", [
    ("TE_00000350#DNA/DTA", TeClass.DNA, "hAT", "hAT"),
    ("TE_00000011#LTR/RLG", TeClass.LTR, "Gypsy", "Gypsy"),
    # EDTA files a Helitron as DNA/DHH or MITE/...: the code is more specific.
    ("TE_00000400#DNA/DHH", TeClass.RC, "Helitron", "Helitron"),
    ("TE_00000009#MITE/DTT", TeClass.DNA, "TcMar", "TcMar"),
])
def test_edta_wicker_codes_are_translated(header, te_class, superfamily, family):
    parts = parse_te_name_parts(header)
    assert (parts.te_class, parts.superfamily, parts.family) == (
        te_class, superfamily, family)


@pytest.mark.parametrize("name, te_class", [
    ("L1HS", TeClass.LINE), ("L1PA2", TeClass.LINE), ("AluYa5", TeClass.SINE),
    ("SVA_E", TeClass.RETROPOSON), ("HERVK", TeClass.LTR), ("MIR3", TeClass.SINE),
])
def test_a_bare_name_is_classified_from_the_name(name, te_class):
    parts = parse_te_name_parts(name)
    assert parts.te_class is te_class
    assert parts.taxon_source == "name"


def test_a_one_letter_key_needs_a_separator():
    """`I` (the LINE superfamily) and `P` (the DNA one) must not claim any name
    that merely starts with that letter."""
    assert class_from_superfamily("I-1_DM") is TeClass.LINE
    assert class_from_superfamily("P-element") is TeClass.DNA
    assert class_from_superfamily("Ikaros") is None
    assert class_from_superfamily("Pogo") is TeClass.DNA   # `pogo`, not `p`


def test_an_uninformative_stated_class_lets_the_superfamily_speak():
    assert classify("Unknown", "Gypsy").te_class is TeClass.LTR
    assert classify("Unknown", "").te_class is TeClass.UNKNOWN


def test_each_class_has_the_mechanism_the_decision_layer_conditions_on():
    assert classify("LINE", "L1").mechanism is Mechanism.TPRT
    assert classify("SINE", "Alu").mechanism is Mechanism.TPRT
    assert classify("LTR", "ERVK").mechanism is Mechanism.INTEGRASE
    assert classify("DNA", "hAT").mechanism is Mechanism.TRANSPOSASE
    assert classify("RC", "Helitron").mechanism is Mechanism.ROLLING_CIRCLE
    assert classify("Satellite", "").is_transposable is False


def test_the_library_summary_counts_what_could_not_be_classified():
    headers = ["AluY#SINE/Alu", "Gypsy:Gypsy-1", "L1HS", "Unknown:Unknown-1", "Foo"]
    summary = summarise_library(
        (h, _taxon(h)) for h in headers)
    assert summary.entries == 5
    assert summary.by_source == {"header": 1, "superfamily": 1, "name": 1, "none": 2}
    assert summary.unknown_fraction == pytest.approx(0.4)
    assert summary.unclassified_names == ["Unknown:Unknown-1", "Foo"]
    assert "unclassified=2" in summary.render()


def _taxon(header):
    parts = parse_te_name_parts(header)
    return classify(parts.class_label,
                    parts.superfamily if parts.superfamily != "NA" else "",
                    parts.subfamily)
