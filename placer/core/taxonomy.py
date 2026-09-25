"""What kind of element a library entry is: its class and superfamily.

WHY THIS EXISTS. The caller used to recognise four human families by name
prefix (Alu, L1, SVA, HERV) and guess everything else's mechanism from
substrings (`policy.family_kind`), so on a fish or a plant library almost every
element was "other". What a mechanism model needs is the CLASS -- whether the
element is copied by target-primed reverse transcription, integrated from a
cDNA, cut and pasted by a transposase or copied by rolling circle -- and that is
a property of the library entry, not of how its name happens to be spelt.

THREE HEADER CONVENTIONS, all seen on the libraries this tool is benchmarked on:

  * RepeatMasker / Dfam / EDTA / RepeatModeler2: `name#Class/Superfamily`,
    e.g. `AluYa5#SINE/Alu`, `LTR19-int#LTR/ERV1`, `TE_00000350#DNA/DTA`,
    `rnd-1_family-12#LINE/L2`. The class is stated; it is trusted as stated.
  * tldr's `Superfamily:Family`, e.g. `Gypsy:Gypsy-100`, `hAT-Ac:hAT-Ac-3`
    (the cichlid MWCichlidTE-3.2 library: 599 entries, 36 superfamilies, no
    class anywhere). The class is inferred from the superfamily.
  * a bare name, e.g. `L1HS`. The class is inferred from the name, which works
    for the well-known prefixes and otherwise gives UNKNOWN.

Inference uses `_SUPERFAMILY_CLASS`, which follows RepeatMasker's classification
(the vocabulary Dfam and RepeatModeler2 emit) plus the Wicker et al. 2007
three-letter codes EDTA uses. A name the table does not know is UNKNOWN, never a
guess -- `summarise_library` counts them so a run can say how much of its
library it could not classify.

This module names classes; it does not model them. The per-class mechanism
terms (TSD length, poly(A), terminal repeats) are the decision layer's.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class TeClass(str, Enum):
    """RepeatMasker's top-level classes, plus one bucket for non-TE repeats.

    The values are RepeatMasker's own spellings, so they can be written to an
    output file and read back by anything that already speaks that vocabulary.
    DIRS and Ngaro stay under LTR and Maverick/Polinton and Crypton under DNA,
    as RepeatMasker files them; where their mechanism differs it is a
    superfamily-level fact for the decision layer, not a class.
    """

    LINE = "LINE"
    SINE = "SINE"
    RETROPOSON = "Retroposon"   # SVA: composite, mobilised by L1 in trans
    PLE = "PLE"                 # Penelope-like elements
    LTR = "LTR"
    DNA = "DNA"
    RC = "RC"                   # rolling-circle: Helitron
    UNKNOWN = "Unknown"
    NON_TE = "NonTE"            # satellites, RNA genes, simple repeats, artefacts


class Mechanism(str, Enum):
    """How a class inserts, which is what the evidence model conditions on."""

    TPRT = "tprt"                     # target-primed reverse transcription
    INTEGRASE = "integrase"           # cDNA integrated by an integrase
    TRANSPOSASE = "transposase"       # cut-and-paste, terminal inverted repeats
    ROLLING_CIRCLE = "rolling_circle"
    UNKNOWN = "unknown"


MECHANISM_OF_CLASS: dict[TeClass, Mechanism] = {
    TeClass.LINE: Mechanism.TPRT,
    TeClass.SINE: Mechanism.TPRT,
    TeClass.RETROPOSON: Mechanism.TPRT,
    TeClass.PLE: Mechanism.TPRT,
    TeClass.LTR: Mechanism.INTEGRASE,
    TeClass.DNA: Mechanism.TRANSPOSASE,
    TeClass.RC: Mechanism.ROLLING_CIRCLE,
    TeClass.UNKNOWN: Mechanism.UNKNOWN,
    TeClass.NON_TE: Mechanism.UNKNOWN,
}


#: Class labels as they appear after the `#`, lower-cased and with any
#: trailing `?` (RepeatMasker's "uncertain") removed.
_CLASS_LABELS: dict[str, TeClass] = {
    "line": TeClass.LINE,
    "sine": TeClass.SINE,
    "retroposon": TeClass.RETROPOSON,
    "ple": TeClass.PLE,
    "ltr": TeClass.LTR,
    "dna": TeClass.DNA,
    "mite": TeClass.DNA,        # EDTA files MITEs as their own class
    "tir": TeClass.DNA,
    "rc": TeClass.RC,
    "helitron": TeClass.RC,
    "unknown": TeClass.UNKNOWN,
    "unspecified": TeClass.UNKNOWN,
    "na": TeClass.UNKNOWN,
    "satellite": TeClass.NON_TE,
    "simple_repeat": TeClass.NON_TE,
    "low_complexity": TeClass.NON_TE,
    "trna": TeClass.NON_TE,
    "rrna": TeClass.NON_TE,
    "snrna": TeClass.NON_TE,
    "scrna": TeClass.NON_TE,
    "srprna": TeClass.NON_TE,
    "rna": TeClass.NON_TE,
    "artefact": TeClass.NON_TE,
    "artifact": TeClass.NON_TE,
    "segmental": TeClass.NON_TE,
    "other": TeClass.UNKNOWN,
}

#: Superfamily -> class, for headers that carry no class. A key matches the
#: name exactly or as a prefix (see `class_from_superfamily` for the rule), so
#: `hat` covers `hAT-Charlie`, `erv` covers `ERV1` and `ERVL-MaLR`, and `l1`
#: covers `L1-Tx1` and `L1HS`. Longest key wins.
_SUPERFAMILY_CLASS: dict[str, TeClass] = {
    # --- LINE (non-LTR retrotransposons)
    "l1": TeClass.LINE, "l2": TeClass.LINE, "cr1": TeClass.LINE,
    "rte": TeClass.LINE, "rex": TeClass.LINE, "rex-babar": TeClass.LINE,
    "dong-r4": TeClass.LINE, "r1": TeClass.LINE, "r2": TeClass.LINE,
    "r4": TeClass.LINE, "i": TeClass.LINE, "jockey": TeClass.LINE,
    "tad1": TeClass.LINE, "proto1": TeClass.LINE, "proto2": TeClass.LINE,
    "crack": TeClass.LINE, "nimb": TeClass.LINE, "kiri": TeClass.LINE,
    "vingi": TeClass.LINE, "tx1": TeClass.LINE, "line": TeClass.LINE,
    "dualen": TeClass.LINE, "ingi": TeClass.LINE, "cre": TeClass.LINE,
    "rtex": TeClass.LINE, "l1-tx1": TeClass.LINE, "rte-bovb": TeClass.LINE,
    # --- SINE
    "alu": TeClass.SINE, "sine": TeClass.SINE, "mir": TeClass.SINE,
    "b2": TeClass.SINE, "b4": TeClass.SINE, "id": TeClass.SINE,
    "trna-core-rte": TeClass.SINE, "trna-rte": TeClass.SINE,
    "trna-core": TeClass.SINE, "trna-l2": TeClass.SINE,
    "trna-deu": TeClass.SINE, "trna-v": TeClass.SINE,
    "5s-deu-l2": TeClass.SINE, "5s": TeClass.SINE, "7sl": TeClass.SINE,
    "ceph": TeClass.SINE, "deu": TeClass.SINE,
    "sine1": TeClass.SINE, "sine2": TeClass.SINE, "sine3": TeClass.SINE,
    # --- Retroposon
    "sva": TeClass.RETROPOSON, "retroposon": TeClass.RETROPOSON,
    # --- PLE
    "penelope": TeClass.PLE, "ple": TeClass.PLE, "poseidon": TeClass.PLE,
    # --- LTR (RepeatMasker files DIRS and Ngaro here)
    "erv": TeClass.LTR, "herv": TeClass.LTR, "ervk": TeClass.LTR,
    "ervl": TeClass.LTR, "ervl-malr": TeClass.LTR, "erv1": TeClass.LTR,
    "gypsy": TeClass.LTR, "copia": TeClass.LTR, "pao": TeClass.LTR,
    "bel-pao": TeClass.LTR, "bel": TeClass.LTR, "dirs": TeClass.LTR,
    "ngaro": TeClass.LTR, "viper": TeClass.LTR, "caulimovirus": TeClass.LTR,
    "ltr": TeClass.LTR, "retrovirus": TeClass.LTR, "malr": TeClass.LTR,
    # --- DNA
    "dna": TeClass.DNA, "hat": TeClass.DNA, "tcmar": TeClass.DNA,
    "tc1": TeClass.DNA, "mariner": TeClass.DNA, "pif": TeClass.DNA,
    "harbinger": TeClass.DNA, "pif-harbinger": TeClass.DNA,
    "cmc": TeClass.DNA, "enspm": TeClass.DNA, "cmc-enspm": TeClass.DNA,
    "cacta": TeClass.DNA, "mule": TeClass.DNA, "mudr": TeClass.DNA,
    "mule-mudr": TeClass.DNA, "mutator": TeClass.DNA,
    "piggybac": TeClass.DNA, "merlin": TeClass.DNA, "p": TeClass.DNA,
    "transib": TeClass.DNA, "kolobok": TeClass.DNA, "zisupton": TeClass.DNA,
    "academ": TeClass.DNA, "ginger": TeClass.DNA, "sola": TeClass.DNA,
    "novosib": TeClass.DNA, "zator": TeClass.DNA, "dada": TeClass.DNA,
    "is3eu": TeClass.DNA, "crypton": TeClass.DNA, "maverick": TeClass.DNA,
    "polinton": TeClass.DNA, "mite": TeClass.DNA, "tir": TeClass.DNA,
    "tc2": TeClass.DNA, "tigger": TeClass.DNA, "pogo": TeClass.DNA,
    "fot1": TeClass.DNA, "isrm11": TeClass.DNA, "tcmar-isrm11": TeClass.DNA,
    "isl2eu": TeClass.DNA, "pif-isl2eu": TeClass.DNA,
    # --- RC
    "helitron": TeClass.RC, "rc": TeClass.RC, "helentron": TeClass.RC,
    # --- not transposable elements
    "satellite": TeClass.NON_TE, "simple_repeat": TeClass.NON_TE,
    "low_complexity": TeClass.NON_TE, "trna": TeClass.NON_TE,
    "rrna": TeClass.NON_TE, "snrna": TeClass.NON_TE, "scrna": TeClass.NON_TE,
    "srprna": TeClass.NON_TE, "artefact": TeClass.NON_TE,
    # --- unknown, stated as such
    "unknown": TeClass.UNKNOWN, "na": TeClass.UNKNOWN,
}

#: Wicker et al. 2007 three-letter codes (the order/superfamily notation EDTA
#: writes, e.g. `DNA/DTA`, `LTR/RLG`), mapped to class and to the superfamily
#: name the rest of the vocabulary uses.
_WICKER_CODES: dict[str, tuple[TeClass, str]] = {
    "RLG": (TeClass.LTR, "Gypsy"), "RLC": (TeClass.LTR, "Copia"),
    "RLB": (TeClass.LTR, "Bel-Pao"), "RLR": (TeClass.LTR, "Retrovirus"),
    "RLE": (TeClass.LTR, "ERV"), "RLX": (TeClass.LTR, "Unknown"),
    "RYD": (TeClass.LTR, "DIRS"), "RYN": (TeClass.LTR, "Ngaro"),
    "RYV": (TeClass.LTR, "Viper"), "RPP": (TeClass.PLE, "Penelope"),
    "RIR": (TeClass.LINE, "R2"), "RIT": (TeClass.LINE, "RTE"),
    "RIJ": (TeClass.LINE, "Jockey"), "RIL": (TeClass.LINE, "L1"),
    "RII": (TeClass.LINE, "I"), "RIX": (TeClass.LINE, "Unknown"),
    "RST": (TeClass.SINE, "tRNA"), "RSL": (TeClass.SINE, "7SL"),
    "RSS": (TeClass.SINE, "5S"), "RSX": (TeClass.SINE, "Unknown"),
    "DTT": (TeClass.DNA, "TcMar"), "DTA": (TeClass.DNA, "hAT"),
    "DTM": (TeClass.DNA, "MULE"), "DTE": (TeClass.DNA, "Merlin"),
    "DTR": (TeClass.DNA, "Transib"), "DTP": (TeClass.DNA, "P"),
    "DTB": (TeClass.DNA, "PiggyBac"), "DTH": (TeClass.DNA, "PIF-Harbinger"),
    "DTC": (TeClass.DNA, "CACTA"), "DTX": (TeClass.DNA, "Unknown"),
    "DYC": (TeClass.DNA, "Crypton"), "DHH": (TeClass.RC, "Helitron"),
    "DMM": (TeClass.DNA, "Maverick"), "DXX": (TeClass.DNA, "Unknown"),
}


@dataclass(frozen=True)
class TeTaxon:
    """A library entry's place in the classification."""

    te_class: TeClass = TeClass.UNKNOWN
    superfamily: str = "NA"
    #: Where `te_class` came from: "header" (stated after `#`), "superfamily"
    #: or "name" (inferred from the table, from the superfamily token or the
    #: element's own name), or "none" (UNKNOWN because nothing matched).
    source: str = "none"

    @property
    def mechanism(self) -> Mechanism:
        return MECHANISM_OF_CLASS[self.te_class]

    @property
    def is_transposable(self) -> bool:
        return self.te_class not in (TeClass.NON_TE,)


def _clean(label: str) -> str:
    return label.strip().rstrip("?").strip()


def class_from_label(label: str) -> TeClass | None:
    """A stated class label onto the enum, or None if it is not a class name."""
    return _CLASS_LABELS.get(_clean(label).lower())


def class_from_superfamily(superfamily: str) -> TeClass | None:
    """Infer the class from a superfamily (or bare element) name.

    Exact match first; then the longest table key that is a prefix of the
    name. A key of two or more characters matches whatever follows it (`alu`
    takes `AluYa5`, `l1` takes `L1HS`, `gypsy` takes `Gypsy100`); the
    one-letter keys `i` and `p` need a separator (`-`, `_`, `.`) or a digit
    after them, so they cannot swallow an unrelated name that merely starts
    with that letter. None when nothing matches -- the caller decides what an
    unknown name means.
    """
    name = _clean(superfamily).lower()
    if not name:
        return None
    wicker = _WICKER_CODES.get(name.upper())
    if wicker is not None:
        return wicker[0]
    if name in _SUPERFAMILY_CLASS:
        return _SUPERFAMILY_CLASS[name]
    best_key = ""
    for key in _SUPERFAMILY_CLASS:
        if len(key) <= len(best_key) or not name.startswith(key):
            continue
        rest = name[len(key):]
        if len(key) >= 2 or rest[0] in "-_." or rest[0].isdigit():
            best_key = key
    return _SUPERFAMILY_CLASS[best_key] if best_key else None


def classify(class_label: str, superfamily: str, element_name: str = "") -> TeTaxon:
    """The class of one library entry, from whatever its header provides.

    `class_label` is the text before the `/` after a `#` ("NA" or "" when the
    header had none), `superfamily` the text after it (or the tldr family
    token), `element_name` the entry's own name, tried last. A stated class
    that is itself uninformative (`Unknown`, `Unspecified`) lets the
    superfamily speak: `Unknown/Gypsy` is an LTR element whose submitter was
    unsure, not an element of no class.
    """
    superfamily_clean = _clean(superfamily) if superfamily else ""
    wicker = _WICKER_CODES.get(superfamily_clean.upper())
    if wicker is not None:
        superfamily_clean = wicker[1]
    canonical_superfamily = superfamily_clean or "NA"

    stated = class_from_label(class_label) if class_label and class_label != "NA" else None
    if stated is not None and stated is not TeClass.UNKNOWN:
        if wicker is not None and stated is not TeClass.NON_TE:
            # EDTA writes MITE/DTA and DNA/DHH: the code is more specific than
            # the class it is filed under (a Helitron is RC, not DNA).
            return TeTaxon(wicker[0], canonical_superfamily, "header")
        return TeTaxon(stated, canonical_superfamily, "header")

    for name, source in ((superfamily, "superfamily"), (element_name, "name")):
        if not name or name == "NA":
            continue
        inferred = class_from_superfamily(name)
        if inferred is not None and inferred is not TeClass.UNKNOWN:
            return TeTaxon(inferred, canonical_superfamily, source)
    return TeTaxon(TeClass.UNKNOWN, canonical_superfamily, "none")


@dataclass
class LibrarySummary:
    """How much of a library could be classified, for the run log."""

    entries: int = 0
    by_class: Counter = field(default_factory=Counter)
    by_source: Counter = field(default_factory=Counter)
    unclassified_names: list[str] = field(default_factory=list)

    @property
    def unknown_fraction(self) -> float:
        if self.entries == 0:
            return 0.0
        return self.by_class[TeClass.UNKNOWN] / self.entries

    def render(self, max_examples: int = 5) -> str:
        classes = ", ".join(f"{cls.value}={count}" for cls, count
                            in sorted(self.by_class.items(), key=lambda kv: -kv[1]))
        line = (f"entries={self.entries} classes: {classes}; class stated in "
                f"header={self.by_source['header']}, inferred from "
                f"superfamily={self.by_source['superfamily']}, from "
                f"name={self.by_source['name']}, "
                f"unclassified={self.by_source['none']}")
        if self.unclassified_names:
            examples = ", ".join(self.unclassified_names[:max_examples])
            line += f" (e.g. {examples})"
        return line


def summarise_library(taxa: Iterable[tuple[str, TeTaxon]]) -> LibrarySummary:
    """Count classes and where they came from, over `(name, taxon)` pairs."""
    summary = LibrarySummary()
    for name, taxon in taxa:
        summary.entries += 1
        summary.by_class[taxon.te_class] += 1
        summary.by_source[taxon.source] += 1
        if taxon.source == "none" and taxon.te_class is TeClass.UNKNOWN:
            summary.unclassified_names.append(name)
    return summary
