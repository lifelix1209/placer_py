"""The structural hallmarks of an inserted element, measured by class.

WHY BY CLASS. Every class leaves a different signature, because each is
inserted by a different machine:

  class               how it inserts               what the insert shows
  ------------------  ---------------------------  ---------------------------------
  LINE, SINE,         target-primed reverse        a 3' poly(A) tail, often 5'
  Retroposon, PLE     transcription (TPRT)         truncation, sometimes a 3'
                                                   transduction past the element end
  LTR                 integrase, from a cDNA       both ends complete, starting TG
                                                   and ending CA
  DNA                 transposase, cut and paste   both ends complete, terminal
                                                   inverted repeats (TIRs)
  RC (Helitron)       rolling circle               5' TC ... 3' CTRR, no tail, no TSD

The caller used to measure the TPRT signature on every insert -- a terminal A
or T run -- whatever the element, and in the reference orientation, so a
minus-strand L1's poly(T) at the insert's START was never seen and an LTR's
A-rich end was credited as a tail.

THE ORIENTATION. Everything here is measured on the insert in the ELEMENT's
orientation: reverse-complemented when the best TE alignment is on the minus
strand. A poly(A) is then always A at the 3' end, and a minus-strand poly(T)
at the 5' end of the reference-oriented insert is the same tail -- A and T are
no longer conflated. With no oriented alignment (`strand` "NA") nothing can be
oriented, and `oriented_insert` returns the insert as given.

This module measures; it does not weigh. The decision layer turns these
observables into likelihood ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

from .seqtools import reverse_complement
from .taxonomy import MECHANISM_OF_CLASS, Mechanism, TeClass, classify

#: How far an alignment end may fall short of the element's end and still count
#: as reaching it: BLAST's local alignment routinely stops a few bases short of
#: a diverged end. Whichever is larger, an absolute floor or a fraction.
END_TOLERANCE_BP = 30
END_TOLERANCE_FRACTION = 0.03

#: A terminal run counts as a tail from this length. Six is where a run of one
#: base stops being unremarkable in random sequence (0.25^6 ~ 1/4096 per site)
#: and it is the threshold the redesign used.
MIN_TAIL_BP = 6
#: Real tails are imperfect: a single other base may interrupt one, provided at
#: least this many of the tail base follow it on the far side. Isolated
#: interruptions are what an aged or sequencing-damaged tail looks like; a
#: purity fraction instead lets the run extend into the element body, since any
#: few bases of body keep the fraction above the bar.
TAIL_RESUME_BP = 3

#: Bases compared between the two ends for a terminal inverted repeat. TIRs run
#: from ~10 bp (hAT) to hundreds (Mutator, CACTA); the first 20 are enough to
#: tell one from chance, which matches ~5 of 20 positions.
TIR_PROBE_BP = 20


@dataclass
class ElementStructure:
    """What the insert shows about how it got there. -1 / "NA" = not measured."""

    te_class: str = TeClass.UNKNOWN.value
    mechanism: str = Mechanism.UNKNOWN.value
    strand: str = "NA"
    #: The insert, in element orientation (see the module docstring).
    oriented: bool = False
    #: Length of the terminal A-rich run at the element's 3' end (T-rich at the
    #: 5' end of the reference-oriented insert on the minus strand). With no
    #: orientation, the A-or-T run at the insert's 3' end, as before.
    polya_len: int = 0
    #: Element-oriented bases between the end of the TE core and the tail.
    transduction_len: int = -1
    #: Alignment geometry on the element consensus.
    five_prime_complete: bool = False
    three_prime_complete: bool = False
    #: LTR: the element starts TG (2 = both bases) and ends CA.
    ltr_start_matches: int = -1
    ltr_end_matches: int = -1
    #: DNA: identity between the first TIR_PROBE_BP bases and the reverse
    #: complement of the last TIR_PROBE_BP, 0..1.
    tir_identity: float = -1.0
    #: Helitron: 5' TC (0-2 matching) and 3' CTRR (0-4 matching).
    helitron_start_matches: int = -1
    helitron_end_matches: int = -1

    @property
    def both_ends_complete(self) -> bool:
        return self.five_prime_complete and self.three_prime_complete

    @property
    def tail_expected(self) -> bool:
        """Whether a poly(A) tail is part of this class's signature."""
        return self.mechanism in (Mechanism.TPRT.value, Mechanism.UNKNOWN.value)


def oriented_insert(insert_seq: str, strand: str) -> str:
    """The insert in element orientation: reverse complement on the minus strand."""
    seq = (insert_seq or "").upper()
    return reverse_complement(seq) if strand == "-" else seq


def terminal_run(seq: str, base: str, resume: int = TAIL_RESUME_BP) -> int:
    """Length of the tail of `base` ending `seq`, allowing lone interruptions.

    Anchored at the last base, which must itself be `base`: a tail that does
    not reach the end of the insert is not a tail at the junction. Reading
    leftwards, a single other base is taken into the tail only when at least
    `resume` more of `base` follow it; the tail ends at the first base that
    does not qualify, and always on `base`.
    """
    i = len(seq)
    total = 0
    while True:
        run = 0
        while i - run > 0 and seq[i - run - 1] == base:
            run += 1
        if run == 0:
            break
        total += run
        i -= run
        if i < 2 or seq[i - 1] == base:
            break
        after = 0
        while i - 1 - after > 0 and seq[i - 2 - after] == base:
            after += 1
        if after < resume:
            break
        total += 1          # the interrupting base
        i -= 1
    return total


def end_reached(position: int, target: int, element_length: int) -> bool:
    if position < 0 or element_length <= 0:
        return False
    tolerance = max(END_TOLERANCE_BP, END_TOLERANCE_FRACTION * element_length)
    return abs(target - position) <= tolerance


def _matches(seq: str, pattern: str) -> int:
    """Positions of `seq` matching `pattern`, where R is A or G."""
    count = 0
    for base, want in zip(seq, pattern):
        if want == "R":
            count += base in "AG"
        else:
            count += base == want
    return count


def _identity(a: str, b: str) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def measure(insert_seq: str, te_class: str, strand: str,
            consensus_start: int = -1, consensus_end: int = -1,
            element_length: int = -1, core_end_on_insert: int = -1,
            superfamily: str = "") -> ElementStructure:
    """Measure the class's hallmarks on one insert.

    `consensus_start`/`consensus_end` are the best alignment's interval on the
    element (0-based half-open), `element_length` the element's length, and
    `core_end_on_insert` where the aligned core ends on the insert in ELEMENT
    orientation (-1 when unknown).
    """
    cls = _class_of(te_class, superfamily)
    out = ElementStructure(te_class=cls.value,
                           mechanism=MECHANISM_OF_CLASS[cls].value,
                           strand=strand if strand in ("+", "-") else "NA")
    out.oriented = out.strand != "NA"
    seq = oriented_insert(insert_seq, out.strand)

    if out.oriented:
        out.polya_len = terminal_run(seq, "A")
    else:
        out.polya_len = max(terminal_run(seq, "A"), terminal_run(seq, "T"))
    if out.polya_len < MIN_TAIL_BP:
        out.polya_len = 0

    out.five_prime_complete = end_reached(consensus_start, 0, element_length)
    out.three_prime_complete = end_reached(consensus_end, element_length, element_length)

    if 0 <= core_end_on_insert <= len(seq):
        tail_start = len(seq) - out.polya_len
        out.transduction_len = max(0, tail_start - core_end_on_insert)

    if len(seq) >= 4:
        out.ltr_start_matches = _matches(seq[:2], "TG")
        out.ltr_end_matches = _matches(seq[-2:], "CA")
        out.helitron_start_matches = _matches(seq[:2], "TC")
        out.helitron_end_matches = _matches(seq[-4:], "CTRR")
    if len(seq) >= 2 * TIR_PROBE_BP:
        out.tir_identity = _identity(seq[:TIR_PROBE_BP],
                                     reverse_complement(seq[-TIR_PROBE_BP:]))
    return out


def _class_of(te_class: str, superfamily: str) -> TeClass:
    try:
        return TeClass(te_class)
    except ValueError:
        return classify("NA", superfamily).te_class
