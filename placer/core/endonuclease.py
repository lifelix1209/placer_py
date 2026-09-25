"""The L1 ORF2p endonuclease target motif at an insertion junction.

Taken from the pre-port redesign (`redesign/model/mechanistic.py`), which was
the only code in the repository that modelled it, with one fix: the minus-strand
window was assembled in the wrong order (see `junction_windows`).

The endonuclease nicks the BOTTOM strand at a degenerate 5'-TTTT|AA-3'
consensus, and the 3'-OH it leaves primes reverse transcription (TPRT). Read on
the top strand, that same site is 5'-TT|AAAA-3'. Which strand carries it
depends on the orientation of the inserted element, so both are scored and the
better one is kept.

This is a LINE/SINE mechanism. An LTR element or a DNA transposon is inserted by
an integrase or a transposase that has no such preference, so a caller must
only apply this term to non-LTR retrotransposons (and to the SINEs and SVAs
they mobilise in trans). Nothing here knows the element's class; that is the
caller's job.

Refs: Flasch et al. 2019 Cell (L1 EN target preference); Miller et al. 2021 NAR
10.1093/nar/gkab826 (EN sequence recognition).
"""

from __future__ import annotations

import math

from .seqtools import reverse_complement

#: The bottom-strand consensus, 5'->3', with the nick between positions 4 and 5.
EN_CONSENSUS = "TTTTAA"
EN_NICK_OFFSET = 4

#: A degenerate position-weight model: each consensus base with this probability,
#: the other three sharing the rest. 0.85 + 3 * 0.05 = 1.
EN_MATCH_P = 0.85
EN_MISMATCH_P = 0.05
UNIFORM_P = 0.25


def pwm_log_odds(window: str) -> float:
    """Log-odds of a 6 bp window under the degenerate TTTTAA model vs uniform.

    Range: 6 * ln(0.85/0.25) = 7.34 nats for an exact match, down to
    6 * ln(0.05/0.25) = -9.66 for a window sharing no base with the consensus.
    """
    if len(window) != len(EN_CONSENSUS):
        raise ValueError(f"window must be {len(EN_CONSENSUS)} bp, got {len(window)}")
    score = 0.0
    for base, consensus in zip(window.upper(), EN_CONSENSUS):
        p = EN_MATCH_P if base == consensus else EN_MISMATCH_P
        score += math.log(p / UNIFORM_P)
    return score


def junction_windows(left_flank: str, right_flank: str) -> tuple[str, str] | None:
    """The two 6 bp windows that could hold the motif, each read 5'->3'.

    `left_flank` ends at the breakpoint and `right_flank` starts at it, both
    on the top strand. The nick is at the breakpoint, 4 bases into the motif
    on whichever strand was cut:

      * "+", the top strand was nicked: the motif reads directly as
        `left[-4:] + right[:2]`.
      * "-", the bottom strand was nicked: the top-strand bases are
        `left[-2:] + right[:4]` (5'-TT|AAAA-3' for a perfect site), and the
        bottom strand reads them reverse-complemented.

    The redesign built the minus window as `revcomp(right[:4] + left[-2:])`,
    which is `revcomp(left[-2:]) + revcomp(right[:4])` -- the right bases but in
    the wrong order, with the nick at offset 2. `test_39` pins the fix.
    """
    left = (left_flank or "").upper()
    right = (right_flank or "").upper()
    if len(left) < 4 or len(right) < 4:
        return None
    plus = left[-4:] + right[:2]
    minus = reverse_complement(left[-2:] + right[:4])
    return plus, minus


def endonuclease_motif_log_odds(left_flank: str,
                                right_flank: str) -> tuple[float, str] | None:
    """Best motif log-odds over both strands, and which strand gave it.

    Returns None when either flank is too short to hold the window, so that
    "not evaluated" is never confused with a score of zero.
    """
    windows = junction_windows(left_flank, right_flank)
    if windows is None:
        return None
    plus, minus = windows
    plus_score = pwm_log_odds(plus)
    minus_score = pwm_log_odds(minus)
    if minus_score > plus_score:
        return minus_score, "-"
    return plus_score, "+"
