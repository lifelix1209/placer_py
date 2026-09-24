"""
The parallel redesign: a second Python implementation, kept alongside the port.

NOT a port of the C++, and not dead code. It reaches the same question from the
other side -- it takes candidates from a Sniffles VCF instead of scanning the
BAM, scores the L1 endonuclease target motif (which the C++ does not model at
all), treats the TSD as a log-LR against a measured background rather than a
`+0.15` bonus, and its TE-body term really is 5'-truncation tolerant.

So the two halves of this package are strong in different places, and the test
suite constrains them differently on purpose:

  * the PORT (`placer_py.*`) is pinned to the C++ golden vectors, because
    reproducing it exactly is the point;
  * this REDESIGN is pinned to invariants and behaviour, because requiring it
    to reproduce the C++ numbers would require it to get worse.

`tests/test_10_mechanistic_invariants.py` is where that second constraint
lives, and `placer_py/core/integrate.py` is where the redesign's
hand-tuned score is put under the port's FDR machinery -- which is the only
place the two actually meet.
"""
