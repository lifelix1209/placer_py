"""Replay-based development of PLACER's decision layer (after Dream-RSI).

Dream-RSI (Zheng et al., arXiv 2609.14858) improves an exploration policy by
replaying it against frozen records of past discovery, instead of paying for
new real evaluations. Here the expensive evaluation is a scan: HG002 chr1 costs
about 8.5 CPU-hours. A finished scan's evidence ledger is a frozen WORLD --
every hypothesis the scan evaluated, with everything it measured -- and a
decision policy is replayed against it in seconds.

    world.py      load a scan's output directory as a world
    policies/     decision policies: `current` is placer's own, the rest are
                  candidates; each maps a world's rows to calls
    objective.py  score a policy's calls the way TEBench scores a caller, and
                  compare two policies by a paired block bootstrap
    run.py        the command line: score, compare, diagnose, log

A policy sees only what the scan recorded. The truth set is held by the
objective and never passed to a policy, and positions are not features: see
`objective.check_invariance`.
"""
