# PLACER: instructions for coding agents

Read [`docs/development-strategy.md`](docs/development-strategy.md) before
changing anything that affects what PLACER calls. It is the development method
for this repository: replay first, adapted from Dream-RSI. `CONTRIBUTING.md`
covers tests, lint and review. The rules below hold in every session.

## Changing what PLACER decides

1. **Classify the change first.**
   - **Replay domain**: selection, labels, naming, placement, thresholds,
     parameters over recorded terms. Develop and accept it against frozen
     worlds with `python3 -m tools.dream.run`. Do not rerun the scan to test
     it.
   - **Scan domain**: discovery, consensus, segmentation, alignment, new
     observables. It needs an online run. First record what the next policy
     will need (`--record-world`, new ledger columns), so the following
     iteration can be replayed.
2. **Each round runs in this order.** `diagnose` → one hypothesis per
   candidate (in `placer_dev/dream/candidates/`) → `compare ... --log` against
   the accepted policy on every world in the pool.
3. **Accept a candidate only if all of these hold.**
   - The bootstrap 5th percentile of its gain is > 0 on the primary world.
   - No world regresses.
   - The invariance check passes.
   - It reads only recorded observables.

   Log rejected candidates too. The one exception is a **validity fix**, which
   removes loci whose null is shown to be misspecified. It may be accepted with
   a reported objective loss, but only with the maintainer's sign-off
   (`docs/development-strategy.md`, section 2).
4. **The objective is TEBench's, exactly**: TE recall at precision ≥ 0.95,
   ±100 bp, confident regions, PASS only, and RepeatMasker re-annotation of
   annotated worlds. Never score with a hand-rolled matcher or a wider window.
   **Score what the output reports**: calls at the VCF's position
   (`bp_left`), with the VCF's FILTER. A replay of a run's own ledger must
   reproduce that run's `calls.vcf` exactly (`tests/test_47_dream.py`).
5. **No peeking.**
   - Policies never see truth.
   - Coordinates and ids are never features.
   - No threshold chosen to flip a listed locus.
6. **Data.**
   - HG002 chr1 and the cichlid slice are dreamt on.
   - chr2–8 is online validation only.
   - The TEBench holdout contigs are never touched before release.
   - The 10 Mb slices are smoke tests only.
7. **Promote accepted logic into `placer/core/`.** The replay must call the
   production function.
8. **Before any default changes**, validate online on chr2–8 from frozen code,
   and record the new scan as a world.
9. **Cite the numbers.** A commit or CHANGELOG entry that changes decisions
   cites the world, TP / FP / FN, precision, recall and the bootstrap interval.

## Cluster runs

- Run SLURM jobs only from frozen code: a snapshot (`rsync --exclude .git` into
  `placer_dev/frozen/<name>`) or a detached worktree that nobody edits until
  the job ends. Workers re-read modules from disk mid-run.
- Use the `placer-dev` env; `te_bench`'s pyabpoa 1.5.3 dies of SIGILL here.
- Add `--exclude=node5` when that node is overloaded.

## Before handing back

```bash
pytest -q
python3 tools/run_tests_without_pytest.py    # must also pass: no tmp_path/monkeypatch
~/anaconda3/bin/ruff check .
```

Commit or push only when the maintainer asks. Commit messages follow the
existing style: an imperative subject, and a body that says what was measured.
