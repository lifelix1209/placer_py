# Contributing

## The one rule

**This is a port with an oracle.** The C++ at
[lifelix1209/PLACER](https://github.com/lifelix1209/PLACER) is authoritative for
anything under `placer_py/` that is not `redesign/`. Before changing behaviour
here, decide which of these you are doing, because they are reviewed
differently:

| change | what it needs |
|---|---|
| **Porting more of the C++** | The module docstring names the C++ file. Match it, including the parts that look wrong — and pin the ones that look wrong in a test that says so. |
| **Fixing a port bug** | A test that fails before and passes after, and a line in the test explaining what the C++ actually does. |
| **Changing what PLACER decides** | Change the C++ first, regenerate the oracle, then port the change. A behaviour change made only here silently forks the two. |

The third row is the one that matters. `tests/oracle/cpp_reference.json` is
frozen out of the C++ and asserted to `rtol=1e-12` — a tolerance that admits a
different order of floating-point operations and nothing else. If a change
makes those tests fail, the contract moved, and that has to be deliberate.

## Running the tests

```bash
python3 tools/run_tests_without_pytest.py          # no dependencies at all
python3 tools/run_tests_without_pytest.py test_24  # one module
pytest -q                                          # the same suite
```

Both must pass. The zero-dependency runner implements only the pytest surface
the suite uses, so if they disagree the runner needs extending — that is a bug
in `tools/run_tests_without_pytest.py`, not a reason to skip it.

That surface is `fixture` (including `scope="session"` and yield-fixtures),
`parametrize`, `mark.<name>`, `mark.xfail(strict=…)`, `skip`, `fail`, `raises`,
`approx` and module-level `pytestmark`, and it is pinned by
`tests/test_34_runner_parity.py` rather than by a sentence — every test in that
file passes only if the two runners agree about what a decorator means. What is
**not** implemented is pytest's builtin fixtures (`tmp_path`, `monkeypatch`,
`capsys`), test classes and plugins; a test that needs one of those is asking
for the zero-dependency run to become a subset, so extend the runner or find
another way to write the test.

## Writing a test

The suite has four kinds, and the marker says which argument the test is
making:

- `@pytest.mark.golden` — equality against a C++ vector. The strongest, and
  only available where `tools/dump_oracle.cpp` can reach the function.
- `@pytest.mark.invariant` — a property any correct implementation must have,
  in any language. Most of the scanner half is pinned this way, because the C++
  cannot reach those stages without a BAM.
- `@pytest.mark.regression` — a bug that was found and fixed. Say what it was.
- `@pytest.mark.contract` — the ledger schema, i.e. the seam.

Write the test first where you can. Most of this port was built that way, which
is why "done" never depended on anyone's memory of what the C++ did.

**A test that pins surprising behaviour should say it is surprising.** There are
several in here — two functions with the same name and opposite polarity, a
segmenter that does not discover TSDs, a family parsed from the class path
rather than the element name. Each is pinned with a docstring explaining why it
looks wrong and why it is kept. Please keep that habit; it is the difference
between a port and a rewrite.

## Regenerating the oracle

```bash
PLACER_SRC=/path/to/PLACER HTSLIB_INCLUDE_DIR=/path/to/htslib/include \
  tools/regenerate_oracle.sh
git diff -- tests/oracle/cpp_reference.json
```

A non-empty diff means the contract moved. Review it before committing.

## What is deliberately not here

- **abPOA.** `placer_py/consensus.py` takes the consensus function as an
  argument and raises rather than approximating. A worse consensus changes the
  insert sequence, the TE identity and the structure decode without changing
  any QC field — the run would look clean and every call would be wrong.
- **The parallel executor.** It only adds cores; the C++ asserts its own
  equivalence to the streaming path.

If you add either, add it behind the same seam rather than inline.
