"""The replay harness (`tools/dream`): worlds, policies and the objective.

The objective imports TEBench's own evaluator, so those tests are skipped
where the TEBench checkout is absent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from placer.core.mechanism_selection import select_loci
from tools.dream import world
from tools.dream.policies import Decision, coverage_rule, current

LEDGER_COLUMNS = ("chrom", "pos", "candidate_retention_reason", "family", "subfamily",
                  "te_annotation_class", "best_te_identity", "best_te_query_coverage",
                  "event_consensus_len", "left_flank_align_len", "right_flank_align_len",
                  "mech_log_lr_vs_non_te", "mech_log_lr_vs_artifact", "mech_decoy_count",
                  "mech_decoy_mean_exp_linkage", "te_union_covered_bp", "te_union_coverage",
                  "te_dominant_family", "te_dominant_class", "ref_span_reads",
                  "bp_left", "bp_right", "final_qc")


def _row(pos, art, nonte=5.0, union_bp=300, union=0.9, cls="SINE", fam="Alu",
         consensus=460, reason="EVALUATED"):
    return {"chrom": "chr1", "pos": str(pos), "candidate_retention_reason": reason,
            "family": fam, "subfamily": "NA", "te_annotation_class": cls,
            "best_te_identity": "0.95", "best_te_query_coverage": str(union),
            "event_consensus_len": str(consensus), "left_flank_align_len": "80",
            "right_flank_align_len": "80", "mech_log_lr_vs_non_te": str(nonte),
            "mech_log_lr_vs_artifact": str(art), "mech_decoy_count": "100",
            "mech_decoy_mean_exp_linkage": "0.5", "te_union_covered_bp": str(union_bp),
            "te_union_coverage": str(union), "te_dominant_family": fam,
            "te_dominant_class": cls, "ref_span_reads": "20",
            "bp_left": str(pos), "bp_right": str(pos), "final_qc": "PASS_TE_CLOSED"}


def _write(rows):
    # tempfile rather than pytest's tmp_path: tools/run_tests_without_pytest.py
    # runs this file too, and it has no builtin fixtures.
    path = Path(tempfile.mkdtemp(prefix="dream_test_")) / "evidence_ledger.tsv"
    lines = ["\t".join(LEDGER_COLUMNS)]
    lines += ["\t".join(r[c] for c in LEDGER_COLUMNS) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _objective():
    """The objective module, or a skip where the TEBench checkout is absent.

    `pytest.skip`, not `importorskip`: the zero-dependency runner has only the
    former."""
    try:
        from tools.dream import objective
    except ImportError as error:
        pytest.skip(f"TEBench not importable: {error}")
    return objective


def _world_rows(rows):
    parsed, missing = world.load_rows(_write(rows))
    return parsed, missing


def test_a_world_keeps_evaluated_rows_as_numbers_and_defaults_what_it_lacks():
    rows, missing = _world_rows([_row(1000, 30.0),
                                           _row(5000, 1.0, reason="LEDGER_ONLY_PRE_EXPENSIVE_STAGE")])
    assert len(rows) == 1
    row = rows[0]
    assert row.pos == 1000 and row.mech_log_lr_vs_artifact == 30.0
    assert row.chrom == "chr1" and row.family == "Alu"
    assert row.tid == 0 and row._row_id == 0
    assert "mech_aligned_len" in missing and row.mech_aligned_len == 0


def test_the_insert_is_the_consensus_less_its_flanks_unless_recorded():
    assert world.insert_length({"event_consensus_len": 460, "left_flank_align_len": 80,
                                "right_flank_align_len": 81}) == 299
    assert world.insert_length({"insert_seq": "A" * 123, "event_consensus_len": 460}) == 123


def test_the_current_policy_is_placers_own_selection():
    raw = [_row(1000 + 10_000 * i, art=4.0 * i, nonte=3.0 * i - 5) for i in range(12)]
    rows, _ = _world_rows(raw)
    direct = [r.copy() for r in rows]
    select_loci(direct, 0.1)
    expected = {(r._row_id, "TE" if r.mech_ebh_selected else "STRUCTURAL")
                for r in direct if r.mech_ebh_selected or r.mech_structural_selected}
    replayed = {(d.row._row_id, d.label) for d in current.select(rows, 0.1)}
    assert replayed == expected
    assert all(not hasattr(r, "mech_ebh_selected") for r in rows)   # rows untouched


def test_the_coverage_rule_labels_by_tebenchs_thresholds():
    raw = [_row(10_000, 60.0, nonte=-50.0, union_bp=300, union=0.9),    # TE, whatever vs_non_te says
           _row(20_000, 60.0, union_bp=300, union=0.4),                 # under 50%: structural
           _row(30_000, 60.0, union_bp=90, union=0.9),                  # under 100 bp: structural
           _row(40_000, 60.0, union_bp=300, union=0.9, cls="NonTE")]    # not a TE class
    rows, _ = _world_rows(raw)
    labels = {int(d.row.pos): d.label for d in coverage_rule.select(rows, 0.1)}
    assert labels == {10_000: "TE", 20_000: "STRUCTURAL", 30_000: "STRUCTURAL",
                      40_000: "STRUCTURAL"}


class _PeeksAtCoordinates:
    """A policy that has memorised where the truth is: it must be caught."""

    @staticmethod
    def select(rows, q, **params):
        return [Decision(row=r, label="TE", e_value=1.0, family="Alu", te_class="SINE")
                for r in rows if int(r.pos) == 10_000]


def test_the_invariance_check_catches_a_policy_that_reads_positions():
    objective = _objective()
    rows, _ = _world_rows([_row(10_000, 60.0), _row(20_000, 50.0)])
    assert objective.check_invariance(coverage_rule, rows, 0.1)
    assert not objective.check_invariance(_PeeksAtCoordinates, rows, 0.1)


def test_scoring_is_tebenchs_matching_and_acceptance_needs_a_real_gain():
    objective = _objective()
    from tebench.model import Call
    from tebench.regions import RegionIndex

    truth_pos = [1_000_000 * k + 500 for k in range(1, 31)]
    truth = objective.Truth(
        calls=[Call(call_id=f"t{i}", sample="S", caller="truth", contig="chr1",
                    pos0=p, end0=p) for i, p in enumerate(truth_pos)],
        confident=RegionIndex.from_intervals({"chr1": [(0, 40_000_000)]}),
        region=("chr1", 0, 40_000_000))
    rows, _ = _world_rows([_row(p - 1 + 50, 60.0) for p in truth_pos]
                          + [_row(35_000_000, 60.0)])

    def decide(subset):
        return [Decision(row=r, label="TE", e_value=1.0, family="Alu", te_class="SINE")
                for r in subset]

    everything = objective.score(decide(rows), truth)
    assert (everything.tp, everything.fp, everything.fn) == (30, 1, 0)   # 50 bp off matches
    half = objective.score(decide(rows[:15]), truth)
    assert (half.tp, half.fp, half.fn) == (15, 0, 15)
    assert objective.compare(half, everything, truth).accepted
    assert not objective.compare(everything, everything, truth).accepted


def test_a_replay_of_a_runs_own_ledger_is_that_runs_calls():
    """The replay must score what the output reports: the same calls, at the
    VCF's own positions, with its FILTER. Until 2026-09-26 it scored the
    midpoint of a wide interval (the VCF writes bp_left) and counted non-PASS
    calls TEBench drops; this is the check that catches that class of error."""
    import pathlib

    from test_38_parallel import _real_run_available, _run

    reason = _real_run_available()
    if reason:
        pytest.skip(reason)
    objective = _objective()
    from tools.dream.policies import load

    out = tempfile.mkdtemp(prefix="dream_online_")
    _run(out, 1, None, "mechanism", "coverage", record_world=True)
    rows, _ = world.load_rows(pathlib.Path(out) / "evidence_ledger.tsv")
    decisions = load("coverage_placed").select(rows, 0.10)
    replayed = sorted((int(d.pos) if d.pos is not None else objective.vcf_pos0(d.row), d.family)
                      for d in decisions if d.label == "TE" and objective.vcf_pass(d))
    online = []
    for line in (pathlib.Path(out) / "calls.vcf").read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        info = dict(kv.split("=", 1) for kv in fields[7].split(";") if "=" in kv)
        if fields[6] == "PASS":
            online.append((int(fields[1]), info.get("FAM", "?")))
    assert replayed == sorted(online)
    assert len(online) >= 8          # not vacuous: the dataset plants nine
