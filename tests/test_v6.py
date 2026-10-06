"""v6 correctness gates. These must pass before any compute is spent on v6.

v6's whole claim is that the *only* thing that moves across its grid is the
budget, and that the ladder is nested — a short run is a prefix of a long one
with the same seed. Most of what follows is about those two properties, because
if either fails the study measures something other than what it says.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.learn import total_error
from bpneat.record import code_fingerprint as v2_fingerprint
from bpneat.record import deserialise_genome
from bpneat.v3.conditions import GENERATIONS as V3_GENERATIONS
from bpneat.v3.conditions import POPULATION, RESTARTS, _random_search
from bpneat.v3.datasets import make_bundle
from bpneat.v3.fingerprint import v3_fingerprint
from bpneat.v4.fingerprint import v4_fingerprint
from bpneat.v5.conditions import BATCH_SIZE, INNER_STEPS
from bpneat.v5.conditions import base_config as v5_base_config
from bpneat.v5.fingerprint import v5_fingerprint
from bpneat.v5.search import V5Config, search
from bpneat.v6.conditions import (
    CONDITIONS,
    base_config,
    config_for,
    random_architecture_search,
    run_condition,
)
from bpneat.v6.fingerprint import V6_SCIENCE_MODULES, fingerprints
from bpneat.v6.protocol import (
    ALL_CONDITIONS,
    ALL_TASKS,
    ARMS,
    BUDGETS,
    BURNED_SEEDS,
    EVERY_CONDITION,
    EXTENSION_BUDGETS,
    FAMILIES,
    FAMILY_SIZE,
    LADDER_GENERATIONS,
    REFERENCE,
    REFERENCE_BUDGET,
    REFERENCE_CANDIDATES,
    REPLICATES,
    SLOPE_AXIS,
    SLOPE_TERMS,
    cells,
    condition_name,
    planned_runs,
    split_condition,
    validate,
)
from bpneat.v6.suite import _order_key, _plan_by_cell, shard_cells

PILOT_DATA, PILOT_SEARCH = 9004, 19004
CONFIRMATORY_RUNS = 1080
EVERY_RUN = 1350
CELLS = 90


# --------------------------------------------------------------------------
# Protocol identity
# --------------------------------------------------------------------------


def test_protocol_validates():
    validate()


def test_plan_size_and_cells():
    assert len(planned_runs()) == CONFIRMATORY_RUNS
    assert len(planned_runs(include_extension=True)) == EVERY_RUN
    assert len(cells()) == CELLS
    ids = {(p["task"], p["condition"], p["replicate"]) for p in planned_runs(True)}
    assert len(ids) == EVERY_RUN


def test_the_geometries_are_v5s_three():
    assert ALL_TASKS == ("spiral", "checkerboard", "spiral3")


def test_the_ladder_is_the_declared_one():
    assert LADDER_GENERATIONS == (4, 9, 20, 62)
    assert [b.candidates for b in BUDGETS] == [500, 1000, 2100, 6300]
    assert [b.candidates for b in EXTENSION_BUDGETS] == [16800]
    # Written out so that moving a rung has to be a deliberate edit here too.
    assert [round(b.multiplier, 4) for b in BUDGETS] == [0.2381, 0.4762, 1.0, 3.0]
    assert EXTENSION_BUDGETS[0].multiplier == 8.0
    assert REFERENCE_CANDIDATES == 2100
    assert REFERENCE_BUDGET.candidates == REFERENCE_CANDIDATES
    assert REFERENCE == "search_b2100"


def test_the_reference_rung_is_what_every_earlier_release_ran():
    """If v3 ever changes its generation count the ladder stops being anchored."""
    for task in ALL_TASKS:
        assert V3_GENERATIONS[task] == REFERENCE_BUDGET.generations
        assert POPULATION * (V3_GENERATIONS[task] + 1) == REFERENCE_CANDIDATES


def test_only_generations_move_across_the_ladder():
    ref = base_config("spiral", REFERENCE_BUDGET)
    fields = [f for f in vars(ref) if f != "generations"]
    for b in BUDGETS + EXTENSION_BUDGETS:
        cfg = base_config("spiral", b)
        assert cfg.generations == b.generations
        for f in fields:
            assert getattr(cfg, f) == getattr(ref, f), f"{b.label} moved {f}"


def test_the_search_arm_at_the_reference_rung_is_v5s_reference_configuration():
    """v6's anchor must be the configuration v5 released, field for field."""
    for task in ALL_TASKS:
        got = config_for(REFERENCE, task)
        want = v5_base_config(task)
        assert vars(got) == vars(want)
        assert got.inner_steps == INNER_STEPS
        assert got.batch_size == BATCH_SIZE


def test_no_replicate_seed_was_spent_before():
    seeds = {r.dataset_seed for r in REPLICATES} | {r.search_seed for r in REPLICATES}
    assert not (seeds & BURNED_SEEDS)
    # v5's thirty replicates are burned, so v6's sealed test is fresh again.
    assert 70_001 in BURNED_SEEDS and 80_001 in BURNED_SEEDS
    assert PILOT_DATA in BURNED_SEEDS and PILOT_SEARCH in BURNED_SEEDS


def test_families_are_declared_in_advance_and_written_out():
    assert FAMILY_SIZE == {
        "search_vs_null": 12,
        "search_vs_fixed": 12,
        "scaling_slopes": 15,
        "equivalence_at_reference": 3,
        "size_slopes": 3,
    }
    for name, members in FAMILIES.items():
        assert len(set(members)) == len(members), name
        assert len(members) == FAMILY_SIZE[name]
    assert set(SLOPE_TERMS) == set(SLOPE_AXIS)
    assert set(SLOPE_AXIS.values()) <= {"gradient_steps", "candidates"}


def test_only_the_primary_outcome_is_a_scored_family():
    """A size contrast is descriptive and must not carry a corrected p-value."""
    from bpneat.v6.analysis import arm_contrasts

    runs, final = [], {}
    assert arm_contrasts(runs, final, "causal_hidden_nodes") == []
    # And the flag the analysis sets is the one the figures and CSVs read.
    import inspect

    src = inspect.getsource(arm_contrasts)
    assert 'scored = metric == "test_accuracy"' in src
    assert "if members and scored:" in src


def test_the_extension_rung_scores_nothing():
    ext = {b.label for b in EXTENSION_BUDGETS}
    for name in ("search_vs_null", "search_vs_fixed", "equivalence_at_reference"):
        labels = {label for _, label in FAMILIES[name]}
        assert not (labels & ext), f"{name} would score the extension rung"
    assert not (set(ALL_CONDITIONS) & {condition_name(a, b)
                                       for b in EXTENSION_BUDGETS for a in ARMS})
    assert set(EVERY_CONDITION) == set(CONDITIONS)


def test_every_arm_exists_at_every_rung():
    for b in BUDGETS + EXTENSION_BUDGETS:
        for arm in ARMS:
            name = condition_name(arm, b)
            assert name in CONDITIONS
            assert split_condition(name) == (arm, b)


# --------------------------------------------------------------------------
# The ladder is nested — the property every paired contrast rests on
# --------------------------------------------------------------------------


def test_a_short_search_is_a_prefix_of_a_long_one():
    """The search's stream must not depend on how many generations it will run.

    This is what makes the budget contrast paired on the strongest terms, and it
    is also why no hypothesis is scored on validation accuracy.
    """
    bundle = make_bundle("spiral", seed=PILOT_DATA)

    def cfg(gens):
        return V5Config(task="spiral", generations=gens, population=16, inner_steps=20)

    short = search(bundle, cfg(3), seed=PILOT_SEARCH)
    long = search(bundle, cfg(7), seed=PILOT_SEARCH)
    assert len(short.history) == 4 and len(long.history) == 8
    assert short.history == long.history[:4]
    assert short.champion.fitness == long.history[3]["champion_fitness"]
    # And the consequence the contract names: never worse with more budget.
    assert long.champion.fitness >= short.champion.fitness


def test_the_null_improves_monotonically_with_its_budget():
    """Same seed, longer stream: the running best can only improve."""
    bundle = make_bundle("spiral", seed=PILOT_DATA)
    cfg = V5Config(task="spiral", generations=0, population=8, inner_steps=20)
    losses = []
    for budget in (3, 6, 12):
        g, w, cands, _ = random_architecture_search(bundle, cfg, PILOT_SEARCH, budget)
        assert cands == budget
        losses.append(
            total_error(g, w, bundle.validation.X, bundle.validation.y, cfg.settle)
        )
    assert losses == sorted(losses, reverse=True) or len(set(losses)) < 3
    assert losses[-1] <= losses[0]


def test_the_null_is_v3s_null_transcribed_not_reinterpreted():
    """v6 owns the lines its fingerprint covers; they must still be v3's lines."""
    bundle = make_bundle("checkerboard", seed=PILOT_DATA)
    cfg = V5Config(task="checkerboard", generations=0, population=8, inner_steps=20)
    for budget in (4, 9):
        mine = random_architecture_search(bundle, cfg, PILOT_SEARCH, budget)
        theirs = _random_search(bundle, cfg, PILOT_SEARCH, budget)
        assert mine[0].ops == theirs[0].ops
        assert mine[0].src == theirs[0].src and mine[0].dst == theirs[0].dst
        assert np.array_equal(np.asarray(mine[1]), np.asarray(theirs[1]))
        assert mine[2] == theirs[2] and mine[3] == theirs[3]


def test_the_null_never_reads_the_sealed_test_split():
    clean = make_bundle("spiral", seed=PILOT_DATA)
    poisoned = make_bundle("spiral", seed=PILOT_DATA)
    poisoned.test.X[:] = np.nan
    poisoned.test.y[:] = np.nan
    cfg = V5Config(task="spiral", generations=0, population=8, inner_steps=20)
    a = random_architecture_search(clean, cfg, PILOT_SEARCH, 6)
    b = random_architecture_search(poisoned, cfg, PILOT_SEARCH, 6)
    assert a[0].ops == b[0].ops
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


# --------------------------------------------------------------------------
# Budget matching
# --------------------------------------------------------------------------


def test_each_fixed_arm_is_matched_to_the_search_arm_at_its_own_rung():
    from bpneat.v6.protocol import MATCHED_TO

    assert len(MATCHED_TO) == len(BUDGETS) + len(EXTENSION_BUDGETS)
    for matched, target in MATCHED_TO.items():
        assert split_condition(matched)[1] is split_condition(target)[1]
        assert split_condition(matched)[0] == "fixed"
        assert split_condition(target)[0] == "search"


def test_a_fixed_arm_refuses_a_budget_it_was_not_matched_to():
    with pytest.raises(ValueError, match="search_b500"):
        run_condition("fixed_b500", "spiral", 99, PILOT_DATA, PILOT_SEARCH)
    with pytest.raises(ValueError, match="search_b500"):
        run_condition("fixed_b500", "spiral", 99, PILOT_DATA, PILOT_SEARCH,
                      matched_steps={"search_b2100": 5_000})


def test_the_fixed_arm_spends_the_budget_it_was_given():
    rec = run_condition("fixed_b500", "spiral", 99, PILOT_DATA, PILOT_SEARCH,
                        matched_steps={"search_b500": 4_000})
    assert rec["config"]["matched_steps"] == 4_000
    assert rec["config"]["matched_to"] == "search_b500"
    assert rec["config"]["restarts"] == RESTARTS
    assert rec["compute"]["candidate_evaluations"] == RESTARTS
    assert rec["compute"]["gradient_steps"] == pytest.approx(4_000, rel=0.02)


# --------------------------------------------------------------------------
# Runs and records
# --------------------------------------------------------------------------


def test_run_condition_stores_a_loadable_champion():
    from bpneat.learn import accuracy

    rec = run_condition("search_b500", "spiral", 99, PILOT_DATA, PILOT_SEARCH)
    blob = json.loads(json.dumps(rec))
    g, w = deserialise_genome(blob["champion"])
    bundle = make_bundle("spiral", seed=PILOT_DATA)
    got = accuracy(g, w, bundle.validation.X, bundle.validation.y, True)
    assert got == pytest.approx(rec["metrics"]["validation_accuracy"], abs=1e-12)
    assert rec["test_evaluated"] is False
    assert rec["arm"] == "search" and rec["budget"] == "b500"
    assert rec["candidate_budget"] == 500
    assert rec["compute"]["candidate_evaluations"] == 500
    assert rec["history"], "the trajectory is what makes the nesting checkable"


def test_a_record_carries_its_rung_so_the_analysis_never_parses_a_name():
    rec = run_condition("search_b500", "checkerboard", 99, PILOT_DATA, PILOT_SEARCH)
    for key in ("arm", "budget", "candidate_budget", "budget_multiplier"):
        assert key in rec
    assert rec["config"]["generations"] == 4
    assert rec["config"]["candidate_budget"] == 500


def test_every_planned_condition_is_defined():
    assert {p["condition"] for p in planned_runs(True)} == set(CONDITIONS)


# --------------------------------------------------------------------------
# Sharding and ordering
# --------------------------------------------------------------------------


def test_shards_partition_the_cells():
    for total in (1, 3, 4, 7):
        seen = [c for i in range(total) for c in shard_cells(i, total)]
        assert sorted(seen) == sorted(cells())
        assert len(seen) == len(cells())
    with pytest.raises(ValueError):
        shard_cells(4, 4)


def test_cells_are_replicate_major_so_a_truncated_run_is_a_smaller_release():
    first = cells()[: len(ALL_TASKS)]
    assert {t for t, _ in first} == set(ALL_TASKS)
    assert {r for _, r in first} == {1}


def test_a_cell_runs_cheap_rungs_first_and_a_search_before_its_fixed_arm():
    items = sorted(_plan_by_cell(True)[("spiral", 1)], key=_order_key)
    order = [p["condition"] for p in items]
    assert len(order) == len(EVERY_CONDITION)
    budgets = [split_condition(c)[1].candidates for c in order]
    assert budgets == sorted(budgets)
    for b in BUDGETS + EXTENSION_BUDGETS:
        assert order.index(condition_name("search", b)) < order.index(
            condition_name("fixed", b)
        )


def test_the_manifest_does_not_call_a_missing_extension_rung_incomplete(tmp_path):
    from bpneat.v6.suite import _manifest

    runs = tmp_path / "raw" / "runs"
    runs.mkdir(parents=True)
    for p in planned_runs():
        (runs / f"{p['task']}__{p['condition']}__r{p['replicate']:02d}.json").write_text("{}")
    m = _manifest(tmp_path, runs, fingerprints(), [], 0.0, 0, 0)
    assert m["complete"] is True
    assert m["missing"] == [] and m["unexpected"] == []
    assert m["extension_present"] == 0
    assert m["extension_planned"] == EVERY_RUN - CONFIRMATORY_RUNS
    assert m["extension_complete"] is False


# --------------------------------------------------------------------------
# Analysis: the scorer is frozen with the protocol, so it is gated with it
# --------------------------------------------------------------------------


def test_holm_corrects_at_the_declared_family_size_not_the_present_one():
    """A missing cell must not buy the surviving cells more power."""
    from bpneat.v6.analysis import _holm

    rows = [{"p": 0.01}, {"p": 0.02}]
    _holm(rows, "p", 12)
    assert rows[0]["holm_p"] == pytest.approx(0.12)
    assert rows[1]["holm_p"] == pytest.approx(0.22)
    assert not any(r["holm_significant"] for r in rows)


def test_equivalence_declares_equivalence_only_inside_the_margin():
    from bpneat.v6.analysis import _wilcoxon_p
    from bpneat.v6.protocol import EQUIVALENCE_DELTA as DELTA

    rng = np.random.default_rng(0)
    tight = rng.normal(0.0, 0.002, 30)
    wide = rng.normal(0.05, 0.002, 30)
    for d, equivalent in ((tight, True), (wide, False)):
        lower = _wilcoxon_p(d + DELTA, alternative="greater")
        upper = _wilcoxon_p(DELTA - d, alternative="greater")
        assert (max(lower, upper) < 0.05) is equivalent


def test_a_slope_needs_three_rungs():
    from bpneat.v6.analysis import MIN_RUNGS_FOR_A_SLOPE

    assert MIN_RUNGS_FOR_A_SLOPE == 3
    assert len(BUDGETS) > MIN_RUNGS_FOR_A_SLOPE


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def test_v6_fingerprint_set_is_disjoint_and_complete():
    pkg = Path(__file__).resolve().parents[1] / "src" / "bpneat" / "v6"
    on_disk = {p.name for p in pkg.glob("*.py")} - {"__init__.py"}
    non_science = {"analysis.py", "figures.py", "release.py", "verify.py",
                   "finaltest.py", "suite.py", "run.py", "fingerprint.py"}
    assert set(V6_SCIENCE_MODULES) == on_disk - non_science
    # v6 has no searcher of its own: the thing under study is the budget, so the
    # searcher is v5's released module, byte for byte.
    assert "search.py" not in on_disk
    fp = fingerprints()
    assert fp["v5_frozen"] == v5_fingerprint()["combined"]
    assert fp["v4_frozen"] == v4_fingerprint()["combined"]
    assert fp["v3_frozen"] == v3_fingerprint()["combined"]
    assert fp["v2_frozen"] == v2_fingerprint()["combined"]
    assert len({fp["v6"], fp["v5_frozen"], fp["v4_frozen"],
                fp["v3_frozen"], fp["v2_frozen"]}) == 5


def test_the_older_fingerprints_are_the_released_ones():
    fp = fingerprints()
    assert fp["v2_frozen"].startswith("cfdf1fa3198adc0e")
    assert fp["v3_frozen"].startswith("8438c9e89c7c72a3")
    assert fp["v4_frozen"].startswith("cd9d0c1f60b44c8d")
    assert fp["v5_frozen"].startswith("33f2c177be4b5335")
