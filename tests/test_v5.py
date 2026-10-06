"""v5 correctness gates. These must pass before any compute is spent on v5.

v5's whole point is that each condition changes exactly one NEAT mechanism, so
the gates are mostly about that: a factor must be reachable, must actually do
something, and must not drag anything else with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.evolve import P_ADD_CONNECTION, P_ADD_NODE
from bpneat.record import SCIENCE_MODULES as V2_MODULES
from bpneat.record import code_fingerprint as v2_fingerprint
from bpneat.record import deserialise_genome
from bpneat.v3.datasets import make_bundle
from bpneat.v3.fingerprint import v3_fingerprint
from bpneat.v4.fingerprint import v4_fingerprint
from bpneat.v5.conditions import (
    CONDITIONS,
    DEEP_FACTOR,
    HIGH_P_ADD_NODE,
    NARROW_POPULATION,
    base_config,
    config_for,
    run_condition,
)
from bpneat.v5.fingerprint import V5_SCIENCE_MODULES, fingerprints
from bpneat.v5.protocol import (
    ALL_CONDITIONS,
    ALL_TASKS,
    BURNED_SEEDS,
    FAMILY_CONDITIONS,
    FAMILY_SIZE,
    REFERENCE,
    REPLICATES,
    cells,
    planned_runs,
    validate,
)
from bpneat.v5.search import REF_P_ADD_CONNECTION, REF_P_ADD_NODE, V5Config, search

RUNS = 720
CELLS = 90


# --------------------------------------------------------------------------
# Protocol identity
# --------------------------------------------------------------------------


def test_protocol_validates():
    validate()


def test_plan_size_and_cells():
    assert len(planned_runs()) == RUNS
    assert len(cells()) == CELLS
    assert len({(p["task"], p["condition"], p["replicate"]) for p in planned_runs()}) == RUNS


def test_the_two_saturated_geometries_are_excluded_by_contract():
    """v4 showed XOR and circles separate nothing; v5 drops them in advance."""
    assert ALL_TASKS == ("spiral", "checkerboard", "spiral3")
    assert "xor" not in ALL_TASKS and "circle" not in ALL_TASKS


def test_no_replicate_seed_was_spent_before():
    seeds = {r.dataset_seed for r in REPLICATES} | {r.search_seed for r in REPLICATES}
    assert not (seeds & BURNED_SEEDS)
    # v4's thirty replicates are burned, so v5's sealed test is fresh again.
    assert 50_001 in BURNED_SEEDS and 60_001 in BURNED_SEEDS


def test_family_is_declared_in_advance():
    assert REFERENCE not in FAMILY_CONDITIONS
    assert set(ALL_CONDITIONS) == {REFERENCE} | set(FAMILY_CONDITIONS)
    assert FAMILY_SIZE == len(FAMILY_CONDITIONS) * len(ALL_TASKS)
    assert FAMILY_SIZE == 21  # written out, so widening the family fails loudly


# --------------------------------------------------------------------------
# The reference condition must be the reference
# --------------------------------------------------------------------------


def test_the_reference_configuration_matches_the_frozen_constants():
    """v5's reference arm must be the algorithm as frozen code runs it."""
    assert REF_P_ADD_NODE == P_ADD_NODE
    assert REF_P_ADD_CONNECTION == P_ADD_CONNECTION
    for task in ALL_TASKS:
        cfg = config_for(REFERENCE, task)
        assert cfg.p_add_node == P_ADD_NODE
        assert cfg.p_add_connection == P_ADD_CONNECTION
        assert cfg.use_penalty is True
        assert cfg.crossover is True
        assert cfg.n_species == 5
        assert cfg.population == 100


def test_v5_mutate_matches_the_frozen_mutate_at_reference_rates():
    """At the reference rates, v5's mutate must be the frozen one, step for step."""
    from bpneat.evolve import Individual, InnovationRegistry, SearchConfig
    from bpneat.evolve import mutate as frozen_mutate
    from bpneat.genome import logistic_genome
    from bpneat.v5.search import mutate as v5_mutate

    for seed in range(12):
        g = logistic_genome(np.random.default_rng(seed))
        a = Individual(genome=g.copy(), weights=np.array(g.weight, dtype=np.float64))
        b = Individual(genome=g.copy(), weights=np.array(g.weight, dtype=np.float64))
        reg_a, reg_b = InnovationRegistry(), InnovationRegistry()
        for _ in range(6):
            frozen_mutate(a, np.random.default_rng(100 + seed), reg_a,
                          SearchConfig(task="spiral"))
            v5_mutate(b, np.random.default_rng(100 + seed), reg_b,
                      base_config("spiral"))
        assert a.genome.ops == b.genome.ops
        assert a.genome.src == b.genome.src and a.genome.dst == b.genome.dst
        assert np.allclose(a.weights, b.weights)


# --------------------------------------------------------------------------
# Each factor must be reachable and must do something
# --------------------------------------------------------------------------


def test_every_condition_changes_exactly_what_it_claims():
    ref = config_for(REFERENCE, "spiral")
    expected = {
        "neat_complexify": {"p_add_node", "p_add_connection"},
        "neat_no_penalty": {"use_penalty"},
        "neat_complexify_no_penalty": {"p_add_node", "p_add_connection", "use_penalty"},
        "neat_no_speciation": {"n_species"},
        "neat_no_crossover": {"crossover"},
        # The budget is held, so the width/depth arm moves both of its terms.
        "neat_deep_narrow": {"population", "generations"},
    }
    for cond, fields in expected.items():
        cfg = config_for(cond, "spiral")
        changed = {
            f for f in ("p_add_node", "p_add_connection", "use_penalty", "n_species",
                        "crossover", "population", "generations")
            if getattr(cfg, f) != getattr(ref, f)
        }
        assert changed == fields, f"{cond} changed {changed}, expected {fields}"


def test_the_width_depth_arm_holds_the_candidate_budget():
    for task in ALL_TASKS:
        ref = config_for(REFERENCE, task)
        deep = config_for("neat_deep_narrow", task)
        assert deep.candidate_budget == ref.candidate_budget
        assert deep.population == NARROW_POPULATION
        assert deep.generations + 1 == (ref.generations + 1) * DEEP_FACTOR


def test_raising_the_node_rate_actually_adds_more_nodes():
    """The factor must bite, or the arm tests nothing."""
    bundle = make_bundle("spiral", seed=9003)
    low = search(bundle, V5Config(task="spiral", generations=4, population=20,
                                  inner_steps=20, p_add_node=REF_P_ADD_NODE), seed=19003)
    high = search(bundle, V5Config(task="spiral", generations=4, population=20,
                                   inner_steps=20, p_add_node=HIGH_P_ADD_NODE), seed=19003)
    assert high.node_additions > low.node_additions


def test_disabling_crossover_stops_crossover():
    bundle = make_bundle("spiral", seed=9003)
    on = search(bundle, V5Config(task="spiral", generations=3, population=20,
                                 inner_steps=20), seed=19003)
    off = search(bundle, V5Config(task="spiral", generations=3, population=20,
                                  inner_steps=20, crossover=False), seed=19003)
    assert on.crossovers > 0
    assert off.crossovers == 0


def test_removing_speciation_leaves_one_species():
    bundle = make_bundle("spiral", seed=9003)
    res = search(bundle, V5Config(task="spiral", generations=3, population=20,
                                  inner_steps=20, n_species=1), seed=19003)
    assert all(h["n_species"] == 1 for h in res.history)


def test_the_penalty_flag_reaches_the_fitness():
    from bpneat.learn import fitness_from_error
    from bpneat.v5.search import V5Config as C

    assert C(task="spiral").use_penalty is True
    assert C(task="spiral", use_penalty=False).use_penalty is False
    g, _ = deserialise_genome(
        run_condition(REFERENCE, "spiral", 99, 9003, 19003)["champion"]
    )
    assert fitness_from_error(g, 0.5, True) != fitness_from_error(g, 0.5, False)


def test_config_rejects_impossible_factor_values():
    for bad in ({"p_add_node": 1.5}, {"p_add_connection": -0.1}, {"n_species": 0}):
        with pytest.raises(ValueError):
            V5Config(task="spiral", **bad)


# --------------------------------------------------------------------------
# Runs and records
# --------------------------------------------------------------------------


def test_search_never_reads_the_sealed_test_split():
    clean = make_bundle("spiral", seed=9003)
    poisoned = make_bundle("spiral", seed=9003)
    poisoned.test.X[:] = np.nan
    poisoned.test.y[:] = np.nan
    cfg = V5Config(task="spiral", generations=3, population=16, inner_steps=20)
    a = search(clean, cfg, seed=19003)
    b = search(poisoned, cfg, seed=19003)
    assert a.champion.fitness == b.champion.fitness
    assert np.isfinite(a.champion.fitness)


def test_search_is_deterministic_given_its_seed():
    bundle = make_bundle("checkerboard", seed=9003)
    cfg = V5Config(task="checkerboard", generations=3, population=16, inner_steps=20)
    a = search(bundle, cfg, seed=19003)
    b = search(bundle, cfg, seed=19003)
    assert a.champion.fitness == b.champion.fitness
    assert a.gradient_steps == b.gradient_steps
    assert np.array_equal(a.champion.weights, b.champion.weights)


def test_run_condition_stores_a_loadable_champion():
    from bpneat.learn import accuracy

    rec = run_condition(REFERENCE, "spiral", 99, 9003, 19003)
    blob = json.loads(json.dumps(rec))
    g, w = deserialise_genome(blob["champion"])
    bundle = make_bundle("spiral", seed=9003)
    got = accuracy(g, w, bundle.validation.X, bundle.validation.y, True)
    assert got == pytest.approx(rec["metrics"]["validation_accuracy"], abs=1e-12)
    assert rec["test_evaluated"] is False
    assert rec["history"], "the size trajectory is a primary series, not a diagnostic"
    assert "mean_nodes" in rec["history"][0]


def test_the_matched_yardstick_refuses_to_run_without_the_reference_budget():
    with pytest.raises(ValueError, match=REFERENCE):
        run_condition("fixed_mixed_matched", "spiral", 99, 9003, 19003)


def test_every_planned_condition_is_defined():
    assert {p["condition"] for p in planned_runs()} == set(CONDITIONS)
    assert set(CONDITIONS) == set(ALL_CONDITIONS)


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def test_v5_fingerprint_set_is_disjoint_and_complete():
    pkg = Path(__file__).resolve().parents[1] / "src" / "bpneat" / "v5"
    on_disk = {p.name for p in pkg.glob("*.py")} - {"__init__.py"}
    non_science = {"analysis.py", "figures.py", "release.py", "verify.py",
                   "finaltest.py", "suite.py", "run.py", "fingerprint.py"}
    assert set(V5_SCIENCE_MODULES) == on_disk - non_science
    fp = fingerprints()
    assert fp["v4_frozen"] == v4_fingerprint()["combined"]
    assert fp["v3_frozen"] == v3_fingerprint()["combined"]
    assert fp["v2_frozen"] == v2_fingerprint()["combined"]
    assert len({fp["v5"], fp["v4_frozen"], fp["v3_frozen"], fp["v2_frozen"]}) == 4


def test_the_older_fingerprints_are_the_released_ones():
    fp = fingerprints()
    assert fp["v2_frozen"].startswith("cfdf1fa3198adc0e")
    assert fp["v3_frozen"].startswith("8438c9e89c7c72a3")
    assert fp["v4_frozen"].startswith("cd9d0c1f60b44c8d")
    assert set(V2_MODULES) <= set(V2_MODULES)
