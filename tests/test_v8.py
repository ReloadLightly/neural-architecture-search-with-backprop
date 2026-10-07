"""v8 correctness gates. These must pass before any compute is spent on v8.

v8 is the first protocol built on the n-dimensional core rather than directly on
the frozen modules, and the first to run on data that is finite rather than
generated. Both of those change what can go wrong, so most of what follows is
about them: that a champion is read back at the width it was found at, that a
split is a partition of specific rows and the sealed part of it is sealed, and
that the one arm with no counterpart in any earlier protocol — the linear model
— really is a model with no architecture.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.nd import encoding as nd
from bpneat.nd.datasets import TASKS as ND_TASKS
from bpneat.nd.datasets import make_bundle
from bpneat.nd.evolve import layout_for
from bpneat.nd.fingerprint import nd_fingerprint
from bpneat.record import code_fingerprint as v2_fingerprint
from bpneat.v3.fingerprint import v3_fingerprint
from bpneat.v5.conditions import base_config as v5_base_config
from bpneat.v5.fingerprint import v5_fingerprint
from bpneat.v8.conditions import CONDITIONS, config_for, run_condition
from bpneat.v8.fingerprint import V8_SCIENCE_MODULES, data_fingerprint, fingerprints
from bpneat.v8.protocol import (
    ALL_TASKS,
    ARM_ORDER,
    ARMS,
    BURNED_SEEDS,
    EQUIVALENCE_DELTA,
    EVERY_TASK,
    EXTENSION_TASKS,
    FAMILIES,
    FAMILY_SIZE,
    FIXED_HIDDEN,
    MATCHED_TO,
    MULTISTART_RESTARTS,
    REFERENCE,
    REFERENCE_CANDIDATES,
    REPLICATES,
    TEST_ROWS,
    cells,
    planned_runs,
    validate,
)
from bpneat.v8.suite import _order_key, _plan_by_cell, shard_cells

PILOT_SPLIT, PILOT_SEARCH = 9005, 19005
CONFIRMATORY_RUNS = 360
EVERY_RUN = 480
CELLS = 90
V8_DIR = Path(__file__).resolve().parents[1] / "src" / "bpneat" / "v8"


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


def test_the_datasets_are_the_vendored_ones_and_the_split_is_declared():
    assert ALL_TASKS == ("iris", "wine", "breast_cancer")
    assert EXTENSION_TASKS == ("digits",)
    assert set(EVERY_TASK) <= set(ND_TASKS)
    assert not (set(ALL_TASKS) & set(EXTENSION_TASKS))


def test_the_recorded_test_split_sizes_are_the_real_ones():
    """The equivalence margin is justified by these, so they cannot be stale."""
    for task in EVERY_TASK:
        assert len(make_bundle(task, seed=1).test) == TEST_ROWS[task], task


def test_the_equivalence_margin_is_a_margin_the_data_can_carry():
    finest = 1.0 / min(TEST_ROWS[t] for t in ALL_TASKS)
    assert EQUIVALENCE_DELTA >= finest, (
        f"delta {EQUIVALENCE_DELTA} is finer than one test row ({finest:.4f})"
    )
    # And not so wide that it would declare equivalence between anything.
    assert EQUIVALENCE_DELTA <= 0.05


def test_no_replicate_seed_was_spent_before():
    seeds = {r.split_seed for r in REPLICATES} | {r.search_seed for r in REPLICATES}
    assert not (seeds & BURNED_SEEDS)
    assert 90_001 in BURNED_SEEDS and 100_001 in BURNED_SEEDS  # v6's
    assert PILOT_SPLIT in BURNED_SEEDS and PILOT_SEARCH in BURNED_SEEDS


def test_families_are_declared_in_advance_and_written_out():
    assert FAMILY_SIZE == {
        "vs_linear": 9,
        "vs_null": 3,
        "vs_fixed": 3,
        "equivalence_with_linear": 3,
        "champion_size": 3,
    }
    for name, members in FAMILIES.items():
        assert len(set(members)) == len(members) == FAMILY_SIZE[name], name
        assert {task for task, _ in members} == set(ALL_TASKS), name


def test_the_extension_dataset_scores_nothing():
    for name, members in FAMILIES.items():
        assert not ({task for task, _ in members} & set(EXTENSION_TASKS)), name


# --------------------------------------------------------------------------
# The four arms
# --------------------------------------------------------------------------


def test_the_arms_are_the_four_declared_ones():
    assert ARMS == ("search", "null", "linear", "fixed")
    assert set(CONDITIONS) == set(ARMS)
    assert REFERENCE == "search"
    for matched, target in MATCHED_TO.items():
        assert ARM_ORDER[target] < ARM_ORDER[matched]


def test_the_search_arm_is_v5s_reference_configuration_at_this_width():
    """Only the layout and the data may differ from the configuration v6 anchors on."""
    reference = v5_base_config("spiral")
    for task in EVERY_TASK:
        bundle = make_bundle(task, seed=1)
        cfg = config_for(task, bundle)
        assert cfg.layout == layout_for(bundle)
        assert cfg.candidate_budget == REFERENCE_CANDIDATES
        for field in ("population", "n_species", "inner_steps", "batch_size",
                      "use_penalty", "crossover", "p_add_node", "p_add_connection",
                      "propagation", "fitness_split", "selector", "lamarckian"):
            assert getattr(cfg, field) == getattr(reference, field), (task, field)


def test_a_binary_dataset_gets_one_logit_and_a_multiclass_one_gets_k():
    for task in EVERY_TASK:
        bundle = make_bundle(task, seed=1)
        layout = layout_for(bundle)
        assert layout.n_inputs == bundle.n_features
        assert layout.n_outputs == (1 if bundle.n_classes == 2 else bundle.n_classes)


def test_a_matched_arm_refuses_a_budget_it_was_not_matched_to():
    for arm in ("linear", "fixed"):
        with pytest.raises(ValueError, match="search"):
            run_condition(arm, "iris", 0, PILOT_SPLIT, PILOT_SEARCH)
        with pytest.raises(ValueError, match="search"):
            run_condition(arm, "iris", 0, PILOT_SPLIT, PILOT_SEARCH,
                          matched_steps={"null": 1000})


def test_the_linear_arm_has_no_architecture_at_all():
    """The whole point of the arm: no hidden units, on any dataset."""
    for task in EVERY_TASK:
        bundle = make_bundle(task, seed=1)
        layout = layout_for(bundle)
        rec = run_condition("linear", task, 0, 1, PILOT_SEARCH,
                            matched_steps={"search": 600})
        g, _ = nd.deserialise(rec["champion"], layout)
        assert g.n_nodes == layout.n_structural, f"{task}: linear arm grew a node"
        assert rec["metrics"]["causal_hidden_nodes"] == 0
        assert rec["config"]["restarts"] == MULTISTART_RESTARTS
        assert rec["config"]["matched_to"] == "search"


def test_the_fixed_arm_is_the_declared_network_at_the_given_budget():
    rec = run_condition("fixed", "iris", 0, 1, PILOT_SEARCH,
                        matched_steps={"search": 3000})
    assert rec["config"]["fixed_hidden"] == list(FIXED_HIDDEN)
    assert rec["config"]["matched_steps"] == 3000
    assert rec["compute"]["gradient_steps"] == pytest.approx(3000, rel=0.03)
    assert rec["compute"]["candidate_evaluations"] == MULTISTART_RESTARTS


def test_a_record_carries_the_width_its_champion_was_found_at():
    """A champion read back as a two-input graph would score, silently wrong."""
    for task in ("iris", "breast_cancer"):
        bundle = make_bundle(task, seed=1)
        rec = run_condition("linear", task, 0, 1, PILOT_SEARCH,
                            matched_steps={"search": 600})
        blob = json.loads(json.dumps(rec))
        assert blob["champion_layout"]["n_inputs"] == bundle.n_features
        layout = nd.Layout(**blob["champion_layout"])
        g, w = nd.deserialise(blob["champion"], layout)
        from bpneat.nd.learn import accuracy

        got = accuracy(g, w, bundle.validation.X, bundle.validation.y, True)
        assert got == pytest.approx(rec["metrics"]["validation_accuracy"], abs=1e-12)
        assert rec["test_evaluated"] is False
        assert rec["config"]["n_features"] == bundle.n_features
        assert rec["config"]["sealed_test_rows"] == TEST_ROWS[task]


def test_run_condition_refuses_an_arm_it_does_not_have():
    with pytest.raises(ValueError, match="unknown arm"):
        run_condition("hillclimb", "iris", 0, 1, 2)


# --------------------------------------------------------------------------
# Sharding, ordering, and the sealed split
# --------------------------------------------------------------------------


def test_shards_partition_the_cells():
    for total in (1, 3, 4, 7):
        seen = [c for i in range(total) for c in shard_cells(i, total)]
        assert sorted(seen) == sorted(cells())
    with pytest.raises(ValueError):
        shard_cells(4, 4)


def test_a_cell_runs_the_search_before_the_arms_matched_to_it():
    items = sorted(_plan_by_cell(True)[("iris", 1)], key=_order_key)
    order = [p["condition"] for p in items]
    assert order == list(ARMS)
    for matched, target in MATCHED_TO.items():
        assert order.index(target) < order.index(matched)


def test_cells_are_replicate_major():
    first = cells()[: len(ALL_TASKS)]
    assert {t for t, _ in first} == set(ALL_TASKS)
    assert {r for _, r in first} == {1}


def test_only_the_firewall_reads_the_sealed_split():
    """The same static check the n-dimensional core gets, for the protocol."""
    allowed = {"finaltest.py"}
    offenders = []
    # Deliberately blunt: any attribute named ``test`` outside the firewall
    # fails, including ``len(bundle.test)``. A rule that allowed "only the
    # length" would be the first step down a slope, and the number that is
    # actually wanted — how many rows the sealed split holds — is in the
    # contract, where a gate pins it to the loader.
    for path in sorted(V8_DIR.glob("*.py")):
        if path.name in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == "test":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"sealed split read at {offenders}"


def test_the_manifest_does_not_call_a_missing_extension_dataset_incomplete(tmp_path):
    from bpneat.v8.suite import _manifest

    runs = tmp_path / "raw" / "runs"
    runs.mkdir(parents=True)
    for p in planned_runs():
        (runs / f"{p['task']}__{p['condition']}__r{p['replicate']:02d}.json").write_text("{}")
    m = _manifest(tmp_path, runs, fingerprints(), [], 0.0, 0, 0)
    assert m["complete"] is True
    assert m["missing"] == [] and m["unexpected"] == []
    assert m["extension_present"] == 0
    assert m["extension_planned"] == EVERY_RUN - CONFIRMATORY_RUNS
    assert sorted(m["datasets"]) == sorted(ALL_TASKS)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def test_holm_corrects_at_the_declared_family_size():
    from bpneat.v8.analysis import _holm

    rows = [{"p": 0.01}, {"p": 0.02}]
    _holm(rows, "p", 9)
    assert rows[0]["holm_p"] == pytest.approx(0.09)
    assert not any(r["holm_significant"] for r in rows)


def test_equivalence_declares_equivalence_only_inside_the_margin():
    from bpneat.v8.analysis import _wilcoxon_p

    rng = np.random.default_rng(0)
    for spread, equivalent in ((0.002, True), (0.08, False)):
        d = rng.normal(spread * 25 if not equivalent else 0.0, spread, 30)
        lower = _wilcoxon_p(d + EQUIVALENCE_DELTA, alternative="greater")
        upper = _wilcoxon_p(EQUIVALENCE_DELTA - d, alternative="greater")
        assert (max(lower, upper) < 0.05) is equivalent


def test_only_the_primary_outcome_is_a_scored_family():
    import inspect

    from bpneat.v8.analysis import arm_contrasts

    src = inspect.getsource(arm_contrasts)
    assert 'scored = metric == "test_accuracy"' in src
    assert "if scored_members and scored:" in src


def test_every_declared_contrast_is_one_the_analysis_computes():
    from bpneat.v8.analysis import CONTRASTS

    assert set(CONTRASTS) == {"vs_linear", "vs_null", "vs_fixed"}
    for family, pairs in CONTRASTS.items():
        assert len(pairs) * len(ALL_TASKS) == FAMILY_SIZE[family], family
        for left, right in pairs:
            assert left in ARMS and right in ARMS


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def test_v8_fingerprint_set_is_disjoint_and_complete():
    on_disk = {p.name for p in V8_DIR.glob("*.py")} - {"__init__.py"}
    non_science = {"analysis.py", "figures.py", "release.py", "verify.py",
                   "finaltest.py", "suite.py", "run.py", "fingerprint.py"}
    assert set(V8_SCIENCE_MODULES) == on_disk - non_science
    fp = fingerprints()
    assert fp["nd"] == nd_fingerprint()["combined"]
    assert fp["v5_frozen"] == v5_fingerprint()["combined"]
    assert fp["v3_frozen"] == v3_fingerprint()["combined"]
    assert fp["v2_frozen"] == v2_fingerprint()["combined"]
    assert len({fp["v8"], fp["data"], fp["nd"], fp["v5_frozen"],
                fp["v3_frozen"], fp["v2_frozen"]}) == 6


def test_the_data_is_fingerprinted_by_content_not_by_loader():
    """A release built on a dataset that later changed must not look sound."""
    fp = data_fingerprint()
    assert set(fp["files"]) == set(ND_TASKS)
    assert all(len(h) == 64 for h in fp["files"].values())


def test_the_older_fingerprints_are_the_released_ones():
    fp = fingerprints()
    assert fp["v2_frozen"].startswith("cfdf1fa3198adc0e")
    assert fp["v3_frozen"].startswith("8438c9e89c7c72a3")
    assert fp["v5_frozen"].startswith("33f2c177be4b5335")
