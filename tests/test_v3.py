"""Pre-freeze gates for protocol v3.

These mirror the v2 gates and add the one v3 depends on: the vectorized dense
evaluator must be the *same model* as the frozen genome path, not merely a
similar one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.baselines import make_mlp
from bpneat.genome import (
    OP_GAUSSIAN,
    OP_RELU,
    OP_SIN,
    OP_TANH,
)
from bpneat.genome import (
    backward as genome_backward,
)
from bpneat.genome import (
    forward as genome_forward,
)
from bpneat.learn import _sigmoid
from bpneat.record import code_fingerprint
from bpneat.v3 import dense
from bpneat.v3.conditions import CONDITIONS, run_condition
from bpneat.v3.datasets import TASKS, make_bundle
from bpneat.v3.fingerprint import V3_SCIENCE_MODULES, fingerprints
from bpneat.v3.learners import mixed_mlp, plain_train
from bpneat.v3.protocol import BLOCKS, BURNED_SEEDS, REPLICATES, cells, planned_runs, validate
from bpneat.v3.search import V3Config, search
from bpneat.v3.selection import SELECTORS

TOL = 1e-10


# ---------------------------------------------------------------- dense path


@pytest.mark.parametrize(
    "hidden,op",
    [((32, 32), OP_TANH), ((16,), OP_SIN), ((8, 8, 8), OP_RELU), ((12, 12), OP_GAUSSIAN)],
)
def test_dense_matches_genome_forward_and_gradient(hidden, op):
    """The optimisation must be the same model, to 1e-10, values and gradients."""
    b = make_bundle("spiral", seed=30001)
    X, y = b.train.X[:32], b.train.y[:32]
    g = make_mlp(hidden, np.random.default_rng(1), op=op)
    noise = np.random.default_rng(7).normal(0, 0.4, g.n_connections)
    w = np.array(g.weight, dtype=np.float64) + noise

    tape = genome_forward(g, X, w, settle=True)
    z_slow = tape.vals[tape.out_var]
    plan = dense.plan(g)
    dtape = dense.forward(plan, w, X)
    z_fast = dtape.V[:, 3]
    assert np.max(np.abs(z_slow - z_fast)) < TOL

    d = (_sigmoid(z_slow) - y) / len(X)
    g_slow = genome_backward(tape, d, w)
    g_fast = dense.backward(plan, w, dtape, d)
    rel = np.abs(g_slow - g_fast) / np.maximum(1.0, np.abs(g_slow))
    assert np.max(rel) < TOL


def test_dense_matches_on_mixed_operators():
    b = make_bundle("checkerboard", seed=30001)
    X, y = b.train.X[:24], b.train.y[:24]
    g = mixed_mlp((16, 16), np.random.default_rng(3))
    noise = np.random.default_rng(5).normal(0, 0.3, g.n_connections)
    w = np.array(g.weight, dtype=np.float64) + noise
    tape = genome_forward(g, X, w, settle=True)
    plan = dense.plan(g)
    dtape = dense.forward(plan, w, X)
    assert np.max(np.abs(tape.vals[tape.out_var] - dtape.V[:, 3])) < TOL
    d = (_sigmoid(tape.vals[tape.out_var]) - y) / len(X)
    rel = np.abs(genome_backward(tape, d, w) - dense.backward(plan, w, dtape, d))
    assert np.max(rel / np.maximum(1.0, np.abs(genome_backward(tape, d, w)))) < TOL


def test_dense_refuses_a_mult_genome():
    from bpneat.evolve import Individual, InnovationRegistry, add_node
    from bpneat.genome import OP_MULT, logistic_genome

    rng = np.random.default_rng(0)
    g = logistic_genome(rng)
    ind = Individual(genome=g, weights=np.array(g.weight))
    add_node(ind, rng, InnovationRegistry())
    ind.genome.ops[-1] = OP_MULT
    with pytest.raises(dense.NotDenseable):
        dense.plan(ind.genome)


def test_plain_train_is_deterministic_and_runs_every_step():
    b = make_bundle("spiral", seed=30001)
    g = make_mlp((8, 8), np.random.default_rng(1))
    w0 = np.array(g.weight, dtype=np.float64)
    a = plain_train(g, w0, b.train.X, b.train.y, np.random.default_rng(2), n_steps=50)
    c = plain_train(g, w0, b.train.X, b.train.y, np.random.default_rng(2), n_steps=50)
    assert np.array_equal(a, c)
    assert not np.allclose(a, w0)


def test_dense_and_genome_training_agree():
    """Same seed, same steps, dense on or off — identical weights."""
    b = make_bundle("xor", seed=30001)
    g = make_mlp((6,), np.random.default_rng(1))
    w0 = np.array(g.weight, dtype=np.float64)
    fast = plain_train(g, w0, b.train.X, b.train.y, np.random.default_rng(4), 60, use_dense=True)
    slow = plain_train(g, w0, b.train.X, b.train.y, np.random.default_rng(4), 60, use_dense=False)
    assert np.max(np.abs(fast - slow)) < TOL


# ---------------------------------------------------------------- protocol


def test_v3_protocol_is_self_consistent():
    validate()
    seeds = {r.dataset_seed for r in REPLICATES} | {r.search_seed for r in REPLICATES}
    assert not (seeds & BURNED_SEEDS)
    assert len(REPLICATES) == 30


def test_plan_dedupes_the_shared_reference_cell():
    plan = planned_runs()
    ids = [(p["task"], p["condition"], p["replicate"]) for p in plan]
    assert len(ids) == len(set(ids))
    # backprop_neat belongs to all four blocks but is planned once per cell.
    n_ref = sum(1 for p in plan if p["condition"] == "backprop_neat")
    assert n_ref == len(cells())


def test_every_block_condition_exists():
    for b in BLOCKS:
        for c in b.conditions:
            assert c in CONDITIONS, f"block {b.name} names unknown condition {c}"


@pytest.mark.parametrize("task", TASKS)
def test_v3_tasks_are_balanced_and_independent(task):
    b = make_bundle(task, seed=30001)
    for split in (b.train, b.validation, b.test):
        assert set(np.unique(split.y)) <= {0.0, 1.0}
        assert 0.3 < split.y.mean() < 0.7
    assert not np.allclose(b.train.X, b.validation.X)
    assert not np.allclose(b.train.X, b.test.X)


# ---------------------------------------------------------------- selection


def test_selection_probabilities_are_distributions():
    f = -np.abs(np.random.default_rng(0).normal(0.5, 0.25, 40))
    for s in SELECTORS.values():
        p = s.probabilities(f)
        assert p.shape == f.shape
        assert np.all(p >= 0)
        assert abs(p.sum() - 1.0) < 1e-12


def test_selection_intensity_orders_the_sweep():
    """Weaker slack must mean weaker pressure; the sweep must actually span."""
    f = -np.abs(np.random.default_rng(1).normal(0.5, 0.25, 100))
    i = {n: s.intensity(f) for n, s in SELECTORS.items()}
    assert i["roulette_s1.0"] < i["roulette_s0.1"] < i["roulette_s0.01"] < i["roulette_s0.001"]
    assert i["tournament_k2"] < i["tournament_k4"]
    assert max(i.values()) > 4 * min(i.values())


# ---------------------------------------------------------------- isolation


def test_v3_search_never_reads_the_sealed_test_split():
    clean = make_bundle("xor", seed=30001)
    poisoned = make_bundle("xor", seed=30001)
    poisoned.test.X[:] = np.nan
    poisoned.test.y[:] = np.nan
    cfg = V3Config(task="xor", generations=1, population=8, inner_steps=20)
    a = search(clean, cfg, seed=40001)
    b = search(poisoned, cfg, seed=40001)
    assert a.metrics["validation_loss"] == pytest.approx(b.metrics["validation_loss"])
    assert np.isfinite(a.metrics["validation_loss"])


@pytest.mark.parametrize("condition", ["backprop_neat", "fixed_mlp_tanh_ha", "evolution_only"])
def test_v3_records_carry_no_test_metrics(condition):
    rec = run_condition(condition, "xor", 1, 30001, 40001)
    blob = json.dumps(rec)
    assert "test_accuracy" not in blob and "test_loss" not in blob
    assert rec["test_evaluated"] is False


def test_matched_condition_refuses_without_a_budget():
    with pytest.raises(ValueError, match="budget"):
        run_condition("fixed_mlp_tanh_matched", "xor", 1, 30001, 40001)


# ---------------------------------------------------------------- fingerprints


def test_v2_frozen_fingerprint_is_unchanged():
    """R3: v3 may never edit the seven v2 science modules."""
    assert code_fingerprint()["combined"] == (
        "cfdf1fa3198adc0e369466800ede0ea5afa24d99af18000a2161357ae5ec0d75"
    )


def test_v3_fingerprint_covers_its_science_modules():
    fp = fingerprints()
    assert fp["v3"] != fp["v2_frozen"]
    root = Path(__file__).resolve().parents[1] / "src" / "bpneat" / "v3"
    for name in V3_SCIENCE_MODULES:
        assert (root / name).exists(), name
    # Suite, analysis and CLI code must not be fingerprint-bound.
    assert "suite.py" not in V3_SCIENCE_MODULES
    assert "finaltest.py" not in V3_SCIENCE_MODULES


# ---------------------------------------------------------------- firewall


def _tiny_release(tmp_path, monkeypatch):
    import bpneat.v3.suite as S

    one = [{"task": "xor", "condition": "evolution_only", "replicate": 1,
            "dataset_seed": 30001, "search_seed": 40001}]
    monkeypatch.setattr(S, "planned_runs", lambda: one)
    monkeypatch.setattr(S, "cells", lambda: [("xor", 1)])
    return S.run_cells(tmp_path, 0, 1, progress=lambda *_: None)


def test_v3_final_test_runs_once_then_refuses(tmp_path, monkeypatch):
    from bpneat.v3.finaltest import FinalTestRefused, run_final_test

    m = _tiny_release(tmp_path, monkeypatch)
    assert m["complete"], m
    rel = run_final_test(tmp_path, progress=lambda *_: None)
    assert rel["n_runs"] == 1
    assert 0.0 <= rel["results"][0]["test_accuracy"] <= 1.0
    with pytest.raises(FinalTestRefused, match="runs once"):
        run_final_test(tmp_path, progress=lambda *_: None)


def test_v3_final_test_refuses_an_incomplete_suite(tmp_path, monkeypatch):
    from bpneat.v3.finaltest import FinalTestRefused, run_final_test

    _tiny_release(tmp_path, monkeypatch)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["complete"] = False
    m["missing"] = ["xor__backprop_neat__r01"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(FinalTestRefused, match="incomplete"):
        run_final_test(tmp_path, progress=lambda *_: None)


def test_v3_final_test_refuses_changed_science_code(tmp_path, monkeypatch):
    from bpneat.v3.finaltest import FinalTestRefused, run_final_test

    _tiny_release(tmp_path, monkeypatch)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["fingerprints"]["v3"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(FinalTestRefused, match="fingerprint"):
        run_final_test(tmp_path, progress=lambda *_: None)


def test_v3_suite_resume_is_exact(tmp_path, monkeypatch):
    """Re-running a finished shard must change nothing on disk."""

    _tiny_release(tmp_path, monkeypatch)
    path = next((tmp_path / "raw" / "runs").glob("*.json"))
    before = path.read_bytes()
    m = _tiny_release(tmp_path, monkeypatch)
    assert m["this_shard"]["skipped"] == 1
    assert path.read_bytes() == before
