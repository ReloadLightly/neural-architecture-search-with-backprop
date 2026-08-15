"""Correctness gates. These must pass before any compute is spent on a study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.baselines import make_mlp, make_random_architecture
from bpneat.datasets import TASKS, make_bundle
from bpneat.evolve import (
    Individual,
    InnovationRegistry,
    SearchConfig,
    mutate,
)
from bpneat.genome import (
    ACTIVATIONS,
    OUT,
    backward,
    causal_subgraph,
    forward,
    logistic_genome,
)
from bpneat.learn import _sigmoid, accuracy


def _loss(g, w, X, y, settle=True):
    tape = forward(g, X, w, settle=settle)
    z = tape.vals[tape.out_var]
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_partitions_are_independent(task):
    """Train/validation/test must be separate draws, not copies."""
    b = make_bundle(task, seed=99)
    assert len(b.train) == len(b.validation) == len(b.test) == 200
    assert not np.allclose(b.train.X, b.validation.X)
    assert not np.allclose(b.train.X, b.test.X)
    assert not np.allclose(b.validation.X, b.test.X)


@pytest.mark.parametrize("task", TASKS)
def test_labels_are_balanced_and_binary(task):
    b = make_bundle(task, seed=7)
    for split in (b.train, b.validation, b.test):
        assert set(np.unique(split.y)) <= {0.0, 1.0}
        assert 0.3 < split.y.mean() < 0.7


def test_spiral_matches_source_geometry():
    """1.75 turns on radius 6, raw unstandardised coordinates."""
    b = make_bundle("spiral", seed=3, noise=0.0)
    radii = np.linalg.norm(b.train.X, axis=1)
    assert radii.max() == pytest.approx(6.0, abs=0.3)
    assert np.abs(b.train.X).max() > 3.0  # not standardised


def test_dataset_seed_is_reproducible():
    a = make_bundle("circle", seed=11)
    b = make_bundle("circle", seed=11)
    assert np.array_equal(a.train.X, b.train.X)
    assert not np.array_equal(a.train.X, make_bundle("circle", seed=12).train.X)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def test_gradients_match_finite_differences():
    """Backprop through the executed trace, across every supported operator."""
    rng = np.random.default_rng(0)
    reg = InnovationRegistry()
    cfg = SearchConfig(task="spiral")
    b = make_bundle("spiral", seed=1, n_train=16, n_validation=16, n_test=16)
    X, y = b.train.X, b.train.y

    worst = 0.0
    for _ in range(10):
        g = logistic_genome(rng)
        ind = Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))
        for _ in range(8):
            mutate(ind, rng, reg, cfg)
        w = ind.weights.astype(np.float64)

        tape = forward(ind.genome, X, w, settle=True)
        analytic = backward(tape, (_sigmoid(tape.vals[tape.out_var]) - y) / len(X), w)

        eps = 1e-6
        for ci in range(len(w)):
            wp, wm = w.copy(), w.copy()
            wp[ci] += eps
            wm[ci] -= eps
            numeric = (_loss(ind.genome, wp, X, y) - _loss(ind.genome, wm, X, y)) / (2 * eps)
            worst = max(worst, abs(numeric - analytic[ci]) / max(1.0, abs(numeric)))
    assert worst < 1e-4, f"worst relative gradient error {worst:.2e}"


def test_every_operator_is_reachable_and_differentiable():
    rng = np.random.default_rng(4)
    reg = InnovationRegistry()
    cfg = SearchConfig(task="xor")
    seen = set()
    for _ in range(60):
        g = logistic_genome(rng)
        ind = Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))
        for _ in range(6):
            mutate(ind, rng, reg, cfg)
        seen.update(ind.genome.ops[4:])
    assert set(ACTIVATIONS) <= seen


# --------------------------------------------------------------------------
# Propagation semantics — the failure this project must not repeat
# --------------------------------------------------------------------------


def test_reference_break_rule_can_zero_a_working_network():
    """Documents *why* this project does not use Ha's break rule by default.

    A fixed MLP is touched everywhere on the first tick, so the reference rule
    stops before any hidden unit has influenced the output node.
    """
    rng = np.random.default_rng(1)
    g = make_mlp((8, 8), rng)
    X = make_bundle("spiral", seed=2).train.X[:16]
    w = np.array(g.weight, dtype=np.float64)

    ref = forward(g, X, w, settle=False)
    assert np.allclose(ref.vals[ref.out_var], 0.0)

    settled = forward(g, X, w, settle=True)
    assert not np.allclose(settled.vals[settled.out_var], 0.0)


def test_mlp_hidden_units_causally_reach_the_output():
    """Gate 2: a baseline must not be silently crippled by propagation order."""
    rng = np.random.default_rng(1)
    g = make_mlp((8, 8), rng)
    b = make_bundle("circle", seed=2)
    w = np.array(g.weight, dtype=np.float64)

    info = causal_subgraph(g, b.train.X[:16], w, settle=True)
    assert info["causal_hidden_nodes"] == 16, info


def test_perturbing_a_hidden_weight_changes_the_output():
    rng = np.random.default_rng(6)
    g = make_mlp((6,), rng)
    X = make_bundle("xor", seed=5).train.X[:8]
    w = np.array(g.weight, dtype=np.float64)
    before = forward(g, X, w, settle=True)
    base = before.vals[before.out_var].copy()

    w2 = w.copy()
    w2[2] += 3.0
    after = forward(g, X, w2, settle=True)
    assert not np.allclose(base, after.vals[after.out_var])


# --------------------------------------------------------------------------
# Test-split isolation
# --------------------------------------------------------------------------


def test_search_never_reads_the_sealed_test_split():
    """Poison the test split; a correct search must be unaffected by it."""
    from bpneat.evolve import search

    cfg = SearchConfig(task="xor", generations=1, population=8, inner_steps=20)
    clean = make_bundle("xor", seed=21)
    poisoned = make_bundle("xor", seed=21)
    poisoned.test.X[:] = np.nan
    poisoned.test.y[:] = np.nan

    a = search(clean, cfg, seed=5)
    b = search(poisoned, cfg, seed=5)
    assert a.metrics["validation_loss"] == pytest.approx(b.metrics["validation_loss"])
    assert np.isfinite(a.metrics["validation_loss"])


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_random_architecture_is_wired_and_alive():
    rng = np.random.default_rng(8)
    g = make_random_architecture(rng, n_hidden=10, n_extra_connections=8, activations=ACTIVATIONS)
    X = make_bundle("spiral", seed=4).train.X[:16]
    w = np.array(g.weight, dtype=np.float64)
    info = causal_subgraph(g, X, w, settle=True)
    assert info["causal_hidden_nodes"] > 0
    assert any(g.dst[i] == OUT for i in range(g.n_connections))


def test_optimiser_state_is_not_shared_between_genomes():
    from bpneat.learn import train

    rng = np.random.default_rng(2)
    b = make_bundle("xor", seed=13)
    g1 = make_mlp((4,), np.random.default_rng(1))
    g2 = make_mlp((4,), np.random.default_rng(1))
    r1 = train(g1, np.array(g1.weight), b.train.X, b.train.y, np.random.default_rng(3), n_cycles=40)
    r2 = train(g2, np.array(g2.weight), b.train.X, b.train.y, np.random.default_rng(3), n_cycles=40)
    assert np.allclose(r1.weights, r2.weights)


def test_accuracy_is_chance_for_an_untrained_logistic_genome():
    rng = np.random.default_rng(0)
    g = logistic_genome(rng)
    b = make_bundle("spiral", seed=1)
    a = accuracy(g, np.array(g.weight), b.validation.X, b.validation.y)
    assert 0.2 < a < 0.8
