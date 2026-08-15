"""Feasibility probe: verify gradients, then measure real throughput on this box.

Answers one question — what does a reference-budget Backprop-NEAT run actually
cost here — so the confirmatory compute forecast rests on measurement rather
than on a runtime quoted from someone else's hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.datasets import make_bundle  # noqa: E402
from bpneat.evolve import (  # noqa: E402
    Individual,
    InnovationRegistry,
    SearchConfig,
    mutate,
    search,
)
from bpneat.genome import backward, forward, logistic_genome  # noqa: E402
from bpneat.learn import train  # noqa: E402


def _loss(g, w, X, y):
    tape = forward(g, X, w)
    p = 1.0 / (1.0 + np.exp(-np.clip(tape.vals[tape.out_var], -60, 60)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradient_check() -> float:
    """Finite-difference check of backprop through the executed trace."""
    rng = np.random.default_rng(0)
    reg = InnovationRegistry()
    bundle = make_bundle("spiral", seed=1, n_train=16, n_validation=16, n_test=16)
    X, y = bundle.train.X, bundle.train.y

    worst = 0.0
    cfg = SearchConfig(task="spiral")
    for trial in range(12):
        g = logistic_genome(rng)
        ind = Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))
        for _ in range(8):  # grow a genome with varied operators
            mutate(ind, rng, reg, cfg)

        w = ind.weights.astype(np.float64)
        tape = forward(ind.genome, X, w)
        p = 1.0 / (1.0 + np.exp(-np.clip(tape.vals[tape.out_var], -60, 60)))
        d_out = (np.clip(p, 1e-12, 1 - 1e-12) - y) / len(X)
        analytic = backward(tape, d_out, w)

        eps = 1e-6
        for ci in range(len(w)):
            wp, wm = w.copy(), w.copy()
            wp[ci] += eps
            wm[ci] -= eps
            numeric = (_loss(ind.genome, wp, X, y) - _loss(ind.genome, wm, X, y)) / (2 * eps)
            denom = max(1.0, abs(numeric), abs(analytic[ci]))
            worst = max(worst, abs(numeric - analytic[ci]) / denom)

        if trial == 0:
            print(
                f"  sample genome: {ind.genome.n_nodes} nodes, "
                f"{ind.genome.n_enabled} enabled connections"
            )
    return worst


def time_single_candidate() -> None:
    """Cost of the inner learner alone, at the reference budget of 600 steps."""
    rng = np.random.default_rng(3)
    reg = InnovationRegistry()
    bundle = make_bundle("spiral", seed=11)
    cfg = SearchConfig(task="spiral")

    for label, n_mutations in (("minimal (logistic)", 0), ("grown", 24), ("large", 60)):
        g = logistic_genome(rng)
        ind = Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))
        for _ in range(n_mutations):
            mutate(ind, rng, reg, cfg)
        t0 = time.time()
        res = train(
            ind.genome, ind.weights, bundle.train.X, bundle.train.y, rng, n_cycles=600
        )
        dt = time.time() - t0
        print(
            f"  {label:20s} {ind.genome.n_nodes:3d} nodes "
            f"{ind.genome.n_enabled:4d} conns -> {dt:6.2f}s "
            f"({res.gradient_steps} realized steps)"
        )


def time_reduced_search() -> tuple[float, int]:
    """A real (small) search, to get seconds-per-candidate under evolution."""
    bundle = make_bundle("spiral", seed=7103)
    cfg = SearchConfig(task="spiral", generations=3, population=20, inner_steps=600)
    t0 = time.time()
    res = search(bundle, cfg, seed=17103)
    dt = time.time() - t0
    per = dt / max(res.candidates, 1)
    print(
        f"  {res.candidates} candidates, {res.gradient_steps} gradient steps, "
        f"{dt:.1f}s  ->  {per:.3f}s/candidate"
    )
    print(
        f"  champion: val acc {res.metrics['validation_accuracy']:.3f}, "
        f"{res.metrics['represented_nodes']} represented nodes, "
        f"{res.metrics['causal_hidden_nodes']} causal hidden nodes"
    )
    return per, res.candidates


def main() -> None:
    print("=" * 68)
    print("1. Gradient check (finite differences through the executed trace)")
    worst = gradient_check()
    status = "PASS" if worst < 1e-4 else "FAIL"
    print(f"  worst relative error: {worst:.3e}   [{status}]")

    print("\n2. Inner-learner cost at the reference budget (600 steps, batch 10)")
    time_single_candidate()

    print("\n3. Reduced search on spirals (real evolution)")
    per, _ = time_reduced_search()

    print("\n4. Forecast for the reference budget")
    for task, cands in (("xor", 1205), ("circle", 1205), ("spiral", 2305)):
        print(f"  {task:8s} {cands:5d} candidates -> {per * cands / 60:6.1f} min/run")
    print("=" * 68)


if __name__ == "__main__":
    main()
