"""What does correct propagation cost, and does evolution work once it is correct?

Settling is what makes the graphs compute, but it is also what makes them
expensive: each extra tick re-evaluates every touched node. This probe reports
ticks-to-settle against genome size, then runs a real search to check that
structure now accumulates and accuracy moves off chance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat import genome as gmod  # noqa: E402
from bpneat.baselines import make_mlp  # noqa: E402
from bpneat.datasets import make_bundle  # noqa: E402
from bpneat.evolve import (  # noqa: E402
    Individual,
    InnovationRegistry,
    SearchConfig,
    add_connection,
    add_node,
    evaluate,
    kmedoids,
    _reproduce,
)
from bpneat.genome import forward, logistic_genome  # noqa: E402
from bpneat.learn import accuracy, train  # noqa: E402


def count_ticks(g, X, w, settle: bool) -> int:
    """Instrumented copy of the tick loop, counting iterations only."""
    n = g.n_nodes
    incoming = g.incoming()
    touched = bytearray(n)
    touched[0] = touched[1] = touched[2] = 1
    vals = {i: np.zeros(len(X)) for i in range(n)}
    vals[0] = np.ones(len(X))
    vals[1] = X[:, 0]
    vals[2] = X[:, 1]
    prev_out = None
    for tick in range(1, gmod.MAX_TICK + 1):
        newly = [i for i in range(n) if not touched[i] and any(touched[s] for _, s in incoming[i])]
        for i in newly:
            touched[i] = 1
        for i in range(n):
            if not touched[i] or not incoming[i]:
                continue
            terms = [vals[s] * w[ci] for ci, s in incoming[i]]
            cum = terms[0]
            for t in terms[1:]:
                cum = cum * t if g.ops[i] == gmod.OP_MULT else cum + t
            if g.ops[i] in gmod._UNARY:
                cum = gmod._apply_unary(g.ops[i], cum)
            elif g.ops[i] == gmod.OP_SQUARE:
                cum = cum * cum
            vals[i] = cum
        if settle:
            cur = vals[3]
            if prev_out is not None and np.allclose(cur, prev_out, rtol=0, atol=1e-12):
                return tick
            prev_out = cur.copy()
        else:
            if all(touched) or not newly:
                return tick
    return gmod.MAX_TICK


def ticks_report() -> None:
    rng = np.random.default_rng(5)
    reg = InnovationRegistry()
    bundle = make_bundle("spiral", seed=11)
    X = bundle.train.X[:20]

    print(f"  {'genome':<22} {'nodes':>6} {'conns':>6} {'ref ticks':>10} {'settle ticks':>13}")
    mlp = make_mlp((32, 32), np.random.default_rng(1))
    print(
        f"  {'fixed MLP 32x32':<22} {mlp.n_nodes:6d} {mlp.n_connections:6d} "
        f"{count_ticks(mlp, X, np.array(mlp.weight), False):10d} "
        f"{count_ticks(mlp, X, np.array(mlp.weight), True):13d}"
    )

    for n_nodes, n_conns in ((10, 20), (20, 60), (40, 130)):
        g = logistic_genome(rng)
        ind = Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))
        while ind.genome.n_nodes < n_nodes:
            add_node(ind, rng, reg)
        while ind.genome.n_connections < n_conns:
            if not add_connection(ind, rng, reg):
                break
        w = ind.weights
        print(
            f"  {'evolved-like':<22} {ind.genome.n_nodes:6d} {ind.genome.n_enabled:6d} "
            f"{count_ticks(ind.genome, X, w, False):10d} "
            f"{count_ticks(ind.genome, X, w, True):13d}"
        )


def cost_report() -> None:
    rng = np.random.default_rng(5)
    reg = InnovationRegistry()
    bundle = make_bundle("spiral", seed=11)
    print(f"  {'nodes':>6} {'conns':>6} {'realized':>9} {'seconds':>9}")
    for n_nodes, n_conns in ((10, 20), (20, 60), (40, 130)):
        g = logistic_genome(rng)
        ind = Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))
        while ind.genome.n_nodes < n_nodes:
            add_node(ind, rng, reg)
        while ind.genome.n_connections < n_conns:
            if not add_connection(ind, rng, reg):
                break
        t0 = time.time()
        res = train(ind.genome, ind.weights, bundle.train.X, bundle.train.y, rng, n_cycles=600)
        print(
            f"  {ind.genome.n_nodes:6d} {ind.genome.n_enabled:6d} "
            f"{res.gradient_steps:9d} {time.time()-t0:9.2f}"
        )


def search_report(generations: int, population: int) -> None:
    bundle = make_bundle("spiral", seed=7103)
    cfg = SearchConfig(task="spiral", generations=generations, population=population)
    rng = np.random.default_rng(17103)
    reg = InnovationRegistry()
    pop = []
    for _ in range(population):
        g = logistic_genome(rng)
        pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    champion = None
    candidates = 0
    t_start = time.time()
    print(f"  {'gen':>4} {'cands':>6} {'sec':>7} {'mean_nodes':>11} {'mean_conns':>11} {'val_acc':>8}")
    for gen in range(generations + 1):
        t0 = time.time()
        for ind in pop:
            if not ind.evaluated:
                evaluate(ind, bundle, cfg, rng)
                candidates += 1
        best = max(pop, key=lambda i: i.fitness)
        if champion is None or best.fitness > champion.fitness:
            champion = best.copy()
        va = accuracy(champion.genome, champion.weights, bundle.validation.X, bundle.validation.y)
        print(
            f"  {gen:4d} {candidates:6d} {time.time()-t0:7.1f} "
            f"{np.mean([i.genome.n_nodes for i in pop]):11.1f} "
            f"{np.mean([i.genome.n_enabled for i in pop]):11.1f} {va:8.3f}"
        )
        if gen == generations:
            break
        kmedoids(pop, cfg.n_species, rng)
        pop = _reproduce(pop, cfg, rng, reg)

    total = time.time() - t_start
    print(f"  {candidates} candidates in {total/60:.2f} min -> {total/max(candidates,1):.3f} s/candidate")


def main() -> None:
    print("A. Ticks required to produce a correct output")
    ticks_report()
    print("\nB. Inner-learner cost under settled propagation (600 nominal steps)")
    cost_report()
    print("\nC. Real search on spirals, settled propagation (pop 30, 6 generations)")
    search_report(generations=6, population=30)


if __name__ == "__main__":
    main()
