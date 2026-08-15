"""Cost at the scale the search actually reaches.

The first probe measured near-minimal genomes and therefore under-reported cost
by an order of magnitude. This one measures (a) a genome at the size Ha's
reference run ends at, and (b) a reference-population search, per generation, so
growth in cost is visible rather than assumed.
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
    add_connection,
    add_node,
    evaluate,
    kmedoids,
    _reproduce,
)
from bpneat.genome import logistic_genome  # noqa: E402
from bpneat.learn import train  # noqa: E402


def time_by_size() -> None:
    """Inner-learner cost as a function of genome size, at 600 nominal steps."""
    rng = np.random.default_rng(5)
    reg = InnovationRegistry()
    bundle = make_bundle("spiral", seed=11)

    print(f"  {'nodes':>6} {'conns':>6} {'realized':>9} {'seconds':>9} {'ms/step':>9}")
    for n_nodes, n_conns in ((10, 20), (20, 60), (40, 130), (60, 232)):
        g = logistic_genome(rng)
        ind = Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))
        while ind.genome.n_nodes < n_nodes:
            add_node(ind, rng, reg)
        while ind.genome.n_connections < n_conns:
            if not add_connection(ind, rng, reg):
                break

        t0 = time.time()
        res = train(
            ind.genome, ind.weights, bundle.train.X, bundle.train.y, rng, n_cycles=600
        )
        dt = time.time() - t0
        per_step = dt / max(res.gradient_steps, 1) * 1000
        print(
            f"  {ind.genome.n_nodes:6d} {ind.genome.n_enabled:6d} "
            f"{res.gradient_steps:9d} {dt:9.2f} {per_step:9.2f}"
        )


def timed_search(task: str, generations: int, population: int, seed: int) -> None:
    """A reference-population search, reporting cost and growth per generation."""
    bundle = make_bundle(task, seed=7103)
    cfg = SearchConfig(task=task, generations=generations, population=population)
    rng = np.random.default_rng(seed)
    reg = InnovationRegistry()

    pop = []
    for _ in range(population):
        g = logistic_genome(rng)
        pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    champion = None
    candidates = 0
    total_start = time.time()
    print(
        f"  {'gen':>4} {'cands':>6} {'sec':>8} {'cum_min':>8} "
        f"{'mean_nodes':>11} {'mean_conns':>11} {'champ_val_acc':>14}"
    )
    for gen in range(generations + 1):
        t0 = time.time()
        for ind in pop:
            if not ind.evaluated:
                evaluate(ind, bundle, cfg, rng)
                candidates += 1
        dt = time.time() - t0

        best = max(pop, key=lambda i: i.fitness)
        if champion is None or best.fitness > champion.fitness:
            champion = best.copy()

        from bpneat.learn import accuracy

        va = accuracy(
            champion.genome, champion.weights, bundle.validation.X, bundle.validation.y
        )
        print(
            f"  {gen:4d} {candidates:6d} {dt:8.2f} {(time.time()-total_start)/60:8.2f} "
            f"{np.mean([i.genome.n_nodes for i in pop]):11.1f} "
            f"{np.mean([i.genome.n_enabled for i in pop]):11.1f} {va:14.3f}"
        )

        if gen == generations:
            break
        kmedoids(pop, cfg.n_species, rng)
        pop = _reproduce(pop, cfg, rng, reg)

    print(
        f"  total {candidates} candidates in "
        f"{(time.time()-total_start)/60:.2f} min "
        f"({(time.time()-total_start)/max(candidates,1):.3f}s/candidate)"
    )


def main() -> None:
    print("=" * 78)
    print("A. Inner-learner cost vs genome size (600 nominal steps, batch 10)")
    time_by_size()

    print("\nB. Reference-population search on spirals (pop 100, 8 generations)")
    timed_search("spiral", generations=8, population=100, seed=17103)
    print("=" * 78)


if __name__ == "__main__":
    main()
