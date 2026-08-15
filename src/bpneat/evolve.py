"""Population-based topology search with inherited learned weights.

The outer loop is NEAT: minimal logistic genomes, structural mutation, union
crossover under an innovation registry, and K-medoids subpopulations that
protect structural diversity. The inner loop is backpropagation, applied to
*every* evaluated candidate under a fixed budget (see :mod:`.learn`).

In the primary (Lamarckian) condition learned weights are inherited by
descendants. The Baldwinian ablation re-initialises weights each generation, so
selection still favours learnable structures but nothing learned is passed on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .datasets import DatasetBundle
from .genome import (
    ACTIVATIONS,
    BIAS,
    IN_X,
    IN_Y,
    N_STRUCTURAL,
    OUT,
    Genome,
    causal_subgraph,
    logistic_genome,
)
from .learn import accuracy, fitness_from_error, total_error, train

MUTATION_RATE = 0.2
MUTATION_SIZE = 0.5
P_ADD_CONNECTION = 0.5
P_ADD_NODE = 0.2


@dataclass
class SearchConfig:
    task: str
    generations: int = 20
    population: int = 100
    n_species: int = 5
    inner_steps: int = 600
    batch_size: int = 10
    use_penalty: bool = True
    backprop: bool = True
    lamarckian: bool = True
    activations: tuple[int, ...] = ACTIVATIONS
    fitness_split: str = "train"  # "train" (Track A) or "validation" (Track B)
    # Declared propagation mode, never inferred and never pooled across tracks.
    #
    #   "ha2016"  — Ha's exact break rule: stop once every node is touched.
    #   "settled" — same propagation, but ticked to a topology-determined fixed
    #               point so every represented path contributes.
    #
    # Under "ha2016" the output node (id 3) is recomputed before every hidden
    # node, so a genome that grows hidden structure while keeping a direct
    # input/bias edge into the output has that structure ignored by fitness.
    # Seed genomes are exactly that shape, which makes the linear solution a
    # trap. This is a property of the source algorithm and a Chapter 4 result
    # in its own right, so it is a reported factor rather than a silent default.
    propagation: str = "settled"
    elitism: int = 1

    def __post_init__(self) -> None:
        if self.propagation not in ("ha2016", "settled"):
            raise ValueError(f"unknown propagation mode {self.propagation!r}")
        if self.fitness_split not in ("train", "validation"):
            raise ValueError(f"unknown fitness split {self.fitness_split!r}")

    @property
    def settle(self) -> bool:
        return self.propagation == "settled"


class InnovationRegistry:
    """Structural innovations are numbered globally so crossover can align."""

    def __init__(self) -> None:
        self._conn: dict[tuple[int, int], int] = {}
        self._next = 0

    def connection(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self._conn:
            self._conn[key] = self._next
            self._next += 1
        return self._conn[key]


@dataclass
class Individual:
    genome: Genome
    weights: np.ndarray
    fitness: float = -np.inf
    error: float = np.inf
    gradient_steps: int = 0
    species: int = 0
    evaluated: bool = False

    def copy(self) -> "Individual":
        return Individual(
            genome=self.genome.copy(),
            weights=self.weights.copy(),
            fitness=self.fitness,
            error=self.error,
            gradient_steps=self.gradient_steps,
            species=self.species,
            evaluated=self.evaluated,
        )


# --------------------------------------------------------------------------
# Mutation and crossover
# --------------------------------------------------------------------------


def mutate_weights(ind: Individual, rng: np.random.Generator) -> None:
    n = len(ind.weights)
    if n == 0:
        return
    mask = rng.random(n) < MUTATION_RATE
    if mask.any():
        ind.weights[mask] += rng.normal(0.0, MUTATION_SIZE, int(mask.sum()))


def add_connection(ind: Individual, rng: np.random.Generator, reg: InnovationRegistry) -> bool:
    g = ind.genome
    n = g.n_nodes
    # Any node may be a source (recurrence is permitted); bias and inputs may
    # never be destinations.
    for _ in range(20):
        src = int(rng.integers(0, n))
        dst = int(rng.integers(OUT, n))
        if dst in (BIAS, IN_X, IN_Y):
            continue
        if g.has_connection(src, dst):
            continue
        g.src.append(src)
        g.dst.append(dst)
        g.weight.append(0.0)
        g.active.append(True)
        g.innovation.append(reg.connection(src, dst))
        ind.weights = np.append(ind.weights, rng.normal(0.0, 1.0))
        return True
    return False


def add_node(ind: Individual, rng: np.random.Generator, reg: InnovationRegistry) -> bool:
    g = ind.genome
    live = [i for i in range(g.n_connections) if g.active[i]]
    if not live:
        return False
    ci = int(rng.choice(live))
    g.active[ci] = False
    old_w = float(ind.weights[ci])
    new_node = g.n_nodes
    g.ops.append(int(rng.choice(ACTIVATIONS)))

    for src, dst, w in ((g.src[ci], new_node, 1.0), (new_node, g.dst[ci], old_w)):
        g.src.append(src)
        g.dst.append(dst)
        g.weight.append(w)
        g.active.append(True)
        g.innovation.append(reg.connection(src, dst))
        ind.weights = np.append(ind.weights, w)
    return True


def mutate(ind: Individual, rng: np.random.Generator, reg: InnovationRegistry, cfg: SearchConfig) -> None:
    mutate_weights(ind, rng)
    if rng.random() < P_ADD_CONNECTION:
        add_connection(ind, rng, reg)
    if rng.random() < P_ADD_NODE:
        add_node(ind, rng, reg)
        # Keep the operator set restricted for the homogeneous ablation.
        if len(cfg.activations) == 1:
            for i in range(N_STRUCTURAL, ind.genome.n_nodes):
                ind.genome.ops[i] = cfg.activations[0]


def crossover(a: Individual, b: Individual, rng: np.random.Generator) -> Individual:
    """Union crossover: the fitter parent supplies structure it alone carries."""
    if b.fitness > a.fitness:
        a, b = b, a
    child = a.copy()
    lookup = {inv: i for i, inv in enumerate(b.genome.innovation)}
    for i, inv in enumerate(child.genome.innovation):
        j = lookup.get(inv)
        if j is not None and rng.random() < 0.5:
            child.weights[i] = b.weights[j]
    child.evaluated = False
    return child


def distance(a: Individual, b: Individual) -> float:
    """NEAT compatibility distance over innovation sets.

    Weights come from ``Individual.weights``, which is the single source of
    truth. ``Genome.weight`` only carries the *initial* vector — mutation and
    learning update the array, not the list — so reading the list here would
    cluster on stale values, and would silently change behaviour across a
    checkpoint round-trip.
    """
    ga, gb = a.genome, b.genome
    ia, ib = set(ga.innovation), set(gb.innovation)
    if not ia and not ib:
        return 0.0
    disjoint = len(ia ^ ib)
    n = max(len(ia), len(ib), 1)
    shared = ia & ib
    if shared:
        wa = {inv: w for inv, w in zip(ga.innovation, a.weights)}
        wb = {inv: w for inv, w in zip(gb.innovation, b.weights)}
        wdiff = float(np.mean([abs(wa[i] - wb[i]) for i in shared]))
    else:
        wdiff = 0.0
    op_diff = abs(ga.n_nodes - gb.n_nodes)
    return disjoint / n + 0.4 * wdiff + 0.2 * op_diff


def kmedoids(pop: list[Individual], k: int, rng: np.random.Generator, iters: int = 8) -> None:
    """Assign ``ind.species`` by K-medoids on compatibility distance."""
    n = len(pop)
    k = min(k, n)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(pop[i], pop[j])
            D[i, j] = D[j, i] = d

    medoids = list(rng.choice(n, size=k, replace=False))
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = np.argmin(D[:, medoids], axis=1)
        moved = False
        for c in range(k):
            members = np.flatnonzero(labels == c)
            if len(members) == 0:
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            best = int(members[int(np.argmin(costs))])
            if best != medoids[c]:
                medoids[c] = best
                moved = True
        if not moved:
            break
    for i, ind in enumerate(pop):
        ind.species = int(labels[i])


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    champion: Individual
    history: list[dict] = field(default_factory=list)
    candidates: int = 0
    gradient_steps: int = 0
    wall_time: float = 0.0
    metrics: dict = field(default_factory=dict)


def evaluate(ind: Individual, bundle: DatasetBundle, cfg: SearchConfig, rng: np.random.Generator) -> None:
    """Train the candidate, then score it on the configured fitness split."""
    if cfg.backprop:
        res = train(
            ind.genome,
            ind.weights,
            bundle.train.X,
            bundle.train.y,
            rng,
            n_cycles=cfg.inner_steps,
            batch_size=cfg.batch_size,
            settle=cfg.settle,
        )
        if cfg.lamarckian:
            ind.weights = res.weights
        train_error = res.error
        ind.gradient_steps = res.gradient_steps
        eval_weights = res.weights
    else:
        train_error = total_error(
            ind.genome, ind.weights, bundle.train.X, bundle.train.y, cfg.settle
        )
        ind.gradient_steps = 0
        eval_weights = ind.weights

    if cfg.fitness_split == "validation":
        err = total_error(
            ind.genome, eval_weights, bundle.validation.X, bundle.validation.y, cfg.settle
        )
    else:
        err = train_error

    ind.error = err
    ind.fitness = fitness_from_error(ind.genome, err, cfg.use_penalty)
    ind.evaluated = True


def search(
    bundle: DatasetBundle,
    cfg: SearchConfig,
    seed: int,
    checkpoint_path=None,
    checkpoint_every: int = 1,
) -> SearchResult:
    """Run one Backprop-NEAT search. Never reads ``bundle.test``.

    If ``checkpoint_path`` exists the run resumes from it exactly; otherwise it
    starts fresh and writes a checkpoint every ``checkpoint_every`` generations.
    """
    from pathlib import Path

    from . import checkpoint as ckpt

    cp = Path(checkpoint_path) if checkpoint_path else None
    started = time.time()

    if cp is not None and cp.exists():
        state = ckpt.load(cp)
        rng = state["rng"]
        reg = state["registry"]
        pop = state["population"]
        champion = state["champion"]
        history = state["history"]
        candidates = state["candidates"]
        grad_steps = state["gradient_steps"]
        start_gen = state["generation"]
        prior_elapsed = state["elapsed"]
    else:
        rng = np.random.default_rng(seed)
        reg = InnovationRegistry()
        pop = []
        for _ in range(cfg.population):
            g = logistic_genome(rng)
            pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))
        champion = None
        history = []
        candidates = 0
        grad_steps = 0
        start_gen = 0
        prior_elapsed = 0.0

    for gen in range(start_gen, cfg.generations + 1):
        for ind in pop:
            if not ind.evaluated:
                evaluate(ind, bundle, cfg, rng)
                candidates += 1
                grad_steps += ind.gradient_steps

        # Hall of fame: the best candidate ever seen survives setbacks.
        best = max(pop, key=lambda i: i.fitness)
        if champion is None or best.fitness > champion.fitness:
            champion = best.copy()

        val_acc = accuracy(
            champion.genome,
            champion.weights,
            bundle.validation.X,
            bundle.validation.y,
            cfg.settle,
        )
        history.append(
            {
                "generation": gen,
                "best_fitness": float(best.fitness),
                "champion_fitness": float(champion.fitness),
                "champion_validation_accuracy": float(val_acc),
                "mean_nodes": float(np.mean([i.genome.n_nodes for i in pop])),
                "mean_connections": float(np.mean([i.genome.n_enabled for i in pop])),
                "species": len({i.species for i in pop}),
                "candidates": candidates,
                "gradient_steps": grad_steps,
            }
        )

        if gen == cfg.generations:
            break

        kmedoids(pop, cfg.n_species, rng)
        pop = _reproduce(pop, cfg, rng, reg)

        if cp is not None and (gen + 1) % checkpoint_every == 0:
            ckpt.save(
                cp,
                generation=gen + 1,
                population=pop,
                champion=champion,
                rng=rng,
                registry=reg,
                history=history,
                candidates=candidates,
                gradient_steps=grad_steps,
                elapsed=prior_elapsed + (time.time() - started),
            )

    assert champion is not None
    return SearchResult(
        champion=champion,
        history=history,
        candidates=candidates,
        gradient_steps=grad_steps,
        wall_time=prior_elapsed + (time.time() - started),
        metrics=_champion_metrics(champion, bundle, cfg.settle),
    )


def _reproduce(
    pop: list[Individual], cfg: SearchConfig, rng: np.random.Generator, reg: InnovationRegistry
) -> list[Individual]:
    """Within-species selection, elitism, crossover, mutation, repopulation."""
    by_species: dict[int, list[Individual]] = {}
    for ind in pop:
        by_species.setdefault(ind.species, []).append(ind)

    next_pop: list[Individual] = []
    n_species = max(len(by_species), 1)
    quota = max(1, cfg.population // n_species)

    for members in by_species.values():
        members.sort(key=lambda i: i.fitness, reverse=True)
        produced: list[Individual] = []

        for elite in members[: cfg.elitism]:
            keep = elite.copy()
            if not cfg.lamarckian:
                keep.weights = rng.normal(0.0, 1.0, len(keep.weights))
                keep.evaluated = False
            produced.append(keep)

        # Selection pressure: parents are drawn from the better half of the
        # species, so each subpopulation fills its own quota.
        pool = members[: max(2, len(members) // 2)]
        while len(produced) < quota:
            a = pool[int(rng.integers(0, len(pool)))]
            b = pool[int(rng.integers(0, len(pool)))]
            child = crossover(a, b, rng) if a is not b else a.copy()
            mutate(child, rng, reg, cfg)
            if not cfg.lamarckian:
                child.weights = rng.normal(0.0, 1.0, len(child.weights))
            child.evaluated = False
            produced.append(child)

        next_pop.extend(produced)

    # Extinction/repopulation: refill any shortfall with fresh minimal genomes.
    while len(next_pop) < cfg.population:
        g = logistic_genome(rng)
        next_pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    return next_pop[: cfg.population]


def _champion_metrics(
    champion: Individual, bundle: DatasetBundle, settle: bool = True
) -> dict:
    """Validation-only metrics. The sealed test split is deliberately untouched."""
    g, w = champion.genome, champion.weights
    causal = causal_subgraph(g, bundle.validation.X, w, settle)
    return {
        "train_accuracy": accuracy(g, w, bundle.train.X, bundle.train.y, settle),
        "train_loss": total_error(g, w, bundle.train.X, bundle.train.y, settle),
        "validation_accuracy": accuracy(
            g, w, bundle.validation.X, bundle.validation.y, settle
        ),
        "validation_loss": total_error(
            g, w, bundle.validation.X, bundle.validation.y, settle
        ),
        **causal,
    }
