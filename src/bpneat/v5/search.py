"""NEAT with its five mechanisms exposed as parameters.

Everything that decides what a candidate *is* — the genome encoding, the
structural operators, the compatibility metric, the inner learner, the fitness —
is imported unchanged from the frozen v2 modules. v5 changes only:

* ``p_add_node`` and ``p_add_connection``, the complexification rates. In frozen
  code these are module constants (``P_ADD_NODE = 0.2``, ``P_ADD_CONNECTION =
  0.5``), which is why this module exists at all: they cannot be varied without
  new code, and editing the frozen module is forbidden.
* ``use_penalty``, the complexity penalty ``1 + 0.03*sqrt(connections)`` that
  opposes complexification, already a flag on the v3 config.
* ``n_species``, the speciation that is supposed to protect new structure.
* ``crossover``, on or off.
* the population/generation split at a fixed candidate budget, which trades
  width for depth.

Selection is v3's reference operator (fitness-proportionate roulette, slack
0.01, no elitism, whole-population replacement, extinction 0.5), because v3
Block C established that the selection *level* moves nothing here, and v5 is not
re-asking that question.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..datasets import DatasetBundle
from ..evolve import (
    Individual,
    InnovationRegistry,
    add_connection,
    add_node,
    evaluate,
    kmedoids,
    mutate_weights,
)
from ..evolve import (
    crossover as union_crossover,
)
from ..genome import ACTIVATIONS, N_STRUCTURAL, causal_subgraph, logistic_genome
from ..learn import accuracy, total_error
from ..v3.selection import SELECTORS

#: The reference rates, as they stand in the frozen module.
REF_P_ADD_NODE = 0.2
REF_P_ADD_CONNECTION = 0.5


@dataclass
class V5Config:
    """Duck-types :class:`bpneat.evolve.SearchConfig` for the frozen evaluator."""

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
    fitness_split: str = "validation"
    propagation: str = "settled"
    elitism: int = 0
    extinction_rate: float = 0.5
    selector: str = "roulette_s0.01"
    # --- the v5 factors ---
    p_add_node: float = REF_P_ADD_NODE
    p_add_connection: float = REF_P_ADD_CONNECTION
    crossover: bool = True

    def __post_init__(self) -> None:
        if self.propagation not in ("ha2016", "settled"):
            raise ValueError(f"unknown propagation {self.propagation!r}")
        if self.selector not in SELECTORS:
            raise ValueError(f"unknown selector {self.selector!r}")
        if not 0.0 <= self.p_add_node <= 1.0:
            raise ValueError(f"p_add_node out of range: {self.p_add_node}")
        if not 0.0 <= self.p_add_connection <= 1.0:
            raise ValueError(f"p_add_connection out of range: {self.p_add_connection}")
        if self.n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {self.n_species}")

    @property
    def settle(self) -> bool:
        return self.propagation == "settled"

    @property
    def selection(self):
        return SELECTORS[self.selector]

    @property
    def candidate_budget(self) -> int:
        return self.population * (self.generations + 1)


@dataclass
class V5Result:
    champion: Individual
    history: list[dict] = field(default_factory=list)
    candidates: int = 0
    gradient_steps: int = 0
    wall_time: float = 0.0
    metrics: dict = field(default_factory=dict)
    #: Structural bookkeeping: how often each operator actually fired.
    node_additions: int = 0
    connection_additions: int = 0
    crossovers: int = 0


def mutate(
    ind: Individual, rng: np.random.Generator, reg: InnovationRegistry, cfg: V5Config
) -> tuple[int, int]:
    """:func:`bpneat.evolve.mutate` with the two structural rates parameterised.

    Identical in every other respect, including the homogeneous-operator clamp,
    so that changing a rate changes a rate and nothing else.
    """
    mutate_weights(ind, rng)
    added_conn = added_node = 0
    if rng.random() < cfg.p_add_connection:
        added_conn = int(add_connection(ind, rng, reg))
    if rng.random() < cfg.p_add_node:
        if add_node(ind, rng, reg):
            added_node = 1
            if len(cfg.activations) == 1:
                for i in range(N_STRUCTURAL, ind.genome.n_nodes):
                    ind.genome.ops[i] = cfg.activations[0]
    return added_node, added_conn


def _reproduce(
    pop: list[Individual],
    cfg: V5Config,
    rng: np.random.Generator,
    reg: InnovationRegistry,
) -> tuple[list[Individual], int, int, int]:
    """One generation of offspring, plus counts of the operators that fired."""
    sel = cfg.selection
    by_species: dict[int, list[Individual]] = {}
    for ind in pop:
        by_species.setdefault(ind.species, []).append(ind)

    ranked = sorted(by_species.items(), key=lambda kv: max(i.fitness for i in kv[1]), reverse=True)
    best_species, worst_species = ranked[0][0], ranked[-1][0]
    extinction = (
        cfg.extinction_rate > 0 and len(ranked) > 1 and rng.random() < cfg.extinction_rate
    )

    quota = max(1, cfg.population // max(len(by_species), 1))
    next_pop: list[Individual] = []
    n_node = n_conn = n_cross = 0

    for species, members in by_species.items():
        source = by_species[best_species] if (extinction and species == worst_species) else members
        produced: list[Individual] = []
        while len(produced) < quota:
            mom = sel.pick(source, rng)
            dad = sel.pick(source, rng)
            if cfg.crossover and mom is not dad:
                child = union_crossover(mom, dad, rng)
                n_cross += 1
            else:
                child = mom.copy()
            a, c = mutate(child, rng, reg, cfg)
            n_node += a
            n_conn += c
            if not cfg.lamarckian:
                child.weights = rng.normal(0.0, 1.0, len(child.weights))
            child.species = species
            child.evaluated = False
            produced.append(child)
        next_pop.extend(produced)

    while len(next_pop) < cfg.population:
        g = logistic_genome(rng)
        next_pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    return next_pop[: cfg.population], n_node, n_conn, n_cross


def search(bundle: DatasetBundle, cfg: V5Config, seed: int) -> V5Result:
    """One v5 search. Never reads ``bundle.test``."""
    rng = np.random.default_rng(seed)
    reg = InnovationRegistry()
    started = time.time()

    pop = []
    for _ in range(cfg.population):
        g = logistic_genome(rng)
        pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    champion: Individual | None = None
    candidates = grad_steps = 0
    n_node = n_conn = n_cross = 0
    history: list[dict] = []

    for gen in range(cfg.generations + 1):
        for ind in pop:
            if not ind.evaluated:
                evaluate(ind, bundle, cfg, rng)
                candidates += 1
                grad_steps += ind.gradient_steps

        best = max(pop, key=lambda i: i.fitness)
        if champion is None or best.fitness > champion.fitness:
            champion = best.copy()

        # Record the size trajectory: v5's question is about complexification,
        # so population size over time is a primary series, not a diagnostic.
        history.append(
            {
                "generation": gen,
                "best_fitness": float(best.fitness),
                "champion_fitness": float(champion.fitness),
                "mean_nodes": float(np.mean([i.genome.n_nodes for i in pop])),
                "max_nodes": int(max(i.genome.n_nodes for i in pop)),
                "mean_connections": float(np.mean([i.genome.n_enabled for i in pop])),
                "champion_nodes": int(champion.genome.n_nodes),
                "n_species": len({i.species for i in pop}),
                "candidates": candidates,
                "gradient_steps": grad_steps,
            }
        )
        if gen == cfg.generations:
            break

        kmedoids(pop, cfg.n_species, rng)
        pop, a, c, x = _reproduce(pop, cfg, rng, reg)
        n_node += a
        n_conn += c
        n_cross += x

    assert champion is not None
    g, w = champion.genome, champion.weights
    causal = causal_subgraph(g, bundle.validation.X, w, cfg.settle)
    train_acc = accuracy(g, w, bundle.train.X, bundle.train.y, cfg.settle)
    val_acc = accuracy(g, w, bundle.validation.X, bundle.validation.y, cfg.settle)

    return V5Result(
        champion=champion,
        history=history,
        candidates=candidates,
        gradient_steps=grad_steps,
        wall_time=time.time() - started,
        node_additions=n_node,
        connection_additions=n_conn,
        crossovers=n_cross,
        metrics={
            "train_accuracy": train_acc,
            "train_loss": total_error(g, w, bundle.train.X, bundle.train.y, cfg.settle),
            "validation_accuracy": val_acc,
            "validation_loss": total_error(
                g, w, bundle.validation.X, bundle.validation.y, cfg.settle
            ),
            "generalization_gap": train_acc - val_acc,
            **causal,
        },
    )
