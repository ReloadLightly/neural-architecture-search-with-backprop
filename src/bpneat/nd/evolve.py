"""The reference search, over graphs of any width.

Almost nothing here is new, and that is deliberate. Of NEAT's machinery, only
two pieces know how many inputs a graph has:

* ``add_connection``, because bias and the inputs may never be destinations,
  and the frozen version expresses "never" as the literal node id 3;
* the homogeneous-operator clamp inside ``mutate``, because it walks the hidden
  nodes and the frozen version knows where those start as the literal 4.

Everything else — ``mutate_weights``, ``add_node``, union crossover, the
compatibility distance, the K-medoids speciation, the innovation registry and
``Individual`` itself — touches only the connection arrays, which this
encoding's genome carries under the same names. So those are *imported from the
frozen module and run on nd genomes directly* rather than reimplemented. There
is no second copy to drift, and `tests/test_nd_evolve.py` asserts that the
frozen operators produce identical structures on both genome types from the
same seed.

The reproduction loop is v5's, which is the configuration v6 anchors on:
fitness-proportionate roulette with slack 0.01, no elitism, whole-population
replacement, a species extinction draw, and the structural rates as parameters
rather than module constants.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..evolve import (
    EXTINCTION_RATE,
    MUTATION_RATE,
    MUTATION_SIZE,
    P_ADD_CONNECTION,
    P_ADD_NODE,
    Individual,
    InnovationRegistry,
    add_node,
    distance,
    kmedoids,
    mutate_weights,
)
from ..evolve import crossover as union_crossover
from ..genome import ACTIVATIONS
from ..v3.selection import SELECTORS
from .datasets import TabularBundle
from .encoding import Layout, causal_subgraph, logistic_genome
from .learn import accuracy, fitness_from_error, total_error, train

__all__ = [
    "ACTIVATIONS",
    "EXTINCTION_RATE",
    "MUTATION_RATE",
    "MUTATION_SIZE",
    "P_ADD_CONNECTION",
    "P_ADD_NODE",
    "Individual",
    "InnovationRegistry",
    "NdConfig",
    "NdResult",
    "add_connection",
    "add_node",
    "distance",
    "evaluate",
    "kmedoids",
    "mutate",
    "mutate_weights",
    "search",
    "union_crossover",
]


@dataclass
class NdConfig:
    """v5's reference configuration, over a declared input/output layout."""

    task: str
    layout: Layout
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
    extinction_rate: float = EXTINCTION_RATE
    selector: str = "roulette_s0.01"
    p_add_node: float = P_ADD_NODE
    p_add_connection: float = P_ADD_CONNECTION
    crossover: bool = True

    def __post_init__(self) -> None:
        if self.propagation not in ("ha2016", "settled"):
            raise ValueError(f"unknown propagation {self.propagation!r}")
        if self.fitness_split not in ("train", "validation"):
            raise ValueError(f"unknown fitness split {self.fitness_split!r}")
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
class NdResult:
    champion: Individual
    history: list[dict] = field(default_factory=list)
    candidates: int = 0
    gradient_steps: int = 0
    wall_time: float = 0.0
    metrics: dict = field(default_factory=dict)
    node_additions: int = 0
    connection_additions: int = 0
    crossovers: int = 0


def add_connection(
    ind: Individual, rng: np.random.Generator, reg: InnovationRegistry
) -> bool:
    """The frozen operator, with "not bias and not an input" said by layout.

    The frozen version draws the destination from ``[OUT, n)`` — node id 3 — and
    then re-checks that it is not one of the three structural sources, which is
    unreachable. Here the lower bound is the first output node, which is 3 when
    there are two inputs and one output, so the draw consumes the generator
    identically at that layout.
    """
    g = ind.genome
    n = g.n_nodes
    first_output = g.layout.outputs[0]
    forbidden = set(g.layout.sources)
    for _ in range(20):
        src = int(rng.integers(0, n))
        dst = int(rng.integers(first_output, n))
        if dst in forbidden:  # unreachable at any layout; kept for exactness
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


def mutate(
    ind: Individual, rng: np.random.Generator, reg: InnovationRegistry, cfg: NdConfig
) -> tuple[int, int]:
    """v5's ``mutate`` with the two structural rates parameterised.

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
                for i in range(ind.genome.layout.n_structural, ind.genome.n_nodes):
                    ind.genome.ops[i] = cfg.activations[0]
    return added_node, added_conn


def evaluate(
    ind: Individual, bundle: TabularBundle, cfg: NdConfig, rng: np.random.Generator
) -> None:
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


def _seed_individual(rng: np.random.Generator, layout: Layout) -> Individual:
    g = logistic_genome(rng, layout)
    return Individual(genome=g, weights=np.array(g.weight, dtype=np.float64))


def _reproduce(
    pop: list[Individual],
    cfg: NdConfig,
    rng: np.random.Generator,
    reg: InnovationRegistry,
) -> tuple[list[Individual], int, int, int]:
    """One generation of offspring, plus counts of the operators that fired."""
    sel = cfg.selection
    by_species: dict[int, list[Individual]] = {}
    for ind in pop:
        by_species.setdefault(ind.species, []).append(ind)

    ranked = sorted(
        by_species.items(), key=lambda kv: max(i.fitness for i in kv[1]), reverse=True
    )
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
        next_pop.append(_seed_individual(rng, cfg.layout))

    return next_pop[: cfg.population], n_node, n_conn, n_cross


def search(bundle: TabularBundle, cfg: NdConfig, seed: int) -> NdResult:
    """One search over graphs of this bundle's width. Never reads ``bundle.test``."""
    if bundle.n_features != cfg.layout.n_inputs:
        raise ValueError(
            f"{bundle.task} has {bundle.n_features} features, the layout takes "
            f"{cfg.layout.n_inputs}"
        )
    if bundle.n_classes != (2 if cfg.layout.n_outputs == 1 else cfg.layout.n_outputs):
        raise ValueError(
            f"{bundle.task} has {bundle.n_classes} classes, the layout has "
            f"{cfg.layout.n_outputs} output(s)"
        )

    rng = np.random.default_rng(seed)
    reg = InnovationRegistry()
    started = time.time()

    pop = [_seed_individual(rng, cfg.layout) for _ in range(cfg.population)]
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

    return NdResult(
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


def layout_for(bundle: TabularBundle) -> Layout:
    """The layout a dataset calls for: one logit for two classes, k for k."""
    return Layout(bundle.n_features, 1 if bundle.n_classes == 2 else bundle.n_classes)
