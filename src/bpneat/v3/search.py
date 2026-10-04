"""The v3 search loop: v2's mechanics with the selection operator exposed.

Everything that decides what a candidate *is* — genome encoding, mutation,
crossover, speciation, the inner learner, fitness — is imported unchanged from
the frozen v2 modules. The only thing v3 changes is how parents are sampled,
because that is the factor under test, and it records the realized selection
intensity so the dose-response has an x-axis measured rather than assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..datasets import DatasetBundle
from ..evolve import (
    Individual,
    InnovationRegistry,
    crossover,
    evaluate,
    kmedoids,
    mutate,
)
from ..genome import ACTIVATIONS, causal_subgraph, logistic_genome
from ..learn import accuracy, total_error
from .selection import DEFAULT_SELECTOR, SELECTORS, Selector


@dataclass
class V3Config:
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
    selector: str = DEFAULT_SELECTOR

    def __post_init__(self) -> None:
        if self.propagation not in ("ha2016", "settled"):
            raise ValueError(f"unknown propagation {self.propagation!r}")
        if self.fitness_split not in ("train", "validation"):
            raise ValueError(f"unknown fitness split {self.fitness_split!r}")
        if self.selector not in SELECTORS:
            raise ValueError(f"unknown selector {self.selector!r}")

    @property
    def settle(self) -> bool:
        return self.propagation == "settled"

    @property
    def selection(self) -> Selector:
        return SELECTORS[self.selector]


@dataclass
class V3Result:
    champion: Individual
    history: list[dict] = field(default_factory=list)
    candidates: int = 0
    gradient_steps: int = 0
    wall_time: float = 0.0
    metrics: dict = field(default_factory=dict)
    selection_intensity: float = 0.0


def _reproduce(
    pop: list[Individual],
    cfg: V3Config,
    rng: np.random.Generator,
    reg: InnovationRegistry,
) -> tuple[list[Individual], float]:
    """One generation of offspring, plus the realized selection intensity."""
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
    intensities: list[float] = []

    for species, members in by_species.items():
        source = by_species[best_species] if (extinction and species == worst_species) else members
        fitness = np.array([m.fitness for m in source], dtype=np.float64)
        intensities.append(sel.intensity(fitness))

        produced: list[Individual] = []
        # Truncation levels carry an elite; the roulette and tournament levels
        # do not, matching the reference.
        if sel.elitism:
            for elite in sorted(members, key=lambda i: i.fitness, reverse=True)[: sel.elitism]:
                keep = elite.copy()
                if not cfg.lamarckian:
                    keep.weights = rng.normal(0.0, 1.0, len(keep.weights))
                    keep.evaluated = False
                produced.append(keep)

        while len(produced) < quota:
            mom = sel.pick(source, rng)
            dad = sel.pick(source, rng)
            child = crossover(mom, dad, rng) if mom is not dad else mom.copy()
            mutate(child, rng, reg, cfg)
            if not cfg.lamarckian:
                child.weights = rng.normal(0.0, 1.0, len(child.weights))
            child.species = species
            child.evaluated = False
            produced.append(child)
        next_pop.extend(produced)

    while len(next_pop) < cfg.population:
        g = logistic_genome(rng)
        next_pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    mean_intensity = float(np.mean(intensities)) if intensities else 0.0
    return next_pop[: cfg.population], mean_intensity


def search(bundle: DatasetBundle, cfg: V3Config, seed: int) -> V3Result:
    """One v3 search. Never reads ``bundle.test``."""
    rng = np.random.default_rng(seed)
    reg = InnovationRegistry()
    started = time.time()

    pop = []
    for _ in range(cfg.population):
        g = logistic_genome(rng)
        pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    champion: Individual | None = None
    candidates = grad_steps = 0
    history: list[dict] = []
    intensities: list[float] = []

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
                "mean_connections": float(np.mean([i.genome.n_enabled for i in pop])),
                "candidates": candidates,
                "gradient_steps": grad_steps,
            }
        )
        if gen == cfg.generations:
            break

        kmedoids(pop, cfg.n_species, rng)
        pop, intensity = _reproduce(pop, cfg, rng, reg)
        intensities.append(intensity)
        history[-1]["selection_intensity"] = intensity

    assert champion is not None
    g, w = champion.genome, champion.weights
    causal = causal_subgraph(g, bundle.validation.X, w, cfg.settle)
    train_acc = accuracy(g, w, bundle.train.X, bundle.train.y, cfg.settle)
    val_acc = accuracy(g, w, bundle.validation.X, bundle.validation.y, cfg.settle)

    return V3Result(
        champion=champion,
        history=history,
        candidates=candidates,
        gradient_steps=grad_steps,
        wall_time=time.time() - started,
        selection_intensity=float(np.mean(intensities)) if intensities else 0.0,
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
