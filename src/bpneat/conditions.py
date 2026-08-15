"""The comparison matrix: one mechanism removed at a time.

Every condition — evolved or fixed — is a graph trained by the same inner
learner under the same declared propagation mode, so a difference between two
conditions is attributable to the mechanism named in the table rather than to
two different pieces of training code.

Compute is never summarised by one number. Each run reports candidate
evaluations, realized gradient steps, and wall time separately, because a
candidate-matched comparison is not a compute-matched one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .baselines import make_logistic, make_mlp, make_random_architecture
from .datasets import DatasetBundle
from .evolve import Individual, SearchConfig, search
from .genome import ACTIVATIONS, OP_TANH, causal_subgraph
from .learn import accuracy, total_error, train
from .record import RunRecord, run_id, serialise_genome

# Reference budgets: population 100, five species, 10 generations for the
# separable geometries and 20 for spirals.
GENERATIONS = {"xor": 10, "circle": 10, "spiral": 20}

# Frozen success thresholds (handoff section 6.6).
SUCCESS_THRESHOLD = {"xor": 0.90, "circle": 0.90, "spiral": 0.80}


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str  # "evolution" | "multistart" | "random_search"
    core: bool
    mechanism: str
    overrides: dict = field(default_factory=dict)
    # Multistart / random-search budget. Declared explicitly rather than
    # inherited, because matching an evolutionary candidate count with fixed
    # 32x32 networks costs far more compute per candidate.
    restarts: int = 60
    hidden: tuple[int, ...] = (32, 32)


CONDITIONS: dict[str, Condition] = {
    "backprop_neat": Condition(
        "backprop_neat", "evolution", True,
        "evolved topology, heterogeneous operators, inherited weights, backprop",
    ),
    "homogeneous_tanh": Condition(
        "homogeneous_tanh", "evolution", True,
        "removes operator diversity, keeps topology search and learning",
        overrides={"activations": (OP_TANH,)},
    ),
    "evolution_only": Condition(
        "evolution_only", "evolution", True,
        "removes gradient learning",
        overrides={"backprop": False},
    ),
    "random_search": Condition(
        "random_search", "random_search", True,
        "removes evolutionary selection, keeps graph irregularity and the learner",
    ),
    "fixed_mlp": Condition(
        "fixed_mlp", "multistart", True,
        "removes topology search",
    ),
    "logistic": Condition(
        "logistic", "multistart", True,
        "minimal linear floor",
        hidden=(),
    ),
    "no_penalty": Condition(
        "no_penalty", "evolution", False,
        "tests the bloat/regularisation trade-off",
        overrides={"use_penalty": False},
    ),
    "baldwinian": Condition(
        "baldwinian", "evolution", False,
        "selection for learnability without inherited learned weights",
        overrides={"lamarckian": False},
    ),
}

CORE_CONDITIONS = tuple(n for n, c in CONDITIONS.items() if c.core)


def base_config(task: str, propagation: str, fitness_split: str) -> SearchConfig:
    return SearchConfig(
        task=task,
        generations=GENERATIONS[task],
        population=100,
        n_species=5,
        inner_steps=600,
        batch_size=10,
        propagation=propagation,
        fitness_split=fitness_split,
    )


def _metrics(g, w, bundle: DatasetBundle, settle: bool) -> dict:
    """Validation-only metrics. The sealed test split is never touched here."""
    causal = causal_subgraph(g, bundle.validation.X, w, settle)
    train_acc = accuracy(g, w, bundle.train.X, bundle.train.y, settle)
    val_acc = accuracy(g, w, bundle.validation.X, bundle.validation.y, settle)
    return {
        "train_accuracy": train_acc,
        "train_loss": total_error(g, w, bundle.train.X, bundle.train.y, settle),
        "validation_accuracy": val_acc,
        "validation_loss": total_error(
            g, w, bundle.validation.X, bundle.validation.y, settle
        ),
        "generalization_gap": train_acc - val_acc,
        **causal,
    }


def _multistart(
    spec: Condition, bundle: DatasetBundle, cfg: SearchConfig, seed: int
) -> tuple:
    """Train N independent fixed architectures; select the best on validation."""
    rng = np.random.default_rng(seed)
    best = None
    candidates = 0
    grad_steps = 0
    history = []

    for i in range(spec.restarts):
        g = (
            make_logistic(rng)
            if not spec.hidden
            else make_mlp(spec.hidden, rng)
        )
        res = train(
            g, np.array(g.weight, dtype=np.float64), bundle.train.X, bundle.train.y,
            rng, n_cycles=cfg.inner_steps, batch_size=cfg.batch_size, settle=cfg.settle,
        )
        candidates += 1
        grad_steps += res.gradient_steps
        val = total_error(g, res.weights, bundle.validation.X, bundle.validation.y, cfg.settle)
        if best is None or val < best[0]:
            best = (val, g, res.weights)
        history.append({"restart": i, "validation_loss": float(val)})

    return best[1], best[2], candidates, grad_steps, history


def _random_search(
    spec: Condition, bundle: DatasetBundle, cfg: SearchConfig, seed: int
) -> tuple:
    """Sample architectures of evolved-like size, train each, select on validation.

    The control for H2: everything evolution has except selection.
    """
    rng = np.random.default_rng(seed)
    best = None
    candidates = 0
    grad_steps = 0
    history = []
    acts = tuple(cfg.activations)

    for i in range(spec.restarts):
        n_hidden = int(rng.integers(2, 16))
        extra = int(rng.integers(0, 12))
        g = make_random_architecture(rng, n_hidden, extra, acts)
        res = train(
            g, np.array(g.weight, dtype=np.float64), bundle.train.X, bundle.train.y,
            rng, n_cycles=cfg.inner_steps, batch_size=cfg.batch_size, settle=cfg.settle,
        )
        candidates += 1
        grad_steps += res.gradient_steps
        val = total_error(g, res.weights, bundle.validation.X, bundle.validation.y, cfg.settle)
        if best is None or val < best[0]:
            best = (val, g, res.weights)
        history.append({"sample": i, "n_hidden": n_hidden, "validation_loss": float(val)})

    return best[1], best[2], candidates, grad_steps, history


def run_condition(
    condition: str,
    bundle: DatasetBundle,
    replicate: int,
    dataset_seed: int,
    search_seed: int,
    propagation: str = "settled",
    track: str = "B",
    fitness_split: str = "validation",
    config_overrides: dict | None = None,
) -> RunRecord:
    """Execute one run and return a record that never contains test metrics."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    spec = CONDITIONS[condition]
    cfg = base_config(bundle.task, propagation, fitness_split)
    if spec.overrides:
        cfg = replace(cfg, **spec.overrides)
    if config_overrides:
        cfg = replace(cfg, **config_overrides)

    started = time.time()
    if spec.kind == "evolution":
        res = search(bundle, cfg, seed=search_seed)
        g, w = res.champion.genome, res.champion.weights
        candidates, grad_steps, history = res.candidates, res.gradient_steps, res.history
    elif spec.kind == "multistart":
        g, w, candidates, grad_steps, history = _multistart(spec, bundle, cfg, search_seed)
    elif spec.kind == "random_search":
        g, w, candidates, grad_steps, history = _random_search(spec, bundle, cfg, search_seed)
    else:
        raise AssertionError(spec.kind)
    wall = time.time() - started

    metrics = _metrics(g, w, bundle, cfg.settle)
    metrics["success"] = bool(
        metrics["validation_accuracy"] >= SUCCESS_THRESHOLD[bundle.task]
    )

    return RunRecord(
        run_id=run_id(bundle.task, condition, replicate),
        task=bundle.task,
        condition=condition,
        replicate=replicate,
        dataset_seed=dataset_seed,
        search_seed=search_seed,
        track=track,
        config={
            "propagation": cfg.propagation,
            "fitness_split": cfg.fitness_split,
            "generations": cfg.generations,
            "population": cfg.population,
            "n_species": cfg.n_species,
            "inner_steps": cfg.inner_steps,
            "batch_size": cfg.batch_size,
            "use_penalty": cfg.use_penalty,
            "backprop": cfg.backprop,
            "lamarckian": cfg.lamarckian,
            "n_activations": len(cfg.activations),
            "kind": spec.kind,
            "restarts": spec.restarts if spec.kind != "evolution" else None,
            "generator": bundle.generator,
            "noise": bundle.noise,
            "success_threshold": SUCCESS_THRESHOLD[bundle.task],
        },
        metrics=metrics,
        compute={
            "candidate_evaluations": candidates,
            "gradient_steps": grad_steps,
            "wall_time_seconds": wall,
        },
        champion=serialise_genome(g, w),
        history=history,
    )
