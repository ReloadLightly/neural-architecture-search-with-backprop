"""The v3 comparison matrix.

Each condition is one way of answering "what does this algorithm conclude?".
Block A removes v2's control starvation; block B separates propagation from the
fitness split; block C sweeps selection pressure; block D varies inheritance.

The matched fixed-architecture conditions receive the gradient budget that
``backprop_neat`` actually spent **in the same replicate**, so the comparison is
paired on compute rather than on candidate count alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..baselines import make_mlp, make_random_architecture
from ..datasets import DatasetBundle
from ..genome import OP_SIN, OP_TANH, causal_subgraph
from ..learn import accuracy, total_error, train
from ..record import serialise_genome
from .datasets import make_bundle
from .learners import ha_multistart, matched_multistart, mixed_mlp
from .protocol import SUCCESS_THRESHOLD
from .search import V3Config, search

#: Evolutionary budget per task. Candidate-matched controls inherit it.
GENERATIONS = {"xor": 10, "circle": 10, "spiral": 20, "checkerboard": 20, "spiral3": 20}
POPULATION = 100
MLP_HIDDEN = (32, 32)
#: Restarts for every fixed-architecture control, as in v2.
RESTARTS = 60


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str  # "evolution" | "ha_multistart" | "matched_multistart" | "random_search"
    blocks: tuple[str, ...]
    mechanism: str
    overrides: dict = field(default_factory=dict)
    builder: str = "tanh"


CONDITIONS: dict[str, Condition] = {
    "backprop_neat": Condition(
        "backprop_neat", "evolution", ("A", "B", "C", "D"),
        "the full mechanism; the shared reference cell of every block",
    ),
    "fixed_mlp_tanh_ha": Condition(
        "fixed_mlp_tanh_ha", "ha_multistart", ("A",),
        "v2's control, kept to exhibit the starvation artifact", builder="tanh",
    ),
    "fixed_mlp_tanh_matched": Condition(
        "fixed_mlp_tanh_matched", "matched_multistart", ("A",),
        "same architecture, same compute, no rollback", builder="tanh",
    ),
    "fixed_mlp_sin_matched": Condition(
        "fixed_mlp_sin_matched", "matched_multistart", ("A",),
        "operator bias without search", builder="sin",
    ),
    "fixed_mlp_mixed_matched": Condition(
        "fixed_mlp_mixed_matched", "matched_multistart", ("A",),
        "heterogeneous operators without search", builder="mixed",
    ),
    "random_search_matched": Condition(
        "random_search_matched", "random_search", ("A",),
        "removes selection, matched on candidates",
    ),
    "homogeneous_tanh": Condition(
        "homogeneous_tanh", "evolution", ("A",),
        "removes operator diversity", overrides={"activations": (OP_TANH,)},
    ),
    "evolution_only": Condition(
        "evolution_only", "evolution", ("A",),
        "removes gradient learning", overrides={"backprop": False},
    ),
    "prop_ha_fit_val": Condition(
        "prop_ha_fit_val", "evolution", ("B",),
        "Ha propagation, validation fitness",
        overrides={"propagation": "ha2016", "fitness_split": "validation"},
    ),
    "prop_ha_fit_train": Condition(
        "prop_ha_fit_train", "evolution", ("B",),
        "Ha propagation, training fitness (v2 track A)",
        overrides={"propagation": "ha2016", "fitness_split": "train"},
    ),
    "prop_settled_fit_train": Condition(
        "prop_settled_fit_train", "evolution", ("B",),
        "settled propagation, training fitness",
        overrides={"propagation": "settled", "fitness_split": "train"},
    ),
    "baldwinian": Condition(
        "baldwinian", "evolution", ("D",),
        "learned weights are not inherited", overrides={"lamarckian": False},
    ),
}

for _lvl in ("roulette_s1.0", "roulette_s0.1", "roulette_s0.001",
             "tournament_k2", "tournament_k4", "v1_truncation"):
    CONDITIONS[f"sel_{_lvl}"] = Condition(
        f"sel_{_lvl}", "evolution", ("C",),
        f"selection pressure level {_lvl}", overrides={"selector": _lvl},
    )

_BUILDERS = {
    "tanh": lambda rng: make_mlp(MLP_HIDDEN, rng, op=OP_TANH),
    "sin": lambda rng: make_mlp(MLP_HIDDEN, rng, op=OP_SIN),
    "mixed": lambda rng: mixed_mlp(MLP_HIDDEN, rng),
}


def base_config(task: str) -> V3Config:
    return V3Config(
        task=task, generations=GENERATIONS[task], population=POPULATION,
        n_species=5, inner_steps=600, batch_size=10,
        propagation="settled", fitness_split="validation",
    )


def candidate_budget(task: str) -> int:
    return POPULATION * (GENERATIONS[task] + 1)


def _metrics(g, w, bundle: DatasetBundle, settle: bool) -> dict:
    causal = causal_subgraph(g, bundle.validation.X, w, settle)
    tr = accuracy(g, w, bundle.train.X, bundle.train.y, settle)
    va = accuracy(g, w, bundle.validation.X, bundle.validation.y, settle)
    return {
        "train_accuracy": tr,
        "train_loss": total_error(g, w, bundle.train.X, bundle.train.y, settle),
        "validation_accuracy": va,
        "validation_loss": total_error(g, w, bundle.validation.X, bundle.validation.y, settle),
        "generalization_gap": tr - va,
        **causal,
    }


def _random_search(bundle, cfg, seed, budget):
    """Candidate-matched random architecture search under the same learner."""
    rng = np.random.default_rng(seed)
    best = None
    steps = 0
    for _ in range(budget):
        n_hidden = int(rng.integers(2, 16))
        extra = int(rng.integers(0, 12))
        g = make_random_architecture(rng, n_hidden, extra, tuple(cfg.activations))
        res = train(g, np.array(g.weight, dtype=np.float64), bundle.train.X, bundle.train.y,
                    rng, n_cycles=cfg.inner_steps, batch_size=cfg.batch_size, settle=cfg.settle)
        steps += res.gradient_steps
        val = total_error(g, res.weights, bundle.validation.X, bundle.validation.y, cfg.settle)
        if best is None or val < best[0]:
            best = (val, g, res.weights)
    return best[1], best[2], budget, steps


def run_condition(
    condition: str,
    task: str,
    replicate: int,
    dataset_seed: int,
    search_seed: int,
    matched_steps: int | None = None,
) -> dict:
    """Execute one v3 run. Returns a record; never reads the sealed test split."""
    spec = CONDITIONS[condition]
    bundle = make_bundle(task, seed=dataset_seed)
    cfg = base_config(task)
    if spec.overrides:
        cfg = replace(cfg, **spec.overrides)

    started = time.time()
    intensity = None
    history: list = []

    if spec.kind == "evolution":
        res = search(bundle, cfg, seed=search_seed)
        g, w = res.champion.genome, res.champion.weights
        candidates, steps, history = res.candidates, res.gradient_steps, res.history
        intensity = res.selection_intensity
    elif spec.kind == "ha_multistart":
        g, w, steps, history = ha_multistart(
            _BUILDERS[spec.builder], bundle, RESTARTS, cfg.inner_steps, search_seed
        )
        candidates = RESTARTS
    elif spec.kind == "matched_multistart":
        if matched_steps is None:
            raise ValueError(f"{condition} needs the cell's backprop_neat budget")
        g, w, steps, history = matched_multistart(
            _BUILDERS[spec.builder], bundle, matched_steps, RESTARTS, search_seed
        )
        candidates = RESTARTS
    elif spec.kind == "random_search":
        g, w, candidates, steps = _random_search(
            bundle, cfg, search_seed, candidate_budget(task)
        )
    else:
        raise AssertionError(spec.kind)

    metrics = _metrics(g, w, bundle, cfg.settle)
    metrics["success"] = bool(metrics["validation_accuracy"] >= SUCCESS_THRESHOLD[task])
    metrics["collapsed"] = bool(metrics["causal_hidden_nodes"] == 0)

    return {
        "run_id": f"{task}__{condition}__r{replicate:02d}",
        "task": task,
        "condition": condition,
        "replicate": replicate,
        "dataset_seed": dataset_seed,
        "search_seed": search_seed,
        "blocks": list(spec.blocks),
        "config": {
            "kind": spec.kind,
            "propagation": cfg.propagation,
            "fitness_split": cfg.fitness_split,
            "selector": cfg.selector,
            "lamarckian": cfg.lamarckian,
            "backprop": cfg.backprop,
            "generations": cfg.generations,
            "population": cfg.population,
            "inner_steps": cfg.inner_steps,
            "n_activations": len(cfg.activations),
            "restarts": RESTARTS if spec.kind.endswith("multistart") else None,
            "matched_steps": matched_steps,
            "builder": spec.builder if spec.kind.endswith("multistart") else None,
            "generator": bundle.generator,
            "noise": bundle.noise,
            "success_threshold": SUCCESS_THRESHOLD[task],
        },
        # The champion is stored whole so the sealed-test pass can load it
        # without retraining or reselecting anything.
        "champion": serialise_genome(g, w),
        "metrics": metrics,
        "compute": {
            "candidate_evaluations": candidates,
            "gradient_steps": steps,
            "wall_time_seconds": time.time() - started,
            "selection_intensity": intensity,
        },
        "history": history,
        "test_evaluated": False,
    }
