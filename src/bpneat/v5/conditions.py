"""The v5 comparison matrix: NEAT's five mechanisms, one factor at a time.

Each condition changes exactly one thing against the reference configuration,
except ``neat_complexify_no_penalty``, which changes the two that oppose each
other and is declared as a planned interaction rather than discovered as one.

The yardstick is kept: ``fixed_mixed_matched`` is the fixed 32x32 network with
heterogeneous operators, trained on the gradient budget the reference actually
spent in the same replicate. It is the arm that beat every evolved champion in
v3 and v4, and it is here so that "does any NEAT variant beat a fixed network"
has an answer rather than an inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..datasets import DatasetBundle
from ..genome import causal_subgraph
from ..learn import accuracy, total_error
from ..record import serialise_genome
from ..v3.conditions import GENERATIONS, MLP_HIDDEN, POPULATION, RESTARTS
from ..v3.datasets import make_bundle
from ..v3.learners import matched_multistart, mixed_mlp
from .protocol import MATCHED_TO, SUCCESS_THRESHOLD
from .search import REF_P_ADD_CONNECTION, REF_P_ADD_NODE, V5Config, search

INNER_STEPS = 600
BATCH_SIZE = 10

#: The amplified complexification rate. 0.5 against the reference's 0.2 roughly
#: triples the expected node additions per lineage, which over twenty
#: generations is the difference between ~4 and ~10 added nodes.
HIGH_P_ADD_NODE = 0.5
HIGH_P_ADD_CONNECTION = 0.8

#: The width/depth trade at a fixed candidate budget: a quarter of the
#: population for four times the generations, so a lineage sees four times as
#: many mutation events for the same number of evaluations.
NARROW_POPULATION = 25
DEEP_FACTOR = 4


@dataclass(frozen=True)
class Condition:
    name: str
    mechanism: str
    kind: str = "evolution"  # or "matched_multistart"
    overrides: dict = field(default_factory=dict)


CONDITIONS: dict[str, Condition] = {
    "neat_reference": Condition(
        "neat_reference",
        "the reference configuration: p_add_node 0.2, penalty on, 5 species, "
        "crossover on, population 100",
    ),
    "neat_complexify": Condition(
        "neat_complexify",
        "raise the structural mutation rates; everything else at reference",
        overrides={
            "p_add_node": HIGH_P_ADD_NODE,
            "p_add_connection": HIGH_P_ADD_CONNECTION,
        },
    ),
    "neat_no_penalty": Condition(
        "neat_no_penalty",
        "remove the complexity penalty that opposes complexification",
        overrides={"use_penalty": False},
    ),
    "neat_complexify_no_penalty": Condition(
        "neat_complexify_no_penalty",
        "both of the above: the planned interaction, declared in advance",
        overrides={
            "p_add_node": HIGH_P_ADD_NODE,
            "p_add_connection": HIGH_P_ADD_CONNECTION,
            "use_penalty": False,
        },
    ),
    "neat_no_speciation": Condition(
        "neat_no_speciation",
        "one species: remove the mechanism that is supposed to protect new structure",
        overrides={"n_species": 1},
    ),
    "neat_no_crossover": Condition(
        "neat_no_crossover",
        "mutation only: remove historical-marking recombination",
        overrides={"crossover": False},
    ),
    "neat_deep_narrow": Condition(
        "neat_deep_narrow",
        "a quarter of the population for four times the generations, same budget",
        overrides={"population": NARROW_POPULATION},
    ),
    "fixed_mixed_matched": Condition(
        "fixed_mixed_matched",
        "the yardstick: a fixed 32x32 heterogeneous-operator network at the "
        "reference's realized gradient budget",
        kind="matched_multistart",
    ),
}


def base_config(task: str) -> V5Config:
    return V5Config(
        task=task,
        generations=GENERATIONS[task],
        population=POPULATION,
        n_species=5,
        inner_steps=INNER_STEPS,
        batch_size=BATCH_SIZE,
        p_add_node=REF_P_ADD_NODE,
        p_add_connection=REF_P_ADD_CONNECTION,
    )


def config_for(condition: str, task: str) -> V5Config:
    cfg = base_config(task)
    spec = CONDITIONS[condition]
    if spec.overrides:
        cfg = replace(cfg, **spec.overrides)
    if condition == "neat_deep_narrow":
        # Hold the candidate budget, not the generation count.
        cfg = replace(cfg, generations=(cfg.generations + 1) * DEEP_FACTOR - 1)
    return cfg


def _metrics(g, w, bundle: DatasetBundle) -> dict:
    causal = causal_subgraph(g, bundle.validation.X, w, True)
    tr = accuracy(g, w, bundle.train.X, bundle.train.y, True)
    va = accuracy(g, w, bundle.validation.X, bundle.validation.y, True)
    return {
        "train_accuracy": tr,
        "train_loss": total_error(g, w, bundle.train.X, bundle.train.y, True),
        "validation_accuracy": va,
        "validation_loss": total_error(g, w, bundle.validation.X, bundle.validation.y, True),
        "generalization_gap": tr - va,
        **causal,
    }


def run_condition(
    condition: str,
    task: str,
    replicate: int,
    dataset_seed: int,
    search_seed: int,
    matched_steps: dict[str, int] | None = None,
) -> dict:
    """Execute one v5 run. Returns a record; never reads the sealed test split."""
    spec = CONDITIONS[condition]
    bundle = make_bundle(task, seed=dataset_seed)
    cfg = config_for(condition, task)
    started = time.time()
    extra: dict = {}
    matched_from: str | None = None
    used_steps: int | None = None

    if spec.kind == "evolution":
        res = search(bundle, cfg, seed=search_seed)
        g, w = res.champion.genome, res.champion.weights
        candidates, steps, history = res.candidates, res.gradient_steps, res.history
        extra.update(
            {
                "node_additions": res.node_additions,
                "connection_additions": res.connection_additions,
                "crossovers": res.crossovers,
            }
        )
    elif spec.kind == "matched_multistart":
        matched_from = MATCHED_TO[condition]
        if not matched_steps or matched_from not in matched_steps:
            raise ValueError(f"{condition} needs this cell's {matched_from} budget")
        used_steps = int(matched_steps[matched_from])
        g, w, steps, history = matched_multistart(
            lambda rng: mixed_mlp(MLP_HIDDEN, rng),
            bundle, used_steps, RESTARTS, search_seed,
        )
        candidates = RESTARTS
    else:  # pragma: no cover - CONDITIONS is closed over the two kinds above
        raise AssertionError(spec.kind)

    metrics = _metrics(g, w, bundle)
    metrics["success"] = bool(metrics["validation_accuracy"] >= SUCCESS_THRESHOLD[task])
    metrics["collapsed"] = bool(metrics["causal_hidden_nodes"] == 0)

    return {
        "run_id": f"{task}__{condition}__r{replicate:02d}",
        "task": task,
        "condition": condition,
        "replicate": replicate,
        "dataset_seed": dataset_seed,
        "search_seed": search_seed,
        "config": {
            "kind": spec.kind,
            "p_add_node": cfg.p_add_node,
            "p_add_connection": cfg.p_add_connection,
            "use_penalty": cfg.use_penalty,
            "n_species": cfg.n_species,
            "crossover": cfg.crossover,
            "population": cfg.population,
            "generations": cfg.generations,
            "candidate_budget": cfg.candidate_budget,
            "inner_steps": INNER_STEPS,
            "restarts": RESTARTS if spec.kind == "matched_multistart" else None,
            "matched_to": matched_from,
            "matched_steps": used_steps,
            "generator": bundle.generator,
            "noise": bundle.noise,
            "success_threshold": SUCCESS_THRESHOLD[task],
        },
        "champion": serialise_genome(g, np.asarray(w, dtype=np.float64)),
        "metrics": metrics,
        "compute": {
            "candidate_evaluations": candidates,
            "gradient_steps": steps,
            "wall_time_seconds": time.time() - started,
            **extra,
        },
        "history": history,
        "test_evaluated": False,
    }
