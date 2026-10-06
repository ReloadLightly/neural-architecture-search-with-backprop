"""The v6 grid: three arms x five budgets, and nothing else moving.

Every arm is an arm an earlier release already ran and already reported. The
only thing v6 introduces is that each of them is run at five budgets instead of
one, so that "the search does not beat its controls" can be checked against a
curve rather than a point.

* ``search`` is v5's reference configuration, imported from
  :mod:`bpneat.v5.search` unchanged. At the reference rung it is v5's
  ``neat_reference`` exactly, on fresh splits.
* ``null`` is v3's candidate-matched random architecture search. It is
  reimplemented here rather than imported, because v3's copy is a private
  function inside a frozen module and v6 must own every line its fingerprint
  covers; a gate asserts the two produce the identical champion from identical
  inputs, so "reimplemented" means transcribed, not reinterpreted.
* ``fixed`` is v3's ``matched_multistart`` on a 32x32 mixed-operator network,
  given the gradient budget the search arm actually spent at the same rung in
  the same cell. It is v5's ``fixed_mixed_matched``, parameterised by rung.

The inner learner, the fitness, the propagation, the selection operator, the
operator set, the population, the complexity penalty, the speciation, the
crossover and the restart count are all held at the reference values. Only the
generation count moves, and the two derived quantities that follow from it: the
candidate count the null is matched to, and the gradient budget the fixed arm is
matched to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from ..baselines import make_random_architecture
from ..datasets import DatasetBundle
from ..genome import causal_subgraph
from ..learn import accuracy, total_error, train
from ..record import serialise_genome
from ..v3.conditions import MLP_HIDDEN, POPULATION, RESTARTS
from ..v3.datasets import make_bundle
from ..v3.learners import matched_multistart, mixed_mlp
from ..v5.conditions import BATCH_SIZE, INNER_STEPS
from ..v5.search import REF_P_ADD_CONNECTION, REF_P_ADD_NODE, V5Config, search
from .protocol import (
    ARM_MEANING,
    EVERY_CONDITION,
    MATCHED_TO,
    SUCCESS_THRESHOLD,
    Budget,
    split_condition,
)

#: The random architecture sampler's range, transcribed from v3's
#: ``_random_search``: 2-15 hidden nodes and 0-11 connections beyond the ones
#: the constructor already lays down. It is deliberately *not* matched to the
#: sizes NEAT reaches — see the declared limitation in the preregistration.
NULL_HIDDEN_RANGE = (2, 16)
NULL_EXTRA_CONNECTION_RANGE = (0, 12)


@dataclass(frozen=True)
class Condition:
    name: str
    arm: str
    budget: Budget
    kind: str  # "evolution" | "random_search" | "matched_multistart"
    mechanism: str

    @property
    def candidate_budget(self) -> int:
        return self.budget.candidates


_KIND = {"search": "evolution", "null": "random_search", "fixed": "matched_multistart"}


def _condition(name: str) -> Condition:
    arm, budget = split_condition(name)
    return Condition(
        name=name,
        arm=arm,
        budget=budget,
        kind=_KIND[arm],
        mechanism=(
            f"{ARM_MEANING[arm]}; {budget.candidates:,} candidates "
            f"({budget.multiplier:.2f}x the reference budget)"
        ),
    )


CONDITIONS: dict[str, Condition] = {
    name: _condition(name) for name in EVERY_CONDITION
}


def base_config(task: str, budget: Budget) -> V5Config:
    """v5's reference configuration at this rung's generation count."""
    return V5Config(
        task=task,
        generations=budget.generations,
        population=POPULATION,
        n_species=5,
        inner_steps=INNER_STEPS,
        batch_size=BATCH_SIZE,
        p_add_node=REF_P_ADD_NODE,
        p_add_connection=REF_P_ADD_CONNECTION,
    )


def config_for(condition: str, task: str) -> V5Config:
    spec = CONDITIONS[condition]
    cfg = base_config(task, spec.budget)
    if spec.kind != "evolution":
        # The non-evolution arms do not reproduce, so a generation count would
        # be a number in the record that nothing reads. The candidate budget
        # they are matched to is carried by the condition, not by the config.
        cfg = replace(cfg, generations=0)
    return cfg


def random_architecture_search(bundle: DatasetBundle, cfg: V5Config, seed: int, budget: int):
    """Candidate-matched random architecture search under the same learner.

    Transcribed from ``bpneat.v3.conditions._random_search``. Selection is the
    only thing removed: every candidate is drawn independently from the sampler
    and trained by the same inner learner for the same number of cycles, and the
    one with the lowest validation error is kept. Because the generator's stream
    does not depend on ``budget``, a shorter run is a prefix of a longer one —
    the nesting the contract relies on.
    """
    rng = np.random.default_rng(seed)
    best = None
    steps = 0
    for _ in range(budget):
        n_hidden = int(rng.integers(*NULL_HIDDEN_RANGE))
        extra = int(rng.integers(*NULL_EXTRA_CONNECTION_RANGE))
        g = make_random_architecture(rng, n_hidden, extra, tuple(cfg.activations))
        res = train(g, np.array(g.weight, dtype=np.float64), bundle.train.X, bundle.train.y,
                    rng, n_cycles=cfg.inner_steps, batch_size=cfg.batch_size, settle=cfg.settle)
        steps += res.gradient_steps
        val = total_error(g, res.weights, bundle.validation.X, bundle.validation.y, cfg.settle)
        if best is None or val < best[0]:
            best = (val, g, res.weights)
    return best[1], best[2], budget, steps


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
    """Execute one v6 run. Returns a record; never reads the sealed test split."""
    spec = CONDITIONS[condition]
    bundle = make_bundle(task, seed=dataset_seed)
    cfg = config_for(condition, task)
    started = time.time()
    extra: dict = {}
    matched_from: str | None = None
    used_steps: int | None = None
    history: list = []

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
    elif spec.kind == "random_search":
        g, w, candidates, steps = random_architecture_search(
            bundle, cfg, search_seed, spec.candidate_budget
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
    else:  # pragma: no cover - CONDITIONS is closed over the three kinds above
        raise AssertionError(spec.kind)

    metrics = _metrics(g, w, bundle)
    metrics["success"] = bool(metrics["validation_accuracy"] >= SUCCESS_THRESHOLD[task])
    metrics["collapsed"] = bool(metrics["causal_hidden_nodes"] == 0)

    return {
        "run_id": f"{task}__{condition}__r{replicate:02d}",
        "task": task,
        "condition": condition,
        "arm": spec.arm,
        "budget": spec.budget.label,
        "candidate_budget": spec.candidate_budget,
        "budget_multiplier": spec.budget.multiplier,
        "replicate": replicate,
        "dataset_seed": dataset_seed,
        "search_seed": search_seed,
        "config": {
            "kind": spec.kind,
            "arm": spec.arm,
            "budget": spec.budget.label,
            "candidate_budget": spec.candidate_budget,
            "population": POPULATION if spec.kind == "evolution" else None,
            "generations": spec.budget.generations if spec.kind == "evolution" else None,
            "p_add_node": cfg.p_add_node,
            "p_add_connection": cfg.p_add_connection,
            "use_penalty": cfg.use_penalty,
            "n_species": cfg.n_species,
            "crossover": cfg.crossover,
            "inner_steps": INNER_STEPS,
            "batch_size": BATCH_SIZE,
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
