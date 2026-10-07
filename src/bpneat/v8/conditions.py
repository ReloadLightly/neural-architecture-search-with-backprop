"""The v8 grid: four arms on real tabular data, one budget.

Three of the four arms are arms an earlier protocol already ran, lifted to
``d`` inputs and ``k`` outputs by :mod:`bpneat.nd` and byte-identical to the
frozen versions at two and one. The fourth is new, and is the reason v8 exists:

* ``search`` — NEAT at v5's reference configuration, 2,100 candidates.
* ``null`` — the same candidate count sampled instead of searched. The sampler
  is the frozen one, so a null here is the null v3 and v6 ran.
* ``linear`` — **no architecture at all.** The seed genome, bias and every input
  wired straight to the outputs, given the gradient budget the search spent and
  the same 60 restarts the fixed arm gets. On these datasets it is within two to
  five points of everything else, and the search's own champions carry zero
  causally active hidden units, so this arm is where the question actually is.
* ``fixed`` — a 32x32 mixed-operator network at the same matched budget.

The inner learner, the fitness, settled propagation, the selection operator, the
operator set, the structural rates, the population and the restart count are all
held at the reference values. What changes from v6 is the data and the fourth
arm, and nothing else.
"""

from __future__ import annotations

import time

import numpy as np

from ..nd.baselines import (
    RANDOM_EXTRA_CONNECTIONS,
    RANDOM_HIDDEN_NODES,
    make_linear,
    make_mixed_mlp,
    make_random_architecture,
    matched_multistart,
)
from ..nd.datasets import TabularBundle, make_bundle
from ..nd.encoding import causal_subgraph
from ..nd.evolve import NdConfig, layout_for, search
from ..nd.learn import accuracy, total_error, train
from ..record import serialise_genome
from ..v5.conditions import BATCH_SIZE, INNER_STEPS
from .protocol import (
    ARM_MEANING,
    ARMS,
    FIXED_HIDDEN,
    MATCHED_TO,
    MULTISTART_RESTARTS,
    REFERENCE_GENERATIONS,
    TEST_ROWS,
)

__all__ = ["CONDITIONS", "config_for", "run_condition"]


CONDITIONS: dict[str, str] = dict(ARM_MEANING)


def config_for(task: str, bundle: TabularBundle) -> NdConfig:
    """v5's reference configuration, over this dataset's own layout."""
    return NdConfig(
        task=task,
        layout=layout_for(bundle),
        generations=REFERENCE_GENERATIONS,
        population=100,
        n_species=5,
        inner_steps=INNER_STEPS,
        batch_size=BATCH_SIZE,
    )


def random_architecture_search(bundle: TabularBundle, cfg: NdConfig, seed: int, budget: int):
    """Candidate-matched random architecture search under the same learner.

    v6's ``random_architecture_search``, over this bundle's layout. Selection is
    the only thing removed: every candidate is drawn independently from the
    frozen sampler and trained by the same inner learner for the same number of
    cycles, and the one with the lowest validation error is kept.
    """
    rng = np.random.default_rng(seed)
    best = None
    steps = 0
    for _ in range(budget):
        n_hidden = int(rng.integers(*RANDOM_HIDDEN_NODES))
        extra = int(rng.integers(*RANDOM_EXTRA_CONNECTIONS))
        g = make_random_architecture(rng, n_hidden, extra, tuple(cfg.activations), cfg.layout)
        res = train(g, np.array(g.weight, dtype=np.float64), bundle.train.X, bundle.train.y,
                    rng, n_cycles=cfg.inner_steps, batch_size=cfg.batch_size, settle=cfg.settle)
        steps += res.gradient_steps
        val = total_error(g, res.weights, bundle.validation.X, bundle.validation.y, cfg.settle)
        if best is None or val < best[0]:
            best = (val, g, res.weights)
    return best[1], best[2], budget, steps


def _metrics(g, w, bundle: TabularBundle) -> dict:
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
    split_seed: int,
    search_seed: int,
    matched_steps: dict[str, int] | None = None,
) -> dict:
    """Execute one v8 run. Returns a record; never reads the sealed test split."""
    if condition not in ARMS:
        raise ValueError(f"unknown arm {condition!r}; have {ARMS}")
    bundle = make_bundle(task, seed=split_seed)
    cfg = config_for(task, bundle)
    started = time.time()
    extra: dict = {}
    matched_from: str | None = None
    used_steps: int | None = None
    history: list = []

    if condition == "search":
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
    elif condition == "null":
        g, w, candidates, steps = random_architecture_search(
            bundle, cfg, search_seed, cfg.candidate_budget
        )
    else:  # "linear" or "fixed": both matched to the search's realised budget
        matched_from = MATCHED_TO[condition]
        if not matched_steps or matched_from not in matched_steps:
            raise ValueError(f"{condition} needs this cell's {matched_from} budget")
        used_steps = int(matched_steps[matched_from])
        build = (
            (lambda rng: make_linear(rng, cfg.layout))
            if condition == "linear"
            else (lambda rng: make_mixed_mlp(FIXED_HIDDEN, rng, cfg.layout))
        )
        g, w, steps, history = matched_multistart(
            build, bundle, used_steps, MULTISTART_RESTARTS, search_seed
        )
        candidates = MULTISTART_RESTARTS

    metrics = _metrics(g, w, bundle)
    metrics["collapsed"] = bool(metrics["causal_hidden_nodes"] == 0)

    return {
        "run_id": f"{task}__{condition}__r{replicate:02d}",
        "task": task,
        "condition": condition,
        "arm": condition,
        "replicate": replicate,
        "split_seed": split_seed,
        "search_seed": search_seed,
        "config": {
            "arm": condition,
            "n_features": bundle.n_features,
            "n_classes": bundle.n_classes,
            "n_outputs": cfg.layout.n_outputs,
            "candidate_budget": cfg.candidate_budget,
            "population": cfg.population if condition == "search" else None,
            "generations": cfg.generations if condition == "search" else None,
            "inner_steps": INNER_STEPS,
            "batch_size": BATCH_SIZE,
            "restarts": MULTISTART_RESTARTS if matched_from else None,
            "fixed_hidden": list(FIXED_HIDDEN) if condition == "fixed" else None,
            "matched_to": matched_from,
            "matched_steps": used_steps,
            "source": bundle.source,
            "train_rows": len(bundle.train),
            "validation_rows": len(bundle.validation),
            # From the contract, not from the bundle. The sealed split's *size*
            # is what the equivalence margin rests on and belongs in a record,
            # but nothing outside the firewall touches ``bundle.test`` — not
            # even its length — because "the firewall is the only reader" is a
            # rule a gate can check and "may read some of it" is not.
            # ``TEST_ROWS`` is pinned to the loader by its own gate.
            "sealed_test_rows": TEST_ROWS[task],
            "constant_features": list(bundle.constant_features),
        },
        "champion": serialise_genome(g, np.asarray(w, dtype=np.float64)),
        "champion_layout": {
            "n_inputs": cfg.layout.n_inputs,
            "n_outputs": cfg.layout.n_outputs,
        },
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
