"""The v4 comparison matrix: two search algorithms through the same controls.

Eight conditions per cell. Two are searches — Backprop-NEAT, imported from the
released v3 code and run unchanged, and CGP, defined in :mod:`bpneat.v4.search`.
One is the unmatched fixed baseline that produced v3's artifact. Four are fixed
baselines given the gradient budget a search actually spent *in that same
replicate*, two per algorithm, because the two algorithms do not spend the same
budget and a shared matched arm would silently favour one of them. The last
removes selection from CGP while keeping its genotype space and its learner.

The design question is the sign of ``search - fixed``. If that sign is set by
the budget protocol, it flips for both algorithms at the same place. If it is
set by the search algorithm, the two algorithms disagree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..baselines import make_mlp
from ..datasets import DatasetBundle
from ..genome import OP_SIN, OP_TANH, causal_subgraph
from ..learn import accuracy, total_error
from ..record import serialise_genome
from ..v3.conditions import GENERATIONS, MLP_HIDDEN, POPULATION, RESTARTS
from ..v3.conditions import base_config as v3_base_config
from ..v3.datasets import make_bundle
from ..v3.learners import ha_multistart, matched_multistart, mixed_mlp
from ..v3.search import search as v3_search
from . import cgp
from .protocol import MATCHED_TO, SUCCESS_THRESHOLD
from .search import LAMBDA, cgp_random_search, cgp_search

#: CGP's inner budget, identical to Backprop-NEAT's.
INNER_STEPS = 600
BATCH_SIZE = 10


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str  # "bpneat" | "cgp" | "cgp_random" | "ha_multistart" | "matched_multistart"
    blocks: tuple[str, ...]
    mechanism: str
    builder: str | None = None


CONDITIONS: dict[str, Condition] = {
    "bpneat": Condition(
        "bpneat", "bpneat", ("E", "G"),
        "released v3 Backprop-NEAT, re-run unchanged on fresh replicates",
    ),
    "cgp": Condition(
        "cgp", "cgp", ("F", "G"),
        "CGP (1+4) with neutral drift; same operators, learner, fitness and budget",
    ),
    "fixed_tanh_ha": Condition(
        "fixed_tanh_ha", "ha_multistart", ("E", "F"),
        "the unmatched fixed control that produced v3's artifact", builder="tanh",
    ),
    "fixed_tanh_matched_bpneat": Condition(
        "fixed_tanh_matched_bpneat", "matched_multistart", ("E",),
        "same architecture at Backprop-NEAT's realized budget", builder="tanh",
    ),
    "fixed_mixed_matched_bpneat": Condition(
        "fixed_mixed_matched_bpneat", "matched_multistart", ("E",),
        "heterogeneous operators at Backprop-NEAT's realized budget", builder="mixed",
    ),
    "fixed_tanh_matched_cgp": Condition(
        "fixed_tanh_matched_cgp", "matched_multistart", ("F",),
        "same architecture at CGP's realized budget", builder="tanh",
    ),
    "fixed_mixed_matched_cgp": Condition(
        "fixed_mixed_matched_cgp", "matched_multistart", ("F",),
        "heterogeneous operators at CGP's realized budget", builder="mixed",
    ),
    "cgp_random_matched": Condition(
        "cgp_random_matched", "cgp_random", ("F",),
        "CGP genotypes drawn rather than selected, matched on candidates",
    ),
}

_BUILDERS = {
    "tanh": lambda rng: make_mlp(MLP_HIDDEN, rng, op=OP_TANH),
    "sin": lambda rng: make_mlp(MLP_HIDDEN, rng, op=OP_SIN),
    "mixed": lambda rng: mixed_mlp(MLP_HIDDEN, rng),
}


def candidate_budget(task: str) -> int:
    """Backprop-NEAT's candidate count, which CGP and the null both inherit."""
    return POPULATION * (GENERATIONS[task] + 1)


def _metrics(g, w, bundle: DatasetBundle, settle: bool = True) -> dict:
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


def run_condition(
    condition: str,
    task: str,
    replicate: int,
    dataset_seed: int,
    search_seed: int,
    matched_steps: dict[str, int] | None = None,
) -> dict:
    """Execute one v4 run. Returns a record; never reads the sealed test split.

    ``matched_steps`` maps a reference condition name to the gradient budget it
    realized in this same cell. A matched arm without its reference's budget is
    an error, never a silent fallback to a nominal number.
    """
    spec = CONDITIONS[condition]
    bundle = make_bundle(task, seed=dataset_seed)
    budget = candidate_budget(task)
    started = time.time()
    extra: dict = {}
    matched_from: str | None = None
    used_steps: int | None = None

    if spec.kind == "bpneat":
        cfg = v3_base_config(task)
        res = v3_search(bundle, cfg, seed=search_seed)
        g, w = res.champion.genome, res.champion.weights
        candidates, steps, history = res.candidates, res.gradient_steps, res.history
        extra["selection_intensity"] = res.selection_intensity

    elif spec.kind == "cgp":
        res = cgp_search(
            bundle, budget=budget, seed=search_seed,
            inner_steps=INNER_STEPS, batch_size=BATCH_SIZE,
        )
        g, w = res.champion.pheno.genome, res.champion.weights
        candidates, steps, history = res.candidates, res.gradient_steps, res.history
        extra.update(
            {
                "cgp_generations": res.generations,
                "cgp_accepted": res.accepted,
                "cgp_neutral_accepted": res.neutral_accepted,
                "cgp_active_nodes": len(res.champion.pheno.active),
                **cgp.genotype_summary(res.champion.genotype),
            }
        )

    elif spec.kind == "cgp_random":
        res = cgp_random_search(
            bundle, budget=budget, seed=search_seed,
            inner_steps=INNER_STEPS, batch_size=BATCH_SIZE,
        )
        g, w = res.champion.pheno.genome, res.champion.weights
        candidates, steps, history = res.candidates, res.gradient_steps, res.history
        extra.update(
            {
                "cgp_active_nodes": len(res.champion.pheno.active),
                **cgp.genotype_summary(res.champion.genotype),
            }
        )

    elif spec.kind == "ha_multistart":
        g, w, steps, history = ha_multistart(
            _BUILDERS[spec.builder], bundle, RESTARTS, INNER_STEPS, search_seed
        )
        candidates = RESTARTS

    elif spec.kind == "matched_multistart":
        matched_from = MATCHED_TO[condition]
        if not matched_steps or matched_from not in matched_steps:
            raise ValueError(f"{condition} needs this cell's {matched_from} budget")
        used_steps = int(matched_steps[matched_from])
        g, w, steps, history = matched_multistart(
            _BUILDERS[spec.builder], bundle, used_steps, RESTARTS, search_seed
        )
        candidates = RESTARTS

    else:  # pragma: no cover - CONDITIONS is closed over the kinds above
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
        "blocks": list(spec.blocks),
        "config": {
            "kind": spec.kind,
            "candidate_budget": budget,
            "inner_steps": INNER_STEPS,
            "batch_size": BATCH_SIZE,
            "population": POPULATION if spec.kind == "bpneat" else None,
            "generations": GENERATIONS[task] if spec.kind == "bpneat" else None,
            "cgp_lambda": LAMBDA if spec.kind == "cgp" else None,
            "cgp_n_func": cgp.N_FUNC if spec.kind in ("cgp", "cgp_random") else None,
            "restarts": RESTARTS if spec.kind.endswith("multistart") else None,
            "builder": spec.builder,
            "matched_to": matched_from,
            "matched_steps": used_steps,
            "generator": bundle.generator,
            "noise": bundle.noise,
            "success_threshold": SUCCESS_THRESHOLD[task],
        },
        # Stored whole so the sealed-test pass loads exactly what validation
        # selected, without retraining or reselecting anything.
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
