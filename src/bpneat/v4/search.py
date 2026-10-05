"""CGP search: ``(1+lambda)`` with neutral drift, and its candidate-matched null.

The loop is the standard CGP evolutionary strategy. One parent produces
``LAMBDA`` offspring per generation; every offspring is trained by
:func:`bpneat.v4.learners.rollback_train` and scored on the validation split
with the same penalised fitness Backprop-NEAT uses; the best offspring replaces
the parent when its fitness is **greater than or equal to** the parent's.

That equality is the mechanism, not a detail. An offspring whose active
phenotype is unchanged but whose inactive genes have moved scores exactly the
same, and is accepted — so the genotype random-walks through neutral space
while fitness is held, and a later single mutation can activate a region that
took many mutations to assemble. It is CGP's answer to the problem NEAT solves
with innovation numbers and weak selection, and it is a genuinely different
answer, which is the point of running it.

The parent is evaluated once at birth and never re-evaluated, so a generation
costs exactly ``LAMBDA`` candidate evaluations and the candidate budget is
comparable with Backprop-NEAT's ``population * (generations + 1)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..datasets import DatasetBundle
from ..learn import fitness_from_error
from ..v3 import dense
from . import cgp
from .learners import dense_accuracy, dense_total_error, rollback_train

#: Offspring per generation. 4 is the common CGP setting (Miller's 1+4 ES).
LAMBDA = 4


@dataclass
class Candidate:
    genotype: cgp.Genotype
    fitness: float
    error: float
    gradient_steps: int
    pheno: cgp.Phenotype
    weights: np.ndarray


@dataclass
class CGPResult:
    champion: Candidate
    history: list[dict] = field(default_factory=list)
    candidates: int = 0
    gradient_steps: int = 0
    wall_time: float = 0.0
    generations: int = 0
    accepted: int = 0
    neutral_accepted: int = 0


def evaluate(
    gt: cgp.Genotype,
    bundle: DatasetBundle,
    inner_steps: int,
    batch_size: int,
    rng: np.random.Generator,
    lamarckian: bool = True,
) -> Candidate:
    """Decode, train, score on validation. Never reads ``bundle.test``."""
    pheno = cgp.decode(gt)
    g = pheno.genome
    plan = dense.plan(g)
    w0 = np.array(g.weight, dtype=np.float64)
    res = rollback_train(
        g, w0, bundle.train.X, bundle.train.y, rng,
        n_cycles=inner_steps, batch_size=batch_size, plan=plan,
    )
    if lamarckian:
        cgp.write_back(gt, pheno, res.weights)
    err = dense_total_error(plan, res.weights, bundle.validation.X, bundle.validation.y)
    return Candidate(
        genotype=gt,
        fitness=fitness_from_error(g, err, use_penalty=True),
        error=err,
        gradient_steps=res.gradient_steps,
        pheno=pheno,
        weights=res.weights,
    )


def cgp_search(
    bundle: DatasetBundle,
    budget: int,
    seed: int,
    inner_steps: int = 600,
    batch_size: int = 10,
    n_func: int = cgp.N_FUNC,
    lam: int = LAMBDA,
) -> CGPResult:
    """One CGP run under a candidate budget. Never reads ``bundle.test``."""
    rng = np.random.default_rng(seed)
    started = time.time()

    parent = evaluate(cgp.random_genotype(rng, n_func), bundle, inner_steps, batch_size, rng)
    champion = parent
    candidates = 1
    grad_steps = parent.gradient_steps
    history: list[dict] = []
    accepted = neutral = 0
    gen = 0

    while candidates + lam <= budget:
        gen += 1
        brood = [
            evaluate(cgp.mutate(parent.genotype, rng), bundle, inner_steps, batch_size, rng)
            for _ in range(lam)
        ]
        candidates += lam
        grad_steps += sum(c.gradient_steps for c in brood)

        best = max(brood, key=lambda c: c.fitness)
        if best.fitness >= parent.fitness:
            if best.fitness == parent.fitness:
                neutral += 1
            accepted += 1
            parent = best
        if best.fitness > champion.fitness:
            champion = best

        # One row per generation would be 525 rows per run; sample instead, and
        # always keep the first and the last.
        if gen == 1 or gen % 25 == 0:
            history.append(
                {
                    "generation": gen,
                    "best_fitness": float(best.fitness),
                    "parent_fitness": float(parent.fitness),
                    "champion_fitness": float(champion.fitness),
                    "active_nodes": len(parent.pheno.active),
                    "candidates": candidates,
                    "gradient_steps": grad_steps,
                }
            )

    if not history or history[-1]["generation"] != gen:
        history.append(
            {
                "generation": gen,
                "best_fitness": float(parent.fitness),
                "parent_fitness": float(parent.fitness),
                "champion_fitness": float(champion.fitness),
                "active_nodes": len(parent.pheno.active),
                "candidates": candidates,
                "gradient_steps": grad_steps,
            }
        )

    return CGPResult(
        champion=champion,
        history=history,
        candidates=candidates,
        gradient_steps=grad_steps,
        wall_time=time.time() - started,
        generations=gen,
        accepted=accepted,
        neutral_accepted=neutral,
    )


def cgp_random_search(
    bundle: DatasetBundle,
    budget: int,
    seed: int,
    inner_steps: int = 600,
    batch_size: int = 10,
    n_func: int = cgp.N_FUNC,
) -> CGPResult:
    """The null for CGP: the same genotype space and learner, no selection.

    ``budget`` genotypes are drawn independently and trained; the one with the
    lowest validation loss is kept. Candidate-matched to :func:`cgp_search`, so
    any difference is attributable to selection and neutral drift alone.
    """
    rng = np.random.default_rng(seed)
    started = time.time()
    best: Candidate | None = None
    grad_steps = 0

    for _ in range(budget):
        cand = evaluate(
            cgp.random_genotype(rng, n_func), bundle, inner_steps, batch_size, rng
        )
        grad_steps += cand.gradient_steps
        if best is None or cand.error < best.error:
            best = cand

    assert best is not None
    return CGPResult(
        champion=best,
        history=[],
        candidates=budget,
        gradient_steps=grad_steps,
        wall_time=time.time() - started,
        generations=0,
    )


def champion_metrics(cand: Candidate, bundle: DatasetBundle) -> dict:
    """Validation-side metrics for a CGP champion, on the dense evaluator."""
    plan = dense.plan(cand.pheno.genome)
    tr = dense_accuracy(plan, cand.weights, bundle.train.X, bundle.train.y)
    va = dense_accuracy(plan, cand.weights, bundle.validation.X, bundle.validation.y)
    return {
        "train_accuracy": tr,
        "train_loss": dense_total_error(plan, cand.weights, bundle.train.X, bundle.train.y),
        "validation_accuracy": va,
        "validation_loss": dense_total_error(
            plan, cand.weights, bundle.validation.X, bundle.validation.y
        ),
        "generalization_gap": tr - va,
    }
