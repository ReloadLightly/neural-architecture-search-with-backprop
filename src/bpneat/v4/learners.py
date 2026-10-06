"""The inner learner for CGP candidates: Ha's rollback rule, on the dense path.

Backprop-NEAT candidates are trained by :func:`bpneat.learn.train` — RMSProp
with a rollback check every twenty updates against the full training split,
breaking at the first increase. The cross-algorithm comparison in v4 only means
something if the second algorithm's candidates are trained by *the same* rule,
so this module reimplements exactly that rule on :mod:`bpneat.v3.dense`.

Two reasons it cannot simply call the frozen function:

* ``forward(..., settle=True)`` runs at most ``SETTLE_MAX_TICK = 16`` ticks. A
  48-node CGP row can decode to a chain longer than that, and the frozen
  evaluator would then return a value taken before the deepest nodes had run.
  The dense evaluator is exact at any depth.
* CGP candidates are small but numerous — 2100 per run, 600 nominal steps each
  — and the Python-per-node path would make the arm cost more than the whole
  rest of the study.

:func:`rollback_train` is the frozen algorithm, line for line, with
:func:`bpneat.genome.forward` / :func:`bpneat.genome.backward` replaced by their
dense equivalents. ``tests/test_v4.py`` asserts the two agree on realized step
count exactly and on final weights to 1e-8 over randomly generated acyclic
genomes, which is the tolerance matrix-product rounding allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..genome import OUT, Genome
from ..learn import (
    BATCH_SIZE,
    ROLLBACK_EVERY,
    RMSProp,
    _sigmoid,
)
from ..v3 import dense


@dataclass
class DenseLearnResult:
    weights: np.ndarray
    error: float
    gradient_steps: int


def dense_total_error(
    plan: dense.DensePlan, weights: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """Mean binary cross-entropy from logits, matching :func:`bpneat.learn.total_error`."""
    z = dense.forward(plan, weights, X).V[:, OUT]
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def dense_accuracy(
    plan: dense.DensePlan, weights: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    z = dense.forward(plan, weights, X).V[:, OUT]
    return float(np.mean((z > 0.0) == (y > 0.5)))


def rollback_train(
    g: Genome,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_cycles: int = 600,
    batch_size: int = BATCH_SIZE,
    plan: dense.DensePlan | None = None,
) -> DenseLearnResult:
    """:func:`bpneat.learn.train` on the dense evaluator. Same rule, same budget."""
    if plan is None:
        plan = dense.plan(g)
    n = len(X)
    weights = weights.astype(np.float64, copy=True)
    init_error = dense_total_error(plan, weights, X, y)
    very_init_error = init_error
    backup = weights.copy()
    orig_backup = weights.copy()
    solver = RMSProp(g.n_connections)
    steps = 0

    for j in range(n_cycles):
        idx = rng.integers(0, n, batch_size)
        xb, yb = X[idx], y[idx]

        tape = dense.forward(plan, weights, xb)
        d_out = (_sigmoid(tape.V[:, OUT]) - yb) / batch_size
        grad = dense.backward(plan, weights, tape, d_out)
        solver.step(weights, grad)
        steps += 1

        if j > 0 and j % ROLLBACK_EVERY == 0:
            err = dense_total_error(plan, weights, X, y)
            if err > init_error:
                weights = backup.copy()
                break
            init_error = err
            backup = weights.copy()

    err = dense_total_error(plan, weights, X, y)
    if err > init_error:
        err = init_error
        weights = backup.copy()
    if err > very_init_error:
        err = very_init_error
        weights = orig_backup.copy()

    return DenseLearnResult(weights=weights, error=err, gradient_steps=steps)
