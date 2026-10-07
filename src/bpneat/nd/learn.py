"""The frozen inner learner, with the single output unfixed.

Read alongside :mod:`bpneat.learn`. The optimiser, every hyperparameter, the
rollback rule and the complexity penalty are the frozen ones, imported rather
than restated. What generalises is the loss: one output keeps the binary
cross-entropy computed from a logit, and ``k`` outputs get softmax
cross-entropy over ``k`` logits.

At ``k = 1`` the softmax branch is never taken and the binary branch is the
frozen expression on the frozen logits, so the loss, the accuracy, the gradient
signal, the rollback decisions and the returned weights are bit-identical.
``tests/test_nd_equivalence.py`` requires that, including the trained weights
after a full inner budget, which is the strongest statement available: the two
learners take the same path through the same loss surface, not merely to the
same place.

Why softmax rather than ``k`` independent sigmoids: the multi-class datasets
this exists for have exactly one true class per row, and a softmax says so,
while independent sigmoids let a graph assert two classes at once and then be
scored as if it had not. The binary case is not a softmax over one logit, which
is degenerate — it stays the frozen expression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..learn import (
    BATCH_SIZE,
    CLIPVAL,
    DECAY_RATE,
    LEARN_RATE,
    PENALTY_CONNECTION_FACTOR,
    PENALTY_NODE_FACTOR,
    REGC,
    ROLLBACK_EVERY,
    SMOOTH_EPS,
    RMSProp,
    _sigmoid,
)
from .encoding import NdGenome, backward, forward

__all__ = [
    "BATCH_SIZE",
    "CLIPVAL",
    "DECAY_RATE",
    "LEARN_RATE",
    "REGC",
    "ROLLBACK_EVERY",
    "SMOOTH_EPS",
    "LearnResult",
    "RMSProp",
    "accuracy",
    "fitness_from_error",
    "penalty_factor",
    "total_error",
    "train",
]


def _logits(tape, k: int) -> np.ndarray:
    if k == 1:
        return tape.vals[tape.out_vars[0]]
    return np.stack([tape.vals[v] for v in tape.out_vars], axis=1)


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def _class_index(y: np.ndarray) -> np.ndarray:
    """Labels as integer class indices, however they were handed over."""
    arr = np.asarray(y)
    if arr.ndim == 2:  # one-hot
        return arr.argmax(axis=1)
    return arr.astype(np.int64)


def total_error(
    g: NdGenome, weights: np.ndarray, X: np.ndarray, y: np.ndarray, settle: bool = True
) -> float:
    """Mean cross-entropy over a whole split.

    One output: ``softplus(z) - y*z``, the frozen expression — smooth and exact
    in the saturated regime, where clipping a probability would flatten the loss
    and silently break the correspondence with the analytic gradient. Several
    outputs: the same quantity for a softmax, computed by the same reasoning as
    ``logsumexp(z) - z[true]`` rather than as the log of a probability.
    """
    tape = forward(g, X, weights, settle=settle)
    k = g.layout.n_outputs
    z = _logits(tape, k)
    if k == 1:
        return float(np.mean(np.logaddexp(0.0, z) - y * z))
    idx = _class_index(y)
    shifted = z - z.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(logsumexp - shifted[np.arange(len(idx)), idx]))


def accuracy(
    g: NdGenome, weights: np.ndarray, X: np.ndarray, y: np.ndarray, settle: bool = True
) -> float:
    tape = forward(g, X, weights, settle=settle)
    k = g.layout.n_outputs
    z = _logits(tape, k)
    if k == 1:
        return float(np.mean((z > 0.0) == (y > 0.5)))
    return float(np.mean(z.argmax(axis=1) == _class_index(y)))


def _d_out(g: NdGenome, z: np.ndarray, yb: np.ndarray, batch_size: int) -> np.ndarray:
    if g.layout.n_outputs == 1:
        return (_sigmoid(z) - yb) / batch_size
    onehot = np.zeros_like(z)
    onehot[np.arange(len(z)), _class_index(yb)] = 1.0
    return (_softmax_rows(z) - onehot) / batch_size


@dataclass
class LearnResult:
    weights: np.ndarray
    error: float
    gradient_steps: int


def train(
    g: NdGenome,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_cycles: int = 600,
    batch_size: int = BATCH_SIZE,
    settle: bool = True,
) -> LearnResult:
    """Train one candidate under the fixed inner budget, with source rollback.

    The frozen loop, including the rollback every 20 updates against the full
    training split and the two final reverts. Realized gradient steps are
    usually well below the nominal budget because of it, which is why they are
    reported rather than assumed.
    """
    n = len(X)
    weights = weights.astype(np.float64, copy=True)
    init_error = total_error(g, weights, X, y, settle)
    very_init_error = init_error
    backup = weights.copy()
    orig_backup = weights.copy()
    solver = RMSProp(g.n_connections)
    steps = 0

    for j in range(n_cycles):
        idx = rng.integers(0, n, batch_size)
        xb, yb = X[idx], y[idx]

        tape = forward(g, xb, weights, settle=settle)
        d_out = _d_out(g, _logits(tape, g.layout.n_outputs), yb, batch_size)
        grad = backward(tape, d_out, weights)
        solver.step(weights, grad)
        steps += 1

        if j > 0 and j % ROLLBACK_EVERY == 0:
            err = total_error(g, weights, X, y, settle)
            if err > init_error:
                weights = backup.copy()
                break
            init_error = err
            backup = weights.copy()

    err = total_error(g, weights, X, y, settle)
    if err > init_error:
        err = init_error
        weights = backup.copy()
    if err > very_init_error:
        err = very_init_error
        weights = orig_backup.copy()

    return LearnResult(weights=weights, error=err, gradient_steps=steps)


def penalty_factor(g: NdGenome) -> float:
    """``1 + 0.03*sqrt(connections)`` — the reference complexity penalty.

    The frozen expression subtracts 3 from the node count, which is its bias and
    two inputs; here it is whatever the layout says those are, so the penalty
    counts the same thing — nodes beyond the ones every graph starts with.
    """
    free = g.layout.n_structural - g.layout.n_outputs
    return (
        1.0
        + PENALTY_NODE_FACTOR * np.sqrt(max(g.n_nodes - free, 0))
        + PENALTY_CONNECTION_FACTOR * np.sqrt(g.n_connections)
    )


def fitness_from_error(g: NdGenome, error: float, use_penalty: bool = True) -> float:
    """Higher is better. Reference fitness is ``-error * penaltyFactor``."""
    return -error * (penalty_factor(g) if use_penalty else 1.0)
