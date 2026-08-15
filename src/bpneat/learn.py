"""The inner learner: RMSProp on every evaluated candidate, with rollback.

This mirrors ``fitnessFunc`` in ``datafit-neat.js``. Every candidate the search
evaluates is trained under the same fixed budget, so fitness measures how well
an architecture *can learn* under that budget rather than how well it could do
with hypothetical optimal weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genome import Genome, backward, forward

LEARN_RATE = 0.01
REGC = 0.001
CLIPVAL = 5.0
DECAY_RATE = 0.999
SMOOTH_EPS = 1e-8
BATCH_SIZE = 10
ROLLBACK_EVERY = 20
PENALTY_NODE_FACTOR = 0.00
PENALTY_CONNECTION_FACTOR = 0.03


class RMSProp:
    """Per-genome optimiser state. Never shared between genomes."""

    __slots__ = ("cache",)

    def __init__(self, n: int) -> None:
        self.cache = np.zeros(n)

    def step(self, weights: np.ndarray, grad: np.ndarray, lr: float = LEARN_RATE) -> None:
        g = np.clip(grad, -CLIPVAL, CLIPVAL)
        self.cache *= DECAY_RATE
        self.cache += (1.0 - DECAY_RATE) * g * g
        weights -= lr * g / np.sqrt(np.maximum(self.cache, SMOOTH_EPS)) + REGC * weights
        np.clip(weights, -CLIPVAL * 10.0, CLIPVAL * 10.0, out=weights)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def total_error(
    g: Genome, weights: np.ndarray, X: np.ndarray, y: np.ndarray, settle: bool = True
) -> float:
    """Mean binary cross-entropy over a whole split.

    Computed from logits as ``softplus(z) - y*z``, which is smooth and exact in
    the saturated regime where clipping a probability would flatten the loss and
    silently break the correspondence with the analytic gradient.
    """
    tape = forward(g, X, weights, settle=settle)
    z = tape.vals[tape.out_var]
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def accuracy(
    g: Genome, weights: np.ndarray, X: np.ndarray, y: np.ndarray, settle: bool = True
) -> float:
    tape = forward(g, X, weights, settle=settle)
    return float(np.mean((tape.vals[tape.out_var] > 0.0) == (y > 0.5)))


@dataclass
class LearnResult:
    weights: np.ndarray
    error: float
    gradient_steps: int


def train(
    g: Genome,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_cycles: int = 600,
    batch_size: int = BATCH_SIZE,
    settle: bool = True,
) -> LearnResult:
    """Train one candidate under the fixed inner budget, with source rollback.

    Rollback is checked every 20 updates against the full training split; a
    candidate that has got worse is reverted, which is why realized gradient
    steps are usually well below the nominal budget and must be reported.
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
        d_out = (_sigmoid(tape.vals[tape.out_var]) - yb) / batch_size
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


def penalty_factor(g: Genome) -> float:
    """``1 + 0.03*sqrt(connections)`` — the reference complexity penalty."""
    return (
        1.0
        + PENALTY_NODE_FACTOR * np.sqrt(max(g.n_nodes - 3, 0))
        + PENALTY_CONNECTION_FACTOR * np.sqrt(g.n_connections)
    )


def fitness_from_error(g: Genome, error: float, use_penalty: bool = True) -> float:
    """Higher is better. Reference fitness is ``-error * penaltyFactor``."""
    return -error * (penalty_factor(g) if use_penalty else 1.0)
