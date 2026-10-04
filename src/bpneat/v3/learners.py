"""Inner learners, and the budget matching v2 lacked.

v2 trained every candidate with Ha's rule: RMSProp with a rollback check every
20 updates against the full training split, breaking at the first increase. For
an 8-node evolved graph that is a mild regulariser. For a 69-node fixed network
it stops training after 42 steps, which is why v2's fixed-MLP control reached
only 0.60 on spirals (see ``docs/audit-2026-10.md``).

``plain_train`` removes the stopping rule and nothing else: same optimiser, same
hyperparameters, same clipping. ``matched_multistart`` then gives a fixed
architecture the *same realized gradient budget* the evolutionary condition
spent in that very replicate, so the comparison is paired on compute.

Both run on :mod:`bpneat.v3.dense` when the genome allows it, which it does for
every fixed architecture here, and fall back to the frozen genome path
otherwise. The two paths agree to 1e-10 on values and gradients (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baselines import make_mlp
from ..datasets import DatasetBundle
from ..genome import Genome, backward, forward
from ..learn import CLIPVAL, REGC, SMOOTH_EPS, _sigmoid, accuracy, total_error
from . import dense

#: Ha's RMSProp decay. Kept so that removing rollback changes exactly one thing.
HA_DECAY = 0.999
LEARN_RATE = 0.01
BATCH_SIZE = 10


@dataclass
class LearnOutcome:
    weights: np.ndarray
    gradient_steps: int
    validation_loss: float


def _loss_from_logits(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def plain_train(
    g: Genome,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_steps: int,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARN_RATE,
    decay: float = HA_DECAY,
    use_dense: bool = True,
) -> np.ndarray:
    """RMSProp for exactly ``n_steps`` updates. No rollback, no early break."""
    w = np.asarray(weights, dtype=np.float64).copy()
    cache = np.zeros(len(w))
    n = len(X)

    plan = None
    if use_dense:
        try:
            plan = dense.plan(g)
        except dense.NotDenseable:
            plan = None

    for _ in range(n_steps):
        idx = rng.integers(0, n, batch_size)
        xb, yb = X[idx], y[idx]
        if plan is not None:
            tape = dense.forward(plan, w, xb)
            z = tape.V[:, 3]
            d = (_sigmoid(z) - yb) / batch_size
            grad = dense.backward(plan, w, tape, d)
        else:
            tape = forward(g, xb, w, settle=True)
            z = tape.vals[tape.out_var]
            d = (_sigmoid(z) - yb) / batch_size
            grad = backward(tape, d, w)
        grad = np.clip(grad, -CLIPVAL, CLIPVAL)
        cache *= decay
        cache += (1.0 - decay) * grad * grad
        w -= lr * grad / np.sqrt(np.maximum(cache, SMOOTH_EPS)) + REGC * w
        np.clip(w, -CLIPVAL * 10.0, CLIPVAL * 10.0, out=w)
    return w


def mixed_mlp(hidden: tuple[int, ...], rng: np.random.Generator) -> Genome:
    """A fixed MLP whose hidden operators are drawn from the reference set.

    Built on the frozen :func:`bpneat.baselines.make_mlp` and then re-typed.
    ``mult`` is excluded: it aggregates by product, which the dense evaluator
    cannot express as one matrix product, and a fixed architecture is exactly
    where the dense path has to be available.
    """
    g = make_mlp(hidden, rng)
    first_hidden = 4
    carrier = first_hidden + sum(hidden)
    for i in range(first_hidden, carrier):
        g.ops[i] = int(rng.choice(dense.DENSE_OPS))
    return g


def matched_multistart(
    build: callable,
    bundle: DatasetBundle,
    total_steps: int,
    restarts: int,
    seed: int,
    batch_size: int = BATCH_SIZE,
    decay: float = HA_DECAY,
) -> tuple[Genome, np.ndarray, int, list[dict]]:
    """Spend ``total_steps`` gradient updates across ``restarts`` fixed nets.

    Selection is on validation loss, exactly as the v2 multistart control did;
    the only things that change are the stopping rule and the size of the
    budget. The sealed test is never read.
    """
    rng = np.random.default_rng(seed)
    per = max(1, total_steps // max(restarts, 1))
    best: tuple[float, Genome, np.ndarray] | None = None
    spent = 0
    history: list[dict] = []

    for i in range(restarts):
        g = build(rng)
        w = plain_train(
            g, np.array(g.weight, dtype=np.float64), bundle.train.X, bundle.train.y,
            rng, n_steps=per, batch_size=batch_size, decay=decay,
        )
        spent += per
        val = total_error(g, w, bundle.validation.X, bundle.validation.y, True)
        history.append({"restart": i, "steps": per, "validation_loss": float(val)})
        if best is None or val < best[0]:
            best = (val, g, w)

    assert best is not None
    return best[1], best[2], spent, history


def ha_multistart(
    build: callable,
    bundle: DatasetBundle,
    restarts: int,
    inner_steps: int,
    seed: int,
    batch_size: int = BATCH_SIZE,
) -> tuple[Genome, np.ndarray, int, list[dict]]:
    """The v2 control, reproduced exactly: Ha's learner with rollback."""
    from ..learn import train

    rng = np.random.default_rng(seed)
    best: tuple[float, Genome, np.ndarray] | None = None
    spent = 0
    history: list[dict] = []

    for i in range(restarts):
        g = build(rng)
        res = train(
            g, np.array(g.weight, dtype=np.float64), bundle.train.X, bundle.train.y,
            rng, n_cycles=inner_steps, batch_size=batch_size, settle=True,
        )
        spent += res.gradient_steps
        val = total_error(g, res.weights, bundle.validation.X, bundle.validation.y, True)
        history.append({"restart": i, "steps": res.gradient_steps, "validation_loss": float(val)})
        if best is None or val < best[0]:
            best = (val, g, res.weights)

    assert best is not None
    return best[1], best[2], spent, history


def validation_accuracy(g: Genome, w: np.ndarray, bundle: DatasetBundle) -> float:
    return accuracy(g, w, bundle.validation.X, bundle.validation.y, True)
