"""Can the inner learner actually fit a spiral given a known-good architecture?

If a 2x32 tanh MLP cannot get well above chance on ha2016 spirals under this
learner, then a near-chance evolutionary result says nothing about evolution.
This isolates the learner from the search.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat import learn  # noqa: E402
from bpneat.baselines import make_mlp  # noqa: E402
from bpneat.datasets import make_bundle  # noqa: E402
from bpneat.genome import backward, forward  # noqa: E402
from bpneat.learn import accuracy, total_error  # noqa: E402


def acc(g, w, X, y, settle):
    tape = forward(g, X, w, settle=settle)
    return float(np.mean((tape.vals[tape.out_var] > 0.0) == (y > 0.5)))


def loss(g, w, X, y, settle):
    tape = forward(g, X, w, settle=settle)
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(tape.vals[tape.out_var], -60, 60))), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(bundle, g, lr, decay, n_steps=600, batch=10, seed=0, settle=False):
    """Train with explicit RMSProp settings and no rollback, to see raw learning."""
    rng = np.random.default_rng(seed)
    w = np.array(g.weight, dtype=np.float64)
    cache = np.zeros(len(w))
    X, y = bundle.train.X, bundle.train.y
    curve = []
    for j in range(n_steps):
        idx = rng.integers(0, len(X), batch)
        tape = forward(g, X[idx], w, settle=settle)
        p = 1.0 / (1.0 + np.exp(-np.clip(tape.vals[tape.out_var], -60, 60)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        grad = backward(tape, (p - y[idx]) / batch, w)
        grad = np.clip(grad, -5.0, 5.0)
        cache *= decay
        cache += (1 - decay) * grad * grad
        w -= lr * grad / np.sqrt(np.maximum(cache, 1e-8)) + 0.001 * w
        np.clip(w, -50, 50, out=w)
    return w, curve


def main() -> None:
    bundle = make_bundle("spiral", seed=7103)
    rng = np.random.default_rng(1)
    g = make_mlp((32, 32), rng)
    print(f"MLP 32x32: {g.n_nodes} nodes, {g.n_connections} connections")
    print(f"initial train acc: {accuracy(g, np.array(g.weight), bundle.train.X, bundle.train.y):.3f}\n")

    print("Effective first-step size for RMSProp is ~lr/sqrt(1-decay):")
    for decay in (0.999, 0.99, 0.9):
        print(f"  decay={decay:<6} -> {0.01/np.sqrt(1-decay):.3f} per weight per step")

    w0 = np.array(g.weight, dtype=np.float64)
    ref_out = forward(g, bundle.train.X[:4], w0, settle=False)
    set_out = forward(g, bundle.train.X[:4], w0, settle=True)
    print("\nUntrained output on 4 points:")
    print(f"  reference propagation : {np.round(ref_out.vals[ref_out.out_var], 4)}")
    print(f"  settled propagation   : {np.round(set_out.vals[set_out.out_var], 4)}")

    print(f"\n{'settle':>7} {'lr':>7} {'decay':>7} {'steps':>7} {'train_loss':>11} {'train_acc':>10} {'val_acc':>9}")
    for settle, lr, decay, steps in (
        (False, 0.01, 0.999, 600),
        (True, 0.01, 0.999, 600),
        (True, 0.01, 0.99, 600),
        (True, 0.01, 0.9, 600),
        (True, 0.003, 0.99, 2000),
    ):
        w, _ = run(bundle, g, lr, decay, n_steps=steps, seed=2, settle=settle)
        print(
            f"{str(settle):>7} {lr:>7} {decay:>7} {steps:>7} "
            f"{loss(g, w, bundle.train.X, bundle.train.y, settle):11.4f} "
            f"{acc(g, w, bundle.train.X, bundle.train.y, settle):10.3f} "
            f"{acc(g, w, bundle.validation.X, bundle.validation.y, settle):9.3f}"
        )


if __name__ == "__main__":
    main()
