"""Control architectures over graphs of any width, as ordinary genomes.

Every baseline is built as an :class:`~bpneat.nd.encoding.NdGenome` and trained
by the same inner learner as an evolved candidate, so a comparison isolates the
mechanism under test rather than the learning code. These are the frozen
baselines with the two inputs and the one output unfixed; read alongside
:mod:`bpneat.baselines` and :mod:`bpneat.v3.learners`.

**The bias carrier.** The frozen module routes each output's bias through a
carrier node rather than wiring bias straight to the output, because under Ha's
exact propagation rule a direct ``bias -> output`` edge makes the output touched
on tick 0 alongside the hidden units, the loop stops, and the output is returned
before a single hidden node has run — a silently dead network. Every protocol
that will use these runs under settled propagation, where that cannot happen;
the carrier is kept anyway so the baselines are correct under both modes, as the
frozen ones are. With ``k`` outputs each output gets its **own** carrier, so each
has an independent learnable bias rather than a shared one scaled per output.

**The linear baseline is an arm, not a footnote.** ``make_linear`` is the seed
genome — bias and every input wired straight to the outputs, no hidden units at
all. On the real tabular datasets in this repository it comes within two to five
points of everything else, which makes "does architecture search beat having no
architecture" the sharpest question available, and it cannot be asked without
measuring it under the same budget as everything else.
"""

from __future__ import annotations

import numpy as np

from ..genome import OP_ADD, OP_NULL, OP_TANH
from ..learn import CLIPVAL, DECAY_RATE, LEARN_RATE, REGC, SMOOTH_EPS
from . import dense
from .datasets import TabularBundle
from .encoding import Layout, NdGenome, backward, forward, logistic_genome
from .learn import BATCH_SIZE, _d_out, _logits, total_error

__all__ = [
    "RANDOM_EXTRA_CONNECTIONS",
    "RANDOM_HIDDEN_NODES",
    "make_linear",
    "make_mixed_mlp",
    "make_mlp",
    "make_random_architecture",
    "matched_multistart",
    "plain_train",
]

#: The random architecture sampler's range, transcribed from v3's
#: ``_random_search`` so that a null here is the null the earlier protocols
#: ran: 2-15 hidden nodes, and 0-11 connections beyond the ones the
#: constructor already lays down.
RANDOM_HIDDEN_NODES = (2, 16)
RANDOM_EXTRA_CONNECTIONS = (0, 12)


def _connect(g: NdGenome, src: int, dst: int, w: float) -> None:
    g.src.append(src)
    g.dst.append(dst)
    g.weight.append(w)
    g.active.append(True)
    g.innovation.append(len(g.innovation))


def _bias_carrier(g: NdGenome) -> int:
    """A learnable bias that does not touch its output node early."""
    g.ops.append(OP_ADD)
    node = g.n_nodes - 1
    _connect(g, g.layout.bias, node, 0.0)
    return node


def make_linear(rng: np.random.Generator, layout: Layout) -> NdGenome:
    """No architecture at all: the seed genome, as an arm in its own right."""
    return logistic_genome(rng, layout)


def make_mlp(
    hidden: tuple[int, ...],
    rng: np.random.Generator,
    layout: Layout,
    op: int = OP_TANH,
) -> NdGenome:
    """A fixed, task-agnostic feed-forward network with He-style init."""
    g = NdGenome(layout=layout, ops=[OP_NULL] * layout.n_structural)
    prev = list(layout.inputs)

    for width in hidden:
        ids = []
        for _ in range(width):
            g.ops.append(op)
            ids.append(g.n_nodes - 1)
        scale = np.sqrt(2.0 / max(len(prev), 1))
        for node in ids:
            _connect(g, g.layout.bias, node, 0.0)
            for p in prev:
                _connect(g, p, node, float(rng.normal(0.0, scale)))
        prev = ids

    scale = np.sqrt(2.0 / max(len(prev), 1))
    for out in layout.outputs:
        carrier = _bias_carrier(g)
        for p in prev:
            _connect(g, p, out, float(rng.normal(0.0, scale)))
        _connect(g, carrier, out, 1.0)
    return g


def make_mixed_mlp(
    hidden: tuple[int, ...], rng: np.random.Generator, layout: Layout
) -> NdGenome:
    """The fixed MLP whose hidden operators come from the reference set.

    ``mult`` is excluded, as in v3: it aggregates by product, which the dense
    evaluator cannot express as one matrix product, and a fixed architecture
    trained to a matched gradient budget is exactly where the dense path has to
    be available or the study is unaffordable.
    """
    g = make_mlp(hidden, rng, layout)
    first_hidden = layout.n_structural
    for i in range(first_hidden, first_hidden + sum(hidden)):
        g.ops[i] = int(rng.choice(dense.DENSE_OPS))
    return g


def make_random_architecture(
    rng: np.random.Generator,
    n_hidden: int,
    n_extra_connections: int,
    activations: tuple[int, ...],
    layout: Layout,
) -> NdGenome:
    """A graph of comparable size to an evolved one, sampled rather than selected.

    The control that removes evolutionary selection while keeping graph
    irregularity, heterogeneous operators and the same inner learner. Each
    hidden node is wired from something already reachable and into **every**
    output, so the reachability guarantee the frozen version gives for one
    output holds for all of them — the same shape an MLP's last layer has.
    """
    g = NdGenome(layout=layout, ops=[OP_NULL] * layout.n_structural)
    for _ in range(n_hidden):
        g.ops.append(int(rng.choice(activations)))

    hidden = list(range(layout.n_structural, g.n_nodes))
    sources = list(layout.sources)
    for h in hidden:
        upstream = sources + [x for x in hidden if x < h]
        src = int(rng.choice(upstream or sources))
        _connect(g, src, h, float(rng.normal(0.0, 1.0)))
        for out in layout.outputs:
            _connect(g, h, out, float(rng.normal(0.0, 0.5)))
    for out in layout.outputs:
        _connect(g, _bias_carrier(g), out, 1.0)

    seen = {(g.src[i], g.dst[i]) for i in range(g.n_connections)}
    targets = list(layout.outputs) + hidden
    forbidden = set(layout.sources)
    for _ in range(n_extra_connections):
        for _ in range(20):
            src = int(rng.integers(0, g.n_nodes))
            dst = int(rng.choice(targets))
            if src == dst or (src, dst) in seen or dst in forbidden:
                continue
            _connect(g, src, dst, float(rng.normal(0.0, 0.5)))
            seen.add((src, dst))
            break
    return g


def plain_train(
    g: NdGenome,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_steps: int,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARN_RATE,
    decay: float = DECAY_RATE,
    use_dense: bool = True,
) -> np.ndarray:
    """RMSProp for exactly ``n_steps`` updates. No rollback, no early break.

    v3's learner, which removes Ha's stopping rule and nothing else, so that a
    fixed architecture given a matched budget actually spends it. The dense
    evaluator is used when the graph allows it and the genome path otherwise;
    the two agree to 1e-10 and a gate says so.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    cache = np.zeros(len(w))
    n = len(X)

    plan = None
    if use_dense:
        try:
            plan = dense.plan(g)
        except dense.NotDenseable:
            plan = None

    outputs = list(g.layout.outputs)
    for _ in range(n_steps):
        idx = rng.integers(0, n, batch_size)
        xb, yb = X[idx], y[idx]
        if plan is not None:
            tape = dense.forward(plan, w, xb)
            z = tape.V[:, outputs]
            if g.layout.n_outputs == 1:
                z = z[:, 0]
            grad = dense.backward(plan, w, tape, _d_out(g, z, yb, batch_size))
        else:
            tape = forward(g, xb, w, settle=True)
            z = _logits(tape, g.layout.n_outputs)
            grad = backward(tape, _d_out(g, z, yb, batch_size), w)
        grad = np.clip(grad, -CLIPVAL, CLIPVAL)
        cache *= decay
        cache += (1.0 - decay) * grad * grad
        w -= lr * grad / np.sqrt(np.maximum(cache, SMOOTH_EPS)) + REGC * w
        np.clip(w, -CLIPVAL * 10.0, CLIPVAL * 10.0, out=w)
    return w


def matched_multistart(
    build,
    bundle: TabularBundle,
    total_steps: int,
    restarts: int,
    seed: int,
    batch_size: int = BATCH_SIZE,
    decay: float = DECAY_RATE,
) -> tuple[NdGenome, np.ndarray, int, list[dict]]:
    """Spend ``total_steps`` gradient updates across ``restarts`` fixed nets.

    Selection is on validation loss, as every matched control in this
    repository does it. The sealed test is never read.
    """
    rng = np.random.default_rng(seed)
    per = max(1, total_steps // max(restarts, 1))
    best: tuple[float, NdGenome, np.ndarray] | None = None
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
