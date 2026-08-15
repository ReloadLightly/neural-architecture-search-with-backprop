"""Control architectures, expressed as ordinary genomes.

Every baseline is built as a :class:`~bpneat.genome.Genome` and trained by the
same inner learner as an evolved candidate, so a comparison isolates the
mechanism under test rather than the learning code.

Node ids are assigned in topological order, so the reference asynchronous
propagation settles a feed-forward stack in a single tick and every hidden unit
causally reaches the output (verified in the test suite).
"""

from __future__ import annotations

import numpy as np

from .genome import (
    BIAS,
    IN_X,
    IN_Y,
    OP_NULL,
    OP_TANH,
    OUT,
    Genome,
    logistic_genome,
)


def _connect(g: Genome, src: int, dst: int, w: float) -> None:
    g.src.append(src)
    g.dst.append(dst)
    g.weight.append(w)
    g.active.append(True)
    g.innovation.append(len(g.innovation))


def make_mlp(hidden: tuple[int, ...], rng: np.random.Generator, op: int = OP_TANH) -> Genome:
    """A fixed, task-agnostic feed-forward network with He-style init."""
    g = Genome(ops=[OP_NULL] * 4)
    layers: list[list[int]] = []
    prev = [IN_X, IN_Y]

    for width in hidden:
        ids = []
        for _ in range(width):
            g.ops.append(op)
            ids.append(g.n_nodes - 1)
        scale = np.sqrt(2.0 / max(len(prev), 1))
        for node in ids:
            _connect(g, BIAS, node, 0.0)
            for p in prev:
                _connect(g, p, node, float(rng.normal(0.0, scale)))
        layers.append(ids)
        prev = ids

    scale = np.sqrt(2.0 / max(len(prev), 1))
    _connect(g, BIAS, OUT, 0.0)
    for p in prev:
        _connect(g, p, OUT, float(rng.normal(0.0, scale)))
    return g


def make_logistic(rng: np.random.Generator) -> Genome:
    return logistic_genome(rng)


def make_random_architecture(
    rng: np.random.Generator,
    n_hidden: int,
    n_extra_connections: int,
    activations: tuple[int, ...],
) -> Genome:
    """A graph of comparable size to an evolved one, but sampled rather than selected.

    This is the control for H2: it removes evolutionary selection while keeping
    graph irregularity, heterogeneous operators, and the same inner learner.
    """
    g = Genome(ops=[OP_NULL] * 4)
    for _ in range(n_hidden):
        g.ops.append(int(rng.choice(activations)))

    nodes = list(range(g.n_nodes))
    hidden = nodes[4:]

    # Guarantee every hidden node is reachable and reaches the output.
    for h in hidden:
        src = int(rng.choice([BIAS, IN_X, IN_Y] + [x for x in hidden if x < h] or [IN_X]))
        _connect(g, src, h, float(rng.normal(0.0, 1.0)))
        _connect(g, h, OUT, float(rng.normal(0.0, 0.5)))
    _connect(g, BIAS, OUT, 0.0)

    seen = {(g.src[i], g.dst[i]) for i in range(g.n_connections)}
    for _ in range(n_extra_connections):
        for _ in range(20):
            src = int(rng.integers(0, g.n_nodes))
            dst = int(rng.choice([OUT] + hidden)) if hidden else OUT
            if src == dst or (src, dst) in seen or dst in (BIAS, IN_X, IN_Y):
                continue
            _connect(g, src, dst, float(rng.normal(0.0, 0.5)))
            seen.add((src, dst))
            break
    return g
