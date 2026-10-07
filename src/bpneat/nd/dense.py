"""v3's vectorized evaluator, over graphs of any width.

Read alongside :mod:`bpneat.v3.dense`, which this generalises and whose reasoning
applies unchanged. It groups nodes by topological depth and evaluates each group
with one matrix product, keeping the clamping and the operator semantics of the
settled genome path. It is an optimisation, never a different model, and the
test suite asserts it against the genome path on both counts.

Without it v8 is not affordable. A fixed 32x32 network trained to a matched
gradient budget is several hundred thousand updates, and the genome path walks
one node at a time in Python; on the real datasets that is hours per run rather
than minutes.

Three places knew how many inputs a graph has — where the topological sort is
seeded, where the input columns are filled, and where the output gradient is
injected — and those are the only lines that differ. Supported: acyclic genomes
whose nodes aggregate by summation. The ``mult`` operator aggregates by product,
is not expressible as one matrix product, and falls back to the genome path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..genome import (
    NODE_CLAMP,
    OP_ABS,
    OP_ADD,
    OP_GAUSSIAN,
    OP_MULT,
    OP_RELU,
    OP_SIGMOID,
    OP_SIN,
    OP_SQUARE,
    OP_TANH,
)
from .encoding import Layout, NdGenome

# Operators this evaluator can express. ``mult`` is excluded by construction.
DENSE_OPS = (
    OP_SIGMOID,
    OP_TANH,
    OP_RELU,
    OP_GAUSSIAN,
    OP_SIN,
    OP_ABS,
    OP_SQUARE,
    OP_ADD,
)

_UNARY = {OP_SIGMOID, OP_TANH, OP_RELU, OP_GAUSSIAN, OP_SIN, OP_ABS}


class NotDenseable(ValueError):
    """The genome cannot be evaluated by this module (cycle, or a mult node)."""


@dataclass
class DensePlan:
    """Topological groups plus the weight matrix that drives them."""

    n_nodes: int
    layout: Layout
    # One entry per topological depth: the node ids settled at that depth, and
    # the (op, column-slice) runs within them. One matrix product per depth, so
    # a layer of mixed operators costs the same as a uniform one.
    groups: list[tuple[np.ndarray, list[tuple[int, np.ndarray]]]]
    conn_src: np.ndarray
    conn_dst: np.ndarray
    conn_active: np.ndarray
    n_conn: int


def plan(g: NdGenome) -> DensePlan:
    """Group nodes by topological depth, or raise :class:`NotDenseable`."""
    n = g.n_nodes
    src = np.array(g.src, dtype=np.int64)
    dst = np.array(g.dst, dtype=np.int64)
    active = np.array(g.active, dtype=bool)

    for i in range(n):
        if g.ops[i] == OP_MULT:
            raise NotDenseable(f"node {i} uses the mult operator")

    # The genome path recomputes every touched node in *node-id* order within a
    # tick, reading values in place. So it reaches the topological fixed point
    # this module computes only when the ids are themselves a topological order
    # — with one documented exception, the outputs, which hold the lowest ids
    # after the inputs and are therefore always recomputed from the previous
    # tick's hidden values. ``settle_ticks`` adds exactly one tick for that.
    #
    # An edge from a higher id to a lower *hidden* id is a second inversion that
    # nothing compensates for, and the two paths then legitimately disagree —
    # measured at up to 0.34 in a logit on sampled graphs with back-edges. Those
    # graphs fall back to the genome path rather than being evaluated by a
    # method that answers a different question. The fixed architectures this
    # module exists for are built in layer order and are unaffected.
    outputs = set(g.layout.outputs)
    for ci in range(len(src)):
        if not active[ci]:
            continue
        s, d = int(src[ci]), int(dst[ci])
        if s > d and d not in outputs:
            raise NotDenseable(
                f"connection {s} -> {d} runs backwards into a hidden node, so "
                "node-id order is not a topological order and the genome path "
                "does not settle to this module's fixed point"
            )

    incoming: list[list[int]] = [[] for _ in range(n)]
    for ci in range(len(src)):
        if active[ci]:
            incoming[dst[ci]].append(int(src[ci]))

    depth = {g.layout.bias: 0}
    for i in g.layout.inputs:
        depth[i] = 0
    remaining = {i for i in range(n) if i not in depth}
    groups: list[tuple[np.ndarray, int]] = []
    while remaining:
        ready = [
            i
            for i in remaining
            if incoming[i] and all(s in depth for s in incoming[i])
        ]
        # A node with no live inputs keeps its default of zero forever; it can
        # be settled at any depth, so retire it here rather than deadlocking.
        inert = [i for i in remaining if not incoming[i]]
        if not ready and not inert:
            raise NotDenseable("graph contains a cycle")
        for i in inert:
            depth[i] = 0
            remaining.discard(i)
        if not ready:
            continue
        d = max(depth[s] for i in ready for s in incoming[i]) + 1
        for i in ready:
            depth[i] = d
            remaining.discard(i)
        # Order the level by operator so each operator is a contiguous slice of
        # one matrix product, rather than a matrix product of its own.
        ordered = sorted(ready, key=lambda i: (g.ops[i], i))
        ids = np.array(ordered, dtype=np.int64)
        runs: list[tuple[int, np.ndarray]] = []
        start = 0
        for pos in range(1, len(ordered) + 1):
            if pos == len(ordered) or g.ops[ordered[pos]] != g.ops[ordered[start]]:
                runs.append((g.ops[ordered[start]], np.arange(start, pos)))
                start = pos
        groups.append((ids, runs))

    return DensePlan(
        n_nodes=n,
        layout=g.layout,
        groups=groups,
        conn_src=src,
        conn_dst=dst,
        conn_active=active,
        n_conn=len(src),
    )


def _weight_matrix(p: DensePlan, weights: np.ndarray) -> np.ndarray:
    M = np.zeros((p.n_nodes, p.n_nodes))
    live = p.conn_active
    np.add.at(M, (p.conn_src[live], p.conn_dst[live]), weights[live])
    return M


def _apply(op: int, x: np.ndarray) -> np.ndarray:
    if op == OP_SIGMOID:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
    if op == OP_TANH:
        return np.tanh(x)
    if op == OP_RELU:
        return np.maximum(x, 0.0)
    if op == OP_GAUSSIAN:
        return np.exp(-np.clip(x * x, 0.0, 60.0))
    if op == OP_SIN:
        return np.sin(x)
    if op == OP_ABS:
        return np.abs(x)
    if op == OP_SQUARE:
        return x * x
    return x  # add / null: the weighted sum itself


def _grad(op: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if op == OP_SIGMOID:
        return y * (1.0 - y)
    if op == OP_TANH:
        return 1.0 - y * y
    if op == OP_RELU:
        return (x > 0.0).astype(np.float64)
    if op == OP_GAUSSIAN:
        return -2.0 * x * y
    if op == OP_SIN:
        return np.cos(x)
    if op == OP_ABS:
        return np.sign(x)
    if op == OP_SQUARE:
        return 2.0 * x
    return np.ones_like(x)


@dataclass
class DenseTape:
    V: np.ndarray  # (B, n) settled node values, post-clamp
    pre: list[np.ndarray]  # per group: the weighted sum before the operator
    post: list[np.ndarray]  # per group: after the operator, before the clamp
    M: np.ndarray


def forward(p: DensePlan, weights: np.ndarray, X: np.ndarray) -> DenseTape:
    """Logits in ``tape.V[:, p.layout.outputs]``, matching the settled genome path."""
    B = X.shape[0]
    if X.shape[1] != p.layout.n_inputs:
        raise ValueError(
            f"plan takes {p.layout.n_inputs} inputs, X has {X.shape[1]} columns"
        )
    V = np.zeros((B, p.n_nodes))
    V[:, p.layout.bias] = 1.0
    V[:, list(p.layout.inputs)] = X
    M = _weight_matrix(p, weights)

    pre_all, post_all = [], []
    for ids, runs in p.groups:
        pre = V @ M[:, ids]
        post = np.empty_like(pre)
        for op, cols in runs:
            post[:, cols] = _apply(op, pre[:, cols])
        V[:, ids] = np.clip(post, -NODE_CLAMP, NODE_CLAMP)
        pre_all.append(pre)
        post_all.append(post)
    return DenseTape(V=V, pre=pre_all, post=post_all, M=M)


def backward(
    p: DensePlan, weights: np.ndarray, tape: DenseTape, d_out: np.ndarray
) -> np.ndarray:
    """Gradient w.r.t. the connection-weight vector, in genome order.

    ``d_out`` is ``(B,)`` for one output or ``(B, k)`` for several, the same two
    spellings the genome path takes.
    """
    B = tape.V.shape[0]
    dV = np.zeros((B, p.n_nodes))
    seeds = np.asarray(d_out)
    if seeds.ndim == 1:
        seeds = seeds.reshape(-1, 1)
    if seeds.shape[1] != p.layout.n_outputs:
        raise ValueError(
            f"d_out has {seeds.shape[1]} columns, the layout has "
            f"{p.layout.n_outputs} outputs"
        )
    dV[:, list(p.layout.outputs)] += seeds
    dM = np.zeros((p.n_nodes, p.n_nodes))

    for gi in range(len(p.groups) - 1, -1, -1):
        ids, runs = p.groups[gi]
        pre, post = tape.pre[gi], tape.post[gi]
        g_node = dV[:, ids].copy()
        dV[:, ids] = 0.0
        # The clamp is the identity inside the band and flat outside it.
        g_node = g_node * (np.abs(post) <= NODE_CLAMP)
        g_pre = np.empty_like(g_node)
        for op, cols in runs:
            g_pre[:, cols] = g_node[:, cols] * _grad(op, pre[:, cols], post[:, cols])

        # pre = V_before @ M[:, ids]. Every node with a live edge into this
        # group sits at a strictly smaller depth, was settled in an earlier
        # group, and is never rewritten — so tape.V equals V_before on exactly
        # the rows that matter. Rows for later-settled nodes pick up values
        # here, but no active edge runs from a later group back to this one
        # (that would be a cycle), so those entries are never read back.
        dM[:, ids] += tape.V.T @ g_pre
        dV += g_pre @ tape.M[:, ids].T

    live = p.conn_active
    grad = np.zeros(p.n_conn)
    grad[live] = dM[p.conn_src[live], p.conn_dst[live]]
    return grad
