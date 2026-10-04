"""A vectorized evaluator for acyclic genomes, exact to the genome path.

The v2 genome evaluator walks one node at a time in Python. That is fine for the
8-node graphs evolution actually produces, and ruinous for a 69-node MLP trained
for 20,000 steps (~5 minutes per run). v3 must train fixed networks to a matched
gradient budget, which is ~132k steps per run, so the slow path would cost about
33 minutes per run and make the study impossible.

This module groups nodes by topological depth and evaluates each group with one
matrix product, keeping the clamping and the operator semantics of
:func:`bpneat.genome.forward` with ``settle=True``. Equivalence to the genome
path — forward values and gradients, to 1e-10 — is asserted in the test suite;
this code is an optimisation, never a different model.

Supported: acyclic genomes whose nodes aggregate their inputs by summation. The
``mult`` operator aggregates by product and is not expressible as a single
matrix product, so a genome containing one falls back to the genome path. Every
other operator in the reference set is supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..genome import (
    BIAS,
    IN_X,
    IN_Y,
    NODE_CLAMP,
    OP_ABS,
    OP_ADD,
    OP_GAUSSIAN,
    OP_MULT,
    OP_NULL,
    OP_RELU,
    OP_SIGMOID,
    OP_SIN,
    OP_SQUARE,
    OP_TANH,
    OUT,
    Genome,
)

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
    groups: list[tuple[np.ndarray, int]]  # (node ids at this depth, shared op)
    conn_src: np.ndarray
    conn_dst: np.ndarray
    conn_active: np.ndarray
    n_conn: int


def plan(g: Genome) -> DensePlan:
    """Group nodes by topological depth, or raise :class:`NotDenseable`."""
    n = g.n_nodes
    src = np.array(g.src, dtype=np.int64)
    dst = np.array(g.dst, dtype=np.int64)
    active = np.array(g.active, dtype=bool)

    for i in range(n):
        if g.ops[i] == OP_MULT:
            raise NotDenseable(f"node {i} uses the mult operator")

    incoming: list[list[int]] = [[] for _ in range(n)]
    for ci in range(len(src)):
        if active[ci]:
            incoming[dst[ci]].append(int(src[ci]))

    depth = {BIAS: 0, IN_X: 0, IN_Y: 0}
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
        ops = {g.ops[i] for i in ready}
        # Split the depth level by operator so each sub-group is one matmul.
        for op in sorted(ops):
            ids = np.array(sorted(i for i in ready if g.ops[i] == op), dtype=np.int64)
            groups.append((ids, op))

    return DensePlan(
        n_nodes=n,
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
    """Logits in ``tape.V[:, OUT]``, matching the genome path under ``settle``."""
    B = X.shape[0]
    V = np.zeros((B, p.n_nodes))
    V[:, BIAS] = 1.0
    V[:, IN_X] = X[:, 0]
    V[:, IN_Y] = X[:, 1]
    M = _weight_matrix(p, weights)

    pre_all, post_all = [], []
    for ids, op in p.groups:
        pre = V @ M[:, ids]
        post = _apply(op, pre)
        V[:, ids] = np.clip(post, -NODE_CLAMP, NODE_CLAMP)
        pre_all.append(pre)
        post_all.append(post)
    return DenseTape(V=V, pre=pre_all, post=post_all, M=M)


def backward(
    p: DensePlan, weights: np.ndarray, tape: DenseTape, d_out: np.ndarray
) -> np.ndarray:
    """Gradient w.r.t. the connection-weight vector, in genome order."""
    B = tape.V.shape[0]
    dV = np.zeros((B, p.n_nodes))
    dV[:, OUT] += d_out
    dM = np.zeros((p.n_nodes, p.n_nodes))

    for gi in range(len(p.groups) - 1, -1, -1):
        ids, op = p.groups[gi]
        pre, post = tape.pre[gi], tape.post[gi]
        g_node = dV[:, ids].copy()
        dV[:, ids] = 0.0
        # The clamp is the identity inside the band and flat outside it.
        g_node = g_node * (np.abs(post) <= NODE_CLAMP)
        g_pre = g_node * _grad(op, pre, post)

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
