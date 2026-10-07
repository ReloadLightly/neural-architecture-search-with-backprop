"""The frozen encoding with ``d`` inputs and ``k`` outputs.

Read alongside :mod:`bpneat.genome`, which this generalises. Everything that
decides *what a graph computes* — the operator set, the aggregation rule, the
asynchronous propagation scheme, the settling bound, the node clamp, the unary
maps and their derivatives — is imported from the frozen module rather than
restated, so the two cannot drift apart in substance. What changes here is only
the structural layout:

* frozen: node 0 is bias, 1 and 2 are the two inputs, 3 is the single output,
  and ``N_STRUCTURAL`` is 4;
* here: node 0 is bias, ``1 .. d`` are the inputs, ``d+1 .. d+k`` are the
  outputs, and ``n_structural`` is ``1 + d + k``.

At ``d = 2, k = 1`` that layout *is* the frozen layout, node for node, which is
the whole design: the general path and the frozen path then execute the same
operations in the same order on the same values, so they agree bitwise rather
than approximately. Every loop below is written to keep that true — the tape is
pushed in the same order, the touch set is seeded in the same order, and the
tick loop breaks on the same condition.

One property of the reference scheme survives generalisation and is worth
naming, because it is not obvious: outputs keep the lowest node ids after the
inputs, so a tick recomputes every output *before* every hidden node, and an
output therefore reads its operands from the previous tick. The frozen module's
docstring says this of node 3; it is true here of all ``k`` of them, and it is
why ``settle_ticks`` adds one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..genome import _UNARY as UNARY
from ..genome import (
    ACTIVATIONS,
    MAX_TICK,
    NODE_CLAMP,
    OP_MULT,
    OP_NAMES,
    OP_NULL,
    OP_SQUARE,
    SETTLE_MAX_TICK,
    _apply_unary,
    _unary_grad,
)

__all__ = [
    "ACTIVATIONS",
    "Layout",
    "NdGenome",
    "Trace",
    "backward",
    "causal_subgraph",
    "forward",
    "logistic_genome",
    "predict_logits",
    "settle_ticks",
]

# Tape entry kinds. Same values as the frozen module's, and a gate asserts it:
# the two tapes are compared entry by entry in the equivalence test, so a
# renumbering here would make a real difference look like a match.
_MUL_W = 0  # out = operand * W[conn]
_SUM = 1  # out = sum(inputs)
_PROD = 2  # out = prod(inputs)
_UNARY_OP = 3  # out = f(input)
_SQUARE = 4  # out = input * input
_CLAMP = 5  # out = clip(input, -NODE_CLAMP, NODE_CLAMP)


@dataclass(frozen=True)
class Layout:
    """Which node id is what, for ``d`` inputs and ``k`` outputs."""

    n_inputs: int
    n_outputs: int = 1

    def __post_init__(self) -> None:
        if self.n_inputs < 1:
            raise ValueError(f"need at least one input, got {self.n_inputs}")
        if self.n_outputs < 1:
            raise ValueError(f"need at least one output, got {self.n_outputs}")

    @property
    def bias(self) -> int:
        return 0

    @property
    def inputs(self) -> tuple[int, ...]:
        return tuple(range(1, 1 + self.n_inputs))

    @property
    def outputs(self) -> tuple[int, ...]:
        return tuple(range(1 + self.n_inputs, 1 + self.n_inputs + self.n_outputs))

    @property
    def n_structural(self) -> int:
        return 1 + self.n_inputs + self.n_outputs

    @property
    def sources(self) -> tuple[int, ...]:
        """Nodes a new connection may start from: bias and the inputs."""
        return (self.bias, *self.inputs)

    def is_hidden(self, node: int) -> bool:
        return node >= self.n_structural


#: The layout every earlier protocol ran under. ``Layout(2, 1).n_structural``
#: is 4 and its node ids are 0, 1, 2, 3 — the frozen module's, exactly.
FROZEN_LAYOUT = Layout(2, 1)


@dataclass
class NdGenome:
    """A computation graph plus its learned connection weights.

    The connection arrays are the frozen module's, field for field. The one
    addition is ``layout``, which says how to read the first ``n_structural``
    node ids; it is structure, not state, and is never mutated.
    """

    layout: Layout = FROZEN_LAYOUT
    ops: list[int] = field(default_factory=list)
    src: list[int] = field(default_factory=list)
    dst: list[int] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    active: list[bool] = field(default_factory=list)
    innovation: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ops:
            self.ops = [OP_NULL] * self.layout.n_structural

    @property
    def n_nodes(self) -> int:
        return len(self.ops)

    @property
    def n_connections(self) -> int:
        return len(self.src)

    @property
    def n_enabled(self) -> int:
        return int(sum(self.active))

    def copy(self) -> NdGenome:
        return NdGenome(
            layout=self.layout,
            ops=list(self.ops),
            src=list(self.src),
            dst=list(self.dst),
            weight=list(self.weight),
            active=list(self.active),
            innovation=list(self.innovation),
        )

    def incoming(self) -> list[list[tuple[int, int]]]:
        """Per node, the list of ``(connection index, source node)`` operands."""
        table: list[list[tuple[int, int]]] = [[] for _ in range(self.n_nodes)]
        for ci in range(self.n_connections):
            if self.active[ci]:
                table[self.dst[ci]].append((ci, self.src[ci]))
        return table

    def has_connection(self, a: int, b: int) -> bool:
        return any(
            self.src[i] == a and self.dst[i] == b for i in range(self.n_connections)
        )


def logistic_genome(rng: np.random.Generator, layout: Layout = FROZEN_LAYOUT) -> NdGenome:
    """``initConfig: "all"`` — bias and every input wired to every output.

    The draw order is bias-then-inputs, output by output. At ``Layout(2, 1)``
    that is three draws in the frozen module's order, from the same generator,
    so the weights come out identical.
    """
    g = NdGenome(layout=layout)
    innovation = 0
    for out in layout.outputs:
        for node in layout.sources:
            g.src.append(node)
            g.dst.append(out)
            g.weight.append(float(rng.normal(0.0, 1.0)))
            g.active.append(True)
            g.innovation.append(innovation)
            innovation += 1
    return g


class Trace:
    """The operations that actually executed, in execution order."""

    __slots__ = ("vals", "entries", "out_vars", "n_conn")

    def __init__(self, n_conn: int) -> None:
        self.vals: list[np.ndarray] = []
        self.entries: list[tuple] = []
        self.out_vars: list[int] = []
        self.n_conn = n_conn

    @property
    def out_var(self) -> int:
        """The single output's variable, for the one-output case."""
        if len(self.out_vars) != 1:
            raise ValueError(f"{len(self.out_vars)} outputs; use out_vars")
        return self.out_vars[0]

    def push(self, value: np.ndarray) -> int:
        self.vals.append(value)
        return len(self.vals) - 1


def settle_ticks(g: NdGenome) -> int:
    """How many ticks settling needs, derived from topology alone.

    The frozen reasoning, unchanged: the count must not depend on the weights,
    or an epsilon change could add or drop a tick, the traced function would be
    discontinuous, and the analytic gradient would stop matching finite
    differences. BFS gives the round at which each node is first touched, and
    the outputs need one more tick than their deepest ancestor because they are
    recomputed before every hidden node in each tick.
    """
    n = g.n_nodes
    incoming = g.incoming()
    touched = bytearray(n)
    touched[g.layout.bias] = 1
    for i in g.layout.inputs:
        touched[i] = 1
    rounds = 0
    for r in range(1, MAX_TICK + 1):
        newly = [
            i
            for i in range(n)
            if not touched[i] and any(touched[s] for _, s in incoming[i])
        ]
        if not newly:
            break
        for i in newly:
            touched[i] = 1
        rounds = r
    return max(1, min(rounds + 1, SETTLE_MAX_TICK))


def forward(
    g: NdGenome, X: np.ndarray, weights: np.ndarray, settle: bool = False
) -> Trace:
    """Run asynchronous propagation and record a trace.

    Line for line the frozen ``forward``, with the three seeded variables
    replaced by ``1 + d`` of them and the single recorded output by ``k``. The
    tick loop, the staging of touches, the node-id ordering, the aggregation,
    the clamp and both stopping rules are untouched.
    """
    layout = g.layout
    if X.shape[1] != layout.n_inputs:
        raise ValueError(
            f"genome takes {layout.n_inputs} inputs, X has {X.shape[1]} columns"
        )
    n = g.n_nodes
    B = X.shape[0]
    tape = Trace(g.n_connections)
    incoming = g.incoming()

    zero = tape.push(np.zeros(B))
    node_var = [zero] * n
    node_var[layout.bias] = tape.push(np.ones(B))
    for col, node in enumerate(layout.inputs):
        node_var[node] = tape.push(np.ascontiguousarray(X[:, col]))

    touched = bytearray(n)
    touched[layout.bias] = 1
    for i in layout.inputs:
        touched[i] = 1

    n_ticks = settle_ticks(g) if settle else MAX_TICK
    for _ in range(n_ticks):
        # forwardTouch: staged so a fresh touch does not cascade this pass.
        newly = [
            i
            for i in range(n)
            if not touched[i] and any(touched[s] for _, s in incoming[i])
        ]
        for i in newly:
            touched[i] = 1

        # forwardTick: recompute every touched node in node-id order, in place.
        for i in range(n):
            if not touched[i]:
                continue
            operands = incoming[i]
            if not operands:
                continue
            op = g.ops[i]

            terms = []
            for ci, s in operands:
                v = tape.push(tape.vals[node_var[s]] * weights[ci])
                tape.entries.append((_MUL_W, v, node_var[s], ci))
                terms.append(v)

            if len(terms) == 1:
                cum = terms[0]
            elif op == OP_MULT:
                acc = tape.vals[terms[0]]
                for t in terms[1:]:
                    acc = acc * tape.vals[t]
                cum = tape.push(acc)
                tape.entries.append((_PROD, cum, tuple(terms)))
            else:
                acc = tape.vals[terms[0]]
                for t in terms[1:]:
                    acc = acc + tape.vals[t]
                cum = tape.push(acc)
                tape.entries.append((_SUM, cum, tuple(terms)))

            if op in UNARY:
                cum2 = tape.push(_apply_unary(op, tape.vals[cum]))
                tape.entries.append((_UNARY_OP, cum2, cum, op))
                cum = cum2
            elif op == OP_SQUARE:
                cum2 = tape.push(tape.vals[cum] * tape.vals[cum])
                tape.entries.append((_SQUARE, cum2, cum))
                cum = cum2

            if settle:
                clamped = tape.push(np.clip(tape.vals[cum], -NODE_CLAMP, NODE_CLAMP))
                tape.entries.append((_CLAMP, clamped, cum))
                cum = clamped
            node_var[i] = cum

        if not settle:
            if all(touched) or not newly:
                break

    tape.out_vars = [node_var[o] for o in layout.outputs]
    return tape


def backward(tape: Trace, d_out: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reverse-mode gradient of the traced computation w.r.t. connection weights.

    ``d_out`` is ``(B,)`` for one output or ``(B, k)`` for several. The reverse
    sweep is the frozen one; only the seeding differs, and with ``k = 1`` the
    seed is a single assignment into an empty slot, which is what the frozen
    module does.

    Two outputs can share a variable only if they are the same node, so seeding
    accumulates rather than assigns — harmless at ``k = 1`` and correct above it.
    """
    grads: list[np.ndarray | None] = [None] * len(tape.vals)
    dW = np.zeros(tape.n_conn)

    def accum(idx: int, value: np.ndarray) -> None:
        cur = grads[idx]
        grads[idx] = value if cur is None else cur + value

    seeds = np.asarray(d_out)
    if seeds.ndim == 1:
        seeds = seeds.reshape(-1, 1)
    if seeds.shape[1] != len(tape.out_vars):
        raise ValueError(
            f"d_out has {seeds.shape[1]} columns, the trace has "
            f"{len(tape.out_vars)} outputs"
        )
    for col, var in enumerate(tape.out_vars):
        accum(var, seeds[:, col])

    for entry in reversed(tape.entries):
        kind = entry[0]
        g_out = grads[entry[1]]
        if g_out is None:
            continue

        if kind == _MUL_W:
            _, out, src, ci = entry
            dW[ci] += float(np.dot(g_out, tape.vals[src]))
            accum(src, g_out * weights[ci])
        elif kind == _SUM:
            for t in entry[2]:
                accum(t, g_out)
        elif kind == _PROD:
            terms = entry[2]
            for k, t in enumerate(terms):
                partial = g_out
                for j, other in enumerate(terms):
                    if j != k:
                        partial = partial * tape.vals[other]
                accum(t, partial)
        elif kind == _UNARY_OP:
            _, out, src, op = entry
            accum(src, g_out * _unary_grad(op, tape.vals[src], tape.vals[out]))
        elif kind == _SQUARE:
            _, out, src = entry
            accum(src, g_out * 2.0 * tape.vals[src])
        elif kind == _CLAMP:
            _, out, src = entry
            accum(src, g_out * (np.abs(tape.vals[src]) <= NODE_CLAMP))

    return dW


def predict_logits(
    g: NdGenome, X: np.ndarray, weights: np.ndarray, settle: bool = True
) -> np.ndarray:
    """``(B,)`` for a single output, ``(B, k)`` for several.

    The one-output case returns a flat vector rather than a ``(B, 1)`` column so
    that it is the frozen module's return value, not something shaped like it.
    """
    tape = forward(g, X, weights, settle=settle)
    if len(tape.out_vars) == 1:
        return tape.vals[tape.out_vars[0]]
    return np.stack([tape.vals[v] for v in tape.out_vars], axis=1)


def causal_subgraph(
    g: NdGenome, X: np.ndarray, weights: np.ndarray, settle: bool = True
) -> dict:
    """Nodes and connections that actually reached a returned output.

    Derived from the executed trace, so it reflects reference propagation
    semantics rather than the represented topology. With several outputs a node
    counts as causal if it reached *any* of them, which is the generalisation
    that keeps the one-output number unchanged.
    """
    tape = forward(g, X, weights, settle=settle)
    live_vars = set(tape.out_vars)
    live_conns: set[int] = set()

    for entry in reversed(tape.entries):
        if entry[1] not in live_vars:
            continue
        kind = entry[0]
        if kind == _MUL_W:
            live_conns.add(entry[3])
            live_vars.add(entry[2])
        elif kind in (_SUM, _PROD):
            live_vars.update(entry[2])
        else:
            live_vars.add(entry[2])  # unary, square, clamp

    nodes = {g.src[ci] for ci in live_conns} | {g.dst[ci] for ci in live_conns}
    hidden = {i for i in nodes if g.layout.is_hidden(i)}
    ops: dict[str, int] = {}
    for i in hidden:
        name = OP_NAMES[g.ops[i]]
        ops[name] = ops.get(name, 0) + 1

    return {
        "causal_nodes": len(nodes),
        "causal_hidden_nodes": len(hidden),
        "causal_connections": len(live_conns),
        "causal_operators": ops,
        "represented_nodes": g.n_nodes,
        "represented_connections": g.n_enabled,
    }
