"""Evolvable computation graphs with backpropagation through the executed trace.

Faithful to ``hardmaru/backprop-neat-js`` (``ml/neat.js``):

* nodes 0..2 are bias, input-x, input-y; node 3 is the output. These four carry
  the null operator, so the output is a plain weighted sum and the sigmoid is
  applied by the loss, exactly as in ``datafit-neat.js``.
* a node aggregates its incoming ``operand * weight`` terms by summation, or by
  elementwise product when the node's operator is ``mult``; the unary operator
  is then applied unless it is null/mult/add; ``square`` sums and then squares.
* propagation is the reference asynchronous scheme: a node becomes *touched*
  when any operand is touched (staged, so a touch does not cascade within one
  pass), every touched node is recomputed in node-id order each tick reading
  in-place values, and ticking stops when all nodes are touched, when a tick
  touches nothing new, or after ``MAX_TICK`` ticks.

Because a tick can read an operand that is still at its default zero, a graph
may return an output before some represented structure has influenced it. That
is a property of the reference algorithm, not a bug, and it is why
:func:`causal_subgraph` — not the represented node count — is the architecture
measure this project reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_TICK = 100
# Settling runs the graph to a fixed point rather than to touch-coverage. A
# recurrent cycle carrying an expansive operator (square, mult) diverges to
# inf/nan within a few ticks, so settling is bounded and node values are
# clamped. Both bounds are part of the declared protocol.
SETTLE_MAX_TICK = 16
NODE_CLAMP = 100.0

# Operators. NULL covers bias/input/output nodes.
OP_NULL = 0
OP_SIGMOID = 1
OP_TANH = 2
OP_RELU = 3
OP_GAUSSIAN = 4
OP_SIN = 5
OP_ABS = 6
OP_MULT = 7
OP_SQUARE = 8
OP_ADD = 9

OP_NAMES = {
    OP_NULL: "null",
    OP_SIGMOID: "sigmoid",
    OP_TANH: "tanh",
    OP_RELU: "relu",
    OP_GAUSSIAN: "gaussian",
    OP_SIN: "sin",
    OP_ABS: "abs",
    OP_MULT: "mult",
    OP_SQUARE: "square",
    OP_ADD: "add",
}

# activations_default in ml/neat.js
ACTIVATIONS = (
    OP_SIGMOID,
    OP_TANH,
    OP_RELU,
    OP_GAUSSIAN,
    OP_SIN,
    OP_ABS,
    OP_MULT,
    OP_SQUARE,
    OP_ADD,
)

# Operators applied as a unary map after aggregation.
_UNARY = {OP_SIGMOID, OP_TANH, OP_RELU, OP_GAUSSIAN, OP_SIN, OP_ABS}

BIAS, IN_X, IN_Y, OUT = 0, 1, 2, 3
N_STRUCTURAL = 4


@dataclass
class Genome:
    """A computation graph plus its learned connection weights."""

    ops: list[int] = field(default_factory=lambda: [OP_NULL] * N_STRUCTURAL)
    # Parallel arrays over connections.
    src: list[int] = field(default_factory=list)
    dst: list[int] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    active: list[bool] = field(default_factory=list)
    innovation: list[int] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.ops)

    @property
    def n_connections(self) -> int:
        return len(self.src)

    @property
    def n_enabled(self) -> int:
        return int(sum(self.active))

    def copy(self) -> "Genome":
        return Genome(
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


def logistic_genome(rng: np.random.Generator) -> Genome:
    """``initConfig: "all"`` — bias and both inputs wired straight to the output."""
    g = Genome()
    for i, node in enumerate((BIAS, IN_X, IN_Y)):
        g.src.append(node)
        g.dst.append(OUT)
        g.weight.append(float(rng.normal(0.0, 1.0)))
        g.active.append(True)
        g.innovation.append(i)
    return g


# --------------------------------------------------------------------------
# Forward pass with an explicit tape, and reverse-mode backpropagation.
# --------------------------------------------------------------------------

# Tape entry kinds.
_MUL_W = 0  # out = operand * W[conn]
_SUM = 1  # out = sum(inputs)
_PROD = 2  # out = prod(inputs)
_UNARY_OP = 3  # out = f(input)
_SQUARE = 4  # out = input * input
_CLAMP = 5  # out = clip(input, -NODE_CLAMP, NODE_CLAMP)


class Trace:
    """The operations that actually executed, in execution order."""

    __slots__ = ("vals", "entries", "out_var", "n_conn")

    def __init__(self, n_conn: int) -> None:
        self.vals: list[np.ndarray] = []
        self.entries: list[tuple] = []
        self.out_var: int = -1
        self.n_conn = n_conn

    def push(self, value: np.ndarray) -> int:
        self.vals.append(value)
        return len(self.vals) - 1


def settle_ticks(g: Genome) -> int:
    """How many ticks settling needs, derived from topology alone.

    The count must not depend on the weights: if it did, an epsilon change
    could add or drop a tick, the traced function would be discontinuous, and
    the analytic gradient would stop matching finite differences. BFS gives the
    round at which each node is first touched; the output needs one more tick
    than its deepest ancestor because node id 3 is recomputed before every
    hidden node in each tick.
    """
    n = g.n_nodes
    incoming = g.incoming()
    touched = bytearray(n)
    touched[BIAS] = touched[IN_X] = touched[IN_Y] = 1
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
    g: Genome, X: np.ndarray, weights: np.ndarray, settle: bool = False
) -> Trace:
    """Run asynchronous propagation and record a trace.

    ``settle=False`` is Ha's exact rule: stop once every node has been touched,
    or once a tick touches nothing new. Because the output node holds id 3 and
    every tick recomputes nodes in id order, the output reads its operands from
    the *previous* tick. For a graph whose nodes all become touched on the first
    tick — anything wired directly from bias or the inputs into both the output
    and the hidden units — that rule returns an output computed before any
    hidden node ran.

    ``settle=True`` keeps the same propagation but stops on value convergence
    instead of on touch coverage, so every represented path has actually
    contributed. Evolved genomes and fixed baselines must use the *same* mode
    for a comparison between them to mean anything.
    """
    n = g.n_nodes
    B = X.shape[0]
    tape = Trace(g.n_connections)
    incoming = g.incoming()

    zero = tape.push(np.zeros(B))
    node_var = [zero] * n
    node_var[BIAS] = tape.push(np.ones(B))
    node_var[IN_X] = tape.push(np.ascontiguousarray(X[:, 0]))
    node_var[IN_Y] = tape.push(np.ascontiguousarray(X[:, 1]))

    touched = bytearray(n)
    touched[BIAS] = touched[IN_X] = touched[IN_Y] = 1

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

            if op in _UNARY:
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

    tape.out_var = node_var[OUT]
    return tape


def _apply_unary(op: int, x: np.ndarray) -> np.ndarray:
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
    raise AssertionError(f"not a unary operator: {op}")


def backward(tape: Trace, d_out: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reverse-mode gradient of the traced computation w.r.t. connection weights."""
    grads: list[np.ndarray | None] = [None] * len(tape.vals)
    grads[tape.out_var] = d_out
    dW = np.zeros(tape.n_conn)

    def accum(idx: int, value: np.ndarray) -> None:
        cur = grads[idx]
        grads[idx] = value if cur is None else cur + value

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


def _unary_grad(op: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    raise AssertionError(f"not a unary operator: {op}")


def predict_logits(
    g: Genome, X: np.ndarray, weights: np.ndarray, settle: bool = True
) -> np.ndarray:
    tape = forward(g, X, weights, settle=settle)
    return tape.vals[tape.out_var]


# --------------------------------------------------------------------------
# Causal structure
# --------------------------------------------------------------------------


def causal_subgraph(
    g: Genome, X: np.ndarray, weights: np.ndarray, settle: bool = True
) -> dict:
    """Nodes and connections that actually reached the returned output.

    Derived from the executed trace, so it reflects reference propagation
    semantics rather than the represented topology.
    """
    tape = forward(g, X, weights, settle=settle)
    live_vars = {tape.out_var}
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
    hidden = {i for i in nodes if i >= N_STRUCTURAL}
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
