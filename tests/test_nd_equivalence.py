"""At two inputs and one output, the general encoding must *be* the frozen one.

This is the gate the whole n-dimensional core rests on. Every result in this
repository was produced by `bpneat.genome`, and a generalisation of it is only
allowed to inherit that credibility if it is the same computation — not a
numerically close one. So the comparisons here are exact: identical genomes,
identical tapes entry for entry, identical float64 values bit for bit,
identical gradients, identical causal structure.

Exactness is also the only check that can catch the failure that matters. A
reordered sum or a reassociated product agrees to 1e-15 and would pass a
tolerance test, while meaning the two paths no longer execute the same
operations in the same order — at which point "the frozen results transfer" is
no longer true, only plausible.

If something here cannot be made to pass, the generalisation has failed and is
reported as failed. It is not fixed by loosening the comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat import genome as frozen
from bpneat.nd import encoding as nd

SEEDS = range(24)
#: Tape kinds are compared entry for entry, so the two modules must agree on
#: what each number means before any comparison below says anything.
KIND_NAMES = ("_MUL_W", "_SUM", "_PROD", "_UNARY_OP", "_SQUARE", "_CLAMP")


def _inputs(rng, n=40):
    return rng.normal(0.0, 1.2, size=(n, 2))


def _grow(g, rng, n_hidden: int, n_extra: int, ops):
    """Add hidden nodes and connections to a genome, structure-first.

    Written against the shared field names so that one function can build the
    *same* graph in both encodings: the frozen genome and the general one carry
    identical connection arrays, which is the point.
    """
    for _ in range(n_hidden):
        # Split a random enabled connection, the way add_node does.
        live = [i for i in range(len(g.src)) if g.active[i]]
        ci = live[int(rng.integers(len(live)))]
        g.active[ci] = False
        new = len(g.ops)
        g.ops.append(int(ops[int(rng.integers(len(ops)))]))
        for a, b, w in ((g.src[ci], new, 1.0), (new, g.dst[ci], float(g.weight[ci]))):
            g.src.append(a)
            g.dst.append(b)
            g.weight.append(w)
            g.active.append(True)
            g.innovation.append(len(g.innovation))
    for _ in range(n_extra):
        a = int(rng.integers(len(g.ops)))
        b = int(rng.integers(len(g.ops)))
        if a == b:
            continue
        g.src.append(a)
        g.dst.append(b)
        g.weight.append(float(rng.normal(0.0, 1.0)))
        g.active.append(True)
        g.innovation.append(len(g.innovation))
    return g


def _pair(seed: int, n_hidden: int = 4, n_extra: int = 3):
    """One graph, built twice: once frozen, once general, from one seed."""
    a = frozen.logistic_genome(np.random.default_rng(seed))
    b = nd.logistic_genome(np.random.default_rng(seed))
    _grow(a, np.random.default_rng(1000 + seed), n_hidden, n_extra, frozen.ACTIVATIONS)
    _grow(b, np.random.default_rng(1000 + seed), n_hidden, n_extra, frozen.ACTIVATIONS)
    return a, b


def _same_genome(a, b) -> None:
    assert a.ops == b.ops
    assert a.src == b.src
    assert a.dst == b.dst
    assert a.weight == b.weight
    assert a.active == b.active
    assert a.innovation == b.innovation
    assert a.n_nodes == b.n_nodes
    assert a.n_connections == b.n_connections
    assert a.n_enabled == b.n_enabled


# --------------------------------------------------------------------------
# The layout is the frozen layout
# --------------------------------------------------------------------------


def test_the_tape_kinds_mean_the_same_thing_in_both_modules():
    """Compared entry for entry below, so a renumbering must fail here first."""
    for name in KIND_NAMES:
        assert getattr(nd, name) == getattr(frozen, name), name


def test_two_inputs_and_one_output_is_the_frozen_node_layout():
    layout = nd.FROZEN_LAYOUT
    assert layout.bias == frozen.BIAS
    assert layout.inputs == (frozen.IN_X, frozen.IN_Y)
    assert layout.outputs == (frozen.OUT,)
    assert layout.n_structural == frozen.N_STRUCTURAL
    for node in range(frozen.N_STRUCTURAL):
        assert not layout.is_hidden(node)
    assert layout.is_hidden(frozen.N_STRUCTURAL)


def test_the_general_default_genome_is_the_frozen_default_genome():
    for seed in SEEDS:
        _same_genome(
            frozen.logistic_genome(np.random.default_rng(seed)),
            nd.logistic_genome(np.random.default_rng(seed)),
        )


def test_the_operator_set_is_the_frozen_one_not_a_copy_of_it():
    assert nd.ACTIVATIONS is frozen.ACTIVATIONS


# --------------------------------------------------------------------------
# The same computation, not a close one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("settle", [False, True])
def test_the_forward_tape_is_identical_entry_for_entry(settle):
    for seed in SEEDS:
        a, b = _pair(seed)
        rng = np.random.default_rng(5000 + seed)
        X = _inputs(rng)
        w = rng.normal(0.0, 1.0, len(a.src))

        ta = frozen.forward(a, X, w, settle=settle)
        tb = nd.forward(b, X, w, settle=settle)

        assert len(ta.vals) == len(tb.vals), f"seed {seed}: different tape length"
        assert ta.entries == tb.entries, f"seed {seed}: different tape entries"
        assert ta.out_var == tb.out_var
        assert tb.out_vars == [ta.out_var]
        for i, (va, vb) in enumerate(zip(ta.vals, tb.vals)):
            assert np.array_equal(va, vb), f"seed {seed}: tape value {i} differs"


@pytest.mark.parametrize("settle", [False, True])
def test_the_logits_are_bit_identical(settle):
    for seed in SEEDS:
        a, b = _pair(seed)
        rng = np.random.default_rng(6000 + seed)
        X = _inputs(rng)
        w = rng.normal(0.0, 1.0, len(a.src))
        pa = frozen.predict_logits(a, X, w, settle=settle)
        pb = nd.predict_logits(b, X, w, settle=settle)
        assert pa.shape == pb.shape == (len(X),)
        assert np.array_equal(pa, pb), f"seed {seed}"


@pytest.mark.parametrize("settle", [False, True])
def test_the_weight_gradients_are_bit_identical(settle):
    for seed in SEEDS:
        a, b = _pair(seed)
        rng = np.random.default_rng(7000 + seed)
        X = _inputs(rng)
        w = rng.normal(0.0, 1.0, len(a.src))
        d_out = rng.normal(0.0, 1.0, len(X))

        ga = frozen.backward(frozen.forward(a, X, w, settle=settle), d_out, w)
        gb = nd.backward(nd.forward(b, X, w, settle=settle), d_out, w)
        assert np.array_equal(ga, gb), f"seed {seed}"
        # And the (B, 1) spelling of the same thing must give the same answer.
        gc = nd.backward(
            nd.forward(b, X, w, settle=settle), d_out.reshape(-1, 1), w
        )
        assert np.array_equal(ga, gc), f"seed {seed}: column form differs"


def test_the_settling_tick_count_is_identical():
    for seed in SEEDS:
        a, b = _pair(seed, n_hidden=6, n_extra=5)
        assert frozen.settle_ticks(a) == nd.settle_ticks(b), f"seed {seed}"


@pytest.mark.parametrize("settle", [False, True])
def test_the_causal_subgraph_is_identical(settle):
    for seed in SEEDS:
        a, b = _pair(seed)
        rng = np.random.default_rng(8000 + seed)
        X = _inputs(rng)
        w = rng.normal(0.0, 1.0, len(a.src))
        assert frozen.causal_subgraph(a, X, w, settle) == nd.causal_subgraph(
            b, X, w, settle
        ), f"seed {seed}"


def test_the_equivalence_is_tested_on_graphs_that_actually_use_the_machinery():
    """A gate that only ever saw three-connection graphs would prove nothing."""
    saw_hidden = saw_mult = saw_square = saw_unary = 0
    for seed in SEEDS:
        _, b = _pair(seed)
        hidden = [b.ops[i] for i in range(b.layout.n_structural, b.n_nodes)]
        saw_hidden += len(hidden)
        saw_mult += sum(op == frozen.OP_MULT for op in hidden)
        saw_square += sum(op == frozen.OP_SQUARE for op in hidden)
        saw_unary += sum(op in frozen._UNARY for op in hidden)
    assert saw_hidden >= 4 * len(SEEDS) - len(SEEDS)
    assert saw_mult and saw_square and saw_unary, (
        f"mult={saw_mult} square={saw_square} unary={saw_unary}: the tape kinds "
        "that differ most between the two paths were never exercised"
    )


# --------------------------------------------------------------------------
# Above two inputs it has to be correct on its own terms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d,k", [(1, 1), (3, 1), (5, 1), (4, 3), (8, 2)])
def test_a_wider_graph_computes_and_differentiates(d, k):
    rng = np.random.default_rng(31 + d * 10 + k)
    layout = nd.Layout(d, k)
    g = nd.logistic_genome(rng, layout)
    _grow(g, rng, 5, 4, frozen.ACTIVATIONS)
    X = rng.normal(0.0, 1.0, size=(32, d))
    w = rng.normal(0.0, 1.0, len(g.src))

    logits = nd.predict_logits(g, X, w, settle=True)
    assert logits.shape == ((len(X),) if k == 1 else (len(X), k))
    assert np.all(np.isfinite(logits))

    d_out = rng.normal(0.0, 1.0, size=(len(X), k))
    grad = nd.backward(nd.forward(g, X, w, settle=True), d_out, w)
    assert grad.shape == (len(g.src),)
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize("d,k", [(3, 1), (4, 2)])
def test_the_analytic_gradient_matches_finite_differences(d, k):
    """The one check that does not compare against the frozen module.

    Above two inputs there is nothing to compare against, so correctness has to
    be established directly. Central differences on a settled graph, which is
    the mode every protocol evaluates in and the one whose tick count is fixed
    by topology rather than by values — without that, an epsilon step could
    change the number of ticks and the comparison would be meaningless.
    """
    rng = np.random.default_rng(97 + d * 10 + k)
    g = nd.logistic_genome(rng, nd.Layout(d, k))
    _grow(g, rng, 4, 3, frozen.ACTIVATIONS)
    X = rng.normal(0.0, 0.8, size=(16, d))
    w = rng.normal(0.0, 0.5, len(g.src))
    target = rng.normal(0.0, 1.0, size=(len(X), k))

    def loss(weights):
        out = nd.predict_logits(g, X, weights, settle=True).reshape(len(X), k)
        return float(np.sum((out - target) ** 2))

    tape = nd.forward(g, X, w, settle=True)
    out = nd.predict_logits(g, X, w, settle=True).reshape(len(X), k)
    analytic = nd.backward(tape, 2.0 * (out - target), w)

    eps = 1e-6
    for ci in range(len(w)):
        up, down = w.copy(), w.copy()
        up[ci] += eps
        down[ci] -= eps
        numeric = (loss(up) - loss(down)) / (2 * eps)
        scale = max(1.0, abs(numeric), abs(analytic[ci]))
        assert abs(numeric - analytic[ci]) / scale < 2e-5, (
            f"d={d} k={k} connection {ci}: analytic {analytic[ci]:.8f} vs "
            f"numeric {numeric:.8f}"
        )


def test_a_genome_refuses_data_of_the_wrong_width():
    g = nd.logistic_genome(np.random.default_rng(0), nd.Layout(3, 1))
    w = np.asarray(g.weight)
    with pytest.raises(ValueError, match="3 inputs"):
        nd.forward(g, np.zeros((4, 2)), w)


def test_a_layout_refuses_to_be_degenerate():
    for bad in ((0, 1), (2, 0), (-1, 1)):
        with pytest.raises(ValueError):
            nd.Layout(*bad)
