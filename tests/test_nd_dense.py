"""The fast evaluator and the control architectures, over any width.

Two obligations, and they are different ones.

The dense evaluator is an *optimisation*: it must agree with the genome path it
replaces, at every width, to numerical tolerance. It does not have to agree
bitwise, because it deliberately computes the same quantity by a different
association — one matrix product per topological level instead of one node at a
time — and 1e-10 is the tolerance v3 established for exactly this comparison.

The baselines are *the frozen baselines*: at two inputs and one output they must
be byte-identical, like everything else in this core, because a control that is
merely similar to the control an earlier protocol ran is a different control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat import baselines as frozen_bl
from bpneat import genome as frozen
from bpneat.nd import baselines as nd_bl
from bpneat.nd import dense as nd_dense
from bpneat.nd import encoding as nd
from bpneat.v3 import dense as v3_dense
from bpneat.v3 import learners as v3_learners

LAYOUTS = [(2, 1), (4, 1), (7, 1), (5, 3), (12, 10)]
HIDDEN = (6, 5)


def _same_genome(a, b) -> None:
    assert a.ops == b.ops
    assert a.src == b.src and a.dst == b.dst
    assert a.weight == b.weight
    assert a.active == b.active and a.innovation == b.innovation


# --------------------------------------------------------------------------
# The dense evaluator against the genome path it replaces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d,k", LAYOUTS)
def test_the_dense_path_matches_the_genome_path(d, k):
    layout = nd.Layout(d, k)
    rng = np.random.default_rng(17 + d * 7 + k)
    g = nd_bl.make_mixed_mlp(HIDDEN, rng, layout)
    X = rng.normal(0.0, 1.0, size=(23, d))
    w = rng.normal(0.0, 0.7, len(g.src))

    plan = nd_dense.plan(g)
    tape = nd_dense.forward(plan, w, X)
    fast = tape.V[:, list(layout.outputs)]
    slow = nd.predict_logits(g, X, w, settle=True).reshape(len(X), k)
    assert np.allclose(fast, slow, atol=1e-10), f"d={d} k={k}: forward"

    d_out = rng.normal(0.0, 1.0, size=(len(X), k))
    g_fast = nd_dense.backward(plan, w, tape, d_out)
    g_slow = nd.backward(nd.forward(g, X, w, settle=True), d_out, w)
    assert np.allclose(g_fast, g_slow, atol=1e-10), f"d={d} k={k}: gradient"


@pytest.mark.parametrize("d,k", [(3, 1), (4, 3)])
def test_a_forward_wired_sampled_graph_agrees_on_both_paths(d, k):
    """Irregular but id-ordered graphs — not just layered ones."""
    layout = nd.Layout(d, k)
    ops = tuple(o for o in frozen.ACTIVATIONS if o != frozen.OP_MULT)
    checked = 0
    for seed in range(8):
        rng = np.random.default_rng(400 + seed)
        g = nd_bl.make_random_architecture(rng, 7, 0, ops, layout)
        X = rng.normal(0.0, 1.0, size=(19, d))
        w = rng.normal(0.0, 0.7, len(g.src))
        plan = nd_dense.plan(g)  # no extra edges, so no backwards wiring
        fast = nd_dense.forward(plan, w, X).V[:, list(layout.outputs)]
        slow = nd.predict_logits(g, X, w, settle=True).reshape(len(X), k)
        assert np.allclose(fast, slow, atol=1e-10), f"d={d} k={k} seed={seed}"
        checked += 1
    assert checked == 8


@pytest.mark.parametrize("d,k", [(3, 1), (4, 3)])
def test_a_backwards_wired_sampled_graph_is_refused_rather_than_approximated(d, k):
    """With extra connections the sampler wires backwards, and must be declined.

    This is not a rare corner: with the sampler's own range of extra edges,
    *every* graph drawn here is refused. That is the right outcome and it costs
    nothing, because the arm these graphs belong to — the candidate-matched
    null — trains on the genome path anyway.
    """
    layout = nd.Layout(d, k)
    ops = tuple(o for o in frozen.ACTIVATIONS if o != frozen.OP_MULT)
    refused = accepted = 0
    for seed in range(12):
        rng = np.random.default_rng(400 + seed)
        g = nd_bl.make_random_architecture(rng, 7, 5, ops, layout)
        X = rng.normal(0.0, 1.0, size=(19, d))
        w = rng.normal(0.0, 0.7, len(g.src))
        try:
            plan = nd_dense.plan(g)
        except nd_dense.NotDenseable:
            refused += 1
            continue
        # Not every draw wires backwards — the extra edges can all land on
        # outputs. The invariant is not "all refused" but "anything accepted is
        # right", which is the property the fast path actually has to have.
        accepted += 1
        fast = nd_dense.forward(plan, w, X).V[:, list(layout.outputs)]
        slow = nd.predict_logits(g, X, w, settle=True).reshape(len(X), k)
        assert np.allclose(fast, slow, atol=1e-10), f"d={d} k={k} seed={seed}"
    assert refused, f"d={d} k={k}: nothing was refused, so the guard is untested"
    assert refused + accepted == 12


def test_the_two_paths_really_do_disagree_on_a_backwards_wired_graph():
    """The guard protects against a measured difference, not a worry.

    Demonstrated through v3's dense evaluator, which has no such guard, on a
    graph from the *frozen* sampler at two inputs. Across 300 seeds, 29 of the
    134 plannable graphs disagree with the genome path and the worst gap is
    0.47 in a logit. v3 never met one — it only ever planned layered fixed
    networks, and v4 only ever planned decoded CGP genomes, which are
    feed-forward by construction — so this is a latent property of the
    optimisation that the general version declines rather than inherits.
    See `docs/audit-dense-ordering.md`.
    """
    ops = tuple(o for o in frozen.ACTIVATIONS if o != frozen.OP_MULT)
    worst = 0.0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        g = frozen_bl.make_random_architecture(rng, 2 + seed % 9, 1 + seed % 7, ops)
        X = rng.normal(size=(16, 2))
        w = rng.normal(0.0, 0.7, len(g.src))
        try:
            plan = v3_dense.plan(g)
        except v3_dense.NotDenseable:
            continue
        fast = v3_dense.forward(plan, w, X).V[:, frozen.OUT]
        slow = frozen.predict_logits(g, X, w, settle=True)
        worst = max(worst, float(np.abs(fast - slow).max()))
    assert worst > 0.1, (
        f"the worst disagreement found was {worst:.2e}; if the two paths now "
        "agree on sampled graphs, the guard in bpneat.nd.dense is unnecessary "
        "and should be removed rather than left as folklore"
    )


def test_the_general_sampler_at_two_inputs_hits_the_same_graphs():
    """And the general version declines exactly those, rather than agreeing."""
    ops = tuple(o for o in frozen.ACTIVATIONS if o != frozen.OP_MULT)
    declined_by_nd = planned_by_v3 = 0
    for seed in range(60):
        a = frozen_bl.make_random_architecture(
            np.random.default_rng(seed), 2 + seed % 9, 1 + seed % 7, ops
        )
        b = nd_bl.make_random_architecture(
            np.random.default_rng(seed), 2 + seed % 9, 1 + seed % 7, ops,
            nd.FROZEN_LAYOUT,
        )
        _same_genome(a, b)
        try:
            v3_dense.plan(a)
            planned_by_v3 += 1
        except v3_dense.NotDenseable:
            continue
        try:
            nd_dense.plan(b)
        except nd_dense.NotDenseable:
            declined_by_nd += 1
    assert planned_by_v3 and declined_by_nd, (
        f"v3 planned {planned_by_v3}, the general version declined "
        f"{declined_by_nd}; the two should differ on exactly these graphs"
    )


def test_the_dense_path_refuses_a_graph_whose_ids_are_not_topological():
    layout = nd.Layout(3, 1)
    rng = np.random.default_rng(2)
    g = nd_bl.make_mlp((4, 4), rng, layout)
    assert nd_dense.plan(g).n_nodes == g.n_nodes  # layered: accepted

    first_hidden = layout.n_structural
    g.src.append(first_hidden + 5)
    g.dst.append(first_hidden)
    g.weight.append(0.9)
    g.active.append(True)
    g.innovation.append(len(g.innovation))
    with pytest.raises(nd_dense.NotDenseable, match="backwards"):
        nd_dense.plan(g)


def test_the_dense_path_refuses_what_it_cannot_express():
    layout = nd.Layout(3, 2)
    rng = np.random.default_rng(5)
    g = nd_bl.make_mlp((4,), rng, layout)
    g.ops[layout.n_structural] = frozen.OP_MULT
    with pytest.raises(nd_dense.NotDenseable, match="mult"):
        nd_dense.plan(g)


def test_the_dense_path_refuses_data_of_the_wrong_width():
    g = nd_bl.make_mlp((3,), np.random.default_rng(1), nd.Layout(4, 1))
    plan = nd_dense.plan(g)
    with pytest.raises(ValueError, match="4 inputs"):
        nd_dense.forward(plan, np.asarray(g.weight), np.zeros((5, 2)))


def test_the_general_dense_path_agrees_with_v3s_at_two_inputs():
    """The same optimisation, so the same numbers, on the same graph."""
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    frozen_mlp = v3_learners.mixed_mlp(HIDDEN, rng_a)
    general = nd_bl.make_mixed_mlp(HIDDEN, rng_b, nd.FROZEN_LAYOUT)
    _same_genome(frozen_mlp, general)

    X = np.random.default_rng(9).normal(size=(17, 2))
    w = np.random.default_rng(10).normal(size=len(general.src))
    a = v3_dense.forward(v3_dense.plan(frozen_mlp), w, X).V[:, frozen.OUT]
    b = nd_dense.forward(nd_dense.plan(general), w, X).V[:, 3]
    assert np.allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------
# The baselines are the frozen baselines at two inputs and one output
# --------------------------------------------------------------------------


def test_the_fixed_network_is_the_frozen_fixed_network():
    for seed in range(12):
        a = frozen_bl.make_mlp(HIDDEN, np.random.default_rng(seed))
        b = nd_bl.make_mlp(HIDDEN, np.random.default_rng(seed), nd.FROZEN_LAYOUT)
        _same_genome(a, b)


def test_the_mixed_operator_network_is_v3s():
    for seed in range(12):
        a = v3_learners.mixed_mlp(HIDDEN, np.random.default_rng(seed))
        b = nd_bl.make_mixed_mlp(HIDDEN, np.random.default_rng(seed), nd.FROZEN_LAYOUT)
        _same_genome(a, b)


def test_the_random_architecture_sampler_is_the_frozen_one():
    """The null arm's distribution must be the distribution v3 and v6 sampled."""
    for seed in range(16):
        n_hidden, extra = 2 + seed % 9, seed % 7
        a = frozen_bl.make_random_architecture(
            np.random.default_rng(seed), n_hidden, extra, frozen.ACTIVATIONS
        )
        b = nd_bl.make_random_architecture(
            np.random.default_rng(seed), n_hidden, extra, frozen.ACTIVATIONS,
            nd.FROZEN_LAYOUT,
        )
        _same_genome(a, b)


def test_the_sampler_ranges_are_the_ones_the_earlier_protocols_used():
    from bpneat.v6.conditions import NULL_EXTRA_CONNECTION_RANGE, NULL_HIDDEN_RANGE

    assert nd_bl.RANDOM_HIDDEN_NODES == NULL_HIDDEN_RANGE
    assert nd_bl.RANDOM_EXTRA_CONNECTIONS == NULL_EXTRA_CONNECTION_RANGE


def test_the_linear_baseline_is_the_seed_genome():
    for seed in range(8):
        a = nd.logistic_genome(np.random.default_rng(seed), nd.FROZEN_LAYOUT)
        b = nd_bl.make_linear(np.random.default_rng(seed), nd.FROZEN_LAYOUT)
        _same_genome(a, b)
        assert b.n_nodes == nd.FROZEN_LAYOUT.n_structural, "a linear model has no hidden units"


def test_plain_training_matches_v3s_at_two_inputs():
    """The matched arm's learner, not just its architecture."""
    from bpneat.v3.datasets import make_bundle

    bundle = make_bundle("spiral", seed=9004)
    for seed in (0, 1, 2):
        a = v3_learners.mixed_mlp((8, 8), np.random.default_rng(seed))
        b = nd_bl.make_mixed_mlp((8, 8), np.random.default_rng(seed), nd.FROZEN_LAYOUT)
        w0 = np.array(a.weight, dtype=np.float64)
        wa = v3_learners.plain_train(
            a, w0, bundle.train.X, bundle.train.y,
            np.random.default_rng(100 + seed), n_steps=150,
        )
        wb = nd_bl.plain_train(
            b, w0, bundle.train.X, bundle.train.y,
            np.random.default_rng(100 + seed), n_steps=150,
        )
        assert np.allclose(wa, wb, atol=1e-12), f"seed {seed}"


@pytest.mark.parametrize("d,k", [(4, 3), (30, 1)])
def test_every_hidden_node_of_a_sampled_graph_reaches_every_output(d, k):
    layout = nd.Layout(d, k)
    for seed in range(6):
        rng = np.random.default_rng(900 + seed)
        g = nd_bl.make_random_architecture(rng, 6, 3, frozen.ACTIVATIONS, layout)
        edges = {(g.src[i], g.dst[i]) for i in range(g.n_connections) if g.active[i]}
        for h in range(layout.n_structural, g.n_nodes):
            if g.ops[h] == frozen.OP_ADD and any(
                s == layout.bias for s, dd in edges if dd == h
            ):
                continue  # a bias carrier, which feeds one output by design
            for out in layout.outputs:
                assert (h, out) in edges, f"hidden {h} misses output {out}"
