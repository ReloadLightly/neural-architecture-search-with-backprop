"""Every graph the released protocols hand to the fast evaluator is id-ordered.

`bpneat.v3.dense` is exact to the genome path only for graphs whose node ids are
a topological order, with the outputs as the one compensated exception. That
holds for everything v3, v4, v5 and v6 ever plan — layered fixed networks, and
decoded CGP genotypes, which may only read earlier nodes — but nothing asserted
it, so a change that made one of them wire backwards would have been found by a
reader comparing numbers rather than by CI.

See `docs/audit-dense-ordering.md`. The frozen module is not edited: it is
released and fingerprinted, and the defect it has is one none of its callers
reaches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat import genome as frozen
from bpneat.v3 import dense as v3_dense
from bpneat.v3.datasets import make_bundle
from bpneat.v3.learners import mixed_mlp
from bpneat.v4 import cgp

#: The fixed architecture every matched control in v3, v4, v5 and v6 builds.
MLP_HIDDEN = (32, 32)


def _backwards_into_hidden(g) -> list[tuple[int, int]]:
    """Active edges that run from a higher id into a node that is not the output."""
    return [
        (g.src[i], g.dst[i])
        for i in range(g.n_connections)
        if g.active[i] and g.src[i] > g.dst[i] and g.dst[i] != frozen.OUT
    ]


@pytest.mark.parametrize("seed", range(8))
def test_the_fixed_networks_are_id_ordered(seed):
    g = mixed_mlp(MLP_HIDDEN, np.random.default_rng(seed))
    assert not _backwards_into_hidden(g)
    v3_dense.plan(g)  # and therefore plannable


def test_every_decoded_cgp_genotype_is_id_ordered():
    """CGP nodes may only read earlier nodes; this asserts the decode keeps that."""
    offenders = 0
    for seed in range(200):
        gt = cgp.random_genotype(np.random.default_rng(seed))
        g = cgp.decode(gt).genome
        offenders += len(_backwards_into_hidden(g))
    assert offenders == 0, f"{offenders} backwards edges across 200 decodes"


def test_the_two_paths_agree_on_everything_the_released_protocols_plan():
    """The claim v3's docstring makes, checked where it is actually relied on."""
    bundle = make_bundle("spiral", seed=9004)
    X = bundle.train.X[:32]
    worst = 0.0

    for seed in range(6):
        g = mixed_mlp(MLP_HIDDEN, np.random.default_rng(seed))
        w = np.asarray(g.weight, dtype=np.float64)
        fast = v3_dense.forward(v3_dense.plan(g), w, X).V[:, frozen.OUT]
        slow = frozen.predict_logits(g, X, w, settle=True)
        worst = max(worst, float(np.abs(fast - slow).max()))

    for seed in range(60):
        g = cgp.decode(cgp.random_genotype(np.random.default_rng(seed))).genome
        w = np.asarray(g.weight, dtype=np.float64)
        fast = v3_dense.forward(v3_dense.plan(g), w, X).V[:, frozen.OUT]
        slow = frozen.predict_logits(g, X, w, settle=True)
        worst = max(worst, float(np.abs(fast - slow).max()))

    assert worst < 1e-10, f"worst disagreement {worst:.3e}"


def test_the_sampled_architectures_never_reach_the_fast_path():
    """The graphs that *do* wire backwards are trained on the genome path only.

    v3's ``random_search_matched`` and v6's ``null`` arm call
    ``bpneat.learn.train``, which has no dense branch. If that ever changes,
    this gate is the one that should stop it.
    """
    import inspect

    from bpneat import learn
    from bpneat.v3 import conditions as v3_conditions
    from bpneat.v6 import conditions as v6_conditions

    assert "dense" not in inspect.getsource(learn)
    for module, name in (
        (v3_conditions, "_random_search"),
        (v6_conditions, "random_architecture_search"),
    ):
        source = inspect.getsource(getattr(module, name))
        assert "dense" not in source, f"{name} now reaches the fast path"
        assert "train(" in source
