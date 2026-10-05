"""v4 correctness gates. These must pass before any compute is spent on v4.

The substantive ones are the equivalence gates. v4's claim is that a *second*
search algorithm behaves the same way under the same controls, which is only
meaningful if the second algorithm really does share the inner learner, the
fitness, the operator semantics and the budget. Each of those is pinned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.baselines import make_mlp
from bpneat.genome import OP_MULT, OUT, causal_subgraph, forward
from bpneat.genome import backward as genome_backward
from bpneat.learn import _sigmoid, fitness_from_error, train
from bpneat.record import SCIENCE_MODULES as V2_MODULES
from bpneat.record import code_fingerprint as v2_fingerprint
from bpneat.record import deserialise_genome
from bpneat.v3 import dense
from bpneat.v3.datasets import make_bundle
from bpneat.v3.fingerprint import V3_SCIENCE_MODULES, v3_fingerprint
from bpneat.v4 import cgp
from bpneat.v4.conditions import CONDITIONS, candidate_budget, run_condition
from bpneat.v4.fingerprint import V4_SCIENCE_MODULES, fingerprints
from bpneat.v4.learners import dense_total_error, rollback_train
from bpneat.v4.protocol import (
    ALL_TASKS,
    BLOCKS,
    BURNED_SEEDS,
    FAMILY_CONDITIONS,
    FAMILY_SIZE,
    MATCHED_TO,
    REFERENCES,
    REPLICATES,
    bridge_runs,
    cells,
    planned_runs,
    validate,
)
from bpneat.v4.search import cgp_random_search, cgp_search

RUNS = 1200
CELLS = 150


def _genotypes(n: int, start: int = 0):
    for i in range(n):
        yield cgp.random_genotype(np.random.default_rng(start + i))


# --------------------------------------------------------------------------
# Protocol identity
# --------------------------------------------------------------------------


def test_protocol_validates():
    validate()


def test_plan_size_and_cells():
    assert len(planned_runs()) == RUNS
    assert len(cells()) == CELLS
    assert len({(p["task"], p["condition"], p["replicate"]) for p in planned_runs()}) == RUNS


def test_no_replicate_seed_was_spent_before():
    seeds = {r.dataset_seed for r in REPLICATES} | {r.search_seed for r in REPLICATES}
    assert not (seeds & BURNED_SEEDS)
    # v3's thirty replicates are among the burned seeds, so this also asserts
    # that v4's sealed test is drawn from splits v3 never opened.
    assert 30_001 in BURNED_SEEDS and 40_001 in BURNED_SEEDS


def test_every_planned_condition_is_defined_and_blocked():
    planned = {p["condition"] for p in planned_runs()}
    assert planned == set(CONDITIONS)
    for name, spec in CONDITIONS.items():
        assert spec.blocks, name
        assert {b.name for b in BLOCKS} >= set(spec.blocks), name


def test_families_are_declared_in_advance():
    assert set(FAMILY_CONDITIONS) == set(REFERENCES)
    assert FAMILY_SIZE == {
        ref: len(conds) * len(ALL_TASKS) for ref, conds in FAMILY_CONDITIONS.items()
    }
    # Pre-declared sizes, written out so a later edit to the condition set is
    # visible as a test failure rather than a silently widened family.
    assert FAMILY_SIZE["bpneat"] == 35
    assert FAMILY_SIZE["cgp"] == 25


def test_matched_arms_name_a_reference_that_runs_in_the_same_cell():
    by_cell: dict[tuple[str, int], set[str]] = {}
    for p in planned_runs():
        by_cell.setdefault((p["task"], p["replicate"]), set()).add(p["condition"])
    for cond, ref in MATCHED_TO.items():
        for cell, conds in by_cell.items():
            if cond in conds:
                assert ref in conds, f"{cond} in {cell} without {ref}"


def test_bridge_uses_v3_seeds_and_only_v3_seeds():
    br = bridge_runs()
    assert len(br) == len(ALL_TASKS) * 10
    v4_seeds = {r.dataset_seed for r in REPLICATES}
    for b in br:
        assert b["dataset_seed"] in BURNED_SEEDS
        assert b["dataset_seed"] not in v4_seeds


# --------------------------------------------------------------------------
# The CGP encoding
# --------------------------------------------------------------------------


def test_operator_set_is_exactly_the_dense_expressible_one():
    assert cgp.OPS == tuple(dense.DENSE_OPS)
    assert OP_MULT not in cgp.OPS


def test_genotype_is_feed_forward_by_construction():
    for gt in _genotypes(50):
        for j in range(gt.n_func):
            for k in range(cgp.ARITY):
                assert gt.src[j, k] < cgp.N_TERMINALS + j


def test_every_phenotype_is_acyclic_and_dense_planable():
    for gt in _genotypes(200):
        pheno = cgp.decode(gt)
        plan = dense.plan(pheno.genome)  # raises NotDenseable on a cycle or mult
        assert plan.n_nodes == pheno.genome.n_nodes
        assert len(pheno.slots) == pheno.genome.n_connections


def test_phenotype_holds_exactly_the_reachable_nodes():
    for gt in _genotypes(100):
        pheno = cgp.decode(gt)
        assert pheno.active == cgp.active_nodes(gt)
        # 4 structural nodes + one per active function node + the bias carrier.
        assert pheno.genome.n_nodes == 4 + len(pheno.active) + 1


def test_phenotype_is_alive_under_settled_propagation():
    X = make_bundle("spiral", seed=9002).train.X[:32]
    for gt in _genotypes(100):
        pheno = cgp.decode(gt)
        g = pheno.genome
        w = np.array(g.weight, dtype=np.float64)
        tape = forward(g, X, w, settle=True)
        assert np.all(np.isfinite(tape.vals[tape.out_var]))
        if pheno.active:
            info = causal_subgraph(g, X, w, settle=True)
            assert info["causal_hidden_nodes"] >= 1, pheno.active


def test_the_output_bias_reaches_the_output():
    """The carrier must actually carry. Under settled propagation, always."""
    X = make_bundle("circle", seed=9002).train.X[:32]
    for gt in _genotypes(60, start=700):
        pheno = cgp.decode(gt)
        g = pheno.genome
        w = np.array(g.weight, dtype=np.float64)
        ci = next(i for i, (kind, _, _) in enumerate(pheno.slots) if kind == "out_wb")
        w2 = w.copy()
        w2[ci] += 5.0
        a = forward(g, X, w, settle=True)
        b = forward(g, X, w2, settle=True)
        assert not np.allclose(a.vals[a.out_var], b.vals[b.out_var])


def test_mutation_always_changes_something():
    rng = np.random.default_rng(11)
    for gt in _genotypes(60, start=300):
        child = cgp.mutate(gt, rng)
        same = (
            np.array_equal(child.op, gt.op)
            and np.array_equal(child.src, gt.src)
            and np.array_equal(child.w, gt.w)
            and np.array_equal(child.wb, gt.wb)
            and child.out_src == gt.out_src
            and child.out_w == gt.out_w
            and child.out_wb == gt.out_wb
            and child.out_cw == gt.out_cw
        )
        assert not same


def test_mutation_does_not_touch_the_parent():
    rng = np.random.default_rng(12)
    gt = cgp.random_genotype(np.random.default_rng(13))
    before = (gt.op.copy(), gt.src.copy(), gt.w.copy(), gt.wb.copy(),
              gt.out_src, gt.out_w, gt.out_wb, gt.out_cw)
    cgp.mutate(gt, rng)
    assert np.array_equal(gt.op, before[0]) and np.array_equal(gt.src, before[1])
    assert np.array_equal(gt.w, before[2]) and np.array_equal(gt.wb, before[3])
    assert (gt.out_src, gt.out_w, gt.out_wb, gt.out_cw) == before[4:]


def test_weights_round_trip_through_the_genotype():
    """Lamarckian inheritance: write_back then decode must give back the weights."""
    rng = np.random.default_rng(14)
    for gt in _genotypes(40, start=400):
        pheno = cgp.decode(gt)
        trained = rng.normal(0.0, 1.0, pheno.genome.n_connections)
        cgp.write_back(gt, pheno, trained)
        again = cgp.decode(gt)
        assert again.active == pheno.active
        assert np.allclose(np.array(again.genome.weight), trained, atol=0, rtol=0)


def test_inactive_genes_do_not_change_the_phenotype():
    """The genotype-phenotype map is what neutral drift exploits."""
    rng = np.random.default_rng(15)
    X = make_bundle("xor", seed=9002).train.X[:32]
    checked = 0
    for gt in _genotypes(80, start=500):
        active = set(cgp.active_nodes(gt))
        inactive = [j for j in range(gt.n_func) if j not in active]
        if not inactive:
            continue
        pheno = cgp.decode(gt)
        w = np.array(pheno.genome.weight, dtype=np.float64)
        before = forward(pheno.genome, X, w, settle=True)
        twin = gt.copy()
        j = inactive[int(rng.integers(0, len(inactive)))]
        twin.op[j] = int(rng.choice(np.array(cgp.OPS, dtype=np.int64)))
        twin.wb[j] += 3.0
        after_pheno = cgp.decode(twin)
        assert after_pheno.active == pheno.active
        after = forward(
            after_pheno.genome, X,
            np.array(after_pheno.genome.weight, dtype=np.float64), settle=True,
        )
        assert np.allclose(before.vals[before.out_var], after.vals[after.out_var])
        checked += 1
    assert checked >= 50


# --------------------------------------------------------------------------
# Learner equivalence — the gate the cross-algorithm claim rests on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 4.0, 20.0])
def test_dense_and_frozen_evaluators_agree_on_values_and_gradients(scale):
    """Including the clamped regime: at scale 20 roughly a fifth of nodes clamp."""
    b = make_bundle("spiral", seed=9002)
    X, y = b.train.X[:32], b.train.y[:32]
    worst_v = worst_g = 0.0
    for i, gt in enumerate(_genotypes(30, start=1000)):
        g = cgp.decode(gt).genome
        plan = dense.plan(g)
        if len(plan.groups) > 14:  # the frozen path settles for at most 16 ticks
            continue
        w = np.random.default_rng(55 + i).normal(0.0, scale, g.n_connections)

        td = dense.forward(plan, w, X)
        tg = forward(g, X, w, settle=True)
        zd, zg = td.V[:, OUT], tg.vals[tg.out_var]
        worst_v = max(worst_v, float(np.max(np.abs(zd - zg))) / max(1.0, float(np.max(np.abs(zg)))))

        gd = dense.backward(plan, w, td, (_sigmoid(zd) - y) / len(X))
        gg = genome_backward(tg, (_sigmoid(zg) - y) / len(X), w)
        worst_g = max(worst_g, float(np.max(np.abs(gd - gg))) / max(1.0, float(np.max(np.abs(gg)))))

    assert worst_v < 1e-10, f"forward disagreement {worst_v:.2e}"
    assert worst_g < 1e-10, f"gradient disagreement {worst_g:.2e}"


@pytest.mark.parametrize("n_cycles", [1, 5])
def test_rollback_train_is_the_frozen_rule(n_cycles):
    """Same optimiser, same schedule, same realized step count, same weights.

    Asserted on CGP phenotypes over the first few updates. Beyond them the two
    trajectories part, and not because they implement different rules: see
    ``test_candidate_training_is_chaotic`` for the measurement, and
    ``test_rollback_matches_the_frozen_rule_on_a_stable_graph`` for the
    full-budget gate on a graph where the trajectory is stable.
    """
    b = make_bundle("spiral", seed=9002)
    worst_w = 0.0
    for gt in _genotypes(25, start=2000):
        g = cgp.decode(gt).genome
        plan = dense.plan(g)
        if len(plan.groups) > 14:
            continue
        w0 = np.array(g.weight, dtype=np.float64)
        a = rollback_train(
            g, w0, b.train.X, b.train.y, np.random.default_rng(7),
            n_cycles=n_cycles, plan=plan,
        )
        c = train(
            g, w0, b.train.X, b.train.y, np.random.default_rng(7),
            n_cycles=n_cycles, settle=True,
        )
        assert a.gradient_steps == c.gradient_steps
        worst_w = max(worst_w, float(np.max(np.abs(a.weights - c.weights))))
    assert worst_w < 1e-10, f"weight disagreement {worst_w:.2e}"


@pytest.mark.parametrize("hidden", [(8,), (16, 16)])
def test_rollback_matches_the_frozen_rule_on_a_stable_graph(hidden):
    """The decisive gate: the full 600-step budget, rollback breaks included.

    A tanh MLP has no expansive operator, so its training trajectory is stable
    and the two implementations can be held to the whole budget — which is where
    the rollback rule actually fires, and therefore where "same rule" has to be
    demonstrated rather than assumed.
    """
    b = make_bundle("spiral", seed=9002)
    broke_early = 0
    for i in range(8):
        g = make_mlp(hidden, np.random.default_rng(900 + i))
        plan = dense.plan(g)
        w0 = np.array(g.weight, dtype=np.float64)
        a = rollback_train(
            g, w0, b.train.X, b.train.y, np.random.default_rng(3),
            n_cycles=600, plan=plan,
        )
        c = train(
            g, w0, b.train.X, b.train.y, np.random.default_rng(3),
            n_cycles=600, settle=True,
        )
        assert a.gradient_steps == c.gradient_steps
        assert np.max(np.abs(a.weights - c.weights)) < 1e-9
        assert a.error == pytest.approx(c.error, rel=1e-10, abs=1e-12)
        broke_early += int(a.gradient_steps < 600)
    # If nothing ever broke early the gate would not have exercised rollback.
    assert broke_early > 0


def test_candidate_training_is_chaotic():
    """A measured property of the protocol, pinned so it cannot drift silently.

    Perturbing a candidate's initial weights by 1e-12 relative and training it
    under the *frozen* learner moves the trained weights by far more than the
    perturbation. The protocol's unit of evidence is therefore a distribution
    over replicates, never an individual run — which is the methodological point
    v4 makes, so it must be a gate and not a remark.
    """
    b = make_bundle("spiral", seed=9002)
    worst = 0.0
    for gt in _genotypes(12, start=2000):
        g = cgp.decode(gt).genome
        plan = dense.plan(g)
        if len(plan.groups) > 14:
            continue
        w0 = np.array(g.weight, dtype=np.float64)
        w1 = w0 * (1.0 + 1e-12)
        a = rollback_train(g, w0, b.train.X, b.train.y,
                           np.random.default_rng(7), n_cycles=600, plan=plan)
        c = rollback_train(g, w1, b.train.X, b.train.y,
                           np.random.default_rng(7), n_cycles=600, plan=plan)
        worst = max(worst, float(np.max(np.abs(a.weights - c.weights))))
    assert worst > 1e-6, (
        f"only {worst:.2e} divergence from a 1e-12 perturbation; if candidate "
        "training has become stable, the paper's sensitivity claim needs redoing"
    )


def test_dense_total_error_matches_the_frozen_one():
    from bpneat.learn import total_error

    b = make_bundle("circle", seed=9002)
    for gt in _genotypes(30, start=3000):
        g = cgp.decode(gt).genome
        plan = dense.plan(g)
        if len(plan.groups) > 14:
            continue
        w = np.array(g.weight, dtype=np.float64)
        a = dense_total_error(plan, w, b.validation.X, b.validation.y)
        c = total_error(g, w, b.validation.X, b.validation.y, True)
        assert a == pytest.approx(c, rel=1e-10, abs=1e-12)


def test_cgp_fitness_is_the_reference_penalised_fitness():
    """CGP must be selected on the same quantity Backprop-NEAT is selected on."""
    from bpneat.v4.search import evaluate

    b = make_bundle("xor", seed=9002)
    rng = np.random.default_rng(21)
    cand = evaluate(cgp.random_genotype(np.random.default_rng(22)), b, 40, 10, rng)
    expect = fitness_from_error(cand.pheno.genome, cand.error, use_penalty=True)
    assert cand.fitness == pytest.approx(expect, rel=0, abs=0)


# --------------------------------------------------------------------------
# The CGP search loop
# --------------------------------------------------------------------------


def test_cgp_search_respects_its_candidate_budget():
    b = make_bundle("xor", seed=9002)
    for budget in (9, 21, 41):
        res = cgp_search(b, budget=budget, seed=19002, inner_steps=10)
        assert res.candidates <= budget
        assert res.candidates > budget - 4 - 1
        assert res.generations == (budget - 1) // 4


def test_cgp_search_accepts_neutral_offspring():
    """Neutral drift is the mechanism, so it must actually fire."""
    b = make_bundle("xor", seed=9002)
    res = cgp_search(b, budget=201, seed=19002, inner_steps=20)
    assert res.neutral_accepted > 0
    assert res.accepted >= res.neutral_accepted


def test_cgp_champion_is_the_best_ever_seen():
    b = make_bundle("circle", seed=9002)
    res = cgp_search(b, budget=101, seed=19002, inner_steps=20)
    assert res.champion.fitness >= max(h["best_fitness"] for h in res.history)


def test_cgp_search_never_reads_the_sealed_test_split():
    clean = make_bundle("xor", seed=9002)
    poisoned = make_bundle("xor", seed=9002)
    poisoned.test.X[:] = np.nan
    poisoned.test.y[:] = np.nan
    a = cgp_search(clean, budget=41, seed=19002, inner_steps=20)
    c = cgp_search(poisoned, budget=41, seed=19002, inner_steps=20)
    assert a.champion.fitness == c.champion.fitness
    assert np.isfinite(a.champion.fitness)


def test_cgp_random_search_never_reads_the_sealed_test_split():
    clean = make_bundle("spiral", seed=9002)
    poisoned = make_bundle("spiral", seed=9002)
    poisoned.test.X[:] = np.nan
    poisoned.test.y[:] = np.nan
    a = cgp_random_search(clean, budget=12, seed=19002, inner_steps=20)
    c = cgp_random_search(poisoned, budget=12, seed=19002, inner_steps=20)
    assert a.champion.error == c.champion.error


def test_cgp_search_is_deterministic_given_its_seed():
    b = make_bundle("spiral", seed=9002)
    a = cgp_search(b, budget=41, seed=19002, inner_steps=20)
    c = cgp_search(b, budget=41, seed=19002, inner_steps=20)
    assert a.champion.fitness == c.champion.fitness
    assert np.array_equal(a.champion.weights, c.champion.weights)
    assert a.gradient_steps == c.gradient_steps


def test_lamarckian_inheritance_is_live_in_the_search():
    """A parent's genotype must carry trained weights, not birth weights."""
    from bpneat.v4.search import evaluate

    b = make_bundle("xor", seed=9002)
    gt = cgp.random_genotype(np.random.default_rng(31))
    before = gt.w.copy()
    evaluate(gt, b, 60, 10, np.random.default_rng(32), lamarckian=True)
    assert not np.array_equal(gt.w, before)

    gt2 = cgp.random_genotype(np.random.default_rng(31))
    kept = gt2.w.copy()
    evaluate(gt2, b, 60, 10, np.random.default_rng(32), lamarckian=False)
    assert np.array_equal(gt2.w, kept)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def test_run_condition_stores_a_loadable_champion():
    rec = run_condition("cgp", "xor", 99, 9002, 19002)
    assert rec["champion"] is not None
    blob = json.loads(json.dumps(rec))  # the record must survive the round trip
    g, w = deserialise_genome(blob["champion"])
    from bpneat.learn import accuracy

    b = make_bundle("xor", seed=9002)
    got = accuracy(g, w, b.validation.X, b.validation.y, True)
    assert got == pytest.approx(rec["metrics"]["validation_accuracy"], abs=1e-12)
    assert rec["test_evaluated"] is False


def test_matched_arm_refuses_to_run_without_its_reference_budget():
    with pytest.raises(ValueError, match="bpneat"):
        run_condition("fixed_tanh_matched_bpneat", "xor", 99, 9002, 19002)
    with pytest.raises(ValueError, match="cgp"):
        run_condition("fixed_tanh_matched_cgp", "xor", 99, 9002, 19002, matched_steps={})


def test_matched_arm_spends_the_budget_it_was_given():
    rec = run_condition(
        "fixed_tanh_matched_cgp", "xor", 99, 9002, 19002, matched_steps={"cgp": 6000}
    )
    # 60 restarts share the budget; integer division loses at most 59 steps.
    assert 6000 - 60 <= rec["compute"]["gradient_steps"] <= 6000
    assert rec["config"]["matched_to"] == "cgp"
    assert rec["config"]["matched_steps"] == 6000


def test_candidate_budget_matches_backprop_neats():
    from bpneat.v3.conditions import candidate_budget as v3_budget

    for task in ALL_TASKS:
        assert candidate_budget(task) == v3_budget(task)


def test_record_never_claims_a_test_evaluation():
    rec = run_condition("fixed_tanh_ha", "circle", 99, 9002, 19002)
    assert rec["test_evaluated"] is False
    assert "test_accuracy" not in rec["metrics"]


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def test_v4_fingerprint_set_is_disjoint_and_complete():
    pkg = Path(__file__).resolve().parents[1] / "src" / "bpneat" / "v4"
    on_disk = {p.name for p in pkg.glob("*.py")} - {"__init__.py"}
    non_science = {"analysis.py", "figures.py", "release.py", "verify.py",
                   "finaltest.py", "suite.py", "run.py", "bridge.py",
                   "fingerprint.py"}
    assert set(V4_SCIENCE_MODULES) == on_disk - non_science
    fp = fingerprints()
    assert fp["v3_frozen"] == v3_fingerprint()["combined"]
    assert fp["v2_frozen"] == v2_fingerprint()["combined"]
    assert fp["v4"] not in (fp["v3_frozen"], fp["v2_frozen"])


def test_the_older_fingerprints_are_the_released_ones():
    """v4 may not edit frozen v2 or released v3 science code."""
    fp = fingerprints()
    assert fp["v2_frozen"].startswith("cfdf1fa3198adc0e")
    assert fp["v3_frozen"].startswith("8438c9e89c7c72a3")
    assert set(fp["v2_modules"]) == set(V2_MODULES)
    assert set(fp["v3_modules"]) == set(V3_SCIENCE_MODULES)


def test_fixed_baselines_are_the_v3_builders():
    """The controls must be the same architectures v3 used, not lookalikes."""
    from bpneat.v3.conditions import MLP_HIDDEN as V3_HIDDEN
    from bpneat.v4.conditions import _BUILDERS

    a = _BUILDERS["tanh"](np.random.default_rng(1))
    b = make_mlp(V3_HIDDEN, np.random.default_rng(1))
    assert a.ops == b.ops and a.src == b.src and a.dst == b.dst
    assert np.allclose(a.weight, b.weight)
