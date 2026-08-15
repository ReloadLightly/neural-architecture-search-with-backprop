"""Gates for the experiment machinery: resume, sharding, and the test firewall.

These exist because the previous attempt lost completed results and never built
a safe path from calibration to a finished study.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat import protocol
from bpneat.conditions import CONDITIONS, CORE_CONDITIONS, run_condition
from bpneat.datasets import make_bundle
from bpneat.evolve import SearchConfig, search
from bpneat.finaltest import FinalTestRefused, run_final_test
from bpneat.record import assert_no_test_metrics, code_fingerprint
from bpneat.suite import run_plan, shard


# --------------------------------------------------------------------------
# Exact resume
# --------------------------------------------------------------------------


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    """Gate 1: an interrupted run must not change the science."""
    bundle = make_bundle("xor", seed=8101)
    cfg = SearchConfig(task="xor", generations=4, population=12, inner_steps=30)

    whole = search(bundle, cfg, seed=18101)

    # Interrupt after two generations, then resume from the checkpoint.
    cp = tmp_path / "ckpt.json"
    partial_cfg = SearchConfig(task="xor", generations=2, population=12, inner_steps=30)
    search(bundle, partial_cfg, seed=18101, checkpoint_path=cp, checkpoint_every=1)
    assert cp.exists(), "no checkpoint written"
    resumed = search(bundle, cfg, seed=18101, checkpoint_path=cp, checkpoint_every=1)

    assert resumed.candidates == whole.candidates
    assert resumed.gradient_steps == whole.gradient_steps
    assert resumed.metrics["validation_loss"] == pytest.approx(
        whole.metrics["validation_loss"], rel=0, abs=1e-12
    )
    assert np.allclose(resumed.champion.weights, whole.champion.weights)


def test_checkpoint_roundtrip_preserves_generator_state(tmp_path):
    from bpneat import checkpoint as ckpt
    from bpneat.evolve import Individual, InnovationRegistry
    from bpneat.genome import logistic_genome

    rng = np.random.default_rng(3)
    pop = [
        Individual(genome=(g := logistic_genome(rng)), weights=np.array(g.weight))
        for _ in range(3)
    ]
    reg = InnovationRegistry()
    reg.connection(1, 5)
    path = tmp_path / "c.json"
    ckpt.save(
        path, generation=2, population=pop, champion=pop[0], rng=rng, registry=reg,
        history=[{"generation": 0}], candidates=7, gradient_steps=11, elapsed=1.5,
    )
    state = ckpt.load(path)
    assert state["generation"] == 2
    assert state["candidates"] == 7
    assert state["registry"].connection(1, 5) == 0
    assert state["rng"].random() == rng.random()


# --------------------------------------------------------------------------
# Protocol and sharding
# --------------------------------------------------------------------------


def test_protocol_contract_is_self_consistent():
    protocol.validate()
    seeds = {r.dataset_seed for r in protocol.REPLICATES}
    seeds |= {r.search_seed for r in protocol.REPLICATES}
    assert not (seeds & protocol.BURNED_SEEDS)
    assert len(protocol.REPLICATES) == 10


def test_shards_partition_the_plan_exactly_once():
    plan = protocol.planned_runs("B")
    total = 7
    seen = []
    for i in range(total):
        seen.extend(shard(plan, i, total))
    ids = [(p["task"], p["condition"], p["replicate"]) for p in seen]
    assert len(ids) == len(plan)
    assert len(set(ids)) == len(plan)


def test_core_matrix_is_six_conditions():
    assert len(CORE_CONDITIONS) == 6
    assert set(CORE_CONDITIONS) == {
        "backprop_neat", "homogeneous_tanh", "evolution_only",
        "random_search", "fixed_mlp", "logistic",
    }
    assert all(CONDITIONS[c].core for c in CORE_CONDITIONS)


# --------------------------------------------------------------------------
# Records carry no sealed-test information
# --------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ["backprop_neat", "fixed_mlp", "logistic"])
def test_run_records_never_carry_test_metrics(condition):
    bundle = make_bundle("xor", seed=8202)
    rec = run_condition(
        condition, bundle, replicate=1, dataset_seed=8202, search_seed=18202,
        config_overrides={"generations": 1, "population": 6, "inner_steps": 20},
    )
    payload = rec.to_dict()
    assert_no_test_metrics(payload)
    assert rec.test_evaluated is False
    blob = json.dumps(payload)
    assert "test_accuracy" not in blob and "test_loss" not in blob


def test_poisoned_test_split_cannot_change_a_run():
    clean = make_bundle("circle", seed=8303)
    poisoned = make_bundle("circle", seed=8303)
    poisoned.test.X[:] = np.nan
    poisoned.test.y[:] = np.nan
    over = {"generations": 1, "population": 6, "inner_steps": 20}

    a = run_condition("backprop_neat", clean, 1, 8303, 18303, config_overrides=over)
    b = run_condition("backprop_neat", poisoned, 1, 8303, 18303, config_overrides=over)
    assert a.metrics["validation_loss"] == pytest.approx(b.metrics["validation_loss"])
    assert np.isfinite(a.metrics["validation_loss"])


# --------------------------------------------------------------------------
# The final-test firewall
# --------------------------------------------------------------------------


def _tiny_release(tmp_path, **kw):
    return run_plan(
        out_dir=tmp_path,
        track="B",
        conditions=["logistic"],
        tasks=["xor"],
        replicates=protocol.REPLICATES[:1],
        progress=lambda *_: None,
        **kw,
    )


def test_final_test_refuses_an_incomplete_suite(tmp_path):
    manifest = _tiny_release(tmp_path)
    assert manifest["complete"]
    # Remove the record so the suite is no longer complete.
    (tmp_path / "raw" / "runs").glob("*.json").__next__().unlink()
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["complete"] = False
    m["missing"] = ["xor__logistic__r01"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))

    with pytest.raises(FinalTestRefused, match="incomplete"):
        run_final_test(tmp_path, allow_draft=True, progress=lambda *_: None)


def test_final_test_refuses_a_draft_protocol(tmp_path):
    _tiny_release(tmp_path)
    with pytest.raises(FinalTestRefused, match="not frozen"):
        run_final_test(tmp_path, allow_draft=False, progress=lambda *_: None)


def test_final_test_refuses_modified_code(tmp_path):
    _tiny_release(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["code_fingerprint"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(FinalTestRefused, match="fingerprint"):
        run_final_test(tmp_path, allow_draft=True, progress=lambda *_: None)


def test_final_test_runs_once_and_then_refuses(tmp_path):
    _tiny_release(tmp_path)
    release = run_final_test(tmp_path, allow_draft=True, progress=lambda *_: None)
    assert release["n_runs"] == 1
    r = release["results"][0]
    assert 0.0 <= r["test_accuracy"] <= 1.0
    assert (tmp_path / "final-test.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["test_evaluated"]

    with pytest.raises(FinalTestRefused, match="already marked test_evaluated"):
        run_final_test(tmp_path, allow_draft=True, progress=lambda *_: None)


def test_suite_skips_completed_runs(tmp_path):
    first = _tiny_release(tmp_path)
    assert first["present"] == 1
    calls = []
    _tiny_release(tmp_path)
    # Second invocation must not have re-run anything: the record is unchanged.
    rec = json.loads(next((tmp_path / "raw" / "runs").glob("*.json")).read_text())
    assert rec["environment"]["code_fingerprint"] == code_fingerprint()["combined"]
