"""Gates for the derived tables.

Every published number is reconstructed from raw records, so the reconstruction
itself needs to be checked: pairing must line up within replicate, differences
must carry the declared sign, and the frontier must actually exclude dominated
points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.analysis import paired_effects, pareto_front, summarise


def _run(task, condition, replicate, val_acc, val_loss, causal=3, steps=100, conns=8):
    return {
        "run_id": f"{task}__{condition}__r{replicate:02d}",
        "task": task,
        "condition": condition,
        "replicate": replicate,
        "track": "B",
        "metrics": {
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
            "generalization_gap": 0.01,
            "causal_hidden_nodes": causal,
            "causal_connections": conns,
            "represented_nodes": causal + 6,
            "causal_operators": {"tanh": 1, "sin": 1},
        },
        "compute": {
            "candidate_evaluations": 100,
            "gradient_steps": steps,
            "wall_time_seconds": 1.0,
        },
    }


def test_summary_reports_spread_not_just_means():
    runs = [
        _run("xor", "backprop_neat", 1, 0.90, 0.30),
        _run("xor", "backprop_neat", 2, 0.70, 0.50),
    ]
    (row,) = summarise(runs)
    assert row["n_replicates"] == 2
    assert row["validation_accuracy_mean"] == pytest.approx(0.80)
    assert row["validation_accuracy_min"] == pytest.approx(0.70)
    assert row["validation_accuracy_max"] == pytest.approx(0.90)
    assert row["validation_accuracy_sd"] > 0
    assert row["success_count"] == 1  # xor threshold is 0.90


def test_paired_effects_pair_within_replicate():
    """A control that is better on replicate 1 and worse on 2 must average out."""
    runs = [
        _run("xor", "backprop_neat", 1, 0.90, 0.20),
        _run("xor", "fixed_mlp", 1, 0.80, 0.30),
        _run("xor", "backprop_neat", 2, 0.60, 0.60),
        _run("xor", "fixed_mlp", 2, 0.70, 0.50),
    ]
    rows = [r for r in paired_effects(runs, metric="validation_accuracy")
            if r["comparison"].endswith("fixed_mlp")]
    assert len(rows) == 1
    row = rows[0]
    assert row["n_pairs"] == 2
    # (0.90-0.80) and (0.60-0.70) -> mean 0.0
    assert row["mean_difference"] == pytest.approx(0.0)
    assert row["wins"] == 1
    assert row["ci95_low"] <= row["mean_difference"] <= row["ci95_high"]


def test_paired_effects_skip_unpaired_replicates():
    runs = [
        _run("xor", "backprop_neat", 1, 0.9, 0.2),
        _run("xor", "fixed_mlp", 1, 0.8, 0.3),
        _run("xor", "backprop_neat", 2, 0.9, 0.2),  # no control for replicate 2
    ]
    (row,) = [r for r in paired_effects(runs, metric="validation_accuracy")
              if r["comparison"].endswith("fixed_mlp")]
    assert row["n_pairs"] == 1


def test_pareto_front_excludes_dominated_points():
    runs = [
        # Strictly better on all three axes.
        _run("xor", "backprop_neat", 1, 0.9, 0.10, causal=2, steps=50, conns=4),
        # Worse loss, more structure, more compute -> dominated.
        _run("xor", "fixed_mlp", 1, 0.7, 0.40, causal=9, steps=900, conns=40),
        # Worse loss but cheapest -> not dominated.
        _run("xor", "logistic", 1, 0.6, 0.50, causal=0, steps=10, conns=3),
    ]
    front = {p["condition"] for p in pareto_front(runs)}
    assert "backprop_neat" in front
    assert "logistic" in front
    assert "fixed_mlp" not in front


def test_pareto_uses_test_loss_when_available():
    runs = [_run("xor", "backprop_neat", 1, 0.9, 0.10)]
    final = {"xor__backprop_neat__r01": {"test_loss": 0.42, "test_accuracy": 0.8,
                                         "validation_to_test_drop": 0.1}}
    (p,) = pareto_front(runs, final)
    assert p["loss_source"] == "test"
    assert p["loss"] == pytest.approx(0.42)
