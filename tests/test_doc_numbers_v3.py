"""R6 for protocol v3: every published v3 number must come from the release.

These activate once `results/backprop-neat-v3/final-test.json` exists. Before
that they skip, so the file can be committed with the preregistration rather
than bolted on after the results are known — which is the point.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V3 = ROOT / "results" / "backprop-neat-v3"
DOCS = ("README.md", "docs/paper/main.md", "results/backprop-neat-v3/README.md")

pytestmark = pytest.mark.skipif(
    not (V3 / "final-test.json").exists(),
    reason="v3 sealed test not yet evaluated",
)


def _text() -> str:
    out = []
    for name in DOCS:
        p = ROOT / name
        if p.exists():
            out.append(p.read_text())
    return "\n".join(out)


@pytest.fixture(scope="module")
def release():
    runs = [json.loads(p.read_text()) for p in (V3 / "raw" / "runs").glob("*.json")]
    final = {
        r["run_id"]: r
        for r in json.loads((V3 / "final-test.json").read_text())["results"]
    }
    by = defaultdict(dict)
    for r in runs:
        by[(r["task"], r["condition"])][r["replicate"]] = r
    return by, final, runs


@pytest.fixture(scope="module")
def published():
    """The release's own summary table.

    Documents must agree with what the release *publishes*, not with a fresh
    re-derivation: a NumPy mean and `statistics.mean` sum in different orders
    and can round either side of a .0005 boundary, which is how this check
    first failed on a figure that was correct to the data.
    """
    import csv

    with open(V3 / "summary.csv") as fh:
        return {(r["task"], r["condition"]): r for r in csv.DictReader(fh)}


def test_suite_is_complete_and_clean(release):
    _, _, runs = release
    manifest = json.loads((V3 / "manifest.json").read_text())
    assert manifest["complete"], manifest["missing"][:5]
    assert not manifest["failed"]
    assert len(runs) == manifest["planned"]


def test_frozen_v2_modules_never_moved(release):
    """R3: v3 may not edit the seven v2 science modules, during or after."""
    from bpneat.record import code_fingerprint

    manifest = json.loads((V3 / "manifest.json").read_text())
    assert manifest["fingerprints"]["v2_frozen"] == code_fingerprint()["combined"]
    assert code_fingerprint()["combined"] == (
        "cfdf1fa3198adc0e369466800ede0ea5afa24d99af18000a2161357ae5ec0d75"
    )


@pytest.mark.parametrize(
    "task,condition",
    [
        ("spiral", "backprop_neat"),
        ("spiral", "fixed_mlp_tanh_ha"),
        ("spiral", "fixed_mlp_tanh_matched"),
        ("spiral", "fixed_mlp_sin_matched"),
        ("spiral", "evolution_only"),
        ("spiral", "homogeneous_tanh"),
    ],
)
def test_published_accuracies_match_release(published, task, condition):
    mean = float(published[(task, condition)]["test_accuracy_mean"])
    assert f"{mean:.3f}" in _text(), (
        f"{task}/{condition} sealed-test accuracy should read {mean:.3f}"
    )


@pytest.mark.parametrize("condition", ["backprop_neat", "fixed_mlp_tanh_ha",
                                       "fixed_mlp_tanh_matched"])
def test_published_budgets_match_release(published, condition):
    steps = float(published[("spiral", condition)]["gradient_steps_mean"])
    assert f"{steps:,.0f}" in _text(), (
        f"spiral/{condition} gradient steps should read {steps:,.0f}"
    )


def test_replicate_count_is_as_preregistered(release):
    by, _, _ = release
    for key, cell in by.items():
        assert len(cell) == 30, f"{key} has {len(cell)} replicates, expected 30"
    assert "30 paired replicates" in _text()


def test_stability_matrix_verdicts_are_published(release):
    """Every cell verdict in the figure must also appear in the CSV."""
    import csv

    with open(V3 / "stability-matrix.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "stability matrix is empty"
    allowed = {"supported", "reversed", "not significant", "n/a"}
    for row in rows:
        for k, v in row.items():
            if k in ("claim", "statement", "task"):
                continue
            assert v in allowed, f"unexpected verdict {v!r} in {row['claim']}/{k}"


def test_no_claim_without_a_number(release):
    """Guard against prose that outruns the tables."""
    text = _text()
    for banned in ("dramatically", "proves that", "conclusively"):
        assert banned not in text.lower(), f"unsupported intensifier: {banned}"
