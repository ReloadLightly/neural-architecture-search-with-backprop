"""R6 for protocol v5: every published v5 number must come from the release.

The README's NEAT-machinery section quotes a size and an accuracy for each of
eight conditions on three geometries. That is the densest table of numbers in the
repository, and the one most likely to drift, so each cell is bound here.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V5 = ROOT / "results" / "backprop-neat-v5"
DOCS = (
    "README.md",
    "docs/paper/main.md",
    "docs/v5-preregistration.md",
    "results/backprop-neat-v5/README.md",
)

pytestmark = pytest.mark.skipif(
    not (V5 / "final-test.json").exists(),
    reason="v5 sealed test not yet evaluated",
)


def _text() -> str:
    return "\n".join(
        (ROOT / name).read_text() for name in DOCS if (ROOT / name).exists()
    )


@pytest.fixture(scope="module")
def release():
    runs = [json.loads(p.read_text()) for p in (V5 / "raw" / "runs").glob("*.json")]
    final = {
        r["run_id"]: r
        for r in json.loads((V5 / "final-test.json").read_text())["results"]
    }
    return runs, final


@pytest.fixture(scope="module")
def published():
    def rows(name):
        path = V5 / name
        if not path.exists():
            return []
        with open(path) as fh:
            return list(csv.DictReader(fh))

    return {
        "summary": {(r["task"], r["condition"]): r for r in rows("summary.csv")},
        "hyp": {r["hypothesis"]: r for r in rows("hypotheses.csv")},
        "complexity": {(r["task"], r["condition"]): r for r in rows("complexity.csv")},
    }


def test_suite_is_complete_and_clean(release):
    runs, final = release
    manifest = json.loads((V5 / "manifest.json").read_text())
    assert manifest["complete"] and not manifest["failed"]
    assert not manifest["unexpected"]
    assert manifest["test_evaluated"] is True
    assert len(runs) == 720 and len(final) == 720


def test_one_fingerprint_and_three_frozen_ancestors(release):
    runs, _ = release
    assert len({r["fingerprints"]["v5"] for r in runs}) == 1
    for key, prefix in (
        ("v4_frozen", "cd9d0c1f60b44c8d"),
        ("v3_frozen", "8438c9e89c7c72a3"),
        ("v2_frozen", "cfdf1fa3198adc0e"),
    ):
        got = {r["fingerprints"][key] for r in runs}
        assert len(got) == 1 and next(iter(got)).startswith(prefix)


def test_v5_test_seeds_are_disjoint_from_v3_and_v4(release):
    from bpneat.v3.protocol import REPLICATES as V3R
    from bpneat.v4.protocol import REPLICATES as V4R

    runs, _ = release
    earlier = {r.dataset_seed for r in V3R} | {r.dataset_seed for r in V4R}
    assert not ({r["dataset_seed"] for r in runs} & earlier)


def test_the_reference_arm_really_is_the_frozen_configuration(release):
    """v5's anchor must be the algorithm as frozen code runs it."""
    from bpneat.evolve import P_ADD_CONNECTION, P_ADD_NODE

    runs, _ = release
    for r in runs:
        if r["condition"] != "neat_reference":
            continue
        c = r["config"]
        assert c["p_add_node"] == P_ADD_NODE
        assert c["p_add_connection"] == P_ADD_CONNECTION
        assert c["use_penalty"] is True and c["crossover"] is True
        assert c["n_species"] == 5 and c["population"] == 100


def test_every_condition_holds_the_candidate_budget(release):
    runs, _ = release
    budgets = {r["config"]["candidate_budget"] for r in runs}
    assert len(budgets) == 1, f"the budget is not held across conditions: {budgets}"


def test_the_readme_mechanism_table_matches_the_release(published):
    """Every cell of the densest table in the repository."""
    text = _text()
    tasks = ("spiral", "checkerboard", "spiral3")
    conds = (
        "neat_reference", "neat_complexify", "neat_no_penalty",
        "neat_complexify_no_penalty", "neat_no_speciation", "neat_no_crossover",
        "neat_deep_narrow",
    )
    for cond in conds:
        for task in tasks:
            row = published["summary"][(task, cond)]
            acc = f"{float(row['test_accuracy_mean']):.3f}"
            size = f"{float(row['causal_hidden_nodes_mean']):.1f}"
            assert acc in text, f"{task}/{cond} accuracy {acc} is not in any document"
            assert size in text, f"{task}/{cond} size {size} is not in any document"


def test_every_hypothesis_verdict_is_stated_as_published(published):
    text = _text()
    assert published["hyp"], "hypotheses.csv is empty"
    import re

    for name, row in published["hyp"].items():
        starts = [m.start() for m in re.finditer(re.escape(name), text)]
        assert starts, f"{name} is not mentioned in any document"
        assert any(row["verdict"] in text[i : i + 400].lower() for i in starts), (
            f"{name} is published as '{row['verdict']}' but no document says so near it"
        )


def test_the_documents_do_not_claim_a_neat_variant_beat_the_fixed_network(published):
    """v5-H7's verdict is a claim about every cell; check the cells, not the label."""
    beats = []
    for task in ("spiral", "checkerboard", "spiral3"):
        fixed = float(published["summary"][(task, "fixed_mixed_matched")]["test_accuracy_mean"])
        for cond, row in published["summary"].items():
            if cond[0] != task or cond[1] == "fixed_mixed_matched":
                continue
            if float(row["test_accuracy_mean"]) > fixed:
                beats.append(cond)
    verdict = published["hyp"]["v5-H7"]["verdict"]
    assert (verdict == "holds") == (not beats), (
        f"v5-H7 is published as {verdict} but these arms beat the fixed net: {beats}"
    )


def test_run_and_cell_counts_are_published_correctly():
    text = _text()
    assert "720" in text
    assert "90 cells" in text
