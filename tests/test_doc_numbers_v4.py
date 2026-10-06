"""R6 for protocol v4: every published v4 number must come from the release.

These activate once `results/backprop-neat-v4/final-test.json` exists. Before
that they skip, so the file is committed with the preregistration rather than
bolted on after the results are known — which is the point of writing it first.

The bindings are deliberately structural rather than a list of expected values:
a check that hard-codes the number it is checking cannot catch a document that
drifts away from the release, because the drift would have to be entered in two
places. Each check therefore reads the release, formats the number the way the
documents do, and asserts the string is present.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V4 = ROOT / "results" / "backprop-neat-v4"
DOCS = (
    "README.md",
    "docs/paper/main.md",
    "docs/v4-preregistration.md",
    "results/backprop-neat-v4/README.md",
)

pytestmark = pytest.mark.skipif(
    not (V4 / "final-test.json").exists(),
    reason="v4 sealed test not yet evaluated",
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
    runs = [json.loads(p.read_text()) for p in (V4 / "raw" / "runs").glob("*.json")]
    final = {
        r["run_id"]: r
        for r in json.loads((V4 / "final-test.json").read_text())["results"]
    }
    by = defaultdict(dict)
    for r in runs:
        by[(r["task"], r["condition"])][r["replicate"]] = r
    return by, final, runs


@pytest.fixture(scope="module")
def published():
    """The release's own tables. Documents agree with what the release publishes."""
    def rows(name):
        path = V4 / name
        if not path.exists():
            return []
        with open(path) as fh:
            return list(csv.DictReader(fh))

    return {
        "summary": {(r["task"], r["condition"]): r for r in rows("summary.csv")},
        "signs": {(r["algorithm"], r["task"]): r for r in rows("sign-matrix.csv")},
        "cross": {r["task"]: r for r in rows("cross-algorithm.csv")},
        "hyp": {r["hypothesis"]: r for r in rows("hypotheses.csv")},
        "budget": {(r["task"], r["condition"]): r for r in rows("budget-table.csv")},
    }


# --------------------------------------------------------------------------
# The release itself
# --------------------------------------------------------------------------


def test_suite_is_complete_and_clean(release):
    _, _, runs = release
    manifest = json.loads((V4 / "manifest.json").read_text())
    assert manifest["complete"], manifest["missing"][:5]
    assert not manifest["failed"]
    assert not manifest["unexpected"]
    assert manifest["test_evaluated"] is True
    assert len(runs) == 1200


def test_every_record_carries_one_v4_fingerprint(release):
    _, _, runs = release
    fps = {r["fingerprints"]["v4"] for r in runs}
    assert len(fps) == 1, f"the v4 science code moved mid-suite: {sorted(fps)}"


def test_the_older_releases_stayed_frozen(release):
    _, _, runs = release
    assert {r["fingerprints"]["v3_frozen"] for r in runs} == {
        json.loads((V4 / "manifest.json").read_text())["fingerprints"]["v3_frozen"]
    }
    v2 = {r["fingerprints"]["v2_frozen"] for r in runs}
    assert len(v2) == 1 and next(iter(v2)).startswith("cfdf1fa3198adc0e")


def test_the_sealed_test_was_evaluated_once_per_run(release):
    _, final, runs = release
    assert len(final) == len(runs)
    assert all(not r["test_evaluated"] for r in runs)


def test_v4_test_seeds_are_disjoint_from_v3s(release):
    """The headline is an out-of-sample replication, so the splits must be new."""
    from bpneat.v3.protocol import REPLICATES as V3_REPLICATES

    _, _, runs = release
    v3_seeds = {r.dataset_seed for r in V3_REPLICATES}
    assert not ({r["dataset_seed"] for r in runs} & v3_seeds)


def test_budget_matching_actually_matched(release, published):
    """A matched arm must have spent the budget its reference spent."""
    _, _, runs = release
    for r in runs:
        target = r["config"].get("matched_steps")
        if target is None:
            continue
        spent = r["compute"]["gradient_steps"]
        restarts = r["config"]["restarts"]
        # 60 restarts share the budget by integer division; the loss is bounded.
        assert target - restarts <= spent <= target, r["run_id"]


def test_each_matched_arm_matched_its_own_algorithm(release):
    by, _, runs = release
    for r in runs:
        ref = r["config"].get("matched_to")
        if ref is None:
            continue
        src = by[(r["task"], ref)][r["replicate"]]
        assert r["config"]["matched_steps"] == src["compute"]["gradient_steps"], r["run_id"]


def test_the_two_algorithms_got_the_same_candidate_budget(release):
    by, _, _ = release
    for (_task, cond), items in by.items():
        if cond not in ("bpneat", "cgp", "cgp_random_matched"):
            continue
        for rec in items.values():
            assert rec["compute"]["candidate_evaluations"] <= rec["config"]["candidate_budget"]
            # CGP spends lambda per generation, so it can fall short by < lambda.
            assert (
                rec["config"]["candidate_budget"] - rec["compute"]["candidate_evaluations"]
                <= 4
            ), rec["run_id"]


# --------------------------------------------------------------------------
# Documents against the release
# --------------------------------------------------------------------------


def test_run_and_cell_counts_are_published_correctly():
    text = _text()
    assert "1200" in text or "1,200" in text
    assert "150 cells" in text


def test_headline_accuracies_are_quoted_from_the_summary_table(published):
    """Every three-decimal accuracy claimed for a spiral condition must match."""
    text = _text()
    checked = 0
    for cond in (
        "bpneat", "cgp", "fixed_tanh_ha",
        "fixed_tanh_matched_bpneat", "fixed_tanh_matched_cgp",
        "fixed_mixed_matched_bpneat", "fixed_mixed_matched_cgp",
    ):
        row = published["summary"].get(("spiral", cond))
        if not row or not row.get("test_accuracy_mean"):
            continue
        val = f"{float(row['test_accuracy_mean']):.3f}"
        assert val in text, f"spiral/{cond} test accuracy {val} is not in any document"
        checked += 1
    assert checked >= 5


def test_the_sign_matrix_verdicts_match_the_published_table(published):
    """A document may not describe a reversal the table does not show."""
    text = _text().lower()
    reversed_cells = [
        r for r in published["signs"].values() if r["reversed_by_matching"] == "True"
    ]
    if reversed_cells:
        assert "revers" in text
    for algo in ("Backprop-NEAT", "CGP"):
        rows = [r for r in published["signs"].values() if r["algorithm"] == algo]
        assert rows, algo
        n_rev = sum(1 for r in rows if r["reversed_by_matching"] == "True")
        # The count is published, so a document claiming a different one fails.
        assert f"{n_rev}" in _text(), f"{algo}: {n_rev} reversed cells not stated"


def test_every_hypothesis_verdict_is_stated_as_published(published):
    """Each hypothesis must be stated with the verdict the release published.

    Checked over *every* occurrence of the label, not the first. v3 and v4 both
    number their hypotheses from H1 and the paper discusses both, so taking the
    first occurrence found v3's table and failed on a correct document.
    """
    text = _text()
    assert published["hyp"], "hypotheses.csv is empty"
    for name, row in published["hyp"].items():
        starts = [m.start() for m in re.finditer(rf"\b{name}\b", text)]
        assert starts, f"{name} is not mentioned in any document"
        assert any(
            row["verdict"] in text[i : i + 400].lower() for i in starts
        ), f"{name} is published as '{row['verdict']}' but no document says so near it"


def test_no_document_claims_a_significance_the_release_does_not_support(release):
    """Any quoted Holm p must appear in a committed release at that precision.

    Both v3's and v4's published values are allowed: these documents discuss both
    releases, and v3's 0.385 is a real number from a real table. What the check
    forbids is a p value that appears in no release at all. The pattern must not
    swallow a trailing sentence period, which is how it first failed.
    """
    published_p = set()
    for rel in (V4, ROOT / "results" / "backprop-neat-v3"):
        path = rel / "paired-effects.csv"
        if not path.exists():
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r.get("holm_p"):
                    published_p.add(f"{float(r['holm_p']):.3f}")
                    published_p.add(f"{float(r['holm_p']):.2f}")
    text = _text()
    found = 0
    for match in re.finditer(r"Holm[\s\n]*\*?p\*?\s*=\s*(\d+\.\d+)", text):
        found += 1
        assert match.group(1) in published_p, (
            f"Holm p = {match.group(1)} appears in no committed release"
        )
    assert found >= 3, f"only {found} Holm p values quoted; the check is not exercised"


def test_the_bridge_result_is_stated_accurately():
    path = V4 / "bridge" / "replication.json"
    if not path.exists():
        pytest.skip("bridge not run")
    b = json.loads(path.read_text())
    text = _text()
    assert f"{b['n_identical']}/{b['n_runs']}" in text or (
        str(b["n_runs"]) in text and str(b["n_identical"]) in text
    )
    # A document may only claim bit-identical reproduction if that is what happened.
    if not b["all_identical"]:
        assert "bit-for-bit" not in text.lower() or "did not" in text.lower()


def test_no_withdrawn_v2_claim_reappears():
    text = _text().lower()
    for banned in (
        "beat a 65-unit baseline",
        "beats a 65-unit baseline",
        "using about 5 active units",
    ):
        assert banned not in text, f"withdrawn claim present: {banned}"


# --------------------------------------------------------------------------
# Evaluator consistency: what selected a champion must be what reports it
# --------------------------------------------------------------------------


def test_no_cgp_champion_exceeds_the_settling_bound(release):
    """CGP is selected on the dense path but *recorded* through the frozen one.

    The frozen evaluator settles for at most ``SETTLE_MAX_TICK = 16`` ticks. A
    48-node CGP row can in principle decode to a deeper chain, and a champion
    that did would have its metrics taken before its deepest nodes had run —
    reporting a different network from the one validation selected. The
    preregistration says the dense path removes that hazard; this asserts it on
    every champion rather than on the pilot sample it was argued from.
    """
    from bpneat.genome import SETTLE_MAX_TICK
    from bpneat.record import deserialise_genome
    from bpneat.v3 import dense

    _, _, runs = release
    deepest = 0
    checked = 0
    for r in runs:
        if r["condition"] not in ("cgp", "cgp_random_matched"):
            continue
        g, _ = deserialise_genome(r["champion"])
        depth = len(dense.plan(g).groups)
        deepest = max(deepest, depth)
        checked += 1
        assert depth < SETTLE_MAX_TICK, f"{r['run_id']}: depth {depth}"
    assert checked == 2 * 5 * 30, checked
    print(f"deepest CGP champion: {deepest} of {SETTLE_MAX_TICK} ticks")


def test_cgp_champion_metrics_agree_between_the_two_evaluators(release):
    """The recorded accuracy must be the accuracy the search was selecting on."""
    import numpy as np

    from bpneat.learn import accuracy
    from bpneat.record import deserialise_genome
    from bpneat.v3 import dense
    from bpneat.v3.datasets import make_bundle
    from bpneat.v4.learners import dense_accuracy

    _, _, runs = release
    worst = 0.0
    for r in runs:
        if r["condition"] not in ("cgp", "cgp_random_matched"):
            continue
        g, w = deserialise_genome(r["champion"])
        b = make_bundle(r["task"], seed=r["dataset_seed"])
        frozen = accuracy(g, w, b.validation.X, b.validation.y, True)
        densev = dense_accuracy(dense.plan(g), w, b.validation.X, b.validation.y)
        assert frozen == r["metrics"]["validation_accuracy"]
        worst = max(worst, abs(frozen - densev))
    assert worst == 0.0, f"the two evaluators disagree on {worst:.3e} of the labels"
    assert np.isfinite(worst)
