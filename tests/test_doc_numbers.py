"""R6 for protocol v2: every published v2 number must come from a release file.

Documents are checked against what the release **publishes** — its own
`summary.csv` — not against a fresh re-derivation. Re-deriving with a different
summation order rounds either side of a half-digit boundary and flags correct
figures as wrong; that happened twice while writing these checks.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "backprop-neat-v2"

#: Documents that may carry v2 numbers. The top README is a paper about v3 and
#: no longer republishes the v2 tables; these do.
V2_DOCS = (
    "results/backprop-neat-v2/README.md",
    "results/backprop-neat-v2/ERRATA.md",
    "docs/v2-errata.md",
    "docs/v1-invalidation.md",
    "docs/reference-targets.md",
    "docs/protocol-freeze-v2.md",
    "docs/writeup.md",
    "docs/paper/main.md",
    "README.md",
)

TASKS = ("xor", "circle", "spiral")
CONDITIONS = (
    "backprop_neat", "homogeneous_tanh", "evolution_only",
    "random_search", "fixed_mlp", "logistic",
)


def _text(*names: str) -> str:
    names = names or V2_DOCS
    return "\n".join((ROOT / n).read_text() for n in names if (ROOT / n).exists())


@pytest.fixture(scope="module")
def published():
    out = {}
    for track in ("track-a", "track-b"):
        with open(V2 / track / "summary.csv") as fh:
            for r in csv.DictReader(fh):
                out[(track, r["task"], r["condition"])] = r
    return out


@pytest.fixture(scope="module")
def raw():
    data = {}
    for track in ("track-a", "track-b"):
        runs = [json.loads(p.read_text()) for p in (V2 / track / "raw" / "runs").glob("*.json")]
        final = {
            r["run_id"]: r
            for r in json.loads((V2 / track / "final-test.json").read_text())["results"]
        }
        by = defaultdict(dict)
        for r in runs:
            by[(r["task"], r["condition"])][r["replicate"]] = r
        data[track] = (by, final, runs)
    return data


# ---------------------------------------------------------------- headline


@pytest.mark.parametrize("task,condition", [(t, c) for t in TASKS for c in CONDITIONS])
def test_headline_accuracy_matches_release(published, task, condition):
    mean = float(published[("track-b", task, condition)]["test_accuracy_mean"])
    assert f"{mean:.3f}" in _text(), (
        f"track B {task}/{condition} sealed-test accuracy should read {mean:.3f}"
    )


@pytest.mark.parametrize(
    "task,condition",
    [(t, c) for t in TASKS for c in ("backprop_neat", "fixed_mlp")],
)
def test_gradient_steps_match_release(published, task, condition):
    steps = float(published[("track-b", task, condition)]["gradient_steps_mean"])
    assert f"{steps:,.0f}" in _text(), (
        f"track B {task}/{condition} gradient steps should read {steps:,.0f}"
    )


def test_starvation_ratio_is_stated_correctly(published):
    bpn = float(published[("track-b", "spiral", "backprop_neat")]["gradient_steps_mean"])
    mlp = float(published[("track-b", "spiral", "fixed_mlp")]["gradient_steps_mean"])
    assert f"{round(bpn / mlp)}×" in _text()


# ---------------------------------------------------------------- E2 / E4


def test_xor_win_counts_match_release(raw):
    by, final, _ = raw["track-b"]
    ref, mlp = by[("xor", "backprop_neat")], by[("xor", "fixed_mlp")]
    acc_wins = sum(
        final[ref[k]["run_id"]]["test_accuracy"] > final[mlp[k]["run_id"]]["test_accuracy"]
        for k in ref
    )
    loss_wins = sum(
        final[ref[k]["run_id"]]["test_loss"] < final[mlp[k]["run_id"]]["test_loss"] for k in ref
    )
    perfect = sum(final[r["run_id"]]["test_accuracy"] == 1.0 for r in ref.values())
    text = _text()
    assert (acc_wins, perfect, loss_wins) == (10, 7, 8)
    for n in (acc_wins, perfect, loss_wins):
        assert f"{n}/10" in text


@pytest.mark.parametrize("task,expected", [("xor", 3), ("spiral", 1)])
def test_collapse_rates_match_release(raw, task, expected):
    by, _, _ = raw["track-a"]
    collapsed = sum(
        r["metrics"]["causal_hidden_nodes"] == 0 for r in by[(task, "backprop_neat")].values()
    )
    assert collapsed == expected
    assert f"{collapsed}/10" in _text()


@pytest.mark.parametrize("task", TASKS)
def test_track_a_means_match_release(published, task):
    mean = float(published[("track-a", task, "backprop_neat")]["test_accuracy_mean"])
    assert f"{mean:.3f}" in _text()


@pytest.mark.parametrize("task", TASKS)
def test_reference_target_sizes_match_release(published, task):
    row = published[("track-a", task, "backprop_neat")]
    text = _text("docs/reference-targets.md")
    for key in ("represented_nodes_mean",):
        assert f"{float(row[key]):.1f}" in text, f"{task} {key} should read {float(row[key]):.1f}"


# ---------------------------------------------------------------- totals


def test_total_compute_matches_release(raw):
    cands = steps = 0
    for track in ("track-a", "track-b"):
        _, _, runs = raw[track]
        cands += sum(r["compute"]["candidate_evaluations"] for r in runs)
        steps += sum(r["compute"]["gradient_steps"] for r in runs)
    text = _text()
    assert f"{cands:,}" in text, f"total candidates should read {cands:,}"
    assert f"{steps:,}" in text, f"total gradient steps should read {steps:,}"


def test_validation_to_test_drop_bound_holds(raw):
    """The release states a bound; the bound must actually hold."""
    by, final, _ = raw["track-b"]
    worst_overall = worst_bpn = 0.0
    for (_task, cond), cell in by.items():
        drop = st.mean(final[r["run_id"]]["validation_to_test_drop"] for r in cell.values())
        worst_overall = max(worst_overall, drop)
        if cond == "backprop_neat":
            worst_bpn = max(worst_bpn, drop)
    assert worst_overall <= 0.032 + 1e-9, worst_overall
    assert worst_bpn <= 0.011 + 1e-9, worst_bpn
    text = _text()
    assert "0.032" in text and "0.011" in text


# ---------------------------------------------------------------- hygiene


def test_no_stale_repository_name_outside_history():
    offenders = []
    for pattern in ("*.md", "*.toml", "*.cff", "*.yml"):
        for path in ROOT.rglob(pattern):
            if any(part in (".git", ".venv", "logs") for part in path.parts):
                continue
            body = path.read_text(errors="ignore")
            if "backprop-neat-summer" in body or "Backprop-NEAT — Summer" in body:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"stale repository name in {offenders}"


def test_withdrawn_claims_carry_a_warning():
    """The v2 release must not present its withdrawn claims unmarked."""
    body = (V2 / "README.md").read_text()
    assert "withdrawn" in body.lower()
    assert "ERRATA.md" in body
    assert (V2 / "ERRATA.md").exists()


def test_links_resolve():
    for doc in ("README.md", "docs/v2-errata.md", "docs/v1-invalidation.md",
                "docs/reference-targets.md", "docs/protocol-freeze-v1.md",
                "docs/paper/main.md"):
        body = (ROOT / doc).read_text()
        for target in ("docs/v2-errata.md", "docs/audit-2026-10.md",
                       "docs/v3-preregistration.md", "docs/protocol-freeze-v2.md",
                       "docs/paper/main.md"):
            if Path(target).name in body:
                assert (ROOT / target).exists(), f"{doc} links to missing {target}"
