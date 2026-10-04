"""Every number published in a README or doc must come from a release file.

R6: no claim may outrun its evidence. This recomputes the published quantities
from ``results/backprop-neat-v2/`` and asserts the exact rendered strings appear
in the documents that claim them. If a release is re-run, or a document is
edited by hand, this fails.

It is deliberately a *presence* check against recomputed values, not a parser:
a number that drifts stops appearing, and the test names which claim broke.
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V2 = ROOT / "results" / "backprop-neat-v2"


def _load(track: str):
    paths = glob.glob(str(V2 / track / "raw" / "runs" / "*.json"))
    runs = [json.loads(Path(p).read_text()) for p in paths]
    final = {
        r["run_id"]: r
        for r in json.loads((V2 / track / "final-test.json").read_text())["results"]
    }
    return runs, final


def _by(runs):
    out = defaultdict(dict)
    for r in runs:
        out[(r["task"], r["condition"])][r["replicate"]] = r
    return out


@pytest.fixture(scope="module")
def release():
    ra, fa = _load("track-a")
    rb, fb = _load("track-b")
    return {"a": (_by(ra), fa, ra), "b": (_by(rb), fb, rb)}


def _text(*names: str) -> str:
    return "\n".join((ROOT / n).read_text() for n in names)


# ---------------------------------------------------------------- headline

@pytest.mark.parametrize(
    "task,condition",
    [(t, c) for t in ("xor", "circle", "spiral")
     for c in ("backprop_neat", "homogeneous_tanh", "evolution_only",
               "random_search", "fixed_mlp", "logistic")],
)
def test_headline_accuracy_matches_release(release, task, condition):
    by, final, _ = release["b"]
    cell = by[(task, condition)].values()
    mean = st.mean(final[r["run_id"]]["test_accuracy"] for r in cell)
    rendered = f"{mean:.3f}"
    assert rendered in _text("README.md"), (
        f"README headline for {task}/{condition} should read {rendered}"
    )


@pytest.mark.parametrize(
    "task,condition,doc",
    [
        ("spiral", "backprop_neat", "docs/v2-errata.md"),
        ("spiral", "fixed_mlp", "docs/v2-errata.md"),
        ("xor", "backprop_neat", "docs/v2-errata.md"),
        ("xor", "fixed_mlp", "docs/v2-errata.md"),
        ("circle", "backprop_neat", "docs/v2-errata.md"),
        ("circle", "fixed_mlp", "docs/v2-errata.md"),
    ],
)
def test_errata_gradient_steps_match_release(release, task, condition, doc):
    by, _, _ = release["b"]
    steps = st.mean(r["compute"]["gradient_steps"] for r in by[(task, condition)].values())
    assert f"{steps:,.0f}" in _text(doc), (
        f"{doc}: {task}/{condition} steps should read {steps:,.0f}"
    )


def test_starvation_ratio_is_stated_correctly(release):
    by, _, _ = release["b"]
    bpn = st.mean(r["compute"]["gradient_steps"] for r in by[("spiral", "backprop_neat")].values())
    mlp = st.mean(r["compute"]["gradient_steps"] for r in by[("spiral", "fixed_mlp")].values())
    ratio = round(bpn / mlp)
    assert f"{ratio}×" in _text("README.md", "docs/v2-errata.md")


# ---------------------------------------------------------------- E2 / E4

def test_xor_win_counts_match_release(release):
    by, final, _ = release["b"]
    ref = by[("xor", "backprop_neat")]
    mlp = by[("xor", "fixed_mlp")]
    acc_wins = sum(
        final[ref[k]["run_id"]]["test_accuracy"] > final[mlp[k]["run_id"]]["test_accuracy"]
        for k in ref
    )
    perfect = sum(final[r["run_id"]]["test_accuracy"] == 1.0 for r in ref.values())
    loss_wins = sum(
        final[ref[k]["run_id"]]["test_loss"] < final[mlp[k]["run_id"]]["test_loss"] for k in ref
    )
    text = _text("README.md", "docs/v2-errata.md")
    assert f"{acc_wins}/10" in text and acc_wins == 10
    assert f"{perfect}/10" in text and perfect == 7
    assert f"{loss_wins}/10" in text and loss_wins == 8


@pytest.mark.parametrize("task,expected", [("xor", 3), ("spiral", 1)])
def test_collapse_rates_match_release(release, task, expected):
    by, _, _ = release["a"]
    runs = by[(task, "backprop_neat")].values()
    collapsed = sum(r["metrics"]["causal_hidden_nodes"] == 0 for r in runs)
    assert collapsed == expected
    assert f"{collapsed}/10" in _text("docs/v2-errata.md", "README.md")


@pytest.mark.parametrize("task", ["xor", "circle", "spiral"])
def test_track_a_means_match_release(release, task):
    by, final, _ = release["a"]
    runs = by[(task, "backprop_neat")].values()
    mean = st.mean(final[r["run_id"]]["test_accuracy"] for r in runs)
    assert f"{mean:.3f}" in _text("docs/v1-invalidation.md", "docs/reference-targets.md")


@pytest.mark.parametrize("task", ["xor", "circle", "spiral"])
def test_reference_target_sizes_match_release(release, task):
    by, _, _ = release["a"]
    runs = by[(task, "backprop_neat")].values()
    nodes = st.mean(r["metrics"]["represented_nodes"] for r in runs)
    conns = st.mean(r["metrics"]["represented_connections"] for r in runs)
    text = _text("docs/reference-targets.md")
    assert f"{nodes:.1f}" in text, f"{task} nodes should read {nodes:.1f}"
    assert f"{conns:.1f}" in text, f"{task} connections should read {conns:.1f}"


# ---------------------------------------------------------------- totals

def test_total_compute_matches_release(release):
    text = _text("README.md")
    cands = steps = 0
    for key in ("a", "b"):
        _, _, runs = release[key]
        cands += sum(r["compute"]["candidate_evaluations"] for r in runs)
        steps += sum(r["compute"]["gradient_steps"] for r in runs)
    assert f"{cands:,}" in text, f"total candidates should read {cands:,}"
    assert f"{steps:,}" in text, f"total gradient steps should read {steps:,}"


def test_validation_to_test_drop_bound_holds(release):
    """The README states a bound; the bound must actually hold."""
    worst_overall = 0.0
    worst_bpn = 0.0
    for key in ("b",):
        by, final, _ = release[key]
        for (_task, cond), d in by.items():
            drop = st.mean(final[r["run_id"]]["validation_to_test_drop"] for r in d.values())
            worst_overall = max(worst_overall, drop)
            if cond == "backprop_neat":
                worst_bpn = max(worst_bpn, drop)
    text = _text("README.md")
    assert worst_overall <= 0.032 + 1e-9, worst_overall
    assert worst_bpn <= 0.011 + 1e-9, worst_bpn
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


def test_errata_links_resolve():
    for doc in ("README.md", "docs/v2-errata.md", "docs/v1-invalidation.md",
                "docs/reference-targets.md", "docs/protocol-freeze-v1.md"):
        body = (ROOT / doc).read_text()
        for target in ("docs/v2-errata.md", "docs/audit-2026-10.md",
                       "docs/v3-preregistration.md", "docs/protocol-freeze-v2.md"):
            name = Path(target).name
            if name in body:
                assert (ROOT / target).exists(), f"{doc} links to missing {target}"
