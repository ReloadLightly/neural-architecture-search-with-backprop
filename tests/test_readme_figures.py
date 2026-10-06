"""The README's own figures must draw the numbers the releases publish.

`bench/readme_figures.py` is the only place in the project that renders a figure
from *two* releases at once, which is exactly where a number can drift without
any single release noticing. These gates read the published CSVs and assert that
the values the script selects are those values, and that the README's prose
agrees with them.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# `bench/` is a directory of scripts, not an installed package, so importing it
# needs the repository root on the path. `python -m pytest` puts the working
# directory there and bare `pytest` does not — which is how this file passed
# locally and failed in CI. Insert it explicitly rather than depend on the
# invocation.
sys.path.insert(0, str(ROOT))

V3 = ROOT / "results" / "backprop-neat-v3"
V4 = ROOT / "results" / "backprop-neat-v4"
FIGURES = ROOT / "docs" / "figures"
README = ROOT / "README.md"

pytestmark = pytest.mark.skipif(
    not (V4 / "summary.csv").exists(), reason="v4 release not built yet"
)


def _summary(release: Path) -> dict[tuple[str, str], dict]:
    with open(release / "summary.csv") as fh:
        return {(r["task"], r["condition"]): r for r in csv.DictReader(fh)}


def test_the_figures_exist_and_are_referenced():
    expected = {"does-search-pay.png", "topologies-found.png", "budget-decides.png"}
    assert expected <= {p.name for p in FIGURES.glob("*.png")}
    text = README.read_text()
    for name in expected:
        assert f"docs/figures/{name}" in text, f"{name} is rendered but never shown"


def test_every_committed_figure_is_shown_and_is_a_result():
    """Nothing is committed to `docs/figures/` that the README does not show.

    The rule this enforces is that every visualization in the repository is a
    measured result from a committed release record. A figure nobody points at
    is how a picture of the problem setup, or of a pilot run that is explicitly
    not evidence, ends up looking like a finding.
    """
    text = README.read_text()
    orphans = [
        p.name for p in sorted(FIGURES.iterdir())
        if p.suffix.lower() in (".png", ".gif", ".svg")
        and f"docs/figures/{p.name}" not in text
    ]
    assert not orphans, f"committed but shown nowhere: {orphans}"


def test_no_figure_draws_the_problem_setup_rather_than_a_result():
    """The input geometries and the pilot-seed animation were drawings of the
    setup, not of a finding. They are gone; this keeps them gone."""
    banned = {"task-geometries.png", "evolution.gif",
              "plate-i-forms.png", "plate-ii-complexification.png"}
    present = banned & {p.name for p in FIGURES.iterdir()}
    assert not present, f"not a measured result: {sorted(present)}"


def test_the_search_figure_draws_published_values():
    """Every accuracy the dumbbell figure plots must be in v4's summary table."""
    from bench.readme_figures import TASKS

    s4 = _summary(V4)
    for task in TASKS:
        for cond in ("cgp", "cgp_random_matched", "fixed_mixed_matched_cgp", "bpneat"):
            row = s4.get((task, cond))
            assert row is not None, f"{task}/{cond} missing from v4 summary.csv"
            assert row["test_accuracy_mean"], f"{task}/{cond} has no sealed-test value"


def test_the_topology_figure_draws_published_values():
    from bench.readme_figures import FIXED_UNITS, HA_CHAMPION_NODES

    s3, s4 = _summary(V3), _summary(V4)
    assert float(s4[("spiral", "bpneat")]["causal_hidden_nodes_mean"]) > 0
    assert float(s4[("spiral", "cgp")]["cgp_active_nodes_mean"]) > 0
    assert float(s4[("spiral", "cgp_random_matched")]["cgp_active_nodes_mean"]) > 0
    assert float(s3[("spiral", "backprop_neat")]["causal_hidden_nodes_mean"]) > 0
    # The two reference lines are published facts, not plot decoration: Ha's
    # Figure 10.3 champion and the control's own width.
    # Read the node count out of the published Figure 10.3 table rather than
    # trusting a constant that happens to say 34.
    targets = (ROOT / "docs" / "reference-targets.md").read_text()
    row = next(ln for ln in targets.splitlines() if ln.startswith("| Spirals |"))
    assert int(row.split("|")[3].strip()) == HA_CHAMPION_NODES
    assert FIXED_UNITS == float(
        s4[("spiral", "fixed_mixed_matched_bpneat")]["causal_hidden_nodes_mean"]
    )


def test_the_readme_quotes_the_sizes_it_plots():
    """The prose beside the topology figure must match the plotted values."""
    s4 = _summary(V4)
    text = README.read_text()
    for cond, key in (
        ("bpneat", "causal_hidden_nodes_mean"),
        ("cgp", "cgp_active_nodes_mean"),
        ("cgp_random_matched", "cgp_active_nodes_mean"),
    ):
        val = f"{float(s4[('spiral', cond)][key]):.1f}"
        assert val in text, f"spiral/{cond} size {val} is plotted but not stated"
    for task, cond, key in (
        ("checkerboard", "cgp", "cgp_active_nodes_mean"),
        ("checkerboard", "cgp_random_matched", "cgp_active_nodes_mean"),
    ):
        val = f"{float(s4[(task, cond)][key]):.1f}"
        assert val in text, f"{task}/{cond} size {val} is claimed but not published"


def test_the_readme_search_table_matches_the_release():
    """The §1 table is the study's headline claim; bind every cell of it."""
    s4 = _summary(V4)
    text = README.read_text()
    for task in ("spiral", "checkerboard", "spiral3"):
        for cond in ("cgp", "cgp_random_matched"):
            val = f"{float(s4[(task, cond)]['test_accuracy_mean']):.3f}"
            assert val in text, f"{task}/{cond} = {val} is in the table but not published"


def _release_totals(release: Path) -> set[str]:
    """Per-run means and the release total, both as the README formats them.

    A release total is the sum over its summary rows of mean x n, which is the
    sum of the per-run values exactly — so this reads the published table rather
    than re-deriving anything from raw records.
    """
    out: set[str] = set()
    path = release / "summary.csv"
    if not path.exists():
        return out
    grad = cand = 0.0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            n = int(r["n"])
            grad += float(r["gradient_steps_mean"]) * n
            cand += float(r["candidate_evaluations_mean"]) * n
            out.add(f"{round(float(r['gradient_steps_mean'])):,}")
    out.add(f"{round(grad):,}")
    out.add(f"{round(cand):,}")
    return out


def test_the_readme_does_not_claim_an_unpublished_budget():
    """Every compute count quoted in prose must come from a committed release.

    Covers both kinds the README quotes: a single run's realized budget, and a
    release total. The totals were unchecked until one of them failed this gate.
    """
    published: set[str] = set()
    for rel in (V4, ROOT / "results" / "backprop-neat-v3",
                ROOT / "results" / "backprop-neat-v5"):
        published |= _release_totals(rel)
    with open(V4 / "budget-table.csv") as fh:
        for r in csv.DictReader(fh):
            published.add(f"{round(float(r['gradient_steps_mean'])):,}")

    text = README.read_text()
    quoted = set(re.findall(r"\b(\d{1,3}(?:,\d{3})+) (?:gradient updates|candidate evaluations)",
                            text))
    assert len(quoted) >= 4, f"only {len(quoted)} compute counts quoted; check not exercised"
    for q in quoted:
        assert q in published, f"{q} appears in no committed release"


def test_the_operator_claims_match_the_published_tables():
    """§2's operator fractions must come from each release's operator-usage.csv.

    This block compares a 30-replicate distribution with a qualitative reading of
    the published demo champions, so the numbers carrying it are bound here
    individually rather than left to prose.
    """
    text = README.read_text()
    reference_of = {"v3": "backprop_neat", "v4": "bpneat"}
    quoted = {
        ("circle", "square"), ("circle", "abs"),
        ("xor", "mult"), ("spiral", "sin"), ("spiral3", "sin"),
    }
    checked = 0
    for tag, cond in reference_of.items():
        path = ROOT / "results" / f"backprop-neat-{tag}" / "operator-usage.csv"
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["condition"] != cond:
                    continue
                if (r["task"], r["operator"]) not in quoted:
                    continue
                val = f"{float(r['fraction']):.2f}"
                assert val in text, (
                    f"{tag} {r['task']}/{r['operator']} fraction {val} is claimed "
                    "but not published at that precision"
                )
                checked += 1
    assert checked == 2 * len(quoted), f"only {checked} operator fractions bound"


def test_the_readme_does_not_misreport_the_predicted_operators():
    """The published reading names specific operators; quote it as it stands."""
    from bench.portrait_figures import PREDICTED

    targets = (ROOT / "docs" / "reference-targets.md").read_text()
    for task, ops in PREDICTED.items():
        for op in ops:
            # `sin` is written "sine" in the book's prose; accept either spelling.
            assert op in targets or op.replace("sin", "sine") in targets, (
                f"{task}: the figure marks {op} as predicted, but "
                "docs/reference-targets.md does not record that prediction"
            )
