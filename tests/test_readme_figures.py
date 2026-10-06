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
