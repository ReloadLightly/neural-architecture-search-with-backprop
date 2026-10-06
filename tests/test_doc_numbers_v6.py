"""R6 for protocol v6: every published v6 number must come from a committed file.

Two things are bound here, and they bind from different moments in the protocol's
life.

Before any confirmatory run exists, the preregistration already quotes numbers:
the v6 science fingerprint, and the pilot cost table that decided which rungs are
confirmatory and which is a declared extension. Those come from
``results/backprop-neat-v6/pilot/cost.json`` and from the live modules, and they
are checked from the freeze onward.

After the sealed test, the release's own numbers are bound the same way v3, v4
and v5 bind theirs.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V6 = ROOT / "results" / "backprop-neat-v6"
PREREG = ROOT / "docs" / "v6-preregistration.md"
DOCS = (
    "README.md",
    "docs/paper/main.md",
    "docs/v6-preregistration.md",
    "results/backprop-neat-v6/README.md",
)


def _text() -> str:
    return "\n".join(
        (ROOT / name).read_text() for name in DOCS if (ROOT / name).exists()
    )


# --------------------------------------------------------------------------
# From the freeze onward
# --------------------------------------------------------------------------


def test_the_preregistration_records_the_live_science_fingerprint():
    """Edit a v6 science module and this fails, which is the point.

    The preregistration is the only record of what was frozen — the proxy refuses
    tag pushes — so it has to be kept honest by a gate rather than by care.
    """
    from bpneat.v6.fingerprint import v6_fingerprint

    assert v6_fingerprint()["combined"] in PREREG.read_text()


def test_the_preregistration_records_the_released_older_fingerprints():
    from bpneat.v6.fingerprint import fingerprints

    text = PREREG.read_text()
    fp = fingerprints()
    for key in ("v5_frozen", "v4_frozen", "v3_frozen", "v2_frozen"):
        assert fp[key][:16] in text, key


def test_the_preregistration_plan_matches_the_protocol():
    from bpneat.v6.protocol import (
        ALL_BUDGETS,
        BUDGETS,
        EQUIVALENCE_DELTA,
        EXTENSION_BUDGETS,
        FAMILY_SIZE,
        cells,
        planned_runs,
    )

    text = PREREG.read_text()
    assert f"{len(planned_runs()):,} confirmatory runs" in text
    assert f"{len(cells())} cells" in text
    n_extension = len(planned_runs(include_extension=True)) - len(planned_runs())
    assert f"{n_extension} runs at `b16800`" in text
    # The whole ladder row, multiplier included, so a rung cannot be described
    # in the document as something other than what the protocol computes.
    for b in ALL_BUDGETS:
        part = "**extension**" if b in EXTENSION_BUDGETS else "confirmatory"
        row = (f"| `{b.label}` | {b.generations} | {b.candidates:,} | "
               f"{b.multiplier:.2f} | {part} |")
        assert row in text, row
    for name, size in FAMILY_SIZE.items():
        assert re.search(rf"`{name}` \| {size} \|", text), name
    assert f"δ = **{EQUIVALENCE_DELTA}" in text
    span = BUDGETS[-1].candidates / BUDGETS[0].candidates
    assert f"{span:.1f}× in candidates" in text


@pytest.mark.skipif(
    not (V6 / "pilot" / "cost.json").exists(), reason="cost pilot not yet run"
)
def test_every_pilot_cost_in_the_preregistration_comes_from_the_pilot_record():
    """The cost table decided the ladder, so each of its cells is bound."""
    pilot = json.loads((V6 / "pilot" / "cost.json").read_text())
    rows = {r["condition"]: r for r in pilot["rows"]}
    text = PREREG.read_text()
    assert pilot["task"] == "spiral"
    for cond in ("search_b500", "search_b1000", "search_b2100", "search_b6300"):
        r = rows[cond]
        assert f"| `{r['budget']}` | {r['wall_time_seconds']:.1f} s | " \
               f"{r['gradient_steps']:,} | {r['steps_per_candidate']:.1f} |" in text, cond
    # The size observation that motivated v6-H7 is bound to the record too.
    lo, hi = rows["search_b500"], rows["search_b6300"]
    assert f"{lo['causal_hidden_nodes']} causally active hidden unit at " \
           f"{lo['candidate_budget']:,} candidates to " \
           f"{hi['causal_hidden_nodes']} at {hi['candidate_budget']:,}" in text
    assert f"{lo['validation_accuracy']:.3f} to " \
           f"{hi['validation_accuracy']:.3f}" in text
    # The pilot must be what the preregistration says it is.
    assert pilot["dataset_seed"] == 9004 and pilot["search_seed"] == 19004
    assert "Seeds 9004 and 19004 only" in text


def test_the_pilot_seeds_are_burned_and_no_replicate_uses_them():
    from bpneat.v6.protocol import BURNED_SEEDS, REPLICATES

    assert {9004, 19004} <= BURNED_SEEDS
    used = {r.dataset_seed for r in REPLICATES} | {r.search_seed for r in REPLICATES}
    assert not (used & {9004, 19004})


# --------------------------------------------------------------------------
# After the sealed test
# --------------------------------------------------------------------------

released = pytest.mark.skipif(
    not (V6 / "final-test.json").exists(),
    reason="v6 sealed test not yet evaluated",
)


@pytest.fixture(scope="module")
def published():
    def rows(name):
        path = V6 / name
        if not path.exists():
            return []
        with open(path) as fh:
            return list(csv.DictReader(fh))

    return {
        "summary": {(r["task"], r["condition"]): r for r in rows("summary.csv")},
        "hypotheses": {r["hypothesis"]: r for r in rows("hypotheses.csv")},
        "slopes": {
            (r["ladder"], r["task"], r["term"]): r for r in rows("scaling-slopes.csv")
        },
    }


@released
def test_every_published_verdict_matches_the_released_hypothesis_table(published):
    text = _text()
    assert published["hypotheses"], "the release has no hypothesis table"
    for name, row in published["hypotheses"].items():
        if name not in text:
            continue
        # A verdict may be quoted only in the words the release file uses.
        other = {"holds", "fails", "incomplete"} - {row["verdict"]}
        for claim in other:
            assert f"{name} {claim}" not in text, f"{name} published as {claim}"


@released
def test_no_hypothesis_is_scored_on_the_extension_rung(published):
    from bpneat.v6.protocol import EXTENSION_BUDGETS

    labels = {b.label for b in EXTENSION_BUDGETS}
    for key in published["slopes"]:
        ladder, _, _ = key
        if ladder == "with_extension":
            continue
        assert ladder == "confirmatory"
    for label in labels:
        for (_, task, term), row in published["slopes"].items():
            if row["ladder"] == "confirmatory":
                assert label not in row.get("axis", ""), (task, term)


@released
def test_the_release_counts_what_the_manifest_counts(published):
    manifest = json.loads((V6 / "manifest.json").read_text())
    final = json.loads((V6 / "final-test.json").read_text())
    n_conf = manifest["planned"]
    n_ext = manifest["extension_present"]
    assert final["n_runs"] == n_conf + n_ext
    assert manifest["test_evaluated"] is True
    text = _text()
    if "1,080" in text or "1080" in text:
        assert n_conf == 1080
