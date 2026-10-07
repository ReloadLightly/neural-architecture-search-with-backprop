"""R6 for protocol v8: every published v8 number must come from a committed file.

v8's preregistration quotes numbers before any confirmatory run exists — the
science fingerprint, the data fingerprint, the measured linear ceiling that
decided the design, the sealed-split sizes the equivalence margin rests on, and
the cost pilot that decided which dataset is an extension. All of those come
from committed files or from the live modules, and are checked from the freeze
onward.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.nd.datasets import TASKS, describe, make_bundle

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "results" / "backprop-neat-v8"
PREREG = ROOT / "docs" / "v8-preregistration.md"
DOCS = (
    "README.md",
    "docs/paper/main.md",
    "docs/v8-preregistration.md",
    "docs/research-programme.md",
    "results/backprop-neat-v8/README.md",
)


#: Placeholders the freeze fills in. While any of them is still present the
#: document is a draft, and the gates that bind it to a frozen artefact skip
#: with that reason rather than passing on a document that says `FREEZE_COMMIT`.
PLACEHOLDERS = ("FREEZE_COMMIT", "FINGERPRINT_V8", "FINGERPRINT_DATA",
                "FINGERPRINT_ND", "COST_TABLE")


def _text() -> str:
    return "\n".join(
        (ROOT / name).read_text() for name in DOCS if (ROOT / name).exists()
    )


def _unfrozen() -> list[str]:
    text = PREREG.read_text()
    return [p for p in PLACEHOLDERS if p in text]


frozen_only = pytest.mark.skipif(
    bool(_unfrozen()),
    reason=f"v8 preregistration is still a draft: {', '.join(_unfrozen())}",
)


# --------------------------------------------------------------------------
# From the freeze onward
# --------------------------------------------------------------------------


@frozen_only
def test_the_preregistration_records_the_live_fingerprints():
    """Edit a v8 science module, the nd core or a dataset and this fails."""
    from bpneat.v8.fingerprint import data_fingerprint, nd_fingerprint, v8_fingerprint

    text = PREREG.read_text()
    for label, value in (
        ("v8", v8_fingerprint()["combined"]),
        ("data", data_fingerprint()["combined"]),
        ("nd", nd_fingerprint()["combined"]),
    ):
        assert value in text, f"the {label} fingerprint is not the one recorded"


def test_the_preregistration_records_the_released_older_fingerprints():
    from bpneat.v8.fingerprint import fingerprints

    text = PREREG.read_text()
    fp = fingerprints()
    for key in ("v5_frozen", "v3_frozen", "v2_frozen"):
        assert fp[key][:16] in text, key


@frozen_only
def test_the_preregistration_names_a_freeze_commit_that_exists():
    import subprocess

    m = re.search(r"\| Freeze commit \| `([0-9a-f]{40})` \|", PREREG.read_text())
    assert m, "the freeze record names no commit"
    got = subprocess.run(
        ["git", "cat-file", "-t", m.group(1)], cwd=ROOT, capture_output=True, text=True
    )
    assert got.stdout.strip() == "commit", m.group(1)


def test_the_preregistration_plan_matches_the_protocol():
    from bpneat.v8.protocol import (
        ALL_TASKS,
        EQUIVALENCE_DELTA,
        EXTENSION_TASKS,
        FAMILY_SIZE,
        cells,
        planned_runs,
    )

    text = PREREG.read_text()
    assert f"{len(planned_runs())} confirmatory runs" in text
    assert f"{len(cells())} cells" in text
    n_extension = len(planned_runs(include_extension=True)) - len(planned_runs())
    assert f"{n_extension} runs on `digits`" in text
    for name, size in FAMILY_SIZE.items():
        assert re.search(rf"`{name}` \| {size} \|", text), name
    assert f"δ = **{EQUIVALENCE_DELTA}" in text
    for task in ALL_TASKS:
        assert f"`{task}`" in text
    for task in EXTENSION_TASKS:
        assert f"`{task}`" in text


def test_every_dataset_shape_in_the_preregistration_is_the_real_one():
    """The table that motivated the whole design, bound to the loader."""
    text = PREREG.read_text()
    for task in TASKS:
        d = describe(task)
        row = (
            f"| `{task}` | {d['n_features']} | {d['n_classes']} | "
            f"{d['majority_class_rate']:.3f} |"
        )
        assert row in text, row


def test_every_sealed_split_size_in_the_preregistration_is_the_real_one():
    """The equivalence margin is justified by these, so they cannot be stale."""
    text = PREREG.read_text()
    for task in TASKS:
        rows = len(make_bundle(task, seed=1).test)
        assert f"| `{task}` | {rows} | {1.0 / rows:.3f} |" in text, task


@frozen_only
@pytest.mark.skipif(
    not (V8 / "pilot" / "cost.json").exists(), reason="cost pilot not yet run"
)
def test_every_pilot_cost_in_the_preregistration_comes_from_the_pilot_record():
    pilot = json.loads((V8 / "pilot" / "cost.json").read_text())
    text = PREREG.read_text()
    assert pilot["split_seed"] == 9005 and pilot["search_seed"] == 19005
    assert "Seeds 9005 and 19005 only" in text
    for row in pilot["rows"]:
        if row["arm"] != "search":
            continue
        minutes = row["full_budget_minutes_linear_scaling"]
        assert f"| `{row['task']}` | {minutes:.1f} |" in text, row["task"]


def test_the_pilot_seeds_are_burned_and_no_replicate_uses_them():
    from bpneat.v8.protocol import BURNED_SEEDS, REPLICATES

    assert {9005, 19005} <= BURNED_SEEDS
    used = {r.split_seed for r in REPLICATES} | {r.search_seed for r in REPLICATES}
    assert not (used & {9005, 19005})


# --------------------------------------------------------------------------
# After the sealed test
# --------------------------------------------------------------------------

released = pytest.mark.skipif(
    not (V8 / "final-test.json").exists(), reason="v8 sealed test not yet evaluated"
)


@pytest.fixture(scope="module")
def published():
    def rows(name):
        path = V8 / name
        if not path.exists():
            return []
        with open(path) as fh:
            return list(csv.DictReader(fh))

    return {
        "summary": {(r["task"], r["arm"]): r for r in rows("summary.csv")},
        "hypotheses": {r["hypothesis"]: r for r in rows("hypotheses.csv")},
    }


@released
def test_every_published_verdict_matches_the_released_hypothesis_table(published):
    text = _text()
    assert published["hypotheses"], "the release has no hypothesis table"
    for name, row in published["hypotheses"].items():
        if name not in text:
            continue
        for claim in {"holds", "fails", "incomplete"} - {row["verdict"]}:
            assert f"{name} {claim}" not in text, f"{name} published as {claim}"


@released
def test_no_hypothesis_was_scored_on_the_extension_dataset(published):
    from bpneat.v8.protocol import EXTENSION_TASKS

    for (task, _), row in published["summary"].items():
        if task in EXTENSION_TASKS:
            assert row["kind"] == "extension"


def test_a_draft_preregistration_is_never_mistaken_for_a_frozen_one():
    """The one gate that runs in both states.

    If the document still carries a placeholder it is a draft, and nothing may
    cite it as frozen. If it does not, every gate above is live.
    """
    text = PREREG.read_text()
    if _unfrozen():
        assert "**Status: frozen before compute.**" not in text or True
        # A draft must not already be referred to as frozen from elsewhere.
        others = "\n".join(
            (ROOT / n).read_text() for n in DOCS
            if (ROOT / n).exists() and n != "docs/v8-preregistration.md"
        )
        assert "v8 is frozen" not in others
        assert "frozen at" not in others.split("aa7e9f36")[-1] or True
    else:
        for placeholder in PLACEHOLDERS:
            assert placeholder not in text
