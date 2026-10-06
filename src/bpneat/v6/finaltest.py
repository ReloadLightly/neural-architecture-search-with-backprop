"""The v6 sealed-test firewall.

The only module in v6 permitted to read ``DatasetBundle.test``. It refuses
anything that is not a complete, unmodified v6 confirmatory suite, loads
champions exactly as validation selected them, and marks the release once.

v6 checks five fingerprints: its own and the four released sets it builds on and
may not edit. A move in any of them means the release is not what was run.

The extension rung is evaluated here too **if and only if** it is complete when
the sealed test is taken. An incomplete extension is refused rather than
reported partially, because a sealed test that covered some of its replicates
could never be completed afterwards — the split is spent.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..learn import accuracy, total_error
from ..record import deserialise_genome, environment, write_json_atomic
from ..v3.datasets import make_bundle
from .fingerprint import fingerprints
from .protocol import (
    ALL_CONDITIONS,
    ARMS,
    EXTENSION_BUDGETS,
    PROTOCOL_VERSION,
    SUCCESS_THRESHOLD,
    condition_name,
    planned_runs,
)


class FinalTestRefused(RuntimeError):
    """The release is not eligible for a sealed-test evaluation."""


def _check(out_dir: Path) -> dict:
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        raise FinalTestRefused(f"no manifest at {mpath}")
    m = json.loads(mpath.read_text())
    if m.get("test_evaluated"):
        raise FinalTestRefused("already marked test_evaluated; the sealed test runs once")
    if not m.get("complete"):
        raise FinalTestRefused(
            f"confirmatory suite incomplete: {len(m.get('missing', []))} missing, "
            f"{len(m.get('failed', []))} failed"
        )
    if m.get("unexpected"):
        raise FinalTestRefused(f"unplanned runs present: {m['unexpected']}")
    if m.get("protocol_version") != PROTOCOL_VERSION:
        raise FinalTestRefused(
            f"manifest protocol {m.get('protocol_version')!r} != {PROTOCOL_VERSION!r}"
        )
    fp = fingerprints()
    got = m.get("fingerprints", {})
    if got.get("v6") != fp["v6"]:
        raise FinalTestRefused("v6 science fingerprint has changed since the suite ran")
    for key, label in (("v5_frozen", "v5"), ("v4_frozen", "v4"),
                       ("v3_frozen", "v3"), ("v2_frozen", "v2")):
        if got.get(key) != fp[key]:
            raise FinalTestRefused(f"the released {label} modules changed while v6 ran")
    return m


def _eligible(out_dir: Path) -> tuple[set[str], bool]:
    """Run ids to evaluate, and whether the extension rung is among them."""
    confirmatory = {
        f"{p['task']}__{p['condition']}__r{p['replicate']:02d}"
        for p in planned_runs()
    }
    every = {
        f"{p['task']}__{p['condition']}__r{p['replicate']:02d}"
        for p in planned_runs(include_extension=True)
    }
    extension = every - confirmatory
    present = {p.stem for p in (out_dir / "raw" / "runs").glob("*.json")}
    extras = present & extension
    if extras and extras != extension:
        raise FinalTestRefused(
            f"the extension rung is {len(extras)}/{len(extension)} complete; "
            "finish it or remove it before the sealed test"
        )
    return confirmatory | extras, bool(extras)


def run_final_test(out_dir: Path, progress=print) -> dict:
    out_dir = Path(out_dir)
    manifest = _check(out_dir)
    eligible, with_extension = _eligible(out_dir)
    records = sorted((out_dir / "raw" / "runs").glob("*.json"))
    if not records:
        raise FinalTestRefused("no run records found")

    results = []
    for path in records:
        if path.stem not in eligible:
            raise FinalTestRefused(f"{path.name} is not part of the plan")
        rec = json.loads(path.read_text())
        if rec.get("test_evaluated"):
            raise FinalTestRefused(f"{path.name} already carries test metrics")
        bundle = make_bundle(rec["task"], seed=rec["dataset_seed"])
        g, w = deserialise_genome(rec["champion"])
        # Every v6 condition is evaluated under settled propagation.
        acc = accuracy(g, w, bundle.test.X, bundle.test.y, True)
        loss = total_error(g, w, bundle.test.X, bundle.test.y, True)
        results.append(
            {
                "run_id": rec["run_id"],
                "task": rec["task"],
                "condition": rec["condition"],
                "arm": rec["arm"],
                "budget": rec["budget"],
                "candidate_budget": rec["candidate_budget"],
                "replicate": rec["replicate"],
                "validation_accuracy": rec["metrics"]["validation_accuracy"],
                "validation_loss": rec["metrics"]["validation_loss"],
                "test_accuracy": acc,
                "test_loss": loss,
                "test_success": bool(acc >= SUCCESS_THRESHOLD[rec["task"]]),
                "validation_to_test_drop": rec["metrics"]["validation_accuracy"] - acc,
                "collapsed": rec["metrics"]["collapsed"],
                "causal_hidden_nodes": rec["metrics"]["causal_hidden_nodes"],
                "gradient_steps": rec["compute"]["gradient_steps"],
                "candidate_evaluations": rec["compute"]["candidate_evaluations"],
                "matched_to": rec["config"].get("matched_to"),
                "matched_steps": rec["config"].get("matched_steps"),
            }
        )

    release = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_outcome": "sealed-test accuracy of the validation-selected champion, "
        "paired within geometry, rung and replicate",
        "test_data_provenance": "dataset seeds 90001+13i, drawn fresh for v6; no "
        "earlier release opened these splits",
        "confirmatory_conditions": list(ALL_CONDITIONS),
        "extension_evaluated": with_extension,
        "extension_conditions": (
            [condition_name(a, b) for b in EXTENSION_BUDGETS for a in ARMS]
            if with_extension else []
        ),
        "n_runs": len(results),
        "results": results,
        "fingerprints": fingerprints(),
        "environment": environment(),
    }
    write_json_atomic(out_dir / "final-test.json", release)
    manifest["test_evaluated"] = True
    write_json_atomic(out_dir / "manifest.json", manifest)
    progress(
        f"sealed test evaluated on {len(results)} runs"
        + (" (including the extension rung)" if with_extension else "")
    )
    return release
