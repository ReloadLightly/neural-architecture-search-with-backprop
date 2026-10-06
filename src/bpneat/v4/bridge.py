"""The v3 bridge: a reproducibility gate, not a result.

v4 draws fresh dataset seeds, so nothing in it is paired with v3 and no v4
number can be compared to a v3 number statistically. That leaves one thing
worth checking directly: that the released v3 pipeline still produces exactly
what it produced, on its own seeds, under the current checkout.

This module re-runs v3's ``backprop_neat`` condition on the first ten v3
replicates of every task, by calling v3's own ``run_condition``, and compares
the result to the committed record field by field: champion topology, champion
weights, realized gradient steps, and every validation-side metric. A mismatch
means something that is supposed to be frozen has moved.

The bridge reads only the training and validation splits — v3's ``run_condition``
cannot reach a test split — so re-running it does not consume anyone's sealed
test. Its output is excluded from every v4 table and claim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..record import write_json_atomic
from ..v3.conditions import run_condition as v3_run_condition
from ..v3.fingerprint import fingerprints as v3_fingerprints
from .fingerprint import fingerprints
from .protocol import BRIDGE_CONDITION, bridge_runs

#: Metrics compared exactly. Floats must agree bitwise: identical code, identical
#: seeds and identical inputs leave no room for a difference.
EXACT_METRICS = (
    "train_accuracy",
    "train_loss",
    "validation_accuracy",
    "validation_loss",
    "causal_hidden_nodes",
    "causal_connections",
    "represented_nodes",
    "represented_connections",
)

V3_RELEASE = Path("results/backprop-neat-v3")


def _compare(fresh: dict, stored: dict) -> list[str]:
    diffs: list[str] = []
    for key in EXACT_METRICS:
        a, b = fresh["metrics"].get(key), stored["metrics"].get(key)
        if a != b:
            diffs.append(f"metrics.{key}: {a!r} != {b!r}")
    if fresh["compute"]["gradient_steps"] != stored["compute"]["gradient_steps"]:
        diffs.append(
            f"gradient_steps: {fresh['compute']['gradient_steps']} "
            f"!= {stored['compute']['gradient_steps']}"
        )
    if fresh["compute"]["candidate_evaluations"] != stored["compute"]["candidate_evaluations"]:
        diffs.append("candidate_evaluations differ")
    fa, fb = fresh.get("champion"), stored.get("champion")
    if fa is None or fb is None:
        diffs.append("champion missing from one side")
    else:
        for key in ("ops", "src", "dst", "active", "weights"):
            if fa[key] != fb[key]:
                diffs.append(f"champion.{key} differs")
    return diffs


def run_bridge(out_dir: Path, v3_dir: Path | None = None, progress=print) -> dict:
    """Re-run the bridge cells and compare them with the committed v3 release."""
    out_dir = Path(out_dir)
    v3_dir = Path(v3_dir) if v3_dir is not None else V3_RELEASE
    stored_dir = v3_dir / "raw" / "runs"
    if not stored_dir.is_dir():
        raise FileNotFoundError(f"no v3 release records at {stored_dir}")

    started = time.time()
    results: list[dict] = []
    for p in bridge_runs():
        rid = f"{p['task']}__{BRIDGE_CONDITION}__r{p['replicate']:02d}"
        stored_path = stored_dir / f"{rid}.json"
        if not stored_path.exists():
            results.append({"run_id": rid, "status": "absent_from_v3", "differences": []})
            progress(f"  {rid}: ABSENT from the v3 release")
            continue
        stored = json.loads(stored_path.read_text())
        fresh = v3_run_condition(
            condition=BRIDGE_CONDITION, task=p["task"], replicate=p["replicate"],
            dataset_seed=p["dataset_seed"], search_seed=p["search_seed"],
        )
        diffs = _compare(fresh, stored)
        results.append(
            {
                "run_id": rid,
                "status": "identical" if not diffs else "differs",
                "differences": diffs,
                "validation_accuracy": fresh["metrics"]["validation_accuracy"],
                "gradient_steps": fresh["compute"]["gradient_steps"],
            }
        )
        progress(f"  {rid}: {'identical' if not diffs else 'DIFFERS: ' + '; '.join(diffs)}")

    identical = sum(1 for r in results if r["status"] == "identical")
    payload = {
        "purpose": (
            "reproducibility gate: v3's released backprop_neat condition, re-run "
            "under the current checkout on its own seeds. Not a v4 result."
        ),
        "v3_release": str(v3_dir),
        "condition": BRIDGE_CONDITION,
        "n_runs": len(results),
        "n_identical": identical,
        "all_identical": identical == len(results) and bool(results),
        "fingerprints": fingerprints(),
        "v3_fingerprints": v3_fingerprints(),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    target = out_dir / "bridge" / "replication.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, payload)
    progress(f"bridge: {identical}/{len(results)} runs reproduce bit-for-bit -> {target}")
    return payload
