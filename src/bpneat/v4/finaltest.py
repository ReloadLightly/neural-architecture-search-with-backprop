"""The v4 sealed-test firewall.

The only module in v4 permitted to read ``DatasetBundle.test``. It refuses
anything that is not a complete, unmodified v4 release, loads champions exactly
as validation selected them, and marks the release once.

v4 checks three fingerprints, not one: its own, the released v3 set and the
frozen v2 set. v4 builds on both older sets and may edit neither, so a move in
any of the three means the release is not the thing that was run.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..learn import accuracy, total_error
from ..record import deserialise_genome, environment, write_json_atomic
from ..v3.datasets import make_bundle
from .fingerprint import fingerprints
from .protocol import PROTOCOL_VERSION, SUCCESS_THRESHOLD


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
            f"suite incomplete: {len(m.get('missing', []))} missing, "
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
    if got.get("v4") != fp["v4"]:
        raise FinalTestRefused("v4 science fingerprint has changed since the suite ran")
    if got.get("v3_frozen") != fp["v3_frozen"]:
        raise FinalTestRefused("the released v3 modules changed while v4 ran")
    if got.get("v2_frozen") != fp["v2_frozen"]:
        raise FinalTestRefused("the frozen v2 modules changed while v4 ran")
    return m


def run_final_test(out_dir: Path, progress=print) -> dict:
    out_dir = Path(out_dir)
    manifest = _check(out_dir)
    records = sorted((out_dir / "raw" / "runs").glob("*.json"))
    if not records:
        raise FinalTestRefused("no run records found")

    results = []
    for path in records:
        rec = json.loads(path.read_text())
        if rec.get("test_evaluated"):
            raise FinalTestRefused(f"{path.name} already carries test metrics")
        bundle = make_bundle(rec["task"], seed=rec["dataset_seed"])
        g, w = deserialise_genome(rec["champion"])
        # Every v4 condition is evaluated under settled propagation; see the
        # preregistration. Asserted rather than inferred from the record.
        acc = accuracy(g, w, bundle.test.X, bundle.test.y, True)
        loss = total_error(g, w, bundle.test.X, bundle.test.y, True)
        results.append(
            {
                "run_id": rec["run_id"],
                "task": rec["task"],
                "condition": rec["condition"],
                "replicate": rec["replicate"],
                "blocks": rec["blocks"],
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
                "cgp_active_nodes": rec["compute"].get("cgp_active_nodes"),
            }
        )

    release = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_outcome": "sealed-test accuracy of the validation-selected champion, "
        "paired within task and replicate",
        "test_data_provenance": "dataset seeds 50001+11i, drawn fresh for v4; no "
        "earlier release opened these splits",
        "n_runs": len(results),
        "results": results,
        "fingerprints": fingerprints(),
        "environment": environment(),
    }
    write_json_atomic(out_dir / "final-test.json", release)
    manifest["test_evaluated"] = True
    write_json_atomic(out_dir / "manifest.json", manifest)
    progress(f"sealed test evaluated on {len(results)} runs")
    return release
