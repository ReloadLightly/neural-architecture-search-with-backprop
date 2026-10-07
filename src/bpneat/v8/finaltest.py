"""The v8 sealed-test firewall.

The only module in v8 permitted to read ``TabularBundle.test``. It refuses
anything that is not a complete, unmodified v8 confirmatory suite, loads
champions exactly as validation selected them, and marks the release once.

v8 checks six fingerprints: its own, the committed datasets by content, the
n-dimensional core, and the three released sets that core is proved identical
to at two inputs. The data is checked by content and not by loader because a
release built on a dataset that later changed would otherwise look sound.

A champion is rebuilt through :func:`bpneat.nd.encoding.deserialise` with the
layout its own record carries. The frozen deserialiser would return a two-input
graph, which is the right object for every protocol before this one and silently
the wrong one for this one.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..nd.datasets import make_bundle
from ..nd.encoding import Layout, deserialise
from ..nd.learn import accuracy, total_error
from ..record import environment, write_json_atomic
from .fingerprint import fingerprints
from .protocol import ALL_TASKS, EXTENSION_TASKS, PROTOCOL_VERSION, planned_runs


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
    if got.get("v8") != fp["v8"]:
        raise FinalTestRefused("v8 science fingerprint has changed since the suite ran")
    if got.get("data") != fp["data"]:
        raise FinalTestRefused("the committed datasets have changed since the suite ran")
    if got.get("nd") != fp["nd"]:
        raise FinalTestRefused("the n-dimensional core changed while v8 ran")
    for key, label in (("v5_frozen", "v5"), ("v3_frozen", "v3"), ("v2_frozen", "v2")):
        if got.get(key) != fp[key]:
            raise FinalTestRefused(f"the released {label} modules changed while v8 ran")
    return m


def _eligible(out_dir: Path) -> tuple[set[str], bool]:
    """Run ids to evaluate, and whether the extension dataset is among them."""
    def ids(include_extension):
        return {
            f"{p['task']}__{p['condition']}__r{p['replicate']:02d}"
            for p in planned_runs(include_extension)
        }

    confirmatory = ids(False)
    extension = ids(True) - confirmatory
    present = {p.stem for p in (out_dir / "raw" / "runs").glob("*.json")}
    extras = present & extension
    if extras and extras != extension:
        raise FinalTestRefused(
            f"the extension dataset is {len(extras)}/{len(extension)} complete; "
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
        bundle = make_bundle(rec["task"], seed=rec["split_seed"])
        layout = Layout(
            rec["champion_layout"]["n_inputs"], rec["champion_layout"]["n_outputs"]
        )
        if layout.n_inputs != bundle.n_features:
            raise FinalTestRefused(
                f"{path.name}: champion takes {layout.n_inputs} inputs, "
                f"{rec['task']} has {bundle.n_features} features"
            )
        g, w = deserialise(rec["champion"], layout)
        acc = accuracy(g, w, bundle.test.X, bundle.test.y, True)
        loss = total_error(g, w, bundle.test.X, bundle.test.y, True)
        results.append(
            {
                "run_id": rec["run_id"],
                "task": rec["task"],
                "condition": rec["condition"],
                "arm": rec["arm"],
                "replicate": rec["replicate"],
                "validation_accuracy": rec["metrics"]["validation_accuracy"],
                "validation_loss": rec["metrics"]["validation_loss"],
                "test_accuracy": acc,
                "test_loss": loss,
                "test_rows": len(bundle.test),
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
        "paired within dataset and replicate",
        "test_data_provenance": "three tenths of each committed dataset, stratified, "
        "by split seeds 110001+13i; a replicate is a different partition of the same "
        "finite rows, and no earlier release has opened any of them",
        "confirmatory_datasets": list(ALL_TASKS),
        "extension_evaluated": with_extension,
        "extension_datasets": list(EXTENSION_TASKS) if with_extension else [],
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
        + (" (including the extension dataset)" if with_extension else "")
    )
    return release
