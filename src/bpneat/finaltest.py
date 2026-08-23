"""The sealed-test firewall.

This is the only module in the project permitted to read ``DatasetBundle.test``.
It refuses to run unless the confirmatory suite is complete, unmodified, and
frozen; it loads champions that were already selected on validation and never
retrains or reselects them; and it marks the release exactly once.

``search`` and ``suite`` cannot reach this code path, and the test suite asserts
that they do not.
"""

from __future__ import annotations

import json
from pathlib import Path

from .conditions import SUCCESS_THRESHOLD
from .datasets import make_bundle
from .learn import accuracy, total_error
from .protocol import PROTOCOL_VERSION
from .record import code_fingerprint, deserialise_genome, environment, write_json_atomic


class FinalTestRefused(RuntimeError):
    """Raised whenever the release is not eligible for a sealed-test evaluation."""


def _check_eligible(out_dir: Path, allow_draft: bool) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FinalTestRefused(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("test_evaluated"):
        raise FinalTestRefused(
            "this release is already marked test_evaluated; the sealed test is "
            "evaluated exactly once per protocol version"
        )
    if not manifest.get("complete"):
        raise FinalTestRefused(
            f"suite incomplete: {len(manifest.get('missing', []))} runs missing, "
            f"{len(manifest.get('failed', []))} failed"
        )
    if manifest.get("unexpected"):
        raise FinalTestRefused(f"unplanned runs present: {manifest['unexpected']}")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise FinalTestRefused(
            f"manifest protocol {manifest.get('protocol_version')!r} != "
            f"code protocol {PROTOCOL_VERSION!r}"
        )
    if PROTOCOL_VERSION.endswith("-draft") and not allow_draft:
        raise FinalTestRefused(
            f"protocol {PROTOCOL_VERSION!r} is not frozen; freeze it at Gate 5 "
            "before opening the sealed test"
        )

    current = code_fingerprint()["combined"]
    if manifest.get("code_fingerprint") != current:
        raise FinalTestRefused(
            "code fingerprint has changed since the suite ran; the runs no "
            "longer correspond to this implementation"
        )
    return manifest


def run_final_test(out_dir: Path, allow_draft: bool = False, progress=print) -> dict:
    """Evaluate every selected champion on its sealed test split, once, atomically."""
    out_dir = Path(out_dir)
    manifest = _check_eligible(out_dir, allow_draft)

    runs_dir = out_dir / "raw" / "runs"
    records = sorted(runs_dir.glob("*.json"))
    if not records:
        raise FinalTestRefused("no run records found")

    results = []
    for path in records:
        rec = json.loads(path.read_text())
        if rec.get("test_evaluated"):
            raise FinalTestRefused(f"{path.name} already carries test metrics")

        bundle = make_bundle(rec["task"], seed=rec["dataset_seed"])
        # The champion is loaded exactly as selected. No training, no reselection.
        g, w = deserialise_genome(rec["champion"])
        settle = rec["config"]["propagation"] == "settled"

        loss = total_error(g, w, bundle.test.X, bundle.test.y, settle)
        acc = accuracy(g, w, bundle.test.X, bundle.test.y, settle)
        results.append(
            {
                "run_id": rec["run_id"],
                "task": rec["task"],
                "condition": rec["condition"],
                "replicate": rec["replicate"],
                "track": rec["track"],
                "validation_loss": rec["metrics"]["validation_loss"],
                "validation_accuracy": rec["metrics"]["validation_accuracy"],
                "test_loss": loss,
                "test_accuracy": acc,
                "test_success": bool(acc >= SUCCESS_THRESHOLD[rec["task"]]),
                "validation_to_test_drop": rec["metrics"]["validation_accuracy"] - acc,
            }
        )
        progress(
            f"{rec['run_id']}: val {rec['metrics']['validation_accuracy']:.3f} "
            f"-> test {acc:.3f}"
        )

    release = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_outcome": "sealed-test binary cross-entropy of the "
        "validation-selected champion, paired within task and replicate",
        "n_runs": len(results),
        "results": results,
        "code_fingerprint": code_fingerprint()["combined"],
        "environment": environment(),
    }
    write_json_atomic(out_dir / "final-test.json", release)

    # Mark the release exactly once, only after the record is safely on disk.
    manifest["test_evaluated"] = True
    write_json_atomic(out_dir / "manifest.json", manifest)

    progress(f"\nsealed test evaluated on {len(results)} runs -> {out_dir/'final-test.json'}")
    return release


def write_checksums(out_dir: Path) -> Path:
    """A checksum manifest so the release can be verified after transport."""
    from .record import sha256_file

    out_dir = Path(out_dir)
    lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "sha256sums.txt":
            lines.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
    target = out_dir / "sha256sums.txt"
    target.write_text("\n".join(lines) + "\n")
    return target
