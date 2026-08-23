"""Gates for the release-verification command and the champion gallery.

``bpneat verify`` is the claim the README makes to a stranger: that the numbers
in this repository regenerate from its own raw records. A verifier that cannot
fail is worth nothing, so most of what follows tampers with a release and
insists the command notices.

The gallery is new code that touches datasets, so it ships with the same
poison-gate discipline as the search: NaN the sealed test split, and every array
it produces must be unchanged.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.analysis import build
from bpneat.champions import (
    boundary_field,
    causal_counts_agree,
    causal_elements,
    champion_bundle,
    select_champions,
    topology_layout,
)
from bpneat.cli import main
from bpneat.datasets import make_bundle
from bpneat.finaltest import write_checksums
from bpneat.record import code_fingerprint
from bpneat.release import DERIVED_FILES, verify

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "results" / "backprop-neat-v2"

requires_release = pytest.mark.skipif(
    not (RELEASE / "track-b" / "raw" / "runs").is_dir(),
    reason="the committed v2 release is not present in this checkout",
)


# --------------------------------------------------------------------------
# A small synthetic release, so the verifier's failure modes are cheap to test
# --------------------------------------------------------------------------


def _record(task: str, condition: str, replicate: int, value: float) -> dict:
    return {
        "run_id": f"{task}__{condition}__r{replicate:02d}",
        "task": task,
        "condition": condition,
        "replicate": replicate,
        "track": "B",
        "dataset_seed": 8000 + replicate,
        "search_seed": 18000 + replicate,
        "config": {"propagation": "settled"},
        "metrics": {
            "validation_accuracy": value,
            "validation_loss": 1.0 - value,
            "causal_hidden_nodes": 3,
            "causal_connections": 7,
            "causal_operators": {"sin": 2, "mult": 1},
            "represented_nodes": 8,
            "represented_connections": 11,
            "generalization_gap": 0.01,
        },
        "compute": {
            "candidate_evaluations": 100,
            "gradient_steps": 1000,
            "wall_time_seconds": 1.5,
        },
    }


def _tiny_release(root: Path) -> Path:
    """A complete, self-consistent release: records, tables, checksums."""
    runs = root / "raw" / "runs"
    runs.mkdir(parents=True)
    results = []
    for rep in (1, 2, 3):
        for condition, base in (("backprop_neat", 0.90), ("fixed_mlp", 0.80)):
            rec = _record("xor", condition, rep, base + 0.01 * rep)
            (runs / f"{rec['run_id']}.json").write_text(json.dumps(rec, indent=2))
            results.append(
                {
                    "run_id": rec["run_id"],
                    "task": "xor",
                    "condition": condition,
                    "replicate": rep,
                    "track": "B",
                    "validation_loss": rec["metrics"]["validation_loss"],
                    "validation_accuracy": rec["metrics"]["validation_accuracy"],
                    "test_loss": rec["metrics"]["validation_loss"] + 0.02,
                    "test_accuracy": rec["metrics"]["validation_accuracy"] - 0.01,
                    "test_success": True,
                    "validation_to_test_drop": 0.01,
                }
            )
    (root / "final-test.json").write_text(
        json.dumps({"protocol_version": "v2", "n_runs": len(results), "results": results}, indent=2)
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "v2",
                "track": "B",
                "complete": True,
                "test_evaluated": True,
                "failed": [],
                "missing": [],
                "code_fingerprint": code_fingerprint()["combined"],
            },
            indent=2,
        )
    )
    build(root, progress=lambda *_: None)
    write_checksums(root)
    return root


@pytest.fixture
def tiny(tmp_path):
    return _tiny_release(tmp_path / "release")


def test_a_pristine_release_verifies(tiny):
    report = verify(tiny)
    assert report.ok, [str(f) for f in report.failures]
    assert not report.errata
    # Every derived table was proven by regeneration, not merely by hash.
    assert "regeneration  5/5" in "\n".join(report.lines)


def test_editing_a_raw_record_is_caught(tiny):
    path = next((tiny / "raw" / "runs").glob("*.json"))
    rec = json.loads(path.read_text())
    rec["metrics"]["validation_accuracy"] = 0.123456
    path.write_text(json.dumps(rec, indent=2))

    report = verify(tiny)
    assert not report.ok
    checks = {f.check for f in report.failures}
    # The record no longer hashes, and the tables no longer follow from it.
    assert "checksums" in checks
    assert "regeneration" in checks


def test_editing_a_derived_table_is_caught(tiny):
    target = tiny / "summary.csv"
    target.write_text(target.read_text().replace("xor", "xor "))

    report = verify(tiny)
    assert not report.ok
    assert any(f.check == "regeneration" and "summary.csv" in f.detail for f in report.failures)


def test_deleting_a_listed_file_is_caught(tiny):
    (tiny / "raw" / "runs" / "xor__fixed_mlp__r03.json").unlink()
    report = verify(tiny)
    assert not report.ok
    assert any("listed but absent" in f.detail for f in report.failures)


def test_an_unaccounted_file_is_caught(tiny):
    (tiny / "raw" / "runs" / "xor__fixed_mlp__r99.json").write_text("{}")
    report = verify(tiny)
    assert not report.ok
    assert any("not accounted for" in f.detail for f in report.failures)


def test_an_incomplete_or_unevaluated_manifest_is_caught(tiny):
    path = tiny / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["complete"] = False
    manifest["test_evaluated"] = False
    manifest["failed"] = ["xor__backprop_neat__r01"]
    path.write_text(json.dumps(manifest, indent=2))
    write_checksums(tiny)

    report = verify(tiny)
    assert not report.ok
    details = " ".join(f.detail for f in report.failures)
    assert "not marked complete" in details
    assert "never evaluated" in details
    assert "recorded as failed" in details


def test_a_moved_code_fingerprint_is_caught(tiny):
    path = tiny / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["code_fingerprint"] = "0" * 64
    path.write_text(json.dumps(manifest, indent=2))
    write_checksums(tiny)

    report = verify(tiny)
    assert not report.ok
    assert any(f.check == "fingerprint" for f in report.failures)


def test_a_stale_derived_hash_is_an_erratum_not_a_failure(tiny):
    """The v2 release's own defect, reproduced deliberately.

    ``write_checksums`` runs before the tables are regenerated with the
    sealed-test columns, so the recorded hash goes stale while the committed
    bytes stay correct. Regeneration proves those bytes independently, so this
    is reported and does not fail the release.
    """
    sums = tiny / "sha256sums.txt"
    lines = []
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        lines.append(f"{'0' * 64}  {name}" if name == "summary.csv" else line)
    sums.write_text("\n".join(lines) + "\n")

    report = verify(tiny)
    assert report.ok, [str(f) for f in report.failures]
    assert any("summary.csv" in f.detail and "stale hash" in f.detail for f in report.errata)


def test_a_missing_derived_entry_is_an_erratum_not_a_failure(tiny):
    """Track B's symptom of the same defect: the entry was never written."""
    sums = tiny / "sha256sums.txt"
    kept = [
        line
        for line in sums.read_text().splitlines()
        if line.partition("  ")[2] not in DERIVED_FILES
    ]
    sums.write_text("\n".join(kept) + "\n")

    report = verify(tiny)
    assert report.ok, [str(f) for f in report.failures]
    assert len(report.errata) == len(DERIVED_FILES)


def test_a_stale_hash_on_an_unprovable_file_still_fails(tiny):
    """The carve-out is conditional: break regeneration and it disappears."""
    target = tiny / "summary.csv"
    target.write_text(target.read_text() + "corrupted\n")

    report = verify(tiny)
    assert not report.ok
    checks = {f.check for f in report.failures}
    assert {"regeneration", "checksums"} <= checks
    assert not any("stale hash" in f.detail for f in report.errata)


def test_verify_is_read_only(tiny):
    before = {
        p: p.stat().st_mtime_ns for p in sorted(tiny.rglob("*")) if p.is_file()
    }
    digests = {p: p.read_bytes() for p in before}
    verify(tiny)
    after = {p: p.stat().st_mtime_ns for p in sorted(tiny.rglob("*")) if p.is_file()}
    assert set(after) == set(before)
    assert all(p.read_bytes() == digests[p] for p in digests)


def test_the_cli_exit_code_follows_the_report(tiny, capsys):
    assert main(["verify", "--dir", str(tiny)]) == 0
    (tiny / "manifest.json").unlink()
    assert main(["verify", "--dir", str(tiny)]) == 1
    assert "FAILED" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The champion gallery
# --------------------------------------------------------------------------


@requires_release
def test_champion_selection_is_deterministic_and_takes_the_lower_median():
    first = select_champions(RELEASE, "track-b")
    second = select_champions(RELEASE, "track-b")
    assert set(first) == {"xor", "circle", "spiral"}
    assert {t: c.run_id for t, c in first.items()} == {t: c.run_id for t, c in second.items()}

    from bpneat.analysis import load_final_test, load_runs

    final = load_final_test(RELEASE / "track-b")
    for task, champ in first.items():
        accs = sorted(
            (final[r["run_id"]]["test_accuracy"], r["replicate"])
            for r in load_runs(RELEASE / "track-b")
            if r["task"] == task and r["condition"] == "backprop_neat"
        )
        assert len(accs) == 10
        # Ten replicates have no single middle; the rule takes the lower of the
        # two central positions, breaking ties on the smaller replicate id.
        assert (champ.test_accuracy, champ.replicate) == accs[4]


@requires_release
def test_the_local_causal_walk_agrees_with_the_frozen_implementation():
    """``genome.py`` cannot be edited to expose the sets, so the walk is
    duplicated. It must never drift from the implementation it copies."""
    for champ in select_champions(RELEASE, "track-b").values():
        bundle = champion_bundle(champ)
        assert causal_counts_agree(
            champ.genome, bundle.train.X[:64], champ.weights, champ.settle
        )


@requires_release
def test_the_gallery_never_reads_the_sealed_test_split():
    """Poison the test split; every array the gallery computes is unchanged."""
    for champ in select_champions(RELEASE, "track-b").values():
        clean = make_bundle(champ.task, seed=champ.dataset_seed)
        poisoned = make_bundle(champ.task, seed=champ.dataset_seed)
        poisoned.test.X[:] = np.nan
        poisoned.test.y[:] = np.nan

        a = boundary_field(champ, clean, resolution=48)
        b = boundary_field(champ, poisoned, resolution=48)
        assert np.array_equal(a["prob"], b["prob"])
        assert np.array_equal(a["xx"], b["xx"])
        assert np.array_equal(a["train_X"], b["train_X"])
        assert np.isfinite(a["prob"]).all()

        assert topology_layout(champ, clean) == topology_layout(champ, poisoned)


@requires_release
def test_the_drawn_structure_matches_the_release_metrics():
    """A drawing that disagreed with the committed record would be a lie."""
    from bpneat.analysis import load_runs

    by_id = {r["run_id"]: r for r in load_runs(RELEASE / "track-b")}
    for champ in select_champions(RELEASE, "track-b").values():
        rec = by_id[champ.run_id]
        lay = topology_layout(champ, champion_bundle(champ))
        assert lay["causal_hidden"] == rec["metrics"]["causal_hidden_nodes"]
        assert lay["causal_connections"] == rec["metrics"]["causal_connections"]
        assert lay["represented_connections"] == rec["metrics"]["represented_connections"]
        assert (
            lay["represented_hidden"] == rec["metrics"]["represented_nodes"] - 4
        )
        drawn = [n for n in lay["nodes"] if not n["structural"]]
        assert sum(1 for n in drawn if n["causal"]) == lay["causal_hidden"]


@requires_release
def test_every_causal_connection_is_an_enabled_one():
    for champ in select_champions(RELEASE, "track-b").values():
        bundle = champion_bundle(champ)
        conns, _ = causal_elements(
            champ.genome, bundle.train.X[:64], champ.weights, champ.settle
        )
        assert all(champ.genome.active[ci] for ci in conns)


def test_the_gallery_declines_an_empty_release(tmp_path):
    from bpneat.champions import build_all

    (tmp_path / "track-b" / "raw" / "runs").mkdir(parents=True)
    assert build_all(tmp_path, progress=lambda *_: None) == []
    assert main(["champions", "--dir", str(tmp_path)]) == 1


@requires_release
def test_the_release_figures_render_from_committed_data_only(tmp_path):
    """Render into a scratch directory: the release is not written to."""
    from bpneat.champions import build_all

    work = tmp_path / "release"
    (work / "track-b").mkdir(parents=True)
    for name in ("raw", "final-test.json"):
        src = RELEASE / "track-b" / name
        if src.is_dir():
            shutil.copytree(src, work / "track-b" / name)
        else:
            shutil.copy2(src, work / "track-b" / name)

    written = build_all(work, progress=lambda *_: None)
    assert {p.name for p in written} == {
        "champion-boundaries.png",
        "champion-topologies.png",
        "paired-test-loss-track-b.png",
    }
    assert all(p.stat().st_size > 10_000 for p in written)
