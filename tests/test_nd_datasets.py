"""The vendored real datasets, and the split that must not leak.

Two kinds of gate. The first is about the data being what the repository says it
is: the committed CSVs must match the checksums recorded when they were
extracted, so a dataset a release was built on cannot change underneath it. The
second is about the split, and is the same firewall the synthetic tasks have —
a finite dataset makes leakage easier, not harder, because the three partitions
share rows' provenance rather than being three independent draws.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.nd import datasets as nd_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tabular"
ND_DIR = ROOT / "src" / "bpneat" / "nd"


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DATA_DIR / "manifest.json").read_text())


# --------------------------------------------------------------------------
# The data is what the repository says it is
# --------------------------------------------------------------------------


def test_every_vendored_file_matches_its_recorded_checksum(manifest):
    """A dataset a release was built on must not change underneath it."""
    for entry in manifest["datasets"]:
        path = ROOT / entry["file"]
        assert path.exists(), entry["file"]
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == entry["sha256"], f"{entry['name']}: {got} != {entry['sha256']}"


def test_the_manifest_covers_exactly_the_datasets_the_loader_offers(manifest):
    assert {d["name"] for d in manifest["datasets"]} == set(nd_data.TASKS)


def test_every_dataset_records_where_it_really_came_from(manifest):
    """scikit-learn is the courier; the manifest must name the source."""
    for entry in manifest["datasets"]:
        assert "UCI" in entry["origin"], entry["name"]
        assert entry["url"].startswith("https://archive.ics.uci.edu/"), entry["name"]
        assert entry["note"].strip()
    assert "not used for anything else" in manifest["courier"]


def test_the_battery_spans_the_two_axes_the_geometries_never_did(manifest):
    widths = sorted(d["n_features"] for d in manifest["datasets"])
    classes = sorted(d["n_classes"] for d in manifest["datasets"])
    assert widths[0] >= 3 and widths[-1] >= 32, widths
    assert classes[0] == 2 and classes[-1] >= 5, classes
    # Every one of them is wider than the two inputs every earlier protocol had.
    assert all(w > 2 for w in widths)


def test_the_loader_reads_the_shape_the_manifest_records(manifest):
    for entry in manifest["datasets"]:
        got = nd_data.describe(entry["name"])
        assert got["n_samples"] == entry["n_samples"]
        assert got["n_features"] == entry["n_features"]
        assert got["n_classes"] == entry["n_classes"]
        assert got["class_counts"] == entry["class_counts"]


def test_scikit_learn_is_not_imported_by_anything_that_runs():
    """It extracted the data once. No module under src/ may depend on it."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] == "sklearn" for n in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"sklearn imported by {sorted(set(offenders))}"


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", nd_data.TASKS)
def test_the_three_splits_partition_the_dataset(task):
    """Checked on indices, not on values.

    Two rows of a real dataset can be identical — iris contains duplicates —
    so "no row appears twice" is false of the *data* while the split is still a
    perfect partition. Checking values here would have failed on iris and the
    obvious repair, excusing iris, would have thrown away the only check that
    the split is disjoint at all.
    """
    b = nd_data.make_bundle(task, seed=11)
    n = nd_data.describe(task)["n_samples"]
    parts = [b.train_index, b.validation_index, b.test_index]
    assert [len(p) for p in parts] == [len(b.train), len(b.validation), len(b.test)]
    every = np.concatenate(parts)
    assert len(every) == n
    assert sorted(every.tolist()) == list(range(n)), f"{task}: not a partition"


@pytest.mark.parametrize("task", nd_data.TASKS)
def test_every_class_appears_in_every_split(task):
    b = nd_data.make_bundle(task, seed=12)
    classes = set(range(b.n_classes))
    for split in (b.train, b.validation, b.test):
        assert set(np.unique(split.y).tolist()) == classes, task


@pytest.mark.parametrize("task", nd_data.TASKS)
def test_the_scaler_is_fitted_on_the_training_split_alone(task):
    """Fitting it on all three would leak the sealed split through the mean."""
    b = nd_data.make_bundle(task, seed=13)
    assert np.allclose(b.train.X.mean(axis=0), 0.0, atol=1e-10)
    # Every feature that varies is scaled to unit deviation. The ones that do
    # not vary are named on the bundle rather than excused by dataset: digits
    # has always-dark pixels, and a gate that said "or task == digits" would go
    # on passing if a different dataset grew a constant column.
    varies = np.setdiff1d(np.arange(b.n_features), np.asarray(b.constant_features, int))
    assert np.allclose(b.train.X[:, varies].std(axis=0), 1.0, atol=1e-10)
    for i in b.constant_features:
        assert np.all(b.train.X[:, i] == 0.0), f"{task}: feature {i} is not constant"
    # And the other two must *not* be centred, or the fit saw them.
    off = max(
        float(np.abs(b.validation.X.mean(axis=0)).max()),
        float(np.abs(b.test.X.mean(axis=0)).max()),
    )
    assert off > 1e-6, f"{task}: held-out splits are centred, so the scaler saw them"


@pytest.mark.parametrize("task", nd_data.TASKS)
def test_a_split_is_decided_by_its_seed_and_by_nothing_else(task):
    a = nd_data.make_bundle(task, seed=21)
    b = nd_data.make_bundle(task, seed=21)
    c = nd_data.make_bundle(task, seed=22)
    assert np.array_equal(a.train_index, b.train_index)
    assert np.array_equal(a.train.X, b.train.X) and np.array_equal(a.train.y, b.train.y)
    assert np.array_equal(a.test_index, b.test_index)
    # Compared on indices, not on labels: the split is stratified and the rows
    # are kept in file order, so the *label vector* of a split is the same for
    # every seed by construction. A label comparison here looks like a test and
    # is one only by accident on datasets whose file is not sorted by class.
    assert not np.array_equal(a.train_index, c.train_index), f"{task}: seed did nothing"
    assert not np.array_equal(a.test_index, c.test_index), f"{task}: seed did nothing"


@pytest.mark.parametrize("task", nd_data.TASKS)
def test_the_sealed_split_does_not_reach_the_training_split(task):
    """Poison the test rows; everything a search can see must be unchanged."""
    clean = nd_data.make_bundle(task, seed=31)
    poisoned = nd_data.make_bundle(task, seed=31)
    poisoned.test.X[:] = np.nan
    assert np.array_equal(clean.train.X, poisoned.train.X)
    assert np.array_equal(clean.validation.X, poisoned.validation.X)
    assert np.all(np.isfinite(clean.train.X))


def test_no_module_in_the_nd_core_reads_the_sealed_split():
    """The same firewall the synthetic protocols have, checked statically.

    A module that needs ``.test`` is a final-test module, and there is not one
    here yet. When there is, it is named in this gate rather than exempted by
    being forgotten.
    """
    allowed: set[str] = set()
    offenders = []
    for path in sorted(ND_DIR.glob("*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "test":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"sealed split read at {offenders}"


@pytest.mark.parametrize("task", nd_data.TASKS)
def test_a_constant_feature_is_reported_rather_than_silently_passed_through(task):
    b = nd_data.make_bundle(task, seed=41)
    raw_X, _, _ = nd_data._raw(task)
    for i in b.constant_features:
        assert raw_X[b.train_index, i].std() <= 1e-12, f"{task}: feature {i} varies"
    assert (len(b.constant_features) > 0) == (task == "digits"), (
        f"{task} reports {len(b.constant_features)} constant features; if that "
        "changed, the battery changed"
    )


def test_the_loader_refuses_a_dataset_it_does_not_have():
    with pytest.raises(ValueError, match="unknown dataset"):
        nd_data.make_bundle("spiral", seed=1)
