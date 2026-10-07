"""Real tabular datasets, split the way the synthetic ones are.

Four datasets, vendored as CSV by ``bench/vendor_datasets.py`` and committed, so
nothing here reaches the network and nothing depends on a library's version to
decide what the data is. They span the two axes the 2-D geometries never did:
four to sixty-four inputs, and two to ten classes.

======================  =======  ========  =======
dataset                 samples  features  classes
======================  =======  ========  =======
``iris``                    150         4        3
``wine``                    178        13        3
``breast_cancer``           569        30        2
``digits``                 1797        64       10
======================  =======  ========  =======

**Splitting.** Three disjoint partitions drawn by one seed, stratified by class,
with ``test`` sealed exactly as it is for the synthetic tasks: no code path
reachable from search or model selection may read it. A real dataset is finite,
so unlike a generator its three partitions are a *partition* rather than three
independent draws, and the seed decides which rows land where.

**Standardisation.** Features are centred and scaled by the **training split's**
own mean and standard deviation, and the validation and test splits are
transformed by those same numbers. Fitting the scaler on all three would leak
the test split into training through the mean, which is the quiet version of the
mistake this repository exists to avoid. Standardisation is not optional here:
the frozen inner learner's step size, weight clip and regularisation constant
were set for inputs on a ``[-5, 5]`` square, and a feature measured in hundreds
would otherwise be a change of learner rather than a change of dataset.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "tabular"

TASKS = ("iris", "wine", "breast_cancer", "digits")

#: How the finite rows are divided. Validation is what every search selects on;
#: test is opened once per release by a final-test command and never here.
#:
#: Three tenths to the sealed test rather than two, which is more than a split
#: convention: a finite test split resolves accuracy only to one row in its
#: size, and a protocol that wants to claim two arms are *equivalent* within
#: some margin cannot state a margin finer than that. Thirty rows resolve 0.033;
#: forty-five resolve 0.022. The training split pays for it, and on datasets
#: this size — the widest has thirty features — it can afford to.
TRAIN_FRACTION = 0.5
VALIDATION_FRACTION = 0.2
#: The remainder is the sealed test split.

#: A floor on how small a stratified partition may get before the split stops
#: meaning anything. iris has 50 rows in its smallest class, so 20% of it is 10.
MIN_ROWS_PER_CLASS = 3


@dataclass(frozen=True)
class Split:
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class TabularBundle:
    """Three disjoint partitions of one real dataset.

    ``test`` is sealed: no code path reachable from search or model selection
    may read it. Only a one-shot final-test command may.
    """

    task: str
    source: str
    seed: int
    n_features: int
    n_classes: int
    train: Split
    validation: Split
    test: Split
    #: Which rows of the source file went where. A finite dataset's split is a
    #: partition of specific rows, so a release that records only the shapes has
    #: not recorded the split; and two rows of a real dataset can be identical —
    #: iris has duplicates — so disjointness is a statement about indices and
    #: cannot be checked on values.
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray
    #: The training split's own mean and scale, kept so a release can show that
    #: the transform was fitted where it says it was.
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    #: Features with no variation in training. They carry no information, are
    #: left centred at zero rather than dropped so the input width stays what
    #: the manifest says, and are named so nothing mistakes them for data.
    constant_features: tuple[int, ...]


@cache
def _raw(task: str) -> tuple[np.ndarray, np.ndarray, str]:
    if task not in TASKS:
        raise ValueError(f"unknown dataset {task!r}; have {TASKS}")
    path = DATA_DIR / f"{task}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is committed to this repository; run "
            "`make vendor-data` only if you are re-creating it from source."
        )
    rows, labels = [], []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            rows.append([float(v) for v in row[:-1]])
            labels.append(int(row[-1]))
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    entry = next(d for d in manifest["datasets"] if d["name"] == task)
    if (X.shape[0], X.shape[1]) != (entry["n_samples"], entry["n_features"]):
        raise ValueError(f"{task}: file shape disagrees with the manifest")
    return X, y, entry["origin"]


def _stratified_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    """Three disjoint index sets, class by class, so every split sees every class."""
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * TRAIN_FRACTION))
        n_validation = int(round(n * VALIDATION_FRACTION))
        n_test = n - n_train - n_validation
        if min(n_train, n_validation, n_test) < MIN_ROWS_PER_CLASS:
            raise ValueError(
                f"class {label} has {n} rows, too few to split three ways with "
                f"at least {MIN_ROWS_PER_CLASS} in each"
            )
        train.append(idx[:n_train])
        validation.append(idx[n_train : n_train + n_validation])
        test.append(idx[n_train + n_validation :])
    # Sorted so the row order inside a split does not depend on class order.
    return tuple(np.sort(np.concatenate(part)) for part in (train, validation, test))


def make_bundle(task: str, seed: int) -> TabularBundle:
    """One dataset, split three ways and standardised on the training split."""
    X, y, origin = _raw(task)
    tr, va, te = _stratified_indices(y, seed)

    mean = X[tr].mean(axis=0)
    raw_scale = X[tr].std(axis=0)
    constant = np.flatnonzero(raw_scale <= 1e-12)
    scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)

    def split(idx: np.ndarray) -> Split:
        return Split(np.ascontiguousarray((X[idx] - mean) / scale), y[idx].copy())

    n_classes = int(len(np.unique(y)))
    return TabularBundle(
        task=task,
        source=origin,
        seed=seed,
        n_features=int(X.shape[1]),
        n_classes=n_classes,
        train=split(tr),
        validation=split(va),
        test=split(te),
        train_index=tr,
        validation_index=va,
        test_index=te,
        feature_mean=mean,
        feature_scale=scale,
        constant_features=tuple(int(i) for i in constant),
    )


def describe(task: str) -> dict:
    """Shape and provenance, without drawing a split."""
    X, y, origin = _raw(task)
    counts = np.bincount(y)
    return {
        "task": task,
        "origin": origin,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(counts)),
        "class_counts": counts.tolist(),
        "majority_class_rate": float(counts.max() / counts.sum()),
    }
