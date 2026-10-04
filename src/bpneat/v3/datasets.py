"""v3 task geometries: the three v2 tasks plus two harder ones.

The v2 generators are imported unchanged from the frozen module. The two new
geometries are defined here so that v3 owns their fingerprint.

``checkerboard`` has a disconnected, parity-structured boundary that no single
radial or sinusoidal feature can express. ``spiral3`` is a three-arm spiral:
arm 0 carries class 1 with half the points, arms 1 and 2 carry class 0 with a
quarter each, so the labels stay balanced while the boundary is strictly harder
than the two-arm original.
"""

from __future__ import annotations

import numpy as np

from ..datasets import DatasetBundle, Split
from ..datasets import make_bundle as _v2_bundle
from ..datasets import make_split as _v2_split

V2_TASKS = ("xor", "circle", "spiral")
V3_TASKS = ("checkerboard", "spiral3")
TASKS = V2_TASKS + V3_TASKS

#: Side of one checkerboard cell on the [-5, 5] square (4x4 grid).
CHECKER_CELL = 2.5
#: Turns in the three-arm spiral, against 1.75 in Ha's two-arm original.
SPIRAL3_TURNS = 2.5


def _checkerboard(rng: np.random.Generator, n: int, noise: float) -> Split:
    x = rng.uniform(-5.0, 5.0, n)
    y = rng.uniform(-5.0, 5.0, n)
    # Parity of the cell index decides the class, before noise is added.
    cx = np.floor((x + 5.0) / CHECKER_CELL)
    cy = np.floor((y + 5.0) / CHECKER_CELL)
    label = ((cx + cy) % 2 == 0).astype(np.float64)
    x = x + rng.normal(0.0, noise, n)
    y = y + rng.normal(0.0, noise, n)
    return Split(np.stack([x, y], axis=1), label)


def _spiral3(rng: np.random.Generator, n: int, noise: float) -> Split:
    half = n // 2
    quarter = (n - half) // 2
    counts = (half, quarter, n - half - quarter)
    labels = (1.0, 0.0, 0.0)
    xs, ys, ls = [], [], []
    for k, (count, label) in enumerate(zip(counts, labels)):
        i = np.arange(count, dtype=np.float64)
        r = i / max(count, 1) * 6.0
        t = SPIRAL3_TURNS * i / max(count, 1) * 2.0 * np.pi + k * 2.0 * np.pi / 3.0
        xs.append(r * np.sin(t) + rng.uniform(-1.0, 1.0, count) * noise)
        ys.append(r * np.cos(t) + rng.uniform(-1.0, 1.0, count) * noise)
        ls.append(np.full(count, label))
    X = np.stack([np.concatenate(xs), np.concatenate(ys)], axis=1)
    return Split(X, np.concatenate(ls))


_GENERATORS = {"checkerboard": _checkerboard, "spiral3": _spiral3}


def make_split(task: str, rng: np.random.Generator, n: int, noise: float) -> Split:
    if task in _GENERATORS:
        split = _GENERATORS[task](rng, n, noise)
        order = rng.permutation(len(split))
        return Split(split.X[order], split.y[order])
    return _v2_split(task, rng, n, noise)


def make_bundle(
    task: str,
    seed: int,
    n_train: int = 200,
    n_validation: int = 200,
    n_test: int = 200,
    noise: float = 0.5,
) -> DatasetBundle:
    """Three independent draws. v2 tasks delegate to the frozen generator."""
    if task in V2_TASKS:
        return _v2_bundle(task, seed, n_train, n_validation, n_test, noise)
    if task not in _GENERATORS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    rng = np.random.default_rng(seed)
    return DatasetBundle(
        task=task,
        generator="v3",
        noise=noise,
        seed=seed,
        train=make_split(task, rng, n_train, noise),
        validation=make_split(task, rng, n_validation, noise),
        test=make_split(task, rng, n_test, noise),
    )
