"""Task geometries.

The ``ha2016`` generators reproduce the geometry in ``hardmaru/backprop-neat-js``
(``dataset.js``) exactly: raw unstandardised coordinates, noise 0.5, a 1.75-turn
spiral on radius 6, and a radius-5 circle. Distributions are matched; the RNG
stream is not (numpy vs. the JS PRNG), so this is a faithful reconstruction of
the task, not a bit-exact replay of any particular sampled dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TASKS = ("xor", "circle", "spiral")


@dataclass(frozen=True)
class Split:
    X: np.ndarray  # (n, 2) float64
    y: np.ndarray  # (n,)   float64 in {0, 1}

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True)
class DatasetBundle:
    """Three independently sampled partitions of the same generator.

    ``test`` is sealed: no code path reachable from search or model selection
    may read it. Only the one-shot final-test command may.
    """

    task: str
    generator: str
    noise: float
    seed: int
    train: Split
    validation: Split
    test: Split


def _xor(rng: np.random.Generator, n: int, noise: float) -> Split:
    # x = randf(-5,5) + randn(0, noise); label 1 iff x and y share a sign.
    x = rng.uniform(-5.0, 5.0, n) + rng.normal(0.0, noise, n)
    y = rng.uniform(-5.0, 5.0, n) + rng.normal(0.0, noise, n)
    label = (((x > 0) & (y > 0)) | ((x < 0) & (y < 0))).astype(np.float64)
    return Split(np.stack([x, y], axis=1), label)


def _circle(rng: np.random.Generator, n: int, noise: float) -> Split:
    radius = 5.0
    half = n // 2

    def arm(r_lo: float, r_hi: float, count: int):
        r = rng.uniform(r_lo, r_hi, count)
        angle = rng.uniform(0.0, 2.0 * np.pi, count)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        # The label is decided on the clean point, then noise is added.
        label = ((x * x + y * y) < (radius * 0.5) ** 2).astype(np.float64)
        x = x + rng.uniform(-radius, radius, count) * noise / 3.0
        y = y + rng.uniform(-radius, radius, count) * noise / 3.0
        return x, y, label

    xi, yi, li = arm(0.0, radius * 0.5, half)
    xo, yo, lo = arm(radius * 0.75, radius, n - half)
    X = np.stack([np.concatenate([xi, xo]), np.concatenate([yi, yo])], axis=1)
    return Split(X, np.concatenate([li, lo]))


def _spiral(rng: np.random.Generator, n: int, noise: float) -> Split:
    half = n // 2

    def arm(delta_t: float, label: float, count: int):
        i = np.arange(count, dtype=np.float64)
        r = i / count * 6.0
        t = 1.75 * i / count * 2.0 * np.pi + delta_t
        x = r * np.sin(t) + rng.uniform(-1.0, 1.0, count) * noise
        y = r * np.cos(t) + rng.uniform(-1.0, 1.0, count) * noise
        return x, y, np.full(count, label)

    x0, y0, l0 = arm(0.0, 0.0, half)
    x1, y1, l1 = arm(np.pi, 1.0, n - half)
    X = np.stack([np.concatenate([x0, x1]), np.concatenate([y0, y1])], axis=1)
    return Split(X, np.concatenate([l0, l1]))


_GENERATORS = {"xor": _xor, "circle": _circle, "spiral": _spiral}


def make_split(task: str, rng: np.random.Generator, n: int, noise: float) -> Split:
    if task not in _GENERATORS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    split = _GENERATORS[task](rng, n, noise)
    order = rng.permutation(len(split))
    return Split(split.X[order], split.y[order])


def make_bundle(
    task: str,
    seed: int,
    n_train: int = 200,
    n_validation: int = 200,
    n_test: int = 200,
    noise: float = 0.5,
    generator: str = "ha2016",
) -> DatasetBundle:
    """Three independent draws from one generator under one dataset seed."""
    if generator != "ha2016":
        raise ValueError(f"unknown generator {generator!r}")
    rng = np.random.default_rng(seed)
    return DatasetBundle(
        task=task,
        generator=generator,
        noise=noise,
        seed=seed,
        train=make_split(task, rng, n_train, noise),
        validation=make_split(task, rng, n_validation, noise),
        test=make_split(task, rng, n_test, noise),
    )
