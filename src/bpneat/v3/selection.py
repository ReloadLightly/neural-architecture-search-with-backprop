"""Selection operators, and one scale on which to compare them.

v1 and v2 of this project differed in their selection operator and reached
opposite conclusions about Ha's propagation rule, so v3 treats selection
pressure as a measured factor rather than a setting. The operators below span
the range from Ha's own near-neutral roulette to the truncation-plus-elitism
rule that invalidated v1.

Comparing them needs a common axis. Breeding odds of the best individual over
the median are undefined for truncation (the median may have zero probability),
so this module reports the **standardised selection intensity**

    I = (sum_i p_i f_i - mean f) / sd f

the fitness advantage of the expected parent over the population mean, in
population standard deviations. It is finite and comparable for every operator
here, and it is the quantity the dose-response in Block C is plotted against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evolve import Individual

#: Ha's slack in ``pickRandomIndex``; weight is ``1/(-fitness + slack)``.
HA_SLACK = 0.01


@dataclass(frozen=True)
class Selector:
    """A parent-sampling rule plus its pick-probability model."""

    name: str
    kind: str  # "roulette" | "tournament" | "truncation"
    slack: float = HA_SLACK
    k: int = 2
    keep_fraction: float = 0.5
    elitism: int = 0

    def probabilities(self, fitness: np.ndarray) -> np.ndarray:
        """Probability that each member is chosen as a parent, one draw."""
        n = len(fitness)
        if n == 0:
            return np.zeros(0)
        if self.kind == "roulette":
            w = 1.0 / (-fitness + self.slack)
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            total = w.sum()
            return np.full(n, 1.0 / n) if total <= 0 else w / total
        if self.kind == "tournament":
            # Rank 1 is best. P(rank r wins a k-tournament with replacement)
            # = ((n-r+1)^k - (n-r)^k) / n^k.
            order = np.argsort(-fitness, kind="stable")
            ranks = np.empty(n, dtype=np.int64)
            ranks[order] = np.arange(1, n + 1)
            better = n - ranks
            p = ((better + 1.0) ** self.k - better**self.k) / float(n) ** self.k
            return p / p.sum()
        if self.kind == "truncation":
            keep = max(1, int(round(n * self.keep_fraction)))
            order = np.argsort(-fitness, kind="stable")
            p = np.zeros(n)
            p[order[:keep]] = 1.0 / keep
            return p
        raise ValueError(f"unknown selector kind {self.kind!r}")

    def intensity(self, fitness: np.ndarray) -> float:
        """Standardised selection intensity; 0.0 means no pressure."""
        f = np.asarray(fitness, dtype=np.float64)
        f = f[np.isfinite(f)]
        if len(f) < 2:
            return 0.0
        sd = f.std(ddof=1)
        if sd <= 0:
            return 0.0
        p = self.probabilities(f)
        return float((float(p @ f) - f.mean()) / sd)

    def pick(self, members: list[Individual], rng: np.random.Generator) -> Individual:
        f = np.array([m.fitness for m in members], dtype=np.float64)
        finite = f[np.isfinite(f)]
        floor = float(finite.min()) if finite.size else 0.0
        f = np.where(np.isfinite(f), f, floor)
        p = self.probabilities(f)
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            return members[int(rng.integers(0, len(members)))]
        return members[int(rng.choice(len(members), p=p / total))]


#: Block C levels, weakest pressure first. ``roulette_s0.01`` is Ha's own rule;
#: ``v1_truncation`` is the operator that invalidated protocol v1.
SELECTORS: dict[str, Selector] = {
    "roulette_s1.0": Selector("roulette_s1.0", "roulette", slack=1.0),
    "roulette_s0.1": Selector("roulette_s0.1", "roulette", slack=0.1),
    "roulette_s0.01": Selector("roulette_s0.01", "roulette", slack=HA_SLACK),
    "roulette_s0.001": Selector("roulette_s0.001", "roulette", slack=0.001),
    "tournament_k2": Selector("tournament_k2", "tournament", k=2),
    "tournament_k4": Selector("tournament_k4", "tournament", k=4),
    "v1_truncation": Selector("v1_truncation", "truncation", keep_fraction=0.5, elitism=1),
}

DEFAULT_SELECTOR = "roulette_s0.01"
