"""Exact checkpoint and resume.

A resumed run must produce bitwise-identical scientific output to an
uninterrupted one, which means the generator state, the innovation registry and
the counters travel with the population — not just the genomes. The test suite
asserts the equivalence rather than assuming it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .evolve import Individual, InnovationRegistry
from .record import deserialise_genome, serialise_genome, write_json_atomic
import json


def dump_individual(ind: Individual) -> dict:
    d = serialise_genome(ind.genome, ind.weights)
    d.update(
        fitness=float(ind.fitness) if np.isfinite(ind.fitness) else None,
        error=float(ind.error) if np.isfinite(ind.error) else None,
        gradient_steps=int(ind.gradient_steps),
        species=int(ind.species),
        evaluated=bool(ind.evaluated),
    )
    return d


def load_individual(d: dict) -> Individual:
    g, w = deserialise_genome(d)
    return Individual(
        genome=g,
        weights=w,
        fitness=-np.inf if d["fitness"] is None else float(d["fitness"]),
        error=np.inf if d["error"] is None else float(d["error"]),
        gradient_steps=int(d["gradient_steps"]),
        species=int(d["species"]),
        evaluated=bool(d["evaluated"]),
    )


def dump_registry(reg: InnovationRegistry) -> dict:
    return {
        "next": reg._next,
        "conn": [[int(a), int(b), int(v)] for (a, b), v in reg._conn.items()],
    }


def load_registry(d: dict) -> InnovationRegistry:
    reg = InnovationRegistry()
    reg._next = int(d["next"])
    reg._conn = {(int(a), int(b)): int(v) for a, b, v in d["conn"]}
    return reg


def save(
    path: Path,
    *,
    generation: int,
    population: list[Individual],
    champion: Individual | None,
    rng: np.random.Generator,
    registry: InnovationRegistry,
    history: list[dict],
    candidates: int,
    gradient_steps: int,
    elapsed: float,
) -> None:
    write_json_atomic(
        Path(path),
        {
            "generation": generation,
            "population": [dump_individual(i) for i in population],
            "champion": dump_individual(champion) if champion is not None else None,
            "rng_state": json.loads(json.dumps(rng.bit_generator.state, default=int)),
            "registry": dump_registry(registry),
            "history": history,
            "candidates": candidates,
            "gradient_steps": gradient_steps,
            "elapsed": elapsed,
        },
    )


def load(path: Path) -> dict:
    with open(path) as fh:
        state = json.load(fh)
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    return {
        "generation": int(state["generation"]),
        "population": [load_individual(d) for d in state["population"]],
        "champion": load_individual(state["champion"]) if state["champion"] else None,
        "rng": rng,
        "registry": load_registry(state["registry"]),
        "history": state["history"],
        "candidates": int(state["candidates"]),
        "gradient_steps": int(state["gradient_steps"]),
        "elapsed": float(state["elapsed"]),
    }
