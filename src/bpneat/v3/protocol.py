"""The frozen v3 contract.

Committed and tagged ``v3-freeze`` before any confirmatory compute. Anything
that changes scientific output requires a new version and a new result
identity; results are never patched under an unchanged version.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_VERSION = "v3"

#: v2 tasks plus the two harder geometries defined in :mod:`bpneat.v3.datasets`.
ALL_TASKS = ("xor", "circle", "spiral", "checkerboard", "spiral3")

#: Seeds spent on v1/v2 confirmatory work, v2 calibration, or v3 piloting.
#: None may be reused as a v3 confirmatory replicate.
BURNED_SEEDS = frozenset(
    {1103, 2203, 3301, 4409, 5501, 6607, 7103, 17103}
    | {8101 + 101 * i for i in range(10)}
    | {18101 + 101 * i for i in range(10)}
    | {8101, 8202, 8303, 8404, 8505, 8606, 8707, 8808, 8909, 9010}
    | {18101, 18202, 18303, 18404, 18505, 18606, 18707, 18808, 18909, 19010}
    | {9001, 19001}  # v3 pilot
)

N_REPLICATES = 30


@dataclass(frozen=True)
class Replicate:
    replicate: int
    dataset_seed: int
    search_seed: int


REPLICATES: tuple[Replicate, ...] = tuple(
    Replicate(i + 1, 30_001 + 7 * i, 40_001 + 7 * i) for i in range(N_REPLICATES)
)


@dataclass(frozen=True)
class Block:
    name: str
    question: str
    conditions: tuple[str, ...]
    tasks: tuple[str, ...]


#: ``backprop_neat`` is the shared reference cell: it is the (settled,
#: validation) corner of block B, the ``roulette_s0.01`` level of block C and
#: the Lamarckian level of block D. It is run once and referenced by all four
#: blocks, so the blocks are analysis views over one run plan.
BLOCKS: tuple[Block, ...] = (
    Block(
        "A",
        "Do v2's conclusions survive controls that are not starved?",
        (
            "backprop_neat",
            "fixed_mlp_tanh_ha",
            "fixed_mlp_tanh_matched",
            "fixed_mlp_sin_matched",
            "fixed_mlp_mixed_matched",
            "random_search_matched",
            "homogeneous_tanh",
            "evolution_only",
        ),
        ALL_TASKS,
    ),
    Block(
        "B",
        "Propagation rule and fitness split, separated.",
        ("backprop_neat", "prop_ha_fit_val", "prop_ha_fit_train", "prop_settled_fit_train"),
        ("xor", "circle", "spiral"),
    ),
    Block(
        "C",
        "How does selection pressure move collapse rate and causal size?",
        (
            "sel_roulette_s1.0",
            "sel_roulette_s0.1",
            "backprop_neat",
            "sel_roulette_s0.001",
            "sel_tournament_k2",
            "sel_tournament_k4",
            "sel_v1_truncation",
        ),
        ("xor", "spiral"),
    ),
    Block(
        "D",
        "Does inheriting learned weights matter?",
        ("backprop_neat", "baldwinian"),
        ("xor", "circle", "spiral"),
    ),
)

#: Pre-declared family for Holm correction: the Block A comparisons of
#: ``backprop_neat`` against every other Block A condition, on every task.
PRIMARY_FAMILY_SIZE = (len(BLOCKS[0].conditions) - 1) * len(ALL_TASKS)

SUCCESS_THRESHOLD = {
    "xor": 0.90,
    "circle": 0.90,
    "spiral": 0.80,
    "checkerboard": 0.80,
    "spiral3": 0.75,
}


def validate() -> None:
    seeds = [r.dataset_seed for r in REPLICATES] + [r.search_seed for r in REPLICATES]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("v3 replicate seeds must be unique")
    clash = BURNED_SEEDS & set(seeds)
    if clash:
        raise AssertionError(f"v3 reuses burned seeds: {sorted(clash)}")
    if len(REPLICATES) != N_REPLICATES:
        raise AssertionError("replicate count disagrees with the contract")
    for b in BLOCKS:
        for t in b.tasks:
            if t not in ALL_TASKS:
                raise AssertionError(f"block {b.name} names unknown task {t}")


def planned_runs() -> list[dict]:
    """The union of distinct (task, condition, replicate) cells across blocks."""
    seen: set[tuple[str, str, int]] = set()
    plan: list[dict] = []
    for block in BLOCKS:
        for task in block.tasks:
            for cond in block.conditions:
                for rep in REPLICATES:
                    key = (task, cond, rep.replicate)
                    if key in seen:
                        continue
                    seen.add(key)
                    plan.append(
                        {
                            "task": task,
                            "condition": cond,
                            "replicate": rep.replicate,
                            "dataset_seed": rep.dataset_seed,
                            "search_seed": rep.search_seed,
                        }
                    )
    return plan


def cells() -> list[tuple[str, int]]:
    """(task, replicate) units. A cell is the shard unit: budget matching needs
    ``backprop_neat`` to run before the matched arms of the same cell."""
    out: list[tuple[str, int]] = []
    seen = set()
    for p in planned_runs():
        key = (p["task"], p["replicate"])
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


validate()
