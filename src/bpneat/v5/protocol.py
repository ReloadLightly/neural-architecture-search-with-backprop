"""The frozen v5 contract.

Committed and tagged ``v5-freeze`` before any confirmatory compute.

**Scope.** Three geometries, not five. v4's two easy geometries saturated above
0.97 for every condition, which contributed nothing except to drag two
hypotheses below their task thresholds; v5 drops them and says so in advance
rather than discovering it again. The three retained geometries are the ones on
which v3 and v4 separated conditions at all.

**Fresh test data,** as in v4 and for the same reason: v5's sealed test must not
be a second look at splits an earlier release already spent.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..v3.protocol import SUCCESS_THRESHOLD as V3_SUCCESS_THRESHOLD
from ..v4.protocol import BURNED_SEEDS as V4_BURNED
from ..v4.protocol import REPLICATES as V4_REPLICATES

PROTOCOL_VERSION = "v5"

#: The geometries on which conditions separate. XOR and circles are excluded by
#: the contract, not by a later choice.
ALL_TASKS = ("spiral", "checkerboard", "spiral3")
SUCCESS_THRESHOLD = {t: V3_SUCCESS_THRESHOLD[t] for t in ALL_TASKS}

N_REPLICATES = 30

BURNED_SEEDS = frozenset(
    set(V4_BURNED)
    | {r.dataset_seed for r in V4_REPLICATES}
    | {r.search_seed for r in V4_REPLICATES}
    | {9003, 19003}  # v5 pilot
)


@dataclass(frozen=True)
class Replicate:
    replicate: int
    dataset_seed: int
    search_seed: int


REPLICATES: tuple[Replicate, ...] = tuple(
    Replicate(i + 1, 70_001 + 13 * i, 80_001 + 13 * i) for i in range(N_REPLICATES)
)

#: The reference condition every contrast is anchored on.
REFERENCE = "neat_reference"

#: Conditions whose fixed-architecture arm inherits the reference's realized
#: gradient budget in the same cell.
MATCHED_TO = {"fixed_mixed_matched": REFERENCE}

#: Pre-declared family for Holm correction: every condition against the
#: reference, on every geometry.
FAMILY_CONDITIONS: tuple[str, ...] = (
    "neat_complexify",
    "neat_no_penalty",
    "neat_complexify_no_penalty",
    "neat_no_speciation",
    "neat_no_crossover",
    "neat_deep_narrow",
    "fixed_mixed_matched",
)
FAMILY_SIZE = len(FAMILY_CONDITIONS) * len(ALL_TASKS)

ALL_CONDITIONS: tuple[str, ...] = (REFERENCE, *FAMILY_CONDITIONS)


def validate() -> None:
    seeds = [r.dataset_seed for r in REPLICATES] + [r.search_seed for r in REPLICATES]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("v5 replicate seeds must be unique")
    clash = BURNED_SEEDS & set(seeds)
    if clash:
        raise AssertionError(f"v5 reuses burned seeds: {sorted(clash)}")
    if len(REPLICATES) != N_REPLICATES:
        raise AssertionError("replicate count disagrees with the contract")
    if REFERENCE in FAMILY_CONDITIONS:
        raise AssertionError("the reference may not be contrasted against itself")
    for target in MATCHED_TO.values():
        if target != REFERENCE:
            raise AssertionError(f"matched arm targets non-reference {target!r}")


def planned_runs() -> list[dict]:
    return [
        {
            "task": task,
            "condition": cond,
            "replicate": rep.replicate,
            "dataset_seed": rep.dataset_seed,
            "search_seed": rep.search_seed,
        }
        for task in ALL_TASKS
        for cond in ALL_CONDITIONS
        for rep in REPLICATES
    ]


def cells() -> list[tuple[str, int]]:
    """(task, replicate) units. The matched arm needs the reference first."""
    return [(task, rep.replicate) for task in ALL_TASKS for rep in REPLICATES]


validate()
