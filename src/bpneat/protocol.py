"""The frozen confirmatory contract.

Everything here must be committed *before* the confirmatory run and must not
change afterwards. If a fix changes scientific output, the protocol version
changes with it and the result identity changes too — results are not silently
patched under an unchanged version.
"""

from __future__ import annotations

from dataclasses import dataclass

from .conditions import CORE_CONDITIONS, GENERATIONS, SUCCESS_THRESHOLD

PROTOCOL_VERSION = "v1-draft"
"""``-draft`` until Gate 5. The final-test firewall refuses a draft protocol."""

TASKS = ("xor", "circle", "spiral")

# Seeds already spent on pilots or calibration, which may never be reused as
# confirmatory replicates (handoff section 6.4). 7103/17103 are added because
# this workspace used them for the propagation smoke runs.
BURNED_SEEDS = frozenset({1103, 2203, 3301, 4409, 5501, 6607, 7103, 17103})


@dataclass(frozen=True)
class Replicate:
    """One paired draw: both the sampled dataset and the search stochasticity vary."""

    replicate: int
    dataset_seed: int
    search_seed: int


REPLICATES: tuple[Replicate, ...] = (
    Replicate(1, 8101, 18101),
    Replicate(2, 8202, 18202),
    Replicate(3, 8303, 18303),
    Replicate(4, 8404, 18404),
    Replicate(5, 8505, 18505),
    Replicate(6, 8606, 18606),
    Replicate(7, 8707, 18707),
    Replicate(8, 8808, 18808),
    Replicate(9, 8909, 18909),
    Replicate(10, 9010, 19010),
)


@dataclass(frozen=True)
class Track:
    """Track A reconstructs Ha; Track B compares mechanisms. Never pooled."""

    name: str
    propagation: str
    fitness_split: str
    purpose: str


TRACK_A = Track(
    "A", "ha2016", "train",
    "historical reconstruction under Ha's exact propagation and training-loss fitness",
)
TRACK_B = Track(
    "B", "settled", "validation",
    "mechanism comparison under settled propagation and validation-selected fitness",
)
TRACKS = {"A": TRACK_A, "B": TRACK_B}


def validate() -> None:
    """Fail loudly if the frozen contract is internally inconsistent."""
    seeds = [r.dataset_seed for r in REPLICATES] + [r.search_seed for r in REPLICATES]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("replicate seeds must be unique")
    if BURNED_SEEDS & set(seeds):
        raise AssertionError(f"calibration seeds reused: {BURNED_SEEDS & set(seeds)}")
    if len(REPLICATES) != 10:
        raise AssertionError("the contract declares ten paired replicates")
    if set(TASKS) != set(GENERATIONS) or set(TASKS) != set(SUCCESS_THRESHOLD):
        raise AssertionError("task tables disagree")


def planned_runs(track: str, conditions=None, tasks=None, replicates=None) -> list[dict]:
    """The complete run plan. Completeness is checked against this, not against
    whatever happens to be on disk."""
    conditions = tuple(conditions or CORE_CONDITIONS)
    tasks = tuple(tasks or TASKS)
    reps = tuple(replicates or REPLICATES)
    plan = []
    for task in tasks:
        for condition in conditions:
            for rep in reps:
                plan.append(
                    {
                        "track": track,
                        "task": task,
                        "condition": condition,
                        "replicate": rep.replicate,
                        "dataset_seed": rep.dataset_seed,
                        "search_seed": rep.search_seed,
                    }
                )
    return plan


validate()
