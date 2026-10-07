"""The frozen v8 contract.

**Scope.** Three real tabular datasets, with a fourth declared as an extension
because it costs about as much as the other three together. The battery spans
the two axes every earlier protocol held fixed at two and one: four to thirty
features, and two or three classes confirmatory, sixty-four features and ten
classes in the extension.

**Four arms, one budget.** 2,100 candidate evaluations — a population of 100 for
twenty generations, the budget v2 through v6 all anchor on. The three
non-evolutionary arms are matched to the gradient budget the search *actually
spent* in the same cell, so the comparison is paired on compute rather than on
candidate count.

**The linear arm is the point.** It is the seed genome: bias and every input
wired straight to the outputs, no hidden units. On these datasets it is within
two to five points of everything else, which makes "does architecture search
beat having no architecture" the question the data can actually answer.

**Granularity, stated in advance, and the split chosen for it.** A finite test
split resolves accuracy only to one row in its size, so no claim finer than that
is available. Three tenths of the rows go to the sealed test for this reason:
iris then resolves 0.022 rather than 0.033, and the equivalence margin below —
0.03, between one and two rows of the smallest sealed test — is a margin the
data can actually carry. It was not chosen first and justified afterwards; a
gate in :func:`validate` refuses a margin finer than the smallest dataset can
resolve, and it refused 0.03 at the previous split before the split changed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nd.datasets import TASKS as ND_TASKS
from ..v3.conditions import MLP_HIDDEN, POPULATION, RESTARTS
from ..v6.protocol import BURNED_SEEDS as V6_BURNED
from ..v6.protocol import REPLICATES as V6_REPLICATES

PROTOCOL_VERSION = "v8"

#: The three datasets the hypotheses are scored on.
ALL_TASKS = ("iris", "wine", "breast_cancer")
#: Declared extension. One search at the reference budget costs 58 minutes on
#: it against 1.6 to 5.3 on the others, so it is named separately rather than
#: discovered to be unaffordable halfway through. It scores no hypothesis.
EXTENSION_TASKS = ("digits",)
EVERY_TASK = ALL_TASKS + EXTENSION_TASKS

N_REPLICATES = 30

#: The budget every earlier protocol ran at.
REFERENCE_GENERATIONS = 20
REFERENCE_CANDIDATES = POPULATION * (REFERENCE_GENERATIONS + 1)

#: The fixed architecture and restart count every matched control uses.
FIXED_HIDDEN = MLP_HIDDEN
MULTISTART_RESTARTS = RESTARTS

#: Arms, in the order a cell must run them. ``search`` first because the other
#: three inherit the gradient budget it realised in that cell.
ARMS = ("search", "null", "linear", "fixed")
ARM_ORDER = {arm: i for i, arm in enumerate(ARMS)}

ARM_MEANING = {
    "search": "NEAT at the reference configuration, 2,100 candidates",
    "null": "the same candidate count sampled instead of searched: random "
            "architectures under the same inner learner, selection removed",
    "linear": "no architecture at all — the seed genome, bias and every input "
              "wired straight to the outputs, at the search's realised gradient "
              "budget across 60 restarts",
    "fixed": "a fixed 32x32 mixed-operator network at the search's realised "
             "gradient budget across 60 restarts",
}

#: The anchor every contrast is read against.
REFERENCE = "search"
#: Arms that inherit the search's realised gradient budget in the same cell.
MATCHED_TO = {"linear": "search", "fixed": "search"}


@dataclass(frozen=True)
class Replicate:
    """One split of the finite data, plus the seed the search runs under.

    A real dataset is finite, so a replicate is a *different partition of the
    same rows* rather than a fresh draw from a generator. ``split_seed`` decides
    which rows land in train, validation and the sealed test.
    """

    replicate: int
    split_seed: int
    search_seed: int


REPLICATES: tuple[Replicate, ...] = tuple(
    Replicate(i + 1, 110_001 + 13 * i, 120_001 + 13 * i) for i in range(N_REPLICATES)
)

BURNED_SEEDS = frozenset(
    set(V6_BURNED)
    | {r.dataset_seed for r in V6_REPLICATES}
    | {r.search_seed for r in V6_REPLICATES}
    | {9005, 19005}  # v8 pilot
)

#: Accuracy a sealed test split of this size can resolve. Written out because
#: it is the reason the equivalence margin is what it is, and because a claim
#: finer than one test row is not available on the smallest dataset.
TEST_ROWS = {"iris": 45, "wine": 52, "breast_cancer": 172, "digits": 540}

#: Equivalence margin, in accuracy: between one and two rows of the smallest
#: sealed test split, and more than four of the largest confirmatory one.
EQUIVALENCE_DELTA = 0.03

ALPHA = 0.05
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_261_008

# --------------------------------------------------------------------------
# The pre-declared test families. Holm runs inside a family, never across.
# --------------------------------------------------------------------------

#: Every arm against the linear model: is there anything an architecture buys?
FAMILY_VS_LINEAR = tuple(
    (task, arm) for task in ALL_TASKS for arm in ("search", "null", "fixed")
)
#: The search against its candidate-matched null.
FAMILY_VS_NULL = tuple((task, "null") for task in ALL_TASKS)
#: The search against the budget-matched fixed network.
FAMILY_VS_FIXED = tuple((task, "fixed") for task in ALL_TASKS)
#: Equivalence of the search and the linear model.
FAMILY_EQUIVALENCE = tuple((task, "linear") for task in ALL_TASKS)
#: Does the search's champion carry any causally active hidden unit at all?
FAMILY_CHAMPION_SIZE = tuple((task, "search") for task in ALL_TASKS)

FAMILIES = {
    "vs_linear": FAMILY_VS_LINEAR,
    "vs_null": FAMILY_VS_NULL,
    "vs_fixed": FAMILY_VS_FIXED,
    "equivalence_with_linear": FAMILY_EQUIVALENCE,
    "champion_size": FAMILY_CHAMPION_SIZE,
}
FAMILY_SIZE = {name: len(members) for name, members in FAMILIES.items()}


def validate() -> None:
    seeds = [r.split_seed for r in REPLICATES] + [r.search_seed for r in REPLICATES]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("v8 replicate seeds must be unique")
    clash = BURNED_SEEDS & set(seeds)
    if clash:
        raise AssertionError(f"v8 reuses burned seeds: {sorted(clash)}")
    if len(REPLICATES) != N_REPLICATES:
        raise AssertionError("replicate count disagrees with the contract")
    if set(EVERY_TASK) - set(ND_TASKS):
        raise AssertionError(f"unknown datasets: {set(EVERY_TASK) - set(ND_TASKS)}")
    if set(ALL_TASKS) & set(EXTENSION_TASKS):
        raise AssertionError("a dataset cannot be both confirmatory and an extension")
    if REFERENCE not in ARMS:
        raise AssertionError("the anchor arm is not in the plan")
    for matched, target in MATCHED_TO.items():
        if ARM_ORDER[target] >= ARM_ORDER[matched]:
            raise AssertionError(f"{matched} runs before the {target} it is matched to")
    scored = {task for task, _ in FAMILY_VS_LINEAR}
    if scored & set(EXTENSION_TASKS):
        raise AssertionError("the extension dataset may not appear in a scored family")
    if EQUIVALENCE_DELTA < 1.0 / min(TEST_ROWS[t] for t in ALL_TASKS):
        raise AssertionError(
            "the equivalence margin is finer than one row of the smallest sealed "
            "test split, so it claims a resolution the data does not have"
        )


def planned_runs(include_extension: bool = False) -> list[dict]:
    tasks = EVERY_TASK if include_extension else ALL_TASKS
    return [
        {
            "task": task,
            "condition": arm,
            "replicate": rep.replicate,
            "split_seed": rep.split_seed,
            "search_seed": rep.search_seed,
        }
        for rep in REPLICATES
        for task in tasks
        for arm in ARMS
    ]


def cells(include_extension: bool = False) -> list[tuple[str, int]]:
    """(dataset, replicate) units, replicate-major.

    Replicate-major so that a run cut short holds complete replicates on every
    dataset rather than two finished datasets and one empty one — a
    complete-replicate prefix is a smaller release of the same design instead of
    a different one.
    """
    tasks = EVERY_TASK if include_extension else ALL_TASKS
    return [(task, rep.replicate) for rep in REPLICATES for task in tasks]


validate()
