"""The frozen v6 contract.

Committed before any v6 confirmatory run existed. The freeze record is at the
end of ``docs/v6-preregistration.md``.

**The ladder.** Population is held at 100 — the value every earlier release used
— and the budget is moved by generations alone, so a budget change is one change
and not a width/depth trade as well (v5's ``neat_deep_narrow`` already asked that
question and is not being re-asked here). Five rungs:

===========  ===========  ==========  ===============  =============
label        generations  candidates  x the reference  part of
===========  ===========  ==========  ===============  =============
``b500``     4            500         0.24             confirmatory
``b1000``    9            1000        0.48             confirmatory
``b2100``    20           2100        1.00             confirmatory
``b6300``    62           6300        3.00             confirmatory
``b16800``   167          16800       8.00             extension
===========  ===========  ==========  ===============  =============

``b2100`` is the budget v2, v3, v4 and v5 all ran at, so the third rung of this
ladder is the point those releases already measured — reached again on fresh
splits with fresh seeds, which makes it a replication as well as a rung.

**Four confirmatory rungs and one declared extension.** The search's cost is
superlinear in its generation count — the networks grow, so each later candidate
costs more than an earlier one — and measured on this machine the top rung alone
would cost roughly as much as the rest of the ladder together. Rather than
discover that halfway through and quietly drop it, the contract names the four
rungs the hypotheses are scored on and names ``b16800`` separately: it is run
only if the confirmatory ladder finishes, before the sealed test, and it is
reported as a supplement. **No hypothesis below is scored on it**, and if it is
incomplete when the sealed test is taken it is discarded rather than reported
partially. The confirmatory ladder still spans 12.6x in candidates and about a
decade in gradient steps, which is what the slope hypotheses need.

**The ladder is nested by construction, and that is deliberate.** Within one
cell every ``search`` arm is given the same search seed, and the search's random
stream does not depend on how many generations it will run for. So the champion
of ``search_b500`` *is* the champion of ``search_b16800`` as it stood at
generation four. The same holds for the ``null`` arm, whose running best over a
stream of sampled architectures is a prefix of the longer stream. This makes the
budget contrast paired on the strongest possible terms — the arms differ only in
where they stopped — and it has one consequence that has to be stated, not
discovered: the champion is selected by *validation* fitness, so validation
fitness is non-decreasing in budget **by construction** and carries no evidence
at all. Every hypothesis below is therefore scored on sealed-test accuracy,
which selection never saw and which is free to fall as the budget grows.

**Fresh test data**, as in v4 and v5 and for the same reason: dataset seeds
``90001 + 13i``, with every seed v1-v5 spent added to the burned set.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..v3.conditions import GENERATIONS as V3_GENERATIONS
from ..v3.conditions import POPULATION
from ..v3.protocol import SUCCESS_THRESHOLD as V3_SUCCESS_THRESHOLD
from ..v5.protocol import BURNED_SEEDS as V5_BURNED
from ..v5.protocol import REPLICATES as V5_REPLICATES

PROTOCOL_VERSION = "v6"

#: The geometries on which conditions separate, inherited from v5's contract.
ALL_TASKS = ("spiral", "checkerboard", "spiral3")
SUCCESS_THRESHOLD = {t: V3_SUCCESS_THRESHOLD[t] for t in ALL_TASKS}

N_REPLICATES = 30

#: The budget every earlier release ran at: 100 x (20 + 1).
REFERENCE_GENERATIONS = 20
REFERENCE_CANDIDATES = POPULATION * (REFERENCE_GENERATIONS + 1)


@dataclass(frozen=True)
class Budget:
    """One rung. ``candidates`` is derived, never typed in twice."""

    generations: int

    @property
    def candidates(self) -> int:
        return POPULATION * (self.generations + 1)

    @property
    def label(self) -> str:
        return f"b{self.candidates}"

    @property
    def multiplier(self) -> float:
        return self.candidates / REFERENCE_CANDIDATES


#: Generations per confirmatory rung. 4 and 9 bracket the reference from below;
#: 62 gives exactly 3x its candidate count, so the top of the confirmatory
#: ladder is a round multiple rather than an artefact of rounding generations.
LADDER_GENERATIONS = (4, 9, REFERENCE_GENERATIONS, 62)
#: The declared extension rung: exactly 8x the reference candidate count.
EXTENSION_GENERATIONS = (167,)

BUDGETS: tuple[Budget, ...] = tuple(Budget(g) for g in LADDER_GENERATIONS)
EXTENSION_BUDGETS: tuple[Budget, ...] = tuple(Budget(g) for g in EXTENSION_GENERATIONS)
ALL_BUDGETS: tuple[Budget, ...] = BUDGETS + EXTENSION_BUDGETS
BUDGET_BY_LABEL = {b.label: b for b in ALL_BUDGETS}
REFERENCE_BUDGET = next(b for b in BUDGETS if b.candidates == REFERENCE_CANDIDATES)

#: The three arms, in the order a cell must run them: the fixed arm inherits the
#: gradient budget the search arm spent at the same rung in the same cell.
ARMS = ("search", "null", "fixed")
ARM_ORDER = {arm: i for i, arm in enumerate(ARMS)}

ARM_MEANING = {
    "search": "v5's reference NEAT configuration, run to this rung's generation count",
    "null": "the same candidate count sampled instead of searched: v3's "
            "candidate-matched random architecture search, same inner learner",
    "fixed": "a fixed 32x32 mixed-operator network at the search arm's realised "
             "gradient budget in this same cell, 60 restarts",
}


def condition_name(arm: str, budget: Budget) -> str:
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown arm {arm!r}")
    return f"{arm}_{budget.label}"


def split_condition(condition: str) -> tuple[str, Budget]:
    arm, _, label = condition.partition("_")
    if arm not in ARM_ORDER or label not in BUDGET_BY_LABEL:
        raise ValueError(f"not a v6 condition: {condition!r}")
    return arm, BUDGET_BY_LABEL[label]


#: Budget-major so that a truncated run has complete low rungs rather than one
#: complete arm. ``ALL_CONDITIONS`` is the confirmatory grid; the extension rung
#: is named separately and is never folded into it.
ALL_CONDITIONS: tuple[str, ...] = tuple(
    condition_name(arm, b) for b in BUDGETS for arm in ARMS
)
EXTENSION_CONDITIONS: tuple[str, ...] = tuple(
    condition_name(arm, b) for b in EXTENSION_BUDGETS for arm in ARMS
)
EVERY_CONDITION: tuple[str, ...] = ALL_CONDITIONS + EXTENSION_CONDITIONS

#: The anchor: the arm and rung every earlier release measured.
REFERENCE = condition_name("search", REFERENCE_BUDGET)

#: Each fixed arm inherits the gradient budget of the search arm at its own rung.
MATCHED_TO = {
    condition_name("fixed", b): condition_name("search", b) for b in ALL_BUDGETS
}


@dataclass(frozen=True)
class Replicate:
    replicate: int
    dataset_seed: int
    search_seed: int


REPLICATES: tuple[Replicate, ...] = tuple(
    Replicate(i + 1, 90_001 + 13 * i, 100_001 + 13 * i) for i in range(N_REPLICATES)
)

BURNED_SEEDS = frozenset(
    set(V5_BURNED)
    | {r.dataset_seed for r in V5_REPLICATES}
    | {r.search_seed for r in V5_REPLICATES}
    | {9004, 19004}  # v6 pilot
)

# --------------------------------------------------------------------------
# The pre-declared test families. Holm runs inside a family, never across.
# --------------------------------------------------------------------------
#
# Four families rather than one, because they answer four different questions
# and a single family of 45 would penalise each of them for the others' size.
# Every family is enumerated here, before any run exists, so it cannot be
# widened or narrowed once the numbers are in.

#: Search against its candidate-matched null, at every rung, on every geometry.
FAMILY_SEARCH_VS_NULL = tuple(
    (task, b.label) for task in ALL_TASKS for b in BUDGETS
)

#: Search against the budget-matched fixed network, at every rung.
FAMILY_SEARCH_VS_FIXED = tuple(
    (task, b.label) for task in ALL_TASKS for b in BUDGETS
)

#: The five scaling slopes per geometry. Each is one least-squares slope per
#: replicate, across the rungs, tested against zero by Wilcoxon signed-rank.
#:
#: The first three are each arm's own sealed-test accuracy against the base-10
#: logarithm of the gradient steps it actually spent: "what does a decade of
#: gradient compute buy this arm". Gradient steps rather than candidates,
#: because it is the one axis on which all three arms are commensurable and the
#: axis the fixed arm is matched on.
#:
#: The last two are paired margins, each regressed on the logarithm of the
#: quantity its two arms are *matched* on: candidates for the search against its
#: candidate-matched null, gradient steps for the search against the
#: budget-matched fixed network.
SLOPE_TERMS = ("search", "null", "fixed", "edge_vs_null", "edge_vs_fixed")
SLOPE_AXIS = {
    "search": "gradient_steps",
    "null": "gradient_steps",
    "fixed": "gradient_steps",
    "edge_vs_null": "candidates",
    "edge_vs_fixed": "gradient_steps",
}
FAMILY_SCALING_SLOPES = tuple(
    (task, term) for task in ALL_TASKS for term in SLOPE_TERMS
)

#: Equivalence of search and null at the reference rung only.
FAMILY_EQUIVALENCE = tuple((task, REFERENCE_BUDGET.label) for task in ALL_TASKS)

#: One slope per geometry: the search champion's causally active hidden nodes
#: against log10 candidates. Declared because the pre-freeze cost pilot, on the
#: burned pilot seed and one geometry, found the champion growing from 1 active
#: hidden unit at 500 candidates to 12 at 6300 — which would make v5's "about
#: four active units" a statement about the budget as much as about the fitness.
#: One seed is not evidence; this family is how v6 finds out.
FAMILY_SIZE_SLOPES = tuple((task, "search_size") for task in ALL_TASKS)

FAMILIES = {
    "search_vs_null": FAMILY_SEARCH_VS_NULL,
    "search_vs_fixed": FAMILY_SEARCH_VS_FIXED,
    "scaling_slopes": FAMILY_SCALING_SLOPES,
    "equivalence_at_reference": FAMILY_EQUIVALENCE,
    "size_slopes": FAMILY_SIZE_SLOPES,
}
FAMILY_SIZE = {name: len(members) for name, members in FAMILIES.items()}

#: The equivalence margin, in accuracy, fixed in advance. Two points of accuracy
#: is below the smallest difference any earlier release treated as meaningful and
#: is about a quarter of the spread between the best and worst arm at the
#: reference budget in v5.
EQUIVALENCE_DELTA = 0.02

ALPHA = 0.05

#: Fixed in advance so the interval is not a choice made after the numbers.
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_261_007


def validate() -> None:
    seeds = [r.dataset_seed for r in REPLICATES] + [r.search_seed for r in REPLICATES]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("v6 replicate seeds must be unique")
    clash = BURNED_SEEDS & set(seeds)
    if clash:
        raise AssertionError(f"v6 reuses burned seeds: {sorted(clash)}")
    if len(REPLICATES) != N_REPLICATES:
        raise AssertionError("replicate count disagrees with the contract")
    if len({b.candidates for b in ALL_BUDGETS}) != len(ALL_BUDGETS):
        raise AssertionError("two rungs of the ladder have the same candidate count")
    if list(ALL_BUDGETS) != sorted(ALL_BUDGETS, key=lambda b: b.candidates):
        raise AssertionError("the ladder must be written in ascending order")
    if set(BUDGETS) & set(EXTENSION_BUDGETS):
        raise AssertionError("a rung cannot be both confirmatory and an extension")
    for task in ALL_TASKS:
        if V3_GENERATIONS[task] != REFERENCE_GENERATIONS:
            raise AssertionError(
                f"{task} ran at {V3_GENERATIONS[task]} generations in v3, not "
                f"{REFERENCE_GENERATIONS}; the ladder is no longer task-independent"
            )
    if REFERENCE not in ALL_CONDITIONS:
        raise AssertionError("the anchor condition is not in the plan")
    for matched, target in MATCHED_TO.items():
        if split_condition(matched)[1] is not split_condition(target)[1]:
            raise AssertionError(f"{matched} is matched across rungs, to {target}")
    if len(ALL_CONDITIONS) != len(ARMS) * len(BUDGETS):
        raise AssertionError("the arm x budget grid is incomplete")
    if len(EVERY_CONDITION) != len(ARMS) * len(ALL_BUDGETS):
        raise AssertionError("the extension grid is incomplete")
    scored = {label for _, label in FAMILY_SEARCH_VS_NULL} | {
        label for _, label in FAMILY_SEARCH_VS_FIXED
    }
    if scored & {b.label for b in EXTENSION_BUDGETS}:
        raise AssertionError("the extension rung may not appear in a scored family")


def planned_runs(include_extension: bool = False) -> list[dict]:
    """The confirmatory plan, optionally with the declared extension rung."""
    conditions = EVERY_CONDITION if include_extension else ALL_CONDITIONS
    return [
        {
            "task": task,
            "condition": cond,
            "replicate": rep.replicate,
            "dataset_seed": rep.dataset_seed,
            "search_seed": rep.search_seed,
        }
        for rep in REPLICATES
        for task in ALL_TASKS
        for cond in conditions
    ]


def cells() -> list[tuple[str, int]]:
    """(task, replicate) units, replicate-major.

    Replicate-major on purpose: a run that is cut short then holds *complete
    replicates on all three geometries* rather than two finished geometries and
    one empty one, and a complete-replicate prefix is a smaller release of the
    same design instead of a different one.
    """
    return [(task, rep.replicate) for rep in REPLICATES for task in ALL_TASKS]


validate()
