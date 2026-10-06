"""The frozen v4 contract.

Committed and tagged ``v4-freeze`` before any confirmatory compute. Anything
that changes scientific output requires a new version and a new result
identity; results are never patched under an unchanged version.

**Fresh test data.** v3's sealed test was opened once, on the splits generated
by v3's dataset seeds. Reusing those splits here would make v4's test numbers a
second look at data already used to confirm a result, so v4 draws thirty
entirely new replicates. The cost is that v4 and v3 are not paired; the benefit
is that v4's headline is an out-of-sample replication rather than a reanalysis.
Every v4 claim is therefore a within-v4 paired contrast, and the comparison to
v3 is qualitative — does the same thing happen — never a pooled statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..v3.protocol import ALL_TASKS as V3_TASKS
from ..v3.protocol import BURNED_SEEDS as V3_BURNED
from ..v3.protocol import REPLICATES as V3_REPLICATES
from ..v3.protocol import SUCCESS_THRESHOLD as V3_SUCCESS_THRESHOLD

PROTOCOL_VERSION = "v4"

#: Identical to v3's task set, so the two releases are read side by side.
ALL_TASKS = V3_TASKS
SUCCESS_THRESHOLD = dict(V3_SUCCESS_THRESHOLD)

N_REPLICATES = 30

#: Every seed any earlier protocol spent, plus v3's confirmatory replicates and
#: v4's own pilots. None may be reused as a v4 confirmatory replicate.
BURNED_SEEDS = frozenset(
    set(V3_BURNED)
    | {r.dataset_seed for r in V3_REPLICATES}
    | {r.search_seed for r in V3_REPLICATES}
    | {9002, 19002}  # v4 pilot
)


@dataclass(frozen=True)
class Replicate:
    replicate: int
    dataset_seed: int
    search_seed: int


REPLICATES: tuple[Replicate, ...] = tuple(
    Replicate(i + 1, 50_001 + 11 * i, 60_001 + 11 * i) for i in range(N_REPLICATES)
)


@dataclass(frozen=True)
class Block:
    name: str
    question: str
    conditions: tuple[str, ...]
    tasks: tuple[str, ...]


#: ``fixed_tanh_ha`` is the shared control cell: it is the unmatched fixed
#: baseline for both algorithms, and it depends on neither, so it is run once
#: and referenced by both blocks E and F.
BLOCKS: tuple[Block, ...] = (
    Block(
        "E",
        "Does v3's budget reversal replicate for Backprop-NEAT on fresh data?",
        (
            "bpneat",
            "fixed_tanh_ha",
            "fixed_tanh_matched_bpneat",
            "fixed_mixed_matched_bpneat",
        ),
        ALL_TASKS,
    ),
    Block(
        "F",
        "Does the same reversal hold for a second, independently derived search?",
        (
            "cgp",
            "fixed_tanh_ha",
            "fixed_tanh_matched_cgp",
            "fixed_mixed_matched_cgp",
            "cgp_random_matched",
        ),
        ALL_TASKS,
    ),
    Block(
        "G",
        "Do the two search algorithms differ from each other?",
        ("bpneat", "cgp"),
        ALL_TASKS,
    ),
)

#: The two reference conditions. Each anchors its own pre-declared test family,
#: Holm-corrected within that family and never pooled across the two.
REFERENCES = ("bpneat", "cgp")

#: Conditions contrasted against each reference, in the pre-declared families.
FAMILY_CONDITIONS: dict[str, tuple[str, ...]] = {
    "bpneat": (
        "cgp",
        "fixed_tanh_ha",
        "fixed_tanh_matched_bpneat",
        "fixed_mixed_matched_bpneat",
        "fixed_tanh_matched_cgp",
        "fixed_mixed_matched_cgp",
        "cgp_random_matched",
    ),
    "cgp": (
        "bpneat",
        "fixed_tanh_ha",
        "fixed_tanh_matched_cgp",
        "fixed_mixed_matched_cgp",
        "cgp_random_matched",
    ),
}

FAMILY_SIZE = {ref: len(conds) * len(ALL_TASKS) for ref, conds in FAMILY_CONDITIONS.items()}

#: The conditions whose arms must wait for a reference's realized budget.
MATCHED_TO = {
    "fixed_tanh_matched_bpneat": "bpneat",
    "fixed_mixed_matched_bpneat": "bpneat",
    "fixed_tanh_matched_cgp": "cgp",
    "fixed_mixed_matched_cgp": "cgp",
}

#: Replicates of the *v3* plan re-run under v4 code as a reproducibility gate.
#: Validation only — the bridge never touches any test split, and it is excluded
#: from every v4 table and claim.
BRIDGE_CONDITION = "backprop_neat"
BRIDGE_REPLICATES: tuple = tuple(V3_REPLICATES[:10])


def validate() -> None:
    seeds = [r.dataset_seed for r in REPLICATES] + [r.search_seed for r in REPLICATES]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("v4 replicate seeds must be unique")
    clash = BURNED_SEEDS & set(seeds)
    if clash:
        raise AssertionError(f"v4 reuses burned seeds: {sorted(clash)}")
    if len(REPLICATES) != N_REPLICATES:
        raise AssertionError("replicate count disagrees with the contract")
    known = {c for b in BLOCKS for c in b.conditions}
    for ref, conds in FAMILY_CONDITIONS.items():
        if ref not in known:
            raise AssertionError(f"family reference {ref!r} is not a planned condition")
        for c in conds:
            if c not in known:
                raise AssertionError(f"family of {ref!r} names unplanned condition {c!r}")
        if ref in conds:
            raise AssertionError(f"family of {ref!r} contrasts it against itself")
    for target in MATCHED_TO.values():
        if target not in REFERENCES:
            raise AssertionError(f"matched arm targets non-reference {target!r}")
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
    """(task, replicate) units. A cell is the shard unit: the matched arms need
    both references to have run in that same cell first."""
    out: list[tuple[str, int]] = []
    seen = set()
    for p in planned_runs():
        key = (p["task"], p["replicate"])
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def bridge_runs() -> list[dict]:
    """The v3 cells the bridge re-runs, under v3's own seeds."""
    return [
        {
            "task": task,
            "condition": BRIDGE_CONDITION,
            "replicate": rep.replicate,
            "dataset_seed": rep.dataset_seed,
            "search_seed": rep.search_seed,
        }
        for task in ALL_TASKS
        for rep in BRIDGE_REPLICATES
    ]


validate()
