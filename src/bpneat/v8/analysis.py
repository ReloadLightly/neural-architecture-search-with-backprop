"""Derive every v8 table from the raw records.

The primary outcome is sealed-test accuracy of the validation-selected champion,
paired within (dataset, replicate). A replicate is a *different stratified split
of the same finite rows*, so the pairing is within a split and the thirty
observations are independent of each other — which is what gives the test its
power on datasets whose sealed split is only forty-five rows wide.

Two things about these data shape the analysis and are stated here rather than
discovered later. A finite test split resolves accuracy only to one row in its
size, so paired differences take values in multiples of that and are heavily
tied; the signed-rank test is run with ``zero_method="zsplit"`` for exactly this
reason. And the arms are close — a linear model is within a few points of
everything else — so the interesting result is as likely to be an *equivalence*
as a difference, which is why one family tests for one.

Nothing here retrains a model or reads data; it reads ``raw/runs/*.json`` and
``final-test.json``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from .protocol import (
    ALL_TASKS,
    ALPHA,
    ARMS,
    BOOTSTRAP,
    BOOTSTRAP_SEED,
    EQUIVALENCE_DELTA,
    EVERY_TASK,
    EXTENSION_TASKS,
    FAMILY_SIZE,
    REFERENCE,
    TEST_ROWS,
)

#: Contrasts, by family. Written as (family, left, right): left minus right.
CONTRASTS = {
    "vs_linear": (("search", "linear"), ("null", "linear"), ("fixed", "linear")),
    "vs_null": (("search", "null"),),
    "vs_fixed": (("search", "fixed"),),
}


def load(release_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    runs = [
        json.loads(p.read_text())
        for p in sorted((Path(release_dir) / "raw" / "runs").glob("*.json"))
    ]
    fpath = Path(release_dir) / "final-test.json"
    final = {}
    if fpath.exists():
        final = {r["run_id"]: r for r in json.loads(fpath.read_text())["results"]}
    return runs, final


def _bootstrap_median_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(d) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _holm(rows: list[dict], key: str, family_size: int) -> None:
    """Holm step-down, in place, at the family's *declared* size.

    Declared rather than present: a cell that is missing must not buy the
    surviving cells more power.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, rows[i][key] * (family_size - rank))
        running = max(running, adj)
        rows[i]["holm_p"] = running
        rows[i]["holm_significant"] = bool(running < ALPHA)


def _wilcoxon_p(d: np.ndarray, alternative: str = "two-sided") -> float:
    try:
        return float(wilcoxon(d, zero_method="zsplit", alternative=alternative).pvalue)
    except ValueError:  # every difference is exactly zero
        return 1.0


def _index(runs: list[dict], final: dict[str, dict]):
    out: dict[tuple[str, str, int], dict] = {}
    for r in runs:
        t = final.get(r["run_id"])
        out[(r["task"], r["condition"], r["replicate"])] = {
            "test_accuracy": None if t is None else t["test_accuracy"],
            "test_loss": None if t is None else t["test_loss"],
            "validation_accuracy": r["metrics"]["validation_accuracy"],
            "causal_hidden_nodes": float(r["metrics"]["causal_hidden_nodes"]),
            "gradient_steps": float(r["compute"]["gradient_steps"]),
        }
    return out


def _replicates(runs: list[dict], task: str) -> list[int]:
    return sorted({r["replicate"] for r in runs if r["task"] == task})


def _tasks_present(runs: list[dict]) -> list[str]:
    seen = {r["task"] for r in runs}
    return [t for t in EVERY_TASK if t in seen]


# --------------------------------------------------------------------------


def summarise(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["condition"])].append(r)

    rows = []
    for (task, arm), items in sorted(groups.items()):
        first = items[0]
        va = np.array([i["metrics"]["validation_accuracy"] for i in items])
        causal = np.array([i["metrics"]["causal_hidden_nodes"] for i in items])
        steps = np.array([i["compute"]["gradient_steps"] for i in items], dtype=float)
        wall = np.array([i["compute"]["wall_time_seconds"] for i in items])

        row = {
            "task": task,
            "arm": arm,
            "kind": "extension" if task in EXTENSION_TASKS else "confirmatory",
            "n_features": first["config"]["n_features"],
            "n_classes": first["config"]["n_classes"],
            "test_rows": first["config"]["test_rows"],
            "n": len(items),
            "validation_accuracy_mean": float(va.mean()),
            "causal_hidden_nodes_mean": float(causal.mean()),
            "causal_hidden_nodes_median": float(np.median(causal)),
            "zero_hidden_rate": float(np.mean(causal == 0)),
            "gradient_steps_mean": float(steps.mean()),
            "wall_time_mean": float(wall.mean()),
            "wall_time_total_hours": float(wall.sum() / 3600.0),
        }
        tests = [final[i["run_id"]] for i in items if i["run_id"] in final]
        if tests:
            ta = np.array([t["test_accuracy"] for t in tests])
            tl = np.array([t["test_loss"] for t in tests])
            row.update(
                test_accuracy_mean=float(ta.mean()),
                test_accuracy_median=float(np.median(ta)),
                test_accuracy_sd=float(ta.std(ddof=1)) if len(ta) > 1 else 0.0,
                test_loss_mean=float(tl.mean()),
            )
        rows.append(row)
    return rows


def arm_contrasts(
    runs: list[dict], final: dict[str, dict], metric: str = "test_accuracy"
) -> list[dict]:
    """Every declared pairwise contrast, paired within (dataset, replicate)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = _index(runs, final)
    scored = metric == "test_accuracy"
    by_family: dict[str, list[dict]] = {name: [] for name in CONTRASTS}

    for family, pairs in CONTRASTS.items():
        for task in _tasks_present(runs):
            for left, right in pairs:
                diffs, per_rep = [], []
                for rep in _replicates(runs, task):
                    a, b = idx.get((task, left, rep)), idx.get((task, right, rep))
                    if a is None or b is None:
                        continue
                    if a[metric] is None or b[metric] is None:
                        continue
                    diffs.append(a[metric] - b[metric])
                    per_rep.append({"replicate": rep, left: a[metric], right: b[metric]})
                if len(diffs) < 2:
                    continue
                d = np.array(diffs, dtype=float)
                lo, hi = _bootstrap_median_ci(d, rng)
                by_family[family].append(
                    {
                        "family": family,
                        "task": task,
                        "kind": "extension" if task in EXTENSION_TASKS
                        else "confirmatory",
                        "metric": metric,
                        "scored": scored and task in ALL_TASKS,
                        "comparison": f"{left} - {right}",
                        "left": left,
                        "right": right,
                        "n_pairs": len(d),
                        "mean_difference": float(d.mean()),
                        "median_difference": float(np.median(d)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "wilcoxon_p": _wilcoxon_p(d),
                        "left_wins": int(np.sum(d > 0)),
                        "ties": int(np.sum(d == 0)),
                        "per_replicate": json.dumps(per_rep),
                    }
                )

    rows: list[dict] = []
    for family, members in by_family.items():
        scored_members = [m for m in members if m["scored"]]
        if scored_members and scored:
            _holm(scored_members, "wilcoxon_p", FAMILY_SIZE[family])
        rows.extend(members)
    return rows


def _contrast(rows, family: str, task: str, left: str, right: str) -> dict | None:
    return next(
        (
            r
            for r in rows
            if r["family"] == family
            and r["task"] == task
            and r["left"] == left
            and r["right"] == right
        ),
        None,
    )


def equivalence(
    runs: list[dict], final: dict[str, dict], delta: float = EQUIVALENCE_DELTA
) -> list[dict]:
    """Two one-sided signed-rank tests on (search - linear) per dataset.

    Equivalence within +/-``delta`` is declared when both one-sided tests
    reject. Holm is applied to the larger of the two p-values, which makes
    declaring equivalence harder rather than easier — the conservative direction
    for a claim that two arms are the same.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    idx = _index(runs, final)
    rows: list[dict] = []

    for task in _tasks_present(runs):
        diffs = []
        for rep in _replicates(runs, task):
            a, b = idx.get((task, REFERENCE, rep)), idx.get((task, "linear", rep))
            if a is None or b is None or a["test_accuracy"] is None:
                continue
            if b["test_accuracy"] is None:
                continue
            diffs.append(a["test_accuracy"] - b["test_accuracy"])
        if len(diffs) < 2:
            continue
        d = np.array(diffs, dtype=float)
        p_lower = _wilcoxon_p(d + delta, alternative="greater")
        p_upper = _wilcoxon_p(delta - d, alternative="greater")
        lo, hi = _bootstrap_median_ci(d, rng)
        rows.append(
            {
                "family": "equivalence_with_linear",
                "task": task,
                "kind": "extension" if task in EXTENSION_TASKS else "confirmatory",
                "delta": delta,
                "delta_in_test_rows": delta * TEST_ROWS.get(task, float("nan")),
                "n_pairs": len(d),
                "median_difference": float(np.median(d)),
                "ci95_low": lo,
                "ci95_high": hi,
                "p_above_minus_delta": p_lower,
                "p_below_plus_delta": p_upper,
                "wilcoxon_p": max(p_lower, p_upper),
            }
        )
    scored = [r for r in rows if r["task"] in ALL_TASKS]
    if scored:
        _holm(scored, "wilcoxon_p", FAMILY_SIZE["equivalence_with_linear"])
        for r in scored:
            r["equivalent"] = bool(r["holm_p"] < ALPHA)
    return rows


def champion_size(runs: list[dict]) -> list[dict]:
    """Does the search's champion carry a causally active hidden unit at all?

    Reported as the rate of champions with none, and tested with an exact
    binomial against a half — "more often than not, the search returns a linear
    model" is the claim, and it is a claim about a proportion rather than about
    a mean of a count that is zero most of the time.
    """
    by_task: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        if r["condition"] == REFERENCE:
            by_task[r["task"]].append(int(r["metrics"]["causal_hidden_nodes"]))

    rows = []
    for task in _tasks_present(runs):
        counts = by_task.get(task, [])
        if len(counts) < 2:
            continue
        arr = np.array(counts)
        zeros = int(np.sum(arr == 0))
        rows.append(
            {
                "family": "champion_size",
                "task": task,
                "kind": "extension" if task in EXTENSION_TASKS else "confirmatory",
                "n": len(arr),
                "zero_hidden": zeros,
                "zero_hidden_rate": float(zeros / len(arr)),
                "median_causal_hidden_nodes": float(np.median(arr)),
                "max_causal_hidden_nodes": int(arr.max()),
                "wilcoxon_p": float(
                    binomtest(zeros, len(arr), 0.5, alternative="greater").pvalue
                ),
            }
        )
    scored = [r for r in rows if r["task"] in ALL_TASKS]
    if scored:
        _holm(scored, "wilcoxon_p", FAMILY_SIZE["champion_size"])
    return rows


def budget_table(runs: list[dict]) -> list[dict]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        by[(r["task"], r["condition"])].append(r)
    rows = []
    for (task, arm), items in sorted(by.items()):
        first = items[0]
        steps = np.array([i["compute"]["gradient_steps"] for i in items], dtype=float)
        target = [i["config"].get("matched_steps") for i in items]
        target = [t for t in target if t is not None]
        wall = np.array([i["compute"]["wall_time_seconds"] for i in items], dtype=float)
        rows.append(
            {
                "task": task,
                "arm": arm,
                "matched_to": first["config"].get("matched_to") or "",
                "candidate_budget": first["config"]["candidate_budget"],
                "candidate_evaluations_mean": float(
                    np.mean([i["compute"]["candidate_evaluations"] for i in items])
                ),
                "gradient_steps_mean": float(steps.mean()),
                "target_steps_mean": float(np.mean(target)) if target else "",
                "wall_time_mean": float(wall.mean()),
            }
        )
    return rows


def hypotheses(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Score every preregistered hypothesis by its own declared rule."""
    contrasts = arm_contrasts(runs, final)
    equiv = equivalence(runs, final)
    sizes = champion_size(runs)
    n = len(ALL_TASKS)
    out: list[dict] = []

    def add(h, statement, rule, got, holds, complete=True):
        out.append(
            {
                "hypothesis": h,
                "statement": statement,
                "decision_rule": rule,
                "observed": got if complete else f"{got} (incomplete)",
                "verdict": ("holds" if holds else "fails") if complete else "incomplete",
            }
        )

    def beats(family: str, left: str, right: str) -> tuple[int, int]:
        hits = scored = 0
        for task in ALL_TASKS:
            c = _contrast(contrasts, family, task, left, right)
            if c is None or "holm_p" not in c:
                continue
            scored += 1
            hits += int(c["holm_significant"] and c["median_difference"] > 0)
        return hits, scored

    h, s = beats("vs_linear", "search", "linear")
    add("v8-H1", "architecture search beats having no architecture",
        "search - linear is significantly positive on >=2 of 3 datasets",
        f"{h}/{n}", h >= 2, complete=s == n)

    h, s = beats("vs_null", "search", "null")
    add("v8-H2", "selecting among architectures beats sampling them",
        "search - null is significantly positive on >=2 of 3 datasets",
        f"{h}/{n}", h >= 2, complete=s == n)

    h, s = beats("vs_fixed", "search", "fixed")
    add("v8-H3", "the search beats a budget-matched fixed network",
        "search - fixed is significantly positive on >=2 of 3 datasets",
        f"{h}/{n}", h >= 2, complete=s == n)

    h, s = beats("vs_linear", "fixed", "linear")
    add("v8-H4", "some architecture beats no architecture on these datasets",
        "fixed - linear is significantly positive on >=2 of 3 datasets",
        f"{h}/{n}", h >= 2, complete=s == n)

    eq = sum(1 for r in equiv if r.get("equivalent"))
    add("v8-H5", "the search and the linear model are equivalent",
        f"two one-sided signed-rank tests reject at delta={EQUIVALENCE_DELTA} "
        "accuracy on >=2 of 3 datasets",
        f"{eq}/{n}", eq >= 2,
        complete=len([r for r in equiv if r["task"] in ALL_TASKS]) == n)

    flat = sum(
        1 for r in sizes
        if r["task"] in ALL_TASKS and r.get("holm_significant")
        and r["median_causal_hidden_nodes"] == 0
    )
    add("v8-H6", "the search returns a linear model",
        "the champion carries no causally active hidden unit more often than "
        "not, by an exact binomial, on >=2 of 3 datasets",
        f"{flat}/{n}", flat >= 2,
        complete=len([r for r in sizes if r["task"] in ALL_TASKS]) == n)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def build(release_dir: Path, progress=print) -> dict:
    release_dir = Path(release_dir)
    runs, final = load(release_dir)
    if not runs:
        raise SystemExit(f"no run records under {release_dir}")

    present = {r["task"] for r in runs}
    has_extension = all(t in present for t in EXTENSION_TASKS)

    summary = summarise(runs, final)
    contrasts = arm_contrasts(runs, final)
    size_contrasts = arm_contrasts(runs, final, "causal_hidden_nodes")
    equiv = equivalence(runs, final)
    sizes = champion_size(runs)
    budgets = budget_table(runs)
    hyp = hypotheses(runs, final) if final else []

    write_csv(release_dir / "summary.csv", summary)
    write_csv(release_dir / "arm-contrasts.csv", contrasts + size_contrasts)
    write_csv(release_dir / "equivalence.csv", equiv)
    write_csv(release_dir / "champion-size.csv", sizes)
    write_csv(release_dir / "budget-table.csv", budgets)
    write_csv(release_dir / "hypotheses.csv", hyp)

    payload = {
        "n_runs": len(runs),
        "test_evaluated": bool(final),
        "family_sizes": FAMILY_SIZE,
        "confirmatory_datasets": list(ALL_TASKS),
        "extension_datasets": list(EXTENSION_TASKS),
        "extension_present": has_extension,
        "arms": list(ARMS),
        "summary": summary,
        "arm_contrasts": [
            {k: v for k, v in c.items() if k != "per_replicate"} for c in contrasts
        ],
        "size_contrasts": [
            {k: v for k, v in c.items() if k != "per_replicate"} for c in size_contrasts
        ],
        "equivalence": equiv,
        "champion_size": sizes,
        "budget_table": budgets,
        "hypotheses": hyp,
    }
    (release_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    progress(
        f"{len(runs)} runs -> summary, arm-contrasts, equivalence, champion-size, "
        f"budget-table, hypotheses"
    )
    return payload
