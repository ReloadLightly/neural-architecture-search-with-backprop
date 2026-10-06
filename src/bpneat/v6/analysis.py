"""Derive every v6 table from the raw records.

v6's question is a curve, not a point, so the tables are organised around the
rung: one contrast per (geometry, rung, arm pair), and one *slope* per
(geometry, term) summarising how a contrast moves as the budget grows.

The slope is computed per replicate and then tested across replicates, rather
than fitted once to the group medians. Three reasons, all of them about what the
nesting does: the rungs within a replicate are not independent — the ladder is
nested by construction, so a short run is a prefix of a long one — a per-replicate
slope keeps that dependence inside the unit instead of pretending four rungs are
four independent points, and the resulting thirty slopes are independent of each
other because the replicates are.

Everything here is scored on **sealed-test** accuracy. Validation accuracy is
reported in the summary as a diagnostic and is used by no decision rule, because
the champion was selected on it and the ladder is nested, which makes it
non-decreasing in budget by construction.

Nothing here retrains a model or reads data; it reads ``raw/runs/*.json`` and
``final-test.json``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .protocol import (
    ALL_CONDITIONS,
    ALL_TASKS,
    ALPHA,
    ARMS,
    BOOTSTRAP,
    BOOTSTRAP_SEED,
    BUDGETS,
    EQUIVALENCE_DELTA,
    EVERY_CONDITION,
    EXTENSION_BUDGETS,
    FAMILY_SIZE,
    REFERENCE_BUDGET,
    SLOPE_AXIS,
    SLOPE_TERMS,
    SUCCESS_THRESHOLD,
    condition_name,
)

CLIP_QUANTILE = 0.05

#: The minimum rungs a slope needs before it is reported at all. Two points give
#: a slope with no residual and no honest interval.
MIN_RUNGS_FOR_A_SLOPE = 3


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


def _clipped_mean(x: np.ndarray) -> float:
    if len(x) < 3:
        return float(np.mean(x))
    lo, hi = np.quantile(x, [CLIP_QUANTILE, 1 - CLIP_QUANTILE])
    kept = x[(x >= lo) & (x <= hi)]
    return float(np.mean(kept if len(kept) else x))


def _bootstrap_median_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(d) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _holm(rows: list[dict], key: str, family_size: int) -> None:
    """Holm step-down, in place, over ``rows`` within one declared family.

    ``family_size`` is the *declared* size, not ``len(rows)``: a family whose
    cells are not all present is still corrected for the family that was
    preregistered, so a missing cell cannot buy the others more power.
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


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summarise(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["condition"])].append(r)

    rows = []
    for (task, cond), items in sorted(groups.items()):
        first = items[0]
        va = np.array([i["metrics"]["validation_accuracy"] for i in items])
        causal = np.array([i["metrics"]["causal_hidden_nodes"] for i in items])
        steps = np.array([i["compute"]["gradient_steps"] for i in items], dtype=float)
        cands = np.array([i["compute"]["candidate_evaluations"] for i in items], dtype=float)
        wall = np.array([i["compute"]["wall_time_seconds"] for i in items])
        coll = np.array([i["metrics"]["collapsed"] for i in items], dtype=bool)

        row = {
            "task": task,
            "condition": cond,
            "arm": first["arm"],
            "budget": first["budget"],
            "candidate_budget": first["candidate_budget"],
            "budget_multiplier": first["budget_multiplier"],
            "rung": "extension" if first["budget"] in {b.label for b in EXTENSION_BUDGETS}
                    else "confirmatory",
            "n": len(items),
            # A diagnostic only: selection saw this, and the ladder is nested.
            "validation_accuracy_mean": float(va.mean()),
            "collapse_rate": float(coll.mean()),
            "causal_hidden_nodes_mean": float(causal.mean()),
            "causal_hidden_nodes_median": float(np.median(causal)),
            "gradient_steps_mean": float(steps.mean()),
            "candidate_evaluations_mean": float(cands.mean()),
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
                test_loss_clipped_mean=_clipped_mean(tl),
                test_success_rate=float(np.mean(ta >= SUCCESS_THRESHOLD[task])),
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# The two arm-contrast families
# --------------------------------------------------------------------------


def _index(runs: list[dict], final: dict[str, dict]):
    """(task, condition, replicate) -> (test_accuracy, gradient_steps)."""
    out: dict[tuple[str, str, int], dict] = {}
    for r in runs:
        t = final.get(r["run_id"])
        out[(r["task"], r["condition"], r["replicate"])] = {
            "test_accuracy": None if t is None else t["test_accuracy"],
            "validation_accuracy": r["metrics"]["validation_accuracy"],
            "gradient_steps": float(r["compute"]["gradient_steps"]),
            "candidates": float(r["compute"]["candidate_evaluations"]),
            "causal_hidden_nodes": float(r["metrics"]["causal_hidden_nodes"]),
        }
    return out


def _replicates(runs: list[dict], task: str) -> list[int]:
    return sorted({r["replicate"] for r in runs if r["task"] == task})


def arm_contrasts(
    runs: list[dict],
    final: dict[str, dict],
    metric: str = "test_accuracy",
) -> list[dict]:
    """Search minus each other arm, paired within (geometry, rung, replicate).

    One row per declared cell of ``search_vs_null`` and ``search_vs_fixed``.
    Holm runs inside each family at its declared size.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = _index(runs, final)
    by_family: dict[str, list[dict]] = {"search_vs_null": [], "search_vs_fixed": []}
    # Only the primary outcome is a scored family. A contrast on any other
    # metric is descriptive, carries ``scored=False``, and gets no corrected
    # p-value, so it cannot be read as a test the contract declared.
    scored = metric == "test_accuracy"

    for other in ("null", "fixed"):
        family = f"search_vs_{other}"
        for task in ALL_TASKS:
            reps = _replicates(runs, task)
            for b in BUDGETS:
                a_name, b_name = condition_name("search", b), condition_name(other, b)
                diffs, pairs = [], []
                for rep in reps:
                    a, c = idx.get((task, a_name, rep)), idx.get((task, b_name, rep))
                    if a is None or c is None:
                        continue
                    if a[metric] is None or c[metric] is None:
                        continue
                    diffs.append(a[metric] - c[metric])
                    pairs.append({"replicate": rep, "search": a[metric], other: c[metric]})
                if len(diffs) < 2:
                    continue
                d = np.array(diffs, dtype=float)
                lo, hi = _bootstrap_median_ci(d, rng)
                by_family[family].append(
                    {
                        "family": family,
                        "task": task,
                        "budget": b.label,
                        "candidate_budget": b.candidates,
                        "budget_multiplier": b.multiplier,
                        "metric": metric,
                        "scored": scored,
                        "comparison": f"search - {other}",
                        "n_pairs": len(d),
                        "mean_difference": float(d.mean()),
                        "median_difference": float(np.median(d)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "wilcoxon_p": _wilcoxon_p(d),
                        "search_wins": int(np.sum(d > 0)),
                        "per_replicate": json.dumps(pairs),
                    }
                )
    rows = []
    for family, members in by_family.items():
        if members and scored:
            _holm(members, "wilcoxon_p", FAMILY_SIZE[family])
        rows.extend(members)
    return rows


def _contrast(rows: list[dict], family: str, task: str, budget: str) -> dict | None:
    return next(
        (r for r in rows
         if r["family"] == family and r["task"] == task and r["budget"] == budget),
        None,
    )


# --------------------------------------------------------------------------
# The scaling-slope family
# --------------------------------------------------------------------------


def _series(idx, task: str, rep: int, term: str, budgets) -> tuple[list[float], list[float]]:
    """(x, y) across rungs for one replicate and one declared slope term."""
    axis = SLOPE_AXIS[term]
    xs: list[float] = []
    ys: list[float] = []
    for b in budgets:
        search = idx.get((task, condition_name("search", b), rep))
        if term in ARMS:
            arm = idx.get((task, condition_name(term, b), rep))
            if arm is None or arm["test_accuracy"] is None:
                continue
            x, y = arm[axis], arm["test_accuracy"]
        else:
            other = "null" if term == "edge_vs_null" else "fixed"
            o = idx.get((task, condition_name(other, b), rep))
            if search is None or o is None:
                continue
            if search["test_accuracy"] is None or o["test_accuracy"] is None:
                continue
            x = search[axis]
            y = search["test_accuracy"] - o["test_accuracy"]
        if x is None or x <= 0:
            continue
        xs.append(float(np.log10(x)))
        ys.append(float(y))
    return xs, ys


def scaling_slopes(
    runs: list[dict],
    final: dict[str, dict],
    budgets=BUDGETS,
    label: str = "confirmatory",
) -> list[dict]:
    """One least-squares slope per replicate, then a signed-rank test on those.

    The slope is accuracy per decade of the matched quantity — gradient steps for
    the single-arm terms and for ``edge_vs_fixed``, candidates for
    ``edge_vs_null`` — as declared in ``protocol.SLOPE_AXIS``.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    idx = _index(runs, final)
    rows: list[dict] = []

    for task in ALL_TASKS:
        reps = _replicates(runs, task)
        for term in SLOPE_TERMS:
            slopes, used = [], []
            for rep in reps:
                xs, ys = _series(idx, task, rep, term, budgets)
                if len(xs) < MIN_RUNGS_FOR_A_SLOPE or len(set(xs)) < 2:
                    continue
                slope = float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
                slopes.append(slope)
                used.append({"replicate": rep, "rungs": len(xs), "slope": slope})
            if len(slopes) < 2:
                continue
            s = np.array(slopes, dtype=float)
            lo, hi = _bootstrap_median_ci(s, rng)
            rows.append(
                {
                    "family": "scaling_slopes",
                    "ladder": label,
                    "task": task,
                    "term": term,
                    "axis": SLOPE_AXIS[term],
                    "n_rungs": len(budgets),
                    "n_replicates": len(s),
                    "mean_slope": float(s.mean()),
                    "median_slope": float(np.median(s)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "wilcoxon_p": _wilcoxon_p(s),
                    "positive": int(np.sum(s > 0)),
                    "per_replicate": json.dumps(used),
                }
            )
    if rows:
        _holm(rows, "wilcoxon_p", FAMILY_SIZE["scaling_slopes"])
    return rows


def _slope(rows: list[dict], task: str, term: str, label: str = "confirmatory") -> dict | None:
    return next(
        (r for r in rows
         if r["task"] == task and r["term"] == term and r["ladder"] == label),
        None,
    )


def size_slopes(runs: list[dict], final: dict[str, dict], budgets=BUDGETS) -> list[dict]:
    """Does the champion grow with the budget it was found under?

    One least-squares slope per replicate of the search champion's *causally
    active* hidden nodes against log10 candidates, then a signed-rank test on
    those slopes. Represented nodes are not used: this project has repeatedly
    found represented structure that never reaches the output.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 3)
    idx = _index(runs, final)
    rows: list[dict] = []

    for task in ALL_TASKS:
        slopes, used = [], []
        for rep in _replicates(runs, task):
            xs, ys = [], []
            for b in budgets:
                rec = idx.get((task, condition_name("search", b), rep))
                if rec is None:
                    continue
                xs.append(float(np.log10(b.candidates)))
                ys.append(rec["causal_hidden_nodes"])
            if len(xs) < MIN_RUNGS_FOR_A_SLOPE or len(set(xs)) < 2:
                continue
            slope = float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
            slopes.append(slope)
            used.append({"replicate": rep, "rungs": len(xs), "slope": slope})
        if len(slopes) < 2:
            continue
        s = np.array(slopes, dtype=float)
        lo, hi = _bootstrap_median_ci(s, rng)
        rows.append(
            {
                "family": "size_slopes",
                "ladder": "confirmatory",
                "task": task,
                "term": "search_size",
                "axis": "candidates",
                "n_rungs": len(budgets),
                "n_replicates": len(s),
                "mean_slope": float(s.mean()),
                "median_slope": float(np.median(s)),
                "ci95_low": lo,
                "ci95_high": hi,
                "wilcoxon_p": _wilcoxon_p(s),
                "positive": int(np.sum(s > 0)),
                "per_replicate": json.dumps(used),
            }
        )
    if rows:
        _holm(rows, "wilcoxon_p", FAMILY_SIZE["size_slopes"])
    return rows


# --------------------------------------------------------------------------
# Equivalence at the reference rung
# --------------------------------------------------------------------------


def equivalence(
    runs: list[dict],
    final: dict[str, dict],
    delta: float = EQUIVALENCE_DELTA,
) -> list[dict]:
    """Two one-sided signed-rank tests on (search - null) at the reference rung.

    Equivalence within +/-``delta`` is declared when *both* one-sided tests
    reject: the paired median is above ``-delta`` and below ``+delta``. Holm
    correction is applied to the larger of the two p-values, which makes
    declaring equivalence harder rather than easier — the conservative direction
    for a claim that two arms are the same.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    idx = _index(runs, final)
    b = REFERENCE_BUDGET
    rows: list[dict] = []

    for task in ALL_TASKS:
        diffs = []
        for rep in _replicates(runs, task):
            a = idx.get((task, condition_name("search", b), rep))
            c = idx.get((task, condition_name("null", b), rep))
            if a is None or c is None or a["test_accuracy"] is None:
                continue
            if c["test_accuracy"] is None:
                continue
            diffs.append(a["test_accuracy"] - c["test_accuracy"])
        if len(diffs) < 2:
            continue
        d = np.array(diffs, dtype=float)
        p_lower = _wilcoxon_p(d + delta, alternative="greater")
        p_upper = _wilcoxon_p(delta - d, alternative="greater")
        lo, hi = _bootstrap_median_ci(d, rng)
        rows.append(
            {
                "family": "equivalence_at_reference",
                "task": task,
                "budget": b.label,
                "delta": delta,
                "n_pairs": len(d),
                "median_difference": float(np.median(d)),
                "ci95_low": lo,
                "ci95_high": hi,
                "p_above_minus_delta": p_lower,
                "p_below_plus_delta": p_upper,
                "wilcoxon_p": max(p_lower, p_upper),
            }
        )
    if rows:
        _holm(rows, "wilcoxon_p", FAMILY_SIZE["equivalence_at_reference"])
        for r in rows:
            r["equivalent"] = bool(r["holm_p"] < ALPHA)
    return rows


# --------------------------------------------------------------------------
# What each rung actually cost
# --------------------------------------------------------------------------


def budget_table(runs: list[dict]) -> list[dict]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        by[(r["task"], r["condition"])].append(r)
    rows = []
    for (task, cond), items in sorted(by.items()):
        first = items[0]
        steps = np.array([i["compute"]["gradient_steps"] for i in items], dtype=float)
        target = [i["config"].get("matched_steps") for i in items]
        target = [t for t in target if t is not None]
        wall = np.array([i["compute"]["wall_time_seconds"] for i in items], dtype=float)
        rows.append(
            {
                "task": task,
                "condition": cond,
                "arm": first["arm"],
                "budget": first["budget"],
                "candidate_budget": first["candidate_budget"],
                "budget_multiplier": first["budget_multiplier"],
                "matched_to": first["config"].get("matched_to") or "",
                "generations": first["config"].get("generations") or "",
                "gradient_steps_mean": float(steps.mean()),
                "target_steps_mean": float(np.mean(target)) if target else "",
                "steps_per_candidate": float(steps.mean() / max(first["candidate_budget"], 1))
                if first["arm"] != "fixed" else "",
                "wall_time_mean": float(wall.mean()),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Hypotheses
# --------------------------------------------------------------------------


def hypotheses(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Score every preregistered hypothesis by its own declared rule."""
    contrasts = arm_contrasts(runs, final)
    slopes = scaling_slopes(runs, final)
    sizes = size_slopes(runs, final)
    equiv = equivalence(runs, final)
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

    def slope_sign(term: str, want: int) -> tuple[int, int]:
        """Geometries where this slope is significant with the declared sign."""
        hits = scored = 0
        for t in ALL_TASKS:
            s = _slope(slopes, t, term)
            if s is None or "holm_p" not in s:
                continue
            scored += 1
            hits += int(s["holm_significant"] and np.sign(s["median_slope"]) == want)
        return hits, scored

    h, s = slope_sign("edge_vs_null", +1)
    add("v6-H1", "the search's edge over its candidate-matched null grows with budget",
        "the per-replicate slope of (search - null) against log10 candidates is "
        "significantly positive on >=2 of 3 geometries",
        f"{h}/{n}", h >= 2, complete=s == n)

    eq = sum(1 for r in equiv if r.get("equivalent"))
    add("v6-H2", "at the reference budget the search and its null are equivalent",
        f"two one-sided signed-rank tests reject at delta={EQUIVALENCE_DELTA} "
        "accuracy on >=2 of 3 geometries",
        f"{eq}/{n}", eq >= 2, complete=len(equiv) == n)

    beats, scored = 0, 0
    for t in ALL_TASKS:
        cells = [_contrast(contrasts, "search_vs_fixed", t, b.label) for b in BUDGETS]
        cells = [c for c in cells if c is not None and "holm_p" in c]
        if len(cells) != len(BUDGETS):
            continue
        scored += 1
        beats += int(any(c["holm_significant"] and c["median_difference"] > 0 for c in cells))
    add("v6-H3", "at some budget the search beats the budget-matched fixed network",
        "search - fixed is significantly positive at >=1 rung, on >=2 of 3 geometries",
        f"{beats}/{n}", beats >= 2, complete=scored == n)

    h, s = slope_sign("edge_vs_fixed", -1)
    add("v6-H4", "the search buys less per decade of compute than the fixed network does",
        "the per-replicate slope of (search - fixed) against log10 gradient steps "
        "is significantly negative on >=2 of 3 geometries",
        f"{h}/{n}", h >= 2, complete=s == n)

    h, s = slope_sign("null", +1)
    add("v6-H5", "sampling more architectures helps even with selection removed",
        "the null's own slope against log10 gradient steps is significantly "
        "positive on >=2 of 3 geometries",
        f"{h}/{n}", h >= 2, complete=s == n)

    flips, scored = 0, 0
    for t in ALL_TASKS:
        cells = [_contrast(contrasts, "search_vs_fixed", t, b.label) for b in BUDGETS]
        cells = [c for c in cells if c is not None and "holm_p" in c]
        if len(cells) != len(BUDGETS):
            continue
        scored += 1
        sig = [c for c in cells if c["holm_significant"]]
        flips += int(
            any(c["median_difference"] > 0 for c in sig)
            and any(c["median_difference"] < 0 for c in sig)
        )
    add("v6-H6", "the reference-budget verdict is not an artefact of the budget",
        "no geometry has one rung where the search significantly wins and another "
        "where it significantly loses",
        f"{flips}/{n} geometries flip", flips == 0, complete=scored == n)

    grew = sum(
        1 for r in sizes
        if r.get("holm_significant") and r["median_slope"] > 0
    )
    add("v6-H7", "the champion's size is set by the budget, not only by the fitness",
        "the per-replicate slope of the search champion's causal hidden nodes "
        "against log10 candidates is significantly positive on >=2 of 3 geometries",
        f"{grew}/{n}", grew >= 2, complete=len(sizes) == n)
    return out


# --------------------------------------------------------------------------


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

    present = {r["condition"] for r in runs}
    has_extension = all(
        condition_name(a, b) in present for b in EXTENSION_BUDGETS for a in ARMS
    )

    summary = summarise(runs, final)
    contrasts = arm_contrasts(runs, final)
    size_contrasts = arm_contrasts(runs, final, "causal_hidden_nodes")
    slopes = scaling_slopes(runs, final) + size_slopes(runs, final)
    if has_extension:
        # Reported, never scored: the extension ladder gets its own rows and its
        # own label so no reader can mistake it for the confirmatory family.
        slopes = slopes + scaling_slopes(
            runs, final, budgets=BUDGETS + EXTENSION_BUDGETS, label="with_extension"
        )
    equiv = equivalence(runs, final)
    budgets = budget_table(runs)
    hyp = hypotheses(runs, final) if final else []

    write_csv(release_dir / "summary.csv", summary)
    write_csv(release_dir / "arm-contrasts.csv", contrasts + size_contrasts)
    write_csv(release_dir / "scaling-slopes.csv", slopes)
    write_csv(release_dir / "equivalence.csv", equiv)
    write_csv(release_dir / "budget-table.csv", budgets)
    write_csv(release_dir / "hypotheses.csv", hyp)

    payload = {
        "n_runs": len(runs),
        "test_evaluated": bool(final),
        "family_sizes": FAMILY_SIZE,
        "confirmatory_rungs": [b.label for b in BUDGETS],
        "extension_rungs": [b.label for b in EXTENSION_BUDGETS],
        "extension_present": has_extension,
        "conditions_expected": list(EVERY_CONDITION if has_extension else ALL_CONDITIONS),
        "summary": summary,
        "arm_contrasts": [
            {k: v for k, v in c.items() if k != "per_replicate"} for c in contrasts
        ],
        "size_contrasts": [
            {k: v for k, v in c.items() if k != "per_replicate"} for c in size_contrasts
        ],
        "scaling_slopes": [
            {k: v for k, v in s.items() if k != "per_replicate"} for s in slopes
        ],
        "equivalence": equiv,
        "budget_table": budgets,
        "hypotheses": hyp,
    }
    (release_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    progress(
        f"{len(runs)} runs -> summary, arm-contrasts, scaling-slopes, equivalence, "
        f"budget-table, hypotheses"
    )
    return payload
