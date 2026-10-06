"""Derive every v5 table from the raw records.

v5's question is about NEAT's own machinery, so the primary outcome is reported
twice: sealed-test **accuracy**, as in v3 and v4, and champion **size** — causal
hidden nodes, the measure that reflects what actually reached the output rather
than what the genome represents. A mechanism study of an algorithm whose name
contains "augmenting topologies" has to report whether the topologies augmented.

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
    FAMILY_CONDITIONS,
    FAMILY_SIZE,
    REFERENCE,
    SUCCESS_THRESHOLD,
)

BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261006
CLIP_QUANTILE = 0.05

#: Ha's published Figure 10.3 champion on spirals, for scale. A target, never
#: pooled with anything measured here.
REFERENCE_CHAMPION = {"task": "spiral", "nodes": 34, "connections": 96}


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


def summarise(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["condition"])].append(r)

    rows = []
    for (task, cond), items in sorted(groups.items()):
        va = np.array([i["metrics"]["validation_accuracy"] for i in items])
        causal = np.array([i["metrics"]["causal_hidden_nodes"] for i in items])
        conns = np.array([i["metrics"]["causal_connections"] for i in items])
        repr_n = np.array([i["metrics"]["represented_nodes"] for i in items])
        repr_c = np.array([i["metrics"]["represented_connections"] for i in items])
        steps = np.array([i["compute"]["gradient_steps"] for i in items])
        cands = np.array([i["compute"]["candidate_evaluations"] for i in items])
        wall = np.array([i["compute"]["wall_time_seconds"] for i in items])
        coll = np.array([i["metrics"]["collapsed"] for i in items], dtype=bool)
        adds = [i["compute"].get("node_additions") for i in items]
        adds = [a for a in adds if a is not None]
        xovers = [i["compute"].get("crossovers") for i in items]
        xovers = [x for x in xovers if x is not None]

        row = {
            "task": task,
            "condition": cond,
            "n": len(items),
            "validation_accuracy_mean": float(va.mean()),
            "collapse_rate": float(coll.mean()),
            "causal_hidden_nodes_mean": float(causal.mean()),
            "causal_hidden_nodes_median": float(np.median(causal)),
            "causal_hidden_nodes_max": int(causal.max()),
            "causal_connections_mean": float(conns.mean()),
            "represented_nodes_mean": float(repr_n.mean()),
            "represented_connections_mean": float(repr_c.mean()),
            "node_additions_mean": float(np.mean(adds)) if adds else "",
            "crossovers_mean": float(np.mean(xovers)) if xovers else "",
            "gradient_steps_mean": float(steps.mean()),
            "candidate_evaluations_mean": float(cands.mean()),
            "wall_time_mean": float(wall.mean()),
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
                test_loss_median=float(np.median(tl)),
                test_loss_clipped_mean=_clipped_mean(tl),
                test_success_rate=float(np.mean(ta >= SUCCESS_THRESHOLD[task])),
            )
        rows.append(row)
    return rows


def paired_effects(
    runs: list[dict],
    final: dict[str, dict],
    metric: str = "test_accuracy",
    conditions: tuple[str, ...] = FAMILY_CONDITIONS,
    holm_family: int | None = FAMILY_SIZE,
) -> list[dict]:
    """Reference minus each condition, paired within (task, replicate)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by = {(r["task"], r["condition"], r["replicate"]): r for r in runs}
    tasks = [t for t in ALL_TASKS if any(r["task"] == t for r in runs)]

    def value(rec):
        if metric.startswith("test_"):
            t = final.get(rec["run_id"])
            return None if t is None else t[metric]
        return rec["metrics"].get(metric)

    rows = []
    for task in tasks:
        reps = sorted({r["replicate"] for r in runs if r["task"] == task})
        for cond in conditions:
            diffs, pairs = [], []
            for rep in reps:
                a, b = by.get((task, REFERENCE, rep)), by.get((task, cond, rep))
                if a is None or b is None:
                    continue
                va, vb = value(a), value(b)
                if va is None or vb is None:
                    continue
                diffs.append(va - vb)
                pairs.append({"replicate": rep, "reference": va, "other": vb})
            if len(diffs) < 2:
                continue
            d = np.array(diffs, dtype=float)
            lo, hi = _bootstrap_median_ci(d, rng)
            try:
                p = float(wilcoxon(d, zero_method="zsplit").pvalue)
            except ValueError:
                p = 1.0
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "comparison": f"{REFERENCE} - {cond}",
                    "condition": cond,
                    "n_pairs": len(d),
                    "mean_difference": float(d.mean()),
                    "median_difference": float(np.median(d)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "wilcoxon_p": p,
                    "wins": int(np.sum(d > 0)),
                    "per_replicate": json.dumps(pairs),
                }
            )

    if holm_family:
        order = sorted(range(len(rows)), key=lambda i: rows[i]["wilcoxon_p"])
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, rows[i]["wilcoxon_p"] * (holm_family - rank))
            running = max(running, adj)
            rows[i]["holm_p"] = running
            rows[i]["holm_significant"] = bool(running < 0.05)
    return rows


def _effect(effects: list[dict], task: str, cond: str) -> dict | None:
    return next((e for e in effects if e["task"] == task and e["condition"] == cond), None)


def complexity(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Did the topologies augment, and did it buy anything?

    One row per (task, condition): the size NEAT reached, how far that is from
    the published reference champion, and the accuracy it bought.
    """
    summ = {(r["task"], r["condition"]): r for r in summarise(runs, final)}
    size_eff = paired_effects(runs, final, "causal_hidden_nodes")
    acc_eff = paired_effects(runs, final, "test_accuracy")
    ref_size = REFERENCE_CHAMPION["nodes"]

    rows = []
    for task in ALL_TASKS:
        base = summ.get((task, REFERENCE))
        if base is None:
            continue
        for cond in ALL_CONDITIONS:
            row = summ.get((task, cond))
            if row is None:
                continue
            se = _effect(size_eff, task, cond)
            ae = _effect(acc_eff, task, cond)
            rows.append(
                {
                    "task": task,
                    "condition": cond,
                    "causal_hidden_nodes_mean": row["causal_hidden_nodes_mean"],
                    "causal_hidden_nodes_max": row["causal_hidden_nodes_max"],
                    "represented_nodes_mean": row["represented_nodes_mean"],
                    "node_additions_mean": row["node_additions_mean"],
                    "test_accuracy_mean": row.get("test_accuracy_mean", ""),
                    "size_vs_reference": row["causal_hidden_nodes_mean"]
                    - base["causal_hidden_nodes_mean"],
                    "size_change_holm_p": se["holm_p"] if se and "holm_p" in se else "",
                    "accuracy_vs_reference": (
                        row["test_accuracy_mean"] - base["test_accuracy_mean"]
                        if "test_accuracy_mean" in row and "test_accuracy_mean" in base
                        else ""
                    ),
                    "accuracy_change_holm_p": ae["holm_p"] if ae and "holm_p" in ae else "",
                    # Ha's Figure 10.3 spirals champion, as a scale marker only.
                    "fraction_of_published_champion": (
                        row["causal_hidden_nodes_mean"] / ref_size if task == "spiral" else ""
                    ),
                }
            )
    return rows


def hypotheses(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Score every preregistered hypothesis by its own declared rule."""
    summ = {(r["task"], r["condition"]): r for r in summarise(runs, final)}
    size = paired_effects(runs, final, "causal_hidden_nodes")
    acc = paired_effects(runs, final, "test_accuracy")
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

    def grew(cond):
        """Tasks where the condition's champions are significantly larger."""
        hits = scored = 0
        for t in ALL_TASKS:
            e = _effect(size, t, cond)
            if e is None or "holm_p" not in e:
                continue
            scored += 1
            # The effect is reference minus condition, so a *negative* median
            # means the condition produced the larger networks.
            hits += int(e["holm_p"] < 0.05 and e["median_difference"] < 0)
        return hits, scored

    def beat(cond):
        hits = scored = 0
        for t in ALL_TASKS:
            e = _effect(acc, t, cond)
            if e is None or "holm_p" not in e:
                continue
            scored += 1
            hits += int(e["holm_p"] < 0.05 and e["median_difference"] < 0)
        return hits, scored

    def lost(cond):
        hits = scored = 0
        for t in ALL_TASKS:
            e = _effect(acc, t, cond)
            if e is None or "holm_p" not in e:
                continue
            scored += 1
            hits += int(e["holm_p"] < 0.05 and e["median_difference"] > 0)
        return hits, scored

    h, s = grew("neat_no_penalty")
    add("v5-H1", "the complexity penalty is what suppresses complexification",
        "removing it grows champions significantly on >=2 of 3 geometries",
        f"{h}/{n}", h >= 2, complete=s == n)

    h, s = grew("neat_complexify")
    add("v5-H2", "the structural mutation rate is a binding constraint",
        "raising it grows champions significantly on >=2 of 3",
        f"{h}/{n}", h >= 2, complete=s == n)

    h, s = grew("neat_complexify_no_penalty")
    add("v5-H3", "the two act together",
        "removing both constraints grows champions significantly on >=2 of 3",
        f"{h}/{n}", h >= 2, complete=s == n)

    # H4: does any larger-topology arm actually buy accuracy?
    arms = ("neat_complexify", "neat_no_penalty", "neat_complexify_no_penalty")
    hits = sum(beat(a)[0] for a in arms)
    scored = sum(beat(a)[1] for a in arms)
    add("v5-H4", "bigger topologies buy accuracy",
        "some complexification arm beats the reference on >=2 of 3",
        f"{hits}/{len(arms) * n} arm-task cells", hits >= 2,
        complete=scored == len(arms) * n)

    h, s = lost("neat_no_speciation")
    add("v5-H5", "speciation is load-bearing",
        "removing it loses significantly on >=2 of 3", f"{h}/{n}", h >= 2,
        complete=s == n)

    h, s = lost("neat_no_crossover")
    add("v5-H6", "crossover is load-bearing",
        "removing it loses significantly on >=2 of 3", f"{h}/{n}", h >= 2,
        complete=s == n)

    # H7: the yardstick. Does the best NEAT variant beat a fixed network?
    beats_fixed = 0
    scored = 0
    for t in ALL_TASKS:
        e = _effect(acc, t, "fixed_mixed_matched")
        best = max(
            (summ[(t, c)].get("test_accuracy_mean") for c in ALL_CONDITIONS
             if c != "fixed_mixed_matched" and (t, c) in summ
             and summ[(t, c)].get("test_accuracy_mean") is not None),
            default=None,
        )
        fixed = summ.get((t, "fixed_mixed_matched"), {}).get("test_accuracy_mean")
        if e is None or best is None or fixed is None:
            continue
        scored += 1
        beats_fixed += int(best > fixed)
    add("v5-H7", "no NEAT variant beats a budget-matched fixed network",
        "the best variant beats the fixed arm on 0 of 3", f"{beats_fixed}/{n}",
        beats_fixed == 0, complete=scored == n)
    return out


def operator_usage(runs: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for r in runs:
        for op, k in r["metrics"].get("causal_operators", {}).items():
            counts[(r["task"], r["condition"], op)] += k
            totals[(r["task"], r["condition"])] += k
    return [
        {"task": t, "condition": c, "operator": op, "count": k,
         "fraction": k / max(totals[(t, c)], 1)}
        for (t, c, op), k in sorted(counts.items())
    ]


def budget_table(runs: list[dict]) -> list[dict]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        by[(r["task"], r["condition"])].append(r)
    rows = []
    for (task, cond), items in sorted(by.items()):
        steps = np.array([i["compute"]["gradient_steps"] for i in items], dtype=float)
        target = [i["config"].get("matched_steps") for i in items]
        target = [t for t in target if t is not None]
        rows.append(
            {
                "task": task,
                "condition": cond,
                "matched_to": items[0]["config"].get("matched_to") or "",
                "candidate_budget": items[0]["config"]["candidate_budget"],
                "population": items[0]["config"]["population"],
                "generations": items[0]["config"]["generations"],
                "gradient_steps_mean": float(steps.mean()),
                "target_steps_mean": float(np.mean(target)) if target else "",
                "candidate_evaluations_mean": float(
                    np.mean([i["compute"]["candidate_evaluations"] for i in items])
                ),
            }
        )
    return rows


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

    summary = summarise(runs, final)
    effects = paired_effects(runs, final, "test_accuracy")
    size_effects = paired_effects(runs, final, "causal_hidden_nodes")
    comp = complexity(runs, final)
    budgets = budget_table(runs)
    ops = operator_usage(runs)
    hyp = hypotheses(runs, final) if final else []

    write_csv(release_dir / "summary.csv", summary)
    write_csv(release_dir / "paired-effects.csv", effects + size_effects)
    write_csv(release_dir / "complexity.csv", comp)
    write_csv(release_dir / "budget-table.csv", budgets)
    write_csv(release_dir / "operator-usage.csv", ops)
    write_csv(release_dir / "hypotheses.csv", hyp)

    payload = {
        "n_runs": len(runs),
        "test_evaluated": bool(final),
        "family_size": FAMILY_SIZE,
        "summary": summary,
        "paired_effects": [
            {k: v for k, v in e.items() if k != "per_replicate"} for e in effects
        ],
        "size_effects": [
            {k: v for k, v in e.items() if k != "per_replicate"} for e in size_effects
        ],
        "complexity": comp,
        "budget_table": budgets,
        "hypotheses": hyp,
    }
    (release_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    progress(
        f"{len(runs)} runs -> summary, paired-effects, complexity, budget-table, "
        f"operator-usage, hypotheses"
    )
    return payload
