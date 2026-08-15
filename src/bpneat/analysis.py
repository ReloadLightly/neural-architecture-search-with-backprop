"""Reconstruct every summary from the raw run records.

Nothing here recomputes a model or touches data. It reads the durable records
and derives the tables, so any published number can be traced back to the run
that produced it. Tracks are never pooled and tasks are never collapsed into a
single headline figure.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .conditions import CORE_CONDITIONS, SUCCESS_THRESHOLD

REFERENCE = "backprop_neat"
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260815


def load_runs(release_dir: Path) -> list[dict]:
    runs = []
    for path in sorted((Path(release_dir) / "raw" / "runs").glob("*.json")):
        runs.append(json.loads(path.read_text()))
    return runs


def load_final_test(release_dir: Path) -> dict[str, dict]:
    path = Path(release_dir) / "final-test.json"
    if not path.exists():
        return {}
    return {r["run_id"]: r for r in json.loads(path.read_text())["results"]}


def _bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Paired bootstrap over replicates. Ten replicates is modest; this is an
    uncertainty summary, not proof of a universal effect."""
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(diffs), size=(BOOTSTRAP, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(runs: list[dict], final: dict[str, dict] | None = None) -> list[dict]:
    """Per (task, condition) aggregates, with every replicate still recoverable."""
    final = final or {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["condition"])].append(r)

    rows = []
    for (task, condition), items in sorted(groups.items()):
        val = np.array([i["metrics"]["validation_accuracy"] for i in items])
        vloss = np.array([i["metrics"]["validation_loss"] for i in items])
        causal = np.array([i["metrics"]["causal_hidden_nodes"] for i in items])
        repr_n = np.array([i["metrics"]["represented_nodes"] for i in items])
        gap = np.array([i["metrics"]["generalization_gap"] for i in items])
        cands = np.array([i["compute"]["candidate_evaluations"] for i in items])
        steps = np.array([i["compute"]["gradient_steps"] for i in items])
        wall = np.array([i["compute"]["wall_time_seconds"] for i in items])

        row = {
            "task": task,
            "condition": condition,
            "n_replicates": len(items),
            "validation_accuracy_mean": val.mean(),
            "validation_accuracy_median": float(np.median(val)),
            "validation_accuracy_sd": val.std(ddof=1) if len(val) > 1 else 0.0,
            "validation_accuracy_min": val.min(),
            "validation_accuracy_max": val.max(),
            "validation_loss_mean": vloss.mean(),
            "success_rate": float(np.mean(val >= SUCCESS_THRESHOLD[task])),
            "success_count": int(np.sum(val >= SUCCESS_THRESHOLD[task])),
            "generalization_gap_mean": gap.mean(),
            "represented_nodes_mean": repr_n.mean(),
            "causal_hidden_nodes_mean": causal.mean(),
            "causal_fraction": float(causal.mean() / max(repr_n.mean() - 4, 1e-9)),
            "candidate_evaluations_mean": cands.mean(),
            "gradient_steps_mean": steps.mean(),
            "wall_time_mean": wall.mean(),
        }

        tests = [final[i["run_id"]] for i in items if i["run_id"] in final]
        if tests:
            tacc = np.array([t["test_accuracy"] for t in tests])
            tloss = np.array([t["test_loss"] for t in tests])
            row.update(
                test_accuracy_mean=tacc.mean(),
                test_accuracy_sd=tacc.std(ddof=1) if len(tacc) > 1 else 0.0,
                test_loss_mean=tloss.mean(),
                test_success_rate=float(np.mean(tacc >= SUCCESS_THRESHOLD[task])),
                validation_to_test_drop_mean=float(
                    np.mean([t["validation_to_test_drop"] for t in tests])
                ),
            )
        rows.append(row)
    return rows


def paired_effects(
    runs: list[dict], final: dict[str, dict] | None = None, metric: str = "validation_loss"
) -> list[dict]:
    """Backprop-NEAT minus each control, paired within task and replicate.

    Pairing is what the replicate design buys: both conditions saw the same
    sampled dataset, so the difference removes data variation.
    """
    final = final or {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by_key = {(r["task"], r["condition"], r["replicate"]): r for r in runs}
    tasks = sorted({r["task"] for r in runs})
    conditions = [c for c in CORE_CONDITIONS if c != REFERENCE]

    def value(rec):
        if metric.startswith("test_"):
            t = final.get(rec["run_id"])
            return None if t is None else t[metric]
        return rec["metrics"][metric]

    rows = []
    for task in tasks:
        reps = sorted({r["replicate"] for r in runs if r["task"] == task})
        for condition in conditions:
            diffs, pairs = [], []
            for rep in reps:
                a = by_key.get((task, REFERENCE, rep))
                b = by_key.get((task, condition, rep))
                if a is None or b is None:
                    continue
                va, vb = value(a), value(b)
                if va is None or vb is None:
                    continue
                diffs.append(va - vb)
                pairs.append({"replicate": rep, "reference": va, "control": vb})
            if not diffs:
                continue
            d = np.array(diffs)
            lo, hi = _bootstrap_ci(d, rng)
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "comparison": f"{REFERENCE} - {condition}",
                    "n_pairs": len(d),
                    "mean_difference": d.mean(),
                    "median_difference": float(np.median(d)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "wins": int(np.sum(d < 0)) if "loss" in metric else int(np.sum(d > 0)),
                    "per_replicate": json.dumps(pairs),
                }
            )
    return rows


def operation_usage(runs: list[dict]) -> list[dict]:
    """Causal operator frequencies by task — the evidence for H4."""
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for r in runs:
        ops = r["metrics"].get("causal_operators", {})
        for op, n in ops.items():
            counts[(r["task"], r["condition"], op)] += n
            totals[(r["task"], r["condition"])] += n
    rows = []
    for (task, condition, op), n in sorted(counts.items()):
        rows.append(
            {
                "task": task,
                "condition": condition,
                "operator": op,
                "count": n,
                "fraction": n / max(totals[(task, condition)], 1),
            }
        )
    return rows


def pareto_front(runs: list[dict], final: dict[str, dict] | None = None) -> list[dict]:
    """Non-dominated on (performance, causal complexity, gradient steps).

    Retaining the trade-off rather than collapsing it into one score is the
    point: accuracy is not the only objective a decision-support tool has.
    """
    final = final or {}
    points = []
    for r in runs:
        t = final.get(r["run_id"])
        points.append(
            {
                "run_id": r["run_id"],
                "task": r["task"],
                "condition": r["condition"],
                "replicate": r["replicate"],
                "loss": (t["test_loss"] if t else r["metrics"]["validation_loss"]),
                "loss_source": "test" if t else "validation",
                "causal_connections": r["metrics"]["causal_connections"],
                "gradient_steps": r["compute"]["gradient_steps"],
            }
        )

    rows = []
    for task in sorted({p["task"] for p in points}):
        group = [p for p in points if p["task"] == task]
        for p in group:
            dominated = any(
                q["loss"] <= p["loss"]
                and q["causal_connections"] <= p["causal_connections"]
                and q["gradient_steps"] <= p["gradient_steps"]
                and (
                    q["loss"] < p["loss"]
                    or q["causal_connections"] < p["causal_connections"]
                    or q["gradient_steps"] < p["gradient_steps"]
                )
                for q in group
                if q is not p
            )
            if not dominated:
                rows.append({**p, "non_dominated": True})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build(release_dir: Path, progress=print) -> dict:
    """Regenerate every derived table from the raw records."""
    release_dir = Path(release_dir)
    runs = load_runs(release_dir)
    if not runs:
        raise SystemExit(f"no run records under {release_dir}")
    final = load_final_test(release_dir)

    summary = summarise(runs, final)
    effects = paired_effects(runs, final, "validation_loss")
    if final:
        effects += paired_effects(runs, final, "test_loss")
    effects += paired_effects(runs, final, "validation_accuracy")

    write_csv(release_dir / "summary.csv", summary)
    write_csv(release_dir / "paired-effects.csv", effects)
    write_csv(release_dir / "operation-usage.csv", operation_usage(runs))
    write_csv(release_dir / "pareto-front.csv", pareto_front(runs, final))

    payload = {
        "n_runs": len(runs),
        "test_evaluated": bool(final),
        "tracks": sorted({r["track"] for r in runs}),
        "summary": summary,
        "paired_effects": [{k: v for k, v in e.items() if k != "per_replicate"} for e in effects],
    }
    (release_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    progress(f"{len(runs)} runs -> summary.csv, paired-effects.csv, operation-usage.csv, pareto-front.csv")
    return payload
