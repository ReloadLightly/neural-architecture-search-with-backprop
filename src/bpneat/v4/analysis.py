"""Derive every v4 table from the raw records.

Nothing here retrains a model or reads data; it reads ``raw/runs/*.json`` and
``final-test.json``. The primary outcome is sealed-test **accuracy**, paired
within (task, replicate), the same choice v3 made after v2's mean-BCE primary
inverted an ordering.

The table that carries the study is :func:`sign_matrix`: for each algorithm and
each task, the sign of ``search - fixed`` under the unmatched protocol and under
the matched one. If the budget protocol sets the sign, the two algorithms agree
cell for cell.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .protocol import (
    ALL_TASKS,
    BLOCKS,
    FAMILY_CONDITIONS,
    FAMILY_SIZE,
    MATCHED_TO,
    REFERENCES,
    SUCCESS_THRESHOLD,
)

BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261005
CLIP_QUANTILE = 0.05

#: Per algorithm: its reference condition, its unmatched control, and its own
#: budget-matched control. The unmatched control is shared; the matched one is
#: not, because the two algorithms do not spend the same budget.
ALGORITHMS = (
    {
        "algorithm": "Backprop-NEAT",
        "reference": "bpneat",
        "unmatched": "fixed_tanh_ha",
        "matched": "fixed_tanh_matched_bpneat",
        "matched_mixed": "fixed_mixed_matched_bpneat",
    },
    {
        "algorithm": "CGP",
        "reference": "cgp",
        "unmatched": "fixed_tanh_ha",
        "matched": "fixed_tanh_matched_cgp",
        "matched_mixed": "fixed_mixed_matched_cgp",
    },
)


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
        repr_n = np.array([i["metrics"]["represented_nodes"] for i in items])
        steps = np.array([i["compute"]["gradient_steps"] for i in items])
        cands = np.array([i["compute"]["candidate_evaluations"] for i in items])
        wall = np.array([i["compute"]["wall_time_seconds"] for i in items])
        coll = np.array([i["metrics"]["collapsed"] for i in items], dtype=bool)
        active = [i["compute"].get("cgp_active_nodes") for i in items]
        active = [x for x in active if x is not None]
        neutral = [i["compute"].get("cgp_neutral_accepted") for i in items]
        neutral = [x for x in neutral if x is not None]

        row = {
            "task": task,
            "condition": cond,
            "n": len(items),
            "validation_accuracy_mean": float(va.mean()),
            "collapse_rate": float(coll.mean()),
            "causal_hidden_nodes_mean": float(causal.mean()),
            "represented_nodes_mean": float(repr_n.mean()),
            "gradient_steps_mean": float(steps.mean()),
            "candidate_evaluations_mean": float(cands.mean()),
            "wall_time_mean": float(wall.mean()),
            "cgp_active_nodes_mean": float(np.mean(active)) if active else "",
            "cgp_neutral_accepted_mean": float(np.mean(neutral)) if neutral else "",
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
    reference: str = "bpneat",
    conditions: tuple[str, ...] | None = None,
    holm_family: int | None = None,
) -> list[dict]:
    """Reference minus each condition, paired within (task, replicate).

    Reports Wilcoxon signed-rank, the median paired difference with a bootstrap
    interval, and — when ``holm_family`` is given — a Holm-corrected p value.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by = {(r["task"], r["condition"], r["replicate"]): r for r in runs}
    tasks = [t for t in ALL_TASKS if any(r["task"] == t for r in runs)]
    conds = conditions or tuple(sorted({r["condition"] for r in runs} - {reference}))

    def value(rec):
        if metric.startswith("test_"):
            t = final.get(rec["run_id"])
            return None if t is None else t[metric]
        return rec["metrics"].get(metric)

    rows = []
    for task in tasks:
        reps = sorted({r["replicate"] for r in runs if r["task"] == task})
        for cond in conds:
            diffs, pairs = [], []
            for rep in reps:
                a, b = by.get((task, reference, rep)), by.get((task, cond, rep))
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
            higher_is_better = "accuracy" in metric
            wins = int(np.sum(d > 0)) if higher_is_better else int(np.sum(d < 0))
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "comparison": f"{reference} - {cond}",
                    "reference": reference,
                    "condition": cond,
                    "n_pairs": len(d),
                    "mean_difference": float(d.mean()),
                    "median_difference": float(np.median(d)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "wilcoxon_p": p,
                    "wins": wins,
                    "per_replicate": json.dumps(pairs),
                }
            )

    if holm_family:
        order = sorted(range(len(rows)), key=lambda i: rows[i]["wilcoxon_p"])
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, rows[i]["wilcoxon_p"] * (holm_family - rank))
            running = max(running, adj)  # Holm is monotone in rank
            rows[i]["holm_p"] = running
            rows[i]["holm_significant"] = bool(running < 0.05)
    return rows


def family_effects(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Both pre-declared families, each Holm-corrected inside itself.

    The two families are never pooled: P1 answers "what does Backprop-NEAT's
    comparison say", P2 answers the same for CGP, and a test belongs to exactly
    one of them.
    """
    out: list[dict] = []
    for ref in REFERENCES:
        conds = FAMILY_CONDITIONS[ref]
        rows = paired_effects(
            runs, final, "test_accuracy", ref, conds, holm_family=FAMILY_SIZE[ref]
        )
        for r in rows:
            r["family"] = ref
            r["family_size"] = FAMILY_SIZE[ref]
            r["blocks"] = ",".join(
                sorted(b.name for b in BLOCKS
                       if ref in b.conditions and r["condition"] in b.conditions)
            )
        out.extend(rows)
    return out


def _effect(effects: list[dict], family: str, task: str, cond: str) -> dict | None:
    for e in effects:
        if e["family"] == family and e["task"] == task and e["condition"] == cond:
            return e
    return None


def _sign(e: dict | None) -> str:
    """The direction of ``reference - control``, with significance applied."""
    if e is None:
        return "n/a"
    if e.get("holm_p", e["wilcoxon_p"]) >= 0.05:
        return "ns"
    return "search>fixed" if e["median_difference"] > 0 else "fixed>search"


def sign_matrix(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """The study's headline: does the protocol or the algorithm set the sign?

    One row per (algorithm, task). Two cells: the sign of ``search - fixed``
    against the unmatched control, and against that algorithm's own
    budget-matched control. If the protocol sets the sign, the two algorithms
    produce identical columns.
    """
    effects = family_effects(runs, final)
    rows = []
    for spec in ALGORITHMS:
        for task in ALL_TASKS:
            un = _effect(effects, spec["reference"], task, spec["unmatched"])
            ma = _effect(effects, spec["reference"], task, spec["matched"])
            mx = _effect(effects, spec["reference"], task, spec["matched_mixed"])
            rows.append(
                {
                    "algorithm": spec["algorithm"],
                    "reference": spec["reference"],
                    "task": task,
                    "vs_unmatched": _sign(un),
                    "vs_matched_tanh": _sign(ma),
                    "vs_matched_mixed": _sign(mx),
                    "median_vs_unmatched": un["median_difference"] if un else "",
                    "median_vs_matched_tanh": ma["median_difference"] if ma else "",
                    "median_vs_matched_mixed": mx["median_difference"] if mx else "",
                    "reversed_by_matching": bool(
                        un and ma
                        and _sign(un) == "search>fixed"
                        and _sign(ma) == "fixed>search"
                    ),
                }
            )
    return rows


def budget_table(runs: list[dict]) -> list[dict]:
    """What each arm actually spent. Budget matching has to be auditable."""
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
                "gradient_steps_mean": float(steps.mean()),
                "gradient_steps_min": float(steps.min()),
                "gradient_steps_max": float(steps.max()),
                "target_steps_mean": float(np.mean(target)) if target else "",
                "shortfall_mean": float(np.mean(target) - steps.mean()) if target else "",
                "candidate_evaluations_mean": float(
                    np.mean([i["compute"]["candidate_evaluations"] for i in items])
                ),
            }
        )
    return rows


def cross_algorithm(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """H5: are the two algorithms close, relative to what matching does?

    Compares ``|median(cgp - bpneat)|`` against the effect of budget-matching
    within each algorithm. Both quantities come from the same paired families.
    """
    effects = family_effects(runs, final)
    rows = []
    for task in ALL_TASKS:
        between = _effect(effects, "cgp", task, "bpneat")
        bp = _effect(effects, "bpneat", task, "fixed_tanh_matched_bpneat")
        cg = _effect(effects, "cgp", task, "fixed_tanh_matched_cgp")
        if between is None or bp is None or cg is None:
            continue
        gap = abs(between["median_difference"])
        protocol = float(np.mean([abs(bp["median_difference"]), abs(cg["median_difference"])]))
        rows.append(
            {
                "task": task,
                "median_cgp_minus_bpneat": between["median_difference"],
                "holm_p_between": between.get("holm_p", between["wilcoxon_p"]),
                "between_significant": bool(
                    between.get("holm_p", between["wilcoxon_p"]) < 0.05
                ),
                "median_matching_effect_bpneat": bp["median_difference"],
                "median_matching_effect_cgp": cg["median_difference"],
                "between_over_protocol": gap / protocol if protocol else "",
                "h5_holds": bool(protocol and gap < 0.5 * protocol),
            }
        )
    return rows


def hypotheses(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Score every preregistered hypothesis by its own declared rule."""
    effects = family_effects(runs, final)
    summary = {(r["task"], r["condition"]): r for r in summarise(runs, final)}
    signs = {(r["algorithm"], r["task"]): r for r in sign_matrix(runs, final)}
    cross = {r["task"]: r for r in cross_algorithm(runs, final)}
    n_tasks = len(ALL_TASKS)
    out: list[dict] = []

    def add(h, statement, rule, got, holds, complete=True):
        # A hypothesis is only scored where the evidence for it exists. Without
        # this, a task with no data makes two "n/a" verdicts compare equal and
        # scores agreement, so an incomplete suite would report H4 as holding.
        out.append(
            {
                "hypothesis": h,
                "statement": statement,
                "decision_rule": rule,
                "observed": got if complete else f"{got} (incomplete)",
                "verdict": ("holds" if holds else "fails") if complete else "incomplete",
            }
        )

    # H1: matched beats unmatched, within the same architecture.
    h1 = 0
    h1_seen = 0
    for task in ALL_TASKS:
        a = summary.get((task, "fixed_tanh_matched_bpneat"))
        b = summary.get((task, "fixed_tanh_ha"))
        if not a or not b:
            continue
        av, bv = a.get("test_accuracy_mean"), b.get("test_accuracy_mean")
        if av in (None, "") or bv in (None, ""):
            continue
        h1_seen += 1
        h1 += int(float(av) > float(bv))
    add("H1", "the unmatched control is starved, not weak",
        "matched beats unmatched on >=4 of 5 tasks", f"{h1}/{n_tasks}", h1 >= 4,
        complete=h1_seen == n_tasks)

    # H2/H3: the reversal, per algorithm.
    for h, algo in (("H2", "Backprop-NEAT"), ("H3", "CGP")):
        rows = [signs.get((algo, t), {}) for t in ALL_TASKS]
        beats_un = sum(1 for r in rows if r.get("vs_unmatched") == "search>fixed")
        loses_ma = sum(1 for r in rows if r.get("vs_matched_tanh") == "fixed>search")
        add(h, f"{algo}'s advantage reverses under a matched budget",
            "search>unmatched on >=4 tasks and matched>search on >=3 tasks",
            f"beats unmatched {beats_un}/{n_tasks}, loses to matched {loses_ma}/{n_tasks}",
            beats_un >= 4 and loses_ma >= 3,
            complete=all(
                r.get("vs_unmatched") not in (None, "n/a")
                and r.get("vs_matched_tanh") not in (None, "n/a")
                for r in rows
            ))

    # H4: the two algorithms agree in sign, on tasks where both have a verdict.
    agree = scored = 0
    for t in ALL_TASKS:
        bp = signs.get(("Backprop-NEAT", t), {})
        cg = signs.get(("CGP", t), {})
        pairs = [
            (bp.get("vs_unmatched"), cg.get("vs_unmatched")),
            (bp.get("vs_matched_tanh"), cg.get("vs_matched_tanh")),
        ]
        if any(a in (None, "n/a") or b in (None, "n/a") for a, b in pairs):
            continue
        scored += 1
        agree += int(all(a == b for a, b in pairs))
    add("H4", "the sign is set by the budget protocol, not the algorithm",
        "both signs agree across the two algorithms on >=4 of 5 tasks",
        f"{agree}/{n_tasks}", agree >= 4, complete=scored == n_tasks)

    # H5: between-algorithm gap is small relative to the protocol effect.
    h5 = sum(1 for t in ALL_TASKS if cross.get(t, {}).get("h5_holds"))
    add("H5", "the algorithms differ less than the protocol does",
        "|median(cgp-bpneat)| < 0.5x the matching effect on >=3 of 5 tasks",
        f"{h5}/{n_tasks}", h5 >= 3, complete=len(cross) == n_tasks)

    # H6: CGP's selection vs its own null.
    null_effects = [_effect(effects, "cgp", t, "cgp_random_matched") for t in ALL_TASKS]
    beats_null = sum(
        1 for e in null_effects
        if e is not None and e.get("holm_p", e["wilcoxon_p"]) < 0.05
        and e["median_difference"] > 0
    )
    add("H6", "CGP's selection adds little over its candidate-matched null",
        "CGP beats its null on <=2 of 5 tasks", f"{beats_null}/{n_tasks}",
        beats_null <= 2, complete=all(e is not None for e in null_effects))
    return out


def operator_usage(runs: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for r in runs:
        for op, n in r["metrics"].get("causal_operators", {}).items():
            counts[(r["task"], r["condition"], op)] += n
            totals[(r["task"], r["condition"])] += n
    return [
        {
            "task": t, "condition": c, "operator": op, "count": n,
            "fraction": n / max(totals[(t, c)], 1),
        }
        for (t, c, op), n in sorted(counts.items())
    ]


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
    effects = family_effects(runs, final)
    budgets = budget_table(runs)
    ops = operator_usage(runs)
    signs = sign_matrix(runs, final) if final else []
    cross = cross_algorithm(runs, final) if final else []
    hyp = hypotheses(runs, final) if final else []

    write_csv(release_dir / "summary.csv", summary)
    write_csv(release_dir / "paired-effects.csv", effects)
    write_csv(release_dir / "budget-table.csv", budgets)
    write_csv(release_dir / "operator-usage.csv", ops)
    write_csv(release_dir / "sign-matrix.csv", signs)
    write_csv(release_dir / "cross-algorithm.csv", cross)
    write_csv(release_dir / "hypotheses.csv", hyp)

    payload = {
        "n_runs": len(runs),
        "test_evaluated": bool(final),
        "matched_arms": dict(MATCHED_TO),
        "family_size": dict(FAMILY_SIZE),
        "summary": summary,
        "paired_effects": [
            {k: v for k, v in e.items() if k != "per_replicate"} for e in effects
        ],
        "budget_table": budgets,
        "sign_matrix": signs,
        "cross_algorithm": cross,
        "hypotheses": hyp,
    }
    (release_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    progress(
        f"{len(runs)} runs -> summary, paired-effects, budget-table, "
        f"operator-usage, sign-matrix, cross-algorithm, hypotheses"
    )
    return payload
