"""Derive every v3 table from the raw records.

Nothing here retrains a model or reads data; it reads ``raw/runs/*.json`` and
``final-test.json``. The primary outcome is sealed-test **accuracy**, paired
within replicate, because v2's mean-BCE primary inverted the XOR ranking under
two confident errors (erratum E2).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .protocol import BLOCKS, PRIMARY_FAMILY_SIZE, SUCCESS_THRESHOLD

REFERENCE = "backprop_neat"
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261004
CLIP_QUANTILE = 0.05


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
        inten = [i["compute"].get("selection_intensity") for i in items]
        inten = [x for x in inten if x is not None]

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
            "selection_intensity_mean": float(np.mean(inten)) if inten else "",
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
    reference: str = REFERENCE,
    conditions: tuple[str, ...] | None = None,
    holm_family: int | None = None,
) -> list[dict]:
    """Reference minus each condition, paired within (task, replicate).

    Reports Wilcoxon signed-rank, the median paired difference with a bootstrap
    interval, and — when ``holm_family`` is given — a Holm-corrected p value.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by = {(r["task"], r["condition"], r["replicate"]): r for r in runs}
    tasks = sorted({r["task"] for r in runs})
    conds = conditions or tuple(
        sorted({r["condition"] for r in runs} - {reference})
    )

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


def block_effects(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Per-block comparisons. Only block A enters the Holm family."""
    out: list[dict] = []
    for block in BLOCKS:
        subset = [r for r in runs if block.name in r["blocks"] and r["task"] in block.tasks]
        others = tuple(c for c in block.conditions if c != REFERENCE)
        rows = paired_effects(
            subset, final, "test_accuracy", REFERENCE, others,
            holm_family=PRIMARY_FAMILY_SIZE if block.name == "A" else None,
        )
        for r in rows:
            r["block"] = block.name
            r["confirmatory"] = block.name == "A"
        out.extend(rows)
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


def selection_dose_response(runs: list[dict], final: dict[str, dict]) -> list[dict]:
    """Block C: outcome against realized selection intensity."""
    block = next(b for b in BLOCKS if b.name == "C")
    rows = []
    for task in block.tasks:
        for cond in block.conditions:
            items = [r for r in runs if r["task"] == task and r["condition"] == cond]
            if not items:
                continue
            inten = [r["compute"].get("selection_intensity") for r in items]
            inten = [x for x in inten if x is not None]
            tests = [final[r["run_id"]] for r in items if r["run_id"] in final]
            rows.append(
                {
                    "task": task,
                    "condition": cond,
                    "selector": items[0]["config"]["selector"],
                    "n": len(items),
                    "selection_intensity_mean": float(np.mean(inten)) if inten else "",
                    "collapse_rate": float(np.mean([r["metrics"]["collapsed"] for r in items])),
                    "causal_hidden_nodes_mean": float(
                        np.mean([r["metrics"]["causal_hidden_nodes"] for r in items])
                    ),
                    "represented_nodes_mean": float(
                        np.mean([r["metrics"]["represented_nodes"] for r in items])
                    ),
                    "test_accuracy_mean": float(
                        np.mean([t["test_accuracy"] for t in tests])
                    ) if tests else "",
                }
            )
    return sorted(rows, key=lambda r: (r["task"], r["selection_intensity_mean"] or 0))


#: The five v2 headline claims, each with the control it actually rests on.
#: ``controls`` maps an evaluator setting to the condition that instantiates
#: the claim's control under that setting. ``v2`` draws on the committed v2
#: release; every other column draws on v3.
V2_CLAIMS = (
    {
        "claim": "C1",
        "statement": "Backprop-NEAT beats a fixed MLP on spirals",
        "task": "spiral",
        "kind": "accuracy",
        "controls": {
            "v2 as run": ("v2", "fixed_mlp"),
            "budget-matched tanh": ("v3", "fixed_mlp_tanh_matched"),
            "sin control": ("v3", "fixed_mlp_sin_matched"),
            "mixed control": ("v3", "fixed_mlp_mixed_matched"),
        },
    },
    {
        "claim": "C2",
        "statement": "Gradient learning is complementary to topology search",
        "task": "spiral",
        "kind": "accuracy",
        "controls": {
            "v2 as run": ("v2", "evolution_only"),
            "v3 re-run": ("v3", "evolution_only"),
        },
    },
    {
        "claim": "C3",
        "statement": "Operator diversity contributes",
        "task": "spiral",
        "kind": "accuracy",
        "controls": {
            "v2 as run": ("v2", "homogeneous_tanh"),
            "v3 re-run": ("v3", "homogeneous_tanh"),
        },
    },
    {
        "claim": "C4",
        "statement": "Few causal nodes beat a 65-unit MLP",
        "task": "spiral",
        "kind": "efficiency",
        "controls": {
            "v2 as run": ("v2", "fixed_mlp"),
            "budget-matched tanh": ("v3", "fixed_mlp_tanh_matched"),
            "sin control": ("v3", "fixed_mlp_sin_matched"),
            "mixed control": ("v3", "fixed_mlp_mixed_matched"),
        },
    },
    {
        "claim": "C5",
        "statement": "Evolutionary selection beats random sampling",
        "task": "spiral",
        "kind": "accuracy",
        "controls": {
            "v2 as run": ("v2", "random_search"),
            "candidate-matched": ("v3", "random_search_matched"),
        },
    },
)

MATRIX_COLUMNS = (
    "v2 as run",
    "v3 re-run",
    "budget-matched tanh",
    "sin control",
    "mixed control",
    "candidate-matched",
)


#: The v2 release is located from the installed package, not from the release
#: directory being analysed. `verify` rebuilds tables in a scratch copy whose
#: parent has no v2 release beside it, so a relative lookup would silently drop
#: the "v2 as run" column and make the output non-reproducible.
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_v2_track_b(_root: Path | None = None) -> tuple[list[dict], dict[str, dict]]:
    """The committed v2 release, for the 'v2 as run' column."""
    v2 = REPO_ROOT / "results" / "backprop-neat-v2" / "track-b"
    if not (v2 / "final-test.json").exists():
        return [], {}
    runs = [json.loads(q.read_text()) for q in (v2 / "raw" / "runs").glob("*.json")]
    final = {
        r["run_id"]: r for r in json.loads((v2 / "final-test.json").read_text())["results"]
    }
    return runs, final


def _verdict(runs, final, task, control, kind, family: int) -> str:
    """Does the claim hold against this control, paired within replicate?

    Significance uses the **Holm-corrected** p over the source's own
    pre-declared family, not the raw Wilcoxon p. The difference is not
    cosmetic: on spirals, Backprop-NEAT beats candidate-matched random search
    at raw p < 0.05 and fails at Holm-corrected p = 0.385.
    """
    conds = tuple(sorted({r["condition"] for r in runs} - {REFERENCE}))
    eff = paired_effects(
        [r for r in runs if r["task"] == task], final, "test_accuracy",
        REFERENCE, conds, holm_family=family,
    )
    eff = [e for e in eff if e["condition"] == control]
    if not eff:
        return "n/a"
    e = eff[0]
    p_value = e.get("holm_p", e["wilcoxon_p"])
    if kind == "efficiency":
        # The claim is that fewer causal nodes do at least as well. It fails if
        # the reference is significantly *worse* on accuracy, regardless of size.
        if p_value < 0.05 and e["median_difference"] < 0:
            return "reversed"
        ref_sz = np.mean([
            r["metrics"]["causal_hidden_nodes"]
            for r in runs if r["task"] == task and r["condition"] == REFERENCE
        ])
        ctl_sz = np.mean([
            r["metrics"]["causal_hidden_nodes"]
            for r in runs if r["task"] == task and r["condition"] == control
        ])
        if ref_sz >= ctl_sz:
            return "reversed"
        return "supported" if p_value < 0.05 else "not significant"
    if p_value >= 0.05:
        return "not significant"
    return "supported" if e["median_difference"] > 0 else "reversed"


def stability_matrix(runs: list[dict], final: dict[str, dict],
                     release_dir: Path | None = None) -> list[dict]:
    """Rows: v2 headline claims. Columns: evaluator settings. Cells: verdict.

    Each claim is tested against *its own* control, re-instantiated under each
    setting. Columns that do not bear on a claim are ``n/a``; the matrix is
    deliberately sparse rather than repeating one comparison five times.
    """
    v2_runs, v2_final = _load_v2_track_b()
    out = []
    for spec in V2_CLAIMS:
        row = {"claim": spec["claim"], "statement": spec["statement"], "task": spec["task"]}
        for col in MATRIX_COLUMNS:
            entry = spec["controls"].get(col)
            if entry is None:
                row[col] = "n/a"
                continue
            source, control = entry
            if source == "v2":
                # v2's analogous family: its five core controls on three tasks.
                row[col] = (
                    _verdict(v2_runs, v2_final, spec["task"], control, spec["kind"], 15)
                    if v2_runs else "n/a"
                )
            else:
                row[col] = _verdict(
                    runs, final, spec["task"], control, spec["kind"],
                    PRIMARY_FAMILY_SIZE,
                )
        out.append(row)
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

    summary = summarise(runs, final)
    effects = block_effects(runs, final)
    dose = selection_dose_response(runs, final)
    ops = operator_usage(runs)
    matrix = stability_matrix(runs, final, release_dir) if final else []

    write_csv(release_dir / "summary.csv", summary)
    write_csv(release_dir / "paired-effects.csv", effects)
    write_csv(release_dir / "selection-dose-response.csv", dose)
    write_csv(release_dir / "operator-usage.csv", ops)
    write_csv(release_dir / "stability-matrix.csv", matrix)

    payload = {
        "n_runs": len(runs),
        "test_evaluated": bool(final),
        "summary": summary,
        "paired_effects": [
            {k: v for k, v in e.items() if k != "per_replicate"} for e in effects
        ],
        "selection_dose_response": dose,
        "stability_matrix": matrix,
    }
    (release_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    progress(
        f"{len(runs)} runs -> summary, paired-effects, selection-dose-response, "
        f"operator-usage, stability-matrix"
    )
    return payload
