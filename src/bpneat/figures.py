"""Publication figures, rendered from the raw records.

Every figure is derived from ``raw/runs/*.json`` — nothing is recomputed and no
model is retrained — so a figure and the CSV beside it cannot disagree.

Colour follows the validated categorical order and is never the only carrier of
identity: bars are directly labelled and every figure has a CSV table view in
the release, which is also the required relief for the three slots that sit
below 3:1 contrast on the light surface.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import load_final_test, load_runs, operation_usage, paired_effects, summarise  # noqa: E402
from .conditions import CORE_CONDITIONS, SUCCESS_THRESHOLD  # noqa: E402
from .datasets import make_bundle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2df"

# Validated categorical order (light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CONDITION_COLOR = dict(zip(CORE_CONDITIONS, SERIES))
LABEL = {
    "backprop_neat": "Backprop-NEAT",
    "homogeneous_tanh": "Homog. tanh",
    "evolution_only": "Evolution only",
    "random_search": "Random arch.",
    "fixed_mlp": "Fixed MLP",
    "logistic": "Logistic",
}
TASK_LABEL = {"xor": "XOR", "circle": "Circles", "spiral": "Spirals"}
TASKS = ("xor", "circle", "spiral")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _cat_axis(ax, x, conds) -> None:
    """Rotated condition labels, anchored so they do not drift into the neighbour."""
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABEL[c] for c in conds], rotation=30, ha="right",
        rotation_mode="anchor", fontsize=8,
    )
    ax.set_xlim(-0.7, len(conds) - 0.3)


def _fig(nrows=1, ncols=1, figsize=(10, 4.2)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path: Path, bottom: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bottom is None:
        fig.tight_layout()
    else:
        # Rotated category labels overflow the axes, so reserve the margin
        # explicitly instead of letting tight_layout fight the annotations.
        fig.subplots_adjust(bottom=bottom, top=0.84, left=0.075, right=0.985, wspace=0.22)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------


def task_geometries(out: Path) -> Path:
    fig, axes = _fig(1, 3, figsize=(10.5, 3.6))
    for ax, task in zip(axes, TASKS):
        b = make_bundle(task, seed=8101)
        X, y = b.train.X, b.train.y
        for cls, colour in ((0, SERIES[0]), (1, SERIES[1])):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=13, c=colour, linewidths=0, alpha=0.85,
                       label=f"class {cls}")
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.suptitle("ha2016 task geometries — raw coordinates, noise 0.5, 200 training points",
                 color=INK, fontsize=11)
    return _save(fig, out / "task-geometries.png")


def validation_performance(rows: list[dict], out: Path, track: str) -> Path:
    fig, axes = _fig(1, 3, figsize=(11.5, 4.4))
    by = {(r["task"], r["condition"]): r for r in rows}
    for ax, task in zip(axes, TASKS):
        conds = [c for c in CORE_CONDITIONS if (task, c) in by]
        vals = [by[(task, c)]["validation_accuracy_mean"] for c in conds]
        sds = [by[(task, c)]["validation_accuracy_sd"] for c in conds]
        x = np.arange(len(conds))
        ax.bar(x, vals, width=0.68, color=[CONDITION_COLOR[c] for c in conds],
               yerr=sds, ecolor=INK_2, capsize=3, error_kw={"linewidth": 1.1})
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
        ax.axhline(SUCCESS_THRESHOLD[task], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(-0.45, SUCCESS_THRESHOLD[task] + 0.015, "success threshold",
                ha="left", fontsize=7, color=INK_2)
        _cat_axis(ax, x, conds)
        ax.set_ylim(0, 1.16)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=11)
    axes[0].set_ylabel("validation accuracy", color=INK_2, fontsize=9)
    fig.suptitle(f"Track {track}: validation accuracy of the selected champion "
                 f"(mean ± SD over replicates)", color=INK, fontsize=11)
    return _save(fig, out / f"validation-performance-track-{track.lower()}.png", bottom=0.30)


def track_contrast(rows_a: list[dict], rows_b: list[dict], out: Path) -> Path:
    """The propagation finding: Ha's exact rule against settled propagation."""
    fig, axes = _fig(1, 3, figsize=(11.5, 4.4))
    a = {(r["task"], r["condition"]): r for r in rows_a}
    b = {(r["task"], r["condition"]): r for r in rows_b}
    for ax, task in zip(axes, TASKS):
        conds = [c for c in CORE_CONDITIONS if (task, c) in a and (task, c) in b]
        x = np.arange(len(conds))
        va = [a[(task, c)]["validation_accuracy_mean"] for c in conds]
        vb = [b[(task, c)]["validation_accuracy_mean"] for c in conds]
        ax.bar(x - 0.2, va, width=0.38, color=SERIES[0], label="Track A — ha2016")
        ax.bar(x + 0.2, vb, width=0.38, color=SERIES[1], label="Track B — settled")
        for xi, v in zip(x - 0.2, va):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        for xi, v in zip(x + 0.2, vb):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        _cat_axis(ax, x, conds)
        ax.set_ylim(0, 1.18)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=11)
    axes[0].set_ylabel("validation accuracy", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    fig.suptitle("Propagation rule changes what evolution can discover", color=INK, fontsize=11)
    return _save(fig, out / "propagation-track-contrast.png", bottom=0.30)


def paired_effects_figure(effects: list[dict], out: Path, track: str) -> Path:
    rows = [e for e in effects if e["metric"] == "validation_accuracy"]
    if not rows:
        return None
    fig, axes = _fig(1, 3, figsize=(11.5, 4.0))
    for ax, task in zip(axes, TASKS):
        items = [e for e in rows if e["task"] == task]
        items.sort(key=lambda e: e["mean_difference"])
        y = np.arange(len(items))
        for yi, e in zip(y, items):
            lo, hi = e["ci95_low"], e["ci95_high"]
            colour = SERIES[0] if e["mean_difference"] > 0 else SERIES[1]
            ax.plot([lo, hi], [yi, yi], color=colour, linewidth=2.4, solid_capstyle="round")
            ax.plot(e["mean_difference"], yi, "o", color=colour, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.6)
            ax.text(hi + 0.012, yi, f"{e['mean_difference']:+.3f}", va="center",
                    fontsize=8, color=INK)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(
            ["vs " + LABEL[e["comparison"].split(" - ")[1]] for e in items], fontsize=8.5
        )
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_xlabel("paired difference in validation accuracy\n(Backprop-NEAT − control)",
                       color=INK_2, fontsize=8.5)
    fig.suptitle(f"Track {track}: paired within-replicate differences, 95% bootstrap interval",
                 color=INK, fontsize=11)
    return _save(fig, out / f"paired-condition-effects-track-{track.lower()}.png")


def represented_vs_causal(rows: list[dict], out: Path, track: str) -> Path:
    """H5: represented structure is not the computation that reaches the output."""
    fig, axes = _fig(1, 3, figsize=(11.5, 4.2))
    by = {(r["task"], r["condition"]): r for r in rows}
    for ax, task in zip(axes, TASKS):
        conds = [c for c in CORE_CONDITIONS if (task, c) in by]
        x = np.arange(len(conds))
        rep = [by[(task, c)]["represented_nodes_mean"] - 4 for c in conds]
        cau = [by[(task, c)]["causal_hidden_nodes_mean"] for c in conds]
        ax.bar(x - 0.2, rep, width=0.38, color=SERIES[0], label="represented hidden")
        ax.bar(x + 0.2, cau, width=0.38, color=SERIES[2], label="causally active")
        for xi, v in zip(x - 0.2, rep):
            ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
        for xi, v in zip(x + 0.2, cau):
            ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
        _cat_axis(ax, x, conds)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=11)
    axes[0].set_ylabel("hidden nodes", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle(f"Track {track}: represented vs causally active hidden structure",
                 color=INK, fontsize=11)
    return _save(fig, out / f"represented-vs-causal-track-{track.lower()}.png", bottom=0.30)


def operator_usage(runs: list[dict], out: Path, track: str) -> Path:
    usage = [u for u in operation_usage(runs) if u["condition"] == "backprop_neat"]
    if not usage:
        return None
    ops = sorted({u["operator"] for u in usage})
    fig, ax = _fig(figsize=(8.6, 4.2))
    frac = defaultdict(dict)
    for u in usage:
        frac[u["task"]][u["operator"]] = u["fraction"]
    x = np.arange(len(TASKS))
    bottom = np.zeros(len(TASKS))
    palette = (SERIES * 3)[: len(ops)]
    for op, colour in zip(ops, palette):
        vals = np.array([frac[t].get(op, 0.0) for t in TASKS])
        ax.bar(x, vals, width=0.6, bottom=bottom, color=colour, label=op,
               edgecolor=SURFACE, linewidth=2)
        for xi, (v, b0) in enumerate(zip(vals, bottom)):
            if v > 0.07:
                ax.text(xi, b0 + v / 2, op, ha="center", va="center", fontsize=8, color=INK)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], fontsize=9.5)
    ax.set_ylabel("share of causal hidden operators", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.set_title(f"Track {track}: which operators reach the output, by task "
                 f"(Backprop-NEAT champions)", color=INK, fontsize=11)
    return _save(fig, out / f"operator-usage-track-{track.lower()}.png")


def performance_vs_compute(rows: list[dict], out: Path, track: str) -> Path:
    """Every point is directly labelled, so identity never rests on colour."""
    fig, axes = _fig(1, 3, figsize=(11.5, 4.2))
    by = {(r["task"], r["condition"]): r for r in rows}
    for ax, task in zip(axes, TASKS):
        for c in CORE_CONDITIONS:
            r = by.get((task, c))
            if r is None:
                continue
            xs = max(r["gradient_steps_mean"], 1)
            ax.plot(xs, r["validation_accuracy_mean"], "o", markersize=9,
                    color=CONDITION_COLOR[c], markeredgecolor=SURFACE, markeredgewidth=1.6)
            ax.annotate(LABEL[c], (xs, r["validation_accuracy_mean"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7.5, color=INK)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=11)
        ax.set_xlabel("realized gradient steps", color=INK_2, fontsize=8.5)
        ax.set_ylim(0.35, 1.12)
    axes[0].set_ylabel("validation accuracy", color=INK_2, fontsize=9)
    fig.suptitle(f"Track {track}: performance against realized compute", color=INK, fontsize=11)
    return _save(fig, out / f"performance-vs-compute-track-{track.lower()}.png")


def build_all(root: Path, progress=print) -> list[Path]:
    """Render every figure that the available records support."""
    root = Path(root)
    out = root / "figures"
    written = [task_geometries(out)]

    per_track = {}
    for track, sub in (("A", "track-a"), ("B", "track-b")):
        d = root / sub
        if not (d / "raw" / "runs").exists():
            continue
        runs = load_runs(d)
        if not runs:
            continue
        final = load_final_test(d)
        rows = summarise(runs, final)
        per_track[track] = rows
        written.append(validation_performance(rows, out, track))
        written.append(represented_vs_causal(rows, out, track))
        written.append(performance_vs_compute(rows, out, track))
        f = operator_usage(runs, out, track)
        if f:
            written.append(f)
        f = paired_effects_figure(paired_effects(runs, final, "validation_accuracy"), out, track)
        if f:
            written.append(f)

    if "A" in per_track and "B" in per_track:
        written.append(track_contrast(per_track["A"], per_track["B"], out))
    return [w for w in written if w]
