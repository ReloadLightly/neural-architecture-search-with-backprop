"""v3 figures, rendered from the raw records.

Colour follows the same validated categorical order as v2 (adjacent-pair CVD
ΔE 9.1, normal-vision 19.6) and never carries identity alone: every bar is
directly labelled and every figure has a CSV beside it in the release.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import (  # noqa: E402
    REFERENCE,
    block_effects,
    load,
    selection_dose_response,
    stability_matrix,
    summarise,
)
from .datasets import TASKS, make_bundle  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GOOD, BAD, NEUTRAL = "#1baf7a", "#e34948", "#c9c8c3"

LABEL = {
    "backprop_neat": "Backprop-NEAT",
    "fixed_mlp_tanh_ha": "Fixed MLP (Ha learner)",
    "fixed_mlp_tanh_matched": "Fixed MLP tanh (matched)",
    "fixed_mlp_sin_matched": "Fixed MLP sin (matched)",
    "fixed_mlp_mixed_matched": "Fixed MLP mixed (matched)",
    "random_search_matched": "Random arch. (matched)",
    "homogeneous_tanh": "Homogeneous tanh",
    "evolution_only": "Evolution only",
    "prop_ha_fit_val": "ha2016 / validation",
    "prop_ha_fit_train": "ha2016 / train",
    "prop_settled_fit_train": "settled / train",
    "baldwinian": "Baldwinian",
}
TASK_LABEL = {"xor": "XOR", "circle": "Circles", "spiral": "Spirals",
              "checkerboard": "Checkerboard", "spiral3": "3-arm spiral"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(nrows=1, ncols=1, figsize=(10, 4.2)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if kw:
        fig.subplots_adjust(**kw)
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def task_geometries(out: Path) -> Path:
    fig, axes = _fig(1, 5, figsize=(15, 3.3))
    for ax, task in zip(axes, TASKS):
        b = make_bundle(task, seed=30001)
        for cls, c in ((0, SERIES[0]), (1, SERIES[1])):
            m = b.train.y == cls
            ax.scatter(b.train.X[m, 0], b.train.X[m, 1], s=11, c=c, linewidths=0, alpha=0.85)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(False)
    fig.suptitle("v3 task geometries — the three v2 tasks plus two harder ones",
                 color=INK, fontsize=11)
    return _save(fig, out / "task-geometries.png")


def stability(matrix: list[dict], out: Path) -> Path | None:
    """The central figure: does each v2 claim survive each evaluator setting?"""
    if not matrix:
        return None
    cols = [k for k in matrix[0] if k not in ("claim", "statement", "task")]
    fig, ax = plt.subplots(figsize=(13, 0.9 * len(matrix) + 2.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    colour = {"supported": GOOD, "reversed": BAD, "not significant": NEUTRAL, "n/a": "#f0efec"}
    for r, row in enumerate(matrix):
        for c, col in enumerate(cols):
            v = row[col]
            ax.add_patch(plt.Rectangle((c, -r), 1, 1, facecolor=colour.get(v, NEUTRAL),
                                       edgecolor=SURFACE, linewidth=3))
            short = {"supported": "supported", "reversed": "REVERSED",
                     "not significant": "n.s.", "n/a": "—"}[v]
            ax.text(c + 0.5, -r + 0.5, short, ha="center", va="center",
                    fontsize=8.5, color=INK if v != "reversed" else "#ffffff")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(-len(matrix) + 1, 1)
    ax.set_xticks([c + 0.5 for c in range(len(cols))])
    ax.set_xticklabels(cols, rotation=28, ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks([-r + 0.5 for r in range(len(matrix))])
    ax.set_yticklabels([f"{m['claim']}  {m['statement']}" for m in matrix],
                       fontsize=8.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Conclusion stability: which v2 claims survive which evaluator",
                 color=INK, fontsize=12, pad=14)
    # C2 and C3 compare against ablations that share the candidate budget by
    # construction, so their row is identical across control definitions. That
    # invariance is the point, not a rendering artifact.
    fig.text(0.30, 0.055,
             "Columns are control definitions. Rows C2 and C3 compare against "
             "budget-matched ablations,\nso they are invariant across columns by "
             "construction — that invariance is the finding.",
             fontsize=8, color=INK2, ha="left")
    return _save(fig, out / "stability-matrix.png", bottom=0.30, left=0.30, top=0.88, right=0.99)


#: Block A only. The compute figure is about controls and budgets; the
#: propagation, selection and inheritance conditions belong to other blocks.
#: Eight conditions over a six-hue validated palette. Hues are never cycled —
#: a repeated hue is disambiguated by marker shape, so every condition has a
#: unique (colour, marker) pair and identity never rests on colour alone.
BLOCK_A_STYLE = {
    "backprop_neat": (0, "o"),
    "fixed_mlp_tanh_ha": (1, "o"),
    "fixed_mlp_tanh_matched": (2, "o"),
    "fixed_mlp_sin_matched": (3, "o"),
    "fixed_mlp_mixed_matched": (4, "o"),
    "random_search_matched": (5, "o"),
    "homogeneous_tanh": (0, "s"),
    "evolution_only": (1, "s"),
}

BLOCK_A = (
    "backprop_neat",
    "fixed_mlp_tanh_ha",
    "fixed_mlp_tanh_matched",
    "fixed_mlp_sin_matched",
    "fixed_mlp_mixed_matched",
    "random_search_matched",
    "homogeneous_tanh",
    "evolution_only",
)


def budget_vs_accuracy(summary: list[dict], out: Path) -> Path:
    """Block A: accuracy against the compute each condition actually spent.

    A shared legend carries identity; only the two conditions that make the
    point — v2's starved control and the best matched one — are labelled in
    place, because labelling eight clustered points makes all eight unreadable.
    """
    by = {(r["task"], r["condition"]): r for r in summary}
    fig, axes = _fig(1, 5, figsize=(16, 4.4))
    handles: dict[str, object] = {}
    for ax, task in zip(axes, TASKS):
        best_matched = max(
            (c for c in BLOCK_A if c.endswith("_matched") and (task, c) in by),
            key=lambda c: by[(task, c)].get("test_accuracy_mean", 0.0),
            default=None,
        )
        for c in BLOCK_A:
            r = by.get((task, c))
            if r is None or "test_accuracy_mean" not in r:
                continue
            slot, marker = BLOCK_A_STYLE[c]
            x = max(r["gradient_steps_mean"], 1)
            y = r["test_accuracy_mean"]
            (h,) = ax.plot(x, y, marker, markersize=9, color=SERIES[slot],
                           markeredgecolor=SURFACE, markeredgewidth=1.5)
            handles.setdefault(LABEL[c], h)
            if c in ("fixed_mlp_tanh_ha", best_matched):
                ax.annotate(
                    f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8, color=INK,
                )
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.set_xlabel("realized gradient steps", color=INK2, fontsize=8)
        ax.set_ylim(0.45, 1.08)
    axes[0].set_ylabel("sealed-test accuracy", color=INK2, fontsize=9)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8.5,
               labelcolor=INK2, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Block A: accuracy against realized compute — v2's starved control sits at the left edge",
        color=INK, fontsize=11,
    )
    return _save(fig, out / "budget-vs-accuracy.png",
                 bottom=0.26, top=0.88, left=0.05, right=0.99, wspace=0.18)


def block_a_effects(effects: list[dict], out: Path) -> Path | None:
    rows = [e for e in effects if e.get("block") == "A"]
    if not rows:
        return None
    tasks = [t for t in TASKS if any(e["task"] == t for e in rows)]
    fig, axes = _fig(1, len(tasks), figsize=(3.4 * len(tasks) + 2, 4.4))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        items = sorted([e for e in rows if e["task"] == task],
                       key=lambda e: e["median_difference"])
        y = np.arange(len(items))
        for yi, e in zip(y, items):
            sig = e.get("holm_significant", False)
            col = GOOD if e["median_difference"] > 0 else BAD
            col = col if sig else NEUTRAL
            ax.plot([e["ci95_low"], e["ci95_high"]], [yi, yi], color=col, linewidth=2.6,
                    solid_capstyle="round")
            ax.plot(e["median_difference"], yi, "o", markersize=7, color=col,
                    markeredgecolor=SURFACE, markeredgewidth=1.4)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL.get(e["condition"], e["condition"]) for e in items],
                           fontsize=7.5)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.suptitle("Block A: median paired difference in sealed-test accuracy "
                 "(Backprop-NEAT − control); grey = not significant after Holm",
                 color=INK, fontsize=10.5)
    return _save(fig, out / "block-a-paired-effects.png")


def dose_response(dose: list[dict], out: Path) -> Path | None:
    if not dose:
        return None
    tasks = sorted({d["task"] for d in dose})
    fig, axes = _fig(1, len(tasks) * 2, figsize=(5.2 * len(tasks), 4.0))
    axes = np.atleast_1d(axes)
    for ti, task in enumerate(tasks):
        items = [d for d in dose if d["task"] == task and d["selection_intensity_mean"] != ""]
        items.sort(key=lambda d: d["selection_intensity_mean"])
        x = [d["selection_intensity_mean"] for d in items]
        for j, (key, lab) in enumerate((("collapse_rate", "collapse rate"),
                                        ("causal_hidden_nodes_mean", "causal hidden nodes"))):
            ax = axes[ti * 2 + j]
            ax.plot(x, [d[key] for d in items], "o-", color=SERIES[j], linewidth=2, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.3)
            for d in items:
                if d["selector"] in ("roulette_s0.01", "v1_truncation"):
                    ax.annotate("Ha" if "roulette" in d["selector"] else "v1",
                                (d["selection_intensity_mean"], d[key]),
                                textcoords="offset points", xytext=(0, 10),
                                ha="center", fontsize=8, color=INK)
            ax.set_xlabel("realized selection intensity", color=INK2, fontsize=8)
            ax.set_ylabel(lab, color=INK2, fontsize=8.5)
            ax.set_title(f"{TASK_LABEL[task]} — {lab}", color=INK, fontsize=9.5)
    fig.suptitle("Block C: selection pressure against collapse and causal size",
                 color=INK, fontsize=11)
    return _save(fig, out / "selection-dose-response.png")


def propagation_grid(summary: list[dict], out: Path) -> Path | None:
    """Block B: the 2x2 that v2 could not separate."""
    cells = {
        ("ha2016", "train"): "prop_ha_fit_train",
        ("ha2016", "validation"): "prop_ha_fit_val",
        ("settled", "train"): "prop_settled_fit_train",
        ("settled", "validation"): REFERENCE,
    }
    by = {(r["task"], r["condition"]): r for r in summary}
    tasks = [t for t in ("xor", "circle", "spiral")
             if all((t, c) in by for c in cells.values())]
    if not tasks:
        return None
    fig, axes = _fig(1, len(tasks), figsize=(4.0 * len(tasks) + 1, 4.2))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        labels, accs, colls = [], [], []
        for (prop, split), cond in cells.items():
            r = by[(task, cond)]
            labels.append(f"{prop}\n{split}")
            accs.append(r.get("test_accuracy_mean", r["validation_accuracy_mean"]))
            colls.append(r["collapse_rate"])
        x = np.arange(len(labels))
        ax.bar(x, accs, width=0.62, color=SERIES[0])
        for xi, (a, c) in enumerate(zip(accs, colls)):
            ax.text(xi, a + 0.02, f"{a:.2f}", ha="center", fontsize=8, color=INK)
            if c > 0:
                ax.text(xi, 0.04, f"collapse {c:.0%}", ha="center", fontsize=7, color="#ffffff")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("sealed-test accuracy", color=INK2, fontsize=9)
    fig.suptitle("Block B: propagation × fitness split, separated", color=INK, fontsize=11)
    return _save(fig, out / "propagation-grid.png")


def build_all(release_dir: Path, progress=print) -> list[Path]:
    release_dir = Path(release_dir)
    out = release_dir / "figures"
    runs, final = load(release_dir)
    summary = summarise(runs, final)
    effects = block_effects(runs, final)
    written = [task_geometries(out), budget_vs_accuracy(summary, out)]
    for fn in (
        lambda: stability(stability_matrix(runs, final, release_dir) if final else [], out),
        lambda: block_a_effects(effects, out),
        lambda: dose_response(selection_dose_response(runs, final), out),
        lambda: propagation_grid(summary, out),
    ):
        p = fn()
        if p:
            written.append(p)
    for p in written:
        progress(f"wrote {p}")
    return written
