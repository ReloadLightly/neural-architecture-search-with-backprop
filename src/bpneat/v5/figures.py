"""v5 figures: what NEAT's mechanisms did to the topologies and to the score.

Same conventions as v3 and v4 — the validated categorical order (adjacent-pair
CVD ΔE 9.1, normal-vision 19.6), hues assigned in fixed order and never cycled,
identity never carried by colour alone, and a CSV beside every figure in the
release.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import (  # noqa: E402
    REFERENCE_CHAMPION,
    complexity,
    hypotheses,
    load,
    paired_effects,
    summarise,
)
from .protocol import ALL_CONDITIONS, ALL_TASKS, REFERENCE  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GOOD, BAD, NEUTRAL = "#1baf7a", "#e34948", "#c9c8c3"

LABEL = {
    "neat_reference": "NEAT (reference)",
    "neat_complexify": "+ higher mutation rates",
    "neat_no_penalty": "− complexity penalty",
    "neat_complexify_no_penalty": "both",
    "neat_no_speciation": "− speciation",
    "neat_no_crossover": "− crossover",
    "neat_deep_narrow": "deep & narrow",
    "fixed_mixed_matched": "fixed net (matched budget)",
}
TASK_LABEL = {"spiral": "Spirals", "checkerboard": "Checkerboard", "spiral3": "3-arm spiral"}

#: One (hue, marker, linestyle) per condition. Eight series over a six-hue
#: palette, so the two repeats are separated by shape and dash, never by hue
#: alone.
STYLE = {
    "neat_reference": (SERIES[0], "o", "-"),
    "neat_complexify": (SERIES[1], "s", "-"),
    "neat_no_penalty": (SERIES[2], "^", "-"),
    "neat_complexify_no_penalty": (SERIES[3], "D", "-"),
    "neat_no_speciation": (SERIES[4], "v", "-"),
    "neat_no_crossover": (SERIES[5], "P", "-"),
    "neat_deep_narrow": (SERIES[0], "X", "--"),
    "fixed_mixed_matched": (SERIES[1], "*", ":"),
}


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


def fig_complexification(runs: list[dict], out: Path) -> Path:
    """The signature plot: does the topology actually augment, and how fast?

    Mean represented nodes across the population, per generation, averaged over
    replicates. The dashed marker is the published Figure 10.3 champion, shown
    as a scale reference and never pooled with anything measured here.
    """
    fig, axes = _fig(1, len(ALL_TASKS), figsize=(14, 4.4))
    for ax, task in zip(np.atleast_1d(axes).ravel(), ALL_TASKS):
        for cond in ALL_CONDITIONS:
            if cond == "fixed_mixed_matched":
                continue  # a fixed network has no trajectory
            series = [
                r["history"] for r in runs if r["task"] == task and r["condition"] == cond
            ]
            if not series:
                continue
            n = min(len(h) for h in series)
            gens = [series[0][i]["generation"] for i in range(n)]
            vals = np.mean(
                [[h[i]["mean_nodes"] for i in range(n)] for h in series], axis=0
            )
            colour, marker, dash = STYLE[cond]
            ax.plot(gens, vals, color=colour, linewidth=2.0, linestyle=dash,
                    marker=marker, markevery=max(n // 6, 1), markersize=5,
                    label=LABEL[cond])
        if task == REFERENCE_CHAMPION["task"]:
            ax.axhline(REFERENCE_CHAMPION["nodes"], color=INK2, linewidth=1.2,
                       linestyle=(0, (4, 3)))
            ax.text(0.98, REFERENCE_CHAMPION["nodes"], " Ha (2016) champion: 34 nodes",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=7.5, color=INK2)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.set_xlabel("generation", fontsize=8, color=INK2)
        if task == ALL_TASKS[0]:
            ax.set_ylabel("mean nodes in the population", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(
        "Do the topologies augment? Population size over the run, by mechanism",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.85, bottom=0.13, left=0.06, right=0.99, wspace=0.16)


def fig_size_vs_accuracy(comp: list[dict], out: Path) -> Path:
    """Does a bigger topology buy accuracy? One point per (task, condition)."""
    fig, axes = _fig(1, len(ALL_TASKS), figsize=(13, 4.3))
    for ax, task in zip(np.atleast_1d(axes).ravel(), ALL_TASKS):
        rows = [r for r in comp if r["task"] == task and r["test_accuracy_mean"] != ""]
        for r in rows:
            colour, marker, _ = STYLE[r["condition"]]
            ax.scatter(r["causal_hidden_nodes_mean"], float(r["test_accuracy_mean"]),
                       s=110, color=colour, marker=marker, zorder=3,
                       edgecolors=SURFACE, linewidths=2, label=LABEL[r["condition"]])
        # Direct-label only the two that make the argument.
        for key, dy in ((REFERENCE, 8), ("fixed_mixed_matched", -14)):
            r = next((x for x in rows if x["condition"] == key), None)
            if r:
                ax.annotate(LABEL[key],
                            (r["causal_hidden_nodes_mean"], float(r["test_accuracy_mean"])),
                            textcoords="offset points", xytext=(0, dy), ha="center",
                            fontsize=7.5, color=INK2)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.set_xlabel("causally active hidden nodes", fontsize=8, color=INK2)
        if task == ALL_TASKS[0]:
            ax.set_ylabel("sealed-test accuracy", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            h2.append(h)
            l2.append(lab)
    fig.legend(h2, l2, fontsize=7.5, frameon=False, ncol=4, loc="lower center")
    fig.suptitle("Bigger is not better: champion size against what it scored",
                 color=INK, fontsize=11)
    return _save(fig, out, top=0.86, bottom=0.26, left=0.06, right=0.99, wspace=0.16)


def fig_mechanisms(effects: list[dict], out: Path) -> Path:
    """Each mechanism's effect on accuracy, paired and Holm-corrected."""
    conds = [c for c in ALL_CONDITIONS if c != REFERENCE]
    fig, ax = _fig(figsize=(11, 4.6))
    width = 0.26
    for ti, task in enumerate(ALL_TASKS):
        xs, med, lo, hi, sig = [], [], [], [], []
        for ci, cond in enumerate(conds):
            e = next((x for x in effects
                      if x["task"] == task and x["condition"] == cond
                      and x["metric"] == "test_accuracy"), None)
            if e is None:
                continue
            xs.append(ci + (ti - 1) * width)
            # Plot condition minus reference, so "up" means the change helped.
            med.append(-e["median_difference"])
            lo.append(e["ci95_high"] - e["median_difference"])
            hi.append(e["median_difference"] - e["ci95_low"])
            sig.append(e.get("holm_p", e["wilcoxon_p"]) < 0.05)
        ax.bar(xs, med, width=width * 0.9,
               color=[SERIES[ti] if s else NEUTRAL for s in sig],
               yerr=[lo, hi], ecolor=INK2, capsize=2, error_kw={"linewidth": 0.9},
               label=TASK_LABEL[task])
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABEL[c] for c in conds], fontsize=8, color=INK2,
                       rotation=18, ha="right")
    ax.set_ylabel("median change in sealed-test accuracy\nagainst the reference",
                  fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(
        "What each NEAT mechanism is worth. Grey bars are not significant after "
        "Holm correction.", color=INK, fontsize=10,
    )
    return _save(fig, out)


def fig_hypotheses(hyp: list[dict], out: Path) -> Path:
    fig, ax = _fig(figsize=(11, 0.78 * len(hyp) + 1.6))
    ax.grid(False)
    for i, h in enumerate(reversed(hyp)):
        good = h["verdict"] == "holds"
        colour = GOOD if good else (BAD if h["verdict"] == "fails" else NEUTRAL)
        ax.add_patch(plt.Rectangle((0, i - 0.38), 0.06, 0.76,
                                   facecolor=colour, edgecolor=SURFACE))
        ax.text(0.09, i + 0.14, f"{h['hypothesis']}  {h['statement']}",
                fontsize=9, color=INK, va="center")
        ax.text(0.09, i - 0.19,
                f"rule: {h['decision_rule']}   ·   observed: {h['observed']}",
                fontsize=7.5, color=INK2, va="center")
        ax.text(0.985, i, h["verdict"], fontsize=9, color=colour, ha="right",
                va="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(hyp) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Preregistered hypotheses, scored by their own declared rules",
                 color=INK, fontsize=11, loc="left")
    return _save(fig, out)


def fig_accuracy(summary: list[dict], out: Path) -> Path:
    by = {(r["task"], r["condition"]): r for r in summary}
    fig, axes = _fig(1, len(ALL_TASKS), figsize=(13, 4.3))
    for ax, task in zip(np.atleast_1d(axes).ravel(), ALL_TASKS):
        vals, colours = [], []
        for c in ALL_CONDITIONS:
            row = by.get((task, c))
            v = row.get("test_accuracy_mean") if row else None
            vals.append(float(v) if v not in (None, "") else np.nan)
            colours.append(STYLE[c][0])
        y = np.arange(len(ALL_CONDITIONS))
        ax.barh(y, vals, color=colours, height=0.7)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in ALL_CONDITIONS], fontsize=7.5, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0.45, 0.92)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        if task != ALL_TASKS[0]:
            ax.set_yticklabels([])
    fig.suptitle("Sealed-test accuracy by condition", color=INK, fontsize=11)
    return _save(fig, out, top=0.87, bottom=0.07, left=0.17, right=0.985, wspace=0.08)


def build_all(release_dir: Path, progress=print) -> list[Path]:
    release_dir = Path(release_dir)
    runs, final = load(release_dir)
    figs = release_dir / "figures"
    made = [fig_complexification(runs, figs / "complexification.png")]
    if final:
        comp = complexity(runs, final)
        made.append(fig_size_vs_accuracy(comp, figs / "size-vs-accuracy.png"))
        made.append(fig_accuracy(summarise(runs, final), figs / "accuracy.png"))
        made.append(
            fig_mechanisms(paired_effects(runs, final, "test_accuracy"),
                           figs / "mechanisms.png")
        )
        made.append(fig_hypotheses(hypotheses(runs, final), figs / "hypotheses.png"))
    progress(f"{len(made)} figures -> {figs}")
    return made
