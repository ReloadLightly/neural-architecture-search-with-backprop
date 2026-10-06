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

from ..style import (  # noqa: E402
    BAD,
    GOOD,
    GRID,
    INK,
    INK2,
    NEUTRAL,
    SEARCH_NULL,
    SEARCH_PRIMARY,
    SEARCH_SECONDARY,
    SURFACE,
    colour_of,
    save,
    style_axes,
    within,
)
from .analysis import (  # noqa: E402
    REFERENCE_CHAMPION,
    complexity,
    hypotheses,
    load,
    paired_effects,
    summarise,
)
from .protocol import ALL_CONDITIONS, ALL_TASKS, REFERENCE  # noqa: E402

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
from ..style import TASK_LABEL  # noqa: E402

#: Every arm here is a search, so they all take the search family and differ by
#: value rather than by hue. The three steps are not a decorative ramp: they are
#: the preregistered question. Does releasing the brake on size make topologies
#: grow? So the deep step is the arm with both brakes on, the middle step is the
#: arms that release one or both, and the pale step is the arms that change the
#: search without touching size at all. The reader sees the answer as three
#: bands before reading a single label.
#:
#: The scheme this replaced assigned colour by position over a six-slot
#: categorical palette. With eight arms it cycled: "deep & narrow" came out the
#: same blue as the reference, and the fixed control the same orange as "higher
#: mutation rates".
BRAKES_ON = ["neat_reference"]
BRAKE_RELEASED = ["neat_no_penalty", "neat_complexify", "neat_complexify_no_penalty"]
SEARCH_ALTERED = ["neat_no_speciation", "neat_no_crossover", "neat_deep_narrow"]

#: Within a band, identity is carried by shape and dash, never by hue alone.
MARKER = {
    "neat_reference": ("o", "-"),
    "neat_complexify": ("s", "-"),
    "neat_no_penalty": ("^", "-"),
    "neat_complexify_no_penalty": ("D", "-"),
    "neat_no_speciation": ("v", "-"),
    "neat_no_crossover": ("P", "-"),
    "neat_deep_narrow": ("X", "--"),
    "fixed_mixed_matched": ("*", ":"),
}

_BAND = (
    dict.fromkeys(BRAKES_ON, SEARCH_PRIMARY)
    | dict.fromkeys(BRAKE_RELEASED, SEARCH_SECONDARY)
    | dict.fromkeys(SEARCH_ALTERED, SEARCH_NULL)
)

#: The one arm that is not a search at all keeps the control family.
STYLE = {c: (_BAND.get(c, colour_of(c)), *MARKER[c]) for c in LABEL}

#: Rows read in band order, so colour and vertical position agree.
READING_ORDER = BRAKES_ON + BRAKE_RELEASED + SEARCH_ALTERED + ["fixed_mixed_matched"]

#: Geometry as a series: three ordered steps inside the search family, used
#: only where every arm in the figure is a search and the facet is the task.
TASK_COLOUR = within("search_primary", ALL_TASKS)


def _style(ax):
    style_axes(ax)


def _fig(nrows=1, ncols=1, figsize=(10, 4.2)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path: Path, **kw):
    # One exit for every figure in the module, so the typographic pass in
    # `style.save` cannot be forgotten at a call site.
    return save(fig, path, **kw)


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
            # x is progress through the run, not the generation index. The arms
            # share an evaluation budget but not a population size, so "deep &
            # narrow" spends it over 84 small generations where the others take
            # 21 — against the raw index, six of the seven arms were crushed
            # into the leftmost quarter of every panel and compared nothing.
            gens = np.linspace(0.0, 1.0, n)
            vals = np.mean(
                [[h[i]["mean_nodes"] for i in range(n)] for h in series], axis=0
            )
            colour, marker, dash = STYLE[cond]
            # The reference is what every effect is measured against, so it is
            # drawn heavier as well as deeper.
            ax.plot(gens, vals, color=colour, linestyle=dash,
                    linewidth=2.8 if cond == REFERENCE else 1.8,
                    marker=marker, markevery=max(n // 6, 1), markersize=5,
                    label=LABEL[cond])
        if task == REFERENCE_CHAMPION["task"]:
            ax.axhline(REFERENCE_CHAMPION["nodes"], color=INK2, linewidth=1.2,
                       linestyle=(0, (4, 3)))
            ax.text(0.98, REFERENCE_CHAMPION["nodes"], " Ha (2016) champion: 34 nodes",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=7.5, color=INK2)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.set_xlabel("progress through the run", fontsize=8, color=INK2)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["start", "half", "end"], fontsize=7.5)
        if task == ALL_TASKS[0]:
            ax.set_ylabel("mean nodes in the population", fontsize=8, color=INK2)
    # A per-axes legend here lands on the reference line and its label, so the
    # key moves below the panels and every panel stays readable.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=7,
               loc="lower center")
    fig.suptitle(
        "Do the topologies augment? Population size over the run, at one shared "
        "evaluation budget.\n"
        "Mid blue released a brake on topology size; pale blue changed the search "
        "without touching size; deep blue is the reference with both brakes on.",
        color=INK, fontsize=10.5,
    )
    return _save(fig, out, top=0.78, bottom=0.23, left=0.055, right=0.99, wspace=0.16)


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
        for key, dy, ha, dx in ((REFERENCE, 10, "center", 0),
                                ("fixed_mixed_matched", -17, "right", -8)):
            r = next((x for x in rows if x["condition"] == key), None)
            if r:
                ax.annotate(LABEL[key],
                            (r["causal_hidden_nodes_mean"], float(r["test_accuracy_mean"])),
                            textcoords="offset points", xytext=(dx, dy), ha=ha,
                            fontsize=7.5, color=INK2)
        # symlog, because one arm sits at 65 units and the rest below 11. Ticks
        # are set explicitly: with linthresh=10 matplotlib drew a single "10^1"
        # and a reader could not tell two causal units from nine.
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 20, 65])
        ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "20", "65"], fontsize=7.5)
        ax.set_xlim(-0.4, 115)
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
        # The series here is the geometry, and every arm is a search, so the
        # three tasks are value steps inside the search family rather than three
        # hues that would read as three different kinds of thing. Significance
        # is a second variable and gets a second channel: a hollow bar did not
        # reach significance but still says which geometry it belongs to, where
        # greying it out erased that on two thirds of the panel.
        ax.bar(xs, med, width=width * 0.9,
               color=[TASK_COLOUR[task] if s else SURFACE for s in sig],
               edgecolor=TASK_COLOUR[task], linewidth=1.1,
               yerr=[lo, hi], ecolor=INK2, capsize=2,
               error_kw={"linewidth": 0.9})
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABEL[c] for c in conds], fontsize=8, color=INK2,
                       rotation=18, ha="right")
    ax.set_ylabel("median change in sealed-test accuracy\nagainst the reference",
                  fontsize=8, color=INK2)
    # Explicit handles: matplotlib takes a bar container's legend colour from
    # its first patch, which here is whichever condition happened to come first,
    # so the key showed the wrong colour for two of the three geometries.
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=TASK_COLOUR[tk], edgecolor="none",
                          label=TASK_LABEL[tk])
            for tk in ALL_TASKS
        ] + [
            plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK2,
                          label="not significant (Holm)")
        ],
        fontsize=8, frameon=False, loc="upper left", ncol=2,
    )
    ax.set_title(
        "What each NEAT mechanism is worth. Hollow bars are not significant after "
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
    """Every arm on every geometry, read as dots on a stem from chance.

    Bars on a 0.45 baseline made the fixed net look twice the reference when the
    ratio was 1.22, and clipped a 0.964 bar flat at the panel edge with its value
    label drawn outside the axes. A dot puts the value in its position, so the
    axis may start where the data live without the length lying about a ratio.
    """
    by = {(r["task"], r["condition"]): r for r in summary}
    chance = 0.5
    fig, axes = _fig(1, len(ALL_TASKS), figsize=(13, 4.6))
    for ax, task in zip(np.atleast_1d(axes).ravel(), ALL_TASKS):
        vals, colours = [], []
        for c in READING_ORDER:
            row = by.get((task, c))
            v = row.get("test_accuracy_mean") if row else None
            vals.append(float(v) if v not in (None, "") else np.nan)
            colours.append(STYLE[c][0])
        y = np.arange(len(READING_ORDER))
        ax.axvline(chance, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        for yi, v, colour in zip(y, vals, colours):
            if not np.isfinite(v):
                continue
            ax.plot([chance, v], [yi, yi], color=colour, linewidth=2.0, alpha=0.5,
                    solid_capstyle="round", zorder=2)
            ax.plot(v, yi, "o", color=colour, markersize=8, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.4)
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in READING_ORDER], fontsize=7.5, color=INK2)
        # Hairlines where the band changes, so the three blues read as three
        # groups rather than as three arbitrary tints.
        for edge in (len(BRAKES_ON) - 0.5,
                     len(BRAKES_ON) + len(BRAKE_RELEASED) - 0.5,
                     len(READING_ORDER) - 1.5):
            ax.axhline(edge, color=INK2, linewidth=0.7, alpha=0.45)
        ax.invert_yaxis()
        ax.set_ylim(len(READING_ORDER) - 0.4, -1.05)
        ax.set_xlim(chance - 0.02, 1.05)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        if task == ALL_TASKS[0]:
            ax.text(chance, -0.9, " chance", ha="left", va="center", fontsize=7.5,
                    color=INK2, style="italic")
        else:
            ax.set_yticklabels([])
    fig.suptitle(
        "Sealed-test accuracy by condition. Deep blue is the reference; mid blue "
        "released a brake on topology size;\npale blue changed the search without "
        "touching size; orange is not a search.",
        color=INK, fontsize=10.5,
    )
    return _save(fig, out, top=0.78, bottom=0.07, left=0.175, right=0.985, wspace=0.08)


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
