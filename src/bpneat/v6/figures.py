"""v6 figures: the budget on an axis, the arm in the hue.

The budget is an ordered quantity, so it goes on an axis and never into a
colour. That leaves exactly three hues for the three arms, and they are the hues
those arms already carry in v3, v4 and v5 — the search is slate, the
candidate-matched null is mauve, the budget-matched fixed network is rose — so a
reader who has seen an earlier figure does not have to relearn them.

Every token — ground, ink, grid, palette, type scale, density — comes from
`bpneat.style`. Nothing here defines a hue or a size of its own, and every
figure leaves through `style.save`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from ..style import (  # noqa: E402
    ANNOT_SIZE,
    BAD,
    GOOD,
    INK,
    INK2,
    LEGEND_SIZE,
    NEUTRAL,
    RULE,
    SURFACE,
    TASK_LABEL,
    TITLE_SIZE,
    arm_colour,
    pale,
    parity,
    save,
    strip,
    title,
)
from ..style import figure as _fig  # noqa: E402
from .analysis import (  # noqa: E402
    _contrast,
    _index,
    _replicates,
    arm_contrasts,
    budget_table,
    hypotheses,
    load,
    scaling_slopes,
)
from .protocol import (  # noqa: E402
    ALL_TASKS,
    ARMS,
    BUDGETS,
    EQUIVALENCE_DELTA,
    EXTENSION_BUDGETS,
    REFERENCE_BUDGET,
    SLOPE_TERMS,
    condition_name,
)

ARM_LABEL = {
    "search": "Backprop-NEAT (searched)",
    "null": "same candidates, sampled not searched",
    "fixed": "fixed 32×32 net at the search's gradient budget",
}

TERM_LABEL = {
    "search": "search",
    "null": "null",
    "fixed": "fixed net",
    "edge_vs_null": "search − null",
    "edge_vs_fixed": "search − fixed",
}

#: Marker per arm, so the three series are separable without colour.
ARM_MARKER = {"search": "o", "null": "s", "fixed": "^"}


def _rungs(runs: list[dict]) -> list:
    """The rungs actually present, confirmatory first, extension last."""
    present = {r["budget"] for r in runs}
    return [b for b in (*BUDGETS, *EXTENSION_BUDGETS) if b.label in present]


def _tick_labels(rungs) -> tuple[list[float], list[str]]:
    xs = [float(np.log10(b.candidates)) for b in rungs]
    labels = [f"{b.candidates:,}\n{b.multiplier:.2g}×" for b in rungs]
    return xs, labels


def _median_band(values: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and a 10,000-sample bootstrap interval, rung by rung."""
    rng = np.random.default_rng(7)
    med, lo, hi = [], [], []
    for vals in values:
        v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
        if len(v) == 0:
            med.append(np.nan), lo.append(np.nan), hi.append(np.nan)
            continue
        med.append(float(np.median(v)))
        if len(v) < 2:
            lo.append(med[-1]), hi.append(med[-1])
            continue
        draws = np.median(v[rng.integers(0, len(v), size=(2000, len(v)))], axis=1)
        lo.append(float(np.percentile(draws, 2.5)))
        hi.append(float(np.percentile(draws, 97.5)))
    return np.array(med), np.array(lo), np.array(hi)


# --------------------------------------------------------------------------


def fig_accuracy_by_budget(runs, final, out: Path):
    """The figure the protocol exists to draw: three arms against the budget."""
    rungs = _rungs(runs)
    xs, labels = _tick_labels(rungs)
    idx = _index(runs, final)
    fig, axes = _fig(len(ALL_TASKS), 1, figsize=(6.6, 2.15 * len(ALL_TASKS)))
    axes = np.atleast_1d(axes).ravel()
    ext = {b.label for b in EXTENSION_BUDGETS}

    for ti, task in enumerate(ALL_TASKS):
        ax = axes[ti]
        reps = _replicates(runs, task)
        for arm in ARMS:
            series = [
                [idx[(task, condition_name(arm, b), r)]["test_accuracy"]
                 for r in reps if (task, condition_name(arm, b), r) in idx]
                for b in rungs
            ]
            med, lo, hi = _median_band(series)
            c = arm_colour(arm)
            ax.fill_between(xs, lo, hi, color=pale(c, 0.80), lw=0, zorder=2)
            ax.plot(xs, med, color=c, lw=1.4, zorder=3,
                    marker=ARM_MARKER[arm], ms=4.0, mec=SURFACE, mew=0.6,
                    label=ARM_LABEL[arm] if ti == 0 else None)
        ax.axvline(float(np.log10(REFERENCE_BUDGET.candidates)), color=RULE,
                   lw=0.8, ls=(0, (4, 3)), zorder=1)
        if ext & {b.label for b in rungs}:
            edge = float(np.log10(BUDGETS[-1].candidates * 1.6))
            ax.axvspan(edge, max(xs) + 0.08, color=pale(NEUTRAL, 0.72), lw=0, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("sealed-test accuracy")
        title(ax, TASK_LABEL[task])
        if ti == len(ALL_TASKS) - 1:
            ax.set_xlabel("candidate evaluations (and multiples of the reference budget)")
    axes[0].annotate("the budget every earlier release ran at",
                     xy=(float(np.log10(REFERENCE_BUDGET.candidates)), 1.0),
                     xycoords=("data", "axes fraction"), xytext=(3, -3),
                     textcoords="offset points", ha="left", va="top",
                     fontsize=ANNOT_SIZE, color=RULE)
    axes[0].legend(loc="lower right", fontsize=LEGEND_SIZE)
    fig.suptitle(
        "Does the architecture search pay at any budget?\n"
        "Median over 30 replicates with a bootstrap interval; the budget is on "
        "the axis, the arm in the hue.",
        color=INK, fontsize=TITLE_SIZE, x=0.0, ha="left",
    )
    return save(fig, out, top=0.90, hspace=0.42)


def fig_search_edge(runs, final, contrasts, out: Path):
    """The paired margin over each control, rung by rung, against zero."""
    fig, axes = _fig(1, len(ALL_TASKS), figsize=(6.6, 2.9))
    axes = np.atleast_1d(axes).ravel()
    labels = [f"{b.candidates:,}" for b in BUDGETS]
    x = np.arange(len(BUDGETS), dtype=float)

    for ti, task in enumerate(ALL_TASKS):
        ax = axes[ti]
        ax.axhspan(-EQUIVALENCE_DELTA, EQUIVALENCE_DELTA,
                   color=pale(NEUTRAL, 0.74), lw=0, zorder=0)
        for other, shift in (("null", -0.14), ("fixed", 0.14)):
            cells = [_contrast(contrasts, f"search_vs_{other}", task, b.label)
                     for b in BUDGETS]
            med = np.array([np.nan if c is None else c["median_difference"] for c in cells])
            lo = np.array([np.nan if c is None else c["ci95_low"] for c in cells])
            hi = np.array([np.nan if c is None else c["ci95_high"] for c in cells])
            c = arm_colour(other)
            ax.errorbar(x + shift, med, yerr=[med - lo, hi - med], fmt="none",
                        ecolor=pale(c, 0.45), elinewidth=1.1, capsize=0, zorder=2)
            ax.plot(x + shift, med, color=c, lw=1.2, zorder=3,
                    marker=ARM_MARKER[other], ms=4.0, mec=SURFACE, mew=0.6,
                    label=f"search − {other}" if ti == 0 else None)
            for xi, cell in zip(x + shift, cells):
                if cell is not None and cell.get("holm_significant"):
                    ax.annotate("*", xy=(xi, cell["ci95_high"]), xytext=(0, 2),
                                textcoords="offset points", ha="center",
                                fontsize=ANNOT_SIZE, color=INK2)
        parity(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        if ti == 0:
            ax.set_ylabel("paired difference in\nsealed-test accuracy")
            ax.legend(loc="best", fontsize=LEGEND_SIZE)
        title(ax, TASK_LABEL[task])
        ax.set_xlabel("candidates")
    fig.suptitle(
        "What the search is worth at each budget. Above the dashed line the "
        "search wins; the grey band is the\n"
        f"±{EQUIVALENCE_DELTA} equivalence margin declared in advance; * is "
        "Holm-corrected p < 0.05 within its family.",
        color=INK, fontsize=TITLE_SIZE, x=0.0, ha="left",
    )
    return save(fig, out, top=0.80, bottom=0.22, wspace=0.30)


def fig_scaling_slopes(slopes, out: Path):
    """Every replicate's slope, so the spread is visible and not inferred."""
    import json as _json

    fig, axes = _fig(1, len(ALL_TASKS), figsize=(6.6, 3.0))
    axes = np.atleast_1d(axes).ravel()
    terms = list(SLOPE_TERMS)
    colours = [arm_colour(t) if t in ARMS else INK2 for t in terms]

    for ti, task in enumerate(ALL_TASKS):
        ax = axes[ti]
        groups = []
        for term in terms:
            row = next((s for s in slopes if s["task"] == task and s["term"] == term
                        and s["ladder"] == "confirmatory"), None)
            groups.append(
                [] if row is None
                else [d["slope"] for d in _json.loads(row["per_replicate"])]
            )
        strip(ax, groups, colours, [TERM_LABEL[t] for t in terms],
              ylabel="accuracy per decade" if ti == 0 else "",
              heading=TASK_LABEL[task])
        parity(ax)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    fig.suptitle(
        "Accuracy per decade of compute, one slope per replicate. Above the "
        "dashed line the term grows with\nbudget; below it, more compute buys "
        "less. The two margins use the axis their arms are matched on.",
        color=INK, fontsize=TITLE_SIZE, x=0.0, ha="left",
    )
    return save(fig, out, top=0.80, bottom=0.24, wspace=0.26)


def fig_what_a_rung_costs(budgets, out: Path):
    """The ladder's own cost, which is why the top rung is an extension."""
    fig, axes = _fig(1, 2, figsize=(6.6, 2.6), grid_axis="both")
    rows = [r for r in budgets if r["task"] == "spiral"] or budgets
    present = sorted({r["candidate_budget"] for r in rows})
    x = np.log10(np.array(present, dtype=float))

    for arm in ARMS:
        wall, steps = [], []
        for cb in present:
            hits = [r for r in rows if r["arm"] == arm and r["candidate_budget"] == cb]
            wall.append(np.mean([h["wall_time_mean"] for h in hits]) if hits else np.nan)
            steps.append(np.mean([h["gradient_steps_mean"] for h in hits]) if hits else np.nan)
        for ax, y in ((axes[0], wall), (axes[1], steps)):
            ax.plot(x, y, color=arm_colour(arm), lw=1.3, marker=ARM_MARKER[arm],
                    ms=4.0, mec=SURFACE, mew=0.6,
                    label=arm if ax is axes[0] else None)
    for ax, lab in ((axes[0], "wall-clock seconds per run"),
                    (axes[1], "gradient steps per run")):
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:,}" for c in present], rotation=45, ha="right")
        ax.set_xlabel("candidates")
        ax.set_ylabel(lab)
    title(axes[0], "What a rung costs")
    title(axes[1], "What a rung spends")
    axes[0].legend(loc="upper left", fontsize=LEGEND_SIZE)
    fig.suptitle(
        "The search's cost is superlinear in its generation count — the networks "
        "grow, so later candidates\ncost more than earlier ones. On spirals; "
        "both axes logarithmic.",
        color=INK, fontsize=TITLE_SIZE, x=0.0, ha="left",
    )
    return save(fig, out, top=0.80, bottom=0.24, wspace=0.30)


def fig_verdicts(hyp, out: Path):
    """The preregistered rules and what happened, in the order declared."""
    fig, ax = _fig(1, 1, figsize=(6.6, 0.52 * max(len(hyp), 1) + 1.1), grid_axis=None)
    ax.set_axis_off()
    colour = {"holds": GOOD, "fails": BAD, "incomplete": NEUTRAL}
    y = 0.0
    for row in hyp:
        ax.annotate(row["hypothesis"], xy=(0.0, -y), xycoords=("axes fraction", "data"),
                    ha="left", va="top", fontsize=ANNOT_SIZE, color=INK2)
        ax.annotate(row["verdict"], xy=(0.105, -y), xycoords=("axes fraction", "data"),
                    ha="left", va="top", fontsize=ANNOT_SIZE,
                    color=colour.get(row["verdict"], NEUTRAL))
        ax.annotate(row["observed"], xy=(0.215, -y), xycoords=("axes fraction", "data"),
                    ha="left", va="top", fontsize=ANNOT_SIZE, color=INK2)
        ax.annotate(row["statement"], xy=(0.36, -y), xycoords=("axes fraction", "data"),
                    ha="left", va="top", fontsize=ANNOT_SIZE, color=INK,
                    wrap=True)
        ax.axhline(-y + 0.35, color=RULE, lw=0.5, alpha=0.5)
        y += 1.0
    ax.set_ylim(-y + 0.4, 0.6)
    ax.set_xlim(0, 1)
    fig.suptitle(
        "Every preregistered hypothesis, scored by its own declared rule.",
        color=INK, fontsize=TITLE_SIZE, x=0.0, ha="left",
    )
    return save(fig, out, top=0.90)


def build_all(release_dir: Path, progress=print) -> list[Path]:
    release_dir = Path(release_dir)
    runs, final = load(release_dir)
    figs = release_dir / "figures"
    made = [fig_what_a_rung_costs(budget_table(runs), figs / "what-a-rung-costs.png")]
    if final:
        contrasts = arm_contrasts(runs, final)
        made.append(fig_accuracy_by_budget(runs, final, figs / "accuracy-by-budget.png"))
        made.append(fig_search_edge(runs, final, contrasts, figs / "search-edge.png"))
        made.append(
            fig_scaling_slopes(scaling_slopes(runs, final), figs / "scaling-slopes.png")
        )
        made.append(fig_verdicts(hypotheses(runs, final), figs / "verdicts.png"))
    progress(f"{len(made)} figures -> {figs}")
    return made
