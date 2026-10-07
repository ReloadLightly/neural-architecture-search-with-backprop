"""v8 figures: four arms, three real datasets, and the gap that is not there.

The arm is the whole identity here — on real data at one budget, there is no
second axis — so colour carries the arm and nothing else, in the hues those arms
already have: the search is slate, its candidate-matched null is mauve, the
linear model is brick and the budget-matched fixed network is rose. Cold is a
thing somebody searched for; warm is a thing decided in advance.

Every token comes from `bpneat.style`. Nothing here defines a hue or a size, and
every figure leaves through `style.save`, which measures its own width and
refuses anything wider than the page.
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
    colour_of,
    pale,
    parity,
    save,
    strip,
    suptitle,
    title,
    wrap_to,
)
from ..style import figure as _fig  # noqa: E402
from .analysis import (  # noqa: E402
    CONTRASTS,
    arm_contrasts,
    budget_table,
    champion_size,
    equivalence,
    hypotheses,
    load,
    summarise,
)
from .protocol import ALL_TASKS, ARMS, EQUIVALENCE_DELTA, EXTENSION_TASKS  # noqa: E402

ARM_LABEL = {
    "search": "Backprop-NEAT",
    "null": "same candidates,\nsampled",
    "linear": "linear\n(no architecture)",
    "fixed": "fixed 32×32\n(matched budget)",
}
ARM_MARKER = {"search": "o", "null": "s", "linear": "D", "fixed": "^"}

DATASET_LABEL = {
    "iris": "Iris",
    "wine": "Wine",
    "breast_cancer": "Breast cancer",
    "digits": "Digits",
}


def _tasks(runs) -> list[str]:
    seen = {r["task"] for r in runs}
    return [t for t in (*ALL_TASKS, *EXTENSION_TASKS) if t in seen]


def _per_replicate(final, runs, task: str, arm: str) -> list[float]:
    ids = {
        r["replicate"]: r["run_id"]
        for r in runs
        if r["task"] == task and r["condition"] == arm
    }
    out = []
    for rep in sorted(ids):
        t = final.get(ids[rep])
        if t is not None:
            out.append(t["test_accuracy"])
    return out


def fig_accuracy_by_arm(runs, final, summary, out: Path):
    """The figure the protocol exists to draw: four arms, every replicate shown."""
    tasks = _tasks(runs)
    fig, axes = _fig(1, len(tasks), figsize=(6.6, 3.2))
    axes = np.atleast_1d(axes).ravel()
    colours = [colour_of(a) for a in ARMS]

    for ti, task in enumerate(tasks):
        ax = axes[ti]
        groups = [_per_replicate(final, runs, task, a) for a in ARMS]
        strip(ax, groups, colours, [ARM_LABEL[a] for a in ARMS],
              ylabel="sealed-test accuracy" if ti == 0 else "",
              heading=DATASET_LABEL.get(task, task))
        # The linear model is the thing to measure against here, so it gets the
        # dashed line every other figure in this project gives its reference.
        linear = groups[ARMS.index("linear")]
        if linear:
            parity(ax, float(np.median(linear)))
        row = next((s for s in summary if s["task"] == task and s["arm"] == "linear"), None)
        if row is not None and ti == 0:
            ax.annotate("median of the linear arm", xy=(0.02, float(np.median(linear))),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=ANNOT_SIZE, color=RULE)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
    suptitle(
        fig,
        "Does architecture search beat having no architecture? One dot per "
        "replicate, thirty of them; the bar is the median; the dashed line is "
        "the arm with no hidden units at all.",
    )
    return save(fig, out, bottom=0.30, wspace=0.14)


def fig_contrasts(contrasts, equiv, out: Path):
    """Every declared contrast, against zero and against the margin."""
    pairs = [(f, left, right) for f, ps in CONTRASTS.items() for left, right in ps]
    tasks = sorted(
        {c["task"] for c in contrasts if c["metric"] == "test_accuracy"},
        key=lambda t: (t in EXTENSION_TASKS, t),
    )
    fig, axes = _fig(1, len(tasks), figsize=(6.6, 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[1:]:
        ax.sharey(axes[0])
        ax.tick_params(labelleft=False)
    y = np.arange(len(pairs))[::-1]

    for ti, task in enumerate(tasks):
        ax = axes[ti]
        ax.axvspan(-EQUIVALENCE_DELTA, EQUIVALENCE_DELTA,
                   color=pale(NEUTRAL, 0.74), lw=0, zorder=0)
        for yi, (family, left, right) in zip(y, pairs):
            c = next(
                (
                    r for r in contrasts
                    if r["family"] == family and r["task"] == task
                    and r["left"] == left and r["right"] == right
                    and r["metric"] == "test_accuracy"
                ),
                None,
            )
            if c is None:
                continue
            colour = colour_of(left)
            ax.plot([c["ci95_low"], c["ci95_high"]], [yi, yi],
                    color=pale(colour, 0.45), lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(c["median_difference"], yi, marker=ARM_MARKER[left], ms=5.0,
                    color=colour, mec=SURFACE, mew=0.6, zorder=3)
            if c.get("holm_significant"):
                ax.annotate("*", xy=(c["ci95_high"], yi), xytext=(3, -2),
                            textcoords="offset points", fontsize=ANNOT_SIZE,
                            color=INK2)
        ax.axvline(0.0, color=RULE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(y)
        if ti == 0:
            ax.set_yticklabels([f"{left} − {right}" for _, left, right in pairs])
        ax.set_ylim(-0.6, len(pairs) - 0.4)
        ax.set_xlabel("difference in accuracy")
        title(ax, DATASET_LABEL.get(task, task))
    suptitle(
        fig,
        f"Paired differences in sealed-test accuracy. The grey band is the "
        f"±{EQUIVALENCE_DELTA} equivalence margin declared in advance — about "
        "one row of the smallest sealed split; * is Holm-corrected p < 0.05 "
        "within the contrast's own family.",
    )
    return save(fig, out, bottom=0.22, wspace=0.12)


def fig_champion_size(runs, sizes, out: Path):
    """How much architecture the search actually returns."""
    tasks = _tasks(runs)
    fig, axes = _fig(1, len(tasks), figsize=(6.6, 2.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[1:]:
        ax.sharey(axes[0])
        ax.tick_params(labelleft=False)

    for ti, task in enumerate(tasks):
        ax = axes[ti]
        counts = [
            r["metrics"]["causal_hidden_nodes"]
            for r in runs
            if r["task"] == task and r["condition"] == "search"
        ]
        if not counts:
            continue
        edges = np.arange(-0.5, max(max(counts), 3) + 1.5)
        ax.hist(counts, bins=edges, color=colour_of("search"),
                edgecolor=SURFACE, linewidth=0.7)
        row = next((s for s in sizes if s["task"] == task), None)
        if row is not None:
            ax.annotate(
                f"{row['zero_hidden']}/{row['n']} champions have none"
                + ("*" if row.get("holm_significant") else ""),
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=ANNOT_SIZE, color=INK2,
            )
        ax.set_xlabel("causally active hidden units")
        if ti == 0:
            ax.set_ylabel("replicates")
        title(ax, DATASET_LABEL.get(task, task))
    suptitle(
        fig,
        "What the search returns. A champion with no causally active hidden "
        "unit is a linear model that the search spent 2,100 candidate "
        "evaluations arriving at.",
    )
    return save(fig, out, bottom=0.22, wspace=0.10)


def fig_what_an_arm_costs(budgets, out: Path):
    """What each arm cost, which is half of what "matched" has to mean."""
    tasks = sorted({r["task"] for r in budgets},
                   key=lambda t: (t in EXTENSION_TASKS, t))
    fig, axes = _fig(1, 2, figsize=(6.6, 2.6), grid_axis="y")
    x = np.arange(len(tasks), dtype=float)
    width = 0.8 / max(len(ARMS), 1)

    for ai, arm in enumerate(ARMS):
        wall, steps = [], []
        for task in tasks:
            row = next((r for r in budgets if r["task"] == task and r["arm"] == arm), None)
            wall.append(np.nan if row is None else row["wall_time_mean"] / 60.0)
            steps.append(np.nan if row is None else row["gradient_steps_mean"])
        offset = (ai - (len(ARMS) - 1) / 2) * width
        axes[0].bar(x + offset, wall, width, color=colour_of(arm),
                    edgecolor=SURFACE, linewidth=0.5,
                    label=arm)
        axes[1].bar(x + offset, steps, width, color=colour_of(arm),
                    edgecolor=SURFACE, linewidth=0.5)
    for ax, lab in ((axes[0], "wall-clock minutes per run"),
                    (axes[1], "gradient steps per run")):
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL.get(t, t) for t in tasks],
                           rotation=30, ha="right")
        ax.set_ylabel(lab)
    axes[1].set_yscale("log")
    title(axes[0], "What an arm costs")
    title(axes[1], "What an arm spends")
    axes[0].legend(loc="upper left", fontsize=LEGEND_SIZE)
    suptitle(
        fig,
        "The three non-evolutionary arms are matched on gradient steps, which "
        "is why their step counts agree and their wall-clock times do not.",
    )
    return save(fig, out, bottom=0.26, wspace=0.32)


def fig_verdicts(hyp, out: Path):
    """The preregistered rules and what happened, in the order declared."""
    rules = [wrap_to(r["decision_rule"], 4.2, ANNOT_SIZE - 0.5, _fig(1, 1)[0]).split("\n")
             for r in hyp] if hyp else []
    heights = [1 + len(w) * 0.62 for w in rules]
    fig, ax = _fig(1, 1, figsize=(6.6, 0.30 * sum(heights) + 1.0), grid_axis=None)
    ax.set_axis_off()
    colour = {"holds": GOOD, "fails": BAD, "incomplete": NEUTRAL}
    name_w = max((len(r["hypothesis"]) for r in hyp), default=6)
    verdict_w = max((len(r["verdict"]) for r in hyp), default=7)
    observed_w = max((len(str(r["observed"])) for r in hyp), default=3)
    unit = 0.0098
    x_verdict = (name_w + 2) * unit
    x_observed = x_verdict + (verdict_w + 2) * unit
    x_statement = x_observed + (observed_w + 2) * unit

    y = 0.0
    for row, lines, height in zip(hyp, rules, heights):
        for x, text, c in (
            (0.0, row["hypothesis"], INK2),
            (x_verdict, row["verdict"], colour.get(row["verdict"], NEUTRAL)),
            (x_observed, str(row["observed"]), INK2),
            (x_statement, row["statement"], INK),
        ):
            ax.annotate(text, xy=(x, -y), xycoords=("axes fraction", "data"),
                        ha="left", va="top", fontsize=ANNOT_SIZE, color=c)
        for li, line in enumerate(lines):
            ax.annotate(line, xy=(x_statement, -y - 0.62 * (li + 1)),
                        xycoords=("axes fraction", "data"), ha="left", va="top",
                        fontsize=ANNOT_SIZE - 0.5, color=INK2)
        ax.axhline(-y + 0.42, color=RULE, lw=0.5, alpha=0.5)
        y += height
    ax.set_ylim(-y + 0.5, 0.6)
    ax.set_xlim(0, 1)
    suptitle(
        fig,
        "Every preregistered hypothesis, scored by its own declared rule. The "
        "rule is printed under the statement it decides, because the rule is "
        "the part that was frozen.",
    )
    return save(fig, out, left=0.0, right=1.0)


def build_all(release_dir: Path, progress=print) -> list[Path]:
    release_dir = Path(release_dir)
    runs, final = load(release_dir)
    figs = release_dir / "figures"
    made = [fig_what_an_arm_costs(budget_table(runs), figs / "what-an-arm-costs.png")]
    sizes = champion_size(runs)
    if sizes:
        made.append(fig_champion_size(runs, sizes, figs / "champion-size.png"))
    if final:
        summary = summarise(runs, final)
        contrasts = arm_contrasts(runs, final)
        made.append(fig_accuracy_by_arm(runs, final, summary,
                                        figs / "accuracy-by-arm.png"))
        made.append(fig_contrasts(contrasts, equivalence(runs, final),
                                  figs / "contrasts.png"))
        made.append(fig_verdicts(hypotheses(runs, final), figs / "verdicts.png"))
    progress(f"{len(made)} figures -> {figs}")
    return made


