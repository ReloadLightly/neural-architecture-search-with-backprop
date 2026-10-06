"""v4 figures, rendered from the raw records.

Colour here is semantic, not positional. Every condition in this protocol is
either something we searched for or something we fixed in advance, so blue
means "searched" and orange means "fixed", with the ordered distinctions inside
each family carried as lightness. The condition-to-colour table lives in
``bpneat.style`` and is shared with every other protocol, so a condition cannot
be one colour here and another in the README.

The scheme this replaced assigned colour by row index over a six-slot
categorical palette. With eight conditions it cycled: Backprop-NEAT and "fixed
mixed @ CGP budget" came out the same blue, and the same control changed hue
between figures. Every row was already labelled, so the hue carried nothing and
actively misled.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..style import (  # noqa: E402
    BAD,
    CONTROL_BEST,
    CONTROL_RAMP,
    GOOD,
    GRID,
    INK,
    INK2,
    NEUTRAL,
    SEARCH_PRIMARY,
    SURFACE,
    colour_of,
    role_of,
    save,
    style_axes,
)
from .analysis import (  # noqa: E402
    ALGORITHMS,
    budget_table,
    cross_algorithm,
    family_effects,
    hypotheses,
    load,
    sign_matrix,
    summarise,
)
from .protocol import ALL_TASKS  # noqa: E402

LABEL = {
    "bpneat": "Backprop-NEAT",
    "cgp": "CGP (1+4)",
    "fixed_tanh_ha": "Fixed tanh (unmatched)",
    "fixed_tanh_matched_bpneat": "Fixed tanh @ BP-NEAT budget",
    "fixed_mixed_matched_bpneat": "Fixed mixed @ BP-NEAT budget",
    "fixed_tanh_matched_cgp": "Fixed tanh @ CGP budget",
    "fixed_mixed_matched_cgp": "Fixed mixed @ CGP budget",
    "cgp_random_matched": "Random CGP (matched)",
}
from ..style import TASK_LABEL  # noqa: E402

#: Marker shapes. Colour comes from the condition's role, so two conditions
#: that *are* the same architecture at two budgets share a hue on purpose; the
#: marker and the label separate them, and the shared hue says what is true.
MARKER = {
    "bpneat": "o",
    "cgp": "s",
    "fixed_tanh_ha": "^",
    "fixed_tanh_matched_bpneat": "D",
    "fixed_mixed_matched_bpneat": "v",
    "fixed_tanh_matched_cgp": "P",
    "fixed_mixed_matched_cgp": "X",
    "cgp_random_matched": "*",
}

STYLE = {c: (colour_of(c), MARKER[c]) for c in LABEL}

#: The reading order of `fig_accuracy_by_task`: the searched block, then the
#: fixed block, each in its own order. Grouping the bars by what they are is
#: what makes a shared hue legible instead of ambiguous.
READING_ORDER = [
    "bpneat", "cgp", "cgp_random_matched",
    "fixed_tanh_ha",
    "fixed_tanh_matched_bpneat", "fixed_tanh_matched_cgp",
    "fixed_mixed_matched_bpneat", "fixed_mixed_matched_cgp",
]

#: A verdict is not a value judgement: "the fixed network won" is a result, not
#: a failure, so it gets the control family's colour rather than the error red.
SIGN_COLOUR = {
    "search>fixed": SEARCH_PRIMARY,
    "fixed>search": CONTROL_BEST,
    "ns": NEUTRAL,
    "n/a": SURFACE,
}
SIGN_TEXT = {
    "search>fixed": "search\nwins",
    "fixed>search": "fixed\nwins",
    "ns": "n.s.",
    "n/a": "—",
}


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


def _test_or_val(row) -> float:
    v = row.get("test_accuracy_mean")
    return float(v) if v not in (None, "") else float(row["validation_accuracy_mean"])


def fig_sign_matrix(signs: list[dict], out: Path) -> Path:
    """The headline: the sign of search-vs-fixed, per algorithm and protocol."""
    algos = [a["algorithm"] for a in ALGORITHMS]
    cols = [("vs_unmatched", "unmatched"),
            ("vs_matched_tanh", "matched\ntanh"),
            ("vs_matched_mixed", "matched\nmixed")]
    by = {(r["algorithm"], r["task"]): r for r in signs}

    fig, axes = _fig(1, len(algos), figsize=(9, 5.0))
    for ax, algo in zip(np.atleast_1d(axes).ravel(), algos):
        ax.grid(False)
        for ci, (key, _) in enumerate(cols):
            for ti, task in enumerate(ALL_TASKS):
                verdict = by.get((algo, task), {}).get(key, "n/a")
                ax.add_patch(
                    plt.Rectangle((ci - 0.47, ti - 0.45), 0.94, 0.9,
                                  facecolor=SIGN_COLOUR[verdict], edgecolor=SURFACE,
                                  linewidth=2)
                )
                ax.text(ci, ti, SIGN_TEXT[verdict], ha="center", va="center",
                        fontsize=8, color="white" if verdict != "ns" else INK,
                        fontweight="bold" if verdict != "ns" else "normal")
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylim(-0.6, len(ALL_TASKS) - 0.4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[1] for c in cols], fontsize=8, color=INK2)
        ax.set_yticks(range(len(ALL_TASKS)))
        ax.set_yticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
        ax.set_title(algo, color=INK, fontsize=10)
        # Square cells: without this the panel stretches three columns across
        # its full width and the matrix reads as a scatter of distant blocks.
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(
        "Sealed-test accuracy: which side wins against each control,\n"
        "by algorithm and by budget protocol",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.80, bottom=0.1, left=0.1, right=0.98, wspace=0.05)


def fig_accuracy_by_task(summary: list[dict], out: Path) -> Path:
    """Every condition on every task, read as two families against chance.

    Dots on a stem from chance, not bars from an arbitrary floor. These are
    accuracies on balanced binary tasks, so the meaningful origin is 0.5 and
    not 0; a bar drawn from 0.4 would make a 0.99 and a 0.97 look like the same
    full block, which is exactly what the earlier version did on XOR and
    Circles. With a dot, position carries the value and the axis may start
    where the data live.
    """
    by = {(r["task"], r["condition"]): r for r in summary}
    conds = READING_ORDER
    split = sum(1 for c in conds if role_of(c).startswith("search"))
    chance = 0.5
    fig, axes = _fig(1, len(ALL_TASKS), figsize=(16, 4.6))
    for ax, task in zip(np.atleast_1d(axes).ravel(), ALL_TASKS):
        vals, colours = [], []
        for c in conds:
            row = by.get((task, c))
            vals.append(_test_or_val(row) if row else np.nan)
            colours.append(STYLE[c][0])
        y = np.arange(len(conds))
        ax.axvline(chance, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)),
                   zorder=1)
        for yi, v, colour in zip(y, vals, colours):
            if not np.isfinite(v):
                continue
            ax.plot([chance, v], [yi, yi], color=colour, linewidth=2.0,
                    alpha=0.5, solid_capstyle="round", zorder=2)
            ax.plot(v, yi, "o", color=colour, markersize=8.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
            ax.text(v + 0.018, yi, f"{v:.3f}", va="center", fontsize=7.5,
                    color=INK2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [LABEL[c] + ("  *" if c in ("bpneat", "cgp") else "") for c in conds],
            fontsize=7.5, color=INK2,
        )
        ax.invert_yaxis()
        ax.set_ylim(len(conds) - 0.4, -1.05)
        ax.set_xlim(chance - 0.03, 1.10)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # The two families, separated by a rule rather than by eight hues.
        ax.axhline(split - 0.5, color=INK2, linewidth=0.8, alpha=0.5)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        if task == ALL_TASKS[0]:
            ax.text(chance, -0.9, " chance", ha="left", va="center",
                    fontsize=7.5, color=INK2, style="italic")
        else:
            ax.set_yticklabels([])
    # Family labels in the left margin, outside every panel.
    for frac, text, colour in (
        (0.80, "searched", SEARCH_PRIMARY),
        (0.42, "fixed in advance", CONTROL_BEST),
    ):
        fig.text(0.012, frac, text, rotation=90, ha="left", va="center",
                 fontsize=9, color=colour)
    fig.suptitle(
        "Sealed-test accuracy by condition. Blue was found by a search; "
        "orange was fixed before the run. * marks a search algorithm.",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.86, bottom=0.08, left=0.175, right=0.99, wspace=0.08)


def fig_reversal(effects: list[dict], out: Path) -> Path:
    """Paired median difference against each control, for both algorithms.

    Above zero the search wins; below it the fixed network does. The figure's
    claim is that both algorithms cross zero at the same place.
    """
    fig, axes = _fig(1, 2, figsize=(11.5, 4.4))
    for ax, spec in zip(np.atleast_1d(axes).ravel(), ALGORITHMS):
        ref = spec["reference"]
        controls = [spec["unmatched"], spec["matched"], spec["matched_mixed"]]
        width = 0.26
        for ci, ctl in enumerate(controls):
            xs, med, lo, hi, sig = [], [], [], [], []
            for ti, task in enumerate(ALL_TASKS):
                e = next(
                    (x for x in effects
                     if x["family"] == ref and x["task"] == task and x["condition"] == ctl),
                    None,
                )
                if e is None:
                    continue
                xs.append(ti + (ci - 1) * width)
                med.append(e["median_difference"])
                lo.append(e["median_difference"] - e["ci95_low"])
                hi.append(e["ci95_high"] - e["median_difference"])
                sig.append(e.get("holm_p", e["wilcoxon_p"]) < 0.05)
            # starved -> matched -> strongest, which is exactly CONTROL_RAMP.
            # Significance is a *second* variable, so it gets a second channel:
            # a bar that did not reach significance is drawn hollow in its own
            # colour rather than greyed out, which previously erased the
            # control's identity on most of the panel.
            colour = CONTROL_RAMP[ci]
            ax.bar(xs, med, width=width * 0.9,
                   color=[colour if s else SURFACE for s in sig],
                   edgecolor=colour, linewidth=1.1,
                   yerr=[lo, hi], ecolor=INK2, capsize=2,
                   error_kw={"linewidth": 0.9})
        # Legend handles built explicitly: matplotlib takes a bar container's
        # legend colour from its first patch, which here is whichever task
        # happened to come first — in the previous version that made the key
        # read grey/grey/orange while the series were three different colours.
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, facecolor=CONTROL_RAMP[ci],
                              edgecolor=CONTROL_RAMP[ci], label=LABEL[ctl])
                for ci, ctl in enumerate(controls)
            ] + [
                plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK2,
                              label="not significant (Holm)")
            ],
            fontsize=7.5, frameon=False, loc="lower left", ncol=1,
        )
        ax.axhline(0.0, color=INK, linewidth=1.0)
        ax.set_xticks(range(len(ALL_TASKS)))
        ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
        ax.set_ylabel("median paired difference\n(search − fixed)", fontsize=8, color=INK2)
        ax.set_title(spec["algorithm"], color=INK, fontsize=10)
    fig.suptitle(
        "The reversal, both algorithms. A hollow bar did not reach significance "
        "after Holm correction in that algorithm's own family.",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.85, bottom=0.11, left=0.085, right=0.985, wspace=0.22)


def fig_budget(budgets: list[dict], out: Path) -> Path:
    """What each arm actually spent. The confound, made visible."""
    by = {(r["task"], r["condition"]): r for r in budgets}
    conds = ["bpneat", "cgp", "fixed_tanh_ha",
             "fixed_tanh_matched_bpneat", "fixed_tanh_matched_cgp"]
    fig, ax = _fig(figsize=(10.5, 4.2))
    width = 0.16
    for ci, c in enumerate(conds):
        xs = [ti + (ci - 2) * width for ti in range(len(ALL_TASKS))]
        vals = [by.get((t, c), {}).get("gradient_steps_mean", np.nan) for t in ALL_TASKS]
        colour, _ = STYLE[c]
        ax.bar(xs, vals, width=width * 0.9, color=colour, label=LABEL[c])
    ax.set_yscale("log")
    # A bar on a log axis measures from wherever autoscale put the floor, which
    # is not a number anyone declared. Pin it to one gradient update, so a bar's
    # length is "orders of magnitude above doing nothing" and the two-decade gap
    # the title claims is the gap the reader sees.
    ax.set_ylim(1, None)
    ax.set_xticks(range(len(ALL_TASKS)))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
    ax.set_ylabel("gradient updates per run (log)", fontsize=8, color=INK2)
    # The decades below 10^3 are empty on every panel; the key goes there
    # rather than on top of the bars.
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center")
    ax.set_title(
        "Realized gradient budget. The unmatched control spends about two orders of "
        "magnitude less than the searches it is compared against.",
        color=INK, fontsize=10,
    )
    return _save(fig, out)


def fig_cross_algorithm(cross: list[dict], out: Path) -> Path:
    """How far apart the algorithms are, against how far the protocol moves things."""
    fig, ax = _fig(figsize=(10.5, 4.2))
    xs = np.arange(len(cross))
    between = [abs(r["median_cgp_minus_bpneat"]) for r in cross]
    protocol = [
        float(np.mean([abs(r["median_matching_effect_bpneat"]),
                       abs(r["median_matching_effect_cgp"])]))
        for r in cross
    ]
    # Which algorithm you run is a question about the search (blue); how much
    # budget the control gets is a question about the protocol (orange).
    ax.bar(xs - 0.19, between, width=0.34, color=SEARCH_PRIMARY,
           label="|CGP − Backprop-NEAT|")
    ax.bar(xs + 0.19, protocol, width=0.34, color=CONTROL_BEST,
           label="|effect of matching the budget| (mean of the two)")
    for x, b, p in zip(xs, between, protocol):
        ax.text(x - 0.19, b + 0.004, f"{b:.3f}", ha="center", fontsize=7, color=INK2)
        ax.text(x + 0.19, p + 0.004, f"{p:.3f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([TASK_LABEL[r["task"]] for r in cross], fontsize=8, color=INK2)
    ax.set_ylabel("median paired difference in accuracy", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(
        "Choosing the algorithm matters less than choosing the budget protocol",
        color=INK, fontsize=10,
    )
    return _save(fig, out)


def fig_hypotheses(hyp: list[dict], out: Path) -> Path:
    """The preregistered scorecard, as declared, with the observed counts."""
    fig, ax = _fig(figsize=(11, 0.78 * len(hyp) + 1.6))
    ax.grid(False)
    for i, h in enumerate(reversed(hyp)):
        good = h["verdict"] == "holds"
        # A thin rule in the margin, not a 110px block of saturated colour:
        # the verdict is one bit and it is already written out in words at the
        # end of the row, so it needs a mark, not a field.
        ax.add_patch(
            plt.Rectangle((0, i - 0.38), 0.006, 0.76,
                          facecolor=GOOD if good else BAD, edgecolor="none")
        )
        ax.text(0.022, i + 0.14, f"{h['hypothesis']}  {h['statement']}",
                fontsize=9.5, color=INK, va="center")
        ax.text(0.022, i - 0.19,
                f"rule: {h['decision_rule']}   ·   observed: {h['observed']}",
                fontsize=8, color=INK2, va="center")
        ax.axhline(i - 0.5, color=GRID, linewidth=0.7)
        ax.text(0.985, i, h["verdict"], fontsize=9, color=GOOD if good else BAD,
                ha="right", va="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(hyp) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Preregistered hypotheses, scored by their own declared rules",
                 color=INK, fontsize=11, loc="left")
    return _save(fig, out)


def fig_cgp_structure(runs: list[dict], out: Path) -> Path:
    """What CGP's genotype-phenotype map actually did: active size, neutral drift."""
    active: dict[str, list[int]] = {}
    neutral: dict[str, list[int]] = {}
    for r in runs:
        if r["condition"] not in ("cgp", "cgp_random_matched"):
            continue
        n = r["compute"].get("cgp_active_nodes")
        if n is not None:
            active.setdefault(f"{r['condition']}|{r['task']}", []).append(n)
        k = r["compute"].get("cgp_neutral_accepted")
        if k is not None:
            neutral.setdefault(r["task"], []).append(k)

    fig, axes = _fig(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    width = 0.34
    for ci, cond in enumerate(("cgp", "cgp_random_matched")):
        xs = [ti + (ci - 0.5) * width for ti in range(len(ALL_TASKS))]
        vals = [
            float(np.mean(active.get(f"{cond}|{t}", [np.nan]))) for t in ALL_TASKS
        ]
        ax.bar(xs, vals, width=width * 0.9, color=STYLE[cond][0], label=LABEL[cond])
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(ALL_TASKS)))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
    ax.set_ylabel("active function nodes in the champion", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Phenotype size", color=INK, fontsize=10)

    ax = axes[1]
    vals = [neutral.get(t, []) for t in ALL_TASKS]
    parts = ax.violinplot([v or [0] for v in vals], showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(colour_of("cgp"))
        body.set_alpha(0.55)
        body.set_edgecolor(INK2)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if key in parts:
            parts[key].set_color(INK2)
    ax.set_xticks(range(1, len(ALL_TASKS) + 1))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
    ax.set_ylabel("generations accepted at equal fitness", fontsize=8, color=INK2)
    ax.set_title("Neutral drift, per run", color=INK, fontsize=10)

    fig.suptitle(
        "CGP's genotype-phenotype map in use: small phenotypes, and how often "
        "neutral offspring were accepted",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.86, bottom=0.1, left=0.07, right=0.985, wspace=0.2)


def build_all(release_dir: Path, progress=print) -> list[Path]:
    release_dir = Path(release_dir)
    runs, final = load(release_dir)
    figs = release_dir / "figures"
    summary = summarise(runs, final)
    effects = family_effects(runs, final)
    budgets = budget_table(runs)

    made = [
        fig_accuracy_by_task(summary, figs / "accuracy-by-task.png"),
        fig_reversal(effects, figs / "reversal.png"),
        fig_budget(budgets, figs / "realized-budget.png"),
        fig_cgp_structure(runs, figs / "cgp-structure.png"),
    ]
    if final:
        signs = sign_matrix(runs, final)
        cross = cross_algorithm(runs, final)
        made.insert(0, fig_sign_matrix(signs, figs / "sign-matrix.png"))
        made.append(fig_cross_algorithm(cross, figs / "cross-algorithm.png"))
        made.append(fig_hypotheses(hypotheses(runs, final), figs / "hypotheses.png"))
    progress(f"{len(made)} figures -> {figs}")
    return made
