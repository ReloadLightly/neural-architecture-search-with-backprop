"""v4 figures, rendered from the raw records.

Colour here is semantic, not positional. Every condition in this protocol is
either something we searched for or something we fixed in advance, so a cool
hue means "searched" and a warm one means "fixed". The ordered distinctions
inside each family are separate hues from the shared palette rather than tints
of one, because tints of a single hue at these mark sizes were the thing that
stopped being legible. The condition-to-colour table lives in ``bpneat.style``
and is shared with every other protocol, so a condition cannot be one colour
here and another in the README.

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
    ANNOT_SIZE,
    BAD,
    CONTROL_BEST,
    CONTROL_RAMP,
    GOOD,
    GRID,
    INK,
    INK2,
    LABEL_SIZE,
    NEUTRAL,
    PANEL_TITLE_SIZE,
    RULE,
    SEARCH_PRIMARY,
    SURFACE,
    TICK_SIZE,
    TITLE_SIZE,
    colour_of,
    pale,
    parity,
    role_of,
    save,
    style_axes,
    title,
    vparity,
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


#: No figure in this repository is wider than 6.8in: GitHub renders a README
#: image at about 870px, so a 13in figure at 200dpi arrives at a third of its
#: size and its 7.5pt ticks land at about 2pt. Panels therefore stack downwards
#: and height is spent freely; width is not.
WIDTH = 6.6


def _fig(nrows=1, ncols=1, figsize=(WIDTH, 3.4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path: Path, **kw):
    # One exit for every figure in the module, so nothing can be written at
    # another density or onto another ground than the project's.
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

    # Two algorithms, each a 3x5 block of square cells. This is the one shape
    # that stays side by side: a square matrix stacked would be a 6.6 x 20in
    # column, and at half of 6.6in a cell is still 0.7in across — wide enough
    # for the two words written in it.
    fig, axes = _fig(1, len(algos), figsize=(WIDTH, 4.3))
    for ax, algo in zip(np.atleast_1d(axes).ravel(), algos):
        ax.grid(False)
        for ci, (key, _) in enumerate(cols):
            for ti, task in enumerate(ALL_TASKS):
                verdict = by.get((algo, task), {}).get(key, "n/a")
                # A pale wash carrying the hue, with the verdict written on it
                # in ink. The version this replaces drew the lettering in
                # SURFACE and reversed it out of a saturated block: on a white
                # ground that is white-on-white for "n.s." and for the empty
                # "n/a" cell — the two cases where the reader most needs to be
                # told that nothing was found.
                ax.add_patch(
                    plt.Rectangle((ci - 0.47, ti - 0.45), 0.94, 0.9,
                                  facecolor=pale(SIGN_COLOUR[verdict]),
                                  edgecolor=SURFACE, linewidth=2)
                )
                ax.text(ci, ti, SIGN_TEXT[verdict], ha="center", va="center",
                        fontsize=LABEL_SIZE,
                        color=RULE if verdict == "n/a" else INK)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylim(-0.6, len(ALL_TASKS) - 0.4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[1] for c in cols], fontsize=TICK_SIZE, color=INK2)
        ax.set_yticks(range(len(ALL_TASKS)))
        ax.set_yticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=TICK_SIZE,
                           color=INK2)
        title(ax, algo)
        # Square cells: without this the panel stretches three columns across
        # its full width and the matrix reads as a scatter of distant blocks.
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
    # The headline is a sentence, flush left like every other title here.
    fig.suptitle(
        "Sealed-test accuracy: which side wins against each control,\n"
        "by algorithm and by budget protocol",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    return _save(fig, out, top=0.84, bottom=0.06, left=0.14, right=0.995, wspace=0.55)


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
    # One task per row, not one task per column. Five panels side by side needed
    # 16in of width; at the 6.6in a README actually renders, that put these
    # eight condition names at about 2pt. Stacked, each panel keeps its own full
    # set of labels and the figure simply gets taller.
    fig, axes = _fig(len(ALL_TASKS), 1,
                     figsize=(WIDTH, 1.95 * len(ALL_TASKS) + 0.9))
    panels = np.atleast_1d(axes).ravel()
    for ax, task in zip(panels, ALL_TASKS):
        vals, colours = [], []
        for c in conds:
            row = by.get((task, c))
            vals.append(_test_or_val(row) if row else np.nan)
            colours.append(STYLE[c][0])
        y = np.arange(len(conds))
        # The reference every dot is read against, in the project's one
        # dashed-grey register. Only the first panel names it; the line is the
        # same line in all five.
        vparity(ax, chance, label="chance" if task == ALL_TASKS[0] else None)
        for yi, v, colour in zip(y, vals, colours):
            if not np.isfinite(v):
                continue
            ax.plot([chance, v], [yi, yi], color=colour, linewidth=2.0,
                    alpha=0.5, solid_capstyle="round", zorder=2)
            ax.plot(v, yi, "o", color=colour, markersize=8.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
            ax.text(v + 0.018, yi, f"{v:.3f}", va="center", fontsize=ANNOT_SIZE,
                    color=INK2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [LABEL[c] + ("  *" if c in ("bpneat", "cgp") else "") for c in conds],
            fontsize=TICK_SIZE, color=INK2,
        )
        ax.invert_yaxis()
        ax.set_ylim(len(conds) - 0.4, -0.85)
        ax.set_xlim(chance - 0.03, 1.12)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # The two families, separated by a rule rather than by eight hues.
        ax.axhline(split - 0.5, color=RULE, linewidth=0.8, alpha=0.9)
        title(ax, TASK_LABEL[task])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    # The two families, named once, each in its own colour, beside the block of
    # rows it covers. The earlier version set them in the left margin of a
    # five-column figure; stacked, there is no such margin, and the right-hand
    # edge of the first panel is empty.
    for rows, text, colour in (
        (range(split), "searched", SEARCH_PRIMARY),
        (range(split, len(conds)), "fixed in advance", CONTROL_BEST),
    ):
        rows = list(rows)
        panels[0].annotate(
            text, xy=(1.012, (rows[0] + rows[-1]) / 2),
            xycoords=("axes fraction", "data"), rotation=90,
            ha="left", va="center", fontsize=ANNOT_SIZE, color=colour,
            annotation_clip=False,
        )
    fig.suptitle(
        "Sealed-test accuracy by condition.\n"
        "A cool hue was found by a search; a warm one was fixed before the run.\n"
        "* marks a search algorithm.",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    return _save(fig, out, top=0.905, bottom=0.04, left=0.27, right=0.955,
                 hspace=0.30)


def fig_reversal(effects: list[dict], out: Path) -> Path:
    """Paired median difference against each control, for both algorithms.

    Above zero the search wins; below it the fixed network does. The figure's
    claim is that both algorithms cross zero at the same place.
    """
    fig, axes = _fig(2, 1, figsize=(WIDTH, 6.6))
    for ax, spec in zip(np.atleast_1d(axes).ravel(), ALGORITHMS):
        # Zero is the line the whole figure is read against, so it is the
        # project's dashed grey one, and it is drawn first so the bars sit on
        # top of it rather than it on top of them.
        parity(ax)
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
            loc="lower left", ncol=1,
        )
        ax.set_xticks(range(len(ALL_TASKS)))
        ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=TICK_SIZE,
                           color=INK2)
        ax.set_ylabel("median paired difference\n(search − fixed)",
                      fontsize=LABEL_SIZE, color=INK2)
        title(ax, spec["algorithm"])
    fig.suptitle(
        "The reversal, both algorithms. A hollow bar did not reach\n"
        "significance after Holm correction in that algorithm's own family.",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    return _save(fig, out, top=0.90, bottom=0.05, left=0.155, right=0.99,
                 hspace=0.30)


def fig_budget(budgets: list[dict], out: Path) -> Path:
    """What each arm actually spent. The confound, made visible."""
    by = {(r["task"], r["condition"]): r for r in budgets}
    conds = ["bpneat", "cgp", "fixed_tanh_ha",
             "fixed_tanh_matched_bpneat", "fixed_tanh_matched_cgp"]
    fig, ax = _fig(figsize=(WIDTH, 3.5))
    width = 0.16
    for ci, c in enumerate(conds):
        xs = [ti + (ci - 2) * width for ti in range(len(ALL_TASKS))]
        vals = [by.get((t, c), {}).get("gradient_steps_mean", np.nan) for t in ALL_TASKS]
        colour, _ = STYLE[c]
        # "@ BP-NEAT budget" and "@ CGP budget" are the same architecture at two
        # budgets, so the role table gives them one hue on purpose. Here nothing
        # labels the bar itself, so the second of the pair takes a hatch: a
        # second channel for a second variable, rather than a hue that would
        # claim they are different architectures.
        ax.bar(xs, vals, width=width * 0.9, color=colour, label=LABEL[c],
               hatch="///" if c.endswith("_cgp") else None,
               edgecolor=SURFACE, linewidth=0.0)
    ax.set_yscale("log")
    # A bar on a log axis measures from wherever autoscale put the floor, which
    # is not a number anyone declared. Pin it to one gradient update, so a bar's
    # length is "orders of magnitude above doing nothing" and the two-decade gap
    # the title claims is the gap the reader sees.
    ax.set_ylim(1, None)
    ax.set_xticks(range(len(ALL_TASKS)))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=TICK_SIZE,
                       color=INK2)
    ax.set_ylabel("gradient updates per run (log)", fontsize=LABEL_SIZE, color=INK2)
    # A log axis has no empty floor: every bar is drawn from the bottom of the
    # frame upward, so the key placed "inside, low" sat across five bars, and
    # with no box behind it — this style's legends are unboxed — it could not be
    # read at all. It goes below the task labels, where nothing is drawn.
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.11))
    title(
        ax,
        "Realized gradient budget. The unmatched control spends about\n"
        "two orders of magnitude less than the searches it is compared against.",
    )
    return _save(fig, out)


def fig_cross_algorithm(cross: list[dict], out: Path) -> Path:
    """How far apart the algorithms are, against how far the protocol moves things."""
    fig, ax = _fig(figsize=(WIDTH, 3.4))
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
        ax.text(x - 0.19, b + 0.004, f"{b:.3f}", ha="center", fontsize=ANNOT_SIZE,
                color=INK2)
        ax.text(x + 0.19, p + 0.004, f"{p:.3f}", ha="center", fontsize=ANNOT_SIZE,
                color=INK2)
    # Headroom for the value label above the tallest bar, which otherwise was
    # drawn outside the axes and ran into the title.
    # Headroom for the value label above the tallest bar *and* for the key,
    # which sits over the short XOR and Circles bars: at 1.15 the label on the
    # tallest bar was written through the second legend entry.
    ax.set_ylim(0, max(between + protocol) * 1.38)
    ax.set_xticks(xs)
    ax.set_xticklabels([TASK_LABEL[r["task"]] for r in cross], fontsize=TICK_SIZE,
                       color=INK2)
    ax.set_ylabel("median paired difference in accuracy", fontsize=LABEL_SIZE,
                  color=INK2)
    ax.legend(loc="upper left")
    title(ax, "Choosing the algorithm matters less than choosing the budget protocol")
    return _save(fig, out)


def fig_hypotheses(hyp: list[dict], out: Path) -> Path:
    """The preregistered scorecard, as declared, with the observed counts."""
    fig, ax = _fig(figsize=(WIDTH, 0.82 * len(hyp) + 0.9))
    ax.grid(False)
    for i, h in enumerate(reversed(hyp)):
        good = h["verdict"] == "holds"
        # A thin rule in the margin, not a 110px block of saturated colour:
        # the verdict is one bit and it is already written out in words at the
        # end of the row, so it needs a mark, not a field.
        ax.add_patch(
            plt.Rectangle((0, i - 0.40), 0.008, 0.80,
                          facecolor=GOOD if good else BAD, edgecolor="none")
        )
        ax.text(0.028, i + 0.26, f"{h['hypothesis']}  {h['statement']}",
                fontsize=PANEL_TITLE_SIZE, color=INK, va="center")
        # Rule and outcome on separate lines. Set end to end they ran to about
        # 115 characters, which is 8in of type and so could only be read on a
        # figure far wider than a README renders.
        ax.text(0.028, i - 0.02, f"rule: {h['decision_rule']}",
                fontsize=LABEL_SIZE, color=INK2, va="center")
        ax.text(0.028, i - 0.28, f"observed: {h['observed']}",
                fontsize=LABEL_SIZE, color=INK2, va="center")
        ax.axhline(i - 0.5, color=GRID, linewidth=0.7)
        ax.text(0.995, i + 0.26, h["verdict"], fontsize=LABEL_SIZE,
                color=GOOD if good else BAD, ha="right", va="center",
                fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(hyp) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    title(ax, "Preregistered hypotheses, scored by their own declared rules")
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

    fig, axes = _fig(2, 1, figsize=(WIDTH, 6.0))
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
                ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=ANNOT_SIZE,
                        color=INK2)
    ax.set_xticks(range(len(ALL_TASKS)))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=TICK_SIZE,
                       color=INK2)
    ax.set_ylabel("active function nodes in the champion", fontsize=LABEL_SIZE,
                  color=INK2)
    ax.legend()
    title(ax, "Phenotype size")

    ax = axes[1]
    vals = [neutral.get(t, []) for t in ALL_TASKS]
    parts = ax.violinplot([v or [0] for v in vals], showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        # A flat pale fill rather than the hue at 55% opacity: a translucent
        # body let the grid lines show through the distribution, which read as
        # structure in the data that is not there.
        body.set_facecolor(pale(colour_of("cgp"), 0.55))
        body.set_alpha(1.0)
        body.set_edgecolor(INK2)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if key in parts:
            parts[key].set_color(INK2)
    ax.set_xticks(range(1, len(ALL_TASKS) + 1))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=TICK_SIZE,
                       color=INK2)
    ax.set_ylabel("generations accepted at equal fitness", fontsize=LABEL_SIZE,
                  color=INK2)
    title(ax, "Neutral drift, per run")

    fig.suptitle(
        "CGP's genotype-phenotype map in use: small phenotypes,\n"
        "and how often neutral offspring were accepted",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    return _save(fig, out, top=0.88, bottom=0.06, left=0.135, right=0.99,
                 hspace=0.28)


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
