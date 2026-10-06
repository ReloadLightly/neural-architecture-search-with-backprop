"""v3 figures, rendered from the raw records.

Every figure here is drawn in the one register this project shares with
`competitive-coevolution-of-slimes`: a white ground, no frame, top and right
spines off, a very light grid behind the data, stock sans on a single type
ladder, unboxed legends, and a left-aligned sentence for a title. That look is
defined once, in `bpneat.style`, and nothing below picks a hue, a type size or
a density of its own.

Colour is semantic rather than positional. Every condition in this protocol is
either something we searched for or something we fixed in advance, so a cool
hue means "searched" and a warm one means "fixed", and the condition-to-colour
table is the shared one in `bpneat.style` — a condition cannot be one colour
here and another in v4 or in the README. Colour never carries identity alone:
every mark is directly labelled and every figure has a CSV beside it in the
release.
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
    GOOD,
    GRID,
    INK,
    INK2,
    LABEL_SIZE,
    LEGEND_SIZE,
    NEUTRAL,
    PANEL_TITLE_SIZE,
    RULE,
    SEARCH_NULL,
    SEARCH_PRIMARY,
    SURFACE,
    TICK_SIZE,
    TITLE_SIZE,
    colour_of,
    pale,
    parity,
    save,
    style_axes,
    title,
    vparity,
)
from .analysis import (  # noqa: E402
    REFERENCE,
    block_effects,
    load,
    selection_dose_response,
    stability_matrix,
    summarise,
)
from .datasets import TASKS  # noqa: E402

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
from ..style import TASK_LABEL  # noqa: E402


def _style(ax):
    style_axes(ax)


def _fig(nrows=1, ncols=1, figsize=(10, 4.2)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path: Path, **kw):
    # One exit for every figure in the module, so nothing can be written at
    # another density or onto another ground than the project's.
    return save(fig, path, **kw)


def stability(matrix: list[dict], out: Path) -> Path | None:
    """The central figure: does each v2 claim survive each evaluator setting?"""
    if not matrix:
        return None
    cols = [k for k in matrix[0] if k not in ("claim", "statement", "task")]
    fig, ax = _fig(figsize=(6.6, 0.52 * len(matrix) + 1.8))
    ax.grid(False)
    # "not applicable" is the page itself with a hairline round it: the cell
    # exists, nothing was measured in it, and it should read as absence rather
    # than as a sixth pigment.
    colour = {"supported": GOOD, "reversed": BAD, "not significant": NEUTRAL,
              "n/a": SURFACE}
    for r, row in enumerate(matrix):
        for c, col in enumerate(cols):
            v = row[col]
            # A pale wash carrying the verdict's hue, with the verdict written
            # on it in ink. The version this replaces reversed the lettering
            # out of a saturated block; on a white ground that sets the two
            # cells the reader most needs to be told about — "n.s." and the
            # empty "n/a" — in white on something close to white.
            ax.add_patch(plt.Rectangle(
                (c, -r), 1, 1, facecolor=pale(colour.get(v, NEUTRAL)),
                edgecolor=RULE if v == "n/a" else SURFACE,
                linewidth=0.7 if v == "n/a" else 3))
            short = {"supported": "supported", "reversed": "REVERSED",
                     "not significant": "n.s.", "n/a": "—"}[v]
            ax.text(c + 0.5, -r + 0.5, short, ha="center", va="center",
                    fontsize=LABEL_SIZE,
                    color=RULE if v == "n/a" else INK)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(-len(matrix) + 1, 1)
    ax.set_xticks([c + 0.5 for c in range(len(cols))])
    ax.set_xticklabels(cols, rotation=28, ha="right", fontsize=TICK_SIZE, color=INK2)
    ax.set_yticks([-r + 0.5 for r in range(len(matrix))])
    ax.set_yticklabels([f"{m['claim']}  {m['statement']}" for m in matrix],
                       fontsize=LABEL_SIZE, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    title(ax, "Conclusion stability: which v2 claims survive which evaluator")
    # C2 and C3 compare against ablations that share the candidate budget by
    # construction, so their row is identical across control definitions. That
    # invariance is the point, not a rendering artifact.
    fig.text(0.30, 0.068,
             "Columns are control definitions. Rows C2 and C3 compare against "
             "budget-matched ablations,\nso they are invariant across columns by "
             "construction — that invariance is the finding.",
             fontsize=LABEL_SIZE, color=INK2, ha="left")
    return _save(fig, out / "stability-matrix.png",
                 bottom=0.32, left=0.30, top=0.88, right=0.99)


#: Block A only. The compute figure is about controls and budgets; the
#: propagation, selection and inheritance conditions belong to other blocks.
#: Shape per condition; colour comes from the condition's role, so a reader who
#: learns "cool is a search, warm is a fixed network" on one figure keeps it on
#: every other. Two conditions share a hue only where they are the same kind of
#: thing, and the marker separates them, so identity never rests on colour
#: alone.
BLOCK_A_STYLE = {
    "backprop_neat": "o",
    "fixed_mlp_tanh_ha": "^",
    "fixed_mlp_tanh_matched": "v",
    "fixed_mlp_sin_matched": "P",
    "fixed_mlp_mixed_matched": "X",
    "random_search_matched": "*",
    "homogeneous_tanh": "s",
    "evolution_only": "D",
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
    fig, axes = _fig(5, 1, figsize=(6.6, 2.0 * 5))
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
            marker = BLOCK_A_STYLE[c]
            x = max(r["gradient_steps_mean"], 1)
            y = r["test_accuracy_mean"]
            # A star's ink sits inside its bounding box, so at a shared size
            # it renders about half the weight of a filled circle. The white
            # keyline is what keeps two conditions that land on top of each
            # other readable as two marks.
            (h,) = ax.plot(x, y, marker, markersize=14 if marker == "*" else 9,
                           color=colour_of(c), markeredgecolor=SURFACE,
                           markeredgewidth=1.5)
            handles.setdefault(LABEL[c], h)
            if c in ("fixed_mlp_tanh_ha", best_matched):
                ax.annotate(
                    f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=ANNOT_SIZE, color=INK,
                )
        # Explicit ticks: with linthresh=1000 the 0 tick and the 10^2 tick land
        # a tenth of the linear segment apart and overprint into one glyph.
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_xticks([0, 1e3, 1e4, 1e5])
        title(ax, TASK_LABEL[task])
        ax.set_xlabel("realized gradient steps", color=INK2, fontsize=LABEL_SIZE)
        ax.set_ylim(0.45, 1.08)
    axes[0].set_ylabel("sealed-test accuracy", color=INK2, fontsize=LABEL_SIZE)
    fig.legend(handles.values(), handles.keys(), fontsize=LEGEND_SIZE,
               labelcolor=INK2, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Block A: accuracy against realized compute — v2's starved control sits "
        "at the left edge",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    return _save(fig, out / "budget-vs-accuracy.png",
                 bottom=0.26, top=0.88, left=0.05, right=0.99, wspace=0.18)


def block_a_effects(effects: list[dict], out: Path) -> Path | None:
    rows = [e for e in effects if e.get("block") == "A"]
    if not rows:
        return None
    tasks = [t for t in TASKS if any(e["task"] == t for e in rows)]
    # One row order for every panel, and one x range. Re-sorting inside each
    # panel put the same condition at a different height in each, and letting
    # each panel autoscale meant a +0.01 effect in XOR sat at the same screen
    # position as a +0.08 effect in the 3-arm spiral.
    order = sorted(
        {e["condition"] for e in rows},
        key=lambda c: np.median([e["median_difference"] for e in rows
                                 if e["condition"] == c]),
    )
    span = max(abs(v) for e in rows for v in (e["ci95_low"], e["ci95_high"]))
    fig, axes = _fig(len(tasks), 1, figsize=(6.6, 1.9 * len(tasks)))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        by_cond = {e["condition"]: e for e in rows if e["task"] == task}
        items = [by_cond[c] for c in order if c in by_cond]
        y = np.arange(len(items))
        # Zero is the line every interval is read against, so it is the
        # project's dashed grey one, and it is drawn first so the intervals sit
        # on top of it rather than it on top of them.
        vparity(ax)
        for yi, e in zip(y, items):
            sig = e.get("holm_significant", False)
            # Cool where the search is ahead, warm where the control is — the
            # project's two families, not a green/red verdict. "The fixed
            # network won" is a result, and colouring it as an error said
            # otherwise.
            col = SEARCH_PRIMARY if e["median_difference"] > 0 else CONTROL_BEST
            col = col if sig else NEUTRAL
            ax.plot([e["ci95_low"], e["ci95_high"]], [yi, yi], color=col, linewidth=2.6,
                    solid_capstyle="round")
            ax.plot(e["median_difference"], yi, "o", markersize=7, color=col,
                    markeredgecolor=SURFACE, markeredgewidth=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [LABEL.get(e["condition"], e["condition"]) for e in items],
            fontsize=TICK_SIZE, color=INK2)
        ax.set_xlim(-span * 1.08, span * 1.08)
        title(ax, TASK_LABEL[task])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.suptitle("Block A: median paired difference in sealed-test accuracy "
                 "(Backprop-NEAT − control).\nBlue: the search is ahead. "
                 "Red: the fixed control is. Grey: not significant after Holm.",
                 x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE)
    return _save(fig, out / "block-a-paired-effects.png",
                 top=0.85, bottom=0.08, left=0.155, right=0.99, wspace=0.08)


def dose_response(dose: list[dict], out: Path) -> Path | None:
    if not dose:
        return None
    tasks = sorted({d["task"] for d in dose})
    fig, axes = _fig(len(tasks), 2, figsize=(6.6, 2.3 * len(tasks)))
    # One row per geometry, two columns: the panels used to run along a single
    # row, which is why the index below was flat.
    axes = np.atleast_2d(axes).reshape(len(tasks), 2)
    for ti, task in enumerate(tasks):
        items = [d for d in dose if d["task"] == task and d["selection_intensity_mean"] != ""]
        items.sort(key=lambda d: d["selection_intensity_mean"])
        x = [d["selection_intensity_mean"] for d in items]
        # Two panels, two different measured quantities of one search. A failure
        # rate is a failure rate, so it takes the project's warning colour; the
        # size of what the search built takes the search colour.
        for j, (key, lab, colour) in enumerate(
            (("collapse_rate", "collapse rate", BAD),
             ("causal_hidden_nodes_mean", "causal hidden nodes", SEARCH_PRIMARY))
        ):
            ax = axes[ti, j]
            ax.plot(x, [d[key] for d in items], "o-", color=colour, linewidth=2, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.3)
            for d in items:
                if d["selector"] in ("roulette_s0.01", "v1_truncation"):
                    ax.annotate("Ha" if "roulette" in d["selector"] else "v1",
                                (d["selection_intensity_mean"], d[key]),
                                textcoords="offset points", xytext=(0, 10),
                                ha="center", fontsize=ANNOT_SIZE, color=INK)
            if key == "collapse_rate":
                # Autoscale on an all-zero series produced a panel of negative
                # collapse rates, which read as a measurement rather than a
                # floor. The quantity is a share; show its whole domain.
                ax.set_ylim(-0.02, 1.0)
                if max(d[key] for d in items) == 0:
                    ax.text(0.5, 0.5, "no run collapsed at any\nselection intensity",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=LABEL_SIZE, color=INK2, style="italic")
            ax.set_xlabel("realized selection intensity", color=INK2, fontsize=LABEL_SIZE)
            ax.set_ylabel(lab, color=INK2, fontsize=LABEL_SIZE)
            title(ax, f"{TASK_LABEL[task]} — {lab}")
    fig.suptitle("Block C: selection pressure against collapse and causal size",
                 x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE)
    # Each row carries its own x axis, so the gap has to clear a tick row and
    # the next panel's heading.
    return _save(fig, out / "selection-dose-response.png",
                 top=0.90, bottom=0.08, hspace=0.75, wspace=0.42)


def propagation_grid(summary: list[dict], out: Path) -> Path | None:
    """Block B: the 2x2 that v2 could not separate.

    Two factors, so two visual channels: the propagation rule is the colour,
    the fitness split is the position within a pair. The earlier version drew
    all four bars in one blue and printed the collapse rate inside the bar,
    where it was clipped against the axis whenever the bar was short — the one
    number the figure exists to show was the one hardest to read.
    """
    cells = {
        ("ha2016", "train"): "prop_ha_fit_train",
        ("ha2016", "validation"): "prop_ha_fit_val",
        ("settled", "train"): "prop_settled_fit_train",
        ("settled", "validation"): REFERENCE,
    }
    #: One hue per propagation rule, two values of the search family: both are
    #: searches, and the rule is the factor under test.
    PROP_COLOUR = {"ha2016": SEARCH_NULL, "settled": SEARCH_PRIMARY}
    by = {(r["task"], r["condition"]): r for r in summary}
    tasks = [t for t in ("xor", "circle", "spiral")
             if all((t, c) in by for c in cells.values())]
    if not tasks:
        return None
    fig, axes = _fig(len(tasks), 1, figsize=(6.6, 2.5 * len(tasks)))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        labels, accs, colls, colours = [], [], [], []
        for (prop, split), cond in cells.items():
            r = by[(task, cond)]
            labels.append(split)
            accs.append(r.get("test_accuracy_mean", r["validation_accuracy_mean"]))
            colls.append(r["collapse_rate"])
            colours.append(PROP_COLOUR[prop])
        # Grouped by rule, with a gap between the groups rather than a uniform row.
        x = np.array([0.0, 0.85, 2.05, 2.90])
        ax.bar(x, accs, width=0.68, color=colours)
        # Chance is the reference the bars are read against, in the project's
        # one dashed-grey register.
        parity(ax, 0.5, "chance")
        for xi, (a, c) in enumerate(zip(accs, colls)):
            ax.text(x[xi], a + 0.022, f"{a:.2f}", ha="center", fontsize=LABEL_SIZE,
                    color=INK)
            if c > 0:
                # Outside the bar, under the axis, in the warning colour: a
                # collapse rate is the figure's second finding, not a footnote.
                # Just the number, because "collapse 20%" under adjacent bars
                # is wider than the bar spacing and the two labels collided.
                ax.text(x[xi], -0.145, f"{c:.0%}", ha="center",
                        fontsize=LABEL_SIZE, color=BAD, clip_on=False)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=TICK_SIZE, color=INK2)
        for cx, prop in ((x[:2].mean(), "ha2016"), (x[2:].mean(), "settled")):
            ax.text(cx, -0.235, prop, ha="center", fontsize=PANEL_TITLE_SIZE,
                    color=PROP_COLOUR[prop], clip_on=False)
        ax.set_ylim(0, 1.15)
        ax.set_xlim(-0.6, x[-1] + 1.1)
        title(ax, TASK_LABEL[task])
    np.atleast_1d(axes)[0].set_ylabel(
        "sealed-test accuracy", color=INK2, fontsize=LABEL_SIZE)
    fig.suptitle(
        "Block B: propagation rule x fitness split, separated. Colour is the "
        "propagation rule; the pair is the split it was scored on.\n"
        "The red figure under a bar is the share of that cell's runs that "
        "collapsed to a constant output.",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    return _save(fig, out / "propagation-grid.png", top=0.80, bottom=0.20)


def build_all(release_dir: Path, progress=print) -> list[Path]:
    release_dir = Path(release_dir)
    out = release_dir / "figures"
    runs, final = load(release_dir)
    summary = summarise(runs, final)
    effects = block_effects(runs, final)
    written = [budget_vs_accuracy(summary, out)]
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
