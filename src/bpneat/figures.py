"""Publication figures for v1 and v2, rendered from the raw records.

Every figure is derived from ``raw/runs/*.json`` — nothing is recomputed and no
model is retrained — so a figure and the CSV beside it cannot disagree. Every
figure also draws a *measured result*: the three input geometries used to be
drawn here as well, and that panel is gone, because a picture of the problem
setup reads as a finding once it sits in a release's figure set.

The look comes from `bpneat.style` and from nowhere else: white ground, no
frame, top and right spines off, a very light grid behind the data, stock sans,
legends without a box, and left-aligned sentences for titles. That is the style
of the sibling project `competitive-coevolution-of-slimes`, so the two read as
one body of work. This module defines no hue, no type size and no density.

Panels stack downwards and no figure is wider than the page. A 11.5in strip of
three panels side by side arrives in a README at well under half size, which
put the 7.5pt tick labels below 4pt — the single largest cause of an
unreadable figure here, and not one any palette could fix.

Colour follows the shared condition-to-role table, so a condition cannot be one
colour here and another in the README, and it is never the only carrier of
identity: every bar and every point is directly labelled, and every figure has
a CSV table view in the release. That labelling is also the relief for the few
hues that sit near the floor of what still separates from white.

The committed PNGs of these two releases are deliberately never regenerated —
they are frozen evidence, and nothing in the README or the paper points a
reader at them — so what follows conforms in code without being re-rendered.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import load_final_test, load_runs, operation_usage, paired_effects, summarise  # noqa: E402
from .conditions import CORE_CONDITIONS, SUCCESS_THRESHOLD  # noqa: E402

from .style import (  # noqa: E402
    ANNOT_SIZE,
    CONTROL_MATCHED,
    GRID,
    INK,
    INK2,
    OP_COLOUR,
    RULE,
    SEARCH_PRIMARY,
    SEARCH_SECONDARY,
    SURFACE,
    TASK_LABEL,
    TICK_SIZE,
    TITLE_SIZE,
    colour_of,
    parity,
    save,
    style_axes,
    title,
    vparity,
)

#: Colour by what the condition *is*, not by where it sits in a list, and from
#: the same table every other protocol uses.
CONDITION_COLOR = {c: colour_of(c) for c in CORE_CONDITIONS}
LABEL = {
    "backprop_neat": "Backprop-NEAT",
    "homogeneous_tanh": "Homog. tanh",
    "evolution_only": "Evolution only",
    "random_search": "Random arch.",
    "fixed_mlp": "Fixed MLP",
    "logistic": "Logistic",
}
TASKS = ("xor", "circle", "spiral")

# Every width below is 6.6in — one column, page width — written as a literal
# so that the gate which reads widths off `figsize=` can actually read them.
# Each per-task figure is three panels of the same quantity, so they stack:
# side by side they cost two thirds of the type size, and the type size is
# what made these unreadable.


def _style(ax) -> None:
    style_axes(ax)


def _cat_axis(ax, x, conds) -> None:
    """Rotated condition labels, anchored so they do not drift into the neighbour."""
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABEL[c] for c in conds], rotation=30, ha="right",
        rotation_mode="anchor", fontsize=TICK_SIZE, color=INK2,
    )
    ax.set_xlim(-0.7, len(conds) - 0.3)


def _fig(nrows=1, ncols=1, figsize=(6.6, 4.2)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _headline(fig, text: str) -> None:
    """The figure-level title, set like every other title here: a sentence,
    flush left, in the shared size."""
    fig.suptitle(text, x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE)


def _shared_ylabel(axes, text: str) -> None:
    """One label for a stacked column that shows one quantity three times.

    Repeating it on all three axes spends width this column does not have;
    dropping it entirely leaves the two outer panels unlabelled. It goes on the
    middle axis, where it reads as belonging to the column.
    """
    axes[len(axes) // 2].set_ylabel(text)


def _label_ink(fill: str) -> str:
    """Whichever of the project's two inks a label *on* this fill can be read in.

    Not a new colour — it only chooses between INK and SURFACE. The operator
    palette includes the ink hue itself, so a band label drawn unconditionally
    in INK disappeared on `add`, and one drawn unconditionally in the surface
    colour disappears on the pale bands now that the surface is white.
    """

    def lum(colour: str) -> float:
        rgb = np.array(mcolors.to_rgb(colour))
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])

    def contrast(a: str, b: str) -> float:
        la, lb = lum(a), lum(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

    return INK if contrast(INK, fill) >= contrast(SURFACE, fill) else SURFACE


def _save(fig, path: Path, **adjust) -> Path:
    """One exit for every figure here, so none can be written at another
    density or onto another ground than the project's.

    Callers pass `hspace` rather than a bottom margin: with the panels stacked,
    the space that has to be reserved is between them — a rotated category
    label under one panel must not reach the title of the panel below it.
    """
    return save(fig, path, **adjust)


# --------------------------------------------------------------------------


def validation_performance(rows: list[dict], out: Path, track: str) -> Path:
    fig, axes = _fig(3, 1, figsize=(6.6, 8.4))
    by = {(r["task"], r["condition"]): r for r in rows}
    for ax, task in zip(axes, TASKS):
        conds = [c for c in CORE_CONDITIONS if (task, c) in by]
        vals = [by[(task, c)]["validation_accuracy_mean"] for c in conds]
        sds = [by[(task, c)]["validation_accuracy_sd"] for c in conds]
        x = np.arange(len(conds))
        # Mean and SD is all these two releases' records carry per condition;
        # the per-replicate strip the project prefers needs the run values,
        # which are not in `rows`. So the bar keeps its error cap here, and the
        # dispersion stays in the ink weight rather than the rule weight — a
        # cap is data, unlike the threshold line below it.
        ax.bar(x, vals, width=0.68, color=[CONDITION_COLOR[c] for c in conds],
               yerr=sds, ecolor=INK2, capsize=3, error_kw={"linewidth": 1.1})
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", fontsize=ANNOT_SIZE, color=INK)
        parity(ax, SUCCESS_THRESHOLD[task])
        # The threshold is labelled on the left, not at the right edge where
        # `parity` would put it: that is where the last bar's value sits.
        ax.text(-0.45, SUCCESS_THRESHOLD[task] + 0.015, "success threshold",
                ha="left", fontsize=ANNOT_SIZE, color=RULE)
        _cat_axis(ax, x, conds)
        ax.set_ylim(0, 1.16)
        title(ax, TASK_LABEL[task])
    _shared_ylabel(axes, "validation accuracy")
    _headline(fig, f"Track {track}: validation accuracy of the selected champion "
                   "(mean ± SD over replicates)")
    return _save(fig, out / f"validation-performance-track-{track.lower()}.png",
                 hspace=0.45)


def track_contrast(rows_a: list[dict], rows_b: list[dict], out: Path) -> Path:
    """The propagation finding: Ha's exact rule against settled propagation."""
    fig, axes = _fig(3, 1, figsize=(6.6, 8.4))
    a = {(r["task"], r["condition"]): r for r in rows_a}
    b = {(r["task"], r["condition"]): r for r in rows_b}
    for ax, task in zip(axes, TASKS):
        conds = [c for c in CORE_CONDITIONS if (task, c) in a and (task, c) in b]
        x = np.arange(len(conds))
        va = [a[(task, c)]["validation_accuracy_mean"] for c in conds]
        vb = [b[(task, c)]["validation_accuracy_mean"] for c in conds]
        # Two propagation rules for the same search: one family, two values.
        ax.bar(x - 0.2, va, width=0.38, color=SEARCH_PRIMARY, label="Track A — ha2016")
        ax.bar(x + 0.2, vb, width=0.38, color=SEARCH_SECONDARY, label="Track B — settled")
        for xi, v in zip(x - 0.2, va):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=ANNOT_SIZE, color=INK)
        for xi, v in zip(x + 0.2, vb):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=ANNOT_SIZE, color=INK)
        _cat_axis(ax, x, conds)
        ax.set_ylim(0, 1.18)
        title(ax, TASK_LABEL[task])
    _shared_ylabel(axes, "validation accuracy")
    axes[0].legend(labelcolor=INK2, loc="lower left")
    _headline(fig, "Propagation rule changes what evolution can discover")
    return _save(fig, out / "propagation-track-contrast.png", hspace=0.45)


def paired_effects_figure(effects: list[dict], out: Path, track: str) -> Path:
    rows = [e for e in effects if e["metric"] == "validation_accuracy"]
    if not rows:
        return None
    fig, axes = _fig(3, 1, figsize=(6.6, 7.2))
    for ax, task in zip(axes, TASKS):
        items = [e for e in rows if e["task"] == task]
        items.sort(key=lambda e: e["mean_difference"])
        y = np.arange(len(items))
        for yi, e in zip(y, items):
            lo, hi = e["ci95_low"], e["ci95_high"]
            # Above zero the search is ahead, below it the control is; that is
            # the project's one diverging encoding, so it uses its two anchors.
            colour = SEARCH_PRIMARY if e["mean_difference"] > 0 else CONTROL_MATCHED
            ax.plot([lo, hi], [yi, yi], color=colour, linewidth=2.4, solid_capstyle="round")
            # The white keyline, so two intervals that land on top of each
            # other still read as two.
            ax.plot(e["mean_difference"], yi, "o", color=colour, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.6)
            ax.text(hi + 0.012, yi, f"{e['mean_difference']:+.3f}", va="center",
                    fontsize=ANNOT_SIZE, color=INK)
        vparity(ax, 0)
        ax.set_yticks(y)
        ax.set_yticklabels(
            ["vs " + LABEL[e["comparison"].split(" - ")[1]] for e in items],
            fontsize=TICK_SIZE, color=INK2,
        )
        title(ax, TASK_LABEL[task])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    # The quantity is the same in all three panels, so it is named once, under
    # the one axis a reader reaches last.
    axes[-1].set_xlabel("paired difference in validation accuracy\n"
                        "(Backprop-NEAT − control)")
    _headline(fig, f"Track {track}: paired within-replicate differences, "
                   "95% bootstrap interval")
    return _save(fig, out / f"paired-condition-effects-track-{track.lower()}.png",
                 hspace=0.45)


def represented_vs_causal(rows: list[dict], out: Path, track: str) -> Path:
    """H5: represented structure is not the computation that reaches the output."""
    fig, axes = _fig(3, 1, figsize=(6.6, 8.4))
    by = {(r["task"], r["condition"]): r for r in rows}
    for ax, task in zip(axes, TASKS):
        conds = [c for c in CORE_CONDITIONS if (task, c) in by]
        x = np.arange(len(conds))
        rep = [by[(task, c)]["represented_nodes_mean"] - 4 for c in conds]
        cau = [by[(task, c)]["causal_hidden_nodes_mean"] for c in conds]
        # Carried structure against the structure that actually computes: the
        # same quantity at two depths, so one family at two values.
        ax.bar(x - 0.2, rep, width=0.38, color=SEARCH_SECONDARY,
               label="represented hidden")
        ax.bar(x + 0.2, cau, width=0.38, color=SEARCH_PRIMARY, label="causally active")
        for xi, v in zip(x - 0.2, rep):
            ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=ANNOT_SIZE, color=INK)
        for xi, v in zip(x + 0.2, cau):
            ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=ANNOT_SIZE, color=INK)
        _cat_axis(ax, x, conds)
        title(ax, TASK_LABEL[task])
    _shared_ylabel(axes, "hidden nodes")
    axes[0].legend(labelcolor=INK2)
    _headline(fig, f"Track {track}: represented vs causally active hidden structure")
    return _save(fig, out / f"represented-vs-causal-track-{track.lower()}.png",
                 hspace=0.45)


def operator_usage(runs: list[dict], out: Path, track: str) -> Path:
    usage = [u for u in operation_usage(runs) if u["condition"] == "backprop_neat"]
    if not usage:
        return None
    ops = sorted({u["operator"] for u in usage})
    fig, ax = _fig(figsize=(6.6, 4.2))
    frac = defaultdict(dict)
    for u in usage:
        frac[u["task"]][u["operator"]] = u["fraction"]
    x = np.arange(len(TASKS))
    bottom = np.zeros(len(TASKS))
    # Operators have their own fixed hues; cycling a six-slot series over them
    # meant the same operator changed colour between figures.
    for op in ops:
        colour = OP_COLOUR[op]
        vals = np.array([frac[t].get(op, 0.0) for t in TASKS])
        # The white keyline the rest of the project uses, so adjacent bands
        # separate from each other. An ink outline round every band instead
        # drew the stack's own scaffolding over its data.
        ax.bar(x, vals, width=0.6, bottom=bottom, color=colour, label=op,
               edgecolor=SURFACE, linewidth=0.8)
        for xi, (v, b0) in enumerate(zip(vals, bottom)):
            if v > 0.07:
                ax.text(xi, b0 + v / 2, op, ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=_label_ink(colour))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], fontsize=TICK_SIZE, color=INK2)
    ax.set_ylabel("share of causal hidden operators")
    ax.legend(labelcolor=INK2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    title(ax, f"Track {track}: which operators reach the output, by task "
              "(Backprop-NEAT champions)")
    return _save(fig, out / f"operator-usage-track-{track.lower()}.png")


def performance_vs_compute(rows: list[dict], out: Path, track: str) -> Path:
    """Every point is directly labelled, so identity never rests on colour."""
    fig, axes = _fig(3, 1, figsize=(6.6, 7.8))
    by = {(r["task"], r["condition"]): r for r in rows}
    for ax, task in zip(axes, TASKS):
        for c in CORE_CONDITIONS:
            r = by.get((task, c))
            if r is None:
                continue
            xs = max(r["gradient_steps_mean"], 1)
            # The white keyline: two conditions that realized the same budget
            # land on the same x and would otherwise merge into one mark.
            ax.plot(xs, r["validation_accuracy_mean"], "o", markersize=9,
                    color=CONDITION_COLOR[c], markeredgecolor=SURFACE, markeredgewidth=1.6)
            ax.annotate(LABEL[c], (xs, r["validation_accuracy_mean"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=ANNOT_SIZE, color=INK)
        ax.set_xscale("symlog", linthresh=100)
        title(ax, TASK_LABEL[task])
        ax.set_ylim(0.35, 1.12)
    _shared_ylabel(axes, "validation accuracy")
    axes[-1].set_xlabel("realized gradient steps")
    _headline(fig, f"Track {track}: performance against realized compute")
    return _save(fig, out / f"performance-vs-compute-track-{track.lower()}.png",
                 hspace=0.45)


def build_all(root: Path, progress=print) -> list[Path]:
    """Render every figure that the available records support."""
    root = Path(root)
    out = root / "figures"
    written: list[Path] = []

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

    # Wiring only: the paired sealed-test-loss forest plot belongs to the same
    # release figure set, but its code lives in `champions.py` so that the
    # seven frozen science modules are not the only thing this file has to
    # stay careful about. Imported late to keep the dependency one-way.
    from .champions import paired_test_loss

    if "B" in per_track:
        written.append(paired_test_loss(root, out, "track-b"))
    return [w for w in written if w]
