"""v5 figures: what NEAT's mechanisms did to the topologies and to the score.

Every token — ground, ink, grid, palette, type scale, density — comes from
`bpneat.style`, which is the `competitive-coevolution-of-slimes` style ported
into this repository so that the two read as one body of work. Nothing here
defines a hue or a size of its own, and every figure leaves through
`style.save`, so the exit cannot be forgotten at a call site.

The conventions this module keeps: a hue is assigned by what a row *is* and
never by where it sits in a list; identity is never carried by colour alone
(shape and dash separate the arms inside a band, and every mark is either
directly labelled or listed in the CSV that ships beside the figure in the
release); titles are left-aligned sentences that say what the panel shows; and
there is a dashed grey reference line to measure everything against.
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
    GOOD,
    INK,
    INK2,
    NEUTRAL,
    RULE,
    SEARCH_NULL,
    SEARCH_PRIMARY,
    SEARCH_SECONDARY,
    SURFACE,
    TASK_LABEL,
    TITLE_SIZE,
    colour_of,
    dot,
    parity,
    save,
    suptitle,
    title,
    vparity,
    within,
)
from ..style import figure as _fig  # noqa: E402
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

#: Every arm here is a search, so they all take the search family. The three
#: bands are not decoration: they are the preregistered question. Does releasing
#: the brake on size make topologies grow? So one band is the arm with both
#: brakes on, one is the arms that release one or both, and one is the arms that
#: change the search without touching size. The reader sees the answer as three
#: bands before reading a single label.
#:
#: Two defects this scheme has had to fix in turn. First, colour was assigned by
#: position over a six-slot categorical palette; with eight arms it cycled, so
#: "deep & narrow" came out the same hue as the reference. Then the bands were
#: drawn as three tints of one hue, and at this type size, with a white keyline
#: round every mark, deep/mid/pale blue collapsed back into one blue and the
#: banding carried nothing again. So the bands now take three *separate* palette
#: hues — the role colours, which are the ones the palette search validated as
#: mutually separable under normal, protanopic and deuteranopic vision.
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

#: How the bands are named in the captions, so the sentence under a figure and
#: the ink in it cannot drift apart. Stated once, here.
_BAND_NAME = {
    SEARCH_PRIMARY: "blue",
    SEARCH_SECONDARY: "green",
    SEARCH_NULL: "purple",
}
_CONTROL_NAME = "rose"

#: The one arm that is not a search at all keeps the control family.
STYLE = {c: (_BAND.get(c, colour_of(c)), *MARKER[c]) for c in LABEL}

#: Rows read in band order, so colour and vertical position agree.
READING_ORDER = BRAKES_ON + BRAKE_RELEASED + SEARCH_ALTERED + ["fixed_mixed_matched"]

#: Geometry as a series: three distinct hues from inside one role, used only
#: where every arm in the figure is a search and the series is the task. Three
#: hues rather than three tints of one, for the reason the band comment gives.
TASK_COLOUR = within("search_primary", ALL_TASKS)


def fig_complexification(runs: list[dict], out: Path) -> Path:
    """The signature plot: does the topology actually augment, and how fast?

    Mean represented nodes across the population, per generation, averaged over
    replicates. The dashed line is the published Figure 10.3 champion, shown as
    a scale reference and never pooled with anything measured here.
    """
    left = 0.135
    # Three geometries, three rows. Side by side at a readable width each panel
    # was 2.2 inches across and the whole figure 2055 px, which GitHub shows at
    # 870 px — the tick labels arrived at about 2pt. Stacked, every panel gets
    # the full 6.6 inches and the shared x axis makes the three comparable.
    fig, axes = _fig(len(ALL_TASKS), 1, figsize=(6.6, 7.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, task in zip(axes, ALL_TASKS):
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
            # drawn heavier as well as first in the key.
            ax.plot(gens, vals, color=colour, linestyle=dash,
                    linewidth=2.2 if cond == REFERENCE else 1.3,
                    marker=marker, markevery=max(n // 6, 1), markersize=4,
                    label=LABEL[cond])
        if task == REFERENCE_CHAMPION["task"]:
            # The house reference line: dashed, grey, behind the data, with its
            # label in the same grey so the two read as one piece of furniture
            # rather than as another series.
            parity(ax, REFERENCE_CHAMPION["nodes"],
                   f"Ha (2016) champion: {REFERENCE_CHAMPION['nodes']} nodes")
        title(ax, TASK_LABEL[task])
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["start", "half", "end"])
        ax.set_ylabel("mean nodes in\nthe population")
    axes[-1].set_xlabel("progress through the run")
    # A per-axes legend here lands on the reference line and its label, so the
    # key moves below the panels. Three columns rather than seven: at this width
    # seven put "fixed net (matched budget)" half outside the figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center")
    suptitle(fig,
        "Do the topologies augment? Mean population size over the run,\n"
        "at one shared evaluation budget.\n"
        f"{_BAND_NAME[SEARCH_PRIMARY].capitalize()}: both brakes on, the "
        f"reference. {_BAND_NAME[SEARCH_SECONDARY].capitalize()}: a brake "
        "released on\ntopology size. "
        f"{_BAND_NAME[SEARCH_NULL].capitalize()}: the search changed without "
        "touching size.",
        color=INK, fontsize=TITLE_SIZE, x=left, ha="left",
    )
    return save(fig, out, top=0.865, bottom=0.115, left=left, right=0.985,
                hspace=0.42)


def fig_size_vs_accuracy(comp: list[dict], out: Path) -> Path:
    """Does a bigger topology buy accuracy? One point per (task, condition)."""
    left = 0.115
    fig, axes = _fig(len(ALL_TASKS), 1, figsize=(6.6, 7.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, task in zip(axes, ALL_TASKS):
        rows = [r for r in comp if r["task"] == task and r["test_accuracy_mean"] != ""]
        for r in rows:
            colour, marker, _ = STYLE[r["condition"]]
            # Every mark carries the white keyline, so two arms landing on the
            # same size stay countable. A star's ink sits inside its bounding
            # box, so at one shared size it renders half the weight of a filled
            # circle and needs roughly twice the area.
            dot(ax, r["causal_hidden_nodes_mean"], float(r["test_accuracy_mean"]),
                colour, marker=marker, size=150 if marker == "*" else 70,
                label=LABEL[r["condition"]])
        # Direct-label only the two that make the argument.
        for key, dy, ha, dx in ((REFERENCE, 10, "center", 0),
                                ("fixed_mixed_matched", -17, "right", -8)):
            r = next((x for x in rows if x["condition"] == key), None)
            if r:
                ax.annotate(LABEL[key],
                            (r["causal_hidden_nodes_mean"], float(r["test_accuracy_mean"])),
                            textcoords="offset points", xytext=(dx, dy), ha=ha,
                            fontsize=ANNOT_SIZE, color=INK2)
        # symlog, because one arm sits at 65 units and the rest below 11. Ticks
        # are set explicitly: with linthresh=10 matplotlib drew a single "10^1"
        # and a reader could not tell two causal units from nine.
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 20, 65])
        ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "20", "65"])
        ax.set_xlim(-0.4, 115)
        # A little air top and bottom: at the stacked height the extreme marks
        # otherwise sit on the spine and lose half their ink to it.
        ax.margins(y=0.14)
        title(ax, TASK_LABEL[task])
        ax.set_ylabel("sealed-test accuracy")
    axes[-1].set_xlabel("causally active hidden nodes")
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            h2.append(h)
            l2.append(lab)
    fig.legend(h2, l2, ncol=3, loc="lower center")
    suptitle(fig, "Bigger is not better: champion size against what it scored",
                 color=INK, fontsize=TITLE_SIZE, x=left, ha="left")
    return save(fig, out, top=0.912, bottom=0.135, left=left, right=0.985,
                hspace=0.46)


def fig_mechanisms(effects: list[dict], out: Path) -> Path:
    """Each mechanism's effect on accuracy, paired and Holm-corrected."""
    conds = [c for c in ALL_CONDITIONS if c != REFERENCE]
    # Horizontal, because the categorical axis carries seven names as long as
    # "fixed net (matched budget)". Upright and 6.6 inches wide they had to be
    # raked over to fit and still collided; on the y axis each one gets its own
    # line, set level, and the effect runs along x where zero is a clean rule.
    fig, ax = _fig(figsize=(6.6, 5.0), grid_axis="x")
    width = 0.26
    for ti, task in enumerate(ALL_TASKS):
        ys, med, lo, hi, sig = [], [], [], [], []
        for ci, cond in enumerate(conds):
            e = next((x for x in effects
                      if x["task"] == task and x["condition"] == cond
                      and x["metric"] == "test_accuracy"), None)
            if e is None:
                continue
            # The axis is inverted, so a smaller y sits higher: this puts the
            # three geometries down each cluster in the order the key lists them.
            ys.append(ci + (ti - 1) * width)
            # Plot condition minus reference, so "right" means the change helped.
            med.append(-e["median_difference"])
            lo.append(e["ci95_high"] - e["median_difference"])
            hi.append(e["median_difference"] - e["ci95_low"])
            sig.append(e.get("holm_p", e["wilcoxon_p"]) < 0.05)
        # The series here is the geometry, and every arm is a search, so the
        # three tasks are three distinct hues from inside the search family.
        # Significance is a second variable and gets a second channel: an
        # unfilled bar did not reach significance but still says which geometry
        # it belongs to by its outline, where greying it out erased that on two
        # thirds of the panel.
        ax.barh(ys, med, height=width * 0.9,
                color=[TASK_COLOUR[task] if s else SURFACE for s in sig],
                edgecolor=TASK_COLOUR[task], linewidth=1.1,
                xerr=[lo, hi], ecolor=INK2, capsize=2,
                error_kw={"linewidth": 0.9})
    # Zero is the line every effect is read against, so it is the house dashed
    # grey reference rather than a second solid rule competing with the bars.
    vparity(ax)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([LABEL[c] for c in conds])
    # Headroom above the first row for the key: left of zero and above the
    # smallest effects is the one region of this panel with no ink in it.
    ax.set_ylim(len(conds) - 0.5, -1.6)
    ax.set_xlabel("median change in sealed-test accuracy against the reference")
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
        loc="upper left", ncol=2,
    )
    title(ax, "What each NEAT mechanism is worth.\nUnfilled bars are not "
              "significant after Holm correction.")
    return save(fig, out)


def fig_hypotheses(hyp: list[dict], out: Path) -> Path:
    # Three text rows per hypothesis rather than two: at 6.6 inches the rule and
    # the observation no longer fit on one line beside a right-hand verdict, and
    # the verdict is the thing a reader scans for, so it keeps the right edge and
    # the rule drops to a line of its own.
    fig, ax = _fig(figsize=(6.6, 0.74 * len(hyp) + 0.9), grid_axis=None)
    for i, h in enumerate(reversed(hyp)):
        good = h["verdict"] == "holds"
        colour = GOOD if good else (BAD if h["verdict"] == "fails" else NEUTRAL)
        ax.add_patch(plt.Rectangle((0, i - 0.34), 0.016, 0.68,
                                   facecolor=colour, edgecolor=SURFACE))
        # The undecided swatch is a pale grey: it reads as a filled block
        # against white, but not as a word at this size. So the verdict that did
        # not resolve is set in ink and lets the swatch alone carry the grey.
        word = colour if h["verdict"] in ("holds", "fails") else INK2
        ax.text(0.038, i + 0.22, f"{h['hypothesis']}  {h['statement']}",
                color=INK, va="center")
        ax.text(0.038, i - 0.06, f"rule: {h['decision_rule']}",
                fontsize=ANNOT_SIZE, color=INK2, va="center")
        ax.text(0.038, i - 0.29, f"observed: {h['observed']}",
                fontsize=ANNOT_SIZE, color=INK2, va="center")
        ax.text(0.995, i + 0.22, h["verdict"], color=word, ha="right",
                va="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(hyp) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    title(ax, "Preregistered hypotheses, scored by their own declared rules")
    return save(fig, out)


def fig_accuracy(summary: list[dict], out: Path) -> Path:
    """Every arm on every geometry, read as dots on a stem from chance.

    Bars on a 0.45 baseline made the fixed net look twice the reference when the
    ratio was 1.22, and clipped a 0.964 bar flat at the panel edge with its value
    label drawn outside the axes. A dot puts the value in its position, so the
    axis may start where the data live without the length lying about a ratio.
    """
    by = {(r["task"], r["condition"]): r for r in summary}
    chance = 0.5
    left = 0.265
    # The values run along x, so the grid that helps read them is the x grid.
    # Stacked, so the eight condition names are set once per panel at full size
    # rather than once for three panels crushed to 2.2 inches each.
    fig, axes = _fig(len(ALL_TASKS), 1, figsize=(6.6, 8.4), grid_axis="x")
    axes = np.atleast_1d(axes).ravel()
    for ax, task in zip(axes, ALL_TASKS):
        vals, colours = [], []
        for c in READING_ORDER:
            row = by.get((task, c))
            v = row.get("test_accuracy_mean") if row else None
            vals.append(float(v) if v not in (None, "") else np.nan)
            colours.append(STYLE[c][0])
        y = np.arange(len(READING_ORDER))
        vparity(ax, chance, "chance" if task == ALL_TASKS[0] else None)
        for yi, v, colour in zip(y, vals, colours):
            if not np.isfinite(v):
                continue
            ax.plot([chance, v], [yi, yi], color=colour, linewidth=2.0, alpha=0.5,
                    solid_capstyle="round", zorder=2)
            dot(ax, v, yi, colour, size=45)
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=ANNOT_SIZE,
                    color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in READING_ORDER])
        # Hairlines where the band changes, so the three hues read as three
        # groups and not as three arbitrary choices.
        for edge in (len(BRAKES_ON) - 0.5,
                     len(BRAKES_ON) + len(BRAKE_RELEASED) - 0.5,
                     len(READING_ORDER) - 1.5):
            ax.axhline(edge, color=RULE, linewidth=0.7, alpha=0.8)
        ax.invert_yaxis()
        ax.set_ylim(len(READING_ORDER) - 0.4, -1.05)
        ax.set_xlim(chance - 0.02, 1.05)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        title(ax, TASK_LABEL[task])
    suptitle(fig,
        "Sealed-test accuracy by condition.\n"
        f"{_BAND_NAME[SEARCH_PRIMARY].capitalize()} is the reference, with both "
        f"brakes on; {_BAND_NAME[SEARCH_SECONDARY]} released a brake\n"
        f"on topology size; {_BAND_NAME[SEARCH_NULL]} changed the search "
        f"without touching size;\n{_CONTROL_NAME} is the one arm that is not a "
        "search.",
        color=INK, fontsize=TITLE_SIZE, x=0.0, ha="left",
    )
    return save(fig, out, top=0.872, bottom=0.045, left=left, right=0.985,
                hspace=0.30)


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
