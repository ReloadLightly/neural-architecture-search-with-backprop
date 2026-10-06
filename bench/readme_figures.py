"""The figures the top README leads with, rendered from the committed releases.

These are cross-release views — they put v3's and v4's numbers side by side —
so they belong to neither release directory and are regenerated here, from the
released CSVs only. No run is re-executed and no raw record is re-derived: every
value is read from a `summary.csv` that `make verify` has already proven
regenerates byte-identically from its own raw records.

`tests/test_readme_figures.py` asserts that every number these figures draw is
the one the release publishes, so a figure cannot drift from the evidence.

Palette: the project's validated categorical order (adjacent-pair CVD ΔE 9.1,
normal-vision ΔE 19.6, all six inside the lightness band). Three of the six sit
below 3:1 against the surface, which obliges visible labels rather than
colour-only identity — so every mark here is directly labelled.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V3 = ROOT / "results" / "backprop-neat-v3"
V4 = ROOT / "results" / "backprop-neat-v4"
OUT = ROOT / "docs" / "figures"

from bpneat.style import (  # noqa: E402
    DISPLAY_FAMILY,
    GRID,
    INK,
    INK2,
    NEUTRAL,
    SURFACE,
    colour_of,
    save,
)

TASKS = ("xor", "circle", "spiral", "checkerboard", "spiral3")
from bpneat.style import TASK_LABEL  # noqa: E402

#: Ha (2016), Figure 10.3, spirals champion. A published target, never pooled
#: with anything measured here.
HA_CHAMPION_NODES = 34
#: The fixed control's architecture: 32 + 32 hidden units plus the bias carrier.
FIXED_UNITS = 65


def summary(release: Path) -> dict[tuple[str, str], dict]:
    with open(release / "summary.csv") as fh:
        return {(r["task"], r["condition"]): r for r in csv.DictReader(fh)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.set_axisbelow(True)


def _headline(fig, head: str, sub: str, x: float = 0.055) -> None:
    """A headline and a sub-line, at two sizes and two weights.

    A two-line `set_title` set both at 11.5pt, so the claim and its gloss had
    the same weight and the figure had no entry point. These are the three
    figures the README opens with; they get a real hierarchy.
    """
    fig.text(x, 0.955, head, ha="left", va="top", color=INK,
             fontsize=15.5, family=DISPLAY_FAMILY)
    fig.text(x, 0.885, sub, ha="left", va="top", color=INK2, fontsize=9.5)


def _save(fig, name: str, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    path = save(fig, OUT / name, **kw)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def fig_does_search_pay() -> Path:
    """The NAS question, in one picture: search against not searching.

    For each geometry, a dumbbell joins a search algorithm to its own
    candidate-matched null — the same genotype space, the same inner learner,
    the same number of candidates, selection removed. A short dumbbell means the
    search bought nothing. The diamond is a fixed architecture given the same
    gradient budget, which is what both are really competing against.
    """
    s4 = summary(V4)

    def acc(task, cond):
        return float(s4[(task, cond)]["test_accuracy_mean"])

    rows = []
    for t in TASKS:
        rows.append(
            {
                "task": t,
                "search": acc(t, "cgp"),
                "null": acc(t, "cgp_random_matched"),
                "fixed": acc(t, "fixed_mixed_matched_cgp"),
                "bpneat": acc(t, "bpneat"),
            }
        )

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(rows))[::-1]

    # A geometry where every condition clears this is one on which nothing can
    # be learned: the marks pile up and a per-mark label would be unreadable, so
    # the row is annotated with the fact instead of four colliding numbers.
    saturated = 0.97

    # Two lanes per geometry: the searched conditions on a connecting bar, and
    # the fixed control on its own line below. Drawn on one line, the fixed
    # network's diamond was the last and largest mark and simply covered the
    # other three — on Circles the row showed one mark where there were four.
    search_dy, fixed_dy = 0.15, -0.23
    for yi, r in zip(y, rows):
        searched = (
            ("bpneat", r["bpneat"], "^"),
            ("cgp", r["search"], "s"),
            ("cgp_random_matched", r["null"], "o"),
        )
        lo, hi = min(v for _, v, _ in searched), max(v for _, v, _ in searched)
        # On a geometry where every searched condition clears 0.97 the three
        # marks occupy 0.4% of the axis: drawing them separately would be three
        # overlapping marks claiming a separation that is not there. One mark
        # is what this scale can honestly show, and the row says the range.
        crowded = lo >= saturated
        ax.plot([lo, hi], [yi + search_dy] * 2, color=GRID, linewidth=6,
                solid_capstyle="round", zorder=1)
        ax.plot([r["fixed"], r["fixed"]], [yi + fixed_dy, yi + search_dy],
                color=GRID, linewidth=1.0, zorder=1)
        if crowded:
            ax.scatter((lo + hi) / 2, yi + search_dy, s=150, color=NEUTRAL,
                       zorder=4, marker="o", edgecolors=SURFACE, linewidths=2)
        else:
            for cond, v, marker in searched:
                ax.scatter(v, yi + search_dy, s=140, color=colour_of(cond),
                           zorder=4, marker=marker, edgecolors=SURFACE,
                           linewidths=2)
        ax.scatter(r["fixed"], yi + fixed_dy, s=150,
                   color=colour_of("fixed_tanh_matched_bpneat"), zorder=4,
                   marker="D", edgecolors=SURFACE, linewidths=2)
        ax.annotate(f"{r['fixed']:.3f}", (r["fixed"], yi + fixed_dy), xytext=(12, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=INK2)

        if crowded:
            # Nothing to separate: say so once, at the left of the row, rather
            # than print three colliding numbers on top of three stacked marks.
            # The span is measured, not asserted.
            ax.annotate(
                f"all three searched conditions fall in {lo:.3f}–{hi:.3f} "
                "— drawn as one mark",
                (0.515, yi + search_dy), ha="left", va="center",
                fontsize=8.5, color=INK2, style="italic")
            continue
        # Otherwise every searched mark is directly labelled. Colour may not
        # carry identity alone, and on a saturated row it cannot carry it at all.
        placed: list[float] = []
        for i, (_cond, v, _m) in enumerate(sorted(searched, key=lambda s: s[1])):
            # Alternate above and below when two marks are closer than a label.
            above = not any(abs(v - q) < 0.035 for q in placed)
            placed.append(v)
            ax.annotate(f"{v:.3f}", (v, yi + search_dy),
                        xytext=(0, 15 if above or i % 2 == 0 else -17),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=8.5, color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABEL[r["task"]] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlabel("sealed-test accuracy, mean of 30 paired replicates", fontsize=9,
                  color=INK2)
    ax.set_xlim(0.50, 1.04)
    # Room under the last row for its below-mark label, which would
    # otherwise land on the x axis.
    ax.set_ylim(-0.85, len(rows) - 0.25)

    handles = [
        plt.Line2D([], [], marker="^", color=colour_of("bpneat"), linestyle="",
                   markersize=9, markeredgecolor=SURFACE, label="Backprop-NEAT"),
        plt.Line2D([], [], marker="s", color=colour_of("cgp"), linestyle="",
                   markersize=9, markeredgecolor=SURFACE, label="CGP"),
        plt.Line2D([], [], marker="o", color=colour_of("cgp_random_matched"),
                   linestyle="", markersize=9, markeredgecolor=SURFACE,
                   label="CGP with selection removed (matched)"),
        plt.Line2D([], [], marker="D", color=colour_of("fixed_tanh_matched_bpneat"),
                   linestyle="", markersize=9, markeredgecolor=SURFACE,
                   label="fixed network, matched budget"),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, -0.30))
    _headline(
        fig,
        "Does the architecture search pay?",
        "Each bar spans the searched conditions on one geometry; the diamond below is the fixed network at the same budget.",
        x=0.055,
    )
    return _save(fig, "does-search-pay.png", top=0.84, bottom=0.26, left=0.11,
                 right=0.98)


def fig_topologies_found() -> Path:
    """What the searches actually build, against what they are compared with.

    Causally active hidden units — the nodes that reached the output, not the
    nodes the genome represents. The two reference lines are the published
    champion and the fixed control's width.
    """
    s3, s4 = summary(V3), summary(V4)
    bars = [
        ("Backprop-NEAT", float(s4[("spiral", "bpneat")]["causal_hidden_nodes_mean"]),
         colour_of("bpneat")),
        ("CGP", float(s4[("spiral", "cgp")]["cgp_active_nodes_mean"]),
         colour_of("cgp")),
        ("CGP, selection removed",
         float(s4[("spiral", "cgp_random_matched")]["cgp_active_nodes_mean"]),
         colour_of("cgp_random_matched")),
        ("Backprop-NEAT (v3)",
         float(s3[("spiral", "backprop_neat")]["causal_hidden_nodes_mean"]),
         colour_of("backprop_neat")),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(bars))[::-1]
    # Dots, not bars: the axis is logarithmic and starts at 2.4, so a bar's
    # length would not be proportional to anything.
    for yi, (_label, v, colour) in zip(y, bars):
        ax.plot([2.4, v], [yi, yi], color=colour, linewidth=2.2, alpha=0.5,
                solid_capstyle="round", zorder=3)
        ax.plot(v, yi, "o", color=colour, markersize=11, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(v * 1.08, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=9.5, color=INK)

    # The two reference lines are labelled along themselves: horizontal labels
    # at the top collided with the title and ran off the right edge.
    for x, label in (
        (HA_CHAMPION_NODES, f"Ha (2016) champion: {HA_CHAMPION_NODES}"),
        (FIXED_UNITS, f"the fixed network it is compared with: {FIXED_UNITS}"),
    ):
        ax.axvline(x, color=INK2, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
        ax.text(x * 0.95, (len(bars) - 1) / 2.0, label, fontsize=8.5, color=INK2,
                rotation=90, va="center", ha="right")

    # Four champions between 3.9 and 8.0 units, against references at 34 and
    # 65. On a linear axis the data occupied the leftmost ninth of the panel
    # and Backprop-NEAT's 4.2 sat four pixels from CGP's 3.9. A log axis gives
    # the measured values the room, and still shows how far off the references
    # are — which is the figure's whole point.
    ax.set_xscale("log")
    ax.set_xlim(2.4, FIXED_UNITS * 1.5)
    ax.set_xticks([3, 4, 5, 6, 8, 10, 20, 34, 65])
    ax.set_xticklabels(["3", "4", "5", "6", "8", "10", "20", "34", "65"],
                       fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlabel("causally active hidden units in the champion (spirals)",
                  fontsize=9, color=INK2)
    _headline(
        fig,
        "The topologies these searches actually find",
        "Augmenting topologies — augmenting by about four causally active units.",
        x=0.055,
    )
    return _save(fig, "topologies-found.png", top=0.78, bottom=0.15, left=0.21,
                 right=0.98)


def fig_budget_decides() -> Path:
    """The evaluator result, kept because it is what makes the rest readable.

    The same fixed architecture under two training protocols, against the search
    it is the control for. Nothing about the network changes between the first
    two bars of each group — only how long it was allowed to train.
    """
    s4 = summary(V4)
    chance = 0.5
    tasks = ("spiral", "checkerboard", "spiral3")
    groups = [
        ("fixed net, unmatched", "fixed_tanh_ha", colour_of("fixed_tanh_ha")),
        ("fixed net, matched budget", "fixed_tanh_matched_bpneat",
         colour_of("fixed_tanh_matched_bpneat")),
        ("Backprop-NEAT", "bpneat", colour_of("bpneat")),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    width = 0.26
    # Dots on a stem from chance, not bars from a 0.45 floor. With the floor,
    # 0.528 and 0.590 rendered as heights 0.078 and 0.145, so the third bar
    # looked nearly twice the first where the real ratio is 1.12.
    for gi, (label, cond, colour) in enumerate(groups):
        xs = [ti + (gi - 1) * width for ti in range(len(tasks))]
        vals = [float(s4[(t, cond)]["test_accuracy_mean"]) for t in tasks]
        for x, v in zip(xs, vals):
            ax.plot([x, x], [chance, v], color=colour, linewidth=2.4, alpha=0.5,
                    solid_capstyle="round", zorder=3)
        ax.plot(xs, vals, "o", color=colour, markersize=11, linestyle="",
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=4)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.016, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(chance, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(len(tasks) - 0.62, chance + 0.008, "chance", ha="right", va="bottom",
            fontsize=8, color=INK2, style="italic")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=9.5, color=INK)
    ax.set_ylim(chance - 0.02, 1.02)
    ax.set_ylabel("sealed-test accuracy", fontsize=9, color=INK2)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    _headline(
        fig,
        "The budget decides the comparison",
        "The first two marks are the same 65-unit network. Only the training budget differs.",
        x=0.055,
    )
    return _save(fig, "budget-decides.png", top=0.78, bottom=0.11, left=0.08,
                 right=0.98)


def main() -> int:
    for release in (V3, V4):
        if not (release / "summary.csv").exists():
            print(f"missing {release}/summary.csv", file=sys.stderr)
            return 1
    fig_does_search_pay()
    fig_topologies_found()
    fig_budget_decides()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
