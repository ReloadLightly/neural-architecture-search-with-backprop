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
    CLASS_COLOURS,
    DPI,
    GRID,
    INK,
    INK2,
    OP_COLOUR,
    colour_of,
    save,
    SURFACE,
    boundary_cmap,
    panel,
    style_axes,
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

    fig, ax = plt.subplots(figsize=(11, 5.0), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(rows))[::-1]

    # A geometry where every condition clears this is one on which nothing can
    # be learned: the marks pile up and a per-mark label would be unreadable, so
    # the row is annotated with the fact instead of four colliding numbers.
    saturated = 0.97

    for yi, r in zip(y, rows):
        values = (r["search"], r["null"], r["bpneat"], r["fixed"])
        lo, hi = sorted((r["search"], r["null"]))
        ax.plot([lo, hi], [yi, yi], color=GRID, linewidth=6, solid_capstyle="round",
                zorder=1)
        ax.scatter(r["null"], yi, s=150, color=colour_of("cgp_random_matched"),
                   zorder=3, marker="o", edgecolors=SURFACE, linewidths=2)
        ax.scatter(r["search"], yi, s=150, color=colour_of("cgp"),
                   zorder=4, marker="s", edgecolors=SURFACE, linewidths=2)
        ax.scatter(r["bpneat"], yi, s=150, color=colour_of("bpneat"),
                   zorder=4, marker="^", edgecolors=SURFACE, linewidths=2)
        ax.scatter(r["fixed"], yi, s=210, color=colour_of("fixed_tanh_matched_bpneat"),
                   zorder=5, marker="D", edgecolors=SURFACE, linewidths=2)

        if min(values) >= saturated:
            ax.annotate("every condition ≥ 0.97 — nothing to separate",
                        (min(values), yi), xytext=(-14, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8.5, color=INK2, style="italic")
            continue
        # Otherwise every mark is directly labelled: three palette slots sit
        # below 3:1 against the surface, so identity may not rest on colour.
        ax.annotate(f"{r['fixed']:.3f}", (r["fixed"], yi), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        left = min(r["search"], r["null"], r["bpneat"])
        ax.annotate(f"{left:.3f}", (left, yi), xytext=(-13, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK2)
        right = max(r["search"], r["null"], r["bpneat"])
        if right - left > 0.01:
            ax.annotate(f"{right:.3f}", (right, yi), xytext=(0, -17),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=8.5, color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABEL[r["task"]] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlabel("sealed-test accuracy, mean of 30 paired replicates", fontsize=9,
                  color=INK2)
    ax.set_xlim(0.50, 1.04)
    # Room under the last row for its below-mark label, which would
    # otherwise land on the x axis.
    ax.set_ylim(-0.75, len(rows) - 0.35)

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
    ax.set_title(
        "Does the architecture search pay?\n"
        "Each bar joins a search to the same space sampled at random. "
        "Short bar = the search bought nothing.",
        color=INK, fontsize=11.5, loc="left", pad=14,
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
    ax.barh(y, [b[1] for b in bars], height=0.55, color=[b[2] for b in bars],
            zorder=3)
    for yi, (label, v, _) in zip(y, bars):
        ax.text(v + 0.9, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=9.5, color=INK)

    # The two reference lines are labelled along themselves: horizontal labels
    # at the top collided with the title and ran off the right edge.
    for x, label in (
        (HA_CHAMPION_NODES, f"Ha (2016) champion: {HA_CHAMPION_NODES}"),
        (FIXED_UNITS, f"the fixed network it is compared with: {FIXED_UNITS}"),
    ):
        ax.axvline(x, color=INK2, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
        ax.text(x - 1.1, (len(bars) - 1) / 2.0, label, fontsize=8.5, color=INK2,
                rotation=90, va="center", ha="right")

    ax.set_xlim(0, FIXED_UNITS * 1.1)
    ax.set_xlabel("causally active hidden units in the champion (spirals)",
                  fontsize=9, color=INK2)
    ax.set_title(
        "The topologies these searches actually find\n"
        "Augmenting topologies, augmenting by about four units.",
        color=INK, fontsize=11.5, loc="left", pad=20,
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
    for gi, (label, cond, colour) in enumerate(groups):
        xs = [ti + (gi - 1) * width for ti in range(len(tasks))]
        vals = [float(s4[(t, cond)]["test_accuracy_mean"]) for t in tasks]
        ax.bar(xs, vals, width=width * 0.88, color=colour, label=label, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=9.5, color=INK)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("sealed-test accuracy", fontsize=9, color=INK2)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    ax.set_title(
        "The first two bars are the same 65-unit network.\n"
        "Only the training budget differs — and it decides the comparison.",
        color=INK, fontsize=11.5, loc="left", pad=14,
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
