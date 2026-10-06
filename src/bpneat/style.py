"""The one place this repository's figures get their look.

**This is the style used in `competitive-coevolution-of-slimes`,** ported so
that the two repositories read as one body of work. The rcParams below are that
project's `make_figures.style()`; the palette is its `COLORS`; the idioms —
median line over an inter-quartile band, strip plots with a median bar, a dashed
grey parity line, a left-aligned sentence for a title — are its idioms.

The look, in one paragraph: white ground, no frame, no border; top and right
spines off; a very light grid behind the data; stock sans at 8.5pt with 7.5pt
ticks; legends without a box; and a small fixed set of muted, mid-saturation
hues assigned to conditions by name, never by position. Figures are saved at
200 dpi with a tight bounding box, so there is no wasted margin and nothing is
drawn that is not either data or the furniture needed to read it.

Every plotting module in the project — `figures.py`, the per-protocol figure
modules, `champions.py`, and the scripts under `bench/` — imports its surface,
ink, grid, palette, colormap, type and save settings from here. Nothing else
defines them. That is enforced rather than intended: `tests/test_style.py`
parses every module that imports matplotlib and fails if it assigns its own.

**The palette is the slimes palette, assigned to be separable.** Those twelve
hues are fixed; which one a given series gets is chosen so that series sharing
a panel stay apart. Searching the twelve for the six roles below gives a worst
pair of ΔE 11.4 across normal, protanopic and deuteranopic vision, and for the
nine operators ΔE 7.7. Colour still never carries identity alone: every figure
either directly labels its marks or ships a CSV table view beside it in the
release.

**What is deliberately not restyled.** `results/backprop-neat-v1/figures/`
belongs to an invalidated release and `results/backprop-neat-v2/` carries
errata; both are frozen evidence, and a frozen release should look like what
was released. Nothing in the README or the paper shows them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

#: White, like the page the figure will sit on.
SURFACE = "#ffffff"
#: Primary ink for titles and values, secondary for labels and ticks. Near
#: black rather than black: pure black on white is harsher than anything in
#: the data deserves.
INK = "#222222"
INK2 = "#555555"
#: The grid is a ruler behind the data, not a feature of it.
GRID = "#E6E6E6"
#: Reference lines, parity lines, annotation leaders.
RULE = "#999999"

#: The twelve hues, exactly as `competitive-coevolution-of-slimes` defines
#: them. Nothing in this repository invents a hue; it only chooses which of
#: these a series gets.
PALETTE = {
    "ink": "#222222",
    "slate": "#3B6EA8",
    "pine": "#1F7A5A",
    "amber": "#E08B3C",
    "brick": "#B0413E",
    "umber": "#8C5A2B",
    "violet": "#7A5EA6",
    "rose": "#C1445E",
    "sage": "#4C956C",
    "mauve": "#946A9E",
    "taupe": "#8A6A55",
    "teal": "#4FA3B8",
}

#: Kept only so `tests/test_style.py` can prove nothing reads it. Colour comes
#: from a condition's role, never from its position in a list.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

#: Status colours, reserved. Never reused as a series.
GOOD = PALETTE["sage"]
BAD = PALETTE["brick"]
NEUTRAL = "#B4B4B0"

#: Class 0 → cool, class 1 → warm, through a near-white middle. The midpoint
#: *is* the decision boundary, so it has to read as undecided rather than as a
#: third colour — which is why this is a diverging ramp and not a rainbow.
BOUNDARY_COLOURS = [PALETTE["slate"], "#f4f3f1", PALETTE["brick"]]
#: The two classes as drawn points: darker than the field, so they sit on it.
CLASS_COLOURS = ("#2b527f", "#7d2e2c")

#: A signed quantity — a connection weight, a difference either side of zero —
#: takes the same two diverging anchors. Named so that a module does not reach
#: into a categorical list for them, which is how blue came to mean "tanh" on a
#: node and "positive" on the edge leaving it.
WEIGHT_POS = BOUNDARY_COLOURS[0]
WEIGHT_NEG = BOUNDARY_COLOURS[-1]

#: Operators are the one genuinely nominal set here: nine names with no order
#: between them, which is the hardest case for colour. These nine are the
#: most separable nine of the twelve — worst pair ΔE 7.7 across normal,
#: protanopic and deuteranopic vision. An operator is never identified by
#: colour alone; every figure that uses these also labels them.
OP_COLOUR = {
    "sin": PALETTE["amber"],
    "gaussian": PALETTE["brick"],
    "mult": PALETTE["rose"],
    "square": PALETTE["mauve"],
    "abs": PALETTE["pine"],
    "sigmoid": PALETTE["sage"],
    "tanh": PALETTE["teal"],
    "relu": PALETTE["slate"],
    "add": PALETTE["ink"],
    "null": "#B4B4B0",
}

#: Every geometry this project uses, under one spelling.
TASK_LABEL = {
    "xor": "XOR", "circle": "Circles", "spiral": "Spirals",
    "checkerboard": "Checkerboard", "spiral3": "3-arm spiral",
}

#: Type scale, from the slimes rcParams. One ladder, so a panel title in v3 is
#: the size of one in v5.
TITLE_SIZE = 9.5
PANEL_TITLE_SIZE = 9.5
LABEL_SIZE = 8.5
TICK_SIZE = 7.5
ANNOT_SIZE = 7.0
LEGEND_SIZE = 7.5

#: Raster density for every committed figure.
DPI = 200


def use_project_typography() -> None:
    """Install the slimes rcParams globally.

    Called at import, so a module that does nothing but ``import bpneat.style``
    already draws in the right register. This is that project's ``style()``
    verbatim, plus the colours it sets per call site.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "axes.facecolor": SURFACE,
        "axes.titlelocation": "left",
        "axes.edgecolor": "#BFBFBF",
        "axes.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
    })


use_project_typography()


def boundary_cmap():
    """The diverging ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bpneat_boundary", BOUNDARY_COLOURS)


# --------------------------------------------------------------------------
# Helpers — the slimes idioms, in one place
# --------------------------------------------------------------------------


def style_axes(ax, grid_axis: str | None = "y") -> None:
    """Recessive frame: no top or right spine, a light grid behind the data."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFBFBF")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=TICK_SIZE, length=0)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def panel(ax) -> None:
    """A bare image panel: framed, no ticks. For boundaries and matrices."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CFCFCF")
        spine.set_linewidth(0.7)
    ax.set_facecolor(SURFACE)


def figure(nrows: int = 1, ncols: int = 1, figsize=(6.6, 3.4), grid_axis="y") -> Any:
    """A styled figure and its axes."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax, grid_axis)
    return fig, axes


def title(ax, text: str) -> None:
    """A title is a left-aligned sentence that says what the panel shows."""
    ax.set_title(text, loc="left", color=INK, fontsize=TITLE_SIZE)


def parity(ax, y: float = 0.0, label: str | None = None) -> None:
    """The dashed grey line a reader measures everything else against."""
    ax.axhline(y, color=RULE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate(label, xy=(0.995, y), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=ANNOT_SIZE, color=RULE)


def vparity(ax, x: float = 0.0, label: str | None = None) -> None:
    """The same line, vertical, for horizontal effect plots."""
    ax.axvline(x, color=RULE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=ANNOT_SIZE, color=RULE)


def strip(ax, groups, colours, labels, ylabel: str = "", heading: str = "") -> None:
    """One dot per replicate, jittered, with the group median as a short bar.

    The slimes strip panel. Showing every replicate and the median is honest
    about spread in a way a bar with an error cap is not.
    """
    import numpy as np

    for i, (vals, colour) in enumerate(zip(groups, colours)):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        x = np.full(len(vals), float(i))
        if len(vals) > 1:
            x = x + np.linspace(-0.13, 0.13, len(vals))
        ax.scatter(x, vals, s=16, color=colour, alpha=0.85, zorder=3,
                   edgecolor=SURFACE, linewidth=0.5)
        ax.plot([i - 0.26, i + 0.26], [np.median(vals)] * 2, color=colour,
                lw=2.0, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel)
    if heading:
        title(ax, heading)


def dot(ax, x, y, colour, marker="o", size=28, **kw):
    """A single mark, with the white keyline that keeps overlaps readable."""
    return ax.scatter(x, y, s=size, color=colour, marker=marker, zorder=3,
                      edgecolor=SURFACE, linewidth=0.6, **kw)


def save(fig, path, **adjust) -> Any:
    """Write a figure at the project's density, onto the project's surface.

    `savefig.bbox="tight"` crops to the ink, so a caller passing explicit
    margins is asking for proportions rather than for padding.
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if adjust:
        fig.subplots_adjust(**adjust)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def pale(colour: str, amount: float = 0.74):
    """Toward the surface, so a filled field never competes with the points
    drawn on it."""
    import matplotlib.colors as mcolors
    import numpy as np

    rgb = np.array(mcolors.to_rgb(colour))
    ground = np.array(mcolors.to_rgb(SURFACE))
    return tuple(rgb + (ground - rgb) * amount)


# --------------------------------------------------------------------------
# Semantic roles — colour encodes the argument, not the row index
# --------------------------------------------------------------------------
#
# The defect this replaces: colour was assigned by position in a six-slot list,
# so with eight conditions it cycled — Backprop-NEAT and "fixed mixed @ CGP
# budget" came out the same blue — while every row was already labelled, so the
# hue carried nothing and actively misled. Worse, the same concept changed
# colour between figures.
#
# Every comparison in this project is between two kinds of thing: something we
# searched for, and something we fixed in advance. So there are two families,
# and the ordered distinctions inside each are separate hues drawn from the
# shared palette rather than tints, because tints of one hue are what stopped
# being legible.
#
# Validated: searching the twelve palette hues for these six gives a worst pair
# of ΔE 11.4 across normal, protanopic and deuteranopic vision, and every one
# of them clears 2.6:1 against white.

#: We searched for it.
SEARCH_PRIMARY = PALETTE["slate"]
SEARCH_SECONDARY = PALETTE["sage"]
SEARCH_NULL = PALETTE["mauve"]
SEARCH_RAMP = [SEARCH_PRIMARY, SEARCH_SECONDARY, SEARCH_NULL]

#: We fixed it in advance.
CONTROL_STARVED = PALETTE["amber"]
CONTROL_MATCHED = PALETTE["brick"]
CONTROL_BEST = PALETTE["rose"]
CONTROL_RAMP = [CONTROL_STARVED, CONTROL_MATCHED, CONTROL_BEST]

#: Neither: a published target, or a verdict that did not reach significance.
REFERENCE = PALETTE["ink"]

ROLE_COLOUR = {
    "search_primary": SEARCH_PRIMARY,
    "search_secondary": SEARCH_SECONDARY,
    "search_null": SEARCH_NULL,
    "control_starved": CONTROL_STARVED,
    "control_matched": CONTROL_MATCHED,
    "control_best": CONTROL_BEST,
    "reference": REFERENCE,
}

#: Condition name -> role, across every protocol. The point of this table is
#: that a condition cannot be coloured differently in two figures.
_ROLE_OF = {
    # the algorithm under study, in each protocol's own naming
    "backprop_neat": "search_primary",
    "bpneat": "search_primary",
    "neat_reference": "search_primary",
    # a second search, or a variant of the first
    "cgp": "search_secondary",
    "homogeneous_tanh": "search_secondary",
    "evolution_only": "search_secondary",
    "baldwinian": "search_secondary",
    "neat_complexify": "search_secondary",
    "neat_no_penalty": "search_secondary",
    "neat_complexify_no_penalty": "search_secondary",
    "neat_no_speciation": "search_secondary",
    "neat_no_crossover": "search_secondary",
    "neat_deep_narrow": "search_secondary",
    # the same search space, sampled instead of searched
    "random_search": "search_null",
    "random_search_matched": "search_null",
    "cgp_random_matched": "search_null",
    "logistic": "search_null",
    # a fixed architecture denied the budget the search spent
    "fixed_mlp": "control_starved",
    "fixed_mlp_tanh_ha": "control_starved",
    "fixed_tanh_ha": "control_starved",
    # a fixed architecture given that budget
    "fixed_mlp_tanh_matched": "control_matched",
    "fixed_tanh_matched_bpneat": "control_matched",
    "fixed_tanh_matched_cgp": "control_matched",
    # the strongest fixed control: a better operator set, matched budget
    "fixed_mlp_sin_matched": "control_best",
    "fixed_mlp_mixed_matched": "control_best",
    "fixed_mixed_matched_bpneat": "control_best",
    "fixed_mixed_matched_cgp": "control_best",
    "fixed_mixed_matched": "control_best",
}

#: v6 names its conditions ``<arm>_b<candidates>``, so the grid is fifteen names
#: for three roles. The budget is an ordered quantity and goes on an axis, never
#: into a colour: a rung never changes a hue, and the three arms keep the hues
#: they already have in v3, v4 and v5 — the search is `slate`, the
#: candidate-matched null is `mauve`, the budget-matched fixed network is `rose`.
V6_BUDGET_CANDIDATES = (500, 1000, 2100, 6300, 16800)
_V6_ARM_ROLE = {
    "search": "search_primary",
    "null": "search_null",
    "fixed": "control_best",
}
for _candidates in V6_BUDGET_CANDIDATES:
    for _arm, _role in _V6_ARM_ROLE.items():
        _ROLE_OF[f"{_arm}_b{_candidates}"] = _role


def role_of(condition: str) -> str:
    """The semantic role of a condition, in any protocol.

    Unknown names fall back to a search variant rather than raising: the v3
    propagation and selection arms are named by factor level
    (``prop_ha_fit_val``, ``sel_tournament_k2``), and they are all searches.
    """
    if condition in _ROLE_OF:
        return _ROLE_OF[condition]
    if condition.startswith(("prop_", "sel_", "neat_")):
        return "search_secondary"
    if condition.startswith("fixed"):
        return "control_matched"
    return "search_secondary"


def arm_role(arm: str) -> str:
    """The role of a v6 arm, by arm name rather than condition name.

    v6's figures put the budget on an axis and colour by arm, so they ask for a
    hue once per arm instead of once per condition.
    """
    return _V6_ARM_ROLE[arm]


def arm_colour(arm: str) -> str:
    return ROLE_COLOUR[arm_role(arm)]


def colour_of(condition: str) -> str:
    """The one colour this condition gets, in every figure in the project."""
    return ROLE_COLOUR[role_of(condition)]


#: Where a figure needs several members of one family separated, it takes the
#: next hues from the palette rather than tinting one. Tints of a single hue at
#: small sizes were the thing that stopped being readable.
_WITHIN = {
    "search_primary": [PALETTE["slate"], PALETTE["teal"], PALETTE["violet"]],
    "search_secondary": [PALETTE["sage"], PALETTE["pine"], PALETTE["teal"],
                         PALETTE["violet"], PALETTE["mauve"], PALETTE["taupe"]],
    "search_null": [PALETTE["mauve"], PALETTE["violet"], PALETTE["taupe"]],
    "control_starved": [PALETTE["amber"], PALETTE["umber"], PALETTE["taupe"]],
    "control_matched": [PALETTE["brick"], PALETTE["rose"], PALETTE["umber"]],
    "control_best": [PALETTE["rose"], PALETTE["brick"], PALETTE["umber"]],
    "reference": [PALETTE["ink"], PALETTE["taupe"], PALETTE["umber"]],
}


def ramp(role: str, n: int) -> list[str]:
    """``n`` distinct hues for members of one role, in a fixed order."""
    base = _WITHIN[role]
    if n <= len(base):
        return base[:n]
    extra = [c for c in PALETTE.values() if c not in base]
    return (base + extra)[:n]


def within(role: str, members) -> dict[str, str]:
    """Distinct hues inside one role, keyed by member name."""
    members = list(members)
    return dict(zip(members, ramp(role, len(members))))
