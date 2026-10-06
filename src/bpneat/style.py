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


def wrap_to(text: str, inches: float, size: float, fig) -> str:
    """``text`` broken to fit ``inches``, measured in this figure's renderer.

    Counting characters does not work: the same 90 characters are 6.0 inches
    of lowercase and 10.4 of capitals at this size. So the string is drawn
    once, invisibly, and its real width per character decides the wrap.
    """
    import textwrap

    flat = " ".join(text.split())
    if not flat:
        return text
    probe = fig.text(0.0, 0.0, flat, fontsize=size)
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # pragma: no cover - non-Agg backends
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    per_char = probe.get_window_extent(renderer).width / fig.dpi / len(flat)
    probe.remove()
    columns = max(12, int(max(inches, 0.5) / max(per_char, 1e-6)))
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph.strip(), columns) or [""])
    return "\n".join(lines)


def title(ax, text: str) -> None:
    """A title is a left-aligned sentence that says what the panel shows.

    Wrapped to the width of its own axes. A left-aligned axes title runs
    rightwards from the axes' left edge with nothing to stop it, so a long one
    used to push the tight-cropped figure past the page — the same defect as an
    over-long figure title, in a different place.
    """
    fig = ax.get_figure()
    width = ax.get_position().width * fig.get_size_inches()[0]
    ax.set_title(wrap_to(text, width, TITLE_SIZE, fig), loc="left", color=INK,
                 fontsize=TITLE_SIZE)


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
    needed = getattr(fig, "_bpneat_title_top", None)
    if needed is not None:
        adjust["top"] = min(adjust.get("top", 1.0), needed)
    if adjust:
        fig.subplots_adjust(**adjust)
    _fit_to_the_page(fig, path)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


#: The widest a figure may be saved. GitHub renders a README image at about
#: 870 px whatever its real width, so a figure saved at 8 inches arrives with
#: its 7.5pt tick labels at about 2pt. 6.8 inches at 200 dpi is 1360 px, which
#: is the sister project's register and the one this repository keeps.
MAX_FIGURE_WIDTH_IN = 6.8


def _tight_width(fig) -> float:
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # pragma: no cover - non-Agg backends
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    return float("nan") if bbox is None else bbox.width


def _fit_to_the_page(fig, path) -> None:
    """Pull the panels in until the whole figure fits the page, or say why not.

    ``savefig.bbox="tight"`` crops to the ink, which means it also *grows* the
    saved image when an artist sits outside the figure — a long row label, a
    key, a title line four characters too long. The figsize literal then says
    6.6 inches while the file is 7.9, and nothing notices until a reader cannot
    read it. GitHub then renders both at the same 870 px, so the wider one
    arrives with smaller type: this is one of the two reasons the figures here
    were once illegible, and the other was colour.

    So the tight box is measured before the write, and when it overflows, the
    overflow is taken out of the panel area rather than added to the page. Only
    a figure that still does not fit once its margins have been widened three
    times raises, and then the message says by how much and where from.
    """
    width = fig.get_size_inches()[0]
    for _ in range(3):
        tight = _tight_width(fig)
        if not tight == tight or tight <= MAX_FIGURE_WIDTH_IN + 0.02:
            return
        over = (tight - min(width, MAX_FIGURE_WIDTH_IN)) / width
        sp = fig.subplotpars
        left = min(0.45, sp.left + over * 0.6)
        right = max(0.55, sp.right - over * 0.4)
        if left >= right:
            break
        fig.subplots_adjust(left=left, right=right)
    # Some overflow is not in the panel area at all — a key or a caption placed
    # in figure coordinates does not move when the panels do. Shrinking the
    # canvas moves it in proportion, so that is the second thing tried.
    for _ in range(3):
        tight = _tight_width(fig)
        if not tight == tight or tight <= MAX_FIGURE_WIDTH_IN + 0.02:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * (MAX_FIGURE_WIDTH_IN / tight), h)

    tight = _tight_width(fig)
    if tight == tight and tight > MAX_FIGURE_WIDTH_IN + 0.02:
        raise ValueError(
            f"{path}: {tight:.2f}in wide once tight-cropped, over the "
            f"{MAX_FIGURE_WIDTH_IN}in page, and widening its margins did not "
            f"recover it. The figsize is {width:.2f}in, so {tight - width:.2f}in "
            "of it is an artist outside the figure — a row label, a key or a "
            "title. Give it fewer columns or a shorter label."
        )


def suptitle(fig, text: str, color: str = INK, fontsize: float | None = None, **kw):
    """A left-aligned figure title that cannot be wider than its figure.

    The defect this replaces: titles were broken into lines by hand, by
    counting characters. A line that was four characters too long made
    ``savefig.bbox="tight"`` grow the saved image past the figure width, and a
    figure saved at 8 inches and rendered by GitHub at the same 870 px as one
    saved at 6.6 arrives with its tick labels a fifth smaller. That is one of
    the two reasons the figures in this repository were once illegible, and
    counting characters is not a fix — measuring is.

    So the text is measured in the figure's own renderer, at the size it will
    actually be drawn, and wrapped to fit. Explicit newlines are kept as hard
    breaks, so a title that is already laid out the way its author wants keeps
    that layout unless a line of it genuinely does not fit.
    """
    import textwrap

    size = TITLE_SIZE if fontsize is None else fontsize
    kw.pop("x", None)
    kw.pop("ha", None)
    flat = " ".join(text.split())
    probe = fig.text(0.0, 0.0, flat, fontsize=size)
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # pragma: no cover - non-Agg backends
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    per_char = probe.get_window_extent(renderer).width / fig.dpi / max(len(flat), 1)
    probe.remove()
    usable = fig.get_size_inches()[0] - 0.22
    columns = max(24, int(usable / max(per_char, 1e-6)))

    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph.strip(), columns) or [""])
    artist = fig.suptitle("\n".join(lines), color=color, fontsize=size,
                          x=0.0, ha="left", **kw)
    # However many lines that came to, the panels have to start below them.
    # Stored rather than applied, because a caller's `save(..., top=...)` runs
    # after this and would otherwise overwrite it; `save` takes whichever of
    # the two leaves more room.
    # 0.98 is where matplotlib hangs a suptitle from; the 0.26in below it is
    # room for a panel title, which is drawn above its axes and so sits in the
    # gap rather than inside it.
    height = artist.get_window_extent(renderer).height / fig.dpi
    fig._bpneat_title_top = 0.98 - (height + 0.26) / fig.get_size_inches()[1]
    return artist


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
