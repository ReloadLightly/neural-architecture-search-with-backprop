"""The one place this repository's figures get their look.

Every plotting module in the project — `figures.py`, the per-protocol figure
modules, `champions.py`, and the scripts under `bench/` — imports its surface,
ink, grid, palette, colormap, fonts and save settings from here. Nothing else
defines them.

That is enforced, not merely intended: `tests/test_style.py` parses every module
that imports matplotlib and fails if it assigns its own `SURFACE`, `SERIES` or
any other token defined below, or if it calls `savefig` with its own settings
instead of :func:`save`. Six separate copies of this palette existed before this
module did, which is six chances for the figures to drift apart.

**The palette is validated, not chosen.** Running the categorical checker over
`SERIES` against `SURFACE` in light mode: all six slots inside the lightness
band, chroma floor met, worst adjacent pair ΔE 9.1 under protanopia, normal
vision floor ΔE 19.6. Three slots fall below 3:1 contrast against the surface,
which is legal **only** with visible relief — so every figure here either
directly labels its marks or ships a CSV table view beside it in the release.
Colour never carries identity alone.

**What is deliberately not restyled.** `results/backprop-neat-v1/figures/`
belongs to an invalidated release and `results/backprop-neat-v2/` carries
errata; both are frozen evidence, and a frozen release should look like what was
released. Their figures stay as they were rendered. Nothing in the README or the
paper shows them.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

#: The chart surface. Slightly warm of white so a white page still frames it.
SURFACE = "#fcfcfb"
#: Primary ink for titles and values, secondary for labels and ticks.
INK = "#0b0b0b"
INK2 = "#52514e"
#: Recessive grid and axis colour — present, never competing with the data.
GRID = "#e3e2df"

#: The validated categorical order. Assigned in this order and never cycled: a
#: seventh series folds into "other", small multiples, or a composite encoding
#: with marker shape.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

#: Status colours, reserved. Never reused as "series 4".
GOOD = "#1baf7a"
BAD = "#e34948"
NEUTRAL = "#c9c8c3"

#: Class 0 → blue, class 1 → orange, through a near-neutral midpoint. The
#: midpoint *is* the decision boundary, so it must read as undecided rather than
#: as a third colour — which is why this is a diverging ramp and not a rainbow.
BOUNDARY_COLOURS = ["#2a78d6", "#dcdcd8", "#eb6834"]
#: The two classes as drawn points: darker than the field so they sit on top.
CLASS_COLOURS = ("#1b4f8f", "#8f3a12")

#: Operators get their own fixed hue order, never cycled. Structural nodes are
#: grey because bias, input and output are not an operator choice.
OP_COLOUR = {
    "tanh": SERIES[0], "relu": SERIES[1], "sigmoid": SERIES[2],
    "gaussian": SERIES[3], "sin": SERIES[4], "abs": SERIES[5],
    "square": "#7b5cd6", "mult": "#b0531f", "add": "#3f8f8f", "null": "#9a9a95",
}

#: Every geometry this project uses, under one spelling.
TASK_LABEL = {
    "xor": "XOR", "circle": "Circles", "spiral": "Spirals",
    "checkerboard": "Checkerboard", "spiral3": "3-arm spiral",
}

#: Type scale. One ladder, so a panel title in v3 is the size of one in v5.
TITLE_SIZE = 11.5
PANEL_TITLE_SIZE = 10.0
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
ANNOT_SIZE = 8.5
LEGEND_SIZE = 8.5

#: Raster density for every committed figure.
DPI = 170


def boundary_cmap():
    """The diverging ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bpneat_boundary", BOUNDARY_COLOURS)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def style_axes(ax, grid_axis: str | None = "y") -> None:
    """Recessive frame: no top or right spine, grey ticks without tick marks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=TICK_SIZE, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def panel(ax) -> None:
    """A bare image panel: framed, no ticks. For boundaries and matrices."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_facecolor(SURFACE)


def figure(nrows: int = 1, ncols: int = 1, figsize=(10, 4.2), grid_axis="y") -> Any:
    """A styled figure and its axes."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax, grid_axis)
    return fig, axes


def suptitle(fig, text: str, size: float = TITLE_SIZE) -> None:
    fig.suptitle(text, color=INK, fontsize=size)


def save(fig, path, **adjust) -> Any:
    """Write a figure at the project's density, onto the project's surface."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if adjust:
        fig.subplots_adjust(**adjust)
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def pale(colour: str, amount: float = 0.74):
    """Toward white, so a filled field never competes with points drawn on it."""
    import matplotlib.colors as mcolors
    import numpy as np

    rgb = np.array(mcolors.to_rgb(colour))
    return tuple(rgb + (1.0 - rgb) * amount)


# --------------------------------------------------------------------------
# Semantic roles — colour encodes the argument, not the row index
# --------------------------------------------------------------------------
#
# The defect this replaces: colour was assigned by position in SERIES, so
# `accuracy-by-task` gave eight rows eight hues, *cycled* — Backprop-NEAT and
# "fixed mixed @ CGP budget" both came out blue — while every row was already
# labelled, so the colour carried nothing and actively misled. Worse, the same
# concept changed colour between figures: "fixed network, matched budget" was
# green in one and pink in another.
#
# Every comparison in this project is between two kinds of thing: something we
# searched for, and something we fixed in advance. So there are two hue
# families, and the ordered distinctions inside each are value steps rather
# than new hues.
#
#   blue   = we searched         deep -> pale : primary algorithm, second
#                                 algorithm, the same space with selection
#                                 removed
#   orange = we did not search   pale -> deep : starved of budget, budget
#                                 matched, strongest control
#   grey   = published reference targets and non-significant verdicts; never
#            a series
#
# Validated: blue vs orange is ΔE 24.7 under protanopia and 33.6 at normal
# vision, both above 3:1 against the surface — materially safer than the
# six-slot categorical order it replaces (worst adjacent pair ΔE 9.1, three
# slots under 3:1). The ramps are sequential, so the governing property is
# lightness monotonicity, which `tests/test_style.py` checks: blue spans
# L 0.49→0.73 and orange L 0.76→0.51, both strictly monotonic.

#: We searched for it.
SEARCH_PRIMARY = "#1a5fb4"
SEARCH_SECONDARY = "#2a78d6"
SEARCH_NULL = "#7aaae8"
SEARCH_RAMP = [SEARCH_PRIMARY, SEARCH_SECONDARY, SEARCH_NULL]

#: We fixed it in advance. Pale is starved, deep is the strongest control.
CONTROL_STARVED = "#f0996a"
CONTROL_MATCHED = "#eb6834"
CONTROL_BEST = "#a8400f"
CONTROL_RAMP = [CONTROL_STARVED, CONTROL_MATCHED, CONTROL_BEST]

#: Neither: a published target, or a verdict that did not reach significance.
REFERENCE = INK2

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


def colour_of(condition: str) -> str:
    """The one colour this condition gets, in every figure in the project."""
    return ROLE_COLOUR[role_of(condition)]


def ramp(role: str, n: int) -> list[str]:
    """``n`` steps inside one role's hue, monotonic in lightness.

    For figures that must separate several members of the same role — v5's
    seven NEAT arms, say. Value steps keep the between-role distinction
    dominant and the within-role distinction secondary, which is the right
    hierarchy: that they are all searches matters more than which one.
    """
    import matplotlib.colors as mcolors
    import numpy as np

    base = ROLE_COLOUR[role]
    if n <= 1:
        return [base]
    rgb = np.array(mcolors.to_rgb(base))
    # Toward white and toward black, never past either.
    out = []
    for i in range(n):
        t = i / (n - 1) * 1.1 - 0.42
        mixed = rgb + (1.0 - rgb) * t if t > 0 else rgb * (1.0 + t)
        out.append(mcolors.to_hex(np.clip(mixed, 0.0, 1.0)))
    return out


# ==========================================================================
# The plate language
# ==========================================================================
#
# One family of figures in this repository is not a chart: the Haeckel plates,
# which draw real champion genomes as naturalist specimens. They need a dark
# ground, and a dark ground is exactly the kind of local decision that six
# copies of a palette used to make differently. So the plate language lives
# here too, next to the chart language, and the plates import it.

#: The plate's paper under ink — warm near-black, never pure black.
PLATE_GROUND = "#17150f"
#: The skeletal line: bone, for structure that reaches the output.
PLATE_BONE = "#ece2cd"
#: Structure the genome carries that never arrives. Drawn, but recessive.
PLATE_BONE_DIM = "#8d8470"
#: The border rule and the plate's lettering.
PLATE_RULE = "#6e6450"
#: Specimen captions.
PLATE_CAPTION = "#c9bda2"

#: Haeckel's accents, used as sparingly as he used them: ochre, rose,
#: verdigris. One hue per operator, in the same assignment as `OP_COLOUR`, so
#: a reader who learns "gaussian is the warm red" in a chart keeps it on a
#: plate. Lithograph pigments rather than screen primaries.
PLATE_ACCENT = {
    "sin": "#d8a521",
    "gaussian": "#c6572f",
    "square": "#9c6ea8",
    "abs": "#7f9f6a",
    "tanh": "#6f98b8",
    "relu": "#c8803a",
    "sigmoid": "#5f9e8a",
    "mult": "#b0485a",
    "add": "#8e8468",
    "null": "#9a927c",
}

#: Plates are lithographs and carry fine line work at small scale, so they are
#: rasterised denser than the charts. Still a token, never a literal.
PLATE_DPI = 200


def plate_accent(op_name: str) -> str:
    """The accent for an operator, or bone for anything unnamed."""
    return PLATE_ACCENT.get(op_name, PLATE_BONE)
