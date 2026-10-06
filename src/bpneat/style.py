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
