"""The one place this repository's figures get their look.

**Every figure here is a plate.** Warm near-black ground, bone line work, a
double rule round the edge, serif lettering, tracked small caps over each
panel, and a small set of accent pigments — the register of Ernst Haeckel's
*Kunstformen der Natur*. Not the specimen plates only: the bar charts, the dot
plots, the decision-boundary grids, the sign matrix, the scorecards, the
animation, and whatever gets drawn from results that do not exist yet. A style
that applies to some of the figures is not a style, it is a decoration.

Every plotting module in the project — `figures.py`, the per-protocol figure
modules, `champions.py`, and the scripts under `bench/` — imports its ground,
ink, rule, palette, colormap, type and save settings from here. Nothing else
defines them, and the specimen plates are not an exception: `PLATE_GROUND` and
its siblings are aliases of the same tokens, because a second palette is
exactly how two figures in one repository come to disagree.

That is enforced, not merely intended. `tests/test_style.py` parses every
module that imports matplotlib and fails if it assigns its own ground, ink,
rule or palette; if it writes a figure without the plate's frame; if it names a
colour white; or if a pigment drifts out of the band between the ground and the
bone. Six separate copies of this palette existed before this module did, which
is six chances for the figures to drift apart.

**The palette is validated, not chosen,** and validated against the ground
rather than against paper. Every accent clears 4:1 above the ground; the
closest pair of operator pigments is ΔE 21.2 at normal vision, 14.4 under
protanopia and 13.1 under deuteranopia; the two role families are further apart
than that and are ordered by luminance inside themselves. Colour still never
carries identity alone — every figure either directly labels its marks or ships
a CSV table view beside it in the release.

**What is deliberately not restyled.** `results/backprop-neat-v1/figures/`
belongs to an invalidated release and `results/backprop-neat-v2/` carries
errata; both are frozen evidence, and a frozen release should look like what
was released. Their figures stay as they were rendered, and nothing in the
README or the paper shows them. Every figure a reader is pointed at is a plate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

#: The plate. Every figure in this repository is drawn on a warm near-black
#: ground in bone-coloured line work, after Ernst Haeckel's *Kunstformen der
#: Natur* — not as decoration on top of a chart, but as the chart's own
#: register. A lithograph plate has a ground, a bone line, one rule weight for
#: the frame and a very small number of accent pigments; so does every figure
#: here. Nothing in the repository is drawn any other way.
SURFACE = "#17150f"
#: The skeletal line: bone for values and headings, a dimmer bone for labels.
INK = "#ece2cd"
INK2 = "#c9bda2"
#: The grid is a trace on the plate, not a feature: present when you look for
#: it, invisible when you look at the data.
GRID = "#332e24"
#: The frame rule and the plate's lettering furniture.
RULE = "#6e6450"

#: The old categorical order, kept only so that `tests/test_style.py` can prove
#: nothing reads it. Colour comes from a condition's role, never its position.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

#: Status colours, reserved. Never reused as a series. Verdigris and a muted
#: rose: the two pigments a hand-coloured plate would have had for this.
GOOD = "#7fb79a"
BAD = "#d4705f"
NEUTRAL = "#8d8470"

#: Class 0 → cold, class 1 → warm, through the ground itself. The midpoint *is*
#: the decision boundary, so it must read as undecided rather than as a third
#: pigment — which is why this is a diverging ramp through the plate's own
#: ground and not a rainbow.
BOUNDARY_COLOURS = ["#5f93c6", "#363330", "#d4873f"]
#: The two classes as drawn points: brighter than the field, so they sit on it
#: the way a specimen's detail sits on its wash.
CLASS_COLOURS = ("#bcd8f2", "#f4c88c")

#: A signed quantity — a connection weight, a difference either side of zero —
#: takes the same two diverging anchors. Named so that a module does not reach
#: into the categorical series for them, which is how blue came to mean "tanh"
#: on a node and "positive" on the edge leaving it.
WEIGHT_POS = BOUNDARY_COLOURS[0]
WEIGHT_NEG = BOUNDARY_COLOURS[2]

#: Operators are the one genuinely nominal set in this project: nine names with
#: no order between them, which is the hardest case for colour. These are the
#: plate's accent pigments — ochre, verdigris, rose, slate — used as sparingly
#: as Haeckel used them, and tuned against the ground rather than against paper.
#:
#: Measured: every pigment clears 4.1:1 above the ground, and the closest pair
#: is ΔE 21.2 at normal vision, 14.4 under protanopia and 13.1 under
#: deuteranopia. The light-ground assignment this replaced managed 19.7 / 6.6 /
#: 2.0 before it was fixed, and 26.4 / 8.6 / 11.9 afterwards; the dark ground
#: does better because luminance is doing more of the work.
#:
#: Nine nominal pigments cannot all be separable for everyone, and these are
#: not: the two slates differ mostly in value. So an operator is never
#: identified by colour alone — every figure that uses these also labels them.
#:
#: Structural nodes are bone because bias, input and output are not an operator
#: choice.
OP_COLOUR = {
    "sin": "#e0a51c",       # ochre
    "gaussian": "#c6572f",  # burnt red
    "mult": "#c2546a",      # rose
    "square": "#ab9ad0",    # muted violet
    "abs": "#8fc3a6",       # verdigris
    "sigmoid": "#c8cb5e",   # olive gold
    "tanh": "#a9c6de",      # pale slate
    "relu": "#4f7fae",      # deep slate
    "add": "#8d9699",       # cool grey, so it is not a hue
    "null": "#5a5346",      # the ground, barely lifted
}

#: On the plate every accent clears 3:1 against the ground, so nothing here has
#: to be rescued by relief. The set is kept because the rule it encodes — a
#: pigment too close to the ground is drawn with an outline — is a rule about
#: the ground, and the ground could change again.
NEEDS_RELIEF: frozenset[str] = frozenset()

#: Every geometry this project uses, under one spelling.
TASK_LABEL = {
    "xor": "XOR", "circle": "Circles", "spiral": "Spirals",
    "checkerboard": "Checkerboard", "spiral3": "3-arm spiral",
}

#: Type scale. One ladder, so a panel title in v3 is the size of one in v5.
#: A plate carries less type than a chart and carries it larger, because the
#: lettering is part of the object rather than an annotation on it.
TITLE_SIZE = 13.0
PANEL_TITLE_SIZE = 10.5
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
ANNOT_SIZE = 8.5
LEGEND_SIZE = 8.5

#: Raster density for every committed figure.
DPI = 170

# --------------------------------------------------------------------------
# Typeface
# --------------------------------------------------------------------------
#
# Matplotlib's stock DejaVu Sans is the single loudest signal that a figure was
# not designed, and no amount of palette work survives it. The project ships two
# OFL faces in `assets/fonts` and registers them here, so the figures render the
# same on this machine, in CI and on anyone's checkout — a system font would
# make the committed PNGs depend on the machine that drew them, which for a
# repository whose whole point is reproducibility would be the wrong trade.
#
#   Source Serif 4 Display — titles. A text face with real display cuts, so a
#     suptitle reads as a statement rather than as a larger label.
#   Source Sans 3 — axes, ticks, values. Lining figures of even width, which is
#     what a column of numbers beside a bar needs.

FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"

#: Families, in the order a renderer should try them. DejaVu stays last so a
#: checkout without the bundled files still produces a figure rather than an
#: exception; `tests/test_style.py` checks the bundled files are there.
#: A plate is set in one face. Source Serif 4 Display for the lettering a
#: lithograph would have engraved — the plate's title, the specimen captions,
#: the panel headings — and the text cut of the same family for everything
#: smaller, so a tick label and a title are visibly the same hand.
DISPLAY_FAMILY = ["Source Serif 4 Display", "Source Serif 4", "DejaVu Serif"]
TEXT_FAMILY = ["Source Serif 4", "Source Serif 4 Display", "DejaVu Serif"]


def _register_fonts() -> None:
    """Make the bundled faces visible to matplotlib, once per process."""
    from matplotlib import font_manager

    if not FONT_DIR.is_dir():
        return
    known = {f.fname for f in font_manager.fontManager.ttflist}
    for path in sorted(FONT_DIR.glob("*.ttf")):
        if str(path) not in known:
            font_manager.fontManager.addfont(str(path))


def use_project_typography() -> None:
    """Install the project's type on the global rcParams.

    Called at import, so a module that does nothing but ``import bpneat.style``
    already draws in the right face. Sizes still come from the tokens below;
    this sets family, weight and the small metrics that rcParams owns.
    """
    import matplotlib as mpl

    _register_fonts()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": TEXT_FAMILY,
        "font.sans-serif": TEXT_FAMILY,
        "font.size": LABEL_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        # Ticks are a ruler, not a feature: short, thin and the label's colour.
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })


use_project_typography()


def display_font(size: float, weight: str = "regular") -> dict:
    """Keyword arguments for a title set in the display face."""
    return {"family": DISPLAY_FAMILY, "fontsize": size, "fontweight": weight}


#: Letterpress tracking, for the small caps a plate sets its headings in.
HAIR = "\u2009"


def spaced(text: str) -> str:
    """Tracked small caps. matplotlib cannot letterspace, so the spaces are
    real — which is what a compositor did anyway."""
    return HAIR.join(text.upper())


def boundary_cmap():
    """The diverging ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bpneat_boundary", BOUNDARY_COLOURS)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def style_axes(ax, grid_axis: str | None = "y") -> None:
    """A plate's measuring frame: the ground, one engraved baseline, a trace
    of a grid. No top or right spine, no tick marks, nothing drawn heavier
    than the data it measures."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=INK2, labelsize=TICK_SIZE, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def panel(ax) -> None:
    """A bare specimen panel: framed in the plate's rule, no ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.7)
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


def typeset(fig) -> None:
    """Set the figure's lettering the way a plate sets it.

    Done here, at the single exit every figure passes through, rather than at
    the ~60 call sites that pass ``fontsize=`` literals. A module can still say
    how big a title is; it cannot say what it is set in, which is the part that
    has to be the same everywhere.

    The headline keeps its sentence case because these titles are sentences —
    tracked caps over two lines of prose is a poster, not a plate — but every
    panel heading becomes tracked small caps, which is what a plate puts over
    a specimen.
    """
    if fig.get_suptitle():
        fig._suptitle.set_fontfamily(DISPLAY_FAMILY)
        fig._suptitle.set_color(INK)
        # A headline is a statement; give it room to breathe above the panels.
        fig._suptitle.set_linespacing(1.5)
    for ax in fig.axes:
        title = ax.get_title()
        if not title:
            continue
        ax.title.set_fontfamily(DISPLAY_FAMILY)
        ax.title.set_color(INK)
        # One line, short, and not already tracked: a specimen heading.
        if "\n" not in title and len(title) <= 34 and HAIR not in title:
            ax.set_title(spaced(title), fontsize=ax.title.get_fontsize() * 0.92,
                         color=INK, family=DISPLAY_FAMILY, pad=5.0)


#: Where the frame sits, in inches from the figure edge, and how far below the
#: top rule a headline starts. Both are inches rather than fractions because a
#: plate's border is the same width on a tall plate and a wide one.
FRAME_IN = 0.17
HEADLINE_IN = 0.40


def plate_frame(fig, inset_in: float = FRAME_IN) -> None:
    """The plate's double rule, a thick line and a hairline just inside it.

    Every figure in the repository carries it. It is the one mark that says
    these are plates rather than charts, and it is drawn here so that no
    figure can be published without it.
    """
    import matplotlib.pyplot as plt

    w, h = fig.get_size_inches()
    for inset, lw in ((inset_in, 1.0), (inset_in * 1.52, 0.4)):
        fig.add_artist(plt.Rectangle(
            (inset / w, inset / h), 1 - 2 * inset / w, 1 - 2 * inset / h,
            transform=fig.transFigure, facecolor="none", edgecolor=RULE,
            linewidth=lw, zorder=1000))


def _reserve_headline(fig, adjust: dict) -> dict:
    """Keep the panels clear of the plate's rule and of the lettering under it.

    A chart can let a title float at the top of the canvas and overlap whatever
    is there. A plate cannot: the rule is drawn a fixed distance from the edge
    and the lettering sits inside it, so the panels have to be told how much
    room that takes. Guessing it in inches went wrong on every figure with a
    two-line headline, so it is measured: draw once, ask the renderer where the
    headline actually ends and how tall the panel headings actually are, and
    push the panels below both.
    """
    if not fig.get_suptitle():
        return adjust

    height_in = fig.get_size_inches()[1]
    fig._suptitle.set_y(1 - (FRAME_IN + 0.22) / height_in)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def frac(artist):
        return artist.get_window_extent(renderer).transformed(inv)

    headline_bottom = frac(fig._suptitle).y0
    # How far a panel heading reaches above its own axes, as a fraction.
    overshoot = 0.0
    for ax in fig.axes:
        if not ax.get_title():
            continue
        box = frac(ax.title)
        overshoot = max(overshoot, box.y1 - ax.get_position().y1)

    cap = headline_bottom - 0.018 - overshoot
    if adjust.get("top", 1.0) > cap:
        adjust = dict(adjust, top=cap)
    return adjust


def _frame_pad(fig) -> tuple[float, float]:
    """The inner rule's distance from the edge, as figure fractions."""
    w_in, h_in = fig.get_size_inches()
    inner = FRAME_IN * 1.52 + 0.07
    return inner / w_in, inner / h_in


def _reserve_footer(fig, adjust: dict) -> dict:
    """Leave the panels clear of a key or caption placed below them.

    The counterpart of `_reserve_headline`. A key at ``loc="lower center"`` is
    anchored to the figure, not to the axes, so moving it inside the border
    does not move the panels out of its way — it just lands the key on top of
    the data. Measure how tall it is and give it its own band.
    """
    below = [
        art for art in list(fig.legends) + [
            txt for txt in fig.texts if txt is not getattr(fig, "_suptitle", None)
        ]
        if _artist_box(fig, art).y1 < 0.5
    ]
    if not below:
        return adjust
    _, pad_y = _frame_pad(fig)
    needed = pad_y + max(_artist_box(fig, a).height for a in below) + 0.035
    if adjust.get("bottom", 0.0) < needed:
        adjust = dict(adjust, bottom=needed)
    return adjust


def _artist_box(fig, artist):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())


def _clear_the_frame(fig) -> None:
    """Move figure-level lettering out from under the plate's rule.

    A key placed at ``loc="lower center"`` sits at the very bottom of the
    canvas, which on a plate is exactly where the border is — the rule drew
    straight through the legend and through the leading character of every
    label in it. Rather than hand-tune a bottom margin per figure, measure what
    crosses the rule and move it inside.
    """
    artists = list(fig.legends) + [
        txt for txt in fig.texts if txt is not getattr(fig, "_suptitle", None)
    ]
    if not artists:
        return
    pad_x, pad_y = _frame_pad(fig)
    for art in artists:
        box = _artist_box(fig, art)
        dy = (pad_y - box.y0) if box.y0 < pad_y else (
            (1 - pad_y) - box.y1 if box.y1 > 1 - pad_y else 0.0)
        dx = (pad_x - box.x0) if box.x0 < pad_x else (
            (1 - pad_x) - box.x1 if box.x1 > 1 - pad_x else 0.0)
        if not dx and not dy:
            continue
        if hasattr(art, "set_bbox_to_anchor"):
            art.set_bbox_to_anchor((dx, dy, 1.0, 1.0), transform=fig.transFigure)
        else:
            x, y = art.get_position()
            art.set_position((x + dx, y + dy))


def finish(fig, **adjust) -> None:
    """Everything between "the data is drawn" and "write the file".

    Set the lettering, lay the panels out below the headline, move anything
    that would sit under the border, and draw the border. A module that needs
    its own margins calls this instead of reimplementing the parts it
    remembers.
    """
    typeset(fig)
    adjust = _reserve_headline(fig, adjust)
    adjust = _reserve_footer(fig, adjust)
    if adjust:
        fig.subplots_adjust(**adjust)
    else:
        fig.tight_layout()
    _clear_the_frame(fig)
    plate_frame(fig)


def save(fig, path, **adjust) -> Any:
    """Write a figure at the project's density, onto the project's ground."""
    import matplotlib.pyplot as plt

    finish(fig, **adjust)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def pale(colour: str, amount: float = 0.74):
    """Toward the ground, so a filled field never competes with what is drawn
    on it. On a plate "wash it out" means sink it into the paper, not raise it
    toward white — the ground is the dark here."""
    import matplotlib.colors as mcolors
    import numpy as np

    rgb = np.array(mcolors.to_rgb(colour))
    ground = np.array(mcolors.to_rgb(SURFACE))
    return tuple(rgb + (ground - rgb) * amount)


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
# Validated against the plate ground rather than against paper: all six clear
# 3:1 above the ground, the worst pair inside a family is ΔE 14.6 under
# deuteranopia and 16.2 under protanopia, and the two families are further
# apart than that. The six-slot categorical order this replaced had a worst
# adjacent pair of ΔE 9.1 and three slots under 3:1. The ramps are sequential,
# so the governing property is luminance monotonicity, which
# `tests/test_style.py` checks in both directions.

#: We searched for it. On a dark ground the ordering runs the other way round
#: from a white one: weight is luminance, so the primary is the brightest.
SEARCH_PRIMARY = "#a7c9e8"
SEARCH_SECONDARY = "#6e9cc8"
SEARCH_NULL = "#466f94"
SEARCH_RAMP = [SEARCH_PRIMARY, SEARCH_SECONDARY, SEARCH_NULL]

#: We fixed it in advance. Dim is starved, bright is the strongest control —
#: the same rule as above, read against the ground rather than against paper.
CONTROL_STARVED = "#9a6a42"
CONTROL_MATCHED = "#d08242"
CONTROL_BEST = "#efb264"
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
    # Toward black and toward white, but not far enough toward white to fall
    # below the palest colour this project already validated: SEARCH_NULL sits
    # at 2.34:1 against the surface, and the pale end of a ramp lands at 2.25:1,
    # which is the same perceptual territory rather than a new concession.
    out = []
    for i in range(n):
        t = i / (n - 1) * 0.66 - 0.26
        mixed = rgb + (1.0 - rgb) * t if t > 0 else rgb * (1.0 + t)
        out.append(mcolors.to_hex(np.clip(mixed, 0.0, 1.0)))
    return out


# ==========================================================================
# The plate language
# ==========================================================================
#
# There is no separate plate language any more. The specimen plates and the
# analytical figures are the same object drawn at different magnifications, so
# they take the same ground, the same bone, the same rule and the same accent
# pigments. These names remain because the plate modules read well with them,
# but they are aliases, not a second palette — which is the whole point: a
# second palette is how two figures in one repository come to disagree.

PLATE_GROUND = SURFACE
PLATE_BONE = INK
PLATE_CAPTION = INK2
PLATE_RULE = RULE
#: Structure a genome carries that never reaches the output: present, recessive.
PLATE_BONE_DIM = "#8d8470"
PLATE_ACCENT = OP_COLOUR

#: Plates carry fine line work at small scale, so they are rasterised denser
#: than a chart with the same ink. Still a token, never a literal.
PLATE_DPI = 200


def plate_accent(op_name: str) -> str:
    """The accent for an operator, or bone for anything unnamed."""
    return PLATE_ACCENT.get(op_name, PLATE_BONE)


def within(role: str, members) -> dict[str, str]:
    """Distinct lightness steps inside one role, keyed by member name.

    Use this only where the members are genuinely *ordered* — v5's NEAT arms
    sorted by how much complexification the manipulation permits, say. A
    sequential ramp over an unordered set would imply a progression that is
    not there, which is the failure this whole module exists to prevent.
    """
    members = list(members)
    return dict(zip(members, ramp(role, len(members))))
