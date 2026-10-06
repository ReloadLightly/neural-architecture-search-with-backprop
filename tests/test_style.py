"""One visual style, enforced rather than intended.

Every figure in this repository is drawn in the style of the sibling project
`competitive-coevolution-of-slimes`: white ground, no frame, top and right
spines off, a very light grid behind the data, stock sans at 8.5pt, legends
without a box, and a small fixed set of muted hues assigned to conditions by
name. The two repositories are one body of work and should read as one.

That is a claim about consistency, and consistency decays quietly. So these
gates parse every module that imports matplotlib and fail if it defines its own
ground, ink, grid or palette; if it hard-codes a colour, a density or a type
size; or if a hue is chosen by a row's position rather than by what the row is.
"""

from __future__ import annotations

import ast
import itertools
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bpneat import style  # noqa: E402

#: Names that define the look. No module but `style` may assign one.
RESERVED = {
    "SURFACE", "INK", "INK2", "INK_2", "GRID", "RULE", "SERIES", "PALETTE",
    "GOOD", "BAD", "NEUTRAL", "DPI",
    "BOUNDARY_COLOURS", "CLASS_COLOURS", "OP_COLOUR", "TASK_LABEL",
    "SEARCH_PRIMARY", "SEARCH_SECONDARY", "SEARCH_NULL",
    "CONTROL_STARVED", "CONTROL_MATCHED", "CONTROL_BEST",
    "ROLE_COLOUR", "WEIGHT_POS", "WEIGHT_NEG",
    "TITLE_SIZE", "PANEL_TITLE_SIZE", "LABEL_SIZE", "TICK_SIZE",
    "ANNOT_SIZE", "LEGEND_SIZE",
}

STYLE_MODULE = ROOT / "src" / "bpneat" / "style.py"

#: The upstream project this style is ported from, for the record.
SLIMES_RCPARAMS = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "savefig.bbox": "tight",
}


def _plotting_modules() -> list[Path]:
    out = []
    for base in (ROOT / "src" / "bpneat", ROOT / "bench"):
        for path in sorted(base.rglob("*.py")):
            if path == STYLE_MODULE:
                continue
            text = path.read_text()
            if "matplotlib" in text or "plt." in text:
                out.append(path)
    return out


def _luminance(colour: str) -> float:
    import matplotlib.colors as mcolors
    import numpy as np

    rgb = np.array(mcolors.to_rgb(colour))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------
# One source
# --------------------------------------------------------------------------


def test_there_are_plotting_modules_to_check():
    mods = _plotting_modules()
    assert len(mods) >= 5, f"only found {len(mods)}; the gate is not exercising anything"


@pytest.mark.parametrize("path", _plotting_modules(), ids=lambda p: p.name)
def test_no_module_defines_its_own_style(path: Path):
    """A reserved name may be imported from `style`, never assigned."""
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            names = (
                [e.id for e in t.elts if isinstance(e, ast.Name)]
                if isinstance(t, ast.Tuple)
                else ([t.id] if isinstance(t, ast.Name) else [])
            )
            offenders += [n for n in names if n in RESERVED]
    assert not offenders, (
        f"{path.relative_to(ROOT)} defines {sorted(set(offenders))} itself; "
        "import them from bpneat.style instead"
    )


@pytest.mark.parametrize("path", _plotting_modules(), ids=lambda p: p.name)
def test_no_module_hard_codes_a_colour(path: Path):
    """Every hue in the project comes from the shared palette.

    This is the gate that keeps the two repositories looking like one: a hex
    literal in a figure module is a hue nobody chose on purpose.
    """
    hexes = re.findall(r"[\"']#[0-9a-fA-F]{3,8}[\"']", path.read_text())
    assert not hexes, (
        f"{path.relative_to(ROOT)} hard-codes {sorted(set(hexes))}; "
        "take the hue from bpneat.style"
    )


@pytest.mark.parametrize("path", _plotting_modules(), ids=lambda p: p.name)
def test_every_figure_is_saved_at_the_project_density(path: Path):
    """`savefig` must use the shared DPI token, never a literal."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("savefig", "saving"):
            continue
        for kw in node.keywords:
            if kw.arg == "dpi" and isinstance(kw.value, ast.Constant):
                raise AssertionError(
                    f"{path.relative_to(ROOT)} hard-codes dpi={kw.value.value}; "
                    "use bpneat.style.DPI"
                )


@pytest.mark.parametrize("path", _plotting_modules(), ids=lambda p: p.name)
def test_nothing_draws_a_page_border(path: Path):
    """The dark-plate experiment put a double rule round every figure. It is
    gone, and `savefig.bbox="tight"` means there is no margin to put one in."""
    text = path.read_text()
    for banned in ("plate_frame", "FRAME_IN", "PLATE_", "DISPLAY_FAMILY"):
        assert banned not in text, (
            f"{path.relative_to(ROOT)} still references {banned}"
        )


# --------------------------------------------------------------------------
# The style is the sibling project's style
# --------------------------------------------------------------------------


def test_importing_style_installs_the_slimes_rcparams():
    """Pinned against the upstream `make_figures.style()` it was ported from."""
    import matplotlib as mpl

    for key, want in SLIMES_RCPARAMS.items():
        got = mpl.rcParams[key]
        assert got == want, f"{key}: {got!r} != {want!r}"


def test_the_ground_is_white_and_titles_are_left_aligned():
    import matplotlib as mpl

    assert style.SURFACE == "#ffffff"
    assert mpl.rcParams["figure.facecolor"] == style.SURFACE
    assert mpl.rcParams["savefig.facecolor"] == style.SURFACE
    assert mpl.rcParams["axes.titlelocation"] == "left"


def test_the_palette_is_the_sibling_projects_palette():
    """These twelve hues are `competitive-coevolution-of-slimes`' `COLORS`.
    Nothing here invents a hue; it only chooses which one a series gets."""
    assert set(style.PALETTE.values()) == {
        "#222222", "#3B6EA8", "#1F7A5A", "#E08B3C", "#B0413E", "#8C5A2B",
        "#7A5EA6", "#C1445E", "#4C956C", "#946A9E", "#8A6A55", "#4FA3B8",
    }
    assert len(style.PALETTE) == 12


def test_every_assigned_hue_comes_from_that_palette():
    """Role colours and operator colours are drawn from the twelve, not mixed."""
    allowed = set(style.PALETTE.values()) | {style.NEUTRAL}
    for name, colour in style.ROLE_COLOUR.items():
        assert colour in allowed, f"role {name} uses {colour}, not a palette hue"
    for name, colour in style.OP_COLOUR.items():
        assert colour in allowed, f"operator {name} uses {colour}, not a palette hue"


def test_every_series_colour_carries_on_white():
    """A hue that does not separate from the page is not a series colour."""
    for name, colour in {**style.ROLE_COLOUR, **style.OP_COLOUR}.items():
        if name == "null":
            continue
        assert _contrast(colour, style.SURFACE) >= 2.5, f"{name} is too pale"


# --------------------------------------------------------------------------
# Colour encodes the argument, not the row index
# --------------------------------------------------------------------------


def test_no_module_colours_by_list_position():
    """`SERIES` is the old index-based palette. Nothing may read it.

    Reading `SERIES[i]` is how Backprop-NEAT and "fixed mixed @ CGP budget"
    came out the same blue, and how one hue named a search in one figure and a
    starved control in the next.
    """
    offenders = []
    for path in _plotting_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "SERIES":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if (isinstance(node, ast.Constant) and node.value == "SERIES"
                    and path.name == "champions.py"):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, (
        "colour taken by list position at " + ", ".join(sorted(set(offenders)))
    )


def test_every_condition_in_every_protocol_has_a_declared_role():
    """A new condition must be given a role, not silently defaulted."""
    from bpneat.conditions import CORE_CONDITIONS
    from bpneat.v3.protocol import BLOCKS
    from bpneat.v4.protocol import BLOCKS as V4_BLOCKS
    from bpneat.v5.protocol import ALL_CONDITIONS as V5
    from bpneat.v6.protocol import EVERY_CONDITION as V6

    v4 = {c for b in V4_BLOCKS for c in b.conditions}
    assert v4, "the v4 blocks declare no conditions; the gate checks nothing"
    declared = set(style._ROLE_OF)
    missing = {
        c for c in set(CORE_CONDITIONS) | v4 | set(V5) | set(V6) if c not in declared
    }
    assert not missing, f"no declared role for {sorted(missing)}"
    v3 = {c for b in BLOCKS for c in b.conditions}
    assert v3, "the v3 blocks declare no conditions; the gate checks nothing"
    for c in v3:
        assert style.role_of(c).startswith(("search", "control")), c


def test_the_budget_ladder_is_an_axis_and_never_a_hue():
    """A rung must not change a colour: ordered quantities belong on an axis.

    The gate also pins the ladder against the protocol, so a rung added in one
    place and not the other fails here rather than silently colouring by
    position again.
    """
    from bpneat.v6.protocol import ALL_BUDGETS, ARMS

    assert style.V6_BUDGET_CANDIDATES == tuple(b.candidates for b in ALL_BUDGETS)
    for arm in ARMS:
        hues = {style.colour_of(f"{arm}_b{c}") for c in style.V6_BUDGET_CANDIDATES}
        assert len(hues) == 1, f"{arm} changes hue with its budget"
        assert hues == {style.arm_colour(arm)}
    assert len({style.arm_colour(a) for a in ARMS}) == len(ARMS)


def test_the_two_families_stay_apart():
    """Cold is "we searched", warm is "we fixed it in advance"."""
    import matplotlib.colors as mcolors

    def warmth(c):
        r, g, b = mcolors.to_rgb(c)
        return r - b

    for c in style.SEARCH_RAMP:
        assert warmth(c) < 0.05, f"{c} is not in the cold family"
    for c in style.CONTROL_RAMP:
        assert warmth(c) > 0.15, f"{c} is not in the warm family"


def test_a_signed_quantity_uses_the_diverging_anchors():
    """Weight sign is diverging, not categorical; it may not borrow a series."""
    assert style.WEIGHT_POS == style.BOUNDARY_COLOURS[0]
    assert style.WEIGHT_NEG == style.BOUNDARY_COLOURS[-1]


def test_status_colours_are_not_reused_as_a_condition():
    """good / failed mark a verdict. No condition may be drawn in either."""
    assert style.NEUTRAL not in style.ROLE_COLOUR.values()


def test_the_boundary_ramp_is_diverging_with_a_neutral_midpoint():
    """Two hues through a near-neutral middle — never a rainbow, never a hue
    at the midpoint, because the midpoint is the decision boundary itself."""
    import matplotlib.colors as mcolors

    lo, mid, hi = (mcolors.to_rgb(c) for c in style.BOUNDARY_COLOURS)
    assert max(mid) - min(mid) < 0.05, f"the midpoint {mid} is a colour, not a neutral"
    assert lo != hi
    assert style.boundary_cmap()(0.5) is not None


def test_every_operator_has_its_own_hue():
    from bpneat.genome import OP_NAMES

    assert set(OP_NAMES.values()) <= set(style.OP_COLOUR), "an operator has no colour"
    evolvable = {k: v for k, v in style.OP_COLOUR.items() if k != "null"}
    assert len(set(evolvable.values())) == len(evolvable), "two operators share a hue"


def test_a_role_ramp_gives_distinct_hues_not_tints():
    """Tints of one hue at small sizes were what stopped being legible."""
    for role in style.ROLE_COLOUR:
        for n in (2, 3, 6):
            hues = style.ramp(role, n)
            assert len(set(hues)) == n, f"{role} at n={n} repeats a hue"
            assert set(hues) <= set(style.PALETTE.values())


def test_every_task_has_one_label():
    from bpneat.v3.protocol import ALL_TASKS

    assert set(ALL_TASKS) <= set(style.TASK_LABEL)
    assert len(set(style.TASK_LABEL.values())) == len(style.TASK_LABEL)


# --------------------------------------------------------------------------
# Colour-vision deficiency
# --------------------------------------------------------------------------


def _linear(colour):
    import matplotlib.colors as mcolors
    import numpy as np

    rgb = np.array(mcolors.to_rgb(colour))
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _simulate(colour, kind):
    """Viénot-Brettel-Mollon dichromat simulation, the standard linear one."""
    import matplotlib.colors as mcolors
    import numpy as np

    to_lms = np.array([[0.31399, 0.63951, 0.04649],
                       [0.15537, 0.75789, 0.08670],
                       [0.01775, 0.10945, 0.87247]])
    collapse = {
        "protan": np.array([[0.0, 1.05118294, -0.05116099], [0, 1, 0], [0, 0, 1]]),
        "deutan": np.array([[1, 0, 0], [0.9513092, 0.0, 0.04866992], [0, 0, 1]]),
    }[kind]
    out = np.clip(np.linalg.inv(to_lms) @ (collapse @ (to_lms @ _linear(colour))), 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    return mcolors.to_hex(np.clip(srgb, 0, 1))


def _delta_e(a, b):
    import numpy as np

    def lab(colour):
        m = np.array([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]])
        xyz = (m @ _linear(colour)) / np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

    return float(np.linalg.norm(lab(a) - lab(b)))


def _worst_pair(palette, kind):
    worst = None
    for a, b in itertools.combinations(palette, 2):
        ca, cb = palette[a], palette[b]
        if kind != "normal":
            ca, cb = _simulate(ca, kind), _simulate(cb, kind)
        d = _delta_e(ca, cb)
        if worst is None or d < worst[0]:
            worst = (d, a, b)
    return worst


def test_the_roles_separate_under_colour_vision_deficiency():
    """Which of the twelve hues each role gets was chosen by searching them
    for the most separable six. These are the floors that search achieved,
    rounded down; they are a ratchet."""
    roles = {k: v for k, v in style.ROLE_COLOUR.items() if k != "reference"}
    for kind, floor in {"normal": 15.0, "protan": 13.5, "deutan": 11.0}.items():
        d, a, b = _worst_pair(roles, kind)
        assert d >= floor, f"{kind}: {a} and {b} are only ΔE {d:.1f} apart"


def test_the_operator_palette_separates_under_colour_vision_deficiency():
    """Nine nominal hues is the hardest case; these are the measured floors."""
    palette = {k: v for k, v in style.OP_COLOUR.items() if k != "null"}
    for kind, floor in {"normal": 11.5, "protan": 7.5, "deutan": 8.5}.items():
        d, a, b = _worst_pair(palette, kind)
        assert d >= floor, f"{kind}: {a} and {b} are only ΔE {d:.1f} apart"


# --------------------------------------------------------------------------
# Size — the largest single cause of an unreadable figure here
# --------------------------------------------------------------------------

#: Inches. GitHub renders a README image at about 870 px; at 200 dpi a figure
#: wider than this arrives below half size, and 7.5pt tick labels land at under
#: 4pt. No palette fixes that. The sibling project's widest figure is 6.8.
MAX_FIGURE_WIDTH_IN = 6.8


@pytest.mark.parametrize("path", _plotting_modules(), ids=lambda p: p.name)
def test_no_figure_is_wider_than_the_page(path: Path):
    """Panels stack downwards, never sideways.

    A literal width is read straight off the `figsize=`; a computed one
    (`figsize=(2.2 * len(tasks), 3.0)`) is evaluated at the largest number of
    panels any protocol has, which is five.
    """
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "figsize" or not isinstance(kw.value, ast.Tuple):
                continue
            width = kw.value.elts[0]
            value = None
            if isinstance(width, ast.Constant):
                value = float(width.value)
            elif (isinstance(width, ast.BinOp) and isinstance(width.op, ast.Mult)
                  and isinstance(width.left, ast.Constant)):
                value = float(width.left.value) * 5  # the widest protocol
            elif (isinstance(width, ast.BinOp) and isinstance(width.op, ast.Add)
                  and isinstance(width.left, ast.BinOp)
                  and isinstance(width.left.left, ast.Constant)
                  and isinstance(width.right, ast.Constant)):
                value = float(width.left.left.value) * 5 + float(width.right.value)
            if value is not None and value > MAX_FIGURE_WIDTH_IN:
                offenders.append(f"line {node.lineno}: {value:.1f}in")
    assert not offenders, (
        f"{path.relative_to(ROOT)} draws figures wider than "
        f"{MAX_FIGURE_WIDTH_IN}in — {offenders}"
    )
