"""One visual style, enforced rather than intended.

Six plotting modules each carried their own copy of the surface, ink, grid and
categorical palette. They happened to agree, which is the dangerous case: six
copies that agree today are six chances to disagree tomorrow, and a reader
cannot tell a deliberate difference from a drifted one.

`src/bpneat/style.py` is now the only place those values are defined. These
gates parse every module that imports matplotlib and fail if it defines its own.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bpneat import style  # noqa: E402

#: Names that define the look. No module but `style` may assign one.
RESERVED = {
    "SURFACE", "INK", "INK2", "INK_2", "GRID", "SERIES",
    "SEARCH_PRIMARY", "SEARCH_SECONDARY", "SEARCH_NULL",
    "CONTROL_STARVED", "CONTROL_MATCHED", "CONTROL_BEST",
    "ROLE_COLOUR", "REFERENCE_COLOUR", "WEIGHT_POS", "WEIGHT_NEG",
    "PLATE_GROUND", "PLATE_BONE", "PLATE_BONE_DIM", "PLATE_RULE",
    "PLATE_CAPTION", "PLATE_ACCENT", "PLATE_DPI",
    "DISPLAY_FAMILY", "TEXT_FAMILY",
    "GOOD", "BAD", "NEUTRAL", "DPI",
    "BOUNDARY_COLOURS", "CLASS_COLOURS", "OP_COLOUR", "TASK_LABEL",
    "TITLE_SIZE", "PANEL_TITLE_SIZE", "LABEL_SIZE", "TICK_SIZE",
    "ANNOT_SIZE", "LEGEND_SIZE",
}

STYLE_MODULE = ROOT / "src" / "bpneat" / "style.py"


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


def test_there_are_plotting_modules_to_check():
    mods = _plotting_modules()
    assert len(mods) >= 6, f"only found {len(mods)}; the gate is not exercising anything"


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
                # An animation writer may legitimately render smaller; it is the
                # one case, and it says so at the call site.
                assert "movie" in path.name, (
                    f"{path.relative_to(ROOT)} hard-codes dpi={kw.value.value}; "
                    "use bpneat.style.DPI"
                )


def test_the_palette_is_the_validated_one():
    """The six slots are a validated set, not a preference; pin them."""
    assert style.SERIES == [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
    ]
    assert style.SURFACE == "#fcfcfb"
    assert len(style.SERIES) == len(set(style.SERIES)) == 6


def test_status_colours_are_not_reused_as_series():
    """good / warning / critical are reserved and must not double as series 4."""
    assert style.BAD not in style.SERIES
    assert style.NEUTRAL not in style.SERIES


def test_the_boundary_ramp_is_diverging_with_a_neutral_midpoint():
    """Two hues through a near-neutral middle — never a rainbow, never a hue
    at the midpoint, because the midpoint is the decision boundary itself."""
    import matplotlib.colors as mcolors

    lo, mid, hi = (mcolors.to_rgb(c) for c in style.BOUNDARY_COLOURS)
    spread = max(mid) - min(mid)
    assert spread < 0.05, f"the diverging midpoint {mid} is a colour, not a neutral"
    assert lo != hi
    assert style.boundary_cmap()(0.5) is not None


def test_every_operator_has_its_own_hue():
    from bpneat.genome import OP_NAMES

    assert set(OP_NAMES.values()) <= set(style.OP_COLOUR), "an operator has no colour"
    evolvable = {k: v for k, v in style.OP_COLOUR.items() if k != "null"}
    assert len(set(evolvable.values())) == len(evolvable), "two operators share a hue"


def test_every_task_has_one_label():
    from bpneat.v3.protocol import ALL_TASKS

    assert set(ALL_TASKS) <= set(style.TASK_LABEL)
    assert len(set(style.TASK_LABEL.values())) == len(style.TASK_LABEL)


# --------------------------------------------------------------------------
# Semantic roles: colour encodes the argument, not the row index
# --------------------------------------------------------------------------


def test_no_module_colours_by_list_position():
    """`SERIES` is the old index-based palette. Nothing may read it any more.

    Reading `SERIES[i]` is how Backprop-NEAT and "fixed mixed @ CGP budget"
    came out the same blue, and how one hue named a search in one figure and a
    starved control in the next. Colour comes from `colour_of` / `ramp` /
    `within`, which take a condition or a role, not a position.
    """
    offenders = []
    for path in _plotting_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "SERIES":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            # champions.py reaches through a dict of style tokens.
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

    v4 = {c for b in V4_BLOCKS for c in b.conditions}
    assert v4, "the v4 blocks declare no conditions; the gate checks nothing"
    declared = set(style._ROLE_OF)
    missing = {
        c for c in set(CORE_CONDITIONS) | v4 | set(V5)
        if c not in declared
    }
    assert not missing, f"no declared role for {sorted(missing)}"
    # v3 names its factor arms by level (prop_*, sel_*); those are all searches
    # and `role_of` says so explicitly rather than by falling off the end.
    v3 = {c for b in BLOCKS for c in b.conditions}
    assert v3, "the v3 blocks declare no conditions; the gate checks nothing"
    for c in v3:
        assert style.role_of(c).startswith(("search", "control")), c


def test_the_two_families_are_one_hue_each():
    """Blue is "we searched", orange is "we fixed it". Two anchors, no third."""
    import matplotlib.colors as mcolors

    def hue(c):
        return mcolors.rgb_to_hsv(mcolors.to_rgb(c))[0]

    for c in style.SEARCH_RAMP:
        assert 0.5 < hue(c) < 0.72, f"{c} is not in the blue family"
    for c in style.CONTROL_RAMP:
        assert hue(c) < 0.09, f"{c} is not in the orange family"


def test_role_ramps_are_monotonic_in_lightness():
    """A sequential ramp's governing property is lightness, in both directions."""
    import matplotlib.colors as mcolors

    def light(c):
        r, g, b = mcolors.to_rgb(c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    assert [light(c) for c in style.SEARCH_RAMP] == sorted(
        light(c) for c in style.SEARCH_RAMP)
    assert [light(c) for c in style.CONTROL_RAMP] == sorted(
        (light(c) for c in style.CONTROL_RAMP), reverse=True)
    for role in ("search_primary", "control_matched"):
        steps = [light(c) for c in style.ramp(role, 6)]
        assert steps == sorted(steps), f"{role} ramp is not monotonic"


def test_a_signed_quantity_uses_the_diverging_anchors():
    """Weight sign is diverging, not categorical; it may not borrow a series."""
    assert style.WEIGHT_POS == style.BOUNDARY_COLOURS[0]
    assert style.WEIGHT_NEG == style.BOUNDARY_COLOURS[-1]


def test_status_colours_cannot_be_confused_with_a_condition():
    """GOOD/BAD mark a verdict. No condition may be drawn in either."""
    verdicts = {style.GOOD, style.BAD}
    assert not (verdicts & set(style.ROLE_COLOUR.values()))


# --------------------------------------------------------------------------
# Typography
# --------------------------------------------------------------------------


def test_the_bundled_faces_are_present_and_registered():
    """The committed PNGs must not depend on the machine that drew them."""
    from matplotlib import font_manager

    files = sorted(p.name for p in style.FONT_DIR.glob("*.ttf"))
    assert len(files) >= 6, f"the bundled faces are missing: {files}"
    assert (style.FONT_DIR / "LICENSE-source-serif.md").exists()
    assert (style.FONT_DIR / "LICENSE-source-sans.md").exists()
    names = {f.name for f in font_manager.fontManager.ttflist}
    assert style.TEXT_FAMILY[0] in names, "the text face did not register"
    assert style.DISPLAY_FAMILY[0] in names, "the display face did not register"


def test_the_families_fall_back_rather_than_fail():
    """A checkout without the bundled files still renders, in DejaVu."""
    assert style.TEXT_FAMILY[-1].startswith("DejaVu")
    assert style.DISPLAY_FAMILY[-1].startswith("DejaVu")


def test_importing_style_installs_the_type():
    import matplotlib as mpl

    assert mpl.rcParams["font.sans-serif"][0] == style.TEXT_FAMILY[0]
    assert mpl.rcParams["font.serif"][0] == style.DISPLAY_FAMILY[0]


def test_titles_are_typeset_at_the_single_exit():
    """`style.save` puts headings into the display face, so a module cannot
    forget to at one of its ~60 title call sites."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.suptitle("headline")
    ax.set_title("panel")
    style.typeset(fig)
    assert fig._suptitle.get_fontfamily()[0] == style.DISPLAY_FAMILY[0]
    assert ax.title.get_fontfamily()[0] == style.DISPLAY_FAMILY[0]
    plt.close(fig)


# --------------------------------------------------------------------------
# The plate language
# --------------------------------------------------------------------------


def test_the_plate_palette_lives_here_too():
    """The one figure family with a dark ground is still a single source."""
    import matplotlib.colors as mcolors

    for name in ("PLATE_GROUND", "PLATE_BONE", "PLATE_BONE_DIM", "PLATE_RULE",
                 "PLATE_CAPTION"):
        mcolors.to_rgb(getattr(style, name))
    from bpneat.genome import OP_NAMES

    assert set(OP_NAMES.values()) <= set(style.PLATE_ACCENT)
    for v in style.PLATE_ACCENT.values():
        mcolors.to_rgb(v)
    assert style.plate_accent("not-an-operator") == style.PLATE_BONE


# --------------------------------------------------------------------------
# The operator palette is validated, not chosen
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
    import itertools

    worst = None
    for a, b in itertools.combinations(palette, 2):
        ca, cb = palette[a], palette[b]
        if kind != "normal":
            ca, cb = _simulate(ca, kind), _simulate(cb, kind)
        d = _delta_e(ca, cb)
        if worst is None or d < worst[0]:
            worst = (d, a, b)
    return worst


def test_the_operator_palette_separates_under_colour_vision_deficiency():
    """Nine nominal hues is the hardest case; these are the measured floors.

    The assignment this replaced took six of nine slots from the categorical
    series, which put `tanh` and `square` at ΔE 2.0 under deuteranopia — the
    same colour, for a deuteranope. The thresholds below are the numbers the
    current set actually achieves, rounded down; they are a ratchet, so a
    future edit cannot quietly give one back.
    """
    palette = {k: v for k, v in style.OP_COLOUR.items() if k != "null"}
    floors = {"normal": 26.0, "protan": 8.5, "deutan": 11.9}
    for kind, floor in floors.items():
        d, a, b = _worst_pair(palette, kind)
        assert d >= floor, f"{kind}: {a} and {b} are only ΔE {d:.1f} apart"


def test_a_low_contrast_operator_colour_is_declared_as_needing_relief():
    """Okabe-Ito's yellow is below 3:1 against the surface. That is allowed
    only because every figure draws it with an edge, and the exception is
    named rather than left for a reader to discover."""
    import numpy as np

    def contrast(a, b):
        def lum(c):
            r = _linear(c)
            return float(0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2])
        hi, lo = max(lum(a), lum(b)), min(lum(a), lum(b))
        return (hi + 0.05) / (lo + 0.05)

    low = {
        op for op, c in style.OP_COLOUR.items()
        if op != "null" and contrast(c, style.SURFACE) < 1.6
    }
    assert low == set(style.NEEDS_RELIEF), (
        f"these sit below 1.6:1 and are not declared: {sorted(low ^ set(style.NEEDS_RELIEF))}"
    )
    assert np.isfinite(contrast(style.OP_COLOUR["relu"], style.SURFACE))


def test_the_plate_pigments_track_the_chart_hues():
    """A reader who learns an operator's colour in a chart keeps it on a plate."""
    import matplotlib.colors as mcolors

    for op in style.OP_COLOUR:
        if op in ("null", "add"):
            continue  # `add` is near-black in a chart; a dark ground cannot carry it
        chart = mcolors.rgb_to_hsv(mcolors.to_rgb(style.OP_COLOUR[op]))[0]
        plate = mcolors.rgb_to_hsv(mcolors.to_rgb(style.PLATE_ACCENT[op]))[0]
        gap = min(abs(chart - plate), 1 - abs(chart - plate))
        assert gap < 0.085, f"{op}: plate pigment is a different hue ({gap:.3f})"
