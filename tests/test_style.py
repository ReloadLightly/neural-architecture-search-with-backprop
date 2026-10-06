"""One visual style, enforced rather than intended.

Every figure in this repository is a plate: a warm near-black ground, bone
line work, a double rule round the edge, serif lettering, and a small set of
accent pigments. Not some of them, and not the plates only — all of them,
including whatever is drawn from results that do not exist yet.

That is a claim about consistency, and consistency is exactly the kind of claim
that decays quietly. So it is enforced here rather than intended: these gates
parse every module that imports matplotlib and fail if it defines its own
ground, its own ink, its own palette, or saves a figure without the frame.
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
    "SURFACE", "INK", "INK2", "INK_2", "GRID", "RULE", "SERIES",
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


def test_the_ground_is_the_plate_ground():
    """One ground, and it is the plate's. Pinned because every other colour in
    the project was chosen against it."""
    assert style.SURFACE == "#17150f"
    assert style.PLATE_GROUND == style.SURFACE
    assert style.PLATE_BONE == style.INK


def test_nothing_is_lighter_than_the_ink_or_darker_than_the_ground():
    """The ground is the floor and the bone is the ceiling: any pigment outside
    that range is either invisible or shouting."""
    named = {
        **{k: v for k, v in style.OP_COLOUR.items()},
        **style.ROLE_COLOUR,
        "good": style.GOOD, "bad": style.BAD, "neutral": style.NEUTRAL,
        "rule": style.RULE, "grid": style.GRID,
    }
    floor, ceiling = _luminance(style.SURFACE), _luminance(style.INK)
    for name, colour in named.items():
        assert floor <= _luminance(colour) <= ceiling, f"{name} is outside the plate"


def test_status_colours_are_not_reused_as_series():
    """good / warning / critical are reserved and must not double as series 4."""
    assert style.BAD not in style.SERIES
    assert style.NEUTRAL not in style.SERIES


def test_the_boundary_ramp_is_diverging_with_a_neutral_midpoint():
    """Two hues through a near-neutral middle — never a rainbow, never a hue
    at the midpoint, because the midpoint is the decision boundary itself. On
    a plate the middle is the ground showing through."""
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
    """Cold is "we searched", warm is "we fixed it". Two anchors, no third."""
    import matplotlib.colors as mcolors

    def hue(c):
        return mcolors.rgb_to_hsv(mcolors.to_rgb(c))[0]

    for c in style.SEARCH_RAMP:
        assert 0.5 < hue(c) < 0.72, f"{c} is not in the cold family"
    for c in style.CONTROL_RAMP:
        assert hue(c) < 0.11, f"{c} is not in the warm family"


def test_role_ramps_are_monotonic_in_luminance():
    """A sequential ramp's governing property is luminance.

    On a dark ground the direction flips: weight is brightness, so the primary
    search and the strongest control are the brightest members of their family
    and the ramps run the opposite way from each other.
    """
    search = [_luminance(c) for c in style.SEARCH_RAMP]
    control = [_luminance(c) for c in style.CONTROL_RAMP]
    assert search == sorted(search, reverse=True), "the search ramp is not ordered"
    assert control == sorted(control), "the control ramp is not ordered"
    for role in ("search_primary", "control_matched"):
        steps = [_luminance(c) for c in style.ramp(role, 6)]
        assert steps == sorted(steps), f"{role} ramp is not monotonic"


def test_every_role_colour_carries_on_the_ground():
    """A pigment that does not clear 3:1 above the ground is not on the plate."""
    for name, colour in style.ROLE_COLOUR.items():
        assert _contrast(colour, style.SURFACE) >= 3.0, f"{name} sinks into the ground"


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
    """A plate is set in one face, and it is a serif."""
    import matplotlib as mpl

    assert mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["font.serif"][0] == style.TEXT_FAMILY[0]
    assert "Serif" in style.TEXT_FAMILY[0] and "Serif" in style.DISPLAY_FAMILY[0]


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
    """There is no second palette: the plate names are aliases of the one set."""
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
    floors = {"normal": 21.0, "protan": 14.0, "deutan": 13.0}
    for kind, floor in floors.items():
        d, a, b = _worst_pair(palette, kind)
        assert d >= floor, f"{kind}: {a} and {b} are only ΔE {d:.1f} apart"
    for name, colour in palette.items():
        assert _contrast(colour, style.SURFACE) >= 4.0, f"{name} sinks into the ground"


def test_any_pigment_too_close_to_the_ground_is_declared():
    """The rule is about the ground, so it survives the ground changing: a
    pigment that does not clear it has to be named as needing an outline."""
    low = {
        op for op, c in style.OP_COLOUR.items()
        if op != "null" and _contrast(c, style.SURFACE) < 2.0
    }
    assert low == set(style.NEEDS_RELIEF), (
        "undeclared pigments that sink into the ground: "
        f"{sorted(low ^ set(style.NEEDS_RELIEF))}"
    )


def test_there_is_no_second_operator_palette():
    """A reader who learns an operator's pigment on one figure keeps it on all
    of them, because there is only one table."""
    assert style.PLATE_ACCENT is style.OP_COLOUR


# --------------------------------------------------------------------------
# The plate register, applied to everything
# --------------------------------------------------------------------------


def test_no_figure_is_written_without_the_frame():
    """A plate without its rule is a chart, and the point of this style is that
    there are no charts in here. Every `savefig` in the repository either goes
    through `style.save`, which draws the frame, or draws it itself two
    statements earlier."""
    offenders = []
    for path in _plotting_modules():
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines):
            if ".savefig(" not in line:
                continue
            window = "\n".join(lines[max(0, i - 6):i])
            if "plate_frame" not in window and "style.save" not in window:
                offenders.append(f"{path.relative_to(ROOT)}:{i + 1}")
    assert not offenders, (
        "figures written without the plate's rule at " + ", ".join(offenders)
    )


def test_the_frame_is_drawn_in_inches_not_fractions():
    """A plate's border is the same width on a tall plate and a wide one, so
    the inset is an absolute measure. Taking it as a figure fraction would give
    a 16-inch-wide figure a hairline down the sides and a band across the top.
    """
    import matplotlib.pyplot as plt

    for size in ((4.0, 4.0), (16.0, 4.4), (8.2, 13.6)):
        fig = plt.figure(figsize=size)
        style.plate_frame(fig)
        rect = fig.artists[0]
        inset_x = (size[0] - rect.get_width() * size[0]) / 2
        inset_y = (size[1] - rect.get_height() * size[1]) / 2
        assert abs(inset_x - style.FRAME_IN) < 1e-6, size
        assert abs(inset_y - style.FRAME_IN) < 1e-6, size
        plt.close(fig)


def test_every_plotting_module_draws_on_the_shared_ground():
    """A module may not set its own facecolor to anything but the ground."""
    import re

    pattern = re.compile(r"facecolor\s*=\s*[\"']#")
    offenders = [
        str(p.relative_to(ROOT)) for p in _plotting_modules()
        if pattern.search(p.read_text())
    ]
    assert not offenders, f"hard-coded facecolors in {offenders}"


def test_nothing_draws_in_white():
    """White is not on this plate. A call site that wants "the paper" means the
    ground, and must say `SURFACE` so it follows the ground if it moves."""
    import re

    pattern = re.compile(r"[\"'](?:white|#fff(?:fff)?)[\"']", re.IGNORECASE)
    offenders = []
    for path in _plotting_modules():
        for i, line in enumerate(path.read_text().split("\n")):
            if pattern.search(line) and "noqa: plate" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{i + 1}")
    assert not offenders, "literal white at " + ", ".join(offenders)
