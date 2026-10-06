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
