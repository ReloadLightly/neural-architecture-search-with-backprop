"""Kunstformen der Architektur — the evolved networks drawn as naturalist plates.

Ernst Haeckel's *Kunstformen der Natur* (1899–1904) drew radiolaria: small,
radially symmetric skeletal forms, many specimens to a plate, pale against a
dark ground, arranged by kind, with one dominant specimen given its own space.

The analogy is not decoration. Every specimen on these plates is a real champion
genome, loaded from a committed run record exactly as validation selected it —
the same objects the analytical figures summarise, drawn as forms rather than as
numbers. A plate shows what a table cannot: that these are *small*, that they are
irregular, that each geometry gets a different build, that removing a constraint
grows them, and that the fixed network they are measured against is a different
order of thing entirely.

Honesty constraints, because a beautiful figure is still a figure:

* No genome is idealised, pruned or symmetrised. The layout is radial, which is
  a presentation choice; the nodes, edges, operators and weights are the record's.
* Structure that never reaches the output is drawn, faintly, drifting outside
  the rim — the plate shows represented *and* causal form, as the analytical
  figures do.
* Each specimen is captioned with its true causal size out of its represented
  size, and the plate states which release it came from.
* Specimens are the **median** replicate of thirty by sealed-test accuracy,
  never the best. A plate of hand-picked champions would be the selection error
  this whole project is about.
* The fixed control appears once, not once per geometry, because its topology
  does not vary with the geometry. Repeating it would have implied otherwise.

The dark ground is the one place in the repository that departs from the light
surface, and it departs on purpose: these are plates, not charts. Nothing here
is a result, and no number on them appears in any claim.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpneat import style  # noqa: E402
from bpneat.champions import causal_elements  # noqa: E402
from bpneat.genome import N_STRUCTURAL, OP_NAMES, OUT  # noqa: E402
from bpneat.record import deserialise_genome  # noqa: E402
from bpneat.v3.datasets import make_bundle  # noqa: E402

OUT_DIR = ROOT / "docs" / "figures"

# --------------------------------------------------------------------------
# The plate's palette lives in bpneat.style, like every other palette here
# --------------------------------------------------------------------------

# No local copies of the palette: the names below are the shared ones, read
# through `style` at every use, so this module cannot drift from the figures.
from bpneat.style import PLATE_BONE_DIM as BONE_DIM  # noqa: E402
from bpneat.style import plate_accent as accent  # noqa: E402
from bpneat.style import spaced  # noqa: E402


# --------------------------------------------------------------------------
# Radial layout — what makes a radiolarian a radiolarian
# --------------------------------------------------------------------------


def radial_layout(g) -> tuple[dict[int, tuple[float, float]], set[int]]:
    """Place the output at the centre, the rest on rings by distance from it.

    A feed-forward graph drawn left to right is a flow chart. The same graph
    drawn with its output at the centre and its sources on the rim is a radial
    form — which is both what Haeckel's subjects look like and a legitimate
    layout, because the ring index *is* the depth.

    Nodes the output cannot reach backwards are placed just outside the rim,
    where they read as carried but detached. They are excluded from the radius
    normalisation so that they cannot shrink the organism.
    """
    incoming: dict[int, list[int]] = {i: [] for i in range(g.n_nodes)}
    for ci in range(g.n_connections):
        if g.active[ci]:
            incoming[g.dst[ci]].append(g.src[ci])

    ring = {OUT: 0}
    frontier = [OUT]
    while frontier:
        nxt = []
        for node in frontier:
            for src in incoming.get(node, ()):
                if src not in ring:
                    ring[src] = ring[node] + 1
                    nxt.append(src)
        frontier = nxt

    attached = set(ring)
    detached = [i for i in range(g.n_nodes) if i not in ring]

    by_ring: dict[int, list[int]] = {}
    for node, r in ring.items():
        by_ring.setdefault(r, []).append(node)

    pos: dict[int, tuple[float, float]] = {}
    n_rings = max(by_ring) or 1
    for r, members in sorted(by_ring.items()):
        members.sort()
        if r == 0:
            pos[members[0]] = (0.0, 0.0)
            continue
        radius = r / n_rings
        # Offset alternate rings by half a step so spokes do not line up into
        # false radial spars that the genome does not have.
        phase = (np.pi / len(members)) * (r % 2)
        for k, node in enumerate(members):
            theta = 2.0 * np.pi * k / len(members) + phase + np.pi / 2
            pos[node] = (radius * np.cos(theta), radius * np.sin(theta))

    for k, node in enumerate(sorted(detached)):
        theta = 2.0 * np.pi * k / max(len(detached), 1) - np.pi / 2
        pos[node] = (1.17 * np.cos(theta), 1.17 * np.sin(theta))

    return pos, attached


def draw_specimen(ax, g, weights, task: str) -> dict:
    """One organism: fine pale line work, accents only where an operator earns it."""
    X = make_bundle(task, seed=50001).train.X[:48]
    live_conns, live_nodes = causal_elements(g, X, weights, True)
    pos, _attached = radial_layout(g)
    wmax = max((abs(float(w)) for w in weights), default=1.0) or 1.0

    # Dense organisms need finer line work, or the skeleton becomes a smear.
    n_drawn = max(g.n_nodes, 1)
    fine = float(np.clip(np.sqrt(9.0 / n_drawn), 0.30, 1.0))

    for ci in range(g.n_connections):
        if not g.active[ci]:
            continue
        s, d = int(g.src[ci]), int(g.dst[ci])
        if s not in pos or d not in pos:
            continue
        causal = ci in live_conns
        w = abs(float(weights[ci])) / wmax
        ax.add_patch(
            FancyArrowPatch(
                pos[s], pos[d],
                connectionstyle="arc3,rad=0.16",
                arrowstyle="-",
                linewidth=((0.30 + 1.45 * w) if causal else 0.28) * fine,
                color=style.INK if causal else BONE_DIM,
                alpha=(0.90 if causal else 0.26) * (1.0 if fine > 0.6 else 0.75),
                zorder=2,
            )
        )

    r_base = 0.072 * fine + 0.012
    for node, (x, y) in pos.items():
        causal = node in live_nodes or node == OUT
        structural = node < N_STRUCTURAL
        op = OP_NAMES[g.ops[node]]
        if node == OUT:
            # Filled bone: the one node that is neither an operator nor an
            # input, so it must not borrow an operator's accent from the key.
            r, face, edge = r_base * 1.4, style.INK, style.SURFACE
        elif structural:
            r, face, edge = r_base * 0.85, style.SURFACE, style.INK if causal else BONE_DIM
        else:
            r, face, edge = r_base, accent(op) if causal else style.SURFACE, (
                style.INK if causal else BONE_DIM)
        ax.add_patch(
            Circle((x, y), r, facecolor=face, edgecolor=edge,
                   linewidth=0.75 * max(fine, 0.5), alpha=1.0 if causal else 0.45,
                   zorder=4)
        )
        if node == OUT:
            # A faint halo, the way Haeckel haloes a central capsule.
            ax.add_patch(
                Circle((x, y), r * 2.0, facecolor="none", edgecolor=style.INK,
                       linewidth=0.4, alpha=0.45, zorder=3)
            )

    ax.set_xlim(-1.26, 1.26)
    ax.set_ylim(-1.26, 1.26)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(style.SURFACE)
    return {
        "causal_hidden": len({i for i in live_nodes if i >= N_STRUCTURAL}),
        "represented_hidden": g.n_nodes - N_STRUCTURAL,
        "ops": {OP_NAMES[g.ops[i]] for i in live_nodes if i >= N_STRUCTURAL},
    }


# --------------------------------------------------------------------------
# Specimen selection — the median, never the best
# --------------------------------------------------------------------------


def median_champion(release: Path, task: str, condition: str):
    """The median replicate by sealed-test accuracy, with its genome and weights."""
    runs = [
        json.loads(p.read_text())
        for p in sorted((release / "raw" / "runs").glob(f"{task}__{condition}__*.json"))
    ]
    if not runs:
        return None
    fpath = release / "final-test.json"
    final = {}
    if fpath.exists():
        final = {r["run_id"]: r for r in json.loads(fpath.read_text())["results"]}
    key = (
        (lambda r: final[r["run_id"]]["test_accuracy"])
        if final and all(r["run_id"] in final for r in runs)
        else (lambda r: r["metrics"]["validation_accuracy"])
    )
    runs.sort(key=key)
    rec = runs[(len(runs) - 1) // 2]
    g, w = deserialise_genome(rec["champion"])
    return g, w


# --------------------------------------------------------------------------
# Plate composition
# --------------------------------------------------------------------------

TOP_IN = 1.30       # title block
ROW_IN = 2.60       # one row of specimens, caption included
CAP_IN = 0.72       # the caption band inside a row
COL_IN = 2.35
LEFT_IN = 0.86      # the row-label margin
RIGHT_IN = 0.28
BOTTOM_IN = 1.05    # footer and operator key
CODA_IN = 3.45      # the comparison specimen, when a plate carries one


def _operator_key(fig, ops: set[str], y: float, aspect: float) -> None:
    """Which accent is which operator, for the operators this plate actually shows.

    `aspect` is width/height in inches: a circle in figure-fraction coordinates
    is an ellipse unless the radii are corrected for it.
    """
    present = [o for o in OP_NAMES.values() if o in ops and o != "null"]
    if not present:
        return
    step = 1.0 / (len(present) + 1)
    r = 0.0055
    for i, op in enumerate(present):
        x = step * (i + 1)
        fig.add_artist(Ellipse(
            (x - 0.016, y), width=2 * r, height=2 * r * aspect,
            transform=fig.transFigure, facecolor=accent(op), edgecolor=style.INK,
            linewidth=0.5, zorder=11))
        fig.text(x + 0.002, y, op, ha="left", va="center", fontsize=6.4,
                 color=style.INK2, family="serif")


def plate(rows, title, subtitle, footer, out: Path, coda=None, ncols=None):
    """`rows` is a list of (row_label | None, [specimen, ...])."""
    ncols = ncols or max(len(specs) for _, specs in rows)
    nrows = len(rows)
    has_row_labels = any(lbl for lbl, _ in rows)
    left_in = LEFT_IN if has_row_labels else RIGHT_IN

    fig_w = left_in + ncols * COL_IN + RIGHT_IN
    fig_h = TOP_IN + nrows * ROW_IN + (CODA_IN if coda else 0.0) + BOTTOM_IN
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=style.SURFACE)

    def fx(inches):
        return inches / fig_w

    def fy(inches):  # inches from the top
        return 1.0 - inches / fig_h

    all_ops: set[str] = set()

    for ri, (row_label, specs) in enumerate(rows):
        row_top = TOP_IN + ri * ROW_IN
        art_h = ROW_IN - CAP_IN
        side = min(COL_IN * 0.88, art_h)
        indent = (ncols - len(specs)) * COL_IN / 2.0
        for ci, spec in enumerate(specs):
            cx = left_in + indent + ci * COL_IN + COL_IN / 2.0
            ax = fig.add_axes([
                fx(cx - side / 2.0), fy(row_top + art_h / 2.0 + side / 2.0),
                fx(side), side / fig_h,
            ])
            info = draw_specimen(ax, spec["genome"], spec["weights"], spec["task"])
            all_ops |= info["ops"]
            label = spec["label"]
            fig.text(fx(cx), fy(row_top + art_h + 0.12), spaced(label),
                     ha="center", va="top", color=style.INK2, family="serif",
                     fontsize=7.0 if len(label) <= 17 else 6.0)
            fig.text(fx(cx), fy(row_top + art_h + 0.34),
                     f"{info['causal_hidden']} of {info['represented_hidden']} units",
                     ha="center", va="top", color=style.RULE, family="serif",
                     style="italic", fontsize=6.6)
        if row_label:
            fig.text(fx(0.40), fy(row_top + art_h / 2.0), spaced(row_label),
                     ha="center", va="center", rotation=90, color=style.INK2,
                     family="serif", fontsize=7.6)

    if coda is not None:
        spec, heading, prose = coda
        band_top = TOP_IN + nrows * ROW_IN
        # A hairline above the comparison specimen: it is of a different kind.
        fig.add_artist(plt.Line2D(
            [fx(left_in), fx(fig_w - RIGHT_IN)], [fy(band_top + 0.10)] * 2,
            color=style.RULE, linewidth=0.5, alpha=0.8, transform=fig.transFigure))
        side = CODA_IN - 1.25
        cx = left_in + COL_IN * 0.72
        prose_x = left_in + COL_IN * 1.42
        prose_w = fig_w - RIGHT_IN - 0.18 - prose_x
        # Wrap to the column that is actually there, rather than to a guess.
        chars = max(28, int(prose_w / 0.0585))
        prose = "\n\n".join(
            textwrap.fill(par, chars) for par in prose.split("\n\n"))
        ax = fig.add_axes([
            fx(cx - side / 2.0), fy(band_top + 0.55 + side), fx(side), side / fig_h,
        ])
        info = draw_specimen(ax, spec["genome"], spec["weights"], spec["task"])
        all_ops |= info["ops"]
        fig.text(fx(cx), fy(band_top + 0.62 + side), spaced(heading),
                 ha="center", va="top", color=style.INK2, family="serif", fontsize=7.4)
        fig.text(fx(cx), fy(band_top + 0.84 + side),
                 f"{info['causal_hidden']} of {info['represented_hidden']} units",
                 ha="center", va="top", color=style.RULE, family="serif",
                 style="italic", fontsize=6.6)
        fig.text(fx(prose_x), fy(band_top + 0.58), prose,
                 ha="left", va="top", color=style.INK2, family="serif",
                 fontsize=7.9, linespacing=1.72)

    fig.text(0.5, fy(0.52), spaced(title), ha="center", va="center",
             fontsize=14.5, color=style.INK, family="serif")
    fig.text(0.5, fy(0.86), subtitle, ha="center", va="center",
             fontsize=9.0, color=style.INK2, family="serif", style="italic")
    # A single rule under the title block, as a plate has.
    fig.add_artist(plt.Line2D([0.30, 0.70], [fy(1.03)] * 2, color=style.RULE,
                              linewidth=0.6, transform=fig.transFigure))

    _operator_key(fig, all_ops, fy(fig_h - 0.74), fig_w / fig_h)
    fig.text(0.5, fy(fig_h - 0.44), footer, ha="center", va="center",
             fontsize=6.9, color=style.RULE, family="serif")

    style.plate_frame(fig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=style.PLATE_DPI, facecolor=style.SURFACE)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return out


# --------------------------------------------------------------------------
# The plates
# --------------------------------------------------------------------------

TAFEL_I_PROSE = (
    "Zum Vergleich: das feste Netz.\n\n"
    "Every specimen above was built by a search. This one was not. It is the "
    "fixed two-layer network the searched architectures are measured against, "
    "and it is the same network on every geometry \u2014 which is why it is "
    "drawn once here rather than three times.\n\n"
    "Its uniform rim is not an artefact of the drawing. Sixty-five units, one "
    "operator, two layers: the thing a search does not have to be told."
)


def plate_i(out: Path) -> Path:
    """What two different searches, and no search at all, build on each geometry."""
    v4 = ROOT / "results" / "backprop-neat-v4"
    rows = []
    for task, task_label in (("spiral", "Spirals"),
                             ("checkerboard", "Checkerboard"),
                             ("spiral3", "Three-arm spiral")):
        specs = []
        for cond, label in (("bpneat", "Backprop-NEAT"),
                            ("cgp", "Cartesian GP"),
                            ("cgp_random_matched", "Sampled, not searched")):
            got = median_champion(v4, task, cond)
            if got is None:
                continue
            g, w = got
            specs.append({"genome": g, "weights": w, "label": label, "task": task})
        if specs:
            rows.append((task_label, specs))

    coda = None
    got = median_champion(v4, "spiral", "fixed_tanh_ha")
    if got is not None:
        g, w = got
        coda = ({"genome": g, "weights": w, "task": "spiral"},
                "The fixed network", TAFEL_I_PROSE)

    return plate(
        rows,
        "Kunstformen der Architektur",
        "Tafel I — Formen, die von zwei Suchverfahren gefunden wurden",
        "Median specimen of thirty replicates, protocol v4. Bright filaments reach the output; "
        "faint ones are carried by the genome and never arrive.",
        out, coda=coda, ncols=3,
    )


def plate_ii(out: Path) -> Path:
    """One organism under seven conditions — complexification, as a progression."""
    v5 = ROOT / "results" / "backprop-neat-v5"
    arms = (
        ("neat_reference", "Reference"),
        ("neat_no_crossover", "Without crossover"),
        ("neat_no_speciation", "Without speciation"),
        ("neat_no_penalty", "Without the penalty"),
        ("neat_complexify", "Higher mutation"),
        ("neat_complexify_no_penalty", "Neither constraint"),
        ("neat_deep_narrow", "Deep and narrow"),
    )
    specs = []
    for cond, label in arms:
        got = median_champion(v5, "spiral", cond)
        if got is None:
            continue
        g, w = got
        specs.append({"genome": g, "weights": w, "label": label, "task": "spiral"})
    rows = [(None, specs[i:i + 4]) for i in range(0, len(specs), 4)]
    return plate(
        rows,
        "Kunstformen der Architektur",
        "Tafel II — Dieselbe Art unter sieben Bedingungen",
        "Median specimen of thirty replicates on spirals, protocol v5. Releasing the complexity "
        "penalty and raising the mutation rate grows the form; neither is the algorithm's limit.",
        out, ncols=4,
    )


def main() -> int:
    if not (ROOT / "results" / "backprop-neat-v4" / "raw" / "runs").is_dir():
        print("v4 release missing", file=sys.stderr)
        return 1
    plate_i(OUT_DIR / "plate-i-forms.png")
    if (ROOT / "results" / "backprop-neat-v5" / "raw" / "runs").is_dir():
        plate_ii(OUT_DIR / "plate-ii-complexification.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
